#!/usr/bin/env python3

"""Evaluate a COMET diffusion policy on the real UR5e via FDCCEnv.

Usage:
    python evaluate_policy.py

    # Override eval settings:
    python evaluate_policy.py eval.num_rollouts=5 eval.max_timesteps=500

    # Point to a different env config (required for first run):
    python evaluate_policy.py +eval.env_config=/path/to/data_collection.yaml

Controls during each rollout:
    Enter  - confirm start of rollout (after reset prompt)
"""

import datetime
import logging
import signal
import sys
import timeit
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import rospy

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ur_control import transformations

from comet.common.utils.utils import load_base_policy
from comet.common.utils.ft_visualizer import FTVisualizer
from comet.common.utils.viz_utils import save_to_video
from comet.common.policies.types import FeatureType

from osx_ur5e.fdcc_env import FDCCEnv

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def _signal_handler(sig, frame):
    logger.info("Interrupted — shutting down.")
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_file: Path) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in root.handlers[:]:
        if not isinstance(h, logging.FileHandler):
            root.removeHandler(h)
    root.addHandler(RichHandler(console=console, rich_tracebacks=True, show_path=False))
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"))
    root.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Observation formatting
# ---------------------------------------------------------------------------

def format_real_robot_observations(
    arm,
    image_recorder,
    features: dict,
    camera_shape: tuple,
) -> dict:
    """Build a policy-ready observation dict from the real robot arm and cameras.

    Mirrors get_observations() in data_collection.py but returns torch tensors
    in the same format expected by the COMET policy (float32 states, uint8 images).

    Args:
        arm: CompliantController instance from FDCCEnv.
        image_recorder: ImageRecorder instance from FDCCEnv.
        features: Feature dict loaded from the policy checkpoint (used to filter keys).
        camera_shape: (H, W) to resize camera images to match training resolution.

    Returns:
        Dict mapping observation keys to torch tensors ready for policy.select_action().
    """
    eef = arm.end_effector()
    eef_velocity = arm.end_effector_velocity()

    raw_obs = {
        "observation.qpos":                    arm.joint_angles(),
        "observation.qvel":                    arm.joint_velocities(),
        "observation.eef.position":            eef[:3],
        "observation.eef.linear_velocity":     eef_velocity[:3],
        "observation.eef.angular_velocity":    eef_velocity[3:],
        "observation.eef.rotation_ortho6":     transformations.ortho6_from_quaternion(eef[3:]),
        "observation.eef.rotation_axis_angle": transformations.axis_angle_from_quaternion(eef[3:]),
        "observation.ft":                      arm.get_wrench(), # TODO (malek): check if this is in the world frame or the tool frame with cristian
    }

    obs = {}

    # State observations — only keep keys the policy actually uses
    for key, value in raw_obs.items():
        if key in features:
            obs[key] = torch.tensor(np.array(value).flatten(), dtype=torch.float32)

    # Camera images
    if image_recorder is not None:
        resize_transform = transforms.Resize(camera_shape, antialias=True)
        raw_images = image_recorder.get_images()
        for cam_name, image_hwc in raw_images.items():
            feat_key = f"observation.images.{cam_name}"
            if feat_key in features:
                image_chw = np.ascontiguousarray(np.transpose(image_hwc, (2, 0, 1)))
                image_tensor = torch.tensor(image_chw, dtype=torch.uint8)
                obs[feat_key] = resize_transform(image_tensor)

    return obs


# ---------------------------------------------------------------------------
# Action conversion
# ---------------------------------------------------------------------------

def convert_policy_action(action_dict: dict, actions_as_deltas: bool) -> dict:
    """Convert COMET policy output tensors to the FDCCEnv action dict format.

    FDCCEnv.set_compliant_control_action() expects:
        action['action.position']     — numpy array (3,)
        action['action.orientation']  — numpy array (3,) for deltas or (6,) for absolute
        action['action.stiffness_diag'] or action['action.stiffness_cholesky']

    The policy outputs action.rotation_axis_angle (3D, delta mode) or
    action.rotation_ortho6 (6D, absolute mode). This function renames the
    appropriate key to 'action.orientation'.

    Args:
        action_dict: Dict of tensors from policy.select_action().
        actions_as_deltas: Matches env.actions_as_deltas config.

    Returns:
        Dict with numpy array values ready for FDCCEnv.step().
    """
    env_action = {}

    for key, value in action_dict.items():
        if isinstance(value, torch.Tensor):
            np_value = value.squeeze(0).cpu().numpy() if value.dim() > 0 else value.cpu().numpy()
        else:
            np_value = np.array(value)

        if actions_as_deltas and key == "action.rotation_axis_angle":
            env_action["action.orientation"] = np_value
        elif not actions_as_deltas and key == "action.rotation_ortho6":
            env_action["action.orientation"] = np_value
        else:
            env_action[key] = np_value

    return env_action


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# config_path is relative to this file's directory
@hydra.main(
    version_base=None,
    config_path="../../../../../dependencies/comet/configs",
    config_name="blackboard_wipe",
)
def main(cfg: DictConfig) -> None:
    # output_dir = Path(HydraConfig.get().runtime.output_dir)
    # output_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = Path(cfg.eval.base.load_ckpt) / "eval" / str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    eval_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(eval_dir / "evaluation.log")

    seed = cfg.eval.seed
    num_rollouts = cfg.eval.num_rollouts
    max_timesteps = cfg.eval.max_timesteps
    save_video = OmegaConf.select(cfg, "eval.save_video", default=False)
    policy_filename = OmegaConf.select(cfg, "eval.policy_filename", default="best_ema_policy.ckpt")

    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info(f"Seed: {seed}")
    logger.info(f"Results will be saved to: {eval_dir}")

    # ------------------------------------------------------------------
    # Load policy
    # ------------------------------------------------------------------
    ckpt_dir = Path(cfg.eval.base.load_ckpt)
    logger.info(f"Loading policy from: {ckpt_dir}")
    policy, base_cfg, features = load_base_policy(ckpt_dir, policy_filename)
    policy.cuda()
    policy.eval()

    action_keys = {key: ft.shape for key, ft in features.items() if ft.type is FeatureType.ACTION}
    logger.info(f"Action keys: {list(action_keys.keys())}")
    logger.info(f"Observation keys: {[k for k, ft in features.items() if ft.type is not FeatureType.ACTION]}")

    # Camera shape expected by the policy (H, W)
    if hasattr(base_cfg, "model_specific") and "camera_shape" in base_cfg.model_specific:
        camera_shape = tuple(base_cfg.model_specific.camera_shape[-2:])
    else:
        camera_shape = (240, 320)
        logger.warning(f"No camera_shape in checkpoint config, defaulting to {camera_shape}")

    control_frequency = int(base_cfg.dataset.dataset.fps)
    logger.info(f"Control frequency: {control_frequency} Hz")

    # ------------------------------------------------------------------
    # Load env config and build FDCCEnv
    # ------------------------------------------------------------------
    # env_config can be overridden via: +eval.env_config=/path/to/yaml
    # env_config_path = OmegaConf.select(cfg, "eval.env_config", default=None)
    # if env_config_path is None:
    #     raise ValueError("eval.env_config must be set")
    # else:
    #     env_config_path = Path(env_config_path)

    # logger.info(f"Loading env config from: {env_config_path}")
    # raw_env_cfg = OmegaConf.load(env_config_path)
    # Support both bare env config and configs that nest it under an 'env' key
    # env_cfg = raw_env_cfg.get("env", raw_env_cfg) if hasattr(raw_env_cfg, "get") else raw_env_cfg
    # env_cfg = cfg

    rospy.init_node("evaluate_policy", anonymous=False)
    logger.info("ROS node initialized")

    env = FDCCEnv(config=cfg, use_torch_for_cameras=False)

    # FDCCEnv.reset() asserts reference_trajectory is not None even though
    # it is unused during the actual reset movement — satisfy the check.
    env.reference_trajectory = []

    actions_as_deltas = env.actions_as_deltas
    logger.info(f"actions_as_deltas: {actions_as_deltas}")

    # ------------------------------------------------------------------
    # FT Visualizer
    # ------------------------------------------------------------------
    include_stiffness = any("stiffness" in k for k in action_keys)
    ft_visualizer = FTVisualizer(
        maxlen=max_timesteps,
        include_stiffness=include_stiffness,
        force_ylim=(-5, 80),
        arm="right",
        figure_size=(5, 4),
        headless=True,
    )

    np.set_printoptions(linewidth=np.inf, formatter={"float": lambda x: f"{x:0.3f}"})
    torch.set_printoptions(linewidth=2000, sci_mode=False, precision=5)

    # ------------------------------------------------------------------
    # Move home once before evaluation
    # ------------------------------------------------------------------
    logger.info("Moving to home position...")
    env.go_home()

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    total_steps_per_episode = []
    force_violations = 0
    all_rollout_frames = []

    eval_start_time = timeit.default_timer()
    total_steps_completed = 0

    def make_step_description(rollout_id: int, steps: int) -> str:
        elapsed = timeit.default_timer() - eval_start_time
        fps = steps / elapsed if elapsed > 0 else 0.0
        return f"Rollout {rollout_id + 1}/{num_rollouts} | FPS: {fps:.1f}"

    progress_columns = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=80),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    with Progress(*progress_columns, console=console) as progress:
        rollout_task = progress.add_task("Evaluation", total=num_rollouts)

        for rollout_id in range(num_rollouts):
            step_task = progress.add_task(
                make_step_description(rollout_id, total_steps_completed),
                total=max_timesteps,
            )

            # Reset and wait for user confirmation
            env.reset(move_robot=True)
            progress.stop()
            input(f"\n  Rollout {rollout_id + 1}/{num_rollouts} — press Enter to start...")
            progress.start()

            env.activate_compliance_control()
            policy.reset()

            episode_frames = []
            force_violation = False

            for t in range(max_timesteps):
                # --- Observe ---
                obs = format_real_robot_observations(
                    env.arm,
                    env.image_recorder,
                    features,
                    camera_shape,
                )

                # Batch dimension for the policy
                policy_obs = {
                    k: v.cuda().float().unsqueeze(0) if isinstance(v, torch.Tensor) else v
                    for k, v in obs.items()
                }

                # --- Act ---
                with torch.no_grad():
                    action = policy.select_action(policy_obs)
                    if False:
                        # action["action.normal_force"] = torch.tensor([10], device="cuda")  # override for now to test the controller
                        print("will edit the force")
                        # action["action.contact_direction"] = torch.tensor([[0.0, 0.0, -1.0]], device="cuda")  # straight down
                        target_force = 5.0
                        action["action.normal_force"] = torch.tensor([target_force], device="cuda")  # override for now to test the controller
                        # f_low = DEFAULT_STIFFNESS_PARAMS['f_low']
                        # f_high = DEFAULT_STIFFNESS_PARAMS['f_high']
                        # K_max = DEFAULT_STIFFNESS_PARAMS['k_max']
                        # K_min = DEFAULT_STIFFNESS_PARAMS['k_min']
                        f_low = 1.0
                        f_high = 10.0
                        K_max = 1000
                        K_min = 500.0
                        # max_displacement = DEFAULT_STIFFNESS_PARAMS['max_displacement']
                        max_displacement = 0.015

                        f_norm = abs(target_force)
                        if f_norm < f_low:
                            K = K_max
                        elif f_norm > f_high:
                            K = K_min
                        else:
                            # K = K_max - (K_max - K_min) * (f_norm - f_low) / (f_high - f_low)
                            K_gap = K_max - K_min
                            f_gap = f_high - f_low
                            K = K_max - K_gap * (f_norm - f_low) / f_gap

                        if max_displacement > 0 and f_norm > f_low:
                            k_floor = f_norm / max_displacement
                            print(f"{k_floor=}")
                            # return max(k_raw, k_floor)
                            K = np.clip(k_floor, K_min, K_max)

                        print(f"{K=}")

                        action["action.estimated_stiffness"] = torch.tensor([K], device="cuda")

                action_dict = { # FIXME: why do this if it exist in the fucntion below as well (convert_policy_action)?
                    k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
                    for k, v in action.items()
                }
                env_action = convert_policy_action(action_dict, actions_as_deltas)

                # --- Step ---
                timestep = env.step(env_action)
                done = timestep.last()

                # --- Record images for video ---
                if save_video and env.image_recorder is not None:
                    raw_images = env.image_recorder.get_images()
                    if raw_images:
                        frames_list = list(raw_images.values())
                        frame = np.concatenate(frames_list, axis=1) if len(frames_list) > 1 else frames_list[0]
                        episode_frames.append(frame)

                # --- FT tracking ---
                wrench = env.arm.get_wrench()
                stiffness = None
                if include_stiffness:
                    if "action.stiffness_diag" in action_dict:
                        val = action_dict["action.stiffness_diag"]
                        stiffness = float(np.mean(val.cpu().numpy() if isinstance(val, torch.Tensor) else val))
                    elif "action.estimated_stiffness" in action_dict:
                        val = action_dict["action.estimated_stiffness"]
                        stiffness = float(val.cpu().item() if isinstance(val, torch.Tensor) else val)
                ft_visualizer.add_data(t, wrench, stiffness)

                total_steps_completed += 1
                progress.update(
                    step_task,
                    advance=1,
                    description=make_step_description(rollout_id, total_steps_completed),
                )

                if done:
                    force_violation = True
                    logger.info(
                        f"Rollout {rollout_id} ended early at step {t + 1} "
                        f"(force/torque limit exceeded)"
                    )
                    break

            env.deactivate_compliance_control()

            steps_taken = t + 1
            total_steps_per_episode.append(steps_taken)
            if force_violation:
                force_violations += 1

            logger.info(
                f"Rollout {rollout_id} complete | steps: {steps_taken} | "
                f"force violation: {force_violation}"
            )

            # --- Save per-rollout artefacts ---
            ft_visualizer.render_now()
            ft_visualizer.save(eval_dir / f"rollout_{rollout_id}.png")
            _, forces_data, _ = ft_visualizer.get_data()
            np.save(eval_dir / f"rollout_{rollout_id}_contact_force.npy", forces_data)
            ft_visualizer.clear()

            if save_video and episode_frames:
                save_to_video(episode_frames, eval_dir / "videos", f"rollout_{rollout_id}.mp4", control_frequency)
            if save_video:
                all_rollout_frames.extend(episode_frames)

            progress.remove_task(step_task)
            violation_pct = f"{force_violations / (rollout_id + 1) * 100:.1f}%"
            progress.update(
                rollout_task,
                advance=1,
                description=f"Evaluation | force violations: {violation_pct}",
            )

    env.go_home()
    logger.info("Returned to home position after evaluation.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"Rollouts completed:        {num_rollouts}")
    logger.info(f"Force violations:          {force_violations} / {num_rollouts} "
                f"({force_violations / num_rollouts * 100:.1f}%)")
    logger.info(f"Mean steps per episode:    {np.mean(total_steps_per_episode):.1f}")
    logger.info(f"Std  steps per episode:    {np.std(total_steps_per_episode):.1f}")
    logger.info("=" * 60)

    if save_video and all_rollout_frames:
        save_to_video(all_rollout_frames, eval_dir, "evaluation_all_rollouts.mp4", control_frequency)

    ft_visualizer.close()
    logger.info(f"Evaluation complete. Results saved to: {eval_dir}")


if __name__ == "__main__":
    if "hydra.run.dir" not in " ".join(sys.argv):
        sys.argv.append("paths.script=/eval")
    main()
