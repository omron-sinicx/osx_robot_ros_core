#!/usr/bin/env python3
"""Evaluate LeRobot ACT / Diffusion baselines on the real UR5e via FDCCEnv.

Usage:
    python evaluate_policy_baseline.py

    python evaluate_policy_baseline.py \\
        --config-name baseline_book_flipping \\
        paths.ckpt_name=baseline/act/2026-06-27/13-25-06 \\
        eval.num_rollouts=5

    # Sync inference (default when eval.inference.mode=sync):
    python evaluate_policy_baseline.py eval.inference.mode=sync

Controls during each rollout:
    Enter  - confirm start of rollout (after reset prompt)
    q      - stop current rollout and end evaluation
"""

from __future__ import annotations

import datetime
import logging
import signal
import sys
import timeit
from pathlib import Path

import hydra
import numpy as np
import rospy
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms
from pynput import keyboard as kb
from tqdm import tqdm
from ur_control import transformations

from comet.common.utils.ft_visualizer import FTVisualizer
from comet.common.utils.utils import load_baseline_policy
from comet.common.utils.viz_utils import save_to_video
from comet.inference import BaselineAsyncInferenceEngine, build_baseline_inference_engine
from comet.inference.utils import split_action_tensor, to_cuda_batch
from osx_ur5e.fdcc_env import FDCCEnv
from osx_ur5e.utils import setup_logging

logger = logging.getLogger(__name__)


def _signal_handler(sig, frame):
    logger.info("Interrupted — shutting down.")
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)


# ---------------------------------------------------------------------------
# Keyboard listener
# ---------------------------------------------------------------------------

def start_stop_listener(events: dict):
    """Start a background keyboard listener; 'q' sets events['stop'] = True."""

    def on_press(key):
        try:
            if key.char and key.char.lower() == "q":
                events["stop"] = True
        except AttributeError:
            pass

    listener = kb.Listener(on_press=on_press)
    listener.start()
    return listener


def format_real_robot_observations(
    arm,
    image_recorder,
    wanted_keys: set[str],
    camera_shape: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Build a policy-ready observation dict from the real robot arm and cameras."""
    eef = arm.end_effector()
    eef_velocity = arm.end_effector_velocity()

    raw_obs = {
        "observation.qpos": arm.joint_angles(),
        "observation.qvel": arm.joint_velocities(),
        "observation.eef.position": eef[:3],
        "observation.eef.linear_velocity": eef_velocity[:3],
        "observation.eef.angular_velocity": eef_velocity[3:],
        "observation.eef.rotation_ortho6": transformations.ortho6_from_quaternion(eef[3:]),
        "observation.eef.rotation_ortho6_site": transformations.ortho6_from_quaternion(eef[3:]),
        "observation.eef.rotation_axis_angle": transformations.axis_angle_from_quaternion(eef[3:]),
        "observation.ft": arm.get_wrench(),
    }

    obs: dict[str, torch.Tensor] = {}
    for key, value in raw_obs.items():
        if key in wanted_keys:
            obs[key] = torch.tensor(np.array(value).flatten(), dtype=torch.float32)

    if image_recorder is not None:
        resize_transform = transforms.Resize(camera_shape, antialias=True)
        raw_images = image_recorder.get_images()
        for cam_name, image_hwc in raw_images.items():
            feat_key = f"observation.images.{cam_name}"
            if feat_key in wanted_keys:
                image_chw = np.ascontiguousarray(np.transpose(image_hwc, (2, 0, 1)))
                image_tensor = torch.tensor(image_chw, dtype=torch.uint8)
                obs[feat_key] = resize_transform(image_tensor)

    missing = wanted_keys - set(obs.keys())
    if missing:
        raise KeyError(f"Missing required observation keys: {sorted(missing)}")

    return obs


def convert_policy_action(action_dict: dict) -> dict:
    """Convert baseline policy output tensors to the FDCCEnv factored action format."""
    env_action = {}
    for key, value in action_dict.items():
        if isinstance(value, torch.Tensor):
            env_action[key] = value.squeeze(0).cpu().numpy() if value.dim() > 0 else value.cpu().numpy()
        else:
            env_action[key] = np.array(value)
    return env_action


@hydra.main(
    version_base=None,
    config_path="../../../../../dependencies/comet/configs",
    config_name="baseline_book_flipping",
)
def main(cfg: DictConfig) -> None:
    stop_events = {"stop": False}
    kb_listener = start_stop_listener(stop_events)
    try:
        main_loop(cfg, stop_events)
    finally:
        kb_listener.stop()
        logger.info("Keyboard listener stopped.")


def main_loop(cfg: DictConfig, stop_events: dict) -> None:
    eval_dir = Path(cfg.eval.base.load_ckpt) / "eval" / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    eval_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg.eval.seed
    num_rollouts = cfg.eval.num_rollouts
    max_timesteps = cfg.eval.max_timesteps
    save_video = OmegaConf.select(cfg, "eval.save_video", default=False)

    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info(f"Seed: {seed}")
    logger.info(f"Results will be saved to: {eval_dir}")

    policy_filename = OmegaConf.select(cfg, "eval.policy_filename", default="best_ema_policy.ckpt")

    ckpt_dir = Path(cfg.eval.base.load_ckpt)
    logger.info(f"Loading baseline policy from: {ckpt_dir}")
    bundle = load_baseline_policy(ckpt_dir, ckpt_name=policy_filename)
    bundle.policy.cuda()
    bundle.policy.eval()

    wanted_keys = set(bundle.feature_groups.state_keys) | set(bundle.feature_groups.image_keys)
    action_keys = list(bundle.feature_groups.action_keys)
    logger.info(f"Action keys: {action_keys}")
    logger.info(f"Observation keys: {sorted(wanted_keys)}")

    required_factored = (
        "action.ref_position",
        "action.ref_rotation_ortho6",
        "action.contact_direction",
        "action.normal_force",
        "action.normal_torque",
        "action.estimated_stiffness",
    )
    required_vt = (
        "action.ref_position",
        "action.ref_rotation_ortho6",
        "action.virtual_target_position",
        "action.virtual_target_rotation",
        "action.estimated_stiffness",
    )
    if all(key in action_keys for key in required_factored):
        action_repr = "factored"
    elif all(key in action_keys for key in required_vt):
        action_repr = "virtual_target"
    else:
        raise ValueError(
            "Unsupported baseline action representation. "
            f"Got: {action_keys}. Expected factored or virtual-target keys."
        )
    logger.info(f"Action representation: {action_repr}")

    model_section = cfg.model_configs[cfg.model.name]
    camera_shape = tuple(model_section.camera_shape[-2:])
    control_frequency = int(cfg.dataset.dataset.fps)
    logger.info(f"Control frequency: {control_frequency} Hz")
    logger.info(f"Camera shape: {camera_shape}")

    inference_engine = build_baseline_inference_engine(bundle, cfg, control_frequency)
    inference_mode = OmegaConf.select(cfg, "eval.inference.mode", default="sync")
    logger.info(f"Inference mode: {inference_mode}")
    inference_engine.start()

    rospy.init_node("evaluate_policy_baseline", anonymous=False)

    hydra_dir = Path(HydraConfig.get().runtime.output_dir)
    hydra_job_log = hydra_dir / f"{HydraConfig.get().job.name}.log"
    eval_log = eval_dir / "evaluation.log"
    setup_logging(hydra_job_log, eval_log)
    logger.info("ROS node initialized")
    logger.info("Log files: %s , %s", hydra_job_log, eval_log)
    logger.info(f"Eval artifacts: {eval_dir}")

    env = FDCCEnv(config=cfg, use_torch_for_cameras=False)
    env.reference_trajectory = []

    if env.actions_as_deltas:
        logger.warning(
            "env.actions_as_deltas is True but baseline actions are absolute. "
            "Set controller.actions_as_deltas=false for factored/virtual-target checkpoints."
        )

    include_stiffness = any("stiffness" in key for key in action_keys)
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

    logger.info("Moving to home position...")
    env.go_home()

    total_steps_per_episode = []
    force_violations = 0
    all_rollout_frames = []
    eval_start_time = timeit.default_timer()
    total_steps_completed = 0

    rollout_bar = tqdm(range(num_rollouts), desc="Evaluation", unit="rollout")
    for rollout_id in rollout_bar:
        inference_engine.pause()
        env.reset(move_robot=False)
        input(f"\n  Rollout {rollout_id + 1}/{num_rollouts} — press Enter to start...")

        env.activate_compliance_control()
        inference_engine.reset()
        inference_engine.resume()

        if isinstance(inference_engine, BaselineAsyncInferenceEngine):
            bootstrap_obs = format_real_robot_observations(
                env.arm,
                env.image_recorder,
                wanted_keys,
                camera_shape,
            )
            inference_engine.notify_observation(to_cuda_batch(bootstrap_obs))
            logger.info("Waiting for first inference chunk (cold start / GPU warmup)...")
            if not inference_engine.wait_for_action(timeout_s=60.0):
                if inference_engine.failed:
                    logger.error("Background inference thread failed on first chunk — skipping rollout.")
                else:
                    logger.error("Timed out waiting for first inference chunk — skipping rollout.")
                inference_engine.pause()
                continue
            logger.info("First chunk ready — starting rollout.")

        episode_frames = []
        force_violation = False

        step_bar = tqdm(
            range(max_timesteps),
            desc=f"Rollout {rollout_id + 1}/{num_rollouts}",
            unit="step",
            leave=False,
        )
        for t in step_bar:
            if stop_events["stop"]:
                logger.info("'q' pressed — stopping rollout early.")
                inference_engine.pause()
                break

            obs = format_real_robot_observations(
                env.arm,
                env.image_recorder,
                wanted_keys,
                camera_shape,
            )
            policy_obs = to_cuda_batch(obs)

            if isinstance(inference_engine, BaselineAsyncInferenceEngine):
                inference_engine.notify_observation(policy_obs)
                action_tensor = inference_engine.get_action(None)
            else:
                action_tensor = inference_engine.get_action(policy_obs)

            if action_tensor is None:
                if inference_engine.failed:
                    logger.error("Inference engine failed; ending rollout early.")
                    break
                rospy.sleep(1.0 / control_frequency)
                continue

            action_dict = split_action_tensor(
                action_tensor,
                action_keys,
                bundle.output_sizes,
            )
            env_action = convert_policy_action(action_dict)
            timestep = env.step(env_action)
            done = timestep.last()

            if save_video and env.image_recorder is not None:
                raw_images = env.image_recorder.get_images()
                if raw_images:
                    frames_list = list(raw_images.values())
                    frame = np.concatenate(frames_list, axis=1) if len(frames_list) > 1 else frames_list[0]
                    episode_frames.append(frame)

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
            elapsed = timeit.default_timer() - eval_start_time
            fps = total_steps_completed / elapsed if elapsed > 0 else 0.0
            step_bar.set_postfix(fps=f"{fps:.1f}")

            if done:
                force_violation = True
                logger.info(
                    f"Rollout {rollout_id} ended early at step {t + 1} "
                    f"(force/torque limit exceeded)"
                )
                break

        step_bar.close()

        # While still compliant, move to the home Cartesian pose so the robot lifts
        # off the surface before the controller switch (avoids force spike on deactivation)
        env.move_to_home(timeout=5.0)

        # NOW safe to switch — robot is at home and clear of the surface
        env.deactivate_compliance_control()

        steps_taken = t + 1
        total_steps_per_episode.append(steps_taken)
        if force_violation:
            force_violations += 1

        logger.info(
            f"Rollout {rollout_id} complete | steps: {steps_taken} | "
            f"force violation: {force_violation}"
        )

        ft_visualizer.render_now()
        ft_visualizer.save(eval_dir / f"rollout_{rollout_id}.png")
        _, forces_data, _ = ft_visualizer.get_data()
        np.save(eval_dir / f"rollout_{rollout_id}_contact_force.npy", forces_data)
        ft_visualizer.clear()

        if save_video and episode_frames:
            save_to_video(
                episode_frames,
                eval_dir / "videos",
                f"rollout_{rollout_id}.mp4",
                control_frequency,
            )
        if save_video:
            all_rollout_frames.extend(episode_frames)

        violation_pct = f"{force_violations / (rollout_id + 1) * 100:.1f}%"
        rollout_bar.set_postfix(violations=violation_pct)

    env.go_home()
    inference_engine.stop()
    logger.info("Returned to home position after evaluation.")

    logger.info("=" * 60)
    logger.info(f"Rollouts completed:        {num_rollouts}")
    logger.info(
        f"Force violations:          {force_violations} / {num_rollouts} "
        f"({force_violations / num_rollouts * 100:.1f}%)"
    )
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
