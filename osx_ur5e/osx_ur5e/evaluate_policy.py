#!/usr/bin/env python3
"""Evaluate a COMET diffusion policy on the UR5e via FDCCEnv.

Usage:
    ros2 run osx_ur5e evaluate_policy
    ros2 run osx_ur5e evaluate_policy eval.num_rollouts=5 eval.max_timesteps=500

Controls during each rollout:
    Enter  - confirm start of rollout (after reset prompt)
"""

import logging
import signal
import sys
import timeit
from pathlib import Path

import hydra
import numpy as np
import rclpy
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
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
from torchvision import transforms

from comet.common.policies.types import FeatureType
from comet.common.utils.ft_visualizer import FTVisualizer
from comet.common.utils.policy_utils import load_base_policy
from comet.common.utils.viz_utils import save_to_video
from ur_control import transformations

from osx_ur5e.fdcc_env import FDCCEnv
from osx_ur5e.ros_node import RosRuntime

logger = logging.getLogger(__name__)
console = Console()


def _signal_handler(sig, frame):
    logger.info("Interrupted — shutting down.")
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)


def setup_logging(log_file: Path) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in root.handlers[:]:
        if not isinstance(h, logging.FileHandler):
            root.removeHandler(h)
    root.addHandler(RichHandler(console=console, rich_tracebacks=True, show_path=False))
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    )
    root.addHandler(file_handler)


def format_real_robot_observations(arm, image_recorder, features, camera_shape):
    eef = arm.end_effector()
    eef_velocity = arm.end_effector_velocity()

    raw_obs = {
        "observation.qpos": arm.joint_angles(),
        "observation.qvel": arm.joint_velocities(),
        "observation.eef.position": eef[:3],
        "observation.eef.linear_velocity": eef_velocity[:3],
        "observation.eef.angular_velocity": eef_velocity[3:],
        "observation.eef.rotation_ortho6": transformations.ortho6_from_quaternion(eef[3:]),
        "observation.eef.rotation_axis_angle": transformations.axis_angle_from_quaternion(eef[3:]),
        "observation.ft": arm.get_wrench(),
    }

    obs = {}
    for key, value in raw_obs.items():
        if key in features:
            obs[key] = torch.tensor(np.array(value).flatten(), dtype=torch.float32)

    if image_recorder is not None:
        resize_transform = transforms.Resize(camera_shape, antialias=True)
        raw_images = image_recorder.get_images()
        for cam_name, image_hwc in raw_images.items():
            feat_key = f"observation.images.{cam_name}"
            if feat_key in features and image_hwc is not None:
                image_chw = np.ascontiguousarray(np.transpose(image_hwc, (2, 0, 1)))
                image_tensor = torch.tensor(image_chw, dtype=torch.uint8)
                obs[feat_key] = resize_transform(image_tensor)

    return obs


def convert_policy_action(action_dict: dict, actions_as_deltas: bool) -> dict:
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


def run(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir / "evaluation.log")

    seed = cfg.eval.seed
    num_rollouts = cfg.eval.num_rollouts
    max_timesteps = cfg.eval.max_timesteps
    save_video = OmegaConf.select(cfg, "eval.save_video", default=False)
    policy_filename = OmegaConf.select(cfg, "eval.policy_filename", default="best_ema_policy.ckpt")

    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info("Seed: %d", seed)
    logger.info("Results will be saved to: %s", output_dir)

    ckpt_dir = Path(cfg.eval.base.load_ckpt)
    logger.info("Loading policy from: %s", ckpt_dir)
    policy, base_cfg, features = load_base_policy(ckpt_dir, policy_filename)
    policy.cuda()
    policy.eval()

    action_keys = {key: ft.shape for key, ft in features.items() if ft.type is FeatureType.ACTION}
    logger.info("Action keys: %s", list(action_keys.keys()))
    logger.info(
        "Observation keys: %s",
        [k for k, ft in features.items() if ft.type is not FeatureType.ACTION],
    )

    if hasattr(base_cfg, "model_specific") and "camera_shape" in base_cfg.model_specific:
        camera_shape = tuple(base_cfg.model_specific.camera_shape[-2:])
    else:
        camera_shape = (240, 320)
        logger.warning("No camera_shape in checkpoint config, defaulting to %s", camera_shape)

    control_frequency = int(base_cfg.dataset.dataset.fps)
    logger.info("Control frequency: %d Hz", control_frequency)

    use_gazebo_sim = bool(OmegaConf.select(cfg, "use_gazebo_sim", default=False))
    runtime = RosRuntime("evaluate_policy")
    runtime.node.declare_parameter("use_gazebo_sim", use_gazebo_sim)
    # use_real_robot is deliberately left undeclared: URServices probes the UR driver's
    # dashboard client for it, which cannot disagree with reality the way a config value can.
    if use_gazebo_sim:
        runtime.node.declare_parameter("use_sim_time", True)

    try:
        logger.info("ROS node initialized")
        env = FDCCEnv(config=cfg, node=runtime.node, use_torch_for_cameras=False)
        env.reference_trajectory = []

        actions_as_deltas = env.actions_as_deltas
        logger.info("actions_as_deltas: %s", actions_as_deltas)

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

        logger.info("Moving to home position...")
        env.go_home()

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
                if not rclpy.ok():
                    break

                step_task = progress.add_task(
                    make_step_description(rollout_id, total_steps_completed),
                    total=max_timesteps,
                )

                env.reset(move_robot=True)
                progress.stop()
                input(f"\n  Rollout {rollout_id + 1}/{num_rollouts} — press Enter to start...")
                progress.start()

                env.activate_compliance_control()
                policy.reset()

                episode_frames = []
                force_violation = False
                t = 0

                for t in range(max_timesteps):
                    if not rclpy.ok():
                        break

                    obs = format_real_robot_observations(
                        env.arm,
                        env.image_recorder,
                        features,
                        camera_shape,
                    )

                    policy_obs = {
                        k: v.cuda().float().unsqueeze(0) if isinstance(v, torch.Tensor) else v
                        for k, v in obs.items()
                    }

                    with torch.no_grad():
                        action = policy.select_action(policy_obs)

                    action_dict = {
                        k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
                        for k, v in action.items()
                    }
                    env_action = convert_policy_action(action_dict, actions_as_deltas)

                    timestep = env.step(env_action)
                    done = timestep.last()

                    if save_video and env.image_recorder is not None:
                        raw_images = env.image_recorder.get_images()
                        if raw_images:
                            frames_list = [f for f in raw_images.values() if f is not None]
                            if frames_list:
                                frame = (
                                    np.concatenate(frames_list, axis=1)
                                    if len(frames_list) > 1
                                    else frames_list[0]
                                )
                                episode_frames.append(frame)

                    wrench = env.arm.get_wrench()
                    stiffness = None
                    if include_stiffness:
                        if "action.stiffness_diag" in action_dict:
                            val = action_dict["action.stiffness_diag"]
                            stiffness = float(
                                np.mean(val.cpu().numpy() if isinstance(val, torch.Tensor) else val)
                            )
                        elif "action.estimated_stiffness" in action_dict:
                            val = action_dict["action.estimated_stiffness"]
                            stiffness = float(
                                val.cpu().item() if isinstance(val, torch.Tensor) else val
                            )
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
                            "Rollout %d ended early at step %d (force/torque limit exceeded)",
                            rollout_id,
                            t + 1,
                        )
                        break

                env.deactivate_compliance_control()

                steps_taken = t + 1
                total_steps_per_episode.append(steps_taken)
                if force_violation:
                    force_violations += 1

                logger.info(
                    "Rollout %d complete | steps: %d | force violation: %s",
                    rollout_id,
                    steps_taken,
                    force_violation,
                )

                ft_visualizer.render_now()
                ft_visualizer.save(output_dir / f"rollout_{rollout_id}.png")
                _, forces_data, _ = ft_visualizer.get_data()
                np.save(output_dir / f"rollout_{rollout_id}_contact_force.npy", forces_data)
                ft_visualizer.clear()

                if save_video and episode_frames:
                    save_to_video(
                        episode_frames,
                        output_dir / "videos",
                        f"rollout_{rollout_id}.mp4",
                        control_frequency,
                    )
                if save_video:
                    all_rollout_frames.extend(episode_frames)

                progress.remove_task(step_task)
                violation_pct = f"{force_violations / (rollout_id + 1) * 100:.1f}%"
                progress.update(
                    rollout_task,
                    advance=1,
                    description=f"Evaluation | force violations: {violation_pct}",
                )

        logger.info("=" * 60)
        logger.info("Rollouts completed:        %d", num_rollouts)
        logger.info(
            "Force violations:          %d / %d (%.1f%%)",
            force_violations,
            num_rollouts,
            force_violations / num_rollouts * 100,
        )
        logger.info("Mean steps per episode:    %.1f", np.mean(total_steps_per_episode))
        logger.info("Std  steps per episode:    %.1f", np.std(total_steps_per_episode))
        logger.info("=" * 60)

        if save_video and all_rollout_frames:
            save_to_video(
                all_rollout_frames, output_dir, "evaluation_all_rollouts.mp4", control_frequency
            )

        ft_visualizer.close()
        logger.info("Evaluation complete. Results saved to: %s", output_dir)
    finally:
        runtime.shutdown()


@hydra.main(
    version_base=None,
    config_path="../../../../../dependencies/comet/configs",
    config_name="test_wipe_osx",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    if "hydra.run.dir" not in " ".join(sys.argv):
        sys.argv.append("paths.script=/eval")
    main()
