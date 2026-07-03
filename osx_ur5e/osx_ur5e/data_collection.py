#!/usr/bin/env python3
"""Data collection via Gello teleoperation, saved as a LeRobotDataset.

Usage:
    ros2 run osx_ur5e data_collection
    ros2 run osx_ur5e data_collection dataset.repo_id=user/my_dataset dataset.num_episodes=5

Controls during recording:
    Enter  - end current episode early
    r      - discard and re-record current episode
    q      - stop recording and finalize dataset
"""

import logging
import math
import shutil
import sys
import time
import yaml
from pathlib import Path

import hydra
import numpy as np
import rclpy
from omegaconf import DictConfig, OmegaConf
from pynput import keyboard as kb
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from ur_control import transformations
from ur_control.fzi_cartesian_compliance_controller import CompliantController

from osx_ur5e.image_recorder import ImageRecorder
from osx_ur5e.ros_node import RosRuntime

console = Console()
log = logging.getLogger(__name__)


def build_features(cfg: DictConfig) -> dict:
    features = {}

    for cam_name, cam_info in cfg.cameras.items():
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": (cam_info.height, cam_info.width, cam_info.channels),
            "names": ["height", "width", "channels"],
        }

    for key, shape in cfg.states.items():
        features[key] = {
            "dtype": "float32",
            "shape": tuple(shape),
            "names": None,
        }

    for key, shape in cfg.actions.items():
        features[key] = {
            "dtype": "float32",
            "shape": tuple(shape),
            "names": None,
        }

    return features


def start_keyboard_listener(events: dict):
    def on_press(key):
        try:
            ch = key.char.lower()
            if ch == "q":
                events["stop"] = True
                events["exit_early"] = True
            elif ch == "r":
                events["rerecord"] = True
                events["exit_early"] = True
        except AttributeError:
            if key in (kb.Key.enter, kb.Key.space):
                events["exit_early"] = True

    listener = kb.Listener(on_press=on_press)
    listener.start()
    return listener


def get_observations(arm: CompliantController, image_recorder: ImageRecorder) -> dict:
    eef = arm.end_effector()
    eef_velocity = arm.end_effector_velocity()
    obs = {
        "observation.qpos": arm.joint_angles(),
        "observation.qvel": arm.joint_velocities(),
        "observation.eef.position": eef[:3],
        "observation.eef.linear_velocity": eef_velocity[:3],
        "observation.eef.angular_velocity": eef_velocity[3:],
        "observation.eef.rotation_ortho6": transformations.ortho6_from_quaternion(eef[3:]),
        "observation.eef.rotation_axis_angle": transformations.axis_angle_from_quaternion(eef[3:]),
        "observation.ft": arm.get_wrench(),
    }
    if image_recorder is not None:
        obs.update(
            {f"observation.images.{k}": v for k, v in image_recorder.get_images().items()}
        )
    return obs


def set_action(arm: CompliantController, gello, safety_cfg: DictConfig) -> dict:
    gello_joints = gello.joint_angles()
    current_pose = arm.end_effector()
    target_pose = arm.end_effector(joint_angles=gello_joints)
    stiffness_diag = arm.current_stiffness
    delta_translation = target_pose[:3] - current_pose[:3]
    delta_rotation = transformations.quaternions_orientation_error(target_pose[3:], current_pose[3:])

    max_delta_rotation = np.deg2rad(safety_cfg.max_delta_rotation)
    clipped_delta_translation = np.clip(
        delta_translation,
        -safety_cfg.max_delta_translation,
        safety_cfg.max_delta_translation,
    )
    clipped_delta_orientation = np.clip(
        delta_rotation, -max_delta_rotation, max_delta_rotation
    )

    next_position = current_pose[:3] + clipped_delta_translation
    next_position[0] = np.clip(
        next_position[0], safety_cfg.workspace_range.x[0], safety_cfg.workspace_range.x[1]
    )
    next_position[1] = np.clip(
        next_position[1], safety_cfg.workspace_range.y[0], safety_cfg.workspace_range.y[1]
    )
    next_position[2] = np.clip(
        next_position[2], safety_cfg.workspace_range.z[0], safety_cfg.workspace_range.z[1]
    )
    next_orientation = transformations.rotate_quaternion_by_rpy(
        *clipped_delta_orientation, current_pose[3:]
    )
    next_target = np.concatenate([next_position, next_orientation])

    arm.set_cartesian_target_pose(pose=next_target)

    return {
        "action.joint": gello_joints,
        "action.position": next_position,
        "action.rotation_ortho6": transformations.ortho6_from_quaternion(next_orientation),
        "action.rotation_axis_angle": transformations.axis_angle_from_quaternion(next_orientation),
        "action.stiffness_diag": stiffness_diag,
        "action.delta_position": clipped_delta_translation,
        "action.delta_rotation": clipped_delta_orientation,
    }


def record_episode(arm, gello, image_recorder, dataset, cfg, events) -> None:
    ds_cfg = cfg.dataset
    safety_cfg = cfg.controller.safety_parameters
    dt = 1.0 / ds_cfg.fps
    total_steps = math.ceil(ds_cfg.episode_time_s * ds_cfg.fps)

    arm.zero_ft_sensor()
    arm.activate_cartesian_controller()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Episode[/bold cyan]"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("/"),
        TimeRemainingColumn(),
        TextColumn("[dim]fps:{task.fields[hz]:.0f}[/dim]"),
        console=console,
        refresh_per_second=10,
    )

    with progress:
        task_id = progress.add_task("record", total=total_steps, hz=ds_cfg.fps)
        start_t = time.perf_counter()

        while time.perf_counter() - start_t < ds_cfg.episode_time_s and rclpy.ok():
            if events["exit_early"] or events["stop"]:
                events["exit_early"] = False
                break

            force_norm = np.linalg.norm(arm.get_wrench())
            torque_norm = np.linalg.norm(arm.get_wrench()[3:])
            if (
                force_norm > safety_cfg.max_force_torque[0]
                or torque_norm > safety_cfg.max_force_torque[1]
            ):
                log.warning("Force/torque norm is too high: %.2f/%.2f", force_norm, torque_norm)
                events["exit_early"] = True
                events["rerecord"] = True
                break

            loop_start = time.perf_counter()

            all_values = {}
            all_values.update(get_observations(arm, image_recorder))
            all_values.update(set_action(arm, gello, safety_cfg))
            all_values = {k: v.astype(np.float32) for k, v in all_values.items()}

            dataset.add_frame({**all_values, "task": ds_cfg.task})

            elapsed = time.perf_counter() - loop_start
            sleep_time = dt - elapsed
            if sleep_time < 0:
                log.warning("Loop running slow: %.1f Hz (target %d Hz)", 1.0 / elapsed, ds_cfg.fps)
            else:
                time.sleep(sleep_time)

            actual_hz = 1.0 / max(time.perf_counter() - loop_start, 1e-6)
            progress.update(task_id, advance=1, hz=actual_hz)

    arm.activate_joint_trajectory_controller()


def wait_for_reset(reset_time_s: float, events: dict) -> None:
    start = time.perf_counter()
    while time.perf_counter() - start < reset_time_s:
        if events["exit_early"] or events["stop"] or not rclpy.ok():
            events["exit_early"] = False
            break
        remaining = reset_time_s - (time.perf_counter() - start)
        print(f"\r  Reset: {remaining:.0f}s remaining (Enter to skip)  ", end="", flush=True)
        time.sleep(0.5)
    print()


def wait_for_keypress_reset(events: dict) -> None:
    events["exit_early"] = False
    print("  Press Enter to start next episode...", flush=True)
    while not events["exit_early"] and not events["stop"] and rclpy.ok():
        time.sleep(0.1)
    events["exit_early"] = False
    print()


def run(cfg: DictConfig) -> None:
    ds_cfg = cfg.dataset
    controller_cfg = cfg.controller
    features = build_features(ds_cfg)
    num_camera_threads = ds_cfg.image_writer.threads_per_camera * len(ds_cfg.cameras)
    repo_id = ds_cfg.repo_id[0]
    dataset_dir = Path(ds_cfg.dir) / repo_id

    log.setLevel(logging.INFO)
    log.propagate = False
    if not any(isinstance(h, RichHandler) for h in log.handlers):
        log.addHandler(
            RichHandler(console=console, show_time=True, show_path=False, markup=True)
        )

    use_gazebo_sim = bool(OmegaConf.select(cfg, "use_gazebo_sim", default=False))
    runtime = RosRuntime("data_collection")
    runtime.node.declare_parameter("use_gazebo_sim", use_gazebo_sim)
    runtime.node.declare_parameter("use_real_robot", not use_gazebo_sim)
    if use_gazebo_sim:
        runtime.node.declare_parameter("use_sim_time", True)

    try:
        log.info("Initializing hardware...")
        from osx_gello.gello import Gello

        gello = Gello()
        arm = CompliantController(node=runtime.node, gripper_type=None)
        arm.set_control_mode(controller_cfg.mode)
        arm.update_pd_gains(
            OmegaConf.to_container(controller_cfg.p_gains),
            OmegaConf.to_container(controller_cfg.d_gains),
        )
        arm.update_selection_matrix(OmegaConf.to_container(controller_cfg.selection_matrix))
        arm.set_solver_parameters(
            error_scale=controller_cfg.error_scale,
            iterations=controller_cfg.iterations,
            publish_state_feedback=True,
        )
        arm.update_stiffness(controller_cfg.stiffness * np.ones(6))
        arm.current_stiffness = np.array(controller_cfg.stiffness * np.ones(6))
        arm.auto_switch_controllers = False
        arm.async_mode = True
        arm.zero_ft_sensor()

        image_recorder = (
            ImageRecorder(node=runtime.node, camera_names=list(ds_cfg.cameras))
            if ds_cfg.cameras
            else None
        )

        if ds_cfg.overwrite or not dataset_dir.exists():
            if dataset_dir.exists() and dataset_dir.is_dir():
                confirm = input(
                    f"Dataset directory {dataset_dir} already exists. Overwrite? (y/n): "
                )
                if confirm.strip().lower() != "y":
                    log.info("Exiting...")
                    sys.exit(1)
                shutil.rmtree(dataset_dir)

            log.info("Creating dataset: %s", repo_id)
            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=ds_cfg.fps,
                features=features,
                root=dataset_dir,
                robot_type=ds_cfg.robot_type,
                use_videos=bool(ds_cfg.cameras),
                image_writer_processes=ds_cfg.image_writer.num_processes,
                image_writer_threads=num_camera_threads,
            )

            config_path = dataset_dir / "meta" / "hydra_config.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as f:
                yaml.safe_dump(OmegaConf.to_container(cfg, resolve=True), f)
        else:
            log.info("Resuming dataset: %s", repo_id)
            dataset = LeRobotDataset(repo_id, root=dataset_dir)
            if num_camera_threads:
                dataset.start_image_writer(
                    num_processes=ds_cfg.image_writer.num_processes,
                    num_threads=num_camera_threads,
                )

        events = {"exit_early": False, "rerecord": False, "stop": False}
        start_keyboard_listener(events)

        wait_for_keypress_reset(events)
        recorded = 0
        while recorded < ds_cfg.num_episodes and not events["stop"] and rclpy.ok():
            log.info(
                "Episode %d/%d  [Enter=end episode, r=redo, q=quit]",
                recorded + 1,
                ds_cfg.num_episodes,
            )
            record_episode(arm, gello, image_recorder, dataset, cfg, events)

            if events["stop"]:
                log.info("Stopping recording...")
                dataset.clear_episode_buffer()
                break

            if events["rerecord"]:
                log.info("Discarding episode, re-recording...")
                events["rerecord"] = False
                dataset.clear_episode_buffer()
                wait_for_keypress_reset(events)
                continue

            dataset.save_episode()
            recorded += 1
            log.info("Saved episode %d (%d total in dataset)", recorded, dataset.num_episodes)

            if recorded < ds_cfg.num_episodes and not events["stop"]:
                wait_for_keypress_reset(events)

        log.info("Finalizing dataset...")
        dataset.finalize()
        log.info("Done. %d episodes saved to %s", dataset.num_episodes, dataset.root)
    finally:
        runtime.shutdown()


@hydra.main(config_path="../../config/hydra", config_name="test_task", version_base=None)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
