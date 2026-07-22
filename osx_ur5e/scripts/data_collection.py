#!/usr/bin/env python3
"""Data collection via Gello teleoperation, saved as a LeRobotDataset.

Uses Hydra for configuration management.

Usage:
    rosrun osx_ur5e data_collection.py
    rosrun osx_ur5e data_collection.py dataset.repo_id=user/my_dataset dataset.num_episodes=5
    rosrun osx_ur5e data_collection.py dataset.overwrite=true

Controls during recording:
    Enter  - end current episode early
    r      - discard and re-record current episode
    q      - stop recording and finalize dataset
"""

import logging
import math
import os
import select
import shutil
import signal
import sys
import termios
import threading
import time
import tty
import yaml
from pathlib import Path

import hydra
import numpy as np
import rospy
from omegaconf import DictConfig, OmegaConf
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
from osx_gello.gello import Gello
from osx_gym_env.utils import ImageRecorder
from osx_claw.claw_controller import ClawController
from bendlabs.bendlabs_recorder import BendLabsRecorder
from ur_control import transformations
from ur_control.fzi_cartesian_compliance_controller import CompliantController

console = Console()
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gripper utilities
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_features(cfg: DictConfig) -> dict:
    """Build a LeRobotDataset feature dict from Hydra cameras/states/actions."""
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

    # gripper is included as the last element of observation.state and action

    if cfg.get("use_bendlabs", False):
        n = cfg.get("num_bendlabs_sensors", 4)
        features["observation.bendlabs"] = {
            "dtype": "float32", "shape": (n * 2,), "names": None}

    return features


# ---------------------------------------------------------------------------
# Keyboard listener
# ---------------------------------------------------------------------------

def start_keyboard_listener(events: dict) -> threading.Thread:
    """Start a background thread that reads keypresses directly from stdin.

    Keys:
        Enter / Space  - end current episode early (save it)
        r              - discard episode and re-record
        q              - stop all recording
    """
    try:
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
    except (termios.error, AttributeError):
        log.warning("stdin is not a tty; keyboard listener disabled.")
        return threading.Thread(target=lambda: None, daemon=True)

    def _reader():
        try:
            while not events.get("_stop_listener"):
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not ready:
                    continue
                ch = sys.stdin.read(1).lower()
                if ch == "q":
                    events["stop"] = True
                    events["exit_early"] = True
                elif ch == "r":
                    events["rerecord"] = True
                    events["exit_early"] = True
                elif ch in ("\r", "\n", " "):
                    events["exit_early"] = True
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# Robot I/O
# ---------------------------------------------------------------------------

def get_observations(arm: CompliantController, image_recorder: ImageRecorder,
                     claw: ClawController = None,
                     bendlabs: BendLabsRecorder = None) -> dict:
    arm_qpos = arm.joint_angles()
    obs = {
        "observation.state": (
            np.append(arm_qpos, claw.get_normalized_position())
            if claw is not None else arm_qpos
        ),
        "observation.ft": arm.get_wrench(),
    }
    if bendlabs is not None:
        obs["observation.bendlabs"] = bendlabs.get_angles()
    if image_recorder is not None:
        images = image_recorder.get_images()
        if any(v is None for v in images.values()):
            raise RuntimeError(
                "One or more camera images are unavailable; cannot record frame.")
        obs.update({f"observation.images.{k}": v for k, v in images.items()})
    return obs


def set_action(arm: CompliantController, gello: Gello, safety_cfg: DictConfig,
               claw: ClawController = None) -> dict:
    gello_joints = gello.joint_angles()
    current_pose = arm.end_effector()
    target_pose = arm.end_effector(joint_angles=gello_joints)
    delta_translation = target_pose[:3] - current_pose[:3]
    delta_rotation = transformations.quaternions_orientation_error(
        target_pose[3:], current_pose[3:])

    max_delta_rotation = np.deg2rad(safety_cfg.max_delta_rotation)
    clipped_delta_translation = np.clip(
        delta_translation, -safety_cfg.max_delta_translation, safety_cfg.max_delta_translation)
    clipped_delta_orientation = np.clip(
        delta_rotation, -max_delta_rotation, max_delta_rotation)

    next_position = current_pose[:3] + clipped_delta_translation
    next_position[0] = np.clip(
        next_position[0], safety_cfg.workspace_range.x[0], safety_cfg.workspace_range.x[1])
    next_position[1] = np.clip(
        next_position[1], safety_cfg.workspace_range.y[0], safety_cfg.workspace_range.y[1])
    next_position[2] = np.clip(
        next_position[2], safety_cfg.workspace_range.z[0], safety_cfg.workspace_range.z[1])
    next_orientation = transformations.rotate_quaternion_by_rpy(
        *clipped_delta_orientation, current_pose[3:])
    next_target = np.concatenate([next_position, next_orientation])

    arm.set_cartesian_target_pose(pose=next_target)

    if claw is not None:
        norm_gripper = gello.gripper_position()
        claw.set_normalized_position(norm_gripper)
        action_joints = np.append(gello_joints, norm_gripper)
    else:
        action_joints = gello_joints

    return {"action": action_joints}


# ---------------------------------------------------------------------------
# Record loop
# ---------------------------------------------------------------------------

def record_episode(
    arm: CompliantController,
    gello: Gello,
    image_recorder: ImageRecorder,
    dataset: LeRobotDataset,
    cfg: DictConfig,
    events: dict,
    claw: ClawController = None,
    bendlabs: BendLabsRecorder = None,
) -> None:
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

        while time.perf_counter() - start_t < ds_cfg.episode_time_s and not rospy.is_shutdown():
            if events["exit_early"] or events["stop"]:
                events["exit_early"] = False
                break

            force_norm = np.linalg.norm(arm.get_wrench())
            torque_norm = np.linalg.norm(arm.get_wrench()[3:])
            if force_norm > safety_cfg.max_force_torque[0] or torque_norm > safety_cfg.max_force_torque[1]:
                log.warning("Force/torque norm is too high: %.2f/%.2f",
                            force_norm, torque_norm)
                events["exit_early"] = True
                events["rerecord"] = True
                break

            loop_start = time.perf_counter()

            all_values = {}
            all_values.update(get_observations(
                arm, image_recorder, claw, bendlabs))
            all_values.update(set_action(arm, gello, safety_cfg, claw))
            all_values = {k: np.asarray(v).astype(np.float32)
                          for k, v in all_values.items()}

            dataset.add_frame({**all_values, "task": ds_cfg.task})

            elapsed = time.perf_counter() - loop_start
            sleep_time = dt - elapsed
            if sleep_time < 0:
                log.warning("Loop running slow: %.1f Hz (target %d Hz)",
                            1.0 / elapsed, ds_cfg.fps)
            else:
                rospy.sleep(sleep_time)

            actual_hz = 1.0 / max(time.perf_counter() - loop_start, 1e-6)
            progress.update(task_id, advance=1, hz=actual_hz)

    arm.activate_joint_trajectory_controller()


def wait_for_reset(reset_time_s: float, events: dict) -> None:
    """Countdown during environment reset, interruptible by Enter."""
    start = time.perf_counter()
    while time.perf_counter() - start < reset_time_s:
        if events["exit_early"] or events["stop"] or rospy.is_shutdown():
            events["exit_early"] = False
            break
        remaining = reset_time_s - (time.perf_counter() - start)
        print(
            f"\r  Reset: {remaining:.0f}s remaining (Enter to skip)  ", end="", flush=True)
        rospy.sleep(0.5)
    print()


def wait_for_keypress_reset(events: dict) -> None:
    """Block until Enter/Space is pressed to start the next episode."""
    events["exit_early"] = False
    print("  Press Enter to start next episode...", flush=True)
    while not events["exit_early"] and not events["stop"] and not rospy.is_shutdown():
        rospy.sleep(0.1)
    events["exit_early"] = False
    print()


def move_to_home(arm, cfg, claw=None) -> None:
    """Drive the arm (and gripper) to a fixed home configuration via the joint
    trajectory controller. Runs OUTSIDE the recorded episode, so the homing
    motion is never added to the dataset.
    """
    ds_cfg = cfg.dataset
    home = ds_cfg.get("home_position", None)
    if home is None:
        log.warning("dataset.home_position not set; skipping arm homing. "
                    "Add a 6-element joint config to enable consistent episode starts.")
        return
    home = np.array(home, dtype=float)

    # Optional small randomization around home for robustness to imperfect resets.
    jitter = float(ds_cfg.get("home_randomization", 0.0))  # radians, per joint
    if jitter > 0.0:
        home = home + np.random.uniform(-jitter, jitter, size=home.shape)

    target_time = float(ds_cfg.get("home_target_time_s", 4.0))
    log.info("Homing arm to %s (%.1fs)", home, target_time)

    # auto_switch is off, so do it explicitly
    arm.activate_joint_trajectory_controller()
    arm.set_joint_positions(positions=home, target_time=target_time, wait=True)

    if claw is not None:
        gripper_home = float(ds_cfg.get("gripper_home", 0.0))
        claw.set_normalized_position(float(np.clip(gripper_home, 0.0, 1.0)))


def teleop_to_start(arm, gello, claw, cfg, events) -> None:
    """Non-recording GELLO teleop loop used as an alternative to auto-homing.

    The user drives the arm to a desired start position and presses Enter to confirm.
    """
    safety_cfg = cfg.controller.safety_parameters
    dt = 1.0 / cfg.dataset.fps

    arm.activate_cartesian_controller()
    print("  Arm homed. TELEOP active — drive to desired start position, "
          "then press Enter to begin recording.", flush=True)

    events["exit_early"] = False
    while not events["exit_early"] and not events["stop"] and not rospy.is_shutdown():
        loop_start = time.perf_counter()
        set_action(arm, gello, safety_cfg, claw)
        elapsed = time.perf_counter() - loop_start
        rospy.sleep(max(dt - elapsed, 0.0))
    events["exit_early"] = False
    print()

    arm.activate_joint_trajectory_controller()


def reset_environment(arm, gello, claw, cfg, events) -> None:
    ds_cfg = cfg.dataset

    move_to_home(arm, cfg, claw)
    if events["stop"] or rospy.is_shutdown():
        return

    home = ds_cfg.get("home_position", None)
    if home is not None:
        try:
            home_arr = np.array(home, dtype=float)
            gello_arr = np.array(gello.joint_angles(), dtype=float)
            err = np.abs(gello_arr - home_arr)
            print(f"  Arm homed. GELLO offset from home (rad): {err}")
        except Exception as e:
            log.debug("GELLO alignment readout skipped: %s", e)

    if ds_cfg.get("teleop_reset", False):
        print("  Reset the scene, then press Enter to begin teleop adjustment...",
              flush=True)
        events["exit_early"] = False
        while not events["exit_early"] and not events["stop"] and not rospy.is_shutdown():
            rospy.sleep(0.1)
        events["exit_early"] = False
        print()
        if events["stop"] or rospy.is_shutdown():
            return
        teleop_to_start(arm, gello, claw, cfg, events)
        return

    print("  Reset the scene and align the GELLO to the arm, "
          "then press Enter to start recording...", flush=True)

    events["exit_early"] = False
    while not events["exit_early"] and not events["stop"] and not rospy.is_shutdown():
        rospy.sleep(0.1)
    events["exit_early"] = False
    print()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(config_path="../config/hydra",
            config_name="test_task",
            version_base=None)
def main(cfg: DictConfig) -> None:

    ds_cfg = cfg.dataset
    controller_cfg = cfg.controller
    features = build_features(ds_cfg)
    num_camera_threads = ds_cfg.image_writer.threads_per_camera * \
        len(ds_cfg.cameras)
    repo_id = ds_cfg.repo_id[0]
    dataset_dir = Path(ds_cfg.dir) / repo_id

    rospy.init_node("data_collection")

    # rospy.init_node() rewires the root logger; set up Rich on our logger explicitly after it.
    log.setLevel(logging.INFO)
    log.propagate = False
    if not any(isinstance(h, RichHandler) for h in log.handlers):
        log.addHandler(RichHandler(console=console,
                       show_time=True, show_path=False, markup=True))

    log.info("Initializing hardware...")
    gello = Gello()
    arm = CompliantController(gripper_type=None)
    arm.set_control_mode(controller_cfg.mode)
    arm.update_pd_gains(OmegaConf.to_container(
        controller_cfg.p_gains), OmegaConf.to_container(controller_cfg.d_gains))
    arm.update_selection_matrix(
        OmegaConf.to_container(controller_cfg.selection_matrix))
    arm.set_solver_parameters(error_scale=controller_cfg.error_scale,
                              iterations=controller_cfg.iterations, publish_state_feedback=True)
    arm.update_stiffness(controller_cfg.stiffness * np.ones(6))
    arm.current_stiffness = np.array(controller_cfg.stiffness * np.ones(6))
    arm.auto_switch_controllers = False
    arm.async_mode = True
    arm.zero_ft_sensor()

    claw = None
    if ds_cfg.get("use_gripper", False):
        log.info("Initializing ClawController...")
        claw = ClawController(init_node=False)

    bendlabs = None
    if ds_cfg.get("use_bendlabs", False):
        log.info("Initializing BendLabsRecorder...")
        bendlabs = BendLabsRecorder(
            init_node=False,
            num_sensors=ds_cfg.get("num_bendlabs_sensors", 4),
        )
        log.info("Waiting for BendLabs sensors...")
        deadline = time.perf_counter() + 10.0
        while not bendlabs.sensors_ready() and not rospy.is_shutdown():
            if time.perf_counter() > deadline:
                log.error(
                    "Timed out waiting for BendLabs sensors. Check that uart_bridge_node is running.")
                sys.exit(1)
            rospy.sleep(0.1)
        log.info("BendLabs sensors ready.")

    image_recorder = (
        ImageRecorder(init_node=False, camera_names=list(ds_cfg.cameras))
        if ds_cfg.cameras
        else None
    )

    if image_recorder is not None:
        log.info("Waiting for cameras...")
        deadline = time.perf_counter() + 10.0
        while not image_recorder.cameras_ready() and not rospy.is_shutdown():
            if time.perf_counter() > deadline:
                log.error(
                    "Timed out waiting for camera images. Check that camera topics are publishing.")
                sys.exit(1)
            rospy.sleep(0.1)
        log.info("All cameras ready.")

    meta_complete = (dataset_dir / "meta" / "tasks.parquet").exists()

    if ds_cfg.overwrite or not dataset_dir.exists() or not meta_complete:
        if dataset_dir.exists() and dataset_dir.is_dir():
            if not meta_complete:
                log.warning(
                    "Incomplete dataset found at %s, recreating.", dataset_dir)
            else:
                confirm = input(
                    f"Dataset directory {dataset_dir} already exists. Overwrite? (y/n): ")
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

    def _sigint_handler(_sig, _frame):
        events["stop"] = True
        events["exit_early"] = True

    signal.signal(signal.SIGINT, _sigint_handler)

    listener = start_keyboard_listener(events)

    reset_environment(arm, gello, claw, cfg, events)
    try:
        recorded = 0
        while recorded < ds_cfg.num_episodes and not events["stop"] and not rospy.is_shutdown():
            log.info(
                "Episode %d/%d  [Enter=end episode, r=redo, q=quit]",
                recorded + 1,
                ds_cfg.num_episodes,
            )
            record_episode(arm, gello, image_recorder, dataset,
                           cfg, events, claw, bendlabs)

            if events["stop"]:
                log.info("Stopping recording...")
                dataset.clear_episode_buffer()
                for cam_key in dataset.meta.camera_keys:
                    img_dir = dataset._get_image_file_dir(
                        dataset.episode_buffer["episode_index"], cam_key
                    )
                    if img_dir.exists() and any(img_dir.iterdir()):
                        log.error(
                            "Leftover images after clear_episode_buffer in %s!", img_dir)
                break

            if events["rerecord"]:
                log.info("Discarding episode, re-recording...")
                events["rerecord"] = False
                dataset.clear_episode_buffer()
                for cam_key in dataset.meta.camera_keys:
                    img_dir = dataset._get_image_file_dir(
                        dataset.episode_buffer["episode_index"], cam_key
                    )
                    if img_dir.exists() and any(img_dir.iterdir()):
                        log.error(
                            "Leftover images after clear_episode_buffer in %s!", img_dir)
                reset_environment(arm, gello, claw, cfg, events)
                continue

            dataset.save_episode()
            recorded += 1
            log.info("Saved episode %d (%d total in dataset)",
                     recorded, dataset.num_episodes)

            if recorded < ds_cfg.num_episodes and not events["stop"]:
                reset_environment(arm, gello, claw, cfg, events)
    finally:
        events["_stop_listener"] = True
        listener.join(timeout=0.2)
        log.info("Finalizing dataset...")
        dataset.finalize()
        log.info("Done. %d episodes saved to %s",
                 dataset.num_episodes, dataset.root)
        rospy.signal_shutdown("Recording complete")
        os._exit(0)


if __name__ == "__main__":
    main()
