#!/usr/bin/env python3
"""Data collection via Gello teleoperation, saved as a LeRobotDataset.

Usage:
    rosrun osx_ur5e data_collection.py --config path/to/data_collection.yaml
    rosrun osx_ur5e data_collection.py --config path/to/data_collection.yaml \
        --repo-id user/override_name --num-episodes 5

Controls during recording:
    Enter  - end current episode early
    r      - discard and re-record current episode
    q      - stop recording and finalize dataset
"""

import argparse
import logging
import math
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml
import rospy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from osx_gello.gello import Gello
from osx_gym_env.utils import ImageRecorder
from ur_control import transformations
from ur_control.fzi_cartesian_compliance_controller import CompliantController

OBS_STR = "observation"
ACTION = "action"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Override YAML dataset values with any CLI args that were explicitly set."""
    overrides = {
        "repo_id": args.repo_id,
        "task": args.task,
        "root": args.root,
        "fps": args.fps,
        "num_episodes": args.num_episodes,
        "episode_time_s": args.episode_time_s,
        "reset_time_s": args.reset_time_s,
    }
    for key, val in overrides.items():
        if val is not None:
            cfg["dataset"][key] = val
    return cfg


def parse_args():
    p = argparse.ArgumentParser(description="Collect teleoperation data with Gello + UR5e")
    p.add_argument(
        "--config",
        default="config/data_collection.yaml",
        help="Path to the YAML config file relative to osx_ur5e directory",
    )
    # Optional overrides for every dataset field
    p.add_argument("--repo-id",        default=None, help="Override dataset.repo_id")
    p.add_argument("--task",           default=None, help="Override dataset.task")
    p.add_argument("--root",           default=None, help="Override dataset.root")
    p.add_argument("--fps",            default=None, type=int)
    p.add_argument("--num-episodes",   default=None, type=int)
    p.add_argument("--episode-time-s", default=None, type=float)
    p.add_argument("--reset-time-s",   default=None, type=float)
    p.add_argument("--resume", action="store_true", help="Resume recording into an existing dataset")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Feature construction (mirrors comet's build_features)
# ---------------------------------------------------------------------------

def build_features(cfg: dict) -> dict:
    """Build a LeRobotDataset feature dict from YAML cameras/states/actions."""
    features = {}

    for cam_name, cam_info in cfg.get("cameras", {}).items():
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": (cam_info["height"], cam_info["width"], cam_info["channels"]),
            "names": ["height", "width", "channels"],
        }

    for key, shape in cfg.get("states", {}).items():
        features[key] = {
            "dtype": "float32",
            "shape": tuple(shape),
            "names": None,
        }

    for key, shape in cfg.get("actions", {}).items():
        features[key] = {
            "dtype": "float32",
            "shape": tuple(shape),
            "names": None,
        }

    return features


# ---------------------------------------------------------------------------
# Keyboard listener
# ---------------------------------------------------------------------------

def start_keyboard_listener(events):
    """Daemon thread reading stdin to set episode control flags."""

    def _listen():
        while not events["stop"]:
            try:
                line = sys.stdin.readline().strip().lower()
            except EOFError:
                break
            if line == "q":
                events["stop"] = True
                events["exit_early"] = True
            elif line == "r":
                events["rerecord"] = True
                events["exit_early"] = True
            else:
                events["exit_early"] = True

    t = threading.Thread(target=_listen, daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# Robot I/O
# ---------------------------------------------------------------------------

def get_observations(arm, image_recorder):
    eef = arm.end_effector()
    obs = {
        "observation.eef_position":           eef[:3],
        "observation.eef_rotation_ortho6":    transformations.ortho6_from_quaternion(eef[3:]),
        "observation.eef_rotation_axis_angle": transformations.axis_angle_from_quaternion(eef[3:]),
        "observation.eef_wrench":             arm.get_wrench(),
    }
    if image_recorder is not None:
        print("Getting images")
        obs.update(image_recorder.get_images())
    return obs


def set_action(arm: CompliantController, gello: Gello):
    gello_joints = gello.joint_angles()
    current_pose = arm.end_effector()
    target_pose = arm.end_effector(joint_angles=gello_joints)

    arm.set_cartesian_target_pose(pose=target_pose)

    return {
        "action.joint_angles":        gello_joints,
        "action.position":            target_pose[:3],
        "action.rotation_ortho6":     transformations.ortho6_from_quaternion(target_pose[3:]),
        "action.rotation_axis_angle": transformations.axis_angle_from_quaternion(target_pose[3:]),
        "action.delta_position":      target_pose[:3] - current_pose[:3],
        "action.delta_rotation":      transformations.quaternions_orientation_error(target_pose[3:], current_pose[3:]),
    }


# ---------------------------------------------------------------------------
# Record loop
# ---------------------------------------------------------------------------

def record_episode(arm: CompliantController, gello: Gello, image_recorder: ImageRecorder, dataset: LeRobotDataset, fps: int, episode_time_s: float, task: str, events: dict):
    dt = 1.0 / fps
    start_t = time.perf_counter()
    arm.activate_cartesian_controller()

    while time.perf_counter() - start_t < episode_time_s and not rospy.is_shutdown():
        if events["exit_early"]:
            events["exit_early"] = False
            break

        loop_start = time.perf_counter()

        all_values = {}
        all_values.update(get_observations(arm, image_recorder))
        all_values.update(set_action(arm, gello))

        all_values = {k: v.astype(np.float32) for k, v in all_values.items()}

        dataset.add_frame({**all_values, "task": task})

        elapsed = time.perf_counter() - loop_start
        sleep_time = dt - elapsed
        if sleep_time < 0:
            log.warning("Loop running slow: %.1f Hz (target %d Hz)", 1.0 / elapsed, fps)
        else:
            rospy.sleep(sleep_time)

    arm.activate_joint_trajectory_controller()


def wait_for_reset(reset_time_s, events):
    """Countdown during environment reset, interruptible by Enter."""
    start = time.perf_counter()
    while time.perf_counter() - start < reset_time_s:
        if events["exit_early"] or events["stop"] or rospy.is_shutdown():
            events["exit_early"] = False
            break
        remaining = reset_time_s - (time.perf_counter() - start)
        print(f"\r  Reset: {remaining:.0f}s remaining (Enter to skip)  ", end="", flush=True)
        rospy.sleep(0.5)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_config(str(Path(__file__).parent.parent / args.config))
    cfg = apply_overrides(cfg, args)

    ds_cfg = cfg["dataset"]
    features = build_features(cfg)
    num_camera_threads = cfg["image_writer"]["threads_per_camera"] * len(cfg.get("cameras", {}))

    rospy.init_node("data_collection")

    log.info("Initializing hardware...")
    gello = Gello()
    arm = CompliantController(gripper_type=None)
    arm.set_control_mode(cfg["controller"]["mode"])
    arm.update_pd_gains(cfg["controller"]["p_gains"], cfg["controller"]["d_gains"])
    arm.set_position_control_mode(enable=True)
    arm.update_selection_matrix(cfg["controller"]["selection_matrix"])
    arm.set_solver_parameters(error_scale=cfg["controller"]["error_scale"], iterations=cfg["controller"]["iterations"], publish_state_feedback=True)
    arm.auto_switch_controllers = False
    arm.async_mode = True
    arm.zero_ft_sensor()

    image_recorder = (
        ImageRecorder(init_node=False, camera_names=list(cfg["cameras"]))
        if cfg.get("cameras")
        else None
    )

    if args.resume:
        log.info("Resuming dataset: %s", ds_cfg["repo_id"])
        dataset = LeRobotDataset(ds_cfg["repo_id"], root=Path(ds_cfg["root"])/ds_cfg["repo_id"])
        if num_camera_threads:
            dataset.start_image_writer(
                num_processes=cfg["image_writer"]["num_processes"],
                num_threads=num_camera_threads,
            )
    else:
        dataset_dir = Path(ds_cfg["root"])/ds_cfg["repo_id"]
        if dataset_dir.exists():
            confirm = input(f"Dataset directory {dataset_dir} already exists. Do you want to overwrite it? (y/n): ")
            if confirm == "y":
                shutil.rmtree(dataset_dir)
            else:
                log.info("Exiting...")
                sys.exit(1)
        log.info("Creating dataset: %s", ds_cfg["repo_id"])
        dataset = LeRobotDataset.create(
            repo_id=ds_cfg["repo_id"],
            fps=ds_cfg["fps"],
            features=features,
            root=dataset_dir,
            robot_type=ds_cfg.get("robot_type", "ur5e"),
            use_videos=bool(cfg.get("cameras")),
            image_writer_processes=cfg["image_writer"]["num_processes"],
            image_writer_threads=num_camera_threads,
        )

    events = {"exit_early": False, "rerecord": False, "stop": False}
    start_keyboard_listener(events)

    try:
        recorded = 0
        while recorded < ds_cfg["num_episodes"] and not events["stop"] and not rospy.is_shutdown():
            log.info(
                "Episode %d/%d  [Enter=end episode, r=redo, q=quit]",
                recorded + 1,
                ds_cfg["num_episodes"],
            )
            record_episode(
                arm, gello, image_recorder, dataset,
                ds_cfg["fps"], ds_cfg["episode_time_s"], ds_cfg["task"], events,
            )

            if events["rerecord"]:
                log.info("Discarding episode, re-recording...")
                events["rerecord"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            recorded += 1
            log.info("Saved episode %d (%d total in dataset)", recorded, dataset.num_episodes)

            if recorded < ds_cfg["num_episodes"] and not events["stop"]:
                wait_for_reset(ds_cfg["reset_time_s"], events)
    finally:
        log.info("Finalizing dataset...")
        dataset.finalize()
        log.info("Done. %d episodes saved to %s", dataset.num_episodes, dataset.root)


if __name__ == "__main__":
    main()
