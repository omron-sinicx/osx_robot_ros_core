#!/usr/bin/env python3
"""High-frequency raw data collection via Gello teleoperation (Stage 1).

Records every source at its native rate into per-episode rosbags:
    /joint_states + /wrench @ 500 Hz, Gello @ 100 Hz, cameras @ native fps
    (hardware header stamps), plus the stamped action topics published by the
    100 Hz teleop loop (target_frame, gello_joints, stiffness_command).

No LeRobot dataset is written here - run convert_bags_to_lerobot.py on the
session directory to build one at any target fps.

Usage:
    rosrun osx_ur5e data_collection_pro.py
    rosrun osx_ur5e data_collection_pro.py recording.camera_transport=raw
    rosrun osx_ur5e data_collection_pro.py dataset.dataset.num_episodes=5

Controls:
    Enter  - start episode / end current episode (save)
    r      - discard current episode and re-record
    q      - stop recording and finalize session
"""

import datetime
import json
import logging
import re
import subprocess
from pathlib import Path

import hydra
import numpy as np
import rospy
import yaml
from omegaconf import DictConfig, OmegaConf
from pynput import keyboard as kb
from rich.console import Console
from rich.logging import RichHandler

from osx_gello.gello import Gello
from osx_ur5e.bag_recorder import RosbagRecorder
from osx_ur5e.teleop import GelloTeleop
from ur_control.fzi_cartesian_compliance_controller import CompliantController

console = Console()
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Keyboard listener
# ---------------------------------------------------------------------------

def start_keyboard_listener(events: dict):
    """pynput listener: Enter/Space=advance, r=discard+rerecord, q=stop all."""

    def on_press(key):
        try:
            ch = key.char.lower()
            if ch == "q":
                events["stop"] = True
                events["advance"] = True
            elif ch == "r":
                events["rerecord"] = True
                events["advance"] = True
        except AttributeError:
            if key in (kb.Key.enter, kb.Key.space):
                events["advance"] = True

    listener = kb.Listener(on_press=on_press)
    listener.start()
    return listener


def wait_for_advance(events: dict, prompt: str, timeout_s: float = None) -> None:
    """Block until Enter/q (or timeout). Clears the advance flag."""
    events["advance"] = False
    start = rospy.get_time()
    print(f"  {prompt}", flush=True)
    while not events["advance"] and not events["stop"] and not rospy.is_shutdown():
        if timeout_s is not None:
            remaining = timeout_s - (rospy.get_time() - start)
            if remaining <= 0:
                break
            print(f"\r  {prompt} ({remaining:.0f}s)  ", end="", flush=True)
        rospy.sleep(0.1)
    events["advance"] = False
    print()


# ---------------------------------------------------------------------------
# Topic resolution / session setup
# ---------------------------------------------------------------------------

def resolve_camera_topics(camera_names, transport: str, wait_s: float = 5.0):
    """Per-camera (image_topic, camera_info_topic), RealSense vs usb_cam layout.

    Raw per-camera topics only (no /sync namespace): true hardware stamps.
    """
    topics = {}
    deadline = rospy.get_time() + wait_s
    remaining = list(camera_names)
    while remaining and not rospy.is_shutdown():
        published = {t for t, _ in rospy.get_published_topics()}
        for cam in list(remaining):
            for base, info in ((f"/{cam}/color/image_raw", f"/{cam}/color/camera_info"),
                               (f"/{cam}/image_raw", f"/{cam}/camera_info")):
                if base in published:
                    image_topic = base + ("/compressed" if transport == "compressed" else "")
                    topics[cam] = (image_topic, info)
                    remaining.remove(cam)
                    break
        if not remaining or rospy.get_time() >= deadline:
            break
        rospy.sleep(0.2)

    for cam in remaining:
        base = f"/{cam}/color/image_raw"
        topics[cam] = (base + ("/compressed" if transport == "compressed" else ""),
                       f"/{cam}/color/camera_info")
        log.warning("Camera %s not seen on the graph; defaulting to %s", cam, topics[cam][0])
    return topics


def check_clock_sanity(topics_to_check: dict, max_skew_s: float) -> None:
    """Warn when a source's header stamps deviate from ROS time.

    Guards against e.g. RealSense global_time_enabled misconfiguration before
    an entire session is recorded on a skewed clock.
    """
    import rostopic

    for name, topic in topics_to_check.items():
        msg_class, _, _ = rostopic.get_topic_class(topic, blocking=False)
        if msg_class is None:
            log.warning("Clock check: %s (%s) not available, skipping", name, topic)
            continue
        try:
            msg = rospy.wait_for_message(topic, msg_class, timeout=3.0)
        except rospy.ROSException:
            log.warning("Clock check: no message on %s within 3s", topic)
            continue
        header = getattr(msg, "header", None)
        if header is None:
            continue
        skew = abs(rospy.get_time() - header.stamp.to_sec())
        if skew > max_skew_s:
            log.error("Clock check FAILED for %s: |stamp - now| = %.3fs (max %.3fs). "
                      "Recorded data would be time-skewed!", name, skew, max_skew_s)
        else:
            log.info("Clock check OK for %s: skew %.1f ms", name, skew * 1e3)


def write_session_meta(session_dir: Path, cfg: DictConfig, topics, camera_topics) -> None:
    """Sidecar with everything Stage 2 needs to be self-contained."""
    meta = {
        "created": datetime.datetime.now().isoformat(),
        "task": cfg.dataset.dataset.task,
        "topics": topics,
        "camera_topics": {cam: list(t) for cam, t in camera_topics.items()},
        "cameras": OmegaConf.to_container(cfg.dataset.cameras, resolve=True),
        "robot_description": rospy.get_param("/robot_description", None),
        "gello_calibration_file": rospy.get_param("calibration_file", None),
        "stamp_offset_s": {},  # per-camera latency corrections (GoPro etc.)
    }
    try:
        meta["git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd="/root/osx-ur", text=True).strip()
    except Exception:
        meta["git_sha"] = None
    if meta["gello_calibration_file"]:
        try:
            meta["gello_calibration"] = yaml.safe_load(open(meta["gello_calibration_file"]))
        except Exception:
            meta["gello_calibration"] = None

    with open(session_dir / "session_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(session_dir / "config.yaml", "w") as f:
        yaml.safe_dump(OmegaConf.to_container(cfg, resolve=True), f)


# ---------------------------------------------------------------------------
# Safety supervision
# ---------------------------------------------------------------------------

def ft_violation(arm, safety_cfg) -> bool:
    wrench = arm.get_wrench()
    force_norm = np.linalg.norm(wrench[:3])
    torque_norm = np.linalg.norm(wrench[3:])
    if force_norm > safety_cfg.max_force_torque[0] or torque_norm > safety_cfg.max_force_torque[1]:
        log.warning("Force/torque too high: %.2f/%.2f", force_norm, torque_norm)
        return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="/root/osx-ur/dependencies/comet/configs",
            config_name="book_flipping",
            version_base=None)
def main(cfg: DictConfig) -> None:
    rec_cfg = cfg.recording
    ds_cfg = cfg.dataset
    controller_cfg = cfg.controller
    safety_cfg = controller_cfg.safety_parameters

    rospy.init_node("data_collection_pro")

    log.setLevel(logging.INFO)
    log.propagate = False
    if not any(isinstance(h, RichHandler) for h in log.handlers):
        log.addHandler(RichHandler(console=console, show_time=True, show_path=False, markup=True))

    # -- hardware ----------------------------------------------------------
    log.info("Initializing hardware...")
    gello = Gello()
    arm = CompliantController(gripper_type=None)
    arm.set_control_mode(controller_cfg.mode)
    arm.update_pd_gains(OmegaConf.to_container(controller_cfg.p_gains),
                        OmegaConf.to_container(controller_cfg.d_gains))
    arm.update_selection_matrix(OmegaConf.to_container(controller_cfg.selection_matrix))
    arm.set_solver_parameters(error_scale=controller_cfg.error_scale,
                              iterations=controller_cfg.iterations,
                              publish_state_feedback=True)
    arm.auto_switch_controllers = False
    arm.async_mode = True
    arm.zero_ft_sensor()

    teleop = GelloTeleop(arm, gello, safety_cfg, rate_hz=rec_cfg.teleop_rate_hz)
    teleop.start()
    # Publishes the initial (latched) stiffness command so bags always have one.
    teleop.set_stiffness(controller_cfg.stiffness * np.ones(6))

    # -- topics + session --------------------------------------------------
    camera_topics = resolve_camera_topics(list(ds_cfg.cameras or []), rec_cfg.camera_transport)
    topics = list(rec_cfg.state_topics) + list(rec_cfg.extra_topics)
    for image_topic, info_topic in camera_topics.values():
        topics += [image_topic, info_topic]

    check_clock_sanity(
        {"joint_states": "/joint_states",
         **{cam: t[0].replace("/compressed", "") for cam, t in camera_topics.items()}},
        rec_cfg.clock_check.max_skew_s,
    )

    task_slug = re.sub(r"[^a-z0-9]+", "_", ds_cfg.dataset.task.lower()).strip("_")
    session_dir = Path(rec_cfg.output_dir) / \
        f"{task_slug}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir.mkdir(parents=True, exist_ok=True)
    write_session_meta(session_dir, cfg, topics, camera_topics)
    log.info("Session: %s", session_dir)
    log.info("Recording %d topics", len(topics))

    recorder = RosbagRecorder(topics, buffsize_mb=rec_cfg.buffsize_mb)
    events = {"advance": False, "rerecord": False, "stop": False}
    start_keyboard_listener(events)

    num_episodes = ds_cfg.dataset.num_episodes
    episode_time_s = ds_cfg.dataset.episode_time_s
    reset_time_s = ds_cfg.dataset.reset_time_s
    saved = 0
    episode_idx = 0

    try:
        while saved < num_episodes and not events["stop"] and not rospy.is_shutdown():
            wait_for_advance(events, f"Press Enter to start episode {saved + 1}/{num_episodes} "
                                     "(q to quit)...")
            if events["stop"] or rospy.is_shutdown():
                break

            # -- ARM: move to start, engage compliance, teleop live --------
            teleop.pause()
            arm.dashboard_services.activate_ros_control_on_ur()
            arm.activate_joint_trajectory_controller()
            arm.set_joint_positions(positions=controller_cfg.init_qpos, target_time=2.0, wait=True)
            arm.zero_ft_sensor()
            arm.activate_cartesian_controller()
            teleop.resume()

            # -- RECORDING --------------------------------------------------
            episode_dir = session_dir / f"episode_{episode_idx:06d}"
            recorder.start(episode_dir, start_timeout_s=rec_cfg.episode_start_timeout_s)
            log.info("Episode %d recording  [Enter=save, r=discard, q=save+quit]", saved + 1)

            events["advance"] = False
            events["rerecord"] = False
            t_start = rospy.get_time()
            supervise = rospy.Rate(50)
            ft_tripped = False
            while not rospy.is_shutdown():
                if events["advance"] or events["stop"]:
                    break
                if rospy.get_time() - t_start >= episode_time_s:
                    log.info("Episode time limit reached (%.0fs)", episode_time_s)
                    break
                if ft_violation(arm, safety_cfg):
                    ft_tripped = True
                    break
                supervise.sleep()
            events["advance"] = False

            # -- SAVE / DISCARD ----------------------------------------------
            if ft_tripped or events["rerecord"]:
                teleop.pause()  # hold pose on violation
                reason = "ft_violation" if ft_tripped else "user"
                recorder.discard(reason=reason)
                events["rerecord"] = False
                log.info("Episode discarded (%s), re-recording", reason)
                if ft_tripped:
                    arm.activate_joint_trajectory_controller()
            else:
                meta = recorder.stop(task=ds_cfg.dataset.task,
                                     extra_meta={"episode_index": episode_idx})
                saved += 1
                episode_idx += 1
                log.info("Saved episode %d/%d (%.1fs, %.1f MB)", saved, num_episodes,
                         meta.get("duration_s") or 0.0,
                         meta.get("bag_size_bytes", 0) / 1e6)

            # -- RESET (teleop stays live so the scene can be re-staged) ----
            if saved < num_episodes and not events["stop"]:
                if not teleop.active:
                    arm.zero_ft_sensor()
                    arm.activate_cartesian_controller()
                    teleop.resume()
                wait_for_advance(events, "Reset the scene - Enter to skip countdown",
                                 timeout_s=reset_time_s)
    finally:
        log.info("Finalizing session...")
        teleop.shutdown()
        if recorder.recording:
            recorder.discard(reason="shutdown")
        try:
            arm.activate_joint_trajectory_controller()
        except Exception:
            pass

        episodes = sorted(p.name for p in session_dir.glob("episode_*") if p.is_dir())
        manifest = {
            "episodes": episodes,
            "num_saved": len(episodes),
            "discarded": sorted(p.name for p in (session_dir / ".discarded").glob("episode_*"))
            if (session_dir / ".discarded").exists() else [],
            "disk_usage_bytes": sum(f.stat().st_size for f in session_dir.rglob("*") if f.is_file()),
        }
        with open(session_dir / "session_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        log.info("Done. %d episodes in %s", len(episodes), session_dir)


if __name__ == "__main__":
    main()
