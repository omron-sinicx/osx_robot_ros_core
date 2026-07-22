#!/usr/bin/env python3
"""
ROS image-only data collection for the Point-Policy ur5e_pipeline.

Collapses collect_data_realsense.py + process_data_human.py into a single
script: this repo already runs synchronized RealSense nodes (see
camera_bringup.launch) and subscribes to them via rospy, so cameras are
captured in lockstep (one frame per camera per loop tick) instead of being
recorded independently and reconciled by timestamp afterward. Output is
written directly in the final processed_data/ layout that
convert_to_pkl_human.py consumes — no extracted_data/ raw stage, no jpg
extraction, no ffmpeg re-encode pass.

Prerequisite: camera_bringup.launch (or equivalent) must already be running
so that /wrist_camera/... and /front_camera/... topics are publishing.

Usage:
    rosrun osx_ur5e collect_image_data.py --task_name pick_cup
    rosrun osx_ur5e collect_image_data.py --task_name pick_cup --no_depth

Controls (OpenCV window must have focus):
    SPACE / ENTER  — start recording a new demo
    ENTER          — stop and SAVE the current demo
    r              — discard current demo and immediately re-record (same demo ID)
    q              — discard current demo (if recording) and quit
"""

import argparse
import pickle as pkl
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import rospy
import yaml

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "collect_image_data_config.yaml"
)

# cam_1/cam_2 identity must match calib/calib.npy (see collect_data_config.yaml
# and collect_image_data_config.yaml for the full explanation). Index 0 -> cam_1,
# index 1 -> cam_2.
CAM_IDS = [1, 2]


# ---------------------------------------------------------------------------
# Depth utilities (ported verbatim from ur5e_pipeline/collect_data_realsense.py)
# ---------------------------------------------------------------------------
def _fill_depth_holes(depth_m: np.ndarray, max_hole_radius: int = 4) -> np.ndarray:
    """Fill small depth holes (0.0 pixels) via dilation of valid neighbours."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * max_hole_radius + 1, 2 * max_hole_radius + 1)
    )
    dilated = cv2.dilate(depth_m, kernel)
    return np.where(depth_m == 0.0, dilated, depth_m)


# ---------------------------------------------------------------------------
# Manual sensor_msgs/Image decoding.
#
# Deliberately not using cv_bridge here: its compiled boost module is built
# against the NumPy 1.x C API and segfaults/raises under NumPy 2.x (the only
# numpy available in this environment). The encodings realsense2_camera
# publishes on the topics we use (rgb8/bgr8 color, 16UC1 aligned depth) are
# trivial to decode by hand, so we just reshape the raw bytes instead.
#
# Frames are returned in BGR channel order (matching what cv2.VideoWriter /
# cv2.VideoCapture produce), since convert_to_pkl_human.py reads the saved
# mp4s with cv2.VideoCapture and explicitly flips BGR->RGB itself.
# ---------------------------------------------------------------------------
def _imgmsg_to_bgr8(msg) -> np.ndarray:
    assert msg.encoding in ("rgb8", "bgr8"), f"expected rgb8/bgr8, got {msg.encoding}"
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    arr = arr[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
    return arr[:, :, ::-1] if msg.encoding == "rgb8" else arr


def _imgmsg_to_depth_mono16(msg) -> np.ndarray:
    assert msg.encoding == "16UC1", f"expected 16UC1, got {msg.encoding}"
    dtype = np.dtype(np.uint16).newbyteorder(">" if msg.is_bigendian else "<")
    arr = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.step // 2)
    return arr[:, : msg.width]


# ---------------------------------------------------------------------------
# ROS camera subscriber: latest color + depth frame for one camera namespace
# ---------------------------------------------------------------------------
class RosCamera:
    def __init__(self, namespace: str, collect_depth: bool):
        from sensor_msgs.msg import Image

        self.namespace = namespace
        self.collect_depth = collect_depth
        self.color_image = None
        self.color_timestamp = None
        self.depth_image = None

        rospy.Subscriber(f"/{namespace}/color/image_raw", Image, self._color_cb)
        if collect_depth:
            rospy.Subscriber(
                f"/{namespace}/aligned_depth_to_color/image_raw", Image, self._depth_cb
            )

    def _color_cb(self, msg):
        self.color_image = _imgmsg_to_bgr8(msg)
        self.color_timestamp = msg.header.stamp.to_sec() * 1000.0  # ms

    def _depth_cb(self, msg):
        depth_raw = _imgmsg_to_depth_mono16(msg)
        depth_m = depth_raw.astype(np.float32) / 1000.0
        self.depth_image = _fill_depth_holes(depth_m)

    def ready(self) -> bool:
        if self.color_image is None:
            return False
        if self.collect_depth and self.depth_image is None:
            return False
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Collect image-only demo data via ROS-synchronized RealSense cameras"
    )
    p.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG_PATH),
        help="YAML file with task_name/data_dir/cam_namespaces/fps. "
             "Any of the flags below, if passed, override the config value."
    )
    p.add_argument("--task_name", default=None, help="Task name (used as folder name)")
    p.add_argument("--data_dir", default=None, help="Root data directory")
    p.add_argument("--no_depth", action="store_true", help="Disable depth recording")
    p.add_argument("--fps", type=int, default=None)
    args = p.parse_args(rospy.myargv()[1:])

    cfg = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    for key, cfg_key in [
        ("task_name", "task_name"), ("data_dir", "data_dir"), ("fps", "fps"),
    ]:
        if getattr(args, key) is None and cfg.get(cfg_key) is not None:
            setattr(args, key, cfg[cfg_key])

    args.cam_namespaces = cfg.get("cam_namespaces", ["wrist_camera", "front_camera"])
    if args.fps is None:
        args.fps = 30

    if not args.task_name:
        p.error("--task_name is required (pass on the CLI or set it in the config file)")
    if not args.data_dir:
        p.error("--data_dir is required (pass on the CLI or set it in the config file)")

    args.collect_depth = not args.no_depth
    return args


def get_next_demo_id(task_dir: Path) -> int:
    existing = [
        d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith("demonstration_")
    ]
    if not existing:
        return 0
    return max(int(d.name.split("_")[-1]) for d in existing) + 1


def write_video(path: Path, frames, fps: int):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def write_depth_video(path: Path, depth_frames, fps: int):
    """Colorized magma_r visualization, matching process_data_human.py's depth output."""
    colormap = matplotlib.colormaps["magma_r"]
    colorized = []
    for frame in depth_frames:
        min_v, max_v = frame.min(), frame.max()
        norm = (frame - min_v) / (max_v - min_v) if max_v > min_v else np.zeros_like(frame)
        rgb = colormap(norm, bytes=True)[:, :, :3]
        colorized.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    write_video(path, colorized, fps)


def write_states_csv(path: Path, timestamps, n_frames: int):
    """Dummy all-zero state, one row per captured frame (image-only collection).

    pose_aa must be formatted as the literal "[np.float64(...), ...]" string
    shape so convert_to_pkl_human.py's extract_number regex can parse it.
    """
    dummy_pose_aa = "[" + ", ".join(["np.float64(0.0)"] * 6) + "]"
    rows = {
        "created timestamp": timestamps,
        "pose_aa": [dummy_pose_aa] * n_frames,
        "gripper_state": [0.0] * n_frames,
        "desired_gripper_state": [0.0] * n_frames,
    }
    pd.DataFrame(rows).to_csv(path, index=False)


def save_demo(task_dir: Path, demo_id: int, cameras, cam_ids,
              rec_rgb, rec_ts, rec_depth, collect_depth, fps):
    demo_dir = task_dir / f"demonstration_{demo_id}"
    demo_dir.mkdir(parents=True, exist_ok=True)

    video_dir = demo_dir / "videos"
    video_dir.mkdir(exist_ok=True)
    n_frames = min(len(rec_rgb[cid]) for cid in cam_ids)
    for cid in cam_ids:
        write_video(video_dir / f"camera{cid}.mp4", rec_rgb[cid][:n_frames], fps)

    if collect_depth:
        depth_dir = demo_dir / "depth"
        depth_dir.mkdir(exist_ok=True)
        for cid in cam_ids:
            frames = rec_depth[cid][:n_frames]
            with open(depth_dir / f"depth{cid}.pkl", "wb") as f:
                pkl.dump(frames, f)
            write_depth_video(depth_dir / f"depth{cid}.mp4", frames, fps)

    write_states_csv(demo_dir / "states.csv", rec_ts[cam_ids[0]][:n_frames], n_frames)
    print(f"[Demo {demo_id}] Saved {n_frames} frames to {demo_dir}")


# ---------------------------------------------------------------------------
# Main loop — continuous OpenCV event loop, state machine
# ---------------------------------------------------------------------------
def run(cameras, cam_ids, task_dir, collect_depth, fps):
    """
    State machine:
        IDLE      — live feed; SPACE/ENTER starts recording
        RECORDING — live feed + REC overlay; ENTER saves, R re-records, Q quits
    """
    IDLE, RECORDING = "idle", "recording"
    state = IDLE
    demo_id = None

    rec_rgb = {cid: [] for cid in cam_ids}
    rec_ts = {cid: [] for cid in cam_ids}
    rec_depth = {cid: [] for cid in cam_ids}

    rate = rospy.Rate(fps)
    print("\nSPACE/ENTER=start recording   Q=quit")

    while not rospy.is_shutdown():
        for cid, cam in zip(cam_ids, cameras):
            frame = cam.color_image
            if frame is None:
                continue
            disp = frame.copy()
            if state == RECORDING:
                cv2.circle(disp, (20, 20), 10, (0, 0, 255), -1)
                cv2.putText(disp, f"REC  {len(rec_rgb[cid])} frames", (35, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(disp, "ENTER=save   R=re-record   Q=quit",
                            (10, disp.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(disp, "SPACE / ENTER = start recording   Q = quit",
                            (10, disp.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow(f"Camera {cid}", disp)

        key = cv2.waitKey(1) & 0xFF

        if state == RECORDING:
            for cid, cam in zip(cam_ids, cameras):
                if cam.color_image is None:
                    continue
                rec_rgb[cid].append(cam.color_image.copy())
                rec_ts[cid].append(cam.color_timestamp)
                if collect_depth and cam.depth_image is not None:
                    rec_depth[cid].append(cam.depth_image.copy())

        if state == IDLE:
            if key in (ord(" "), 13, 10):       # SPACE or ENTER -> start recording
                demo_id = get_next_demo_id(task_dir)
                for cid in cam_ids:
                    rec_rgb[cid].clear(); rec_ts[cid].clear(); rec_depth[cid].clear()
                state = RECORDING
                print(f"\n[Demo {demo_id}] Recording...  ENTER=save  R=re-record  Q=quit")
            elif key == ord("q"):
                break

        elif state == RECORDING:
            if key in (13, 10):                 # ENTER -> save
                print(f"[Demo {demo_id}] Saving...")
                save_demo(task_dir, demo_id, cameras, cam_ids,
                          rec_rgb, rec_ts, rec_depth, collect_depth, fps)
                state = IDLE
                print("SPACE/ENTER=next demo   Q=quit")

            elif key == ord("r"):               # R -> discard + re-record same ID
                n = min(len(rec_rgb[cid]) for cid in cam_ids)
                print(f"[Demo {demo_id}] Discarded ({n} frames). Re-recording...")
                for cid in cam_ids:
                    rec_rgb[cid].clear(); rec_ts[cid].clear(); rec_depth[cid].clear()

            elif key == ord("q"):               # Q -> discard + quit
                print(f"[Demo {demo_id}] Discarded. Quitting.")
                break

        rate.sleep()

    cv2.destroyAllWindows()


def main():
    args = parse_args()
    rospy.init_node("collect_image_data", anonymous=True)

    task_dir = Path(args.data_dir) / "processed_data" / args.task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    cameras = [
        RosCamera(namespace, args.collect_depth) for namespace in args.cam_namespaces
    ]
    for cid, ns in zip(CAM_IDS, args.cam_namespaces):
        print(f"Subscribed to camera '{ns}' (cam_{cid}).")

    print("Waiting for cameras...")
    deadline = rospy.Time.now() + rospy.Duration(10.0)
    while not all(cam.ready() for cam in cameras) and rospy.Time.now() < deadline:
        rospy.sleep(0.1)
    if not all(cam.ready() for cam in cameras):
        rospy.logerr("Timed out waiting for camera topics. Is camera_bringup.launch running?")
        return
    print("All cameras ready.")

    run(cameras, CAM_IDS, task_dir, args.collect_depth, args.fps)


if __name__ == "__main__":
    main()
