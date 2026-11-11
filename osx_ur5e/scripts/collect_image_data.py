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

Each saved demo also gets a calibration.npy: live intrinsics/distortion per
camera (from its camera_info topic) merged with extrinsics from --calib_path
(no live ROS topic for hand-eye extrinsics), same {'int','ext','dist_coeff'}
layout as calib/calib.npy.

Usage:
    rosrun osx_ur5e collect_image_data.py --task_name pick_cup
    rosrun osx_ur5e collect_image_data.py --task_name pick_cup --depth
    rosrun osx_ur5e collect_image_data.py --task_name pick_cup --calib_path /path/to/calib.npy
    rosrun osx_ur5e collect_image_data.py --task_name pick_cup --cam_namespaces front_camera

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
import numpy as np
import pandas as pd
import rospy
import yaml

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "collect_image_data_config.yaml"
)

# Extrinsics have no live ROS topic (they come from the offline hand-eye solve
# in osx_camera_calibration), so this is the only source for them. Intrinsics
# are read live from each camera's camera_info topic instead of from here,
# since that's always in sync with whatever firmware/resolution is actually
# streaming.
DEFAULT_CALIB_PATH = "/root/osx-ur/dependencies/Point-Policy/calib/calib.npy"

# cam_1/cam_2 identity must match calib/calib.npy (see collect_data_config.yaml
# and collect_image_data_config.yaml for the full explanation): cam_1 =
# wrist_camera, cam_2 = front_camera. Fixed by namespace, not list position,
# so temporarily dropping one camera (e.g. --cam_namespaces front_camera)
# doesn't relabel the other.
NAMESPACE_TO_CAM_ID = {"wrist_camera": 1, "front_camera": 2}


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
        from sensor_msgs.msg import CameraInfo, Image

        self.namespace = namespace
        self.collect_depth = collect_depth
        self.color_image = None
        self.color_timestamp = None
        self.depth_image = None
        self.camera_info = None

        rospy.Subscriber(f"/{namespace}/color/image_raw", Image, self._color_cb)
        rospy.Subscriber(f"/{namespace}/color/camera_info", CameraInfo, self._info_cb)
        if collect_depth:
            rospy.Subscriber(
                f"/{namespace}/aligned_depth_to_color/image_raw", Image, self._depth_cb
            )

    def _color_cb(self, msg):
        self.color_image = _imgmsg_to_bgr8(msg)
        self.color_timestamp = msg.header.stamp.to_sec() * 1000.0  # ms

    def _depth_cb(self, msg):
        # Keep the raw uint16 depth exactly as published (no unit conversion,
        # no hole filling, no normalization) so saved depth matches the
        # camera's original values bit-for-bit.
        self.depth_image = _imgmsg_to_depth_mono16(msg)

    def _info_cb(self, msg):
        self.camera_info = msg

    def ready(self) -> bool:
        if self.color_image is None:
            return False
        if self.camera_info is None:
            return False
        if self.collect_depth and self.depth_image is None:
            return False
        return True

    def intrinsics(self) -> dict:
        """Live camera_info -> calib.npy-style {'int', 'dist_coeff'} dict."""
        msg = self.camera_info
        return {
            "int": np.array(msg.K, dtype=np.float64).reshape(3, 3),
            "dist_coeff": np.array(msg.D, dtype=np.float64).reshape(1, -1),
            "width": msg.width,
            "height": msg.height,
            "distortion_model": msg.distortion_model,
        }


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
    p.add_argument("--depth", action="store_true", help="Enable depth recording (off by default)")
    p.add_argument("--fps", type=int, default=None)
    p.add_argument(
        "--calib_path", type=str, default=None,
        help="calib.npy providing per-camera extrinsics (and fallback intrinsics). "
             f"Defaults to {DEFAULT_CALIB_PATH}."
    )
    p.add_argument(
        "--cam_namespaces", type=str, default=None,
        help="Comma-separated subset of camera namespaces to record this run, e.g. "
             "'front_camera' to temporarily skip the wrist camera. Defaults to the "
             "config's full cam_namespaces list."
    )
    args = p.parse_args(rospy.myargv()[1:])

    cfg = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    for key, cfg_key in [
        ("task_name", "task_name"), ("data_dir", "data_dir"), ("fps", "fps"),
        ("calib_path", "calib_path"),
    ]:
        if getattr(args, key) is None and cfg.get(cfg_key) is not None:
            setattr(args, key, cfg[cfg_key])
    if args.calib_path is None:
        args.calib_path = DEFAULT_CALIB_PATH

    if args.cam_namespaces is not None:
        args.cam_namespaces = [ns.strip() for ns in args.cam_namespaces.split(",")]
    else:
        args.cam_namespaces = cfg.get("cam_namespaces", ["wrist_camera", "front_camera"])

    unknown = [ns for ns in args.cam_namespaces if ns not in NAMESPACE_TO_CAM_ID]
    if unknown:
        p.error(f"Unknown camera namespace(s) {unknown}; expected one of {list(NAMESPACE_TO_CAM_ID)}")
    args.cam_ids = [NAMESPACE_TO_CAM_ID[ns] for ns in args.cam_namespaces]

    if args.fps is None:
        args.fps = 30

    if not args.task_name:
        p.error("--task_name is required (pass on the CLI or set it in the config file)")
    if not args.data_dir:
        p.error("--data_dir is required (pass on the CLI or set it in the config file)")

    args.collect_depth = args.depth
    return args


def build_calibration(cameras, cam_ids, calib_path: str) -> dict:
    """Per-camera {'int', 'ext', 'dist_coeff'} dict, calib.npy-compatible.

    Intrinsics/distortion come live from each camera's camera_info topic
    (always in sync with whatever is actually streaming); extrinsics come
    from calib_path since hand-eye calibration has no live ROS topic.
    """
    calib_npy = {}
    if Path(calib_path).exists():
        calib_npy = np.load(calib_path, allow_pickle=True).item()
    else:
        rospy.logwarn(
            f"calib_path '{calib_path}' not found — saved calibration will have "
            "no extrinsics ('ext')."
        )

    calibration = {}
    for cid, cam in zip(cam_ids, cameras):
        key = f"cam_{cid}"
        entry = cam.intrinsics()
        if key in calib_npy and "ext" in calib_npy[key]:
            entry["ext"] = calib_npy[key]["ext"]
        calibration[key] = entry
    return calibration


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
              rec_rgb, rec_ts, rec_depth, collect_depth, fps, calibration):
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

    np.save(demo_dir / "calibration.npy", calibration, allow_pickle=True)
    write_states_csv(demo_dir / "states.csv", rec_ts[cam_ids[0]][:n_frames], n_frames)
    print(f"[Demo {demo_id}] Saved {n_frames} frames to {demo_dir}")


# ---------------------------------------------------------------------------
# Main loop — continuous OpenCV event loop, state machine
# ---------------------------------------------------------------------------
def run(cameras, cam_ids, task_dir, collect_depth, fps, calibration):
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
                          rec_rgb, rec_ts, rec_depth, collect_depth, fps, calibration)
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
    for cid, ns in zip(args.cam_ids, args.cam_namespaces):
        print(f"Subscribed to camera '{ns}' (cam_{cid}).")

    print("Waiting for cameras...")
    deadline = rospy.Time.now() + rospy.Duration(10.0)
    while not all(cam.ready() for cam in cameras) and rospy.Time.now() < deadline:
        rospy.sleep(0.1)
    if not all(cam.ready() for cam in cameras):
        rospy.logerr("Timed out waiting for camera topics. Is camera_bringup.launch running?")
        return
    print("All cameras ready.")

    calibration = build_calibration(cameras, args.cam_ids, args.calib_path)
    run(cameras, args.cam_ids, task_dir, args.collect_depth, args.fps, calibration)


if __name__ == "__main__":
    main()
