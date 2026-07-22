#!/usr/bin/env python3
"""Standalone ROS node: DIFT localization + causal TAPIR point tracking.

Owns all of the DIFT/TAPIR/PointsClass state that evaluate_policy_points.py
used to run in-process (in a background thread). Runs independently so it
can be started/stopped/monitored separately from the evaluation node, and so
its tracking-visualization image can be watched live (e.g. via
rqt_image_view) regardless of what the evaluation node is doing.

cam_1 = wrist_camera, cam_2 = front_camera -- must match the assignment
calib.npy was calibrated with (see collect_data_config.yaml). Do not reorder.

Interface (all under this node's private namespace, e.g. /tracking_node/...):
    Services:
        ~reset_and_track (std_srvs/Trigger)
            Re-runs DIFT localization on the current camera frames and
            (re)starts publishing tracked points. Called by a client (e.g.
            evaluate_policy_points.py) at the start of every rollout, since
            the tracked object may have moved between episodes.
        ~stop_track (std_srvs/Trigger)
            Stops publishing tracked points (pauses tracking) until the next
            ~reset_and_track call.
    Publishers:
        ~environment_state (std_msgs/Float64MultiArray)
            Flattened (N*3,) world-frame tracked-point vector, published at
            ~tracking_fps while active. No header/stamp -- subscribers should
            timestamp their own receipt time for staleness checks (same
            pattern RawImageRecorder already uses for camera frames).
        ~visualization_image (sensor_msgs/Image, bgr8)
            The two-camera panel with tracked point(s) overlaid, published
            every tracking tick while active, and once immediately inside
            ~reset_and_track (so a caller doesn't have to wait a full period
            for a fresh localization preview).
    Params:
        ~num_points (int, set once at startup)
            Number of tracked points N, so a client can sanity-check its
            policy's expected observation.environment_state dimension
            (N*3) before running.

Usage:
    rosrun osx_ur5e tracking_node.py \
        --task_name pick_place_00 \
        --calib_path /root/osx-ur/dependencies/Point-Policy/calib/calib.npy
"""

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

import rospy
from rich.console import Console
from rich.logging import RichHandler
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger, TriggerResponse

# Point-Policy's point_utils (DIFT + TAPIR tracking) and franka/utils.py
# (triangulate_points) -- not installed as packages, so add their
# directories to sys.path, matching evaluate_policy_points.py's convention.
POINT_POLICY_ROOT = Path("/root/osx-ur/dependencies/Point-Policy/point_policy")
sys.path.insert(0, str(POINT_POLICY_ROOT))
sys.path.insert(0, str(POINT_POLICY_ROOT / "robot_utils" / "franka"))
from point_utils.points_class import PointsClass  # noqa: E402
from utils import triangulate_points  # noqa: E402 -- franka/utils.py

logger = logging.getLogger(__name__)
console = Console()

# cam_1/cam_2 identity must match calib/calib.npy (see collect_data_config.yaml).
CAM1_TOPIC = "wrist_camera"   # cam_1
CAM2_TOPIC = "front_camera"  # cam_2
PIXEL_KEY_1 = "pixels1"
PIXEL_KEY_2 = "pixels2"


def _signal_handler(sig, frame):
    logger.info("Interrupted — shutting down.")
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)


class RawImageRecorder:
    """Subscribes directly to /{cam}/color/image_raw and decodes the raw
    sensor_msgs/Image manually -- cv_bridge is broken under NumPy 2.x in
    this environment (confirmed elsewhere this session), so this bypasses
    it entirely. Identical to evaluate_policy_points.py's class of the same
    name."""

    def __init__(self, camera_names):
        self.camera_names = camera_names
        self._images = {name: None for name in camera_names}
        self._timestamps = {name: None for name in camera_names}
        for name in camera_names:
            rospy.Subscriber(f"/{name}/color/image_raw", Image, self._callback, callback_args=name)

    def _callback(self, msg, cam_name):
        assert msg.encoding == "rgb8", f"expected rgb8 on /{cam_name}/color/image_raw, got {msg.encoding}"
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        arr = arr[:, : msg.width * 3].reshape(msg.height, msg.width, 3).copy()
        self._images[cam_name] = arr
        self._timestamps[cam_name] = rospy.get_time()

    def cameras_ready(self) -> bool:
        return all(self._images[name] is not None for name in self.camera_names)

    def get_images(self) -> dict:
        result = {}
        for name in self.camera_names:
            img = self._images[name]
            if img is None:
                rospy.logerr_throttle(1, f"No image received yet for {name}")
                result[name] = None
            elif rospy.get_time() - self._timestamps[name] > 0.5:
                rospy.logerr_throttle(1, f"Image for {name} is too old! ignoring")
                result[name] = None
            else:
                result[name] = img
        return result


def setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(RichHandler(console=console, rich_tracebacks=True, show_path=False))


def load_points_class(task_name: str) -> PointsClass:
    """Same construction as evaluate_policy_points.py's function of the same
    name: object_labels=["objects"] only (no hand to track on a live robot),
    pixel_keys fixed to both live cameras."""
    cfg_path = POINT_POLICY_ROOT / "cfgs" / "suite" / "points_cfg.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    root_dir = cfg["root_dir"]
    cfg["dift_path"] = f"{root_dir}/{cfg['dift_path']}"
    cfg["cotracker_checkpoint"] = f"{root_dir}/{cfg['cotracker_checkpoint']}"
    cfg["tapir_checkpoint"] = f"{root_dir}/{cfg['tapir_checkpoint']}"
    cfg["task_name"] = task_name
    cfg["pixel_keys"] = [PIXEL_KEY_1, PIXEL_KEY_2]
    cfg["object_labels"] = ["objects"]
    cfg["use_gt_depth"] = False

    return PointsClass(**cfg)


def build_projection_matrices(calib: dict) -> dict:
    """P = K_padded @ ext per camera, cached once."""
    P = {}
    for cam_name in ("cam_1", "cam_2"):
        intr = calib[cam_name]["int"]
        extr = calib[cam_name]["ext"]
        intr_padded = np.concatenate([intr, np.zeros((3, 1))], axis=1)
        P[cam_name] = intr_padded @ extr
    return P


def undistort_points(pts_xy: np.ndarray, calib_cam: dict) -> np.ndarray:
    """(N,2) raw pixel coords -> (N,2) undistorted pixel coords."""
    K = calib_cam["int"]
    D = calib_cam["dist_coeff"]
    return cv2.undistortPoints(
        pts_xy.reshape(-1, 1, 2).astype(np.float32), K, D, P=K
    ).reshape(-1, 2)


def localize_and_init_tracking(points_class: PointsClass, image_recorder: RawImageRecorder):
    """One-time-per-rollout: DIFT-localize the object keypoint(s) on the
    current frame and initialize the causal tracker's queries from it.
    Identical to evaluate_policy_points.py's function of the same name."""
    images = image_recorder.get_images()
    frame1, frame2 = images[CAM1_TOPIC], images[CAM2_TOPIC]

    points_class.add_to_image_list(frame1, PIXEL_KEY_1)
    points_class.add_to_image_list(frame2, PIXEL_KEY_2)
    points_class.find_semantic_similar_points(PIXEL_KEY_1, "objects")
    points_class.find_semantic_similar_points(PIXEL_KEY_2, "objects")
    points_class.track_points(PIXEL_KEY_1, is_first_step=True)
    points_class.track_points(PIXEL_KEY_2, is_first_step=True)

    # semantic_similar_points is (N,3): [label, x, y], same layout label_points.py saves.
    pts1_xy = points_class.semantic_similar_points[f"{PIXEL_KEY_1}_objects"][:, 1:3].cpu().numpy()
    pts2_xy = points_class.semantic_similar_points[f"{PIXEL_KEY_2}_objects"][:, 1:3].cpu().numpy()
    return frame1, frame2, pts1_xy, pts2_xy


def draw_tracked_points(frame_rgb: np.ndarray, pts_xy: np.ndarray, radius: int = 6) -> np.ndarray:
    """frame_rgb: HxWx3 uint8 RGB. pts_xy: (N,2). Returns a BGR image with
    each point circled and index-labeled. Identical to
    evaluate_policy_points.py's function of the same name."""
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    for i, (x, y) in enumerate(pts_xy):
        x, y = int(round(float(x))), int(round(float(y)))
        cv2.circle(frame_bgr, (x, y), radius, (0, 0, 255), -1)
        cv2.circle(frame_bgr, (x, y), radius + 1, (255, 255, 255), 1)
        cv2.putText(frame_bgr, str(i), (x + radius + 2, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return frame_bgr


def triangulate_world_points(pts1_xy: np.ndarray, pts2_xy: np.ndarray, calib: dict, P: dict) -> np.ndarray:
    """(N,2) native-resolution pixel coords per camera -> (N*3,) flattened
    world-frame 3D points. Identical to evaluate_policy_points.py's function
    of the same name."""
    pts1_undist = undistort_points(pts1_xy, calib["cam_1"])
    pts2_undist = undistort_points(pts2_xy, calib["cam_2"])
    world_pts = triangulate_points([P["cam_1"], P["cam_2"]], [pts1_undist, pts2_undist])
    return world_pts[:, :3].reshape(-1).astype(np.float64)  # (N*3,)


def get_live_environment_state(
    points_class: PointsClass, image_recorder: RawImageRecorder, calib: dict, P: dict,
):
    """One live frame -> ((N*3,) world-frame tracked-point vector, combined
    BGR visualization panel). Identical tracking logic to
    evaluate_policy_points.py's function of the same name, but also returns
    the annotated frames so the caller can publish them."""
    images = image_recorder.get_images()
    frame1, frame2 = images[CAM1_TOPIC], images[CAM2_TOPIC]

    points_class.add_to_image_list(frame1, PIXEL_KEY_1)
    points_class.add_to_image_list(frame2, PIXEL_KEY_2)
    points_class.track_points(PIXEL_KEY_1, last_n_frames=1)
    points_class.track_points(PIXEL_KEY_2, last_n_frames=1)

    pts1 = points_class.get_points_on_image(PIXEL_KEY_1, last_n_frames=1)[0].numpy()  # (N, 2)
    pts2 = points_class.get_points_on_image(PIXEL_KEY_2, last_n_frames=1)[0].numpy()  # (N, 2)

    env_state = triangulate_world_points(pts1, pts2, calib, P)

    panel1 = draw_tracked_points(frame1, pts1)
    panel2 = draw_tracked_points(frame2, pts2)
    combined = np.hstack([panel1, panel2])

    return env_state, combined


def build_image_msg(frame_bgr: np.ndarray) -> Image:
    """Manually build a sensor_msgs/Image (bgr8) -- the mirror-image of
    RawImageRecorder's manual decode, since cv_bridge is unusable here."""
    msg = Image()
    msg.header.stamp = rospy.Time.now()
    msg.height, msg.width = frame_bgr.shape[:2]
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(frame_bgr).tobytes()
    return msg


class TrackingNode:
    """Owns PointsClass/calib/cameras and publishes tracked points + a
    visualization image at ~tracking_fps, gated on/off via two services.

    A single rospy.Timer (created once, for the node's whole lifetime) is
    the only thing that ever calls into TAPIR -- this is deliberate, not
    incidental: torch.backends.cudnn.benchmark=True benchmarks convolution
    algorithms per OS thread, so a freshly-created thread pays a multi-second
    warm-up the first time it calls predict() (confirmed on hardware this
    session: ~2.9s on a new thread's first call vs. ~90ms after warm-up).
    Using one persistent Timer thread for the whole process means that cost
    is paid at most once, ever -- not once per rollout.

    A lock serializes all PointsClass access, since the service callbacks
    (~reset_and_track, on rospy's service-handling thread) and the Timer
    callback (on the Timer's own thread) would otherwise race on the same
    PointsClass/TapirTracker state.
    """

    def __init__(self, points_class: PointsClass, image_recorder: RawImageRecorder,
                 calib: dict, P: dict, target_hz: float):
        self.points_class = points_class
        self.image_recorder = image_recorder
        self.calib = calib
        self.P = P

        self.active = False
        self._lock = threading.Lock()

        self.env_state_pub = rospy.Publisher("~environment_state", Float64MultiArray, queue_size=1)
        self.viz_pub = rospy.Publisher("~visualization_image", Image, queue_size=1)

        rospy.Service("~reset_and_track", Trigger, self._handle_reset_and_track)
        rospy.Service("~stop_track", Trigger, self._handle_stop_track)

        self.timer = rospy.Timer(rospy.Duration(1.0 / target_hz), self._tick)

    def _handle_reset_and_track(self, req) -> TriggerResponse:
        with self._lock:
            self.active = False
            try:
                self.points_class.reset_episode()
                # inference_mode() here too, so the causal_state set_queries()
                # creates is an inference tensor from the start -- consistent
                # with every predict() call in _tick (see its own note) and
                # avoids any cross-mode tensor issues later in the rollout.
                with torch.inference_mode():
                    frame1, frame2, pts1_xy, pts2_xy = localize_and_init_tracking(
                        self.points_class, self.image_recorder
                    )
                panel1 = draw_tracked_points(frame1, pts1_xy)
                panel2 = draw_tracked_points(frame2, pts2_xy)
                self.viz_pub.publish(build_image_msg(np.hstack([panel1, panel2])))
            except Exception as e:
                logger.exception("reset_and_track failed")
                return TriggerResponse(success=False, message=str(e))
            self.active = True
        logger.info("Tracking (re)started after DIFT re-localization.")
        return TriggerResponse(success=True, message="Localized and tracking.")

    def _handle_stop_track(self, req) -> TriggerResponse:
        with self._lock:
            self.active = False
        logger.info("Tracking stopped.")
        return TriggerResponse(success=True, message="Tracking stopped.")

    def _tick(self, event) -> None:
        if not self.active:
            return
        with self._lock:
            if not self.active:  # re-check: may have been stopped while waiting for the lock
                return
            try:
                # Load-bearing, not just a speed optimization: TapirTracker.__init__
                # calls torch.set_grad_enabled(False), but PyTorch's grad-mode is
                # thread-local, and that call happened on the main thread, not this
                # Timer's thread. Without an explicit inference_mode() scope here,
                # predict() would build an autograd graph on top of the causal_state
                # carried across every call -- an unbounded-growth leak, confirmed on
                # hardware previously as a steady CUDA OOM.
                with torch.inference_mode():
                    env_state, combined = get_live_environment_state(
                        self.points_class, self.image_recorder, self.calib, self.P
                    )
            except Exception:
                logger.exception("Tracking tick failed; will retry next tick.")
                return

        self.env_state_pub.publish(Float64MultiArray(data=env_state.tolist()))
        self.viz_pub.publish(build_image_msg(combined))


def parse_args():
    argv = rospy.myargv(argv=sys.argv)[1:]
    parser = argparse.ArgumentParser(
        description="DIFT localization + causal TAPIR point tracking as a standalone ROS node")
    parser.add_argument("--task_name", type=str, required=True,
                        help="Task name -- selects which coordinates/<task_name>/ manual "
                             "annotation to DIFT-localize against.")
    parser.add_argument("--calib_path", type=str,
                        default="/root/osx-ur/dependencies/Point-Policy/calib/calib.npy")
    parser.add_argument("--tracking_fps", type=float, default=5.0,
                        help="Target rate (Hz) for tracking + publishing. TAPIR tracking "
                             "is far slower than a typical 30Hz control loop (~85-90ms/"
                             "camera/frame), so this runs at its own sustainable rate.")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    setup_logging()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    rospy.init_node("tracking_node", anonymous=False)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(isinstance(h, RichHandler) for h in logger.handlers):
        logger.addHandler(RichHandler(console=console, rich_tracebacks=True, show_path=False))

    logger.info(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    logger.info(f"Loading point tracker for task '{args.task_name}'...")
    points_class = load_points_class(args.task_name)
    n_points = points_class.initial_coords[f"{PIXEL_KEY_1}_objects"].shape[0]
    rospy.set_param("~num_points", int(n_points))
    logger.info(f"Tracking {n_points} object point(s) for task '{args.task_name}'.")

    for pk in (PIXEL_KEY_1, PIXEL_KEY_2):
        tracker_model = points_class.tracker[pk].model
        model_device = next(tracker_model.parameters()).device
        logger.info(f"TapirTracker[{pk}] model device: {model_device}")

    calib = np.load(args.calib_path, allow_pickle=True).item()
    P = build_projection_matrices(calib)

    image_recorder = RawImageRecorder(camera_names=[CAM1_TOPIC, CAM2_TOPIC])
    logger.info("Waiting for cameras...")
    deadline = time.perf_counter() + 10.0
    while not image_recorder.cameras_ready() and not rospy.is_shutdown():
        if time.perf_counter() > deadline:
            logger.error("Timed out waiting for cameras.")
            sys.exit(1)
        rospy.sleep(0.1)
    logger.info("Cameras ready.")

    TrackingNode(points_class, image_recorder, calib, P, args.tracking_fps)
    logger.info(
        f"tracking_node ready: ~reset_and_track / ~stop_track services, "
        f"~environment_state / ~visualization_image topics, {args.tracking_fps} Hz."
    )
    rospy.spin()


if __name__ == "__main__":
    main()
