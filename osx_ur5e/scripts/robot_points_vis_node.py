#!/usr/bin/env python3
"""Standalone live visualization: paint the UR5e's actual gripper keypoints
onto both raw Realsense camera feeds.

Fully independent of any control script (replay_pkl.py, evaluate_policy_points.py,
replay_pkl_episode.py): it reads the robot's current TCP pose passively via
tf2 (base_link -> tool0), the exact same base/tip frames
ur_control.arm.Arm.end_effector() uses for its own forward-kinematics lookup
(over /joint_states), instead of constructing a CompliantController -- doing
that here would risk switching controllers out from under whatever script is
actually driving the arm (CompliantController's constructor activates ROS
control and registers an on_shutdown hook that reactivates the joint
trajectory controller on exit). tf2 has no such side effects and works
whether or not anything else is running.

Each tick, the looked-up pose is expanded into the 9 gripper keypoints (same
layout convert_pkl_human_to_robot.py writes to robot_tracks_3d_*, via
gripper_points.py's `extrapoints`), reprojected onto each raw camera frame
via calib.npy, and republished as a single side-by-side annotated image on
~visualization_image. Same subscribe/project/draw/publish pattern as
tracking_node.py and replay_pkl.py's live visualization (manual
sensor_msgs/Image encode/decode -- cv_bridge is broken under NumPy 2.x here).

Gripper open/closed state (only used for the small fingertip-narrowing
visual) is read from ClawController in read-only mode -- only
get_normalized_position() is ever called, never a setter, so this never
competes with a real controlling process for the gripper. Pass
--use_gripper false to skip it entirely (always draws the open geometry).

Usage:
    rosrun osx_ur5e robot_points_vis_node.py
    rosrun osx_ur5e robot_points_vis_node.py --rate 15 --use_gripper false

View with:
    rosrun rqt_image_view rqt_image_view /robot_points_vis_node/visualization_image
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rospy
import tf2_ros
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image
from rich.console import Console
from rich.logging import RichHandler

from osx_claw.claw_controller import ClawController

# gripper_points.py's `extrapoints` (8 body-point offsets from the TCP
# center) -- not installed as a package, matching replay_pkl.py's own
# sys.path convention for this file.
sys.path.insert(0, str(Path("/root/osx-ur/dependencies/Point-Policy/point_policy/robot_utils/ur5e")))
from gripper_points import extrapoints  # noqa: E402

logger = logging.getLogger(__name__)
console = Console()

# cam_1/cam_2 identity must match calib/calib.npy (see tracking_node.py,
# replay_pkl.py, collect_data_config.yaml). Do not reorder.
CAM1_TOPIC = "wrist_camera"   # cam_1
CAM2_TOPIC = "front_camera"  # cam_2


def _signal_handler(sig, frame):
    logger.info("Interrupted — shutting down.")
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)


def setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(RichHandler(console=console, rich_tracebacks=True, show_path=False))


# ---------------------------------------------------------------------------
# Live camera visualization: subscribe to both cameras, reproject the
# current gripper's 9 keypoints onto each raw frame, republish the annotated
# pair. Identical helpers to replay_pkl.py's (duplicated inline rather than
# shared, matching this codebase's existing convention -- tracking_node.py
# and replay_pkl.py already each carry their own copies).
# ---------------------------------------------------------------------------

class RawImageRecorder:
    """Subscribes directly to /{cam}/color/image_raw and decodes the raw
    sensor_msgs/Image manually. Identical to tracking_node.py's class of the
    same name."""

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
                result[name] = None
            elif rospy.get_time() - self._timestamps[name] > 0.5:
                rospy.logwarn_throttle(1, f"Image for {name} is too old! skipping visualization this tick")
                result[name] = None
            else:
                result[name] = img
        return result


def build_image_msg(frame_bgr: np.ndarray) -> Image:
    """Manually build a sensor_msgs/Image (bgr8) -- the mirror-image of
    RawImageRecorder's manual decode. Identical to tracking_node.py's
    function of the same name."""
    msg = Image()
    msg.header.stamp = rospy.Time.now()
    msg.height, msg.width = frame_bgr.shape[:2]
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(frame_bgr).tobytes()
    return msg


def project(pts3d: np.ndarray, K: np.ndarray, E: np.ndarray, D: np.ndarray) -> np.ndarray:
    """(N,3) world points -> (N,2) pixel coords, given calib.npy's per-camera
    int/ext/dist_coeff. Identical to visualize_reproj.py's function of the
    same name."""
    r, t = E[:3, :3], E[:3, 3]
    rvec, _ = cv2.Rodrigues(r)
    proj, _ = cv2.projectPoints(pts3d.astype(np.float32), rvec, t, K, D)
    return proj.squeeze(axis=1)  # (N, 2)


def gripper_world_points(pose7: np.ndarray, is_closed: bool = False) -> np.ndarray:
    """pose7 = [x, y, z, qx, qy, qz, qw] (a TCP pose, e.g. from base_link ->
    tool0 tf) -> (9, 3) world points: [TCP center, *8 body points].

    Same layout/order convert_pkl_human_to_robot.py writes to
    robot_tracks_3d_* (points3d = [T_g_world[:3,3]] + [T_g_world @ Tp for Tp
    in extrapoints]). gripper_points.py's extrapoints are defined relative to
    the TCP frame -- matches replay_pkl.py's gripper_world_points exactly.

    is_closed: when True, narrows the two fingertip points (indices 0/1 of
    extrapoints -> points3d[1]/[2]), same values convert_pkl_human_to_robot.py
    uses when it detects a pinch.
    """
    T = np.eye(4)
    T[:3, :3] = R.from_quat(pose7[3:]).as_matrix()
    T[:3, 3] = pose7[:3]
    points3d = [T[:3, 3]]
    for ep_idx, Tp in enumerate(extrapoints):
        Tp_local = Tp.copy()
        if is_closed and ep_idx in (0, 1):
            Tp_local[1, 3] = 0.015 if ep_idx == 0 else -0.015
        pt = T @ Tp_local
        points3d.append(pt[:3, 3])
    return np.array(points3d)  # (9, 3)


_GRIPPER_POINT_COLORS = [
    (0, 255, 255),    # 0: TCP center -- yellow
    (0, 255, 0), (0, 200, 0),        # 1-2: fingertips -- green
    (255, 128, 0), (255, 160, 0), (255, 200, 0),  # 3-5: mid-finger -- orange
    (255, 0, 200), (255, 0, 150), (255, 0, 100),  # 6-8: near-base -- magenta
]


def draw_gripper_points(frame_rgb: np.ndarray, pts_xy: np.ndarray, radius: int = 5) -> np.ndarray:
    """frame_rgb: HxWx3 uint8 RGB. pts_xy: (9,2). Returns a BGR image with
    the current gripper's 9 keypoints overlaid (same style as
    tracking_node.py's draw_tracked_points)."""
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    for i, (x, y) in enumerate(pts_xy):
        x, y = int(round(float(x))), int(round(float(y)))
        color = _GRIPPER_POINT_COLORS[i % len(_GRIPPER_POINT_COLORS)]
        cv2.circle(frame_bgr, (x, y), radius, color, -1)
        cv2.circle(frame_bgr, (x, y), radius + 1, (255, 255, 255), 1)
    return frame_bgr


def publish_robot_points_visualization(pose7: np.ndarray, is_closed: bool,
                                       image_recorder: RawImageRecorder,
                                       calib: dict, viz_pub) -> None:
    """One live frame from each camera -> annotated side-by-side panel,
    published on viz_pub. Any failure here is logged and swallowed -- this
    node must keep ticking even through a transient camera/projection hiccup."""
    try:
        images = image_recorder.get_images()
        frame1, frame2 = images[CAM1_TOPIC], images[CAM2_TOPIC]
        if frame1 is None or frame2 is None:
            return

        pts3d = gripper_world_points(pose7, is_closed=is_closed)
        pts1_xy = project(pts3d, calib["cam_1"]["int"], calib["cam_1"]["ext"], calib["cam_1"]["dist_coeff"])
        pts2_xy = project(pts3d, calib["cam_2"]["int"], calib["cam_2"]["ext"], calib["cam_2"]["dist_coeff"])

        panel1 = draw_gripper_points(frame1, pts1_xy)
        panel2 = draw_gripper_points(frame2, pts2_xy)
        viz_pub.publish(build_image_msg(np.hstack([panel1, panel2])))
    except Exception:
        logger.exception("Visualization failed this tick; will retry next tick.")


# ---------------------------------------------------------------------------
# Pose reading: plain tf2 lookup, no CompliantController -- see module
# docstring for why. Numerically identical to arm.end_effector() since both
# ultimately derive from /joint_states + the same base_link/tool0 kinematic
# chain (ur_control.arm.Arm.end_effector() does FK to the same tip_link this
# node looks up via tf).
# ---------------------------------------------------------------------------

def transform_to_pose7(transform) -> np.ndarray:
    """geometry_msgs/TransformStamped -> [x, y, z, qx, qy, qz, qw]."""
    t = transform.transform.translation
    q = transform.transform.rotation
    return np.array([t.x, t.y, t.z, q.x, q.y, q.z, q.w], dtype=np.float64)


def parse_args():
    argv = rospy.myargv(argv=sys.argv)[1:]
    parser = argparse.ArgumentParser(
        description="Paint the UR5e's live gripper keypoints onto both raw camera feeds")
    parser.add_argument("--calib_path", type=str,
                        default="/root/osx-ur/dependencies/Point-Policy/calib/calib.npy",
                        help="Path to calib.npy")
    parser.add_argument("--base_frame", type=str, default="base_link",
                        help="tf base frame (default: base_link, matches ur_control.arm.Arm)")
    parser.add_argument("--tip_frame", type=str, default="tool0",
                        help="tf tip frame (default: tool0, matches ur_control.arm.Arm's ee_link)")
    parser.add_argument("--rate", type=float, default=30.0, help="Publish rate in Hz (default: 30)")
    parser.add_argument("--use_gripper", type=lambda s: s.lower() != "false", default=True,
                        help="Read gripper open/closed state via ClawController (read-only) for the "
                             "fingertip-narrowing visual (default: true; pass false to skip it and "
                             "always draw the open geometry, e.g. if the gripper topic isn't up)")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    setup_logging()

    rospy.init_node("robot_points_vis_node", anonymous=False)
    # rospy.init_node() disrupts logging set up beforehand (same issue
    # worked around in evaluate_policy_points.py) -- re-establish our handler
    # and stop propagating to the root logger's now-altered handlers.
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(isinstance(h, RichHandler) for h in logger.handlers):
        logger.addHandler(RichHandler(console=console, rich_tracebacks=True, show_path=False))
    logger.info("ROS node initialized")

    calib = np.load(args.calib_path, allow_pickle=True).item()

    image_recorder = RawImageRecorder(camera_names=[CAM1_TOPIC, CAM2_TOPIC])
    viz_pub = rospy.Publisher("~visualization_image", Image, queue_size=1)

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)  # noqa: F841 -- must stay alive

    claw = None
    if args.use_gripper:
        logger.info("Initializing ClawController (read-only -- only get_normalized_position() is called)...")
        claw = ClawController(init_node=False)

    logger.info(
        f"Publishing to {rospy.get_name()}/visualization_image at {args.rate} Hz "
        f"(tf: {args.base_frame} -> {args.tip_frame}). Waiting up to 5s for cameras..."
    )
    deadline = time.perf_counter() + 5.0
    while not image_recorder.cameras_ready() and time.perf_counter() < deadline and not rospy.is_shutdown():
        rospy.sleep(0.1)
    if not image_recorder.cameras_ready():
        logger.warning(
            "Cameras not ready after 5s -- will publish once frames arrive, or stay silent if the "
            "camera nodes aren't running."
        )

    rate = rospy.Rate(args.rate)
    while not rospy.is_shutdown():
        try:
            transform = tf_buffer.lookup_transform(
                args.base_frame, args.tip_frame, rospy.Time(0), rospy.Duration(0.2)
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(2.0, f"tf lookup {args.base_frame} -> {args.tip_frame} failed: {e}")
            rate.sleep()
            continue

        pose7 = transform_to_pose7(transform)
        # Flipped relative to ClawController.get_normalized_position()'s own
        # docstring (0=closed/1=open) -- observed inverted on this gripper,
        # scoped fix here only (evaluate_policy_points.py is unaffected/fine).
        is_closed = claw is not None and claw.get_normalized_position() > 0.5
        publish_robot_points_visualization(pose7, is_closed, image_recorder, calib, viz_pub)
        rate.sleep()


if __name__ == "__main__":
    main()
