#!/usr/bin/env python3
"""Replay a human-demo-converted episode (from the pkl produced by the
human->robot conversion script) directly on a UR5e with a custom gripper.

Unlike replay_episode.py, this does NOT read a LeRobot dataset and does NOT
need FK-from-joint-angles: the pkl already stores cartesian TCP targets
(position + rotvec) per step in `robot_tcp_poses`, so we send those straight
into the same cartesian-compliance controller used elsewhere.

Also (optionally, on by default -- dataset.publish_visualization) subscribes
to both cameras and publishes a live annotated view on ~replay_visualization:
each step's commanded gripper pose is expanded into its 9 keypoints (same
layout convert_pkl_human_to_robot.py writes to robot_tracks_3d_*) and
reprojected onto each raw camera frame via calib.npy, same pattern as
tracking_node.py's ~visualization_image. View with e.g.:
    rosrun rqt_image_view rqt_image_view /replay_from_pkl/replay_visualization

Usage:
    rosrun osx_ur5e replay_from_pkl.py
    rosrun osx_ur5e replay_from_pkl.py dataset.episode_idx=3

    # Replay a ROBOT_PKL demo directly by task name + demo number, instead of
    # the full path (constructs {dataset.dir}/processed_data_pkl/expert_demos/
    # {env_name}/{task_name}/demo_{episode_idx:04d}.pkl -- the real,
    # vision-tracking-derived poses written by convert_pkl_human_to_robot.py):
    rosrun osx_ur5e replay_from_pkl.py dataset.task_name=bottle_open_06 dataset.episode_idx=0
    rosrun osx_ur5e replay_from_pkl.py dataset.task_name=bottle_open_06 dataset.env_name=franka_env dataset.episode_idx=0

Assumes the pkl has one of three structures (auto-detected, see
load_episode -- in priority order):
  1. A task_pkl_io ROBOT_PKL demo file (dataset.task_name, or dataset.pkl_path
     pointed directly at expert_demos/<env>/<task>/demo_XXXX.pkl -- written by
     convert_pkl_human_to_robot.py):
       DATA["robot_tcp_poses"] -> (T, 6) [x,y,z,rx,ry,rz] (rotvec)
       DATA["gripper_states"]  -> (T,)   -1 (open) / 1 (closed)
  2. A task_pkl_io human pkl demo file (dataset.pkl_path pointed directly at
     processed_data_pkl/<task>/demo_XXXX.pkl -- written by
     convert_to_pkl_human*.py). WARNING: for demos collected via
     collect_image_data.py, cartesian_states/gripper_states here are always
     dummy all-zero placeholders, not real robot state -- prefer (1):
       DATA["cartesian_states"] -> (T, 6) [x,y,z,rx,ry,rz] (rotvec)
       DATA["gripper_states"]   -> (T,)   -1 (open) / 1 (closed)
  3. The original converted format:
       DATA["observations"] -> list of episodes
       DATA["observations"][i]["robot_tcp_poses"]  -> (T, 6) [x,y,z,rx,ry,rz] (rotvec), robot-base frame
       DATA["observations"][i]["gripper_states"]   -> (T,)   -1 (open) / 1 (closed)

load_episode also refuses to replay a degenerate trajectory (position barely
moves across the whole episode) rather than silently driving the arm into it
-- this is what dummy/placeholder pose data (case 2, on an image-only
collection) looks like.

NOTE: if robot_tcp_poses/gripper_states are written without a {pixel_key}
suffix by an upstream per-camera loop (as human_poses/gripper_states were in
the original human->robot conversion script), only the LAST camera's version
ends up in the pkl. Make sure that's the calibration frame you actually want,
or fix the upstream script to suffix these keys and update GRIPPER_KEY /
POSE_KEY below accordingly.
"""

from ur_control.fzi_cartesian_compliance_controller import CompliantController
from ur_control import transformations
from osx_claw.claw_controller import ClawController
from rich.logging import RichHandler
from rich.console import Console
import rospy
from sensor_msgs.msg import Image
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import hydra
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2
import logging
import pickle as pkl
import signal
import sys
import time
from pathlib import Path

import numpy as np
import tqdm

import matplotlib
matplotlib.use("Agg")  # must be before pyplot import

# gripper_points.py's `extrapoints` (8 body-point offsets from the TCP
# center) -- not installed as a package, matching convert_pkl_human_to_robot.py's
# own sys.path convention for this file.
sys.path.insert(0, str(Path("/root/osx-ur/dependencies/Point-Policy/point_policy/robot_utils/ur5e")))
from gripper_points import extrapoints  # noqa: E402


logger = logging.getLogger(__name__)
console = Console()

# (T, 6): [x, y, z, rx, ry, rz] rotvec, robot-base frame
POSE_KEY = "robot_tcp_poses"
GRIPPER_KEY = "gripper_states"    # (T,):   -1 open / 1 closed

_DIM_LABELS = ["x", "y", "z", "rx", "ry", "rz", "gripper"]

# cam_1/cam_2 identity must match calib/calib.npy (see tracking_node.py,
# collect_data_config.yaml). Do not reorder.
CAM1_TOPIC = "wrist_camera"   # cam_1
CAM2_TOPIC = "front_camera"  # cam_2


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
    root.addHandler(RichHandler(console=console,
                    rich_tracebacks=True, show_path=False))
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"))
    root.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Pkl loading helpers
# ---------------------------------------------------------------------------

MIN_POSITION_RANGE_M = 0.01  # 1cm -- see the degenerate-trajectory guard below


def load_episode(pkl_path: Path, episode_idx: int):
    """Load one episode's pose/gripper trajectory from a pkl.

    Supports three pkl layouts, auto-detected by top-level key:
      - A task_pkl_io ROBOT_PKL demo file (e.g.
        data/processed_data_pkl/expert_demos/<env_name>/<task>/demo_0000.pkl,
        written by convert_pkl_human_to_robot.py) -- a single dict with
        POSE_KEY ("robot_tcp_poses") / GRIPPER_KEY ("gripper_states")
        directly at the top level, reconstructed from vision-tracked hand
        poses. This is the real, replayable data and is checked first.
      - A task_pkl_io human pkl demo file (e.g.
        data/processed_data_pkl/<task>/demo_0000.pkl, written by
        convert_to_pkl_human*.py) -- a single dict with "cartesian_states"
        (T,6) / "gripper_states" (T,). WARNING: for demos collected via
        collect_image_data.py (image-only collection), these are always
        dummy all-zero placeholders (see write_states_csv there) -- not
        real robot state. The degenerate-trajectory guard below exists
        mainly to catch this case before it reaches the arm.
      - The original converted format: DATA["observations"] -> list of
        episode dicts, each with POSE_KEY/GRIPPER_KEY. episode_idx selects
        which one.
    For the two task_pkl_io (single-dict) layouts, the file itself is one
    demo, so episode_idx is ignored (it was already consumed to pick which
    demo_XXXX.pkl to load, via dataset.task_name or an explicit
    dataset.pkl_path).
    """
    logger.info(f"Loading pkl: {pkl_path}")
    # pkl may have been pickled with numpy>=2.0 (which renamed numpy.core to
    # numpy._core); alias the submodules to their already-imported numpy.core
    # counterparts so it unpickles under older numpy installs like the one in
    # this ROS environment. Aliasing submodules individually (rather than
    # just the top-level package) avoids re-importing/re-initializing
    # numpy's C extensions under a second name, which crashes.
    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = np.core
        for _name in dir(np.core):
            _mod = getattr(np.core, _name)
            if isinstance(_mod, type(np.core)) and _mod.__name__.startswith("numpy.core."):
                sys.modules[f"numpy._core.{_name}"] = _mod
    with open(pkl_path, "rb") as f:
        DATA = pkl.load(f)

    if POSE_KEY in DATA:
        # task_pkl_io ROBOT_PKL demo file -- real, vision-tracking-derived
        # per-step target poses (see convert_pkl_human_to_robot.py).
        logger.info(f"Pkl is a task_pkl_io ROBOT_PKL demo file ({POSE_KEY}/{GRIPPER_KEY})")
        poses = np.asarray(DATA[POSE_KEY])          # (T, 6)
        grippers = np.asarray(DATA[GRIPPER_KEY])    # (T,)
    elif "cartesian_states" in DATA:
        logger.info(f"Pkl is a task_pkl_io human pkl demo file (cartesian_states/gripper_states)")
        poses = np.asarray(DATA["cartesian_states"])   # (T, 6)
        grippers = np.asarray(DATA["gripper_states"])  # (T,)
    else:
        observations = DATA["observations"]
        num_episodes = len(observations)
        logger.info(
            f"Pkl has {num_episodes} episode(s); replaying episode {episode_idx}")

        if episode_idx >= num_episodes:
            raise ValueError(
                f"episode_idx {episode_idx} out of range (pkl has {num_episodes} episodes)")

        ep = observations[episode_idx]
        if POSE_KEY not in ep or GRIPPER_KEY not in ep:
            raise KeyError(
                f"Episode {episode_idx} missing '{POSE_KEY}' or '{GRIPPER_KEY}'. "
                f"Available keys: {list(ep.keys())}"
            )

        poses = np.asarray(ep[POSE_KEY])          # (T, 6)
        grippers = np.asarray(ep[GRIPPER_KEY])    # (T,)

    if poses.shape[0] != grippers.shape[0]:
        raise ValueError(
            f"poses ({poses.shape[0]}) and gripper_states ({grippers.shape[0]}) "
            "length mismatch for this episode."
        )

    # Degenerate-trajectory guard: a pose trajectory that barely moves across
    # the whole episode is almost certainly dummy/placeholder data (e.g.
    # collect_image_data.py's all-zero states.csv for image-only
    # collections), not a real demo -- refuse to command the arm into it
    # rather than silently driving to a fixed (here, the origin) pose.
    position_range = np.ptp(poses[:, :3], axis=0)
    if np.all(position_range < MIN_POSITION_RANGE_M):
        raise ValueError(
            f"{pkl_path} looks like a degenerate/dummy trajectory -- position "
            f"barely moves across the episode (range={position_range} m, "
            f"threshold={MIN_POSITION_RANGE_M} m). Refusing to replay this "
            "onto the arm. If this is a human pkl (cartesian_states) from an "
            "image-only collection (collect_image_data.py), its state is "
            "always dummy zero by design -- replay the corresponding "
            "ROBOT_PKL instead (data/processed_data_pkl/expert_demos/"
            "<env_name>/<task>/demo_XXXX.pkl, produced by "
            "convert_pkl_human_to_robot.py)."
        )

    return poses, grippers


def pose_vec_to_quat_pose(pose_vec: np.ndarray) -> np.ndarray:
    """[x, y, z, rx, ry, rz] rotvec -> [x, y, z, qx, qy, qz, qw]."""
    pos = pose_vec[:3]
    rotvec = pose_vec[3:6]
    quat = R.from_rotvec(rotvec).as_quat()  # scipy: [qx, qy, qz, qw]
    return np.concatenate([pos, quat])


def gripper_state_to_normalized(state: float) -> float:
    """-1 (open) / 1 (closed) -> 0.0-1.0 for ClawController.set_normalized_position.

    Inverted relative to the naive -1->0.0 / 1->1.0 mapping: here state > 0
    (closed) maps to 0.0 and state <= 0 (open) maps to 1.0, to match this
    gripper's normalized-position convention.

    Adjust this mapping again if your custom gripper's open/close convention
    changes, or replace with a continuous mapping if you regenerate
    gripper_states as a continuous value upstream instead of the binary -1/1
    used by the conversion script.
    """
    return 0.0 if state > 0 else 1.0


# ---------------------------------------------------------------------------
# Live camera visualization: subscribe to both cameras, reproject the
# commanded gripper's 9 keypoints (same layout convert_pkl_human_to_robot.py
# writes to robot_tracks_3d_*) onto each raw frame, and republish the
# annotated pair. Same subscribe/publish pattern as tracking_node.py
# (manual sensor_msgs/Image encode/decode -- cv_bridge is broken under
# NumPy 2.x in this environment).
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
                rospy.logwarn_throttle(1, f"Image for {name} is too old! skipping visualization this step")
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
    """pose7 = [x, y, z, qx, qy, qz, qw] (a TCP pose, e.g. arm.end_effector())
    -> (9, 3) world points: [TCP center, *8 body points].

    Same layout/order convert_pkl_human_to_robot.py writes to
    robot_tracks_3d_* (points3d = [T_g_world[:3,3]] + [T_g_world @ Tp for Tp
    in extrapoints]). gripper_points.py's extrapoints are defined relative to
    the TCP frame -- arm.end_effector() already *is* that TCP pose (Tshift,
    the flange->TCP offset, is what produces a valid TCP target from a
    hand-tracking-derived flange estimate upstream; it doesn't apply again
    here), so extrapoints can be applied to it directly.

    is_closed: when True, narrows the two fingertip points (indices 0/1 of
    extrapoints -> points3d[1]/[2]) the same way convert_pkl_human_to_robot.py
    does when it detects a pinch -- without this, the overlay never visually
    shows the gripper opening/closing, even though the commanded pose and
    gripper_states both change correctly.
    """
    T = np.eye(4)
    T[:3, :3] = R.from_quat(pose7[3:]).as_matrix()
    T[:3, 3] = pose7[:3]
    points3d = [T[:3, 3]]
    for ep_idx, Tp in enumerate(extrapoints):
        Tp_local = Tp.copy()
        if is_closed and ep_idx in (0, 1):
            # Same values as convert_pkl_human_to_robot.py's pinch-close case.
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
    the commanded gripper's 9 keypoints overlaid (same style as
    tracking_node.py's draw_tracked_points)."""
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    for i, (x, y) in enumerate(pts_xy):
        x, y = int(round(float(x))), int(round(float(y)))
        color = _GRIPPER_POINT_COLORS[i % len(_GRIPPER_POINT_COLORS)]
        cv2.circle(frame_bgr, (x, y), radius, color, -1)
        cv2.circle(frame_bgr, (x, y), radius + 1, (255, 255, 255), 1)
    return frame_bgr


def publish_replay_visualization(pose7: np.ndarray, is_closed: bool, image_recorder: RawImageRecorder,
                                 calib: dict, viz_pub) -> None:
    """One live frame from each camera -> annotated side-by-side panel,
    published on viz_pub. Any failure here is logged and swallowed -- this
    is a non-critical add-on and must never interrupt the actual replay."""
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
        logger.exception("Replay visualization failed this step; will retry next step.")


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

def compute_clipped_target(target_pose_quat: np.ndarray, current_pose: np.ndarray,
                           safety_cfg) -> tuple:
    """Shared clipping logic: one safety-bounded step from current -> target.

    Returns (next_target_pose, pos_err, rot_err_rad) where the errors are the
    *unclipped* remaining distance to target (useful for convergence checks).
    """
    delta_translation = target_pose_quat[:3] - current_pose[:3]
    delta_rotation = transformations.quaternions_orientation_error(
        target_pose_quat[3:], current_pose[3:]
    )

    max_delta_rotation = np.deg2rad(safety_cfg.max_delta_rotation)
    clipped_translation = np.clip(delta_translation,
                                  -safety_cfg.max_delta_translation,
                                  safety_cfg.max_delta_translation)
    clipped_rotation = np.clip(
        delta_rotation, -max_delta_rotation, max_delta_rotation)

    next_pos = current_pose[:3] + clipped_translation
    next_pos[0] = np.clip(
        next_pos[0], safety_cfg.workspace_range.x[0], safety_cfg.workspace_range.x[1])
    next_pos[1] = np.clip(
        next_pos[1], safety_cfg.workspace_range.y[0], safety_cfg.workspace_range.y[1])
    next_pos[2] = np.clip(
        next_pos[2], safety_cfg.workspace_range.z[0], safety_cfg.workspace_range.z[1])
    next_orient = transformations.rotate_quaternion_by_rpy(
        *clipped_rotation, current_pose[3:])
    next_target = np.concatenate([next_pos, next_orient])

    pos_err = np.linalg.norm(delta_translation)
    rot_err = np.linalg.norm(delta_rotation)
    return next_target, pos_err, rot_err


def execute_cartesian_action(target_pose_quat: np.ndarray, gripper_norm: float,
                             arm, claw, safety_cfg) -> None:
    """Send one safety-clipped cartesian step toward target (used in the main replay loop)."""
    current_pose = arm.end_effector()
    next_target, _, _ = compute_clipped_target(
        target_pose_quat, current_pose, safety_cfg)

    arm.set_cartesian_target_pose(pose=next_target)

    if claw is not None:
        claw.set_normalized_position(float(np.clip(gripper_norm, 0.0, 1.0)))


def ramp_to_start_pose(target_pose_quat: np.ndarray, arm, safety_cfg,
                       pos_tol: float = 0.005, rot_tol_deg: float = 1.0,
                       control_hz: float = 20.0, timeout_s: float = 20.0) -> None:
    """Walk the arm to the episode's first pose using the same per-step clipping
    as the replay loop, instead of one unclipped jump.

    This mirrors what set_joint_positions(..., target_time=5.0, wait=True) gave
    the original joint-space replay for free (a smooth, rate-limited approach
    to the start), but does it in cartesian space step-by-step since we only
    have a cartesian target here.
    """
    rot_tol = np.deg2rad(rot_tol_deg)
    dt = 1.0 / control_hz
    deadline = time.perf_counter() + timeout_s

    logger.info(
        f"Ramping to start pose (pos_tol={pos_tol} m, rot_tol={rot_tol_deg} deg, "
        f"timeout={timeout_s}s)..."
    )

    step = 0
    while True:
        current_pose = np.array(arm.end_effector())
        next_target, pos_err, rot_err = compute_clipped_target(
            target_pose_quat, current_pose, safety_cfg
        )

        if pos_err < pos_tol and rot_err < rot_tol:
            logger.info(f"Start pose reached after {step} ramp steps "
                        f"(pos_err={pos_err:.4f} m, rot_err={np.rad2deg(rot_err):.2f} deg)")
            return

        if time.perf_counter() > deadline:
            logger.warning(
                f"Ramp to start pose timed out after {timeout_s}s — "
                f"pos_err={pos_err:.4f} m, rot_err={np.rad2deg(rot_err):.2f} deg. "
                "Proceeding anyway; check the arm before continuing."
            )
            return

        arm.set_cartesian_target_pose(pose=next_target)
        step += 1
        time.sleep(dt)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_visualization(target_log: list, actual_log: list, out_path: Path,
                       pkl_path: Path, episode_idx: int) -> None:
    """Plot converted-demo targets vs actual arm/gripper state.

    target_log: [{t, pose_vec (6,), gripper_norm}]  — from the pkl.
    actual_log: [{t, pose (7,) xyz+quat, gripper}]  — measured from the arm.
    """
    n_dims = 7
    n_cols = 2
    n_rows = (n_dims + 1) // n_cols

    t_arr = np.array([e["t"] for e in target_log])
    target_pos = np.array([e["pose_vec"][:3]
                          for e in target_log])       # (T, 3)
    target_rotvec = np.array([e["pose_vec"][3:6]
                             for e in target_log])   # (T, 3)
    target_grip = np.array([e["gripper_norm"] for e in target_log])      # (T,)

    actual_pos = np.array([e["pose"][:3]
                          for e in actual_log])           # (T, 3)
    actual_quat = np.array([e["pose"][3:]
                           for e in actual_log])          # (T, 4)
    actual_rotvec = R.from_quat(
        actual_quat).as_rotvec()                 # (T, 3)
    actual_grip = np.array([e["gripper"] for e in actual_log])           # (T,)

    target_all = np.concatenate(
        [target_pos, target_rotvec, target_grip[:, None]], axis=1
    )
    actual_all = np.concatenate(
        [actual_pos, actual_rotvec, actual_grip[:, None]], axis=1
    )

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))
    axes = axes.flatten()

    for dim in range(n_dims):
        ax = axes[dim]
        ax.plot(t_arr, target_all[:, dim], color="#377eb8",
                linewidth=1.8, label="pkl target")
        ax.plot(t_arr, actual_all[:, dim], color="#e41a1c", linewidth=1.2,
                linestyle="--", alpha=0.8, label="arm actual")
        gap = np.abs(target_all[:, dim] - actual_all[:, dim])
        ax.fill_between(t_arr, target_all[:, dim], actual_all[:, dim],
                        alpha=0.12, color="orange", label=f"gap (max={gap.max():.3f})")
        ax.set_title(_DIM_LABELS[dim], fontsize=11, fontweight="bold")
        ax.set_xlabel("Episode step")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    for dim in range(n_dims, len(axes)):
        axes[dim].set_visible(False)

    fig.suptitle(
        f"Pkl Replay  |  {pkl_path.name}  episode={episode_idx}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(
    version_base=None,
    config_path="../config/hydra",
    config_name="test_task",
)
def main(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "replay_from_pkl.log")

    np.set_printoptions(linewidth=np.inf, formatter={
                        "float": lambda x: f"{x:0.3f}"})

    # ------------------------------------------------------------------
    # Load episode from pkl
    # ------------------------------------------------------------------
    episode_idx = int(cfg.dataset.episode_idx)

    task_name = cfg.dataset.get("task_name", None)
    if task_name:
        # ROBOT_PKL demo file, addressed by task name + demo number
        # (episode_idx) instead of a full path -- matches ROBOT_PKL's
        # directory layout in run_pipeline.sh, written by
        # convert_pkl_human_to_robot.py (real, vision-tracking-derived
        # poses -- NOT the Step-2 human pkl under processed_data_pkl/{task}/
        # directly, whose cartesian_states/gripper_states are dummy zero
        # placeholders for image-only collections; see load_episode):
        #   {root}/data/processed_data_pkl/expert_demos/{env_name}/{task_name}/demo_{episode_idx:04d}.pkl
        env_name = cfg.dataset.get("env_name", "franka_env")
        pkl_path = (
            Path(cfg.dataset.dir) / "processed_data_pkl" / "expert_demos" / env_name / task_name
            / f"demo_{episode_idx:04d}.pkl"
        )
    else:
        pkl_path = Path(cfg.dataset.pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"Pkl file not found: {pkl_path}")
    fps = cfg.dataset.fps
    step_duration_s = 1.0 / fps
    use_gripper = cfg.dataset.get("use_gripper", True)

    pose_vecs, gripper_states = load_episode(pkl_path, episode_idx)
    ep_len = pose_vecs.shape[0]
    logger.info(f"Episode {episode_idx}: {ep_len} steps")

    # ------------------------------------------------------------------
    # Initialize hardware
    # ------------------------------------------------------------------
    rospy.init_node("replay_from_pkl", anonymous=False)
    logger.info("ROS node initialized")

    # ------------------------------------------------------------------
    # Live camera visualization (optional, non-critical): subscribe now so
    # frames have time to arrive before the replay loop starts publishing.
    # ------------------------------------------------------------------
    publish_visualization = cfg.dataset.get("publish_visualization", True)
    image_recorder = None
    calib = None
    viz_pub = None
    if publish_visualization:
        calib_path = cfg.dataset.get(
            "calib_path", "/root/osx-ur/dependencies/Point-Policy/calib/calib.npy")
        calib = np.load(calib_path, allow_pickle=True).item()
        image_recorder = RawImageRecorder(camera_names=[CAM1_TOPIC, CAM2_TOPIC])
        viz_pub = rospy.Publisher("~replay_visualization", Image, queue_size=1)
        logger.info(
            f"Live visualization enabled -> publishing to "
            f"{rospy.get_name()}/replay_visualization (waiting up to 5s for cameras)"
        )
        deadline = time.perf_counter() + 5.0
        while not image_recorder.cameras_ready() and time.perf_counter() < deadline:
            rospy.sleep(0.1)
        if not image_recorder.cameras_ready():
            logger.warning(
                "Cameras not ready after 5s -- visualization will publish once frames "
                "arrive, or stay silent if the camera nodes aren't running. Replay is "
                "unaffected either way."
            )

    controller_cfg = cfg.controller
    safety_cfg = controller_cfg.safety_parameters

    arm = CompliantController(gripper_type=None)
    arm.set_control_mode(controller_cfg.mode)
    arm.update_pd_gains(
        OmegaConf.to_container(controller_cfg.p_gains),
        OmegaConf.to_container(controller_cfg.d_gains),
    )
    arm.update_selection_matrix(
        OmegaConf.to_container(controller_cfg.selection_matrix))
    arm.set_solver_parameters(
        error_scale=controller_cfg.error_scale,
        iterations=controller_cfg.iterations,
    )
    arm.update_stiffness(controller_cfg.stiffness * np.ones(6))
    arm.auto_switch_controllers = False
    arm.async_mode = True
    arm.zero_ft_sensor()

    claw = None
    if use_gripper:
        logger.info("Initializing ClawController...")
        claw = ClawController(init_node=False)

    # ------------------------------------------------------------------
    # Move to first frame's cartesian pose
    # ------------------------------------------------------------------
    first_pose = pose_vec_to_quat_pose(pose_vecs[0])
    first_gripper = gripper_state_to_normalized(gripper_states[0])

    logger.info(
        f"Moving to start pose: pos={first_pose[:3]} quat={first_pose[3:]}")
    arm.activate_cartesian_controller()
    ramp_to_start_pose(
        first_pose, arm, safety_cfg,
        pos_tol=cfg.dataset.get("start_pos_tol", 0.005),
        rot_tol_deg=cfg.dataset.get("start_rot_tol_deg", 1.0),
        control_hz=cfg.dataset.get("start_ramp_hz", 20.0),
        timeout_s=cfg.dataset.get("start_ramp_timeout_s", 20.0),
    )
    if claw is not None:
        claw.set_normalized_position(first_gripper)

    input(
        f"\n  Episode {episode_idx} ({ep_len} steps) — press Enter to start replay...")

    # ------------------------------------------------------------------
    # Replay loop
    # ------------------------------------------------------------------
    logger.info(f"Replaying episode {episode_idx} at {fps} Hz...")

    target_log = []  # {t, pose_vec (6,), gripper_norm}
    actual_log = []  # {t, pose (7,), gripper}

    # On every gripper open<->close transition, stretch out (rather than
    # freeze) the next `gripper_slowdown_duration_s` of real time by
    # multiplying each step's sleep by `gripper_slowdown_factor` -- gives the
    # slow claw a head start on reaching the commanded position/making
    # contact before the arm has moved far, without depending on gripper
    # position feedback (which ClawController.get_position() can echo back
    # instantly from the just-issued command rather than real sensor state).
    gripper_slowdown_enabled = cfg.dataset.get("gripper_slowdown_enabled", True)
    gripper_slowdown_factor = cfg.dataset.get("gripper_slowdown_factor", 4.0)
    gripper_slowdown_duration_s = cfg.dataset.get("gripper_slowdown_duration_s", 0.5)
    prev_gripper_state = gripper_states[0]  # already commanded + settled before the loop
    slowdown_until = -1.0  # perf_counter deadline; step_start past this -> normal speed

    with tqdm.tqdm(total=ep_len, desc=f"Episode {episode_idx}") as pbar:
        for step in range(ep_len):
            step_start = time.perf_counter()

            pose_vec = pose_vecs[step]
            gripper_norm = gripper_state_to_normalized(gripper_states[step])
            target_pose_quat = pose_vec_to_quat_pose(pose_vec)

            if claw is not None and gripper_states[step] != prev_gripper_state:
                claw.set_normalized_position(gripper_norm)
                if gripper_slowdown_enabled:
                    slowdown_until = step_start + gripper_slowdown_duration_s
                    logger.info(
                        f"Gripper transition -> slowing replay {gripper_slowdown_factor}x "
                        f"for {gripper_slowdown_duration_s}s"
                    )
            prev_gripper_state = gripper_states[step]

            execute_cartesian_action(
                target_pose_quat, gripper_norm, arm, claw, safety_cfg)

            _pose = np.array(arm.end_effector())
            _grip = claw.get_normalized_position() if claw is not None else 0.0
            target_log.append(
                {"t": step, "pose_vec": pose_vec.copy(), "gripper_norm": gripper_norm})
            actual_log.append(
                {"t": step, "pose": _pose.copy(), "gripper": _grip})

            if publish_visualization:
                # gripper_states convention: >0 = closed (see gripper_state_to_normalized).
                is_closed = bool(gripper_states[step] > 0)
                publish_replay_visualization(_pose, is_closed, image_recorder, calib, viz_pub)

            tqdm.tqdm.write(
                f"t={step:03d}  target_pos={pose_vec[:3].round(3)}  "
                f"actual_pos={_pose[:3].round(3)}  "
                f"pos_gap={np.abs(pose_vec[:3] - _pose[:3]).max():.4f} m  "
                f"gripper_cmd={gripper_norm:.2f}"
            )

            dt_s = time.perf_counter() - step_start
            target_step_duration = step_duration_s
            if step_start < slowdown_until:
                target_step_duration *= gripper_slowdown_factor
            remaining = target_step_duration - dt_s
            if remaining < 0:
                logger.debug(f"Step slow: {1.0/dt_s:.1f} Hz (target {fps} Hz)")
            else:
                time.sleep(remaining)

            pbar.update(1)

    arm.activate_joint_trajectory_controller()
    print(f"Replay complete: {ep_len} steps")

    vis_path = output_dir / f"replay_from_pkl_vis_ep{episode_idx:03d}.png"
    print(f"Saving visualization to: {vis_path}")
    try:
        save_visualization(target_log, actual_log,
                           vis_path, pkl_path, episode_idx)
        print(f"Visualization saved: {vis_path}")
    except Exception as e:
        import traceback
        print(f"ERROR saving visualization: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
