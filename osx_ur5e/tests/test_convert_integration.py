"""End-to-end integration test for the Stage-2 converter.

Synthesizes a realistic multi-rate episode bag (500 Hz joints/wrench, 100 Hz
action topics, 30 Hz JPEG camera), runs the real convert_episode() path into
a real LeRobotDataset, and validates tick math, ZOH values, and dataset
structure. Needs the in-container stack (rosbag, lerobot, osx_msgs).
"""

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

rosbag = pytest.importorskip("rosbag")
pytest.importorskip("lerobot")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

UR5E_URDF = Path("/root/osx-ur/underlay_ws/src/ur_python_utilities/ur_pykdl/urdf/ur5e.urdf")

ALPHABETICAL_JOINTS = [
    "elbow_joint", "shoulder_lift_joint", "shoulder_pan_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

T0 = 1000.0        # synthetic ROS epoch start
DURATION = 3.0
FPS = 25


def qpos_at(t):
    """Ground-truth joint trajectory (deterministic, smooth)."""
    base = np.array([0.1, -1.2, 1.8, -0.9, -1.5, 0.3])
    return base + 0.1 * np.sin(2 * np.pi * 0.5 * (t - T0) + np.arange(6))


def _stamp(t):
    import rospy
    return rospy.Time.from_sec(t)


def write_synthetic_bag(bag_path: Path):
    import cv2
    from geometry_msgs.msg import PoseStamped, WrenchStamped
    from osx_msgs.msg import Float64ArrayStamped
    from sensor_msgs.msg import CompressedImage, JointState

    # JOINT_ORDER index of each alphabetical name, to publish driver-style.
    from osx_ur5e.sample_feeders import JOINT_ORDER

    with rosbag.Bag(str(bag_path), "w") as bag:
        # 500 Hz joint states + wrench
        for i in range(int(DURATION * 500)):
            t = T0 + i / 500.0
            q = qpos_at(t)
            msg = JointState()
            msg.header.stamp = _stamp(t)
            msg.name = ALPHABETICAL_JOINTS
            msg.position = [q[JOINT_ORDER.index(n)] for n in ALPHABETICAL_JOINTS]
            msg.velocity = [0.05 * k for k in range(6)]
            bag.write("/joint_states", msg, _stamp(t))

            w = WrenchStamped()
            w.header.stamp = _stamp(t)
            w.wrench.force.x = math.sin(t - T0)
            w.wrench.torque.z = 0.1
            bag.write("/wrench/filtered", w, _stamp(t))

        # 100 Hz target_frame + gello joints
        for i in range(int(DURATION * 100)):
            t = T0 + i / 100.0
            p = PoseStamped()
            p.header.stamp = _stamp(t)
            p.pose.position.x = 0.4 + 0.01 * math.sin(t - T0)
            p.pose.position.y = 0.1
            p.pose.position.z = 0.3
            p.pose.orientation.w = 1.0
            bag.write("/cartesian_compliance_controller/target_frame", p, _stamp(t))

            g = JointState()
            g.header.stamp = _stamp(t)
            g.position = list(qpos_at(t) + 0.02)
            bag.write("/data_collection/gello_joints", g, _stamp(t))

        # 1 Hz stiffness keepalive
        for i in range(int(DURATION) + 1):
            t = T0 + float(i)
            s = Float64ArrayStamped()
            s.header.stamp = _stamp(t)
            s.data = [400.0] * 6
            bag.write("/data_collection/stiffness_command", s, _stamp(t))

        # 30 Hz JPEG camera; pixel value encodes the frame index.
        for i in range(int(DURATION * 30)):
            t = T0 + i / 30.0
            img = np.full((48, 64, 3), i % 256, dtype=np.uint8)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            assert ok
            m = CompressedImage()
            m.header.stamp = _stamp(t)
            m.format = "rgb8; jpeg compressed bgr8"
            m.data = buf.tobytes()
            bag.write("/front_camera/color/image_raw/compressed", m, _stamp(t))

    return bag_path


@pytest.fixture(scope="module")
def dataset_cfg():
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "cameras": {"front_camera": {"height": 48, "width": 64, "channels": 3}},
        "states": {
            "observation.qpos": [6], "observation.qvel": [6],
            "observation.eef.position": [3], "observation.eef.linear_velocity": [3],
            "observation.eef.angular_velocity": [3], "observation.eef.rotation_ortho6": [6],
            "observation.eef.rotation_axis_angle": [3], "observation.ft": [6],
        },
        "actions": {
            "action.joint": [6], "action.position": [3], "action.rotation_ortho6": [6],
            "action.rotation_axis_angle": [3], "action.stiffness_diag": [6],
            "action.delta_position": [3], "action.delta_rotation": [3],
        },
    })


def test_full_conversion(tmp_path, dataset_cfg):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from ur_pykdl import ur_kinematics

    from convert_bags_to_lerobot import convert_episode
    from osx_ur5e.dataset_features import build_features
    from osx_ur5e.observation_assembler import ObservationAssembler

    bag_path = write_synthetic_bag(tmp_path / "episode.bag")

    session_meta = {
        "camera_topics": {"front_camera": ["/front_camera/color/image_raw/compressed",
                                           "/front_camera/color/camera_info"]},
        "robot_description": UR5E_URDF.read_text(),
        "task": "synthetic test",
    }
    episode_meta = {"t_record_start": T0}
    conversion_cfg = SimpleNamespace(
        wrench_topic="/wrench/filtered", staleness_budget_s={}, report=True)

    assembler = ObservationAssembler(
        ur_kinematics(urdf_string=session_meta["robot_description"]), ["front_camera"])

    dataset = LeRobotDataset.create(
        repo_id="synthetic_test",
        fps=FPS,
        features=build_features(dataset_cfg),
        root=tmp_path / "dataset",
        robot_type="ur5e",
        use_videos=True,
    )

    with rosbag.Bag(str(bag_path)) as bag:
        report = convert_episode(bag, episode_meta, session_meta, assembler,
                                 dataset, fps=FPS, task="synthetic test",
                                 conversion_cfg=conversion_cfg)
    dataset.finalize()
    # Re-open through the reader path (as training does); instances returned
    # by create() are write-oriented in this lerobot version.
    dataset = LeRobotDataset("synthetic_test", root=tmp_path / "dataset")

    # --- tick math -------------------------------------------------------
    # t0 = latest first-sample among sources (all start at T0 here).
    assert report["t0"] == pytest.approx(T0, abs=1e-6)
    expected_ticks = int(math.floor((report["t_end"] - report["t0"]) * FPS)) + 1
    assert report["num_ticks"] == expected_ticks
    assert report["num_ticks"] == dataset.num_frames
    assert dataset.num_episodes == 1

    # No staleness: every source is dense and starts at t0.
    assert not report["stale_ticks"], report["stale_ticks"]

    # --- content spot checks ----------------------------------------------
    frame0 = dataset[0]
    frame_mid = dataset[report["num_ticks"] // 2]

    # qpos at tick k must be the latest 500 Hz sample <= t_k (ZOH, <=2ms old).
    for idx, frame in ((0, frame0), (report["num_ticks"] // 2, frame_mid)):
        t_k = T0 + idx / FPS
        sample_t = math.floor((t_k - T0) * 500) / 500 + T0
        np.testing.assert_allclose(
            frame["observation.qpos"].numpy(), qpos_at(sample_t), atol=1e-5)

    # Delta actions: target - FK(eef); target x oscillates around 0.4.
    assert abs(frame_mid["action.position"][0].item() - 0.4) < 0.011
    np.testing.assert_allclose(
        frame_mid["action.stiffness_diag"].numpy(), np.full(6, 400.0), atol=1e-4)

    # frame_time is the real tick time relative to t0.
    k = report["num_ticks"] // 2
    assert frame_mid["observation.frame_time"].item() == pytest.approx(k / FPS, abs=1e-5)

    # Video was encoded and frames decode to the expected ramp value.
    videos = list((tmp_path / "dataset").rglob("*.mp4"))
    assert videos, "no video encoded"
    img = frame_mid["observation.images.front_camera"]
    t_k = T0 + k / FPS
    expected_cam_idx = math.floor((t_k - T0) * 30)
    value = float(img.mean()) * (255.0 if img.dtype.is_floating_point else 1.0)
    assert abs(value - (expected_cam_idx % 256)) < 6  # JPEG + video codec tolerance

    # image_time records the camera stamp (<= 1/30 s behind the tick).
    image_time = frame_mid["observation.image_time.front_camera"].item()
    assert 0.0 <= (k / FPS) - image_time <= (1 / 30 + 1e-6)


def test_conversion_is_deterministic(tmp_path, dataset_cfg):
    """Two conversions of the same bag produce identical parquet content."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from ur_pykdl import ur_kinematics

    from convert_bags_to_lerobot import convert_episode
    from osx_ur5e.dataset_features import build_features
    from osx_ur5e.observation_assembler import ObservationAssembler

    bag_path = write_synthetic_bag(tmp_path / "episode.bag")
    session_meta = {
        "camera_topics": {"front_camera": ["/front_camera/color/image_raw/compressed",
                                           "/front_camera/color/camera_info"]},
        "robot_description": UR5E_URDF.read_text(),
    }
    conversion_cfg = SimpleNamespace(
        wrench_topic="/wrench/filtered", staleness_budget_s={}, report=True)

    frames = []
    for run in (1, 2):
        assembler = ObservationAssembler(
            ur_kinematics(urdf_string=session_meta["robot_description"]), ["front_camera"])
        dataset = LeRobotDataset.create(
            repo_id="det_test", fps=FPS, features=build_features(dataset_cfg),
            root=tmp_path / f"dataset_{run}", robot_type="ur5e", use_videos=True)
        with rosbag.Bag(str(bag_path)) as bag:
            convert_episode(bag, {"t_record_start": T0}, session_meta, assembler,
                            dataset, fps=FPS, task="det", conversion_cfg=conversion_cfg)
        dataset.finalize()
        frames.append(LeRobotDataset("det_test", root=tmp_path / f"dataset_{run}"))

    a, b = frames
    assert a.num_frames == b.num_frames
    for idx in (0, a.num_frames // 2, a.num_frames - 1):
        fa, fb = a[idx], b[idx]
        for key in ("observation.qpos", "observation.ft", "action.delta_position",
                    "observation.frame_time"):
            np.testing.assert_array_equal(fa[key].numpy(), fb[key].numpy())
