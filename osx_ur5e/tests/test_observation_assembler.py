"""Unit tests for ObservationAssembler (no robot, no ROS master needed)."""

from pathlib import Path

import numpy as np
import pytest

from osx_ur5e.observation_assembler import ObservationAssembler, Sample

UR5E_URDF = Path("/root/osx-ur/underlay_ws/src/ur_python_utilities/ur_pykdl/urdf/ur5e.urdf")

QPOS = np.array([0.1, -1.2, 1.8, -0.9, -1.5, 0.3])
QVEL = np.array([0.02, -0.05, 0.10, 0.01, -0.02, 0.03])
WRENCH = np.array([1.0, -2.0, 3.0, 0.1, -0.2, 0.3])


@pytest.fixture(scope="module")
def kinematics():
    from ur_pykdl import ur_kinematics
    return ur_kinematics(urdf_string=UR5E_URDF.read_text())


@pytest.fixture()
def assembler(kinematics):
    return ObservationAssembler(kinematics, camera_names=["front_camera"])


def make_samples(t0=100.0):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    return {
        "joint_states": Sample(t0 + 0.010, np.concatenate([QPOS, QVEL])),
        "wrench": Sample(t0 + 0.011, WRENCH),
        "images.front_camera": Sample(t0 + 0.005, img),
        "target_frame": None,  # replaced per-test
    }


def test_observation_keys_shapes_dtypes(assembler):
    samples = make_samples()
    obs = assembler.assemble_observation(samples, tick_time=100.02, episode_start=100.0)

    expected_shapes = {
        "observation.qpos": (6,),
        "observation.qvel": (6,),
        "observation.eef.position": (3,),
        "observation.eef.linear_velocity": (3,),
        "observation.eef.angular_velocity": (3,),
        "observation.eef.rotation_ortho6": (6,),
        "observation.eef.rotation_axis_angle": (3,),
        "observation.ft": (6,),
        "observation.image_time.front_camera": (1,),
        "observation.frame_time": (1,),
    }
    for key, shape in expected_shapes.items():
        assert key in obs, key
        assert obs[key].shape == shape, key
        assert obs[key].dtype == np.float32, key
    assert obs["observation.images.front_camera"].shape == (480, 640, 3)
    assert obs["observation.images.front_camera"].dtype == np.uint8


def test_observation_values(assembler, kinematics):
    samples = make_samples()
    obs = assembler.assemble_observation(samples, tick_time=100.02, episode_start=100.0)

    np.testing.assert_allclose(obs["observation.qpos"], QPOS.astype(np.float32))
    np.testing.assert_allclose(obs["observation.qvel"], QVEL.astype(np.float32))
    np.testing.assert_allclose(obs["observation.ft"], WRENCH.astype(np.float32))

    eef = np.asarray(kinematics.forward(QPOS))
    np.testing.assert_allclose(obs["observation.eef.position"], eef[:3].astype(np.float32), rtol=1e-6)

    eef_vel = np.asarray(kinematics.forward_velocity(QPOS, QVEL))
    np.testing.assert_allclose(obs["observation.eef.linear_velocity"], eef_vel[:3].astype(np.float32), rtol=1e-5)
    np.testing.assert_allclose(obs["observation.eef.angular_velocity"], eef_vel[3:].astype(np.float32), rtol=1e-5)

    # Times relative to episode start on a shared axis.
    np.testing.assert_allclose(obs["observation.frame_time"], [0.02], atol=1e-6)
    np.testing.assert_allclose(obs["observation.image_time.front_camera"], [0.005], atol=1e-6)


def test_rotation_representations_consistent(assembler, kinematics):
    from ur_control import transformations

    obs = assembler.assemble_observation(make_samples(), tick_time=100.02, episode_start=100.0)
    eef = np.asarray(kinematics.forward(QPOS))

    # ortho6 roundtrips back to the same quaternion (up to sign).
    q_back = np.asarray(transformations.quaternion_from_ortho6(
        obs["observation.eef.rotation_ortho6"].astype(np.float64)))
    q = eef[3:]
    assert min(np.linalg.norm(q_back - q), np.linalg.norm(q_back + q)) < 1e-5


def test_missing_camera_raises(assembler):
    samples = make_samples()
    del samples["images.front_camera"]
    with pytest.raises(KeyError):
        assembler.assemble_observation(samples, tick_time=100.02)


def test_action_zero_delta_when_target_equals_pose(assembler, kinematics):
    eef = np.asarray(kinematics.forward(QPOS))
    samples = {
        "target_frame": Sample(100.0, eef.copy()),
        "stiffness": Sample(100.0, np.full(6, 300.0)),
        "gello_joints": Sample(100.0, QPOS + 0.01),
    }
    action = assembler.assemble_action(samples, eef_pose=eef)
    np.testing.assert_allclose(action["action.delta_position"], np.zeros(3), atol=1e-7)
    np.testing.assert_allclose(action["action.delta_rotation"], np.zeros(3), atol=1e-6)
    np.testing.assert_allclose(action["action.position"], eef[:3].astype(np.float32))
    np.testing.assert_allclose(action["action.stiffness_diag"], np.full(6, 300.0, dtype=np.float32))
    np.testing.assert_allclose(action["action.joint"], (QPOS + 0.01).astype(np.float32))


def test_action_delta_matches_hand_computed(assembler, kinematics):
    from ur_control import transformations

    eef = np.asarray(kinematics.forward(QPOS))
    target = eef.copy()
    target[:3] += np.array([0.010, -0.005, 0.002])
    # Rotate the target slightly around z.
    target[3:] = transformations.rotate_quaternion_by_rpy(0.0, 0.0, 0.05, eef[3:])

    samples = {
        "target_frame": Sample(100.0, target),
        "stiffness": Sample(100.0, np.full(6, 300.0)),
        "gello_joints": Sample(100.0, QPOS),
    }
    action = assembler.assemble_action(samples, eef_pose=eef)

    np.testing.assert_allclose(action["action.delta_position"],
                               [0.010, -0.005, 0.002], atol=1e-6)
    expected_rot = np.asarray(transformations.quaternions_orientation_error(target[3:], eef[3:]))
    np.testing.assert_allclose(action["action.delta_rotation"],
                               expected_rot.astype(np.float32), atol=1e-6)


def test_required_keys(assembler):
    keys = assembler.required_keys(with_action=True, with_images=True)
    assert set(keys) == {
        "joint_states", "wrench", "images.front_camera",
        "target_frame", "stiffness", "gello_joints",
    }
    assert set(assembler.required_keys(with_action=False, with_images=False)) == {
        "joint_states", "wrench",
    }
