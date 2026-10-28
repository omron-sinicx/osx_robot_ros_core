"""SpaceMouse teleoperation tests with fake device, arm and policy (no robot, no ROS master).

Run inside the container with the workspace sourced:
  python3 -m unittest discover -s osx_ur5e/tests -p test_spacemouse_teleop.py
"""

import importlib.util
import time
import unittest
from pathlib import Path

import numpy as np
import yaml

from osx_ur5e.policy import Policy
from osx_ur5e.spacemouse import MouseReading, RightTap, SpaceMouse, map_to_base
from osx_ur5e.target_tracker import rotate, rotation_error

PACKAGE = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("spacemouse_control_fdcc",
                                               PACKAGE / "scripts/spacemouse_control_fdcc.py")
smc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smc)

CFG = yaml.safe_load(smc.DEFAULT_CONFIG.read_text())
DT = 1.0 / CFG["control"]["rate_hz"]
DOWN = np.array([-0.0832, 0.9964, 0.0051, 0.0181]) / np.linalg.norm([-0.0832, 0.9964, 0.0051, 0.0181])
HOME = np.concatenate([[-0.005, 0.46, 0.618], DOWN])


class FakeState:
    def __init__(self, t, axes=(0.0,) * 6, buttons=(0, 0)):
        self.t = t
        self.x, self.y, self.z, self.roll, self.pitch, self.yaw = axes
        self.buttons = list(buttons)


class FakeDevice:
    def __init__(self):
        self.state = FakeState(t=-1.0)
        self.unplugged = False
        self.closed = False

    def push(self, axes=(0.0,) * 6, buttons=(0, 0)):
        self.state = FakeState(time.monotonic(), axes, buttons)

    def read(self):
        if self.unplugged:
            raise OSError("device unplugged")
        return self.state

    def close(self):
        self.closed = True


class FakeMouse:
    def __init__(self):
        self.set()

    def set(self, twist=(0.0,) * 6, left=False, right=False, connected=True):
        self.reading = MouseReading(np.array(twist, dtype=float), left, right, connected, 0.0)

    def read(self):
        return self.reading


class FakeArm:
    """Kinematic stand-in: the measured pose follows the commanded target exactly."""

    def __init__(self, pose):
        self.pose = np.array(pose, dtype=float)
        self.wrench = np.zeros(6)
        self.zero_count = 0

    def end_effector(self, joint_angles=None):
        return HOME.copy() if joint_angles is not None else self.pose.copy()

    def joint_angles(self):
        return np.zeros(6)

    def get_wrench(self):
        return self.wrench.copy()

    def zero_ft_sensor(self):
        self.zero_count += 1
        self.wrench[:] = 0.0

    def set_cartesian_target_pose(self, pose):
        self.pose = np.array(pose, dtype=float)


class ConstantPolicy(Policy):
    def __init__(self, delta):
        self.delta = np.asarray(delta, dtype=float)
        self.acts = 0
        self.resets = 0

    def reset(self, state):
        self.resets += 1

    def act(self, state):
        self.acts += 1
        return self.delta


def wait_for(condition, timeout=1.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.002)


class TestMapping(unittest.TestCase):
    def test_deadzone_zeroes_small_input_and_keeps_full_scale(self):
        np.testing.assert_allclose(map_to_base([0.04, -0.05, 0, 0, 0, 0], 0.0, 0.05), np.zeros(6))
        np.testing.assert_allclose(map_to_base([1.2, -1, 0.525, 0, 0, 0], 0.0, 0.05), [1, -1, 0.5, 0, 0, 0])

    def test_operator_yaw_rotates_linear_and_angular_axes(self):
        np.testing.assert_allclose(map_to_base([1, 0, 0, 1, 0, 0], 90.0, 0.0), [0, 1, 0, 0, 1, 0], atol=1e-12)


class TestRightTap(unittest.TestCase):
    def run_sequence(self, samples):
        tap = RightTap()
        return [tap.update(left, right) for left, right in samples]

    def test_solo_press_taps_on_release(self):
        self.assertEqual(self.run_sequence([(0, 1), (0, 1), (0, 0)]), [False, False, True])

    def test_chord_never_taps(self):
        self.assertFalse(any(self.run_sequence([(0, 1), (1, 1), (1, 0), (0, 0)])))
        self.assertFalse(any(self.run_sequence([(1, 0), (1, 1), (0, 1), (0, 0)])))

    def test_tap_after_chord(self):
        self.assertEqual(self.run_sequence([(1, 1), (0, 0), (0, 1), (0, 0)]), [False, False, False, True])


class TestSpaceMouseReader(unittest.TestCase):
    def setUp(self):
        self.device = FakeDevice()
        self.mouse = SpaceMouse(operator_yaw_deg=0.0, deadzone=0.0, stale_timeout_s=0.1, device=self.device)

    def tearDown(self):
        self.mouse.close()

    def test_reports_reach_reading(self):
        self.device.push(axes=(0.5, 0, 0, 0, 0, -0.25), buttons=(1, 0))
        wait_for(lambda: self.mouse.read().left)
        reading = self.mouse.read()
        np.testing.assert_allclose(reading.twist, [0.5, 0, 0, 0, 0, -0.25])
        self.assertTrue(reading.connected)

    def test_stale_axes_zero_but_buttons_persist(self):
        self.device.push(axes=(1, 0, 0, 0, 0, 0), buttons=(0, 1))
        wait_for(lambda: self.mouse.read().right)
        time.sleep(0.15)
        reading = self.mouse.read()
        np.testing.assert_allclose(reading.twist, np.zeros(6))
        self.assertTrue(reading.right)

    def test_unplug_reports_disconnected(self):
        self.device.push(axes=(1, 0, 0, 0, 0, 0), buttons=(1, 1))
        wait_for(lambda: self.mouse.read().left)
        self.device.unplugged = True
        wait_for(lambda: not self.mouse.read().connected)
        reading = self.mouse.read()
        np.testing.assert_allclose(reading.twist, np.zeros(6))
        self.assertFalse(reading.left or reading.right)


class TestRotationHelpers(unittest.TestCase):
    def test_rotate_identity_about_z(self):
        q = rotate([0, 0, 0, 1], [0, 0, np.pi / 2])
        np.testing.assert_allclose(q, [0, 0, np.sqrt(0.5), np.sqrt(0.5)], atol=1e-12)

    def test_rotation_error_inverts_rotate(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            v = rng.uniform(-0.5, 0.5, size=3)
            np.testing.assert_allclose(rotation_error(rotate(q, v), q), v, atol=1e-9)


class TestTargetTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = smc.make_tracker(CFG["limits"], DT)
        self.max_velocity = CFG["limits"]["max_linear_velocity"]

    def test_velocity_limit_per_axis(self):
        self.tracker.reanchor(HOME)
        for _ in range(50):
            before = self.tracker.pose[:3].copy()
            pose = self.tracker.step(np.array([1.0, -1.0, 0, 0, 0, 0]), self.tracker.pose)
            self.assertTrue(np.all(np.abs(pose[:3] - before) <= self.max_velocity * DT + 1e-12))

    def test_workspace_blocks_outward_motion_only(self):
        start = HOME.copy()
        start[0] = self.tracker.workspace_max[0]
        self.tracker.reanchor(start)
        pose = self.tracker.step(np.array([0.001, 0, 0, 0, 0, 0]), start)
        self.assertAlmostEqual(pose[0], start[0])
        self.tracker.reanchor(start)
        pose = self.tracker.step(np.array([-0.0005, 0, 0, 0, 0, 0]), start)
        self.assertLess(pose[0], start[0])

    def test_start_outside_box_is_not_pushed(self):
        start = HOME.copy()
        start[2] = self.tracker.workspace_min[2] - 0.05
        self.tracker.reanchor(start)
        np.testing.assert_allclose(self.tracker.step(np.zeros(6), start), start)
        self.tracker.reanchor(start)
        self.assertAlmostEqual(self.tracker.step(np.array([0, 0, -0.0005, 0, 0, 0]), start)[2], start[2])
        self.tracker.reanchor(start)
        self.assertGreater(self.tracker.step(np.array([0, 0, 0.0005, 0, 0, 0]), start)[2], start[2])

    def test_target_stays_within_tracking_error_of_measured(self):
        self.tracker.reanchor(HOME)
        for _ in range(100):
            pose = self.tracker.step(np.array([0.001, 0, 0, 0, 0, 0]), HOME)
        self.assertLessEqual(np.linalg.norm(pose[:3] - HOME[:3]), CFG["limits"]["max_tracking_error"] + 1e-12)


class TestController(unittest.TestCase):
    def make(self, start=HOME, policy=None):
        self.arm = FakeArm(start)
        self.mouse = FakeMouse()
        self.controller = smc.SpaceMouseController(CFG, self.arm, self.mouse, policy=policy,
                                                   clock=time.monotonic)
        self.controller.tracker.reanchor(start)

    def ticks(self, n, **mouse):
        self.mouse.set(**mouse)
        for _ in range(n):
            self.controller.tick()

    def test_teleop_full_deflection_moves_at_configured_speed(self):
        self.make()
        self.ticks(100, twist=(1, 0, 0, 0, 0, 0))
        travelled = self.arm.pose[0] - HOME[0]
        speed = CFG["teleop"]["max_linear_speed"]
        self.assertGreater(travelled, 0.9 * speed)
        self.assertLessEqual(travelled, speed + 1e-9)

    def test_rotation_follows_twist(self):
        self.make()
        self.ticks(100, twist=(0, 0, 0, 0, 0, 1))
        angle = rotation_error(self.arm.pose[3:], HOME[3:])
        self.assertGreater(angle[2], 0.9 * np.deg2rad(CFG["teleop"]["max_angular_speed_deg"]))

    def test_right_tap_zeroes_ft_without_motion(self):
        self.make()
        self.ticks(3, right=True)
        self.ticks(1)
        self.assertEqual(self.arm.zero_count, 1)
        np.testing.assert_allclose(self.arm.pose, HOME)

    def test_force_fault_holds_until_right_tap(self):
        self.make()
        self.arm.wrench[:3] = [0, 0, CFG["safety"]["max_force"] + 10]
        self.ticks(1, twist=(1, 0, 0, 0, 0, 0))
        self.assertIs(self.controller.mode, smc.Mode.FAULT)
        held = self.arm.pose.copy()
        self.ticks(20, twist=(1, 0, 0, 0, 0, 0))
        np.testing.assert_allclose(self.arm.pose, held)
        self.ticks(2, right=True)
        self.ticks(1)
        self.assertEqual(self.arm.zero_count, 1)
        self.assertIs(self.controller.mode, smc.Mode.TELEOP)
        self.ticks(10, twist=(1, 0, 0, 0, 0, 0))
        self.assertGreater(self.arm.pose[0], held[0])

    def test_disconnected_mouse_faults(self):
        self.make()
        self.ticks(1, connected=False)
        self.assertIs(self.controller.mode, smc.Mode.FAULT)

    def test_homing_converges_with_limited_speed(self):
        start = np.concatenate([HOME[:3] + [0.05, -0.03, 0.02], rotate(HOME[3:], [0, 0, np.deg2rad(8)])])
        self.make(start)
        positions = []
        self.mouse.set(left=True, right=True)
        for _ in range(400):
            self.controller.tick()
            positions.append(self.arm.pose[:3].copy())
        steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        self.assertLessEqual(steps.max(), CFG["home"]["max_linear_speed"] * DT + 1e-9)
        self.assertLess(np.linalg.norm(self.arm.pose[:3] - HOME[:3]), CFG["home"]["position_tolerance"])
        self.assertLess(np.linalg.norm(rotation_error(HOME[3:], self.arm.pose[3:])),
                        np.deg2rad(CFG["home"]["orientation_tolerance_deg"]))
        self.assertEqual(self.arm.zero_count, 0)

    def test_left_without_policy_stays_in_teleop(self):
        self.make()
        self.ticks(5, left=True)
        self.assertIs(self.controller.mode, smc.Mode.TELEOP)
        np.testing.assert_allclose(self.arm.pose, HOME)

    def test_policy_delta_spread_over_its_period(self):
        policy = ConstantPolicy([0.0022, 0, 0, 0, 0, 0])
        self.make(policy=policy)
        ticks_per_act = round(policy.period_s * CFG["control"]["rate_hz"])
        self.ticks(4 * ticks_per_act, left=True)
        self.assertEqual(policy.resets, 1)
        self.assertEqual(policy.acts, 4)
        self.assertGreater(self.arm.pose[0], HOME[0])
        self.assertLessEqual(self.arm.pose[0] - HOME[0], 4 * 0.0022 + 1e-9)


if __name__ == "__main__":
    unittest.main()
