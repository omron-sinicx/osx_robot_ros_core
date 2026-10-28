#!/usr/bin/env python3

# The MIT License (MIT)
#
# Copyright (c) 2026 OMRON SINIC X
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Author: Cristian Beltran

"""Teleoperate the UR5e with a 3Dconnexion SpaceMouse under the FZI Cartesian compliance controller.

SpaceMouse Compact buttons:
  LEFT held          run the policy (ignored when no policy is loaded)
  RIGHT tap          zero the F/T sensor, re-anchor the target and clear a fault
  LEFT + RIGHT held  move to the home pose

Usage:
  rosrun osx_ur5e spacemouse_control_fdcc.py                 # teleoperation
  rosrun osx_ur5e spacemouse_control_fdcc.py --mouse-only    # print the mapped mouse input; no ROS/robot
  rosrun osx_ur5e spacemouse_control_fdcc.py --config FILE
"""

import argparse
import enum
import time
from pathlib import Path

import numpy as np
import rospy
import yaml

from osx_ur5e.action_limits import DeltaActionLimiter
from osx_ur5e.policy import Policy, RobotState
from osx_ur5e.spacemouse import RightTap, SpaceMouse
from osx_ur5e.target_tracker import TargetTracker, limit_norm, rotation_error
from ur_control.fzi_cartesian_compliance_controller import CompliantController

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config/control_fdcc.yaml"


class Mode(enum.Enum):
    TELEOP = "teleop"
    POLICY = "policy"
    HOMING = "homing"
    FAULT = "fault"


def configure_controller(arm, mode, stiffness, p_gains, d_gains, error_scale, iterations):
    """Apply the `controller` config block to the compliance controller."""
    arm.set_control_mode(mode)
    arm.update_selection_matrix(np.ones(6))
    arm.update_stiffness(np.asarray(stiffness, dtype=float))
    arm.update_pd_gains(p_gains, d_gains)
    arm.set_solver_parameters(error_scale=error_scale, iterations=iterations)


def make_tracker(limits, dt):
    """Build the TargetTracker from the `limits` config block."""
    linear_velocity = limits["max_linear_velocity"]
    angular_velocity = np.deg2rad(limits["max_angular_velocity_deg"])
    limiter = DeltaActionLimiter(
        max_delta_translation=linear_velocity * dt,
        max_delta_rotation=angular_velocity * dt,
        max_linear_velocity=linear_velocity,
        max_linear_acceleration=limits["max_linear_acceleration"],
        max_angular_velocity=angular_velocity,
        max_angular_acceleration=np.deg2rad(limits["max_angular_acceleration_deg"]))
    workspace = limits["workspace"]
    return TargetTracker(
        limiter, dt,
        workspace_min=[workspace[axis][0] for axis in "xyz"],
        workspace_max=[workspace[axis][1] for axis in "xyz"],
        max_tracking_error=limits["max_tracking_error"])


class SpaceMouseController:
    """Each tick: read the mouse and robot, pick a mode from the buttons, command one limited step."""

    def __init__(self, cfg, arm, mouse, policy: Policy | None = None, clock=rospy.get_time):
        self.arm = arm
        self.mouse = mouse
        self.policy = policy
        self.clock = clock

        self.rate_hz = cfg["control"]["rate_hz"]
        self.dt = 1.0 / self.rate_hz
        self.tracker = make_tracker(cfg["limits"], self.dt)

        teleop = cfg["teleop"]
        self.teleop_speed = np.array(3 * [teleop["max_linear_speed"]] +
                                     3 * [np.deg2rad(teleop["max_angular_speed_deg"])])
        if not teleop["enable_rotation"]:
            self.teleop_speed[3:] = 0.0

        self.max_force = cfg["safety"]["max_force"]
        self.max_torque = cfg["safety"]["max_torque"]

        home = cfg["home"]
        self.home_pose = np.asarray(arm.end_effector(joint_angles=home["joints"]), dtype=float)
        self.home_linear_speed = home["max_linear_speed"]
        self.home_angular_speed = np.deg2rad(home["max_angular_speed_deg"])
        self.home_position_tolerance = home["position_tolerance"]
        self.home_orientation_tolerance = np.deg2rad(home["orientation_tolerance_deg"])
        if not self.tracker.in_workspace(self.home_pose[:3]):
            rospy.logwarn(f"Home position {self.home_pose[:3].round(3)} is outside the workspace box")

        self.policy_ticks = max(1, round(policy.period_s * self.rate_hz)) if policy else 1
        self.right_tap = RightTap()
        self.mode = Mode.TELEOP
        self._reset_mode_state()

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    def run(self):
        """Activate the compliance controller and run until shutdown or a protective stop."""
        rate = rospy.Rate(self.rate_hz)
        self.arm.zero_ft_sensor()
        state = self.read_state()
        self.tracker.reanchor(state.pose)
        if not self.tracker.in_workspace(state.pose[:3]):
            rospy.logwarn(f"Start position {state.pose[:3].round(3)} is outside the workspace box; "
                          "only motion back toward it is allowed")
        self.arm.activate_cartesian_controller()
        rospy.loginfo("Teleoperation active")
        try:
            while not rospy.is_shutdown():
                if self.arm.dashboard_services.is_protective_stopped():
                    rospy.logerr("Protective stop: exiting")
                    break
                start = time.monotonic()
                self.tick()
                if time.monotonic() - start > self.dt:
                    rospy.logwarn_throttle(5, f"Control tick took longer than {self.dt * 1e3:.0f} ms")
                rate.sleep()
        except rospy.ROSInterruptException:
            pass
        finally:
            self.arm.activate_joint_trajectory_controller()
            if self.policy:
                self.policy.close()

    def tick(self):
        reading = self.mouse.read()
        state = self.read_state()
        right_tap = self.right_tap.update(reading.left, reading.right)

        if self.mode is Mode.FAULT:
            if right_tap and reading.connected:
                self.arm.zero_ft_sensor()
                self.switch_mode(Mode.TELEOP, self.read_state())
                rospy.loginfo("F/T sensor zeroed, fault cleared")
            self.arm.set_cartesian_target_pose(self.tracker.pose)
            return

        fault = self.detect_fault(reading, state)
        if fault:
            rospy.logerr(f"{fault}; holding pose. Tap RIGHT to zero the F/T sensor and resume.")
            self.switch_mode(Mode.FAULT, state)
            self.arm.set_cartesian_target_pose(self.tracker.pose)
            return

        if right_tap:
            self.arm.zero_ft_sensor()
            self.tracker.reanchor(self.arm.end_effector())
            rospy.loginfo("F/T sensor zeroed, target re-anchored")
            return

        mode = self.requested_mode(reading)
        if mode is not self.mode:
            self.switch_mode(mode, state)

        if self.mode is Mode.TELEOP:
            delta = reading.twist * self.teleop_speed * self.dt
        elif self.mode is Mode.POLICY:
            delta = self.policy_delta(state)
        else:
            delta = self.homing_delta(state)
        self.arm.set_cartesian_target_pose(self.tracker.step(delta, state.pose))

    # ------------------------------------------------------------------
    # Modes
    # ------------------------------------------------------------------

    def requested_mode(self, reading):
        if reading.left and reading.right:
            return Mode.HOMING
        if reading.left and self.policy is not None:
            return Mode.POLICY
        if reading.left and not self._warned_no_policy:
            rospy.logwarn("LEFT pressed but no policy is loaded")
            self._warned_no_policy = True
        return Mode.TELEOP

    def switch_mode(self, mode, state):
        rospy.loginfo(f"Mode {self.mode.value} -> {mode.value}")
        self.mode = mode
        self.tracker.reanchor(state.pose)
        self._reset_mode_state()
        if mode is Mode.POLICY:
            self.policy.reset(state)

    def _reset_mode_state(self):
        self._policy_step = np.zeros(6)
        self._policy_ticks_left = 0
        self._home_reached = False
        self._warned_no_policy = False

    def policy_delta(self, state):
        if self._policy_ticks_left == 0:
            self._policy_step = np.asarray(self.policy.act(state), dtype=float) / self.policy_ticks
            self._policy_ticks_left = self.policy_ticks
        self._policy_ticks_left -= 1
        return self._policy_step

    def homing_delta(self, state):
        target = self.tracker.pose
        position_error = self.home_pose[:3] - target[:3]
        orientation_error = rotation_error(self.home_pose[3:], target[3:])
        reached = (np.linalg.norm(self.home_pose[:3] - state.pose[:3]) < self.home_position_tolerance and
                   np.linalg.norm(rotation_error(self.home_pose[3:], state.pose[3:])) <
                   self.home_orientation_tolerance)
        if reached and not self._home_reached:
            rospy.loginfo("Home reached")
        self._home_reached = reached
        return np.concatenate([limit_norm(position_error, self.home_linear_speed * self.dt),
                               limit_norm(orientation_error, self.home_angular_speed * self.dt)])

    # ------------------------------------------------------------------
    # Robot I/O
    # ------------------------------------------------------------------

    def read_state(self):
        return RobotState(stamp=self.clock(),
                          joints=np.asarray(self.arm.joint_angles(), dtype=float),
                          pose=np.asarray(self.arm.end_effector(), dtype=float),
                          wrench=np.asarray(self.arm.get_wrench(), dtype=float))

    def detect_fault(self, reading, state):
        if not reading.connected:
            return "SpaceMouse disconnected"
        force, torque = np.linalg.norm(state.wrench[:3]), np.linalg.norm(state.wrench[3:])
        if force > self.max_force or torque > self.max_torque:
            return f"F/T limit exceeded (|F|={force:.1f} N, |T|={torque:.2f} Nm)"
        return None


def run_mouse_only(mouse, period_s=0.1):
    """Print the mapped base_link command until Ctrl-C."""
    previous_count = mouse.report_count
    try:
        while True:
            time.sleep(period_s)
            reading = mouse.read()
            count = mouse.report_count
            twist = np.array2string(reading.twist, precision=2, suppress_small=True, sign=" ")
            print(f"twist {twist}  left={int(reading.left)} right={int(reading.right)}  "
                  f"reports/s={(count - previous_count) / period_s:4.0f}  "
                  f"age={reading.report_age_s:5.2f}s  connected={reading.connected}")
            previous_count = count
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mouse-only", action="store_true", help="print the mapped mouse input; no ROS/robot")
    args = parser.parse_args(rospy.myargv()[1:])
    cfg = yaml.safe_load(args.config.read_text())

    mouse = SpaceMouse(**cfg["spacemouse"])
    try:
        if args.mouse_only:
            run_mouse_only(mouse)
            return
        rospy.init_node("spacemouse_control_fdcc")
        arm = CompliantController(gripper_type=None, **cfg["robot"])
        if not arm.dashboard_services.activate_ros_control_on_ur():
            raise RuntimeError("Failed to activate ROS control on the UR")
        configure_controller(arm, **cfg["controller"])
        SpaceMouseController(cfg, arm, mouse).run()
    finally:
        mouse.close()


if __name__ == "__main__":
    main()
