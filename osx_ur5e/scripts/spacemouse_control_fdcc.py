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

"""C2 handoff: SpaceMouse manual teleoperation and exactly two policy systems.

Original C2 uses the paired 445 mm / 35 mm Student + D1 + Teacher models.
c2_force_div500 uses the same models with all policy force inputs attenuated
from native F/50 to F/500; both retain Teacher-only Fz negation and T/5.
This input experiment is not physical force limiting or a calibrated sign fix.

Use deploy/rri/run_c2.sh c2 or force500 after reading docs/C2_HANDOFF_zh_TW.md.
Loading --system produces a startup preview; policy motion starts only after
an operator button action. Starting this script itself initializes robot IO.
Code RIGHT tap-release starts policy, RIGHT press stops; physical button sides
must be identified. Optional other-toggle Home uses code LEFT press-release;
any button stops Home. Home completes on TCP, not on saved joint branch.
Normal policy start does not tare; fault acknowledgement and manual controls
retain their documented tare behavior. SPACE/ESC exits. No automatic Home.
"""

import argparse
import enum
import sys
import time
from pathlib import Path

import numpy as np
import rospy
import yaml

from osx_ur5e.action_limits import DeltaActionLimiter
from osx_ur5e.keyboard import ExitKeys
from osx_ur5e.policy import Policy, RobotState
from osx_ur5e.spacemouse import RightTap, SpaceMouse
from osx_ur5e.target_tracker import TargetTracker, PolicyTargetManager, limit_norm, rotate, rotation_error
from ur_control.fzi_cartesian_compliance_controller import CompliantController

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config/control_fdcc.yaml"


class Mode(enum.Enum):
    TELEOP = "teleop"
    BUTTON_WAIT = "button_wait"
    POLICY = "policy"
    HOMING = "homing"
    HOME_HOLD = "home_hold"
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

    @staticmethod
    def validate_direction_reset(cfg):
        """The C2 handoff excludes the retired direction-reset controller."""
        if cfg.get("policy_direction_reset") is not None:
            raise ValueError("C2_HANDOFF_DIRECTION_RESET_REMOVED")
        return None

    @staticmethod
    def validate_target_management(cfg):
        value = cfg.get("policy_target_management")
        if value is None:
            return None
        options = PolicyTargetManager.validate(value)
        if options is not None:
            if cfg.get("policy_direction_reset", {}).get("enabled", False):
                raise ValueError("C1 and C2 target management are mutually exclusive")
            if not cfg["limits"].get("policy_acceleration_limit_enabled", True):
                raise ValueError("C2 requires existing acceleration limiting")
            if cfg.get("robot") != {"base_link": "base_link", "ee_link": "tool0"}:
                raise ValueError("C2 requires measured and target base_link/tool0 reference")
        return options

    @staticmethod
    def validate_tracking_limits(limits):
        """Require a finite positive lead cap and boolean policy-only limiter overrides."""
        cap = limits["max_tracking_error"]
        if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not 0 < cap < float("inf"):
            raise ValueError("limits.max_tracking_error must be a finite positive number")
        for name in ("policy_tracking_cap_enabled", "policy_acceleration_limit_enabled"):
            if not isinstance(limits.get(name, True), bool):
                raise ValueError(f"limits.{name} must be true or false")
        return limits.get("policy_tracking_cap_enabled", True)

    @staticmethod
    def validate_measured_anchor(cfg, policy_target_mode=None):
        """Reject the retired measured-anchor path before creating any device."""
        mode = cfg.get('policy_target_mode', 'legacy') if policy_target_mode is None else policy_target_mode
        if mode != 'legacy' or cfg.get('measured_anchor') is not None:
            raise ValueError('C2_HANDOFF_MEASURED_ANCHOR_REMOVED')
        return None

    def __init__(self, cfg, arm, mouse, policy: Policy | None = None, clock=rospy.get_time,
                 policy_trigger="right-toggle", home_trigger="both-hold", home_timeout_s=30.,
                 policy_target_mode=None, pose_snapshot_provider=None, monotonic_clock=None):
        if policy_trigger not in ("right-toggle", "left-hold"):
            raise ValueError("policy_trigger must be right-toggle or left-hold")
        if home_trigger not in ("both-hold", "other-toggle"):
            raise ValueError("home_trigger must be both-hold or other-toggle")
        if home_trigger == "other-toggle" and (policy is None or policy_trigger != "right-toggle"):
            raise ValueError("other-toggle Home requires a policy and right-toggle")
        if (isinstance(home_timeout_s, bool) or not isinstance(home_timeout_s, (int, float))
                or not 0 < home_timeout_s <= 120):
            raise ValueError("home_timeout_s must be finite and in (0, 120]")
        self.home_trigger = home_trigger
        self.home_timeout_s = float(home_timeout_s)
        # main passes this exact timeout to SpaceMouse; stale reads synthesize zero twist.
        self._home_neutral_report_timeout_s = cfg["spacemouse"]["stale_timeout_s"]
        # Require a complete release after startup, cancellation, completion or fault.
        self._home_gesture_blocked = True
        self._home_press_pending = False
        self._home_latched = False
        self._home_hold = False
        self._home_started_s = None
        self._home_next_progress_s = None
        self._home_status = "idle"
        self._home_progress = None
        self.arm = arm
        self.mouse = mouse
        self.policy = policy
        self.policy_trigger = policy_trigger
        self.clock = clock
        self.validate_measured_anchor(cfg, policy_target_mode)

        self.rate_hz = cfg["control"]["rate_hz"]
        self.dt = 1.0 / self.rate_hz
        self.policy_tracking_cap_enabled = self.validate_tracking_limits(cfg["limits"])
        self.policy_acceleration_limit_enabled = cfg["limits"].get("policy_acceleration_limit_enabled", True)
        self.tracker = make_tracker(cfg["limits"], self.dt)

        teleop = cfg["teleop"]
        self.teleop_lead = np.array(3 * [teleop["max_lead"]] + 3 * [np.deg2rad(teleop["max_lead_angle_deg"])])
        if not teleop["enable_rotation"]:
            self.teleop_lead[3:] = 0.0

        self.max_force = cfg["safety"]["max_force"]
        self.max_torque = cfg["safety"]["max_torque"]

        home = cfg["home"]
        self.home_joints = np.asarray(home["joints"], dtype=float).copy()
        self.home_pose = np.asarray(arm.end_effector(joint_angles=home["joints"]), dtype=float)
        self.home_linear_speed = home["max_linear_speed"]
        self.home_angular_speed = np.deg2rad(home["max_angular_speed_deg"])
        self.home_position_tolerance = home["position_tolerance"]
        self.home_orientation_tolerance = np.deg2rad(home["orientation_tolerance_deg"])
        if not self.tracker.in_workspace(self.home_pose[:3]):
            rospy.logwarn(f"Home position {self.home_pose[:3].round(3)} is outside the workspace box")

        self.policy_ticks = max(1, round(policy.period_s * self.rate_hz)) if policy else 1
        self.right_tap = RightTap()
        # Human chord presses/releases arrive as separate HID reports. Delay
        # solo starts and latch Home until both buttons have been released.
        self.policy_button_grace_s = .2
        self._policy_press_since = None
        self._policy_latched = False
        self._policy_start_pending = False
        self._right_was_down = False
        self._ignore_right_release = False
        self._buttons_down = (False, False)
        self._fault_ack_ready = True
        self._home_chord = False
        # Lightweight diagnostics only; no per-tick file/ROS IO. Counts are
        # cumulative for this controller instance, including across mode resets.
        self._policy_hold_counts = dict(held_valid_target=0, no_source_measured=0,
                                       expired_measured=0, post_step_expired=0)
        self.validate_direction_reset(cfg)
        attach_counts = getattr(policy, "attach_controller_hold_counts", None)
        if callable(attach_counts):
            attach_counts(self._policy_hold_counts)
        target_options = self.validate_target_management(cfg)
        self.target_management = PolicyTargetManager(target_options) if target_options is not None else None
        self._last_finite_published_target = None
        self.mode = Mode.TELEOP
        self._reset_mode_state()
        self.trace_writer = None
        self._trace_context = None
        trace_options = cfg.get('controller_trace')
        if trace_options is not None:
            if (not isinstance(trace_options, dict) or set(trace_options) != {'enabled', 'queue_capacity'}
                    or type(trace_options['enabled']) is not bool):
                raise ValueError('controller_trace requires enabled and queue_capacity')
            if trace_options['enabled']:
                if policy is None or not hasattr(policy, 'output'):
                    raise ValueError('controller trace requires a policy launch output directory')
                from osx_softwrist_bridge.controller_trace import ControllerTraceWriter
                self.trace_writer = ControllerTraceWriter(policy.output,
                    dict(controller_config=cfg, model_binding=getattr(policy.model, 'binding', {}),
                         operator_triggers=dict(policy_trigger=self.policy_trigger, home_trigger=self.home_trigger,
                             home_timeout_s=self.home_timeout_s, home_path='FDCC Cartesian pose',
                             home_joint_error_semantics='diagnostic only; no joint branch or turn guarantee'),
                         policy_target_mode='legacy',
                         sensor_wrench_semantics='arm.get_wrench safety/control state; may be filtered, not raw ROS',
                         source_wrench_semantics='raw model /wrench saved in source NPZ; separate from current safety wrench'),
                    capacity=trace_options['queue_capacity'])
                self.tracker.trace_enabled = True

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    def run(self):
        """Activate the compliance controller and run until SPACE/ESC, shutdown or a protective stop."""
        rate = rospy.Rate(self.rate_hz)
        # URServices methods return True whenever use_real_robot is false (Gazebo), so the
        # protective-stop query is only meaningful on the real robot.
        on_real_robot = self.arm.dashboard_services.use_real_robot
        self.arm.zero_ft_sensor()
        state = self.read_state()
        self.tracker.reanchor(state.pose)
        if not self.tracker.in_workspace(state.pose[:3]):
            rospy.logwarn(f"Start position {state.pose[:3].round(3)} is outside the workspace box; "
                          "only motion back toward it is allowed")
        self.arm.activate_cartesian_controller()
        try:
            with ExitKeys() as keys:
                if keys.active:
                    rospy.loginfo("Teleoperation active: SPACE or ESC to exit")
                else:
                    rospy.loginfo("Teleoperation active: exit with Ctrl-C (stdin is not a terminal)")
                while not rospy.is_shutdown():
                    if keys.pressed():
                        rospy.loginfo("Exit key pressed")
                        break
                    if on_real_robot and self.arm.dashboard_services.is_protective_stopped():
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
            try:
                self.cancel_policy_trigger()
                self.clear_single_home("controller_exit")
                if self.policy:
                    self.policy.deactivate("controller_exit")
                self.arm.activate_joint_trajectory_controller()
            finally:
                try:
                    if self.policy:
                        self.policy.close()
                finally:
                    if self.trace_writer is not None:
                        self.trace_writer.close()

    def tick(self):
        reading = self.mouse.read()
        state = self.read_state()
        if self.trace_writer is not None:
            self._trace_context = dict(tick_monotonic_s=time.monotonic(), reason='normal',
                measured_pose_base_link_tool0=np.asarray(state.pose).tolist(),
                measured_safety_wrench_N_Nm=np.asarray(state.wrench).tolist())
            self.tracker.last_step_diagnostics = None
        self._buttons_down = (reading.left, reading.right)
        right_pressed = reading.right and not self._right_was_down
        self._right_was_down = reading.right
        right_tap = self.right_tap.update(reading.left, reading.right)

        if not self.valid_control_state(state):
            self.invalid_state_fault(state)
            return

        if self.mode is Mode.FAULT:
            # A button held when the fault occurred must be released before a
            # fresh RIGHT gesture can acknowledge it.
            if not self._fault_ack_ready:
                if not (reading.left or reading.right):
                    self._fault_ack_ready = True
            elif right_tap and reading.connected:
                self.arm.zero_ft_sensor()
                self.switch_mode(Mode.TELEOP, self.read_state())
                rospy.loginfo("F/T sensor zeroed, fault cleared; policy remains stopped")
            self.publish_target(self.tracker.pose)
            return

        fault = self.detect_fault(reading, state)
        if fault:
            rospy.logerr(f"{fault}; holding pose. Tap RIGHT to zero F/T and clear fault; policy stays stopped.")
            self.switch_mode(Mode.FAULT, state)
            self.publish_target(self.tracker.pose)
            return

        mode = self.requested_mode(reading, right_pressed, right_tap)
        if right_tap and not self.right_toggle_policy:
            self.switch_mode(Mode.TELEOP, state)
            self.arm.zero_ft_sensor()
            self.tracker.reanchor(self.arm.end_effector())
            self._reset_mode_state()
            rospy.loginfo("F/T sensor zeroed, target re-anchored")
            return

        if mode is not self.mode:
            try:
                self.switch_mode(mode, state)
            except Exception as exc:
                if mode is not Mode.POLICY:
                    raise
                self.policy_fault(state, exc)
                return

        if self.mode is Mode.TELEOP:
            delta = self.teleop_delta(reading, state)
        elif self.mode is Mode.POLICY:
            try:
                delta = self.policy_delta(state)
                if delta is None:
                    # Hold the selected target without integrating limiter
                    # velocity. policy_delta chooses valid target vs measured
                    # reanchor according to the last consumed source deadline.
                    if self._policy_hold_target_is_live and self._policy_source_expired():
                        self._discard_policy_cached_motion(state)
                    self.publish_target(self.tracker.pose)
                    return
            except Exception as exc:
                self.policy_fault(state, exc)
                return
        elif self.mode is Mode.HOMING:
            if self.home_trigger == "other-toggle":
                now = time.monotonic()
                self.report_home_progress(state, now)
                if self._home_started_s is None or now - self._home_started_s >= self.home_timeout_s:
                    self.finish_single_home(state, "timeout")
                    return
            delta = self.homing_delta(state)
            if self.home_trigger == "other-toggle" and self._home_reached:
                self.finish_single_home(state, "tcp_reached")
                return
        else:
            delta = np.zeros(6)
        enforce_tracking_error = self.mode is not Mode.POLICY or self.policy_tracking_cap_enabled
        enforce_acceleration = self.mode is not Mode.POLICY or self.policy_acceleration_limit_enabled
        if self.trace_writer is not None:
            self._trace_context.update(native_substep_delta_m_rad=np.asarray(delta).tolist(),
                raw_goal_xyz_m=(self.tracker.pose[:3] + delta[:3]).tolist(),
                substep_index=(self.policy_ticks-self._policy_ticks_left) if self.mode is Mode.POLICY else None)
        if self.target_management is not None and self.mode is Mode.POLICY:
            delta = self.target_management.apply(delta, self.tracker.pose, state.pose)
        try:
            target = self.tracker.step(
                delta, state.pose, enforce_tracking_error=enforce_tracking_error,
                enforce_acceleration=enforce_acceleration)
        except (ValueError, FloatingPointError) as exc:
            self.policy_fault(state, exc)
            return
        if self.mode is Mode.POLICY:
            try:
                # Computation can cross the deadline after policy_delta's
                # check. Never publish that computed candidate or its queued
                # remainder; reanchor also clears limiter velocity.
                if self._policy_source_expired():
                    self._discard_policy_cached_motion(state, reason="post_step_expired")
                    target = self.tracker.pose
                else:
                    if self.target_management is not None:
                        self.target_management.complete_step(target)
            except Exception as exc:
                self.policy_fault(state, exc)
                return
        self.publish_target(target)

    # ------------------------------------------------------------------
    # Modes
    # ------------------------------------------------------------------

    @property
    def right_toggle_policy(self):
        return self.policy is not None and self.policy_trigger == "right-toggle"

    def cancel_policy_trigger(self):
        self._policy_latched = False
        self._policy_start_pending = False
        self._policy_press_since = None
        self._ignore_right_release = False

    def requested_mode(self, reading, right_pressed=False, right_tap=False):
        if self.home_trigger == "other-toggle":
            return self.requested_single_home_mode(reading, right_pressed, right_tap)
        if reading.left and reading.right:
            self.cancel_policy_trigger()
            self._home_chord = True
            return Mode.HOMING
        if self._home_chord:
            self.cancel_policy_trigger()
            if reading.left or reading.right:
                return Mode.BUTTON_WAIT
            self._home_chord = False
        if self.right_toggle_policy:
            return self.requested_toggle_mode(reading, right_pressed, right_tap)
        if reading.left and self.policy is not None:
            if self._policy_press_since is None:
                self._policy_press_since = self.clock()
            if self.clock()-self._policy_press_since < self.policy_button_grace_s:
                return Mode.BUTTON_WAIT
            return Mode.POLICY
        self._policy_press_since = None
        if reading.left and not self._warned_no_policy:
            rospy.logwarn("LEFT pressed but no policy is loaded")
            self._warned_no_policy = True
        return Mode.TELEOP

    def clear_single_home(self, reason, *, hold=False):
        """Clear only the opt-in Home gesture; never tare or restart policy."""
        self._home_latched = False
        self._home_press_pending = False
        self._home_started_s = None
        self._home_next_progress_s = None
        self._home_gesture_blocked = True
        self._home_hold = hold
        self._home_status = reason

    def requested_single_home_mode(self, reading, right_pressed, right_tap):
        """Code LEFT solo press/release starts Home; any new button stops it.

        Code RIGHT remains the existing policy toggle. Consume chord/cancel
        gestures through full release so their release cannot start either mode.
        """
        down = reading.left or reading.right
        if self._home_gesture_blocked:
            self.cancel_policy_trigger()
            if not down:
                self._home_gesture_blocked = False
            return Mode.HOME_HOLD if self._home_hold else Mode.BUTTON_WAIT
        if self._home_latched:
            if down:
                self.cancel_policy_trigger()
                self.clear_single_home("button_cancelled", hold=True)
                rospy.loginfo("Home stopped by button; release all buttons before a new gesture")
                return Mode.HOME_HOLD
            return Mode.HOMING
        if (reading.left and reading.right) or (self._home_press_pending and reading.right):
            self.cancel_policy_trigger()
            self.clear_single_home("chord_cancelled", hold=True)
            return Mode.HOME_HOLD
        if reading.left:
            # Stop policy on press, but do not move Home until the solo release.
            self.cancel_policy_trigger()
            self._home_press_pending = True
            self._home_hold = False
            return Mode.BUTTON_WAIT
        if self._home_press_pending:
            self.cancel_policy_trigger()
            self._home_press_pending = False
            self._home_latched = True
            self._home_started_s = time.monotonic()
            self._home_next_progress_s = self._home_started_s
            self._home_status = "running"
            self._home_progress = None
            rospy.loginfo(f"Home started: FDCC Cartesian pose, timeout {self.home_timeout_s:g} s; "
                          "any button stops. Joint angles are diagnostic; same branch/turns are not guaranteed.")
            return Mode.HOMING
        if self._home_hold and not down:
            # Require a neutral observation after the consumed button gesture.
            # The neutral tick only reanchors; a later deflection resumes manual control.
            twist = np.asarray(reading.twist, dtype=float)
            age = getattr(reading, "report_age_s", None)
            timeout = self._home_neutral_report_timeout_s
            fresh = (isinstance(age, (int, float)) and not isinstance(age, bool)
                     and isinstance(timeout, (int, float)) and not isinstance(timeout, bool)
                     and 0 < timeout < float("inf") and 0 <= age <= timeout)
            if fresh and twist.shape == (6,) and np.isfinite(twist).all() and not np.any(twist):
                self._home_hold = False
                return Mode.TELEOP
            return Mode.HOME_HOLD
        self._home_hold = False
        return self.requested_toggle_mode(reading, right_pressed, right_tap)

    def finish_single_home(self, state, reason):
        self._home_next_progress_s = None
        self.report_home_progress(state, time.monotonic())
        self.cancel_policy_trigger()
        self.clear_single_home(reason, hold=True)
        self.switch_mode(Mode.HOME_HOLD, state)
        if reason == "timeout":
            rospy.logwarn(f"Home timeout after {self.home_timeout_s:g} s; holding measured pose. "
                          "Release all buttons, then center the mouse to resume manual control; use a fresh gesture for policy/Home.")
        else:
            rospy.loginfo("Home TCP target reached; holding measured pose until buttons are released and mouse is neutral. Joint branch/turns are not verified.")
        if self._trace_context is not None:
            self._trace_context['reason'] = 'home_' + reason
        self.publish_target(self.tracker.pose)

    def report_home_progress(self, state, now):
        if self._home_next_progress_s is not None and now < self._home_next_progress_s:
            return
        position_mm = float(np.linalg.norm(self.home_pose[:3] - state.pose[:3]) * 1000.)
        orientation_deg = float(np.rad2deg(np.linalg.norm(rotation_error(self.home_pose[3:], state.pose[3:]))))
        joint_deg = None
        try:
            joints = np.asarray(state.joints, dtype=float)
            if (joints.shape == (6,) and self.home_joints.shape == (6,)
                    and np.isfinite(joints).all() and np.isfinite(self.home_joints).all()):
                joint_deg = float(np.rad2deg(np.max(np.abs(self.home_joints - joints))))
        except (AttributeError, TypeError, ValueError):
            pass
        joint_text = f"{joint_deg:.2f} deg" if joint_deg is not None else "unavailable (invalid joints)"
        elapsed = now - self._home_started_s if self._home_started_s is not None else 0.
        self._home_progress = dict(tcp_remaining_mm=position_mm, orientation_remaining_deg=orientation_deg,
            max_saved_joint_error_deg=joint_deg, elapsed_s=elapsed, joint_error_is_diagnostic=True)
        rospy.loginfo(f"Home: TCP remaining {position_mm:.2f} mm / orientation {orientation_deg:.2f} deg; "
                      f"max saved-joint difference {joint_text} (diagnostic only); "
                      f"elapsed {elapsed:.1f}/{self.home_timeout_s:g} s")
        self._home_next_progress_s = now + 2.

    def requested_toggle_mode(self, reading, right_pressed, right_tap):
        if right_pressed:
            self._ignore_right_release = self._policy_latched or self._policy_start_pending
            self._policy_press_since = self.clock()
            self._policy_start_pending = False
            if self._ignore_right_release:
                rospy.loginfo("RIGHT toggle: policy stopped")
            self._policy_latched = False
        if reading.left:
            # LEFT is an explicit stop, never a policy start in toggle mode.
            self.cancel_policy_trigger()
            return Mode.BUTTON_WAIT
        if reading.right:
            return Mode.BUTTON_WAIT
        if right_tap:
            if self._ignore_right_release:
                self.cancel_policy_trigger()
            elif self._policy_press_since is not None:
                self._policy_start_pending = True
        if self._policy_start_pending:
            if self.clock() - self._policy_press_since < self.policy_button_grace_s:
                return Mode.BUTTON_WAIT
            self._policy_start_pending = False
            self._policy_latched = True
            home_gesture = ("LEFT solo release returns Home" if self.home_trigger == "other-toggle"
                            else "LEFT+RIGHT returns Home")
            rospy.loginfo(f"RIGHT toggle: policy started; RIGHT or LEFT stops, {home_gesture}")
        return Mode.POLICY if self._policy_latched else Mode.TELEOP

    def switch_mode(self, mode, state):
        if not self.valid_control_state(state):
            self.invalid_state_fault(state)
            return
        if getattr(self, '_trace_context', None) is not None:
            self._trace_context.update(reason='mode_switch_reanchor', previous_mode=self.mode.value)
        rospy.loginfo(f"Mode {self.mode.value} -> {mode.value}")
        if self.mode is Mode.POLICY and self.policy is not None:
            self.policy.deactivate(mode.value)
        self.mode = mode
        if mode is Mode.FAULT:
            self.clear_single_home("fault")
            self.cancel_policy_trigger()
            self._fault_ack_ready = not any(self._buttons_down)
        self.tracker.reanchor(state.pose)
        self._reset_mode_state()
        if mode is Mode.POLICY:
            self.policy.reset(state)

    def policy_fault(self, state, error):
        rospy.logerr(f"Policy failed: {error}; holding measured pose")
        self.switch_mode(Mode.FAULT, state)
        self.publish_target(self.tracker.pose)

    def _reset_mode_state(self):
        if self.target_management is not None:
            self.target_management.reset()
        self._teleop_moving = False
        self._teleop_idle_goal = None
        self._policy_step = np.zeros(6)
        self._policy_ticks_left = 0
        self._home_reached = False
        self._warned_no_policy = False

    def teleop_delta(self, reading, state):
        """Delta toward a goal placed relative to the measured pose, not the commanded target.

        While the mouse is deflected the goal leads the measured pose by max_lead times the
        deflection, so the target cannot run away from a robot that lags behind. On release the
        goal becomes the measured pose at that moment and is held until the mouse moves again.
        """
        lead = reading.twist * self.teleop_lead
        target = self.tracker.pose
        if np.any(lead):
            self._teleop_moving = True
            goal = np.concatenate([state.pose[:3] + lead[:3],
                                   rotate(state.pose[3:], lead[3:]) if np.any(lead[3:]) else target[3:]])
        else:
            if self._teleop_moving:
                self._teleop_moving = False
                orientation = state.pose[3:] if np.any(self.teleop_lead[3:]) else target[3:]
                self._teleop_idle_goal = np.concatenate([state.pose[:3], orientation])
            goal = target if self._teleop_idle_goal is None else self._teleop_idle_goal
        return np.concatenate([goal[:3] - target[:3], rotation_error(goal[3:], target[3:])])

    def _policy_source_expired(self):
        deadline = getattr(self.policy, "action_source_deadline_monotonic_s", None)
        if deadline is None:
            return False  # Historical policies do not declare a source deadline.
        if (not isinstance(deadline, (int, float)) or isinstance(deadline, bool)
                or not np.isfinite(deadline)):
            raise ValueError("POLICY_SOURCE_DEADLINE_INVALID: finite monotonic deadline required")
        return time.monotonic() > deadline

    def _discard_policy_cached_motion(self, state, *, reason="expired_measured"):
        self._policy_step = np.zeros(6)
        self._policy_ticks_left = 0
        self.tracker.reanchor(state.pose)
        if self.target_management is not None:
            self.target_management.reset()
        self._policy_hold_counts[reason] += 1
        if getattr(self, '_trace_context', None) is not None:
            self._trace_context['reason'] = reason


    def policy_delta(self, state):
        self._policy_hold_target_is_live = False
        if self._policy_ticks_left and self._policy_source_expired():
            self._discard_policy_cached_motion(state)
            # Try a newly delivered result on this same tick. act still runs
            # its producer watchdog; an expired result is never reused.
        if self._policy_ticks_left == 0:
            delta = np.asarray(self.policy.act(state), dtype=float)
            if delta.shape != (6,) or not np.isfinite(delta).all():
                raise ValueError("Policy must return six finite Cartesian deltas")
            expired = self._policy_source_expired()
            if expired:
                self._discard_policy_cached_motion(state)
                return None
            if getattr(self.policy, "awaiting_fresh_result", False) is True:
                self._policy_step = np.zeros(6)
                self._policy_ticks_left = 0
                if getattr(self.policy, "action_source_deadline_monotonic_s", None) is None:
                    self._discard_policy_cached_motion(state, reason="no_source_measured")
                else:
                    # A consumed action need not be repeated to retain its
                    # already-issued target while its source is still fresh.
                    # Do not cancel target lead or restart acceleration here.
                    self._policy_hold_counts["held_valid_target"] += 1
                    self._policy_hold_target_is_live = True
                    if getattr(self, '_trace_context', None) is not None:
                        self._trace_context['reason'] = 'waiting_valid_source_hold'
                return None
            if self.target_management is not None:
                self.target_management.observe(delta, state.pose, self.tracker.pose, time.monotonic())
            self._policy_step = delta / self.policy_ticks
            self._policy_ticks_left = self.policy_ticks
        if self._policy_source_expired():
            self._discard_policy_cached_motion(state)
            return None
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

    @staticmethod
    def valid_pose(pose):
        try:
            p = np.asarray(pose, dtype=float)
            return p.shape == (7,) and np.isfinite(p).all() and np.linalg.norm(p[3:]) > 1e-8
        except (TypeError, ValueError):
            return False

    @classmethod
    def valid_control_state(cls, state):
        try:
            w = np.asarray(state.wrench, dtype=float)
            return cls.valid_pose(state.pose) and w.shape == (6,) and np.isfinite(w).all()
        except (AttributeError, TypeError, ValueError):
            return False

    def invalid_state_fault(self, state=None):
        # Never reanchor to NaN/Inf. No tare, retry, or automatic restart.
        if self._trace_context is not None:
            self._trace_context['reason'] = 'invalid_state'
        if self.mode is Mode.POLICY and self.policy is not None:
            self.policy.deactivate("nonfinite_control_state")
        self.cancel_policy_trigger()
        self.clear_single_home("invalid_state")
        self.mode = Mode.FAULT
        self._fault_ack_ready = not any(self._buttons_down)
        self._reset_mode_state()
        if hasattr(self.tracker, 'limiter'):
            self.tracker.limiter.reset()
        hold = self._last_finite_published_target
        if state is not None and self.valid_pose(getattr(state, 'pose', None)):
            # A valid measured pose can still discharge lead when only wrench
            # is invalid, exactly as the existing safety-fault hold does.
            hold = np.asarray(state.pose, dtype=float).copy()
            self.tracker.reanchor(hold)
        # With invalid pose, never treat an unissued tracker candidate as an
        # executed/dispatched target. No known dispatched pose => no publish.
        rospy.logerr_throttle(1, "Invalid measured control state; policy stopped; holding last finite target if available")
        if hold is not None:
            if self._trace_context is not None:
                self._trace_context['reason'] = 'invalid_state_hold'
            self.tracker.pose = hold.copy()
            self.publish_target(hold)
        elif self.trace_writer is not None:
            self.trace_writer.submit(dict(self._trace_context or {},
                event='invalid_state_no_dispatch', mode=self.mode.value, physical_ack=False))

    def publish_target(self, target):
        if not self.valid_pose(target):
            self.invalid_state_fault()
            return
        try:
            self.arm.set_cartesian_target_pose(target)
        except Exception as exc:
            if self.trace_writer is not None:
                self.trace_writer.submit(dict(self._trace_context or {}, event='dispatch_failed',
                    candidate_target_base_link_tool0=np.asarray(target).tolist(),
                    dispatch_attempt_monotonic_s=time.monotonic(), error=repr(exc)[:512],
                    mode=self.mode.value, physical_ack=False))
            raise  # Preserve existing exception/stop path, never acknowledge.
        # Dispatch return is not a physical execution acknowledgement.
        self._last_finite_published_target = np.asarray(target, dtype=float).copy()
        if self.trace_writer is not None:
            # Only after the real/stub publication API returned successfully.
            # This is a dispatch record, never robot execution acknowledgement.
            row = dict(self._trace_context or {}, event='dispatch_returned', dispatch_return_monotonic_s=time.monotonic(),
                mode=self.mode.value, published_target_base_link_tool0=np.asarray(target).tolist(),
                source=getattr(self.policy, 'controller_source_context', None) if self.mode is Mode.POLICY else None,
                target_layers=self.tracker.last_step_diagnostics, physical_ack=False)
            if self.target_management is not None:
                row['target_management'] = dict(blocked=self.target_management.blocked.tolist(),
                    release=self.target_management.release.tolist(),reason=list(self.target_management.last_reason))
            row['home_control'] = dict(trigger=self.home_trigger, timeout_s=self.home_timeout_s,
                status=self._home_status, latched=self._home_latched, progress=self._home_progress)
            self.trace_writer.submit(row)

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
    """Print the mapped base_link command until SPACE, ESC or Ctrl-C."""
    previous_count = mouse.report_count
    try:
        with ExitKeys() as keys:
            while not keys.pressed():
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


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=None, help='Controller YAML; a policy launch defaults to the retained C2 YAML')
    parser.add_argument('--mouse-only', action='store_true', help='Read mouse only; no ROS/Arm initialization')
    parser.add_argument('--home-config', type=Path, help='Saved Home JSON; startup does not move Home')
    parser.add_argument('--home-trigger', choices=('both-hold', 'other-toggle'), default='both-hold')
    parser.add_argument('--home-timeout-s', type=float, default=30.)
    parser.add_argument('--system', choices=(
        'surface035_measured140_tool445_field_hole_only_task35_z0_time_aligned_250ms_teacher_fz_flip',
        'c2_force_div500'), help='Original C2 or identical paired models with policy force divisor 500')
    parser.add_argument('--teacher', choices=('teacher_paired_surface035_measured140_tool445',), default=None)
    parser.add_argument('--policy-trigger', choices=('right-toggle', 'left-hold'), default='right-toggle')
    parser.add_argument('--videomode', choices=('2d', '3d'), default='2d')
    parser.add_argument('--startup-preview', choices=('3d', 'off'), default=None)
    args = parser.parse_args(argv)
    if not 0 < args.home_timeout_s <= 120:
        parser.error('--home-timeout-s must be finite and in (0, 120]')
    if args.mouse_only and (args.system or args.home_config or args.teacher):
        parser.error('--mouse-only cannot select a policy or Home')
    if args.teacher and not args.system:
        parser.error('--teacher requires --system')
    if args.home_trigger == 'other-toggle' and (not args.system or args.policy_trigger != 'right-toggle'):
        parser.error('--home-trigger other-toggle requires --system and --policy-trigger right-toggle')
    if (args.videomode == '3d' or args.startup_preview == '3d') and not args.system:
        parser.error('Policy video/preview requires --system')
    args.teacher = 'teacher_paired_surface035_measured140_tool445'
    args.policy_config = args.profile = args.policy_target_mode = args.d1_peg_depth_mode = None
    args.execute_native_policy = False
    if args.config is None:
        args.config = (Path(__file__).resolve().parents[6] / 'configs/rri/local/control-fdcc-z425mm-c2-target-management.yaml'
                       if args.system else DEFAULT_CONFIG)
    return args


def policy_bridge():
    # The ROS core also works as a standalone repo. Only a policy launch needs
    # the parent bridge; bootstrap it without requiring shell-specific PYTHONPATH.
    for parent in Path(__file__).resolve().parents:
        bridge = parent / "bridge/src"
        if (bridge / "osx_softwrist_bridge/spacemouse_policy.py").is_file():
            sys.path.insert(0, str(bridge))
            break
    from osx_softwrist_bridge import spacemouse_policy
    return spacemouse_policy


def home_bridge():
    # Manual teleoperation can select Home without loading policy/model modules.
    for parent in Path(__file__).resolve().parents:
        bridge = parent / "bridge/src"
        if (bridge / "osx_softwrist_bridge/spacemouse_home.py").is_file():
            sys.path.insert(0, str(bridge))
            break
    from osx_softwrist_bridge import spacemouse_home
    return spacemouse_home


def preflight_selected_policy(args):
    return policy_bridge().preflight_policy(args.policy_config, profile=args.profile,
                       execute_native=args.execute_native_policy, system=args.system, teacher=args.teacher,
                       d1_peg_depth_mode=args.d1_peg_depth_mode)


def load_selected_policy(args):
    return policy_bridge().load_policy(args.policy_config, profile=args.profile,
                       execute_native=args.execute_native_policy, system=args.system, teacher=args.teacher,
                       videomode=args.videomode, d1_peg_depth_mode=args.d1_peg_depth_mode,
                       startup_preview=args.startup_preview)


def main():
    args = parse_arguments(rospy.myargv()[1:])
    controller_config_data = args.config.read_bytes()
    source_controller_config = yaml.safe_load(controller_config_data)
    cfg = source_controller_config
    if args.policy_target_mode is not None:
        cfg = dict(cfg, policy_target_mode=args.policy_target_mode)
    if (args.execute_native_policy or args.system) and (cfg["robot"]["base_link"] != "base_link" or cfg["robot"]["ee_link"] != "tool0"):
        raise ValueError("Policy control requires base_link / tool0 controller frames")
    SpaceMouseController.validate_measured_anchor(cfg)
    SpaceMouseController.validate_direction_reset(cfg)
    if args.system and SpaceMouseController.validate_target_management(cfg) is None:
        raise ValueError('C2_HANDOFF_REQUIRES_C2_TARGET_MANAGEMENT')
    home_selection = None
    if args.home_config:
        cfg, home_selection = home_bridge().select_home(args.home_config, cfg)
        gesture = "code LEFT solo press-release returns Home" if args.home_trigger == "other-toggle" else "LEFT+RIGHT hold returns Home"
        print(f"Home selected for this launch: {home_selection['name']} ({home_selection['path']}); "
              f"{gesture}; startup does not move Home.", flush=True)

    if args.system:
        preflight_selected_policy(args)

    policy_cap_enabled = SpaceMouseController.validate_tracking_limits(cfg["limits"])
    target_options = (SpaceMouseController.validate_target_management(cfg)
                      if "policy_target_management" in cfg else None)
    if not policy_cap_enabled:
        print("POLICY target-to-measured tracking cap DISABLED for this launch; "
              f"manual/Home cap remains {cfg['limits']['max_tracking_error'] * 1000:g} mm. "
              "Workspace, velocity, force/torque and data timeout limits remain active; "
              f"POLICY acceleration clipping is {'enabled' if cfg['limits'].get('policy_acceleration_limit_enabled', True) else 'disabled'}.", flush=True)
    if not cfg["limits"].get("policy_acceleration_limit_enabled", True):
        print("POLICY acceleration clipping DISABLED for this launch; manual/Home acceleration "
              "limits remain active. Velocity, workspace, force/torque and data timeouts remain active.",
              flush=True)

    if args.home_trigger == "other-toggle":
        print(f"Home trigger other-toggle: code LEFT press stops policy; solo release starts FDCC Cartesian Home "
              f"(timeout {args.home_timeout_s:g} s); any button stops Home. Code RIGHT policy toggle unchanged. "
              "Physical left/right require identification. Release all buttons after startup. "
              "TCP completion does not guarantee saved joint branch/turns; completion/timeout holds pose until full release and neutral mouse.", flush=True)
    mouse = SpaceMouse(**cfg["spacemouse"])
    policy = None
    try:
        if args.mouse_only:
            run_mouse_only(mouse)
            return
        rospy.init_node("spacemouse_control_fdcc")
        if home_selection is not None:
            home_selection = home_bridge().verify_selected_home(home_selection, rospy)
        if args.policy_config or args.system:
            policy = load_selected_policy(args)
            if args.config != DEFAULT_CONFIG:
                controller_receipt = home_bridge().record_controller_config(
                    args.config, controller_config_data, source_controller_config, cfg, policy.output)
                rospy.loginfo(f"Selected controller config: {controller_receipt}")
            if home_selection is not None:
                home_receipt = home_bridge().record_home_selection(home_selection, policy.output)
                rospy.loginfo(f"Selected Home {home_selection['name']}: {home_receipt}")
            mode = "motion enabled" if args.execute_native_policy or args.system else "observation only"
            trigger = ("RIGHT tap-release starts / RIGHT press stops (no F/T zero); LEFT also stops"
                       if args.policy_trigger == "right-toggle" else "LEFT hold starts / release stops; RIGHT zeros F/T")
            home_gesture = ("code LEFT solo press-release Home; any button stops Home"
                            if args.home_trigger == "other-toggle" else "LEFT+RIGHT hold Home")
            rospy.logwarn(f"{policy.system} / {policy.teacher}: {trigger}; {mode}. "
                          "Fault: RIGHT only zeros F/T and clears fault; a separate click is needed to start. "
                          f"Manual teleoperation and {home_gesture} still move the robot.")
        arm = CompliantController(gripper_type=None, **cfg["robot"])
        if not arm.dashboard_services.activate_ros_control_on_ur():
            raise RuntimeError("Failed to activate ROS control on the UR")
        configure_controller(arm, **cfg["controller"])
        SpaceMouseController(cfg, arm, mouse, policy=policy, policy_trigger=args.policy_trigger,
                             home_trigger=args.home_trigger, home_timeout_s=args.home_timeout_s).run()
    finally:
        try:
            if policy is not None:
                policy.close()
        finally:
            mouse.close()


if __name__ == "__main__":
    main()
