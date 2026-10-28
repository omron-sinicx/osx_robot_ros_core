#!/usr/bin/env python3
"""Manual test: hold the UR5e's current pose via cartesian_compliance_controller
while modulating its stiffness live, PER AXIS, based on the MLP-estimated wrist
deflection (see bendlabs/scripts/estimate_pose_node.py, topic /estimated_pose by
default -- a geometry_msgs/PoseStamped already relative to a zeroed origin:
position is displacement-from-start, orientation is rotation-since-start).

Physically deflect the wrist and feel the arm get softer the further the
estimated deflection is from its zeroed origin, and stiffer near it.

WHY THIS DOESN'T USE ur_control.fzi_cartesian_compliance_controller.CompliantController:
historical. When this script was written, that module's unconditional
`from cartesian_compliance_controller.srv import SetStiffness` at import time
could not be satisfied: the cartesian_compliance_controller package vendored in
this repo's underlay_ws did not define that service (its CMakeLists.txt had the
srv-generation section commented out), so neither it nor any script using
CompliantController could be imported at all.

THAT IS NO LONGER TRUE. The vendored submodule now ships SetStiffness.srv and
generates it into underlay_ws/devel, so CompliantController imports cleanly and
data_collection.py / evaluate_policy.py / deformation_measurement.py work as
written. Nothing below needs to change -- the bare-Arm approach is equivalent
and still correct -- but do not propagate the old claim to new scripts: prefer
CompliantController for new code.

This script builds directly on `ur_control.arm.Arm` plus raw
`dynamic_reconfigure` clients and `ur_control.fzi_utils`'s parameter-dict
converters, reproducing exactly what CompliantController's relevant methods do
(same topic/parameter-server names, same dict schemas) without importing the
broken module. Reference behavior mirrored 1:1 from
fzi_cartesian_compliance_controller.py: `set_cartesian_target_pose`
(-> `conversions.to_pose_stamped(self.base_link, pose)` on `<ns><ctrl>/target_frame`),
`update_stiffness`'s 6-vector path (-> `fzi_utils.convert_stiffness_to_parameters`
on the `<ns><ctrl>/stiffness` dynamic_reconfigure client), `set_control_mode` /
`update_selection_matrix` (both live on that same "stiffness" reconfigure
config), `update_pd_gains` (one reconfigure client per axis under
`<ns><ctrl>/pd_gains/<axis>`), `set_solver_parameters` (`<ns><ctrl>/solver`),
and `activate_joint_trajectory_controller` (`Arm`'s own method, already usable
as-is). Activation order also mirrors
ur_control/scripts/cartesian_compliance_controller_examples.py::enable_compliance_control
(also used by FDCCEnv): zero_ft_sensor -> configure -> switch to the
compliance controller -> publish current pose as target. The FZI C++
controller itself resets its target to the current pose on start, but a stale
queued target_frame message (subscriber queue_size=3) could otherwise override
that, so the current pose is defensively republished right after activation,
same as every example in this repo.

The control-mode defaults (p_gains, d_gains, selection_matrix, error_scale,
iterations) mirror config/hydra/controller/ur5e.yaml, hardcoded here so this
script has no Hydra dependency.

Ctrl-C returns the arm to the joint trajectory controller before exiting.

Params (rospy private, ~):
    ~pose_topic       relative pose to track (default "/estimated_pose")
    ~mode             control mode, "spring-mass-damper" or "parallel"
                      (default "spring-mass-damper")
    ~rate             stiffness-update rate, Hz (default 40.0)
    ~min_change       skip an update if every axis changed less than this,
                      N/m or Nm/rad (default 5.0)
    ~trans_near       [m] |displacement| at/below which translation axes are
                      at max stiffness -- also the SOFT->STIFF unlatch point
                      for ~profile=step (default 0.003; kept > 0 since the
                      pose estimate is noisy and rarely reads back exactly 0)
    ~trans_far        [m] |displacement| at/above which translation axes are
                      at min stiffness (default 0.01)
    ~rot_near         [rad] rotation angle at/below which rotation axes are
                      at max stiffness -- same unlatch role as ~trans_near
                      (default 0.05)
    ~rot_far          [rad] rotation angle at/above which rotation axes are
                      at min stiffness (default 0.5)
    ~trans_stiffness_min / ~trans_stiffness_max   N/m (default 300 / 3000)
    ~rot_stiffness_min / ~rot_stiffness_max       Nm/rad (default 10 / 150 --
                      much lower than trans's ceiling on purpose: comparable
                      *felt* softness needs roughly stiffness_rot ~=
                      stiffness_trans * r^2, where r is the distance (m) from
                      the wrist's rotation center to where force is applied
                      -- see controller/ur5e.yaml's stiffness_trans/
                      stiffness_rot comment for the full reasoning)
    ~profile          "linear" / "tanh" (continuous ramp between near and far)
                      or "step" (2-stage: only s_max or s_min, no values in
                      between -- near/far double as a hysteresis band: STIFF
                      -> SOFT once dist >= far, back SOFT -> STIFF only once
                      dist <= near, holds its last stage in between so it
                      doesn't chatter at a single threshold) (default "linear")
    ~pose_timeout     seconds without a message before the fail-safe kicks in
                      (default 0.5)
    ~duration         auto-stop after this many seconds, 0 = run until Ctrl-C
                      (default 0)

Usage:
    rosrun osx_ur5e dynamic_compliance_test.py
    rosrun osx_ur5e dynamic_compliance_test.py _pose_topic:=/mocap_relative_pose _trans_far:=0.08
    rosrun osx_ur5e dynamic_compliance_test.py _profile:=step   # 2-stage instead of a ramp
"""
import numpy as np
import rospy
import dynamic_reconfigure.client
from geometry_msgs.msg import PoseStamped

from ur_control.arm import Arm
from ur_control import conversions
from ur_control.constants import CARTESIAN_COMPLIANCE_CONTROLLER
from ur_control.fzi_utils import (
    convert_stiffness_to_parameters,
    convert_pd_gains_to_parameters,
    convert_selection_matrix_to_parameters,
)
from osx_ur5e.compliance_utils import quat_to_rotvec_abs, distance_to_stiffness, step_with_hysteresis

PD_GAIN_AXES = ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z")

# Mirrors config/hydra/controller/ur5e.yaml -- hardcoded so this script has no
# Hydra dependency.
P_GAINS = [0.05, 0.05, 0.05, 0.1, 0.1, 0.1]
D_GAINS = [0, 0, 0, 0, 0, 0]
SELECTION_MATRIX = [1, 1, 1, 1, 1, 1]
ERROR_SCALE = 0.9
ITERATIONS = 1.0


class DynamicComplianceTest:
    def __init__(self):
        rospy.init_node("dynamic_compliance_test")

        self.pose_topic = rospy.get_param("~pose_topic", "/estimated_pose")
        self.mode = rospy.get_param("~mode", "spring-mass-damper")
        rate_hz = rospy.get_param("~rate", 40.0)
        self.min_change = rospy.get_param("~min_change", 5.0)

        # near defaults are NOT 0.0: the MLP pose estimate is noisy (recall its
        # ~4-5mm / ~2-4deg typical error) and essentially never reads back
        # exactly 0 even when fully released, so near=0 would make the
        # SOFT->STIFF hysteresis unlatch condition (dist<=near, profile=step)
        # practically unreachable -- once soft, stuck soft. These defaults
        # give a real dead-zone above that noise floor; watch the ~1Hz
        # "deflection:" log while at rest and raise them further if needed.
        self.trans_near = rospy.get_param("~trans_near", 0.003)
        self.trans_far = rospy.get_param("~trans_far", 0.02)
        self.rot_near = rospy.get_param("~rot_near", 0.05)
        self.rot_far = rospy.get_param("~rot_far", 0.5)

        self.trans_min = rospy.get_param("~trans_stiffness_min", 8000.0)
        self.trans_max = rospy.get_param("~trans_stiffness_max", 8000.0)
        # Much lower than trans_min/max -- see controller/ur5e.yaml's
        # stiffness_trans/stiffness_rot comment: comparable *felt* softness
        # needs roughly stiffness_rot ~= stiffness_trans * r^2 (r = distance
        # from the wrist's rotation center to where force is applied), not
        # the same raw N/m-vs-Nm/rad number.
        self.rot_min = rospy.get_param("~rot_stiffness_min", 10.0)
        self.rot_max = rospy.get_param("~rot_stiffness_max", 150.0)
        self.profile = rospy.get_param("~profile", "linear")
        self.pose_timeout = rospy.get_param("~pose_timeout", 0.5)
        duration = rospy.get_param("~duration", 0.0)

        self.max_stiffness = np.array([self.trans_max] * 3 + [self.rot_max] * 3)
        self.last_pose = None  # (pos: np.array(3), quat: np.array(4))
        self.last_pose_stamp = None
        self.last_sent = None
        # per-axis STIFF/SOFT latch for ~profile=step (hysteresis state); starts
        # STIFF (not soft), matching the zeroed pose at startup.
        self._trans_soft = np.zeros(3, dtype=bool)
        self._rot_soft = np.zeros(3, dtype=bool)

        rospy.loginfo("dynamic_compliance_test: bringing up compliance controller "
                      "(bare Arm + raw dynamic_reconfigure -- see module docstring)...")
        self.arm = Arm(ee_link="tool0", base_link="base_link", gripper_type=None)

        ns = self.arm.ns
        ctrl = CARTESIAN_COMPLIANCE_CONTROLLER
        self.stiffness_client = dynamic_reconfigure.client.Client(f"{ns}{ctrl}/stiffness", timeout=10)
        self.solver_client = dynamic_reconfigure.client.Client(f"{ns}{ctrl}/solver", timeout=10)
        self.pd_gain_clients = {
            axis: dynamic_reconfigure.client.Client(f"{ns}{ctrl}/pd_gains/{axis}", timeout=10)
            for axis in PD_GAIN_AXES
        }
        self.target_pub = rospy.Publisher(f"{ns}{ctrl}/target_frame", PoseStamped, queue_size=10)

        # control mode + selection matrix both live on the controller's "stiffness"
        # dynamic_reconfigure config (see ComplianceController.cfg)
        selection_params = convert_selection_matrix_to_parameters(SELECTION_MATRIX)["stiffness"]
        self.stiffness_client.update_configuration({
            "use_parallel_force_position_control": self.mode == "parallel",
            **selection_params,
        })
        for axis, params in convert_pd_gains_to_parameters(P_GAINS, D_GAINS).items():
            self.pd_gain_clients[axis].update_configuration(params)
        self.solver_client.update_configuration({
            "error_scale": ERROR_SCALE, "iterations": ITERATIONS, "publish_state_feedback": True,
        })

        self.arm.zero_ft_sensor()
        if not self.arm.controller_manager.switch_controllers(
                controllers_on=[ctrl], controllers_off=[self.arm.running_controller_name]):
            raise RuntimeError("dynamic_compliance_test: failed to activate cartesian_compliance_controller")
        self.arm.running_controller_name = ctrl  # so activate_joint_trajectory_controller() works on shutdown

        hold_pose = self.arm.end_effector()
        self._publish_target(hold_pose)
        rospy.loginfo(f"dynamic_compliance_test: holding pose {np.round(hold_pose, 4)}, "
                      f"tracking {self.pose_topic} (mode={self.mode})")

        rospy.on_shutdown(self._shutdown)
        rospy.Subscriber(self.pose_topic, PoseStamped, self._pose_cb, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / rate_hz), self._update)
        rospy.Timer(rospy.Duration(1.0), self._log_status)

        if duration > 0:
            rospy.Timer(rospy.Duration(duration), lambda _e: rospy.signal_shutdown(
                f"~duration ({duration}s) elapsed"), oneshot=True)

    def _publish_target(self, pose):
        msg = conversions.to_pose_stamped(self.arm.base_link, pose)
        msg.header.stamp = rospy.Time.now()
        self.target_pub.publish(msg)

    def _pose_cb(self, msg):
        p = msg.pose.position
        o = msg.pose.orientation
        self.last_pose = (np.array([p.x, p.y, p.z]), np.array([o.x, o.y, o.z, o.w]))
        self.last_pose_stamp = rospy.get_time()

    def _send(self, stiffness):
        if self.last_sent is not None and np.all(np.abs(stiffness - self.last_sent) < self.min_change):
            return
        params = convert_stiffness_to_parameters(stiffness.astype(np.int64))["stiffness"]
        self.stiffness_client.update_configuration(params)
        self.last_sent = stiffness

    def _update(self, _event):
        if self.last_pose is None:
            return  # nothing received yet -- don't command anything

        if rospy.get_time() - self.last_pose_stamp > self.pose_timeout:
            rospy.logwarn_throttle(
                2.0, f"dynamic_compliance_test: no pose on {self.pose_topic} for "
                     f">{self.pose_timeout}s -- forcing max stiffness (fail-safe)")
            self._trans_soft[:] = False  # reset the step latch so we resume STIFF, not
            self._rot_soft[:] = False    # stuck thinking we're still SOFT from before the dropout
            self._send(self.max_stiffness)
            return

        pos, quat = self.last_pose
        d_trans = np.abs(pos)
        d_rot = quat_to_rotvec_abs(*quat)

        if self.profile == "step":
            stiff_trans, self._trans_soft = step_with_hysteresis(
                d_trans, self.trans_near, self.trans_far, self.trans_min, self.trans_max, self._trans_soft)
            stiff_rot, self._rot_soft = step_with_hysteresis(
                d_rot, self.rot_near, self.rot_far, self.rot_min, self.rot_max, self._rot_soft)
        else:
            stiff_trans = distance_to_stiffness(d_trans, self.trans_near, self.trans_far,
                                                self.trans_min, self.trans_max, self.profile)
            stiff_rot = distance_to_stiffness(d_rot, self.rot_near, self.rot_far,
                                              self.rot_min, self.rot_max, self.profile)
        self._send(np.concatenate([stiff_trans, stiff_rot]))

    def _log_status(self, _event):
        if self.last_pose is None or self.last_sent is None:
            return
        pos, quat = self.last_pose
        rospy.loginfo(
            f"deflection: pos={np.round(np.abs(pos), 4)} m  "
            f"rot={np.round(quat_to_rotvec_abs(*quat), 3)} rad  "
            f"-> stiffness trans={np.round(self.last_sent[:3], 0)} N/m  "
            f"rot={np.round(self.last_sent[3:], 0)} Nm/rad"
        )

    def _shutdown(self):
        rospy.loginfo("dynamic_compliance_test: returning to joint trajectory controller...")
        self.arm.activate_joint_trajectory_controller()


if __name__ == "__main__":
    DynamicComplianceTest()
    rospy.spin()
