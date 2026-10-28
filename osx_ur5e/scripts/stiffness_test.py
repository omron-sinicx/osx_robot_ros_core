#!/usr/bin/env python3
"""Manual test: hold the UR5e's current pose via cartesian_compliance_controller
and let you type in different stiffness values live, to feel how each one
responds -- unlike dynamic_compliance_test.py, nothing here reacts to
deflection or moves the target pose; the arm just holds still at whatever
stiffness you last entered.

WHY THIS DOESN'T USE ur_control.fzi_cartesian_compliance_controller.CompliantController:
historical -- when this script was written, that module's unconditional
`from cartesian_compliance_controller.srv import SetStiffness` at import time
could not be satisfied, because the cartesian_compliance_controller package
vendored in this repo's underlay_ws did not define that service. THAT IS NO
LONGER TRUE: the submodule now ships SetStiffness.srv (generated into
underlay_ws/devel), so CompliantController imports fine and scripts that use it
(data_collection.py, evaluate_policy.py, deformation_measurement.py) work. This
script simply has no reason to be rewritten -- it builds directly on
`ur_control.arm.Arm` plus raw
`dynamic_reconfigure` clients and `ur_control.fzi_utils`'s parameter-dict
converters, exactly mirroring dynamic_compliance_test.py's activation
sequence: zero_ft_sensor -> configure (PD gains, selection matrix, solver
params) -> switch to the compliance controller -> publish current pose as a
one-time hold target (never re-published after that -- there is no pose
tracking in this script).

The control-mode defaults (p_gains, d_gains, selection_matrix, error_scale,
iterations) mirror config/hydra/controller/ur5e.yaml, hardcoded here so this
script has no Hydra dependency -- same as dynamic_compliance_test.py.

Usage:
    rosrun osx_ur5e stiffness_test.py
    rosrun osx_ur5e stiffness_test.py _stiffness_trans:=500 _stiffness_rot:=30

Then, at the prompt, type either:
    "<trans> <rot>"           -- e.g. "500 30"  -> N/m applied to x/y/z, Nm/rad to roll/pitch/yaw
    "<x> <y> <z> <rx> <ry> <rz>"  -- 6 values, one per axis
    Enter (blank)              -- re-print the current stiffness
    q                          -- quit (returns to joint trajectory controller)

Params (rospy private, ~):
    ~stiffness_trans   initial N/m applied to x/y/z (default 3000, matching
                       controller/ur5e.yaml's stiffness_trans)
    ~stiffness_rot     initial Nm/rad applied to roll/pitch/yaw (default 50,
                       matching controller/ur5e.yaml's stiffness_rot -- see
                       that file's comment for why this is so much lower
                       than stiffness_trans: comparable *felt* softness needs
                       roughly stiffness_rot ~= stiffness_trans * r^2, where
                       r is the distance (m) from the wrist's rotation
                       center to where force is applied)
    ~mode              control mode, "spring-mass-damper" or "parallel"
                       (default "spring-mass-damper")
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

PD_GAIN_AXES = ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z")

# Mirrors config/hydra/controller/ur5e.yaml -- hardcoded so this script has no
# Hydra dependency. Same as dynamic_compliance_test.py.
P_GAINS = [0.05, 0.05, 0.05, 0.1, 0.1, 0.1]
D_GAINS = [0, 0, 0, 0, 0, 0]
SELECTION_MATRIX = [1, 1, 1, 1, 1, 1]
ERROR_SCALE = 0.9
ITERATIONS = 1.0


def parse_stiffness(text: str, current: np.ndarray):
    """'<trans> <rot>' -> uniform 6-vector, '<6 values>' -> per-axis, '' ->
    current unchanged. Raises ValueError on anything else (caller reprompts)."""
    tokens = text.replace(",", " ").split()
    if not tokens:
        return current
    values = [float(t) for t in tokens]
    if len(values) == 2:
        trans, rot = values
        return np.array([trans] * 3 + [rot] * 3)
    if len(values) == 6:
        return np.array(values)
    raise ValueError(f"expected 2 values (trans rot) or 6 (per-axis), got {len(values)}")


class StiffnessTest:
    def __init__(self):
        rospy.init_node("stiffness_test")

        self.mode = rospy.get_param("~mode", "spring-mass-damper")
        stiffness_trans = rospy.get_param("~stiffness_trans", 3000.0)
        stiffness_rot = rospy.get_param("~stiffness_rot", 50.0)
        self.stiffness = np.array([stiffness_trans] * 3 + [stiffness_rot] * 3)

        rospy.loginfo("stiffness_test: bringing up compliance controller "
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
            raise RuntimeError("stiffness_test: failed to activate cartesian_compliance_controller")
        self.arm.running_controller_name = ctrl  # so activate_joint_trajectory_controller() works on shutdown

        # Hold pose is published exactly once here -- nothing in this script
        # ever moves the target again, unlike dynamic_compliance_test.py.
        hold_pose = self.arm.end_effector()
        msg = conversions.to_pose_stamped(self.arm.base_link, hold_pose)
        msg.header.stamp = rospy.Time.now()
        self.target_pub.publish(msg)
        rospy.loginfo(f"stiffness_test: holding pose {np.round(hold_pose, 4)} (mode={self.mode})")

        rospy.on_shutdown(self._shutdown)
        self._send(self.stiffness)

    def _send(self, stiffness: np.ndarray) -> None:
        params = convert_stiffness_to_parameters(stiffness.astype(np.int64))["stiffness"]
        self.stiffness_client.update_configuration(params)
        self.stiffness = stiffness
        rospy.loginfo(f"stiffness_test: stiffness -> trans={np.round(stiffness[:3], 0)} N/m  "
                      f"rot={np.round(stiffness[3:], 0)} Nm/rad")

    def run(self) -> None:
        print("\nEnter stiffness as '<trans> <rot>' (N/m Nm/rad) or 6 per-axis values, "
              "blank to reprint current, 'q' to quit.\n")
        while not rospy.is_shutdown():
            try:
                text = input("stiffness> ").strip()
            except EOFError:
                break
            if text.lower() in ("q", "quit", "exit"):
                break
            try:
                new_stiffness = parse_stiffness(text, self.stiffness)
            except ValueError as e:
                print(f"  invalid input ({e}) -- try again.")
                continue
            if np.array_equal(new_stiffness, self.stiffness) and text:
                continue
            self._send(new_stiffness)

    def _shutdown(self):
        rospy.loginfo("stiffness_test: returning to joint trajectory controller...")
        self.arm.activate_joint_trajectory_controller()


if __name__ == "__main__":
    node = StiffnessTest()
    try:
        node.run()
    finally:
        rospy.signal_shutdown("stiffness_test: exiting")
