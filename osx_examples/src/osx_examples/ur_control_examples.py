#!/usr/bin/env python3
"""Direct ur_control arm examples without motion planning (use with care).

Usage:
    ros2 run osx_examples ur_control_examples

Requires a running gz-sim bringup (ur_gripper_gz or osx_ur5e gz_bringup).
"""

import argparse
import signal
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from ur_control.arm import Arm
from ur_control.constants import GripperType, IKSolverType
from ur_control.gripper_configs import GRIPPER_CONFIGS, apply_gripper_config, read_active_gripper

np.set_printoptions(linewidth=np.inf, suppress=True)


def _signal_handler(sig, frame):
    print("Interrupted — shutting down.")
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)


def confirm_to_proceed(message: str) -> None:
    input("%s — press Enter to continue..." % message)


def main(args=None):
    parser = argparse.ArgumentParser(description="ur_control direct motion examples")
    parser.add_argument(
        "--gripper",
        default="auto",
        choices=["auto", "none"] + sorted(GRIPPER_CONFIGS.keys()),
        help="Gripper config (auto reads /active_gripper from bringup)",
    )
    parser.add_argument(
        "--no-use-gazebo-sim",
        action="store_false",
        dest="use_gazebo_sim",
        help="Disable use_gazebo_sim node parameter",
    )
    parser.set_defaults(use_gazebo_sim=True)
    parser.add_argument(
        "--real-robot",
        action="store_true",
        default=False,
        help="Target a real robot instead of simulation",
    )

    argv = remove_ros_args(args if args is not None else sys.argv)
    cli = parser.parse_args(argv[1:])

    rclpy.init(args=args)
    node = Node("ur_control_examples")
    node.declare_parameter("use_gazebo_sim", cli.use_gazebo_sim and not cli.real_robot)
    node.declare_parameter("use_real_robot", cli.real_robot)
    if cli.use_gazebo_sim and not cli.real_robot:
        node.declare_parameter("use_sim_time", True)

    gripper_name = cli.gripper
    if gripper_name == "auto":
        gripper_name = read_active_gripper(node) or "none"
    if gripper_name != "none":
        apply_gripper_config(node, gripper_name)

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    time.sleep(1.0)

    try:
        arm = Arm(
            node,
            gripper_type=GripperType.GENERIC if gripper_name != "none" else None,
            ik_solver=IKSolverType.EAIK,
            robot_version="UR5e",
        )
        arm.activate_joint_trajectory_controller()

        print("Current joint configuration:", np.round(arm.joint_angles(), 4).tolist())
        print("Current EEF pose:", np.round(arm.end_effector(), 5).tolist())

        confirm_to_proceed("Go to joint configuration")
        joint_config_goal = [1.03477, -1.51666, 1.53914, -1.45612, -1.34467, 1.01749]
        result = arm.set_joint_positions(
            target_time=5.0, positions=joint_config_goal, wait=True
        )
        print("Joint motion result:", result)

        confirm_to_proceed("Go to Cartesian pose (base_link)")
        eef_pose = [-0.131, 0.181, 0.508, -0.503, 0.507, 0.493, 0.497]
        result = arm.set_target_pose(pose=eef_pose, target_time=5.0, wait=True)
        print("Cartesian motion result:", result)

        print("Done.")
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
