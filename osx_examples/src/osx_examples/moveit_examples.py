#!/usr/bin/env python3
"""Interactive MoveIt 2 examples using ur_gripper_gz_moveit_config.

Requires gz-sim bringup + move_group running (see README).

Usage:
    ros2 run osx_examples moveit_examples
    ros2 run osx_examples moveit_examples --ur-type ur3e --gripper hande
"""

import argparse
import os
import signal
import sys
import tempfile
import threading

import rclpy
import tf2_ros
import yaml
from math import tau
from moveit.planning import MoveItPy
from rclpy.executors import MultiThreadedExecutor
from rclpy.logging import get_logger
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from osx_examples.moveit_config import build_moveit_config
from osx_examples.moveit_helpers import (
    ARM_GROUP,
    confirm_to_proceed,
    set_joint_position_goal,
    set_named_pose_goal,
    set_pose_goal,
    set_relative_motion_goal,
)
from ur_control import conversions


def _signal_handler(sig, frame):
    print("Interrupted — shutting down.")
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)


def main(args=None):
    parser = argparse.ArgumentParser(description="MoveIt 2 interactive examples")
    parser.add_argument("--ur-type", default="ur5e", help="UR variant (ur3e, ur5e, …)")
    parser.add_argument(
        "--gripper",
        default="robotiq_2f85",
        choices=["hande", "robotiq_2f85"],
        help="Gripper matching the running gz bringup",
    )
    parser.add_argument(
        "--load-gripper",
        action="store_true",
        default=True,
        help="Include gripper MoveIt group (default: true)",
    )
    parser.add_argument(
        "--no-load-gripper",
        action="store_false",
        dest="load_gripper",
        help="Arm-only MoveIt planning",
    )

    argv = remove_ros_args(args if args is not None else sys.argv)
    cli = parser.parse_args(argv[1:])

    rclpy.init(args=args)
    logger = get_logger("moveit_examples")

    moveit_config = build_moveit_config(
        ur_type=cli.ur_type,
        gripper=cli.gripper,
        load_gripper=cli.load_gripper,
    )

    # MoveItPy's config_dict path bakes the node_name into the generated params file
    # instead of "/**", which trips a known moveit_py bug where it can't set the
    # qos_overrides parameters declared for the /clock subscription (needed since
    # use_sim_time=True) — https://github.com/moveit/moveit2/issues/2940. Writing a
    # "/**" wildcard params file and loading it via launch_params_filepaths avoids it.
    config_dict = moveit_config.to_dict()
    config_dict["use_sim_time"] = True
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(tmp_fd, "w") as f:
        yaml.dump({"/**": {"ros__parameters": config_dict}}, f)

    try:
        moveit = MoveItPy(node_name="moveit_examples", launch_params_filepaths=[tmp_path])
    finally:
        os.unlink(tmp_path)

    arm = moveit.get_planning_component(ARM_GROUP)
    logger.info(f"MoveItPy ready (group={ARM_GROUP}, ur_type={cli.ur_type}, gripper={cli.gripper})")

    tf_node = Node("moveit_examples_tf")
    executor = MultiThreadedExecutor()
    executor.add_node(tf_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer, tf_node)
    del tf_listener

    try:
        confirm_to_proceed("Go to joint position")
        joint_pose_goal = [1.03477, -1.51666, 1.53914, -1.45612, -1.34467, 1.01749]
        set_joint_position_goal(arm, moveit, joint_pose_goal, logger)

        confirm_to_proceed("Go to cartesian position (base_link)")
        pose_goal = conversions.to_pose_stamped(
            frame_id="base_link",
            pose=[0.4, -0.22, 0.35, tau / 4, tau / 4, 0.0],
        )
        set_pose_goal(arm, moveit, pose_goal, logger)

        confirm_to_proceed("Go to another cartesian position (base_link)")
        pose_goal = conversions.to_pose_stamped(
            frame_id="base_link",
            pose=[0.35, -0.20, 0.30, tau / 4, tau / 4, 0.0],
        )
        set_pose_goal(arm, moveit, pose_goal, logger)

        confirm_to_proceed("Move relative to robot base frame (-Z 10 cm)")
        set_relative_motion_goal(
            arm, moveit, tf_buffer, relative_translation=[0, 0, -0.1], logger=logger
        )

        confirm_to_proceed("Move relative to tool0 frame (-X 10 cm)")
        set_relative_motion_goal(
            arm,
            moveit,
            tf_buffer,
            relative_translation=[-0.1, 0, 0],
            logger=logger,
            relative_to_tcp=True,
        )

        confirm_to_proceed("Move relative to world/base frame (+X 10 cm)")
        set_relative_motion_goal(
            arm, moveit, tf_buffer, relative_translation=[0.1, 0, 0], logger=logger
        )

        confirm_to_proceed("Go to named pose 'home'")
        set_named_pose_goal(arm, moveit, "home", logger)

        logger.info("All examples completed.")
    finally:
        moveit.shutdown()
        executor.shutdown()
        tf_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
