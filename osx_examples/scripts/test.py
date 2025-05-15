#!/usr/bin/env python

import copy
from math import tau

import numpy as np
from osx_assembly_database.assembly_reader import AssemblyReader
from osx_robot_control.core import OSXCore
from osx_robot_control.common import OSXCommon
import rospy
import sys
import signal

from ur_control import conversions


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def main():
    rospy.init_node("Moveit Examples")
    osx = OSXCommon()
    osx.assembly_database = AssemblyReader("cooking")
    osx.reset_scene_and_robots()
    osx.spawn_object("apple", conversions.to_pose_stamped("tray_center", [0.0, -0.1, 0.001] + np.deg2rad([0, 0., 0]).tolist()))
    osx.spawn_object("carrot", conversions.to_pose_stamped("tray_center", [0.0, -0.0, 0.001] + np.deg2rad([0, 0., 0]).tolist()))
    osx.spawn_object("cucumber", conversions.to_pose_stamped("tray_center", [0.0, 0.1, 0.001] + np.deg2rad([0, 0., 0]).tolist()))
    osx.spawn_object("plate1", conversions.to_pose_stamped("left_centering_link", [0.0, 0, 0.001] + np.deg2rad([0, 90., 0]).tolist()))
    osx.spawn_object("plate2", conversions.to_pose_stamped("right_centering_link", [0.0, 0, 0.001] + np.deg2rad([0, 90., 0]).tolist()))
    osx.spawn_object("cucumber_slice", conversions.to_pose_stamped("tray_center", [0.1, 0.05, 0.001] + np.deg2rad([0, 0, 0]).tolist()))

    # initial_q = [1.1242, -1.7731, 2.0603, -1.8568, -1.5654, 1.1249]
    # osx.b_bot.move_joints(joint_pose_goal=initial_q, speed=0.3, wait=True)

    # init_pose = osx.b_bot.get_current_pose_stamped(end_effector_link="b_bot_knife_center")
    # next_pose = copy.deepcopy(init_pose)
    # next_pose.pose.position.y += 0.01
    # next_pose2 = copy.deepcopy(next_pose)
    # next_pose2.pose.position.z += 0.01
    # next_pose3 = copy.deepcopy(next_pose2)
    # next_pose3.pose.position.y += 0.01
    # next_pose4 = copy.deepcopy(next_pose3)
    # next_pose4.pose.position.z += 0.01
    # next_pose5 = copy.deepcopy(next_pose4)
    # next_pose5.pose.position.y += 0.01
    # next_pose6 = copy.deepcopy(next_pose5)
    # next_pose6.pose.position.z += 0.01
    # # Test move trajectory lin
    # trajectory = [[init_pose, 0.0, 1.0], [next_pose, 0.0, 1.0], [next_pose2, 0.0, 1.0], [next_pose3, 0.0, 1.0], [next_pose4, 0.0, 1.0], [next_pose5, 0.0, 1.0], [next_pose6, 0.0, 1.0]]
    # osx.b_bot.move_lin_trajectory(trajectory, end_effector_link="b_bot_knife_center", retime=False)


if __name__ == "__main__":
    main()
