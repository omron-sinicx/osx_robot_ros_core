#!/usr/bin/env python

from math import tau
from osx_robot_control.core import OSXCore
import rospy
import rospkg
import sys
import signal

from copy import deepcopy

from ur_control import conversions
import numpy as np

#grasping related
import geometry_msgs.msg
from osx_robot_control.common import OSXCommon
from osx_assembly_database.assembly_reader import AssemblyReader

# Global variables
robot_name = "b_bot"


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


def main():
    rospy.init_node("pick knife example")
    #osx = OSXCore()
    osx = OSXCommon()
    osx.reset_scene_and_robots()
    osx.ab_bot.go_to_named_pose("home")
    osx.a_bot.gripper.open()
    osx.b_bot.gripper.open()

    object_name = "cucumber"

    # Load cooking objects
    osx.assembly_database = AssemblyReader("cooking")
    # Load tools (Knife)
    osx.define_tool_collision_objects()

    current_joint_values = osx.active_robots[robot_name].robot_group.get_current_joint_values()


    rospy.sleep(1)

    osx.spawn_tool("knife")
    osx.spawn_object("cucumber", conversions.to_pose_stamped("tray_center", [0.0, 0.05, 0.01] + np.deg2rad([0, 90., 0]).tolist()))

    equip_knife_pose = conversions.to_pose_stamped("knife_pickup_link", [0.039, -0.004, 0.017, -0.016, -0.000, -0.009, 1.000])

    grasp_names = osx.assembly_database.get_grasp_names(object_name)
    print(grasp_names)

    for grasp_name in grasp_names:
        print(grasp_name)

    robot = osx.active_robots[robot_name]
    # compute actual grasp pose from grasp name
    
    grasp_pose = osx.get_transformed_grasp_pose(object_name, grasp_names[0], target_frame="workspace_center")
            
    # Sample poses for approach
    approach_pose = deepcopy(grasp_pose)
    approach_pose.pose.position.z += 0.10
    res = robot.go_to_pose_goal(approach_pose, initial_joints=current_joint_values, plan_only = True)

    if res is None:
        return None
    approach_trajectory = res[0]

    res = robot.go_to_pose_goal(grasp_pose, initial_joints=approach_trajectory.joint_trajectory.points[-1].positions, plan_only = True)

    if res is None:
        return None
    pickup_trajectory = res[0]

    object_pose = conversions.to_pose_stamped("tray_center", [0.0, 0.05, 0.01] + np.deg2rad([0, 90., 0]).tolist())

    osx.simple_pick(robot_name, object_pose, approach_height=0.1, axis="z", item_id_to_attach="cucumber", attach_with_collisions=True)
    # osx.active_robots[robot_name].move_joints(robot_conf.joints)

    place_pose = conversions.to_pose_stamped("cutting_board_surface", [0.02, -0.03, 0, tau/4, tau/4, 0])

    # osx.simple_place(robot_name, place_pose, approach_height=0.1, axis="z")

    current_joint_values = osx.active_robots[robot_name].robot_group.get_current_joint_values()
    
    # Sample poses for approach
    approach_pose = deepcopy(place_pose)
    approach_pose.pose.position.z += 0.10
    res = robot.go_to_pose_goal(approach_pose, initial_joints=current_joint_values, plan_only=True)

    if res is None:
        return None
    
    approach_trajectory = res[0]

    robot.execute_plan(approach_trajectory)

    placing_pose = deepcopy(place_pose)
    placing_pose.pose.position.z += 0.05
    res = robot.go_to_pose_goal(placing_pose, initial_joints=approach_trajectory.joint_trajectory.points[-1].positions, plan_only=True)

    if res is None:
        return None
            
    placedown_trajectory = res[0]

    robot.execute_plan(placedown_trajectory)


if __name__ == "__main__":
    main()