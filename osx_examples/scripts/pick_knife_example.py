#!/usr/bin/env python

from math import tau
from osx_robot_control.core import OSXCore
import rospy
import rospkg
import sys
import signal

from ur_control import conversions

#grasping related
import geometry_msgs.msg
from osx_robot_control.common import OSXCommon
from osx_assembly_database.assembly_reader import AssemblyReader
from osx_robot_control.helpers import stack_two_plans


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


def main():
    rospy.init_node("pick knife example")
    #osx = OSXCore()
    osx = OSXCommon()
    
    object_name = "knife"
    robot_name = "b_bot"

    osx.define_tool_collision_objects()
    osx.assembly_database = AssemblyReader("cooking")

    osx.planning_scene_interface.remove_attached_object("knife")
    osx.planning_scene_interface.remove_world_object("knife")
    rospy.sleep(1)

    osx.active_robots[robot_name].go_to_named_pose("home", speed=0.3) # Move to initial position
    osx.active_robots[robot_name].gripper.open() # Reset gripper

    osx.spawn_tool("knife")

    equip_pose = conversions.to_pose_stamped(
            "knife_pickup_link", [0.039, -0.004, 0.017, -0.016, -0.000, -0.009, 1.000])

    # You can use a pre-defined approach pose and grasping pose here

    # approach_t, _ = osx.active_robots[robot_name].go_to_named_pose("knife_pick_up_ready", speed = 0.2, plan_only=True)
    # osx.active_robots[robot_name].execute_plan(approach_t)

    # equip_pose = conversions.to_pose_stamped(
    #         "knife_pickup_link", [0.039, -0.004, 0.017, -0.016, -0.000, -0.009, 1.000])

    # grasp_t, _ = osx.active_robots[robot_name].go_to_pose_goal(equip_pose, speed=.1, move_lin=True, plan_only = True)
    # osx.active_robots[robot_name].execute_plan(grasp_t)



    # You can use simple_pick with the object_pose
    
    approach_pose, grasp_pose, seq = osx.simple_pick(robot_name, equip_pose, item_id_to_attach= "knife", axis="x",
                                                sign=-1, attach_with_collisions=True, approach_height=0.1,
                                                lift_up_after_pick=False, speed_fast=0.25,  grasp_height=-0.0,
                                                approach_with_move_lin=True)
    
    print("Approach pose: ", approach_pose) # Wrap into trajectory?
    print("Grasping pose: ", grasp_pose) # Wrap into trajectory

    osx.active_robots[robot_name].go_to_pose_goal(pose_goal_stamped = approach_pose, speed = 0.3)
    osx.active_robots[robot_name].go_to_pose_goal(pose_goal_stamped = grasp_pose, speed = 0.3)

    # osx.execute_sequence(robot_name, seq, "pick")


    rospy.sleep(1)

    osx.allow_collisions_with_robot_hand("knife", "b_bot")
    osx.active_robots[robot_name].gripper.close(velocity=0.03, force=100)

    osx.active_robots[robot_name].gripper.attach_object("knife", with_collisions = True)

    osx.planning_scene_interface.allow_collisions("knife", "knife_holder")
    osx.active_robots[robot_name].move_lin_rel(relative_translation=[0, 0, 0.1], speed=0.1, relative_to_robot_base=True)
        # self.b_bot.go_to_named_pose("knife_pick_up_ready", speed=0.2)
    osx.planning_scene_interface.disallow_collisions("knife", "knife_holder")

    osx.active_robots[robot_name].go_to_named_pose("home", speed=0.25)

if __name__ == "__main__":
    main()