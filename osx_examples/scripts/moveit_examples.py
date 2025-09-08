#!/usr/bin/env python

from math import tau
from osx_robot_control.core import OSXCore
import rospy
import sys
import signal

from ur_control import conversions


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def main():
    rospy.init_node("osx_moveit_examples")
    osx = OSXCore()

    # Go to a given joint position
    osx.confirm_to_proceed("Go to joint position")
    joint_pose_goal = [1.03477, -1.51666, 1.53914, -1.45612, -1.34467, 1.01749]
    osx.a_bot.set_joint_position_goal(joint_pose_goal=joint_pose_goal, speed=0.3)

    # Go to a given cartesian position
    # Convert a list of cartesian pose [x, y, z, qx, qy, qz, qw] to a geometry_msgs.msg.PoseStamped
    osx.confirm_to_proceed("Go to cartesian position")
    pose_goal = conversions.to_pose_stamped(frame_id="cutting_board", pose=[0.0, -0.22, 0.27, tau/4, tau/4, 0])
    osx.a_bot.set_pose_goal(pose_goal_stamped=pose_goal, speed=0.3)

    # Go to a given cartesian position in a linear way
    osx.confirm_to_proceed("Go to cartesian position (Linear)")
    pose_goal = conversions.to_pose_stamped(frame_id="cutting_board", pose=[0.0, -0.20, 0.20, tau/4, tau/4, 0])
    osx.a_bot.set_pose_goal(pose_goal_stamped=pose_goal, speed=0.3)

    # Move relative to the current pose (Base Frame)
    osx.confirm_to_proceed("Move relative to robot base frame")
    # Move 10cm downward
    osx.a_bot.set_relative_motion_goal(relative_translation=[0, 0, -0.1], relative_to_robot_base=True, speed=0.3)

    # Move relative to the current pose (End-Effector Frame "a_bot_gripper_tip_link")
    osx.confirm_to_proceed("Move relative to robot's end effector frame")
    # Move 10cm upward
    osx.a_bot.set_relative_motion_goal(relative_translation=[-0.1, 0, 0], relative_to_tcp=True, speed=0.3)

    # Move relative to the current pose (World Frame)
    osx.confirm_to_proceed("Move relative to the world frame")
    # Move away from the TV screen
    osx.a_bot.set_relative_motion_goal(relative_translation=[0.1, 0, 0], speed=0.3)

    # Go to named pose (Defined in the moveit config file: osx_moveit_config/config/osx_base_scene.srdf)
    osx.confirm_to_proceed("Go to the named pose 'Home'")
    osx.a_bot.go_to_named_pose("home", speed=0.3)


if __name__ == "__main__":
    main()
