#!/usr/bin/env python

from math import tau
import rospy
from ur_control.arm import Arm
import sys
import signal

from ur_control.constants import TRAC_IK


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def main():
    rospy.init_node('ur3e_script_control')

    # Initialize controller for an specific arm
    ns = "b_bot"
    tcp_link = "gripper_tip_link"
    joint_names_prefix = ns+'_' if ns else ''

    global arm
    arm = Arm(namespace=ns,
              joint_names_prefix=joint_names_prefix,
              ee_link=tcp_link,
              ft_topic='wrench',
              ik_solver=TRAC_IK)

    # See current joint configuration of the robot
    print("current joint configuration", arm.joint_angles())

    # See current Cartesian pose of the robot's end effector
    print("current EEF pose", arm.end_effector())

    # Go to a given target joint configuration in 5 seconds, wait until the motion is completed
    # Optionally it can not wait for the action to be completed and to something else.
    input("Enter to proceed: Go to joint configuration")
    joint_config_goal = [1.03477, -1.51666, 1.53914, -1.45612, -1.34467, 1.01749]
    arm.set_joint_positions(position=joint_config_goal, t=5, wait=True)

    # Go to a given cartesian pose with the end effector. The pose is only understood in the base_link frame of the robot
    input("Enter to proceed: Go to Cartesian pose")
    eef_pose = [-0.131, 0.181, 0.508, -0.503, 0.507, 0.493, 0.497]
    arm.set_target_pose(pose=eef_pose, t=5, wait=True)


if __name__ == "__main__":
    main()
