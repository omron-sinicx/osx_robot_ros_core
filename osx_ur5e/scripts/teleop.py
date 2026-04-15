#!/usr/bin/env python3

from osx_gello.gello import Gello
import rospy
from ur_control.arm import Arm

def main():
    rospy.init_node('teleop')
    gello = Gello()
    arm = Arm(gripper_type=None)

    while not rospy.is_shutdown():
        print("gello joint angles:", [f"{angle:.4f}" for angle in gello.joint_angles()])
        print("arm joint angles:", [f"{angle:.4f}" for angle in arm.joint_angles()])
        arm.set_joint_positions(target_time=0.05, positions=gello.joint_angles(), wait=False)
        rospy.sleep(0.05)

if __name__ == "__main__":
    main()
