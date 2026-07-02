#!/usr/bin/env python3
"""Offline validation for multi_camera_sync using synthetic image publishers."""

import subprocess
import sys
import time

import rospy
from sensor_msgs.msg import Image


def _make_image(stamp, width=4, height=4):
    msg = Image()
    msg.header.stamp = stamp
    msg.height = height
    msg.width = width
    msg.encoding = "rgb8"
    msg.step = width * 3
    msg.data = bytes([0] * (height * msg.step))
    return msg


def main():
    rospy.init_node("multi_camera_sync_offline_test")

    pub_a = rospy.Publisher("/front_camera/color/image_raw", Image, queue_size=5)
    pub_b = rospy.Publisher("/wrist_camera/color/image_raw", Image, queue_size=5)
    received = {}

    def _cb(name):
        def callback(msg):
            received[name] = msg.header.stamp.to_sec()

        return callback

    rospy.Subscriber("/sync/front_camera/image_raw", Image, _cb("front"))
    rospy.Subscriber("/sync/wrist_camera/image_raw", Image, _cb("wrist"))

    deadline = time.time() + 10.0
    rate = rospy.Rate(30)
    seq = 0
    while time.time() < deadline and not rospy.is_shutdown():
        stamp = rospy.Time.now()
        pub_a.publish(_make_image(stamp))
        pub_b.publish(_make_image(stamp + rospy.Duration(0.005)))
        seq += 1
        if len(received) == 2:
            break
        rate.sleep()

    if len(received) != 2:
        print("FAIL: did not receive synced images within timeout", file=sys.stderr)
        return 1

    if received["front"] != received["wrist"]:
        print(
            "FAIL: synced stamps differ",
            received,
            file=sys.stderr,
        )
        return 1

    print("PASS: received synced image set with common stamp", received)
    return 0


if __name__ == "__main__":
    sys.exit(main())
