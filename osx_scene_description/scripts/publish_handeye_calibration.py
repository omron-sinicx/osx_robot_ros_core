#!/usr/bin/env python3
"""Publish hand-eye calibration result from YAML as a static TF transform.

Usage:
    rosrun osx_scene_description publish_handeye_calibration.py \
        --parent base_link \
        --child d435i_color_optical_frame
"""
import argparse
from pathlib import Path

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
import yaml


def main():
    parser = argparse.ArgumentParser(description="Publish hand-eye calibration as static TF")
    parser.add_argument("--parent", required=True, help="Parent frame (e.g. base_link)")
    parser.add_argument("--child", required=True, help="Child frame (e.g. d435i_color_optical_frame)")
    parser.add_argument("--file", default=None, help="YAML file path (default: config/camera_calibration/<parent>-to-<child>.yaml)")
    args, _ = parser.parse_known_args()

    if args.file:
        yaml_path = Path(args.file)
    else:
        pkg_dir = Path(__file__).resolve().parent.parent
        yaml_path = pkg_dir / "config" / "camera_calibration" / f"{args.parent}-to-{args.child}.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {yaml_path}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    tr = data["transform"]

    rospy.init_node("publish_handeye_calibration", anonymous=True)

    broadcaster = tf2_ros.StaticTransformBroadcaster()
    msg = TransformStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = args.parent
    msg.child_frame_id = args.child
    msg.transform.translation.x = tr["x"]
    msg.transform.translation.y = tr["y"]
    msg.transform.translation.z = tr["z"]
    msg.transform.rotation.x = tr["qx"]
    msg.transform.rotation.y = tr["qy"]
    msg.transform.rotation.z = tr["qz"]
    msg.transform.rotation.w = tr["qw"]

    broadcaster.sendTransform(msg)
    rospy.loginfo(f"Publishing static TF: {args.parent} -> {args.child} (from {yaml_path})")
    rospy.spin()


if __name__ == "__main__":
    main()
