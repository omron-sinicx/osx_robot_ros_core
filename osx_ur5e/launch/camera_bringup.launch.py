# Brings up the two Intel RealSense cameras. ROS 2 port of camera_bringup.launch (ROS 1).
#
#   ros2 launch osx_ur5e camera_bringup.launch.py
#   ros2 launch osx_ur5e camera_bringup.launch.py enable_depth:=true
#
# Topic contract: osx_ur5e's ImageRecorder subscribes to the absolute topic
# /<camera_name>/color/image_raw (image_recorder.py), so each node runs in the root
# namespace under its own name. The names must match the keys under `dataset.cameras` in
# config/hydra/test_task.yaml (front_camera / wrist_camera).
#
# Only the color stream is enabled by default -- it is the only stream anything in this
# package reads, and two devices at 60 fps leave little USB3 headroom. enable_depth:=true
# turns the depth stream plus color-aligned depth back on, matching the ROS 1 defaults.
#
# This drives realsense2_camera_node directly rather than including the driver's
# rs_launch.py, for two reasons: rs_launch.py warns about every launch configuration in
# scope that is not one of its own parameters (launch configurations are inherited by
# included launch files, so our own arguments produce a wall of warnings), and its
# serial_no plumbing infers a numeric serial as an integer, which the driver rejects --
# see the ParameterValue(value_type=str) below. Every parameter not set here keeps the
# driver's own default.
#
# Dropped from the ROS 1 file: tf_prefix (the ROS 2 driver derives frames from camera_name,
# so they are front_camera_* / wrist_camera_* rather than calibrated_*; nothing in this repo
# consumes camera frames) and fisheye_fps (no equivalent).

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

CAMERAS = [
    ("front_camera", "front_serial_no", "242322072216"),
    ("wrist_camera", "wrist_serial_no", "242322075793"),
]


def generate_launch_description():
    camera_fps = LaunchConfiguration("camera_fps")
    camera_width = LaunchConfiguration("camera_width")
    camera_height = LaunchConfiguration("camera_height")
    enable_depth = LaunchConfiguration("enable_depth")
    initial_reset = LaunchConfiguration("initial_reset")

    declared_arguments = [
        DeclareLaunchArgument(
            "front_serial_no",
            default_value=CAMERAS[0][2],
            description="Serial number of the front camera.",
        ),
        DeclareLaunchArgument(
            "wrist_serial_no",
            default_value=CAMERAS[1][2],
            description="Serial number of the wrist camera.",
        ),
        DeclareLaunchArgument("camera_fps", default_value="60"),
        DeclareLaunchArgument("camera_width", default_value="640"),
        DeclareLaunchArgument("camera_height", default_value="480"),
        DeclareLaunchArgument(
            "enable_depth",
            default_value="false",
            description="Also stream depth and publish color-aligned depth. Off by default: "
            "nothing in this package subscribes to depth.",
        ),
        DeclareLaunchArgument(
            "initial_reset",
            default_value="false",
            description="Hardware-reset each device before streaming. Useful when a camera "
            "was left in a bad state by a killed process.",
        ),
    ]

    # The driver takes resolution+fps as one "width,height,fps" profile string per stream.
    profile = [camera_width, ",", camera_height, ",", camera_fps]

    cameras = [
        Node(
            package="realsense2_camera",
            executable="realsense2_camera_node",
            name=camera_name,
            namespace="",
            output="screen",
            parameters=[{
                "camera_name": camera_name,
                "camera_namespace": "",
                # A bare numeric serial would be inferred as an integer and the driver
                # rejects it ("parameter {serial_no} is of type {string}").
                "serial_no": ParameterValue(
                    LaunchConfiguration(serial_arg), value_type=str),
                "enable_color": True,
                "rgb_camera.color_profile": ParameterValue(profile, value_type=str),
                "enable_depth": ParameterValue(enable_depth, value_type=bool),
                "depth_module.depth_profile": ParameterValue(profile, value_type=str),
                "align_depth.enable": ParameterValue(enable_depth, value_type=bool),
                "pointcloud.enable": False,
                "enable_infra1": False,
                "enable_infra2": False,
                "initial_reset": ParameterValue(initial_reset, value_type=bool),
            }],
        )
        for camera_name, serial_arg, _ in CAMERAS
    ]

    return LaunchDescription(declared_arguments + cameras)
