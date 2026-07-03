"""Launch moveit_examples (convenience wrapper — the node manages its own MoveIt config).

Usage:
    ros2 launch osx_examples moveit_examples.launch.py
    ros2 launch osx_examples moveit_examples.launch.py ur_type:=ur3e gripper:=hande
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    ur_type = LaunchConfiguration("ur_type").perform(context)
    gripper = LaunchConfiguration("gripper").perform(context)
    load_gripper = LaunchConfiguration("load_gripper").perform(context)

    load_gripper_arg = ["--load-gripper"] if load_gripper == "true" else ["--no-load-gripper"]
    moveit_examples_node = Node(
        package="osx_examples",
        executable="moveit_examples",
        output="screen",
        arguments=["--ur-type", ur_type, "--gripper", gripper] + load_gripper_arg,
    )

    return [moveit_examples_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("ur_type", default_value="ur5e",
                              description="UR variant (ur3e, ur5e, …)"),
        DeclareLaunchArgument("gripper", default_value="robotiq_2f85",
                              choices=["hande", "robotiq_2f85"],
                              description="Gripper matching the running gz bringup"),
        DeclareLaunchArgument("load_gripper", default_value="true",
                              description="Include gripper MoveIt group (true/false)"),
        OpaqueFunction(function=launch_setup),
    ])
