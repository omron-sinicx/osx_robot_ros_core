# Bringup for the real UR5e. ROS 2 port of connect_real_robot.launch (ROS 1).
#
#   ros2 launch osx_ur5e connect_real_robot.launch.py robot_ip:=10.0.2.3
#   ros2 launch osx_ur5e connect_real_robot.launch.py use_mock_hardware:=true  # no robot
#
# Wraps ur_robot_driver's ur_control.launch.py (driver + robot_state_publisher +
# the standard UR controllers) and adds the two OSX-specific pieces the ROS 1
# launch provided:
#
#   * ft_filter  -- Butterworth-filtered, zeroable republish of the wrist FT sensor on
#     /wrench/filtered (+ a /wrench/filtered/zero_ftsensor service). ur_control's Arm
#     prefers that topic over the raw one and needs the service for zero_ft_sensor().
#   * cartesian_compliance_controller -- loaded inactive for CompliantController (FDCC).
#
# The UR5e calibration is applied by declaring kinematics_params_file here:
# launch configurations are inherited by included launch files, so ur_rsp.launch.py
# (included by ur_control.launch.py, which forwards only robot_ip/ur_type) picks this
# value up instead of ur_description's default_kinematics.yaml.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare("osx_ur5e")

    ur_driver_launch = PathJoinSubstitution(
        [FindPackageShare("ur_robot_driver"), "launch", "ur_control.launch.py"]
    )

    declared_arguments = [
        DeclareLaunchArgument(
            "robot_ip",
            default_value="10.0.2.15",
            description="IP address by which the robot can be reached.",
        ),
        DeclareLaunchArgument("ur_type", default_value="ur5e"),
        DeclareLaunchArgument(
            "kinematics_params_file",
            default_value=PathJoinSubstitution([pkg, "config", "ur5e_calibration.yaml"]),
            description="Calibration of this specific UR5e (extracted with ur_calibration). "
            "Inherited by ur_rsp.launch.py; see the note at the top of this file.",
        ),
        DeclareLaunchArgument(
            "headless_mode",
            default_value="false",
            description="Send URScript to the robot directly instead of using the External "
            "Control URCap. On e-Series this requires the robot in 'remote control' mode.",
        ),
        DeclareLaunchArgument(
            "use_mock_hardware",
            default_value="false",
            description="Start ros2_control with mock hardware instead of the real robot. "
            "Useful to check this launch file without a robot on the network.",
        ),
        DeclareLaunchArgument("launch_rviz", default_value="false"),
        DeclareLaunchArgument("launch_dashboard_client", default_value="true"),
        DeclareLaunchArgument(
            "initial_joint_controller",
            default_value="scaled_joint_trajectory_controller",
            description="ur_control's Arm drives this controller "
            "(constants.JOINT_POSITION_TRAJECTORY_CONTROLLER).",
        ),
        DeclareLaunchArgument("activate_joint_controller", default_value="true"),
        DeclareLaunchArgument("controller_spawner_timeout", default_value="120"),
        # Tool (RS485) communication, off by default -- as in the ROS 1 launch.
        DeclareLaunchArgument("use_tool_communication", default_value="false"),
        DeclareLaunchArgument("tool_device_name", default_value="/tmp/ttyUR"),
        DeclareLaunchArgument("tool_tcp_port", default_value="54322"),
        DeclareLaunchArgument("tool_voltage", default_value="0"),
        # Driver <-> robot socket ports (same values as the ROS 1 launch).
        DeclareLaunchArgument("reverse_port", default_value="50001"),
        DeclareLaunchArgument("script_sender_port", default_value="50002"),
        DeclareLaunchArgument("trajectory_port", default_value="50003"),
        DeclareLaunchArgument("script_command_port", default_value="50004"),
    ]

    forwarded = [
        "robot_ip",
        "ur_type",
        "headless_mode",
        "use_mock_hardware",
        "launch_rviz",
        "launch_dashboard_client",
        "initial_joint_controller",
        "activate_joint_controller",
        "controller_spawner_timeout",
        "use_tool_communication",
        "tool_device_name",
        "tool_tcp_port",
        "tool_voltage",
        "reverse_port",
        "script_sender_port",
        "trajectory_port",
        "script_command_port",
    ]
    ur_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(ur_driver_launch),
        launch_arguments={name: LaunchConfiguration(name) for name in forwarded}.items(),
    )

    # The driver's force_torque_sensor_broadcaster publishes on
    # /force_torque_sensor_broadcaster/wrench; filter it into /wrench/filtered, which is
    # where ur_control's Arm looks (ns + FT_SUBSCRIBER + '/filtered').
    ft_filter = Node(
        package="ur_control_examples",
        executable="ft_filter",
        output="screen",
        arguments=[
            "-t", "force_torque_sensor_broadcaster/wrench",
            "-ot", "wrench/filtered",
        ],
    )

    # Loaded inactive; CompliantController activates it. --controller-ros-args remaps the
    # controller node itself (a Node-level remap would only touch the short-lived spawner
    # process, not the controller running inside controller_manager).
    compliance_spawner = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=[
            "cartesian_compliance_controller",
            "-c", "/controller_manager",
            "--param-file", PathJoinSubstitution(
                [pkg, "config", "ur5e_cartesian_controllers.yaml"]
            ),
            "--inactive",
            "--controller-manager-timeout", LaunchConfiguration("controller_spawner_timeout"),
            "--controller-ros-args",
            "-r /cartesian_compliance_controller/ft_sensor_wrench:=/wrench/filtered",
        ],
    )

    return LaunchDescription(
        declared_arguments + [ur_driver, ft_filter, compliance_spawner]
    )
