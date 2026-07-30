# Minimal, self-contained Gazebo (gz-sim / Harmonic) bringup for a single UR5e
# driven through gz_ros2_control. ROS 2 port of gazebo.launch.
#
#   ros2 launch osx_ur5e gz_bringup.launch.py            # with GUI
#   ros2 launch osx_ur5e gz_bringup.launch.py gui:=false # headless server (CI)
#
# Provides everything ur_control's Arm and CompliantController need: /robot_description + TF
# (robot_state_publisher), /joint_states (joint_state_broadcaster), the
# scaled_joint_trajectory_controller FollowJointTrajectory action, /wrench +
# /wrench/filtered (force_torque_sensor_broadcaster + ft_filter) and an inactive
# cartesian_compliance_controller for FDCC.
#
# Note: use_gazebo_sim / use_real_robot are node parameters read from the hydra config by the
# entry points (both default to False), not launch arguments -- run the env with
# `+use_gazebo_sim=true` against this bringup.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            RegisterEventHandler, SetEnvironmentVariable, TimerAction)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (Command, FindExecutable, LaunchConfiguration,
                                  PathJoinSubstitution, PythonExpression)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Let gz resolve package:// mesh URIs (e.g. ur_description meshes). gz searches each
    # GZ_SIM_RESOURCE_PATH entry as a root, so add the share dir that contains the packages.
    share_root = os.path.dirname(get_package_share_directory("ur_description"))
    gz_resource_path = SetEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        os.pathsep.join([share_root, os.environ.get("GZ_SIM_RESOURCE_PATH", "")]),
    )

    ur_type = LaunchConfiguration("ur_type")
    gui = LaunchConfiguration("gui")

    pkg = FindPackageShare("osx_ur5e")
    controllers_file = PathJoinSubstitution([pkg, "config", "ur5e_gz_controllers.yaml"])
    xacro_file = PathJoinSubstitution([pkg, "urdf", "ur5e_gz.urdf.xacro"])

    robot_description_content = Command([
        FindExecutable(name="xacro"), " ", xacro_file,
        " ur_type:=", ur_type,
        " name:=", ur_type,
        " controllers_file:=", controllers_file,
    ])
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description, {"use_sim_time": True}],
    )

    # gz <-> ROS clock bridge so use_sim_time nodes get /clock.
    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        output="screen",
    )

    # Headless server (-s) when gui:=false, otherwise full GUI. -r = run on start.
    gz_args = PythonExpression(
        ["'-r -v3 empty.sdf' if '", gui, "' == 'true' else '-s -r -v3 empty.sdf'"]
    )
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]
        ),
        launch_arguments={"gz_args": gz_args}.items(),
    )

    spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=["-topic", "robot_description", "-name", ur_type, "-allow_renaming", "true"],
    )
    # Give gz a moment to bring up the world service before spawning.
    spawn_entity_delayed = TimerAction(period=4.0, actions=[spawn_entity])

    jsb_spawner = Node(
        package="controller_manager", executable="spawner", output="screen",
        arguments=["joint_state_broadcaster", "-c", "/controller_manager",
                   "--controller-manager-timeout", "120"],
    )
    # Publishes the wrist FT sensor on /wrench so ur_control's Arm reads it unchanged
    # (FT_SUBSCRIBER='wrench'). --controller-ros-args remaps the controller node itself (the
    # broadcaster publishes ~/wrench); a Node-level remap would only touch the short-lived
    # spawner process, not the controller running inside controller_manager.
    fts_spawner = Node(
        package="controller_manager", executable="spawner", output="screen",
        arguments=["force_torque_sensor_broadcaster", "-c", "/controller_manager",
                   "--controller-manager-timeout", "120",
                   "--controller-ros-args",
                   "-r /force_torque_sensor_broadcaster/wrench:=/wrench"],
    )
    jtc_spawner = Node(
        package="controller_manager", executable="spawner", output="screen",
        arguments=["scaled_joint_trajectory_controller", "-c", "/controller_manager",
                   "--controller-manager-timeout", "120"],
    )
    fvc_spawner = Node(
        package="controller_manager", executable="spawner", output="screen",
        arguments=["forward_velocity_controller", "-c", "/controller_manager",
                   "--inactive", "--controller-manager-timeout", "120"],
    )
    # Loaded inactive; ur_control's CompliantController activates it. Its wrench input
    # (~/ft_sensor_wrench) is remapped to /wrench/filtered, ft_filter's output.
    compliance_spawner = Node(
        package="controller_manager", executable="spawner", output="screen",
        arguments=["cartesian_compliance_controller", "-c", "/controller_manager",
                   "--inactive", "--controller-manager-timeout", "120",
                   "--controller-ros-args",
                   "-r /cartesian_compliance_controller/ft_sensor_wrench:=/wrench/filtered"],
    )

    # Butterworth-filtered, zeroable republish of the FT sensor: subscribes to /wrench (from
    # the broadcaster) and publishes /wrench/filtered plus a /wrench/filtered/zero_ftsensor
    # service. ur_control's Arm prefers the filtered topic and needs that service for
    # zero_ft_sensor(). (The real-robot launch wires this differently: there the UR driver
    # spawns its own broadcaster with no remap, so ft_filter reads
    # /force_torque_sensor_broadcaster/wrench directly.)
    ft_filter = Node(
        package="ur_control_examples", executable="ft_filter", output="screen",
        arguments=["-t", "wrench"],
        parameters=[{"use_sim_time": True}],
    )

    # Order: spawn model -> joint_state_broadcaster -> FT broadcaster + trajectory
    # controller -> velocity and compliance controllers.
    return LaunchDescription([
        gz_resource_path,
        DeclareLaunchArgument("ur_type", default_value="ur5e"),
        DeclareLaunchArgument("gui", default_value="true",
                              description="Run the Gazebo GUI (false = headless server)"),
        robot_state_publisher,
        clock_bridge,
        gz_sim,
        ft_filter,
        spawn_entity_delayed,
        RegisterEventHandler(OnProcessExit(target_action=spawn_entity, on_exit=[jsb_spawner])),
        RegisterEventHandler(
            OnProcessExit(target_action=jsb_spawner, on_exit=[fts_spawner, jtc_spawner])),
        RegisterEventHandler(
            OnProcessExit(target_action=jtc_spawner,
                          on_exit=[fvc_spawner, compliance_spawner])),
    ])
