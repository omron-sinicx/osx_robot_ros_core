cmake_minimum_required(VERSION 3.8)
project(osx_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Slimmed for the ROS 2 migration: only RobotStatus remains (vision interfaces were
# dropped). RobotStatus uses primitive fields only, so there are no interface deps.
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RobotStatus.msg"
)

ament_package()
