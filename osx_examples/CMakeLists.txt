cmake_minimum_required(VERSION 3.1.3)
project(osx_examples)

## Add support for C++11, supported in ROS Kinetic and newer
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  tf
  tf_conversions
  moveit_core 
  moveit_commander
  moveit_ros_planning
  moveit_ros_planning_interface
  robotiq_msgs
  ur_msgs
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES osx_examples
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(include ${catkin_INCLUDE_DIRS})

#############
## Testing ##
#############

