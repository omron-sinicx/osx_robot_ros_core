cmake_minimum_required(VERSION 3.0.2)
project(osx_splines)

find_package(catkin REQUIRED COMPONENTS
  rospy
)

## Python modules under src/osx_splines (see setup.py)
catkin_python_setup()

catkin_package(
)

include_directories(${catkin_INCLUDE_DIRS})

## casp entry points, runnable with `rosrun osx_splines <script>`
catkin_install_python(PROGRAMS
  scripts/evaluate_policy.py
  scripts/replay_episode.py
  scripts/test_replay_episode.py
  scripts/test_replay_episode_bspline.py
  scripts/test_waypoint_stride.py
  scripts/test_waypoint_sync.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
