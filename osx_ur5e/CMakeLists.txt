cmake_minimum_required(VERSION 3.8)
project(osx_ur5e)

find_package(ament_cmake REQUIRED)

# Install the description, configs and launch files.
# NOTE: the RL/data-collection Python (src/osx_ur5e, scripts/) is deferred with the
# RL stack and intentionally not built/installed here yet.
install(DIRECTORY
  launch
  config
  urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
