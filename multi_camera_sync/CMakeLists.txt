cmake_minimum_required(VERSION 3.0.2)
project(multi_camera_sync)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  nodelet
  pluginlib
  roscpp
  sensor_msgs
  std_msgs
)
find_package(yaml-cpp REQUIRED)

add_message_files(
  FILES
  SyncStatus.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES multi_camera_sync
  CATKIN_DEPENDS message_runtime nodelet pluginlib roscpp sensor_msgs std_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(multi_camera_sync
  src/sync_config.cpp
  src/sync_engine.cpp
)
add_dependencies(multi_camera_sync ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(multi_camera_sync
  ${catkin_LIBRARIES}
  yaml-cpp
)

add_executable(multi_camera_sync_node src/multi_camera_sync_node.cpp)
add_dependencies(multi_camera_sync_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(multi_camera_sync_node multi_camera_sync ${catkin_LIBRARIES})

add_library(multi_camera_sync_nodelet src/multi_camera_sync_nodelet.cpp)
add_dependencies(multi_camera_sync_nodelet ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(multi_camera_sync_nodelet multi_camera_sync ${catkin_LIBRARIES})

install(TARGETS multi_camera_sync multi_camera_sync_node multi_camera_sync_nodelet
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES nodelet_plugins.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
