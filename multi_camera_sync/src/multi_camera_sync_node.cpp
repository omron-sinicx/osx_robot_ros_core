#include <memory>
#include <string>

#include <ros/ros.h>

#include "multi_camera_sync/sync_config.hpp"
#include "multi_camera_sync/sync_engine.hpp"

int main(int argc, char** argv) {
  ros::init(argc, argv, "multi_camera_sync");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  std::string config_path;
  if (!pnh.getParam("config", config_path)) {
    ROS_FATAL("~config param is required (path to YAML config)");
    return 1;
  }

  try {
    multi_camera_sync::SyncConfig config = multi_camera_sync::loadConfigFromYaml(config_path);

    if (pnh.hasParam("slop")) {
      pnh.getParam("slop", config.slop);
    }
    if (pnh.hasParam("queue_size")) {
      pnh.getParam("queue_size", config.queue_size);
    }
    if (pnh.hasParam("output_namespace")) {
      pnh.getParam("output_namespace", config.output_namespace);
    }
    if (pnh.hasParam("reference_camera")) {
      pnh.getParam("reference_camera", config.reference_camera);
    }

    multi_camera_sync::SyncEngine engine(nh, pnh, config);
    engine.start();
    ros::spin();
  } catch (const std::exception& ex) {
    ROS_FATAL("multi_camera_sync failed to start: %s", ex.what());
    return 1;
  }

  return 0;
}
