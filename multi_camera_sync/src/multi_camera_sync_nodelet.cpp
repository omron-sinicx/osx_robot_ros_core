#include <memory>
#include <string>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include "multi_camera_sync/sync_config.hpp"
#include "multi_camera_sync/sync_engine.hpp"

namespace multi_camera_sync {

class MultiCameraSyncNodelet : public nodelet::Nodelet {
 public:
  MultiCameraSyncNodelet() = default;

 private:
  void onInit() override {
    ros::NodeHandle nh = getNodeHandle();
    ros::NodeHandle pnh = getPrivateNodeHandle();

    std::string config_path;
    if (!pnh.getParam("config", config_path)) {
      NODELET_FATAL("~config param is required (path to YAML config)");
      throw std::runtime_error("missing ~config param");
    }

    SyncConfig config = loadConfigFromYaml(config_path);

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

    engine_.reset(new SyncEngine(nh, pnh, config));
    engine_->start();
  }

  std::unique_ptr<SyncEngine> engine_;
};

}  // namespace multi_camera_sync

PLUGINLIB_EXPORT_CLASS(multi_camera_sync::MultiCameraSyncNodelet, nodelet::Nodelet)
