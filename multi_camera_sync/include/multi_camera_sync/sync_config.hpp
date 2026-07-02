#pragma once

#include <string>
#include <vector>

namespace multi_camera_sync {

struct CameraConfig {
  std::string name;
  std::string topic;
};

struct SyncConfig {
  double slop = 0.025;
  int queue_size = 30;
  std::string reference_camera;
  std::string output_namespace = "sync";
  std::vector<CameraConfig> cameras;
};

SyncConfig loadConfigFromYaml(const std::string& path);

}  // namespace multi_camera_sync
