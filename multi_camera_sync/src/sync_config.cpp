#include "multi_camera_sync/sync_config.hpp"

#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace multi_camera_sync {

SyncConfig loadConfigFromYaml(const std::string& path) {
  YAML::Node root = YAML::LoadFile(path);
  SyncConfig config;

  if (root["slop"]) {
    config.slop = root["slop"].as<double>();
  }
  if (root["queue_size"]) {
    config.queue_size = root["queue_size"].as<int>();
  }
  if (root["output_namespace"]) {
    config.output_namespace = root["output_namespace"].as<std::string>();
  }
  if (root["reference_camera"]) {
    config.reference_camera = root["reference_camera"].as<std::string>();
  }

  const YAML::Node cameras = root["cameras"];
  if (!cameras || !cameras.IsMap()) {
    throw std::runtime_error("multi_camera_sync config requires a 'cameras' map");
  }

  for (const auto& entry : cameras) {
    CameraConfig camera;
    camera.name = entry.first.as<std::string>();
    const YAML::Node camera_node = entry.second;
    if (!camera_node["topic"]) {
      throw std::runtime_error("camera '" + camera.name + "' is missing 'topic'");
    }
    camera.topic = camera_node["topic"].as<std::string>();
    config.cameras.push_back(camera);
  }

  if (config.cameras.size() < 2) {
    throw std::runtime_error("multi_camera_sync requires at least 2 cameras");
  }

  if (config.reference_camera.empty()) {
    config.reference_camera = config.cameras.front().name;
  }

  return config;
}

}  // namespace multi_camera_sync
