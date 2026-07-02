#pragma once

#include <deque>
#include <memory>
#include <string>
#include <vector>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#include "multi_camera_sync/SyncStatus.h"
#include "multi_camera_sync/sync_config.hpp"

namespace multi_camera_sync {

class SyncEngine {
 public:
  SyncEngine(ros::NodeHandle& nh, ros::NodeHandle& pnh, const SyncConfig& config);

  void start();

 private:
  void onImage(size_t camera_index, const sensor_msgs::ImageConstPtr& msg);
  void tryPublish();
  void publishStatus(const ros::Time& reference_stamp, double max_skew_s,
                     const std::vector<double>& stamp_skew_s);
  int findBestMatch(const std::deque<sensor_msgs::ImageConstPtr>& buffer,
                    const ros::Time& anchor_stamp) const;
  void removeThrough(const std::vector<int>& match_idx);

  SyncConfig config_;
  size_t reference_index_ = 0;

  ros::Publisher status_pub_;
  std::vector<ros::Subscriber> subscribers_;
  std::vector<ros::Publisher> publishers_;
  std::vector<std::deque<sensor_msgs::ImageConstPtr>> buffers_;

  uint64_t sync_count_ = 0;
  ros::Time last_status_stamp_;
  double sync_rate_hz_ = 0.0;
};

}  // namespace multi_camera_sync
