#include "multi_camera_sync/sync_engine.hpp"

#include <algorithm>
#include <cmath>

namespace multi_camera_sync {

namespace {

double stampToSec(const ros::Time& stamp) {
  return stamp.sec + stamp.nsec * 1e-9;
}

}  // namespace

SyncEngine::SyncEngine(ros::NodeHandle& nh, ros::NodeHandle& pnh, const SyncConfig& config)
    : config_(config) {
  if (config_.queue_size < 1) {
    throw std::runtime_error("queue_size must be >= 1");
  }

  reference_index_ = 0;
  for (size_t i = 0; i < config_.cameras.size(); ++i) {
    if (config_.cameras[i].name == config_.reference_camera) {
      reference_index_ = i;
      break;
    }
  }
  if (config_.cameras[reference_index_].name != config_.reference_camera) {
    throw std::runtime_error("reference_camera '" + config_.reference_camera + "' not found");
  }

  const std::string status_topic = "/" + config_.output_namespace + "/status";
  status_pub_ = nh.advertise<SyncStatus>(status_topic, 1);

  buffers_.resize(config_.cameras.size());
  publishers_.reserve(config_.cameras.size());
  subscribers_.reserve(config_.cameras.size());

  for (size_t i = 0; i < config_.cameras.size(); ++i) {
    const auto& camera = config_.cameras[i];
    const std::string output_topic =
        "/" + config_.output_namespace + "/" + camera.name + "/image_raw";
    publishers_.push_back(nh.advertise<sensor_msgs::Image>(output_topic, 1));

    const size_t camera_index = i;
    subscribers_.push_back(nh.subscribe<sensor_msgs::Image>(
        camera.topic, static_cast<uint32_t>(config_.queue_size),
        [this, camera_index](const sensor_msgs::ImageConstPtr& msg) {
          onImage(camera_index, msg);
        }));
  }

  ROS_INFO_STREAM("multi_camera_sync ready: cameras="
                  << config_.reference_camera << " + "
                  << (config_.cameras.size() - 1)
                  << " others, slop=" << config_.slop << "s, queue="
                  << config_.queue_size << ", output=/" << config_.output_namespace << "/");
}

void SyncEngine::start() {}

void SyncEngine::onImage(size_t camera_index, const sensor_msgs::ImageConstPtr& msg) {
  auto& buffer = buffers_[camera_index];
  buffer.push_back(msg);
  while (buffer.size() > static_cast<size_t>(config_.queue_size)) {
    buffer.pop_front();
  }
  tryPublish();
}

int SyncEngine::findBestMatch(const std::deque<sensor_msgs::ImageConstPtr>& buffer,
                              const ros::Time& anchor_stamp) const {
  int best_idx = -1;
  double best_diff = config_.slop + 1.0;
  for (size_t i = 0; i < buffer.size(); ++i) {
    const double diff = std::abs((buffer[i]->header.stamp - anchor_stamp).toSec());
    if (diff <= config_.slop && diff < best_diff) {
      best_diff = diff;
      best_idx = static_cast<int>(i);
    }
  }
  return best_idx;
}

void SyncEngine::tryPublish() {
  const auto& ref_buffer = buffers_[reference_index_];
  if (ref_buffer.empty()) {
    return;
  }

  for (size_t ref_pos = 0; ref_pos < ref_buffer.size(); ++ref_pos) {
    const sensor_msgs::ImageConstPtr& anchor = ref_buffer[ref_pos];
    const ros::Time anchor_stamp = anchor->header.stamp;

    std::vector<int> match_idx(config_.cameras.size(), -1);
    match_idx[reference_index_] = static_cast<int>(ref_pos);

    bool all_found = true;
    for (size_t i = 0; i < config_.cameras.size(); ++i) {
      if (i == reference_index_) {
        continue;
      }
      match_idx[i] = findBestMatch(buffers_[i], anchor_stamp);
      if (match_idx[i] < 0) {
        all_found = false;
        break;
      }
    }

    if (!all_found) {
      continue;
    }

    std::vector<double> stamp_skew_s;
    stamp_skew_s.reserve(config_.cameras.size());
    double max_skew_s = 0.0;

    for (size_t i = 0; i < config_.cameras.size(); ++i) {
      const sensor_msgs::ImageConstPtr& source = buffers_[i][static_cast<size_t>(match_idx[i])];
      const double skew_s =
          std::abs((source->header.stamp - anchor_stamp).toSec());
      stamp_skew_s.push_back(skew_s);
      max_skew_s = std::max(max_skew_s, skew_s);

      sensor_msgs::ImagePtr output(new sensor_msgs::Image(*source));
      output->header.stamp = anchor_stamp;
      publishers_[i].publish(output);
    }

    removeThrough(match_idx);
    sync_count_++;
    publishStatus(anchor_stamp, max_skew_s, stamp_skew_s);
    return;
  }
}

void SyncEngine::removeThrough(const std::vector<int>& match_idx) {
  for (size_t i = 0; i < buffers_.size(); ++i) {
    const size_t remove_through = static_cast<size_t>(match_idx[i]) + 1;
    if (remove_through >= buffers_[i].size()) {
      buffers_[i].clear();
    } else {
      buffers_[i].erase(buffers_[i].begin(),
                        buffers_[i].begin() + static_cast<std::ptrdiff_t>(remove_through));
    }
  }
}

void SyncEngine::publishStatus(const ros::Time& reference_stamp, double max_skew_s,
                               const std::vector<double>& stamp_skew_s) {
  SyncStatus status;
  status.header.stamp = reference_stamp;
  status.camera_names.reserve(config_.cameras.size());
  for (const auto& camera : config_.cameras) {
    status.camera_names.push_back(camera.name);
  }
  status.stamp_skew_s = stamp_skew_s;
  status.max_skew_s = max_skew_s;
  status.sync_count = sync_count_;

  if (!last_status_stamp_.isZero()) {
    const double dt = (reference_stamp - last_status_stamp_).toSec();
    if (dt > 0.0) {
      sync_rate_hz_ = 1.0 / dt;
    }
  }
  last_status_stamp_ = reference_stamp;
  status.sync_rate_hz = sync_rate_hz_;

  status_pub_.publish(status);
}

}  // namespace multi_camera_sync
