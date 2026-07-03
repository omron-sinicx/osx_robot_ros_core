"""Camera image subscriptions for data collection (rclpy port of osx_gym_env.ImageRecorder)."""

import time
from collections import deque

import numpy as np
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class ImageRecorder:
    def __init__(self, node, camera_names=None, is_debug=False, use_torch=False, data_type="float32"):
        self.node = node
        self.is_debug = is_debug
        self.bridge = CvBridge()
        self.camera_names = list(camera_names or [])
        self.data_type = data_type
        self.use_torch = use_torch

        for cam_name in self.camera_names:
            setattr(self, f"{cam_name}_image", None)
            setattr(self, f"{cam_name}_timestamp", None)
            if self.is_debug:
                setattr(self, f"{cam_name}_timestamps", deque(maxlen=50))
            topic = f"/{cam_name}/color/image_raw"
            node.create_subscription(
                Image,
                topic,
                lambda msg, name=cam_name: self._image_cb(msg, name),
                qos_profile_sensor_data,
            )

        time.sleep(0.5)

    def _image_cb(self, data, cam_name):
        if self.use_torch:
            import torch

            image = torch.from_numpy(
                self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            ).cuda()
        else:
            image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

        setattr(self, f"{cam_name}_image", image)
        setattr(self, f"{cam_name}_timestamp", time.monotonic())

        if self.is_debug:
            stamp = data.header.stamp
            getattr(self, f"{cam_name}_timestamps").append(stamp.sec + stamp.nanosec * 1e-9)

    def get_images(self):
        image_dict = {}
        for cam_name in self.camera_names:
            timestamp = getattr(self, f"{cam_name}_timestamp", None)
            if timestamp is not None and time.monotonic() - timestamp > 0.5:
                self.node.get_logger().error(
                    "Image is too old; ignoring", throttle_duration_sec=1.0
                )
                image_dict[cam_name] = None
            else:
                image = getattr(self, f"{cam_name}_image", None)
                if image is not None and self.data_type == "float32":
                    image = (image / 255.0).astype(np.float32)
                image_dict[cam_name] = image
        return image_dict

    def print_diagnostics(self):
        def dt_helper(values):
            values = np.array(values)
            return np.mean(values[1:] - values[:-1])

        for cam_name in self.camera_names:
            timestamps = getattr(self, f"{cam_name}_timestamps", None)
            if timestamps and len(timestamps) > 1:
                image_freq = 1.0 / dt_helper(list(timestamps))
                print(f"{cam_name} image_freq={image_freq:.2f}")
        print()
