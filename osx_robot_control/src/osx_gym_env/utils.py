import numpy as np
import time

import torch

import rospy
from ur_control import transformations


class ImageRecorder:
    def __init__(self, init_node=True, is_debug=False, camera_names=[], use_torch=False, data_type='float32'):
        from collections import deque
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image
        self.is_debug = is_debug
        self.bridge = CvBridge()
        self.camera_names = camera_names
        self.data_type = data_type
        if init_node:
            rospy.init_node('image_recorder', anonymous=True)
        for cam_name in self.camera_names:
            setattr(self, f'{cam_name}_image', None)
            setattr(self, f'{cam_name}_secs', None)
            setattr(self, f'{cam_name}_nsecs', None)
            setattr(self, f'{cam_name}_timestamps', deque(maxlen=50))
            rospy.Subscriber(f"/{cam_name}/color/image_raw", Image, self.image_cb, callback_args={'cam_name': cam_name, 'use_torch': use_torch})
        rospy.sleep(0.5)

    def image_cb(self, data, args):
        cam_name = args['cam_name']
        if args['use_torch']:
            setattr(self, f'{cam_name}_image', torch.from_numpy(self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')).cuda())
        else:
            setattr(self, f'{cam_name}_image', self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough'))
        setattr(self, f'{cam_name}_timestamp', rospy.get_time())

        if self.is_debug:
            getattr(self, f'{cam_name}_timestamps').append(data.header.stamp.secs + (data.header.stamp.nsecs * 1e-9))

    def get_images(self):
        image_dict = dict()
        for cam_name in self.camera_names:
            image = getattr(self, f'{cam_name}_image')
            if image is None:
                rospy.logerr_throttle(1, f"No image received yet for {cam_name}")
                image_dict[cam_name] = None
            elif hasattr(self, f'{cam_name}_timestamp') and rospy.get_time() - getattr(self, f'{cam_name}_timestamp') > 0.5:
                rospy.logerr_throttle(1, "Image is too old! ignoring")
                image_dict[cam_name] = None
            else:
                if self.data_type == 'float32':
                    image = (image/255.0).astype(np.float32)
                image_dict[cam_name] = image
        return image_dict

    def cameras_ready(self):
        """Return True once all cameras have received at least one frame."""
        return all(getattr(self, f'{cam_name}_image') is not None for cam_name in self.camera_names)

    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)
        for cam_name in self.camera_names:
            image_freq = 1 / dt_helper(getattr(self, f'{cam_name}_timestamps'))
            print(f'{cam_name} {image_freq=:.2f}')
        print()


def compute_eef_velocity(current_pose, previous_pose, dt):
    """
        Assume that the poses are given as [x,y,z] + [quat(4)]
    """
    linear_velocity = (current_pose[:3] - previous_pose[:3]) / dt
    angular_velocity = transformations.angular_velocity_from_quaternions(current_pose[3:], previous_pose[3:], dt)
    return np.concatenate((linear_velocity, angular_velocity))


if __name__ == '__main__':
    names = ["extra_camera"]
    im = ImageRecorder(init_node=True, is_debug=True, camera_names=names)
    for _ in range(2):
        rospy.sleep(1)
        im.print_diagnostics()
