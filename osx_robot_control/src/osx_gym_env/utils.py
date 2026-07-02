import numpy as np

import torch

import rospy
from ur_control import transformations


class ImageRecorder:
    def __init__(
        self,
        init_node=True,
        is_debug=False,
        camera_names=None,
        use_torch=False,
        data_type='float32',
        max_image_age_s=1.0,
        sync_ns=None,
        image_topic_template=None,
    ):
        from collections import deque
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image
        self.is_debug = is_debug
        self.bridge = CvBridge()
        self.camera_names = list(camera_names or [])
        self.use_torch = use_torch
        self.data_type = data_type
        self.max_image_age_s = max_image_age_s
        self.sync_ns = sync_ns
        if image_topic_template is None:
            if sync_ns:
                image_topic_template = '/{sync_ns}/{cam_name}/image_raw'
            else:
                image_topic_template = '/{cam_name}/color/image_raw'
        self.image_topic_template = image_topic_template
        if init_node:
            rospy.init_node('image_recorder', anonymous=True)
        for cam_name in self.camera_names:
            setattr(self, f'{cam_name}_msg', None)
            setattr(self, f'{cam_name}_received_at', None)
            setattr(self, f'{cam_name}_timestamps', deque(maxlen=50))
            topic = self.image_topic_template.format(
                sync_ns=sync_ns or '',
                cam_name=cam_name,
            )
            rospy.Subscriber(
                topic,
                Image,
                self.image_cb,
                callback_args={'cam_name': cam_name},
                queue_size=1,
            )
        rospy.sleep(0.5)

    def image_cb(self, data, args):
        cam_name = args['cam_name']
        setattr(self, f'{cam_name}_msg', data)
        setattr(self, f'{cam_name}_received_at', rospy.Time.now())

        if self.is_debug:
            getattr(self, f'{cam_name}_timestamps').append(
                data.header.stamp.secs + (data.header.stamp.nsecs * 1e-9)
            )

    def _message_age_s(self, msg, cam_name=None) -> float:
        import rospy
        if msg is None:
            return float('inf')
        if cam_name is not None:
            received_at = getattr(self, f'{cam_name}_received_at', None)
            if received_at is not None:
                return (rospy.Time.now() - received_at).to_sec()
        if msg.header.stamp == rospy.Time():
            return float('inf')
        return (rospy.Time.now() - msg.header.stamp).to_sec()

    def _decode_image(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if self.use_torch:
            image = torch.from_numpy(image).cuda()
        elif self.data_type == 'float32':
            image = (image / 255.0).astype(np.float32)
        return image

    def get_images(self, camera_names=None, max_image_age_s=None):
        import rospy
        image_dict = dict()
        max_age = self.max_image_age_s if max_image_age_s is None else max_image_age_s
        for cam_name in (camera_names or self.camera_names):
            msg = getattr(self, f'{cam_name}_msg', None)
            if msg is None:
                image_dict[cam_name] = None
                continue

            age_s = self._message_age_s(msg, cam_name=cam_name)
            if age_s > max_age:
                rospy.logerr_throttle(
                    1,
                    "Image is too old for %s (age=%.2fs, max=%.2fs); ignoring",
                    cam_name,
                    age_s,
                    max_age,
                )
                image_dict[cam_name] = None
                continue

            image_dict[cam_name] = self._decode_image(msg)
        return image_dict

    def wait_for_fresh_images(
        self,
        camera_names=None,
        timeout_s=2.0,
        max_image_age_s=None,
    ):
        """Wait until all requested cameras have a recent frame."""
        import rospy

        requested = list(camera_names or self.camera_names)
        if not requested:
            return {}

        deadline = rospy.get_time() + timeout_s
        last_images = {}
        while rospy.get_time() <= deadline and not rospy.is_shutdown():
            last_images = self.get_images(requested, max_image_age_s=max_image_age_s)
            missing = [cam for cam in requested if last_images.get(cam) is None]
            if not missing:
                return last_images
            rospy.sleep(0.05)

        return last_images

    def get_diagnostics(self):
        import rospy
        diagnostics = {}
        stamps = []
        for cam_name in self.camera_names:
            msg = getattr(self, f'{cam_name}_msg', None)
            stamp = None if msg is None else msg.header.stamp.to_sec()
            if stamp is not None:
                stamps.append(stamp)
            diagnostics[cam_name] = {
                'has_message': msg is not None,
                'age_s': None if msg is None else self._message_age_s(msg, cam_name=cam_name),
                'header_age_s': None if msg is None else (
                    float('inf') if msg.header.stamp == rospy.Time()
                    else (rospy.Time.now() - msg.header.stamp).to_sec()
                ),
                'stamp': stamp,
                'now': rospy.Time.now().to_sec(),
            }
        if len(stamps) >= 2:
            diagnostics['max_stamp_skew_s'] = max(stamps) - min(stamps)
        diagnostics['sync_ns'] = self.sync_ns
        return diagnostics

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
