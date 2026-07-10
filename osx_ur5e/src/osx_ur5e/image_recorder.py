import rospy
from collections import deque
from dataclasses import dataclass, field


_REALSENSE_IMAGE_TOPIC = '/{cam_name}/color/image_raw'
_USB_CAM_IMAGE_TOPIC = '/{cam_name}/image_raw'


@dataclass
class _CameraState:
    """All per-camera state, grouped in one place."""
    topic: str
    msg: object = None                 # latest sensor_msgs/Image
    received_at: object = None         # rospy.Time when msg arrived
    timestamps: deque = field(default_factory=lambda: deque(maxlen=50))


def _resolve_image_topic(cam_name, sync_ns=None, wait_s=2.0):
    """Pick RealSense vs usb_cam/GoPro topic layout for a camera."""
    if sync_ns:
        return f'/{sync_ns}/{cam_name}/image_raw'

    candidates = [
        _REALSENSE_IMAGE_TOPIC.format(cam_name=cam_name),
        _USB_CAM_IMAGE_TOPIC.format(cam_name=cam_name),
    ]

    deadline = rospy.get_time() + wait_s
    while not rospy.is_shutdown():
        published = {topic for topic, _ in rospy.get_published_topics()}
        for topic in candidates:
            if topic in published:
                return topic
        if rospy.get_time() >= deadline:
            break
        rospy.sleep(0.1)

    rospy.logwarn(
        'ImageRecorder: no image topic found for %s within %.1fs; '
        'defaulting to %s',
        cam_name,
        wait_s,
        candidates[0],
    )
    return candidates[0]


class ImageRecorder:
    def __init__(
        self,
        init_node=True,
        camera_names=None,
        max_image_age_s=1.0,
        sync_ns=None,
        image_topic_template=None,
    ):
        from cv_bridge import CvBridge
        self.bridge = CvBridge()
        self.camera_names = list(camera_names or [])
        self.max_image_age_s = max_image_age_s
        self.sync_ns = sync_ns
        self.image_topic_template = image_topic_template
        if init_node:
            rospy.init_node('image_recorder', anonymous=True)
        self.cameras = {}
        for cam_name in self.camera_names:
            topic = self._topic_for(cam_name)
            self.cameras[cam_name] = _CameraState(topic=topic)
            rospy.loginfo('ImageRecorder: %s -> %s', cam_name, topic)
            self._subscribe(cam_name, topic)
        rospy.sleep(0.5)

    def _topic_for(self, cam_name):
        if self.image_topic_template is None:
            return _resolve_image_topic(cam_name, sync_ns=self.sync_ns)
        return self.image_topic_template.format(
            sync_ns=self.sync_ns or '',
            cam_name=cam_name,
        )

    def _subscribe(self, cam_name, topic):
        from sensor_msgs.msg import Image
        rospy.Subscriber(
            topic,
            Image,
            self.image_cb,
            callback_args={'cam_name': cam_name},
            queue_size=1,
        )

    def image_cb(self, data, args):
        cam = self.cameras[args['cam_name']]
        cam.msg = data
        cam.received_at = rospy.Time.now()
        cam.timestamps.append(
            data.header.stamp.secs + (data.header.stamp.nsecs * 1e-9)
        )

    def _message_age_s(self, cam) -> float:
        """Age of a camera's latest frame in seconds (inf if none/undated)."""
        if cam.msg is None:
            return float('inf')
        if cam.received_at is not None:
            return (rospy.Time.now() - cam.received_at).to_sec()
        if cam.msg.header.stamp == rospy.Time():
            return float('inf')
        return (rospy.Time.now() - cam.msg.header.stamp).to_sec()

    def _decode_image(self, msg):
        return self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def _resolve_max_age(self, max_image_age_s):
        """Fall back to the instance default when no override is given."""
        return self.max_image_age_s if max_image_age_s is None else max_image_age_s

    def _header_stamp_s(self, cam_name):
        """Header stamp (acquisition instant) of a camera's latest frame, in seconds."""
        return self.cameras[cam_name].msg.header.stamp.to_sec()

    def _fresh_image(self, cam_name, max_age):
        """Decoded frame for one camera, or None if missing/stale."""
        cam = self.cameras.get(cam_name)
        if cam is None or cam.msg is None:
            return None
        age_s = self._message_age_s(cam)
        if age_s > max_age:
            rospy.logerr_throttle(
                1,
                "Image is too old for %s (age=%.2fs, max=%.2fs); ignoring",
                cam_name,
                age_s,
                max_age,
            )
            return None
        return self._decode_image(cam.msg)

    def get_images(self, camera_names=None, max_image_age_s=None):
        max_age = self._resolve_max_age(max_image_age_s)
        return {
            cam_name: self._fresh_image(cam_name, max_age)
            for cam_name in (camera_names or self.camera_names)
        }

    def get_images_with_stamp(self, camera_names=None, max_image_age_s=None):
        """Like get_images(), but also return each frame's capture time.

        Returns (image_dict, stamp_dict) where stamp_dict[cam] is the ROS
        header stamp in seconds (the frame's true acquisition instant), or
        None when the frame is missing/stale. Use it to align vision against
        the higher-rate proprioception recorded in the same loop iteration.
        """
        max_age = self._resolve_max_age(max_image_age_s)
        image_dict = dict()
        stamp_dict = dict()
        for cam_name in (camera_names or self.camera_names):
            image = self._fresh_image(cam_name, max_age)
            image_dict[cam_name] = image
            stamp_dict[cam_name] = (
                None if image is None else self._header_stamp_s(cam_name)
            )
        return image_dict, stamp_dict

    def wait_for_fresh_images(
        self,
        camera_names=None,
        timeout_s=2.0,
        max_image_age_s=None,
    ):
        """Wait until all requested cameras have a recent frame."""
        requested = list(camera_names or self.camera_names)
        if not requested:
            return {}

        deadline = rospy.get_time() + timeout_s
        last_images = {}
        while rospy.get_time() <= deadline and not rospy.is_shutdown():
            last_images = self.get_images(requested, max_image_age_s=max_image_age_s)
            if all(last_images.get(cam) is not None for cam in requested):
                return last_images
            rospy.sleep(0.05)

        return last_images

    def _header_age_s(self, cam):
        """Age from the frame's header stamp (inf if undated, None if no frame)."""
        if cam.msg is None:
            return None
        if cam.msg.header.stamp == rospy.Time():
            return float('inf')
        return (rospy.Time.now() - cam.msg.header.stamp).to_sec()
