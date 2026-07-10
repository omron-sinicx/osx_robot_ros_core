"""Sample feeders: turn live topics or recorded bags into per-source Samples.

Two feeders produce the same ``{source_key: Sample}`` view consumed by
ObservationAssembler (see observation_assembler.py for the canonical keys):

    RosSampleFeeder - online: subscribes the hot topics (tcp_nodelay), caches
        the latest message per source, decodes on read. Grab-latest semantics.
    BagCursor       - offline: per-source time-sorted sample lists from a
        rosbag; advance(t) returns the latest sample with stamp <= t per
        source (causal zero-order hold).

rospy / rosbag / cv_bridge imports are kept out of module level so the module
is importable in any environment.
"""

import bisect
from typing import Callable, Dict, List, Optional

import numpy as np

from osx_ur5e.observation_assembler import Sample

# UR joint order used across ur_control / ur_pykdl (constants.JOINT_ORDER).
JOINT_ORDER = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def reorder_joint_state(names, values, joint_order=None) -> np.ndarray:
    """Reorder a JointState field into JOINT_ORDER (drivers publish alphabetically)."""
    joint_order = joint_order or JOINT_ORDER
    values = list(values)
    return np.array([values[list(names).index(j)] for j in joint_order], dtype=np.float64)


def joint_state_to_value(msg, joint_order=None) -> np.ndarray:
    """sensor_msgs/JointState -> (12,) qpos ++ qvel in JOINT_ORDER."""
    qpos = reorder_joint_state(msg.name, msg.position, joint_order)
    qvel = (
        reorder_joint_state(msg.name, msg.velocity, joint_order)
        if msg.velocity else np.zeros(len(qpos))
    )
    return np.concatenate([qpos, qvel])


def wrench_to_value(msg) -> np.ndarray:
    """geometry_msgs/WrenchStamped -> (6,)."""
    w = msg.wrench
    return np.array([w.force.x, w.force.y, w.force.z,
                     w.torque.x, w.torque.y, w.torque.z], dtype=np.float64)


def pose_to_value(msg) -> np.ndarray:
    """geometry_msgs/PoseStamped -> (7,) xyz + quaternion xyzw."""
    p, q = msg.pose.position, msg.pose.orientation
    return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float64)


def decode_image_msg(msg) -> np.ndarray:
    """sensor_msgs/Image or CompressedImage -> HWC uint8, canonical RGB.

    All sources are canonicalized to RGB channel order regardless of the
    publisher's convention (RealSense publishes rgb8, usb_cam/GoPro bgr8,
    compressed_image_transport JPEGs decode to BGR via OpenCV), so the
    dataset and the online feeder always agree.
    """
    if hasattr(msg, "format"):  # CompressedImage
        import cv2
        bgr = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # Raw Image: decode without cv_bridge (pure numpy) for portability.
    channels = {"rgb8": 3, "bgr8": 3, "mono8": 1}.get(msg.encoding)
    if channels is None:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
    if msg.encoding == "bgr8":
        img = img[..., ::-1]
    return img.squeeze() if channels == 1 else img


def stamp_of(msg, fallback: float) -> float:
    """Header stamp in seconds, or ``fallback`` when absent/zero."""
    header = getattr(msg, "header", None)
    if header is None:
        return fallback
    stamp = header.stamp.to_sec()
    return stamp if stamp > 0.0 else fallback


# ---------------------------------------------------------------------------
# Online feeder
# ---------------------------------------------------------------------------

class RosSampleFeeder:
    """Subscribe the hot topics and expose the latest Sample per source key.

    Grab-latest semantics identical to what the offline BagCursor reproduces:
    each source is whatever was most recently received, stamped with its
    acquisition instant (header stamp).
    """

    def __init__(
        self,
        camera_names: List[str],
        wrench_topic: Optional[str] = None,
        joint_states_topic: str = "/joint_states",
        image_topic_template: Optional[str] = None,
        with_action_sources: bool = False,
        gello_ns: str = "/dynamixel_workbench",
    ):
        import rospy
        from geometry_msgs.msg import PoseStamped, WrenchStamped
        from sensor_msgs.msg import CompressedImage, Image, JointState

        self._rospy = rospy
        self.camera_names = list(camera_names or [])
        # (receive_time, Sample-with-raw-msg) per canonical key
        self._latest: Dict[str, tuple] = {}
        self._decoders: Dict[str, Callable] = {}

        def subscribe(key, topic, msg_type, to_value):
            self._decoders[key] = to_value
            rospy.Subscriber(topic, msg_type, self._make_cb(key),
                             queue_size=1, tcp_nodelay=True)

        subscribe("joint_states", joint_states_topic, JointState, joint_state_to_value)
        subscribe("wrench", wrench_topic or self._resolve_wrench_topic(), WrenchStamped, wrench_to_value)

        for cam in self.camera_names:
            topic = (
                image_topic_template.format(cam_name=cam)
                if image_topic_template else self._resolve_image_topic(cam)
            )
            msg_type = CompressedImage if topic.endswith("/compressed") else Image
            subscribe(f"images.{cam}", topic, msg_type, decode_image_msg)
            rospy.loginfo("RosSampleFeeder: %s -> %s", cam, topic)

        if with_action_sources:
            from osx_msgs.msg import Float64ArrayStamped

            subscribe("target_frame", "/cartesian_compliance_controller/target_frame",
                      PoseStamped, pose_to_value)
            subscribe("stiffness", "/data_collection/stiffness_command",
                      Float64ArrayStamped,
                      lambda m: np.asarray(m.data, dtype=np.float64))
            subscribe("gello_joints", "/data_collection/gello_joints",
                      JointState, lambda m: np.asarray(m.position, dtype=np.float64))

    # -- topic resolution ---------------------------------------------------

    def _published_topics(self):
        return {t for t, _ in self._rospy.get_published_topics()}

    def _resolve_wrench_topic(self) -> str:
        published = self._published_topics()
        return "/wrench/filtered" if "/wrench/filtered" in published else "/wrench"

    def _resolve_image_topic(self, cam_name: str, wait_s: float = 2.0) -> str:
        """RealSense vs usb_cam topic layout, raw per-camera topics only."""
        rospy = self._rospy
        candidates = [
            f"/{cam_name}/color/image_raw",
            f"/{cam_name}/image_raw",
        ]
        deadline = rospy.get_time() + wait_s
        while not rospy.is_shutdown():
            published = self._published_topics()
            for topic in candidates:
                if topic in published:
                    return topic
            if rospy.get_time() >= deadline:
                break
            rospy.sleep(0.1)
        rospy.logwarn("RosSampleFeeder: no image topic found for %s; defaulting to %s",
                      cam_name, candidates[0])
        return candidates[0]

    # -- caching ------------------------------------------------------------

    def _make_cb(self, key: str):
        def cb(msg):
            now = self._rospy.get_time()
            self._latest[key] = (now, msg, stamp_of(msg, fallback=now))
        return cb

    def get_latest(self, keys: Optional[List[str]] = None) -> Dict[str, Sample]:
        """Latest Sample per source key (decoded). Missing sources are omitted."""
        out = {}
        for key in (keys or list(self._latest)):
            entry = self._latest.get(key)
            if entry is None:
                continue
            _, msg, stamp = entry
            out[key] = Sample(stamp=stamp, value=self._decoders[key](msg))
        return out

    def get_images(self, camera_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Latest decoded frame per camera (None when not yet received).

        Kept API-compatible with what comet's eval artifacts expect from the
        environment's camera feed (capture_video_frame).
        """
        out = {}
        for cam in (camera_names or self.camera_names):
            sample = self.get_latest([f"images.{cam}"]).get(f"images.{cam}")
            out[cam] = None if sample is None else sample.value
        return out

    def age_s(self, key: str) -> float:
        """Seconds since the source's latest message was received (inf if none)."""
        entry = self._latest.get(key)
        if entry is None:
            return float("inf")
        return self._rospy.get_time() - entry[0]

    def wait_until_fresh(self, max_age_s: float = 0.5, timeout_s: float = 5.0,
                         keys: Optional[List[str]] = None) -> bool:
        """Block until every requested source has a recent message."""
        rospy = self._rospy
        keys = keys or ["joint_states", "wrench"] + [f"images.{c}" for c in self.camera_names]
        deadline = rospy.get_time() + timeout_s
        while not rospy.is_shutdown():
            if all(self.age_s(k) <= max_age_s for k in keys):
                return True
            if rospy.get_time() >= deadline:
                stale = [k for k in keys if self.age_s(k) > max_age_s]
                rospy.logwarn("RosSampleFeeder: stale sources after %.1fs: %s", timeout_s, stale)
                return False
            rospy.sleep(0.02)
        return False


# ---------------------------------------------------------------------------
# Offline feeder
# ---------------------------------------------------------------------------

class BagCursor:
    """Causal zero-order-hold cursor over per-source sample streams.

    ``streams[key]`` is a time-sorted list of (stamp, value_or_thunk); a thunk
    (zero-arg callable) is materialized lazily on access so image streams can
    be decoded per tick instead of held in memory. ``advance(t)`` must be
    called with non-decreasing t (two-pointer walk, O(total samples) overall).
    """

    def __init__(self, streams: Dict[str, List[tuple]]):
        self._streams = {k: sorted(v, key=lambda e: e[0]) for k, v in streams.items()}
        self._idx = {k: -1 for k in self._streams}  # index of latest sample <= t
        self._decoded: Dict[str, tuple] = {}        # key -> (idx, value) memo
        self._last_t = -float("inf")

    @property
    def keys(self) -> List[str]:
        return list(self._streams)

    def first_stamp(self, key: str) -> float:
        stream = self._streams[key]
        return stream[0][0] if stream else float("inf")

    def last_stamp(self, key: str) -> float:
        stream = self._streams[key]
        return stream[-1][0] if stream else -float("inf")

    def advance(self, tick_time: float) -> Dict[str, Sample]:
        """Move to ``tick_time`` and return the latest causal Sample per key.

        Sources with no sample yet (stamp <= tick_time) are omitted.
        """
        if tick_time < self._last_t:
            raise ValueError(
                f"BagCursor.advance must be called with non-decreasing time "
                f"({tick_time} < {self._last_t})"
            )
        self._last_t = tick_time

        out = {}
        for key, stream in self._streams.items():
            i = self._idx[key]
            while i + 1 < len(stream) and stream[i + 1][0] <= tick_time:
                i += 1
            self._idx[key] = i
            if i < 0:
                continue
            stamp, value = stream[i]
            if callable(value):
                memo = self._decoded.get(key)
                if memo is not None and memo[0] == i:
                    value = memo[1]
                else:
                    value = value()
                    self._decoded[key] = (i, value)
            out[key] = Sample(stamp=stamp, value=value)
        return out

    def seek(self, tick_time: float) -> Dict[str, Sample]:
        """Jump (possibly backwards) to ``tick_time`` via binary search."""
        self._last_t = tick_time
        for key, stream in self._streams.items():
            stamps = [e[0] for e in stream]
            self._idx[key] = bisect.bisect_right(stamps, tick_time) - 1
        self._decoded.clear()
        return self.advance(tick_time)
