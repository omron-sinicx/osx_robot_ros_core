"""Per-rollout video capture from cameras that are not part of the policy's input.

The env's feeder is deliberately not used here: its camera list is the
checkpoint's frozen observation contract, and everything in it is also tiled
into ``rollout_X.mp4``. A recording-only camera gets its own subscriber instead,
so it cannot change what the policy sees or what the main video shows.

Duck-typed for ``comet.eval.runner.RealRobotEvalRunner``'s
``extra_camera_recorder`` hook (``camera_names`` / ``start_rollout`` / ``capture``
/ ``finish_rollout``), the same way ``osx_ur5e.rosbag_recorder.RosbagRecorder``
is duck-typed for ``rollout_recorder``.

``capture`` runs inside the control loop, so it only keeps a reference to the
latest message; decoding happens in ``finish_rollout``, once the robot is home.
"""

from __future__ import annotations

import logging

import numpy as np
import rospy
from sensor_msgs.msg import Image

from osx_ur5e.sample_feeders import decode_image_msg, resolve_image_topic

logger = logging.getLogger(__name__)

# rospy's default 64 KiB receive buffer is smaller than one raw frame, which
# defeats queue_size=1 and delivers increasingly old images.
_SUBSCRIBER_BUFF_SIZE = 2**24


class ExtraCameraRecorder:
    """Buffers one frame per control step for each recording-only camera."""

    def __init__(self, camera_names: list[str], max_image_age_s: float = 1.0) -> None:
        self.camera_names = list(dict.fromkeys(camera_names))
        self.max_image_age_s = max_image_age_s
        self._latest: dict[str, tuple[float, Image]] = {}
        self._msgs: dict[str, list[Image]] = {cam: [] for cam in self.camera_names}
        self._stale_steps: dict[str, int] = {cam: 0 for cam in self.camera_names}
        self._subscribers = []
        for cam in self.camera_names:
            topic = resolve_image_topic(cam)
            self._subscribers.append(
                rospy.Subscriber(
                    topic,
                    Image,
                    self._make_cb(cam),
                    queue_size=1,
                    buff_size=_SUBSCRIBER_BUFF_SIZE,
                    tcp_nodelay=True,
                )
            )
            logger.info("ExtraCameraRecorder: %s -> %s", cam, topic)

    def _make_cb(self, cam: str):
        def cb(msg: Image) -> None:
            self._latest[cam] = (rospy.get_time(), msg)

        return cb

    def start_rollout(self) -> None:
        """Drop anything buffered by a previous (possibly discarded) attempt."""
        for cam in self.camera_names:
            self._msgs[cam] = []
            self._stale_steps[cam] = 0

    def capture(self) -> None:
        """Keep the latest message of each camera for this control step.

        A stale frame is still kept, so the video stays step-aligned with
        ``rollout_X.mp4``; ``finish_rollout`` reports how many there were.
        """
        now = rospy.get_time()
        for cam in self.camera_names:
            entry = self._latest.get(cam)
            if entry is None:
                self._stale_steps[cam] += 1
                continue
            received_at, msg = entry
            if now - received_at > self.max_image_age_s:
                self._stale_steps[cam] += 1
            self._msgs[cam].append(msg)

    def finish_rollout(self) -> dict[str, list[np.ndarray]]:
        """Decode the buffered messages to HWC uint8 RGB frames, per camera.

        A camera that cannot be decoded yields no frames rather than raising.
        """
        frames: dict[str, list[np.ndarray]] = {}
        for cam in self.camera_names:
            msgs, self._msgs[cam] = self._msgs[cam], []
            if self._stale_steps[cam]:
                logger.warning(
                    "ExtraCameraRecorder: %s had no fresh frame on %d steps this rollout.",
                    cam,
                    self._stale_steps[cam],
                )
            try:
                frames[cam] = [_to_rgb_frame(msg) for msg in msgs]
            except ValueError as exc:
                logger.warning("ExtraCameraRecorder: cannot decode %s: %s", cam, exc)
                frames[cam] = []
        return frames


def _to_rgb_frame(msg: Image) -> np.ndarray:
    frame = decode_image_msg(msg)
    return np.stack([frame] * 3, axis=-1) if frame.ndim == 2 else frame
