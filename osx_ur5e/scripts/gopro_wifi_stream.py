#!/usr/bin/env python3
"""Publish GoPro WiFi preview stream (UDP MPEG-TS) as a ROS image topic."""

import os
import queue
import threading
import urllib.error
import urllib.request

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# GoPro sends HEVC inside MPEG-TS. Keep enough UDP fifo for the decoder to
# receive reference frames; latency is managed by dropping stale frames in the
# reader thread instead of starving the HEVC decoder.
_DEFAULT_STREAM_URL = "udp://127.0.0.1:8554?overrun_nonfatal=1&fifo_size=131072"


def _set_gopro_stream(base_url: str, start: bool, mode: str = "webcam",
                      port: int = 8554, resolution: int = 7, fov: int = 0,
                      protocol: int = 1) -> None:
    base = base_url.rstrip("/")
    if mode == "webcam":
        if start:
            url = f"{base}/gopro/webcam/start?res={resolution}&fov={fov}&port={port}&protocol={protocol}"
        else:
            url = f"{base}/gopro/webcam/stop"
    else:
        url = f"{base}/gopro/camera/stream/{'start' if start else 'stop'}"
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=10) as response:
        response.read()


class LatestFrameReader:
    """Continuously read frames and keep only the most recent one."""

    def __init__(self, stream_url: str, buffer_size: int, max_read_failures: int) -> None:
        self._stream_url = stream_url
        self._buffer_size = buffer_size
        self._max_read_failures = max_read_failures
        self._stop = threading.Event()
        self._capture: cv2.VideoCapture | None = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _open_capture(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self._stream_url, cv2.CAP_FFMPEG)
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open GoPro stream at {self._stream_url}")
        capture.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)
        return capture

    def _reset_capture(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _read_loop(self) -> None:
        read_failures = 0
        while not self._stop.is_set():
            if self._capture is None:
                try:
                    self._capture = self._open_capture()
                    read_failures = 0
                    rospy.loginfo("Opened GoPro UDP stream")
                except RuntimeError as exc:
                    rospy.logwarn_throttle(5.0, "%s", exc)
                    rospy.sleep(0.5)
                    continue

            ok, frame = self._capture.read()
            if not ok:
                read_failures += 1
                if read_failures >= self._max_read_failures:
                    rospy.logwarn_throttle(5.0, "Lost GoPro stream, reconnecting...")
                    self._reset_capture()
                    read_failures = 0
                continue

            read_failures = 0
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    def read_latest(self, timeout: float) -> tuple[bool, object | None]:
        try:
            return True, self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return False, None

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._reset_capture()


class GoProWifiStreamNode:
    def __init__(self) -> None:
        camera_name = rospy.get_param("~camera_name", "head_camera")
        self._frame_id = rospy.get_param("~camera_frame_id", camera_name)
        self._stream_url = rospy.get_param("~stream_url", _DEFAULT_STREAM_URL)
        self._publish_rate = rospy.get_param("~framerate", 30.0)
        self._buffer_size = int(rospy.get_param("~buffer_size", 1))
        self._max_read_failures = int(rospy.get_param("~max_read_failures", 30))
        self._stream_warmup_s = float(rospy.get_param("~stream_warmup_s", 1.0))
        self._gopro_base_url = rospy.get_param("~gopro_base_url", "http://10.5.5.9:8080")
        self._start_stream = rospy.get_param("~start_stream", True)
        self._stop_stream_on_shutdown = rospy.get_param("~stop_stream_on_shutdown", True)
        self._stream_mode = rospy.get_param("~stream_mode", "webcam")  # 'webcam' or 'preview'
        self._stream_port = int(rospy.get_param("~stream_port", 8554))
        self._webcam_resolution = int(rospy.get_param("~webcam_resolution", 7))
        self._webcam_fov = int(rospy.get_param("~webcam_fov", 0))
        self._webcam_protocol = int(rospy.get_param("~webcam_protocol", 1))
        ffmpeg_options = rospy.get_param("~ffmpeg_capture_options", "")
        if ffmpeg_options:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ffmpeg_options

        self._publisher = rospy.Publisher(
            f"/{camera_name}/image_raw",
            Image,
            queue_size=1,
        )
        self._bridge = CvBridge()
        self._reader: LatestFrameReader | None = None

        if self._start_stream:
            self._start_gopro_stream()

        if self._stop_stream_on_shutdown:
            rospy.on_shutdown(self._shutdown)

    def _start_gopro_stream(self) -> None:
        try:
            # Reset any previous stream before starting a fresh preview session.
            _set_gopro_stream(self._gopro_base_url, start=False, mode=self._stream_mode)
            _set_gopro_stream(self._gopro_base_url, start=True, mode=self._stream_mode,
                              port=self._stream_port,
                              resolution=self._webcam_resolution,
                              fov=self._webcam_fov,
                              protocol=self._webcam_protocol)
            rospy.loginfo("Started GoPro preview stream at %s", self._gopro_base_url)
            if self._stream_warmup_s > 0.0:
                rospy.sleep(self._stream_warmup_s)
        except (urllib.error.URLError, TimeoutError) as exc:
            rospy.logwarn(
                "Could not start GoPro preview stream via HTTP (%s). "
                "Assuming the stream is already running.",
                exc,
            )

    def _shutdown(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        if not self._stop_stream_on_shutdown:
            return
        try:
            _set_gopro_stream(self._gopro_base_url, start=False, mode=self._stream_mode)
            rospy.loginfo("Stopped GoPro preview stream")
        except (urllib.error.URLError, TimeoutError) as exc:
            rospy.logwarn("Could not stop GoPro preview stream via HTTP: %s", exc)

    def spin(self) -> None:
        self._reader = LatestFrameReader(
            self._stream_url,
            self._buffer_size,
            self._max_read_failures,
        )
        rospy.loginfo("Receiving GoPro WiFi stream from %s", self._stream_url)

        rate = rospy.Rate(self._publish_rate)
        frame_timeout = max(1.0, 2.0 / self._publish_rate)
        while not rospy.is_shutdown():
            ok, frame = self._reader.read_latest(timeout=frame_timeout)
            if not ok or frame is None:
                rospy.logwarn_throttle(5.0, "Waiting for GoPro frames...")
                continue

            message = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            message.header.stamp = rospy.Time.now()
            message.header.frame_id = self._frame_id
            self._publisher.publish(message)
            rate.sleep()


def main() -> None:
    rospy.init_node("gopro_wifi_stream")
    GoProWifiStreamNode().spin()


if __name__ == "__main__":
    main()
