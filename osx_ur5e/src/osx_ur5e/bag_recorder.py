"""Per-episode rosbag recording via a managed ``rosbag record`` subprocess.

``rosbag record`` is the C++ recorder: it writes every topic at native rate
with no Python in the hot path. This wrapper owns the subprocess lifecycle
per episode and the episode directory layout:

    episode_NNNNNN/
        episode.bag
        meta.json       # task, t_record_start/end, per-topic counts, ...

Discarded episodes are moved to ``.discarded/`` (not deleted): auto-discards
triggered by force/torque violations are exactly the ones worth inspecting.
"""

import json
import os
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import rospy
from std_msgs.msg import Empty


class RosbagRecorderError(RuntimeError):
    pass


class RosbagRecorder:
    """Manage one ``rosbag record`` subprocess per episode."""

    def __init__(self, topics: List[str], buffsize_mb: int = 4096):
        self.topics = list(topics)
        self.buffsize_mb = buffsize_mb
        self._proc: Optional[subprocess.Popen] = None
        self._episode_dir: Optional[Path] = None
        self._begin_write_seen = False
        self._t_record_start: Optional[float] = None
        # `rosbag record -p` publishes on <node_name>/begin_write once the
        # bag file is actually open - our exact "recording live" gate.
        self._node_name = "data_collection_bag_recorder"
        rospy.Subscriber(f"/{self._node_name}/begin_write", Empty,
                         self._begin_write_cb, queue_size=1)

    def _begin_write_cb(self, _msg):
        if not self._begin_write_seen:
            self._begin_write_seen = True
            self._t_record_start = rospy.get_time()

    @property
    def recording(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def t_record_start(self) -> Optional[float]:
        return self._t_record_start

    # ------------------------------------------------------------------

    def start(self, episode_dir: Path, start_timeout_s: float = 10.0) -> float:
        """Start recording into ``episode_dir/episode.bag``.

        Blocks until the recorder confirms the bag is open (via -p publish)
        and returns the ROS time at which recording became live.
        """
        if self.recording:
            raise RosbagRecorderError("Recorder already running")

        episode_dir = Path(episode_dir)
        episode_dir.mkdir(parents=True, exist_ok=True)
        self._episode_dir = episode_dir
        self._begin_write_seen = False
        self._t_record_start = None

        cmd = [
            "rosbag", "record",
            "-O", str(episode_dir / "episode.bag"),
            f"--buffsize={self.buffsize_mb}",
            "--tcpnodelay",
            "-p",
            "-q",
            f"__name:={self._node_name}",
        ] + self.topics
        # Own process group so stop() can signal rosbag and its children
        # without touching our process.
        self._proc = subprocess.Popen(cmd, preexec_fn=os.setsid)

        deadline = rospy.get_time() + start_timeout_s
        while not self._begin_write_seen and not rospy.is_shutdown():
            if self._proc.poll() is not None:
                raise RosbagRecorderError(
                    f"rosbag record exited early (code {self._proc.returncode})")
            if rospy.get_time() > deadline:
                self._kill()
                raise RosbagRecorderError(
                    f"rosbag record did not start within {start_timeout_s}s")
            rospy.sleep(0.02)

        rospy.loginfo("RosbagRecorder: recording live at %s", episode_dir)
        return self._t_record_start

    def stop(self, task: str = "", extra_meta: Optional[dict] = None,
             stop_timeout_s: float = 30.0) -> dict:
        """Stop recording, verify the bag closed cleanly, write meta.json."""
        if self._proc is None:
            raise RosbagRecorderError("Recorder not running")
        episode_dir = self._episode_dir
        t_end = rospy.get_time()

        os.killpg(os.getpgid(self._proc.pid), signal.SIGINT)
        try:
            self._proc.wait(timeout=stop_timeout_s)
        except subprocess.TimeoutExpired:
            rospy.logwarn("RosbagRecorder: SIGINT timeout, killing")
            self._kill()
        self._proc = None

        bag_path = episode_dir / "episode.bag"
        active = episode_dir / "episode.bag.active"
        if active.exists() and not bag_path.exists():
            rospy.logwarn("RosbagRecorder: bag left active, reindexing")
            subprocess.run(["rosbag", "reindex", str(active)], check=True)
            active.rename(bag_path)
        if not bag_path.exists():
            raise RosbagRecorderError(f"No bag produced in {episode_dir}")

        meta = {
            "task": task,
            "t_record_start": self._t_record_start,
            "t_record_end": t_end,
            "duration_s": (t_end - self._t_record_start) if self._t_record_start else None,
            "topics": self.topics,
            "message_counts": self._topic_counts(bag_path),
            "bag_size_bytes": bag_path.stat().st_size,
        }
        meta.update(extra_meta or {})
        with open(episode_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        return meta

    def discard(self, reason: str = "user") -> Optional[Path]:
        """Stop and move the episode into ``.discarded/`` for later inspection."""
        if self._proc is not None:
            try:
                self.stop(extra_meta={"discarded": True, "discard_reason": reason})
            except RosbagRecorderError as e:
                rospy.logwarn("RosbagRecorder: discard stop failed: %s", e)
        if self._episode_dir is None:
            return None
        discard_root = self._episode_dir.parent / ".discarded"
        discard_root.mkdir(exist_ok=True)
        target = discard_root / self._episode_dir.name
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(self._episode_dir), str(target))
        rospy.loginfo("RosbagRecorder: discarded episode -> %s", target)
        self._episode_dir = None
        return target

    # ------------------------------------------------------------------

    def _kill(self):
        if self._proc is not None and self._proc.poll() is None:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            self._proc.wait()
        self._proc = None

    @staticmethod
    def _topic_counts(bag_path: Path) -> Dict[str, int]:
        import rosbag
        with rosbag.Bag(str(bag_path)) as bag:
            info = bag.get_type_and_topic_info().topics
            return {topic: t.message_count for topic, t in info.items()}
