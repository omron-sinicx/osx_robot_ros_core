"""Per-rollout rosbag recording of the raw topics the policy pipeline consumes.

Records with ``rosbag record`` subprocesses so bags contain the untouched
message streams (full camera rate, hardware stamps). Bags written this way
replay through the same ``BagCursor``/``ObservationAssembler`` pair the offline
bag->LeRobot converter uses, which makes them directly usable by the offline
sanity checks (comet ``docs/offline_sanity_checks.md``, tool 3: temporal skew,
FT source/zeroing, live-camera parity).

Fully guarded: a missing ``rosbag`` binary or a failed spawn logs a warning and
degrades to a no-op — it never crashes or stalls a rollout.
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Commanded target poses (published by set_cartesian_target_pose) plus TF, so
# the action side and kinematic frames are in the bag alongside observations.
EXTRA_DEFAULT_TOPICS = [
    "/cartesian_compliance_controller/target_frame",
    "/tf",
    "/tf_static",
]


class RosbagRecorder:
    """Start/stop one ``rosbag record`` subprocess per rollout."""

    def __init__(self, topics: list[str], subscribe_grace_s: float = 0.5) -> None:
        self.topics = list(dict.fromkeys(topics))  # dedupe, keep order
        self.subscribe_grace_s = subscribe_grace_s
        self.out_dir: Path | None = None
        self._proc: subprocess.Popen | None = None
        self._bag_path: Path | None = None
        self._available = shutil.which("rosbag") is not None
        if not self._available:
            logger.warning("rosbag binary not found — rollout recording disabled.")
        atexit.register(self._kill_running)

    @staticmethod
    def default_topics(feeder, extra_topics: list[str] | None = None) -> list[str]:
        """Everything the policy consumes (feeder-resolved) + commands/TF."""
        return list(feeder.topics.values()) + EXTRA_DEFAULT_TOPICS + list(extra_topics or [])

    def set_output_dir(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)

    def start_rollout(self, rollout_id: int) -> None:
        if not self._available:
            return
        if self._proc is not None:
            logger.warning("Previous rosbag record still running — stopping it first.")
            self.end_rollout()
        assert self.out_dir is not None, "set_output_dir() must be called before recording"
        self._bag_path = self.out_dir / f"rollout_{rollout_id}.bag"
        cmd = [
            "rosbag", "record", "--lz4",
            "-O", str(self._bag_path),
            f"__name:=eval_rosbag_{rollout_id}",
            *self.topics,
        ]
        try:
            # Own process group so SIGINT reaches rosbag record and its
            # recorder children without touching this process.
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )
        except OSError as e:
            logger.warning("Failed to start rosbag record: %s — recording disabled.", e)
            self._available = False
            self._proc = None
            return
        # rosbag record needs a moment to connect its subscriptions.
        time.sleep(self.subscribe_grace_s)
        logger.info("Recording rollout rosbag: %s (%d topics)", self._bag_path, len(self.topics))

    def end_rollout(self) -> None:
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        try:
            # SIGINT lets rosbag record close the bag and write its index.
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            proc.wait(timeout=15.0)
        except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
            logger.warning("rosbag record did not exit cleanly — killing.")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
        if self._bag_path is not None:
            if self._bag_path.exists():
                size_mb = self._bag_path.stat().st_size / 1e6
                logger.info("Rollout rosbag saved: %s (%.1f MB)", self._bag_path, size_mb)
            else:
                active = self._bag_path.with_suffix(".bag.active")
                if active.exists():
                    logger.warning(
                        "Bag left unindexed: %s — recover with 'rosbag reindex'.", active
                    )
                else:
                    logger.warning("Expected bag not found: %s", self._bag_path)
        self._bag_path = None

    def discard_rollout(self) -> None:
        """Stop recording and delete the bag (rollout attempt was restarted)."""
        bag_path = self._bag_path
        self.end_rollout()
        if bag_path is None:
            return
        for p in (bag_path, bag_path.with_suffix(".bag.active")):
            if p.exists():
                try:
                    p.unlink()
                    logger.info("Discarded rollout rosbag: %s", p)
                except OSError as e:
                    logger.warning("Failed to delete discarded bag %s: %s", p, e)

    def _kill_running(self) -> None:
        if self._proc is not None:
            self.end_rollout()
