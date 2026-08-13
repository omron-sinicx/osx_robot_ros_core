"""Per-rollout wrist-camera snapshots at the two at-home moments: right before
the policy starts and right after it returns.

Capture-only — no segmentation, no metric, no ``comet`` usage. The automatic
on-robot marker-wipe measurement this module used to compute turned out to be
untrustworthy per-rollout, so scoring now happens post-hoc, by hand, with the
interactive annotator ``comet/scripts/utils/marker_wipe.py``.

Duck-typed for ``comet.eval.runner.RealRobotEvalRunner``'s ``scene_snapshotter``
hook (``set_output_dir`` / ``capture_before`` / ``capture_after``) the same way
``osx_ur5e.rosbag_recorder.RosbagRecorder`` is duck-typed for ``rollout_recorder``.

Fully guarded: a missing/stale camera frame degrades to "skip this snapshot"
rather than raising — and the runner wraps every hook call too, so a bug here
never kills a rollout.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SceneMarkerSnapshotter:
    """Saves wrist-camera scene snapshots at the two per-rollout home moments.

    Duck-typed for RealRobotEvalRunner's scene_snapshotter hook:
    set_output_dir / capture_before / capture_after -> dict | None. This
    implementation always returns None — it only captures PNGs; the
    marker-wipe percentage is scored afterwards, by hand, with
    comet/scripts/utils/marker_wipe.py. The hook contract stays dict | None
    for future use.
    """

    def __init__(
        self,
        feeder,
        camera: str = "wrist_camera",
    ) -> None:
        self.feeder = feeder
        self.camera = camera
        self._output_dir: Path | None = None

    def set_output_dir(self, eval_dir: str | Path) -> None:
        self._output_dir = Path(eval_dir)

    def _grab(self) -> np.ndarray | None:
        """Latest wrist-camera frame as BGR uint8, or ``None`` if unavailable."""
        key = f"images.{self.camera}"
        if not self.feeder.wait_until_fresh(max_age_s=1.0, timeout_s=2.0, keys=[key]):
            logger.warning("SceneMarkerSnapshotter: %s not fresh — skipping snapshot.", self.camera)
            return None
        img = self.feeder.get_images([self.camera]).get(self.camera)
        if img is None:
            logger.warning("SceneMarkerSnapshotter: no frame available for %s.", self.camera)
            return None
        return img[:, :, ::-1].copy()  # RGB (feeder) -> BGR (marker tooling / cv2)

    def capture_before(self, rollout_id: int) -> None:
        """Grab and save the at-home frame right before the policy starts."""
        if self._output_dir is None:
            logger.warning("SceneMarkerSnapshotter: set_output_dir() not called — skipping capture.")
            return None
        bgr = self._grab()
        if bgr is None:
            return None
        path = self._output_dir / f"rollout_{rollout_id}_marker_before.png"
        cv2.imwrite(str(path), bgr)
        logger.info("Rollout %d marker-before snapshot: %s", rollout_id, path)
        return None

    def capture_after(self, rollout_id: int) -> dict | None:
        """Grab and save the at-home frame right after the policy returns."""
        if self._output_dir is None:
            logger.warning("SceneMarkerSnapshotter: set_output_dir() not called — skipping capture.")
            return None
        after = self._grab()
        if after is None:
            return None
        path = self._output_dir / f"rollout_{rollout_id}_marker_after.png"
        cv2.imwrite(str(path), after)
        logger.info("Rollout %d marker-after snapshot: %s", rollout_id, path)
        return None
