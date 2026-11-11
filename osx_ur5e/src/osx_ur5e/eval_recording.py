"""Per-rollout recorders and plots for evaluate_policy_points.py."""

import logging
import time
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Image

import matplotlib
matplotlib.use("Agg")  # must be before any pyplot import
import matplotlib.pyplot as plt

from ur_control import transformations

logger = logging.getLogger(__name__)


def write_video(path: Path, frames, fps: float) -> None:
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


class VisualizationRecorder:
    """Buffers tracking_node's ~visualization_image (bgr8) during a rollout and
    writes an mp4 on stop_and_save(). fps is measured, not assumed."""

    def __init__(self, topic: str):
        self._recording = False
        self._frames = []
        self._start_time = None
        rospy.Subscriber(topic, Image, self._callback)

    def _callback(self, msg: Image) -> None:
        if not self._recording:
            return
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        arr = arr[:, : msg.width * 3].reshape(msg.height, msg.width, 3).copy()
        self._frames.append(arr)

    def start(self) -> None:
        self._frames = []
        self._start_time = time.perf_counter()
        self._recording = True

    def stop(self) -> None:
        """Stop capturing without saving."""
        self._recording = False

    def stop_and_save(self, path: Path) -> bool:
        """Write buffered frames to `path` as mp4. False if < 2 frames."""
        self._recording = False
        elapsed = time.perf_counter() - self._start_time if self._start_time else 0.0
        n = len(self._frames)
        if n < 2:
            logger.warning(
                f"Visualization recording: only {n} frame(s) captured -- skipping mp4 save "
                f"(is tracking_node publishing {path.name}'s source topic?)."
            )
            return False
        fps = n / elapsed if elapsed > 0 else 5.0
        path.parent.mkdir(parents=True, exist_ok=True)
        write_video(path, self._frames, fps)
        logger.info(f"Saved rollout visualization video: {path} ({n} frames @ {fps:.1f} fps)")
        return True


FTTrip = namedtuple("FTTrip", ["index", "stamp", "force"])


class WrenchRecorder:
    """Records the FT sensor at its full rate during a rollout, and latches a
    trip when |F| exceeds force_limit for limit_consecutive samples in a row
    (polled by the control loop via check_trip()).
    """

    def __init__(self, topic: str, force_limit: float = None, limit_consecutive: int = 3):
        self._topic = topic
        self._force_limit = float(force_limit) if force_limit else None
        self._limit_consecutive = max(1, int(limit_consecutive))
        self._recording = False
        self._samples = []      # (stamp_s, fx, fy, fz, tx, ty, tz)
        self._step_marks = []   # (step_index, len(self._samples) at that step)
        self._over_count = 0    # consecutive samples above _force_limit
        self._trip = None       # FTTrip, latched on the FIRST breach of a rollout
        rospy.Subscriber(topic, WrenchStamped, self._callback)

    def _callback(self, msg: WrenchStamped) -> None:
        if not self._recording:
            return
        stamp = msg.header.stamp.to_sec() or rospy.get_time()
        f, tq = msg.wrench.force, msg.wrench.torque
        self._samples.append((stamp, f.x, f.y, f.z, tq.x, tq.y, tq.z))

        if self._force_limit is None:
            return
        magnitude = float(np.sqrt(f.x * f.x + f.y * f.y + f.z * f.z))
        if magnitude <= self._force_limit:
            self._over_count = 0
            return
        self._over_count += 1
        if self._over_count >= self._limit_consecutive and self._trip is None:
            self._trip = FTTrip(index=len(self._samples) - 1, stamp=stamp, force=magnitude)

    def start(self) -> None:
        self._samples = []
        self._step_marks = []
        self._over_count = 0
        self._trip = None
        self._recording = True

    def mark_step(self, step: int) -> None:
        """Note that control step `step` begins at the current sample count."""
        self._step_marks.append((step, len(self._samples)))

    def check_trip(self):
        """The latched FTTrip for this rollout, or None."""
        return self._trip

    def stop(self) -> None:
        """Stop capturing without saving."""
        self._recording = False

    def stop_and_save(self, path: Path = None):
        """Stop recording and return the trace (stamp, t_rel, wrench (N,6),
        step_marks (M,2), trip info); also saved as npz when `path` is given.
        None if < 2 samples."""
        self._recording = False
        n = len(self._samples)
        if n < 2:
            logger.warning(
                f"FT recording: only {n} sample(s) captured on {self._topic} -- skipping save "
                f"(is the topic publishing? check `rostopic hz {self._topic}`)."
            )
            return None

        arr = np.asarray(self._samples, dtype=np.float64)
        data = {
            "stamp": arr[:, 0],
            "t_rel": arr[:, 0] - arr[0, 0],
            "wrench": arr[:, 1:].astype(np.float32),
            "step_marks": np.asarray(self._step_marks, dtype=np.int64).reshape(-1, 2),
            "topic": np.asarray(self._topic),
            "trip_index": np.asarray(self._trip.index if self._trip else -1),
            "trip_force": np.asarray(self._trip.force if self._trip else np.nan),
            "trip_t_rel": np.asarray(
                self._trip.stamp - arr[0, 0] if self._trip else np.nan),
            "force_limit": np.asarray(
                self._force_limit if self._force_limit is not None else np.nan),
        }
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **data)
            duration = data["t_rel"][-1]
            logger.info(
                f"Saved rollout FT data: {path} ({n} samples, {duration:.1f} s, "
                f"{n / duration if duration > 0 else float('nan'):.0f} Hz, "
                f"max |F|={np.linalg.norm(data['wrench'][:, :3], axis=1).max():.1f} N)"
            )
        return data


def save_ft_visualization(ft_data: dict, out_path: Path, rollout_id: int) -> bool:
    """Plot a rollout's FT trace (components + magnitudes) to PNG."""
    wrench = np.asarray(ft_data["wrench"])
    t = np.asarray(ft_data["t_rel"])
    if len(t) < 2:
        logger.warning(f"Rollout {rollout_id}: only {len(t)} FT sample(s) -- skipping FT plot.")
        return False

    force, torque = wrench[:, :3], wrench[:, 3:]
    force_mag = np.linalg.norm(force, axis=1)
    torque_mag = np.linalg.norm(torque, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, vals, labels, unit, title in (
        (axes[0, 0], force, ["Fx", "Fy", "Fz"], "N", "Force"),
        (axes[0, 1], torque, ["Tx", "Ty", "Tz"], "Nm", "Torque"),
    ):
        for i, (label, color) in enumerate(zip(labels, ["#377eb8", "#4daf4a", "#e41a1c"])):
            ax.plot(t, vals[:, i], color=color, linewidth=1.0, label=label)
        ax.set_title(f"{title} ({unit})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    for ax, mag, unit, title, color in (
        (axes[1, 0], force_mag, "N", "Force magnitude", "#ff7f00"),
        (axes[1, 1], torque_mag, "Nm", "Torque magnitude", "#984ea3"),
    ):
        ax.plot(t, mag, color=color, linewidth=1.2,
                label=f"max={mag.max():.2f} {unit}, mean={mag.mean():.2f} {unit}")
        ax.axhline(mag.max(), color=color, linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_title(f"{title} ({unit})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    trip_t = float(ft_data.get("trip_t_rel", np.nan))
    trip_note = ""
    if np.isfinite(trip_t):
        trip_force = float(ft_data["trip_force"])
        for ax in axes.flat:
            ax.axvline(trip_t, color="#e41a1c", linestyle="--", linewidth=1.4, alpha=0.9)
        trip_note = (f" -- FORCE LIMIT TRIPPED at {trip_t:.2f} s "
                     f"(|F|={trip_force:.1f} N > {float(ft_data['force_limit']):.1f} N)")

    fig.suptitle(
        f"Rollout {rollout_id}: FT sensor ({ft_data['topic']}, {len(t)} samples, "
        f"{t[-1]:.1f} s){trip_note}", fontsize=12
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        f"Saved rollout FT plot: {out_path} (max |F|={force_mag.max():.1f} N, "
        f"max |T|={torque_mag.max():.2f} Nm)"
    )
    return True


def save_bend_data(bend_log: list, baseline, out_path: Path, rollout_id: int):
    """Save a per-step BendLabs trace as npz and return it as a dict.
    bend_log entries: (ros_stamp_s, step, twist0, bend0, twist1, ...).
    None if < 2 steps."""
    if len(bend_log) < 2:
        logger.warning(
            f"Rollout {rollout_id}: only {len(bend_log)} flexible-sensor sample(s) -- "
            "skipping save (are the BendLabs sensors publishing?)."
        )
        return None

    arr = np.asarray(bend_log, dtype=np.float64)
    n_sensors = (arr.shape[1] - 2) // 2
    data = {
        "stamp": arr[:, 0],
        "t_rel": arr[:, 0] - arr[0, 0],
        "step": arr[:, 1].astype(np.int64),
        "angles": arr[:, 2:].astype(np.float32),
        "baseline": np.asarray(baseline if baseline is not None
                               else np.full(2 * n_sensors, np.nan), dtype=np.float32),
        "labels": np.asarray([f"{axis}{i}" for i in range(n_sensors)
                              for axis in ("twist", "bend")]),
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **data)
        peak = np.abs(data["angles"] - data["baseline"]).max()
        logger.info(
            f"Saved rollout flexible-sensor data: {out_path} ({len(bend_log)} steps, "
            f"{n_sensors} sensors, peak deviation {peak:.2f} deg)"
        )
    return data


def save_bend_visualization(bend_data: dict, out_path: Path, rollout_id: int) -> bool:
    """Plot BendLabs angles and their deviation from the rollout start to PNG."""
    angles = np.asarray(bend_data["angles"])
    step = np.asarray(bend_data["step"])
    if len(step) < 2:
        logger.warning(f"Rollout {rollout_id}: only {len(step)} flexible-sensor sample(s) "
                       "-- skipping plot.")
        return False

    labels = [str(l) for l in np.asarray(bend_data["labels"])]
    baseline = np.asarray(bend_data["baseline"])
    deviation = np.abs(angles - baseline)
    n_sensors = angles.shape[1] // 2
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for col, axis in enumerate(("twist", "bend")):
        idx = [i for i, l in enumerate(labels) if l.startswith(axis)]
        for k, i in enumerate(idx):
            axes[0, col].plot(step, angles[:, i], color=colors[k % 10], linewidth=1.2,
                              label=labels[i])
            axes[1, col].plot(step, deviation[:, i], color=colors[k % 10], linewidth=1.2,
                              label=f"{labels[i]} (max {deviation[:, i].max():.2f})")
        axes[0, col].set_title(f"{axis.capitalize()} angle (deg)", fontsize=11,
                               fontweight="bold")
        axes[1, col].set_title(f"|{axis} deviation from rollout start| (deg)", fontsize=11,
                               fontweight="bold")
        for row in (0, 1):
            axes[row, col].set_xlabel("Step")
            axes[row, col].legend(fontsize=8, loc="upper right")
            axes[row, col].grid(True, alpha=0.3)

    fig.suptitle(
        f"Rollout {rollout_id}: flexible sensors ({n_sensors} sensors, {len(step)} steps, "
        f"peak deviation {deviation.max():.2f} deg)", fontsize=12
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved rollout flexible-sensor plot: {out_path} "
                f"(peak deviation {deviation.max():.2f} deg)")
    return True


def save_pose_error_visualization(pose_error_log: list, out_path: Path, rollout_id: int) -> bool:
    """Plot commanded (post-clip) vs actual EE pose over a rollout to PNG."""
    if len(pose_error_log) < 2:
        logger.warning(
            f"Rollout {rollout_id}: only {len(pose_error_log)} pose-error sample(s) -- "
            "skipping pose-error plot."
        )
        return False

    t_arr = np.array([e["t"] for e in pose_error_log])
    commanded = np.array([e["commanded"] for e in pose_error_log])  # (T, 7)
    actual = np.array([e["actual"] for e in pose_error_log])        # (T, 7)

    pos_error_mm = np.linalg.norm(commanded[:, :3] - actual[:, :3], axis=1) * 1000.0
    orient_error_deg = np.array([
        np.degrees(np.linalg.norm(transformations.quaternions_orientation_error(c[3:], a[3:])))
        for c, a in zip(commanded, actual)
    ])

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axis_labels = ["X", "Y", "Z"]
    for i, ax in enumerate(axes.flat[:3]):
        cmd_mm = commanded[:, i] * 1000.0
        act_mm = actual[:, i] * 1000.0
        ax.plot(t_arr, cmd_mm, color="#377eb8", linewidth=1.6, label="commanded", zorder=3)
        ax.plot(t_arr, act_mm, color="#e41a1c", linewidth=1.2, linestyle="--", alpha=0.8,
                label="actual", zorder=4)
        ax.fill_between(t_arr, cmd_mm, act_mm, alpha=0.15, color="orange")
        ax.set_title(f"{axis_labels[i]} (mm)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Step")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    ax = axes.flat[3]
    ax.plot(t_arr, pos_error_mm, color="#4daf4a", linewidth=1.6,
            label=f"position error (max={pos_error_mm.max():.1f} mm)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Position error (mm)", color="#4daf4a")
    ax.tick_params(axis="y", labelcolor="#4daf4a")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(t_arr, orient_error_deg, color="#984ea3", linewidth=1.2, linestyle="--",
             label=f"orientation error (max={orient_error_deg.max():.1f} deg)")
    ax2.set_ylabel("Orientation error (deg)", color="#984ea3")
    ax2.tick_params(axis="y", labelcolor="#984ea3")
    ax.set_title("Combined pose error", fontsize=11, fontweight="bold")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    fig.suptitle(f"Rollout {rollout_id}: commanded vs actual pose", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        f"Saved rollout pose-error plot: {out_path} (max pos err {pos_error_mm.max():.1f} mm, "
        f"max orient err {orient_error_deg.max():.1f} deg)"
    )
    return True
