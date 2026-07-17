#!/usr/bin/env python3
"""Convert a Stage-1 recording session (rosbags) into a LeRobotDataset (Stage 2).

Builds a uniform tick grid at any target fps and samples every source with
causal zero-order hold (latest sample <= tick): slower sources duplicate,
faster sources are subsampled. FK-derived features are recomputed from
/joint_states through the session's own /robot_description snapshot, so the
offline observations are bit-exact with the online assembler. Delta actions
are recomputed per tick as target(t) (-) measured eef(t) - execution-matching
semantics, valid for any target fps.

Runs in-container (needs the python rosbag API + lerobot); no roscore needed.

Usage:
    rosrun osx_ur5e convert_bags_to_lerobot.py \
        recording.conversion.session_dir=/home/malek/osx-ros-comet/dependencies/comet/raw_data/wiping_the_edge_of_the_sink_20260711_043437 \
        dataset.dataset.fps=25 \
        dataset.dataset.repo_id=[test_25hz]

Re-run with a different fps / repo_id at any time - the bags are the ground
truth, the LeRobotDataset is a single-rate view of them.
"""

import json
import logging
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import hydra
import comet  # noqa: F401  # registers the ${comet_root:} resolver used by the configs
import numpy as np
import yaml
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from osx_ur5e.dataset_features import build_features
from osx_ur5e.observation_assembler import ObservationAssembler
from osx_ur5e.sample_feeders import (
    JOINT_ORDER,
    BagCursor,
    decode_image_msg,
    joint_state_to_value,
    pose_to_value,
    stamp_of,
    wrench_to_value,
)

console = Console()
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bag indexing
# ---------------------------------------------------------------------------

def index_episode_bag(bag, session_meta: dict, wrench_topic: str):
    """One sequential pass -> {source_key: [(stamp, value_or_thunk), ...]}.

    States are decoded eagerly (tiny); images stay in the bag and are fetched
    + decoded lazily per tick via the bag index (constant memory).

    Timestamp rule: header stamp when nonzero, else bag receive time (covers
    the header-less dynamixel stream and any legacy zero-stamp messages).
    """
    camera_by_topic = {
        topics[0]: cam for cam, topics in session_meta["camera_topics"].items()
    }

    decoders = {
        "/joint_states": ("joint_states", joint_state_to_value),
        wrench_topic: ("wrench", wrench_to_value),
        "/cartesian_compliance_controller/target_frame": ("target_frame", pose_to_value),
        "/data_collection/stiffness_command":
            ("stiffness", lambda m: np.asarray(m.data, dtype=np.float64)),
        "/data_collection/gello_joints":
            ("gello_joints", lambda m: np.asarray(m.position, dtype=np.float64)),
    }

    streams = defaultdict(list)
    topics = list(decoders) + list(camera_by_topic)
    for topic, msg, recv_t in bag.read_messages(topics=topics):
        stamp = stamp_of(msg, fallback=recv_t.to_sec())
        if topic in camera_by_topic:
            cam = camera_by_topic[topic]
            streams[f"images.{cam}"].append(
                (stamp, _make_image_thunk(bag, topic, recv_t)))
            continue
        key, to_value = decoders[topic]
        if key == "joint_states" and not set(JOINT_ORDER).issubset(msg.name):
            continue  # other JointState publishers (pan/tilt etc.) share /joint_states
        streams[key].append((stamp, to_value(msg)))

    return dict(streams)


def _make_image_thunk(bag, topic: str, recv_t):
    """Lazy single-message fetch through the bag index (keyed by receive time)."""
    def thunk():
        for _, msg, _ in bag.read_messages(topics=[topic], start_time=recv_t, end_time=recv_t):
            return decode_image_msg(msg)
        raise RuntimeError(f"Bag message vanished: {topic} @ {recv_t.to_sec()}")
    return thunk


# ---------------------------------------------------------------------------
# Stream statistics
# ---------------------------------------------------------------------------

def stream_stats(streams: dict) -> dict:
    """Per-source native rate and gap detection from the indexed stamps."""
    stats = {}
    for key, samples in streams.items():
        stamps = np.array([s for s, _ in samples])
        if len(stamps) < 3:
            stats[key] = {"count": len(stamps), "median_period_s": None, "gaps": 0}
            continue
        periods = np.diff(np.sort(stamps))
        median = float(np.median(periods))
        stats[key] = {
            "count": int(len(stamps)),
            "median_period_s": median,
            "native_hz": (1.0 / median) if median > 0 else None,
            "gaps": int(np.count_nonzero(periods > 2.0 * median)),
            "max_period_s": float(periods.max()),
        }
    return stats


# ---------------------------------------------------------------------------
# Episode conversion
# ---------------------------------------------------------------------------

def convert_episode(bag, episode_meta: dict, session_meta: dict, assembler,
                    dataset: LeRobotDataset, fps: float, task: str,
                    conversion_cfg) -> dict:
    """Tick-grid + causal-ZOH assembly of one episode into the dataset."""
    streams = index_episode_bag(bag, session_meta, conversion_cfg.wrench_topic)
    stats = stream_stats(streams)

    required = assembler.required_keys(with_action=True, with_images=True)
    missing = [k for k in required if not streams.get(k)]
    if missing:
        raise RuntimeError(f"Episode has no samples for required sources: {missing}")

    cursor = BagCursor(streams)

    # Tick grid: start once every required source has a causal sample and
    # recording is confirmed live; end at the earliest source drop-out.
    t0 = max(max(cursor.first_stamp(k) for k in required),
             episode_meta.get("t_record_start") or -np.inf)
    t_end = min(cursor.last_stamp(k) for k in required)
    if t_end <= t0:
        raise RuntimeError(f"Empty tick grid (t0={t0:.3f} >= t_end={t_end:.3f})")
    num_ticks = int(math.floor((t_end - t0) * fps)) + 1

    # Staleness budgets: 3x measured native period unless overridden.
    budgets = {}
    for key in required:
        override = conversion_cfg.staleness_budget_s.get(key)
        median = stats[key]["median_period_s"] or (1.0 / fps)
        budgets[key] = float(override) if override is not None else 3.0 * median

    skews = defaultdict(list)
    stale_counts = defaultdict(int)

    for k in range(num_ticks):
        t_k = t0 + k / fps
        samples = cursor.advance(t_k)

        for key in required:
            skew = t_k - samples[key].stamp
            skews[key].append(skew)
            if skew > budgets[key]:
                stale_counts[key] += 1

        obs = assembler.assemble_observation(samples, tick_time=t_k, episode_start=t0)
        eef_pose = assembler.eef_pose_from_samples(samples)
        action = assembler.assemble_action(samples, eef_pose=eef_pose)

        # No explicit "timestamp": LeRobot labels frame_index/fps (exactly
        # uniform -> tolerance check passes by construction). Real times live
        # in observation.frame_time / observation.image_time.*.
        dataset.add_frame({**obs, **action, "task": task})

    dataset.save_episode()

    report = {
        "num_ticks": num_ticks,
        "t0": t0,
        "t_end": t_end,
        "trimmed_head_s": t0 - (episode_meta.get("t_record_start") or t0),
        "source_stats": stats,
        "staleness_budget_s": budgets,
        "stale_ticks": dict(stale_counts),
        "skew_s": {
            key: {
                "p50": float(np.percentile(v, 50)),
                "p95": float(np.percentile(v, 95)),
                "max": float(np.max(v)),
            }
            for key, v in skews.items()
        },
    }
    for key, count in stale_counts.items():
        if count:
            log.warning("  %s: %d/%d ticks staler than %.0f ms",
                        key, count, num_ticks, budgets[key] * 1e3)
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="/root/osx-ur/dependencies/comet/configs",
            config_name="blank",
            version_base=None)
def main(cfg: DictConfig) -> None:
    import rosbag

    from ur_pykdl import ur_kinematics

    log.setLevel(logging.INFO)
    log.propagate = False
    if not any(isinstance(h, RichHandler) for h in log.handlers):
        log.addHandler(RichHandler(console=console, show_time=True, show_path=False, markup=True))

    ds_cfg = cfg.dataset
    conversion_cfg = cfg.recording.conversion
    fps = ds_cfg.dataset.fps

    if not conversion_cfg.session_dir:
        log.error("Set recording.conversion.session_dir=/path/to/session")
        sys.exit(1)
    session_dir = Path(conversion_cfg.session_dir)
    session_meta = json.loads((session_dir / "session_meta.json").read_text())

    episode_dirs = sorted(
        p for p in session_dir.glob("episode_*")
        if p.is_dir() and (p / "episode.bag").exists()
    )
    if not episode_dirs:
        log.error("No episodes found in %s", session_dir)
        sys.exit(1)

    # Rate sanity: above the slowest camera the video just duplicates frames.
    if fps > 500:
        log.error("fps=%s exceeds the 500 Hz state rate", fps)
        sys.exit(1)

    # Offline kinematics from the session's calibrated URDF snapshot:
    # bit-exact FK parity with the online assembler.
    urdf_string = session_meta.get("robot_description")
    if not urdf_string:
        log.error("session_meta.json has no robot_description snapshot")
        sys.exit(1)
    camera_names = list(session_meta["camera_topics"])
    assembler = ObservationAssembler(ur_kinematics(urdf_string=urdf_string), camera_names)

    # -- dataset -------------------------------------------------------------
    features = build_features(ds_cfg)
    repo_id = ds_cfg.dataset.repo_id[0]
    dataset_dir = Path(ds_cfg.dataset.dir) / repo_id
    num_camera_threads = ds_cfg.image_writer.threads_per_camera * len(camera_names)

    if dataset_dir.exists():
        if not ds_cfg.dataset.overwrite:
            confirm = input(f"Dataset {dataset_dir} exists. Overwrite? (y/n): ")
            if confirm.strip().lower() != "y":
                log.info("Exiting...")
                sys.exit(1)
        shutil.rmtree(dataset_dir)

    log.info("Creating dataset %s at %d fps from %d episodes",
             repo_id, fps, len(episode_dirs))
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=dataset_dir,
        robot_type=ds_cfg.dataset.robot_type,
        use_videos=True,
        image_writer_processes=ds_cfg.image_writer.num_processes,
        image_writer_threads=num_camera_threads,
    )

    task = session_meta.get("task", ds_cfg.dataset.task)
    episode_reports = {}
    for episode_dir in track(episode_dirs, description="Converting", console=console):
        episode_meta = json.loads((episode_dir / "meta.json").read_text()) \
            if (episode_dir / "meta.json").exists() else {}
        log.info("Episode %s", episode_dir.name)
        with rosbag.Bag(str(episode_dir / "episode.bag")) as bag:
            episode_reports[episode_dir.name] = convert_episode(
                bag, episode_meta, session_meta, assembler, dataset,
                fps=fps, task=task, conversion_cfg=conversion_cfg,
            )

    log.info("Finalizing dataset...")
    dataset.finalize()

    # -- sidecars -------------------------------------------------------------
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "hydra_config.yaml", "w") as f:
        yaml.safe_dump(OmegaConf.to_container(cfg, resolve=True), f)
    if conversion_cfg.report:
        with open(meta_dir / "conversion_report.json", "w") as f:
            json.dump({
                "session_dir": str(session_dir),
                "fps": fps,
                "episodes": episode_reports,
            }, f, indent=2)

    total_ticks = sum(r["num_ticks"] for r in episode_reports.values())
    log.info("Done: %d episodes, %d frames at %d fps -> %s",
             len(episode_reports), total_ticks, fps, dataset_dir)
    log.info("Reminder: training must set policy.fps=%d to match this dataset.", fps)


if __name__ == "__main__":
    main()
