#!/usr/bin/env python3
"""Validation harness for the two-stage data collection pipeline.

Subcommands:
    determinism  - convert the same session twice -> datasets must match
                   (parquet bytes, decoded video pixels, reports).
    equivalence  - replay a bag through live rospy subscribers (rosbag play
                   --clock) and compare the online RosSampleFeeder+assembler
                   against the offline BagCursor at the converter's tick grid.
    soak         - inspect a recorded episode bag for message drops: per-topic
                   counts vs the rate implied by the stamps.

Usage:
    rosrun osx_ur5e validate_conversion.py determinism --session <dir> --config-name book_flipping
    rosrun osx_ur5e validate_conversion.py equivalence --episode <episode_dir> --report <conversion_report.json>
    rosrun osx_ur5e validate_conversion.py soak --episode <episode_dir>
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _decoded_video_hash(path: Path) -> str:
    """Hash of decoded pixels (encoder threading may perturb container bytes)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    h = hashlib.sha256()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h.update(frame.tobytes())
    cap.release()
    return h.hexdigest()


def cmd_determinism(args, passthrough):
    convert = [
        "rosrun", "osx_ur5e", "convert_bags_to_lerobot.py",
        f"recording.conversion.session_dir={args.session}",
        "dataset.dataset.overwrite=true",
    ] + passthrough

    roots = []
    for run in (1, 2):
        out_dir = Path(tempfile.mkdtemp(prefix=f"det_run{run}_"))
        print(f"[determinism] conversion run {run} -> {out_dir}")
        subprocess.run(convert + [f"dataset.dataset.dir={out_dir}"], check=True)
        repos = [p for p in out_dir.iterdir() if p.is_dir()]
        assert len(repos) == 1, f"expected one repo dir in {out_dir}"
        roots.append(repos[0])

    failures = []
    for a in sorted(roots[0].rglob("*.parquet")):
        b = roots[1] / a.relative_to(roots[0])
        if _hash_file(a) != _hash_file(b):
            failures.append(f"parquet differs: {a.relative_to(roots[0])}")
    for a in sorted(roots[0].rglob("*.mp4")):
        b = roots[1] / a.relative_to(roots[0])
        if _decoded_video_hash(a) != _decoded_video_hash(b):
            failures.append(f"video pixels differ: {a.relative_to(roots[0])}")
    for name in ("meta/conversion_report.json",):
        ra, rb = roots[0] / name, roots[1] / name
        if ra.exists() and json.loads(ra.read_text()) != json.loads(rb.read_text()):
            failures.append(f"report differs: {name}")

    if failures:
        print("[determinism] FAIL:\n  " + "\n  ".join(failures))
        return 1
    print("[determinism] PASS: two conversions are identical")
    return 0


# ---------------------------------------------------------------------------
# equivalence (online feeder vs offline cursor)
# ---------------------------------------------------------------------------

def cmd_equivalence(args, _passthrough):
    """Requires: roscore running, /use_sim_time=true, and this script started
    BEFORE `rosbag play --clock <episode.bag>` so subscribers are connected.
    """
    import rospy

    from osx_ur5e.sample_feeders import BagCursor, RosSampleFeeder
    import rosbag as rosbag_api
    from convert_bags_to_lerobot import index_episode_bag  # same indexing path

    episode_dir = Path(args.episode)
    session_meta = json.loads((episode_dir.parent / "session_meta.json").read_text())
    report = json.loads(Path(args.report).read_text())
    ep_report = report["episodes"][episode_dir.name]
    fps = report["fps"]
    t0, num_ticks = ep_report["t0"], ep_report["num_ticks"]

    camera_names = list(session_meta["camera_topics"])
    rospy.init_node("validate_equivalence")
    feeder = RosSampleFeeder(
        camera_names=camera_names,
        wrench_topic="/wrench/filtered",
        with_action_sources=True,
    )

    with rosbag_api.Bag(str(episode_dir / "episode.bag")) as bag:
        streams = index_episode_bag(bag, session_meta, "/wrench/filtered")
        cursor = BagCursor(streams)

        print(f"[equivalence] waiting for rosbag play (sim time)...  "
              f"start: rosbag play --clock {episode_dir}/episode.bag")
        while rospy.get_time() == 0.0 and not rospy.is_shutdown():
            rospy.sleep(0.05)

        state_keys = ["joint_states", "wrench", "target_frame", "gello_joints", "stiffness"]
        matches, mismatches, skipped = 0, 0, 0
        for k in range(num_ticks):
            t_k = t0 + k / fps
            # Wait for sim time to reach the tick (rosbag play drives the clock).
            while rospy.get_time() < t_k and not rospy.is_shutdown():
                rospy.sleep(0.001)
            online = feeder.get_latest(state_keys)
            offline = cursor.advance(t_k)
            for key in state_keys:
                on, off = online.get(key), offline.get(key)
                if on is None or off is None:
                    skipped += 1
                    continue
                if on.stamp == off.stamp and np.allclose(on.value, off.value):
                    matches += 1
                elif abs(on.stamp - off.stamp) <= 2.0 / 500.0:
                    # rosbag play timing jitter: off-by-one sample is tolerable
                    matches += 1
                else:
                    mismatches += 1

    total = matches + mismatches
    ratio = matches / max(total, 1)
    print(f"[equivalence] {matches}/{total} matched ({ratio:.4%}), {skipped} skipped")
    if ratio < 0.999:
        print("[equivalence] FAIL (< 99.9% match)")
        return 1
    print("[equivalence] PASS")
    return 0


# ---------------------------------------------------------------------------
# soak
# ---------------------------------------------------------------------------

def cmd_soak(args, _passthrough):
    import rosbag as rosbag_api

    episode_dir = Path(args.episode)
    failures = []
    with rosbag_api.Bag(str(episode_dir / "episode.bag")) as bag:
        info = bag.get_type_and_topic_info().topics
        duration = bag.get_end_time() - bag.get_start_time()
        print(f"[soak] bag duration {duration:.1f}s")
        for topic, t in sorted(info.items()):
            if t.frequency is None or t.message_count < 10:
                print(f"  {topic:60s} {t.message_count:8d} msgs")
                continue
            expected = t.frequency * duration
            ratio = t.message_count / expected
            status = "OK " if ratio >= 0.995 else "DROP"
            print(f"  {topic:60s} {t.message_count:8d} msgs @ {t.frequency:6.1f} Hz "
                  f"({ratio:6.2%} of expected) {status}")
            if ratio < 0.995:
                failures.append(topic)

    if failures:
        print(f"[soak] FAIL: drops detected on {failures}")
        return 1
    print("[soak] PASS: all topics >= 99.5% of expected rate")
    return 0


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("determinism")
    p.add_argument("--session", required=True)

    p = sub.add_parser("equivalence")
    p.add_argument("--episode", required=True)
    p.add_argument("--report", required=True)

    p = sub.add_parser("soak")
    p.add_argument("--episode", required=True)

    args, passthrough = parser.parse_known_args()
    sys.exit({"determinism": cmd_determinism,
              "equivalence": cmd_equivalence,
              "soak": cmd_soak}[args.cmd](args, passthrough))


if __name__ == "__main__":
    main()
