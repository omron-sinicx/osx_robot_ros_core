#!/usr/bin/env python3
"""Chunk-vs-demo comparison with NO robot and NO ROS master -- pure dataset + checkpoint.

Same computation and same plots as replay_pkl_episode.py's
dataset.compare_policy_path=... dataset.compare_env_source=dataset mode, minus the
hardware: the policy is fed the recorded observation at each inference tick and its
predicted chunk is compared against the demo's actual future actions. Because nothing
moves, this runs in seconds on CPU, so it is the right tool for sweeping episodes or
A/B-ing checkpoints. Use the replay_pkl_episode.py path instead when you specifically
want LIVE tracker keypoints (compare_env_source=live), which is the one thing this
cannot tell you.

The prediction and plotting helpers are imported from replay_pkl_episode.py rather
than copied, so the offline and on-hardware plots cannot drift apart.

Usage:
    python3 compare_chunks_offline.py \
        --policy_path /root/osx-ur/outputs/20260826_060655/best/pretrained_model \
        --task_name peg_hole_dir1_03 --episodes 0,9,29,40,49

    # sweep every episode, compare two checkpoints
    python3 compare_chunks_offline.py --policy_path <A> --task_name peg_hole_dir1_03 --episodes all
"""

import argparse
import glob
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_pkl_episode import (  # noqa: E402  (also sets matplotlib's Agg backend)
    plot_chunk_vs_demo,
    predict_chunk,
    setup_chunk_compare,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy_path", required=True, type=Path)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--task_name", help='repo_id is "<task_name>_lerobot"')
    g.add_argument("--repo_id", help="exact dataset directory name")
    p.add_argument("--data_dir", type=Path, default=Path("/root/osx-ur/data"))
    p.add_argument("--episodes", default="all",
                   help='"all", a list "0,9,29", or ranges "0-9,20,30-32"')
    p.add_argument("--chunk_every", type=int, default=None,
                   help="capture cadence in steps (default: the policy's n_action_steps)")
    p.add_argument("--out_dir", type=Path, default=None,
                   help="default: <policy_path>/../chunk_compare")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="cpu keeps the GPU free for a concurrent training run")
    p.add_argument("--no_plots", action="store_true", help="numbers only, skip the PNGs")
    return p.parse_args()


def parse_episodes(spec, available):
    if spec == "all":
        return list(available)
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    missing = [e for e in out if e not in available]
    if missing:
        raise SystemExit(f"episode(s) {missing} not in the dataset (has {len(available)})")
    return out


def main():
    args = parse_args()
    if args.device != "auto":
        import os
        if args.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

    repo_id = args.repo_id or f"{args.task_name}_lerobot"
    root = args.data_dir / repo_id
    if not root.exists():
        raise SystemExit(f"Dataset not found: {root}")
    files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"No parquet files under {root}/data")
    df = pd.concat([pd.read_parquet(f) for f in files])
    available = sorted(int(e) for e in df.episode_index.unique())
    episodes = parse_episodes(args.episodes, available)

    policy, pre, post, pcfg = setup_chunk_compare(args.policy_path)
    every = args.chunk_every or int(pcfg.n_action_steps)
    out_dir = args.out_dir or (args.policy_path.parent / "chunk_compare")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"{repo_id}: {len(episodes)} episode(s), capture every {every} steps, "
                f"chunk_size={pcfg.chunk_size}, out={out_dir}")

    # err[k] = 3D position error at lookahead k, pooled over every chunk/episode.
    err = {}
    # y-branch accuracy: the diagnostic that matters most here -- a non-monotonic
    # axis lets a model score well overall while predicting the minority branch
    # backwards, which a pooled error number hides completely.
    branch = {"desc": [0, 0], "asc": [0, 0]}
    K = min(30, int(pcfg.chunk_size) - 1)

    for e in episodes:
        g = df[df.episode_index == e].sort_values("frame_index")
        S = np.stack(g["observation.state"].to_numpy()).astype(np.float32)
        E = np.stack(g["observation.environment_state"].to_numpy()).astype(np.float32)
        A = np.stack(g["action"].to_numpy()).astype(np.float32)
        chunks, at = [], []
        for t0 in range(0, len(S), max(1, every)):
            ch = predict_chunk(policy, pre, post, S[t0], E[t0], int(pcfg.chunk_size))
            chunks.append(ch)
            at.append(t0)
            for k in range(len(ch)):
                if t0 + k < len(A):
                    err.setdefault(k, []).append(
                        float(np.linalg.norm(ch[k, :3] - A[t0 + k, :3]) * 1000))
            if t0 + K < len(A):
                ddy = (A[t0 + K, 1] - A[t0, 1]) * 1000
                pdy = (ch[K, 1] - ch[0, 1]) * 1000
                if abs(ddy) >= 2.0:
                    b = branch["desc" if ddy < 0 else "asc"]
                    b[1] += 1
                    b[0] += int(np.sign(pdy) == np.sign(ddy))
        ep_err = [np.linalg.norm(c[0, :3] - A[t, :3]) * 1000 for c, t in zip(chunks, at)]
        logger.info(f"  ep {e:>3}: {len(chunks):>2} chunks, bias@0 = {np.mean(ep_err):6.2f} mm")
        if not args.no_plots:
            plot_chunk_vs_demo(chunks, at, A, None,
                               out_dir / f"chunk_vs_demo_ep{e:03d}.png",
                               repo_id, e, "dataset (offline)")
            np.savez(out_dir / f"chunk_vs_demo_ep{e:03d}.npz",
                     chunks=np.stack(chunks), chunk_at=np.array(at), demo_actions=A)

    print(f"\n=== {repo_id}  |  {args.policy_path} ===")
    print(f"{'lookahead':>10}{'mean':>10}{'p95':>10}{'n':>8}")
    for k in [0, 1, 5, 10, 20, 30, 45, int(pcfg.chunk_size) - 1]:
        if k in err:
            v = np.array(err[k])
            print(f"{('+' + str(k)):>10}{v.mean():>10.2f}{np.percentile(v, 95):>10.2f}{len(v):>8}")
    print("   (error at +0 is prediction bias; growth with lookahead is staleness)")
    print(f"\n  y-branch sign accuracy at lookahead +{K}:")
    for nm, key in [("demo y DESCENDING", "desc"), ("demo y ASCENDING ", "asc")]:
        ok, n = branch[key]
        if n:
            print(f"     {nm}: {100.0 * ok / n:5.1f}%  (n={n})")
    print("   (a big gap here means the policy has learned only the dominant branch,")
    print("    which a pooled error number cannot show)")


if __name__ == "__main__":
    main()
