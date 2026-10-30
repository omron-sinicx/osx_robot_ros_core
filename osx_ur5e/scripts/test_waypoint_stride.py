#!/usr/bin/env python3
"""Replay one dataset episode at several waypoint strides to test faster execution.

A deployed policy cannot be made to infer faster — its control rate is fixed at
``policy.fps``. But because ``actions_as_deltas`` is false, every action FDCCEnv
receives is an *absolute* pose target that it servos toward with a per-tick clip
(``fdcc_env.set_compliant_control_action``). So a policy that emitted targets
further along the path would traverse it faster at the same rate.

This script emulates that with recorded data: stride ``k`` commands only every
k-th recorded waypoint, so the episode completes in ``ep_len/k`` control ticks
and the arm covers the same path ~k times faster. What it costs you — contact
force overshoot, position tracking error — is what the figures show.

With ``dataset.replay_comparison=true`` the whole sweep runs for the primary
action type (``dataset.replay``) *and* for ``raw_actions``, so you can see
whether a factored/virtual-target representation tolerates being sped up better
than raw pose targets do.

Figures per episode:
  - ``ep{N}_{atype}_stride_overlay.png``  every stride for one action type on
    one plot, one colour each, against the dataset. Panels are on the dataset
    time base so strides are directly comparable, plus a wall-clock panel that
    shows the compression, plus a summary table (speedup, force cost, clips).
  - ``ep{N}_{atype}_stride{k}_detail.png``  each stride alone vs the dataset,
    in the full per-axis layout from test_replay_episode.py.
  - ``ep{N}_stride{k}_actions.png``  the action types against each other at a
    fixed stride (comparison mode only).

Usage:
    # default sweep (1x and 2x), primary action type only
    python test_waypoint_stride.py --config-name edge_sink_wipe_dm

    # explicit sweep
    python test_waypoint_stride.py --config-name edge_sink_wipe_dm '+eval.waypoint_strides=[1,2,3]'

    # factored AND raw actions at every stride
    python test_waypoint_stride.py --config-name edge_sink_wipe_dm \
        '+eval.waypoint_strides=[1,2,3]' dataset.replay_comparison=true

    # a later episode, three of them
    python test_waypoint_stride.py --config-name edge_sink_wipe_dm \
        dataset.dataset.episode_idx=4 +eval.num_episodes=3

Nothing moves without confirmation: replay_single_episode prompts for Enter
before each replay. Start at stride 2 — the safety limiter's velocity and
acceleration caps sit far above recorded frame deltas at 60 Hz, so it will not
throttle the stride for you; the binding constraint is the compliance spring
force against controller.safety_parameters.max_force_torque.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

import rospy
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from osx_ur5e.fdcc_env import FDCCEnv

# The replay primitive, dataset extraction, env-config projection and the
# per-replay detail plot are shared verbatim with test_replay_episode.py -
# this script only adds the stride sweep and its overlay figure. Importing
# rather than copying keeps one implementation of the loop that drives the arm.
from test_replay_episode import (
    ReplayResult,
    _action_keys_for,
    _env_config_from,
    extract_dataset_ground_truth,
    move_to_init_qpos,
    plot_episode_comparison,
    replay_single_episode,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stride colouring
# ---------------------------------------------------------------------------

_STRIDE_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red",
                  "tab:purple", "tab:brown", "tab:pink", "tab:olive"]


def _stride_color(idx: int, n: int):
    """One distinct colour per stride; falls back to viridis beyond the list."""
    if n <= len(_STRIDE_COLORS):
        return _STRIDE_COLORS[idx]
    return plt.cm.viridis(idx / max(n - 1, 1))


def _stride_label(replay: ReplayResult) -> str:
    return f"stride {replay.stride}x"


# ---------------------------------------------------------------------------
# Overlay figure: every stride, one colour each
# ---------------------------------------------------------------------------

def plot_stride_overlay(
    episode_idx: int,
    dataset_gt: dict,
    replays: list[ReplayResult],
    fps: float,
    save_path: Path,
) -> None:
    """All strides on one figure, one colour per stride, against the dataset."""
    ds_force = dataset_gt["force_norm"]
    ds_torque = dataset_gt["torque_norm"]
    ds_eef = dataset_gt["eef_pos"]
    ds_stiffness = dataset_gt["stiffness"]
    ds_steps = np.arange(len(ds_force))
    ds_wall = ds_steps / fps

    n = len(replays)
    has_stiffness = ds_stiffness is not None and any(r.stiffness is not None for r in replays)

    fig = plt.figure(figsize=(15, 20))
    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.45, wspace=0.28,
                           height_ratios=[1.0, 1.0, 1.0, 1.0, 1.1])

    # ── Row 0: force norm on the dataset time base (strides comparable) ──
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(ds_steps, ds_force, color="tab:grey", linestyle="--", linewidth=1.0, label="dataset")
    for i, r in enumerate(replays):
        ax0.plot(r.ds_steps, r.force_norm, color=_stride_color(i, n), linewidth=0.9,
                 label=_stride_label(r))
    ax0.set_title(f"Episode {episode_idx} [{replays[0].action_type}] — force norm vs dataset "
                  "timestep (same path position: strides are directly comparable)")
    ax0.set_ylabel("||F|| (N)")
    ax0.set_xlabel("Dataset timestep")
    ax0.set_ylim(bottom=0)
    ax0.legend(fontsize=8, ncol=min(n + 1, 6))
    ax0.grid(True, linewidth=0.4)

    # ── Row 1: same force, but on measured wall-clock — shows the speedup ──
    ax1 = fig.add_subplot(gs[1, :])
    ax1.plot(ds_wall, ds_force, color="tab:grey", linestyle="--", linewidth=1.0,
             label=f"dataset ({ds_wall[-1]:.1f}s)")
    for i, r in enumerate(replays):
        wall = np.linspace(0.0, r.duration_s, len(r.force_norm))
        ax1.plot(wall, r.force_norm, color=_stride_color(i, n), linewidth=0.9,
                 label=f"{_stride_label(r)} ({r.duration_s:.1f}s)")
    ax1.set_title("Force norm vs measured wall-clock (the actual time compression)")
    ax1.set_ylabel("||F|| (N)")
    ax1.set_xlabel("Elapsed time (s)")
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=8, ncol=min(n + 1, 6))
    ax1.grid(True, linewidth=0.4)

    # ── Row 2 left: force error ──
    ax_fd = fig.add_subplot(gs[2, 0])
    for i, r in enumerate(replays):
        ax_fd.plot(r.ds_steps, r.force_norm - ds_force[r.ds_indices],
                   color=_stride_color(i, n), linewidth=0.9, label=_stride_label(r))
    ax_fd.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax_fd.set_title("Force error (replay − dataset)")
    ax_fd.set_ylabel("ΔF (N)")
    ax_fd.set_xlabel("Dataset timestep")
    ax_fd.legend(fontsize=7)
    ax_fd.grid(True, linewidth=0.4)

    # ── Row 2 right: L2 position error ──
    ax_l2 = fig.add_subplot(gs[2, 1])
    for i, r in enumerate(replays):
        l2 = np.linalg.norm(r.eef_pos - ds_eef[r.ds_indices], axis=1)
        ax_l2.plot(r.ds_steps, l2, color=_stride_color(i, n), linewidth=0.9,
                   label=_stride_label(r))
    ax_l2.set_title("L2 position error (replay − dataset)")
    ax_l2.set_ylabel("||error|| (m)")
    ax_l2.set_xlabel("Dataset timestep")
    ax_l2.legend(fontsize=7)
    ax_l2.grid(True, linewidth=0.4)

    # ── Row 3 left: torque norm ──
    ax_tn = fig.add_subplot(gs[3, 0])
    ax_tn.plot(ds_steps, ds_torque, color="tab:grey", linestyle="--", linewidth=1.0, label="dataset")
    for i, r in enumerate(replays):
        ax_tn.plot(r.ds_steps, r.torque_norm, color=_stride_color(i, n), linewidth=0.9,
                   label=_stride_label(r))
    ax_tn.set_title("Torque norm (Nm)")
    ax_tn.set_ylabel("||τ|| (Nm)")
    ax_tn.set_xlabel("Dataset timestep")
    ax_tn.set_ylim(bottom=0)
    ax_tn.legend(fontsize=7)
    ax_tn.grid(True, linewidth=0.4)

    # ── Row 3 right: stiffness ──
    ax_st = fig.add_subplot(gs[3, 1])
    if has_stiffness:
        if ds_stiffness is not None:
            ax_st.plot(ds_steps, ds_stiffness, color="tab:grey", linestyle="--",
                       linewidth=1.0, label="dataset")
        for i, r in enumerate(replays):
            if r.stiffness is not None:
                ax_st.plot(r.ds_steps, r.stiffness, color=_stride_color(i, n), linewidth=0.9,
                           label=_stride_label(r))
        ax_st.set_title("Commanded stiffness")
        ax_st.set_ylabel("Stiffness")
        ax_st.legend(fontsize=7)
    else:
        ax_st.set_title("Commanded stiffness (not applicable)")
        ax_st.axis("off")
    ax_st.set_xlabel("Dataset timestep")
    ax_st.grid(True, linewidth=0.4)

    # ── Row 4: summary table — what each stride cost ──
    ax_tb = fig.add_subplot(gs[4, :])
    ax_tb.axis("off")
    base = replays[0]
    demo_arc = float(np.linalg.norm(np.diff(ds_eef, axis=0), axis=1).sum())
    lines = [
        f"Episode {episode_idx} — speedup sweep "
        f"({base.action_type}, dataset episode {len(ds_force)} steps / {ds_wall[-1]:.1f}s at {fps:g} Hz, "
        f"path {1000 * demo_arc:.0f} mm)",
        "",
        f"{'variant':>10} {'steps':>6} {'wall(s)':>8} {'speedup':>8} {'coverage':>9} {'mean|dF|':>9} "
        f"{'max|F|':>7} {'meanL2':>8} {'clips':>6} {'F-viol':>7}",
        "-" * 96,
    ]
    for r in replays:
        if len(r.force_norm) == 0:
            lines.append(f"{_stride_label(r):>10} {0:>6}   (no steps recorded)")
            continue
        d_force = r.force_norm - ds_force[r.ds_indices]
        mean_df = float(np.mean(np.abs(d_force)))
        l2 = float(np.mean(np.linalg.norm(r.eef_pos - ds_eef[r.ds_indices], axis=1)))
        speedup = (ds_wall[-1] / r.duration_s) if r.duration_s > 0 else float("nan")
        arc = float(np.linalg.norm(np.diff(r.eef_pos, axis=0), axis=1).sum()) if len(r.eef_pos) > 1 else 0.0
        coverage = 100.0 * arc / demo_arc if demo_arc > 0 else float("nan")
        lines.append(
            f"{_stride_label(r):>10} {len(r.force_norm):>6} {r.duration_s:>8.1f} {speedup:>7.2f}x "
            f"{coverage:>8.1f}% {mean_df:>9.2f} {r.force_norm.max():>7.2f} {1000 * l2:>6.1f}mm "
            f"{r.clip_count:>6} {str(r.force_violation):>7}"
        )
    lines += [
        "",
        "coverage = path the arm actually traversed / demo path. A speedup with coverage",
        "well below 100% is a truncated wipe, not a faster one - the open-loop reference",
        "reversed strokes before the arm reached them. Judge speedup and coverage together.",
        "Read mean|dF| (not a summed total): fewer samples shrink any sum for free.",
        "clips>0: the safety limiter throttled the commanded step; clips=0: the compliance",
        "controller's own tracking is the limit.",
    ]
    ax_tb.text(0.01, 0.98, "\n".join(lines), va="top", ha="left",
               family="monospace", fontsize=9)

    fig.suptitle(f"Waypoint stride sweep — episode {episode_idx} "
                 f"({replays[0].action_type})", fontsize=12)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Stride overlay saved to: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(
    version_base=None,
    config_path="../../../../../../dependencies/comet/configs",
    config_name="blank",
)
def main(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "test_waypoint_stride.log")

    np.set_printoptions(linewidth=np.inf, formatter={"float": lambda x: f"{x:0.3f}"})

    # ------------------------------------------------------------------
    # Strides
    # ------------------------------------------------------------------
    strides = OmegaConf.select(cfg, "eval.waypoint_strides", default=None)
    strides = [int(s) for s in strides] if strides else [1, 2]
    if any(s < 1 for s in strides):
        raise ValueError(f"eval.waypoint_strides must all be >= 1, got {strides}")
    if len(set(strides)) != len(strides):
        raise ValueError(f"eval.waypoint_strides must be unique, got {strides}")
    strides = sorted(strides)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    repo_id = cfg.dataset.dataset.repo_id
    if isinstance(repo_id, (list, ListConfig)):
        repo_id = str(repo_id[0])

    dataset_root = Path(cfg.dataset.dataset.dir) / repo_id
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    logger.info(f"Loading dataset: {repo_id} from {dataset_root}")
    start_episode = int(cfg.dataset.dataset.episode_idx)
    num_episodes = int(OmegaConf.select(cfg, "eval.num_episodes", default=1))
    meta = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    end_episode = min(start_episode + num_episodes, meta.total_episodes)
    dataset = LeRobotDataset(repo_id, root=dataset_root, video_backend="pyav",
                             use_videos=False,
                             episodes=list(range(start_episode, end_episode)))
    # episodes=[...] filters rows: dataset[] is positional over the loaded
    # subset while meta.episodes reports absolute rows. Subtract at access.
    row_offset = int(dataset.meta.episodes[start_episode]["dataset_from_index"])

    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset.meta.info, f, indent=2)

    fps = cfg.dataset.dataset.fps

    # Action types to sweep. The primary comes from dataset.replay; comparison
    # mode adds raw_actions, so every stride runs for both and the two
    # representations can be judged on how well they tolerate the speedup.
    comparison = bool(OmegaConf.select(cfg, "dataset.replay_comparison", default=False))
    action_types = [str(cfg.dataset.replay)]
    if comparison and "raw_actions" not in action_types:
        action_types.append("raw_actions")

    action_specs = []
    for atype in action_types:
        akeys = _action_keys_for(cfg, atype)
        action_specs.append((atype, akeys, any("stiffness" in k for k in akeys)))
        logger.info(f"Action type: {atype} | keys: {akeys}")

    include_stiffness = any(spec[2] for spec in action_specs)

    logger.info(f"Waypoint strides: {strides} | action types: {action_types} | "
                f"{len(action_specs) * len(strides)} replays per episode | "
                f"episodes {start_episode} – {end_episode - 1}")
    if max(strides) > 1:
        logger.warning(
            "Stride > 1 moves the arm faster: each target sits stride steps further along "
            "the path, so the compliance spring force scales roughly with the stride until "
            "the arm catches up. max_force_torque="
            f"{list(cfg.controller.safety_parameters.max_force_torque)}. Every replay still "
            "waits for Enter before moving."
        )

    # ------------------------------------------------------------------
    # Env
    # ------------------------------------------------------------------
    rospy.init_node("test_waypoint_stride", anonymous=False)
    env = FDCCEnv(config=_env_config_from(cfg, dataset))
    env.reference_trajectory = []
    logger.info(f"actions_as_deltas: {env.actions_as_deltas} | "
                f"control_frequency: {env.control_frequency} Hz")

    move_to_init_qpos(env, reason="initial safe position")

    ft_in_tool_frame = bool(
        dataset.meta.info.get("virtual_target_displacement_config", {})
        .get("ft_in_tool_frame", True)
    )

    # ------------------------------------------------------------------
    # Episode loop
    # ------------------------------------------------------------------
    for episode_idx in range(start_episode, end_episode):
        ds_gt = extract_dataset_ground_truth(
            dataset, episode_idx, include_stiffness,
            stiffness_key="action.stiffness_diag",
            ft_in_tool_frame=ft_in_tool_frame,
            row_offset=row_offset,
        )

        # Action-type major: each action type's stride sweep runs contiguously,
        # so the stride comparison - the experiment - is the one measured
        # back-to-back with the least scene drift between its runs.
        by_action: dict[str, list[ReplayResult]] = {}
        for atype, akeys, a_stiffness in action_specs:
            for stride in strides:
                logger.info(f"Episode {episode_idx} | {atype} | waypoint stride {stride}")
                by_action.setdefault(atype, []).append(replay_single_episode(
                    env, dataset, episode_idx,
                    action_type=atype,
                    replay_action_keys=akeys,
                    fps=fps,
                    include_stiffness=a_stiffness,
                    stride=stride,
                    row_offset=row_offset,
                ))

        all_replays = [r for rs in by_action.values() for r in rs]

        # -- Arrays --
        for r in all_replays:
            tag = f"{r.action_type}_stride{r.stride}"
            np.save(output_dir / f"ep{episode_idx}_{tag}_eef_pos.npy", r.eef_pos)
            np.save(output_dir / f"ep{episode_idx}_{tag}_force_norm.npy", r.force_norm)
            np.save(output_dir / f"ep{episode_idx}_{tag}_torque_norm.npy", r.torque_norm)
            np.save(output_dir / f"ep{episode_idx}_{tag}_ds_indices.npy", r.ds_indices)
        np.save(output_dir / f"ep{episode_idx}_dataset_eef_pos.npy", ds_gt["eef_pos"])
        np.save(output_dir / f"ep{episode_idx}_dataset_force_norm.npy", ds_gt["force_norm"])

        for atype, replays in by_action.items():
            # -- All strides for this action type, one colour each --
            plot_stride_overlay(
                episode_idx=episode_idx,
                dataset_gt=ds_gt,
                replays=replays,
                fps=fps,
                save_path=output_dir / f"ep{episode_idx}_{atype}_stride_overlay.png",
            )
            # -- Each stride alone, full per-axis layout --
            for r in replays:
                plot_episode_comparison(
                    episode_idx=episode_idx,
                    dataset_gt=ds_gt,
                    replays=[r],
                    save_path=output_dir / f"ep{episode_idx}_{atype}_stride{r.stride}_detail.png",
                )

        # -- Action types against each other at a fixed stride --
        if len(by_action) > 1:
            for stride in strides:
                same = [r for r in all_replays if r.stride == stride]
                plot_episode_comparison(
                    episode_idx=episode_idx,
                    dataset_gt=ds_gt,
                    replays=same,
                    save_path=output_dir / f"ep{episode_idx}_stride{stride}_actions.png",
                )

    move_to_init_qpos(env, reason="all episodes finished")
    logger.info("Returned to home position after replay.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"Episodes replayed: {end_episode - start_episode} | strides: {strides} | "
                f"action types: {action_types}")
    logger.info(f"Results saved to:  {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    if "hydra.run.dir" not in " ".join(sys.argv):
        sys.argv.append("paths.script=/waypoint_stride")
    main()
