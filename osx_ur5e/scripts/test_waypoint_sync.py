#!/usr/bin/env python3
"""CONCEPT TEST: drive the reference by the arm's measured progress, not the clock.

Open-loop waypoint striding (test_waypoint_stride.py) advances the reference on
a fixed clock. When the arm cannot keep up it falls behind, and on a
back-and-forth wipe the reference reverses each stroke before the arm reaches
the stroke end — so the motion is truncated rather than sped up (measured: 63%
of the demo path traversed at stride 3).

This script tries the alternative: each tick the measured EEF is projected onto
the demo path to find the row the arm is actually at, and the row ``lookahead``
frames further along is commanded. The reference can never run more than
``lookahead`` ahead of the arm, so strokes complete and the commanded action
belongs to the spot the tool is on. Speed is not prescribed — it is whatever
the compliance controller achieves at that pull, so ``lookahead`` is the knob.

This is exploratory. It deliberately carries its own copy of the replay loop
rather than adding a hook to test_replay_episode.py, so the stable replay path
stays untouched. Everything else (env config, dataset extraction, homing,
per-replay detail figures) is imported from there.

Usage:
    python test_waypoint_sync.py --config-name edge_sink_wipe_dm \
        '+eval.sync_lookaheads=[30,40]'

    # both action types
    python test_waypoint_sync.py --config-name edge_sink_wipe_dm \
        '+eval.sync_lookaheads=[30,40,55]' dataset.replay_comparison=true

Scheduler knobs (all optional): eval.sync_window, eval.sync_max_advance,
eval.sync_tie_tol, eval.sync_move_tol, eval.sync_done_tol, eval.sync_stall_ticks,
eval.sync_max_ticks_factor.

Nothing moves without confirmation: each replay prompts for Enter first.
"""

import json
import logging
import sys
import timeit
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

import rospy
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from osx_ur5e.fdcc_env import FDCCEnv

from test_replay_episode import (
    _action_keys_for,
    _env_config_from,
    build_env_action,
    extract_dataset_ground_truth,
    move_to_init_qpos,
    plot_episode_comparison,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container (duck-types into plot_episode_comparison)
# ---------------------------------------------------------------------------

@dataclass
class SyncResult:
    action_type: str
    lookahead: int
    eef_pos: np.ndarray
    force_norm: np.ndarray
    torque_norm: np.ndarray
    stiffness: np.ndarray | None = None
    force_violation: bool = False
    # Demo row (relative to episode start) the arm was judged to be AT on each
    # tick — the projection, not the commanded row. Comparisons are therefore
    # made at the spot the tool actually occupied, which is NOT the same
    # convention test_replay_episode.py's strided ReplayResult uses (there it
    # is the commanded row). Do not put the two in one error table.
    ds_indices: np.ndarray | None = None
    duration_s: float = 0.0
    clip_count: int = 0
    end_row: int = 0
    stalled: bool = False

    @property
    def stride(self) -> int:      # plot_episode_comparison reads this
        return 1

    @property
    def label(self) -> str:
        return f"{self.action_type}-sync{self.lookahead}"

    @property
    def ds_steps(self) -> np.ndarray:
        if self.ds_indices is None or not len(self.ds_indices):
            return np.arange(len(self.force_norm))
        return self.ds_indices - self.ds_indices[0]


# ---------------------------------------------------------------------------
# Closed-loop reference scheduler
# ---------------------------------------------------------------------------

class ProgressScheduler:
    """Advance the reference by the arm's measured progress, not by the clock.

    Every tick the measured EEF position is projected onto the demo path inside
    a forward window from the current cursor; the cursor moves to that
    projection and the row ``lookahead`` frames further along is commanded.
    Four guards make this safe on a back-and-forth wipe, whose return stroke
    passes within millimetres of the outbound one:

      * DIRECTION-AWARE projection: only rows whose path tangent agrees with
        the arm's motion direction are eligible. That is what tells an
        outbound row from an overlapping return row - an index tie-break
        cannot. The direction is the measured velocity while the arm moves
        more than ``move_tol`` per tick; when it is slower - it decelerates
        onto its target at every stroke end, exactly where the guard matters -
        the hint is the direction from the arm to the row it was last told to
        go to, since that is where the compliance spring is pulling it;
      * the search window is forced to at least ``lookahead`` + ``window``
        rows: the arm can never be further along than the row it was told to
        go to, so its true location is always inside the search;
      * the cursor is monotone and may advance at most ``max_advance`` frames
        per tick, so it cannot leap across a reversal in one step;
      * residual ties within ``tie_tol`` metres resolve to the LATEST row
        (still capped by ``max_advance``): the arm reached where it was sent,
        so it gets credit for it. A demo pause - a run of identical rows - is
        therefore walked through at ``max_advance`` rows/tick instead of
        pinning the cursor at its first row. That compresses in-contact
        dwells, a known force-preservation caveat; this scheduler preserves
        path ORDER, not demo timing. Keep ``tie_tol`` well below the demo's
        per-frame spacing (a slow press creeps at ~0.2 mm/frame).

    ``next()`` returns the absolute dataset row to command, or ``None`` when the
    cursor is within ``done_tol`` frames of the end, or when the cursor has not
    advanced for ``stall_ticks`` ticks (``stalled`` is then set - the replay
    loop ends and the arm homes). ``max_ticks`` is the hard iteration cap.
    """

    def __init__(self, ds_path, ep_start: int, lookahead: int, *,
                 window: int = 30, max_advance: int = 8, tie_tol: float = 0.0002,
                 move_tol: float = 0.0002, done_tol: int = 5, stall_ticks: int = 180,
                 max_ticks_factor: float = 4.0):
        self.path = np.asarray(ds_path, dtype=float)
        if self.path.ndim != 2 or self.path.shape[1] != 3 or len(self.path) < 2:
            raise ValueError(f"ds_path must be (N>=2, 3), got {self.path.shape}")
        tan = np.diff(self.path, axis=0)
        tan = np.vstack([tan, tan[-1:]])                # last row keeps its incoming tangent
        norm = np.linalg.norm(tan, axis=1, keepdims=True)
        self.tangent = np.where(norm > 1e-12, tan / np.maximum(norm, 1e-12), 0.0)
        self._tan_norm = norm[:, 0]
        self.ep_start = int(ep_start)
        self.ep_len = len(self.path)
        self.lookahead = int(lookahead)
        # Invariant, not a tuning knob: the arm is never beyond the commanded
        # row (cursor + lookahead), so the projection must be able to see it.
        self.window = self.lookahead + max(int(window), 2)
        self.max_advance = max(int(max_advance), 1)
        self.tie_tol = float(tie_tol)
        self.move_tol = float(move_tol)
        self.done_tol = int(done_tol)
        self.stall_ticks = int(stall_ticks)
        self.max_ticks = int(np.ceil(max_ticks_factor * self.ep_len))
        self.cursor = 0
        self.ticks = 0
        self._stall = 0
        self.max_stall = 0
        self.stalled = False
        self._prev_pos = None
        self._last_target = None   # path point the arm was last commanded toward
        self._dir = None

    def next(self, eef_pos):
        self.ticks += 1
        pos = np.asarray(eef_pos, dtype=float)
        hint = None
        if self._prev_pos is not None:
            vel = pos - self._prev_pos
            speed = np.linalg.norm(vel)
            if speed > self.move_tol:
                hint = vel / speed
        if hint is None and self._last_target is not None:
            pull = self._last_target - pos
            n = np.linalg.norm(pull)
            if n > 1e-9:
                hint = pull / n
        if hint is not None:
            self._dir = hint
        self._prev_pos = pos.copy()

        lo = self.cursor
        hi = min(self.cursor + self.window, self.ep_len)
        d = np.linalg.norm(self.path[lo:hi] - pos[None, :], axis=1)
        if self._dir is not None:
            # Rows with no tangent (the demo paused there) carry no direction
            # information and stay eligible; only a genuinely opposing tangent
            # is masked out.
            dots = self.tangent[lo:hi] @ self._dir
            agree = (dots > 0.0) | (self._tan_norm[lo:hi] < 1e-9)
            if agree.any():                             # never mask everything away
                d = np.where(agree, d, np.inf)
        j = int(np.flatnonzero(d <= d.min() + self.tie_tol)[-1])   # latest near-minimum
        j = min(j, self.max_advance)
        if j == 0:
            self._stall += 1
            self.max_stall = max(self.max_stall, self._stall)
        else:
            self._stall = 0
        self.cursor += j
        if self.cursor >= self.ep_len - 1 - self.done_tol:
            return None
        if self._stall >= self.stall_ticks:
            self.stalled = True
            return None
        cmd = min(self.cursor + self.lookahead, self.ep_len - 1)
        self._last_target = self.path[cmd]
        return self.ep_start + cmd


# ---------------------------------------------------------------------------
# Closed-loop replay (own copy: the stable loop in test_replay_episode.py stays
# open-loop, so this concept cannot regress it)
# ---------------------------------------------------------------------------

def replay_sync_episode(
    env: FDCCEnv,
    dataset: LeRobotDataset,
    episode_idx: int,
    action_type: str,
    replay_action_keys: list[str],
    fps: float,
    include_stiffness: bool,
    scheduler: ProgressScheduler,
    row_offset: int = 0,
) -> SyncResult:
    """Replay one episode with the reference driven by measured progress."""
    ep = dataset.meta.episodes[episode_idx]
    ep_start = int(ep["dataset_from_index"])
    ep_end = int(ep["dataset_to_index"])
    ep_len = ep_end - ep_start
    sleep_time = 1.0 / fps
    max_steps = int(scheduler.max_ticks)

    eef_pos = np.zeros((max_steps, 3))
    force_norm = np.zeros(max_steps)
    torque_norm = np.zeros(max_steps)
    stiffness = np.zeros(max_steps) if include_stiffness else None
    rec_indices: list[int] = []

    env.reset(move_robot=False)
    env.set_controller_parameters()   # same reason as in test_replay_episode.py

    frame = dataset[ep_start - row_offset]
    logger.info("Moving to episode start qpos...")
    env.arm.set_joint_positions(target_time=1.0, positions=frame["observation.qpos"], wait=True)

    tag = f"{action_type} sync L={scheduler.lookahead}"
    input(f"\n  [{tag}] Episode {episode_idx} (closed-loop, {ep_len} demo steps, "
          f"cap {max_steps} ticks) — press Enter to start...")
    env.activate_compliance_control()

    force_violation = False
    clip_count = 0
    n_rec = 0
    replay_start = timeit.default_timer()
    with tqdm.tqdm(total=max_steps, desc=f"Ep {episode_idx} ({tag})") as pbar:
        for i in range(max_steps):
            step_start = timeit.default_timer()

            actual_eef = env.arm.end_effector()
            t = scheduler.next(actual_eef[:3])
            if t is None:
                break                          # path complete (or stalled)
            frame = dataset[t - row_offset]

            eef_pos[i] = actual_eef[:3]
            rec_indices.append(int(scheduler.cursor))

            env_action = build_env_action(frame, action_type, replay_action_keys)
            timestep = env.step(env_action)

            wrench = env.arm.get_wrench()
            force_norm[i] = np.linalg.norm(wrench[:3])
            torque_norm[i] = np.linalg.norm(wrench[3:])
            if include_stiffness:
                stiffness[i] = env.last_compliance_stiffness

            triggers = env.action_limiter.last_clip_triggers
            if triggers["translation"] or triggers["orientation"]:
                clip_count += 1
            n_rec = i + 1

            if timestep.last():
                logger.warning(f"Episode {episode_idx} ended early at step {i} (force limit exceeded)")
                force_violation = True
                break

            elapsed = timeit.default_timer() - step_start
            remaining = sleep_time - elapsed
            if remaining < 0:
                logger.debug(f"Step slow: {1.0 / elapsed:.1f} Hz (target {fps} Hz)")
            else:
                rospy.sleep(remaining)
            pbar.update(1)

    duration_s = timeit.default_timer() - replay_start   # before homing, like the stride script

    env.move_to_home(timeout=5.0)
    env.deactivate_compliance_control()

    eef_pos = eef_pos[:n_rec]
    force_norm = force_norm[:n_rec]
    torque_norm = torque_norm[:n_rec]
    if include_stiffness:
        stiffness = stiffness[:n_rec]

    logger.info(
        f"  [{tag}] {n_rec} steps in {duration_s:.1f}s "
        f"(dataset episode spans {ep_len / fps:.1f}s at {fps:g} Hz "
        f"-> {ep_len / fps / max(duration_s, 1e-9):.2f}x) | limiter clipped {clip_count} steps"
        f" | reached demo row {scheduler.cursor}/{ep_len - 1}"
        f" | max stall {scheduler.max_stall} ticks"
        + (" | STALLED" if scheduler.stalled else "")
    )

    return SyncResult(
        action_type=action_type,
        lookahead=scheduler.lookahead,
        eef_pos=eef_pos,
        force_norm=force_norm,
        torque_norm=torque_norm,
        stiffness=stiffness,
        force_violation=force_violation,
        ds_indices=np.asarray(rec_indices[:n_rec], dtype=int),
        duration_s=duration_s,
        clip_count=clip_count,
        end_row=int(scheduler.cursor),
        stalled=bool(scheduler.stalled),
    )


# ---------------------------------------------------------------------------
# Overlay figure: every lookahead, one colour each
# ---------------------------------------------------------------------------

_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red",
           "tab:purple", "tab:brown", "tab:pink", "tab:olive"]


def _color(idx: int, n: int):
    if n <= len(_COLORS):
        return _COLORS[idx]
    return plt.cm.viridis(idx / max(n - 1, 1))


def _coverage(replay, ds_eef, radii=(10, 20, 30)) -> dict:
    """Task coverage: fraction of demo path points with a replay point within R mm.

    Arc-length ratio penalises the arm for smoothing the path, which leaves
    nothing unwiped; this measures what the tool actually passed over.
    """
    if len(replay.eef_pos) == 0:
        return {r: float("nan") for r in radii}
    d = np.linalg.norm(ds_eef[:, None, :] - replay.eef_pos[None, :, :], axis=2).min(axis=1) * 1000
    return {r: 100.0 * float((d <= r).mean()) for r in radii}


def plot_sync_overlay(episode_idx, dataset_gt, replays, fps, save_path) -> None:
    ds_force = dataset_gt["force_norm"]
    ds_eef = dataset_gt["eef_pos"]
    ds_steps = np.arange(len(ds_force))
    ds_wall = ds_steps / fps
    demo_arc = float(np.linalg.norm(np.diff(ds_eef, axis=0), axis=1).sum())
    n = len(replays)

    fig = plt.figure(figsize=(15, 17))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.28,
                           height_ratios=[1.0, 1.0, 1.0, 1.3])

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(ds_steps, ds_force, color="tab:grey", linestyle="--", linewidth=1.0, label="dataset")
    for i, r in enumerate(replays):
        ax0.plot(r.ds_steps, r.force_norm, color=_color(i, n), linewidth=0.9, label=r.label)
    ax0.set_title(f"Episode {episode_idx} [{replays[0].action_type}] — force vs demo timestep "
                  "(x = row the tool was judged to be at)")
    ax0.set_ylabel("||F|| (N)"); ax0.set_xlabel("Demo timestep"); ax0.set_ylim(bottom=0)
    ax0.legend(fontsize=8, ncol=min(n + 1, 6)); ax0.grid(True, linewidth=0.4)

    ax1 = fig.add_subplot(gs[1, :])
    ax1.plot(ds_wall, ds_force, color="tab:grey", linestyle="--", linewidth=1.0,
             label=f"dataset ({ds_wall[-1]:.1f}s)")
    for i, r in enumerate(replays):
        wall = np.linspace(0.0, r.duration_s, len(r.force_norm))
        ax1.plot(wall, r.force_norm, color=_color(i, n), linewidth=0.9,
                 label=f"{r.label} ({r.duration_s:.1f}s)")
    ax1.set_title("Force vs measured wall-clock (the actual time compression)")
    ax1.set_ylabel("||F|| (N)"); ax1.set_xlabel("Elapsed time (s)"); ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=8, ncol=min(n + 1, 6)); ax1.grid(True, linewidth=0.4)

    ax2 = fig.add_subplot(gs[2, 0])
    for i, r in enumerate(replays):
        if len(r.eef_pos):
            ax2.plot(r.ds_steps, np.linalg.norm(r.eef_pos - ds_eef[r.ds_indices], axis=1) * 1000,
                     color=_color(i, n), linewidth=0.9, label=r.label)
    ax2.set_title("Projection residual (distance to the row the arm was matched to)")
    ax2.set_ylabel("mm"); ax2.set_xlabel("Demo timestep")
    ax2.legend(fontsize=7); ax2.grid(True, linewidth=0.4)

    ax3 = fig.add_subplot(gs[2, 1])
    for i, r in enumerate(replays):
        if len(r.ds_indices):
            ax3.plot(np.arange(len(r.ds_indices)), r.ds_indices, color=_color(i, n),
                     linewidth=0.9, label=r.label)
    ax3.plot([0, len(ds_steps)], [0, len(ds_steps)], color="black", linewidth=0.6,
             linestyle=":", label="1 row / tick")
    ax3.set_title("Cursor progress (slope = rows advanced per tick)")
    ax3.set_ylabel("Demo row"); ax3.set_xlabel("Tick")
    ax3.legend(fontsize=7); ax3.grid(True, linewidth=0.4)

    ax_tb = fig.add_subplot(gs[3, :])
    ax_tb.axis("off")
    lines = [
        f"Episode {episode_idx} — closed-loop lookahead sweep ({replays[0].action_type}, "
        f"demo {len(ds_force)} steps / {ds_wall[-1]:.1f}s, path {1000 * demo_arc:.0f} mm)",
        "",
        f"{'variant':>10} {'ticks':>6} {'wall(s)':>8} {'speedup':>8} {'arc%':>6} "
        f"{'cov@10':>7} {'cov@20':>7} {'cov@30':>7} {'meanF':>6} {'maxF':>6} {'clips':>6} "
        f"{'endrow':>7} {'stall':>6}",
        "-" * 118,
    ]
    for r in replays:
        if len(r.eef_pos) == 0:
            lines.append(f"{r.label:>10} {0:>6}   (no steps recorded)")
            continue
        arc = float(np.linalg.norm(np.diff(r.eef_pos, axis=0), axis=1).sum())
        cov = _coverage(r, ds_eef)
        speedup = ds_wall[-1] / r.duration_s if r.duration_s > 0 else float("nan")
        lines.append(
            f"{r.label:>10} {len(r.force_norm):>6} {r.duration_s:>8.1f} {speedup:>7.2f}x "
            f"{100 * arc / demo_arc:>5.1f}% {cov[10]:>6.1f}% {cov[20]:>6.1f}% {cov[30]:>6.1f}% "
            f"{r.force_norm.mean():>6.2f} {r.force_norm.max():>6.2f} {r.clip_count:>6} "
            f"{r.end_row:>7} {str(r.stalled):>6}"
        )
    lines += [
        "",
        "cov@R = % of demo path points with a replay point within R mm. This is the",
        "task-relevant coverage: arc% alone penalises the arm for smoothing the path,",
        "which leaves nothing unwiped. Judge speedup and cov@R together.",
        "endrow = demo row the cursor reached (of the episode length); stall=True means",
        "the scheduler gave up because the arm stopped making progress.",
    ]
    ax_tb.text(0.01, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8)

    fig.suptitle(f"Closed-loop sync sweep — episode {episode_idx} "
                 f"({replays[0].action_type})", fontsize=12)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Sync overlay saved to: {save_path}")
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
    setup_logging(output_dir / "test_waypoint_sync.log")
    np.set_printoptions(linewidth=np.inf, formatter={"float": lambda x: f"{x:0.3f}"})

    lookaheads = OmegaConf.select(cfg, "eval.sync_lookaheads", default=None)
    lookaheads = sorted({int(x) for x in lookaheads}) if lookaheads else [30, 40]
    if any(x < 1 for x in lookaheads):
        raise ValueError(f"eval.sync_lookaheads must all be >= 1, got {lookaheads}")
    sync_kw = dict(
        window=int(OmegaConf.select(cfg, "eval.sync_window", default=30)),
        max_advance=int(OmegaConf.select(cfg, "eval.sync_max_advance", default=8)),
        tie_tol=float(OmegaConf.select(cfg, "eval.sync_tie_tol", default=0.0002)),
        move_tol=float(OmegaConf.select(cfg, "eval.sync_move_tol", default=0.0002)),
        done_tol=int(OmegaConf.select(cfg, "eval.sync_done_tol", default=5)),
        stall_ticks=int(OmegaConf.select(cfg, "eval.sync_stall_ticks", default=180)),
        max_ticks_factor=float(OmegaConf.select(cfg, "eval.sync_max_ticks_factor", default=4.0)),
    )

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

    logger.info(f"Sync lookaheads: {lookaheads} | action types: {action_types} | "
                f"{len(action_specs) * len(lookaheads)} replays per episode | "
                f"episodes {start_episode} – {end_episode - 1}")
    logger.info(f"Scheduler params: {sync_kw}")
    logger.warning(
        "Closed-loop sync moves the arm faster than the demo: the reference sits up to "
        "`lookahead` frames ahead, so the compliance spring pulls harder. "
        f"max_force_torque={list(cfg.controller.safety_parameters.max_force_torque)}. "
        "Every replay waits for Enter before moving."
    )

    rospy.init_node("test_waypoint_sync", anonymous=False)
    env = FDCCEnv(config=_env_config_from(cfg, dataset))
    env.reference_trajectory = []
    logger.info(f"actions_as_deltas: {env.actions_as_deltas} | "
                f"control_frequency: {env.control_frequency} Hz")

    move_to_init_qpos(env, reason="initial safe position")

    ft_in_tool_frame = bool(
        dataset.meta.info.get("virtual_target_displacement_config", {})
        .get("ft_in_tool_frame", True)
    )

    for episode_idx in range(start_episode, end_episode):
        ds_gt = extract_dataset_ground_truth(
            dataset, episode_idx, include_stiffness,
            stiffness_key="action.stiffness_diag",
            ft_in_tool_frame=ft_in_tool_frame,
            row_offset=row_offset,
        )
        ep_start = int(dataset.meta.episodes[episode_idx]["dataset_from_index"])

        by_action: dict[str, list[SyncResult]] = {}
        for atype, akeys, a_stiffness in action_specs:
            for L in lookaheads:
                logger.info(f"Episode {episode_idx} | {atype} | lookahead {L}")
                scheduler = ProgressScheduler(ds_gt["eef_pos"], ep_start, L, **sync_kw)
                by_action.setdefault(atype, []).append(replay_sync_episode(
                    env, dataset, episode_idx,
                    action_type=atype,
                    replay_action_keys=akeys,
                    fps=fps,
                    include_stiffness=a_stiffness,
                    scheduler=scheduler,
                    row_offset=row_offset,
                ))

        all_replays = [r for rs in by_action.values() for r in rs]
        for r in all_replays:
            tag = f"{r.action_type}_sync{r.lookahead}"
            np.save(output_dir / f"ep{episode_idx}_{tag}_eef_pos.npy", r.eef_pos)
            np.save(output_dir / f"ep{episode_idx}_{tag}_force_norm.npy", r.force_norm)
            np.save(output_dir / f"ep{episode_idx}_{tag}_torque_norm.npy", r.torque_norm)
            np.save(output_dir / f"ep{episode_idx}_{tag}_ds_indices.npy", r.ds_indices)
        np.save(output_dir / f"ep{episode_idx}_dataset_eef_pos.npy", ds_gt["eef_pos"])
        np.save(output_dir / f"ep{episode_idx}_dataset_force_norm.npy", ds_gt["force_norm"])

        for atype, replays in by_action.items():
            plot_sync_overlay(episode_idx, ds_gt, replays, fps,
                              output_dir / f"ep{episode_idx}_{atype}_sync_overlay.png")
            for r in replays:
                plot_episode_comparison(
                    episode_idx=episode_idx,
                    dataset_gt=ds_gt,
                    replays=[r],
                    save_path=output_dir / f"ep{episode_idx}_{atype}_sync{r.lookahead}_detail.png",
                )

    move_to_init_qpos(env, reason="all episodes finished")
    logger.info("Returned to home position after replay.")
    logger.info("=" * 60)
    logger.info(f"Episodes replayed: {end_episode - start_episode} | lookaheads: {lookaheads} | "
                f"action types: {action_types}")
    logger.info(f"Results saved to:  {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    if "hydra.run.dir" not in " ".join(sys.argv):
        sys.argv.append("paths.script=/waypoint_sync")
    main()
