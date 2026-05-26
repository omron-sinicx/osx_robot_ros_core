#!/usr/bin/env python3
"""Replay dataset episodes on the real UR5e, optionally comparing action representations.

Extends test_replay_episode.py with an ``+eval.comparison=true`` flag. When enabled,
each episode is replayed *twice* — once with the primary action type (``dataset.replay``,
e.g. ``factored_actions``) and once with ``raw_actions`` — so both replays can be
compared against the original dataset on a single plot.

Plots per episode (comparison mode):
  - EEF position tracking (primary replay vs raw replay vs dataset), with per-axis error
  - Force norm (primary vs raw vs dataset)
  - Force difference from dataset for each replay
  - Stiffness (primary vs raw vs dataset), when applicable

Usage:
    # Standard (same as test_replay_episode.py):
    python test_replay_episode_pro.py

    # With comparison:
    python test_replay_episode_pro.py +eval.comparison=true

    # Start from episode 2, compare 3 episodes:
    python test_replay_episode_pro.py +eval.comparison=true dataset.dataset.episode_idx=2 +eval.num_episodes=3
"""

import logging
import signal
import sys
import timeit
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

from comet.common.datasets.utils import tensors_to_numpy
import rospy

from rich.console import Console
from rich.logging import RichHandler

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from osx_ur5e.fdcc_env import FDCCEnv
from ur_control import transformations

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Data container for a single replay run
# ---------------------------------------------------------------------------

@dataclass
class ReplayResult:
    """Stores the recorded data from a single episode replay."""
    action_type: str
    eef_pos: np.ndarray
    force_norm: np.ndarray
    stiffness: np.ndarray | None = None
    force_violation: bool = False


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def _signal_handler(sig, frame):
    logger.info("Interrupted — shutting down.")
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_file: Path) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in root.handlers[:]:
        if not isinstance(h, logging.FileHandler):
            root.removeHandler(h)
    root.addHandler(RichHandler(console=console, rich_tracebacks=True, show_path=False))
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"))
    root.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Action extraction
# ---------------------------------------------------------------------------

def build_env_action(frame: dict, action_type: str, replay_action_keys: list) -> dict:
    """Extract replay actions from a dataset frame and convert to FDCCEnv format."""
    env_action = {}
    frame_np = tensors_to_numpy(frame)

    if action_type == "raw_actions":
        env_action["action.position"] = frame_np["action.position"]
        env_action["action.orientation"] = transformations.quaternion_from_ortho6(frame_np["action.rotation_ortho6"])
        env_action["action.stiffness_diag"] = frame_np["action.stiffness_diag"]
    elif action_type == "virtual_target_actions":
        env_action["action.virtual_target_position"] = frame_np["action.virtual_target_position"]
        env_action["action.virtual_target_rotation"] = frame_np["action.virtual_target_rotation"]
        env_action["action.estimated_stiffness"] = frame_np["action.estimated_stiffness"]
        env_action["action.ref_position"] = frame_np["observation.eef.position"]
        env_action["action.ref_rotation_ortho6"] = frame_np["observation.eef.rotation_ortho6"]
    elif action_type == "factored_actions":
        env_action["action.ref_position"] = frame_np["action.ref_position"]
        env_action["action.ref_rotation_ortho6"] = frame_np["action.ref_rotation_ortho6"]
        env_action["action.contact_direction"] = frame_np["action.contact_direction"]
        env_action["action.normal_force"] = frame_np["action.normal_force"]
        env_action["action.normal_torque"] = frame_np["action.normal_torque"]
        env_action["action.estimated_stiffness"] = frame_np["action.estimated_stiffness"]

    return env_action


def _action_keys_for(cfg: DictConfig, action_type: str) -> list[str]:
    """Return the replay action key names for a given action type from the config."""
    return list(cfg.dataset[action_type].keys())


def move_to_init_qpos(env: FDCCEnv, reason: str = "") -> None:
    """Move the robot to the safe init_qpos configuration via env.go_home()."""
    tag = f" ({reason})" if reason else ""
    logger.info(f"Moving to init_qpos{tag}...")
    try:
        env.deactivate_compliance_control()
    except Exception:
        pass
    env.go_home()
    logger.info("Reached init_qpos.")


# ---------------------------------------------------------------------------
# Single episode replay
# ---------------------------------------------------------------------------

def replay_single_episode(
    env: FDCCEnv,
    dataset: LeRobotDataset,
    episode_idx: int,
    action_type: str,
    replay_action_keys: list[str],
    fps: float,
    include_stiffness: bool,
) -> ReplayResult:
    """Replay one episode through the robot and return recorded data."""
    ep = dataset.meta.episodes[episode_idx]
    ep_start = int(ep["dataset_from_index"])
    ep_end = int(ep["dataset_to_index"])
    ep_len = ep_end - ep_start
    sleep_time = 1.0 / fps

    eef_pos = np.zeros((ep_len, 3))
    force_norm = np.zeros(ep_len)
    stiffness = np.zeros(ep_len) if include_stiffness else None

    env.reset(move_robot=True)

    frame = dataset[ep_start]
    logger.info("Moving to episode start qpos...")
    env.arm.set_joint_positions(target_time=1.0, positions=frame["observation.qpos"], wait=True)

    input(f"\n  [{action_type}] Episode {episode_idx} ({ep_len} steps) — press Enter to start...")
    env.activate_compliance_control()

    force_violation = False
    with tqdm.tqdm(total=ep_len, desc=f"Ep {episode_idx} ({action_type})") as pbar:
        for i, t in enumerate(range(ep_start, ep_end)):
            step_start = timeit.default_timer()
            frame = dataset[t]

            actual_eef = env.arm.end_effector()
            eef_pos[i] = actual_eef[:3]

            env_action = build_env_action(frame, action_type, replay_action_keys)
            timestep = env.step(env_action)

            wrench = env.arm.get_wrench()
            force_norm[i] = np.linalg.norm(wrench[:3])

            if include_stiffness:
                stiffness[i] = float(np.mean(env.last_stiffness_params))

            if timestep.last():
                logger.warning(f"Episode {episode_idx} ended early at step {i} (force limit exceeded)")
                force_violation = True
                eef_pos = eef_pos[:i + 1]
                force_norm = force_norm[:i + 1]
                if include_stiffness:
                    stiffness = stiffness[:i + 1]
                break

            elapsed = timeit.default_timer() - step_start
            remaining = sleep_time - elapsed
            if remaining < 0:
                logger.debug(f"Step slow: {1.0 / elapsed:.1f} Hz (target {fps} Hz)")
            else:
                rospy.sleep(remaining)
            pbar.update(1)

    env.deactivate_compliance_control()

    return ReplayResult(
        action_type=action_type,
        eef_pos=eef_pos,
        force_norm=force_norm,
        stiffness=stiffness,
        force_violation=force_violation,
    )


# ---------------------------------------------------------------------------
# Dataset ground-truth extraction
# ---------------------------------------------------------------------------

def extract_dataset_ground_truth(
    dataset: LeRobotDataset,
    episode_idx: int,
    include_stiffness: bool,
    stiffness_key: str | None,
) -> dict:
    """Pull EEF positions, force norms, and stiffness from the dataset for one episode."""
    ep = dataset.meta.episodes[episode_idx]
    ep_start = int(ep["dataset_from_index"])
    ep_end = int(ep["dataset_to_index"])
    ep_len = ep_end - ep_start

    eef_pos = np.zeros((ep_len, 3))
    force_norm = np.zeros(ep_len)
    stiffness = np.zeros(ep_len) if include_stiffness else None

    for i, t in enumerate(range(ep_start, ep_end)):
        frame = dataset[t]

        if "observation.eef.position" in frame:
            ds_pos = frame["observation.eef.position"]
            eef_pos[i] = (ds_pos.cpu().numpy() if isinstance(ds_pos, torch.Tensor) else np.array(ds_pos)).flatten()

        if "observation.ft" in frame:
            ds_ft = frame["observation.ft"]
            ds_ft_np = (ds_ft.cpu().numpy() if isinstance(ds_ft, torch.Tensor) else np.array(ds_ft)).flatten()
            force_norm[i] = np.linalg.norm(ds_ft_np[:3])

        if include_stiffness and stiffness_key and stiffness_key in frame:
            sv = frame[stiffness_key]
            stiffness[i] = float(np.mean(sv.cpu().numpy() if isinstance(sv, torch.Tensor) else sv))

    return dict(eef_pos=eef_pos, force_norm=force_norm, stiffness=stiffness)


# ---------------------------------------------------------------------------
# Plotting (comparison-aware)
# ---------------------------------------------------------------------------

_REPLAY_STYLES = [
    dict(color_set=["tab:blue", "tab:orange", "tab:green"], force_color="tab:red",    stiff_color="tab:green",  alpha=1.0),
    dict(color_set=["#1f77b4",  "#ff7f0e",    "#2ca02c"],   force_color="tab:purple", stiff_color="tab:purple", alpha=0.65),
]


def plot_episode_comparison(
    episode_idx: int,
    dataset_gt: dict,
    replays: list[ReplayResult],
    save_path: Path,
) -> None:
    """Plot dataset ground-truth vs one or more replay runs.

    When ``replays`` has a single entry this produces the same layout as the
    original script.  With two entries the second replay is overlaid.
    """
    ds_eef = dataset_gt["eef_pos"]
    ds_force = dataset_gt["force_norm"]
    ds_stiffness = dataset_gt["stiffness"]

    has_stiffness = ds_stiffness is not None and any(r.stiffness is not None for r in replays)
    n_rows = 4 if has_stiffness else 3
    fig = plt.figure(figsize=(14, 3.5 * n_rows))
    gs = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.45, wspace=0.35)

    ds_steps = np.arange(len(ds_force))
    axis_labels = ["X", "Y", "Z"]

    # ── Row 0: EEF position per axis ──
    ax_pos = fig.add_subplot(gs[0, :])
    dataset_colors = ["tab:cyan", "tab:red", "tab:olive"]
    for i, (lbl, cd) in enumerate(zip(axis_labels, dataset_colors)):
        ax_pos.plot(ds_steps, ds_eef[:, i], color=cd, linestyle="--", linewidth=0.8, label=f"{lbl} dataset")

    for ridx, replay in enumerate(replays):
        style = _REPLAY_STYLES[ridx % len(_REPLAY_STYLES)]
        steps = np.arange(len(replay.force_norm))
        tag = replay.action_type
        for i, (lbl, ca) in enumerate(zip(axis_labels, style["color_set"])):
            ax_pos.plot(steps, replay.eef_pos[:, i], color=ca, linewidth=0.8, alpha=style["alpha"],
                        label=f"{lbl} {tag}")

    ax_pos.set_title(f"Episode {episode_idx} — EEF position (m)")
    ax_pos.set_ylabel("Position (m)")
    ax_pos.legend(ncol=3, fontsize=6, loc="upper right")
    ax_pos.grid(True, linewidth=0.4)

    # ── Row 1 left: Position error per axis (per replay) ──
    ax_err = fig.add_subplot(gs[1, 0])
    for ridx, replay in enumerate(replays):
        style = _REPLAY_STYLES[ridx % len(_REPLAY_STYLES)]
        T = min(len(replay.eef_pos), len(ds_eef))
        pos_error = replay.eef_pos[:T] - ds_eef[:T]
        steps = np.arange(T)
        for i, (lbl, col) in enumerate(zip(axis_labels, style["color_set"])):
            ax_err.plot(steps, pos_error[:, i], color=col, linewidth=0.8, alpha=style["alpha"],
                        label=f"{lbl} {replay.action_type}")
    ax_err.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax_err.set_title("Position error (replay − dataset)")
    ax_err.set_ylabel("Error (m)")
    ax_err.legend(fontsize=5, ncol=2)
    ax_err.grid(True, linewidth=0.4)

    # ── Row 1 right: L2 position error ──
    ax_l2 = fig.add_subplot(gs[1, 1])
    l2_colors = ["tab:purple", "tab:brown"]
    for ridx, replay in enumerate(replays):
        T = min(len(replay.eef_pos), len(ds_eef))
        l2_err = np.linalg.norm(replay.eef_pos[:T] - ds_eef[:T], axis=1)
        steps = np.arange(T)
        ax_l2.plot(steps, l2_err, color=l2_colors[ridx % len(l2_colors)], linewidth=0.8,
                   label=replay.action_type)
    ax_l2.set_title("L2 position error")
    ax_l2.set_ylabel("||error|| (m)")
    ax_l2.legend(fontsize=7)
    ax_l2.grid(True, linewidth=0.4)

    # ── Row 2 left: Force norm comparison ──
    ax_fn = fig.add_subplot(gs[2, 0])
    ax_fn.plot(ds_steps, ds_force, color="tab:grey", linestyle="--", linewidth=0.8, label="dataset")
    for ridx, replay in enumerate(replays):
        style = _REPLAY_STYLES[ridx % len(_REPLAY_STYLES)]
        steps = np.arange(len(replay.force_norm))
        ax_fn.plot(steps, replay.force_norm, color=style["force_color"], linewidth=0.8,
                   alpha=style["alpha"], label=replay.action_type)
    ax_fn.set_title("Force norm (N)")
    ax_fn.set_ylabel("||F|| (N)")
    ax_fn.set_ylim(bottom=0)
    ax_fn.legend(fontsize=7)
    ax_fn.grid(True, linewidth=0.4)

    # ── Row 2 right: Force difference (replay − dataset) ──
    ax_fd = fig.add_subplot(gs[2, 1])
    fd_colors = ["tab:cyan", "tab:pink"]
    for ridx, replay in enumerate(replays):
        T = min(len(replay.force_norm), len(ds_force))
        force_diff = replay.force_norm[:T] - ds_force[:T]
        steps = np.arange(T)
        ax_fd.plot(steps, force_diff, color=fd_colors[ridx % len(fd_colors)], linewidth=0.8,
                   label=replay.action_type)
    ax_fd.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax_fd.set_title("Force difference (replay − dataset)")
    ax_fd.set_ylabel("ΔF (N)")
    ax_fd.legend(fontsize=7)
    ax_fd.grid(True, linewidth=0.4)

    # ── Row 3: Stiffness comparison (optional) ──
    if has_stiffness:
        ax_st = fig.add_subplot(gs[3, :])
        if ds_stiffness is not None:
            ax_st.plot(ds_steps, ds_stiffness, color="tab:grey", linestyle="--", linewidth=0.8, label="dataset")
        for ridx, replay in enumerate(replays):
            if replay.stiffness is not None:
                style = _REPLAY_STYLES[ridx % len(_REPLAY_STYLES)]
                steps = np.arange(len(replay.stiffness))
                ax_st.plot(steps, replay.stiffness, color=style["stiff_color"], linewidth=0.8,
                           alpha=style["alpha"], label=replay.action_type)
        ax_st.set_title("Mean stiffness")
        ax_st.set_ylabel("Stiffness")
        ax_st.set_xlabel("Timestep")
        ax_st.legend(fontsize=7)
        ax_st.grid(True, linewidth=0.4)
    else:
        fig.add_subplot(gs[n_rows - 1, :]).set_xlabel("Timestep")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Comparison plot saved to: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary logging
# ---------------------------------------------------------------------------

def _log_replay_summary(episode_idx: int, replay: ReplayResult, ds_gt: dict) -> float:
    """Log per-replay metrics and return total force error."""
    T = min(len(replay.force_norm), len(ds_gt["force_norm"]))
    force_diff = replay.force_norm[:T] - ds_gt["force_norm"][:T]
    episode_force_error = float(np.sum(np.abs(force_diff)))

    T_pos = min(len(replay.eef_pos), len(ds_gt["eef_pos"]))
    l2_pos_error = np.linalg.norm(replay.eef_pos[:T_pos] - ds_gt["eef_pos"][:T_pos], axis=1)

    logger.info(
        f"  [{replay.action_type}] Episode {episode_idx} | "
        f"force violation: {replay.force_violation} | "
        f"mean L2 pos err: {np.mean(l2_pos_error):.4f} m | "
        f"mean force err: {np.mean(np.abs(force_diff)):.2f} N | "
        f"total force err: {episode_force_error:.2f} N"
    )
    return episode_force_error


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(
    version_base=None,
    config_path="../../../../../dependencies/comet/configs",
    config_name="test_wipe_osx",
)
def main(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir / "test_replay.log")

    np.set_printoptions(linewidth=np.inf, formatter={"float": lambda x: f"{x:0.3f}"})

    comparison = bool(OmegaConf.select(cfg, "dataset.replay_comparison", default=False))

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    repo_id = cfg.dataset.dataset.repo_id
    if isinstance(repo_id, (list, ListConfig)):
        repo_id = str(repo_id[0])

    dataset_root = Path(cfg.dataset.dataset.dir) / repo_id
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    logger.info(f"Loading dataset: {repo_id} from {dataset_root}")
    dataset = LeRobotDataset(repo_id, root=dataset_root, video_backend="pyav", use_videos=False)

    fps = cfg.dataset.dataset.fps

    start_episode = int(cfg.dataset.dataset.episode_idx)
    num_episodes = int(OmegaConf.select(cfg, "eval.num_episodes", default=1))  # FIXME i dont know from where it imports this
    total_episodes = dataset.meta.total_episodes
    end_episode = min(start_episode + num_episodes, total_episodes)

    logger.info(f"Dataset has {total_episodes} episodes | "
                f"replaying episodes {start_episode} – {end_episode - 1}")

    primary_action_type = cfg.dataset.replay
    primary_keys = _action_keys_for(cfg, primary_action_type)
    primary_has_stiffness = any("stiffness" in k for k in primary_keys)
    logger.info(f"Primary action type: {primary_action_type} | keys: {primary_keys}")

    comparison_action_type = "raw_actions"
    if comparison:
        comparison_keys = _action_keys_for(cfg, comparison_action_type)
        comparison_has_stiffness = any("stiffness" in k for k in comparison_keys)
        logger.info(f"Comparison action type: {comparison_action_type} | keys: {comparison_keys}")
    else:
        comparison_keys = []
        comparison_has_stiffness = False

    include_stiffness = primary_has_stiffness or comparison_has_stiffness

    # Find the stiffness key in the primary action keys (for dataset extraction)
    ds_stiffness_key = next((k for k in primary_keys if "stiffness" in k), None)
    if ds_stiffness_key is None and comparison:
        ds_stiffness_key = next((k for k in comparison_keys if "stiffness" in k), None)

    # ------------------------------------------------------------------
    # Build FDCCEnv
    # ------------------------------------------------------------------
    rospy.init_node("test_replay_episode_pro", anonymous=False)
    logger.info("ROS node initialized")

    env = FDCCEnv(config=cfg, use_torch_for_cameras=False)
    env.reference_trajectory = []

    logger.info(f"actions_as_deltas: {env.actions_as_deltas}")
    logger.info(f"comparison mode: {comparison}")

    # ------------------------------------------------------------------
    # Episode loop
    # ------------------------------------------------------------------
    total_force_errors: dict[str, float] = {}

    move_to_init_qpos(env, reason="initial safe position")

    for episode_idx in range(start_episode, end_episode):
        ep = dataset.meta.episodes[episode_idx]
        ep_len = int(ep["dataset_to_index"]) - int(ep["dataset_from_index"])
        logger.info(f"Episode {episode_idx}: {ep_len} steps")

        # -- Collect dataset ground truth --
        ds_gt = extract_dataset_ground_truth(dataset, episode_idx, include_stiffness, stiffness_key="action.stiffness_diag")

        # -- Run primary replay --
        logger.info(f"Running primary replay: {primary_action_type}")
        primary_result = replay_single_episode(
            env, dataset, episode_idx,
            action_type=primary_action_type,
            replay_action_keys=primary_keys,
            fps=fps,
            include_stiffness=primary_has_stiffness,
        )
        replays = [primary_result]

        move_to_init_qpos(env, reason=f"after {primary_action_type} ep {episode_idx}")

        # -- Run comparison replay (if enabled) --
        if comparison:
            logger.info(f"Running comparison replay: {comparison_action_type}")
            comparison_result = replay_single_episode(
                env, dataset, episode_idx,
                action_type=comparison_action_type,
                replay_action_keys=comparison_keys,
                fps=fps,
                include_stiffness=comparison_has_stiffness,
            )
            replays.append(comparison_result)

            move_to_init_qpos(env, reason=f"after {comparison_action_type} ep {episode_idx}")

        # -- Summaries --
        for replay in replays:
            err = _log_replay_summary(episode_idx, replay, ds_gt)
            total_force_errors[replay.action_type] = total_force_errors.get(replay.action_type, 0.0) + err

        # -- Save numpy arrays --
        for replay in replays:
            tag = replay.action_type
            np.save(output_dir / f"ep{episode_idx}_{tag}_eef_pos.npy", replay.eef_pos)
            np.save(output_dir / f"ep{episode_idx}_{tag}_force_norm.npy", replay.force_norm)
        np.save(output_dir / f"ep{episode_idx}_dataset_eef_pos.npy", ds_gt["eef_pos"])
        np.save(output_dir / f"ep{episode_idx}_dataset_force_norm.npy", ds_gt["force_norm"])

        # -- Plot --
        plot_episode_comparison(
            episode_idx=episode_idx,
            dataset_gt=ds_gt,
            replays=replays,
            save_path=output_dir / f"ep{episode_idx}_comparison.png",
        )

    # ------------------------------------------------------------------
    # Return to safe position
    # ------------------------------------------------------------------
    move_to_init_qpos(env, reason="all episodes finished")

    # ------------------------------------------------------------------
    # Overall summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"Episodes replayed:    {end_episode - start_episode}")
    for atype, ferr in total_force_errors.items():
        logger.info(f"  [{atype}] total force error: {ferr:.2f} N·steps")
    logger.info(f"Results saved to:     {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    if "hydra.run.dir" not in " ".join(sys.argv):
        sys.argv.append("paths.script=/test_replay")
    main()
