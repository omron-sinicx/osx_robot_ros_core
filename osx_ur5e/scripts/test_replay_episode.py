#!/usr/bin/env python3
"""Replay dataset episodes on the real UR5e and compare robot vs. dataset data.

Mirrors comet/scripts/utils/postprocess/test_replay_episode.py but uses FDCCEnv
instead of robosuite, and replaces the FTVisualizer with per-episode comparison plots.

Plots per episode:
  - EEF position tracking (actual vs. dataset), with per-axis error
  - Force norm (actual vs. dataset)
  - Force difference (actual - dataset)
  - Stiffness (actual vs. dataset), if action.stiffness_diag is replayed

Usage:
    python test_replay_episode.py

    # Start from episode 2, compare 3 episodes:
    python test_replay_episode.py dataset.episode_idx=2 +eval.num_episodes=3
"""

import logging
import signal
import sys
import timeit
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
# Action extraction  (shared with replay_episode.py)
# ---------------------------------------------------------------------------

def build_env_action(frame: dict, action_type: str, replay_action_keys: list) -> dict:
    """Extract replay actions from a dataset frame and convert to FDCCEnv format.

    Args:
        frame: Single dataset frame dict (tensors keyed by feature name).
        action_type: Type of action to replay (raw_actions, virtual_target_actions, factored_actions).
        replay_action_keys: List of action key names to replay (from cfg.dataset.replay_actions).

    Returns:
        Dict with numpy array values ready for FDCCEnv.step().
    """
    env_action = {}
    frame_np = tensors_to_numpy(frame)

    if action_type == "raw_actions":
        env_action["action.position"] = frame_np["action.position"]
        env_action["action.orientation"] = transformations.quaternion_from_ortho6(frame_np["action.rotation_ortho6"])
        env_action["action.stiffness_diag"] = frame_np["action.stiffness_diag"]
    elif action_type == "virtual_target_actions":
        env_action["action.virtual_target_position"] = frame_np["action.virtual_target_position"]
        env_action["action.virtual_target_orientation"] = transformations.quaternion_from_ortho6(frame["action.virtual_target_orientation"])
        env_action["action.stiffness_diag"] = frame_np["action.stiffness_diag"]
    elif action_type == "factored_actions":
        env_action["action.ref_position"] = frame_np["action.ref_position"]
        env_action["action.ref_rotation_ortho6"] = frame_np["action.ref_rotation_ortho6"]
        env_action["action.contact_direction"] = frame_np["action.contact_direction"]
        env_action["action.normal_force"] = frame_np["action.normal_force"]
        env_action["action.normal_torque"] = frame_np["action.normal_torque"]
        env_action["action.estimated_stiffness"] = frame_np["action.estimated_stiffness"]

    return env_action


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_episode_comparison(
    episode_idx: int,
    actual_eef_pos: np.ndarray,
    dataset_eef_pos: np.ndarray,
    actual_force_norm: np.ndarray,
    dataset_force_norm: np.ndarray,
    actual_stiffness: np.ndarray | None,
    dataset_stiffness: np.ndarray | None,
    save_path: Path,
) -> None:
    """Plot actual vs. dataset comparison for one replayed episode.

    Args:
        episode_idx: Episode index (for title).
        actual_eef_pos: (T, 3) actual EEF positions recorded during replay.
        dataset_eef_pos: (T, 3) EEF positions stored in the dataset.
        actual_force_norm: (T,) ||wrench[:3]|| measured on the robot.
        dataset_force_norm: (T,) ||observation.ft[:3]|| from the dataset.
        actual_stiffness: (T,) mean stiffness sent to the controller, or None.
        dataset_stiffness: (T,) mean stiffness stored in the dataset, or None.
        save_path: File path to save the figure to.
    """
    has_stiffness = actual_stiffness is not None and dataset_stiffness is not None
    n_rows = 4 if has_stiffness else 3
    fig = plt.figure(figsize=(12, 3.5 * n_rows))
    gs = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.45, wspace=0.35)

    steps = np.arange(len(actual_force_norm))
    axis_labels = ["X", "Y", "Z"]
    colors_actual = ["tab:blue",  "tab:orange", "tab:green"]
    colors_dataset = ["tab:cyan",  "tab:red",    "tab:olive"]

    # ---- Row 0: EEF position per axis ----
    ax_pos = fig.add_subplot(gs[0, :])
    for i, (lbl, ca, cd) in enumerate(zip(axis_labels, colors_actual, colors_dataset)):
        ax_pos.plot(steps, dataset_eef_pos[:, i], color=cd, linestyle="--", linewidth=0.8,
                    label=f"{lbl} dataset")
        ax_pos.plot(steps, actual_eef_pos[:, i],  color=ca, linewidth=0.8,
                    label=f"{lbl} actual")
    ax_pos.set_title(f"Episode {episode_idx} — EEF position (m)")
    ax_pos.set_ylabel("Position (m)")
    ax_pos.legend(ncol=3, fontsize=7)
    ax_pos.grid(True, linewidth=0.4)

    # ---- Row 1 left: Position error per axis ----
    ax_err = fig.add_subplot(gs[1, 0])
    pos_error = actual_eef_pos - dataset_eef_pos
    for i, (lbl, col) in enumerate(zip(axis_labels, colors_actual)):
        ax_err.plot(steps, pos_error[:, i], color=col, linewidth=0.8, label=lbl)
    ax_err.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax_err.set_title("Position error (actual − dataset)")
    ax_err.set_ylabel("Error (m)")
    ax_err.legend(fontsize=7)
    ax_err.grid(True, linewidth=0.4)

    # ---- Row 1 right: L2 position error ----
    ax_l2 = fig.add_subplot(gs[1, 1])
    l2_err = np.linalg.norm(pos_error, axis=1)
    ax_l2.plot(steps, l2_err, color="tab:purple", linewidth=0.8)
    ax_l2.set_title("L2 position error")
    ax_l2.set_ylabel("||error|| (m)")
    ax_l2.grid(True, linewidth=0.4)

    # ---- Row 2 left: Force norm comparison ----
    ax_fn = fig.add_subplot(gs[2, 0])
    ax_fn.plot(steps, dataset_force_norm, color="tab:grey",   linestyle="--", linewidth=0.8, label="dataset")
    ax_fn.plot(steps, actual_force_norm,  color="tab:red",    linewidth=0.8, alpha=0.8,      label="actual")
    ax_fn.set_title("Force norm (N)")
    ax_fn.set_ylabel("||F|| (N)")
    ax_fn.set_ylim(bottom=0)
    ax_fn.legend(fontsize=7)
    ax_fn.grid(True, linewidth=0.4)

    # ---- Row 2 right: Force difference ----
    ax_fd = fig.add_subplot(gs[2, 1])
    force_diff = actual_force_norm - dataset_force_norm
    ax_fd.plot(steps, force_diff, color="tab:cyan", linewidth=0.8)
    ax_fd.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax_fd.set_title("Force difference (actual − dataset)")
    ax_fd.set_ylabel("ΔF (N)")
    ax_fd.grid(True, linewidth=0.4)

    # ---- Row 3: Stiffness comparison (optional) ----
    if has_stiffness:
        ax_st = fig.add_subplot(gs[3, :])
        ax_st.plot(steps, dataset_stiffness, color="tab:grey",  linestyle="--", linewidth=0.8, label="dataset")
        ax_st.plot(steps, actual_stiffness,  color="tab:green", linewidth=0.8, alpha=0.8,       label="actual")
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
    sleep_time = 1.0 / fps

    start_episode = int(cfg.dataset.dataset.episode_idx)
    num_episodes = int(OmegaConf.select(cfg, "eval.num_episodes", default=1))
    total_episodes = dataset.meta.total_episodes
    end_episode = min(start_episode + num_episodes, total_episodes)

    logger.info(f"Dataset has {total_episodes} episodes | "
                f"replaying episodes {start_episode} – {end_episode - 1}")

    action_type = cfg.dataset.replay
    replay_action_keys = list(cfg.dataset[action_type].keys())
    include_stiffness = any("stiffness" in k for k in replay_action_keys)
    logger.info(f"Replay action keys: {replay_action_keys}")

    # ------------------------------------------------------------------
    # Build FDCCEnv from Hydra config (dataset + controller groups)
    # ------------------------------------------------------------------
    rospy.init_node("test_replay_episode", anonymous=False)
    logger.info("ROS node initialized")

    env = FDCCEnv(config=cfg, use_torch_for_cameras=False)
    env.reference_trajectory = []  # satisfy reset() assertion

    actions_as_deltas = env.actions_as_deltas
    logger.info(f"actions_as_deltas: {actions_as_deltas}")

    # ------------------------------------------------------------------
    # Episode loop
    # ------------------------------------------------------------------
    total_force_error = 0.0

    for episode_idx in range(start_episode, end_episode):
        ep = dataset.meta.episodes[episode_idx]
        ep_start = int(ep["dataset_from_index"])
        ep_end = int(ep["dataset_to_index"])
        ep_len = ep_end - ep_start

        logger.info(f"Episode {episode_idx}: frames {ep_start}–{ep_end} ({ep_len} steps)")

        # Pre-allocate collection arrays
        actual_eef_pos = np.zeros((ep_len, 3))
        dataset_eef_pos = np.zeros((ep_len, 3))
        actual_force_norm = np.zeros(ep_len)
        dataset_force_norm = np.zeros(ep_len)
        actual_stiffness = np.zeros(ep_len) if include_stiffness else None
        dataset_stiffness = np.zeros(ep_len) if include_stiffness else None

        # Reset and wait for user confirmation
        env.reset(move_robot=True)

        # ------------------------------------------------------------------
        # Move to the first step's qpos position
        # ------------------------------------------------------------------
        frame = dataset[0]
        logger.info("Moving to home position...")
        env.arm.set_joint_positions(target_time=1.0, positions=frame["observation.qpos"], wait=True)

        input(f"\n  Episode {episode_idx} ({ep_len} steps) — press Enter to start test-replay...")
        env.activate_compliance_control()

        force_violation = False
        with tqdm.tqdm(total=ep_len, desc=f"Episode {episode_idx}") as pbar:
            for i, t in enumerate(range(ep_start, ep_end)):
                step_start = timeit.default_timer()

                frame = dataset[t]

                # --- Record actual robot state BEFORE the step ---
                actual_eef = env.arm.end_effector()
                actual_eef_pos[i] = actual_eef[:3]

                # --- Record dataset state at this timestep ---
                if "observation.eef.position" in frame:
                    ds_pos = frame["observation.eef.position"]
                    dataset_eef_pos[i] = (ds_pos.cpu().numpy() if isinstance(ds_pos, torch.Tensor) else np.array(ds_pos)).flatten()

                if "observation.ft" in frame:
                    ds_ft = frame["observation.ft"]
                    ds_ft_np = (ds_ft.cpu().numpy() if isinstance(ds_ft, torch.Tensor) else np.array(ds_ft)).flatten()
                    dataset_force_norm[i] = np.linalg.norm(ds_ft_np[:3])

                if include_stiffness:
                    stiff_key = next((k for k in replay_action_keys if "stiffness" in k), None)
                    if stiff_key and stiff_key in frame:
                        sv = frame[stiff_key]
                        dataset_stiffness[i] = float(np.mean(sv.cpu().numpy() if isinstance(sv, torch.Tensor) else sv))

                # --- Build and apply action ---
                env_action = build_env_action(frame, action_type, replay_action_keys)
                timestep = env.step(env_action)

                # --- Record actual state AFTER the step ---
                wrench = env.arm.get_wrench()
                actual_force_norm[i] = np.linalg.norm(wrench[:3])

                if include_stiffness:
                    actual_stiffness[i] = float(np.mean(env.last_stiffness_params))

                if timestep.last():
                    logger.warning(f"Episode {episode_idx} ended early at step {i} (force limit exceeded)")
                    force_violation = True
                    # Trim arrays to actual length
                    actual_eef_pos = actual_eef_pos[:i+1]
                    dataset_eef_pos = dataset_eef_pos[:i+1]
                    actual_force_norm = actual_force_norm[:i+1]
                    dataset_force_norm = dataset_force_norm[:i+1]
                    if include_stiffness:
                        actual_stiffness = actual_stiffness[:i+1]
                        dataset_stiffness = dataset_stiffness[:i+1]
                    break

                elapsed = timeit.default_timer() - step_start
                remaining = sleep_time - elapsed
                if remaining < 0:
                    logger.debug(f"Step slow: {1.0/elapsed:.1f} Hz (target {fps} Hz)")
                else:
                    rospy.sleep(remaining)

                pbar.update(1)
        env.deactivate_compliance_control()

        # --- Per-episode summary ---
        force_diff = actual_force_norm - dataset_force_norm
        episode_force_error = float(np.sum(np.abs(force_diff)))
        total_force_error += episode_force_error

        l2_pos_error = np.linalg.norm(actual_eef_pos - dataset_eef_pos, axis=1)
        logger.info(
            f"Episode {episode_idx} complete | "
            f"force violation: {force_violation} | "
            f"mean L2 pos err: {np.mean(l2_pos_error):.4f} m | "
            f"mean force err: {np.mean(np.abs(force_diff)):.2f} N | "
            f"total force err: {episode_force_error:.2f} N"
        )

        # --- Save per-episode numpy arrays ---
        np.save(output_dir / f"ep{episode_idx}_actual_eef_pos.npy",     actual_eef_pos)
        np.save(output_dir / f"ep{episode_idx}_dataset_eef_pos.npy",    dataset_eef_pos)
        np.save(output_dir / f"ep{episode_idx}_actual_force_norm.npy",  actual_force_norm)
        np.save(output_dir / f"ep{episode_idx}_dataset_force_norm.npy", dataset_force_norm)

        # --- Plot comparison ---
        plot_episode_comparison(
            episode_idx=episode_idx,
            actual_eef_pos=actual_eef_pos,
            dataset_eef_pos=dataset_eef_pos,
            actual_force_norm=actual_force_norm,
            dataset_force_norm=dataset_force_norm,
            actual_stiffness=actual_stiffness,
            dataset_stiffness=dataset_stiffness,
            save_path=output_dir / f"ep{episode_idx}_comparison.png",
        )

    # ------------------------------------------------------------------
    # Overall summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"Episodes replayed:    {end_episode - start_episode}")
    logger.info(f"Total force error:    {total_force_error:.2f} N·steps")
    logger.info(f"Results saved to:     {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    if "hydra.run.dir" not in " ".join(sys.argv):
        sys.argv.append("paths.script=/test_replay")
    main()
