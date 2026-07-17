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

import json
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
    torque_norm: np.ndarray
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
        env_action["action.torque_direction"] = frame_np["action.torque_direction"]
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
    torque_norm = np.zeros(ep_len)
    stiffness = np.zeros(ep_len) if include_stiffness else None

    env.reset(move_robot=False)

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
            torque_norm[i] = np.linalg.norm(wrench[3:])

            if include_stiffness:
                stiffness[i] = env.last_compliance_stiffness

            if timestep.last():
                logger.warning(f"Episode {episode_idx} ended early at step {i} (force limit exceeded)")
                force_violation = True
                eef_pos = eef_pos[:i + 1]
                force_norm = force_norm[:i + 1]
                torque_norm = torque_norm[:i + 1]
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
        torque_norm=torque_norm,
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
    ft_in_tool_frame: bool = True,
) -> dict:
    """Pull EEF positions, force norms, stiffness and the quantities needed for
    the sensor-vs-implied-spring consistency check from the dataset for one episode."""
    ep = dataset.meta.episodes[episode_idx]
    ep_start = int(ep["dataset_from_index"])
    ep_end = int(ep["dataset_to_index"])
    ep_len = ep_end - ep_start

    eef_pos = np.zeros((ep_len, 3))
    action_pos = np.zeros((ep_len, 3))
    obs_rot6 = np.zeros((ep_len, 6))
    ft_tool = np.zeros((ep_len, 6))
    k_demo = np.zeros((ep_len, 3))   # translational stiffness diag per step
    force_norm = np.zeros(ep_len)
    torque_norm = np.zeros(ep_len)
    stiffness = np.zeros(ep_len) if include_stiffness else None

    def _np(x):
        return (x.cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)).flatten()

    for i, t in enumerate(range(ep_start, ep_end)):
        frame = dataset[t]

        if "observation.eef.position" in frame:
            eef_pos[i] = _np(frame["observation.eef.position"])
        if "observation.eef.rotation_ortho6" in frame:
            obs_rot6[i] = _np(frame["observation.eef.rotation_ortho6"])
        if "action.position" in frame:
            action_pos[i] = _np(frame["action.position"])
        if "action.stiffness_diag" in frame:
            sd = _np(frame["action.stiffness_diag"])
            if sd.size >= 3:
                k_demo[i] = sd[:3]

        if "observation.ft" in frame:
            ft = _np(frame["observation.ft"])
            ft_tool[i, :ft.size] = ft[:6] if ft.size >= 6 else ft
            force_norm[i] = np.linalg.norm(ft_tool[i, :3])
            torque_norm[i] = np.linalg.norm(ft_tool[i, 3:])

        if include_stiffness and stiffness_key and stiffness_key in frame:
            sv = _np(frame[stiffness_key])
            stiffness[i] = float(np.mean(sv))

    # ── Derived: implied spring force (controller side) and sensor force (world) ──
    # Spring force the demo controller was driving with, in world frame:
    #   F_spring = K_demo * (action.position - obs.eef.position)
    F_spring_world = k_demo * (action_pos - eef_pos)

    # Sensor force expressed in world frame (sign-flipped to get force ON env):
    F_exerted_world = np.zeros_like(F_spring_world)
    for i in range(ep_len):
        f_tool = ft_tool[i, :3]
        if ft_in_tool_frame and np.linalg.norm(obs_rot6[i]) > 0:
            R = transformations.rotation_matrix_from_ortho6(obs_rot6[i])[:3, :3]
            f_world = R @ f_tool
        else:
            f_world = f_tool
        F_exerted_world[i] = -f_world

    # Residual: the part of the sensor reading NOT explained by the spring
    delta_tr_world = F_spring_world - F_exerted_world

    return dict(
        eef_pos=eef_pos,
        force_norm=force_norm,
        torque_norm=torque_norm,
        stiffness=stiffness,
        # new fields for Plot A
        action_pos=action_pos,
        k_demo=k_demo,
        ft_tool=ft_tool,
        F_spring_world=F_spring_world,
        F_exerted_world=F_exerted_world,
        delta_tr_world=delta_tr_world,
    )


# ---------------------------------------------------------------------------
# Plotting (comparison-aware)
# ---------------------------------------------------------------------------

_REPLAY_STYLES = [
    dict(color_set=["tab:blue", "tab:orange", "tab:green"], force_color="tab:red",    stiff_color="tab:green",  alpha=1.0),
    dict(color_set=["#1f77b4",  "#ff7f0e",    "#2ca02c"],   force_color="tab:purple", stiff_color="tab:purple", alpha=0.65),
]


def plot_sensor_vs_spring(
    episode_idx: int,
    dataset_gt: dict,
    save_path: Path,
    contact_threshold_N: float = 2.0,
) -> None:
    """Plot A — sensor force vs. implied controller spring force, from the dataset.

    Produces four panels:
      1) |F_sensor|, |F_spring|, |delta_tr| vs time, with contact shading.
      2) Per-axis F_sensor[i] (solid) vs F_spring[i] (dashed) in world frame.
      3) Scatter |F_spring| vs |F_sensor| coloured by contact / no-contact.
      4) Phase portrait |action - obs| vs |F_sensor| (slope ≈ 1/K_demo at SS).
    """
    F_s = dataset_gt["F_exerted_world"]
    F_k = dataset_gt["F_spring_world"]
    d = dataset_gt["delta_tr_world"]
    Fs_norm = np.linalg.norm(F_s, axis=1)
    Fk_norm = np.linalg.norm(F_k, axis=1)
    d_norm = np.linalg.norm(d, axis=1)
    in_contact = Fs_norm > contact_threshold_N

    ds_eef = dataset_gt["eef_pos"]
    action_pos = dataset_gt["action_pos"]
    disp_norm = np.linalg.norm(action_pos - ds_eef, axis=1)
    if in_contact.any():
        k_demo0 = float(np.median(dataset_gt["k_demo"][in_contact, 0]))
    else:
        k_demo0 = float(np.median(dataset_gt["k_demo"][:, 0])) if dataset_gt["k_demo"].size else 800.0
    if not np.isfinite(k_demo0) or k_demo0 <= 0:
        k_demo0 = 800.0

    steps = np.arange(len(Fs_norm))
    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.30,
                           height_ratios=[1.0, 1.0, 1.2, 0.8])

    # ── Panel 1: norms vs time ──
    ax1 = fig.add_subplot(gs[0, :])
    y_top = max(Fs_norm.max(), Fk_norm.max(), 1.0) * 1.05
    ax1.fill_between(steps, 0, y_top, where=in_contact, color="tab:grey", alpha=0.08,
                     step="mid", label=f"contact (|F_s|>{contact_threshold_N:.1f}N)")
    ax1.plot(steps, Fs_norm, label="|F_sensor| (= -R·ft, world)", color="tab:blue", linewidth=0.9)
    ax1.plot(steps, Fk_norm, label="|F_spring| = |K_demo·(action - obs)|",
             color="tab:orange", linewidth=0.9)
    ax1.plot(steps, d_norm,  label="|F_spring - F_sensor|  (Δ_tr)",
             color="tab:red", linewidth=0.9, alpha=0.85)
    ax1.set_ylim(0, y_top)
    ax1.set_title(f"Episode {episode_idx} — sensor vs implied-spring force (dataset only)")
    ax1.set_ylabel("Force (N)")
    ax1.set_xlabel("Timestep")
    ax1.legend(fontsize=8, ncol=2, loc="upper right")
    ax1.grid(True, linewidth=0.4)

    # ── Panel 2: per-axis time series ──
    ax2 = fig.add_subplot(gs[1, :])
    colours = ["tab:blue", "tab:orange", "tab:green"]
    for i, lbl in enumerate(["X", "Y", "Z"]):
        ax2.plot(steps, F_s[:, i], color=colours[i], linewidth=0.8, label=f"F_sensor {lbl}")
        ax2.plot(steps, F_k[:, i], color=colours[i], linewidth=0.8, linestyle="--",
                 alpha=0.7, label=f"F_spring {lbl}")
    ax2.axhline(0, color="black", linewidth=0.4, linestyle=":")
    ax2.set_title("Per-axis (world frame): sensor (solid) vs implied spring (dashed)")
    ax2.set_ylabel("Force (N)")
    ax2.set_xlabel("Timestep")
    ax2.legend(fontsize=7, ncol=3)
    ax2.grid(True, linewidth=0.4)

    # ── Panel 3: scatter |F_spring| vs |F_sensor| ──
    ax3 = fig.add_subplot(gs[2, 0])
    if (~in_contact).any():
        ax3.scatter(Fs_norm[~in_contact], Fk_norm[~in_contact], s=4, alpha=0.35,
                    color="tab:grey", label="no contact")
    if in_contact.any():
        ax3.scatter(Fs_norm[in_contact],  Fk_norm[in_contact],  s=4, alpha=0.55,
                    color="tab:purple", label="contact")
    lim = max(Fs_norm.max(), Fk_norm.max(), 1.0) * 1.05
    ax3.plot([0, lim], [0, lim], color="black", linewidth=0.6, linestyle=":", label="y=x")
    ax3.set_xlim(0, lim)
    ax3.set_ylim(0, lim)
    ax3.set_aspect("equal", adjustable="box")
    ax3.set_xlabel("|F_sensor| (N)")
    ax3.set_ylabel("|F_spring| (N)")
    ax3.set_title("|F_spring| vs |F_sensor|  (y=x ⇒ quasi-static)")
    ax3.legend(fontsize=7, loc="upper left")
    ax3.grid(True, linewidth=0.4)

    # ── Panel 4: phase portrait |action - obs| vs |F_sensor| ──
    ax4 = fig.add_subplot(gs[2, 1])
    if (~in_contact).any():
        ax4.scatter(Fs_norm[~in_contact], disp_norm[~in_contact], s=4, alpha=0.35,
                    color="tab:grey", label="no contact")
    if in_contact.any():
        ax4.scatter(Fs_norm[in_contact],  disp_norm[in_contact],  s=4, alpha=0.55,
                    color="tab:purple", label="contact")
    F_line = np.linspace(0, max(Fs_norm.max(), 1.0) * 1.05, 50)
    ax4.plot(F_line, F_line / k_demo0, color="black", linewidth=0.6, linestyle=":",
             label=f"slope 1/K_demo (K≈{k_demo0:.0f} N/m)")
    ax4.set_xlabel("|F_sensor| (N)")
    ax4.set_ylabel("|action.position - obs.position| (m)")
    ax4.set_title("Phase portrait: displacement vs sensor force")
    ax4.legend(fontsize=7, loc="upper left")
    ax4.grid(True, linewidth=0.4)

    # ── Panel 5: Δ_tr statistics summary text ──
    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis("off")
    if in_contact.any():
        rms = float(np.sqrt(np.mean(d_norm[in_contact] ** 2)))
        txt = (
            f"Contact-phase statistics (frames with |F_sensor|>{contact_threshold_N} N, "
            f"n={int(in_contact.sum())})\n"
            f"  mean |F_sensor|  = {Fs_norm[in_contact].mean():6.2f} N\n"
            f"  mean |F_spring|  = {Fk_norm[in_contact].mean():6.2f} N\n"
            f"  mean |Δ_tr|      = {d_norm[in_contact].mean():6.2f} N    (RMS = {rms:6.2f} N)\n"
            f"  max  |Δ_tr|      = {d_norm[in_contact].max():6.2f} N\n"
            f"  median K_demo[x] = {k_demo0:6.1f} N/m\n"
            "\n"
            "Interpretation:\n"
            "  - Small |Δ_tr| in contact => quasi-static demo; F_sensor/K_e encoding is K_e-invariant.\n"
            "  - Sustained |F_spring| > |F_sensor| => demo controller was driving harder than\n"
            "    the sensor saw; that gap is re-scaled by K_demo/K_e at replay time and produces\n"
            "    the asymmetric over/undershoot."
        )
    else:
        txt = "No contact frames detected (|F_sensor| never exceeded threshold)."
    ax5.text(0.01, 0.98, txt, va="top", ha="left", family="monospace", fontsize=9)

    fig.suptitle(f"Plot A — sensor vs implied-spring consistency (ep {episode_idx})",
                 fontsize=11)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Plot A saved to: {save_path}")
    plt.close(fig)


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
    ds_torque = dataset_gt["torque_norm"]
    ds_stiffness = dataset_gt["stiffness"]

    has_stiffness = ds_stiffness is not None and any(r.stiffness is not None for r in replays)
    n_rows = 5 if has_stiffness else 4
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
    fd_colors = ["tab:blue", "tab:orange"]
    ax_fn.plot(ds_steps, ds_force, color="tab:grey", linestyle="--", linewidth=0.8, label="dataset")
    for ridx, replay in enumerate(replays):
        style = _REPLAY_STYLES[ridx % len(_REPLAY_STYLES)]
        steps = np.arange(len(replay.force_norm))
        ax_fn.plot(steps, replay.force_norm, color=fd_colors[ridx % len(fd_colors)], linewidth=0.8,
                   alpha=style["alpha"], label=replay.action_type)
    ax_fn.set_title("Force norm (N)")
    ax_fn.set_ylabel("||F|| (N)")
    ax_fn.set_ylim(bottom=0)
    ax_fn.legend(fontsize=7)
    ax_fn.grid(True, linewidth=0.4)

    # ── Row 2 right: Force difference (replay − dataset) ──
    ax_fd = fig.add_subplot(gs[2, 1])
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

    # ── Row 3 left: Torque norm comparison ──
    ax_tn = fig.add_subplot(gs[3, 0])
    ax_tn.plot(ds_steps, ds_torque, color="tab:grey", linestyle="--", linewidth=0.8, label="dataset")
    td_colors = ["tab:blue", "tab:orange"]
    for ridx, replay in enumerate(replays):
        style = _REPLAY_STYLES[ridx % len(_REPLAY_STYLES)]
        steps = np.arange(len(replay.torque_norm))
        color = td_colors[ridx % len(td_colors)]
        ax_tn.plot(steps, replay.torque_norm, color=color, linewidth=0.8,
                   alpha=style["alpha"], label=replay.action_type)
    ax_tn.set_title("Torque norm (Nm)")
    ax_tn.set_ylabel("||τ|| (Nm)")
    ax_tn.set_ylim(bottom=0)
    ax_tn.legend(fontsize=7)
    ax_tn.grid(True, linewidth=0.4)

    # ── Row 3 right: Torque difference (replay − dataset) ──
    ax_td = fig.add_subplot(gs[3, 1])
    for ridx, replay in enumerate(replays):
        T = min(len(replay.torque_norm), len(ds_torque))
        torque_diff = replay.torque_norm[:T] - ds_torque[:T]
        steps = np.arange(T)
        ax_td.plot(steps, torque_diff, color=td_colors[ridx % len(td_colors)], linewidth=0.8,
                   label=replay.action_type)
    ax_td.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax_td.set_title("Torque difference (replay − dataset)")
    ax_td.set_ylabel("Δτ (Nm)")
    ax_td.legend(fontsize=7)
    ax_td.grid(True, linewidth=0.4)

    # ── Row 4: Stiffness comparison (optional) ──
    if has_stiffness:
        ax_st = fig.add_subplot(gs[4, :])
        if ds_stiffness is not None:
            ax_st.plot(ds_steps, ds_stiffness, color="tab:grey", linestyle="--", linewidth=0.8, label="dataset")
        for ridx, replay in enumerate(replays):
            if replay.stiffness is not None:
                style = _REPLAY_STYLES[ridx % len(_REPLAY_STYLES)]
                steps = np.arange(len(replay.stiffness))
                ax_st.plot(steps, replay.stiffness, color=style["stiff_color"], linewidth=0.8,
                           alpha=style["alpha"], label=replay.action_type)
        ax_st.set_title("Stiffness")
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
    config_path="../../../../../../dependencies/comet/configs",
    config_name="blank",
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

    info_path = output_dir / "dataset_info.json"
    with open(info_path, "w") as f:
        json.dump(dataset.meta.info, f, indent=2)
    logger.info(f"Saved dataset.meta.info to {info_path}")

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

    # comparison_action_type = "raw_actions"
    comparison_action_type = "virtual_target_actions"
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

    env = FDCCEnv(config=cfg)
    env.reference_trajectory = []

    logger.info(f"actions_as_deltas: {env.actions_as_deltas}")
    # comparison = False  # FIXME
    logger.info(f"comparison mode: {comparison}")

    # ------------------------------------------------------------------
    # Episode loop
    # ------------------------------------------------------------------
    total_force_errors: dict[str, float] = {}

    move_to_init_qpos(env, reason="initial safe position")

    ft_in_tool_frame = bool(
        dataset.meta.info.get("virtual_target_displacement_config", {})
        .get("ft_in_tool_frame", True)
    )
    logger.info(f"ft_in_tool_frame (from dataset info): {ft_in_tool_frame}")

    for episode_idx in range(start_episode, end_episode):
        ep = dataset.meta.episodes[episode_idx]
        ep_len = int(ep["dataset_to_index"]) - int(ep["dataset_from_index"])
        logger.info(f"Episode {episode_idx}: {ep_len} steps")

        # -- Collect dataset ground truth --
        ds_gt = extract_dataset_ground_truth(
            dataset, episode_idx, include_stiffness,
            stiffness_key="action.stiffness_diag",
            ft_in_tool_frame=ft_in_tool_frame,
        )

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

        # move_to_init_qpos(env, reason=f"after {primary_action_type} ep {episode_idx}")

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

            # move_to_init_qpos(env, reason=f"after {comparison_action_type} ep {episode_idx}")

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
        np.save(output_dir / f"ep{episode_idx}_dataset_F_sensor_world.npy", ds_gt["F_exerted_world"])
        np.save(output_dir / f"ep{episode_idx}_dataset_F_spring_world.npy", ds_gt["F_spring_world"])
        np.save(output_dir / f"ep{episode_idx}_dataset_delta_tr.npy",       ds_gt["delta_tr_world"])

        # -- Plot --
        plot_episode_comparison(
            episode_idx=episode_idx,
            dataset_gt=ds_gt,
            replays=replays,
            save_path=output_dir / f"ep{episode_idx}_comparison.png",
        )
        plot_sensor_vs_spring(
            episode_idx=episode_idx,
            dataset_gt=ds_gt,
            save_path=output_dir / f"ep{episode_idx}_sensor_vs_spring.png",
        )

    # ------------------------------------------------------------------
    # Return to safe position
    # ------------------------------------------------------------------
    # move_to_init_qpos(env, reason="all episodes finished")

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
