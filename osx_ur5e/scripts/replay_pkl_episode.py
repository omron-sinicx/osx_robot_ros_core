#!/usr/bin/env python3
"""Replay a recorded episode from a pkl-converted LeRobot dataset on the real UR5e.

This is the counterpart to replay_episode.py, for datasets produced by
point_policy/robot_utils/ur5e/convert_pkl_to_lerobot.py. Those datasets store
"action" as a cartesian EE pose (xyz + ortho6d, the continuous 6D rotation
representation from Zhou et al.) + gripper, not joint angles — so unlike
replay_episode.py, no forward-kinematics-from-joint-angles step is needed to
get a cartesian target: the target pose is used directly, decoded from
ortho6d to quaternion via ur_control.transformations.quaternion_from_ortho6,
then sent through the same delta-clip-and-command path as
data_collection.py's set_action().

gripper_states in these datasets are -1 (open) / 1 (closed) (see
convert_pkl_human_to_robot.py's debounce_gripper), rescaled here to
ClawController's normalized [0, 1] range (1=open, 0=closed).

Usage:
    rosrun osx_ur5e replay_pkl_episode.py dataset.repo_id=bottle_open_lerobot
    rosrun osx_ur5e replay_pkl_episode.py dataset.repo_id=bottle_open_lerobot dataset.episode_idx=2
"""

import logging
import signal
import sys
import time
from pathlib import Path

import numpy as np
import tqdm

import matplotlib
matplotlib.use("Agg")  # must be before pyplot import
import matplotlib.pyplot as plt

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

import rospy

from rich.console import Console
from rich.logging import RichHandler

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from osx_claw.claw_controller import ClawController
from ur_control import transformations
from ur_control.fzi_cartesian_compliance_controller import CompliantController

logger = logging.getLogger(__name__)
console = Console()

_POSE_LABELS = ["X", "Y", "Z", "a1x", "a1y", "a1z", "a2x", "a2y", "a2z", "Gripper"]


def _signal_handler(sig, frame):
    logger.info("Interrupted — shutting down.")
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)


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


def gripper_state_to_normalized(gripper_state: float) -> float:
    """-1 (open) / 1 (closed) -> ClawController's normalized [0, 1] (1=open, 0=closed)."""
    return float(np.clip((1.0 - gripper_state) / 2.0, 0.0, 1.0))


def action_to_target_pose(action: np.ndarray) -> np.ndarray:
    """action = [x, y, z, ortho6d(6), gripper] -> [x, y, z, qx, qy, qz, qw]."""
    quat = transformations.quaternion_from_ortho6(action[3:9])  # [qx, qy, qz, qw], matches arm.end_effector()
    return np.concatenate([action[:3], quat])


def execute_cartesian_target(target_pose: np.ndarray, arm, claw, gripper_norm, safety_cfg) -> None:
    """Clip target_pose to a safe delta from the current pose and command it.

    Mirrors data_collection.py's set_action() / replay_episode.py's
    execute_cartesian_action(), but the target pose is already cartesian
    (no forward-kinematics-from-joint-angles step needed).
    """
    current_pose = arm.end_effector()

    delta_translation = target_pose[:3] - current_pose[:3]
    delta_rotation = transformations.quaternions_orientation_error(target_pose[3:], current_pose[3:])

    max_delta_rotation = np.deg2rad(safety_cfg.max_delta_rotation)
    clipped_translation = np.clip(delta_translation,
                                  -safety_cfg.max_delta_translation,
                                  safety_cfg.max_delta_translation)
    clipped_rotation = np.clip(delta_rotation, -max_delta_rotation, max_delta_rotation)

    next_pos = current_pose[:3] + clipped_translation
    next_pos[0] = np.clip(next_pos[0], safety_cfg.workspace_range.x[0], safety_cfg.workspace_range.x[1])
    next_pos[1] = np.clip(next_pos[1], safety_cfg.workspace_range.y[0], safety_cfg.workspace_range.y[1])
    next_pos[2] = np.clip(next_pos[2], safety_cfg.workspace_range.z[0], safety_cfg.workspace_range.z[1])
    next_orient = transformations.rotate_quaternion_by_rpy(*clipped_rotation, current_pose[3:])
    next_target = np.concatenate([next_pos, next_orient])

    arm.set_cartesian_target_pose(pose=next_target)

    if claw is not None:
        claw.set_normalized_position(gripper_norm)


def move_to_pose_via_cartesian_control(
    target_pose: np.ndarray, arm, claw, gripper_norm, safety_cfg, fps: float,
    pos_tol: float = 0.01, rot_tol: float = 0.05, timeout_s: float = 15.0,
) -> None:
    """Ease into target_pose by repeatedly calling execute_cartesian_target
    against the same target, instead of jumping there via IK + joint
    trajectory controller. Each call only moves by safety_cfg's clipped
    per-step delta, so this just re-issues the same target every tick until
    the arm converges (or timeout_s elapses, in case the target is outside
    the safe workspace and can only be approached, never reached exactly).
    """
    step_duration_s = 1.0 / fps
    deadline = time.perf_counter() + timeout_s

    while True:
        step_start = time.perf_counter()

        current_pose = arm.end_effector()
        pos_err = np.linalg.norm(target_pose[:3] - current_pose[:3])
        rot_err = np.linalg.norm(
            transformations.quaternions_orientation_error(target_pose[3:], current_pose[3:])
        )
        if pos_err < pos_tol and rot_err < rot_tol:
            logger.info(f"Reached start pose (pos_err={pos_err:.4f} m, rot_err={rot_err:.4f} rad)")
            return
        if time.perf_counter() > deadline:
            logger.warning(
                f"Timed out easing into start pose after {timeout_s}s "
                f"(pos_err={pos_err:.4f} m, rot_err={rot_err:.4f} rad) -- proceeding anyway."
            )
            return

        execute_cartesian_target(target_pose, arm, claw, gripper_norm, safety_cfg)

        dt_s = time.perf_counter() - step_start
        remaining = step_duration_s - dt_s
        if remaining > 0:
            time.sleep(remaining)


def save_visualization(action_log: list, actual_log: list, out_path: Path,
                       repo_id: str, episode_idx: int) -> None:
    """Plot dataset actions (cartesian xyz + ortho6d + gripper) vs actual arm state."""
    n_dims = 10
    n_cols = 2
    n_rows = (n_dims + 1) // n_cols

    t_arr = np.array([e["t"] for e in action_log])
    actions = np.array([e["action_pose6d"] for e in action_log])  # (T, 9)
    action_gripper = np.array([e["gripper_norm"] for e in action_log])  # (T,)
    actual_pose = np.array([e["pose6d"] for e in actual_log])     # (T, 9)
    actual_gripper = np.array([e["gripper"] for e in actual_log])  # (T,)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    axes = axes.flatten()

    for dim in range(n_dims):
        ax = axes[dim]
        gt = actions[:, dim] if dim < 9 else action_gripper
        actual = actual_pose[:, dim] if dim < 9 else actual_gripper

        ax.plot(t_arr, gt, color="#377eb8", linewidth=1.8, label="dataset GT", zorder=3)
        ax.plot(t_arr, actual, color="#e41a1c", linewidth=1.2, linestyle="--",
                alpha=0.8, label="arm actual", zorder=4)
        gap = np.abs(gt - actual)
        ax.fill_between(t_arr, gt, actual, alpha=0.12, color="orange",
                        label=f"tracking gap (max={gap.max():.3f})")

        ax.set_title(_POSE_LABELS[dim], fontsize=11, fontweight="bold")
        ax.set_xlabel("Episode step")
        ax.set_ylabel("Value (m / rad / norm)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    for dim in range(n_dims, len(axes)):
        axes[dim].set_visible(False)

    fig.suptitle(f"Replay Visualization  |  {repo_id}  episode={episode_idx}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


@hydra.main(
    version_base=None,
    config_path="../config/hydra",
    config_name="test_task",
)
def main(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "replay_pkl.log")

    np.set_printoptions(linewidth=np.inf, formatter={"float": lambda x: f"{x:0.3f}"})

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    repo_id = cfg.dataset.repo_id
    if isinstance(repo_id, (list, ListConfig)):
        repo_id = str(repo_id[0])

    dataset_root = Path(cfg.dataset.dir) / repo_id
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    logger.info(f"Loading dataset: {repo_id} from {dataset_root}")
    dataset = LeRobotDataset(repo_id, root=dataset_root, video_backend="pyav", use_videos=False)

    fps = cfg.dataset.fps
    step_duration_s = 1.0 / fps
    use_gripper = cfg.dataset.get("use_gripper", True)

    episode_idx = int(cfg.dataset.episode_idx)
    num_episodes = dataset.meta.total_episodes
    logger.info(f"Dataset has {num_episodes} episodes | replaying episode {episode_idx}")

    if episode_idx >= num_episodes:
        raise ValueError(f"episode_idx {episode_idx} out of range (dataset has {num_episodes} episodes)")

    ep = dataset.meta.episodes[episode_idx]
    ep_start = int(ep["dataset_from_index"])
    ep_end = int(ep["dataset_to_index"])
    ep_len = ep_end - ep_start
    logger.info(f"Episode {episode_idx}: frames {ep_start} – {ep_end} ({ep_len} steps)")

    # ------------------------------------------------------------------
    # Initialize hardware
    # ------------------------------------------------------------------
    rospy.init_node("replay_pkl_episode", anonymous=False)
    logger.info("ROS node initialized")

    controller_cfg = cfg.controller
    safety_cfg = controller_cfg.safety_parameters

    arm = CompliantController(gripper_type=None)
    arm.set_control_mode(controller_cfg.mode)
    arm.update_pd_gains(
        OmegaConf.to_container(controller_cfg.p_gains),
        OmegaConf.to_container(controller_cfg.d_gains),
    )
    arm.update_selection_matrix(OmegaConf.to_container(controller_cfg.selection_matrix))
    arm.set_solver_parameters(
        error_scale=controller_cfg.error_scale,
        iterations=controller_cfg.iterations,
    )
    arm.update_stiffness(controller_cfg.stiffness * np.ones(6))
    arm.auto_switch_controllers = False
    arm.async_mode = True
    arm.zero_ft_sensor()

    claw = None
    if use_gripper:
        logger.info("Initializing ClawController...")
        claw = ClawController(init_node=False)

    # ------------------------------------------------------------------
    # Move to first frame's pose via the same cartesian delta-clip control
    # used during replay, instead of IK + joint trajectory controller.
    # ------------------------------------------------------------------
    first_action = dataset[ep_start]["action"].numpy()
    first_target_pose = action_to_target_pose(first_action)
    first_gripper_norm = gripper_state_to_normalized(first_action[9])

    logger.info(f"Easing into start pose: {first_target_pose}")
    arm.activate_cartesian_controller()
    move_to_pose_via_cartesian_control(first_target_pose, arm, claw, first_gripper_norm, safety_cfg, fps)

    input(f"\n  Episode {episode_idx} ({ep_len} steps) — press Enter to start replay...")

    # ------------------------------------------------------------------
    # Replay loop
    # ------------------------------------------------------------------
    logger.info(f"Replaying episode {episode_idx} at {fps} Hz...")

    action_log = []   # {t, action_pose6d (9,), gripper_norm}
    actual_log = []   # {t, pose6d (9,), gripper}

    with tqdm.tqdm(total=ep_len, desc=f"Episode {episode_idx}") as pbar:
        for t in range(ep_start, ep_end):
            step_start = time.perf_counter()
            step = t - ep_start

            action = dataset[t]["action"].numpy()  # [x, y, z, ortho6d(6), gripper]
            target_pose = action_to_target_pose(action)
            gripper_norm = gripper_state_to_normalized(action[9])

            execute_cartesian_target(target_pose, arm, claw, gripper_norm, safety_cfg)

            actual_pose = arm.end_effector()
            actual_pose6d = np.concatenate([
                actual_pose[:3], transformations.ortho6_from_quaternion(actual_pose[3:]),
            ])
            actual_gripper = claw.get_normalized_position() if claw is not None else 0.0

            action_log.append({"t": step, "action_pose6d": action[:9].copy(), "gripper_norm": gripper_norm})
            actual_log.append({"t": step, "pose6d": actual_pose6d.copy(), "gripper": actual_gripper})

            tqdm.tqdm.write(
                f"t={step:03d}  actual={actual_pose6d.round(3)}  "
                f"act={action[:9].round(3)}  "
                f"max_gap={np.abs(action[:9] - actual_pose6d).max():.4f}  "
                f"gripper_cmd={gripper_norm:.3f}"
            )

            dt_s = time.perf_counter() - step_start
            remaining = step_duration_s - dt_s
            if remaining < 0:
                logger.debug(f"Step slow: {1.0/dt_s:.1f} Hz (target {fps} Hz)")
            else:
                time.sleep(remaining)

            pbar.update(1)

    arm.activate_joint_trajectory_controller()
    print(f"Replay complete: {ep_len} steps")

    vis_path = output_dir / f"replay_pkl_vis_ep{episode_idx:03d}.png"
    print(f"Saving visualization to: {vis_path}")
    try:
        save_visualization(action_log, actual_log, vis_path, repo_id, episode_idx)
        print(f"Visualization saved: {vis_path}")
    except Exception as e:
        import traceback
        print(f"ERROR saving visualization: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
