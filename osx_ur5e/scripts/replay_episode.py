#!/usr/bin/env python3
"""Replay a recorded episode from a LeRobot dataset on the real UR5e.

Uses the same cartesian compliance control as data_collection.py:
  action joints → FK → EEF delta → clip → set_cartesian_target_pose

Usage:
    rosrun osx_ur5e replay_episode.py
    rosrun osx_ur5e replay_episode.py dataset.episode_idx=3

    # With policy inference (predicts live from robot observations and compares to GT):
    rosrun osx_ur5e replay_episode.py ++policy_path=/root/osx-ur/outputs/train/.../checkpoints/140000/pretrained_model
"""

import logging
import signal
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import tqdm

import matplotlib
matplotlib.use("Agg")  # must be before pyplot import
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

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

_JOINT_LABELS = ["J1", "J2", "J3", "J4", "J5", "J6", "Gripper"]
_CHUNK_COLORS = [
    "#e41a1c", "#ff7f00", "#4daf4a", "#984ea3",
    "#a65628", "#f781bf", "#8dd3c7", "#377eb8",
]


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


# ---------------------------------------------------------------------------
# Policy helpers (mirrors evaluate_policy.py)
# ---------------------------------------------------------------------------

def load_policy(policy_path: Path, device: str):
    """Load custom_act policy + pre/post processors from a checkpoint directory."""
    import lerobot_policy_custom_act  # noqa: F401 — registers "custom_act"
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    print(f"Loading policy from {policy_path} ...")
    config = PreTrainedConfig.from_pretrained(str(policy_path))
    policy_cls = get_policy_class(config.type)
    policy = policy_cls.from_pretrained(str(policy_path))
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor, config


def build_live_obs(arm, claw, image_recorder, bendlabs, features: dict, camera_shape: tuple) -> dict:
    """Build a policy-ready observation dict from the live robot state."""
    arm_q = np.array(arm.joint_angles())
    gripper = np.array([claw.get_normalized_position()]) if claw is not None else np.array([0.0])
    state = np.concatenate([arm_q, gripper])

    obs = {}
    if "observation.state" in features:
        obs["observation.state"] = torch.tensor(state, dtype=torch.float32)
    if "observation.ft" in features:
        obs["observation.ft"] = torch.tensor(np.array(arm.get_wrench()), dtype=torch.float32)
    if bendlabs is not None and "observation.bendlabs" in features:
        obs["observation.bendlabs"] = torch.tensor(bendlabs.get_angles(), dtype=torch.float32)
    if image_recorder is not None:
        resize = transforms.Resize(camera_shape, antialias=True)
        for cam_name, img_hwc in image_recorder.get_images().items():
            key = f"observation.images.{cam_name}"
            if key in features:
                img_chw = np.ascontiguousarray(np.transpose(img_hwc, (2, 0, 1)))
                obs[key] = resize(torch.tensor(img_chw, dtype=torch.float32) / 255.0)
    return obs


def predict_full_chunk(policy, preprocessor, postprocessor, obs: dict) -> np.ndarray:
    """Run one forward pass and return the full unnormalized chunk (chunk_size, 7)."""
    policy_obs = preprocessor(obs)
    with torch.inference_mode():
        chunk_norm = policy.predict_action_chunk(policy_obs)  # (1, chunk_size, 7)
    chunk_cpu = chunk_norm.squeeze(0).cpu()  # (chunk_size, 7)
    unnorm = [postprocessor(step.unsqueeze(0)).squeeze(0).numpy() for step in chunk_cpu]
    return np.stack(unnorm)  # (chunk_size, 7)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_visualization(
    action_log: list,
    actual_log: list,
    chunk_log: list,
    out_path: Path,
    repo_id: str,
    episode_idx: int,
) -> None:
    """Plot dataset actions, actual arm state, and policy-predicted chunks.

    Args:
        action_log: [{t, action (7,)}] — dataset ground truth actions.
        actual_log: [{t, arm_q (6,), gripper float}] — real arm state each step.
        chunk_log:  [{t, chunk (chunk_size, 7)}] — policy chunk at each inference step.
                    Empty list if no policy was loaded.
        out_path: Output PNG path.
        repo_id: Dataset name for the title.
        episode_idx: Episode number for the title.
    """
    n_dims = 7
    n_cols = 2
    n_rows = (n_dims + 1) // n_cols

    t_arr = np.array([e["t"] for e in action_log])
    actions = np.array([e["action"] for e in action_log])       # (T, 7)
    arm_actual = np.array([e["arm_q"] for e in actual_log])     # (T, 6)
    grip_actual = np.array([e["gripper"] for e in actual_log])  # (T,)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    axes = axes.flatten()

    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    for dim in range(n_dims):
        ax = axes[dim]

        # Dataset ground truth
        ax.plot(t_arr, actions[:, dim], color="#377eb8", linewidth=1.8,
                label="dataset GT", zorder=3)

        # Actual arm state
        ref = arm_actual[:, dim] if dim < 6 else grip_actual
        ax.plot(t_arr, ref, color="#e41a1c", linewidth=1.2,
                linestyle="--", alpha=0.8, label="arm actual", zorder=4)

        if dim < 6:
            gap = np.abs(actions[:, dim] - arm_actual[:, dim])
            ax.fill_between(t_arr, actions[:, dim], arm_actual[:, dim],
                            alpha=0.12, color="orange",
                            label=f"tracking gap (max={gap.max():.3f})")

        # Policy-predicted chunks
        for i, entry in enumerate(chunk_log):
            t0 = entry["t"]
            chunk = entry["chunk"]                      # (chunk_size, 7)
            chunk_t = t0 + np.arange(len(chunk))
            color = _CHUNK_COLORS[i % len(_CHUNK_COLORS)]
            ax.plot(chunk_t, chunk[:, dim],
                    color=color, linewidth=1.5, alpha=0.85,
                    linestyle=linestyles[i % len(linestyles)],
                    label=f"policy t={t0}", zorder=5)
            ax.axvline(t0, color=color, linewidth=0.6, linestyle=":", alpha=0.5)

        ax.set_title(_JOINT_LABELS[dim], fontsize=11, fontweight="bold")
        ax.set_xlabel("Episode step")
        ax.set_ylabel("Value (rad / norm)")
        ax.legend(fontsize=6, loc="upper right", ncol=2)
        ax.grid(True, alpha=0.3)

    for dim in range(n_dims, len(axes)):
        axes[dim].set_visible(False)

    policy_label = f"  |  {len(chunk_log)} policy chunks" if chunk_log else "  (no policy)"
    fig.suptitle(
        f"Replay Visualization  |  {repo_id}  episode={episode_idx}{policy_label}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

def execute_cartesian_action(action_joints: np.ndarray, arm, claw, safety_cfg) -> None:
    """Send a cartesian target derived from action joint angles, mirroring data_collection.py."""
    current_pose = arm.end_effector()
    target_pose = arm.end_effector(joint_angles=action_joints[:6])

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
        claw.set_normalized_position(float(np.clip(action_joints[6], 0.0, 1.0)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(
    version_base=None,
    config_path="../config/hydra",
    config_name="test_task",
)
def main(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "replay.log")

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
    # Optional: load policy for live inference comparison
    # ------------------------------------------------------------------
    policy_path_str = cfg.get("policy_path", None)
    policy = preprocessor = postprocessor = policy_config = None
    n_action_steps = 1
    camera_shape = (480, 640)

    if policy_path_str:
        policy_path = Path(policy_path_str)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        policy, preprocessor, postprocessor, policy_config = load_policy(policy_path, device)
        n_action_steps = policy_config.n_action_steps
        for key, feat in policy_config.input_features.items():
            if key.startswith("observation.images."):
                camera_shape = (feat.shape[-2], feat.shape[-1])
                break
        print(f"Policy loaded: chunk_size={policy_config.chunk_size}  n_action_steps={n_action_steps}  device={device}")

    # ------------------------------------------------------------------
    # Initialize hardware
    # ------------------------------------------------------------------
    rospy.init_node("replay_episode", anonymous=False)
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

    # Camera + BendLabs only needed when running policy inference
    image_recorder = None
    bendlabs = None
    if policy is not None:
        features = policy_config.input_features
        has_images = any(k.startswith("observation.images.") for k in features)
        has_bendlabs = "observation.bendlabs" in features

        if has_images:
            from osx_gym_env.utils import ImageRecorder
            camera_names = [
                k.replace("observation.images.", "")
                for k in features if k.startswith("observation.images.")
            ]
            image_recorder = ImageRecorder(init_node=False, camera_names=camera_names)
            print("Waiting for cameras...")
            deadline = time.perf_counter() + 10.0
            while not image_recorder.cameras_ready() and not rospy.is_shutdown():
                if time.perf_counter() > deadline:
                    print("ERROR: timed out waiting for cameras")
                    sys.exit(1)
                rospy.sleep(0.1)
            print("Cameras ready.")

        if has_bendlabs:
            _num = int(cfg.dataset.get("num_bendlabs_sensors", 4))
            from bendlabs.bendlabs_recorder import BendLabsRecorder
            bendlabs = BendLabsRecorder(init_node=False, num_sensors=_num)
            print(f"Waiting for {_num} BendLabs sensors...")
            deadline = time.perf_counter() + 10.0
            while not bendlabs.sensors_ready() and not rospy.is_shutdown():
                if time.perf_counter() > deadline:
                    print("ERROR: timed out waiting for BendLabs sensors")
                    sys.exit(1)
                rospy.sleep(0.1)
            print("BendLabs ready.")

        policy.reset()

    # ------------------------------------------------------------------
    # Move to first frame's joint configuration
    # ------------------------------------------------------------------
    first_frame = dataset[ep_start]
    first_action = first_frame["action"].numpy()

    logger.info(f"Moving to start position: {first_action[:6]}")
    arm.activate_joint_trajectory_controller()
    arm.set_joint_positions(positions=first_action[:6], target_time=5.0, wait=True)
    if claw is not None:
        claw.set_normalized_position(float(np.clip(first_action[6], 0.0, 1.0)))

    input(f"\n  Episode {episode_idx} ({ep_len} steps) — press Enter to start replay...")

    # ------------------------------------------------------------------
    # Replay loop
    # ------------------------------------------------------------------
    arm.activate_cartesian_controller()
    logger.info(f"Replaying episode {episode_idx} at {fps} Hz...")

    action_log = []   # {t, action (7,)}
    actual_log = []   # {t, arm_q (6,), gripper float}
    chunk_log = []    # {t, chunk (chunk_size, 7)} — only when policy loaded

    with tqdm.tqdm(total=ep_len, desc=f"Episode {episode_idx}") as pbar:
        for t in range(ep_start, ep_end):
            step_start = time.perf_counter()
            step = t - ep_start

            frame = dataset[t]
            action = frame["action"].numpy()

            # Policy inference at every n_action_steps
            if policy is not None and step % n_action_steps == 0:
                obs = build_live_obs(
                    arm, claw, image_recorder, bendlabs,
                    policy_config.input_features, camera_shape,
                )
                with torch.inference_mode():
                    chunk = predict_full_chunk(policy, preprocessor, postprocessor, obs)
                chunk_log.append({"t": step, "chunk": chunk})
                tqdm.tqdm.write(
                    f"  [policy t={step:03d}] "
                    f"pred_act={chunk[0, :6].round(3)}  "
                    f"gt_act={action[:6].round(3)}  "
                    f"gripper_pred={chunk[0, 6]:.3f}  gt_gripper={action[6]:.3f}"
                )

            execute_cartesian_action(action, arm, claw, safety_cfg)

            _arm_q = np.array(arm.joint_angles())
            _grip = claw.get_normalized_position() if claw is not None else 0.0
            action_log.append({"t": step, "action": action.copy()})
            actual_log.append({"t": step, "arm_q": _arm_q.copy(), "gripper": _grip})

            tqdm.tqdm.write(
                f"t={step:03d}  arm={_arm_q.round(3)}  "
                f"act={action[:6].round(3)}  "
                f"max_gap={np.abs(action[:6] - _arm_q).max():.4f} rad  "
                f"gripper_cmd={action[6]:.3f}"
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

    vis_path = output_dir / f"replay_vis_ep{episode_idx:03d}.png"
    print(f"Saving visualization to: {vis_path}")
    try:
        save_visualization(action_log, actual_log, chunk_log, vis_path, repo_id, episode_idx)
        print(f"Visualization saved: {vis_path}")
    except Exception as e:
        import traceback
        print(f"ERROR saving visualization: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
