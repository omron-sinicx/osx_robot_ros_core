#!/usr/bin/env python3
"""Replay a recorded episode from a LeRobot dataset on the real UR5e via FDCCEnv.

Mirrors comet/scripts/utils/replay_episode.py but uses FDCCEnv instead of robosuite.

Usage:
    python replay_episode.py

    # Select episode and dataset dir:
    python replay_episode.py dataset.dataset.episode_idx=3

    # Point to a different env config:
    python replay_episode.py +eval.env_config=/path/to/data_collection.yaml

Controls:
    Enter  - confirm start of replay (after reset prompt)
    Ctrl+C - abort
"""

import logging
import signal
import sys
import time
import timeit
from pathlib import Path

import numpy as np
import torch
import tqdm

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

from comet.common.datasets.utils import tensors_to_numpy
import rospy

from rich.console import Console
from rich.logging import RichHandler

from comet.common.utils.ft_visualizer import FTVisualizer
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
# Action extraction
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
# Main
# ---------------------------------------------------------------------------

@hydra.main(
    version_base=None,
    config_path="../../../../../../dependencies/comet/configs",
    config_name="test_wipe_osx",
)
def main(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir / "replay.log")

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

    episode_idx = int(cfg.dataset.dataset.episode_idx)
    num_episodes = dataset.meta.total_episodes
    logger.info(f"Dataset has {num_episodes} episodes | replaying episode {episode_idx}")

    if episode_idx >= num_episodes:
        raise ValueError(f"episode_idx {episode_idx} out of range (dataset has {num_episodes} episodes)")

    ep = dataset.meta.episodes[episode_idx]
    ep_start = int(ep["dataset_from_index"])
    ep_end = int(ep["dataset_to_index"])
    ep_len = ep_end - ep_start
    logger.info(f"Episode {episode_idx}: frames {ep_start} – {ep_end} ({ep_len} steps)")

    # Determine which action keys to replay
    action_type = cfg.dataset.replay
    replay_action_keys = list(cfg.dataset[action_type].keys())
    logger.info(f"Replay action keys: {replay_action_keys}")
    input(f"Press Enter to continue...")

    # ------------------------------------------------------------------
    # Load env config and build FDCCEnv
    # ------------------------------------------------------------------
    env_config_path = OmegaConf.select(cfg, "eval.env_config", default=None)
    if env_config_path is None:
        env_config_path = Path(__file__).parent.parent / "config" / "data_collection.yaml"
        logger.info(f"eval.env_config not set, falling back to: {env_config_path}")
    else:
        env_config_path = Path(env_config_path)

    logger.info(f"Loading env config from: {env_config_path}")
    raw_env_cfg = OmegaConf.load(env_config_path)
    env_cfg = raw_env_cfg.get("env", raw_env_cfg) if hasattr(raw_env_cfg, "get") else raw_env_cfg

    rospy.init_node("replay_episode", anonymous=False)
    logger.info("ROS node initialized")

    env = FDCCEnv(config=env_cfg, use_torch_for_cameras=False)

    # ------------------------------------------------------------------
    # FT Visualizer
    # ------------------------------------------------------------------
    include_stiffness = "action.stiffness_diag" in replay_action_keys
    ft_visualizer = FTVisualizer(
        maxlen=ep_len,
        include_stiffness=include_stiffness,
        force_ylim=(-10, 100),
        arm="right",
        figure_size=(5, 3),
        headless=True,
        keep_focus=False,
    )

    # ------------------------------------------------------------------
    # Move home and reset
    # ------------------------------------------------------------------
    logger.info("Moving to home position...")
    env.go_home()

    env.reset(move_robot=True)
    input(f"\n  Episode {episode_idx} ({ep_len} steps) — press Enter to start replay...")

    env.activate_compliance_control()

    # ------------------------------------------------------------------
    # Replay loop
    # ------------------------------------------------------------------
    logger.info(f"Replaying episode {episode_idx}...")

    force_violation = False
    eval_start = timeit.default_timer()

    with tqdm.tqdm(total=ep_len, desc=f"Episode {episode_idx}") as pbar:
        for i, t in enumerate(range(ep_start, ep_end)):
            step_start = timeit.default_timer()

            frame = dataset[t]
            env_action = build_env_action(frame, action_type, replay_action_keys)

            if i % 50 == 0:
                logger.info(f"step {i:4d} | action.position: {env_action.get('action.position')}")

            timestep = env.step(env_action)
            if i == 0:
                rospy.sleep(1.0)

            if timestep.last():
                logger.warning(f"Episode ended early at step {i} (force/torque limit exceeded)")
                force_violation = True
                break

            # FT tracking
            wrench = env.arm.get_wrench()
            stiffness = None
            if include_stiffness and "action.stiffness_diag" in env_action:
                stiffness = float(np.mean(env_action["action.stiffness_diag"]))
            ft_visualizer.add_data(i, wrench, stiffness)

            if i % 10 == 0:
                ft_visualizer.render_now()

            elapsed = timeit.default_timer() - step_start
            remaining = sleep_time - elapsed
            if remaining < 0:
                logger.debug(f"Step running slow: {1.0/elapsed:.1f} Hz (target {fps} Hz)")
            else:
                rospy.sleep(remaining)

            pbar.update(1)

    env.deactivate_compliance_control()

    total_time = timeit.default_timer() - eval_start
    steps_done = ep_len if not force_violation else i + 1
    logger.info(f"Replay complete | steps: {steps_done}/{ep_len} | "
                f"time: {total_time:.1f}s | force violation: {force_violation}")

    # ------------------------------------------------------------------
    # Save FT plot
    # ------------------------------------------------------------------
    ft_visualizer.render_now()
    ft_path = output_dir / f"episode_{episode_idx}_ft.png"
    ft_visualizer.save(ft_path)
    logger.info(f"FT plot saved to: {ft_path}")

    _, forces_data, _ = ft_visualizer.get_data()
    np.save(output_dir / f"episode_{episode_idx}_contact_force.npy", forces_data)
    ft_visualizer.close()


if __name__ == "__main__":
    if "hydra.run.dir" not in " ".join(sys.argv):
        sys.argv.append("paths.script=/replay")
    main()
