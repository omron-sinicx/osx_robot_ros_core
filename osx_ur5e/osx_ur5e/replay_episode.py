#!/usr/bin/env python3
"""Replay a recorded episode from a LeRobot dataset on the UR5e via FDCCEnv.

Usage:
    ros2 run osx_ur5e replay_episode
    ros2 run osx_ur5e replay_episode dataset.episode_idx=3

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

import hydra
import numpy as np
import rclpy
import tqdm
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf
from rich.console import Console
from rich.logging import RichHandler

from comet.common.datasets.utils import tensors_to_numpy
from comet.common.utils.ft_visualizer import FTVisualizer
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from osx_ur5e.fdcc_env import FDCCEnv
from osx_ur5e.ros_node import RosRuntime
from ur_control import transformations

logger = logging.getLogger(__name__)
console = Console()


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
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    )
    root.addHandler(file_handler)


def build_env_action(frame: dict, action_type: str, replay_action_keys: list) -> dict:
    env_action = {}
    frame_np = tensors_to_numpy(frame)

    if action_type == "raw_actions":
        env_action["action.position"] = frame_np["action.position"]
        env_action["action.orientation"] = transformations.quaternion_from_ortho6(
            frame_np["action.rotation_ortho6"]
        )
        env_action["action.stiffness_diag"] = frame_np["action.stiffness_diag"]
    else:
        raise ValueError(f"Invalid action type: {action_type}")
    return env_action


def run(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir / "replay.log")

    np.set_printoptions(linewidth=np.inf, formatter={"float": lambda x: f"{x:0.3f}"})

    repo_id = cfg.dataset.repo_id
    if isinstance(repo_id, (list, ListConfig)):
        repo_id = str(repo_id[0])

    dataset_root = Path(cfg.dataset.dir) / repo_id
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    logger.info("Loading dataset: %s from %s", repo_id, dataset_root)
    dataset = LeRobotDataset(repo_id, root=dataset_root, video_backend="pyav", use_videos=False)

    fps = cfg.dataset.fps
    sleep_time = 1.0 / fps

    episode_idx = int(cfg.dataset.episode_idx)
    num_episodes = dataset.meta.total_episodes
    logger.info("Dataset has %d episodes | replaying episode %d", num_episodes, episode_idx)

    if episode_idx >= num_episodes:
        raise ValueError(
            f"episode_idx {episode_idx} out of range (dataset has {num_episodes} episodes)"
        )

    ep = dataset.meta.episodes[episode_idx]
    ep_start = int(ep["dataset_from_index"])
    ep_end = int(ep["dataset_to_index"])
    ep_len = ep_end - ep_start
    logger.info("Episode %d: frames %d – %d (%d steps)", episode_idx, ep_start, ep_end, ep_len)

    action_type = cfg.dataset.replay
    replay_action_keys = list(cfg.dataset[action_type].keys())
    logger.info("Replay action keys: %s", replay_action_keys)
    input("Press Enter to continue...")

    use_gazebo_sim = bool(OmegaConf.select(cfg, "use_gazebo_sim", default=False))
    runtime = RosRuntime("replay_episode")
    runtime.node.declare_parameter("use_gazebo_sim", use_gazebo_sim)
    # use_real_robot is deliberately left undeclared: URServices probes the UR driver's
    # dashboard client for it, which cannot disagree with reality the way a config value can.
    if use_gazebo_sim:
        runtime.node.declare_parameter("use_sim_time", True)

    try:
        logger.info("ROS node initialized")
        env = FDCCEnv(config=cfg, node=runtime.node, use_torch_for_cameras=False)

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

        logger.info("Moving to home position...")
        env.go_home()

        env.reset(move_robot=True)
        input(f"\n  Episode {episode_idx} ({ep_len} steps) — press Enter to start replay...")

        env.activate_compliance_control()

        logger.info("Replaying episode %d...", episode_idx)

        force_violation = False
        eval_start = timeit.default_timer()
        i = 0

        with tqdm.tqdm(total=ep_len, desc=f"Episode {episode_idx}") as pbar:
            for i, t in enumerate(range(ep_start, ep_end)):
                if not rclpy.ok():
                    break

                step_start = timeit.default_timer()

                frame = dataset[t]
                env_action = build_env_action(frame, action_type, replay_action_keys)

                if i % 50 == 0:
                    logger.info(
                        "step %4d | action.position: %s",
                        i,
                        env_action.get("action.position"),
                    )

                timestep = env.step(env_action)
                if i == 0:
                    time.sleep(1.0)

                if timestep.last():
                    logger.warning(
                        "Episode ended early at step %d (force/torque limit exceeded)", i
                    )
                    force_violation = True
                    break

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
                    logger.debug("Step running slow: %.1f Hz (target %d Hz)", 1.0 / elapsed, fps)
                else:
                    time.sleep(remaining)

                pbar.update(1)

        env.deactivate_compliance_control()

        total_time = timeit.default_timer() - eval_start
        steps_done = ep_len if not force_violation else i + 1
        logger.info(
            "Replay complete | steps: %d/%d | time: %.1fs | force violation: %s",
            steps_done,
            ep_len,
            total_time,
            force_violation,
        )

        ft_visualizer.render_now()
        ft_path = output_dir / f"episode_{episode_idx}_ft.png"
        ft_visualizer.save(ft_path)
        logger.info("FT plot saved to: %s", ft_path)

        _, forces_data, _ = ft_visualizer.get_data()
        np.save(output_dir / f"episode_{episode_idx}_contact_force.npy", forces_data)
        ft_visualizer.close()
    finally:
        runtime.shutdown()


@hydra.main(version_base=None, config_path="../../config/hydra", config_name="test_task")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    if "hydra.run.dir" not in " ".join(sys.argv):
        sys.argv.append("paths.script=/replay")
    main()
