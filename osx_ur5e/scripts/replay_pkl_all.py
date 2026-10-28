#!/usr/bin/env python3
"""Replay EVERY episode of a pkl-converted LeRobot dataset back-to-back, non-stop.

This is the multi-episode counterpart to replay_pkl_episode.py. It reuses that
script's helpers verbatim (import, not copy) so the two cannot drift in how a
pose is decoded, offset, clipped, or commanded.

Between episodes it performs the same reset evaluate_policy_points.py does
between rollouts: retract straight up by dataset.reset_lift_distance_m (world
+Z, gripper untouched so the object stays held), then ease into the next
episode's start pose. Unlike that script it does NOT prompt between episodes --
the whole point here is an unattended sweep of all 50 demos.

The gripper "insert the object by hand" ritual (dataset.start_gripper=closed)
runs ONCE before the first episode. After that the object is assumed held.

Usage:
    rosrun osx_ur5e replay_pkl_all.py dataset.task_name=peg_hole_dir1_03
    # a subset, and loop forever:
    rosrun osx_ur5e replay_pkl_all.py dataset.task_name=peg_hole_dir1_03 \
        dataset.replay_episodes=0-9 dataset.replay_loop=true
    # pause for Enter between episodes (closer to evaluate_policy_points.py)
    rosrun osx_ur5e replay_pkl_all.py dataset.task_name=... dataset.replay_pause_between=true

Config keys (all optional, defaults keep it non-stop over every episode):
    dataset.replay_episodes       null = all; list [0,3,7]; or "0-9,20,30-35"
    dataset.exclude_episodes      episode ids to skip (e.g. a known-bad demo)
    dataset.reset_lift_distance_m 0.05 -- straight-up retract between episodes
    dataset.replay_pause_between  false -- true waits for Enter between episodes
    dataset.replay_loop           false -- true repeats the list forever
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import tqdm

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig

import rospy

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from osx_claw.claw_controller import ClawController
from osx_ur5e import pose_offset
from ur_control import transformations
from ur_control.fzi_cartesian_compliance_controller import CompliantController

# Reuse replay_pkl_episode.py's helpers so pose decoding / offsetting / delta
# clipping / plotting stay byte-identical between the two scripts.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_pkl_episode import (  # noqa: E402
    GRIPPER_CLOSED_NORM,
    GRIPPER_OPEN_NORM,
    action_to_target_pose,
    execute_cartesian_target,
    gripper_state_to_normalized,
    move_to_pose_via_cartesian_control,
    save_visualization,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_episode_spec(spec, num_episodes: int) -> list:
    """null -> every episode; a list -> as given; "0-9,20,30-35" -> expanded.

    Kept permissive on purpose: the same run is likely to be launched from a
    shell (string form) and from a config file (list form).
    """
    if spec is None:
        return list(range(num_episodes))
    if isinstance(spec, (list, ListConfig)):
        return [int(x) for x in spec]
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def episode_bounds(dataset, idx: int):
    ep = dataset.meta.episodes[idx]
    return int(ep["dataset_from_index"]), int(ep["dataset_to_index"])


def retract_and_reset(arm, claw, next_target_pose, lift_m, safety_cfg, fps):
    """Straight-up retract, then ease into next_target_pose.

    Mirrors evaluate_policy_points.py's between-rollout reset: world +Z (not
    the tool axis -- this is about clearing the fixture, whatever the wrist is
    doing), and the gripper command is whatever it currently is, so a held
    object stays held.
    """
    arm.zero_ft_sensor()
    arm.activate_cartesian_controller()
    hold = claw.get_normalized_position() if claw is not None else GRIPPER_CLOSED_NORM

    lift_pose = arm.end_effector().copy()
    lift_pose[2] += lift_m
    logger.info(f"Retracting {lift_m * 100:.0f} cm straight up (gripper held at {hold:.2f})")
    move_to_pose_via_cartesian_control(lift_pose, arm, claw, hold, safety_cfg, fps)

    logger.info("Easing into next episode's start pose")
    move_to_pose_via_cartesian_control(next_target_pose, arm, claw, hold, safety_cfg, fps)


@hydra.main(version_base=None, config_path="../config/hydra", config_name="test_task")
def main(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "replay_pkl_all.log")
    np.set_printoptions(linewidth=np.inf, formatter={"float": lambda x: f"{x:0.3f}"})

    # ---------------- dataset ----------------
    task_name = cfg.dataset.get("task_name", None)
    if task_name is not None:
        repo_id = f"{task_name}_lerobot"
    else:
        repo_id = cfg.dataset.get("repo_id", None)
        if isinstance(repo_id, (list, ListConfig)):
            repo_id = str(repo_id[0])
        if repo_id is None:
            raise ValueError("Provide dataset.task_name=<name> (preferred) or dataset.repo_id=<name>")

    dataset_root = Path(cfg.dataset.dir) / repo_id
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")
    logger.info(f"Loading dataset: {repo_id} from {dataset_root}")
    dataset = LeRobotDataset(repo_id, root=dataset_root, video_backend="pyav", use_videos=False)

    fps = cfg.dataset.fps
    step_duration_s = 1.0 / fps
    use_gripper = cfg.dataset.get("use_gripper", True)
    num_episodes = dataset.meta.total_episodes

    episodes = parse_episode_spec(cfg.dataset.get("replay_episodes", None), num_episodes)
    excluded = set(int(x) for x in (cfg.dataset.get("exclude_episodes", None) or []))
    episodes = [e for e in episodes if e not in excluded]
    bad = [e for e in episodes if not (0 <= e < num_episodes)]
    if bad:
        raise ValueError(f"episode id(s) {bad} out of range (dataset has {num_episodes})")
    if not episodes:
        raise ValueError("No episodes left to replay after applying exclusions")

    lift_m = float(cfg.dataset.get("reset_lift_distance_m", 0.05))
    pause_between = bool(cfg.dataset.get("replay_pause_between", False))
    loop_forever = bool(cfg.dataset.get("replay_loop", False))
    start_step = int(cfg.dataset.get("start_step", 0))

    offset = pose_offset.resolve_offset(cfg.dataset)
    if pose_offset.is_enabled(offset):
        logger.info(f"dataset.eval_offset: {pose_offset.describe(offset)}")

    logger.info(
        f"Replaying {len(episodes)} episode(s) of {repo_id}: {episodes}"
        + (f"  (excluded {sorted(excluded)})" if excluded else "")
    )
    logger.info(
        f"reset lift {lift_m * 100:.0f} cm | pause_between={pause_between} | loop={loop_forever}"
    )

    # ---------------- hardware ----------------
    rospy.init_node("replay_pkl_all", anonymous=False)
    controller_cfg = cfg.controller
    safety_cfg = controller_cfg.safety_parameters

    from omegaconf import OmegaConf
    arm = CompliantController(gripper_type=None)
    arm.set_control_mode(controller_cfg.mode)
    arm.update_pd_gains(
        OmegaConf.to_container(controller_cfg.p_gains),
        OmegaConf.to_container(controller_cfg.d_gains),
    )
    arm.update_selection_matrix(OmegaConf.to_container(controller_cfg.selection_matrix))
    arm.set_solver_parameters(error_scale=controller_cfg.error_scale, iterations=controller_cfg.iterations)
    arm.update_stiffness(np.array([controller_cfg.stiffness_trans] * 3 + [controller_cfg.stiffness_rot] * 3))
    arm.auto_switch_controllers = False
    arm.async_mode = True
    arm.zero_ft_sensor()

    claw = ClawController(init_node=False) if use_gripper else None

    start_gripper = cfg.dataset.get("start_gripper", "auto")
    if start_gripper not in ("auto", "open", "closed"):
        raise ValueError(f"Invalid dataset.start_gripper: {start_gripper!r}")

    arm.activate_cartesian_controller()
    summary = []

    try:
        first = True
        while True:
            for n, episode_idx in enumerate(episodes):
                ep_start, ep_end = episode_bounds(dataset, episode_idx)
                replay_start = ep_start + start_step
                if not (ep_start <= replay_start < ep_end):
                    raise ValueError(
                        f"start_step={start_step} out of range for episode {episode_idx} "
                        f"({ep_end - ep_start} steps)"
                    )
                replay_len = ep_end - replay_start

                first_action = dataset[replay_start]["action"].numpy()
                target0 = action_to_target_pose(first_action, offset)

                if first:
                    # Only the first episode needs a human: the object has to be
                    # in the gripper before anything can be replayed.
                    if start_gripper == "closed":
                        logger.info("Easing into start pose OPEN for object insertion")
                        move_to_pose_via_cartesian_control(
                            target0, arm, claw, GRIPPER_OPEN_NORM, safety_cfg, fps)
                        input("\n  Insert the object into the gripper, then press Enter to close...")
                        if claw is not None:
                            claw.set_normalized_position(GRIPPER_CLOSED_NORM)
                            time.sleep(1.0)
                    else:
                        g0 = GRIPPER_OPEN_NORM if start_gripper == "open" \
                            else gripper_state_to_normalized(first_action[9])
                        move_to_pose_via_cartesian_control(target0, arm, claw, g0, safety_cfg, fps)
                    input(f"\n  Ready. Press Enter to begin the sweep of {len(episodes)} episode(s)...")
                    first = False
                else:
                    retract_and_reset(arm, claw, target0, lift_m, safety_cfg, fps)
                    if pause_between:
                        input(f"\n  Reset. Press Enter to replay episode {episode_idx}...")

                # ---------------- replay one episode ----------------
                logger.info(f"[{n + 1}/{len(episodes)}] replaying episode {episode_idx} ({replay_len} steps)")
                action_log, actual_log, gaps = [], [], []
                with tqdm.tqdm(total=replay_len, desc=f"ep {episode_idx}") as pbar:
                    for t in range(replay_start, ep_end):
                        step_start = time.perf_counter()
                        step = t - ep_start
                        action = dataset[t]["action"].numpy()
                        target_pose = action_to_target_pose(action, offset)
                        gripper_norm = gripper_state_to_normalized(action[9])
                        execute_cartesian_target(target_pose, arm, claw, gripper_norm, safety_cfg)

                        actual_pose = arm.end_effector()
                        actual_pose6d = np.concatenate([
                            actual_pose[:3], transformations.ortho6_from_quaternion(actual_pose[3:]),
                        ])
                        gap = float(np.abs(action[:9] - actual_pose6d).max())
                        gaps.append(gap)
                        action_log.append({"t": step, "action_pose6d": action[:9].copy(),
                                           "gripper_norm": gripper_norm})
                        actual_log.append({"t": step, "pose6d": actual_pose6d.copy(),
                                           "gripper": claw.get_normalized_position() if claw else 0.0})

                        dt_s = time.perf_counter() - step_start
                        remaining = step_duration_s - dt_s
                        if remaining > 0:
                            time.sleep(remaining)
                        else:
                            logger.debug(f"Step slow: {1.0 / max(dt_s, 1e-9):.1f} Hz (target {fps})")
                        pbar.update(1)

                gaps = np.array(gaps)
                summary.append((episode_idx, replay_len, gaps.mean(), gaps.max()))
                logger.info(
                    f"episode {episode_idx} done: {replay_len} steps, "
                    f"tracking gap mean {gaps.mean():.4f} max {gaps.max():.4f}"
                )
                try:
                    save_visualization(action_log, actual_log,
                                       output_dir / f"replay_ep{episode_idx:03d}.png",
                                       repo_id, episode_idx)
                except Exception:
                    logger.exception(f"Could not save visualization for episode {episode_idx}")

            if not loop_forever:
                break
            logger.info("Loop enabled -- starting the list again")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Always hand the arm back to the joint trajectory controller, even on
        # an exception -- leaving it in cartesian mode is a footgun for whatever
        # runs next.
        try:
            arm.activate_joint_trajectory_controller()
        except Exception:
            logger.exception("Could not restore the joint trajectory controller")

    if summary:
        print(f"\n=== replayed {len(summary)} episode(s) of {repo_id} ===")
        print(f"{'ep':>5}{'steps':>8}{'gap_mean':>11}{'gap_max':>10}")
        for e, n, gm, gx in summary:
            print(f"{e:>5}{n:>8}{gm:>11.4f}{gx:>10.4f}")
        allmax = max(s[3] for s in summary)
        worst = max(summary, key=lambda s: s[3])[0]
        print(f"  worst tracking gap {allmax:.4f} (episode {worst})")
        print(f"  plots + log: {output_dir}")


if __name__ == "__main__":
    main()
