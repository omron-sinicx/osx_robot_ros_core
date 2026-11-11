#!/usr/bin/env python3
"""Evaluate a LeRobot ACT policy trained on live-tracked object keypoints.

Observations: EE pose (pos + ortho6d) + gripper as observation.state, and
tracked object keypoints (N*3) from tracking_node.py as
observation.environment_state. No images go to the policy.

Requires tracking_node.py running for the matching task:
    rosrun osx_ur5e tracking_node.py --task_name pick_place_00
    python evaluate_policy_points.py --policy_path outputs/train/act_pick_place_00

Settings come from the hydra config (--hydra_config_name, default test_task);
override any of them as trailing arguments:
    python evaluate_policy_points.py --policy_path ... dataset.n_action_steps=10 \
        dataset.temporal_ensemble_coeff=null

Gripper: dataset.gripper_type = claw (default) | robotiq.

Live keys during a rollout (TTY only):
    Enter - success: end the rollout here.
    f     - fail: stop the arm, open the gripper, lift, and reset.
A rollout that reaches max_timesteps is labelled at a prompt afterwards.
Exceeding dataset.ft_force_limit_n triggers the same reset as 'f'.

Per rollout, saved as rollout<NN>_<outcome>.*: tracking video, FT trace
(ft/), BendLabs trace (bend/) and pose-error plot.
"""

import argparse
import dataclasses
import json
import logging
import os
import select
import signal
import sys
import tempfile
import termios
import time
import timeit
import tty
from collections import Counter, namedtuple
from pathlib import Path

import numpy as np
import torch
from omegaconf import ListConfig, OmegaConf

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import rospy
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

from lerobot.utils.robot_utils import precise_sleep
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import lerobot_policy_custom_act  # noqa: F401 -- registers "custom_act" with draccus

from osx_claw.claw_controller import ClawController
from osx_ur5e import pose_offset
from osx_ur5e.eval_recording import (
    VisualizationRecorder,
    WrenchRecorder,
    save_bend_data,
    save_bend_visualization,
    save_ft_visualization,
    save_pose_error_visualization,
)
from osx_ur5e.eval_recording import logger as eval_recording_logger
from ur_control.fzi_cartesian_compliance_controller import CompliantController
from ur_control import transformations
from bendlabs.bendlabs_recorder import BendLabsRecorder

logger = logging.getLogger(__name__)
console = Console()


def _signal_handler(sig, frame):
    logger.info("Interrupted — shutting down.")
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)


FAIL_KEYS = ("f",)
SUCCESS_KEYS = ("\r", "\n")

# Rollout outcome -> file-name suffix.
OUTCOME_SUFFIXES = {
    "success": "_success",
    "fail": "_fail",
    "ft_limit": "_ft_limit",
}


class KeyPoller:
    """Non-blocking keypress reader for the rollout loop (stdin in cbreak mode).

    Only active between start() and stop(), so the input() prompts between
    rollouts keep normal line editing. No-op when stdin is not a TTY.
    """

    def __init__(self):
        self._fd = None
        self._saved = None
        self._warned = False

    @property
    def enabled(self) -> bool:
        return self._fd is not None

    def start(self) -> bool:
        """Put stdin in cbreak mode. Returns whether key polling is live."""
        if self._fd is not None:
            return True
        try:
            if sys.stdin is None or not sys.stdin.isatty():
                return False
            fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            # Disable echo so keys don't print into the progress bar.
            attrs = termios.tcgetattr(fd)
            attrs[3] &= ~termios.ECHO
            termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
            # Drop keys typed before the rollout started.
            termios.tcflush(fd, termios.TCIFLUSH)
            self._fd = fd
        except (termios.error, ValueError, OSError) as e:
            if not self._warned:
                logger.warning(f"Live keyboard control unavailable ({e}); "
                               "the in-rollout fail/success keys are disabled.")
                self._warned = True
            self._fd, self._saved = None, None
            return False
        return True

    def stop(self) -> None:
        """Restore the terminal. Safe to call repeatedly / when never started."""
        if self._fd is None:
            return
        fd, saved, self._fd, self._saved = self._fd, self._saved, None, None
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        except (termios.error, ValueError, OSError):
            pass

    def poll(self):
        """Return the last key pressed since the previous poll (lowercased), or None."""
        if self._fd is None:
            return None
        key = None
        while select.select([self._fd], [], [], 0)[0]:
            try:
                ch = os.read(self._fd, 1)
            except OSError:
                break
            if not ch:
                break
            key = ch.decode("utf-8", errors="ignore").lower()
        return key


class EnvironmentStateSubscriber:
    """Latest tracking_node ~environment_state plus its age, so the control loop
    can hold position when tracking stalls. `seq` counts messages received."""

    def __init__(self, topic: str):
        self._value = None
        self._timestamp = None
        self.seq = 0
        rospy.Subscriber(topic, Float64MultiArray, self._callback)

    def _callback(self, msg: Float64MultiArray) -> None:
        self._value = np.array(msg.data, dtype=np.float32)
        self._timestamp = time.time()
        self.seq += 1

    def get(self):
        """Returns (env_state, age_seconds); age is inf before the first message."""
        if self._timestamp is None:
            return self._value, float("inf")
        return self._value, time.time() - self._timestamp


def prompt_rollout_outcome(rollout_id: int, steps: int) -> str:
    """Ask whether a rollout that ran to max_timesteps succeeded.
    Returns "success", "fail", or None when there is no terminal."""
    prompt = (f"\n  Rollout {rollout_id} ran to the end ({steps} steps) -- press Enter if it "
              f"SUCCEEDED, or 'f' then Enter if it FAILED: ")
    while True:
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            logger.warning(
                f"Rollout {rollout_id}: no terminal to ask for the outcome -- leaving its "
                "files unlabelled."
            )
            return None
        if answer == "":
            return "success"
        if answer in FAIL_KEYS:
            return "fail"
        print(f"  Unrecognized '{answer}' -- Enter for success, 'f' for fail.")


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


def load_hydra_config(config_path: str, config_name: str, overrides=()):
    """Load the Hydra config used by data_collection.py / replay_episode.py / evaluate_policy.py."""
    config_dir_abs = str((Path(__file__).resolve().parent / config_path).resolve())
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir_abs, version_base=None):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    return cfg


def load_policy_config_compat(policy_path: Path):
    """PreTrainedConfig.from_pretrained, dropping config.json fields unknown to
    the installed lerobot (checkpoints trained with a newer version)."""
    raw = json.loads((policy_path / "config.json").read_text())
    known = {f.name for f in dataclasses.fields(get_policy_class(raw["type"]).config_class)}
    dropped = sorted(k for k in raw if k not in known and k != "type")
    if not dropped:
        return PreTrainedConfig.from_pretrained(str(policy_path))

    logger.warning(
        f"Checkpoint config.json has {len(dropped)} field(s) unknown to the installed "
        f"lerobot ({', '.join(dropped)}) -- it was trained under a newer version. "
        "Ignoring them."
    )
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "config.json").write_text(
            json.dumps({k: v for k, v in raw.items() if k in known or k == "type"})
        )
        return PreTrainedConfig.from_pretrained(td)


def gripper_state_to_normalized(gripper_state: float) -> float:
    """Training encoding (-1 open / +1 closed) -> normalized [0, 1] (1 = open)."""
    return float(np.clip((1.0 - gripper_state) / 2.0, 0.0, 1.0))


def gripper_normalized_to_state(gripper_norm: float) -> float:
    """Normalized [0, 1] (1 = open) -> training encoding (-1 open / +1 closed)."""
    return float(1.0 - 2.0 * gripper_norm)


GRIPPER_OPEN_NORM = 1.0
GRIPPER_CLOSED_NORM = 0.0

# Both backends expose set_normalized_position / get_normalized_command /
# get_normalized_position in [0, 1] (1 = open).
GRIPPER_TYPES = ("claw", "robotiq")


class RobotiqGripperAdapter:
    """ClawController-compatible wrapper around ur_control's RobotiqGripper.

    Remembers the last command (for read_grasp) and sends non-blocking.
    open_fraction maps normalized 1.0 to that fraction of the full stroke,
    e.g. to match the claw's opening width.
    """

    def __init__(self, namespace: str = "", timeout: float = 5.0,
                 open_fraction: float = 1.0):
        # Lazy import: claw-only machines don't need robotiq_msgs.
        from ur_control.grippers import RobotiqGripper

        if not 0.0 < open_fraction <= 1.0:
            raise ValueError(
                f"dataset.robotiq_open_fraction must be in (0, 1], got {open_fraction} "
                f"(it divides the measured position, and 0 would command a gripper that "
                f"can never open)"
            )
        self._open_fraction = float(open_fraction)

        ns = namespace.strip("/")
        self._ns = f"{ns}/" if ns else ""
        try:
            self._gripper = RobotiqGripper(namespace=self._ns, prefix="", timeout=timeout)
        except NameError as e:
            raise RuntimeError(
                "Robotiq support needs the robotiq_msgs package, which is not importable "
                "(ur_control.grippers logged 'Robotiq gripper can't be load' at import). "
                "Build underlay_ws/src/third_party/robotiq-cri and source the workspace, "
                "or set dataset.gripper_type=claw."
            ) from e

        # RobotiqGripper doesn't raise when the action server is missing.
        if not self._gripper.gripper.wait_for_server(rospy.Duration(0.5)):
            raise RuntimeError(
                f"No action server on /{self._ns}gripper_action_controller -- commands would "
                f"be silently dropped.\n"
                f"  The driver node can be running and publishing /{self._ns}status, "
                f"/{self._ns}gripper_status and joint states while still having NO action "
                f"server: cmodel_action_controller.py gives the gripper ~6s to report "
                f"activation (gSTA==3) at startup and, if it does not, returns before "
                f"starting its servers -- look for 'Failed to activate gripper' in its log.\n"
                f"  Check the gripper itself:   rostopic echo -n1 /{self._ns}status\n"
                f"  (gACT=1, gSTA=3, gFLT=0 means it IS activated and the node just gave up "
                f"too early.) If so, restart the node -- it comes up immediately once the "
                f"gripper is already activated:\n"
                f"    rosnode kill /cmodel_action_controller\n"
                f"    rosrun robotiq_control cmodel_action_controller.py"
            )

        # percentage_command() silently ignores any other joint name.
        if self._gripper.gripper_type not in ("finger_joint", "hand-e"):
            raise RuntimeError(
                f"ur_control's RobotiqGripper cannot command joint "
                f"{self._gripper.gripper_type!r} (from /{self._ns}gripper_action_controller/"
                f"joint_name) -- it only handles 'finger_joint' and 'hand-e', and would "
                f"silently ignore every command otherwise. Load one of robotiq_control's "
                f"cmodel_action_controller_{{85,140,hande}}.yaml configs, which all set "
                f"joint_name: finger_joint."
            )

        # None = no gripper_status received yet.
        self._gripper.opening_width = None
        self._cmd_norm = GRIPPER_OPEN_NORM
        self._wait_for_status(timeout)
        if self._gripper.opening_width is not None:
            self._cmd_norm = self.get_normalized_position()
        logger.info(
            f"Robotiq gripper ready (ns='/{self._ns}', max_gap={self._gripper._max_gap:.3f} m, "
            f"position={self._cmd_norm:.2f} normalized). Normalized 1.0 = "
            f"{self._open_fraction:.3f} of the stroke = "
            f"{self._open_fraction * self._gripper._max_gap * 1000:.1f} mm"
            + (" (full jaw width)" if self._open_fraction >= 1.0 else
               f" instead of {self._gripper._max_gap * 1000:.1f} mm")
        )

    def _wait_for_status(self, timeout: float) -> None:
        deadline = time.perf_counter() + timeout
        while self._gripper.opening_width is None and not rospy.is_shutdown():
            if time.perf_counter() > deadline:
                logger.warning(
                    f"No message on /{self._ns}gripper_status within {timeout:.1f}s -- is "
                    f"cmodel_action_controller running? Falling back to reporting the last "
                    f"commanded position as the measured one (grasp detection will never "
                    f"see a gap, so dataset.grasp_binary_enabled will always read 'empty')."
                )
                return
            rospy.sleep(0.05)

    def set_normalized_position(self, value: float) -> None:
        """Command a normalized [0, 1] position (0 = closed, 1 = open)."""
        self._cmd_norm = float(np.clip(value, 0.0, 1.0))
        self._gripper.percentage_command(self._cmd_norm * self._open_fraction, wait=False)

    def get_normalized_command(self) -> float:
        """Last commanded normalized position."""
        return self._cmd_norm

    def get_normalized_position(self) -> float:
        """Measured normalized position; the last command if the driver is silent."""
        if self._gripper.opening_width is None:
            return self._cmd_norm
        return float(np.clip(
            self._gripper.get_opening_percentage() / self._open_fraction, 0.0, 1.0))


def resolve_gripper_type(ds_cfg) -> str:
    """dataset.gripper_type, validated."""
    gripper_type = str(ds_cfg.gripper_type)
    if gripper_type not in GRIPPER_TYPES:
        raise ValueError(
            f"Invalid dataset.gripper_type: {gripper_type!r} "
            f"(expected one of {', '.join(repr(g) for g in GRIPPER_TYPES)})"
        )
    return gripper_type


def make_gripper(ds_cfg):
    """Build the gripper named by dataset.gripper_type."""
    gripper_type = resolve_gripper_type(ds_cfg)
    if gripper_type == "claw":
        logger.info("Initializing ClawController (dataset.gripper_type=claw)...")
        return ClawController(init_node=False)

    namespace = str(ds_cfg.robotiq_namespace)
    timeout = float(ds_cfg.robotiq_timeout)
    open_fraction = float(ds_cfg.robotiq_open_fraction)
    logger.info(
        f"Initializing Robotiq gripper (dataset.gripper_type=robotiq, "
        f"namespace='/{namespace.strip('/')}', open_fraction={open_fraction})..."
    )
    return RobotiqGripperAdapter(namespace=namespace, timeout=timeout,
                                 open_fraction=open_fraction)


GraspConfig = namedtuple("GraspConfig", ["enabled", "cmd_close_threshold", "gap_threshold"])
GraspReading = namedtuple("GraspReading", ["holding", "cmd_norm", "act_norm", "gap"])


def read_grasp(gripper, grasp_cfg: GraspConfig) -> GraspReading:
    """Holding = commanded (near) shut while the measured jaws stay open by at
    least gap_threshold, i.e. blocked by an object."""
    cmd_norm = gripper.get_normalized_command()
    act_norm = gripper.get_normalized_position()
    gap = act_norm - cmd_norm
    holding = bool(cmd_norm <= grasp_cfg.cmd_close_threshold and gap >= grasp_cfg.gap_threshold)
    return GraspReading(holding, cmd_norm, act_norm, gap)


def get_current_state(arm, gripper, grasp_cfg: GraspConfig = None, offset=None, yaw: float = 0.0) -> np.ndarray:
    """observation.state (10,): EE pos + ortho6d + gripper (-1 open / +1 closed).

    With grasp_cfg enabled the gripper element is the binary holding signal
    instead of the measured position. offset/yaw (eval offset) are undone so
    the policy sees its training frame."""
    pose = arm.end_effector().copy()  # [x, y, z, qx, qy, qz, qw]
    if offset is not None:
        pose = pose_offset.undo_offset(pose, offset, yaw)
    ortho6 = transformations.ortho6_from_quaternion(pose[3:])
    if gripper is None:
        gripper_raw = gripper_normalized_to_state(1.0)
    elif grasp_cfg is not None and grasp_cfg.enabled:
        gripper_raw = 1.0 if read_grasp(gripper, grasp_cfg).holding else -1.0
    else:
        gripper_raw = gripper_normalized_to_state(gripper.get_normalized_position())
    return np.concatenate([pose[:3], ortho6, [gripper_raw]]).astype(np.float32)


def execute_cartesian_target(target_pose: np.ndarray, arm, gripper, gripper_norm, safety_cfg) -> np.ndarray:
    """Clip target_pose to a safe step from the current pose, command it, and
    return the clipped target."""
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
    if gripper is not None:
        gripper.set_normalized_position(gripper_norm)
    return next_target


def move_to_pose_via_cartesian_control(
    target_pose: np.ndarray, arm, gripper, gripper_norm, safety_cfg, fps: float,
    pos_tol: float = 0.01, rot_tol: float = 0.05, timeout_s: float = 15.0,
) -> None:
    """Re-issue the same clipped cartesian target each tick until the arm
    converges or timeout_s elapses."""
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
            logger.info(f"Reached target pose (pos_err={pos_err:.4f} m, rot_err={rot_err:.4f} rad)")
            return
        if time.perf_counter() > deadline:
            logger.warning(
                f"Timed out easing into target pose after {timeout_s}s "
                f"(pos_err={pos_err:.4f} m, rot_err={rot_err:.4f} rad) -- proceeding anyway."
            )
            return

        execute_cartesian_target(target_pose, arm, gripper, gripper_norm, safety_cfg)

        dt_s = time.perf_counter() - step_start
        remaining = step_duration_s - dt_s
        if remaining > 0:
            time.sleep(remaining)


def move_to_home(arm, ds_cfg, gripper=None) -> None:
    """Joint-space move to dataset.home_position."""
    home = ds_cfg.home_position
    if home is None:
        logger.warning("dataset.home_position not set; skipping arm homing.")
        return
    home = np.array(home, dtype=float)

    jitter = float(ds_cfg.home_randomization)
    if jitter > 0.0:
        home = home + np.random.uniform(-jitter, jitter, size=home.shape)

    if gripper is not None:
        gripper.set_normalized_position(float(np.clip(ds_cfg.gripper_home, 0.0, 1.0)))
        time.sleep(1.0)

    target_time = float(ds_cfg.home_target_time_s)
    logger.info(f"Homing arm to {home} ({target_time:.1f}s)")
    arm.activate_joint_trajectory_controller()
    arm.set_joint_positions(positions=home, target_time=target_time, wait=True)


def load_start_pose_dataset(ds_cfg):
    """Load the dataset rollout start poses come from. Returns (dataset, repo_id)
    or None to use joint homing instead."""
    if not ds_cfg.use_dataset_start_pose:
        logger.info("dataset.use_dataset_start_pose=false -- using joint-space homing only.")
        return None

    # Prefer the dataset the policy was trained on.
    repo_id = ds_cfg.start_pose_repo_id
    if repo_id is None:
        if ds_cfg.task_name is not None:
            repo_id = f"{ds_cfg.task_name}_lerobot"
        else:
            repo_id = ds_cfg.repo_id
            if isinstance(repo_id, (list, ListConfig)):
                repo_id = str(repo_id[0])
    if repo_id is None:
        logger.warning(
            "No dataset.start_pose_repo_id / task_name / repo_id set -- "
            "falling back to joint-space homing only."
        )
        return None

    dataset_root = Path(ds_cfg.dir) / repo_id
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Start-pose dataset not found: {dataset_root} (from dataset.start_pose_repo_id="
            f"{repo_id!r}). Set dataset.use_dataset_start_pose=false to home to "
            "dataset.home_position instead."
        )

    logger.info(f"Reading start pose from dataset {repo_id} ({dataset_root})")
    dataset = LeRobotDataset(repo_id, root=dataset_root, video_backend="pyav", use_videos=False)
    return dataset, repo_id


def parse_episode_spec(spec, num_episodes: int) -> list:
    """null -> every episode; a list -> as given; "0-9,20,30-35" -> expanded."""
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


def pick_start_pose_episode_idx(ds_cfg, dataset, rollout_id):
    """Episode whose first frame this rollout starts from: "fixed" or
    "sequential" (cycles through dataset.start_pose_episode_idxs)."""
    mode = ds_cfg.start_pose_episode_idx_mode
    if mode == "fixed":
        return int(ds_cfg.start_pose_episode_idx)
    if mode == "sequential":
        pool = parse_episode_spec(ds_cfg.get("start_pose_episode_idxs", None), dataset.meta.total_episodes)
        return pool[rollout_id % len(pool)]
    raise ValueError(
        f"Invalid dataset.start_pose_episode_idx_mode: {mode!r} (expected 'fixed' or 'sequential')"
    )


def resolve_dataset_start_pose(dataset, repo_id, episode_idx, offset, yaw: float = 0.0):
    """First-frame observation.state of episode_idx as a pose, with the eval
    offset applied. Returns (pose7, gripper_norm, label)."""
    if not (0 <= episode_idx < dataset.meta.total_episodes):
        raise ValueError(
            f"dataset start-pose episode_idx={episode_idx} out of range "
            f"({repo_id} has {dataset.meta.total_episodes} episodes)"
        )
    ep_start = int(dataset.meta.episodes[episode_idx]["dataset_from_index"])

    state = dataset[ep_start]["observation.state"].numpy()  # [x,y,z, ortho6d(6), gripper]
    pose = np.concatenate(
        [state[:3], transformations.quaternion_from_ortho6(state[3:9])]
    )
    gripper_norm = gripper_state_to_normalized(float(state[9]))

    pose = pose_offset.apply_offset(pose, offset, yaw)

    label = f"{repo_id} episode {episode_idx} (frame {ep_start})"
    logger.info(
        f"Dataset start pose from {label}: pos={np.round(pose[:3], 4)} "
        f"quat={np.round(pose[3:], 4)} gripper_norm={gripper_norm:.2f}"
        + (f" (includes eval_offset {pose_offset.describe(offset, yaw)})"
           if pose_offset.is_enabled(offset, yaw) else "")
    )
    return pose, gripper_norm, label


def apply_random_start_pose_offset(dataset_start_pose, magnitude_m, rng):
    """Shift the start pose by magnitude_m in a random 3D direction (rng is
    seeded per rollout, so runs are reproducible). Unlike eval_offset, the
    policy sees this shift."""
    if dataset_start_pose is None or magnitude_m == 0.0:
        return dataset_start_pose
    pose, gripper_norm, label = dataset_start_pose
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    offset = magnitude_m * direction
    pose = pose.copy()
    pose[:3] += offset
    logger.info(
        f"Start pose random offset: {magnitude_m * 1000:.1f} mm along "
        f"[{direction[0]:+.2f}, {direction[1]:+.2f}, {direction[2]:+.2f}] "
        f"(dx={offset[0]:+.4f} dy={offset[1]:+.4f} dz={offset[2]:+.4f})"
    )
    return pose, gripper_norm, f"{label} + random offset {magnitude_m * 1000:.1f}mm"


def ease_into_dataset_start_pose(dataset_start_pose, arm, gripper, safety_cfg, fps) -> None:
    """Cartesian move to the start pose, re-issuing the gripper's last command."""
    if dataset_start_pose is None:
        return
    start_pose, _, label = dataset_start_pose

    if gripper is not None:
        hold_norm = gripper.get_normalized_command()
    else:
        hold_norm = GRIPPER_OPEN_NORM

    arm.zero_ft_sensor()
    arm.activate_cartesian_controller()
    logger.info(f"Easing into dataset start pose [{label}]")
    move_to_pose_via_cartesian_control(start_pose, arm, gripper, hold_norm, safety_cfg, fps)


def stop_and_reset(arm, gripper, safety_cfg, fps, lift_distance_m,
                   dataset_start_pose, ds_cfg, open_gripper: bool = True) -> bool:
    """Fail key / FT trip: freeze at the current pose, optionally open the
    gripper, lift lift_distance_m, then return to the start pose (or home).
    Returns whether the gripper was opened."""
    hold_norm = gripper.get_normalized_command() if gripper is not None else GRIPPER_OPEN_NORM

    freeze_pose = arm.end_effector().copy()
    execute_cartesian_target(freeze_pose, arm, gripper, hold_norm, safety_cfg)
    logger.warning(f"STOPPED at pos={np.round(freeze_pose[:3], 4)} -- holding position.")
    time.sleep(0.5)

    released = bool(open_gripper and gripper is not None)
    if released:
        logger.info("Opening the gripper before the reset motion...")
        gripper.set_normalized_position(GRIPPER_OPEN_NORM)
        hold_norm = GRIPPER_OPEN_NORM
        time.sleep(1.0)

    logger.info(f"Retracting {lift_distance_m * 100:.0f} cm straight up before resetting...")
    lift_pose = arm.end_effector().copy()
    lift_pose[2] += lift_distance_m
    move_to_pose_via_cartesian_control(lift_pose, arm, gripper, hold_norm, safety_cfg, fps)

    if dataset_start_pose is None:
        move_to_home(arm, ds_cfg, gripper=None)
    ease_into_dataset_start_pose(dataset_start_pose, arm, gripper, safety_cfg, fps)
    return released


def parse_args():
    argv = rospy.myargv(argv=sys.argv)[1:]
    parser = argparse.ArgumentParser(
        description="Evaluate a LeRobot ACT policy trained on live-tracked object keypoints")
    parser.add_argument("--policy_path", type=str, required=True,
                        help="Path to the trained policy checkpoint directory")
    parser.add_argument("--tracking_node_ns", type=str, default="tracking_node",
                        help="Namespace of the running tracking_node.py.")
    parser.add_argument("--num_rollouts", type=int, default=60)
    parser.add_argument("--max_timesteps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--use_gripper", action="store_true", default=True)
    parser.add_argument("--fps", type=int, default=None,
                        help="Control rate in Hz (default: dataset.fps).")
    parser.add_argument("--hydra_config_path", type=str, default="../config/hydra")
    parser.add_argument("--hydra_config_name", type=str, default="test_task")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_tracking_staleness_s", type=float, default=1.0,
                        help="Hold position while the tracked points are older than this.")
    parser.add_argument("overrides", nargs="*",
                        help="Hydra overrides, e.g. dataset.n_action_steps=10")
    return parser.parse_args(argv)


def init_ros() -> None:
    rospy.init_node("evaluate_policy_points", anonymous=False)
    # init_node replaces the root handlers, so attach the console directly.
    for lg in (logger, eval_recording_logger):
        lg.setLevel(logging.INFO)
        lg.propagate = False
        if not any(isinstance(h, RichHandler) for h in lg.handlers):
            lg.addHandler(RichHandler(console=console, rich_tracebacks=True, show_path=False))
    logger.info("ROS node initialized")


def load_policy(policy_path: Path, ds):
    """Load the checkpoint with dataset.temporal_ensemble_coeff / n_action_steps
    applied. Returns (policy, preprocessor, postprocessor, env_dim)."""
    logger.info(f"Loading policy from: {policy_path}")
    config = load_policy_config_compat(policy_path)

    coeff = ds.temporal_ensemble_coeff
    if coeff is not None:
        if not hasattr(config, "temporal_ensemble_coeff"):
            raise ValueError(
                f"dataset.temporal_ensemble_coeff set, but policy type '{config.type}' has no "
                "temporal_ensemble_coeff field (it is an ACT-family option)."
            )
        config.temporal_ensemble_coeff = float(coeff)
        config.n_action_steps = 1
        logger.info(
            f"Temporal ensembling ENABLED (coeff={coeff}, chunk_size={config.chunk_size}); "
            "n_action_steps forced to 1, so the policy runs a full forward pass every "
            "control step."
        )

    n_action_steps = ds.n_action_steps
    if n_action_steps is not None:
        if coeff is not None:
            raise ValueError(
                "dataset.n_action_steps conflicts with dataset.temporal_ensemble_coeff, which "
                "requires n_action_steps=1. Set temporal_ensemble_coeff to null."
            )
        if n_action_steps < 1:
            raise ValueError(f"dataset.n_action_steps must be >= 1, got {n_action_steps}")
        if n_action_steps > config.chunk_size:
            raise ValueError(
                f"dataset.n_action_steps {n_action_steps} exceeds this checkpoint's "
                f"chunk_size {config.chunk_size}; the policy only predicts "
                f"{config.chunk_size} actions per forward pass. Retrain with a larger "
                "chunk_size to go beyond it."
            )
        previous = config.n_action_steps
        config.n_action_steps = int(n_action_steps)
        logger.info(
            f"n_action_steps overridden {previous} -> {n_action_steps} "
            f"(chunk_size={config.chunk_size}): re-running inference every "
            f"{n_action_steps} control step(s)."
        )

    policy_cls = get_policy_class(config.type)
    policy = policy_cls.from_pretrained(str(policy_path), config=config)
    policy.cuda()
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    logger.info(f"Input features:  {list(config.input_features.keys())}")
    logger.info(f"Output features: {list(config.output_features.keys())}")

    env_feature = config.input_features.get("observation.environment_state")
    if env_feature is None:
        raise ValueError(
            "Policy has no observation.environment_state input feature -- "
            "was it trained on tracked points (convert_pkl_to_lerobot.py output)?"
        )
    return policy, preprocessor, postprocessor, env_feature.shape[0]


class TrackingClient:
    """tracking_node.py's services and environment_state topic, with
    dataset.drop_env_keypoints / env_state_offset applied to what the policy sees."""

    def __init__(self, ns: str, keep_flat: np.ndarray, offset):
        self.ns = ns
        self.keep_flat = keep_flat
        self.offset = offset
        self.reset_srv = rospy.ServiceProxy(f"{ns}/reset_and_track", Trigger)
        self.reset_manual_srv = rospy.ServiceProxy(f"{ns}/reset_and_track_manual", Trigger)
        self.stop_srv = rospy.ServiceProxy(f"{ns}/stop_track", Trigger)
        self.sub = EnvironmentStateSubscriber(f"{ns}/environment_state")
        self._seq_at_reset = 0

    def get(self):
        """Returns (env_state, age_seconds) as the policy should see it."""
        env_state, age_s = self.sub.get()
        if env_state is not None and len(self.keep_flat) != len(env_state):
            env_state = np.asarray(env_state)[self.keep_flat]
        if env_state is not None and self.offset is not None:
            env_state = np.asarray(env_state, dtype=float) + self.offset
        return env_state, age_s

    def relocalize(self) -> bool:
        """Re-localize the object; the operator can retry, pick manually, or skip.
        Returns False if the operator skipped this rollout."""
        use_manual = False
        while True:
            try:
                resp = (self.reset_manual_srv if use_manual else self.reset_srv)()
            except rospy.ServiceException as e:
                resp = None
                logger.warning(f"reset_and_track{'_manual' if use_manual else ''} call failed: {e}")

            if resp is None or not resp.success:
                logger.warning(
                    "tracking_node localization failed"
                    + (f" ({resp.message})" if resp is not None else "") + "."
                )
                choice = input(
                    "  Press Enter to retry localization, 'm' to pick the point(s) "
                    "manually, or 's' to skip this rollout: "
                ).strip().lower()
                if choice == "s":
                    return False
                use_manual = choice == "m"
                continue

            self._seq_at_reset = self.sub.seq
            choice = input(
                "  Tracking (re)initialized -- check "
                f"{self.ns}/visualization_image in rqt_image_view to verify the tracked "
                "point looks right. Press Enter to proceed with this rollout, "
                "'r' to reset tracking again, 'm' to pick the point(s) manually, "
                "or 's' to skip this rollout: "
            ).strip().lower()
            if choice in ("r", "m"):
                use_manual = choice == "m"
                continue
            return choice != "s"

    def wait_first_estimate(self, timeout_s: float = 10.0) -> None:
        """Block until an estimate newer than the last relocalize() arrives."""
        logger.info("Waiting for tracking_node's first estimate (only slow the very first time)...")
        deadline = time.perf_counter() + timeout_s
        while (self.sub.seq <= self._seq_at_reset
               and time.perf_counter() < deadline
               and not rospy.is_shutdown()):
            time.sleep(0.05)
        if self.sub.seq <= self._seq_at_reset:
            logger.warning(
                f"tracking_node produced no estimate within {timeout_s:.0f}s; starting rollout "
                "anyway (the staleness check will hold position until it catches up)."
            )

    def stop(self) -> None:
        self.stop_srv()


def connect_tracking(ns: str, env_dim: int, ds) -> TrackingClient:
    """Check tracking_node's keypoint count against the policy and connect to it."""
    num_points_param = f"{ns}/num_points"
    logger.info(f"Waiting for tracking_node ('{ns}') to report {num_points_param}...")
    n_points = None
    deadline = time.perf_counter() + 15.0
    while n_points is None and time.perf_counter() < deadline and not rospy.is_shutdown():
        if rospy.has_param(num_points_param):
            n_points = rospy.get_param(num_points_param)
        else:
            rospy.sleep(0.2)
    if n_points is None:
        raise RuntimeError(
            f"Timed out waiting for rosparam {num_points_param} -- is tracking_node.py "
            f"running (rosrun osx_ur5e tracking_node.py --task_name <task>)?"
        )

    dropped = sorted(set(ds.drop_env_keypoints))
    bad = [i for i in dropped if not (0 <= i < n_points)]
    if bad:
        raise ValueError(
            f"dataset.drop_env_keypoints {bad} out of range: tracking_node publishes "
            f"{n_points} keypoint(s) (valid indices 0..{n_points - 1})."
        )
    keep_idx = [i for i in range(n_points) if i not in set(dropped)]
    if not keep_idx:
        raise ValueError("dataset.drop_env_keypoints would discard every tracked keypoint.")
    if env_dim != len(keep_idx) * 3:
        raise ValueError(
            f"Policy expects observation.environment_state of dim {env_dim}, but "
            f"tracking_node reports {n_points} point(s) and dataset.drop_env_keypoints={dropped} "
            f"leaves {len(keep_idx)} ({len(keep_idx) * 3} dims). Either "
            "tracking_node.py was started with a --task_name that doesn't match this "
            "policy's task, or dataset.drop_env_keypoints doesn't match the --drop_keypoints "
            "used when the dataset was exported."
        )
    if dropped:
        logger.info(
            f"tracking_node reports {n_points} tracked point(s); dropping {dropped} to match "
            f"the policy -- feeding keypoints {keep_idx} ({env_dim} dims)."
        )
    else:
        logger.info(f"tracking_node reports {n_points} tracked point(s) -- matches policy.")
    keep_flat = np.array([3 * i + c for i in keep_idx for c in range(3)], dtype=int)

    offset = None
    if ds.env_state_offset is not None:
        offset = np.asarray(OmegaConf.to_container(ds.env_state_offset), dtype=float)
        n_dims = len(keep_flat)
        if offset.size == 3:
            offset = np.tile(offset, n_dims // 3)
        elif offset.size != n_dims:
            raise ValueError(
                f"dataset.env_state_offset needs 3 values (a per-keypoint xyz shift) or "
                f"{n_dims} (one per kept dimension); got {offset.size}."
            )
        logger.warning(
            "TEMPORARY dataset.env_state_offset active: adding %s (mm) to the tracked "
            "environment_state. This compensates for a scene shift rather than fixing "
            "it; remove it once the rig is restored or the policy retrained.",
            np.round(offset * 1000, 2).tolist(),
        )

    try:
        for srv in ("reset_and_track", "reset_and_track_manual", "stop_track"):
            rospy.wait_for_service(f"{ns}/{srv}", timeout=15.0)
    except rospy.ROSException as e:
        raise RuntimeError(f"tracking_node services not available: {e}")
    return TrackingClient(ns, keep_flat, offset)


def make_arm(controller_cfg):
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
    stiffness = np.array([controller_cfg.stiffness_trans] * 3 + [controller_cfg.stiffness_rot] * 3)
    arm.update_stiffness(np.diag(stiffness))
    logger.info(f"Initial stiffness: trans={stiffness[:3].round(0)} N/m  "
                f"rot={stiffness[3:].round(0)} Nm/rad")
    arm.auto_switch_controllers = False
    arm.async_mode = True
    arm.zero_ft_sensor()
    return arm


def make_bendlabs(ds):
    """BendLabsRecorder when bend recording is on and the sensors respond, else None."""
    if not (ds.record_bend_data or ds.record_bend_plot):
        return None
    logger.info(f"Initializing BendLabsRecorder ({ds.num_bendlabs_sensors} sensors)...")
    bendlabs = BendLabsRecorder(init_node=False, num_sensors=int(ds.num_bendlabs_sensors))
    deadline = time.perf_counter() + 10.0
    while not bendlabs.sensors_ready() and not rospy.is_shutdown():
        if time.perf_counter() > deadline:
            logger.warning(
                "Timed out waiting for BendLabs sensors -- is uart_bridge_node running? "
                "Disabling bend-sensor recording for this run."
            )
            return None
        rospy.sleep(0.1)
    logger.info("BendLabs sensors ready.")
    return bendlabs


def make_ft_recorder(ds):
    """WrenchRecorder (also implements the force-limit safety stop), or None."""
    force_limit = float(ds.ft_force_limit_n or 0.0)
    if force_limit:
        logger.info(
            f"FT safety stop ARMED: rollout is abandoned when |F| on {ds.ft_topic} exceeds "
            f"{force_limit:.1f} N for {ds.ft_force_limit_consecutive} consecutive samples."
        )
    else:
        logger.warning("FT safety stop DISABLED (dataset.ft_force_limit_n is null/0).")
    if not (ds.record_ft_data or ds.record_ft_plot or force_limit):
        return None
    logger.info(f"Recording FT sensor from {ds.ft_topic} for each rollout.")
    return WrenchRecorder(str(ds.ft_topic), force_limit=force_limit,
                          limit_consecutive=int(ds.ft_force_limit_consecutive))


@dataclasses.dataclass
class EvalContext:
    """Everything that stays fixed across rollouts."""
    args: argparse.Namespace
    ds: object                  # hydra_cfg.dataset
    safety_cfg: object
    fps: int
    output_dir: Path
    policy: object
    preprocessor: object
    postprocessor: object
    tracking: TrackingClient
    arm: object
    gripper: object             # None when --use_gripper is off
    grasp_cfg: GraspConfig
    eval_offset: np.ndarray
    eval_yaw_offset: float
    viz_recorder: object        # VisualizationRecorder or None
    ft_recorder: object         # WrenchRecorder or None
    bendlabs: object            # BendLabsRecorder or None
    keys: KeyPoller


@dataclasses.dataclass
class RolloutResult:
    outcome: str        # key of OUTCOME_SUFFIXES; None = ran to max_timesteps unlabelled
    abandoned: bool     # cut short mid-motion ('f' / force limit) -> stop_and_reset
    steps: int
    pose_error_log: list
    bend_log: list
    bend_baseline: object


def reset_for_rollout(ctx: EvalContext, start_pose, rollout_id: int, reinsert_needed: bool) -> None:
    """Bring the arm (and object) to the rollout's start and wait for the operator.

    reinsert_needed: a reset released the object, so it must be inserted again.
    """
    arm, gripper, ds = ctx.arm, ctx.gripper, ctx.ds
    safety_cfg, fps = ctx.safety_cfg, ctx.fps
    num_rollouts = ctx.args.num_rollouts

    # Object still held from the previous rollout: lift it clear first.
    still_holding = ds.start_gripper == "closed" and rollout_id > 0 and not reinsert_needed
    if still_holding:
        lift_m = float(ds.reset_lift_distance_m)
        logger.info(f"Retracting {lift_m * 100:.0f} cm straight up before resetting to the start pose...")
        arm.zero_ft_sensor()
        arm.activate_cartesian_controller()
        lift_pose = arm.end_effector().copy()
        lift_pose[2] += lift_m
        hold_norm = gripper.get_normalized_position() if gripper is not None else GRIPPER_CLOSED_NORM
        move_to_pose_via_cartesian_control(lift_pose, arm, gripper, hold_norm, safety_cfg, fps)

    if still_holding and not ds.reinsert_every_rollout:
        # Keep holding the object and go straight to the start pose.
        if start_pose is None:
            move_to_home(arm, ds, gripper=None)
        ease_into_dataset_start_pose(start_pose, arm, gripper, safety_cfg, fps)
        input(f"\n  Reset. Press Enter to start rollout {rollout_id + 1}/{num_rollouts}...")
        return

    if start_pose is None:
        move_to_home(arm, ds, gripper)
    elif gripper is not None and ds.start_gripper == "auto":
        # move_to_home is skipped, so apply gripper_home here.
        gripper.set_normalized_position(float(np.clip(ds.gripper_home, 0.0, 1.0)))
        time.sleep(1.0)

    if still_holding and gripper is not None:
        logger.info(
            "reinsert_every_rollout: releasing the object at the retracted "
            "pose -- it will be re-inserted by hand at the start pose."
        )
    if gripper is not None and ds.start_gripper != "auto":
        gripper.set_normalized_position(GRIPPER_OPEN_NORM)

    # Move before grasping/tracking reset, so both match the policy's start.
    ease_into_dataset_start_pose(start_pose, arm, gripper, safety_cfg, fps)

    at_pose = "start pose" if start_pose is not None else "home pose"
    if ds.start_gripper == "closed":
        input(f"\n  At {at_pose} — insert the object into the gripper, then press Enter to close...")
        if gripper is not None:
            gripper.set_normalized_position(GRIPPER_CLOSED_NORM)
            time.sleep(1.0)
        input(f"\n  Closed. Press Enter to start rollout {rollout_id + 1}/{num_rollouts}...")
    else:
        input(f"\n  Reset scene to start position, then press Enter to start rollout "
              f"{rollout_id + 1}/{num_rollouts}...")


def run_rollout(ctx: EvalContext, rollout_id: int, on_step) -> RolloutResult:
    """Run the timed policy loop until max_timesteps, a live key, or the force limit.
    on_step() is called after every executed step."""
    arm, gripper, ds, args = ctx.arm, ctx.gripper, ctx.ds, ctx.args
    step_duration_s = 1.0 / ctx.fps
    last_stale_warn_time = 0.0
    prev_holding = None
    last_grasp_log_time = 0.0
    result = RolloutResult(
        outcome=None, abandoned=False, steps=0, pose_error_log=[], bend_log=[],
        bend_baseline=ctx.bendlabs.get_angles().copy() if ctx.bendlabs is not None else None,
    )

    if ctx.keys.start() and rollout_id == 0:
        logger.info("During a rollout: press 'f' to mark it FAILED, stop the arm, "
                    "abandon that episode and reset; press Enter to mark it "
                    "SUCCEEDED and end it there. A rollout that runs to the end "
                    "asks you afterwards.")

    for t in range(args.max_timesteps):
        result.steps = t + 1
        step_start = time.perf_counter()
        if ctx.ft_recorder is not None:
            ctx.ft_recorder.mark_step(t)

        key = ctx.keys.poll()
        if key in FAIL_KEYS:
            logger.warning(
                f"t={t:03d}  'f' pressed -- rollout marked FAIL; stopping the arm "
                "and abandoning this episode."
            )
            result.outcome, result.abandoned = "fail", True
            break
        if key in SUCCESS_KEYS:
            logger.info(f"t={t:03d}  Enter pressed -- rollout marked SUCCESS, ending here.")
            result.outcome = "success"
            break

        ft_trip = ctx.ft_recorder.check_trip() if ctx.ft_recorder is not None else None
        if ft_trip is not None:
            logger.error(
                f"t={t:03d}  FT LIMIT: |F|={ft_trip.force:.1f} N > "
                f"{float(ds.ft_force_limit_n):.1f} N -- stopping the arm and abandoning "
                "this episode."
            )
            result.outcome, result.abandoned = "ft_limit", True
            break

        if ctx.bendlabs is not None:
            result.bend_log.append((rospy.get_time(), t, *ctx.bendlabs.get_angles()))

        env_state, env_age_s = ctx.tracking.get()
        state = get_current_state(arm, gripper, ctx.grasp_cfg,
                                  offset=ctx.eval_offset, yaw=ctx.eval_yaw_offset)

        if ctx.grasp_cfg.enabled and gripper is not None:
            gr = read_grasp(gripper, ctx.grasp_cfg)
            nums = (f"cmd={gr.cmd_norm:.2f} meas={gr.act_norm:.2f} gap={gr.gap:+.2f} "
                    f"(thr {ctx.grasp_cfg.gap_threshold:.2f})")
            if prev_holding is not None and gr.holding != prev_holding:
                logger.info(
                    f"t={t:03d}  Gripper grasp signal -> "
                    f"{'HOLDING (closed, +1)' if gr.holding else 'EMPTY (open, -1)'}  {nums}"
                )
                last_grasp_log_time = step_start
            elif step_start - last_grasp_log_time > 1.0:
                logger.info(f"t={t:03d}  Gripper: {'HOLDING' if gr.holding else 'empty '}  {nums}")
                last_grasp_log_time = step_start
            prev_holding = gr.holding

        if env_age_s > args.max_tracking_staleness_s:
            if step_start - last_stale_warn_time > 1.0:
                logger.warning(
                    f"t={t:03d}  tracked point is {env_age_s:.2f}s stale "
                    f"(> {args.max_tracking_staleness_s}s) -- holding position, "
                    "not commanding new targets until tracking recovers."
                )
                last_stale_warn_time = step_start
        else:
            obs = {
                "observation.state": torch.tensor(state, dtype=torch.float32),
                "observation.environment_state": torch.tensor(env_state, dtype=torch.float32),
            }
            with torch.inference_mode():
                action = ctx.policy.select_action(ctx.preprocessor(obs))
            action = ctx.postprocessor(action).squeeze().cpu().numpy()  # (10,)

            target_pose = np.concatenate(
                [action[:3], transformations.quaternion_from_ortho6(action[3:9])]
            )
            target_pose = pose_offset.apply_offset(target_pose, ctx.eval_offset, ctx.eval_yaw_offset)
            gripper_norm = gripper_state_to_normalized(action[9])

            next_target = execute_cartesian_target(target_pose, arm, gripper, gripper_norm,
                                                   ctx.safety_cfg)
            if ds.record_pose_error_plot:
                result.pose_error_log.append({"t": t, "commanded": next_target,
                                              "actual": arm.end_effector().copy()})

        dt_s = time.perf_counter() - step_start
        sleep_s = step_duration_s - dt_s
        if sleep_s < 0:
            logger.warning(
                f"Step {t} overran budget: {dt_s * 1000:.1f} ms > {step_duration_s * 1000:.1f} ms "
                f"({1 / dt_s:.1f} Hz actual vs {1.0 / step_duration_s:.1f} Hz target)."
            )
        precise_sleep(max(sleep_s, 0.0))
        on_step()

    ctx.keys.stop()
    return result


def save_artifacts(ctx: EvalContext, rollout_id: int, result: RolloutResult) -> None:
    """Save the rollout's video, pose-error plot, FT and BendLabs traces."""
    ds, out = ctx.ds, ctx.output_dir
    stem = f"rollout{rollout_id:02d}{OUTCOME_SUFFIXES.get(result.outcome, '')}"

    if ctx.viz_recorder is not None:
        ctx.viz_recorder.stop_and_save(out / "videos" / f"{stem}.mp4")

    if ds.record_pose_error_plot:
        save_pose_error_visualization(result.pose_error_log, out / "pose_error" / f"{stem}.png",
                                      rollout_id)

    if ctx.ft_recorder is not None:
        ft_data = ctx.ft_recorder.stop_and_save(
            out / "ft" / f"{stem}.npz" if ds.record_ft_data else None)
        if ft_data is not None and ds.record_ft_plot:
            save_ft_visualization(ft_data, out / "ft" / f"{stem}.png", rollout_id)

    if ctx.bendlabs is not None:
        bend_data = save_bend_data(
            result.bend_log, result.bend_baseline,
            out / "bend" / f"{stem}.npz" if ds.record_bend_data else None, rollout_id)
        if bend_data is not None and ds.record_bend_plot:
            save_bend_visualization(bend_data, out / "bend" / f"{stem}.png", rollout_id)


def log_summary(num_rollouts: int, rollout_outcomes: list, steps_per_episode: list,
                output_dir: Path) -> None:
    logger.info("=" * 60)
    logger.info(f"Rollouts completed:        {num_rollouts}")
    logger.info(f"Rollouts with steps recorded: {len(steps_per_episode)} "
                f"({num_rollouts - len(steps_per_episode)} abandoned)")

    counts = Counter(outcome for _, outcome in rollout_outcomes)
    labelled = counts["success"] + counts["fail"]
    logger.info(
        "Outcomes: "
        + ", ".join(f"{counts[k]} {label}" for k, label in
                    (("success", "success"), ("fail", "fail"), ("ft_limit", "FT limit")))
        + (f", {counts[None]} unlabelled" if counts[None] else "")
    )
    if labelled:
        logger.info(f"Success rate (of {labelled} judged): "
                    f"{100.0 * counts['success'] / labelled:.0f}%")
    successes = [i for i, outcome in rollout_outcomes if outcome == "success"]
    if successes:
        logger.info(f"Succeeded: rollouts {', '.join(str(i) for i in successes)}")
    if steps_per_episode:
        logger.info(f"Mean steps per episode:    {np.mean(steps_per_episode):.1f}")
        logger.info(f"Std  steps per episode:    {np.std(steps_per_episode):.1f}")
    else:
        logger.info("Mean/std steps per episode: n/a -- every rollout was abandoned")
    logger.info("=" * 60)
    logger.info(f"Evaluation complete. Results saved to: {output_dir}")


def main():
    args = parse_args()

    # A missing local path would otherwise be treated as a HF Hub repo id.
    policy_path = Path(args.policy_path).resolve()
    if not policy_path.is_dir():
        raise FileNotFoundError(
            f"Policy checkpoint directory not found: {policy_path} "
            f"(from --policy_path={args.policy_path!r}, resolved relative to cwd={Path.cwd()})"
        )
    if not (policy_path / "config.json").exists():
        raise FileNotFoundError(
            f"{policy_path} exists but has no config.json -- is this a valid "
            "policy checkpoint directory (not e.g. its parent outputs/train/ dir)?"
        )

    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs/eval") / policy_path.name
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "evaluation.log")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Results will be saved to: {output_dir}")

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    hydra_cfg = load_hydra_config(args.hydra_config_path, args.hydra_config_name, args.overrides)
    logger.info(f"Loaded hydra config '{args.hydra_config_name}'"
                + (f" with overrides {args.overrides}" if args.overrides else ""))
    ds = hydra_cfg.dataset
    safety_cfg = hydra_cfg.controller.safety_parameters
    fps = args.fps or int(ds.fps)
    logger.info(f"Control frequency: {fps} Hz | step budget: {1000.0 / fps:.1f} ms")

    if ds.start_gripper not in ("auto", "open", "closed"):
        raise ValueError(f"Invalid dataset.start_gripper: {ds.start_gripper!r} "
                         "(expected 'auto', 'open', or 'closed')")
    if ds.reinsert_every_rollout and ds.start_gripper != "closed":
        logger.warning(
            "dataset.reinsert_every_rollout is set but dataset.start_gripper is "
            f"{ds.start_gripper!r}, not 'closed' -- there is no insert-by-hand step to repeat, "
            "so it has no effect."
        )
    # Validate early so a typo fails before the policy loads.
    resolve_gripper_type(ds)

    # Tool-frame offset/yaw added to every command and removed from
    # observation.state, so the policy keeps its training frame.
    eval_offset = pose_offset.resolve_offset(ds)
    eval_yaw_offset = pose_offset.resolve_yaw_offset(ds)
    if pose_offset.is_enabled(eval_offset, eval_yaw_offset):
        logger.info(
            f"dataset.eval_offset: {pose_offset.describe(eval_offset, eval_yaw_offset)} -- "
            "physical pose will be shifted/rotated along the gripper's own axes relative to "
            "the policy's frame."
        )

    # Rollouts start from a demo's first-frame pose (re-picked per rollout).
    start_pose_dataset = load_start_pose_dataset(ds)

    def get_start_pose(rollout_id):
        if start_pose_dataset is None:
            return None
        dataset, repo_id = start_pose_dataset
        episode_idx = pick_start_pose_episode_idx(ds, dataset, rollout_id)
        pose = resolve_dataset_start_pose(dataset, repo_id, episode_idx, eval_offset, eval_yaw_offset)
        rng = np.random.default_rng(args.seed + rollout_id)
        return apply_random_start_pose_offset(pose, float(ds.start_pose_random_offset_m), rng)

    get_start_pose(0)  # fail early on a bad start-pose episode

    # ------------------------------------------------------------------
    # Policy, tracking, hardware
    # ------------------------------------------------------------------
    policy, preprocessor, postprocessor, env_dim = load_policy(policy_path, ds)

    init_ros()
    ns = args.tracking_node_ns.rstrip("/")
    tracking = connect_tracking(ns, env_dim, ds)

    ctx = EvalContext(
        args=args,
        ds=ds,
        safety_cfg=safety_cfg,
        fps=fps,
        output_dir=output_dir,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        tracking=tracking,
        viz_recorder=VisualizationRecorder(f"{ns}/visualization_image") if ds.record_rollout_video else None,
        ft_recorder=make_ft_recorder(ds),
        arm=make_arm(hydra_cfg.controller),
        gripper=make_gripper(ds) if args.use_gripper else None,
        bendlabs=make_bendlabs(ds),
        grasp_cfg=GraspConfig(bool(ds.grasp_binary_enabled), float(ds.grasp_cmd_close_threshold),
                              float(ds.grasp_gap_threshold)),
        eval_offset=eval_offset,
        eval_yaw_offset=eval_yaw_offset,
        keys=KeyPoller(),
    )
    np.set_printoptions(linewidth=np.inf, formatter={"float": lambda x: f"{x:0.3f}"})

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    num_rollouts = args.num_rollouts
    steps_per_episode = []
    rollout_outcomes = []     # (rollout_id, outcome)
    eval_start_time = timeit.default_timer()
    total_steps_completed = 0
    reinsert_needed = False   # a reset released the object

    def make_step_description(rollout_id: int) -> str:
        elapsed = timeit.default_timer() - eval_start_time
        actual_fps = total_steps_completed / elapsed if elapsed > 0 else 0.0
        return f"Rollout {rollout_id + 1}/{num_rollouts} | FPS: {actual_fps:.1f}"

    progress_columns = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=80),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    try:
        with Progress(*progress_columns, console=console) as progress:
            rollout_task = progress.add_task("Evaluation", total=num_rollouts)

            for rollout_id in range(num_rollouts):
                start_pose = get_start_pose(rollout_id)
                step_task = progress.add_task(make_step_description(rollout_id),
                                              total=args.max_timesteps)

                def finish_rollout():
                    progress.remove_task(step_task)
                    progress.update(rollout_task, advance=1)

                progress.stop()
                reset_for_rollout(ctx, start_pose, rollout_id, reinsert_needed)
                reinsert_needed = False
                ctx.arm.zero_ft_sensor()
                ctx.arm.activate_cartesian_controller()
                proceed = tracking.relocalize()
                progress.start()

                if not proceed:
                    logger.warning(f"Rollout {rollout_id + 1}: skipped by user.")
                    tracking.stop()
                    finish_rollout()
                    continue

                policy.reset()
                if ctx.viz_recorder is not None:
                    ctx.viz_recorder.start()
                if ctx.ft_recorder is not None:
                    ctx.ft_recorder.start()

                progress.stop()
                tracking.wait_first_estimate()
                progress.start()

                def on_step():
                    nonlocal total_steps_completed
                    total_steps_completed += 1
                    progress.update(step_task, advance=1,
                                    description=make_step_description(rollout_id))

                result = run_rollout(ctx, rollout_id, on_step)
                tracking.stop()

                if result.outcome is None:
                    progress.stop()
                    result.outcome = prompt_rollout_outcome(rollout_id, result.steps)
                    progress.start()

                save_artifacts(ctx, rollout_id, result)
                rollout_outcomes.append((rollout_id, result.outcome))

                if result.abandoned:
                    reason = "the FT force limit" if result.outcome == "ft_limit" else "the user"
                    progress.stop()
                    released = stop_and_reset(ctx.arm, ctx.gripper, safety_cfg, fps,
                                              float(ds.reset_lift_distance_m), start_pose,
                                              ds, open_gripper=bool(ds.reset_open_gripper))
                    reinsert_needed = released
                    progress.start()
                    logger.warning(
                        f"Rollout {rollout_id + 1}: abandoned by {reason} at t={result.steps - 1}, "
                        "reset complete -- excluded from the step statistics, but it still "
                        "counts against --num_rollouts."
                    )
                else:
                    ctx.arm.activate_joint_trajectory_controller()
                    steps_per_episode.append(result.steps)
                    logger.info(
                        f"Rollout {rollout_id} complete | steps: {result.steps} | "
                        f"{(result.outcome or 'unlabelled').upper()}"
                    )
                finish_rollout()
    finally:
        ctx.keys.stop()
        if ctx.viz_recorder is not None:
            ctx.viz_recorder.stop()
        if ctx.ft_recorder is not None:
            ctx.ft_recorder.stop()
        try:
            tracking.stop()
        except Exception:
            pass

    log_summary(num_rollouts, rollout_outcomes, steps_per_episode, output_dir)


if __name__ == "__main__":
    main()
