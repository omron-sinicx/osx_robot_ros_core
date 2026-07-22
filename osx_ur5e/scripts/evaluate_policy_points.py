#!/usr/bin/env python3
"""Evaluate a LeRobot ACT policy trained on live-tracked object keypoints.

Counterpart to evaluate_policy.py, for policies trained on datasets produced
by point_policy/robot_utils/ur5e/convert_pkl_to_lerobot.py:
    observation.state             -- EE cartesian pose (pos + ortho6d) + gripper (10,)
    observation.environment_state -- tracked object keypoints, world-frame 3D (N*3,)
    action                        -- next EE cartesian pose + gripper (10,)
No images are fed to the policy at all (use_videos=False) -- instead this
script consumes the offline pipeline's tracked-point signal live, produced by
a separate tracking_node.py ROS node (DIFT localization + causal TAPIR
tracking + two-camera triangulation). This script itself does no DIFT/TAPIR
work and never touches the cameras -- it only calls tracking_node's services
and subscribes to its topics.

Requires tracking_node.py to already be running for the matching task:
    rosrun osx_ur5e tracking_node.py --task_name pick_place_00

Usage:
    python evaluate_policy_points.py \
        --policy_path outputs/train/act_pick_place_00

Controls during each rollout:
    Enter  - confirm start of rollout (after reset prompt)
"""

import argparse
import logging
import signal
import sys
import time
import timeit
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import matplotlib
matplotlib.use("Agg")  # must be before any pyplot import

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

import lerobot_policy_custom_act  # noqa: F401 -- registers "custom_act" with draccus

from osx_claw.claw_controller import ClawController
from ur_control.fzi_cartesian_compliance_controller import CompliantController
from ur_control import transformations
from bendlabs.bendlabs_recorder import BendLabsRecorder

logger = logging.getLogger(__name__)
console = Console()


def _signal_handler(sig, frame):
    logger.info("Interrupted — shutting down.")
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)


class EnvironmentStateSubscriber:
    """Subscribes to tracking_node's ~environment_state (std_msgs/Float64MultiArray)
    and records the wall-clock time each message was received, so the 30Hz
    control loop can detect a stalled/dead tracking node and stop commanding
    the arm instead of acting on an indefinitely-aging tracked point (same
    role SharedValue/TrackingWorker played when tracking ran in-process).
    `seq` counts messages received, so a caller can detect "at least one new
    message since I last checked" (see wait loops in main()). No lock
    needed -- single writer (rospy's subscriber callback thread), same
    GIL-atomic-attribute-assignment reasoning as RawImageRecorder used."""

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
        """Returns (env_state, age_seconds). age_seconds is inf until the
        first message ever arrives."""
        if self._timestamp is None:
            return self._value, float("inf")
        return self._value, time.time() - self._timestamp


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


def load_hydra_config(config_path: str, config_name: str):
    """Load the Hydra config used by data_collection.py / replay_episode.py / evaluate_policy.py."""
    config_dir_abs = str((Path(__file__).resolve().parent / config_path).resolve())
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir_abs, version_base=None):
        cfg = compose(config_name=config_name)
    return cfg


def gripper_state_to_normalized(gripper_state: float) -> float:
    """-1 (open) / 1 (closed) -> ClawController's normalized [0, 1] (1=open, 0=closed).
    Same convention as replay_pkl_episode.py -- used to command the predicted
    gripper action."""
    return float(np.clip((1.0 - gripper_state) / 2.0, 0.0, 1.0))


def gripper_normalized_to_state(gripper_norm: float) -> float:
    """Inverse of the above -- ClawController's normalized [0, 1] -> raw -1/1
    training encoding. Used to read the *current* gripper position into
    observation.state, matching how the training data encoded it."""
    return float(1.0 - 2.0 * gripper_norm)


def get_current_state(arm, claw) -> np.ndarray:
    """Current EE cartesian pose (pos + ortho6d) + gripper -> (10,), matching
    convert_pkl_to_lerobot.py's observation.state encoding exactly."""
    pose = arm.end_effector()  # [x, y, z, qx, qy, qz, qw]
    ortho6 = transformations.ortho6_from_quaternion(pose[3:])
    gripper_norm = claw.get_normalized_position() if claw is not None else 1.0
    gripper_raw = gripper_normalized_to_state(gripper_norm)
    return np.concatenate([pose[:3], ortho6, [gripper_raw]]).astype(np.float32)


def execute_cartesian_target(target_pose: np.ndarray, arm, claw, gripper_norm, safety_cfg) -> None:
    """Clip target_pose to a safe delta from the current pose and command it.
    Identical to replay_pkl_episode.py's function of the same name."""
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


def move_to_home(arm, cfg, claw=None) -> None:
    """Same as evaluate_policy.py's function of the same name."""
    if cfg is None:
        logger.warning("Hydra config unavailable; skipping arm homing.")
        return
    ds_cfg = cfg.dataset
    home = ds_cfg.get("home_position", None)
    if home is None:
        logger.warning("dataset.home_position not set; skipping arm homing.")
        return
    home = np.array(home, dtype=float)

    jitter = float(ds_cfg.get("home_randomization", 0.0))
    if jitter > 0.0:
        home = home + np.random.uniform(-jitter, jitter, size=home.shape)

    target_time = float(ds_cfg.get("home_target_time_s", 4.0))
    logger.info(f"Homing arm to {home} ({target_time:.1f}s)")
    arm.activate_joint_trajectory_controller()
    arm.set_joint_positions(positions=home, target_time=target_time, wait=True)

    if claw is not None:
        gripper_home = float(ds_cfg.get("gripper_home", 0.0))
        claw.set_normalized_position(float(np.clip(gripper_home, 0.0, 1.0)))


def parse_args():
    argv = rospy.myargv(argv=sys.argv)[1:]
    parser = argparse.ArgumentParser(
        description="Evaluate a LeRobot ACT policy trained on live-tracked object keypoints")
    parser.add_argument("--policy_path", type=str, required=True,
                        help="Path to the trained policy checkpoint directory")
    parser.add_argument("--tracking_node_ns", type=str, default="tracking_node",
                        help="Namespace tracking_node.py was started in -- selects which "
                             "services/topics to talk to (<ns>/reset_and_track, "
                             "<ns>/stop_track, <ns>/environment_state, <ns>/visualization_image, "
                             "<ns>/num_points).")
    parser.add_argument("--num_rollouts", type=int, default=5)
    parser.add_argument("--max_timesteps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--use_gripper", action="store_true", default=True)
    parser.add_argument("--fps", type=int, default=None,
                        help="Control frequency in Hz. If not given, read from dataset.fps "
                             "in the hydra config (falls back to 30 if unavailable).")
    parser.add_argument("--hydra_config_path", type=str, default="../config/hydra")
    parser.add_argument("--hydra_config_name", type=str, default="test_task")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_tracking_staleness_s", type=float, default=1.0,
                        help="If tracking_node hasn't produced a fresh environment_state "
                             "within this many seconds (stalled, crashed, or not running), "
                             "stop commanding new cartesian targets (hold at the last "
                             "commanded pose) until tracking recovers.")
    return parser.parse_args(argv)


def main():
    args = parse_args()

    # Resolve to absolute up front: rosrun inherits the calling shell's cwd,
    # which is easy to lose track of, and PreTrainedConfig.from_pretrained
    # silently treats any path that doesn't resolve to a real directory as a
    # HuggingFace Hub repo id instead -- producing a confusing
    # HFValidationError with no mention of the actual (missing) local path.
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
    # Hydra config (controller/safety parameters + fps/home fallback)
    # ------------------------------------------------------------------
    hydra_cfg = None
    try:
        hydra_cfg = load_hydra_config(args.hydra_config_path, args.hydra_config_name)
        logger.info(f"Loaded hydra config '{args.hydra_config_name}'")
    except Exception as e:
        logger.warning(f"Could not load hydra config '{args.hydra_config_name}': {e}")

    dataset_fps = int(hydra_cfg.dataset.fps) if hydra_cfg is not None else None
    controller_cfg = hydra_cfg.controller if hydra_cfg is not None else None
    safety_cfg = controller_cfg.safety_parameters if controller_cfg is not None else None

    if args.fps is not None:
        fps = args.fps
    elif dataset_fps is not None:
        fps = dataset_fps
        logger.info(f"--fps not given; using dataset.fps={fps} from hydra config.")
    else:
        fps = 30
        logger.warning(f"Hydra config unavailable; falling back to fps={fps}.")

    # On a large step-to-step change in the policy's predicted gripper
    # command, stretch out (not freeze) the next gripper_slowdown_duration_s
    # of real time by multiplying the per-step sleep by
    # gripper_slowdown_factor -- gives the slow claw a head start before the
    # arm moves far. Same mechanism as replay_pkl.py, adapted for a
    # continuous per-step command instead of a recorded discrete array.
    ds_cfg_for_slowdown = hydra_cfg.dataset if hydra_cfg is not None else None
    gripper_slowdown_enabled = bool(ds_cfg_for_slowdown.get("gripper_slowdown_enabled", True)) \
        if ds_cfg_for_slowdown is not None else True
    gripper_slowdown_factor = float(ds_cfg_for_slowdown.get("gripper_slowdown_factor", 4.0)) \
        if ds_cfg_for_slowdown is not None else 4.0
    gripper_slowdown_duration_s = float(ds_cfg_for_slowdown.get("gripper_slowdown_duration_s", 2.0)) \
        if ds_cfg_for_slowdown is not None else 2.0
    gripper_slowdown_change_threshold = float(ds_cfg_for_slowdown.get("gripper_slowdown_change_threshold", 0.1)) \
        if ds_cfg_for_slowdown is not None else 0.1

    # Slow the rollout while any BendLabs bend/twist sensor deviates from its
    # start-of-episode baseline by more than bendlab_slowdown_threshold -- a
    # direct physical-contact signal, independent of (and combined via max()
    # with) the gripper_slowdown_* trigger above. Level-based, not a timed
    # window: reverts to normal speed as soon as the deviation drops back
    # under threshold.
    bendlab_slowdown_enabled = bool(ds_cfg_for_slowdown.get("bendlab_slowdown_enabled", True)) \
        if ds_cfg_for_slowdown is not None else True
    bendlab_slowdown_threshold = float(ds_cfg_for_slowdown.get("bendlab_slowdown_threshold", 5.0)) \
        if ds_cfg_for_slowdown is not None else 5.0
    bendlab_slowdown_factor = float(ds_cfg_for_slowdown.get("bendlab_slowdown_factor", 4.0)) \
        if ds_cfg_for_slowdown is not None else 4.0
    num_bendlabs_sensors = int(ds_cfg_for_slowdown.get("num_bendlabs_sensors", 4)) \
        if ds_cfg_for_slowdown is not None else 4

    # ------------------------------------------------------------------
    # Load policy
    # ------------------------------------------------------------------
    logger.info(f"Loading policy from: {policy_path}")
    config = PreTrainedConfig.from_pretrained(str(policy_path))
    policy_cls = get_policy_class(config.type)
    policy = policy_cls.from_pretrained(str(policy_path))
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

    # ------------------------------------------------------------------
    # Connect to tracking_node.py: sanity-check N against the policy, wait
    # for its services, subscribe to its topics.
    # ------------------------------------------------------------------
    rospy.init_node("evaluate_policy_points", anonymous=False)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(isinstance(h, RichHandler) for h in logger.handlers):
        logger.addHandler(RichHandler(console=console, rich_tracebacks=True, show_path=False))
    logger.info("ROS node initialized")

    ns = args.tracking_node_ns.rstrip("/")
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
    expected_dim = env_feature.shape[0]
    if expected_dim != n_points * 3:
        raise ValueError(
            f"Policy expects observation.environment_state of dim {expected_dim}, but "
            f"tracking_node reports {n_points} point(s) ({n_points * 3} dims). "
            "tracking_node.py was likely started with a --task_name that doesn't match "
            "the task this policy was trained on."
        )
    logger.info(f"tracking_node reports {n_points} tracked point(s) -- matches policy.")

    try:
        rospy.wait_for_service(f"{ns}/reset_and_track", timeout=15.0)
        rospy.wait_for_service(f"{ns}/stop_track", timeout=15.0)
    except rospy.ROSException as e:
        raise RuntimeError(f"tracking_node services not available: {e}")
    reset_and_track_srv = rospy.ServiceProxy(f"{ns}/reset_and_track", Trigger)
    stop_track_srv = rospy.ServiceProxy(f"{ns}/stop_track", Trigger)

    env_state_sub = EnvironmentStateSubscriber(f"{ns}/environment_state")

    # ------------------------------------------------------------------
    # Initialize hardware
    # ------------------------------------------------------------------
    arm = CompliantController(gripper_type=None)
    if controller_cfg is not None:
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
    if args.use_gripper:
        logger.info("Initializing ClawController...")
        claw = ClawController(init_node=False)

    bendlabs = None
    if bendlab_slowdown_enabled:
        logger.info(f"Initializing BendLabsRecorder ({num_bendlabs_sensors} sensors)...")
        bendlabs = BendLabsRecorder(init_node=False, num_sensors=num_bendlabs_sensors)
        bendlabs_deadline = time.perf_counter() + 10.0
        while not bendlabs.sensors_ready() and not rospy.is_shutdown():
            if time.perf_counter() > bendlabs_deadline:
                logger.warning(
                    "Timed out waiting for BendLabs sensors -- is uart_bridge_node running? "
                    "Disabling bend-sensor slowdown for this run."
                )
                bendlabs = None
                break
            rospy.sleep(0.1)
        if bendlabs is not None:
            logger.info("BendLabs sensors ready.")

    np.set_printoptions(linewidth=np.inf, formatter={"float": lambda x: f"{x:0.3f}"})

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    num_rollouts = args.num_rollouts
    max_timesteps = args.max_timesteps
    step_duration_s = 1.0 / fps
    total_steps_per_episode = []
    eval_start_time = timeit.default_timer()
    total_steps_completed = 0

    logger.info(f"Control frequency: {fps} Hz | step budget: {step_duration_s * 1000:.1f} ms")

    def make_step_description(rollout_id: int, steps: int) -> str:
        elapsed = timeit.default_timer() - eval_start_time
        actual_fps = steps / elapsed if elapsed > 0 else 0.0
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
                step_task = progress.add_task(
                    make_step_description(rollout_id, total_steps_completed), total=max_timesteps,
                )

                progress.stop()
                move_to_home(arm, hydra_cfg, claw)
                input(f"\n  Reset scene to start position, then press Enter to start rollout "
                      f"{rollout_id + 1}/{num_rollouts}...")
                progress.start()

                arm.zero_ft_sensor()
                arm.activate_cartesian_controller()

                # Ask tracking_node to re-run DIFT localization on the current
                # frames and (re)start publishing tracked points -- the object
                # may have moved between episodes, same reason
                # points_class.reset_episode() ran every rollout before. Lets
                # the user retry as many times as needed (e.g. after checking
                # tracking_node's ~visualization_image in rqt_image_view and
                # deciding the tracked point looks wrong) before committing to
                # running the policy.
                progress.stop()
                skip_rollout = False
                while True:
                    try:
                        resp = reset_and_track_srv()
                    except rospy.ServiceException as e:
                        resp = None
                        logger.warning(f"reset_and_track call failed: {e}")

                    if resp is None or not resp.success:
                        logger.warning(
                            "tracking_node localization failed"
                            + (f" ({resp.message})" if resp is not None else "") + "."
                        )
                        choice = input(
                            "  Press Enter to retry localization, or 's' to skip this rollout: "
                        ).strip().lower()
                        if choice == "s":
                            skip_rollout = True
                            break
                        continue

                    env_seq_before_reset = env_state_sub.seq
                    choice = input(
                        "  Tracking (re)initialized -- check "
                        f"{ns}/visualization_image in rqt_image_view to verify the tracked "
                        "point looks right. Press Enter to proceed with this rollout, "
                        "'r' to reset tracking again, or 's' to skip this rollout: "
                    ).strip().lower()
                    if choice == "r":
                        continue
                    if choice == "s":
                        skip_rollout = True
                    break
                progress.start()

                if skip_rollout:
                    logger.warning(f"Rollout {rollout_id + 1}: skipped by user.")
                    stop_track_srv()
                    progress.remove_task(step_task)
                    progress.update(rollout_task, advance=1)
                    continue

                policy.reset()

                # Block here (not inside the timed per-step loop) until
                # tracking_node has published at least one real estimate
                # since our reset_and_track call -- normally near-instant,
                # only slow on tracking_node's very first-ever tracking tick
                # (one-time cuDNN warm-up, see tracking_node.py's TrackingNode
                # docstring). Absorbing that here means it shows up as this
                # one log line instead of a string of "stale" warnings during
                # the timed rollout.
                progress.stop()
                logger.info("Waiting for tracking_node's first estimate (only slow the very first time)...")
                warmup_deadline = time.perf_counter() + 10.0
                while (env_state_sub.seq <= env_seq_before_reset
                       and time.perf_counter() < warmup_deadline
                       and not rospy.is_shutdown()):
                    time.sleep(0.05)
                if env_state_sub.seq <= env_seq_before_reset:
                    logger.warning(
                        "tracking_node produced no estimate within 10s; starting rollout anyway "
                        "(the staleness check will hold position until it catches up)."
                    )
                progress.start()

                last_stale_warn_time = 0.0
                # Reset against the gripper_home value move_to_home just
                # actually commanded, so t=0 doesn't spuriously trigger (or
                # fail to trigger) a slowdown against a stale prior rollout's
                # last command.
                prev_gripper_norm = float(np.clip(
                    hydra_cfg.dataset.get("gripper_home", 0.0) if hydra_cfg is not None else 0.0,
                    0.0, 1.0))
                slowdown_until = -1.0
                bendlab_baseline = bendlabs.get_angles().copy() if bendlabs is not None else None
                was_slowing = False

                for t in range(max_timesteps):
                    step_start = time.perf_counter()

                    bendlab_triggered = False
                    if bendlabs is not None:
                        bendlab_deviation = float(np.max(np.abs(bendlabs.get_angles() - bendlab_baseline)))
                        bendlab_triggered = bendlab_deviation > bendlab_slowdown_threshold

                    env_state, env_age_s = env_state_sub.get()
                    state = get_current_state(arm, claw)

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
                        policy_obs = preprocessor(obs)

                        with torch.inference_mode():
                            action = policy.select_action(policy_obs)
                        action = postprocessor(action).squeeze().cpu().numpy()  # (10,)

                        target_pose = np.concatenate(
                            [action[:3], transformations.quaternion_from_ortho6(action[3:9])]
                        )
                        gripper_norm = gripper_state_to_normalized(action[9])

                        if claw is not None and abs(gripper_norm - prev_gripper_norm) > gripper_slowdown_change_threshold:
                            if gripper_slowdown_enabled:
                                slowdown_until = step_start + gripper_slowdown_duration_s
                        prev_gripper_norm = gripper_norm

                        execute_cartesian_target(target_pose, arm, claw, gripper_norm, safety_cfg)

                    dt_s = time.perf_counter() - step_start
                    target_step_duration = step_duration_s
                    slowdown_reasons = []
                    if step_start < slowdown_until:
                        target_step_duration = max(target_step_duration, step_duration_s * gripper_slowdown_factor)
                        slowdown_reasons.append("gripper")
                    if bendlab_triggered:
                        target_step_duration = max(target_step_duration, step_duration_s * bendlab_slowdown_factor)
                        slowdown_reasons.append("bendlab")

                    is_slowing = target_step_duration > step_duration_s
                    if is_slowing and not was_slowing:
                        logger.info(
                            f"t={t:03d}  Slowing down execution ({'+'.join(slowdown_reasons)}, "
                            f"{target_step_duration / step_duration_s:.1f}x)"
                        )
                    elif was_slowing and not is_slowing:
                        logger.info(f"t={t:03d}  Resuming normal speed")
                    was_slowing = is_slowing

                    sleep_s = target_step_duration - dt_s
                    if sleep_s < 0:
                        logger.warning(
                            f"Step {t} overran budget: {dt_s * 1000:.1f} ms > {target_step_duration * 1000:.1f} ms "
                            f"({1 / dt_s:.1f} Hz actual vs {1.0 / target_step_duration:.1f} Hz target)."
                        )
                    precise_sleep(max(sleep_s, 0.0))

                    total_steps_completed += 1
                    progress.update(
                        step_task, advance=1,
                        description=make_step_description(rollout_id, total_steps_completed),
                    )

                stop_track_srv()

                arm.activate_joint_trajectory_controller()
                steps_taken = t + 1
                total_steps_per_episode.append(steps_taken)
                logger.info(f"Rollout {rollout_id} complete | steps: {steps_taken}")

                progress.remove_task(step_task)
                progress.update(rollout_task, advance=1)
    finally:
        try:
            stop_track_srv()
        except Exception:
            pass  # best-effort -- don't mask an earlier exception with a cleanup failure

    logger.info("=" * 60)
    logger.info(f"Rollouts completed:        {num_rollouts}")
    logger.info(f"Mean steps per episode:    {np.mean(total_steps_per_episode):.1f}")
    logger.info(f"Std  steps per episode:    {np.std(total_steps_per_episode):.1f}")
    logger.info("=" * 60)
    logger.info(f"Evaluation complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
