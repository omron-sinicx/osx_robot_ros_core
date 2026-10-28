#!/usr/bin/env python3
"""Diagnostic companion to evaluate_policy_points.py: visualize the ACT
policy's *raw predicted action chunk* during a live rollout, to check whether
it is degenerate (e.g. near-constant across the prediction horizon instead of
tracking the live observation).

Runs the exact same connect-to-tracking_node + home + reset_and_track + step
loop as evaluate_policy_points.py, but instead of only consuming one action
per control step from the policy's internal queue, it captures the *entire*
chunk every time the policy actually runs inference (i.e. every
config.n_action_steps ticks, when select_action() would otherwise silently
refill its queue -- or every tick if config.temporal_ensemble_coeff is set,
since ACT reruns inference every step in that mode). Saves, after every
newly-captured chunk:
    <output_dir>/action_chunks.png  -- one subplot per action dimension
        (pos_x, pos_y, pos_z, ortho6_0..5, gripper), x-axis = steps ahead
        within the chunk, one line per captured chunk, colored by capture
        order (viridis, dark=later) so drift across the rollout is visible.
        A flat, overlapping bundle of lines is the "constant chunk" symptom.
    <output_dir>/action_chunks.npz -- every postprocessed chunk (raw arrays)
        for offline analysis, plus per-chunk per-dim std (near-zero std means
        that dimension barely moves across the whole predicted horizon).

Requires tracking_node.py to already be running for the matching task, same
as evaluate_policy_points.py:
    rosrun osx_ur5e tracking_node.py --task_name pick_place_00

Usage:
    python visualize_action_chunk.py --policy_path outputs/train/act_pick_place_00

By default this also drives the real arm/gripper (identical safety clipping
to evaluate_policy_points.py) so the captured chunks reflect an actual
rollout. Pass --no_execute to only home the arm once and then hold still
while capturing chunks against the live (but not policy-driven) observation
stream, if you'd rather not move the robot.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import matplotlib
matplotlib.use("Agg")  # must be before any pyplot import
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import rospy
from std_srvs.srv import Trigger

from lerobot.utils.robot_utils import precise_sleep
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig

import lerobot_policy_custom_act  # noqa: F401 -- registers "custom_act" with draccus

from osx_claw.claw_controller import ClawController
from osx_ur5e import pose_offset
from ur_control.fzi_cartesian_compliance_controller import CompliantController
from ur_control import transformations

# Reuse setup/execution helpers from evaluate_policy_points.py verbatim, so
# this diagnostic can't silently drift from the real eval pipeline's
# observation encoding / safety clipping / homing behavior.
from evaluate_policy_points import (  # noqa: E402
    GRIPPER_CLOSED_NORM,
    GRIPPER_OPEN_NORM,
    EnvironmentStateSubscriber,
    GraspConfig,
    _signal_handler,
    execute_cartesian_target,
    get_current_state,
    load_hydra_config,
    move_to_home,
    move_to_pose_via_cartesian_control,
    setup_logging,
)
import signal

signal.signal(signal.SIGINT, _signal_handler)

logger = logging.getLogger(__name__)

ACTION_DIM_LABELS = [
    "pos_x", "pos_y", "pos_z",
    "ortho6_0", "ortho6_1", "ortho6_2", "ortho6_3", "ortho6_4", "ortho6_5",
    "gripper",
]


def parse_args():
    argv = rospy.myargv(argv=sys.argv)[1:]
    parser = argparse.ArgumentParser(
        description="Visualize an ACT policy's predicted action chunks during a live rollout")
    parser.add_argument("--policy_path", type=str, required=True)
    parser.add_argument("--tracking_node_ns", type=str, default="tracking_node")
    parser.add_argument("--num_rollouts", type=int, default=1)
    parser.add_argument("--max_timesteps", type=int, default=300)
    parser.add_argument("--fps", type=int, default=None,
                        help="Control frequency in Hz. Falls back to dataset.fps in the hydra "
                             "config, then 30.")
    parser.add_argument("--hydra_config_path", type=str, default="../config/hydra")
    parser.add_argument("--hydra_config_name", type=str, default="test_task")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_tracking_staleness_s", type=float, default=1.0)
    parser.add_argument("--use_gripper", action="store_true", default=True)
    parser.add_argument("--no_execute", action="store_true",
                        help="Never command the arm/claw from the policy's output -- home once, "
                             "then only capture/plot chunks against the live observation stream.")
    parser.add_argument("--max_chunks_plotted", type=int, default=12,
                        help="Only the most recent N captured chunks are drawn (older ones fade "
                             "out of the plot but stay in the .npz).")
    return parser.parse_args(argv)


def plot_chunks(chunks, output_path, max_chunks_plotted):
    """chunks: list of (n_action_steps, action_dim) postprocessed numpy arrays,
    oldest first. Overwrites output_path each call so it can be watched live."""
    if not chunks:
        return
    shown = chunks[-max_chunks_plotted:]
    n_dims = shown[0].shape[1]
    cols = 2
    rows = (n_dims + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.3 * rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=max(len(shown) - 1, 1))

    for d in range(n_dims):
        ax = axes[d]
        for i, chunk in enumerate(shown):
            ax.plot(chunk[:, d], color=cmap(norm(i)), linewidth=1.2, alpha=0.85)
        ax.set_title(ACTION_DIM_LABELS[d], fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, linewidth=0.4, alpha=0.3)
    for d in range(n_dims, len(axes)):
        axes[d].axis("off")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.6, pad=0.02)
    cbar.set_label("chunk capture order (older -> newer)", fontsize=8)

    fig.suptitle(
        "Predicted action chunk per dimension  (x-axis = steps ahead within chunk)", fontsize=10)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def summarize_chunk(chunk):
    """Per-dim std across the chunk horizon -- near-zero std means the policy
    predicted (almost) the same value for every step ahead, i.e. a constant
    chunk for that dimension."""
    stds = chunk.std(axis=0)
    return {label: round(float(s), 5) for label, s in zip(ACTION_DIM_LABELS, stds)}


def postprocess_chunk(raw_chunk_t, postprocessor):
    """raw_chunk_t: (1, n_action_steps, action_dim) tensor straight out of
    predict_action_chunk. Returns (n_action_steps, action_dim) numpy array,
    run one step at a time through postprocessor since it expects a
    (batch, action_dim) tensor (matching how select_action feeds it one
    popped action at a time)."""
    n_steps = raw_chunk_t.shape[1]
    steps = [
        postprocessor(raw_chunk_t[:, k]).squeeze(0).cpu().numpy()
        for k in range(n_steps)
    ]
    return np.stack(steps, axis=0)


def main():
    args = parse_args()

    policy_path = Path(args.policy_path).resolve()
    if not policy_path.is_dir():
        raise FileNotFoundError(f"Policy checkpoint directory not found: {policy_path}")
    if not (policy_path / "config.json").exists():
        raise FileNotFoundError(f"{policy_path} exists but has no config.json")

    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs/eval") / policy_path.name / "action_chunk_diag"
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "visualize_action_chunk.log")
    logger.info(f"Results will be saved to: {output_dir}")

    hydra_cfg = None
    try:
        hydra_cfg = load_hydra_config(args.hydra_config_path, args.hydra_config_name)
        logger.info(f"Loaded hydra config '{args.hydra_config_name}'")
    except Exception as e:
        logger.warning(f"Could not load hydra config '{args.hydra_config_name}': {e}")

    dataset_fps = int(hydra_cfg.dataset.fps) if hydra_cfg is not None else None
    controller_cfg = hydra_cfg.controller if hydra_cfg is not None else None
    safety_cfg = controller_cfg.safety_parameters if controller_cfg is not None else None
    ds_cfg = hydra_cfg.dataset if hydra_cfg is not None else None

    if args.fps is not None:
        fps = args.fps
    elif dataset_fps is not None:
        fps = dataset_fps
    else:
        fps = 30
    step_duration_s = 1.0 / fps

    eval_offset = pose_offset.resolve_offset(ds_cfg)
    grasp_binary_enabled = bool(ds_cfg.get("grasp_binary_enabled", True)) if ds_cfg is not None else True
    grasp_cmd_close_threshold = float(ds_cfg.get("grasp_cmd_close_threshold", 0.5)) if ds_cfg is not None else 0.5
    grasp_gap_threshold = float(ds_cfg.get("grasp_gap_threshold", 0.15)) if ds_cfg is not None else 0.15
    grasp_cfg = GraspConfig(grasp_binary_enabled, grasp_cmd_close_threshold, grasp_gap_threshold)

    # Force the gripper OPEN or CLOSED at the start of each rollout, instead of
    # move_to_home's usual dataset.gripper_home value -- same start_gripper
    # ritual as evaluate_policy_points.py (see its docstring/main() for the
    # full rationale): "closed" eases in with the gripper OPEN, pauses for you
    # to insert an object, then closes it before the rollout starts; every
    # rollout after the first instead retracts reset_lift_distance_m straight
    # up (assuming the object is still held) and homes without touching the
    # gripper.
    start_gripper = ds_cfg.get("start_gripper", "auto") if ds_cfg is not None else "auto"
    if start_gripper not in ("auto", "open", "closed"):
        raise ValueError(f"Invalid dataset.start_gripper: {start_gripper!r} (expected 'auto', 'open', or 'closed')")
    reset_lift_distance_m = float(ds_cfg.get("reset_lift_distance_m", 0.05)) if ds_cfg is not None else 0.05

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
    logger.info(f"chunk_size={config.chunk_size}  n_action_steps={config.n_action_steps}  "
                f"temporal_ensemble_coeff={config.temporal_ensemble_coeff}")

    env_feature = config.input_features.get("observation.environment_state")
    if env_feature is None:
        raise ValueError("Policy has no observation.environment_state input feature.")

    # ------------------------------------------------------------------
    # Connect to tracking_node.py
    # ------------------------------------------------------------------
    rospy.init_node("visualize_action_chunk", anonymous=False)
    logger.setLevel(logging.INFO)

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
        raise RuntimeError(f"Timed out waiting for rosparam {num_points_param} -- is tracking_node.py running?")
    expected_dim = env_feature.shape[0]
    if expected_dim != n_points * 3:
        raise ValueError(
            f"Policy expects observation.environment_state of dim {expected_dim}, but "
            f"tracking_node reports {n_points} point(s) ({n_points * 3} dims).")
    logger.info(f"tracking_node reports {n_points} tracked point(s) -- matches policy.")

    rospy.wait_for_service(f"{ns}/reset_and_track", timeout=15.0)
    rospy.wait_for_service(f"{ns}/stop_track", timeout=15.0)
    reset_and_track_srv = rospy.ServiceProxy(f"{ns}/reset_and_track", Trigger)
    stop_track_srv = rospy.ServiceProxy(f"{ns}/stop_track", Trigger)
    env_state_sub = EnvironmentStateSubscriber(f"{ns}/environment_state")

    # ------------------------------------------------------------------
    # Initialize hardware (arm always, so we can read state; claw only if
    # requested; both identical to evaluate_policy_points.py's setup so the
    # captured chunks are commanded/read exactly the same way).
    # ------------------------------------------------------------------
    arm = CompliantController(gripper_type=None)
    if controller_cfg is not None:
        arm.set_control_mode(controller_cfg.mode)
        arm.update_pd_gains(
            OmegaConf.to_container(controller_cfg.p_gains),
            OmegaConf.to_container(controller_cfg.d_gains),
        )
        arm.update_selection_matrix(OmegaConf.to_container(controller_cfg.selection_matrix))
        arm.set_solver_parameters(error_scale=controller_cfg.error_scale, iterations=controller_cfg.iterations)
        arm.current_stiffness = np.array(
            [controller_cfg.stiffness_trans] * 3 + [controller_cfg.stiffness_rot] * 3)
        arm.update_stiffness(np.diag(arm.current_stiffness))
        arm.auto_switch_controllers = False
        arm.async_mode = True
    arm.zero_ft_sensor()

    claw = None
    if args.use_gripper:
        logger.info("Initializing ClawController...")
        claw = ClawController(init_node=False)

    np.set_printoptions(linewidth=np.inf, formatter={"float": lambda x: f"{x:0.3f}"})

    chunk_log = []  # postprocessed (n_action_steps, action_dim) arrays, oldest first
    chunk_meta = []  # (rollout_id, t) for each captured chunk, same order as chunk_log

    def capture_chunk(policy_obs, rollout_id, t):
        with torch.inference_mode():
            raw_chunk_t = policy.predict_action_chunk(policy_obs)[:, : policy.config.n_action_steps]
        chunk = postprocess_chunk(raw_chunk_t, postprocessor)
        chunk_log.append(chunk)
        chunk_meta.append((rollout_id, t))
        stds = summarize_chunk(chunk)
        logger.info(f"t={t:03d}  new chunk captured (n={chunk.shape[0]})  per-dim std: {stds}")
        plot_chunks(chunk_log, output_dir / "action_chunks.png", args.max_chunks_plotted)
        np.savez(
            output_dir / "action_chunks.npz",
            chunks=np.stack(chunk_log),
            rollout_id=np.array([m[0] for m in chunk_meta]),
            t=np.array([m[1] for m in chunk_meta]),
            dim_labels=np.array(ACTION_DIM_LABELS),
        )
        return raw_chunk_t

    ensembling = config.temporal_ensemble_coeff is not None

    try:
        for rollout_id in range(args.num_rollouts):
            if start_gripper == "closed" and rollout_id > 0 and not args.no_execute:
                # Not the first rollout -- the object is presumably still
                # held from the previous attempt, so skip the open/
                # insert-by-hand/close ritual below and instead retract
                # straight up to disengage, then home without touching the
                # gripper (still closed, still holding the object).
                logger.info(
                    f"Retracting {reset_lift_distance_m * 100:.0f} cm straight up "
                    "before resetting to home..."
                )
                arm.zero_ft_sensor()
                arm.activate_cartesian_controller()
                lift_pose = arm.end_effector().copy()
                lift_pose[2] += reset_lift_distance_m
                hold_gripper_norm = claw.get_normalized_position() if claw is not None else GRIPPER_CLOSED_NORM
                move_to_pose_via_cartesian_control(lift_pose, arm, claw, hold_gripper_norm, safety_cfg, fps)
                move_to_home(arm, hydra_cfg, gripper=None)
                input(f"\n  Reset. Press Enter to start rollout {rollout_id + 1}/{args.num_rollouts}...")
            else:
                move_to_home(arm, hydra_cfg, claw if not args.no_execute else None)

                if claw is not None and not args.no_execute and start_gripper != "auto":
                    # Override move_to_home's gripper_home command -- both
                    # "open" and "closed" start the reset phase with the
                    # gripper OPEN (for "closed", so an object can be
                    # inserted by hand next).
                    claw.set_normalized_position(GRIPPER_OPEN_NORM)

                if start_gripper == "closed":
                    input("\n  At home pose — insert the object into the gripper, then press Enter to close...")
                    if claw is not None and not args.no_execute:
                        claw.set_normalized_position(GRIPPER_CLOSED_NORM)
                        time.sleep(1.0)  # let the claw finish closing before the arm moves
                    input(f"\n  Closed. Press Enter to start rollout {rollout_id + 1}/{args.num_rollouts}...")
                else:
                    input(f"\n  Reset scene to start position, then press Enter to start rollout "
                          f"{rollout_id + 1}/{args.num_rollouts}...")

            arm.zero_ft_sensor()
            if not args.no_execute:
                arm.activate_cartesian_controller()

            skip_rollout = False
            while True:
                try:
                    resp = reset_and_track_srv()
                except rospy.ServiceException as e:
                    resp = None
                    logger.warning(f"reset_and_track call failed: {e}")
                if resp is None or not resp.success:
                    choice = input(
                        "  tracking_node localization failed. Press Enter to retry, or 's' to skip: "
                    ).strip().lower()
                    if choice == "s":
                        skip_rollout = True
                        break
                    continue
                env_seq_before_reset = env_state_sub.seq
                choice = input(
                    "  Tracking (re)initialized -- check "
                    f"{ns}/visualization_image to verify the tracked point. Press Enter to "
                    "proceed, 'r' to reset tracking again, or 's' to skip: "
                ).strip().lower()
                if choice == "r":
                    continue
                if choice == "s":
                    skip_rollout = True
                break

            if skip_rollout:
                logger.warning(f"Rollout {rollout_id + 1}: skipped by user.")
                stop_track_srv()
                continue

            policy.reset()
            if not ensembling:
                # For the non-ensembling path we manage the queue ourselves
                # below so we can intercept the full chunk on every refill.
                policy._action_queue.clear()

            logger.info("Waiting for tracking_node's first estimate...")
            warmup_deadline = time.perf_counter() + 10.0
            while (env_state_sub.seq <= env_seq_before_reset
                   and time.perf_counter() < warmup_deadline
                   and not rospy.is_shutdown()):
                time.sleep(0.05)

            for t in range(args.max_timesteps):
                step_start = time.perf_counter()

                env_state, env_age_s = env_state_sub.get()
                state = get_current_state(arm, claw, grasp_cfg, offset=eval_offset)

                if env_age_s > args.max_tracking_staleness_s:
                    logger.warning(f"t={t:03d}  tracked point is {env_age_s:.2f}s stale -- holding position.")
                else:
                    obs = {
                        "observation.state": torch.tensor(state, dtype=torch.float32),
                        "observation.environment_state": torch.tensor(env_state, dtype=torch.float32),
                    }
                    policy_obs = preprocessor(obs)

                    if ensembling:
                        # ACT reruns full inference every step in this mode --
                        # capture it directly, then let select_action's own
                        # (separate, identical-input) call drive the temporal
                        # ensembler for the actually-executed action.
                        capture_chunk(policy_obs, rollout_id, t)
                        with torch.inference_mode():
                            action = policy.select_action(policy_obs)
                    else:
                        if len(policy._action_queue) == 0:
                            raw_chunk_t = capture_chunk(policy_obs, rollout_id, t)
                            policy._action_queue.extend(raw_chunk_t.transpose(0, 1))
                        action = policy._action_queue.popleft()

                    action = postprocessor(action).squeeze().cpu().numpy()  # (10,)

                    if not args.no_execute:
                        target_pose = np.concatenate(
                            [action[:3], transformations.quaternion_from_ortho6(action[3:9])]
                        )
                        target_pose = pose_offset.apply_offset(target_pose, eval_offset)
                        gripper_norm = float(np.clip((1.0 - action[9]) / 2.0, 0.0, 1.0))
                        execute_cartesian_target(target_pose, arm, claw, gripper_norm, safety_cfg)

                dt_s = time.perf_counter() - step_start
                precise_sleep(max(step_duration_s - dt_s, 0.0))

            stop_track_srv()
            if not args.no_execute:
                arm.activate_joint_trajectory_controller()
            logger.info(f"Rollout {rollout_id + 1} complete | chunks captured so far: {len(chunk_log)}")
    finally:
        try:
            stop_track_srv()
        except Exception:
            pass

    logger.info(f"Done. {len(chunk_log)} chunk(s) captured -- see {output_dir}/action_chunks.png "
                f"and {output_dir}/action_chunks.npz")


if __name__ == "__main__":
    main()
