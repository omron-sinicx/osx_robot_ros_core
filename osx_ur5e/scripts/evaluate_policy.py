#!/usr/bin/env python3
"""Evaluate a LeRobot ACT policy on the real UR5e with custom gripper.

The policy expects:
    observation.state       — shape (7,): 6 arm joint angles + 1 gripper position
    observation.ft          — shape (6,): force-torque wrench
    observation.images.*    — camera images (uint8, CHW)
    action                  — shape (7,): 6 arm joint targets + 1 gripper target

Usage:
    python evaluate_policy.py \
        --policy_path outputs/train/act_test_task \
        --num_rollouts 5 --max_timesteps 300

    # Override the control frequency instead of reading it from the hydra config:
    python evaluate_policy.py \
        --policy_path outputs/train/act_test_task --fps 25

Controls during each rollout:
    Enter  - confirm start of rollout (after reset prompt)
"""

from lerobot.utils.robot_utils import precise_sleep
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from bendlabs.bendlabs_recorder import BendLabsRecorder
from osx_claw.claw_controller import ClawController
from osx_gym_env.utils import ImageRecorder
from ur_control.fzi_cartesian_compliance_controller import CompliantController
from ur_control import transformations
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.logging import RichHandler
from rich.console import Console
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize_config_dir
import rospy
import matplotlib.pyplot as plt
import argparse
import logging
import signal
import sys
import time
import timeit
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")


import lerobot_policy_custom_act  # noqa: F401 — registers "custom_act" with draccus

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
    root.addHandler(RichHandler(console=console,
                    rich_tracebacks=True, show_path=False))
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"))
    root.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Hydra config helpers
# ---------------------------------------------------------------------------

def load_hydra_config(config_path: str, config_name: str):
    """Load the Hydra config used by data_collection.py and replay_episode.py."""
    config_dir_abs = str(
        (Path(__file__).resolve().parent / config_path).resolve())

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=config_dir_abs, version_base=None):
        cfg = compose(config_name=config_name)

    return cfg


# ---------------------------------------------------------------------------
# Observation formatting
# ---------------------------------------------------------------------------

def format_real_robot_observations(
    arm,
    image_recorder,
    claw,
    features: dict,
    camera_shape: tuple,
    bendlabs=None,
) -> dict:
    """Build a policy-ready observation dict matching the ACT training format.

    Mirrors get_observations() in data_collection.py exactly:
        observation.state    — 6 arm joint angles + 1 gripper normalized position
        observation.ft       — 6-dim force/torque wrench
        observation.bendlabs — flat BendLabs angles [twist0, bend0, ...]
        observation.images.<cam> — uint8 CHW tensors resized to camera_shape

    Args:
        arm: CompliantController instance.
        image_recorder: ImageRecorder instance (or None).
        claw: ClawController instance (or None if no gripper).
        features: Feature keys from the policy checkpoint (used to filter keys).
        camera_shape: (H, W) to resize images to match training resolution.
        bendlabs: BendLabsRecorder instance (or None).

    Returns:
        Dict of torch tensors ready for policy.select_action().
    """
    arm_qpos = np.array(arm.joint_angles())  # shape (6,)
    # gripper_pos = np.array([claw.get_normalized_position()]
    #                        ) if claw is not None else np.array([0.0])
    gripper_pos = np.array([1.0])  # temporary fix until data is recollected

    state = np.concatenate([arm_qpos, gripper_pos])  # shape (7,)

    obs = {}

    if "observation.state" in features:
        obs["observation.state"] = torch.tensor(state, dtype=torch.float32)
    if "observation.ft" in features:
        obs["observation.ft"] = torch.tensor(
            np.array(arm.get_wrench()), dtype=torch.float32)

    if bendlabs is not None and "observation.bendlabs" in features:
        obs["observation.bendlabs"] = torch.tensor(
            bendlabs.get_angles(), dtype=torch.float32)

    if image_recorder is not None:
        resize_transform = transforms.Resize(camera_shape, antialias=True)
        raw_images = image_recorder.get_images()
        for cam_name, image_hwc in raw_images.items():
            feat_key = f"observation.images.{cam_name}"
            if feat_key in features:
                image_chw = np.ascontiguousarray(
                    np.transpose(image_hwc, (2, 0, 1)))
                obs[feat_key] = resize_transform(torch.tensor(
                    image_chw, dtype=torch.float32))

    return obs


def move_to_home(arm, cfg, claw=None) -> None:
    """Drive the arm (and gripper) to the fixed home configuration via the joint
    trajectory controller, before switching to cartesian control for the rollout.
    """
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

# ---------------------------------------------------------------------------
# Action execution
# ---------------------------------------------------------------------------


def execute_cartesian_action(action_tensor: torch.Tensor, arm, claw, safety_cfg) -> None:
    """Execute a 7-dim ACT action using cartesian compliance control.

    Mirrors set_action() in data_collection.py exactly:
      1. Interpret action[:6] as target joint angles (same as GELLO joints during collection).
      2. Compute FK to get the target EEF pose.
      3. Compute a delta from the current EEF pose, clip it for safety.
      4. Send the clipped cartesian target to the compliance controller.

    Args:
        action_tensor: Shape (7,) or (1, 7) torch tensor from policy.select_action().
        arm: CompliantController instance (must be in cartesian controller mode).
        claw: ClawController instance (or None).
        safety_cfg: OmegaConf DictConfig with max_delta_translation, max_delta_rotation,
                    and workspace_range fields (from controller.safety_parameters).
    """
    action = action_tensor.squeeze().cpu().numpy()  # shape (7,)

    current_pose = arm.end_effector()
    target_pose = arm.end_effector(joint_angles=action[:6])

    delta_translation = target_pose[:3] - current_pose[:3]
    delta_rotation = transformations.quaternions_orientation_error(
        target_pose[3:], current_pose[3:])

    max_delta_rotation = np.deg2rad(safety_cfg.max_delta_rotation)
    clipped_translation = np.clip(delta_translation,
                                  -safety_cfg.max_delta_translation,
                                  safety_cfg.max_delta_translation)
    clipped_rotation = np.clip(
        delta_rotation, -max_delta_rotation, max_delta_rotation)

    next_pos = current_pose[:3] + clipped_translation
    next_pos[0] = np.clip(
        next_pos[0], safety_cfg.workspace_range.x[0], safety_cfg.workspace_range.x[1])
    next_pos[1] = np.clip(
        next_pos[1], safety_cfg.workspace_range.y[0], safety_cfg.workspace_range.y[1])
    next_pos[2] = np.clip(
        next_pos[2], safety_cfg.workspace_range.z[0], safety_cfg.workspace_range.z[1])
    next_orient = transformations.rotate_quaternion_by_rpy(
        *clipped_rotation, current_pose[3:])
    next_target = np.concatenate([next_pos, next_orient])

    arm.set_cartesian_target_pose(pose=next_target)

    if claw is not None:
        claw.set_normalized_position(float(np.clip(action[6], 0.0, 1.0)))


# ---------------------------------------------------------------------------
# Chunk visualization
# ---------------------------------------------------------------------------

_JOINT_LABELS = ["J1", "J2", "J3", "J4", "J5", "J6", "Gripper"]
_CHUNK_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628",
                 "#f781bf", "#999999", "#8dd3c7", "#ffffb3"]


def save_chunk_visualization(
    chunk_log: list,
    actual_log: list,
    out_path: Path,
    policy_name: str,
    chunk_size: int,
) -> None:
    """Plot all predicted action chunks + actual executed joint positions and save to PNG.

    Args:
        chunk_log: List of dicts {t: int, chunk: np.ndarray (chunk_size, 7)}.
        actual_log: List of dicts {t: int, arm_q: np.ndarray (6,), gripper: float, action: np.ndarray (7,)}.
        out_path: Where to write the PNG.
        policy_name: Label shown in the figure title.
        chunk_size: Number of steps per chunk (x-axis length for each prediction).
    """
    n_dims = 7
    n_cols = 2
    n_rows = (n_dims + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    axes = axes.flatten()

    # Actual robot state over the rollout (arm joints + gripper)
    if actual_log:
        actual_t = np.array([e["t"] for e in actual_log])
        actual_arm = np.array([e["arm_q"] for e in actual_log])       # (T, 6)
        actual_grip = np.array([e["gripper"] for e in actual_log])     # (T,)
        actual_action = np.array([e["action"] for e in actual_log])    # (T, 7)

    for dim in range(n_dims):
        ax = axes[dim]

        # Actual arm state
        if actual_log:
            if dim < 6:
                ax.plot(actual_t, actual_arm[:, dim], color="black",
                        linewidth=1.5, alpha=0.5, label="arm actual", zorder=3)
            else:
                ax.plot(actual_t, actual_grip, color="black",
                        linewidth=1.5, alpha=0.5, label="gripper actual", zorder=3)
            # Policy action actually executed
            ax.plot(actual_t, actual_action[:, dim], color="gray",
                    linewidth=1.0, alpha=0.4, linestyle="--", label="action sent", zorder=2)

        # Predicted chunks
        linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
        for i, entry in enumerate(chunk_log):
            t0 = entry["t"]
            chunk = entry["chunk"]          # (chunk_size, 7)
            chunk_t = t0 + np.arange(len(chunk))
            color = _CHUNK_COLORS[i % len(_CHUNK_COLORS)]
            ax.plot(chunk_t, chunk[:, dim],
                    color=color, linewidth=1.6, alpha=0.85,
                    linestyle=linestyles[i % len(linestyles)],
                    label=f"chunk t={t0}", zorder=4)
            ax.axvline(t0, color=color, linewidth=0.6,
                       linestyle=":", alpha=0.5)

        ax.set_title(_JOINT_LABELS[dim], fontsize=11, fontweight="bold")
        ax.set_xlabel("Rollout step")
        ax.set_ylabel("Value (rad / norm)")
        ax.legend(fontsize=6, loc="upper right", ncol=2)
        ax.grid(True, alpha=0.3)

    for dim in range(n_dims, len(axes)):
        axes[dim].set_visible(False)

    fig.suptitle(
        f"Live Chunk Visualization  |  policy={policy_name}  chunk_size={chunk_size}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    # rospy.myargv() strips ROS-internal args (__name:=, __log:=, remappings)
    # so argparse only sees our own --flags.
    # Usage via rosrun:  rosrun osx_ur5e evaluate_policy.py --policy_path <path> --fps 50
    argv = rospy.myargv(argv=sys.argv)[1:]
    parser = argparse.ArgumentParser(
        description="Evaluate a LeRobot ACT policy on the UR5e")
    parser.add_argument("--policy_path", type=str, required=True,
                        help="Path to the trained policy checkpoint directory")
    parser.add_argument("--num_rollouts", type=int, default=3)
    parser.add_argument("--max_timesteps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--use_gripper", action="store_true", default=True,
                        help="Whether a ClawController gripper is attached")
    parser.add_argument("--camera_names", nargs="+", default=["front_camera"],
                        help="Camera topic names (must match training)")
    parser.add_argument("--fps", type=int, default=None,
                        help="Control frequency in Hz (must match training dataset fps). "
                             "If not given, this is read from dataset.fps in the hydra "
                             "config specified by --hydra_config_path/--hydra_config_name "
                             "(the same config used by replay_episode.py).")
    parser.add_argument("--hydra_config_path", type=str, default="../config/hydra",
                        help="Path to the hydra config directory (relative to this file), "
                             "used to look up dataset.fps. Matches replay_episode.py's "
                             "@hydra.main config_path.")
    parser.add_argument("--hydra_config_name", type=str, default="test_task",
                        help="Hydra config name to compose when looking up dataset.fps. "
                             "Matches replay_episode.py's @hydra.main config_name.")
    parser.add_argument("--use_amp", action="store_true", default=False,
                        help="Enable automatic mixed precision (float16) for faster GPU inference")
    parser.add_argument("--num_bendlabs_sensors", type=int, default=None,
                        help="Number of BendLabs sensors to read (overrides dataset.num_bendlabs_sensors "
                             "from hydra config). Set to 0 to disable even if the config enables it.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save logs (default: outputs/eval/<policy_name>)")
    return parser.parse_args(argv)


def main():
    args = parse_args()

    policy_path = Path(args.policy_path)
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        "outputs/eval") / policy_path.name
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir / "evaluation.log")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Results will be saved to: {output_dir}")

    # ------------------------------------------------------------------
    # Load Hydra config (same as data_collection.py / replay_episode.py)
    # ------------------------------------------------------------------
    hydra_cfg = None
    try:
        hydra_cfg = load_hydra_config(
            args.hydra_config_path, args.hydra_config_name)
        logger.info(f"Loaded hydra config '{args.hydra_config_name}'")
    except Exception as e:
        logger.warning(
            f"Could not load hydra config '{args.hydra_config_name}': {e}")

    dataset_fps = int(hydra_cfg.dataset.fps) if hydra_cfg is not None else None
    controller_cfg = hydra_cfg.controller if hydra_cfg is not None else None
    safety_cfg = controller_cfg.safety_parameters if controller_cfg is not None else None

    if args.fps is not None:
        fps = args.fps
        if dataset_fps is not None and fps != dataset_fps:
            logger.warning(
                f"--fps={fps} differs from dataset.fps={dataset_fps}; "
                f"policy will run at {fps} Hz instead of {dataset_fps} Hz."
            )
    elif dataset_fps is not None:
        fps = dataset_fps
        logger.info(
            f"--fps not given; using dataset.fps={fps} from hydra config.")
    else:
        fps = 50
        logger.warning(f"Hydra config unavailable; falling back to fps={fps}.")

    # ------------------------------------------------------------------
    # Load LeRobot ACT policy
    # ------------------------------------------------------------------
    logger.info(f"Loading policy from: {policy_path}")
    # PreTrainedPolicy is abstract — load the config first to discover the
    # concrete type (e.g. "act"), then call from_pretrained on that class.
    config = PreTrainedConfig.from_pretrained(str(policy_path))
    policy_cls = get_policy_class(config.type)
    policy = policy_cls.from_pretrained(str(policy_path))
    policy.cuda()
    policy.eval()

    # Load pre/post-processors saved alongside the checkpoint.
    # Override device so the DeviceProcessorStep targets the actual eval device,
    # not whatever was saved at training time (matches lerobot_eval.py lines 537-546).
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )

    rename_step = preprocessor.steps[0]

    # Check what it's actually renaming
    print("Rename map:")
    # or rename_step.mapping / rename_step.obs_rename_map
    print(rename_step.rename_map)
    # Try all possible attribute names:
    print(vars(rename_step))
    normalizer_step = preprocessor.steps[3]  # NormalizerProcessorStep

    print("=== NORMALIZER STATS KEYS ===")
    # Try these attribute names depending on LeRobot version:
    if hasattr(normalizer_step, 'stats'):
        for key in normalizer_step.stats:
            s = normalizer_step.stats[key]
            print(
                f"  {key}: mean={s.get('mean', 'N/A')}, std={s.get('std', 'N/A')}")
    elif hasattr(normalizer_step, 'normalizer'):
        print(vars(normalizer_step.normalizer))
    else:
        print(vars(normalizer_step))

    print("=== PREPROCESSOR PIPELINE STEPS ===")
    for i, step in enumerate(preprocessor.steps):
        print(f"  [{i}] {type(step).__name__}")

    print("\n=== POSTPROCESSOR PIPELINE STEPS ===")
    for i, step in enumerate(postprocessor.steps):
        print(f"  [{i}] {type(step).__name__}")

    features = config.input_features | config.output_features
    logger.info(f"Input features:  {list(config.input_features.keys())}")
    logger.info(f"Output features: {list(config.output_features.keys())}")

    # Derive camera shape from the policy's image input feature (H, W)
    camera_shape = (240, 320)  # fallback
    for key, feat in config.input_features.items():
        if key.startswith("observation.images."):
            # feat.shape is (C, H, W) or (H, W, C) depending on training
            camera_shape = (feat.shape[-2], feat.shape[-1])
            break
    logger.info(f"Camera shape (H, W): {camera_shape}")

    # ------------------------------------------------------------------
    # Initialize hardware (same controller setup as data_collection.py)
    # ------------------------------------------------------------------
    rospy.init_node("evaluate_policy", anonymous=False)
    # rospy.init_node() rewires the root logger — re-attach our handlers
    # directly to the module logger with propagate=False so ROS can't swallow them.
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(isinstance(h, RichHandler) for h in logger.handlers):
        logger.addHandler(RichHandler(
            console=console, rich_tracebacks=True, show_path=False))
    logger.info("ROS node initialized")

    arm = CompliantController(gripper_type=None)
    if controller_cfg is not None:
        from omegaconf import OmegaConf
        arm.set_control_mode(controller_cfg.mode)
        arm.update_pd_gains(
            OmegaConf.to_container(controller_cfg.p_gains),
            OmegaConf.to_container(controller_cfg.d_gains),
        )
        arm.update_selection_matrix(
            OmegaConf.to_container(controller_cfg.selection_matrix))
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

    image_recorder = None
    if args.camera_names:
        image_recorder = ImageRecorder(
            init_node=False, camera_names=args.camera_names)
        logger.info("Waiting for cameras...")
        deadline = time.perf_counter() + 10.0
        while not image_recorder.cameras_ready() and not rospy.is_shutdown():
            if time.perf_counter() > deadline:
                logger.error("Timed out waiting for cameras.")
                sys.exit(1)
            rospy.sleep(0.1)
        logger.info("Cameras ready.")

    bendlabs = None
    ds_cfg = hydra_cfg.dataset if hydra_cfg is not None else None
    _use_bendlabs = (
        args.num_bendlabs_sensors is not None and args.num_bendlabs_sensors > 0
    ) or (
        args.num_bendlabs_sensors is None
        and ds_cfg is not None
        and ds_cfg.get("use_bendlabs", False)
    )
    if _use_bendlabs:
        _num_sensors = (
            args.num_bendlabs_sensors
            if args.num_bendlabs_sensors is not None
            else int(ds_cfg.get("num_bendlabs_sensors", 4))
        )
        logger.info(
            f"Initializing BendLabsRecorder ({_num_sensors} sensors)...")
        bendlabs = BendLabsRecorder(init_node=False, num_sensors=_num_sensors)
        logger.info("Waiting for BendLabs sensors...")
        deadline = time.perf_counter() + 10.0
        while not bendlabs.sensors_ready() and not rospy.is_shutdown():
            if time.perf_counter() > deadline:
                logger.error(
                    "Timed out waiting for BendLabs sensors. Check that uart_bridge_node is running.")
                sys.exit(1)
            rospy.sleep(0.1)
        logger.info("BendLabs sensors ready.")

    np.set_printoptions(linewidth=np.inf, formatter={
                        "float": lambda x: f"{x:0.3f}"})
    torch.set_printoptions(linewidth=2000, sci_mode=False, precision=5)

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    num_rollouts = args.num_rollouts
    max_timesteps = args.max_timesteps
    use_amp = args.use_amp
    step_duration_s = 1.0 / fps
    total_steps_per_episode = []
    eval_start_time = timeit.default_timer()
    total_steps_completed = 0

    logger.info(
        f"Control frequency: {fps} Hz | step budget: {step_duration_s * 1000:.1f} ms | AMP: {use_amp}")

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

    with Progress(*progress_columns, console=console) as progress:
        rollout_task = progress.add_task("Evaluation", total=num_rollouts)

        for rollout_id in range(num_rollouts):
            step_task = progress.add_task(
                make_step_description(rollout_id, total_steps_completed),
                total=max_timesteps,
            )

            progress.stop()

            move_to_home(arm, hydra_cfg, claw)

            input(f"\n  Reset robot to start position, then press Enter to start rollout "
                  f"{rollout_id + 1}/{num_rollouts}...")
            progress.start()

            arm.zero_ft_sensor()
            arm.activate_cartesian_controller()
            policy.reset()

            # {t, chunk (chunk_size, 7)} — one entry per inference
            chunk_log = []
            # {t, arm_q (6,), gripper float, action (7,)} — one per step
            actual_log = []
            n_action_steps = policy.config.n_action_steps
            chunk_size_cfg = policy.config.chunk_size

            for t in range(max_timesteps):
                step_start = time.perf_counter()

                # --- Observe ---
                obs = format_real_robot_observations(
                    arm, image_recorder, claw, features, camera_shape,
                    bendlabs=bendlabs,
                )

                # Preprocessor handles: add batch dim, move to device, normalize.
                # Pass raw CPU tensors — do NOT manually unsqueeze or .cuda() here.
                policy_obs = preprocessor(obs)

                if t == 0 and rollout_id == 0:
                    from lerobot.datasets.lerobot_dataset import LeRobotDataset
                    _dataset = LeRobotDataset(
                        repo_id="bottle_open_06241205",
                        root="/root/osx-ur/data/bottle_open_06241205",
                    )
                    _sample = _dataset[0]
                    _ds_obs = {k: v for k, v in _sample.items()
                               if k in features
                               and "action" not in k
                               and "index" not in k
                               and "timestamp" not in k}
                    _ds_preprocessed = preprocessor(_ds_obs)

                    print("\n=== RAW VALUE COMPARISON (t=0, rollout=0) ===")
                    for key in ["observation.state", "observation.ft", "observation.bendlabs"]:
                        if key in obs and key in _ds_obs:
                            live = obs[key].numpy()
                            ds = _ds_obs[key].numpy()
                            print(f"\n{key}:")
                            print(f"  Dataset raw : {ds.round(3)}")
                            print(f"  Live raw    : {live.round(3)}")
                            print(
                                f"  Abs diff    : {np.abs(ds - live).round(3)}")

                    print("\n=== NORMALIZED VALUE COMPARISON (t=0, rollout=0) ===")
                    for key in ["observation.state", "observation.ft", "observation.bendlabs"]:
                        if key in policy_obs and key in _ds_preprocessed:
                            live_n = policy_obs[key].cpu()
                            ds_n = _ds_preprocessed[key].cpu()
                            print(f"\n{key}:")
                            print(
                                f"  Dataset normalized : {ds_n.numpy().round(3)}")
                            print(
                                f"  Live normalized    : {live_n.numpy().round(3)}")
                            print(
                                f"  Max abs (live)     : {live_n.abs().max():.4f}")
                            print(
                                f"  Max abs (dataset)  : {ds_n.abs().max():.4f}")
                            if live_n.abs().max() > 5.0:
                                print(
                                    f"  ❌ OUT OF RANGE — likely causing constant output")
                            else:
                                print(f"  ✅ In range")

                # --- Act ---
                # select_action manages the action chunk internally via a deque.
                # On the first call (or when the queue empties after n_action_steps),
                # it runs inference and predicts chunk_size actions, stores them in
                # the queue, then pops and returns one (B, action_dim) tensor per call.
                # torch.autocast halves GPU compute time when use_amp=True.
                amp_ctx = torch.autocast("cuda") if use_amp else nullcontext()
                with torch.inference_mode(), amp_ctx:
                    # Tensor (1, 7), normalized
                    action = policy.select_action(policy_obs)

                    # Capture the full chunk at every inference step (queue just refilled).
                    # The queue refills every n_action_steps calls; at t=0 it always refills.
                    if t % n_action_steps == 0:
                        full_chunk_norm = policy.predict_action_chunk(
                            policy_obs)
                        # full_chunk_norm: (1, chunk_size, 7) normalized
                        chunk_np = full_chunk_norm.squeeze(
                            0).cpu()  # (chunk_size, 7)
                        unnorm_steps = [
                            postprocessor(step.unsqueeze(0)).squeeze(0).numpy()
                            for step in chunk_np
                        ]
                        chunk_log.append(
                            {"t": t, "chunk": np.stack(unnorm_steps)})

                # Postprocessor unnormalizes the action and moves it back to CPU.
                # Tensor (1, 7), in original joint-angle units
                action = postprocessor(action)

                # --- Execute on hardware ---
                execute_cartesian_action(action, arm, claw, safety_cfg)

                # --- Debug: compare arm joints vs policy action ---
                _arm_q = np.array(arm.joint_angles())
                _act_q = action.squeeze().cpu().numpy()
                _grip = claw.get_normalized_position() if claw is not None else 0.0
                actual_log.append({
                    "t": t,
                    "arm_q": _arm_q.copy(),
                    "gripper": _grip,
                    "action": _act_q.copy(),
                })
                logger.info(
                    f"t={t:03d}  arm={_arm_q.round(3)}  "
                    f"act={_act_q[:6].round(3)}  "
                    f"max_gap={np.abs(_act_q[:6] - _arm_q).max():.4f} rad  "
                    f"gripper_cmd={_act_q[6]:.3f}"
                )

                # --- Rate control ---
                # Sleep whatever remains of the step budget so the arm receives
                # commands at a steady fps rather than in bursts.
                dt_s = time.perf_counter() - step_start
                sleep_s = step_duration_s - dt_s
                if sleep_s < 0:
                    logger.warning(
                        f"Step {t} overran budget: {dt_s * 1000:.1f} ms > {step_duration_s * 1000:.1f} ms "
                        f"({1 / dt_s:.1f} Hz actual vs {fps} Hz target). "
                        "Consider enabling --use_amp or reducing --fps."
                    )
                precise_sleep(max(sleep_s, 0.0))

                total_steps_completed += 1
                progress.update(
                    step_task,
                    advance=1,
                    description=make_step_description(
                        rollout_id, total_steps_completed),
                )

            arm.activate_joint_trajectory_controller()
            steps_taken = t + 1
            total_steps_per_episode.append(steps_taken)

            logger.info(
                f"Rollout {rollout_id} complete | steps: {steps_taken}")

            # Save chunk visualization for this rollout
            vis_path = output_dir / f"chunks_rollout{rollout_id:02d}.png"
            try:
                save_chunk_visualization(
                    chunk_log, actual_log, vis_path,
                    policy_name=policy_path.name,
                    chunk_size=chunk_size_cfg,
                )
                logger.info(f"Chunk visualization saved to: {vis_path}")
            except Exception as e:
                logger.warning(f"Could not save chunk visualization: {e}")

            progress.remove_task(step_task)
            progress.update(rollout_task, advance=1)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"Rollouts completed:        {num_rollouts}")
    logger.info(
        f"Mean steps per episode:    {np.mean(total_steps_per_episode):.1f}")
    logger.info(
        f"Std  steps per episode:    {np.std(total_steps_per_episode):.1f}")
    logger.info("=" * 60)
    logger.info(f"Evaluation complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
