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
    rosrun osx_ur5e replay_pkl_episode.py dataset.task_name=bottle_open_06
    rosrun osx_ur5e replay_pkl_episode.py dataset.task_name=bottle_open_06 dataset.episode_idx=2

dataset.task_name is the preferred way to point at a dataset: repo_id
defaults to "<task_name>_lerobot" (matching train_point_policy.py's
convention -- what run_pipeline.sh's LEROBOT_REPO_ID produces by default), so
there's no need to know/type the "_lerobot" suffix or place/rename anything
under dataset.dir. Pass dataset.repo_id=<exact_name> instead if your dataset
doesn't follow that naming convention -- when dataset.task_name IS given, it
always takes priority over dataset.repo_id (including test_task.yaml's
hardcoded repo_id default, kept only for backward compatibility when neither
is overridden).

Force the starting gripper state instead of using the dataset's own recorded
value at the replay start frame:
    rosrun osx_ur5e replay_pkl_episode.py dataset.task_name=... dataset.start_gripper=open
    rosrun osx_ur5e replay_pkl_episode.py dataset.task_name=... dataset.start_gripper=closed
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
from osx_ur5e import pose_offset
from ur_control import transformations
from ur_control.fzi_cartesian_compliance_controller import CompliantController

logger = logging.getLogger(__name__)
console = Console()

_POSE_LABELS = ["X", "Y", "Z", "a1x", "a1y", "a1z", "a2x", "a2y", "a2z", "Gripper"]

# ClawController normalized positions (see gripper_state_to_normalized).
GRIPPER_OPEN_NORM = 1.0    # == gripper_state -1
GRIPPER_CLOSED_NORM = 0.0  # == gripper_state +1


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


def action_to_target_pose(action: np.ndarray, offset=None) -> np.ndarray:
    """action = [x, y, z, ortho6d(6), gripper] -> [x, y, z, qx, qy, qz, qw].

    `offset` (dataset.eval_offset, see osx_ur5e.pose_offset) shifts the
    commanded pose along its OWN local axes -- the same correction
    evaluate_policy_points.py applies to policy actions, so a replay and a
    rollout put the physical tool tip in the same place. Applied here rather
    than at the call sites so it covers both the start-pose easing and the
    per-step targets.
    """
    quat = transformations.quaternion_from_ortho6(action[3:9])  # [qx, qy, qz, qw], matches arm.end_effector()
    pose = np.concatenate([action[:3], quat])
    if offset is not None:
        pose = pose_offset.apply_offset(pose, offset)
    return pose


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


# ---------------------------------------------------------------------------
# Optional: compare a policy's predicted action chunk against the demo
# ---------------------------------------------------------------------------
# Replay drives the arm from the RECORDED actions, so the trajectory is ground
# truth. Running a policy alongside it therefore gives a teacher-forced read on
# hardware: at each inference tick we ask "given the state the demo is in and
# what the tracker currently sees, what would the policy have done?" -- without
# letting the policy's own errors move the arm. That isolates prediction quality
# from closed-loop drift, which a normal rollout conflates.


def setup_chunk_compare(policy_path: Path):
    """Load a checkpoint for chunk comparison. Imports are local so a plain
    replay never pays for torch/lerobot."""
    import torch  # noqa: F401
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from evaluate_policy_points import load_policy_config_compat

    cfg = load_policy_config_compat(policy_path)
    policy = get_policy_class(cfg.type).from_pretrained(str(policy_path), config=cfg)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    policy.to(device).eval()
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    logger.info(
        f"Chunk comparison ENABLED: {policy_path} on {device} "
        f"(chunk_size={cfg.chunk_size}, n_action_steps={cfg.n_action_steps})"
    )
    return policy, pre, post, cfg


def predict_chunk(policy, pre, post, state_vec, env_vec, chunk_len):
    """One full predicted chunk in physical units -> (chunk_len, action_dim).

    postprocessor expects (batch, action_dim), so the chunk is unnormalized one
    step at a time -- same as visualize_action_chunk.py's postprocess_chunk.
    """
    import torch
    obs = {
        "observation.state": torch.tensor(np.asarray(state_vec, dtype=np.float32)),
        "observation.environment_state": torch.tensor(np.asarray(env_vec, dtype=np.float32)),
    }
    with torch.inference_mode():
        raw = policy.predict_action_chunk(pre(obs))[:, :chunk_len]
    return np.stack([post(raw[:, k]).squeeze(0).cpu().numpy() for k in range(raw.shape[1])])


def plot_chunk_vs_demo(chunks, chunk_at, demo_actions, actual_pose6d, out_path,
                       repo_id, episode_idx, env_source):
    """chunks: list of (L, D) predicted chunks; chunk_at: episode step each was
    predicted at; demo_actions: (T, D) recorded actions.

    Top three panels overlay every predicted chunk on the demo trajectory at the
    step it was predicted from, so a chunk that veers away from the black line is
    immediately visible. The bottom panel is the quantitative version: error
    against lookahead depth, which separates "predicts badly" from "predicts
    fine but goes stale".
    """
    if not chunks:
        return
    T = len(demo_actions)
    fig, axes = plt.subplots(4, 1, figsize=(13, 12))
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0, vmax=max(1, len(chunks) - 1))

    for d, (ax, name) in enumerate(zip(axes[:3], ["pos_x", "pos_y", "pos_z"])):
        ax.plot(np.arange(T), demo_actions[:, d] * 1000.0, "k-", lw=2.0,
                label="demo action (commanded)", zorder=3)
        if actual_pose6d is not None:
            ax.plot(np.arange(len(actual_pose6d)), actual_pose6d[:, d] * 1000.0,
                    color="0.55", ls="--", lw=1.0, label="arm actual", zorder=2)
        for i, (ch, t0) in enumerate(zip(chunks, chunk_at)):
            xs = np.arange(t0, t0 + len(ch))
            ax.plot(xs, ch[:, d] * 1000.0, color=cmap(norm(i)), lw=1.1, alpha=0.9,
                    zorder=4, label="predicted chunk" if i == 0 else None)
            ax.plot(t0, ch[0, d] * 1000.0, ".", color=cmap(norm(i)), ms=5, zorder=5)
        ax.set_ylabel(f"{name} (mm)")
        ax.grid(alpha=0.3)
        if d == 0:
            ax.legend(fontsize=8, loc="best")
    axes[2].set_xlabel("episode step")

    # error vs lookahead: how far into a chunk can you trust it?
    L = max(len(c) for c in chunks)
    err = np.full((len(chunks), L), np.nan)
    for i, (ch, t0) in enumerate(zip(chunks, chunk_at)):
        for k in range(len(ch)):
            t = t0 + k
            if t < T:
                err[i, k] = np.linalg.norm(ch[k, :3] - demo_actions[t, :3]) * 1000.0
    ax = axes[3]
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(err, axis=0)
        p95 = np.nanpercentile(err, 95, axis=0)
    ax.plot(np.arange(L), mean, "b-", lw=2, label="mean |predicted - demo| (3D)")
    ax.plot(np.arange(L), p95, "b--", lw=1, alpha=0.7, label="p95")
    ax.set_xlabel("steps ahead within the predicted chunk")
    ax.set_ylabel("position error (mm)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(
        f"error vs lookahead   (step 0 = fresh observation; "
        f"growth here = staleness, offset at step 0 = prediction bias)", fontsize=9)

    fig.suptitle(
        f"Predicted action chunk vs demo -- {repo_id} episode {episode_idx}  "
        f"({len(chunks)} chunks, env_state from {env_source})", fontsize=11)
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
    # dataset.task_name is the preferred way to point at a dataset (matches
    # train_point_policy.py's convention: repo_id defaults to
    # "<task_name>_lerobot", i.e. what run_pipeline.sh's LEROBOT_REPO_ID
    # produces by default) -- no need to know/type the "_lerobot" suffix or
    # otherwise place/rename anything under cfg.dataset.dir. When given, it
    # takes priority over dataset.repo_id's config-file default (test_task.yaml
    # still has a hardcoded repo_id for backward compatibility when neither is
    # overridden); pass dataset.repo_id=<exact_name> directly instead if you
    # need a repo_id that doesn't follow the "<task_name>_lerobot" convention.
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
    # Decide which step of the episode to start the replay from. By default
    # this is the first frame; dataset.start_step=N starts N steps into the
    # episode instead.
    # ------------------------------------------------------------------
    start_step = int(cfg.dataset.get("start_step", 0))
    replay_start = ep_start + start_step

    # Local-frame TCP correction applied to every commanded pose (start-pose
    # easing included -- see action_to_target_pose). Nothing is subtracted back
    # out here: unlike evaluate_policy_points.py there is no policy observing
    # the state, so the recorded actions are simply commanded `offset` away
    # from where they were recorded.
    offset = pose_offset.resolve_offset(cfg.dataset)
    if pose_offset.is_enabled(offset):
        logger.info(f"dataset.eval_offset: {pose_offset.describe(offset)} "
                    "-- commanded poses shifted along the gripper's own axes")

    # ------------------------------------------------------------------
    # Optional chunk comparison (dataset.compare_policy_path). Off by default,
    # so plain replay behaviour is untouched.
    # ------------------------------------------------------------------
    compare_path = cfg.dataset.get("compare_policy_path", None)
    chunk_cmp = None
    if compare_path:
        chunk_cmp = setup_chunk_compare(Path(compare_path))
    compare_every = cfg.dataset.get("compare_every", None)
    env_source_cfg = str(cfg.dataset.get("compare_env_source", "auto"))
    if env_source_cfg not in ("auto", "live", "dataset"):
        raise ValueError(
            f"Invalid dataset.compare_env_source: {env_source_cfg!r} "
            "(expected 'auto', 'live', or 'dataset')")

    if not (ep_start <= replay_start < ep_end):
        raise ValueError(
            f"Resolved replay start frame {replay_start} out of episode range "
            f"[{ep_start}, {ep_end}) (start_step={start_step})"
        )
    replay_len = ep_end - replay_start
    if replay_start != ep_start:
        logger.info(
            f"Starting replay at step {replay_start - ep_start} ({replay_len} steps)"
        )

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
    arm.update_stiffness(np.array([controller_cfg.stiffness_trans] * 3 + [controller_cfg.stiffness_rot] * 3))
    arm.auto_switch_controllers = False
    arm.async_mode = True
    arm.zero_ft_sensor()

    claw = None
    if use_gripper:
        logger.info("Initializing ClawController...")
        claw = ClawController(init_node=False)

    # Live tracked keypoints, if tracking_node is up. The dataset's own recorded
    # env_state is the fallback -- comparing against live keypoints is the whole
    # reason to do this on hardware rather than offline, so which one was used
    # is recorded in the plot title.
    live_env = {"v": None}
    env_source = "dataset"
    if chunk_cmp is not None and env_source_cfg in ("auto", "live"):
        from std_msgs.msg import Float64MultiArray
        rospy.Subscriber("/tracking_node/environment_state", Float64MultiArray,
                         lambda m: live_env.__setitem__("v", np.array(m.data, dtype=np.float32)))
        deadline = time.time() + 3.0
        while live_env["v"] is None and time.time() < deadline:
            time.sleep(0.1)
        if live_env["v"] is not None:
            env_source = "live"
        elif env_source_cfg == "live":
            raise RuntimeError(
                "dataset.compare_env_source=live but no /tracking_node/environment_state "
                "messages arrived. Is tracking_node running and started "
                "(rosservice call /tracking_node/reset_and_track)?")
        else:
            logger.warning("No live env_state; falling back to the dataset's recorded keypoints")
        logger.info(f"chunk comparison env_state source: {env_source}")

    # ------------------------------------------------------------------
    # Move to first frame's pose via the same cartesian delta-clip control
    # used during replay, instead of IK + joint trajectory controller.
    # ------------------------------------------------------------------
    first_action = dataset[replay_start]["action"].numpy()
    first_target_pose = action_to_target_pose(first_action, offset)
    first_gripper_norm = gripper_state_to_normalized(first_action[9])

    # dataset.start_gripper overrides the gripper command used while easing
    # into the start pose -- "auto" (default) keeps first_gripper_norm as
    # recorded in the dataset at replay_start; "open" forces it open;
    # "closed" eases in OPEN first, then prompts you to insert the object and
    # closes to GRIPPER_CLOSED_NORM before replay starts (see the dedicated
    # branch below).
    start_gripper = cfg.dataset.get("start_gripper", "auto")
    if start_gripper not in ("auto", "open", "closed"):
        raise ValueError(f"Invalid dataset.start_gripper: {start_gripper!r} (expected 'auto', 'open', or 'closed')")
    if start_gripper == "open":
        first_gripper_norm = GRIPPER_OPEN_NORM

    arm.activate_cartesian_controller()

    if start_gripper == "closed":
        # Ease in OPEN so you can insert the object by hand, then close to
        # GRIPPER_CLOSED_NORM (not the dataset's own first-frame value --
        # "closed" is an explicit override) before replay starts.
        logger.info(f"Easing into start pose (gripper open, for object insertion): {first_target_pose}")
        move_to_pose_via_cartesian_control(
            first_target_pose, arm, claw, GRIPPER_OPEN_NORM, safety_cfg, fps
        )
        input("\n  At start pose — insert the object into the gripper, then press Enter to close...")
        if claw is not None:
            claw.set_normalized_position(GRIPPER_CLOSED_NORM)
            time.sleep(1.0)  # let the claw finish closing before the arm moves
        input(f"\n  Closed. Episode {episode_idx} ({replay_len} steps) — press Enter to start replay...")
    else:
        logger.info(f"Easing into start pose: {first_target_pose}")
        move_to_pose_via_cartesian_control(
            first_target_pose, arm, claw, first_gripper_norm, safety_cfg, fps
        )
        input(f"\n  Episode {episode_idx} ({replay_len} steps) — press Enter to start replay...")

    # ------------------------------------------------------------------
    # Replay loop
    # ------------------------------------------------------------------
    logger.info(f"Replaying episode {episode_idx} at {fps} Hz...")

    action_log = []   # {t, action_pose6d (9,), gripper_norm}
    actual_log = []   # {t, pose6d (9,), gripper}
    chunks, chunk_at = [], []   # predicted chunks + the episode step each came from

    with tqdm.tqdm(total=replay_len, desc=f"Episode {episode_idx}") as pbar:
        for t in range(replay_start, ep_end):
            step_start = time.perf_counter()
            step = t - ep_start

            action = dataset[t]["action"].numpy()  # [x, y, z, ortho6d(6), gripper]
            target_pose = action_to_target_pose(action, offset)
            gripper_norm = gripper_state_to_normalized(action[9])

            execute_cartesian_target(target_pose, arm, claw, gripper_norm, safety_cfg)

            if chunk_cmp is not None:
                policy, pre, post, pcfg = chunk_cmp
                every = int(compare_every) if compare_every else int(pcfg.n_action_steps)
                if (step % max(1, every)) == 0:
                    # observation.state comes from the DATASET so the comparison is
                    # anchored to the demo the arm is actually following (and dodges
                    # the gripper-channel question entirely); env_state is live when
                    # available, which is the part worth testing on hardware.
                    obs_state = dataset[t]["observation.state"].numpy()
                    obs_env = (live_env["v"] if env_source == "live"
                               else dataset[t]["observation.environment_state"].numpy())
                    if obs_env is not None:
                        try:
                            chunks.append(predict_chunk(policy, pre, post, obs_state,
                                                        obs_env, int(pcfg.chunk_size)))
                            chunk_at.append(step)
                        except Exception:
                            logger.exception(f"chunk prediction failed at step {step}")

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
    print(f"Replay complete: {replay_len} steps")

    if chunk_cmp is not None:
        if chunks:
            demo_actions = np.stack([dataset[t]["action"].numpy()
                                     for t in range(ep_start, ep_end)])
            actual_arr = np.array([e["pose6d"] for e in actual_log]) if actual_log else None
            cmp_png = output_dir / f"chunk_vs_demo_ep{episode_idx:03d}.png"
            try:
                plot_chunk_vs_demo(chunks, chunk_at, demo_actions, actual_arr, cmp_png,
                                   repo_id, episode_idx, env_source)
                print(f"Chunk-vs-demo plot saved: {cmp_png}")
            except Exception:
                logger.exception("Could not save the chunk-vs-demo plot")
            np.savez(
                output_dir / f"chunk_vs_demo_ep{episode_idx:03d}.npz",
                chunks=np.stack(chunks), chunk_at=np.array(chunk_at),
                demo_actions=demo_actions, env_source=env_source,
            )
            # Same decomposition the offline analysis uses: the error at
            # lookahead 0 is prediction bias, its growth with k is staleness.
            errs = {}
            for ch, t0 in zip(chunks, chunk_at):
                for k in range(len(ch)):
                    t = t0 + k
                    if t < len(demo_actions):
                        errs.setdefault(k, []).append(
                            np.linalg.norm(ch[k, :3] - demo_actions[t, :3]) * 1000.0)
            print(f"\n  chunk-vs-demo position error by lookahead ({len(chunks)} chunks, env={env_source}):")
            for k in [0, 1, 5, 10, 20, 30, 45, 59]:
                if k in errs:
                    print(f"     +{k:>2} steps: mean {np.mean(errs[k]):6.2f} mm   "
                          f"p95 {np.percentile(errs[k], 95):6.2f} mm   (n={len(errs[k])})")
        else:
            logger.warning("Chunk comparison was enabled but no chunks were captured")

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
