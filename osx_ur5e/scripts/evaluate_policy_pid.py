#!/usr/bin/env python3
"""Evaluate a COMET diffusion policy on the real UR5e with PI force tracking.

The diffusion policy handles trajectory generation (pose head) while force
tracking is handled by an analytical compliance mapper + PI controller. The
compliance diffusion head outputs (``action.normal_force``,
``action.normal_torque``, ``action.estimated_stiffness``) are replaced at
inference time with values computed from the target force, the current EEF
position, and the measured wrench.

This is the real-robot analogue of
``dependencies/comet/comet/scripts/eval/robosuite/evaluate_robosuite_pid.py``.

Usage:
    python evaluate_policy_pid.py

    # Override eval settings:
    python evaluate_policy_pid.py eval.num_rollouts=5 eval.max_timesteps=500

    # Set the PI target force (N):
    python evaluate_policy_pid.py eval.pid_target_force=40.0

Controls during each rollout:
    Enter  - confirm start of rollout (after reset prompt)
"""

# Portions adapted from evaluate_robosuite_pid.py (MIT License, OMRON SINIC X).

import logging
import datetime
import signal
import sys
import timeit
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torchvision import transforms

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

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

from ur_control import transformations

from comet.common.utils.utils import load_base_policy
from comet.common.utils.ft_visualizer import FTVisualizer
from comet.common.utils.viz_utils import save_to_video
from comet.common.policies.types import FeatureType

from osx_ur5e.fdcc_env import FDCCEnv

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
# Analytical force mapper + PI controller
# ---------------------------------------------------------------------------

@dataclass
class ForceMapperConfig:
    """All tunables for the analytical force path."""
    target_force: float = 30.0
    target_torque: float = 0.0

    # Stiffness schedule: maps |F_target| -> K
    k_max: float = 1000.0
    k_min: float = 500.0
    f_low: float = 1.0
    f_high: float = 10.0
    max_displacement: float = 0.015

    # Non-contact / default stiffness for orthogonal directions
    # default_stiffness: float = 2400.0
    # default_stiffness_rot: float = 2400.0
    default_stiffness: float = 1000.0
    default_stiffness_rot: float = 1000.0

    characteristic_length: float = 0.1
    use_isotropic_stiffness: bool = False

    # PI controller gains
    kp_force: float = 0.2
    ki_force: float = 0.03
    max_integral: float = 70.0  # prevents integral windup

    # Contact detection threshold (N) — below this, PI correction is skipped
    contact_threshold: float = 3.5

    # Smoothing on the measured force to reject sensor noise
    force_ema_alpha: float = 0.3

    # ---- Stability safeguards (real-robot specific) ----
    # When True, feedback uses projected force |f · d̂| instead of ‖f‖.
    # Projected force is the physically meaningful "normal" force and is
    # much less noisy / less coupled to tangential motion on hardware.
    use_projected_force_feedback: bool = True

    # Clamp for the parasitic-spring offset (m). Prevents feedforward
    # from exploding when the policy's nominal position runs far ahead
    # of the actual EEF (e.g. in free space before contact).
    max_tracking_offset: float = 0.01

    # Only apply parasitic-spring feedforward once the EEF is close enough
    # to actually be in contact. Before that, F_cmd falls back to
    # `pre_contact_force_cmd` (usually a small constant push).
    feedforward_requires_contact: bool = True
    pre_contact_force_cmd: float = 0.0

    # Final clamp on the commanded force (N). Keeps the command in a
    # physically reasonable band around the target, even if feedforward
    # or integral windup misbehaves. Applied as target * [lo, hi] bounds
    # so it scales with the target.
    f_cmd_rel_min: float = 0.0    # never command negative push
    f_cmd_rel_max: float = 2.0    # at most 2x target

    # Structured debug logging every N calls (set 0 to disable)
    debug_log_every: int = 10


class ForceTrackingPI:
    """Simple PI controller on force error."""

    def __init__(self, kp: float, ki: float, max_integral: float):
        self.kp = kp
        self.ki = ki
        self.max_integral = max_integral
        self._integral = 0.0

    def reset(self):
        self._integral = 0.0

    def update(self, error: float, dt: float) -> float:
        self._integral = np.clip(
            self._integral + error * dt,
            -self.max_integral, self.max_integral,
        )
        return self.kp * error + self.ki * self._integral


class AnalyticalForceMapper:
    """Replaces the compliance diffusion head at inference time.

    Given a target force and the current robot state, computes
    normal_force, normal_torque, and estimated_stiffness analytically,
    with optional closed-loop PI correction from the measured wrench.
    """

    def __init__(self, cfg: ForceMapperConfig):
        self.cfg = cfg
        self.pi = ForceTrackingPI(cfg.kp_force, cfg.ki_force, cfg.max_integral)
        self._force_ema: Optional[float] = None
        self._call_count: int = 0

    def reset(self):
        self.pi.reset()
        self._force_ema = None
        self._call_count = 0

    def stiffness_schedule(self, f_target_abs: float) -> float:
        """Map target force magnitude to stiffness."""
        c = self.cfg
        if f_target_abs < c.f_low:
            K = c.k_max
        elif f_target_abs > c.f_high:
            K = c.k_min
        else:
            K = c.k_max - (c.k_max - c.k_min) * (f_target_abs - c.f_low) / (c.f_high - c.f_low)

        if c.max_displacement > 0 and f_target_abs > c.f_low:
            k_floor = f_target_abs / c.max_displacement
            K = max(K, k_floor)

        return float(np.clip(K, c.k_min, c.k_max))

    def smooth_force(self, raw_force_norm: float) -> float:
        """EMA filter on measured force magnitude."""
        if self._force_ema is None:
            self._force_ema = raw_force_norm
        else:
            a = self.cfg.force_ema_alpha
            self._force_ema = a * raw_force_norm + (1.0 - a) * self._force_ema
        return self._force_ema

    def compute(
        self,
        contact_direction: np.ndarray,
        x_nominal: np.ndarray,
        x_actual: np.ndarray,
        wrench: np.ndarray,
        dt: float,
    ) -> tuple[float, float, float, dict]:
        """Compute compliance parameters for the target force.

        Returns:
            (normal_force_cmd, normal_torque_cmd, stiffness, diagnostics)
        """
        c = self.cfg
        eps = 1e-8

        d_norm = np.linalg.norm(contact_direction)
        d = contact_direction / (d_norm + eps) if d_norm > eps else np.array([0., 0., -1.])

        K = self.stiffness_schedule(abs(c.target_force))

        # --- Force feedback ---------------------------------------------------
        # Projected force |f · d̂| is the normal-direction contact force and is
        # much less noisy than ‖f‖ on hardware (rejects tangential/shear).
        f_vec = np.asarray(wrench, dtype=np.float64).reshape(-1)[:3]
        print(f"f_vec shape: {f_vec.shape}")
        print(f"f_vec: {f_vec}")

        f_norm_full = float(np.linalg.norm(f_vec))
        print(f"f_norm_full: {f_norm_full}")
        f_projected = float(abs(np.dot(f_vec, d)))
        print(f"f_projected: {f_projected}")
        f_meas_raw = f_projected if c.use_projected_force_feedback else f_norm_full
        F_measured = self.smooth_force(f_meas_raw)
        print(f"F_measured: {F_measured}")
        in_contact = F_measured > c.contact_threshold

        # --- Feedforward: compensate parasitic spring force ------------------
        # FDCC sees: F_achieved = F_cmd + K * dot(x_nominal - x_actual, d)
        # ⇒ F_cmd = F_target - K * tracking_offset
        tracking_offset_raw = float(np.dot(x_nominal - x_actual, d))
        tracking_offset = float(
            np.clip(tracking_offset_raw, -c.max_tracking_offset, c.max_tracking_offset)
        )
        print(f"tracking_offset_raw: {tracking_offset_raw}")
        print(f"tracking_offset: {tracking_offset}")

        # The parasitic force exist because the policy nominal position is not the same
        # as the actual position, so we need to compensate for that.
        # F_parasitic = (K / 2) * tracking_offset
        F_parasitic = K * tracking_offset
        print(f"F_parasitic: {F_parasitic}")

        if c.feedforward_requires_contact and not in_contact:
            # Before contact, don't let a large position mismatch drive F_cmd
            # wildly negative / positive. Use a fixed small pre-contact push.
            F_cmd = float(c.pre_contact_force_cmd)
            F_correction = 0.0
            force_error = 0.0
            ff_enabled = False
        else:
            # F_cmd = (c.target_force - F_parasitic) / 2.0  # THIS fixed the bug, i want my money back
            F_cmd = c.target_force - F_parasitic
            print(f"F_cmd: {F_cmd}")
            ff_enabled = True
            # --- Feedback: PI correction -------------------------------------
            if in_contact:
                force_error = c.target_force - F_measured
                F_correction = self.pi.update(force_error, dt)
                F_cmd += F_correction
            else:
                force_error = 0.0
                F_correction = 0.0

        # --- Final safety clamp ---------------------------------------------
        F_cmd_unclamped = F_cmd
        print(f"F_cmd_unclamped: {F_cmd_unclamped}")
        lo = c.f_cmd_rel_min * c.target_force
        hi = c.f_cmd_rel_max * c.target_force
        if lo > hi:
            lo, hi = hi, lo
        F_cmd = float(np.clip(F_cmd, lo, hi))
        print(f"F_cmd: {F_cmd}")

        diag = {
            "K": K,
            "d": d,
            "d_norm_raw": d_norm,
            "x_nominal": x_nominal,
            "x_actual": x_actual,
            "tracking_offset_raw": tracking_offset_raw,
            "tracking_offset": tracking_offset,
            "F_parasitic": F_parasitic,
            "F_measured": F_measured,
            "f_projected": f_projected,
            "f_norm_full": f_norm_full,
            "in_contact": in_contact,
            "ff_enabled": ff_enabled,
            "force_error": force_error,
            "F_correction": F_correction,
            "integral": self.pi._integral,
            "F_cmd_unclamped": F_cmd_unclamped,
            "F_cmd": F_cmd,
            "clamped": F_cmd_unclamped != F_cmd,
        }
        return F_cmd, c.target_torque, K, diag

    def override_action(
        self,
        action_dict: dict,
        arm,
        dt: float,
    ) -> dict:
        """Override the compliance head outputs in the (batched) action dict.

        Reads the current EEF position and wrench from the real robot arm,
        replaces ``action.normal_force``, ``action.normal_torque``, and
        ``action.estimated_stiffness`` with analytical/PI outputs, and leaves
        the pose-head outputs (position, rotation, compliance_direction) untouched.
        """
        cd_key = "action.compliance_direction" if "action.compliance_direction" in action_dict else "action.contact_direction"
        contact_dir_t = action_dict[cd_key]
        nominal_pos_t = action_dict["action.ref_position"]

        contact_dir = contact_dir_t.detach().cpu().numpy().reshape(-1)[:3] \
            if isinstance(contact_dir_t, torch.Tensor) else np.asarray(contact_dir_t).reshape(-1)[:3]
        x_nominal = nominal_pos_t.detach().cpu().numpy().reshape(-1)[:3] \
            if isinstance(nominal_pos_t, torch.Tensor) else np.asarray(nominal_pos_t).reshape(-1)[:3]

        x_actual = np.asarray(arm.end_effector()[:3], dtype=np.float64).reshape(3)
        print(f"x_actual: {x_actual}")
        wrench = np.asarray(arm.get_wrench(), dtype=np.float64).reshape(-1) # force felt by the robot (direction is opposite of the contact direction)
        print(f"wrench: {wrench}")

        # Transform wrench from tool frame to world frame TODO (malek): check if this is correct with cristian
        eef_quat = np.asarray(arm.end_effector()[3:], dtype=np.float64)
        R_eef = transformations.rotation_matrix_from_quaternion(eef_quat)[:3, :3]
        wrench[:3] = R_eef @ wrench[:3]   # forces: tool → world
        wrench[3:] = R_eef @ wrench[3:]   # torques: tool → world

        F_cmd, tau_cmd, K, diag = self.compute(
            contact_direction=contact_dir,
            x_nominal=x_nominal,
            x_actual=x_actual,
            wrench=wrench,
            dt=dt,
        )

        # ---- Structured per-step diagnostics -------------------------------
        self._call_count += 1
        if self.cfg.debug_log_every and (self._call_count % self.cfg.debug_log_every) == 0:
            logger.info(
                "[pid] t=%04d | contact=%s ff=%s | F_meas=%.2f (proj=%.2f, full=%.2f) "
                "| off_raw=%+.4fm off=%+.4fm | K=%.0f | F_para=%+.2f F_corr=%+.2f "
                "int=%+.2f | F_cmd=%+.2f (unclamped=%+.2f, clamped=%s)",
                self._call_count,
                diag["in_contact"],
                diag["ff_enabled"],
                diag["F_measured"],
                diag["f_projected"],
                diag["f_norm_full"],
                diag["tracking_offset_raw"],
                diag["tracking_offset"],
                diag["K"],
                diag["F_parasitic"],
                diag["F_correction"],
                diag["integral"],
                diag["F_cmd"],
                diag["F_cmd_unclamped"],
                diag["clamped"],
            )
            # Extra detail when raw tracking offset hit the clamp
            if abs(diag["tracking_offset_raw"]) > self.cfg.max_tracking_offset:
                logger.warning(
                    "[pid]   x_nominal=%s x_actual=%s d=%s |d|=%.3f",
                    np.array2string(diag["x_nominal"], precision=3),
                    np.array2string(diag["x_actual"], precision=3),
                    np.array2string(diag["d"], precision=3),
                    diag["d_norm_raw"],
                )

        device = contact_dir_t.device if isinstance(contact_dir_t, torch.Tensor) else torch.device("cuda")

        def write_like(key: str, value: float):
            t = action_dict[key]
            if isinstance(t, torch.Tensor):
                action_dict[key] = torch.full_like(t, float(value))
            else:
                action_dict[key] = torch.tensor([[value]], device=device, dtype=torch.float32)

        write_like("action.normal_force", F_cmd)
        write_like("action.normal_torque", tau_cmd)
        write_like("action.estimated_stiffness", K)

        return action_dict


# ---------------------------------------------------------------------------
# Observation formatting (mirrors evaluate_policy.py)
# ---------------------------------------------------------------------------

def format_real_robot_observations(
    arm,
    image_recorder,
    features: dict,
    camera_shape: tuple,
) -> dict:
    """Build a policy-ready observation dict from the real robot arm and cameras."""
    eef = arm.end_effector()
    eef_velocity = arm.end_effector_velocity()

    raw_obs = {
        "observation.qpos":                    arm.joint_angles(),
        "observation.qvel":                    arm.joint_velocities(),
        "observation.eef.position":            eef[:3],
        "observation.eef.linear_velocity":     eef_velocity[:3],
        "observation.eef.angular_velocity":    eef_velocity[3:],
        "observation.eef.rotation_ortho6":     transformations.ortho6_from_quaternion(eef[3:]),
        "observation.eef.rotation_axis_angle": transformations.axis_angle_from_quaternion(eef[3:]),
        "observation.ft":                      arm.get_wrench(),
    }

    obs = {}
    for key, value in raw_obs.items():
        if key in features:
            obs[key] = torch.tensor(np.array(value).flatten(), dtype=torch.float32)

    if image_recorder is not None:
        resize_transform = transforms.Resize(camera_shape, antialias=True)
        raw_images = image_recorder.get_images()
        for cam_name, image_hwc in raw_images.items():
            feat_key = f"observation.images.{cam_name}"
            if feat_key in features:
                image_chw = np.ascontiguousarray(np.transpose(image_hwc, (2, 0, 1)))
                image_tensor = torch.tensor(image_chw, dtype=torch.uint8)
                obs[feat_key] = resize_transform(image_tensor)

    return obs


# ---------------------------------------------------------------------------
# Action conversion
# ---------------------------------------------------------------------------

def convert_policy_action(action_dict: dict) -> dict:
    """Strip the policy batch dimension and convert tensors to numpy arrays.

    FDCCEnv.prepare_action dispatches on the presence of
    ``action.contact_direction`` to handle factored actions internally via
    ``process_factored_action_dict`` (variable-kp, quaternion), so we just
    hand the factored numpy dict through.
    """
    env_action = {}
    for key, value in action_dict.items():
        if isinstance(value, torch.Tensor):
            env_action[key] = value.squeeze(0).cpu().numpy() if value.dim() > 0 else value.cpu().numpy()
        else:
            env_action[key] = np.array(value)
    return env_action


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# config_path is relative to this file's directory
@hydra.main(
    version_base=None,
    config_path="../../../../../dependencies/comet/configs",
    config_name="blackboard_wipe",
)
def main(cfg: DictConfig) -> None:
    eval_dir = Path(cfg.eval.base.load_ckpt) / "eval" / str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    eval_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(eval_dir / "evaluation_pid.log")

    seed = cfg.eval.seed
    num_rollouts = cfg.eval.num_rollouts
    max_timesteps = cfg.eval.max_timesteps
    save_video = OmegaConf.select(cfg, "eval.save_video", default=False)
    policy_filename = OmegaConf.select(cfg, "eval.policy_filename", default="best_ema_policy.ckpt")

    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info(f"Seed: {seed}")
    logger.info(f"Results will be saved to: {eval_dir}")

    # ------------------------------------------------------------------
    # Load policy (pose head is what matters; compliance head is overridden)
    # ------------------------------------------------------------------
    ckpt_dir = Path(cfg.eval.base.load_ckpt)
    logger.info(f"Loading policy from: {ckpt_dir}")
    policy, base_cfg, features = load_base_policy(ckpt_dir, policy_filename)
    policy.cuda()
    policy.eval()

    action_keys = {key: ft.shape for key, ft in features.items() if ft.type is FeatureType.ACTION}
    logger.info(f"Action keys: {list(action_keys.keys())}")
    logger.info(f"Observation keys: {[k for k, ft in features.items() if ft.type is not FeatureType.ACTION]}")

    required_factored = (
        "action.ref_position",
        "action.ref_rotation_ortho6",
        "action.compliance_direction",
        "action.normal_force",
        "action.normal_torque",
        "action.estimated_stiffness",
    )
    missing = [k for k in required_factored if k not in action_keys]
    if missing:
        raise ValueError(
            "This script requires a factored-action policy. Missing action keys: "
            + ", ".join(missing)
        )

    if hasattr(base_cfg, "model_specific") and "camera_shape" in base_cfg.model_specific:
        camera_shape = tuple(base_cfg.model_specific.camera_shape[-2:])
    else:
        camera_shape = (240, 320)
        logger.warning(f"No camera_shape in checkpoint config, defaulting to {camera_shape}")

    control_frequency = int(base_cfg.dataset.dataset.fps)
    dt = 1.0 / control_frequency
    logger.info(f"Control frequency: {control_frequency} Hz (dt={dt:.4f}s)")

    # ------------------------------------------------------------------
    # Build the analytical force mapper
    # ------------------------------------------------------------------

    force_cfg = ForceMapperConfig(
        target_force=cfg.eval.pid.target_force,
        use_isotropic_stiffness=cfg.eval.pid.use_isotropic_stiffness,
        kp_force=cfg.eval.pid.kp_force,
        ki_force=cfg.eval.pid.ki_force,
        max_integral=cfg.eval.pid.max_integral,
        contact_threshold=cfg.eval.pid.contact_threshold,
        force_ema_alpha=cfg.eval.pid.force_ema_alpha,
        max_tracking_offset=cfg.eval.pid.max_tracking_offset,
        pre_contact_force_cmd=cfg.eval.pid.pre_contact_force_cmd,
        f_cmd_rel_min=cfg.eval.pid.f_cmd_rel_min,
        f_cmd_rel_max=cfg.eval.pid.f_cmd_rel_max,
        k_min=cfg.eval.pid.k_min,
        k_max=cfg.eval.pid.k_max,
        debug_log_every=cfg.eval.pid.debug_log_every,
        feedforward_requires_contact=cfg.eval.pid.feedforward_requires_contact,
        use_projected_force_feedback=cfg.eval.pid.use_projected_force_feedback,
    )
    force_mapper = AnalyticalForceMapper(force_cfg)

    # ------------------------------------------------------------------
    # Build FDCCEnv
    # ------------------------------------------------------------------
    rospy.init_node("evaluate_policy_pid", anonymous=False)
    logger.info("ROS node initialized")

    env = FDCCEnv(config=cfg, use_torch_for_cameras=False)

    # FDCCEnv.reset() asserts reference_trajectory is not None even though
    # it is unused during the actual reset movement — satisfy the check.
    env.reference_trajectory = []

    actions_as_deltas = env.actions_as_deltas
    logger.info(f"actions_as_deltas: {actions_as_deltas}")
    if actions_as_deltas:
        logger.warning(
            "env.actions_as_deltas is True but factored actions are absolute. "
            "FDCCEnv.prepare_action emits a quaternion target; make sure the "
            "env config has actions_as_deltas=false for factored-action checkpoints."
        )

    # ------------------------------------------------------------------
    # FT Visualizer
    # ------------------------------------------------------------------
    ft_visualizer = FTVisualizer(
        maxlen=max_timesteps,
        include_stiffness=True,
        force_ylim=(-5, 80),
        arm="right",
        figure_size=(5, 4),
        headless=True,
    )

    np.set_printoptions(linewidth=np.inf, formatter={"float": lambda x: f"{x:0.3f}"})
    torch.set_printoptions(linewidth=2000, sci_mode=False, precision=5)

    # ------------------------------------------------------------------
    # Move home once before evaluation
    # ------------------------------------------------------------------
    logger.info("Moving to home position...")
    env.go_home()

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    total_steps_per_episode = []
    force_violations = 0
    all_rollout_frames = []

    # Force-tracking metrics (computed only during contact)
    required_force_errors = []
    excess_force_errors = []
    force_tracking_errors = []

    eval_start_time = timeit.default_timer()
    total_steps_completed = 0

    def make_step_description(rollout_id: int, steps: int) -> str:
        elapsed = timeit.default_timer() - eval_start_time
        fps = steps / elapsed if elapsed > 0 else 0.0
        return f"Rollout {rollout_id + 1}/{num_rollouts} | FPS: {fps:.1f}"

    progress_columns = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=80),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    with Progress(*progress_columns, console=console) as progress:
        rollout_task = progress.add_task("Evaluation (PID compliance)", total=num_rollouts)

        for rollout_id in range(num_rollouts):
            step_task = progress.add_task(
                make_step_description(rollout_id, total_steps_completed),
                total=max_timesteps,
            )

            # Reset and wait for user confirmation
            env.reset(move_robot=True)
            progress.stop()
            input(f"\n  Rollout {rollout_id + 1}/{num_rollouts} — press Enter to start...")
            progress.start()

            env.activate_compliance_control()
            policy.reset()
            force_mapper.reset()

            desired_force = force_cfg.target_force
            logger.info(f"Rollout {rollout_id} | target force: {desired_force:.1f} N")

            episode_frames = []
            force_violation = False

            for t in range(max_timesteps):
                # --- Observe ---
                obs = format_real_robot_observations(
                    env.arm,
                    env.image_recorder,
                    features,
                    camera_shape,
                )

                policy_obs = {
                    k: v.cuda().float().unsqueeze(0) if isinstance(v, torch.Tensor) else v
                    for k, v in obs.items()
                }

                # --- Act (pose head from policy, compliance head from PI) ---
                with torch.no_grad():
                    action = policy.select_action(policy_obs)

                action = force_mapper.override_action(action, env.arm, dt)

                action_dict = {
                    k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
                    for k, v in action.items()
                }
                env_action = convert_policy_action(action_dict)

                # --- Step ---
                timestep = env.step(env_action)
                done = timestep.last()

                # --- Record images for video ---
                if save_video and env.image_recorder is not None:
                    raw_images = env.image_recorder.get_images()
                    if raw_images:
                        frames_list = list(raw_images.values())
                        frame = np.concatenate(frames_list, axis=1) if len(frames_list) > 1 else frames_list[0]
                        episode_frames.append(frame)

                # --- FT / force-tracking metrics ---
                wrench = env.arm.get_wrench()
                measured_force = float(np.linalg.norm(np.asarray(wrench)[:3]))
                is_contact = measured_force > force_cfg.contact_threshold
                if is_contact:
                    tracking_error = desired_force - measured_force
                    force_tracking_errors.append(abs(tracking_error))
                    if tracking_error > 0:
                        required_force_errors.append(tracking_error)
                        excess_force_errors.append(0.0)
                    else:
                        required_force_errors.append(0.0)
                        excess_force_errors.append(abs(tracking_error))
                else:
                    required_force_errors.append(0.0)
                    excess_force_errors.append(0.0)

                stiffness_val = action_dict["action.estimated_stiffness"]
                stiffness = float(
                    stiffness_val.cpu().item() if isinstance(stiffness_val, torch.Tensor) else stiffness_val
                )
                ft_visualizer.add_data(t, wrench, stiffness)

                total_steps_completed += 1
                progress.update(
                    step_task,
                    advance=1,
                    description=make_step_description(rollout_id, total_steps_completed),
                )

                if done:
                    force_violation = True
                    logger.info(
                        f"Rollout {rollout_id} ended early at step {t + 1} "
                        f"(force/torque limit exceeded)"
                    )
                    break

            env.deactivate_compliance_control()

            steps_taken = t + 1
            total_steps_per_episode.append(steps_taken)
            if force_violation:
                force_violations += 1

            logger.info(
                f"Rollout {rollout_id} complete | steps: {steps_taken} | "
                f"force violation: {force_violation}"
            )

            # --- Save per-rollout artefacts ---
            ft_visualizer.render_now()
            ft_visualizer.save(eval_dir / f"rollout_{rollout_id}.png")
            _, forces_data, _ = ft_visualizer.get_data()
            np.save(eval_dir / f"rollout_{rollout_id}_contact_force.npy", forces_data)
            ft_visualizer.clear()

            if save_video and episode_frames:
                save_to_video(episode_frames, eval_dir / "videos", f"rollout_{rollout_id}.mp4", control_frequency)
            if save_video:
                all_rollout_frames.extend(episode_frames)

            progress.remove_task(step_task)
            violation_pct = f"{force_violations / (rollout_id + 1) * 100:.1f}%"
            progress.update(
                rollout_task,
                advance=1,
                description=f"Evaluation | force violations: {violation_pct}",
            )

    env.go_home()
    logger.info("Returned to home position after evaluation.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"Rollouts completed:        {num_rollouts}")
    logger.info(
        f"Force violations:          {force_violations} / {num_rollouts} "
        f"({force_violations / num_rollouts * 100:.1f}%)"
    )
    logger.info(f"Mean steps per episode:    {np.mean(total_steps_per_episode):.1f}")
    logger.info(f"Std  steps per episode:    {np.std(total_steps_per_episode):.1f}")
    if required_force_errors:
        logger.info(f"Mean required force error: {np.mean(required_force_errors):.2f} N")
        logger.info(f"Mean excess force error:   {np.mean(excess_force_errors):.2f} N")
    if force_tracking_errors:
        logger.info(
            f"Mean |force tracking error| during contact: "
            f"{np.mean(force_tracking_errors):.2f} N"
        )
    logger.info("=" * 60)

    if save_video and all_rollout_frames:
        save_to_video(all_rollout_frames, eval_dir, "evaluation_all_rollouts.mp4", control_frequency)

    ft_visualizer.close()
    logger.info(f"Evaluation complete. Results saved to: {eval_dir}")


if __name__ == "__main__":
    if "hydra.run.dir" not in " ".join(sys.argv):
        sys.argv.append("paths.script=/eval")
    main()
