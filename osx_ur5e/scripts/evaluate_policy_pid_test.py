#!/usr/bin/env python3
"""Evaluate a COMET diffusion policy with Z-axis force PID override.

Unlike ``evaluate_policy_pid.py`` which overrides the factored compliance
outputs (normal_force, normal_torque, stiffness) along the learned contact
direction, this script:

1. Lets the policy predict the full factored action (pose + compliance).
2. Converts it to a virtual-target + directional-stiffness command via
   ``process_factored_action_dict`` (preserving the policy's XY motion
   intent, contact direction, and stiffness structure).
3. Applies a closed-loop PI controller **only** on the surface-normal axis
   (default: world Z) of the virtual-target position, so that the
   downward contact force tracks a desired setpoint while the tangential
   (XY) sliding motion remains fully policy-controlled.

This decouples "how hard to push" from "where to slide", which the original
all-direction override conflated through the contact_direction vector.

Usage:
    python evaluate_policy_pid_test.py

    # Override target Z-force:
    python evaluate_policy_pid_test.py eval.pid.target_force=20.0

Controls during each rollout:
    Enter  - confirm start of rollout (after reset prompt)
"""

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
from comet.common.utils.vt_utils import process_factored_action_dict
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
# Z-axis force PI controller
# ---------------------------------------------------------------------------

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


@dataclass
class ZForceConfig:
    """Tunables for the surface-normal (Z-axis) force controller."""
    target_force: float = 15.0

    # Which world-frame axis is the surface normal, and the push direction.
    # axis=2 + sign=-1 means push in -Z (downward onto a horizontal table).
    surface_axis: int = 2
    surface_push_sign: float = -1.0

    # PI gains
    kp_force: float = 0.5
    ki_force: float = 0.03
    max_integral: float = 70.0

    # Contact detection
    contact_threshold: float = 3.5
    force_ema_alpha: float = 0.3

    # Safety clamps
    max_force_correction: float = 50.0   # N — caps the PI output
    max_vt_offset: float = 0.05          # m — max virtual-target offset from actual

    # When True, PID only kicks in after contact is detected; before that
    # the policy's full virtual target is passed through unchanged.
    override_only_in_contact: bool = True

    debug_log_every: int = 10


class ZAxisForceController:
    """Closed-loop force controller acting on a single world-frame axis.

    Operates on the *controller-ready* action dict (virtual-target position +
    orientation + stiffness diagonal) that ``process_factored_action_dict``
    already produced. Only the surface-axis component of the virtual-target
    position is modified; everything else is left as the policy predicted.
    """

    def __init__(self, cfg: ZForceConfig):
        self.cfg = cfg
        self.pi = ForceTrackingPI(cfg.kp_force, cfg.ki_force, cfg.max_integral)
        self._force_ema: Optional[float] = None
        self._call_count: int = 0

    def reset(self):
        self.pi.reset()
        self._force_ema = None
        self._call_count = 0

    def smooth_force(self, raw: float) -> float:
        if self._force_ema is None:
            self._force_ema = raw
        else:
            a = self.cfg.force_ema_alpha
            self._force_ema = a * raw + (1.0 - a) * self._force_ema
        return self._force_ema

    def override_controller_action(
        self,
        controller_action: dict,
        arm,
        dt: float,
    ) -> tuple[dict, dict]:
        """Modify only the surface-axis virtual-target to track the desired force.

        Parameters
        ----------
        controller_action : dict
            Must contain ``action.position`` (3,), ``action.orientation`` (4,),
            ``action.stiffness_diag`` (6,) — the output of
            ``process_factored_action_dict``.
        arm : CompliantController
            Live robot handle for reading EEF pose and wrench.
        dt : float
            Control timestep (seconds).

        Returns
        -------
        controller_action : dict
            Same dict with ``action.position[surface_axis]`` potentially modified.
        diag : dict
            Diagnostic data for logging.
        """
        c = self.cfg
        ax = c.surface_axis

        eef_pose = arm.end_effector()
        x_actual = np.asarray(eef_pose[:3], dtype=np.float64)
        eef_quat = np.asarray(eef_pose[3:], dtype=np.float64)

        wrench_raw = np.asarray(arm.get_wrench(), dtype=np.float64).reshape(-1)
        R_eef = transformations.rotation_matrix_from_quaternion(eef_quat)[:3, :3]
        f_world = R_eef @ wrench_raw[:3]

        # Measured push-force along surface axis.
        # Sensor reads reaction (opposite of push direction), so negate with
        # push_sign to get a positive value when the robot is pushing.
        f_push_raw = float(-c.surface_push_sign * f_world[ax])
        f_push = self.smooth_force(max(f_push_raw, 0.0))
        in_contact = f_push > c.contact_threshold

        K_ax = float(controller_action["action.stiffness_diag"][ax])
        K_ax = max(K_ax, 50.0)

        vt_pos = controller_action["action.position"].copy()
        vt_policy = float(vt_pos[ax])

        if in_contact or not c.override_only_in_contact:
            # Feedforward: set virtual target so FDCC generates desired force.
            # FDCC: F_ax = K * (vt_ax - actual_ax)
            # Desired: F_ax = push_sign * target_force
            # => vt_ax = actual_ax + push_sign * target_force / K
            vt_ff = x_actual[ax] + c.surface_push_sign * c.target_force / K_ax

            # PI feedback correction
            if in_contact:
                error = c.target_force - f_push
                correction = self.pi.update(error, dt)
                correction = float(np.clip(correction, -c.max_force_correction, c.max_force_correction))
            else:
                error = 0.0
                correction = 0.0

            vt_ax = vt_ff + c.surface_push_sign * correction / K_ax

            # Safety: clamp offset from current actual position
            vt_ax = float(np.clip(
                vt_ax,
                x_actual[ax] - c.max_vt_offset,
                x_actual[ax] + c.max_vt_offset,
            ))

            vt_pos[ax] = vt_ax
        else:
            error = 0.0
            correction = 0.0

        controller_action["action.position"] = vt_pos

        diag = {
            "f_world_ax": float(f_world[ax]),
            "f_push_raw": f_push_raw,
            "f_push": f_push,
            "in_contact": in_contact,
            "K_ax": K_ax,
            "vt_policy": vt_policy,
            "vt_final": float(vt_pos[ax]),
            "x_actual_ax": float(x_actual[ax]),
            "error": error,
            "correction": correction,
            "integral": self.pi._integral,
        }

        self._call_count += 1
        if c.debug_log_every and (self._call_count % c.debug_log_every) == 0:
            logger.info(
                "[z-pid] t=%04d | contact=%s | F_push=%.2f (raw=%.2f) "
                "| K=%.0f | err=%+.2f corr=%+.2f int=%+.2f "
                "| vt_policy=%.4f vt_final=%.4f actual=%.4f",
                self._call_count,
                in_contact,
                f_push,
                f_push_raw,
                K_ax,
                error,
                correction,
                self.pi._integral,
                vt_policy,
                float(vt_pos[ax]),
                float(x_actual[ax]),
            )

        return controller_action, diag


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
    """Strip batch dimension and convert tensors to numpy."""
    env_action = {}
    for key, value in action_dict.items():
        if isinstance(value, torch.Tensor):
            env_action[key] = value.squeeze(0).cpu().numpy() if value.dim() > 0 else value.cpu().numpy()
        else:
            env_action[key] = np.array(value)
    return env_action


def factored_to_controller(
    env_action: dict,
    default_stiffness: float,
    default_stiffness_rot: float,
) -> dict:
    """Run ``process_factored_action_dict`` and unpack into a controller dict.

    Returns a dict with ``action.position`` (3,), ``action.orientation`` (4,),
    and ``action.stiffness_diag`` (6,).
    """
    fvt = process_factored_action_dict(
        env_action,
        default_stiffness=default_stiffness,
        default_stiffness_rot=default_stiffness_rot,
        characteristic_length=0.1,
        use_isotropic_stiffness=False,
        controller_type="variable_kp",
        orientation_representation="quaternion",
    )
    return {
        "action.position": fvt[0:3],
        "action.orientation": fvt[3:7],
        "action.stiffness_diag": fvt[7:13],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(
    version_base=None,
    config_path="../../../../../dependencies/comet/configs",
    config_name="blackboard_wipe",
)
def main(cfg: DictConfig) -> None:
    eval_dir = Path(cfg.eval.base.load_ckpt) / "eval" / str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    eval_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(eval_dir / "evaluation_z_pid.log")

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
    # Load policy (all heads used; Z-axis PID acts post-factored)
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
    # Build the Z-axis force controller (reads from eval.pid section)
    # ------------------------------------------------------------------
    pid_cfg = cfg.eval.pid

    z_force_cfg = ZForceConfig(
        target_force=pid_cfg.target_force,
        surface_axis=int(OmegaConf.select(cfg, "eval.pid.surface_axis", default=2)),
        surface_push_sign=float(OmegaConf.select(cfg, "eval.pid.surface_push_sign", default=-1.0)),
        kp_force=pid_cfg.kp_force,
        ki_force=pid_cfg.ki_force,
        max_integral=pid_cfg.max_integral,
        contact_threshold=pid_cfg.contact_threshold,
        force_ema_alpha=pid_cfg.force_ema_alpha,
        max_force_correction=float(OmegaConf.select(cfg, "eval.pid.max_force_correction", default=50.0)),
        max_vt_offset=float(OmegaConf.select(cfg, "eval.pid.max_vt_offset", default=0.05)),
        override_only_in_contact=bool(OmegaConf.select(cfg, "eval.pid.override_only_in_contact", default=True)),
        debug_log_every=pid_cfg.debug_log_every,
    )
    z_force_ctrl = ZAxisForceController(z_force_cfg)

    logger.info(
        "Z-axis PID: target=%.1f N, axis=%d, push_sign=%.0f, Kp=%.3f, Ki=%.3f",
        z_force_cfg.target_force,
        z_force_cfg.surface_axis,
        z_force_cfg.surface_push_sign,
        z_force_cfg.kp_force,
        z_force_cfg.ki_force,
    )

    # ------------------------------------------------------------------
    # Build FDCCEnv
    # ------------------------------------------------------------------
    rospy.init_node("evaluate_policy_pid_test", anonymous=False)
    logger.info("ROS node initialized")

    env = FDCCEnv(config=cfg, use_torch_for_cameras=False)
    env.reference_trajectory = []

    actions_as_deltas = env.actions_as_deltas
    logger.info(f"actions_as_deltas: {actions_as_deltas}")
    if actions_as_deltas:
        logger.warning(
            "env.actions_as_deltas is True but factored actions are absolute. "
            "Make sure the env config has actions_as_deltas=false."
        )

    default_stiffness = float(env.controller_config.stiffness)
    default_stiffness_rot = float(env.controller_config.stiffness)

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
        rollout_task = progress.add_task("Evaluation (Z-axis PID)", total=num_rollouts)

        for rollout_id in range(num_rollouts):
            step_task = progress.add_task(
                make_step_description(rollout_id, total_steps_completed),
                total=max_timesteps,
            )

            env.reset(move_robot=True)
            progress.stop()
            input(f"\n  Rollout {rollout_id + 1}/{num_rollouts} — press Enter to start...")
            progress.start()

            env.activate_compliance_control()
            policy.reset()
            z_force_ctrl.reset()

            logger.info(
                "Rollout %d | target Z-force: %.1f N (axis=%d, sign=%.0f)",
                rollout_id, z_force_cfg.target_force,
                z_force_cfg.surface_axis, z_force_cfg.surface_push_sign,
            )

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

                # --- Policy predicts full factored action (no override) ---
                with torch.no_grad():
                    action = policy.select_action(policy_obs)

                action_dict = {
                    k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
                    for k, v in action.items()
                }
                env_action = convert_policy_action(action_dict)

                # --- Convert factored → controller-ready (virtual target + stiffness) ---
                controller_action = factored_to_controller(
                    env_action,
                    default_stiffness=default_stiffness,
                    default_stiffness_rot=default_stiffness_rot,
                )

                # --- Z-axis PID: override only surface-axis of virtual target ---
                controller_action, diag = z_force_ctrl.override_controller_action(
                    controller_action, env.arm, dt,
                )

                # --- Step (controller_action has position+orientation+stiffness,
                #     so FDCCEnv.prepare_action passes it through unchanged) ---
                timestep = env.step(controller_action)
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
                is_contact = measured_force > z_force_cfg.contact_threshold
                if is_contact:
                    tracking_error = z_force_cfg.target_force - diag["f_push"]
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

                stiffness_val = diag["K_ax"]
                ft_visualizer.add_data(t, wrench, stiffness_val)

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
