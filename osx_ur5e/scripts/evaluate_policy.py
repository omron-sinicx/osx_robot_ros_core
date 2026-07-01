#!/usr/bin/env python3

"""Evaluate a COMET diffusion policy on the real UR5e via FDCCEnv.

Usage:
    python evaluate_policy.py

    # Override eval settings:
    python evaluate_policy.py eval.num_rollouts=5 eval.max_timesteps=500

    # Point to a different env config (required for first run):
    python evaluate_policy.py +eval.env_config=/path/to/data_collection.yaml

Controls during each rollout:
    Enter  - confirm start of rollout (after reset prompt)
    q      - stop current rollout and end evaluation
"""

import datetime
import logging
import signal
import sys
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision import transforms

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import rospy

from tqdm import tqdm
from pynput import keyboard as kb

from ur_control import transformations

from comet.common.utils.utils import load_base_policy
from comet.common.utils.ft_visualizer import FTVisualizer
from comet.common.utils.viz_utils import save_to_video
from comet.common.policies.types import FeatureType
from comet.common.policies.guidance_utils import setup_guidance, feed_force_to_guidance
from comet.inference import AsyncInferenceEngine, build_inference_engine
from comet.inference.utils import split_action_tensor, to_cuda_batch

from osx_ur5e.fdcc_env import FDCCEnv
from osx_ur5e.debug_utils import install_horizon_logger
from osx_ur5e.utils import convert_policy_action, format_real_robot_observations, setup_logging
from comet.scripts.utils.visualize_episode import plot_factored_from_arrays, plot_virtual_from_arrays


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def _signal_handler(sig, frame):
    logger.info("Interrupted — shutting down.")
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)


# ---------------------------------------------------------------------------
# Keyboard listener
# ---------------------------------------------------------------------------

def start_stop_listener(events: dict):
    """Start a background keyboard listener; 'q' sets events['stop'] = True."""

    def on_press(key):
        try:
            if key.char and key.char.lower() == "q":
                events["stop"] = True
        except AttributeError:
            pass

    listener = kb.Listener(on_press=on_press)
    listener.start()
    return listener


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
    stop_events = {"stop": False}
    kb_listener = start_stop_listener(stop_events)
    try:
        main_loop(cfg, stop_events)
    finally:
        kb_listener.stop()
        logger.info("Keyboard listener stopped.")


def main_loop(cfg: DictConfig, stop_events: dict) -> None:
    eval_dir = Path(cfg.eval.base.load_ckpt) / "eval" / str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    eval_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg.eval.seed
    num_rollouts = cfg.eval.num_rollouts
    max_timesteps = cfg.eval.max_timesteps
    save_video = OmegaConf.select(cfg, "eval.save_video", default=False)
    log_horizons = OmegaConf.select(cfg, "eval.log_horizons", default=False)
    policy_filename = OmegaConf.select(cfg, "eval.policy_filename", default="best_ema_policy.ckpt")

    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info(f"Seed: {seed}")
    logger.info(f"Results will be saved to: {eval_dir}")

    # ------------------------------------------------------------------
    # Load policy
    # ------------------------------------------------------------------
    ckpt_dir = Path(cfg.eval.base.load_ckpt)
    logger.info(f"Loading policy from: {ckpt_dir}")
    policy, base_cfg, features = load_base_policy(ckpt_dir, policy_filename)
    policy.cuda()
    policy.eval()

    action_keys = {key: ft.shape for key, ft in features.items() if ft.type is FeatureType.ACTION}
    logger.info(f"Action keys: {list(action_keys.keys())}")
    logger.info(f"Observation keys: {[k for k, ft in features.items() if ft.type is not FeatureType.ACTION]}")

    # Camera shape expected by the policy (H, W)
    if hasattr(base_cfg, "model_specific") and "camera_shape" in base_cfg.model_specific:
        camera_shape = tuple(base_cfg.model_specific.camera_shape[-2:])
    else:
        camera_shape = (240, 320)
        logger.warning(f"No camera_shape in checkpoint config, defaulting to {camera_shape}")

    control_frequency = int(base_cfg.dataset.dataset.fps)
    logger.info(f"Control frequency: {control_frequency} Hz")

    inference_engine = build_inference_engine(policy, cfg, control_frequency)
    inference_mode = OmegaConf.select(cfg, "eval.inference.mode", default="sync")
    logger.info(f"Inference mode: {inference_mode}")
    inference_engine.start()

    # ------------------------------------------------------------------
    # Guidance setup
    # ------------------------------------------------------------------
    setup_guidance(policy, cfg, base_cfg, features, control_frequency)

    # ------------------------------------------------------------------
    # Horizon prediction logger (optional)
    # ------------------------------------------------------------------
    if log_horizons:
        horizon_log_dir = eval_dir / "horizons"
        install_horizon_logger(policy, horizon_log_dir)
        logger.info(f"Horizon logging enabled → {horizon_log_dir}")

    # ------------------------------------------------------------------
    # Load env config and build FDCCEnv
    # ------------------------------------------------------------------
    rospy.init_node("evaluate_policy", anonymous=False)

    hydra_dir = Path(HydraConfig.get().runtime.output_dir)
    hydra_job_log = hydra_dir / f"{HydraConfig.get().job.name}.log"
    eval_log = eval_dir / "evaluation.log"
    setup_logging(hydra_job_log, eval_log)
    logger.info("ROS node initialized")
    logger.info("Log files: %s , %s", hydra_job_log, eval_log)
    logger.info(f"Eval artifacts: {eval_dir}")

    env = FDCCEnv(config=base_cfg, use_torch_for_cameras=False)

    # FDCCEnv.reset() asserts reference_trajectory is not None even though
    # it is unused during the actual reset movement — satisfy the check.
    env.reference_trajectory = []

    actions_as_deltas = env.actions_as_deltas
    logger.info(f"actions_as_deltas: {actions_as_deltas}")

    # ------------------------------------------------------------------
    # FT Visualizer
    # ------------------------------------------------------------------
    include_stiffness = any("stiffness" in k for k in action_keys)
    ft_visualizer = FTVisualizer(
        maxlen=max_timesteps,
        include_stiffness=include_stiffness,
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

    eval_start_time = timeit.default_timer()
    total_steps_completed = 0

    rollout_bar = tqdm(range(num_rollouts), desc="Evaluation", unit="rollout")
    for rollout_id in rollout_bar:
        # Reset and wait for user confirmation
        inference_engine.pause()
        env.reset(move_robot=False)
        input(f"\n  Rollout {rollout_id + 1}/{num_rollouts} — press Enter to start...")

        env.activate_compliance_control()
        inference_engine.reset()
        inference_engine.resume()

        if isinstance(inference_engine, AsyncInferenceEngine):
            # BG thread waits for obs_holder before inferring; seed one observation
            # here so cold-start wait can complete before the control loop runs.
            bootstrap_obs = format_real_robot_observations(
                env.arm,
                env.image_recorder,
                features,
                camera_shape,
            )
            inference_engine.notify_observation(to_cuda_batch(bootstrap_obs))
            logger.info("Waiting for first inference chunk (cold start / GPU warmup)...")
            if not inference_engine.wait_for_action(timeout_s=60.0):
                if inference_engine.failed:
                    logger.error("Background inference thread failed on first chunk — skipping rollout.")
                else:
                    logger.error("Timed out waiting for first inference chunk — skipping rollout.")
                inference_engine.pause()
                continue
            logger.info("First chunk ready — starting rollout.")

        episode_frames = []
        force_violation = False
        logged_obs_eef = []
        logged_actions = []

        step_bar = tqdm(range(max_timesteps), desc=f"Rollout {rollout_id + 1}/{num_rollouts}", unit="step", leave=False)
        for t in step_bar:
            if stop_events["stop"]:
                logger.info("'q' pressed — stopping rollout early.")
                inference_engine.pause()
                break

            # --- Observe ---
            obs = format_real_robot_observations(
                env.arm,
                env.image_recorder,
                features,
                camera_shape,
            )

            policy_obs = to_cuda_batch(obs)

            if isinstance(inference_engine, AsyncInferenceEngine):
                inference_engine.notify_observation(policy_obs)
                action_tensor = inference_engine.get_action(None)
            else:
                action_tensor = inference_engine.get_action(policy_obs)

            if action_tensor is None:
                if inference_engine.failed:
                    logger.error("Inference engine failed; ending rollout early.")
                    break
                rospy.sleep(1.0 / control_frequency)
                continue

            action = split_action_tensor(
                action_tensor,
                policy.config.action_features,
                policy.output_sizes,
            )
            if False:
                # action["action.normal_force"] = torch.tensor([10], device="cuda")
                print("will edit the force")
                target_force = 5.0
                action["action.normal_force"] = torch.tensor([target_force], device="cuda")
                f_low = 1.0
                f_high = 10.0
                K_max = 1000
                K_min = 500.0
                max_displacement = 0.015

                f_norm = abs(target_force)
                if f_norm < f_low:
                    K = K_max
                elif f_norm > f_high:
                    K = K_min
                else:
                    K_gap = K_max - K_min
                    f_gap = f_high - f_low
                    K = K_max - K_gap * (f_norm - f_low) / f_gap

                if max_displacement > 0 and f_norm > f_low:
                    k_floor = f_norm / max_displacement
                    print(f"{k_floor=}")
                    K = np.clip(k_floor, K_min, K_max)

                print(f"{K=}")
                action["action.estimated_stiffness"] = torch.tensor([K], device="cuda")

            action_dict = {  # FIXME: why do this if it exist in the fucntion below as well (convert_policy_action)?
                k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
                for k, v in action.items()
            }

            logged_actions.append({
                k: v.cpu().numpy() if isinstance(v, torch.Tensor) else np.array(v)
                for k, v in action_dict.items()
            })

            env_action = convert_policy_action(action_dict, actions_as_deltas)

            # --- Step ---
            timestep = env.step(env_action)
            done = timestep.last()

            # --- Feed force measurement back to guidance ---
            wrench = env.arm.get_wrench()
            feed_force_to_guidance(policy, np.linalg.norm(wrench[:3]))

            # --- Record images for video ---
            if save_video and env.image_recorder is not None:
                raw_images = env.image_recorder.get_images()
                if raw_images:
                    frames_list = list(raw_images.values())
                    frame = np.concatenate(frames_list, axis=1) if len(frames_list) > 1 else frames_list[0]
                    episode_frames.append(frame)

            # --- FT tracking (wrench already fetched above for guidance) ---
            stiffness = None
            if include_stiffness:
                if "action.stiffness_diag" in action_dict:
                    val = action_dict["action.stiffness_diag"]
                    stiffness = float(np.mean(val.cpu().numpy() if isinstance(val, torch.Tensor) else val))
                elif "action.estimated_stiffness" in action_dict:
                    val = action_dict["action.estimated_stiffness"]
                    stiffness = float(val.cpu().item() if isinstance(val, torch.Tensor) else val)
            ft_visualizer.add_data(t, wrench, stiffness)

            total_steps_completed += 1
            elapsed = timeit.default_timer() - eval_start_time
            fps = total_steps_completed / elapsed if elapsed > 0 else 0.0
            step_bar.set_postfix(fps=f"{fps:.1f}")

            if done:
                force_violation = True
                logger.info(
                    f"Rollout {rollout_id} ended early at step {t + 1} "
                    f"(force/torque limit exceeded)"
                )
                break

        step_bar.close()

        # While still compliant, move to the home Cartesian pose so the robot lifts
        # off the surface before the controller switch (avoids force spike on deactivation)
        env.move_to_home(timeout=5.0)

        # NOW safe to switch — robot is at home and clear of the surface
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

        # --- Save 3D trajectory plot ---
        try:
            obs_arr = np.stack(logged_obs_eef)
            act = {k: np.stack([a[k] for a in logged_actions])
                   for k in logged_actions[0]}

            if "action.contact_direction" in act:
                traj_fig = plot_factored_from_arrays(
                    obs_positions=obs_arr,
                    ref_positions=act["action.ref_position"],
                    contact_directions=act["action.contact_direction"],
                    normal_forces=act["action.normal_force"].flatten(),
                    stiffnesses=act["action.estimated_stiffness"].flatten(),
                    title=f"Rollout {rollout_id} — Factored Trajectory",
                )
            elif "action.virtual_target_position" in act:
                traj_fig = plot_virtual_from_arrays(
                    obs_positions=obs_arr,
                    vt_positions=act["action.virtual_target_position"],
                    stiffnesses=act["action.estimated_stiffness"].flatten(),
                    ref_positions=act.get("action.ref_position"),
                    title=f"Rollout {rollout_id} — Virtual Displacement Trajectory",
                )
            else:
                traj_fig = None

            if traj_fig is not None:
                traj_fig.write_html(eval_dir / f"rollout_{rollout_id}_trajectory.html")
                logger.info(f"Saved trajectory plot: rollout_{rollout_id}_trajectory.html")

            # Save rollout data as CSV
            csv_data = {
                "obs.eef.position[0]": obs_arr[:, 0],
                "obs.eef.position[1]": obs_arr[:, 1],
                "obs.eef.position[2]": obs_arr[:, 2],
            }
            for key, arr in act.items():
                if arr.ndim == 1:
                    csv_data[key] = arr
                else:
                    for d in range(arr.shape[1]):
                        csv_data[f"{key}[{d}]"] = arr[:, d]
            pd.DataFrame(csv_data).to_csv(
                eval_dir / f"rollout_{rollout_id}_trajectory.csv", index_label="step")
            logger.info(f"Saved trajectory CSV: rollout_{rollout_id}_trajectory.csv")
        except Exception as e:
            logger.warning(f"Failed to save trajectory plot/csv: {e}")

        if save_video and episode_frames:
            save_to_video(episode_frames, eval_dir / "videos", f"rollout_{rollout_id}.mp4", control_frequency)
        if save_video:
            all_rollout_frames.extend(episode_frames)

        violation_pct = f"{force_violations / (rollout_id + 1) * 100:.1f}%"
        rollout_bar.set_postfix(violations=violation_pct)
    env.go_home()
    inference_engine.stop()
    logger.info("Returned to home position after evaluation.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"Rollouts completed:        {num_rollouts}")
    logger.info(f"Force violations:          {force_violations} / {num_rollouts} "
                f"({force_violations / num_rollouts * 100:.1f}%)")
    logger.info(f"Mean steps per episode:    {np.mean(total_steps_per_episode):.1f}")
    logger.info(f"Std  steps per episode:    {np.std(total_steps_per_episode):.1f}")
    logger.info("=" * 60)

    if save_video and all_rollout_frames:
        save_to_video(all_rollout_frames, eval_dir, "evaluation_all_rollouts.mp4", control_frequency)

    ft_visualizer.close()
    logger.info(f"Evaluation complete. Results saved to: {eval_dir}")


if __name__ == "__main__":
    if "hydra.run.dir" not in " ".join(sys.argv):
        sys.argv.append("paths.script=/eval")
    main()
