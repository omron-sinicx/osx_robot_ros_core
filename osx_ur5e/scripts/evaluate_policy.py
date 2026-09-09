#!/usr/bin/env python3

"""Evaluate a COMET or baseline policy on the real UR5e via FDCCEnv.

Dispatches on ``model.trainer`` (``comet`` | ``baseline``) from the Hydra config.

Usage:
    python evaluate_policy.py

    # COMET
    python evaluate_policy.py model=comet eval.base.ckpt_name=comet/train/bf_fvt_minmax

    # Baseline ACT
    python evaluate_policy.py model=lerobot_act eval.base.ckpt_name=lerobot_act/train/fvt_mean_std

    # Override eval settings:
    python evaluate_policy.py eval.num_rollouts=5 eval.max_timesteps=500

    # Resume a crashed eval in its original directory (continues at the next
    # rollout id from results.json; stats recomputed over old + new rollouts):
    python evaluate_policy.py eval.resume_dir=/path/to/ckpt/eval/2026-07-27_10-52-01

    # Async inference:
    python evaluate_policy.py eval.inference.mode=async

    # DMP force execution layer (dry run first):
    python evaluate_policy.py eval.force_layer.enabled=true eval.force_layer.dry_run=true

    # On-demand force via ROS topic (external mode):
    rostopic pub /comet/target_force std_msgs/Float32 "data: 25.0" -r 5

Controls during each rollout (read from this terminal; when the process has no
terminal they fall back to a global X11 hook, which only sees keys delivered to
$DISPLAY -- override with COMET_STOP_KEYS=stdin|x11|off):
    Enter  - confirm start of rollout (after reset prompt)
    q      - early-stop current rollout, record, and continue to the next
    r      - restart current rollout: discard the attempt (no artifacts, no
             record) and rerun the same rollout id after a scene reset
    y/n    - after each rollout, label it success/failure (an optional note may
             follow, e.g. "n slipped off the edge"); labels land in results.json
             and MLflow
"""

from __future__ import annotations

import logging
import sys
import rospy
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from comet.eval import RealRobotEvalRunner, build_policy_adapter
from comet.eval.io import install_sigint_handler, start_stop_listener
from osx_ur5e.fdcc_env import FDCCEnv
from osx_ur5e.utils import (
    convert_policy_action,
    convert_raw_baseline_action,
    format_real_robot_observations,
    setup_logging,
    tensor_dict_to_numpy,
)

logger = logging.getLogger(__name__)

install_sigint_handler()


def _to_env_action(adapter, action_dict, actions_as_deltas):
    if adapter.trainer == "baseline":
        if adapter.action_repr == "raw":
            return convert_raw_baseline_action(action_dict, actions_as_deltas)
        return tensor_dict_to_numpy(action_dict)
    return convert_policy_action(action_dict, actions_as_deltas)


@hydra.main(
    version_base=None,
    config_path="../../../../../../dependencies/comet/configs",
    config_name="blank",
)
def main(cfg: DictConfig) -> None:
    rospy.init_node("evaluate_policy")
    stop_events = {"stop": False, "restart": False}
    kb_listener = start_stop_listener(stop_events)
    try:
        policy_filename = OmegaConf.select(cfg, "eval.policy_filename", default="best_ema_policy.ckpt")
        adapter = build_policy_adapter(cfg)
        adapter.load(Path(cfg.eval.base.load_ckpt), policy_filename)

        env = FDCCEnv(config=adapter.env_config)

        rollout_recorder = None
        if OmegaConf.select(cfg, "eval.record_rosbag", default=False):
            from osx_ur5e.rosbag_recorder import RosbagRecorder

            extra_topics = list(
                OmegaConf.select(cfg, "eval.rosbag_extra_topics", default=[]) or []
            )
            topics = RosbagRecorder.default_topics(env.feeder, extra_topics)
            rollout_recorder = RosbagRecorder(topics)
            rospy.loginfo(f"Rosbag rollout recording enabled: {topics}")

        scene_snapshotter = None
        if OmegaConf.select(cfg, "eval.marker_wipe.enabled", default=False):
            from osx_ur5e.marker_snapshotter import SceneMarkerSnapshotter

            scene_snapshotter = SceneMarkerSnapshotter(
                env.feeder,
                camera=OmegaConf.select(cfg, "eval.marker_wipe.camera", default="wrist_camera"),
            )
            rospy.loginfo(
                f"Marker-wipe snapshots enabled (camera={scene_snapshotter.camera}); "
                "snapshots are saved per rollout, scoring is post-hoc via "
                "comet/scripts/utils/marker_wipe.py"
            )

        rospy.loginfo(f"Pad to square: {adapter.pad_to_square}")
        if adapter.pad_to_square:
            rospy.loginfo("Images: padding to square before resizing (matches training)")

        def _format_obs(arm, feeder, features_or_keys, camera_shape):
            return format_real_robot_observations(
                arm,
                feeder,
                features_or_keys,
                camera_shape,
                pad_to_square=adapter.pad_to_square,
            )

        runner = RealRobotEvalRunner(
            cfg,
            adapter,
            env,
            format_obs=_format_obs,
            to_env_action=lambda action_dict, actions_as_deltas: _to_env_action(
                adapter, action_dict, actions_as_deltas
            ),
            setup_logging_fn=setup_logging,
            stop_events=stop_events,
            key_listener=kb_listener,
            rollout_recorder=rollout_recorder,
            scene_snapshotter=scene_snapshotter,
        )
        runner.run()
    finally:
        # Always restores the terminal mode the stdin backend took.
        kb_listener.stop()
        logger.info("Keyboard listener stopped.")


if __name__ == "__main__":
    # eval.resume_dir=<existing eval dir> reruns in place: the runner picks up
    # prior rollouts from its results.json and continues the numbering.
    resume_dir = next(
        (arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("eval.resume_dir=")),
        None,
    )
    if resume_dir in ("", "null", "None"):
        resume_dir = None
    if "hydra.run.dir" not in " ".join(sys.argv):
        # Everything (hydra files, logs, eval artifacts) lands under the checkpoint.
        run_dir = resume_dir or "${eval.base.load_ckpt}/eval/${now:%Y-%m-%d_%H-%M-%S}"
        sys.argv.append(f"hydra.run.dir={run_dir}")
    main()
