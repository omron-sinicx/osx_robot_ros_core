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

    # Async inference:
    python evaluate_policy.py eval.inference.mode=async

    # DMP force execution layer (dry run first):
    python evaluate_policy.py eval.force_layer.enabled=true eval.force_layer.dry_run=true

    # On-demand force via ROS topic (external mode):
    rostopic pub /comet/target_force std_msgs/Float32 "data: 25.0" -r 5

Controls during each rollout:
    Enter  - confirm start of rollout (after reset prompt)
    q      - stop current rollout and end evaluation
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
    format_real_robot_observations,
    setup_logging,
    tensor_dict_to_numpy,
)

logger = logging.getLogger(__name__)

install_sigint_handler()


def _to_env_action(adapter, action_dict, actions_as_deltas):
    if adapter.trainer == "baseline":
        return tensor_dict_to_numpy(action_dict)
    return convert_policy_action(action_dict, actions_as_deltas)


@hydra.main(
    version_base=None,
    config_path="../../../../../dependencies/comet/configs",
    config_name="book_flipping",
)
def main(cfg: DictConfig) -> None:
    rospy.init_node("evaluate_policy")
    stop_events = {"stop": False}
    kb_listener = start_stop_listener(stop_events)
    try:
        policy_filename = OmegaConf.select(cfg, "eval.policy_filename", default="best_ema_policy.ckpt")
        adapter = build_policy_adapter(cfg)
        adapter.load(Path(cfg.eval.base.load_ckpt), policy_filename)

        env = FDCCEnv(config=adapter.env_config)

        runner = RealRobotEvalRunner(
            cfg,
            adapter,
            env,
            format_obs=format_real_robot_observations,
            to_env_action=lambda action_dict, actions_as_deltas: _to_env_action(
                adapter, action_dict, actions_as_deltas
            ),
            setup_logging_fn=setup_logging,
            stop_events=stop_events,
        )
        runner.run()
    finally:
        kb_listener.stop()
        logger.info("Keyboard listener stopped.")


if __name__ == "__main__":
    if "hydra.run.dir" not in " ".join(sys.argv):
        sys.argv.append("paths.script=/eval")
    main()
