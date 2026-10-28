#!/usr/bin/env python3
"""Train an ACT-family policy on a local LeRobotDataset via fluxtrain.

Sibling to train_point_policy.py (which uses stock lerobot-train); this
script trains through fluxtrain's faster Trainer instead
(dependencies/fluxtrain), loading one of dependencies/fluxtrain/examples/*.yaml
as the base TrainConfig and overriding the per-task fields (repo_id, root,
run_name, ...). Any other config field can be tuned from the command line the
same way fluxtrain's own CLI works: pass dotted `section.key=value` overrides.

Reads the local dataset directory produced by
Point-Policy/ur5e_pipeline/run_pipeline.sh's step 7
(convert_pkl_human_to_robot.py -> convert_pkl_to_lerobot.py) -- repo_id is
still required by the config but is never resolved against the Hub once
a local root is set. push_to_hub=False keeps the checkpoint local too.

Usage:
    python train_fluxtrain.py bottle_open_06
    python train_fluxtrain.py bottle_open_06 --wandb_enable false
    python train_fluxtrain.py bottle_open_06 optim.lr=1e-4 cache.level=jpeg
    python train_fluxtrain.py bottle_open_06 --config dependencies/fluxtrain/examples/act.yaml
"""

import argparse
from pathlib import Path

# Registers the "custom_act" policy config in lerobot's PreTrainedConfig
# registry.
from lerobot_policy_custom_act.configuration_custom_act import ACTConfig as CustomACTConfig

OSX_UR_ROOT = Path(__file__).resolve().parents[5]  # scripts/osx_ur5e/osx_core/src/catkin_ws/osx-ur
DEFAULT_CONFIG = OSX_UR_ROOT / "dependencies" / "fluxtrain" / "examples" / "act_point_jitter.yaml"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("task_name", help="Task name -- must match the LeRobotDataset directory "
                                       "<data_dir>/<repo_id>/ (default repo_id: <task_name>_lerobot, "
                                       "i.e. what run_pipeline.sh's LEROBOT_REPO_ID produces by default)")
parser.add_argument("--data_dir", default=str(OSX_UR_ROOT / "data"),
                     help="Root data directory (default: osx-ur/data)")
parser.add_argument("--repo_id", default=None, help="Dataset repo_id (default: <task_name>_lerobot)")
parser.add_argument("--policy_type", default="custom_act",
                     help="lerobot policy registry type, e.g. custom_act or act (default: custom_act)")
parser.add_argument("--job_name", default=None, help="Job name / fluxtrain run_name (default: task_name)")
parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
parser.add_argument("--push_to_hub", action="store_true", help="Push the trained checkpoint to the Hub")
parser.add_argument("--resume", action="store_true",
                     help="Resume training from the run's last checkpoint")
parser.add_argument("--wandb_enable", type=lambda s: s.lower() != "false", default=False,
                     help="Also log to W&B, in addition to tensorboard (default: false)")
parser.add_argument("--steps", type=int, default=None,
                     help="Optional hard cap on optimizer steps (fluxtrain's optim.max_steps); "
                          "default: none, train for the config's full optim.epochs")
parser.add_argument("--batch_size", type=int, default=None,
                     help="Batch size (default: the config's own optim.batch_size)")
parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                     help="fluxtrain TrainConfig yaml to start from (default: "
                          "dependencies/fluxtrain/examples/act_point_jitter.yaml, tailored to this "
                          "camera-free UR5e Point-Policy dataset shape)")
args, extra_overrides = parser.parse_known_args()

repo_id = args.repo_id or f"{args.task_name}_lerobot"
job_name = args.job_name or args.task_name
dataset_root = Path(args.data_dir) / repo_id

if not dataset_root.is_dir():
    raise SystemExit(f"Dataset not found: {dataset_root} "
                      f"(run Point-Policy/ur5e_pipeline/run_pipeline.sh first, or pass --data_dir/--repo_id)")


def train_fluxtrain():
    from fluxtrain import TrainConfig, Trainer

    overrides = [
        f"data.repo_ids=[{repo_id}]",
        f"data.root={args.data_dir}",
        f"policy.type={args.policy_type}",
        f"run_name={job_name}",
        f"output_dir={OSX_UR_ROOT / 'outputs' / 'train'}",
        f"device={args.device}",
        f"resume={args.resume}",
        f"hub.push_to_hub={args.push_to_hub}",
    ]
    if args.push_to_hub:
        overrides.append(f"hub.repo_id={repo_id}")
    if args.wandb_enable:
        overrides.append("logging.backends=[tensorboard, wandb]")
    if args.steps is not None:
        overrides.append(f"optim.max_steps={args.steps}")
    if args.batch_size is not None:
        overrides.append(f"optim.batch_size={args.batch_size}")
    overrides += extra_overrides  # e.g. cache.level=jpeg, optim.lr=1e-4

    cfg = TrainConfig.from_yaml(args.config, overrides=overrides)  # validates internally
    Trainer(cfg).train()


train_fluxtrain()
