#!/usr/bin/env python3
"""Train an ACT-family policy on a local LeRobotDataset (no HuggingFace Hub).

Uses stock lerobot-train via TrainPipelineConfig. Equivalent to:
    lerobot-train \
        --dataset.repo_id=<task_name>_lerobot \
        --dataset.root=<data_dir>/<task_name>_lerobot/ \
        --policy.type=custom_act \
        --job_name=<task_name> \
        --policy.device=cuda \
        --policy.push_to_hub=false \
        --wandb.enable=true

Reads the local dataset directory produced by
Point-Policy/ur5e_pipeline/run_pipeline.sh's step 7
(convert_pkl_human_to_robot.py -> convert_pkl_to_lerobot.py) -- repo_id is
still required by the config but is never resolved against the Hub once
a local root is set. push_to_hub=False keeps the checkpoint local too.

Usage:
    python train_point_policy.py bottle_open_06
    python train_point_policy.py bottle_open_06 --wandb_enable false
    python train_point_policy.py bottle_open_06 --steps 50000 --batch_size 16
"""

import argparse
from pathlib import Path

# Registers the "custom_act" policy config in lerobot's PreTrainedConfig
# registry.
from lerobot_policy_custom_act.configuration_custom_act import ACTConfig as CustomACTConfig

OSX_UR_ROOT = Path(__file__).resolve().parents[5]  # scripts/osx_ur5e/osx_core/src/catkin_ws/osx-ur

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("task_name", help="Task name -- must match the LeRobotDataset directory "
                                       "<data_dir>/<repo_id>/ (default repo_id: <task_name>_lerobot, "
                                       "i.e. what run_pipeline.sh's LEROBOT_REPO_ID produces by default)")
parser.add_argument("--data_dir", default=str(OSX_UR_ROOT / "data"),
                     help="Root data directory (default: osx-ur/data)")
parser.add_argument("--repo_id", default=None, help="Dataset repo_id (default: <task_name>_lerobot)")
parser.add_argument("--policy_type", default="custom_act",
                     help="lerobot policy registry type, e.g. custom_act or act (default: custom_act)")
parser.add_argument("--job_name", default=None, help="Job name (default: task_name)")
parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
parser.add_argument("--push_to_hub", action="store_true", help="Push the trained checkpoint to the Hub")
parser.add_argument("--resume", action="store_true",
                     help="Resume training from the last checkpoint")
parser.add_argument("--wandb_enable", type=lambda s: s.lower() != "false", default=False,
                     help="Enable W&B logging (default: true; pass --wandb_enable false to disable)")
parser.add_argument("--steps", type=int, default=400000,
                     help="Training steps -- lerobot's total step count (default: 400000)")
parser.add_argument("--batch_size", type=int, default=None,
                     help="Batch size (default: 8)")
args = parser.parse_args()

repo_id = args.repo_id or f"{args.task_name}_lerobot"
job_name = args.job_name or args.task_name
dataset_root = Path(args.data_dir) / repo_id

if not dataset_root.is_dir():
    raise SystemExit(f"Dataset not found: {dataset_root} "
                      f"(run Point-Policy/ur5e_pipeline/run_pipeline.sh first, or pass --data_dir/--repo_id)")


def train_lerobot():
    from lerobot.scripts.lerobot_train import train
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.policies import PreTrainedConfig

    policy_cfg = PreTrainedConfig.get_choice_class(args.policy_type)(
        device=args.device,
        push_to_hub=args.push_to_hub,
        repo_id=repo_id if args.push_to_hub else None,
    )

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=repo_id,
            root=str(dataset_root),
        ),
        policy=policy_cfg,
        job_name=job_name,
        steps=args.steps if args.steps is not None else 100_000,
        batch_size=args.batch_size if args.batch_size is not None else 8,
        wandb=WandBConfig(enable=args.wandb_enable),
        resume=args.resume,
    )
    cfg.validate()
    train(cfg)


train_lerobot()
