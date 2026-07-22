#!/usr/bin/env python3
"""Train an ACT-family policy on a local LeRobotDataset (no HuggingFace Hub).

Two interchangeable backends:
  --backend lerobot   (default) -- stock lerobot-train, via TrainPipelineConfig.
    Equivalent to:
        lerobot-train \
            --dataset.repo_id=<task_name>_lerobot \
            --dataset.root=<data_dir>/<task_name>_lerobot/ \
            --policy.type=custom_act \
            --job_name=<task_name> \
            --policy.device=cuda \
            --policy.push_to_hub=false \
            --wandb.enable=true
  --backend fluxtrain -- dependencies/fluxtrain's faster Trainer harness
    (same lerobot policy registry underneath, so --policy_type custom_act
    works identically; see dependencies/fluxtrain/README.md).

Both read the same local dataset directory produced by
Point-Policy/ur5e_pipeline/run_pipeline.sh's step 7
(convert_pkl_human_to_robot.py -> convert_pkl_to_lerobot.py) -- repo_id is
still required by both configs but is never resolved against the Hub once
a local root is set. push_to_hub=False keeps the checkpoint local too.

Usage:
    python train_point_policy.py bottle_open_06
    python train_point_policy.py bottle_open_06 --backend fluxtrain
    python train_point_policy.py bottle_open_06 --wandb_enable false
    python train_point_policy.py bottle_open_06 --steps 50000 --batch_size 16
"""

import argparse
from pathlib import Path

# Registers the "custom_act" policy config in lerobot's PreTrainedConfig
# registry -- needed by both backends (fluxtrain resolves policy.type
# through the same shared lerobot registry, it has no registry of its own).
from lerobot_policy_custom_act.configuration_custom_act import ACTConfig as CustomACTConfig

OSX_UR_ROOT = Path(__file__).resolve().parents[5]  # scripts/osx_ur5e/osx_core/src/catkin_ws/osx-ur
CUSTOM_ACT_DIR = OSX_UR_ROOT / "dependencies" / "lerobot_policy_custom_act"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("task_name", help="Task name -- must match the LeRobotDataset directory "
                                       "<data_dir>/<repo_id>/ (default repo_id: <task_name>_lerobot, "
                                       "i.e. what run_pipeline.sh's LEROBOT_REPO_ID produces by default)")
parser.add_argument("--backend", choices=["lerobot", "fluxtrain"], default="lerobot",
                     help="Training harness to use (default: lerobot)")
parser.add_argument("--data_dir", default=str(OSX_UR_ROOT / "data"),
                     help="Root data directory (default: osx-ur/data)")
parser.add_argument("--repo_id", default=None, help="Dataset repo_id (default: <task_name>_lerobot)")
parser.add_argument("--policy_type", default="custom_act",
                     help="lerobot policy registry type, e.g. custom_act or act (default: custom_act)")
parser.add_argument("--job_name", default=None, help="Job name (default: task_name)")
parser.add_argument("--output_dir", default=None,
                     help="Checkpoint output dir (default: "
                          "dependencies/lerobot_policy_custom_act/outputs/train/<job_name>)")
parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
parser.add_argument("--push_to_hub", action="store_true", help="Push the trained checkpoint to the Hub")
parser.add_argument("--wandb_enable", type=lambda s: s.lower() != "false", default=True,
                     help="Enable W&B logging (default: true; pass --wandb_enable false to disable)")
parser.add_argument("--steps", type=int, default=None,
                     help="Training steps -- lerobot's total step count, or a hard cap on top of "
                          "fluxtrain's epoch loop (default: each backend's own default)")
parser.add_argument("--batch_size", type=int, default=None,
                     help="Batch size (default: each backend's own default -- fluxtrain auto-tunes "
                          "this at startup, so leave it unset there unless you need to override)")
args = parser.parse_args()

repo_id = args.repo_id or f"{args.task_name}_lerobot"
job_name = args.job_name or args.task_name
output_dir = Path(args.output_dir or CUSTOM_ACT_DIR / "outputs" / "train" / job_name)
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
        output_dir=output_dir,
        job_name=job_name,
        steps=args.steps if args.steps is not None else 100_000,
        batch_size=args.batch_size if args.batch_size is not None else 8,
        wandb=WandBConfig(enable=args.wandb_enable),
    )
    cfg.validate()
    train(cfg)


def train_fluxtrain():
    from fluxtrain import TrainConfig, Trainer
    from fluxtrain.config import DataConfig, PolicyConfig, OptimConfig, LoggingConfig, HubConfig

    optim_kwargs = {}
    if args.steps is not None:
        optim_kwargs["max_steps"] = args.steps
    if args.batch_size is not None:
        optim_kwargs["batch_size"] = args.batch_size

    logging_backends = ["tensorboard"]
    if args.wandb_enable:
        logging_backends.append("wandb")

    cfg = TrainConfig(
        data=DataConfig(repo_ids=[repo_id], root=str(args.data_dir)),
        policy=PolicyConfig(type=args.policy_type, overrides={"push_to_hub": args.push_to_hub}),
        optim=OptimConfig(**optim_kwargs),
        logging=LoggingConfig(backends=logging_backends, wandb_project=job_name),
        hub=HubConfig(push_to_hub=args.push_to_hub, repo_id=repo_id if args.push_to_hub else None),
        # fluxtrain joins output_dir/run_name into the actual run dir -- split
        # the requested output_dir so the final path matches the lerobot backend's.
        output_dir=str(output_dir.parent),
        run_name=output_dir.name,
        device=args.device,
    )
    cfg.validate()
    Trainer(cfg).train()


if args.backend == "lerobot":
    train_lerobot()
else:
    train_fluxtrain()
