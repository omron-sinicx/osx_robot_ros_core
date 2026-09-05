#!/usr/bin/env python3
"""Replay a dataset episode as a B-spline curve through FDCCEnv at a chosen speed factor.

The spline sibling of test_replay_episode.py (docs/bspline_fct.md section 7,
step 2). Instead of stepping dataset rows, the episode is one continuous 18D
FCT curve with the parameter in demo samples; a playhead advances it at speed
factor ``m`` (``+eval.speed_up``), every control tick evaluates the curve,
projects the sample onto the FCT manifold and hands the dense FCT action dict
to ``FDCCEnv.step`` (``prepare_action`` -> ``process_factored_action_dict`` ->
controller, unchanged). The curve comes either from the stored per-frame
``action.bspline.*`` segments (``+eval.source=dataset``, the default: exactly
the representation a spline policy is trained on, through ``decode_segment``)
or from a fresh fit (``+eval.source=fit``). The ROS-free core is
``osx_ur5e/bspline_replay.py``.

Per episode and speed factor: reset, controller parameters, move to the
episode start qpos, Enter prompt, compliance on, the tick loop until the curve
end, then ``move_to_home`` + ``deactivate_compliance_control`` in the sibling's
order. Outputs: ``ep<i>_<source>_x<m>_*.npy`` (u, rate, commanded FCT, eef pose,
wrench, stiffness, clip flags), a comparison plot on the demo time base
(dataset frame = round(u)), a text summary, and under ROS also the sibling's
``plot_episode_comparison`` figure.

Usage (real robot):
    python test_replay_episode_bspline.py --config-name edge_sink_wipe +eval.speed_up=[1,2]
    python test_replay_episode_bspline.py --config-name edge_sink_wipe +eval.source=fit +eval.control_fps=60
    # force-dependent rate (section 3.3), constant otherwise:
    python test_replay_episode_bspline.py --config-name edge_sink_wipe +eval.playhead.alpha=0.2 +eval.playhead.deadband=1.0

Mock env (no ROS, no robot; runs on the training machine):
    python test_replay_episode_bspline.py --config-name edge_sink_wipe +eval.mock_env=true \\
        +eval.speed_up=[1,2] +eval.source=dataset dataset.dataset.dir=<dir> \\
        dataset.dataset.repo_id=[edge_sink_wipe] dataset.dataset.episode_idx=0 hydra.run.dir=<out>

+eval.* knobs (all optional): mock_env (false), source (dataset|fit), speed_up
(list or "1,2" string, default [1]), control_fps (dataset fps), realtime
(true on the robot, false in mock mode; false = no sleeping), prompt (true;
skipped in mock mode),
num_episodes (1), playhead.{alpha (0), deadband (1 N), rate_min (0.1),
contact_cap (off), contact_force_threshold (2 N)}, bspline.<field> overrides of
the fit config (e.g. +eval.bspline.ref_source=dataset), sibling_plot (true).
"""

import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import tqdm

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

import comet  # noqa: F401  registers the ${comet_root:} resolver the configs need
from comet.common.datasets.dataset_info_utils import load_characteristic_length

from rich.console import Console
from rich.logging import RichHandler

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

# Outside a catkin workspace (mock mode on a training machine) the package is
# not installed; its src dir sits next to this scripts dir. osx_ur5e/__init__.py
# is empty, so importing bspline_replay never pulls in rospy.
_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from osx_ur5e.bspline_replay import (  # noqa: E402
    CURVE_SOURCES,
    MockFDCCEnv,
    Playhead,
    PlayheadConfig,
    SplineReplayResult,
    build_episode_curve,
    compare_with_dataset,
    episode_columns,
    episode_ground_truth,
    episode_table_from_frames,
    fit_config_from_info,
    plot_spline_replay,
    run_tick_loop,
    save_result_arrays,
    summary_text,
)

logger = logging.getLogger(__name__)
console = Console()


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


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _select(cfg: DictConfig, key: str, default=None):
    value = OmegaConf.select(cfg, key, default=default)
    if isinstance(value, (DictConfig, ListConfig)):
        value = OmegaConf.to_container(value, resolve=True)
    return value


def _speed_factors(cfg: DictConfig) -> list[float]:
    raw = _select(cfg, "eval.speed_up", default=[1.0])
    if isinstance(raw, str):
        raw = [s for s in raw.replace(";", ",").split(",") if s.strip()]
    elif not isinstance(raw, (list, tuple)):
        raw = [raw]
    speeds = [float(s) for s in raw]
    if not speeds or any(s <= 0 for s in speeds):
        raise ValueError(f"+eval.speed_up must be positive speed factors, got {raw}")
    return speeds


def _playhead_config(cfg: DictConfig, speed: float, demo_fps: float, control_fps: float) -> PlayheadConfig:
    ph = _select(cfg, "eval.playhead", default={}) or {}
    return PlayheadConfig(
        speed=speed,
        alpha=float(ph.get("alpha", 0.0)),
        deadband=float(ph.get("deadband", 1.0)),
        rate_min=float(ph.get("rate_min", 0.1)),
        contact_cap=ph.get("contact_cap", None),
        contact_force_threshold=float(ph.get("contact_force_threshold", 2.0)),
        demo_fps=demo_fps,
        control_fps=control_fps,
    )


def _move_to_init_qpos(env: Any, reason: str = "") -> None:
    """Same as the sibling's move_to_init_qpos (works on the mock env too)."""
    tag = f" ({reason})" if reason else ""
    logger.info(f"Moving to init_qpos{tag}...")
    try:
        env.deactivate_compliance_control()
    except Exception:
        pass
    env.go_home()
    logger.info("Reached init_qpos.")


# ---------------------------------------------------------------------------
# One replay: controller setup -> start pose -> prompt -> compliance -> tick loop -> home
# ---------------------------------------------------------------------------

def replay_curve_on_env(
    env: Any,
    curve,
    playhead_cfg: PlayheadConfig,
    start_qpos: np.ndarray,
    *,
    episode_idx: int,
    prompt: bool,
    sleep_fn: Callable[[float], None] | None,
) -> SplineReplayResult:
    """Mirror of test_replay_episode.replay_single_episode for a spline curve."""
    playhead = Playhead(playhead_cfg, *curve.domain)
    n_ticks_nominal = int(np.ceil((curve.domain[1] - curve.domain[0]) / (playhead_cfg.speed * playhead_cfg.samples_per_tick))) + 1
    tag = f"{curve.source} x{playhead_cfg.speed:g}"

    env.reset(move_robot=False)
    # reset(move_robot=False) skips the controller parameters (gains, mode,
    # selection matrix); push them explicitly like the sibling does.
    env.set_controller_parameters()

    logger.info("Moving to episode start qpos...")
    env.arm.set_joint_positions(target_time=1.0, positions=np.asarray(start_qpos, dtype=float), wait=True)

    if prompt:
        input(f"\n  [{tag}] Episode {episode_idx} (~{n_ticks_nominal} ticks at {playhead_cfg.control_fps:g} Hz, "
              f"alpha={playhead_cfg.alpha:g}) — press Enter to start...")
    env.activate_compliance_control()

    with tqdm.tqdm(total=n_ticks_nominal, desc=f"Ep {episode_idx} ({tag})") as pbar:
        result = run_tick_loop(env, curve, playhead, sleep_fn=sleep_fn, on_tick=lambda i, u: pbar.update(1))

    if result.force_violation:
        logger.warning(f"Episode {episode_idx} [{tag}] ended early at tick {result.n_ticks} (force limit exceeded)")

    # Same order as the sibling / eval runner: servo home under compliance,
    # then drop compliance (move_to_home falls back to go_home by itself after
    # a force violation switched controllers).
    env.move_to_home(timeout=5.0)
    env.deactivate_compliance_control()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(
    version_base=None,
    config_path="../../../../../../dependencies/comet/configs",
    config_name="blank",
)
def main(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "test_replay_bspline.log")
    np.set_printoptions(linewidth=np.inf, formatter={"float": lambda x: f"{x:0.3f}"})

    mock = bool(_select(cfg, "eval.mock_env", default=False))
    source = str(_select(cfg, "eval.source", default="dataset"))
    if source not in CURVE_SOURCES:
        raise ValueError(f"+eval.source must be one of {CURVE_SOURCES}, got {source!r}")
    speeds = _speed_factors(cfg)
    # Real time on the robot; the mock runs as fast as it can unless +eval.realtime=true.
    realtime = bool(_select(cfg, "eval.realtime", default=not mock))
    prompt = bool(_select(cfg, "eval.prompt", default=True)) and not mock
    sibling_plot = bool(_select(cfg, "eval.sibling_plot", default=True))

    # ------------------------------------------------------------------
    # Dataset (same conventions as the sibling: subset load + row_offset)
    # ------------------------------------------------------------------
    repo_id = cfg.dataset.dataset.repo_id
    if isinstance(repo_id, (list, ListConfig)):
        repo_id = str(repo_id[0])
    dataset_root = Path(cfg.dataset.dataset.dir) / repo_id
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    logger.info(f"Loading dataset: {repo_id} from {dataset_root}")
    start_episode = int(cfg.dataset.dataset.episode_idx)
    num_episodes = int(_select(cfg, "eval.num_episodes", default=1))
    meta = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    end_episode = min(start_episode + num_episodes, meta.total_episodes)
    dataset = LeRobotDataset(repo_id, root=dataset_root, video_backend="pyav", use_videos=False,
                             episodes=list(range(start_episode, end_episode)))
    row_offset = int(dataset.meta.episodes[start_episode]["dataset_from_index"])
    info = dataset.meta.info
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)

    demo_fps = float(info["fps"])
    override_fps = _select(cfg, "eval.control_fps", default=None)
    control_fps = float(override_fps) if override_fps is not None else demo_fps

    fit_config = fit_config_from_info(info, _select(cfg, "eval.bspline", default=None))
    if source == "dataset" and "bspline_config" not in info:
        raise ValueError("source=dataset needs the stored action.bspline.* features "
                         "(run add_bspline_segments.py --write) — or use +eval.source=fit")
    columns = episode_columns(source, fit_config, available=list(info["features"]))
    logger.info(
        f"Dataset {repo_id}: {meta.total_episodes} episodes | replaying {start_episode} – {end_episode - 1} | "
        f"source={source} speeds={speeds} control_fps={control_fps:g} demo_fps={demo_fps:g} mock={mock}"
    )
    logger.info(f"B-spline config: degree {fit_config.degree}, chunk_size {fit_config.chunk_size} "
                f"({fit_config.rows} rows), ref_source {fit_config.ref_source}, relative_knots {fit_config.relative_knots}")

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    if mock:
        env = None  # built per episode so the mock starts at the episode's first pose
        sleep_fn = time.sleep if realtime else None
        sibling = None
    else:
        import rospy  # noqa: WPS433  ROS only here
        import test_replay_episode as sibling  # the dense replay script (imports rospy / FDCCEnv)
        from osx_ur5e.fdcc_env import FDCCEnv

        rospy.init_node("test_replay_episode_bspline", anonymous=False)
        logger.info("ROS node initialized")
        env = FDCCEnv(config=sibling._env_config_from(cfg, dataset))
        env.reference_trajectory = []
        logger.info(f"actions_as_deltas: {env.actions_as_deltas}")
        sleep_fn = rospy.sleep if realtime else None
        _move_to_init_qpos(env, reason="initial safe position")

    ft_in_tool_frame = bool(info.get("virtual_target_displacement_config", {}).get("ft_in_tool_frame", True))
    total_force_errors: dict[str, float] = {}
    summaries: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Episode loop
    # ------------------------------------------------------------------
    for episode_idx in range(start_episode, end_episode):
        ep = dataset.meta.episodes[episode_idx]
        ep_start, ep_end = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        logger.info(f"Episode {episode_idx}: {ep_end - ep_start} frames")

        frames = [dataset[t - row_offset] for t in range(ep_start, ep_end)]
        table = episode_table_from_frames(frames, columns)
        gt = episode_ground_truth(table, fit_config)
        curve = build_episode_curve(table, source, fit_config)
        logger.info(f"Curve ({source}): domain u in [{curve.domain[0]:.1f}, {curve.domain[1]:.1f}] demo samples"
                    + (f", {curve.fit.n_knots} knots, seeding {curve.fit.seeding}" if source == "fit" else ""))
        start_qpos = gt["qpos"][0]

        if mock:
            env = MockFDCCEnv(
                control_fps=control_fps,
                safety_parameters=cfg.controller.safety_parameters,
                default_stiffness=float(cfg.controller.stiffness),
                characteristic_length=load_characteristic_length(info),
                initial_position=gt["eef_pos"][0],
                initial_ortho6=gt["eef_rot6"][0],
            )
            ds_gt_sibling = None
        else:
            ds_gt_sibling = sibling.extract_dataset_ground_truth(
                dataset, episode_idx, include_stiffness=True, stiffness_key="action.estimated_stiffness",
                ft_in_tool_frame=ft_in_tool_frame, row_offset=row_offset,
            )

        for name in ("eef_pos", "force_norm", "gated_force", "label_ref_position", "label_force", "stiffness"):
            np.save(output_dir / f"ep{episode_idx}_dataset_{name}.npy", gt[name])

        for speed in speeds:
            playhead_cfg = _playhead_config(cfg, speed, demo_fps, control_fps)
            result = replay_curve_on_env(env, curve, playhead_cfg, start_qpos,
                                         episode_idx=episode_idx, prompt=prompt, sleep_fn=sleep_fn)
            comparison = compare_with_dataset(result, gt)
            text = summary_text(episode_idx, result, comparison)
            logger.info("\n" + text)
            (output_dir / f"ep{episode_idx}_{result.label}_summary.txt").write_text(text + "\n")
            with open(output_dir / f"ep{episode_idx}_{result.label}_summary.json", "w") as f:
                json.dump(comparison, f, indent=2)
            if result.clip_triggers:
                (output_dir / f"ep{episode_idx}_{result.label}_clips.txt").write_text("\n".join(result.clip_triggers) + "\n")
            save_result_arrays(output_dir, episode_idx, result)
            plot_spline_replay(episode_idx, gt, result, output_dir / f"ep{episode_idx}_{result.label}_spline_replay.png",
                               comparison=comparison)
            summaries[f"ep{episode_idx}_{result.label}"] = comparison
            total_force_errors[result.label] = total_force_errors.get(result.label, 0.0) + comparison["total_meas_force_err_N"]

            if ds_gt_sibling is not None and sibling_plot and result.n_ticks:
                n = int(gt["n_frames"])
                sib_result = sibling.ReplayResult(
                    action_type=result.label, eef_pos=result.eef_pos, force_norm=result.force_norm,
                    torque_norm=result.torque_norm, stiffness=result.stiffness,
                    force_violation=result.force_violation, ds_indices=np.clip(result.ds_indices, 0, n - 1),
                    stride=1, duration_s=result.duration_s, clip_count=result.clip_count,
                )
                sibling._log_replay_summary(episode_idx, sib_result, ds_gt_sibling)
                sibling.plot_episode_comparison(episode_idx, ds_gt_sibling, [sib_result],
                                                output_dir / f"ep{episode_idx}_{result.label}_comparison.png")

    if not mock:
        _move_to_init_qpos(env, reason="all episodes finished")

    with open(output_dir / "summaries.json", "w") as f:
        json.dump(summaries, f, indent=2)
    logger.info("=" * 60)
    logger.info(f"Episodes replayed:    {end_episode - start_episode} | source={source} | speeds={speeds}")
    for label, ferr in total_force_errors.items():
        logger.info(f"  [{label}] total measured-force error: {ferr:.2f} N·ticks")
    logger.info(f"Results saved to:     {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    if "hydra.run.dir" not in " ".join(sys.argv):
        sys.argv.append("paths.script=/test_replay_bspline")
    main()
