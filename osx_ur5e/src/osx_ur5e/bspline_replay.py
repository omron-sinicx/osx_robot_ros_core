"""B-spline replay of a dataset episode through FDCCEnv: the ROS-free core.

Step 2 of ``docs/bspline_fct.md`` section 7 (replay fitted demos through FDCC
at 1x and 2x). ``scripts/test_replay_episode_bspline.py`` is the thin ROS
wrapper; everything that can run without a robot lives here so it can be
exercised with a mock environment and under pytest:

* **episode curve** (:func:`build_episode_curve`): the 18D FCT curve of one
  episode, parameter ``u`` in demo samples at the dataset fps. ``source="fit"``
  fits it on the fly (``build_fct_channels`` + ``fit_episode``);
  ``source="dataset"`` evaluates the stored per-frame ``action.bspline.*``
  segments through ``decode_segment`` / ``evaluate_segment``, i.e. exactly the
  representation a spline policy is trained on;
* **playhead** (:class:`Playhead`): the shared-contract rate law. Constant
  speed factor ``m`` by default (BSP); with ``alpha > 0`` the rate drops with
  the normal-force error (section 3.3 of the design note);
* **per-tick mapping** (:func:`fct_to_env_action`): curve value ->
  ``project_fct`` -> the dense FCT action dict ``FDCCEnv.prepare_action``
  already understands (``action.ref_position`` ... ``action.estimated_stiffness``);
* **tick loop and records** (:func:`run_tick_loop`, :class:`SplineReplayResult`),
  the dataset ground truth on the demo time base, the comparison numbers and a
  compact matplotlib plot;
* :class:`MockFDCCEnv`, a fake env with the surface the loop uses, tracking the
  commanded reference perfectly and measuring a zero wrench.

Imports: numpy / scipy / comet only (plus the ROS-free ``osx_ur5e.action_limits``
when the package is importable). ``osx_ur5e/__init__.py`` is empty, so
``import osx_ur5e.bspline_replay`` never pulls in rospy; outside a catkin
workspace put ``osx_ur5e/src`` on ``sys.path`` first.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from comet.common.utils import math_utils as mu
from comet.common.utils.bspline_utils import (
    FCT_CHANNEL_GROUPS,
    FCT_CHANNEL_SLICES,
    FCT_DIM,
    BSplineFitConfig,
    FitResult,
    build_fct_channels,
    evaluate_segment,
    fit_episode,
    project_fct,
    required_columns,
    segment_domain,
)
from comet.common.utils.vt_utils import process_factored_action_dict

try:  # ROS-free sibling; missing only when this file is imported by path without the package
    from osx_ur5e.action_limits import limiter_from_safety_config
except ImportError:  # pragma: no cover
    limiter_from_safety_config = None

# ==================== Names ====================

SEGMENT_FEATURES: tuple[str, ...] = ("action.bspline.knots",) + tuple(f"action.bspline.{g}" for g in FCT_CHANNEL_GROUPS)
DENSE_ACTION_KEYS: tuple[str, ...] = tuple(f"action.{g}" for g in FCT_CHANNEL_GROUPS)
CURVE_SOURCES: tuple[str, ...] = ("fit", "dataset")

OBS_EEF_POSITION = "observation.eef.position"
OBS_EEF_ROTATION = "observation.eef.rotation_ortho6"
OBS_QPOS = "observation.qpos"
OBS_FT = "observation.ft"
COL_CONTACT_FLAG = "action.contact_flag"
COL_ESTIMATED_STIFFNESS = "action.estimated_stiffness"

POS = FCT_CHANNEL_SLICES["ref_position"]
ROT = FCT_CHANNEL_SLICES["ref_rotation_ortho6"]
DIR = FCT_CHANNEL_SLICES["contact_direction"]
FORCE = FCT_CHANNEL_SLICES["normal_force"]
STIFF = FCT_CHANNEL_SLICES["estimated_stiffness"]

IDENTITY_ORTHO6 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


# ==================== Episode tables ====================


def _as_float_array(value: Any) -> np.ndarray:
    """One cell -> float ndarray (torch tensors, nested lists, object arrays of arrays)."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray) and value.dtype != object:
        return value.astype(float)
    if isinstance(value, (list, tuple, np.ndarray)):
        items = [_as_float_array(v) for v in value]
        if items and items[0].ndim > 0:
            return np.stack(items)
        return np.asarray(items, dtype=float)
    return np.asarray(value, dtype=float)


def table_column(table: Any, key: str) -> np.ndarray:
    """Column ``key`` of a DataFrame / dict of arrays / list of frames as a float ``(T, ...)`` array.

    Scalars and 1-D features come back as ``(T, d)``; the 2-D ``action.bspline.*``
    features (parquet ``list<list<float>>`` cells or ``(rows, d)`` tensors) as
    ``(T, rows, d)``.
    """
    values = table[key]
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    if isinstance(values, np.ndarray) and values.dtype != object:
        out = values.astype(float)
    else:
        out = np.stack([_as_float_array(v) for v in values])
    if out.ndim == 1:
        out = out[:, None]
    return out


def episode_columns(source: str, config: BSplineFitConfig, available: Sequence[str] | None = None) -> list[str]:
    """Dataset columns the replay reads for ``source`` (fit inputs, ground truth, stored segments)."""
    cols = list(required_columns(config.ref_source))
    for c in (OBS_EEF_POSITION, OBS_EEF_ROTATION, OBS_QPOS, OBS_FT, COL_CONTACT_FLAG, COL_ESTIMATED_STIFFNESS):
        if c not in cols:
            cols.append(c)
    if source == "dataset":
        cols += [f for f in SEGMENT_FEATURES if f not in cols]
    if available is not None:
        missing = [c for c in cols if c not in available]
        if missing:
            raise KeyError(f"dataset lacks columns {missing} needed for source={source!r}")
    return cols


def episode_table_from_frames(frames: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> dict[str, np.ndarray]:
    """Stack per-frame dicts (``LeRobotDataset[i]``) into ``{column: (T, ...) float array}``."""
    return {c: np.stack([_as_float_array(f[c]) for f in frames]) for c in columns}


def segment_matrices(table: Any) -> np.ndarray:
    """Concatenate the stored ``action.bspline.*`` features -> ``(T, rows, 1 + 18)`` segment matrices."""
    parts = [table_column(table, f) for f in SEGMENT_FEATURES]
    for name, part in zip(SEGMENT_FEATURES, parts):
        if part.ndim != 3:
            raise ValueError(f"{name}: expected (T, rows, d), got {part.shape}")
    matrices = np.concatenate(parts, axis=-1)
    if matrices.shape[-1] != 1 + FCT_DIM:
        raise ValueError(f"segment matrices must have {1 + FCT_DIM} columns, got {matrices.shape}")
    return matrices


def fit_config_from_info(info: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> BSplineFitConfig:
    """``info.json['bspline_config']`` (or the defaults) with ``overrides`` applied.

    Without a stored config the fps follows the dataset so that the curve
    parameter is in demo samples at the recording rate.
    """
    data: dict[str, Any] = dict(info.get("bspline_config") or {})
    data.setdefault("fps", float(info.get("fps", BSplineFitConfig().fps)))
    data.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return BSplineFitConfig.from_dict(data)


# ==================== Episode curves ====================


class EpisodeCurve:
    """An 18D FCT curve over one episode; ``u`` is in demo samples at the dataset fps."""

    source: str = ""
    n_frames: int = 0
    domain: tuple[float, float] = (0.0, 0.0)

    def evaluate(self, u: float) -> np.ndarray:
        """Raw (unprojected) 18-vector at ``u``; ``u`` is clamped to ``domain``."""
        raise NotImplementedError

    def evaluate_many(self, u: np.ndarray) -> np.ndarray:
        return np.stack([self.evaluate(float(v)) for v in np.asarray(u, dtype=float).reshape(-1)])


class FittedEpisodeCurve(EpisodeCurve):
    """``source="fit"``: one B-spline fitted to the whole episode (``bspline_utils.fit_episode``)."""

    source = "fit"

    def __init__(self, channels: np.ndarray, contact_flag: np.ndarray, config: BSplineFitConfig):
        self.config = config
        self.fit: FitResult = fit_episode(channels, contact_flag, config)
        self.spline = self.fit.spline()
        self.n_frames = int(len(channels))
        t = self.spline.t
        self.domain = (float(t[config.degree]), float(t[-(config.degree + 1)]))

    def evaluate(self, u: float) -> np.ndarray:
        lo, hi = self.domain
        u = min(max(float(u), lo), hi)
        if u == hi:  # left limit, see bspline_utils.evaluate_segment
            u = np.nextafter(hi, -np.inf)
        return np.asarray(self.spline(u), dtype=float)


class StoredSegmentCurve(EpisodeCurve):
    """``source="dataset"``: the per-frame ``action.bspline.*`` segments as stored.

    A segment is stored for every frame ``k`` with its knots relative to ``k``.
    By the BSP assignment rule the segment of frame ``k`` is the first one whose
    first interior knot is >= ``k``, so its domain usually starts a few samples
    *after* ``k`` (on ``edge_sink_wipe`` episode 0 for 90 % of the frames, up to
    16 samples) and, for the frames after the last segment start, before it.
    Demo parameter ``u`` is therefore evaluated on the segment stored for the
    latest frame ``j <= floor(u)`` whose domain contains ``u - j``: the segment
    stored for ``floor(u)`` when it covers ``u``, otherwise the one stored at
    the preceding knot site. All of them are windows of the same fitted curve,
    so this reproduces the episode curve exactly (BSP instead clamps the
    parameter to the segment start, which would hold the first value).
    """

    source = "dataset"

    def __init__(self, segments: np.ndarray, config: BSplineFitConfig):
        segments = np.asarray(segments, dtype=np.float64)
        if segments.ndim != 3 or segments.shape[1:] != (config.rows, 1 + FCT_DIM):
            raise ValueError(f"segments must be (T, {config.rows}, {1 + FCT_DIM}), got {segments.shape}")
        self.config = config
        self.segments = segments
        self.n_frames = int(segments.shape[0])
        self._domains: dict[int, tuple[float, float]] = {}
        lo = self.relative_domain(0)[0]
        hi = (self.n_frames - 1) + self.relative_domain(self.n_frames - 1)[1]
        self.domain = (float(lo), float(hi))

    def relative_domain(self, frame: int) -> tuple[float, float]:
        """``(u_min, u_max)`` of the segment stored for ``frame``, relative to that frame."""
        if frame not in self._domains:
            self._domains[frame] = segment_domain(self.segments[frame], self.config)
        return self._domains[frame]

    def covering_frame(self, u: float) -> int:
        """Frame whose stored segment is evaluated for demo parameter ``u`` (see class doc)."""
        k = int(min(max(np.floor(u), 0), self.n_frames - 1))
        for j in range(k, -1, -1):
            lo, hi = self.relative_domain(j)
            if lo <= u - j <= hi:
                return j
        raise ValueError(f"no stored segment covers u={u}")

    def evaluate(self, u: float) -> np.ndarray:
        lo, hi = self.domain
        u = min(max(float(u), lo), hi)
        j = self.covering_frame(u)
        return evaluate_segment(self.segments[j], np.array([u - j]), self.config)[0]


def episode_labels(table: Any, config: BSplineFitConfig) -> tuple[np.ndarray, np.ndarray]:
    """The 18 FCT channels the curve is fitted to (gated, filtered) and the contact flag."""
    return build_fct_channels(table, config.wrench_filter_window, ref_source=config.ref_source)


def build_episode_curve(table: Any, source: str, config: BSplineFitConfig) -> EpisodeCurve:
    """Episode curve from a frame table: ``"fit"`` fits it, ``"dataset"`` uses the stored segments."""
    if source == "fit":
        channels, flag = episode_labels(table, config)
        return FittedEpisodeCurve(channels, flag, config)
    if source == "dataset":
        return StoredSegmentCurve(segment_matrices(table), config)
    raise ValueError(f"source must be one of {CURVE_SOURCES}, got {source!r}")


# ==================== Playhead ====================


@dataclass
class PlayheadConfig:
    """Rate law of docs/bspline_fct.md section 3.3 (``alpha = 0``: constant speed factor, BSP).

    ``speed`` is the speed factor ``m``; ``u`` advances by
    ``rate * demo_fps / control_fps`` demo samples per control tick, with
    ``rate = clamp(m / (1 + alpha * max(0, |f_pred - f_meas| - deadband)), rate_min, contact_cap)``
    where the cap only applies in contact. Contact is the measured normal force
    above ``contact_force_threshold`` or a commanded press (``f_pred > 0``); the
    measured normal force is ``|F|`` of the wrench (magnitude, no projection on
    the contact direction, no bias capture: the simplification of the replay
    script). ``contact_cap <= 0`` / ``None`` disables the cap.
    """

    speed: float = 1.0
    alpha: float = 0.0
    deadband: float = 1.0
    rate_min: float = 0.1
    contact_cap: float | None = None
    contact_force_threshold: float = 2.0
    demo_fps: float = 60.0
    control_fps: float = 60.0

    def __post_init__(self) -> None:
        if self.speed <= 0:
            raise ValueError(f"speed must be > 0, got {self.speed}")
        if self.alpha < 0 or self.deadband < 0 or self.rate_min < 0:
            raise ValueError("alpha, deadband and rate_min must be >= 0")
        if self.demo_fps <= 0 or self.control_fps <= 0:
            raise ValueError("demo_fps and control_fps must be > 0")
        if self.contact_cap is not None and self.contact_cap <= 0:
            self.contact_cap = None

    @property
    def samples_per_tick(self) -> float:
        """Demo samples one control tick covers at unit rate."""
        return self.demo_fps / self.control_fps

    @property
    def dt(self) -> float:
        return 1.0 / self.control_fps


class Playhead:
    """Curve parameter ``u`` (demo samples) advanced once per control tick by the rate law."""

    def __init__(self, config: PlayheadConfig, u_start: float, u_end: float):
        if u_end < u_start:
            raise ValueError(f"u_end ({u_end}) < u_start ({u_start})")
        self.config = config
        self.u_start = float(u_start)
        self.u_end = float(u_end)
        self.reset()

    def reset(self) -> None:
        self.u = self.u_start
        self.last_rate = self.config.speed
        self.last_gap = 0.0
        self.ticks = 0

    @property
    def finished(self) -> bool:
        """``u`` has reached the curve end (that tick still has to be executed)."""
        return self.u >= self.u_end

    def in_contact(self, f_pred: float, f_meas: float) -> bool:
        return f_meas > self.config.contact_force_threshold or f_pred > 0.0

    def rate(self, f_pred: float = 0.0, f_meas: float = 0.0, in_contact: bool | None = None) -> float:
        """Speed factor for the next tick (section 3.3)."""
        c = self.config
        gap = max(0.0, abs(float(f_pred) - float(f_meas)) - c.deadband)
        rate = c.speed / (1.0 + c.alpha * gap)
        rate = max(rate, c.rate_min)
        if in_contact is None:
            in_contact = self.in_contact(f_pred, f_meas)
        if in_contact and c.contact_cap is not None:
            rate = min(rate, c.contact_cap)
        self.last_gap = gap
        return rate

    def advance(self, f_pred: float = 0.0, f_meas: float = 0.0, in_contact: bool | None = None) -> float:
        """Advance ``u`` by one control tick and return the new ``u`` (clamped to the curve end)."""
        rate = self.rate(f_pred, f_meas, in_contact)
        self.last_rate = rate
        self.u = min(self.u + rate * self.config.samples_per_tick, self.u_end)
        self.ticks += 1
        return self.u


# ==================== Curve sample -> env action ====================


def fct_to_env_action(projected: np.ndarray) -> dict[str, np.ndarray]:
    """A projected 18-vector -> the dense FCT action dict ``FDCCEnv.prepare_action`` consumes.

    Keys ``action.ref_position`` (3), ``action.ref_rotation_ortho6`` (6),
    ``action.contact_direction`` (3), ``action.torque_direction`` (3),
    ``action.normal_force`` (1), ``action.normal_torque`` (1),
    ``action.estimated_stiffness`` (1), float64 arrays like the dataset replay.
    """
    v = np.asarray(projected, dtype=np.float64).reshape(-1)
    if v.shape != (FCT_DIM,):
        raise ValueError(f"expected an 18-vector, got {v.shape}")
    return {f"action.{g}": v[sl].copy() for g, sl in FCT_CHANNEL_SLICES.items()}


def curve_tick_action(curve: EpisodeCurve, u: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Evaluate ``curve`` at ``u``, project onto the FCT manifold, build the env action dict."""
    projected = project_fct(curve.evaluate(u))
    return projected, fct_to_env_action(projected)


# ==================== Records ====================


@dataclass
class SplineReplayResult:
    """Everything recorded during one replay (one episode, one speed factor)."""

    source: str
    speed: float
    demo_fps: float
    control_fps: float
    u_start: float
    u_end: float
    u: np.ndarray                      # (n,) curve parameter of every executed tick
    rate: np.ndarray                   # (n,) rate used to reach that tick
    command: np.ndarray                # (n, 18) projected FCT command
    eef_pos: np.ndarray                # (n, 3)
    eef_quat: np.ndarray               # (n, 4)
    wrench: np.ndarray                 # (n, 6) measured after the step
    stiffness: np.ndarray              # (n,) env.last_compliance_stiffness
    clipped: np.ndarray                # (n,) bool, limiter clipped the step
    clip_triggers: list[str] = field(default_factory=list)
    tick_time: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (n,) s since the first tick
    force_violation: bool = False
    duration_s: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.source}_x{self.speed:g}"

    @property
    def n_ticks(self) -> int:
        return int(len(self.u))

    @property
    def ds_indices(self) -> np.ndarray:
        """Dataset frame (relative to the episode start) each tick corresponds to: ``round(u)``."""
        return np.rint(self.u).astype(int)

    @property
    def commanded_force(self) -> np.ndarray:
        return self.command[:, FORCE][:, 0]

    @property
    def commanded_stiffness(self) -> np.ndarray:
        return self.command[:, STIFF][:, 0]

    @property
    def force_norm(self) -> np.ndarray:
        return np.linalg.norm(self.wrench[:, :3], axis=1)

    @property
    def torque_norm(self) -> np.ndarray:
        return np.linalg.norm(self.wrench[:, 3:], axis=1)

    @property
    def clip_count(self) -> int:
        return int(np.count_nonzero(self.clipped))

    @property
    def demo_span_s(self) -> float:
        """Demo time the executed part of the curve covers."""
        return float(self.u[-1] - self.u[0]) / self.demo_fps if self.n_ticks else 0.0

    @property
    def nominal_duration_s(self) -> float:
        """Wall time the ticks take at ``control_fps`` (what a real-time loop spends)."""
        return self.n_ticks / self.control_fps

    @property
    def achieved_speedup(self) -> float:
        return self.demo_span_s / self.duration_s if self.duration_s > 0 else float("nan")

    def arrays(self) -> dict[str, np.ndarray]:
        """The per-tick arrays to save as npy."""
        return {
            "u": self.u, "rate": self.rate, "cmd_fct": self.command, "eef_pos": self.eef_pos,
            "eef_quat": self.eef_quat, "wrench": self.wrench, "stiffness": self.stiffness,
            "clipped": self.clipped, "tick_time": self.tick_time,
        }


def save_result_arrays(output_dir: Path, episode_idx: int, result: SplineReplayResult) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, arr in result.arrays().items():
        p = output_dir / f"ep{episode_idx}_{result.label}_{name}.npy"
        np.save(p, arr)
        paths.append(p)
    return paths


# ==================== The tick loop ====================


def run_tick_loop(
    env: Any,
    curve: EpisodeCurve,
    playhead: Playhead,
    *,
    sleep_fn: Callable[[float], None] | None = None,
    on_tick: Callable[[int, float], None] | None = None,
) -> SplineReplayResult:
    """Execute ``curve`` on ``env`` under ``playhead`` until the curve end (or a force violation).

    Per tick: evaluate the curve at ``u``, project, ``env.step(dense FCT dict)``,
    read the pose and wrench (after the step, i.e. after the controller had one
    control period on the new target), record, then advance the playhead with
    ``f_pred`` = the commanded normal force and ``f_meas`` = ``|wrench[:3]|``.
    ``sleep_fn`` (``rospy.sleep`` / ``time.sleep``) keeps the loop at
    ``control_fps``; ``None`` runs as fast as possible (mock mode). Controller
    setup, the start pose and homing are the caller's job.
    """
    cfg = playhead.config
    playhead.reset()
    rec: dict[str, list] = {k: [] for k in ("u", "rate", "command", "eef_pos", "eef_quat", "wrench", "stiffness", "clipped", "tick_time")}
    triggers_log: list[str] = []
    force_violation = False
    t_start = time.perf_counter()
    i = 0
    while True:
        tick_start = time.perf_counter()
        u = playhead.u
        projected, env_action = curve_tick_action(curve, u)
        timestep = env.step(env_action)

        eef = np.asarray(env.arm.end_effector(), dtype=float).reshape(-1)
        wrench = np.asarray(env.arm.get_wrench(), dtype=float).reshape(-1)
        f_meas = float(np.linalg.norm(wrench[:3]))
        triggers = env.action_limiter.last_clip_triggers
        clipped = bool(triggers["translation"] or triggers["orientation"])
        if clipped:
            triggers_log.append(f"tick {i} u={u:.2f}: {triggers}")

        rec["u"].append(u)
        rec["rate"].append(playhead.last_rate)
        rec["command"].append(projected)
        rec["eef_pos"].append(eef[:3])
        rec["eef_quat"].append(eef[3:7] if eef.size >= 7 else np.array([0.0, 0.0, 0.0, 1.0]))
        rec["wrench"].append(wrench[:6])
        rec["stiffness"].append(float(env.last_compliance_stiffness))
        rec["clipped"].append(clipped)
        rec["tick_time"].append(tick_start - t_start)
        if on_tick is not None:
            on_tick(i, u)
        i += 1

        if timestep.last():
            force_violation = True
            break
        if playhead.finished:
            break
        playhead.advance(f_pred=float(projected[FORCE][0]), f_meas=f_meas)

        if sleep_fn is not None:
            remaining = cfg.dt - (time.perf_counter() - tick_start)
            if remaining > 0:
                sleep_fn(remaining)
    duration_s = time.perf_counter() - t_start

    return SplineReplayResult(
        source=curve.source, speed=cfg.speed, demo_fps=cfg.demo_fps, control_fps=cfg.control_fps,
        u_start=playhead.u_start, u_end=playhead.u_end,
        u=np.asarray(rec["u"], dtype=float), rate=np.asarray(rec["rate"], dtype=float),
        command=np.asarray(rec["command"], dtype=float).reshape(-1, FCT_DIM),
        eef_pos=np.asarray(rec["eef_pos"], dtype=float).reshape(-1, 3),
        eef_quat=np.asarray(rec["eef_quat"], dtype=float).reshape(-1, 4),
        wrench=np.asarray(rec["wrench"], dtype=float).reshape(-1, 6),
        stiffness=np.asarray(rec["stiffness"], dtype=float), clipped=np.asarray(rec["clipped"], dtype=bool),
        clip_triggers=triggers_log, tick_time=np.asarray(rec["tick_time"], dtype=float),
        force_violation=force_violation, duration_s=duration_s,
    )


# ==================== Dataset ground truth and comparison ====================


def episode_ground_truth(table: Any, config: BSplineFitConfig) -> dict[str, np.ndarray]:
    """What the replay is compared against, on the demo time base (one row per frame).

    ``eef_pos`` / ``eef_rot6`` are the demonstrated pose, ``force_norm`` the raw
    ``|F|`` of ``observation.ft``, ``gated_force`` that magnitude gated by the
    contact flag (the wrench relabel's ``normal_force``), and ``label_*`` the
    channels the curve was fitted to (``build_fct_channels``: gated and filtered
    force, reference rebuilt per ``config.ref_source``).
    """
    channels, flag = episode_labels(table, config)
    ft = table_column(table, OBS_FT)
    force_norm = np.linalg.norm(ft[:, :3], axis=1)
    torque_norm = np.linalg.norm(ft[:, 3:6], axis=1)
    return dict(
        n_frames=np.array(len(channels)),
        eef_pos=table_column(table, OBS_EEF_POSITION)[:, :3],
        eef_rot6=table_column(table, OBS_EEF_ROTATION)[:, :6],
        qpos=table_column(table, OBS_QPOS),
        force_norm=force_norm,
        torque_norm=torque_norm,
        contact_flag=flag.astype(bool),
        gated_force=force_norm * flag,
        stiffness=table_column(table, COL_ESTIMATED_STIFFNESS)[:, 0],
        label_channels=channels,
        label_ref_position=channels[:, POS],
        label_force=channels[:, FORCE][:, 0],
        label_stiffness=channels[:, STIFF][:, 0],
    )


def compare_with_dataset(result: SplineReplayResult, gt: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Per-replay metrics against the dataset at frame ``round(u)`` of every tick."""
    n = int(gt["n_frames"])
    if result.n_ticks == 0:
        return {"n_ticks": 0}
    idx = np.clip(result.ds_indices, 0, n - 1)
    pos_err_cmd = np.linalg.norm(result.eef_pos - result.command[:, POS], axis=1)
    pos_err_label = np.linalg.norm(result.eef_pos - gt["label_ref_position"][idx], axis=1)
    pos_err_obs = np.linalg.norm(result.eef_pos - gt["eef_pos"][idx], axis=1)
    cmd_force = result.commanded_force
    force_err_gated = cmd_force - gt["gated_force"][idx]
    force_err_label = cmd_force - gt["label_force"][idx]
    meas_force_err = result.force_norm - gt["gated_force"][idx]
    stiff_err = result.commanded_stiffness - gt["stiffness"][idx]
    return {
        "n_ticks": result.n_ticks,
        "duration_s": result.duration_s,
        "nominal_duration_s": result.nominal_duration_s,
        "demo_span_s": result.demo_span_s,
        "achieved_speedup": result.achieved_speedup,
        "nominal_speedup": result.demo_span_s / result.nominal_duration_s if result.n_ticks else float("nan"),
        "clip_count": result.clip_count,
        "force_violation": float(result.force_violation),
        "mean_pos_err_vs_command_m": float(pos_err_cmd.mean()),
        "max_pos_err_vs_command_m": float(pos_err_cmd.max()),
        "mean_pos_err_vs_label_m": float(pos_err_label.mean()),
        "max_pos_err_vs_label_m": float(pos_err_label.max()),
        "mean_pos_err_vs_obs_m": float(pos_err_obs.mean()),
        "mean_cmd_force_err_vs_gated_N": float(np.abs(force_err_gated).mean()),
        "max_cmd_force_err_vs_gated_N": float(np.abs(force_err_gated).max()),
        "mean_cmd_force_err_vs_label_N": float(np.abs(force_err_label).mean()),
        "max_cmd_force_err_vs_label_N": float(np.abs(force_err_label).max()),
        "mean_meas_force_err_N": float(np.abs(meas_force_err).mean()),
        "total_meas_force_err_N": float(np.abs(meas_force_err).sum()),
        "mean_stiffness_err": float(np.abs(stiff_err).mean()),
        "mean_rate": float(result.rate.mean()),
        "min_rate": float(result.rate.min()),
    }


def summary_text(episode_idx: int, result: SplineReplayResult, comparison: Mapping[str, float]) -> str:
    c = comparison
    lines = [
        f"[{result.label}] episode {episode_idx}: {c['n_ticks']} ticks at {result.control_fps:g} Hz "
        f"over u in [{result.u_start:.1f}, {result.u_end:.1f}] demo samples "
        f"({c['demo_span_s']:.2f} s of demo)",
        f"  duration: {c['duration_s']:.2f} s wall (nominal {c['nominal_duration_s']:.2f} s at control rate) | "
        f"achieved speedup {c['achieved_speedup']:.2f}x (nominal {c['nominal_speedup']:.2f}x, requested {result.speed:g}x) | "
        f"rate mean {c['mean_rate']:.2f} min {c['min_rate']:.2f}",
        f"  limiter clipped {c['clip_count']} ticks | force violation: {bool(c['force_violation'])}",
        f"  position error vs commanded reference (tracking): mean {c['mean_pos_err_vs_command_m'] * 1e3:.2f} mm, "
        f"max {c['max_pos_err_vs_command_m'] * 1e3:.2f} mm | vs fitted reference at frame round(u): "
        f"mean {c['mean_pos_err_vs_label_m'] * 1e3:.2f} mm, max {c['max_pos_err_vs_label_m'] * 1e3:.2f} mm | "
        f"vs demonstrated eef: mean {c['mean_pos_err_vs_obs_m'] * 1e3:.2f} mm",
        f"  commanded normal_force vs gated |F|: mean {c['mean_cmd_force_err_vs_gated_N']:.3f} N, "
        f"max {c['max_cmd_force_err_vs_gated_N']:.3f} N | vs fit label (filtered): mean "
        f"{c['mean_cmd_force_err_vs_label_N']:.3f} N, max {c['max_cmd_force_err_vs_label_N']:.3f} N",
        f"  measured |F| vs gated demo |F|: mean {c['mean_meas_force_err_N']:.2f} N, total {c['total_meas_force_err_N']:.1f} N | "
        f"stiffness error mean {c['mean_stiffness_err']:.1f} N/m",
    ]
    return "\n".join(lines)


# ==================== Compact ROS-free plot ====================


def plot_spline_replay(
    episode_idx: int,
    gt: Mapping[str, np.ndarray],
    result: SplineReplayResult,
    save_path: Path,
    comparison: Mapping[str, float] | None = None,
) -> Path:
    """Dataset vs replay on the demo time base (x = dataset frame ``round(u)``)."""
    import matplotlib
    if not matplotlib.get_backend().lower().startswith("agg") and not _display_available():
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = int(gt["n_frames"])
    steps = np.arange(n)
    idx = np.clip(result.ds_indices, 0, n - 1)
    x = result.ds_indices
    axis_labels = ["X", "Y", "Z"]
    ds_colors = ["tab:cyan", "tab:red", "tab:olive"]
    rp_colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, axes = plt.subplots(6, 1, figsize=(13, 20), sharex=True)
    fig.subplots_adjust(hspace=0.35, top=0.965)

    ax = axes[0]
    for i, lbl in enumerate(axis_labels):
        ax.plot(steps, gt["eef_pos"][:, i], color=ds_colors[i], linestyle="--", linewidth=0.8, label=f"{lbl} dataset eef")
        ax.plot(steps, gt["label_ref_position"][:, i], color=ds_colors[i], linestyle=":", linewidth=0.8, label=f"{lbl} fitted reference")
        ax.plot(x, result.eef_pos[:, i], color=rp_colors[i], linewidth=0.9, label=f"{lbl} replay eef")
    ax.set_title(f"Episode {episode_idx} [{result.label}] — EEF position (m)")
    ax.set_ylabel("m")
    ax.legend(ncol=3, fontsize=6, loc="upper right")
    ax.grid(True, linewidth=0.4)

    ax = axes[1]
    ax.plot(x, np.linalg.norm(result.eef_pos - result.command[:, POS], axis=1) * 1e3, color="tab:green",
            linewidth=0.9, label="|replay eef − commanded reference| (tracking)")
    ax.plot(x, np.linalg.norm(result.eef_pos - gt["label_ref_position"][idx], axis=1) * 1e3, color="tab:purple",
            linewidth=0.9, label="|replay eef − fitted reference at round(u)|")
    ax.plot(x, np.linalg.norm(result.eef_pos - gt["eef_pos"][idx], axis=1) * 1e3, color="tab:brown",
            linewidth=0.9, alpha=0.8, label="|replay eef − dataset eef|")
    ax.set_title("Position error (mm)")
    ax.set_ylabel("mm")
    ax.legend(fontsize=7)
    ax.grid(True, linewidth=0.4)

    ax = axes[2]
    ax.plot(steps, gt["force_norm"], color="lightgrey", linewidth=0.7, label="dataset |F| raw")
    ax.plot(steps, gt["gated_force"], color="tab:grey", linestyle="--", linewidth=0.9, label="dataset |F| gated")
    ax.plot(steps, gt["label_force"], color="black", linestyle=":", linewidth=0.9, label="fit label (filtered)")
    ax.plot(x, result.commanded_force, color="tab:blue", linewidth=0.9, label="commanded normal_force")
    ax.plot(x, result.force_norm, color="tab:red", linewidth=0.9, alpha=0.85, label="replay measured |F|")
    ax.set_title("Normal force (N)")
    ax.set_ylabel("N")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, linewidth=0.4)

    ax = axes[3]
    ax.plot(x, result.commanded_force - gt["gated_force"][idx], color="tab:blue", linewidth=0.9, label="commanded − gated dataset")
    ax.plot(x, result.force_norm - gt["gated_force"][idx], color="tab:red", linewidth=0.9, alpha=0.85, label="measured − gated dataset")
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_title("Force difference (N)")
    ax.set_ylabel("ΔN")
    ax.legend(fontsize=7)
    ax.grid(True, linewidth=0.4)

    ax = axes[4]
    ax.plot(steps, gt["stiffness"], color="tab:grey", linestyle="--", linewidth=0.9, label="dataset estimated_stiffness")
    ax.plot(x, result.commanded_stiffness, color="tab:green", linewidth=0.9, label="commanded")
    ax.plot(x, result.stiffness, color="tab:olive", linewidth=0.8, alpha=0.7, label="env.last_compliance_stiffness")
    ax.set_title("Stiffness (N/m)")
    ax.set_ylabel("N/m")
    ax.legend(fontsize=7)
    ax.grid(True, linewidth=0.4)

    ax = axes[5]
    ax.plot(x, result.rate, color="tab:blue", linewidth=0.9, label="playhead rate")
    ax.axhline(result.speed, color="black", linewidth=0.5, linestyle=":", label=f"m = {result.speed:g}")
    if result.clipped.any():
        ax.scatter(x[result.clipped], result.rate[result.clipped], s=8, color="tab:red", label="limiter clipped")
    ax.set_title("Playhead rate per tick (x = dataset frame round(u))")
    ax.set_ylabel("rate")
    ax.set_xlabel("Dataset frame")
    ax.legend(fontsize=7)
    ax.grid(True, linewidth=0.4)

    if comparison is not None:
        fig.suptitle(
            f"{result.label}: {comparison['n_ticks']} ticks, nominal {comparison['nominal_speedup']:.2f}x, "
            f"tracking err {comparison['mean_pos_err_vs_command_m'] * 1e3:.2f} mm, "
            f"pos err vs fitted reference {comparison['mean_pos_err_vs_label_m'] * 1e3:.2f} mm, "
            f"cmd force err {comparison['mean_cmd_force_err_vs_gated_N']:.2f} N, clipped {comparison['clip_count']}",
            fontsize=10,
        )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _display_available() -> bool:
    import os
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


# ==================== Mock environment ====================


class _PassThroughLimiter:
    """Stands in for DeltaActionLimiter when osx_ur5e.action_limits is not importable."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.last_clip_triggers = {"translation": [], "orientation": []}

    def clip(self, delta_translation, delta_orientation, dt, **_):
        self.reset()
        return np.asarray(delta_translation, dtype=float), np.asarray(delta_orientation, dtype=float)


class MockTimeStep:
    """Minimal stand-in for osx_ur5e.timestep.TimeStep (``last()`` is what the loop checks)."""

    STEP_MID = 1
    STEP_LAST = 2

    def __init__(self, step_type: int = STEP_MID, observation: Any = None):
        self.step_type = step_type
        self.observation = observation
        self.reward = None
        self.discount = None

    def last(self) -> bool:
        return self.step_type == self.STEP_LAST


class MockArm:
    """The ``env.arm`` surface the loop uses: pose = last commanded reference, wrench = zeros."""

    def __init__(self, position: np.ndarray | None = None, ortho6: np.ndarray | None = None):
        self.position = np.zeros(3) if position is None else np.asarray(position, dtype=float).copy()
        self.ortho6 = IDENTITY_ORTHO6.copy() if ortho6 is None else np.asarray(ortho6, dtype=float).copy()
        self.qpos = np.zeros(6)
        self.wrench = np.zeros(6)
        self.joint_moves: list[np.ndarray] = []

    def end_effector(self, joint_angles: np.ndarray | None = None) -> np.ndarray:
        """``[x, y, z, qx, qy, qz, qw]`` like ``ur_control``'s ``end_effector()``."""
        return np.concatenate([self.position, mu.quaternion_from_ortho6(self.ortho6)])

    def get_wrench(self) -> np.ndarray:
        return self.wrench.copy()

    def set_joint_positions(self, positions, target_time: float = 1.0, wait: bool = True) -> None:
        self.qpos = np.asarray(positions, dtype=float).reshape(-1).copy()
        self.joint_moves.append(self.qpos.copy())


class MockFDCCEnv:
    """FDCCEnv stand-in for running the spline replay without ROS.

    ``step`` validates the dense FCT keys, runs the real
    ``process_factored_action_dict`` (virtual target + 6x6 stiffness, as
    ``FDCCEnv.prepare_action`` does), feeds the virtual-target delta through the
    real ``DeltaActionLimiter`` built from ``safety_parameters`` (so the clip
    count is the one the robot would see at this control rate) and then moves
    the arm *exactly* to the commanded reference pose: no surface, no spring,
    zero wrench. Method calls are logged in ``calls``.
    """

    def __init__(
        self,
        control_fps: float = 60.0,
        safety_parameters: Any = None,
        default_stiffness: float = 500.0,
        characteristic_length: float = 0.1,
        initial_position: np.ndarray | None = None,
        initial_ortho6: np.ndarray | None = None,
    ):
        self.control_frequency = float(control_fps)
        self.dt = 1.0 / self.control_frequency
        self.default_stiffness = float(default_stiffness)
        self.characteristic_length = float(characteristic_length)
        self.actions_as_deltas = False
        self.arm = MockArm(initial_position, initial_ortho6)
        if safety_parameters is not None and limiter_from_safety_config is not None:
            self.action_limiter = limiter_from_safety_config(safety_parameters)
        else:
            self.action_limiter = _PassThroughLimiter()
        self.last_compliance_stiffness = 0.0
        self.compliance_active = False
        self.reference_trajectory: list = []
        self.calls: list[str] = []
        self.actions: list[dict[str, np.ndarray]] = []
        self.controller_actions: list[dict[str, np.ndarray]] = []

    # -- lifecycle (same names/order the replay scripts call) --
    def reset(self, move_robot: bool = False):
        self.calls.append(f"reset(move_robot={move_robot})")
        self.action_limiter.reset()
        if move_robot:
            self.set_controller_parameters()
        return MockTimeStep(observation=None)

    def set_controller_parameters(self) -> None:
        self.calls.append("set_controller_parameters")

    def activate_compliance_control(self) -> None:
        self.calls.append("activate_compliance_control")
        self.action_limiter.reset()
        self.compliance_active = True

    def deactivate_compliance_control(self) -> None:
        self.calls.append("deactivate_compliance_control")
        self.compliance_active = False

    def move_to_home(self, timeout: float = 5.0, **_) -> None:
        self.calls.append("move_to_home")

    def go_home(self) -> None:
        self.calls.append("go_home")

    def check_contact_force_limits(self) -> bool:
        return True

    # -- the step --
    def step(self, action: Mapping[str, Any]) -> MockTimeStep:
        missing = [k for k in DENSE_ACTION_KEYS if k not in action]
        if missing:
            raise KeyError(f"mock env: dense FCT action lacks {missing}")
        self.last_compliance_stiffness = float(np.asarray(action["action.estimated_stiffness"]).reshape(-1)[0])
        controller_action = process_factored_action_dict(
            action,
            default_stiffness=self.default_stiffness,
            default_stiffness_rot=self.default_stiffness,
            characteristic_length=self.characteristic_length,
            use_isotropic_stiffness=False,
            orientation_representation="quaternion",
            full_stiffness_matrix=True,
        )
        current = self.arm.end_effector()
        target_pos = np.asarray(controller_action["action.position"], dtype=float).reshape(3)
        target_quat = np.asarray(controller_action["action.orientation"], dtype=float).reshape(4)
        delta_t = target_pos - current[:3]
        delta_r = (Rotation.from_quat(target_quat) * Rotation.from_quat(current[3:7]).inv()).as_rotvec()
        self.action_limiter.clip(delta_t, delta_r, self.dt)

        self.arm.position = np.asarray(action["action.ref_position"], dtype=float).reshape(3).copy()
        self.arm.ortho6 = np.asarray(action["action.ref_rotation_ortho6"], dtype=float).reshape(6).copy()
        self.actions.append({k: np.array(v, dtype=float) for k, v in action.items()})
        self.controller_actions.append(controller_action)
        return MockTimeStep(MockTimeStep.STEP_MID)
