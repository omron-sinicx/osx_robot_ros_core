"""Tests for osx_ur5e.bspline_replay (spline replay core, no ROS).

Runnable with the comet pixi python from the workspace root:
    .pixi/envs/default/bin/python -m pytest ws/src/osx/osx_core/osx_ur5e/tests/test_bspline_replay.py -q

The stored-segment test needs a dataset that carries the ``action.bspline.*``
features: ``$BSPLINE_REPLAY_TEST_DATASET`` (a LeRobot dataset dir) or
``<comet_root>/data/edge_sink_wipe``; it is skipped otherwise.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# osx_ur5e is a catkin package; outside ROS put its src dir on the path
# (osx_ur5e/__init__.py is empty, so nothing ROS-related is imported).
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from comet.common.utils.bspline_utils import FCT_CHANNEL_SLICES, FCT_DIM, BSplineFitConfig, project_fct  # noqa: E402

from osx_ur5e.bspline_replay import (  # noqa: E402
    DENSE_ACTION_KEYS,
    DENSE_REPLAY_ACTION_TYPES,
    FORCE,
    POS,
    ROT,
    STIFF,
    DenseReplayResult,
    FittedEpisodeCurve,
    MockFDCCEnv,
    Playhead,
    PlayheadConfig,
    StoredSegmentCurve,
    build_episode_curve,
    compare_dense_with_dataset,
    compare_with_dataset,
    compare_with_dense,
    dense_action_columns,
    dense_summary_text,
    episode_ground_truth,
    fct_to_env_action,
    fit_config_from_info,
    pair_spline_with_dense,
    run_dense_loop,
    run_tick_loop,
    segment_matrices,
    summary_text,
)

DEMO_FPS = 60.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_episode(T=360, onset=120, offset=260, ramp=60.0):
    """Synthetic (channels (T, 18), contact_flag (T,)) like tests/test_bspline_utils.py."""
    t = np.arange(T, dtype=float)
    flag = np.zeros(T, dtype=bool)
    flag[onset:offset] = True
    w = 2 * np.pi * t / T
    pos = np.stack([0.30 + 0.05 * np.sin(w), 0.10 + 0.03 * np.cos(0.7 * w), 0.20 - 0.04 * np.sin(0.5 * w)], axis=1)
    ang = 0.3 * np.sin(0.6 * w)
    rot = np.stack([np.cos(ang), np.sin(ang), 0 * ang, -np.sin(ang), np.cos(ang), 0 * ang], axis=1)
    direction = flag[:, None] * np.array([0.0, 0.0, -1.0])
    torque_direction = flag[:, None] * np.array([1.0, 0.0, 0.0])
    force = (flag * 8.0 * np.clip((t - onset) / ramp, 0.0, 1.0))[:, None]
    torque = (flag * 0.05)[:, None]
    stiffness = np.full((T, 1), 800.0)
    channels = np.hstack([pos, rot, direction, torque_direction, force, torque, stiffness])
    assert channels.shape == (T, FCT_DIM)
    return channels, flag


def dense_table(channels: np.ndarray) -> dict:
    """A synthetic episode table carrying both the raw and the factored dense columns."""
    stiffness = channels[:, STIFF]                       # (T, 1)
    table = {
        # raw_actions: the recorded pose command + the 6-vector stiffness diagonal
        "action.position": channels[:, POS].copy(),
        "action.rotation_ortho6": channels[:, ROT].copy(),
        "action.stiffness_diag": np.repeat(stiffness, 6, axis=1),
        # what a replay reads back / the vt fallback reference
        "observation.eef.position": channels[:, POS].copy(),
        "observation.eef.rotation_ortho6": channels[:, ROT].copy(),
    }
    # factored_actions: the seven FCT channel groups
    for key in DENSE_ACTION_KEYS:
        table[key] = channels[:, FCT_CHANNEL_SLICES[key[len("action."):]]].copy()
    return table


def count_ticks(cfg: PlayheadConfig, u_end: float, f_pred=0.0, f_meas=0.0) -> int:
    ph = Playhead(cfg, 0.0, u_end)
    n = 0
    while True:
        n += 1
        if ph.finished:
            return n
        ph.advance(f_pred=f_pred, f_meas=f_meas)


def _dataset_dir() -> Path | None:
    env = os.environ.get("BSPLINE_REPLAY_TEST_DATASET")
    candidates = [Path(env)] if env else []
    try:
        from comet.paths import comet_root
        candidates.append(Path(comet_root()) / "data" / "edge_sink_wipe")
    except Exception:  # pragma: no cover
        pass
    for c in candidates:
        if (c / "meta" / "info.json").exists():
            return c
    return None


@pytest.fixture(scope="module")
def dataset_episode():
    """(table DataFrame of episode 0, info dict) from a dataset with stored segments, else skip."""
    import json
    import pandas as pd

    root = _dataset_dir()
    if root is None:
        pytest.skip("no dataset with action.bspline.* features (set BSPLINE_REPLAY_TEST_DATASET)")
    info = json.loads((root / "meta" / "info.json").read_text())
    if "bspline_config" not in info:
        pytest.skip(f"{root} has no bspline_config")
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    ep0 = LeRobotDatasetMetadata(root.name, root=root).episodes[0]
    data_path = root / info["data_path"].format(chunk_index=int(ep0["data/chunk_index"]), file_index=int(ep0["data/file_index"]))
    df = pd.read_parquet(data_path)
    table = df[df["episode_index"] == 0].reset_index(drop=True)
    return table, info


# ---------------------------------------------------------------------------
# playhead
# ---------------------------------------------------------------------------


def test_playhead_constant_speed_covers_domain_in_fewer_ticks():
    u_end = 300.0
    n1 = count_ticks(PlayheadConfig(speed=1.0, demo_fps=DEMO_FPS, control_fps=DEMO_FPS), u_end)
    n2 = count_ticks(PlayheadConfig(speed=2.0, demo_fps=DEMO_FPS, control_fps=DEMO_FPS), u_end)
    n4 = count_ticks(PlayheadConfig(speed=4.0, demo_fps=DEMO_FPS, control_fps=DEMO_FPS), u_end)
    assert n1 == int(u_end) + 1                      # one tick per demo sample, plus the end tick
    assert abs(n2 - n1 / 2) <= 1
    assert abs(n4 - n1 / 4) <= 1
    # control rate below the demo rate: each tick covers more demo samples
    n20 = count_ticks(PlayheadConfig(speed=1.0, demo_fps=DEMO_FPS, control_fps=20.0), u_end)
    assert abs(n20 - n1 / 3) <= 1
    # the last executed tick sits exactly on the curve end
    ph = Playhead(PlayheadConfig(speed=2.0, demo_fps=DEMO_FPS, control_fps=DEMO_FPS), 0.0, 7.0)
    while not ph.finished:
        ph.advance()
    assert ph.u == 7.0


def test_playhead_dwells_on_force_error():
    cfg = PlayheadConfig(speed=2.0, alpha=0.5, deadband=1.0, rate_min=0.2, demo_fps=DEMO_FPS, control_fps=DEMO_FPS)
    ph = Playhead(cfg, 0.0, 100.0)
    # no force error -> constant speed factor
    assert ph.rate(f_pred=10.0, f_meas=10.0) == pytest.approx(2.0)
    # error inside the deadband -> no slowdown
    assert ph.rate(f_pred=10.0, f_meas=9.2) == pytest.approx(2.0)
    # error beyond the deadband -> slower, symmetric in the sign of the error
    slow = ph.rate(f_pred=10.0, f_meas=4.0)
    assert slow == pytest.approx(2.0 / (1.0 + 0.5 * 5.0))
    assert ph.rate(f_pred=4.0, f_meas=10.0) == pytest.approx(slow)
    # a huge error never stops the playhead: floored at rate_min
    assert ph.rate(f_pred=0.0, f_meas=1000.0) == pytest.approx(0.2)
    # dwelling: the same domain takes more ticks under a persistent force error
    n_free = count_ticks(cfg, 60.0, f_pred=10.0, f_meas=10.0)
    n_dwell = count_ticks(cfg, 60.0, f_pred=10.0, f_meas=0.0)
    assert n_dwell > 2 * n_free
    # alpha = 0 reproduces BSP: the error is ignored
    cfg0 = PlayheadConfig(speed=2.0, alpha=0.0, demo_fps=DEMO_FPS, control_fps=DEMO_FPS)
    assert Playhead(cfg0, 0.0, 10.0).rate(f_pred=10.0, f_meas=0.0) == pytest.approx(2.0)


def test_playhead_contact_cap():
    cfg = PlayheadConfig(speed=3.0, contact_cap=1.5, contact_force_threshold=2.0, demo_fps=DEMO_FPS, control_fps=DEMO_FPS)
    ph = Playhead(cfg, 0.0, 10.0)
    assert ph.rate(f_pred=0.0, f_meas=0.0) == pytest.approx(3.0)      # free space: full speed
    assert ph.rate(f_pred=5.0, f_meas=0.0) == pytest.approx(1.5)      # commanded press counts as contact
    assert ph.rate(f_pred=0.0, f_meas=3.0) == pytest.approx(1.5)      # measured contact too
    assert ph.rate(f_pred=0.0, f_meas=3.0, in_contact=False) == pytest.approx(3.0)
    assert PlayheadConfig(contact_cap=0).contact_cap is None           # 0 disables the cap


# ---------------------------------------------------------------------------
# env action dict
# ---------------------------------------------------------------------------


def test_fct_to_env_action_keys_and_shapes():
    channels, _ = make_episode()
    raw = channels[150] + 0.01  # slightly off-manifold
    action = fct_to_env_action(project_fct(raw))
    assert tuple(action) == DENSE_ACTION_KEYS == (
        "action.ref_position", "action.ref_rotation_ortho6", "action.contact_direction",
        "action.torque_direction", "action.normal_force", "action.normal_torque", "action.estimated_stiffness",
    )
    assert not any(k.startswith("action.bspline") for k in action)
    for key, arr in action.items():
        group = key[len("action."):]
        sl = FCT_CHANNEL_SLICES[group]
        assert arr.shape == (sl.stop - sl.start,) and arr.dtype == np.float64
    assert np.linalg.norm(action["action.contact_direction"]) == pytest.approx(1.0)
    assert action["action.normal_force"].item() >= 0.0            # what FDCCEnv.prepare_action calls
    assert action["action.estimated_stiffness"].item() == pytest.approx(800.01)
    with pytest.raises(ValueError):
        fct_to_env_action(np.zeros(17))


# ---------------------------------------------------------------------------
# mock env + tick loop
# ---------------------------------------------------------------------------


def test_mock_env_tick_loop_tracks_the_curve():
    channels, flag = make_episode()
    config = BSplineFitConfig(fps=DEMO_FPS)
    curve = FittedEpisodeCurve(channels, flag, config)
    env = MockFDCCEnv(control_fps=DEMO_FPS, initial_position=channels[0, POS])
    cfg = PlayheadConfig(speed=2.0, demo_fps=DEMO_FPS, control_fps=DEMO_FPS)
    env.reset(move_robot=False)
    env.activate_compliance_control()
    result = run_tick_loop(env, curve, Playhead(cfg, *curve.domain))

    assert result.n_ticks == count_ticks(cfg, curve.domain[1] - curve.domain[0])
    assert result.u[0] == curve.domain[0] and result.u[-1] == curve.domain[1]
    assert np.all(np.diff(result.u) > 0)
    assert result.command.shape == (result.n_ticks, FCT_DIM)
    # the mock moves exactly to the commanded reference and measures nothing
    np.testing.assert_allclose(result.eef_pos, result.command[:, POS], atol=1e-12)
    assert np.all(result.wrench == 0.0)
    assert np.all(result.stiffness == result.command[:, FCT_CHANNEL_SLICES["estimated_stiffness"]][:, 0])
    assert not result.force_violation and result.clip_count == 0
    # every step received the dense keys and produced a controller action
    assert len(env.actions) == result.n_ticks
    assert set(env.actions[0]) == set(DENSE_ACTION_KEYS)
    assert set(env.controller_actions[0]) == {"action.position", "action.orientation", "action.stiffness_diag"}
    assert env.controller_actions[0]["action.stiffness_diag"].shape == (6, 6)
    # commanded force follows the fitted label within the fit tolerance
    idx = np.clip(result.ds_indices, 0, len(channels) - 1)
    assert np.abs(result.commanded_force - channels[idx, FORCE][:, 0]).max() < 2.0
    assert env.calls[:2] == ["reset(move_robot=False)", "activate_compliance_control"]


# ---------------------------------------------------------------------------
# dense (per-row) replay of the dataset actions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("action_type", ["raw_actions", "factored_actions"])
def test_dense_loop_replays_every_row(action_type):
    channels, _ = make_episode()
    table = dense_table(channels)
    env = MockFDCCEnv(control_fps=DEMO_FPS, initial_position=channels[0, POS], initial_ortho6=channels[0, ROT])
    env.reset(move_robot=False)
    env.activate_compliance_control()
    result = run_dense_loop(env, table, action_type, control_fps=DEMO_FPS, demo_fps=DEMO_FPS)

    T = len(channels)
    assert result.label == action_type and action_type in DENSE_REPLAY_ACTION_TYPES
    assert result.n_ticks == T
    np.testing.assert_array_equal(result.ds_indices, np.arange(T))
    # every type commands the demonstrated reference position here, and the mock
    # goes exactly there
    np.testing.assert_allclose(result.command_pos, channels[:, POS], atol=1e-12)
    np.testing.assert_allclose(result.eef_pos, result.command_pos, atol=1e-12)
    assert np.all(result.wrench == 0.0)
    assert not result.force_violation and result.clip_count == 0 and not result.clip_triggers
    assert np.all(result.stiffness == channels[:, STIFF][:, 0])
    assert result.demo_span_s == pytest.approx((T - 1) / DEMO_FPS)
    assert result.nominal_duration_s == pytest.approx(T / DEMO_FPS)
    # the env saw the right dict shape for this action type
    assert len(env.actions) == T and len(env.controller_actions) == T
    assert set(env.controller_actions[0]) == {"action.position", "action.orientation", "action.stiffness_diag"}
    if action_type == "raw_actions":
        assert set(env.actions[0]) == {"action.position", "action.orientation", "action.stiffness_diag"}
        assert env.actions[0]["action.orientation"].shape == (4,)
    else:
        assert set(env.actions[0]) == set(DENSE_ACTION_KEYS)
        assert env.controller_actions[0]["action.stiffness_diag"].shape == (6, 6)
    assert set(result.arrays()) == {"ds_indices", "command_pos", "eef_pos", "eef_quat",
                                    "wrench", "stiffness", "clipped", "tick_time"}


def test_dense_action_columns():
    assert dense_action_columns("raw_actions") == [
        "action.position", "action.rotation_ortho6", "action.stiffness_diag"]
    assert dense_action_columns("factored_actions") == list(DENSE_ACTION_KEYS)
    vt_all = dense_action_columns("virtual_target_actions")
    assert "action.vt_ref_position" in vt_all and "observation.eef.position" in vt_all
    # optional columns are dropped when the dataset lacks them, required ones raise
    vt_some = dense_action_columns("virtual_target_actions", available=[
        "action.virtual_target_position", "action.virtual_target_rotation",
        "action.vt_estimated_stiffness", "observation.eef.position", "observation.eef.rotation_ortho6"])
    assert "action.vt_ref_position" not in vt_some
    with pytest.raises(KeyError):
        dense_action_columns("raw_actions", available=["action.position"])
    with pytest.raises(ValueError):
        dense_action_columns("nope")


def test_compare_with_dense_pairs_on_the_dataset_frame():
    channels, flag = make_episode()
    config = BSplineFitConfig(fps=DEMO_FPS)
    table = dense_table(channels)

    dense_env = MockFDCCEnv(control_fps=DEMO_FPS, initial_position=channels[0, POS], initial_ortho6=channels[0, ROT])
    dense = run_dense_loop(dense_env, table, "factored_actions", control_fps=DEMO_FPS, demo_fps=DEMO_FPS)

    curve = FittedEpisodeCurve(channels, flag, config)
    spline_env = MockFDCCEnv(control_fps=DEMO_FPS, initial_position=channels[0, POS], initial_ortho6=channels[0, ROT])
    cfg = PlayheadConfig(speed=2.0, demo_fps=DEMO_FPS, control_fps=DEMO_FPS)
    spline = run_tick_loop(spline_env, curve, Playhead(cfg, *curve.domain))

    vs = compare_with_dense(spline, dense)
    assert vs["n_paired"] == spline.n_ticks
    # both replays track their command exactly, so what is left is the fit error
    assert vs["mean_pos_err_vs_dense_m"] < 3e-3 and vs["max_pos_err_vs_dense_m"] < 3e-3
    assert vs["mean_cmd_pos_err_vs_dense_m"] < 3e-3
    assert vs["mean_meas_force_err_vs_dense_N"] == 0.0
    assert vs["total_meas_force_err_vs_dense_N"] == 0.0
    assert vs["mean_stiffness_err_vs_dense"] == pytest.approx(0.0, abs=1e-9)
    assert all(isinstance(v, float) for v in vs.values())
    # the extra summary line only appears when the dense comparison is passed
    assert "vs dense replay" in summary_text(0, spline, compare_with_dataset(spline, {
        "n_frames": np.array(len(channels)), "eef_pos": channels[:, POS],
        "label_ref_position": channels[:, POS], "gated_force": channels[:, FORCE][:, 0],
        "label_force": channels[:, FORCE][:, 0], "stiffness": channels[:, STIFF][:, 0],
    }), vs_dense=vs)

    # a dense replay that ended early pairs only the frames it reached
    cut = dense.n_ticks // 3
    truncated = DenseReplayResult(
        action_type=dense.action_type, demo_fps=dense.demo_fps, control_fps=dense.control_fps,
        ds_indices=dense.ds_indices[:cut], command_pos=dense.command_pos[:cut], eef_pos=dense.eef_pos[:cut],
        eef_quat=dense.eef_quat[:cut], wrench=dense.wrench[:cut], stiffness=dense.stiffness[:cut],
        clipped=dense.clipped[:cut], tick_time=dense.tick_time[:cut], force_violation=True,
        duration_s=dense.duration_s,
    )
    si, di = pair_spline_with_dense(spline, truncated)
    assert len(si) == len(di) < spline.n_ticks
    assert np.all(spline.ds_indices[si] == truncated.ds_indices[di])
    vs_cut = compare_with_dense(spline, truncated)
    assert 0 < vs_cut["n_paired"] < vs["n_paired"]
    # nothing in common -> the empty dict
    empty = DenseReplayResult(
        action_type="raw_actions", demo_fps=DEMO_FPS, control_fps=DEMO_FPS,
        ds_indices=np.zeros(0, dtype=int), command_pos=np.zeros((0, 3)), eef_pos=np.zeros((0, 3)),
        eef_quat=np.zeros((0, 4)), wrench=np.zeros((0, 6)), stiffness=np.zeros(0),
        clipped=np.zeros(0, dtype=bool),
    )
    assert compare_with_dense(spline, empty) == {"n_paired": 0}


def test_quaternion_from_ortho6_matches_ur_control():
    """The dense raw replay converts ortho6 with comet's math_utils; FDCCEnv uses ur_control."""
    from comet.common.utils import math_utils as mu
    tr = pytest.importorskip("ur_control.transformations")

    rng = np.random.default_rng(7)
    for _ in range(8):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        ortho6 = q[:, :2].T.reshape(-1)
        np.testing.assert_allclose(mu.quaternion_from_ortho6(ortho6),
                                   tr.quaternion_from_ortho6(ortho6), atol=1e-12)


# ---------------------------------------------------------------------------
# stored segments (source="dataset") on a real episode
# ---------------------------------------------------------------------------


def test_stored_segments_reproduce_episode_curve(dataset_episode):
    table, info = dataset_episode
    config = fit_config_from_info(info)
    assert config.rows == 24 and config.relative_knots

    stored = build_episode_curve(table, "dataset", config)
    fitted = build_episode_curve(table, "fit", config)
    assert isinstance(stored, StoredSegmentCurve) and isinstance(fitted, FittedEpisodeCurve)
    n = len(table)
    assert stored.n_frames == n == fitted.n_frames
    assert stored.domain == pytest.approx((0.0, n - 1.0))
    assert fitted.domain == pytest.approx((0.0, n - 1.0))

    segments = segment_matrices(table)
    assert segments.shape == (n, config.rows, 1 + FCT_DIM)
    # the BSP assignment rule: the stored segment of most frames starts after the frame
    first_knot = segments[:, 0, 0]
    assert np.any(first_knot > 0)
    # ... so the covering frame is at or before floor(u) and its domain contains u
    for u in (0.0, 5.5, 100.25, 512.0, n - 1.0):
        j = stored.covering_frame(u)
        lo, hi = stored.relative_domain(j)
        assert j <= int(np.floor(u)) and lo <= u - j <= hi
    assert stored.covering_frame(0.0) == 0

    # stored segments == windows of the fitted curve (float32 storage)
    u = np.arange(0.0, n - 1.0, 0.25)
    a = stored.evaluate_many(u)
    b = fitted.evaluate_many(u)
    assert np.isfinite(a).all()
    np.testing.assert_allclose(a[:, POS], b[:, POS], atol=1e-4)
    np.testing.assert_allclose(a[:, FORCE], b[:, FORCE], atol=1e-2)
    np.testing.assert_allclose(a, b, atol=1e-2, rtol=1e-4)

    # ... and within the fit tolerances of the labels at the frames
    channels, flag = make_labels(table, config)
    at_frames = stored.evaluate_many(np.arange(n, dtype=float))
    assert np.abs(at_frames[:, POS] - channels[:, POS]).max() < config.tolerances["ref_position"]
    assert np.abs(at_frames[:, FORCE] - channels[:, FORCE]).max() < config.tolerances["normal_force"]

    # the projected command reproduces the gated measured force at every frame
    projected = project_fct(at_frames)
    gated = np.linalg.norm(np.stack([np.asarray(v, dtype=float) for v in table["observation.ft"]])[:, :3], axis=1) * flag
    assert np.abs(projected[:, FORCE][:, 0] - channels[:, FORCE][:, 0]).max() < 0.2
    assert np.abs(projected[:, FORCE][:, 0] - gated).mean() < 0.2


def make_labels(table, config):
    from osx_ur5e.bspline_replay import episode_labels
    return episode_labels(table, config)


def test_mock_replay_of_dataset_episode_matches_ground_truth(dataset_episode):
    table, info = dataset_episode
    config = fit_config_from_info(info)
    curve = build_episode_curve(table, "dataset", config)
    gt = episode_ground_truth(table, config)
    env = MockFDCCEnv(control_fps=DEMO_FPS, initial_position=gt["eef_pos"][0], initial_ortho6=gt["eef_rot6"][0])
    cfg = PlayheadConfig(speed=2.0, demo_fps=DEMO_FPS, control_fps=DEMO_FPS)
    result = run_tick_loop(env, curve, Playhead(cfg, *curve.domain))
    comparison = compare_with_dataset(result, gt)
    assert comparison["n_ticks"] == count_ticks(cfg, curve.domain[1])
    assert comparison["nominal_speedup"] == pytest.approx(2.0, abs=0.02)
    # the mock tracks the commanded reference perfectly; against the fitted
    # label at frame round(u) the residual is the fit error + half-sample sampling
    assert comparison["mean_pos_err_vs_command_m"] == 0.0
    assert comparison["mean_pos_err_vs_label_m"] < 1.5e-3
    assert comparison["mean_cmd_force_err_vs_label_N"] < 0.2
    assert comparison["clip_count"] == 0


def test_mock_dense_replay_of_dataset_episode(dataset_episode):
    """The raw dataset commands replayed row by row on the mock (the 1x baseline)."""
    table, info = dataset_episode
    config = fit_config_from_info(info)
    gt = episode_ground_truth(table, config)
    for col in dense_action_columns("raw_actions", available=list(table.columns)):
        assert col in table.columns

    env = MockFDCCEnv(control_fps=DEMO_FPS, initial_position=gt["eef_pos"][0], initial_ortho6=gt["eef_rot6"][0])
    env.reset(move_robot=False)
    env.activate_compliance_control()
    result = run_dense_loop(env, table, "raw_actions", control_fps=DEMO_FPS, demo_fps=DEMO_FPS)

    n = int(gt["n_frames"])
    assert result.n_ticks == n and not result.force_violation
    np.testing.assert_array_equal(result.ds_indices, np.arange(n))
    # the mock jumps to the commanded pose, so tracking is exact ...
    np.testing.assert_allclose(result.eef_pos, result.command_pos, atol=1e-12)

    comparison = compare_dense_with_dataset(result, gt)
    assert comparison["mean_pos_err_vs_command_m"] == 0.0
    # ... so what is left is the demo's own command-vs-pose offset: the recorded
    # action leads the eef by the compliance displacement (~2 cm on this episode)
    assert 1e-3 < comparison["mean_pos_err_vs_obs_m"] < 5e-2
    assert comparison["clip_count"] == 0
    assert comparison["n_ticks"] == n
    assert all(isinstance(v, float) for v in comparison.values())
    assert f"[raw_actions] episode 0" in dense_summary_text(0, result, comparison)

    # ... and the spline replay can be paired against it on the dataset frame
    curve = build_episode_curve(table, "dataset", config)
    spline_env = MockFDCCEnv(control_fps=DEMO_FPS, initial_position=gt["eef_pos"][0], initial_ortho6=gt["eef_rot6"][0])
    cfg = PlayheadConfig(speed=2.0, demo_fps=DEMO_FPS, control_fps=DEMO_FPS)
    spline = run_tick_loop(spline_env, curve, Playhead(cfg, *curve.domain))
    vs = compare_with_dense(spline, result)
    assert vs["n_paired"] == spline.n_ticks
    # the two replays command different things (the fitted FCT reference vs the
    # recorded pose action), so on the mock this is their offset, not a tracking
    # error — it stays in the centimetre range the compliance displacement spans
    assert vs["mean_pos_err_vs_dense_m"] < 5e-2
