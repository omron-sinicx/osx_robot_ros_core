"""Shared pure-math helpers for pose-deflection-driven variable stiffness.

Used by dynamic_compliance_test.py / stiffness_test.py (manual hold-and-feel
tests) and evaluate_policy_points.py (continuous linear stiffness during
rollouts, see PoseStiffnessModulator) -- kept here, in the osx_ur5e library
package, rather than duplicated or imported between scripts, since neither is
a natural place for the other to depend on.
"""
import numpy as np


def quat_to_rotvec_abs(qx, qy, qz, qw):
    """Per-axis |rotation vector| (rad) from a relative unit quaternion --
    i.e. how much the tool has rotated about its own x/y/z since the origin.
    angle = 2*acos(|w|) (double-cover safe), axis = (qx,qy,qz) normalized."""
    w = min(abs(qw), 1.0)
    angle = 2.0 * np.arccos(w)
    v = np.array([qx, qy, qz], dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-9 or angle < 1e-9:
        return np.zeros(3)
    axis = v / n
    return np.abs(axis * angle)


def distance_to_stiffness(dist, near, far, s_min, s_max, profile):
    """Monotonically DECREASING map: dist<=near -> s_max, dist>=far -> s_min,
    interpolated in between ('linear' or 'tanh', which saturates faster near
    'far')."""
    dist = np.asarray(dist, dtype=float)
    if far <= near:
        frac = np.where(dist <= near, 0.0, 1.0)
    elif profile == "tanh":
        u = np.clip((dist - near) / (far - near), 0.0, 1.0)
        frac = np.tanh(3.0 * u) / np.tanh(3.0)
    else:
        frac = np.clip((dist - near) / (far - near), 0.0, 1.0)
    stiffness = s_max - frac * (s_max - s_min)
    return np.clip(stiffness, min(s_min, s_max), max(s_min, s_max))


def step_with_hysteresis(dist, near, far, s_min, s_max, was_soft):
    """2-stage (bang-bang) map: only two output values, s_max (STIFF) or s_min
    (SOFT) -- no values in between. 'near' and 'far' double as a hysteresis
    band instead of a ramp: switches STIFF -> SOFT once dist >= far, and only
    back SOFT -> STIFF once dist <= near; while near < dist < far it holds
    whatever stage it was already in (was_soft, latched per-axis state from
    the previous call) so it doesn't chatter back and forth right at a single
    threshold. Returns (stiffness, now_soft) -- now_soft is the new latch to
    pass back in next call.

    near should be kept comfortably above the pose source's noise floor: at
    near=0.0 the SOFT->STIFF unlatch condition (dist<=near) is practically
    unreachable for a noisy real sensor/estimator signal, since it rarely
    reads back exactly 0 even at rest -- so the latch can get stuck SOFT."""
    dist = np.asarray(dist, dtype=float)
    was_soft = np.asarray(was_soft, dtype=bool)
    now_soft = np.where(dist >= far, True, np.where(dist <= near, False, was_soft))
    stiffness = np.where(now_soft, s_min, s_max)
    return stiffness, now_soft
