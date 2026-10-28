#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect a sensors -> wrist-deflection dataset by moving the UR5e while the claw
grips a post BOLTED TO THE TABLE, bounding every motion by force and by deflection.

WHY THIS EXISTS
The mocap label path (bendlabs/scripts/record_pose_dataset.py) needs NatNet and has
been fragile: Motive re-derives a rigid body's local axes when the body is recreated,
which silently flipped the correct frame rotation by 180 deg on 2026-08-18. The FK
label path is both mocap-free and measurably better (0.47mm / 0.48deg val RMSE vs
4.32mm / 1.65deg). This script is the FK path with a rigid anchor and a force bound.

WHAT MAKES THE LABEL WORK
With the claw gripping an anchored post, the tool end cannot move, so the flange's
own displacement IS (minus) the wrist deflection. prepare_deflection_dataset.py turns
that into the training label, per segment, relative to the segment's FIRST ROW:

    pos  = -(R0^T (p - p0))      metres, in the start flange frame
    quat = quat((R0^T R)^T)      tool-relative-to-flange

Two consequences drive this whole file:
  * row 0 of every CSV must be a VERIFIED zero-deflection pose, hence the
    wrench-nulling servo in ExcursionRunner.establish_zero() and one CSV per
    excursion;
  * deflection length and flange displacement are the SAME quantity, which is why
    the deflection bound below costs nothing extra to compute.

THE BOUND
Every outward excursion stops on a single normalised load fraction:

    lam = max( ||F||/force_cap, ||T||/torque_cap,
               ||u[:3]||/defl_trans_cap, ||u[3:]||/defl_rot_cap,
               max_i |s_i - s_i0| / sensor_cap_i )

Boundary at lam >= 1. Because every mechanism (adaptive step, predictive stop, trip
latch, envelope learning) reads only this scalar, adding limits is structurally free.
stop_reason records WHICH term bound.

Note the runaway detector deliberately uses the FORCE-ONLY fraction: with deflection
in lam, a dead wrench topic no longer pins lam at zero (good -- it means the 247mm
runaway in force_dataset_20260812_022007 cannot recur), but the combined lam would
then reach 1 via deflection and mask exactly the failure being detected.

THE GEOMETRIC CAGE IS SEPARATE AND HARDER
DisplacementLimiter is copied verbatim and every commanded pose goes through
pose_from_u, so the geometric limit is enforced BY CONSTRUCTION. The force cap is
reactive and fails open; the cage is constructive and fails closed.

OPERATING ORDER MATTERS -- uart_bridge_node.py averages each sensor's first 10 raw
samples into an offset, so whatever the wrist is doing at bridge startup DEFINES
sensor zero. Start the bridge with the claw OPEN and the wrist hanging free, before
gripping the post. See --park_only and --require_sensor_zero_check.

This is a standalone script. It deliberately does NOT import collect_force_dataset:
that module installs a module-level SIGINT handler which sys.exit(0)s mid-command --
the exact Ctrl+C behaviour replaced here -- and pulls in matplotlib. Blocks copied
from it carry a provenance comment naming the source lines.

Usage:
    # off-hardware: full session against a simulated wrist, no robot, no roscore
    ./collect_deflection_force_bounded.py --dry_run --self_test

    ./collect_deflection_force_bounded.py --park_only
    ./collect_deflection_force_bounded.py --force_cap 5 --torque_cap 0.5 \
        --probe_excursions 6 --hop_waypoints 0
    ./collect_deflection_force_bounded.py            # full session
"""

import argparse
import atexit
import csv
import datetime
import os
import signal
import sys
from abc import ABC, abstractmethod

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# ROS imports are deferred: --dry_run must work with no master and no roscore,
# and importing rospy is harmless but calling init_node is not.
# ---------------------------------------------------------------------------
try:
    import rospy
    from ur_control import transformations
    from ur_control.math_utils import quaternion_from_axis_angle
    _ROS_IMPORTS_OK = True
except Exception as _e:                                    # pragma: no cover
    rospy = None
    transformations = None
    quaternion_from_axis_angle = None
    _ROS_IMPORTS_OK = False
    _ROS_IMPORT_ERROR = _e


# ==================== LOGGING SHIM ====================
# rospy.loginfo before init_node prints to stdout anyway, but in --dry_run we may
# not have rospy at all, so route everything through here.

_USE_ROSLOG = False


def _set_roslog(enabled):
    global _USE_ROSLOG
    _USE_ROSLOG = bool(enabled)


def log_info(msg):
    if _USE_ROSLOG and rospy is not None:
        rospy.loginfo(msg)
    else:
        print(f"[INFO] {msg}")


def log_warn(msg):
    if _USE_ROSLOG and rospy is not None:
        rospy.logwarn(msg)
    else:
        print(f"[WARN] {msg}")


def log_err(msg):
    if _USE_ROSLOG and rospy is not None:
        rospy.logerr(msg)
    else:
        print(f"[ERROR] {msg}")


# ==================== CONSTANTS ====================

# copied from collect_force_dataset.py:274-285 -- the geometric cage, unchanged.
DISPLACEMENT_THRESHOLDS = {
    'x': 0.018,
    'y': 0.018,
    'z': 0.012,
    'xy_pos45': 0.018,
    'xy_neg45': 0.018,
    'yaw': 10.0,
    'pitch': 10.0,
    'roll': 10.0,
    'roll_pitch_pos45': 10.0,
    'roll_pitch_neg45': 10.0,
}

# Measured from deformation_test_20260819_071530 rep1 (grasped-specimen setup):
#   x    18.52mm -> 11.53N / 1.14Nm      roll             2.57mm -> 20.15N / 1.79Nm
#   z    12.38mm -> 13.00N / 0.20Nm      yaw              0.63mm ->  4.14N / 0.51Nm
#   roll_pitch_neg45  2.16mm -> 26.85N / 2.39Nm
# Rotation sweeps are FORCE-limited, not torque-limited, which is why the caps below
# are resultant magnitudes rather than per-DoF.
DEFAULT_FORCE_CAP = 15.0        # N
DEFAULT_TORQUE_CAP = 1.5        # Nm
DEFAULT_HARD_ABORT_FT = [30.0, 30.0, 30.0, 3.0, 3.0, 3.0]

# Matches pose_stiffness_trans_far / rot_far in osx_ur5e/config/hydra/test_task.yaml,
# so the training data covers exactly the deflection range the live controller uses.
DEFAULT_DEFL_TRANS_CAP = 0.015  # m
DEFAULT_DEFL_ROT_CAP = 0.16     # rad (9.2 deg), just inside the 10 deg cage

# copied from collect_force_dataset.py:344 (use_default_config)
DEFAULT_INITIAL_UR = np.radians([75.59, -85.07, 114.92, -120.95, -90.70, -13.37])

(PHASE_NULL, PHASE_LOAD, PHASE_HOLD, PHASE_UNLOAD, PHASE_RELAX, PHASE_TRANSIT,
 PHASE_DWELL) = range(7)


# ==================== COPIED GEOMETRY ====================
# copied from collect_force_dataset.py:379-446 (ForceDatasetCollector.calculate_displacement),
# lifted to module scope: it never touched `self`, and as a free function it works with
# no robot, no ROS master and no controller stack -- which is what lets --dry_run
# validate the cage.

def calculate_displacement(initial_pose, current_pose, direction):
    """Displacement of `current_pose` from `initial_pose` along `direction`.

    Translation keys return metres, rotation keys return degrees.
    """
    if direction == 'x':
        return abs(current_pose[0] - initial_pose[0])
    elif direction == 'y':
        return abs(current_pose[1] - initial_pose[1])
    elif direction == 'z':
        return abs(current_pose[2] - initial_pose[2])
    elif direction in ['xy_pos45', 'xy_neg45']:
        dx = current_pose[0] - initial_pose[0]
        dy = current_pose[1] - initial_pose[1]
        return np.sqrt(dx ** 2 + dy ** 2)
    elif direction in ['yaw', 'pitch', 'roll', 'roll_pitch_pos45', 'roll_pitch_neg45']:
        initial_quat = np.asarray(initial_pose[3:7], dtype=float)
        current_quat = np.asarray(current_pose[3:7], dtype=float)

        if direction in ['roll_pitch_pos45', 'roll_pitch_neg45']:
            # axis-angle, to avoid Euler singularities on the diagonal
            iq = initial_quat / np.linalg.norm(initial_quat)
            cq = current_quat / np.linalg.norm(current_quat)
            iq_inv = np.array([-iq[0], -iq[1], -iq[2], iq[3]])
            rel = transformations.quaternion_multiply(cq, iq_inv)
            return np.rad2deg(np.linalg.norm(
                transformations.axis_angle_from_quaternion(rel)))
        else:
            ie = transformations.euler_from_quaternion(initial_quat)
            ce = transformations.euler_from_quaternion(current_quat)
            idx = {'roll': 0, 'pitch': 1, 'yaw': 2}[direction]
            diff = ce[idx] - ie[idx]
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            return abs(np.rad2deg(diff))

    return 0.0


class DisplacementLimiter:
    """copied from collect_force_dataset.py:130-220, verbatim in behaviour.

    Decides whether a pose is within DISPLACEMENT_THRESHOLDS of `center`, checking
    EVERY threshold key -- including the magnitude-based diagonal keys that a
    per-axis box would miss.

    Here its role is the geometric CAGE, not the primary bound: because every
    commanded pose is built by pose_from_u and clamped by max_scale, the cage cannot
    be violated even if the wrench topic dies.
    """

    def __init__(self, center, thresholds, margin=1.0):
        self.center = np.asarray(center, dtype=float)
        self.thresholds = dict(thresholds)
        self.margin = float(margin)

        self.trans_extent = np.array([self.thresholds.get('x', 0.0),
                                      self.thresholds.get('y', 0.0),
                                      self.thresholds.get('z', 0.0)], dtype=float)
        rot_keys = ('roll', 'pitch', 'yaw', 'roll_pitch_pos45', 'roll_pitch_neg45')
        rot_deg = [self.thresholds[k] for k in rot_keys if k in self.thresholds]
        self.rot_extent = np.deg2rad(min(rot_deg)) if rot_deg else 0.0

    def measure(self, pose):
        pose = np.asarray(pose, dtype=float)
        return {k: float(calculate_displacement(self.center, pose, k))
                for k in self.thresholds}

    def violations(self, pose, margin=None):
        m = self.margin if margin is None else margin
        out = {}
        for k, v in self.measure(pose).items():
            allowed = self.thresholds[k] * m
            if v > allowed + 1e-12:
                out[k] = (v, allowed)
        return out

    def is_admissible(self, pose, margin=None):
        return not self.violations(pose, margin)

    def pose_from_u(self, u):
        """u = [dx,dy,dz,rx,ry,rz] (m, rad, relative to center) -> [x,y,z,qx,qy,qz,qw].

        Rotation composed as quaternion_multiply(rot, center_quat), the same order
        the scripted sweeps use.
        """
        u = np.asarray(u, dtype=float)
        pose = np.empty(7)
        pose[:3] = self.center[:3] + u[:3]
        rot = np.asarray(quaternion_from_axis_angle(u[3:]), dtype=float)
        pose[3:] = transformations.quaternion_multiply(rot, self.center[3:])
        return pose

    def u_of(self, pose):
        """Inverse of pose_from_u for a measured pose."""
        pose = np.asarray(pose, dtype=float)
        u = np.empty(6)
        u[:3] = pose[:3] - self.center[:3]
        c_inv = transformations.quaternion_conjugate(self.center[3:])
        rel = transformations.quaternion_multiply(pose[3:], c_inv)
        u[3:] = transformations.axis_angle_from_quaternion(rel)
        return u

    def max_scale(self, direction, hi=2.0, iters=24):
        """Largest s >= 0 with pose_from_u(s*direction) still admissible.

        Bisection rather than rejection sampling: it always returns an admissible
        point, so sampling cannot fail or bias toward the loosest constraint.
        """
        direction = np.asarray(direction, dtype=float)
        if not self.is_admissible(self.pose_from_u(np.zeros(6))):
            return 0.0
        lo, hi_s = 0.0, hi
        while self.is_admissible(self.pose_from_u(hi_s * direction)) and hi_s < 1e3:
            lo, hi_s = hi_s, hi_s * 2.0
        for _ in range(iters):
            mid = 0.5 * (lo + hi_s)
            if self.is_admissible(self.pose_from_u(mid * direction)):
                lo = mid
            else:
                hi_s = mid
        return lo


# ==================== LOAD METRIC ====================

class LoadMetric:
    """Turns (wrench, u, sensors) into the single normalised load fraction lam.

    Every stopping decision in this file reads only lam and lam_force, which is what
    makes force and deflection limits compose without touching the motion logic.
    """

    def __init__(self, force_cap, torque_cap, defl_trans_cap, defl_rot_cap,
                 hard_abort_ft, sensor_caps=None, trip_samples=2):
        self.force_cap = float(force_cap)
        self.torque_cap = float(torque_cap)
        self.defl_trans_cap = float(defl_trans_cap)
        self.defl_rot_cap = float(defl_rot_cap)
        self.hard_abort_ft = np.asarray(hard_abort_ft, dtype=float)
        self.sensor_caps = None if sensor_caps is None else np.asarray(sensor_caps, float)
        self.trip_samples = int(trip_samples)
        self._above = 0

    def terms(self, wrench, u, sensors=None, sensors0=None):
        """{name -> normalised fraction}. The max over these is lam."""
        w = np.asarray(wrench, dtype=float)
        u = np.asarray(u, dtype=float)
        out = {
            'force': float(np.linalg.norm(w[:3]) / self.force_cap),
            'torque': float(np.linalg.norm(w[3:]) / self.torque_cap),
            'defl_trans': float(np.linalg.norm(u[:3]) / self.defl_trans_cap),
            'defl_rot': float(np.linalg.norm(u[3:]) / self.defl_rot_cap),
        }
        if self.sensor_caps is not None and sensors is not None and sensors0 is not None:
            s = np.asarray(sensors, dtype=float)
            s0 = np.asarray(sensors0, dtype=float)
            if s.shape == s0.shape == self.sensor_caps.shape and np.all(np.isfinite(s)):
                out['sensor'] = float(np.max(np.abs(s - s0) / self.sensor_caps))
        return out

    def lam(self, wrench, u, sensors=None, sensors0=None):
        return max(self.terms(wrench, u, sensors, sensors0).values())

    def lam_force(self, wrench):
        """Force-only fraction.

        The runaway detector MUST use this, not the combined lam: with deflection in
        lam, a dead wrench topic would still drive lam to 1 via the deflection term
        and mask exactly the failure being detected.
        """
        w = np.asarray(wrench, dtype=float)
        return max(float(np.linalg.norm(w[:3]) / self.force_cap),
                   float(np.linalg.norm(w[3:]) / self.torque_cap))

    def binding_term(self, wrench, u, sensors=None, sensors0=None):
        t = self.terms(wrench, u, sensors, sensors0)
        return max(t, key=t.get)

    def hard_abort(self, wrench):
        """Per-axis absolute abort. copied in spirit from collect_force_dataset.py:1716."""
        return bool(np.any(np.abs(np.asarray(wrench, float)) > self.hard_abort_ft))

    def update_trip(self, lam):
        """Latching trip: lam is not monotone in s (hysteresis + controller lag), so a
        single-sample comparison chatters."""
        if lam >= 1.0:
            self._above += 1
        else:
            self._above = 0
        return self._above >= self.trip_samples

    def reset_trip(self):
        self._above = 0


# ==================== RADIAL ENVELOPE ====================

class RadialEnvelope:
    """Nonparametric estimate of the admissible radius per direction on S^5.

    The force-bounded region cannot be pre-validated (evaluating admissibility means
    physically going there), is not convex -- roll reaches 20N of FORCE at only
    1.8Nm of torque, so the coupling has no clean structure -- and drifts with creep.
    A radial kernel estimate needs no structural assumption and degrades to a
    conservative prior where it has no data.

    Safety NEVER depends on this: it only decides where to aim. Every commanded pose
    is still clamped by the geometric cage and interrupted by lam.
    """

    def __init__(self, sigma=0.5, halflife=300.0, prior_radius=0.3, w_min=0.15):
        self.sigma = float(sigma)
        self.halflife = float(halflife)
        self.prior_radius = float(prior_radius)
        self.w_min = float(w_min)
        self.dirs = []       # unit vectors in normalised u-space
        self.radii = []
        self.censored = []
        self.times = []

    def add(self, d_hat, r, censored, t_now):
        d = np.asarray(d_hat, dtype=float)
        n = np.linalg.norm(d)
        if n < 1e-12:
            return
        self.dirs.append(d / n)
        self.radii.append(float(r))
        self.censored.append(bool(censored))
        self.times.append(float(t_now))

    def query(self, d_hat, t_now, s_backstop):
        """Estimated admissible radius along d_hat, always clamped to the cage."""
        d = np.asarray(d_hat, dtype=float)
        d = d / max(np.linalg.norm(d), 1e-12)
        if not self.dirs:
            return self.prior_radius * s_backstop

        D = np.asarray(self.dirs)
        R = np.asarray(self.radii)
        T = np.asarray(self.times)
        cos = np.clip(D @ d, -1.0, 1.0)
        w = np.exp((cos - 1.0) / (self.sigma ** 2))
        if self.halflife > 0:
            # Age is clamped at zero: an observation must never be treated as being
            # from the FUTURE. A loaded envelope carries the previous session's
            # absolute timestamps, and a negative age makes 0.5**(negative/halflife)
            # overflow to inf, which poisons the weighted mean to NaN.
            age = np.maximum(t_now - T, 0.0)
            w = w * 0.5 ** (age / self.halflife)
        tot = float(np.sum(w))
        if not np.isfinite(tot) or tot < self.w_min:
            return self.prior_radius * s_backstop
        r_hat = float(np.sum(w * R) / tot)
        if not np.isfinite(r_hat):
            return self.prior_radius * s_backstop
        # Censored observations are lower bounds, and the cage clamp below means no
        # query can extrapolate past the cage regardless.
        return float(min(r_hat, s_backstop))

    def save(self, path):
        np.savez(path,
                 dirs=np.asarray(self.dirs, dtype=float).reshape(-1, 6),
                 radii=np.asarray(self.radii, dtype=float),
                 censored=np.asarray(self.censored, dtype=bool),
                 times=np.asarray(self.times, dtype=float))

    def load(self, path, t_now=None):
        """Load a saved envelope, rebasing its timestamps onto the current clock.

        Timestamps are absolute (rospy epoch seconds), so a file from an earlier
        session would otherwise be aged out entirely by the half-life decay -- the
        whole envelope would be silently ignored and every query would fall back to
        prior_radius, which is exactly the opposite of what --load_envelope is for.
        Rebasing so the newest observation lands at t_now preserves the relative
        spacing within the file while treating the file as a whole as current.
        """
        z = np.load(path)
        self.dirs = [np.asarray(d, float) for d in z['dirs']]
        self.radii = list(map(float, z['radii']))
        self.censored = list(map(bool, z['censored']))
        times = np.asarray(z['times'], dtype=float)
        if t_now is not None and len(times):
            times = times - float(times.max()) + float(t_now)
        self.times = list(map(float, times))


# ==================== DIRECTION SAMPLER ====================

class DirectionSampler:
    """Unit directions in normalised u-space.

    The first `n_seed` are a deterministic axis spread so coverage matches the old
    scripted sweeps and a run is reproducible; after that, max-min rejection, because
    plain isotropic sampling clumps badly in 6-D.
    """

    def __init__(self, rng, n_seed=12, n_candidates=16, axis_frac=0.35):
        self.rng = rng
        self.axis_frac = float(axis_frac)
        self.n_candidates = int(n_candidates)
        self.used = []
        self.seeds = []
        for axis in range(6):
            for sign in (+1.0, -1.0):
                d = np.zeros(6)
                d[axis] = sign
                self.seeds.append(d)
        self.seeds = self.seeds[:int(n_seed)]
        self._seed_i = 0

    def _isotropic(self):
        d = self.rng.normal(size=6)
        n = np.linalg.norm(d)
        if n < 1e-9:
            d = np.array([1.0, 0, 0, 0, 0, 0])
            n = 1.0
        return d / n

    def _sparse(self, k):
        """A direction with only k non-zero components.

        Isotropic 6-D directions split their magnitude across all six axes, so no
        single axis ever gets near its own cap -- coverage of each individual DoF
        stays narrow even though the joint space is well covered. Mixing in sparse
        directions restores per-axis reach, which is what the MLP's output range
        actually depends on.
        """
        d = np.zeros(6)
        idx = self.rng.choice(6, size=k, replace=False)
        d[idx] = self.rng.normal(size=k)
        n = np.linalg.norm(d)
        return d / n if n > 1e-9 else self._isotropic()

    def next(self):
        if self._seed_i < len(self.seeds):
            d = self.seeds[self._seed_i]
            self._seed_i += 1
        elif self.rng.random() < self.axis_frac:
            d = self._sparse(int(self.rng.integers(1, 3)))     # 1 or 2 live axes
        elif not self.used:
            d = self._isotropic()
        else:
            U = np.asarray(self.used)
            best, best_score = None, -np.inf
            for _ in range(self.n_candidates):
                c = self._isotropic()
                score = -np.max(np.abs(U @ c))   # maximise angular distance to all used
                if score > best_score:
                    best, best_score = c, score
            d = best
        self.used.append(d)
        return d


# ==================== SLIP MONITOR ====================

class SlipMonitor:
    """Detects the anchor moving. Check (b) is the capability the bolted post buys --
    prepare_deflection_dataset.py:40-43 states nothing in the CSV can detect slip, and
    with a free specimen that was true.
    """

    def __init__(self, pose_tol=0.002, rot_tol_deg=1.0, wrench_tol=2.0,
                 wrench_abs=5.0, claw_gap_tol=0.05):
        self.pose_tol = float(pose_tol)
        self.rot_tol = np.deg2rad(float(rot_tol_deg))
        self.wrench_tol = float(wrench_tol)
        self.wrench_abs = float(wrench_abs)
        self.claw_gap_tol = float(claw_gap_tol)
        self.residuals = []
        self.prev_zero_pose = None
        self.prev_claw_gap = None

    def check(self, zero_pose, residual_wrench, claw_gap, limiter, null_travel=0.0,
              null_travel_rot=0.0):
        """-> (verdicts dict, fatal bool). Fatal ends the session."""
        v = {}
        fatal = False

        # (a) round-trip residual, vs the RUNNING MEDIAN rather than zero: the
        # measured hysteretic residual is 0.8-3.05N even on a healthy run, so an
        # absolute test against zero would fire constantly.
        rf = float(np.linalg.norm(np.asarray(residual_wrench, float)[:3]))
        if self.residuals:
            med = float(np.median(self.residuals))
            v['residual_dev'] = abs(rf - med)
            if abs(rf - med) > self.wrench_tol:
                v['suspect_residual'] = True
        v['residual_force'] = rf
        if rf > self.wrench_abs:
            v['suspect_residual'] = True
        self.residuals.append(rf)

        # (b) zero-load fixed-point drift. Under a rigid anchor the zero-load flange
        # pose is a fixed point of the closed chain; a jump between consecutive
        # excursions is slip or a post that moved.
        if self.prev_zero_pose is not None:
            du = limiter.u_of(zero_pose) - limiter.u_of(self.prev_zero_pose)
            dt_ = float(np.linalg.norm(du[:3]))
            dr_ = float(np.linalg.norm(du[3:]))
            v['zero_drift_trans'] = dt_
            v['zero_drift_rot'] = dr_
            # The wrench-nulling servo deliberately repositions the origin to find
            # zero deflection; only drift BEYOND what nulling can explain is slip.
            allow_t = self.pose_tol + float(null_travel)
            allow_r = self.rot_tol + float(null_travel_rot)
            if dt_ > allow_t or dr_ > allow_r:
                v['slip_fixed_point'] = True
                fatal = True
        self.prev_zero_pose = np.asarray(zero_pose, float).copy()

        # (c) claw command-vs-measured gap. The motor is in current-based position
        # control (Operating_Mode 5, Goal_Current 100) so the jaws yield under torque;
        # a collapsing gap means the post was lost.
        if claw_gap is not None:
            v['claw_gap'] = float(claw_gap)
            if self.prev_claw_gap is not None:
                if abs(claw_gap - self.prev_claw_gap) > self.claw_gap_tol:
                    v['slip_claw'] = True
                    fatal = True
            self.prev_claw_gap = float(claw_gap)

        return v, fatal


# ==================== RECORDER ====================

class Recorder:
    """Per-excursion sample buffers -> one CSV.

    The first 21 columns are byte-identical to collect_force_dataset.py:1322-1327 so
    prepare_deflection_dataset.py consumes them unchanged; it builds its output from
    an explicit column list, so the extra diagnostic columns are ignored.
    """

    BASE_HEADER = ['Time', 'Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz',
                   'Pos_X', 'Pos_Y', 'Pos_Z', 'Orient_1', 'Orient_2', 'Orient_3']
    EXTRA_HEADER = ['Fx_s', 'Fy_s', 'Fz_s', 'Tx_s', 'Ty_s', 'Tz_s',
                    'Lambda', 'Lam_force', 'Defl_trans', 'Defl_rot',
                    'Phase', 'S_cmd', 'Claw_cmd', 'Claw_meas', 'Track_err']

    def __init__(self, num_sensors):
        self.num_sensors = int(num_sensors)
        self.rows = []

    def reset(self):
        self.rows = []

    def add(self, t, wrench_base, eef, bend, wrench_raw, lam, lam_force,
            defl_trans, defl_rot, phase, s_cmd, claw_cmd, claw_meas, track_err):
        row = [t]
        row += list(np.asarray(wrench_base, float))
        row += list(np.asarray(eef, float))
        row += list(np.asarray(bend, float))
        row += list(np.asarray(wrench_raw, float))
        row += [lam, lam_force, defl_trans, defl_rot, phase, s_cmd,
                claw_cmd if claw_cmd is not None else np.nan,
                claw_meas if claw_meas is not None else np.nan,
                track_err]
        self.rows.append(row)

    def header(self):
        h = list(self.BASE_HEADER)
        for s in range(self.num_sensors):
            h += [f'S{s}_twist', f'S{s}_bend']
        h += list(self.EXTRA_HEADER)
        return h

    def write(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(self.header())
            w.writerows(self.rows)
        return path


# ==================== CLOCK ====================

class Clock(ABC):
    @abstractmethod
    def now(self): ...
    @abstractmethod
    def sleep(self, dt): ...
    @abstractmethod
    def ok(self): ...


class RosClock(Clock):
    def now(self):
        return rospy.get_time()

    def sleep(self, dt):
        rospy.sleep(dt)

    def ok(self):
        return not rospy.is_shutdown()


class SimClock(Clock):
    # Deliberately NOT zero: rospy.get_time() returns epoch seconds, so a sim clock
    # starting at 0 lets "absolute vs elapsed" confusions pass the dry run and fail
    # on the robot. Starting in the same numeric range makes the dry run exercise the
    # same arithmetic.
    EPOCH0 = 1787000000.0

    def __init__(self):
        self.t = self.EPOCH0

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += float(dt)

    def ok(self):
        return True


# ==================== PLANT ====================

class Plant(ABC):
    """Everything that touches hardware. Keeping this boundary is what makes
    --dry_run a real end-to-end exercise of Session rather than a mock of it."""

    @abstractmethod
    def end_effector(self): ...
    @abstractmethod
    def set_target(self, pose): ...
    @abstractmethod
    def wrench_raw(self): ...
    @abstractmethod
    def wrench_base(self): ...
    @abstractmethod
    def bend(self): ...
    def claw_cmd(self): return None
    def claw_meas(self): return None
    def claw_open(self): pass
    def claw_close(self): pass
    def zero_ft(self): pass
    def protective_stopped(self): return False
    def park(self, joints, target_time=5.0): pass
    def activate_compliance(self): pass
    def activate_trajectory(self): pass
    def stop(self): pass


class RosPlant(Plant):
    def __init__(self, args):
        from ur_control.fzi_cartesian_compliance_controller import CompliantController
        from bendlabs.bendlabs_recorder import BendLabsRecorder

        self.args = args
        self.num_sensors = args.num_sensors

        # copied from collect_force_dataset.py:310-328
        self.arm = CompliantController(
            namespace=None,
            joint_names_prefix=None,
            ee_link="tool0",
            ft_topic="wrench",
            gripper_type=None,
        )
        self.arm.set_position_control_mode(False)
        self.arm.set_control_mode(mode="spring-mass-damper")
        self.arm.set_solver_parameters(error_scale=1.5, iterations=1)
        self.arm.update_stiffness([3000, 3000, 3000, 100, 100, 100])
        self.arm.update_selection_matrix(np.array([1, 1, 1, 1, 1, 1]))
        self.arm.update_pd_gains([0.02, 0.02, 0.02, 1.5, 1.5, 1.5],
                                 d_gains=[0.0] * 6)

        self.gripper = None
        if not args.no_claw:
            from osx_claw.claw_controller import ClawController
            self.gripper = ClawController(init_node=False)

        try:
            self.bend_recorder = BendLabsRecorder(init_node=False,
                                                  num_sensors=self.num_sensors)
        except Exception as e:
            log_warn(f"Could not start BendLabsRecorder: {e}. Bend data will be NaN.")
            self.bend_recorder = None

    def end_effector(self):
        return np.asarray(self.arm.end_effector(), dtype=float)

    def eef_axis_angle(self):
        """copied from collect_force_dataset.py:448-465 -- Pos_X..Orient_3.

        prepare_deflection_dataset.py's aa_to_R depends on exactly this axis-angle
        representation, so it must not change.
        """
        p = self.arm.end_effector()
        return np.concatenate([p[:3],
                               transformations.axis_angle_from_quaternion(p[3:])])

    def set_target(self, pose):
        self.arm.set_cartesian_target_pose(list(np.asarray(pose, float)))

    def wrench_raw(self):
        return np.asarray(self.arm.get_wrench(), dtype=float)

    def wrench_base(self):
        return np.asarray(self.arm.get_wrench(base_frame_control=True), dtype=float)

    def bend(self):
        # copied from collect_force_dataset.py:370-377
        if self.bend_recorder is not None and self.bend_recorder.sensors_ready():
            return np.asarray(self.bend_recorder.get_angles(), dtype=float)
        return np.full(2 * self.num_sensors, np.nan, dtype=float)

    def claw_cmd(self):
        if self.gripper is None:
            return None
        try:
            return float(self.gripper.get_normalized_command())
        except Exception:
            return None

    def claw_meas(self):
        if self.gripper is None:
            return None
        try:
            return float(self.gripper.get_normalized_position())
        except Exception:
            return None

    def claw_open(self):
        if self.gripper is not None:
            self.gripper.open_gripper()

    def claw_close(self):
        if self.gripper is not None:
            self.gripper.close_gripper()

    def zero_ft(self):
        self.arm.zero_ft_sensor()

    def protective_stopped(self):
        try:
            return bool(self.arm.dashboard_services.is_protective_stopped())
        except Exception:
            return False

    def park(self, joints, target_time=5.0):
        self.activate_trajectory()
        self.arm.set_joint_positions(target_time=target_time,
                                     positions=np.asarray(joints, float), wait=True)

    def activate_compliance(self):
        self.arm.activate_cartesian_controller()

    def activate_trajectory(self):
        self.arm.activate_joint_trajectory_controller()

    def stop(self):
        try:
            self.arm.stop_robot()
        except Exception:
            pass


class SimPlant(Plant):
    """Anisotropic, stiffening, hysteretic wrist with first-order command lag.

    Deliberately 3:1 anisotropic so --self_test can check the envelope RECOVERS the
    anisotropy rather than merely staying inside the cage.
    """

    def __init__(self, args, rng):
        self.args = args
        self.rng = rng
        self.num_sensors = args.num_sensors

        k = float(args.sim_stiffness) * 1000.0        # N/mm -> N/m
        self.K = np.array([k, k * 3.0, k * 1.7, 0, 0, 0], dtype=float)
        # Calibrated against deformation_test_20260819_071530 and the first hardware
        # run of this script: rotation excursions there reached the 9.2 deg deflection
        # cap with torque still well under 1.5 Nm (lam_F 0.30-0.80), i.e. they were
        # FORCE-limited, not torque-limited. The old 40 Nm/rad made rotation stop on
        # torque at s~0.2, an unrepresentative regime that also starved the envelope.
        self.Krot = np.array([0, 0, 0, 8.0, 8.0, 5.0], dtype=float)      # Nm/rad
        # Rotating the flange about a point offset from the anchor produces FORCE as
        # well as torque -- this coupling is why the reference data shows roll_pitch
        # hitting 26.9 N against only 2.4 Nm.
        self.rot_force_coupling = 60.0                                   # N/rad
        self.cubic = 4.0e5
        self.hyst_alpha = float(args.sim_hysteresis)
        self.lag = int(args.sim_lag)
        self.fault = args.sim_fault
        self.noise = float(getattr(args, 'sim_noise', 0.02))
        self.rate = float(args.rate)

        self.center_pose = np.array([0.4, 0.1, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=float)
        self.cur = self.center_pose.copy()
        self._queue = []
        self._hyst = np.zeros(6)
        self._steps = 0
        self._zero_offset = np.zeros(6)
        self._anchor_shift = np.zeros(3)
        self._limiter = DisplacementLimiter(self.center_pose, DISPLACEMENT_THRESHOLDS)

    # Simulated TIME before the injected slip fires -- deliberately not a step
    # count, which would land at a different point in the session whenever --rate
    # changes. It must be late enough that several excursions have already reported
    # a healthy anchor, since the slip checks compare CONSECUTIVE excursions rather
    # than absolute values, and a fault present from the first measurement offers
    # nothing to detect.
    SLIP_AFTER_SECONDS = 90.0

    # -- internals ---------------------------------------------------------
    def _slipped(self):
        return (self.fault == 'slip'
                and self._steps * self.rate > self.SLIP_AFTER_SECONDS)

    def _u(self):
        u = self._limiter.u_of(self.cur)
        if self._slipped():
            self._anchor_shift = np.array([0.004, 0.0, 0.0])
        u[:3] = u[:3] - self._anchor_shift
        return u

    def _true_wrench(self):
        u = self._u()
        f = -(self.K[:3] * u[:3] + self.cubic * u[:3] ** 3
              + self.rot_force_coupling * u[3:])
        t = -(self.Krot[3:] * u[3:])
        w = np.concatenate([f, t])
        # rate-independent backlash: the load lags the displacement
        self._hyst = self._hyst + self.hyst_alpha * (w - self._hyst)
        out = self._hyst - self._zero_offset
        # A real F/T sensor never reads exactly zero on all six axes -- that is the
        # signature check_ft_alive() looks for. Without a noise floor the simulator
        # is indistinguishable from a dead topic and the guard fires on a healthy run.
        return out + self.rng.normal(scale=self.noise, size=6)

    # -- Plant API ---------------------------------------------------------
    def end_effector(self):
        return self.cur.copy()

    def eef_axis_angle(self):
        p = self.cur
        return np.concatenate([p[:3],
                               transformations.axis_angle_from_quaternion(p[3:])])

    def set_target(self, pose):
        self._queue.append(np.asarray(pose, float).copy())
        if len(self._queue) > self.lag:
            self.cur = self._queue.pop(0)
        self._steps += 1

    def wrench_raw(self):
        if self.fault == 'dead_ft':
            return np.zeros(6)
        if self.fault == 'ft_dies_mid' and self._steps > 300:
            return np.zeros(6)
        return self._true_wrench()

    def wrench_base(self):
        return self.wrench_raw()

    def bend(self):
        u = self._u()
        # Crude but monotone in deflection. Every channel mixes translation and
        # rotation so none goes constant on a pure-axis excursion (which would trip
        # prepare_deflection_dataset.py's dead-channel warning). The last bend
        # channel gets the wide range the real sensor3_bend shows
        # (train_pose_mlp.py:22-24), which is what makes robust normalisation the
        # default there.
        rot = float(np.linalg.norm(u[3:]))
        s = []
        for i in range(self.num_sensors):
            s.append(u[i % 3] * 1000.0 * (1.0 + 0.3 * i) + rot * (2.0 + i))
            scale = 300.0 if i == self.num_sensors - 1 else 3.0
            s.append(rot * scale + u[(i + 1) % 3] * 400.0)
        return np.asarray(s, dtype=float)

    def claw_cmd(self):
        return 0.0

    def claw_meas(self):
        return 0.5 if self._slipped() else 0.12

    def claw_open(self):
        pass

    def claw_close(self):
        pass

    def zero_ft(self):
        self._zero_offset = np.zeros(6)
        self._zero_offset = self._true_wrench()

    def park(self, joints, target_time=5.0):
        self.cur = self.center_pose.copy()


# ==================== EXCURSION RUNNER ====================

class ExcursionRunner:
    """One excursion: establish zero -> advance -> trip -> hold -> retreat -> relax."""

    def __init__(self, plant, clock, metric, args, recorder):
        self.plant = plant
        self.clock = clock
        self.metric = metric
        self.args = args
        self.rec = recorder
        self.limiter = None
        self.sensors0 = None
        self.max_cmd_trans = 0.0
        self.max_cmd_rot = 0.0
        self.lam_seen_max = 0.0

    # -- sampling ----------------------------------------------------------
    def _sample(self, phase, s_cmd, commanded_pose):
        w_raw = self.plant.wrench_raw()
        w_base = self.plant.wrench_base()
        eef = self.plant.eef_axis_angle()
        bend = self.plant.bend()
        pose = self.plant.end_effector()
        u = self.limiter.u_of(pose)

        lam = self.metric.lam(w_raw, u, bend, self.sensors0)
        lam_f = self.metric.lam_force(w_raw)
        dtr = float(np.linalg.norm(u[:3]))
        drot = float(np.linalg.norm(u[3:]))

        track = 0.0
        if commanded_pose is not None:
            track = float(np.linalg.norm(np.asarray(commanded_pose)[:3] - pose[:3]))

        self.rec.add(self.clock.now(), w_base, eef, bend, w_raw, lam, lam_f,
                     dtr, drot, phase, s_cmd,
                     self.plant.claw_cmd(), self.plant.claw_meas(), track)
        self.lam_seen_max = max(self.lam_seen_max, lam)
        return lam, lam_f, track, w_raw, u

    def _command(self, u, phase, s_cmd):
        """Command a pose, then HOLD it until the wrench stops changing.

        The wrench trails the command by both the controller's own lag and the
        material's relaxation, so the reading on the tick right after a step
        systematically UNDERSTATES the load -- measured in simulation at roughly 40%
        of the settled value near the boundary. Deciding the trip on that unsettled
        number is what lets lam sail past the cap no matter how small the step is,
        and no amount of step-size tuning fixes it. Settling to convergence does.
        """
        pose = self.limiter.pose_from_u(u)
        self.plant.set_target(pose)
        self.max_cmd_trans = max(self.max_cmd_trans, float(np.linalg.norm(u[:3])))
        self.max_cmd_rot = max(self.max_cmd_rot, float(np.linalg.norm(u[3:])))
        self.clock.sleep(self.args.rate)
        out = self._sample(phase, s_cmd, pose)
        hist = [out[0]]

        # Hold for at least three samples ALWAYS. How long to settle must not be
        # decided from the first reading: that reading is the one that understates
        # the load, so a "this is low, move on" decision made from it is exactly
        # wrong near the boundary. Three samples is also the minimum that makes the
        # extrapolation below defined.
        # Settle for a TIME, not a sample count: the wrench's settling time is a
        # physical property of the wrist, so a fixed sample budget would mean 1.0s
        # of settling at 10Hz but only 0.1s at 100Hz -- silently under-settling and
        # letting lam sail past the cap exactly when the rate is raised.
        budget = (self.args.settle_seconds_near if hist[0] > self.args.lam_slow
                  else self.args.settle_seconds_far)
        k = max(1, int(round(self.args.extrapolate_dt / self.args.rate)))
        n_min = max(3, int(self.args.settle_min), 2 * k + 1)
        n_max = max(n_min, int(round(budget / self.args.rate)))

        n = 1
        while n < n_max:
            self.plant.set_target(pose)          # hold, which also drains the lag
            self.clock.sleep(self.args.rate)
            out = self._sample(phase, s_cmd, pose)
            hist.append(out[0])
            n += 1
            if n < n_min:
                continue
            # Convergence expressed per SECOND, so the criterion does not get
            # 10x easier merely because the loop runs 10x faster.
            if abs(hist[-1] - hist[-2]) / self.args.rate < self.args.settle_rate_tol:
                break
            # only keep paying for samples while the PREDICTED load is near the cap
            if self._extrapolate(hist) <= self.args.lam_slow:
                break

        lam_eff = self._extrapolate(hist)
        return (lam_eff,) + tuple(out[1:])

    # -- zero reference ----------------------------------------------------
    def establish_zero(self, full_rezero):
        """Return the verified zero-deflection pose, or None on failure.

        Ordering is load-bearing and differs from collect_force_dataset.py:728-735,
        which zeroes the F/T sensor AFTER closing the gripper. Against a bolted post
        that zeroes out the preload and blinds the force cap.
        """
        if full_rezero:
            self.plant.claw_open()
            self.clock.sleep(1.0)
            self.plant.zero_ft()               # claw OPEN, wrist unloaded
            self.clock.sleep(0.2)
            self.plant.claw_close()            # now grip the post
            self.clock.sleep(1.0)

        pose = self.plant.end_effector()
        self.limiter = DisplacementLimiter(pose, DISPLACEMENT_THRESHOLDS,
                                           margin=self.args.backstop_margin)

        # --- wrench-nulling servo -------------------------------------------
        # Secant-estimated Newton step, NOT a fixed gain. Two reasons this matters:
        #   * the sign of dw/du depends on the F/T sensor's frame convention, which
        #     must be measured rather than assumed -- a fixed-sign gain that guesses
        #     wrong diverges, walks the origin, and then trips the slip check;
        #   * a fixed gain is only stable while gain*stiffness < 1, and the
        #     rigid-anchor stiffness is the one thing we cannot predict.
        # Estimating dw/du from our own steps gets both right for free.
        def _null_axis(idx, tol, probe, k_init, is_rot):
            """Null wrench components `idx` by moving u[idx]. Returns iters used."""
            nonlocal u, travel, travel_rot
            k_hat = np.full(len(idx), float(k_init))     # dw/du, signed
            prev_u = prev_w = None
            it = 0
            while it < self.args.null_max_iters:
                w_now = self.plant.wrench_raw()[idx]
                if np.linalg.norm(w_now) <= tol:
                    return it, True
                if prev_w is not None:
                    du_prev = u[idx] - prev_u
                    dw = w_now - prev_w
                    for j in range(len(idx)):
                        if abs(du_prev[j]) > 1e-9:
                            est = dw[j] / du_prev[j]
                            if abs(est) > 1e-6:
                                k_hat[j] = est
                prev_u, prev_w = u[idx].copy(), w_now.copy()

                if prev_w is not None and it == 0:
                    du = np.full(len(idx), probe) * np.sign(
                        np.where(w_now == 0, 1.0, w_now))   # probe to learn the sign
                else:
                    du = -self.args.null_damping * w_now / k_hat
                    du = np.clip(du, -probe * 20, probe * 20)

                step = float(np.linalg.norm(du))
                if is_rot:
                    if travel_rot + step > self.args.null_max_rot:
                        return it, False
                    travel_rot += step
                else:
                    if travel + step > self.args.null_max_travel:
                        return it, False
                    travel += step

                u[idx] = u[idx] + du
                self.plant.set_target(self.limiter.pose_from_u(u))
                self.clock.sleep(self.args.rate)
                it += 1
            return it, False

        start_pose = self.plant.end_effector()
        u = np.zeros(6)
        travel = 0.0
        travel_rot = 0.0
        it_t, ok_t = _null_axis([0, 1, 2], self.args.preload_force_tol,
                                self.args.null_probe, 1.0 / max(self.args.null_kf, 1e-9),
                                is_rot=False)
        it_r, ok_r = _null_axis([3, 4, 5], self.args.preload_torque_tol,
                                np.deg2rad(0.05), 1.0 / max(self.args.null_kt, 1e-9),
                                is_rot=True)
        iters = it_t + it_r
        suspect = not (ok_t and ok_r)

        if suspect:
            # Revert rather than accept a half-nulled pose: keeping wherever the
            # servo wandered to would make it the next segment's label origin AND
            # drift the zero-load fixed point, which slip check (b) would then
            # report as a moved anchor.
            w_now = self.plant.wrench_raw()
            log_warn(f"wrench null did not converge ({iters} iters, "
                     f"|F|={np.linalg.norm(w_now[:3]):.2f}N, "
                     f"|T|={np.linalg.norm(w_now[3:]):.3f}Nm) -- reverting to the "
                     "pre-null pose and marking this excursion suspect")
            self.plant.set_target(start_pose)
            for _ in range(3):
                self.clock.sleep(self.args.rate)
                self.plant.set_target(start_pose)

        # The nulled pose becomes the origin. Deliberately NOT re-zeroing the F/T
        # sensor here: that would erase the preload evidence the slip checks need.
        zero_pose = self.plant.end_effector()
        self.limiter = DisplacementLimiter(zero_pose, DISPLACEMENT_THRESHOLDS,
                                           margin=self.args.backstop_margin)
        self.sensors0 = self.plant.bend()
        return zero_pose, iters, travel, travel_rot, suspect

    # -- the excursion ------------------------------------------------------
    def run(self, d_hat, s_target, s_backstop, scale):
        """Advance along d_hat until lam trips or the cage stops us, then retreat.

        Returns a dict of per-excursion outcomes.
        """
        self.metric.reset_trip()
        self.lam_seen_max = 0.0
        lam_force_max = 0.0
        d_scaled = d_hat * scale

        # s is dimensionless (s = 1 is roughly the cage), so a step in s means
        # different physical distances along different directions. Convert the
        # metre/radian step limits into an s-step for THIS direction, taking
        # whichever of translation or rotation is the binding one.
        self.s_step_max = self._s_step_max(d_scaled)

        # row 0: the verified zero.
        self._sample(PHASE_NULL, 0.0, None)

        s = 0.0
        stop_reason = 'target'
        lam_hist = []
        # Start at the FLOOR and ramp up, rather than starting at step_max. On a stiff
        # axis a full 1mm step can cross the cap outright, and the predictive stop has
        # no history to work from on step 1. Ramping costs a few samples and removes
        # the one overshoot mode the predictor structurally cannot catch.
        step = self.s_step_max * self.args.step_min_frac

        while s < s_target:
            if self._interrupted():
                stop_reason = 'interrupt'
                break
            if self.plant.protective_stopped():
                stop_reason = 'protective_stop'
                break

            # --- predictive stop: shrink the step so the PREDICTED lam lands at
            # ~0.98 rather than overshooting. Reacting after the fact cannot handle
            # controller lag; this is what makes the loop rate a non-issue.
            if len(lam_hist) >= 2:
                k = -3 if len(lam_hist) >= 3 else -2
                ds = lam_hist[-1][0] - lam_hist[k][0]
                dl = lam_hist[-1][1] - lam_hist[k][1]
                slope = (dl / ds) if abs(ds) > 1e-12 else 0.0
                if slope > 1e-9:
                    # Aiming at predict_target bounds where the step LANDS; capping
                    # the lam jump bounds how far past it can throw us when the wrist
                    # is stiff enough that the whole usable range is about one step
                    # wide -- the regime a rigid anchor may well be in. Together they
                    # make the overshoot bound independent of stiffness.
                    room = (self.args.predict_target - lam_hist[-1][1]) / slope
                    step = min(step, self.args.max_dlam_step / slope, max(room, 0.0))
                # Absolute progress floor, in METRES/RADIANS -- deliberately not a
                # fraction of s_step_max. s_step_max is max_speed*rate, so a fractional
                # floor shrinks with the loop rate: at 100Hz it became 0.004mm, and
                # since the predictor pins the step at the floor once lam passes
                # predict_target, the last ~1mm took ~300 commands and the arm looked
                # like it had stalled. The smallest useful displacement increment is a
                # physical quantity, independent of how fast the loop runs.
                step = max(step, self._s_step_floor(d_scaled))

            s_next = min(s + step, s_target, s_backstop)
            u = s_next * d_scaled
            lam, lam_f, track, w, u_meas = self._command(u, PHASE_LOAD, s_next)
            lam_force_max = max(lam_force_max, lam_f)
            lam_hist.append((s_next, lam))
            s = s_next

            if self.metric.hard_abort(w):
                stop_reason = 'hard_abort'
                break
            if track > self.args.max_track_err:
                stop_reason = 'track_err'
                break
            if self.metric.update_trip(lam):
                stop_reason = 'force'
                break
            if s_next >= s_backstop - 1e-12:
                stop_reason = 'backstop'
                break

            # --- adaptive step: 1mm below lam_slow, tapering to step_min_frac at
            # the boundary.
            frac = (1.0 - lam) / max(1.0 - self.args.lam_slow, 1e-6)
            target_step = self.s_step_max * float(
                np.clip(frac, self.args.step_min_frac, 1.0))
            step = min(target_step, step * self.args.step_growth)


        binding = self.metric.binding_term(self.plant.wrench_raw(),
                                           self.limiter.u_of(self.plant.end_effector()),
                                           self.plant.bend(), self.sensors0)
        if stop_reason == 'force':
            stop_reason = binding      # 'force' | 'torque' | 'defl_trans' | ...

        s_reached = s

        # --- hold at the boundary (creep at constant displacement)
        if self.args.hold_at_boundary > 0 and stop_reason not in (
                'interrupt', 'protective_stop', 'hard_abort'):
            n = max(int(self.args.hold_at_boundary / self.args.rate), 1)
            for _ in range(n):
                self.clock.sleep(self.args.rate)
                self._sample(PHASE_HOLD, s_reached,
                             self.limiter.pose_from_u(s_reached * d_scaled))

        # --- retreat. Absolute commands from a fixed origin, so s=0 really is the
        # origin -- this is what fixes the 0.23-1.64mm under-travel measured in the
        # existing data, where each return target was rebuilt from end_effector().
        if stop_reason != 'protective_stop':
            self._retreat(d_scaled, s_reached)

        # --- relax tail: creep/relaxation data, and the input to slip check (a).
        # Keep COMMANDING the origin rather than merely sampling: the controller lags
        # the command, so without this the last retreat command is still in flight and
        # the excursion would end a few tens of microns off u=0 -- exactly the
        # under-travel that corrupts the next segment's label origin.
        origin = self.limiter.pose_from_u(np.zeros(6))
        n = max(int(self.args.relax_seconds / self.args.rate), 1)
        for _ in range(n):
            self.plant.set_target(origin)
            self.clock.sleep(self.args.rate)
            self._sample(PHASE_RELAX, 0.0, origin)

        return {
            'stop_reason': stop_reason,
            's_reached': s_reached,
            'censored': stop_reason == 'backstop',
            'lam_max': self.lam_seen_max,
            'lam_force_max': lam_force_max,
            'residual_wrench': self.plant.wrench_raw(),
            'n_rows': len(self.rec.rows),
        }

    # -- waypoint hop -------------------------------------------------------
    def hop(self, u_from, u_to):
        """Move straight from u_from to u_to at max_speed, recording every tick.

        This is the Phase B primitive and it deliberately does NOT settle per step.
        Settling exists to FIND the force boundary; once the envelope is known there
        is nothing to search for, and paying for it again is what made every
        excursion spend ~68% of its samples creeping through the last 1.2mm.

        lam is still evaluated every tick, but as an ABORT rather than a stopping
        criterion. Per-tick checking is not optional: the geometric cage is convex in
        u so interpolation cannot leave it, but the force-bounded region has no such
        guarantee -- a straight line between two admissible waypoints can pass
        outside it.
        """
        u_from = np.asarray(u_from, dtype=float)
        u_to = np.asarray(u_to, dtype=float)
        if not np.all(np.isfinite(u_to)):
            log_warn(f"non-finite hop target {u_to} -- holding position")
            u_to = u_from.copy()
        delta = u_to - u_from

        # tick count comes from the SPEED limit, same bound the probe phase uses
        n_lin = np.linalg.norm(delta[:3]) / max(self.args.max_speed * self.args.rate, 1e-12)
        n_rot = np.linalg.norm(delta[3:]) / max(self.args.max_rot_speed * self.args.rate, 1e-12)
        n = max(1, int(np.ceil(max(n_lin, n_rot))))

        stop = 'target'
        u = u_from.copy()
        lam_force_max = 0.0
        lam_hist = []
        for k in range(1, n + 1):
            if self._interrupted():
                stop = 'interrupt'
                break
            if self.plant.protective_stopped():
                stop = 'protective_stop'
                break
            u = u_from + delta * (float(k) / n)
            pose = self.limiter.pose_from_u(u)
            self.plant.set_target(pose)
            self.max_cmd_trans = max(self.max_cmd_trans, float(np.linalg.norm(u[:3])))
            self.max_cmd_rot = max(self.max_cmd_rot, float(np.linalg.norm(u[3:])))
            self.clock.sleep(self.args.rate)
            lam, lam_f, track, w, _ = self._sample(PHASE_TRANSIT, float(k) / n, pose)
            lam_force_max = max(lam_force_max, lam_f)
            lam_hist.append(lam)

            # The wrench lags the commanded pose, and a hop never holds still, so
            # there is no settled reading to extrapolate from the way the probe phase
            # does. Predict forward by the lag instead: abort on where lam is HEADING,
            # not where it currently reads. Without this a stiff wrist is already well
            # past the cap by the time the reading crosses it.
            lam_pred = lam
            if len(lam_hist) >= 2:
                dlam_dt = (lam_hist[-1] - lam_hist[-2]) / self.args.rate
                if dlam_dt > 0:
                    lam_pred = lam + dlam_dt * self.args.hop_lookahead

            if self.metric.hard_abort(w):
                stop = 'hard_abort'
                break
            if track > self.args.max_track_err:
                stop = 'track_err'
                break
            if lam_pred >= self.args.hop_lam_abort:
                stop = self.metric.binding_term(
                    w, self.limiter.u_of(self.plant.end_effector()),
                    self.plant.bend(), self.sensors0)
                break

        # On a load abort, shed load immediately rather than merely stopping: the next
        # waypoint is sampled from a corrected envelope, but we should not sit at the
        # boundary while that happens.
        if stop not in ('target', 'interrupt', 'protective_stop'):
            u_back = u * self.args.hop_abort_retreat
            pose = self.limiter.pose_from_u(u_back)
            self.plant.set_target(pose)
            self.clock.sleep(self.args.rate)
            self._sample(PHASE_UNLOAD, 0.0, pose)
            u = u_back

        # Dwell: the one place settling is still paid for, and the samples that are
        # immune to any sensor lag during transit.
        pose = self.limiter.pose_from_u(u)
        for _ in range(max(int(round(self.args.dwell / self.args.rate)), 1)):
            self.plant.set_target(pose)
            self.clock.sleep(self.args.rate)
            self._sample(PHASE_DWELL, 1.0, pose)

        return {'stop': stop, 'u': u, 'lam_max': self.lam_seen_max,
                'lam_force_max': lam_force_max}

    def _retreat(self, d_scaled, s_from, phase=PHASE_UNLOAD, deadline=None):
        """Step back to u=0, recording. `deadline` bounds the wall time spent: an
        unload that cannot finish must not hang the one path every abort goes
        through."""
        step = self._s_step_max(d_scaled) * self.args.retreat_step_scale
        s = s_from
        while s > 0.0:
            if deadline is not None and self.clock.now() > deadline:
                log_warn(f"unload deadline reached with s={s:.4f} still to go -- "
                         "stopping here, the chain may still be loaded")
                return False
            s = max(0.0, s - step)
            self.plant.set_target(self.limiter.pose_from_u(s * d_scaled))
            self.clock.sleep(self.args.rate)
            self._sample(phase, s, self.limiter.pose_from_u(s * d_scaled))
        return True

    def _extrapolate(self, hist):
        """Asymptote of a monotone, decelerating approach; falls back to the last
        sample whenever the sequence does not look like one."""
        # Stride by a fixed time, not by adjacent samples: at a high rate the
        # sample-to-sample change is tiny and the geometric ratio degenerates
        # toward 1, which makes the extrapolated asymptote explode.
        k = max(1, int(round(self.args.extrapolate_dt / self.args.rate)))
        if len(hist) < 2 * k + 1:
            return hist[-1]
        d1 = hist[-1] - hist[-1 - k]
        d2 = hist[-1 - k] - hist[-1 - 2 * k]
        if abs(d2) < 1e-12:
            return hist[-1]
        r = d1 / d2
        if not (0.0 < r < 1.0):        # not converging geometrically -- don't guess
            return hist[-1]
        lam_inf = hist[-1] + d1 * r / (1.0 - r)
        # never trust the extrapolation beyond a bounded multiple of what was measured
        return float(min(lam_inf, hist[-1] * self.args.extrapolate_cap))

    def _s_step_floor(self, d_scaled):
        """Smallest step in s that still moves a physically meaningful amount."""
        tr = float(np.linalg.norm(d_scaled[:3]))
        ro = float(np.linalg.norm(d_scaled[3:]))
        cand = []
        if tr > 1e-12:
            cand.append(self.args.min_step / tr)
        if ro > 1e-12:
            cand.append(self.args.min_rot_step / ro)
        return min(cand) if cand else 0.0

    def _s_step_max(self, d_scaled):
        """Largest step in s along this direction.

        Bounded by SPEED (max_speed * rate) as well as by an absolute per-command
        distance. Speed is the physically meaningful quantity: bounding only the
        per-command distance means the arm moves N times faster whenever the loop
        rate is raised N times, which silently turns a rate change into a speed
        change -- dangerous against a rigid anchor.
        """
        tr = float(np.linalg.norm(d_scaled[:3]))
        ro = float(np.linalg.norm(d_scaled[3:]))
        d_lin = min(self.args.step_max, self.args.max_speed * self.args.rate)
        d_rot = min(self.args.rot_step_max, self.args.max_rot_speed * self.args.rate)
        cand = []
        if tr > 1e-12:
            cand.append(d_lin / tr)
        if ro > 1e-12:
            cand.append(d_rot / ro)
        return min(cand) if cand else d_lin

    def _interrupted(self):
        return bool(getattr(self.args, '_interrupt_flag', [False])[0])


# ==================== SESSION ====================

class Session:
    def __init__(self, args, plant, clock):
        self.args = args
        self.plant = plant
        self.clock = clock
        self.rng = np.random.default_rng(args.seed)

        sensor_caps = None
        if args.sensor_defl_cap is not None:
            sensor_caps = np.full(2 * args.num_sensors, float(args.sensor_defl_cap))

        self.metric = LoadMetric(args.force_cap, args.torque_cap,
                                 args.defl_trans_cap, args.defl_rot_cap,
                                 args.hard_abort_ft, sensor_caps, args.trip_samples)
        self.envelope = RadialEnvelope(args.kernel_sigma, args.envelope_halflife,
                                       args.prior_radius)
        if args.load_envelope:
            self.envelope.load(args.load_envelope, t_now=clock.now())
            log_info(f"loaded envelope with {len(self.envelope.dirs)} observations")
        self.sampler = DirectionSampler(self.rng, n_seed=args.seed_directions,
                                        axis_frac=args.axis_frac)
        self.slip = SlipMonitor(args.slip_pose_tol, args.slip_rot_tol,
                                args.slip_wrench_tol, args.slip_wrench_abs,
                                args.claw_gap_tol)
        self.recorder = Recorder(args.num_sensors)
        self.out_dir = None
        self.excursion_log = []
        self.dead_backstops = 0
        self.commanded_poses = []       # for --self_test cage assertions

    # -- pre-flight ---------------------------------------------------------
    def check_ft_alive(self, samples=20, settle=0.05):
        """copied from collect_force_dataset.py:1687-1714.

        get_wrench() returns np.zeros(6) when no message has ever arrived, so a dead
        topic is indistinguishable from "no load". force_dataset_20260812_022007 is
        what that looks like in practice: 1596 rows, wrench identically zero, 247mm
        of travel.
        """
        readings = []
        for _ in range(samples):
            readings.append(np.asarray(self.plant.wrench_raw(), dtype=float))
            self.clock.sleep(settle)
        readings = np.asarray(readings)
        if np.all(readings == 0.0):
            log_err(f"F/T sensor reads EXACTLY zero on all 6 axes across {samples} "
                    "samples -- the wrench topic is not delivering. The force limit "
                    "cannot fire. Check `rostopic hz /wrench` and the ft_filter node. "
                    "Refusing to run (--skip_ft_check overrides).")
            return False
        if np.max(np.abs(readings - readings.mean(axis=0))) == 0.0:
            log_warn(f"F/T readings non-zero but perfectly constant "
                     f"({np.round(readings[0], 4)}) -- the topic may be stale.")
        log_info(f"F/T sensor alive (latest {np.round(readings[-1], 3)})")
        return True

    def check_sensor_zero(self):
        """Refuse to run if the bend sensors are not near zero while unloaded.

        Catches the gross case of the UART bridge having been started while already
        gripping the post -- which would silently bias every label in the session.
        """
        b = np.asarray(self.plant.bend(), dtype=float)
        if not np.all(np.isfinite(b)):
            log_warn("bend sensors are NaN -- is uart_bridge_node.py running?")
            return not self.args.require_sensor_zero_check
        worst = float(np.max(np.abs(b)))
        if worst > self.args.sensor_zero_tol:
            log_err(f"bend sensor baseline is {worst:.2f} deg (> "
                    f"{self.args.sensor_zero_tol}) with the wrist supposedly "
                    "unloaded. The UART bridge was probably started while the claw "
                    "was already gripping. Restart it with the claw OPEN.")
            return False
        log_info(f"bend sensor baseline ok (max |angle| = {worst:.2f} deg)")
        return True

    # -- output -------------------------------------------------------------
    def make_out_dir(self):
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "force_deformation_data")
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = self.args.folder or f"deflection_forcebound_{stamp}"
        d = os.path.abspath(os.path.join(base, folder))
        os.makedirs(d, exist_ok=True)
        self.out_dir = d
        self.stamp = stamp
        return d

    def write_meta(self):
        os.makedirs(self.out_dir, exist_ok=True)

        def _plain(v):
            # argparse defaults and computed args can be numpy scalars/arrays, which
            # yaml.safe_dump refuses -- coerce everything to builtins.
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, np.generic):
                return v.item()
            return v

        meta = {k: _plain(v) for k, v in vars(self.args).items()
                if not k.startswith('_')}
        meta['n_excursions_completed'] = len(self.excursion_log)
        with open(os.path.join(self.out_dir, 'session_meta.yaml'), 'w') as f:
            yaml.safe_dump(meta, f, default_flow_style=False)

    def write_excursion_log(self):
        """Rewrite the whole log. Called after every excursion, not just at the end:
        a session that crashes or is killed must not take the log for the excursions
        it already completed with it."""
        if not self.excursion_log:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        keys = sorted({k for r in self.excursion_log for k in r})
        p = os.path.join(self.out_dir, 'excursion_log.csv')
        tmp = p + '.tmp'
        with open(tmp, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in self.excursion_log:
                w.writerow(r)
        os.replace(tmp, p)          # atomic, so a kill mid-write cannot truncate it
        return p

    # -- the session --------------------------------------------------------
    def run(self):
        self.make_out_dir()
        self.t_start = self.clock.now()
        log_info(f"output -> {self.out_dir}")

        if not self.args.skip_ft_check:
            if not self.check_ft_alive():
                return 1
        else:
            if not self.args.i_understand_no_force_limit:
                log_err("--skip_ft_check requires --i_understand_no_force_limit")
                return 1
            self.args.backstop_margin = min(self.args.backstop_margin, 0.5)
            log_warn("F/T check skipped -- backstop margin forced to 0.5. The "
                     "deflection term in lam is now the only active bound.")

        if not self.check_sensor_zero():
            if self.args.require_sensor_zero_check:
                return 1

        self.plant.activate_compliance()

        runner = ExcursionRunner(self.plant, self.clock, self.metric,
                                 self.args, self.recorder)
        n_blocks = (int(np.ceil(self.args.hop_waypoints /
                                max(self.args.waypoints_per_file, 1)))
                    if self.args.hop_waypoints > 0 else 0)
        total = self.args.probe_excursions + n_blocks
        aborted = False

        # ---- Phase A: probe. Creep-and-settle, because this IS the boundary search.
        for i in range(1, self.args.probe_excursions + 1):
            if self._stop_now() or self._probe_once(runner, i, total):
                aborted = True
                break

        # ---- Phase B: hop. The envelope is known now, so move DIRECTLY between
        # waypoints instead of re-discovering the boundary on every excursion.
        if not aborted:
            for b in range(n_blocks):
                if self._stop_now() or self._hop_block(runner, b, total):
                    break

        self.envelope.save(os.path.join(self.out_dir, 'envelope.npz'))
        self.write_excursion_log()
        self.write_meta()
        log_info(f"session complete: {len(self.excursion_log)} excursions -> "
                 f"{self.out_dir}")
        return 0

    def _stop_now(self):
        if self._interrupted():
            log_warn("interrupted -- ending session")
            return True
        if self.clock.now() - self.t_start > self.args.max_session_seconds:
            log_warn("session time budget exhausted")
            return True
        return False

    def _probe_once(self, runner, i, total):
        """One creep-and-settle excursion. Returns True if the session must abort."""
        full_rezero = (i == 1) or (self.args.rezero_every > 0 and
                                   (i - 1) % self.args.rezero_every == 0)
        self.recorder.reset()
        zero_pose, ni, nt, ntr, nsus = runner.establish_zero(full_rezero)

        d_hat = self.sampler.next()
        scale = np.concatenate([runner.limiter.trans_extent,
                                np.full(3, runner.limiter.rot_extent)])
        d_scaled = d_hat * scale
        s_backstop = runner.limiter.max_scale(d_scaled)

        res = runner.run(d_hat, s_backstop, s_backstop, scale)
        self.commanded_poses.append((runner.max_cmd_trans, runner.max_cmd_rot))

        if res['stop_reason'] in ('force', 'torque', 'defl_trans', 'defl_rot',
                                  'sensor'):
            self.envelope.add(d_hat, res['s_reached'], False, self.clock.now())
        elif res['stop_reason'] == 'backstop':
            self.envelope.add(d_hat, res['s_reached'], True, self.clock.now())

        extra = {'s_backstop': s_backstop, 'censored': int(res['censored'])}
        for j in range(6):
            extra[f'd{j}'] = float(d_hat[j])
        return self._finalize(runner, zero_pose, (ni, nt, ntr, nsus), res,
                              'probe', i, total, extra)

    def _hop_block(self, runner, b, total):
        """One CSV worth of waypoint hops. Returns True if the session must abort."""
        i = self.args.probe_excursions + b + 1
        full_rezero = (self.args.rezero_every > 0 and
                       (i - 1) % self.args.rezero_every == 0)
        self.recorder.reset()
        zero_pose, ni, nt, ntr, nsus = runner.establish_zero(full_rezero)
        runner.lam_seen_max = 0.0
        runner.max_cmd_trans = runner.max_cmd_rot = 0.0

        # Row 0 of the file MUST be the verified zero pose -- the label is built
        # relative to it (prepare_deflection_dataset.py:122-125).
        runner._sample(PHASE_NULL, 0.0, None)

        scale = np.concatenate([runner.limiter.trans_extent,
                                np.full(3, runner.limiter.rot_extent)])
        inv_scale = 1.0 / np.maximum(scale, 1e-12)
        u_cur = np.zeros(6)
        done = b * self.args.waypoints_per_file
        n_wp = max(min(self.args.waypoints_per_file,
                       self.args.hop_waypoints - done), 0)

        aborts = nonradial = 0
        lam_force_max = 0.0
        stop_reason = 'target'

        for _ in range(n_wp):
            d_hat = self.sampler.next()
            d_scaled = d_hat * scale
            s_backstop = runner.limiter.max_scale(d_scaled)
            r_hat = self.envelope.query(d_hat, self.clock.now(), s_backstop)
            rho = (1.0 if self.rng.random() < self.args.boundary_frac
                   else self.rng.random() ** (1.0 / 3.0))
            s_target = rho * min(r_hat * self.args.envelope_shrink, s_backstop)

            # Starting from the CURRENT pose rather than from zero is what makes the
            # path non-radial, which is where the hysteresis information lives.
            if float(np.linalg.norm(u_cur)) > 1e-9:
                nonradial += 1

            res = runner.hop(u_cur, s_target * d_scaled)
            lam_force_max = max(lam_force_max, res['lam_force_max'])
            u_cur = res['u']

            if res['stop'] in ('force', 'torque', 'defl_trans', 'defl_rot', 'sensor'):
                aborts += 1
                # the point actually reached is a fresh uncensored boundary sample,
                # which is how the envelope self-corrects for creep
                v = u_cur * inv_scale
                nrm = float(np.linalg.norm(v))
                if nrm > 1e-9:
                    self.envelope.add(v / nrm, nrm, False, self.clock.now())
            elif res['stop'] in ('interrupt', 'protective_stop', 'hard_abort',
                                 'track_err'):
                stop_reason = res['stop']
                break

        if stop_reason != 'protective_stop':
            runner.hop(u_cur, np.zeros(6))      # reopen the next file at zero

        self.commanded_poses.append((runner.max_cmd_trans, runner.max_cmd_rot))
        res = {'stop_reason': stop_reason,
               's_reached': float(np.linalg.norm(u_cur * inv_scale)),
               'lam_max': runner.lam_seen_max, 'lam_force_max': lam_force_max,
               'residual_wrench': self.plant.wrench_raw(),
               'n_rows': len(self.recorder.rows), 'censored': False}
        return self._finalize(runner, zero_pose, (ni, nt, ntr, nsus), res,
                              'hop', i, total,
                              {'waypoints': n_wp, 'hop_aborts': aborts,
                               'nonradial': nonradial})

    def _finalize(self, runner, zero_pose, null_info, res, kind, i, total, extra):
        """Slip checks, CSV, log row, guards. Returns True if the session must abort."""
        ni, nt, ntr, nsus = null_info
        claw_gap = None
        cc, cm = self.plant.claw_cmd(), self.plant.claw_meas()
        if cc is not None and cm is not None:
            claw_gap = cm - cc
        verdicts, fatal = self.slip.check(zero_pose, res['residual_wrench'],
                                          claw_gap, runner.limiter,
                                          null_travel=nt, null_travel_rot=ntr)

        name = (f"deformation_data_{self.args.direction_label}"
                f"_rep{i}_{self.stamp}.csv")
        path = self.recorder.write(os.path.join(self.out_dir, name))

        row = {'rep': i, 'kind': kind, 'stop_reason': res['stop_reason'],
               's_reached': res['s_reached'],
               'lam_max': res['lam_max'], 'lam_force_max': res['lam_force_max'],
               'null_iters': ni, 'null_travel': nt, 'null_travel_rot': ntr,
               'null_suspect': int(nsus),
               'n_rows': res['n_rows'], 'csv': os.path.basename(path)}
        row.update(extra)
        row.update({k: (int(v) if isinstance(v, bool) else v)
                    for k, v in verdicts.items()})
        self.excursion_log.append(row)
        self.write_excursion_log()

        tail = (f"wp={extra['waypoints']:3d} aborts={extra['hop_aborts']:2d}"
                if kind == 'hop' else f"s={res['s_reached']:.3f}")
        log_info(f"[{i}/{total}] {kind:5s} stop={res['stop_reason']:12s} {tail} "
                 f"lam={res['lam_max']:.2f} lam_F={res['lam_force_max']:.2f} "
                 f"rows={res['n_rows']}")

        # Runaway rule uses the FORCE-ONLY fraction: the combined lam would reach 1
        # via the deflection term and mask exactly this failure.
        if res['stop_reason'] in ('backstop', 'defl_trans', 'defl_rot') and \
                res['lam_force_max'] < self.args.dead_lam:
            self.dead_backstops += 1
            log_warn(f"unit {i} reached its limit at only "
                     f"lam_force={res['lam_force_max']:.2f} -- the anchor may not be "
                     f"holding ({self.dead_backstops}/{self.args.max_dead_backstops})")
            if self.dead_backstops >= self.args.max_dead_backstops:
                log_err("too many low-force limit stops -- aborting session")
                self.controlled_unload(runner, "dead_backstops")
                return True
        else:
            self.dead_backstops = 0

        if fatal:
            log_err(f"slip detected ({[k for k in verdicts if k.startswith('slip')]}) "
                    "-- aborting session")
            self.controlled_unload(runner, "slip")
            return True
        if res['stop_reason'] in ('hard_abort', 'protective_stop'):
            self.controlled_unload(runner, res['stop_reason'])
            return True
        return False

    def controlled_unload(self, runner, reason):
        """The single exit path. Never switch controllers or open the claw while the
        chain is loaded."""
        log_warn(f"controlled unload ({reason})")
        try:
            if reason == 'protective_stop':
                log_err("protective stop: NOT auto-retreating. The arm is braked and "
                        "commanding into it is what caused the stop. Release the "
                        "protective stop at the pendant with the claw open.")
                if self.args.auto_unlock_protective_stop:
                    self.plant.claw_open()      # break the chain FIRST
                    self.clock.sleep(1.0)
                return
            if runner.limiter is not None:
                pose = self.plant.end_effector()
                u = runner.limiter.u_of(pose)
                n = float(np.linalg.norm(u))
                if n > 1e-9:
                    runner._retreat(u / n, n, phase=PHASE_UNLOAD,
                                    deadline=self.clock.now() +
                                    self.args.max_unload_seconds)
            self.clock.sleep(0.5)
            self.plant.claw_open()
            self.clock.sleep(0.5)
            self.plant.zero_ft()
            self.plant.activate_trajectory()
        except Exception as e:
            log_err(f"controlled unload failed: {e}")

    def _interrupted(self):
        return bool(getattr(self.args, '_interrupt_flag', [False])[0])


# ==================== SELF TEST ====================

def _sim_session(args, stiffness=None, fault=None, seed=0):
    a = argparse.Namespace(**vars(args))
    if stiffness is not None:
        a.sim_stiffness = stiffness
    if fault is not None:
        a.sim_fault = fault
    a.seed = seed
    a._interrupt_flag = [False]
    clock = SimClock()
    plant = SimPlant(a, np.random.default_rng(seed))
    plant.park(None)
    s = Session(a, plant, clock)
    s.run()
    return s


def self_test(args):
    print("=" * 72)
    print("SELF TEST (no robot, no ROS master)")
    print("=" * 72)
    ok = True

    base = argparse.Namespace(**vars(args))
    base.probe_excursions = 12
    base.hop_waypoints = 24
    base.waypoints_per_file = 12
    base.relax_seconds = 0.3
    base.hold_at_boundary = 0.2
    base.folder = None
    base.skip_ft_check = False
    base.i_understand_no_force_limit = False
    base.require_sensor_zero_check = False

    # ---- 1. geometric cage regression cases -----------------------------
    print("\n1. cage regression cases (must be REJECTED):")
    center = np.array([0.4, 0.1, 0.3, 0.0, 0.0, 0.0, 1.0])
    limiter = DisplacementLimiter(center, DISPLACEMENT_THRESHOLDS)
    cases = []
    u = np.zeros(6); u[0] = 0.018; u[1] = 0.018
    cases.append(("+18mm x AND +18mm y (25.5mm XY magnitude)", u.copy(), "xy_pos45"))
    u = np.zeros(6); u[3] = np.deg2rad(10.0); u[4] = np.deg2rad(10.0)
    cases.append(("10deg roll AND 10deg pitch (~14deg total)", u.copy(), "roll_pitch_pos45"))
    u = np.zeros(6); u[2] = 0.018
    cases.append(("+18mm z (Z limit is 12mm)", u.copy(), "z"))
    for label, uu, key in cases:
        bad = limiter.violations(limiter.pose_from_u(uu), margin=1.0)
        hit = key in bad
        ok &= hit
        print(f"   [{'ok' if hit else 'FAIL'}] {label}: {key} "
              f"{'rejected' if hit else 'NOT rejected'}")

    # ---- 2 / 2b. overshoot + deflection bound across a 25x stiffness sweep
    print("\n2. force/deflection overshoot across stiffness sweep:")
    reasons_seen = set()
    for k in [0.2, 0.6, 1.5, 3.0, 5.0]:
        s = _sim_session(base, stiffness=k, fault='none', seed=1)
        lam_max = max((r['lam_max'] for r in s.excursion_log), default=0.0)
        dt_max = max((t for t, _ in s.commanded_poses), default=0.0)
        dr_max = max((r for _, r in s.commanded_poses), default=0.0)
        good = (lam_max <= base.overshoot_tol and
                dt_max <= base.defl_trans_cap * base.overshoot_tol + 1e-9 and
                dr_max <= base.defl_rot_cap * base.overshoot_tol + 1e-9)
        ok &= good
        for r in s.excursion_log:
            reasons_seen.add(r['stop_reason'])
        print(f"   [{'ok' if good else 'FAIL'}] K={k:4.1f} N/mm  lam_max={lam_max:5.2f}  "
              f"|u_t|={dt_max*1000:6.2f}mm  |u_r|={np.rad2deg(dr_max):5.2f}deg")

    force_bound = {'force', 'torque'} & reasons_seen
    defl_bound = {'defl_trans', 'defl_rot'} & reasons_seen
    both = bool(force_bound) and bool(defl_bound)
    ok &= both
    print(f"   [{'ok' if both else 'FAIL'}] both limit types bind across the sweep: "
          f"{sorted(reasons_seen)}")

    # ---- 2c. dead-wrench containment -------------------------------------
    print("\n3. dead-wrench containment (the 247mm regression):")
    dead = argparse.Namespace(**vars(base))
    dead.skip_ft_check = True            # the whole point: bypass the guard that
    dead.i_understand_no_force_limit = True   # would otherwise refuse to start
    s = _sim_session(dead, stiffness=1.0, fault='dead_ft', seed=2)
    dt_max = max((t for t, _ in s.commanded_poses), default=0.0)
    contained = dt_max <= dead.defl_trans_cap * dead.overshoot_tol + 1e-9
    ok &= contained
    print(f"   [{'ok' if contained else 'FAIL'}] max travel with dead F/T = "
          f"{dt_max*1000:.2f}mm (cap {dead.defl_trans_cap*1000*dead.overshoot_tol:.2f}mm)")

    # ---- 3. exact return --------------------------------------------------
    print("\n4. every excursion returns to u=0:")
    s = _sim_session(base, stiffness=1.0, fault='none', seed=3)
    worst = 0.0
    for f in sorted(os.listdir(s.out_dir)):
        if not f.startswith('deformation_data_'):
            continue
        import csv as _csv
        with open(os.path.join(s.out_dir, f)) as fh:
            rows = list(_csv.DictReader(fh))
        if rows:
            worst = max(worst, abs(float(rows[-1]['Defl_trans'])))
    ret_ok = worst < 1e-6
    ok &= ret_ok
    print(f"   [{'ok' if ret_ok else 'FAIL'}] worst final |u_trans| = {worst:.3e} m")

    # ---- 5. downstream contract ------------------------------------------
    print("\n5. prepare_deflection_dataset.py contract:")
    try:
        import importlib.util
        conv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "..", "..", "..",
                                 "bendlabs", "scripts",
                                 "prepare_deflection_dataset.py")
        conv_path = os.path.abspath(conv_path)
        spec = importlib.util.spec_from_file_location("pdd", conv_path)
        pdd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pdd)
        target = None
        for f in sorted(os.listdir(s.out_dir)):
            if f.startswith('deformation_data_'):
                target = os.path.join(s.out_dir, f)
                break
        df = pdd.convert_segment(target, 0.1)
        need = ['sensor0_twist', 'sensor0_bend', 'pos_x', 'pos_y', 'pos_z',
                'quat_x', 'quat_y', 'quat_z', 'quat_w', 'force_x', 'torque_z',
                'fk_disp_mm', 't', 'segment', 'rep']
        missing = [c for c in need if c not in df.columns]
        zero0 = (abs(df['pos_x'].iloc[0]) < 1e-12 and
                 abs(df['pos_y'].iloc[0]) < 1e-12 and
                 abs(df['pos_z'].iloc[0]) < 1e-12)
        good = df is not None and not missing and zero0
        ok &= good
        print(f"   [{'ok' if good else 'FAIL'}] converted {os.path.basename(target)}: "
              f"{len(df)} rows, missing={missing}, pos[0]==0: {zero0}")
        print(f"        segment={df['segment'].iloc[0]!r} rep={df['rep'].iloc[0]}")
    except Exception as e:
        ok = False
        print(f"   [FAIL] contract test raised: {e}")

    # ---- 6. fault injection ----------------------------------------------
    print("\n6. fault injection:")
    s = _sim_session(base, stiffness=1.0, fault='slip', seed=5)
    slipped = any(r.get('slip_fixed_point') or r.get('slip_claw')
                  for r in s.excursion_log)
    ok &= slipped
    print(f"   [{'ok' if slipped else 'FAIL'}] slip detected by check (b)/(c)")

    a = argparse.Namespace(**vars(base))
    a.skip_ft_check = False
    a._interrupt_flag = [False]
    plant = SimPlant(a, np.random.default_rng(6))
    plant.fault = 'dead_ft'
    sess = Session(a, plant, SimClock())
    sess.make_out_dir()
    refused = not sess.check_ft_alive()
    ok &= refused
    print(f"   [{'ok' if refused else 'FAIL'}] dead F/T refuses to start")

    # ---- 7. hops are genuinely non-radial ---------------------------------
    print("\n7. Phase B hops start from a non-zero pose (non-radial):")
    s7 = _sim_session(base, stiffness=0.55, fault='none', seed=7)
    hops = [r for r in s7.excursion_log if r.get('kind') == 'hop']
    wp = sum(int(r['waypoints']) for r in hops)
    nr = sum(int(r['nonradial']) for r in hops)
    frac = nr / max(wp, 1)
    good = frac > 0.5
    ok &= good
    print(f"   [{'ok' if good else 'FAIL'}] {nr}/{wp} waypoints started from a non-zero u "
          f"({frac*100:.0f}%). If this collapses, Phase B has silently regressed to "
          f"out-and-back and the path-dependence is gone.")

    # ---- 8. coverage is spread, not piled on the boundary ------------------
    print("\n8. deflection coverage is spread across the range:")
    import csv as _csv8
    vals = []
    for f in sorted(os.listdir(s7.out_dir)):
        if not f.startswith('deformation_data_'):
            continue
        with open(os.path.join(s7.out_dir, f)) as fh:
            for r in _csv8.DictReader(fh):
                if int(float(r['Phase'])) in (PHASE_TRANSIT, PHASE_DWELL):
                    vals.append(abs(float(r['Defl_trans'])))
    if vals:
        v = np.asarray(vals)
        top = v.max() * 0.9
        pile = float((v >= top).mean())
        reach = float(np.percentile(v, 90)) / base.defl_trans_cap
        good = pile < 0.40 and reach > 0.25
        ok &= good
        print(f"   [{'ok' if good else 'FAIL'}] {pile*100:.0f}% of samples in the top 10% "
              f"of the range (must be < 40%), and p90 reaches {reach*100:.0f}% of the "
              f"deflection cap (must be > 25%).")
        print(f"        Both bounds matter: the creep design piled ~68% of samples into "
              f"the top 8%, but an over-smoothed envelope fails the other way -- "
              f"everything tiny would otherwise pass trivially.")
        qs = np.percentile(v, [10, 25, 50, 75, 90]) * 1000
        print(f"        |u_trans| percentiles (mm): "
              f"p10={qs[0]:.2f} p25={qs[1]:.2f} p50={qs[2]:.2f} "
              f"p75={qs[3]:.2f} p90={qs[4]:.2f}")
    else:
        ok = False
        print("   [FAIL] no transit/dwell samples found")

    print("\n" + "=" * 72)
    print(f"overall: {'PASS' if ok else 'FAIL'}")
    print("=" * 72)
    return 0 if ok else 1


# ==================== CLI ====================

def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group("output")
    g.add_argument("--folder", default=None,
                   help="output folder under force_deformation_data/")
    g.add_argument("--direction_label", default="random",
                   help="the <direction> slot in deformation_data_<direction>_rep<N>_<ts>.csv")
    g.add_argument("--num_sensors", type=int, default=4)
    g.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("force / deflection bound")
    g.add_argument("--force_cap", type=float, default=DEFAULT_FORCE_CAP)
    g.add_argument("--torque_cap", type=float, default=DEFAULT_TORQUE_CAP)
    g.add_argument("--defl_trans_cap", type=float, default=DEFAULT_DEFL_TRANS_CAP,
                   help="m; FK deflection translation cap (= flange displacement)")
    g.add_argument("--defl_rot_cap", type=float, default=DEFAULT_DEFL_ROT_CAP,
                   help="rad; FK deflection rotation cap. Must sit just INSIDE the "
                        "geometric cage (10 deg = 0.1745 rad) or it can never bind and "
                        "every rotation excursion clips at the cage instead of easing "
                        "into a smooth stop -- which is what the first hardware run showed")
    g.add_argument("--sensor_defl_cap", type=float, default=None,
                   help="deg; per-channel bend-sensor cap. OFF by default -- sensor3_bend "
                        "swings ~1500 while others stay within 5, so this needs "
                        "per-channel scaling measured on hardware first")
    g.add_argument("--hard_abort_ft", type=float, nargs=6, default=DEFAULT_HARD_ABORT_FT)
    g.add_argument("--trip_samples", type=int, default=1,
                   help="consecutive settled samples at/above the cap to declare the "
                        "boundary. 1 is right now that _command settles: a second "
                        "sample would cost another step past the cap")

    g = p.add_argument_group("stepping")
    g.add_argument("--step_max", type=float, default=0.001, help="m per step at low load")
    g.add_argument("--rot_step_max", type=float, default=0.01, help="rad per step at low load")
    g.add_argument("--step_min_frac", type=float, default=0.1,
                   help="floor for the load-adaptive taper, as a fraction of step_max")
    g.add_argument("--min_step", type=float, default=5e-5,
                   help="m; absolute minimum commanded translation increment. Exists to "
                        "guarantee forward progress near the limit -- too small and the "
                        "arm creeps in hundreds of micro-steps and appears stalled")
    g.add_argument("--min_rot_step", type=float, default=5e-4,
                   help="rad; rotation counterpart of --min_step")
    g.add_argument("--max_dlam_step", type=float, default=0.2,
                   help="largest predicted increase in lam a single step may cause")
    g.add_argument("--predict_target", type=float, default=0.92,
                   help="lam the predictive stop aims the next step at")
    g.add_argument("--step_growth", type=float, default=1.6,
                   help="max factor the step may grow by per iteration")
    g.add_argument("--lam_slow", type=float, default=0.5)
    g.add_argument("--max_speed", type=float, default=0.02,
                   help="m/s; commanded translation speed. THIS, not step_max, is what "
                        "bounds how fast the arm moves, and it is independent of --rate")
    g.add_argument("--max_rot_speed", type=float, default=0.2,
                   help="rad/s; commanded rotation speed")
    g.add_argument("--rate", type=float, default=0.01,
                   help="s between commands (default 0.01 = 100 Hz). Controls sample "
                        "DENSITY, not speed -- speed is --max_speed, and settling is a "
                        "time budget, so both are independent of this. Higher rates are "
                        "if anything safer: more samples per step give a better settling "
                        "estimate (lam_max 1.33 at 10Hz vs 1.04 at 100Hz)")
    g.add_argument("--settle_min", type=int, default=3,
                   help="minimum samples held per step (3 is the fewest that lets the "
                        "settled-load extrapolation be computed)")
    g.add_argument("--settle_seconds_near", type=float, default=0.5,
                   help="s held per step above lam_slow, waiting for the wrench to "
                        "converge (a TIME, so it is independent of --rate)")
    g.add_argument("--settle_seconds_far", type=float, default=0.1,
                   help="s held per step at low load")
    g.add_argument("--extrapolate_cap", type=float, default=3.0,
                   help="max factor the settled-load extrapolation may exceed the "
                        "last measured sample by")
    g.add_argument("--settle_rate_tol", type=float, default=0.1,
                   help="|dlam/dt| (per second) below which the load counts as settled")
    g.add_argument("--extrapolate_dt", type=float, default=0.05,
                   help="s; time stride used to estimate the settling asymptote")
    g.add_argument("--max_track_err", type=float, default=0.005)

    g = p.add_argument_group("cage")
    g.add_argument("--backstop_margin", type=float, default=1.0)
    g.add_argument("--max_dead_backstops", type=int, default=3)
    g.add_argument("--dead_lam", type=float, default=0.25)

    g = p.add_argument_group("unloading")
    g.add_argument("--retreat_step_scale", type=float, default=1.0)
    g.add_argument("--hold_at_boundary", type=float, default=0.3,
                   help="s dwelling at the trip point (creep-at-constant-displacement "
                        "data); pure time/data tradeoff, no safety effect")
    g.add_argument("--relax_seconds", type=float, default=0.8,
                   help="s held at u=0 after the retreat; feeds slip check (a)")

    g = p.add_argument_group("zero reference")
    g.add_argument("--preload_force_tol", type=float, default=3.5,
                   help="N; must sit ABOVE the hysteresis band -- measured round-trip "
                        "residuals on deformation_test_20260819_071530 are 0.80-3.05N, "
                        "and a hysteretic residual cannot be nulled by moving")
    g.add_argument("--preload_torque_tol", type=float, default=0.15)
    g.add_argument("--null_kf", type=float, default=0.0005,
                   help="m per N; INITIAL compliance guess only -- the servo estimates "
                        "dw/du from its own steps, so this need not be accurate and its "
                        "sign is not assumed")
    g.add_argument("--null_kt", type=float, default=0.035, help="rad per Nm; initial guess")
    g.add_argument("--null_probe", type=float, default=0.0002,
                   help="m; first null step, used to learn the sign of dw/du")
    g.add_argument("--null_damping", type=float, default=0.7)
    g.add_argument("--null_max_iters", type=int, default=40)
    g.add_argument("--null_max_travel", type=float, default=0.003)
    g.add_argument("--null_max_rot", type=float, default=np.deg2rad(3.0),
                   help="rad; rotation counterpart of --null_max_travel")
    g.add_argument("--rezero_every", type=int, default=20)
    g.add_argument("--require_sensor_zero_check", action="store_true", default=True)
    g.add_argument("--no_sensor_zero_check", dest="require_sensor_zero_check",
                   action="store_false")
    g.add_argument("--sensor_zero_tol", type=float, default=2.0, help="deg")

    g = p.add_argument_group("sampling")
    g.add_argument("--probe_excursions", type=int, default=12,
                   help="Phase A creep-and-settle probes. These SEARCH for the force "
                        "boundary and populate the envelope, so the slow stepping is "
                        "correct here. 12 = the deterministic seed directions")
    g.add_argument("--hop_waypoints", type=int, default=400,
                   help="Phase B waypoints. The arm moves DIRECTLY between these at "
                        "max_speed with no per-step settling")
    g.add_argument("--waypoints_per_file", type=int, default=40,
                   help="waypoints per CSV. Each file returns to zero first, because "
                        "prepare_deflection_dataset.py builds the label relative to "
                        "row 0")
    g.add_argument("--dwell", type=float, default=0.3,
                   help="s held at each waypoint. The only place settling is still "
                        "paid for, and the samples immune to sensor lag in transit")
    g.add_argument("--hop_lam_abort", type=float, default=1.0,
                   help="lam at which a hop aborts. Checked every tick because the "
                        "force-bounded region is not known to be convex, so a straight "
                        "line between two admissible waypoints can leave it")
    g.add_argument("--hop_lookahead", type=float, default=0.15,
                   help="s of lag the hop abort predicts ahead by, compensating the "
                        "wrench trailing the commanded pose during continuous motion")
    g.add_argument("--hop_abort_retreat", type=float, default=0.5,
                   help="fraction of u retained after a load abort -- sheds load "
                        "instead of sitting at the boundary")
    g.add_argument("--seed_directions", type=int, default=12)
    g.add_argument("--axis_frac", type=float, default=0.35,
                   help="fraction of Phase B directions drawn with only 1-2 live axes, "
                        "to recover per-DoF reach that isotropic 6-D sampling dilutes")
    g.add_argument("--boundary_frac", type=float, default=0.25)
    g.add_argument("--envelope_shrink", type=float, default=0.85)
    g.add_argument("--kernel_sigma", type=float, default=0.25,
                   help="angular kernel width for the envelope. The admissible radius "
                        "varies several-fold between translation and rotation "
                        "directions, so a broad kernel over-smooths and systematically "
                        "under-reaches in the loose ones")
    g.add_argument("--envelope_halflife", type=float, default=300.0)
    g.add_argument("--prior_radius", type=float, default=0.3)
    g.add_argument("--load_envelope", default=None)

    g = p.add_argument_group("slip")
    g.add_argument("--slip_pose_tol", type=float, default=0.002)
    g.add_argument("--slip_rot_tol", type=float, default=1.0, help="deg")
    g.add_argument("--slip_wrench_tol", type=float, default=2.0)
    g.add_argument("--slip_wrench_abs", type=float, default=5.0)
    g.add_argument("--claw_gap_tol", type=float, default=0.05)

    g = p.add_argument_group("safety / ops")
    g.add_argument("--skip_ft_check", action="store_true")
    g.add_argument("--i_understand_no_force_limit", action="store_true")
    g.add_argument("--auto_unlock_protective_stop", action="store_true")
    g.add_argument("--max_unload_seconds", type=float, default=15.0,
                   help="wall-time cap on the controlled unload retreat")
    g.add_argument("--max_session_seconds", type=float, default=3600.0)
    g.add_argument("--park_only", action="store_true")
    g.add_argument("--no_claw", action="store_true")
    g.add_argument("--wrench_frame_check", action="store_true")

    g = p.add_argument_group("testing")
    g.add_argument("--dry_run", action="store_true")
    g.add_argument("--self_test", action="store_true")
    g.add_argument("--overshoot_tol", type=float, default=1.35,
                   help="self-test bound on lam. NOT 1.0, and deliberately so: the "
                        "peak occurs at a pose that was ALREADY COMMANDED while the "
                        "reading was still below the cap, after which the load relaxes "
                        "to its asymptote. No rule that observes the wrench can beat "
                        "'the asymptote of the last committed command'. Measured worst "
                        "case is 1.03-1.29 across a 25x stiffness sweep (0.2-5.0 N/mm, "
                        "vs 0.62-1.05 measured on the real rig). The real protection "
                        "against a genuine excursion is hard_abort_ft at 2x the cap, "
                        "which no longer fires anywhere in the sweep.")
    g.add_argument("--sim_stiffness", type=float, default=0.6, help="N/mm")
    g.add_argument("--sim_hysteresis", type=float, default=0.15)
    g.add_argument("--sim_lag", type=int, default=2)
    g.add_argument("--sim_noise", type=float, default=0.02,
                   help="N; simulated F/T noise floor, so a healthy sim is "
                        "distinguishable from a dead topic")
    g.add_argument("--sim_fault", default="none",
                   choices=["none", "dead_ft", "ft_dies_mid", "slip"])
    return p


def wrench_frame_check(plant, clock):
    """Print ||F|| from the raw and base-frame wrench at the current pose.

    arm.get_wrench(base_frame_control=True) routes through
    spalg.force_frame_transform = motion_frame_transform(...).T, which is built for
    (tau; f) ordering, while conversions.from_wrench delivers (f; tau). If that
    pairing is wrong the returned "force" picks up a -R skew(p) tau term, and the two
    norms below will differ by more than sensor noise. A pure rotation cannot change
    ||F||, so any difference is diagnostic.
    """
    log_info("wrench frame check -- hang a known mass and compare across poses")
    for i in range(5):
        raw = np.asarray(plant.wrench_raw(), float)
        base = np.asarray(plant.wrench_base(), float)
        log_info(f"  [{i}] |F|raw={np.linalg.norm(raw[:3]):7.3f}  "
                 f"|F|base={np.linalg.norm(base[:3]):7.3f}  "
                 f"delta={abs(np.linalg.norm(raw[:3])-np.linalg.norm(base[:3])):7.3f}  "
                 f"|T|raw={np.linalg.norm(raw[3:]):6.3f}")
        clock.sleep(0.5)


def main():
    args = build_parser().parse_args()
    args._interrupt_flag = [False]

    if args.dry_run or args.self_test:
        if not _ROS_IMPORTS_OK:
            print(f"[ERROR] ur_control imports failed: {_ROS_IMPORT_ERROR}")
            return 1
        if args.self_test:
            return self_test(args)
        clock = SimClock()
        plant = SimPlant(args, np.random.default_rng(args.seed))
        plant.park(None)
        return Session(args, plant, clock).run()

    if not _ROS_IMPORTS_OK:
        print(f"[ERROR] ROS imports failed: {_ROS_IMPORT_ERROR}")
        return 1

    # We own shutdown: the default handler would sys.exit(0) mid-command with the
    # chain loaded and never release the claw.
    rospy.init_node("deflection_force_bounded_collector", disable_signals=True)
    _set_roslog(True)

    plant = RosPlant(args)
    clock = RosClock()

    last_sigint = [0.0]

    def _sigint(_sig, _frm):
        now = clock.now()
        if now - last_sigint[0] < 2.0:
            log_err("second Ctrl+C -- hard stop")
            plant.stop()
            plant.claw_open()
            sys.exit(1)
        last_sigint[0] = now
        args._interrupt_flag[0] = True
        log_warn("Ctrl+C -- unloading before exit (press again within 2s to hard stop)")

    signal.signal(signal.SIGINT, _sigint)
    atexit.register(lambda: plant.claw_open())
    rospy.on_shutdown(lambda: plant.claw_open())

    if args.park_only:
        plant.claw_open()
        clock.sleep(1.0)
        plant.park(DEFAULT_INITIAL_UR, target_time=5.0)
        log_info("parked with the claw OPEN. Start uart_bridge_node.py NOW, with the "
                 "wrist hanging free, before gripping the post.")
        return 0

    if args.wrench_frame_check:
        wrench_frame_check(plant, clock)
        return 0

    return Session(args, plant, clock).run()


if __name__ == "__main__":
    sys.exit(main())
