"""Commanded tool pose with per-step motion limits for Cartesian teleoperation."""

import numpy as np

from ur_control import transformations


def rotate(quaternion, rotation_vector):
    """Rotate an xyzw quaternion by a base-frame rotation vector [rad]."""
    delta = transformations.quaternion_from_axis_angle(np.asarray(rotation_vector, dtype=float))
    result = transformations.quaternion_multiply(delta, quaternion)
    return result / np.linalg.norm(result)


def rotation_error(target, current):
    """Base-frame rotation vector [rad] that rotates xyzw quaternion `current` onto `target`."""
    error = transformations.quaternion_multiply(target, transformations.quaternion_conjugate(current))
    return transformations.axis_angle_from_quaternion(error)


def limit_norm(vector, max_norm):
    norm = np.linalg.norm(vector)
    return vector * (max_norm / norm) if norm > max_norm else vector


def limit_lead(current, proposed, measured, max_lead):
    """Shorten the step current -> proposed so it does not take the target further than
    max(max_lead, current lead) from the measured position.

    The step is only ever shortened: a lead the target already has (the robot drifted, or contact
    holds it back) is kept, so the target never moves without a commanded step.
    """
    step = proposed - current
    lead = current - measured
    radius = max(max_lead, np.linalg.norm(lead))
    if np.linalg.norm(proposed - measured) <= radius:
        return proposed
    # Largest s in [0, 1] with |lead + s * step| <= radius.
    a, b, c = step @ step, lead @ step, lead @ lead - radius ** 2
    s = (-b + np.sqrt(max(b * b - a * c, 0.0))) / a
    return current + np.clip(s, 0.0, 1.0) * step




class PolicyTargetManager:
    """C2 kinematic anti-windup; opt-in, per-axis and fresh-action authorized.

    A low measured progress rate toward EXISTING lead is only an obstruction
    proxy, not force/contact ground truth. No hidden integrated target is kept:
    outward blocked requests are discarded; reversal release uses a frozen
    measured tool0 anchor. Every output still goes through TargetTracker.step.
    """
    KEYS = {'enabled', 'axes', 'action_neutral_band_m', 'lead_enter_m',
            'lead_exit_m', 'blocked_speed_m_s', 'blocked_duration_s',
            'max_observation_gap_s'}

    @classmethod
    def validate(cls, value):
        if value is None:
            return None
        if not isinstance(value, dict) or set(value) != cls.KEYS:
            raise ValueError('policy_target_management requires exact version-v1 options')
        if type(value['enabled']) is not bool:
            raise ValueError('target management enabled must be boolean')
        axes = value['axes']
        if not isinstance(axes, list) or not axes or len(set(axes)) != len(axes) or any(not isinstance(a, str) or len(a) != 1 or a not in 'xyz' for a in axes):
            raise ValueError('target management axes must be unique x/y/z')
        for key in cls.KEYS - {'enabled', 'axes'}:
            v = value[key]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0:
                raise ValueError('target management numeric options must be finite positive')
        if not value['lead_exit_m'] < value['lead_enter_m'] or not value['max_observation_gap_s'] <= value['blocked_duration_s']:
            raise ValueError('target management hysteresis/dwell order invalid')
        return value if value['enabled'] else None

    def __init__(self, options):
        self.options = dict(options)
        self.axes = tuple('xyz'.index(a) for a in options['axes'])
        self.counts = dict(releases=[0, 0, 0], discarded_m=[0., 0., 0.],
                           blocked_ticks=[0, 0, 0], gap_resets=0)
        self.reset()

    def reset(self):
        self.blocked = np.zeros(3, dtype=bool)
        self.release = np.zeros(3, dtype=bool)
        self.anchor = np.zeros(3)
        self.sign = np.zeros(3, dtype=int)
        self.slow_duration = np.zeros(3)
        self._previous_eligible = np.zeros(3, dtype=bool)
        self._last_measurement = None
        self._last_observed_s = None
        self._step_error = np.zeros(3)
        self.last_reason = ['inactive'] * 3

    def observe(self, action, measured, published, now_s):
        action, measured, published = map(lambda x: np.asarray(x, dtype=float), (action, measured, published))
        if (action.shape != (6,) or measured.shape != (7,) or published.shape != (7,) or
            not all(np.isfinite(x).all() for x in (action, measured, published)) or not np.isfinite(now_s)):
            raise ValueError('FINITE_TARGET_MANAGEMENT_INPUT_REQUIRED')
        lead = published[:3] - measured[:3]
        dt = None if self._last_observed_s is None else now_s - self._last_observed_s
        valid_interval = dt is not None and 0 < dt <= self.options['max_observation_gap_s'] + 1e-12
        if dt is not None and dt <= 0:
            raise ValueError('TARGET_MANAGEMENT_CLOCK_NOT_INCREASING')
        progress = np.zeros(3) if not valid_interval else (measured[:3] - self._last_measurement) / dt * np.sign(lead)
        if not valid_interval:
            self.slow_duration[:] = 0.
            self._previous_eligible[:] = False
            if dt is not None:
                self.blocked[:] = False
                self.release[:] = False
                self.counts['gap_resets'] += 1
        for axis in self.axes:
            value = action[axis]
            sign = 0 if abs(value) <= self.options['action_neutral_band_m'] else (1 if value > 0 else -1)
            self.sign[axis] = sign
            # Hysteresis releases the obstruction latch once tracking catches up.
            caught_up = abs(lead[axis]) <= self.options['lead_exit_m']
            progressing = valid_interval and progress[axis] > self.options['blocked_speed_m_s']
            eligible = abs(lead[axis]) >= self.options['lead_enter_m'] and not progressing
            if caught_up or progressing:
                self.blocked[axis] = False
                self.slow_duration[axis] = 0.
            elif eligible:
                if valid_interval and self._previous_eligible[axis]:
                    self.slow_duration[axis] += dt
                else:
                    self.slow_duration[axis] = 0.
                if self.slow_duration[axis] >= self.options['blocked_duration_s'] - 1e-12:
                    self.blocked[axis] = True
            elif not self.blocked[axis]:
                self.slow_duration[axis] = 0.
            self._previous_eligible[axis] = eligible
            # Neutral pauses release; it never adds autonomous escape motion.
            # An opposite requested direction cancels the old anchor.
            if self.release[axis] and (caught_up or (sign and (
                    sign * (self.anchor[axis] - published[axis]) <= 0 or
                    sign * lead[axis] >= -self.options['lead_exit_m']))):
                self.release[axis] = False
            if (not self.release[axis] and self.blocked[axis] and sign and
                lead[axis] * sign < 0 and abs(lead[axis]) >= self.options['lead_enter_m']):
                self.anchor[axis] = measured[axis]
                self.release[axis] = True
                self.counts['releases'][axis] += 1
        self._last_measurement = measured[:3].copy()
        self._last_observed_s = float(now_s)

    def apply(self, delta, published, measured):
        delta, published, measured = map(lambda x: np.asarray(x, dtype=float), (delta, published, measured))
        if (delta.shape != (6,) or published.shape != (7,) or measured.shape != (7,) or
                not all(np.isfinite(x).all() for x in (delta, published, measured))):
            raise ValueError('FINITE_TARGET_MANAGEMENT_STEP_REQUIRED')
        request = delta.copy()
        lead = published[:3] - measured[:3]
        self._step_error = self.anchor - published[:3]
        self.last_reason = ['normal'] * 3
        for axis in self.axes:
            if self.release[axis] and (abs(lead[axis]) <= self.options['lead_exit_m'] or
                    (self.sign[axis] and lead[axis] * self.sign[axis] >= -self.options['lead_exit_m'])):
                self.release[axis] = False
            if self.release[axis] and self.sign[axis] and self._step_error[axis] * self.sign[axis] > 0:
                # No accumulating reference: a held, bounded anchor consumes the
                # reverse request on this substep. The original delta is NOT added again.
                request[axis] = self._step_error[axis]
                self.last_reason[axis] = 'release_opposing_lead'
            elif self.blocked[axis] and lead[axis] * request[axis] > 0:
                self.counts['discarded_m'][axis] += abs(float(request[axis]))
                self.counts['blocked_ticks'][axis] += 1
                request[axis] = 0.
                self.last_reason[axis] = 'discard_blocked_outward_request'
            elif self.release[axis] and not self.sign[axis]:
                self.last_reason[axis] = 'neutral_pauses_release'
        return request

    def complete_step(self, published):
        # Consume only the actually accepted limited output; no leftover queue.
        error = self.anchor - np.asarray(published[:3])
        reached = (abs(error) <= 1e-12) | (error * self._step_error <= 0)
        self.release &= ~reached



class TargetTracker:
    """Owns the commanded tool0 pose [x y z qx qy qz qw] and limits every step applied to it.

    A step is limited, in order, by the DeltaActionLimiter (per-axis velocity and acceleration),
    the workspace box, and max_tracking_error (see limit_lead). A policy-only experiment can
    explicitly skip the last limit or acceleration clipping for a step, retaining
    velocity and workspace limits. Without a step the target stays put.
    """

    def __init__(self, limiter, dt, workspace_min, workspace_max, max_tracking_error):
        self.limiter = limiter
        self.dt = dt
        self.workspace_min = np.asarray(workspace_min, dtype=float)
        self.workspace_max = np.asarray(workspace_max, dtype=float)
        self.max_tracking_error = max_tracking_error
        self.pose = None
        self.trace_enabled = False
        self.last_step_diagnostics = None

    def reanchor(self, measured_pose):
        """Set the target to the measured pose and clear the limiter's velocity history."""
        p = np.asarray(measured_pose, dtype=float)
        if p.shape != (7,) or not np.isfinite(p).all() or np.linalg.norm(p[3:]) <= 1e-8:
            raise ValueError('FINITE_REANCHOR_POSE_REQUIRED')
        self.pose = p.copy()
        self.limiter.reset()

    def in_workspace(self, position):
        return bool(np.all(position >= self.workspace_min) and np.all(position <= self.workspace_max))

    def step(self, delta, measured_pose, *, enforce_tracking_error=True, enforce_acceleration=True, dt_s=None):
        """Move by delta [dx dy dz rx ry rz] (base_link, m / rad), retaining motion/workspace limits.

        enforce_tracking_error=False skips only the target-to-measured translation lead cap.
        enforce_acceleration=False skips only acceleration clipping in the action limiter.
        """
        if (np.asarray(delta).shape != (6,) or np.asarray(measured_pose).shape != (7,) or
                self.pose is None or not all(np.isfinite(x).all() for x in (delta, measured_pose, self.pose))):
            raise ValueError('FINITE_TRACKER_STATE_REQUIRED')
        d_position, d_rotation = self.limiter.clip(
            delta[:3], delta[3:], self.dt if dt_s is None else dt_s, enforce_acceleration=enforce_acceleration)
        current = self.pose[:3]
        # The box only blocks motion that leaves it or goes further out; it never pushes the target.
        position = np.clip(current + d_position,
                           np.minimum(self.workspace_min, current),
                           np.maximum(self.workspace_max, current))
        after_workspace = position.copy() if self.trace_enabled else None
        if enforce_tracking_error:
            position = limit_lead(current, position, np.asarray(measured_pose[:3], dtype=float),
                                  self.max_tracking_error)
        if self.trace_enabled:
            self.last_step_diagnostics = dict(
                managed_raw_goal_xyz_m=(current + np.asarray(delta[:3])).tolist(),
                motion_limited_goal_xyz_m=(current + d_position).tolist(),
                workspace_limited_goal_xyz_m=after_workspace.tolist(),
                tracking_limited_goal_xyz_m=position.tolist(),
                acceleration_limiting=bool(enforce_acceleration), tracking_limiting=bool(enforce_tracking_error),
                internal_limiter_velocity_m_s=self.limiter._prev_linear_velocity.tolist())
        self.pose = np.concatenate([position, rotate(self.pose[3:], d_rotation)])
        return self.pose.copy()
