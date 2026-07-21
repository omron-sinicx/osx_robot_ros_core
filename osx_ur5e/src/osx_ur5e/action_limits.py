import numpy as np

_AXIS_LABELS = ("x", "y", "z")


def _as_limit_array(value, size=3):
    """Normalize a scalar/sequence limit into a length-``size`` array, or None."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.full(size, float(arr))
    return arr


def _changed_axes(before, after):
    """Return axis labels where ``before`` and ``after`` differ."""
    return [label for label, b, a in zip(_AXIS_LABELS, before, after) if not np.isclose(b, a)]


class DeltaActionLimiter:
    """Clip per-step Cartesian delta actions by magnitude, velocity, and acceleration.

    Given a control period ``dt``, each translation/rotation delta is limited in
    three successive stages:
      1. magnitude:     ``|delta| <= max_delta``
      2. velocity:      ``|delta / dt| <= max_velocity``
      3. acceleration:  ``|(v - v_prev) / dt| <= max_acceleration``

    Acceleration limiting is stateful: the previously commanded velocity is stored
    and must be cleared with :meth:`reset` at the start of each episode/reset so the
    first step of a new episode is not accelerated relative to a stale velocity.

    Rotation quantities are treated as per-axis RPY deltas in radians; the velocity
    and acceleration limits must therefore be provided in radians as well. All
    velocity/acceleration limits accept either a scalar or a length-3 (per-axis)
    value; ``None`` disables that particular stage.

    After each :meth:`clip` call, :attr:`last_clip_triggers` holds which stages
    fired, e.g. ``{"translation": ["magnitude[x]"], "orientation": []}``.
    """

    def __init__(self,
                 max_delta_translation,
                 max_delta_rotation,
                 max_linear_velocity=None,
                 max_linear_acceleration=None,
                 max_angular_velocity=None,
                 max_angular_acceleration=None):
        self.max_delta_translation = max_delta_translation
        self.max_delta_rotation = max_delta_rotation
        self.max_linear_velocity = _as_limit_array(max_linear_velocity)
        self.max_linear_acceleration = _as_limit_array(max_linear_acceleration)
        self.max_angular_velocity = _as_limit_array(max_angular_velocity)
        self.max_angular_acceleration = _as_limit_array(max_angular_acceleration)
        self.reset()

    def reset(self):
        """Clear the stored velocities. Call at the start of each episode/reset."""
        self._prev_linear_velocity = np.zeros(3)
        self._prev_angular_velocity = np.zeros(3)
        self.last_clip_triggers = {"translation": [], "orientation": []}

    def clip(self, delta_translation, delta_orientation, dt,
             max_delta_translation=None, max_delta_rotation=None):
        """Return velocity/acceleration-limited copies of the input deltas.

        ``max_delta_translation`` / ``max_delta_rotation`` optionally override the
        configured magnitude limits for this call (e.g. slower homing moves).
        """
        delta_translation = np.asarray(delta_translation, dtype=float)
        delta_orientation = np.asarray(delta_orientation, dtype=float)

        if max_delta_translation is None:
            max_delta_translation = self.max_delta_translation
        if max_delta_rotation is None:
            max_delta_rotation = self.max_delta_rotation

        clipped_translation, self._prev_linear_velocity, trans_triggers = self._clip_axis(
            delta_translation, dt, max_delta_translation,
            self.max_linear_velocity, self.max_linear_acceleration,
            self._prev_linear_velocity)
        clipped_orientation, self._prev_angular_velocity, ori_triggers = self._clip_axis(
            delta_orientation, dt, max_delta_rotation,
            self.max_angular_velocity, self.max_angular_acceleration,
            self._prev_angular_velocity)
        self.last_clip_triggers = {
            "translation": trans_triggers,
            "orientation": ori_triggers,
        }

        return clipped_translation, clipped_orientation

    @staticmethod
    def _clip_axis(delta, dt, max_delta, max_velocity, max_acceleration, prev_velocity):
        triggers = []

        mag_clipped = np.clip(delta, -max_delta, max_delta)
        mag_axes = _changed_axes(delta, mag_clipped)
        if mag_axes:
            triggers.append(f"magnitude[{','.join(mag_axes)}]")
        delta = mag_clipped

        velocity = delta / dt
        if max_velocity is not None:
            vel_clipped = np.clip(velocity, -max_velocity, max_velocity)
            vel_axes = _changed_axes(velocity, vel_clipped)
            if vel_axes:
                triggers.append(f"velocity[{','.join(vel_axes)}]")
            velocity = vel_clipped
        if max_acceleration is not None:
            dv_max = max_acceleration * dt
            acc_clipped = np.clip(velocity, prev_velocity - dv_max, prev_velocity + dv_max)
            acc_axes = _changed_axes(velocity, acc_clipped)
            if acc_axes:
                triggers.append(f"acceleration[{','.join(acc_axes)}]")
            velocity = acc_clipped

        return velocity * dt, velocity, triggers


def limiter_from_safety_config(safety_cfg):
    """Build a :class:`DeltaActionLimiter` from a controller ``safety_parameters`` config.

    ``max_delta_rotation`` and the angular velocity/acceleration limits are stored in
    degrees in the config (to match ``max_delta_rotation``) and are converted to radians
    here. Missing velocity/acceleration keys leave the corresponding stage disabled.
    """
    def _get(key):
        getter = getattr(safety_cfg, "get", None)
        return getter(key, None) if getter is not None else safety_cfg[key]

    def _deg2rad_or_none(value):
        return None if value is None else np.deg2rad(value)

    return DeltaActionLimiter(
        max_delta_translation=safety_cfg.max_delta_translation,
        max_delta_rotation=np.deg2rad(safety_cfg.max_delta_rotation),
        max_linear_velocity=_get("max_linear_velocity"),
        max_linear_acceleration=_get("max_linear_acceleration"),
        max_angular_velocity=_deg2rad_or_none(_get("max_angular_velocity")),
        max_angular_acceleration=_deg2rad_or_none(_get("max_angular_acceleration")),
    )
