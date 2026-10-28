"""3Dconnexion SpaceMouse input through pyspacemouse, mapped to base_link axes."""

import threading
import time
from dataclasses import dataclass

import numpy as np
import pyspacemouse

# Button indices on the SpaceMouse Compact.
LEFT = 0
RIGHT = 1


@dataclass(frozen=True)
class MouseReading:
    """One sample of the device in base_link axes."""
    twist: np.ndarray       # (6,) [vx vy vz wx wy wz] in [-1, 1]; zero when stale or disconnected
    left: bool
    right: bool
    connected: bool
    report_age_s: float     # time since the last HID report


def apply_deadzone(values, deadzone):
    """Zero |v| <= deadzone and rescale the rest so the output still spans [-1, 1]."""
    values = np.clip(values, -1.0, 1.0)
    return np.sign(values) * np.maximum(np.abs(values) - deadzone, 0.0) / (1.0 - deadzone)


def map_to_base(raw, operator_yaw_deg, deadzone):
    """Map raw HID_Z_UP axes [x y z roll pitch yaw] to a base_link twist in [-1, 1]."""
    shaped = apply_deadzone(np.asarray(raw, dtype=float), deadzone)
    c, s = np.cos(np.deg2rad(operator_yaw_deg)), np.sin(np.deg2rad(operator_yaw_deg))
    yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return np.concatenate([yaw @ shaped[:3], yaw @ shaped[3:]])


class SpaceMouse:
    """Reads the device in a background thread and serves the latest sample.

    pyspacemouse's read() consumes one HID report per call, so reading once per control tick
    lets reports queue up in the kernel and the command lags behind the hand.
    """

    def __init__(self, operator_yaw_deg, deadzone, stale_timeout_s, device=None):
        self.operator_yaw_deg = operator_yaw_deg
        self.deadzone = deadzone
        self.stale_timeout_s = stale_timeout_s
        self._device = device or pyspacemouse.open(axis_convention=pyspacemouse.AxisConvention.HID_Z_UP)

        self._lock = threading.Lock()
        self._raw = np.zeros(6)
        self._buttons = (False, False)
        self._last_report = time.monotonic()
        self._report_count = 0
        self._error = None

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read_loop, name="spacemouse", daemon=True)
        self._thread.start()

    @property
    def report_count(self):
        with self._lock:
            return self._report_count

    def read(self) -> MouseReading:
        with self._lock:
            raw, (left, right) = self._raw, self._buttons
            last_report, error = self._last_report, self._error
        age = time.monotonic() - last_report
        if error is not None:
            return MouseReading(np.zeros(6), False, False, False, age)
        # Buttons report only on change, so they stay valid while the axes go stale.
        twist = map_to_base(raw, self.operator_yaw_deg, self.deadzone) \
            if age <= self.stale_timeout_s else np.zeros(6)
        return MouseReading(twist, left, right, True, age)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._device.close()

    def _read_loop(self):
        last_t = None
        while not self._stop.is_set():
            try:
                state = self._device.read()
            except Exception as error:  # easyhid raises once the device is unplugged
                with self._lock:
                    self._error = error
                return
            if state.t == last_t:
                time.sleep(0.001)
                continue
            last_t = state.t
            raw = np.array([state.x, state.y, state.z, state.roll, state.pitch, state.yaw])
            buttons = (bool(state.buttons[LEFT]), bool(state.buttons[RIGHT]))
            with self._lock:
                self._raw, self._buttons = raw, buttons
                self._last_report = time.monotonic()
                self._report_count += 1


class RightTap:
    """Detects a RIGHT press-and-release during which LEFT was never pressed (LEFT+RIGHT is a chord)."""

    def __init__(self):
        self._held = False
        self._chorded = False

    def update(self, left, right):
        """Return True on the release that ends a solo RIGHT press."""
        if right and not self._held:
            self._chorded = False
        self._chorded = self._chorded or (left and right)
        released = self._held and not right
        self._held = right
        return released and not self._chorded
