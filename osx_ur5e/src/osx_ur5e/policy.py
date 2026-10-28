"""Interface for an autonomous policy run from the SpaceMouse control loop."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobotState:
    """Measured robot state handed to the policy."""
    stamp: float            # ROS time [s]
    joints: np.ndarray      # (6,) joint positions [rad]
    pose: np.ndarray        # (7,) tool0 in base_link: position [m], quaternion xyzw
    wrench: np.ndarray      # (6,) F/T sensor reading: force [N], torque [Nm]


class Policy:
    """Proposes Cartesian motion; the control loop applies every motion and safety limit.

    The loop calls act() once per period_s and spreads the returned delta evenly over the
    control ticks of that period.
    """

    period_s = 0.05

    def reset(self, state: RobotState) -> None:
        """Start a new episode; called each time the operator engages the policy."""

    def act(self, state: RobotState) -> np.ndarray:
        """Return the (6,) delta for one period: base_link translation [m], rotation vector [rad]."""
        raise NotImplementedError

    def close(self) -> None:
        """Release models and subscriptions."""
