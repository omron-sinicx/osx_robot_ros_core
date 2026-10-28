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


class TargetTracker:
    """Owns the commanded tool0 pose [x y z qx qy qz qw] and limits every step applied to it.

    A step is limited, in order, by the DeltaActionLimiter (per-axis velocity and acceleration),
    the workspace box, and the max distance between commanded and measured position.
    """

    def __init__(self, limiter, dt, workspace_min, workspace_max, max_tracking_error):
        self.limiter = limiter
        self.dt = dt
        self.workspace_min = np.asarray(workspace_min, dtype=float)
        self.workspace_max = np.asarray(workspace_max, dtype=float)
        self.max_tracking_error = max_tracking_error
        self.pose = None

    def reanchor(self, measured_pose):
        """Set the target to the measured pose and clear the limiter's velocity history."""
        self.pose = np.array(measured_pose, dtype=float)
        self.limiter.reset()

    def in_workspace(self, position):
        return bool(np.all(position >= self.workspace_min) and np.all(position <= self.workspace_max))

    def step(self, delta, measured_pose):
        """Move the target by delta [dx dy dz rx ry rz] (base_link, m / rad); return the new pose."""
        d_position, d_rotation = self.limiter.clip(delta[:3], delta[3:], self.dt)
        current = self.pose[:3]
        # The box only blocks motion that leaves it or goes further out; it never pushes the target.
        position = np.clip(current + d_position,
                           np.minimum(self.workspace_min, current),
                           np.maximum(self.workspace_max, current))
        position = measured_pose[:3] + limit_norm(position - measured_pose[:3], self.max_tracking_error)
        self.pose = np.concatenate([position, rotate(self.pose[3:], d_rotation)])
        return self.pose.copy()
