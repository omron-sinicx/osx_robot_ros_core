"""Local-frame TCP offset shared by the replay and evaluation scripts.

Why this exists
---------------
The gripper geometry that Point-Policy's pipeline derives from human hand
tracking (gripper_points.py's Tshift + extrapoints) is a hand-measured
approximation, so the physically-correct TCP can sit a centimetre or two away
-- and a few degrees rotated -- from where the recorded actions place it.
Rather than re-deriving the dataset, these scripts shift the *commanded*
pose by a fixed translation (and optionally yaw) offset at runtime.

The offset is applied in the pose's OWN local frame (rotated by whatever
orientation is being commanded), not in the world frame -- so "+z" always means
"further out along the gripper's reach axis" and "+y" always means "toward the
same finger", and yaw always means "spin about the gripper's own current
approach axis", regardless of which way the gripper happens to be pointing. A
world-frame offset would smear across different axes as the wrist rotates,
which is almost never what you want for a tool-tip correction.

Config
------
    dataset.eval_offset: [x, y, z]        # metres, local frame; [0,0,0] = disabled
    dataset.eval_yaw_offset_deg: float    # degrees, local frame; 0 = disabled

The older scalar `dataset.eval_z_offset` is still honoured (as [0, 0, z]) so
existing configs keep working, but `eval_offset` wins when both are present.
"""

import numpy as np

from ur_control import transformations

OFFSET_KEY = "eval_offset"
LEGACY_Z_KEY = "eval_z_offset"
YAW_OFFSET_KEY = "eval_yaw_offset_deg"


def resolve_offset(ds_cfg) -> np.ndarray:
    """Read the local-frame offset from a dataset config node.

    Returns a (3,) float array in metres. All-zeros means disabled. Prefers
    `eval_offset` ([x, y, z]); falls back to the legacy scalar `eval_z_offset`
    ([0, 0, z]); returns zeros when neither is set or ds_cfg is None.
    """
    if ds_cfg is None:
        return np.zeros(3, dtype=float)

    raw = ds_cfg.get(OFFSET_KEY, None)
    if raw is not None:
        offset = np.asarray([float(v) for v in raw], dtype=float)
        if offset.shape != (3,):
            raise ValueError(
                f"dataset.{OFFSET_KEY} must be 3 elements [x, y, z] in metres, "
                f"got {len(offset)}: {list(offset)}"
            )
        return offset

    return np.array([0.0, 0.0, float(ds_cfg.get(LEGACY_Z_KEY, 0.0))], dtype=float)


def resolve_yaw_offset(ds_cfg) -> float:
    """Read the local-frame yaw offset (radians) from a dataset config node.

    Returns 0.0 (disabled) when ds_cfg is None or dataset.eval_yaw_offset_deg
    is unset. Config is in degrees (matches execute_cartesian_target's
    safety_cfg.max_delta_rotation convention); returned in radians, since
    that's what rotate_quaternion_by_rpy takes.
    """
    if ds_cfg is None:
        return 0.0
    return float(np.deg2rad(float(ds_cfg.get(YAW_OFFSET_KEY, 0.0))))


def is_enabled(offset, yaw: float = 0.0) -> bool:
    """Whether `offset` and/or `yaw` is a real (non-zero) shift worth applying."""
    return bool(np.any(np.asarray(offset, dtype=float) != 0.0)) or yaw != 0.0


def apply_offset(pose: np.ndarray, offset, yaw: float = 0.0) -> np.ndarray:
    """Shift `pose` ([x,y,z,qx,qy,qz,qw]) by `offset` along its own local axes,
    then rotate it by `yaw` radians about its own (post-translation) local
    z-axis -- both relative to whatever orientation is currently commanded.

    Returns `pose` unchanged when offset and yaw are both zero, so callers
    can pass them unconditionally.
    """
    if not is_enabled(offset, yaw):
        return pose
    pose = transformations.translate_pose(
        pose, np.asarray(offset, dtype=float), rotated_frame=True
    )
    if yaw != 0.0:
        pose[3:] = transformations.rotate_quaternion_by_rpy(0.0, 0.0, yaw, pose[3:], rotated_frame=True)
    return pose


def undo_offset(pose: np.ndarray, offset, yaw: float = 0.0) -> np.ndarray:
    """Exact inverse of apply_offset -- same local-frame shift and yaw, negated,
    applied in REVERSE order (undo the yaw first, then the translation) so it
    is an exact inverse even when both offset and yaw are non-zero: since
    apply_offset's translation is rotated by the PRE-yaw orientation, undoing
    the translation must likewise use the PRE-yaw (i.e. post-undo-yaw)
    orientation, not the pose's current (post-yaw) one.

    Used to strip the offset back out of an observation so a policy keeps
    seeing the frame it was trained in while the arm physically operates
    `offset`/`yaw` away from it.
    """
    if not is_enabled(offset, yaw):
        return pose
    if yaw != 0.0:
        pose = pose.copy()
        pose[3:] = transformations.rotate_quaternion_by_rpy(0.0, 0.0, -yaw, pose[3:], rotated_frame=True)
    return transformations.translate_pose(
        pose, -np.asarray(offset, dtype=float), rotated_frame=True
    )


def describe(offset, yaw: float = 0.0) -> str:
    """Human-readable one-liner for logging."""
    o = np.asarray(offset, dtype=float)
    base = f"x={o[0]:+.4f} y={o[1]:+.4f} z={o[2]:+.4f} m (local frame)"
    if yaw != 0.0:
        base += f" yaw={np.degrees(yaw):+.2f} deg (local frame)"
    return base
