"""Shared observation/action assembly for data collection and inference.

This module is the single implementation of the observation contract: given
per-source timestamped samples, it produces the exact ``observation.*`` /
``action.*`` dicts used both by the offline bag->LeRobot converter and by the
online inference environment. Keeping one implementation guarantees that
training data and inference observations are assembled identically.

Pure module: only numpy and ur_control.transformations at import time.
Kinematics (ur_pykdl.ur_kinematics) is dependency-injected so the module works
online (URDF from the parameter server) and offline (URDF snapshot from a
recording sidecar) without importing rospy or rosbag.

Canonical source keys (produced by the feeders in sample_feeders.py):
    joint_states   -> np.ndarray (12,): qpos(6) ++ qvel(6)
    wrench         -> np.ndarray (6,)
    target_frame   -> np.ndarray (7,): xyz + quaternion xyzw   (action side)
    stiffness      -> np.ndarray (6,)                          (action side)
    gello_joints   -> np.ndarray (6,)                          (action side)
    images.{cam}   -> np.ndarray HWC uint8
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from ur_control import transformations


@dataclass(frozen=True)
class Sample:
    """One timestamped sample from a source stream.

    stamp: ROS epoch seconds (header stamp, or receive time for unstamped
    sources). value: np.ndarray (states) or HWC uint8 image.
    """
    stamp: float
    value: Any


class ObservationAssembler:
    """Assemble observation/action dicts from per-source samples.

    Args:
        kinematics: ur_pykdl.ur_kinematics instance (injected).
        camera_names: cameras expected in ``images.{cam}`` samples.
    """

    STATE_KEYS = ("joint_states", "wrench")

    def __init__(self, kinematics, camera_names: List[str]):
        self.kinematics = kinematics
        self.camera_names = list(camera_names or [])

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def assemble_observation(
        self,
        samples: Dict[str, Sample],
        tick_time: float,
        episode_start: float = 0.0,
        include_images: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Build the ``observation.*`` dict for one tick.

        Every value is derived from the latest causal sample per source
        (grab-latest semantics, identical online and offline). FK/twist are
        recomputed from joint samples through the injected kinematics - the
        same math as ur_control.Arm.end_effector/end_effector_velocity.

        ``observation.frame_time`` is the tick instant and
        ``observation.image_time.{cam}`` each frame's capture instant, both
        relative to ``episode_start`` and on the same clock, so
        vision-vs-state skew stays measurable in the dataset.
        """
        js = samples["joint_states"]
        qpos = np.asarray(js.value[:6], dtype=np.float64)
        qvel = np.asarray(js.value[6:12], dtype=np.float64)
        wrench = np.asarray(samples["wrench"].value, dtype=np.float64)

        eef = np.asarray(self.kinematics.forward(qpos))
        eef_vel = np.asarray(self.kinematics.forward_velocity(qpos, qvel))

        obs = {
            "observation.qpos":                    qpos,
            "observation.qvel":                    qvel,
            "observation.eef.position":            eef[:3],
            "observation.eef.linear_velocity":     eef_vel[:3],
            "observation.eef.angular_velocity":    eef_vel[3:],
            "observation.eef.rotation_ortho6":     np.asarray(transformations.ortho6_from_quaternion(eef[3:])),
            "observation.eef.rotation_axis_angle": np.asarray(transformations.axis_angle_from_quaternion(eef[3:])),
            "observation.ft":                      wrench,
        }
        obs = {k: v.astype(np.float32) for k, v in obs.items()}

        if include_images:
            for cam in self.camera_names:
                sample = samples.get(f"images.{cam}")
                if sample is None:
                    raise KeyError(f"Missing camera sample: images.{cam}")
                obs[f"observation.images.{cam}"] = sample.value
                obs[f"observation.image_time.{cam}"] = np.array(
                    [sample.stamp - episode_start], dtype=np.float32
                )

        obs["observation.frame_time"] = np.array(
            [tick_time - episode_start], dtype=np.float32
        )
        return obs

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def assemble_action(
        self,
        samples: Dict[str, Sample],
        eef_pose: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Build the ``action.*`` dict for one tick.

        The action is the recorded commanded target (target_frame topic,
        already safety-clipped by the live teleop loop), labeled with
        execution-matching delta semantics: delta = target(t) (-) measured
        eef pose(t), exactly how FDCCEnv applies delta actions. Deltas are
        therefore rate-correct for whatever tick grid this is sampled on,
        and are NOT re-clipped here.

        Args:
            samples: must contain target_frame, stiffness, gello_joints.
            eef_pose: (7,) xyz+quat measured pose at the same tick
                (i.e. from assemble_observation's joint sample).
        """
        target = np.asarray(samples["target_frame"].value, dtype=np.float64)
        stiffness = np.asarray(samples["stiffness"].value, dtype=np.float64)
        gello_joints = np.asarray(samples["gello_joints"].value, dtype=np.float64)
        eef_pose = np.asarray(eef_pose, dtype=np.float64)

        delta_position = target[:3] - eef_pose[:3]
        delta_rotation = np.asarray(
            transformations.quaternions_orientation_error(target[3:], eef_pose[3:])
        )

        action = {
            "action.joint":               gello_joints,
            "action.position":            target[:3],
            "action.rotation_ortho6":     np.asarray(transformations.ortho6_from_quaternion(target[3:])),
            "action.rotation_axis_angle": np.asarray(transformations.axis_angle_from_quaternion(target[3:])),
            "action.stiffness_diag":      stiffness,
            "action.delta_position":      delta_position,
            "action.delta_rotation":      delta_rotation,
        }
        return {k: v.astype(np.float32) for k, v in action.items()}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def eef_pose_from_samples(self, samples: Dict[str, Sample]) -> np.ndarray:
        """Measured eef pose (7,) xyz+quat from the tick's joint sample."""
        qpos = np.asarray(samples["joint_states"].value[:6], dtype=np.float64)
        return np.asarray(self.kinematics.forward(qpos))

    def required_keys(self, with_action: bool = True, with_images: bool = True) -> List[str]:
        """Source keys that must have a causal sample before a tick is valid."""
        keys = list(self.STATE_KEYS)
        if with_images:
            keys += [f"images.{cam}" for cam in self.camera_names]
        if with_action:
            keys += ["target_frame", "stiffness", "gello_joints"]
        return keys
