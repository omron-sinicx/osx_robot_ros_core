import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from ur_control import transformations


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

_FMT = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"


class _FlushingFileHandler(logging.FileHandler):
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def setup_logging(*log_files: Path) -> None:
    """Install eval log handlers on the root logger.

    Call after ``rospy.init_node()`` — ROS reconfigures logging via
    ``logging.config`` and drops handlers installed earlier (including Hydra's).

    Removes ROS's ``RosStreamHandler`` from the ``rosout`` logger so
    ``rospy.loginfo()`` is not printed twice (once by ROS, once via root
    propagation). ``RosOutHandler`` is kept so messages still publish to
    ``/rosout``.
    """
    formatter = logging.Formatter(_FMT)
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    for handler in root.handlers[:]:
        handler.close()
        root.removeHandler(handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    for log_file in log_files:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if log_file.exists() and log_file.stat().st_size > 0 else "w"
        file_handler = _FlushingFileHandler(log_file, mode=mode)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    rosout = logging.getLogger("rosout")
    for handler in rosout.handlers[:]:
        if handler.__class__.__name__ == "RosStreamHandler":
            handler.close()
            rosout.removeHandler(handler)

    logging.getLogger("rospy.internal").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Observation formatting
# ---------------------------------------------------------------------------

def format_real_robot_observations(
    arm,
    image_recorder,
    features: dict,
    camera_shape: tuple,
) -> dict:
    """Build a policy-ready observation dict from the real robot arm and cameras.

    Mirrors get_observations() in data_collection.py but returns torch tensors
    in the same format expected by the COMET policy (float32 states, uint8 images).

    Args:
        arm: CompliantController instance from FDCCEnv.
        image_recorder: ImageRecorder instance from FDCCEnv.
        features: Feature dict loaded from the policy checkpoint (used to filter keys).
        camera_shape: (H, W) to resize camera images to match training resolution.

    Returns:
        Dict mapping observation keys to torch tensors ready for policy.select_action().
    """
    eef = arm.end_effector()
    eef_velocity = arm.end_effector_velocity()

    raw_obs = {
        "observation.qpos":                    arm.joint_angles(),
        "observation.qvel":                    arm.joint_velocities(),
        "observation.eef.position":            eef[:3],
        "observation.eef.linear_velocity":     eef_velocity[:3],
        "observation.eef.angular_velocity":    eef_velocity[3:],
        "observation.eef.rotation_ortho6":     transformations.ortho6_from_quaternion(eef[3:]),
        "observation.eef.rotation_axis_angle": transformations.axis_angle_from_quaternion(eef[3:]),
        "observation.ft":                      arm.get_wrench(),  # TODO (malek): check if this is in the world frame or the tool frame with cristian
    }

    obs = {}

    # State observations — only keep keys the policy actually uses
    for key, value in raw_obs.items():
        if key in features:
            obs[key] = torch.tensor(np.array(value).flatten(), dtype=torch.float32)

    # Camera images
    if image_recorder is not None:
        resize_transform = transforms.Resize(camera_shape, antialias=True)
        raw_images = image_recorder.get_images()
        for cam_name, image_hwc in raw_images.items():
            feat_key = f"observation.images.{cam_name}"
            if feat_key in features:
                image_chw = np.ascontiguousarray(np.transpose(image_hwc, (2, 0, 1)))
                image_tensor = torch.tensor(image_chw, dtype=torch.uint8)
                obs[feat_key] = resize_transform(image_tensor)

    return obs


# ---------------------------------------------------------------------------
# Action conversion
# ---------------------------------------------------------------------------

def convert_policy_action(action_dict: dict, actions_as_deltas: bool) -> dict:
    """Convert COMET policy output tensors to the FDCCEnv action dict format.

    FDCCEnv.set_compliant_control_action() expects:
        action['action.position']     — numpy array (3,)
        action['action.orientation']  — numpy array (3,) for deltas or (6,) for absolute
        action['action.stiffness_diag'] or action['action.stiffness_cholesky']

    The policy outputs action.rotation_axis_angle (3D, delta mode) or
    action.rotation_ortho6 (6D, absolute mode). This function renames the
    appropriate key to 'action.orientation'.

    Args:
        action_dict: Dict of tensors from policy.select_action().
        actions_as_deltas: Matches env.actions_as_deltas config.

    Returns:
        Dict with numpy array values ready for FDCCEnv.step().
    """
    env_action = {}

    for key, value in action_dict.items():
        if isinstance(value, torch.Tensor):
            np_value = value.squeeze(0).cpu().numpy() if value.dim() > 0 else value.cpu().numpy()
        else:
            np_value = np.array(value)

        if actions_as_deltas and key == "action.rotation_axis_angle":
            env_action["action.orientation"] = np_value
        elif not actions_as_deltas and key == "action.rotation_ortho6":
            env_action["action.orientation"] = np_value
        else:
            env_action[key] = np_value

    return env_action
