import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from ur_control import transformations

logger = logging.getLogger(__name__)


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

def _obs_key_set(features_or_keys: dict | set[str]) -> set[str]:
    if isinstance(features_or_keys, dict):
        return set(features_or_keys.keys())
    return set(features_or_keys)


def _camera_names_from_obs_keys(wanted_keys: set[str]) -> list[str]:
    prefix = "observation.images."
    return [key[len(prefix):] for key in wanted_keys if key.startswith(prefix)]


def _image_hwc_to_chw(image: np.ndarray) -> np.ndarray:
    """Convert HWC (or CHW) uint8/float image array to CHW contiguous uint8 layout."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image array, got shape {arr.shape}")

    if arr.shape[-1] in (1, 3, 4):
        return np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))
    if arr.shape[0] in (1, 3, 4):
        return np.ascontiguousarray(arr)
    raise ValueError(f"Cannot infer HWC/CHW layout for image shape {arr.shape}")


def _get_fresh_camera_images(image_recorder, camera_names: list[str], max_wait_s: float = 2.0):
    """Return camera images, retrying briefly when frames are stale or missing."""
    if not camera_names:
        return {}

    if hasattr(image_recorder, "wait_for_fresh_images"):
        last_images = image_recorder.wait_for_fresh_images(
            camera_names=camera_names,
            timeout_s=max_wait_s,
        )
    else:
        import rospy

        deadline = rospy.get_time() + max_wait_s
        last_images = {}
        while rospy.get_time() <= deadline and not rospy.is_shutdown():
            last_images = image_recorder.get_images()
            missing = [
                cam_name
                for cam_name in camera_names
                if last_images.get(cam_name) is None
            ]
            if not missing:
                return last_images
            rospy.sleep(0.05)

    missing = [cam for cam in camera_names if last_images.get(cam) is None]
    if missing:
        diagnostics = (
            image_recorder.get_diagnostics()
            if hasattr(image_recorder, "get_diagnostics")
            else {}
        )
        details = ", ".join(
            f"{cam}={diagnostics.get(cam, {})}" for cam in missing
        )
        raise RuntimeError(
            "Camera image(s) unavailable (stale or not received yet): "
            f"{missing}. Check camera topics and that streams are publishing. "
            f"Diagnostics: {details}"
        )
    return last_images


def format_real_robot_observations(
    arm,
    image_recorder,
    features_or_keys: dict | set[str],
    camera_shape: tuple,
) -> dict:
    """Build a policy-ready observation dict from the real robot arm and cameras.

    Mirrors get_observations() in data_collection.py but returns torch tensors
    in the same format expected by the COMET policy (float32 states, uint8 images).

    Args:
        arm: CompliantController instance from FDCCEnv.
        image_recorder: ImageRecorder instance from FDCCEnv.
        features_or_keys: Feature dict from checkpoint or a set of observation key names.
        camera_shape: (H, W) to resize camera images to match training resolution.

    Returns:
        Dict mapping observation keys to torch tensors ready for policy.select_action().
    """
    wanted_keys = _obs_key_set(features_or_keys)

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

    for key, value in raw_obs.items():
        if key in wanted_keys:
            obs[key] = torch.tensor(np.array(value).flatten(), dtype=torch.float32)

    if image_recorder is not None:
        resize_transform = transforms.Resize(camera_shape, antialias=True)
        camera_names = _camera_names_from_obs_keys(wanted_keys)
        raw_images = _get_fresh_camera_images(image_recorder, camera_names)
        for cam_name, image in raw_images.items():
            feat_key = f"observation.images.{cam_name}"
            if feat_key not in wanted_keys:
                continue
            if image is None:
                raise RuntimeError(f"Camera '{cam_name}' returned no image after wait.")
            image_chw = _image_hwc_to_chw(image)
            if image_chw.dtype != np.uint8:
                image_chw = np.clip(image_chw, 0, 255).astype(np.uint8)
            image_tensor = torch.tensor(image_chw, dtype=torch.uint8)
            obs[feat_key] = resize_transform(image_tensor)

    missing = wanted_keys - set(obs.keys())
    if missing:
        raise KeyError(f"Missing required observation keys: {sorted(missing)}")

    return obs


def tensor_dict_to_numpy(action_dict: dict) -> dict:
    """Convert policy output tensors to numpy arrays (baseline / factored actions)."""
    env_action = {}
    for key, value in action_dict.items():
        if isinstance(value, torch.Tensor):
            env_action[key] = value.squeeze(0).cpu().numpy() if value.dim() > 0 else value.cpu().numpy()
        else:
            env_action[key] = np.array(value)
    return env_action


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
