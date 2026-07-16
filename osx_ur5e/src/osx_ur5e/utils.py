import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from comet.common.utils.image_transforms import pad_to_square as _pad_to_square

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


def format_real_robot_observations(
    arm,
    feeder,
    features_or_keys: dict | set[str],
    camera_shape: tuple,
    pad_to_square: bool = False,
) -> dict:
    """Build a policy-ready observation dict from the live topic feeder.

    Raw observations come from the shared ObservationAssembler - the exact
    implementation the offline bag converter uses to build training data -
    then get torch-ified into the format the COMET policy expects (float32
    states, uint8 CHW images resized to training resolution).

    Args:
        arm: CompliantController from FDCCEnv (provides the kinematics).
        feeder: RosSampleFeeder from FDCCEnv (env.image_recorder).
        features_or_keys: Feature dict from checkpoint or a set of observation key names.
        camera_shape: (H, W) to resize camera images to match training resolution.
        pad_to_square: Pad images to square before resizing, matching checkpoints
            trained with model_configs.square_crop enabled.

    Returns:
        Dict mapping observation keys to torch tensors ready for policy.select_action().
    """
    import rospy

    from osx_ur5e.observation_assembler import ObservationAssembler

    wanted_keys = _obs_key_set(features_or_keys)
    camera_names = _camera_names_from_obs_keys(wanted_keys)

    image_keys = [f"images.{cam}" for cam in camera_names]
    if image_keys and not feeder.wait_until_fresh(
            max_age_s=1.0, timeout_s=2.0, keys=image_keys):
        stale = [k for k in image_keys if feeder.age_s(k) > 1.0]
        raise RuntimeError(
            f"Camera image(s) unavailable (stale or not received yet): {stale}. "
            "Check camera topics and that streams are publishing."
        )

    assembler = ObservationAssembler(arm.kdl, camera_names)
    samples = feeder.get_latest(["joint_states", "wrench"] + image_keys)
    raw_obs = assembler.assemble_observation(samples, tick_time=rospy.get_time())

    obs = {}
    resize_transform = transforms.Resize(camera_shape, antialias=True)
    for key, value in raw_obs.items():
        if key not in wanted_keys:
            continue
        if key.startswith("observation.images."):
            image_chw = _image_hwc_to_chw(value)
            if image_chw.dtype != np.uint8:
                image_chw = np.clip(image_chw, 0, 255).astype(np.uint8)
            image = torch.tensor(image_chw, dtype=torch.uint8)
            if pad_to_square:
                image = _pad_to_square(image)
            obs[key] = resize_transform(image)
        else:
            obs[key] = torch.tensor(np.array(value).flatten(), dtype=torch.float32)

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
