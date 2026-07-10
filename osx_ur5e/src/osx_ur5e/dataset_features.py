"""LeRobotDataset feature-dict construction from the Hydra dataset config.

Shared by the bag->LeRobot converter and any tooling that needs the dataset
schema. Feature key names are the contract with comet training/eval configs.
"""

from omegaconf import DictConfig


def build_features(cfg: DictConfig) -> dict:
    """Build a LeRobotDataset feature dict from Hydra cameras/states/actions.

    ``cfg`` is the dataset config group (cfg.dataset in the top-level config):
    ``cfg.cameras`` maps camera name -> {height, width, channels}, and
    ``cfg.states`` / ``cfg.actions`` map feature key -> shape.
    """
    features = {}

    for cam_name, cam_info in cfg.cameras.items():
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": (cam_info.height, cam_info.width, cam_info.channels),
            "names": ["height", "width", "channels"],
        }
        # Frame capture time (seconds, relative to episode start): the image's
        # ROS header stamp minus episode t0. Same clock/axis as
        # observation.frame_time, so vision aligns to state directly. Kept out
        # of the "observation.images." namespace so LeRobot does not treat it
        # as a video stream.
        features[f"observation.image_time.{cam_name}"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        }

    for key, shape in cfg.states.items():
        features[key] = {
            "dtype": "float32",
            "shape": tuple(shape),
            "names": None,
        }

    for key, shape in cfg.actions.items():
        features[key] = {
            "dtype": "float32",
            "shape": tuple(shape),
            "names": None,
        }

    # Real time of the tick, relative to episode start (seconds). LeRobot
    # labels frames as uniform 1/fps; this records the actual tick time so
    # any conversion-time trimming or source gaps stay visible.
    features["observation.frame_time"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    }

    return features
