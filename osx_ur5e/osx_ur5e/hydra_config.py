"""Resolve Hydra config directories for source and installed layouts."""

import os
from pathlib import Path


def get_hydra_config_dir(config_subdir="hydra") -> str:
    """Return absolute path to ``config/<config_subdir>`` for this package."""
    try:
        from ament_index_python.packages import get_package_share_directory

        return os.path.join(get_package_share_directory("osx_ur5e"), "config", config_subdir)
    except Exception:
        return str(Path(__file__).resolve().parents[2] / "config" / config_subdir)
