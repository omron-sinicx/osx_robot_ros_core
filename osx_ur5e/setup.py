import os
from glob import glob

from setuptools import setup

package_name = "osx_ur5e"


def _collect_share(directory):
    base = os.path.join("share", package_name, directory)
    return [
        (base, [path])
        for path in glob(os.path.join(directory, "**", "*"), recursive=True)
        if os.path.isfile(path)
    ]


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        *_collect_share("config"),
        *_collect_share("launch"),
        *_collect_share("urdf"),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Cristian Beltran",
    maintainer_email="cristian.beltran@sinicx.com",
    description="Single UR5e configuration and FDCC environment library (ROS 2 port).",
    license="BSD",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "replay_episode = osx_ur5e.replay_episode:main",
            "evaluate_policy = osx_ur5e.evaluate_policy:main",
            "data_collection = osx_ur5e.data_collection:main",
        ],
    },
)
