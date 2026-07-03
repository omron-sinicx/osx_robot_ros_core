import os
from glob import glob

from setuptools import setup

package_name = "osx_examples"


def _collect_share(directory):
    base = os.path.join("share", package_name, directory)
    return [
        (base, [path])
        for path in glob(os.path.join(directory, "**", "*"), recursive=True)
        if os.path.isfile(path)
    ]


setup(
    name=package_name,
    version="0.2.0",
    packages=[package_name],
    package_dir={"": "src"},
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        *_collect_share("config"),
        *_collect_share("launch"),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Cristian Beltran",
    maintainer_email="cristian.beltran@sinicx.com",
    description="Interactive motion-planning and direct-control examples (ROS 2 port).",
    license="BSD",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "moveit_examples = osx_examples.moveit_examples:main",
            "ur_control_examples = osx_examples.ur_control_examples:main",
        ],
    },
)
