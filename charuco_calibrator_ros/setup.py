import os
from glob import glob

from setuptools import setup

package_name = "charuco_calibrator_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="TODO",
    maintainer_email="todo@todo.com",
    description="ROS 2 wrapper for the ChArUco camera calibrator",
    license="MIT",
    entry_points={
        "console_scripts": [
            "charuco_calibrator_node = charuco_calibrator_ros.calibrator_node:main",
        ],
    },
)
