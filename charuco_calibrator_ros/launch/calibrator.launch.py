"""Launch file for the ChArUco calibrator ROS node."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("charuco_calibrator_ros")
    default_params = os.path.join(pkg_share, "config", "default_params.yaml")

    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params,
            description="Path to the ROS 2 parameters YAML file",
        ),
        DeclareLaunchArgument(
            "image_topic",
            default_value="/camera/image_raw",
            description="Image topic to subscribe to",
        ),
        Node(
            package="charuco_calibrator_ros",
            executable="charuco_calibrator_node",
            name="charuco_calibrator",
            parameters=[LaunchConfiguration("params_file")],
            remappings=[
                ("image_raw", LaunchConfiguration("image_topic")),
            ],
            output="screen",
        ),
    ])
