"""Launch IMU-only dead reckoning and optionally RViz for an external bag."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config_path = DeclareLaunchArgument(
        "config_path",
        default_value=os.path.join(
            get_package_share_directory("gelslam"), "config", "config.yaml"
        ),
        description="YAML containing imu_config",
    )
    use_sim_time = DeclareLaunchArgument("use_sim_time", default_value="true")
    rviz = DeclareLaunchArgument("rviz", default_value="true")
    common = {
        "config_path": LaunchConfiguration("config_path"),
        "use_sim_time": LaunchConfiguration("use_sim_time"),
    }
    return LaunchDescription([
        config_path, use_sim_time, rviz,
        Node(package="gelslam", executable="imu_propagation_node",
             name="imu_propagation_node", output="screen", parameters=[common]),
        Node(package="rviz2", executable="rviz2", name="imu_only_rviz",
             output="screen", condition=IfCondition(LaunchConfiguration("rviz"))),
    ])
