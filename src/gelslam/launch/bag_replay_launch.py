"""Launch GelSLAM processing nodes for an externally played ROS 2 bag.

The bag player is intentionally not launched here.  Start this launch file
first, then play a bag that publishes tactile images on the ``/image`` topic.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    data_dir_arg = DeclareLaunchArgument(
        "data_dir", default_value="", description="Optional directory for saved state"
    )
    config_path_arg = DeclareLaunchArgument(
        "config_path",
        default_value=os.path.join(
            get_package_share_directory("gelslam"), "config", "config.yaml"
        ),
        description="Path of the GelSLAM configuration file",
    )
    rendering_arg = DeclareLaunchArgument(
        "rendering", default_value="true", description="Start the Open3D visualizer"
    )
    rviz_arg = DeclareLaunchArgument(
        "rviz", default_value="true", description="Start RViz mesh and TF visualization"
    )
    publish_coverage_arg = DeclareLaunchArgument(
        "publish_coverage",
        default_value="true",
        description="Publish pose-graph coverage meshes for a visualizer",
    )
    skip_background_check_arg = DeclareLaunchArgument(
        "skip_background_check",
        default_value="true",
        description="Accept the initial bag frames as tracker background",
    )
    save_states_arg = DeclareLaunchArgument(
        "save_states", default_value="false", description="Save tracker/backend state"
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use the /clock published by ros2 bag play",
    )

    common_parameters = {
        "config_path": LaunchConfiguration("config_path"),
        "data_dir": LaunchConfiguration("data_dir"),
        "use_sim_time": LaunchConfiguration("use_sim_time"),
    }
    tracker_node = Node(
        package="gelslam",
        executable="tracker_node",
        name="tracker_node",
        output="screen",
        parameters=[
            common_parameters,
            {
                "skip_background_check": LaunchConfiguration("skip_background_check"),
                "save_gelslam_state": LaunchConfiguration("save_states"),
            },
        ],
    )
    pose_graph_node = Node(
        package="gelslam",
        executable="pose_graph_node",
        name="pose_graph_node",
        output="screen",
        parameters=[
            common_parameters,
            {
                "rendering": LaunchConfiguration("publish_coverage"),
                "save_gelslam_states": LaunchConfiguration("save_states"),
            },
        ],
    )
    visualizer_node = Node(
        package="gelslam",
        executable="visualizer_node",
        name="visualizer_node",
        output="screen",
        parameters=[common_parameters],
        condition=IfCondition(LaunchConfiguration("rendering")),
    )
    rviz_bridge_node = Node(
        package="gelslam",
        executable="rviz_visualizer_node",
        name="rviz_visualizer_node",
        output="screen",
        parameters=[common_parameters],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=[
            "-d",
            os.path.join(
                get_package_share_directory("gelslam"), "config", "gelslam_rviz.rviz"
            ),
        ],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription(
        [
            data_dir_arg,
            config_path_arg,
            rendering_arg,
            rviz_arg,
            publish_coverage_arg,
            skip_background_check_arg,
            save_states_arg,
            use_sim_time_arg,
            tracker_node,
            pose_graph_node,
            visualizer_node,
            rviz_bridge_node,
            rviz_node,
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=tracker_node,
                    on_exit=[EmitEvent(event=Shutdown())],
                )
            ),
        ]
    )
