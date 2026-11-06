"""Launch GelSLAM with live GelSight Mini and EBIMU24GV6 inputs.

Sensor publishers:
- ``gelsight_streamer_node`` (gs_sdk FastCamera, ffmpeg) publishes
  ``device_config.image_topic``. Each stamp is the completed raw-frame read.
- ``ebimu_publisher`` (``ebimu_pkg``, separate workspace) publishes the raw
  6-axis IMU. Each stamp is the device counter mapped to host time.

Source both workspaces first::

    source ~/IMU/ros2_pkg/install/setup.bash
    source ~/gelslam/install/setup.bash

Keep the gel free of contact for the first frames (background). Static IMU
initialization uses operator rest events: ``rest_event_node`` opens the window
"GelSLAM rest events"; press SPACE there after the background is confirmed,
hold the sensor still during the countdown, then approach at GO.

T_GI follows the IMU mounting slot: ``imu_slot:=near`` or ``imu_slot:=far``
(``imu_config.gelsight_T_imu_calibration``); the tracker logs the selected entry.
"""

import datetime
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

CAMERA_BY_ID = (
    "/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._GelSight_Mini_R0B_2DD6-KVL8_"
    "2DD6KVL8-video-index0"
)
IMU_BY_ID = (
    "/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-"
    "if00-port0"
)


def generate_launch_description():
    share_dir = get_package_share_directory("gelslam")
    default_data_dir = os.path.join(
        os.path.expanduser("~/gelslam/output"),
        "live_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    arguments = [
        DeclareLaunchArgument(
            "data_dir", default_value=default_data_dir,
            description="Output directory (gelslam_online/ is created inside)",
        ),
        DeclareLaunchArgument(
            "sequence_name", default_value="",
            description="Output folder name; defaults to the data_dir name",
        ),
        DeclareLaunchArgument(
            "config_path",
            default_value=os.path.join(share_dir, "config", "live_thick_far_config.yaml"),
            description="Path of the GelSLAM configuration file",
        ),
        DeclareLaunchArgument(
            "imu_slot", default_value="",
            description="IMU mounting slot for T_GI: near or far (Inear / Ifar in "
                        "imu_config.gelsight_T_imu_calibration); empty uses its slot",
        ),
        DeclareLaunchArgument(
            "camera_device", default_value=CAMERA_BY_ID,
            description="GelSight Mini capture device (by-id path to /dev/videoN)",
        ),
        DeclareLaunchArgument(
            "imu_port", default_value=IMU_BY_ID, description="EBRCV24GV6 serial port"
        ),
        DeclareLaunchArgument(
            "imu_configure_receiver", default_value="false",
            description="Send the slam_raw receiver profile once (stored in the receiver)",
        ),
        DeclareLaunchArgument(
            "imu_output_interval_ms", default_value="5",
            description="Receiver <sorN> interval used with imu_configure_receiver (5 = 200 Hz)",
        ),
        DeclareLaunchArgument(
            "imu_frame_id", default_value="ebimu_sensor",
            description="Must match imu_config.frame_id in the configuration file",
        ),
        DeclareLaunchArgument(
            "imu_topic", default_value="/imu/data_raw",
            description="Must match imu_config.topic in the configuration file",
        ),
        DeclareLaunchArgument(
            "slam", default_value="true",
            description="Start the GelSLAM nodes; false starts only the sensor publishers",
        ),
        DeclareLaunchArgument(
            "rendering", default_value="true", description="Start the Open3D visualizer"
        ),
        DeclareLaunchArgument(
            "backend", default_value="true",
            description="Enable loop closure and pose-graph optimization",
        ),
        DeclareLaunchArgument(
            "backend_feedback", default_value=LaunchConfiguration("backend"),
            description="Send backend corrections to tracking",
        ),
        DeclareLaunchArgument(
            "rviz", default_value="true", description="Start RViz mesh and TF visualization"
        ),
        DeclareLaunchArgument(
            "publish_coverage", default_value="true",
            description="Publish pose-graph coverage meshes for a visualizer",
        ),
        DeclareLaunchArgument(
            "skip_background_check", default_value="false",
            description="Accept the first frames as background without the confirmation window",
        ),
        DeclareLaunchArgument(
            "rest_events", default_value="true",
            description="Start rest_event_node (operator window that publishes the rest/GO events "
                        "for static_initialization mode: recorded_rest_event)",
        ),
        DeclareLaunchArgument(
            "rest_duration_s", default_value="5.0",
            description="Rest duration published by rest_event_node",
        ),
        DeclareLaunchArgument(
            "save_video", default_value="false",
            description="Also write the published tactile frames to data_dir/gelsight.avi",
        ),
        DeclareLaunchArgument(
            "save_states", default_value="false", description="Save tracker/backend state"
        ),
    ]

    slam = IfCondition(LaunchConfiguration("slam"))

    def slam_and(name):
        return IfCondition(PythonExpression([
            "'", LaunchConfiguration("slam"), "'.lower() == 'true' and '",
            LaunchConfiguration(name), "'.lower() == 'true'",
        ]))

    common_parameters = {
        "config_path": LaunchConfiguration("config_path"),
        "data_dir": LaunchConfiguration("data_dir"),
        "use_sim_time": False,
    }
    # Nodes that read T_GI from the configuration resolve the same IMU slot.
    slot_parameters = {
        "imu_slot": ParameterValue(LaunchConfiguration("imu_slot"), value_type=str),
    }
    imu_node = Node(
        package="ebimu_pkg",
        executable="ebimu_publisher",
        name="ebimu_publisher",
        output="screen",
        parameters=[
            {
                "port": LaunchConfiguration("imu_port"),
                "baudrate": 921600,
                "sensor_id": 0,
                "profile": "slam_raw",
                "configure_receiver": LaunchConfiguration("imu_configure_receiver"),
                "output_rate": LaunchConfiguration("imu_output_interval_ms"),
                "imu_topic": LaunchConfiguration("imu_topic"),
                "frame_id": LaunchConfiguration("imu_frame_id"),
                "gravity_mps2": 9.80665,
                "stamp_source": "device_clock",
            }
        ],
    )
    camera_node = Node(
        package="gelslam",
        executable="gelsight_streamer_node",
        name="gelsight_streamer_node",
        output="screen",
        parameters=[
            common_parameters,
            {
                "streamer": "ffmpeg",
                "device_path": LaunchConfiguration("camera_device"),
                # Same topic as device_config.image_topic in live_thick_far_config.yaml.
                "image_topic": "/gelsight/image_raw",
                "frame_id": "gel_frame",
                "save_video": LaunchConfiguration("save_video"),
            },
        ],
    )
    tracker_node = Node(
        package="gelslam",
        executable="tracker_node",
        name="tracker_node",
        output="screen",
        parameters=[
            common_parameters,
            slot_parameters,
            {
                "skip_background_check": LaunchConfiguration("skip_background_check"),
                "backend_feedback": LaunchConfiguration("backend_feedback"),
                "save_gelslam_state": LaunchConfiguration("save_states"),
                "sequence_name": LaunchConfiguration("sequence_name"),
            },
        ],
        condition=slam,
    )
    pose_graph_node = Node(
        sigterm_timeout="120",
        sigkill_timeout="30",
        package="gelslam",
        executable="pose_graph_node",
        name="pose_graph_node",
        output="screen",
        parameters=[
            common_parameters,
            slot_parameters,
            {
                "rendering": LaunchConfiguration("publish_coverage"),
                "save_gelslam_states": LaunchConfiguration("save_states"),
                "sequence_name": LaunchConfiguration("sequence_name"),
                "backend_optimization": LaunchConfiguration("backend"),
                "backend_feedback": LaunchConfiguration("backend_feedback"),
            },
        ],
        condition=slam,
    )
    visualizer_node = Node(
        package="gelslam",
        executable="visualizer_node",
        name="visualizer_node",
        output="screen",
        parameters=[common_parameters, slot_parameters,
                    {"use_backend_poses": LaunchConfiguration("backend")}],
        condition=slam_and("rendering"),
    )
    rviz_bridge_node = Node(
        package="gelslam",
        executable="rviz_visualizer_node",
        name="rviz_visualizer_node",
        output="screen",
        parameters=[common_parameters, slot_parameters,
                    {"use_backend_poses": LaunchConfiguration("backend")}],
        condition=slam_and("rviz"),
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        parameters=[{"use_sim_time": False}],
        arguments=["-d", os.path.join(share_dir, "config", "gelslam_rviz.rviz")],
        condition=slam_and("rviz"),
    )

    rest_event_node = Node(
        package="gelslam",
        executable="rest_event_node",
        name="rest_event_node",
        output="screen",
        parameters=[{
            "config_path": LaunchConfiguration("config_path"),
            "imu_slot": ParameterValue(LaunchConfiguration("imu_slot"), value_type=str),
            "rest_duration_s": ParameterValue(LaunchConfiguration("rest_duration_s"), value_type=float),
        }],
        condition=slam_and("rest_events"),
    )

    def shutdown_on_exit(action, reason):
        return RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=action, on_exit=[EmitEvent(event=Shutdown(reason=reason))]
            )
        )

    return LaunchDescription(
        arguments
        + [
            tracker_node,
            pose_graph_node,
            visualizer_node,
            rviz_bridge_node,
            rviz_node,
            rest_event_node,
            imu_node,
            camera_node,
            shutdown_on_exit(pose_graph_node, "Backend process stopped"),
            shutdown_on_exit(tracker_node, "Tracker process stopped"),
            # Without a sensor the tracker cannot continue; stop and save the outputs.
            shutdown_on_exit(imu_node, "IMU publisher stopped"),
            shutdown_on_exit(camera_node, "Camera publisher stopped"),
        ]
    )
