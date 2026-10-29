from setuptools import find_packages, setup

package_name = "gelslam"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/bag_replay_launch.py"]),
        ("share/" + package_name + "/launch", ["launch/imu_propagation_launch.py"]),
        ("share/" + package_name + "/launch", ["launch/live_replay_demo_launch.py"]),
        ("share/" + package_name + "/launch", ["launch/live_demo_launch.py"]),
        (
            "share/" + package_name + "/config",
            [
                "config/config.yaml",
                "config/bag_simulation_config.yaml",
                "config/figure8_imu_refined_experiment.yaml",
                "config/real_config.yaml",
                "config/real_far_config.yaml",
                "config/real_objects_config.yaml",
                "config/gelslam_rviz.rviz",
            ],
        ),
        (
            "share/" + package_name + "/resources/example_calibration/model",
            ["resources/example_calibration/model/nnmodel.pth"],
        ),
    ],
    install_requires=["setuptools", "gtsam>=4.3.0,<5"],
    zip_safe=True,
    maintainer="rpl",
    maintainer_email="hungjuih@andrew.cmu.edu",
    description="Gelslam Algorithm",
    license="MIT License",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "saved_video_streamer_node = gelslam.nodes.saved_video_streamer_node:main",
            "gelsight_streamer_node = gelslam.nodes.gelsight_streamer_node:main",
            "gelslam_node = gelslam.nodes.gelslam_node:main",
            "tracker_node = gelslam.nodes.tracker_node:main",
            "pose_graph_node = gelslam.nodes.pose_graph_node:main",
            "visualizer_node = gelslam.nodes.visualizer_node:main",
            "rviz_visualizer_node = gelslam.nodes.rviz_visualizer_node:main",
            "imu_propagation_node = gelslam.nodes.imu_propagation_node:main",
            "gelslam_offline = gelslam.scripts.gelslam_offline:main",
            "visualize_pose_graph = gelslam.scripts.visualize_pose_graph:main",
        ],
    },
)
