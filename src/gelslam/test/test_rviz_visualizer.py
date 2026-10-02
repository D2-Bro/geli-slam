"""Small geometry tests for the RViz marker/TF bridge."""

import numpy as np
from builtin_interfaces.msg import Time

from gelslam.nodes.rviz_visualizer_node import (
    rotation_matrix_to_quaternion,
    vertices_to_pointcloud,
)


def test_rotation_matrix_to_quaternion_for_identity_and_half_turn_x():
    """TF quaternions follow ROS's xyzw ordering."""
    assert np.allclose(rotation_matrix_to_quaternion(np.eye(3)), [0.0, 0.0, 0.0, 1.0])
    half_turn_x = np.diag([1.0, -1.0, -1.0])
    quaternion = rotation_matrix_to_quaternion(half_turn_x)
    assert np.allclose(np.abs(quaternion), [1.0, 0.0, 0.0, 0.0])


def test_vertices_to_pointcloud_serializes_xyz32():
    """Surface rendering uses one XYZ32 point for every mesh vertex."""
    vertices = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 0.5]])
    cloud = vertices_to_pointcloud(vertices, Time(sec=7, nanosec=8))
    assert cloud.header.frame_id == "gelslam_map"
    assert cloud.header.stamp.sec == 7
    assert cloud.width == 2
    assert cloud.point_step == 12
    assert np.allclose(np.frombuffer(cloud.data, dtype=np.float32).reshape(-1, 3), vertices)
