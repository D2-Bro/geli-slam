"""Small geometry tests for the RViz marker/TF bridge."""

import numpy as np

from gelslam.nodes.rviz_visualizer_node import rotation_matrix_to_quaternion


def test_rotation_matrix_to_quaternion_for_identity_and_half_turn_x():
    """TF quaternions follow ROS's xyzw ordering."""
    assert np.allclose(rotation_matrix_to_quaternion(np.eye(3)), [0.0, 0.0, 0.0, 1.0])
    half_turn_x = np.diag([1.0, -1.0, -1.0])
    quaternion = rotation_matrix_to_quaternion(half_turn_x)
    assert np.allclose(np.abs(quaternion), [1.0, 0.0, 0.0, 0.0])
