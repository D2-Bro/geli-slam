"""Physical observability when a local window has no tactile observations."""
import gtsam
import numpy as np
import pytest

from backend_test_support import config, frame
from gelslam.core.backend_factors import velocity_key, variable_width, state_keys
from gelslam.core.gtsam_pose_graph import PoseGraph


@pytest.mark.parametrize('marginalization', [False, True])
def test_imu_only_window_requires_navigation_information_at_its_boundary(marginalization):
    graph = PoseGraph(config(marginalization_enabled=marginalization))
    graph.add_keyframes([frame(k, new=k == 0, tactile=False) for k in range(6)], 0, 6)
    window, values, fixed = graph._window_graph(3)
    ordering = gtsam.Ordering()
    columns, offset = [], 0
    for key in values.keys():
        ordering.push_back(key)
        width = variable_width(key)
        if key not in fixed:
            columns.extend(range(offset, offset + width))
        offset += width
    A, _ = window.linearize(values).jacobian(ordering)
    A = A[:, columns]
    A /= np.maximum(np.linalg.norm(A, axis=0), 1e-15)
    rank = np.linalg.matrix_rank(A, tol=1e-9)
    assert A.shape[1] - rank == 0

    # The former velocity gauge is now constrained by either the 15D fixed
    # predecessor or the marginal prior. Perturb only free variables.
    shifted = gtsam.Values(values)
    delta_velocity = np.array([.01, -.02, .03])
    boundary = 3 if marginalization else 2
    anchor_time = graph.states[boundary].timestamp
    for k in range(boundary, 6):
        pose = values.atPose3(k).matrix()
        pose[:3, 3] += delta_velocity * (graph.states[k].timestamp - anchor_time)
        shifted.update(k, gtsam.Pose3(pose))
        if velocity_key(k) not in fixed:
            shifted.update(velocity_key(k), values.atVector(velocity_key(k)) + delta_velocity)
    assert np.linalg.norm(shifted.atPose3(5).translation()) > .005
    assert window.error(shifted) > 1e-3
    if not marginalization:
        assert fixed == set(state_keys(2))
