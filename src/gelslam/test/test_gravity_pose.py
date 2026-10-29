"""Check reduced pose Jacobians and gravity preservation with tilted axes."""
import gtsam
import numpy as np
import pytest

from gelslam.core.gravity_pose import (
    align_directions, up_direction,
)
from gelslam.core.loop_pose_graph import LoopPoseGraph
from backend_test_support import loop_records


@pytest.mark.parametrize('world_rotation', [[0., 0., 0.], [.4, -.6, .1]])
def test_global_navigation_loop_can_update_roll_pitch_in_tilted_world_axes(world_rotation):
    world = gtsam.Rot3.Expmap(np.array(world_rotation)).matrix()
    up = world[:, 2]
    local, ideal = [], []
    for index in range(3):
        pose = np.eye(4)
        tilt = gtsam.Rot3.Expmap(np.array([.2, -.3, 0.])).matrix()
        pose[:3, :3] = world @ gtsam.Rot3.Expmap(np.array([0., 0., .1*index])).matrix() @ tilt
        pose[:3, 3] = world @ np.array([.01*index, 0., .002*index])
        local.append(pose)
        truth = pose.copy()
        truth[:3, :3] = world @ tilt
        truth[:3, 3] *= .9
        ideal.append(truth)
    records = loop_records(local, [-9.81*up] * 3)
    graph = LoopPoseGraph({'max_iterations': 30})
    graph.synchronize(local, records, [(0, 1), (1, 2)])
    measured = np.linalg.inv(ideal[0]) @ ideal[2]
    # Full rotation measurements now constrain roll and pitch as well as yaw.
    measured[:3, :3] = measured[:3, :3] @ gtsam.Rot3.Expmap(np.array([.1, -.15, .02])).matrix()
    assert graph.add_loop(0, 2, np.linalg.inv(measured))
    graph.solve()
    assert graph.last_final_cost < graph.last_initial_cost
    np.testing.assert_allclose(graph.map_poses()[0], local[0], atol=1e-14)
    assert np.linalg.norm(graph.corrections[2][:3, :3] @ up - up) > 1e-4
    for key, (pose, corrected) in enumerate(zip(local, graph.map_poses())):
        assert graph.graph_result.atPose3(key).matrix().shape == (4, 4)
    np.testing.assert_array_equal(graph.local_poses, local)
    # Later local VI updates retain the last full global correction.
    changed = [p.copy() for p in local]
    changed[2][:3, :3] = changed[2][:3, :3] @ gtsam.Rot3.Expmap(np.array([.03, .02, 0.])).matrix()
    graph.synchronize(changed, records, [(0, 1), (1, 2)])
    np.testing.assert_allclose(graph.map_poses()[2], graph.corrections[2] @ changed[2], atol=1e-14)


def test_disconnected_trial_loop_can_align_all_six_pose_coordinates():
    first, second = np.eye(4), np.eye(4)
    second[:3, 3] = [.15, .02, .01]
    up = np.array([0., 0., 1.])
    other_up = np.array([0., 1., 0.])
    records = loop_records([first, second], [-9.81*up, -9.81*other_up], [0, 1])
    graph = LoopPoseGraph({'max_iterations': 30})
    graph.synchronize([first, second], records, [])
    desired = np.eye(4)
    desired[:3, :3] = gtsam.Rot3.Expmap(np.array([.3, -.2, .4])).matrix()
    desired[:3, 3] = [.002, .003, .004]
    assert graph.add_loop(0, 1, np.linalg.inv(desired))
    graph.remove_prior_factors([1])
    before = graph.map_poses()
    np.testing.assert_allclose(before[1], desired, atol=1e-14)
    graph.solve()
    np.testing.assert_allclose(graph.map_poses()[1], desired, atol=1e-14)
    np.testing.assert_array_equal(graph.local_poses, [first, second])


def test_historical_pose_corrections_initialize_a_full_navigation_solve():
    local = [np.eye(4), np.eye(4)]
    local[1][0, 3] = .01
    records = loop_records(local)
    graph = LoopPoseGraph({})
    graph.synchronize(local, records, [(0, 1)])
    measured = np.eye(4)
    measured[0, 3] = .008
    graph.add_loop(0, 1, np.linalg.inv(measured))
    graph.solve()
    snapshot = graph.snapshot()
    assert snapshot.pop('degrees_of_freedom') == 15
    snapshot['corrections'][1][:3, :3] = gtsam.Rot3.Expmap(np.array([.2, -.1, .3])).matrix()
    restored = LoopPoseGraph({})
    restored.synchronize(local, records, [(0, 1)])
    restored.restore(snapshot)
    assert not np.allclose(restored.map_poses()[1][:3, :3].T @ restored.up, restored.up)
    restored.solve()
    assert restored.graph_result.dim() == 30
    assert restored.last_final_cost < restored.last_initial_cost
    assert restored.loops and restored.snapshot()['degrees_of_freedom'] == 15


@pytest.mark.parametrize('target', [[0., 0., 1.], [0., 0., -1.], [.2, .3, -.7]])
def test_gravity_axis_alignment_handles_parallel_opposite_and_tilted_directions(target):
    source = np.array([0., 0., 1.])
    target = np.array(target) / np.linalg.norm(target)
    rotation = align_directions(source, target)
    np.testing.assert_allclose(rotation @ source, target, atol=1e-14)
    assert np.linalg.det(rotation) == pytest.approx(1.)


@pytest.mark.parametrize('gravity', [[0., 0., 0.], [0., np.nan, -1.], [0., -1.]])
def test_invalid_gravity_is_rejected(gravity):
    with pytest.raises(ValueError, match='nonzero gravity'):
        up_direction(gravity)
