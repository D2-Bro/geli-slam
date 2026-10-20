"""Check reduced pose Jacobians and gravity preservation with tilted axes."""
from types import SimpleNamespace

import gtsam
import numpy as np
import pytest

from gelslam.core.gravity_pose import (
    align_directions, gravity_between_factor, gravity_pose, gravity_prior_factor,
    up_direction, yaw_angle,
)
from gelslam.core.loop_pose_graph import LoopPoseGraph


def numeric_jacobian(factor, values, key):
    vector = values.atVector(key)
    columns = []
    for axis, epsilon in enumerate([1e-6, 1e-7, 1e-7, 1e-7]):
        step = np.eye(4)[axis] * epsilon
        plus, minus = gtsam.Values(values), gtsam.Values(values)
        plus.update(key, vector + step)
        minus.update(key, vector - step)
        columns.append((factor.unwhitenedError(plus) - factor.unwhitenedError(minus)) / (2 * epsilon))
    return factor.noiseModel().R() @ np.array(columns).T


@pytest.mark.parametrize('up', [[0., 0., 1.], [.3, -.7, .2]])
def test_six_error_rows_have_exact_four_column_endpoint_jacobians(up):
    up = np.array(up) / np.linalg.norm(up)
    rotations = [gtsam.Rot3.Expmap(np.array(v)).matrix()
                 for v in ([.2, -.3, .1], [-.3, .1, -.2])]
    measurement = np.eye(4)
    measurement[:3, :3] = gtsam.Rot3.Expmap(np.array([.1, -.2, .4])).matrix()
    measurement[:3, 3] = [.01, -.02, .03]
    noise = gtsam.noiseModel.Diagonal.Sigmas(np.r_[np.full(3, .04), np.full(3, .002)])
    values = gtsam.Values()
    values.insert(0, np.array([.3, .02, -.01, .04]))
    values.insert(1, np.array([-.1, -.01, .03, .02]))
    factor = gravity_between_factor(0, 1, measurement, noise, rotations, up)
    analytic, _ = factor.linearize(values).jacobian()
    assert analytic.shape == (6, 8) and factor.dim() == 6
    for key in (0, 1):
        np.testing.assert_allclose(analytic[:, 4*key:4*(key+1)],
                                   numeric_jacobian(factor, values, key), atol=2e-6, rtol=1e-7)
    prior = gravity_prior_factor(0, measurement, noise, rotations[0], up)
    analytic, _ = prior.linearize(values).jacobian()
    np.testing.assert_allclose(analytic, numeric_jacobian(prior, values, 0), atol=2e-6, rtol=1e-7)


@pytest.mark.parametrize('world_rotation', [[0., 0., 0.], [.4, -.6, .1]])
def test_loop_updates_yaw_and_xyz_but_never_local_gravity_attitude(world_rotation):
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
    records = [SimpleNamespace(new_trial_flag=i == 0, trial_group=0, gravity=-9.81*up)
               for i in range(3)]
    graph = LoopPoseGraph({'max_iterations': 30})
    graph.synchronize(local, records, [(0, 1), (1, 2)])
    measured = np.linalg.inv(ideal[0]) @ ideal[2]
    # Registration also contains a tilt error, which PGO must not apply.
    measured[:3, :3] = measured[:3, :3] @ gtsam.Rot3.Expmap(np.array([.1, -.15, .02])).matrix()
    assert graph.add_loop(0, 2, np.linalg.inv(measured))
    graph.solve()
    assert graph.last_final_cost < graph.last_initial_cost
    np.testing.assert_allclose(graph.map_poses()[0], local[0], atol=1e-14)
    assert abs(yaw_angle(graph.corrections[2][:3, :3], up)) > .01
    for key, (pose, corrected) in enumerate(zip(local, graph.map_poses())):
        assert graph.graph_result.atVector(key).shape == (4,)
        np.testing.assert_allclose(corrected[:3, :3].T @ up, pose[:3, :3].T @ up, atol=1e-14)
        np.testing.assert_allclose(graph.corrections[key][:3, :3] @ up, up, atol=1e-14)
    np.testing.assert_array_equal(graph.local_poses, local)
    # A later local-VI tilt update remains visible through the saved yaw correction.
    changed = [p.copy() for p in local]
    changed[2][:3, :3] = changed[2][:3, :3] @ gtsam.Rot3.Expmap(np.array([.03, .02, 0.])).matrix()
    graph.synchronize(changed, records, [(0, 1), (1, 2)])
    np.testing.assert_allclose(graph.map_poses()[2][:3, :3].T @ up,
                               changed[2][:3, :3].T @ up, atol=1e-14)


def test_disconnected_trial_alignment_also_preserves_gravity_before_solving():
    first, second = np.eye(4), np.eye(4)
    second[:3, 3] = [.15, .02, .01]
    up = np.array([0., 0., 1.])
    other_up = np.array([0., 1., 0.])
    records = [SimpleNamespace(new_trial_flag=True, trial_group=i, gravity=-9.81*g)
               for i, g in enumerate([up, other_up])]
    graph = LoopPoseGraph({'max_iterations': 30})
    graph.synchronize([first, second], records, [])
    desired = np.eye(4)
    desired[:3, :3] = gtsam.Rot3.Expmap(np.array([.3, -.2, .4])).matrix()
    desired[:3, 3] = [.002, .003, .004]
    assert graph.add_loop(0, 1, np.linalg.inv(desired))
    graph.remove_prior_factors([1])
    before = graph.map_poses()
    np.testing.assert_allclose(before[1][:3, :3].T @ up, other_up, atol=1e-14)
    np.testing.assert_allclose(before[1][:3, 3], desired[:3, 3], atol=1e-14)
    graph.solve()
    np.testing.assert_allclose(graph.map_poses()[1][:3, :3].T @ up, other_up, atol=1e-14)
    np.testing.assert_array_equal(graph.local_poses, [first, second])


def test_historical_six_dof_corrections_are_projected_at_the_next_solve():
    local = [np.eye(4), np.eye(4)]
    local[1][0, 3] = .01
    records = [SimpleNamespace(new_trial_flag=i == 0, trial_group=0) for i in range(2)]
    graph = LoopPoseGraph({})
    graph.synchronize(local, records, [(0, 1)])
    measured = np.eye(4)
    measured[0, 3] = .008
    graph.add_loop(0, 1, np.linalg.inv(measured))
    graph.solve()
    snapshot = graph.snapshot()
    assert snapshot.pop('degrees_of_freedom') == 4
    snapshot['corrections'][1][:3, :3] = gtsam.Rot3.Expmap(np.array([.2, -.1, .3])).matrix()
    restored = LoopPoseGraph({})
    restored.synchronize(local, records, [(0, 1)])
    restored.restore(snapshot)
    assert not np.allclose(restored.map_poses()[1][:3, :3].T @ restored.up, restored.up)
    restored.solve()
    np.testing.assert_allclose(restored.map_poses()[1][:3, :3].T @ restored.up, restored.up, atol=1e-14)
    assert restored.loops and restored.snapshot()['degrees_of_freedom'] == 4


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
