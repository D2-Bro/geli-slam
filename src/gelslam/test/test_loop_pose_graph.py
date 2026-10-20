"""Loop measurements affect map poses, never the local navigation graph."""
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from gelslam.core.gtsam_pose_graph import PoseGraph
from gelslam.core.loop_pose_graph import LoopPoseGraph
from backend_test_support import config, frame


def translation(x):
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


def test_pose_only_loop_distributes_closure_error_without_changing_local_odometry():
    graph = LoopPoseGraph({'max_iterations': 20})
    local = [translation(x) for x in (0., .01, .02)]
    records = [SimpleNamespace(new_trial_flag=k == 0, trial_group=0) for k in range(3)]
    graph.synchronize(local, records, [(0, 1), (1, 2)])
    assert graph.add_loop(0, 2, translation(-.018))
    graph.solve()
    assert graph.factor_kinds == ['pose_anchor', 'odometry', 'odometry', 'loop']
    assert np.allclose([p[0, 3] for p in graph.map_poses()],
                       [0., .009333333333, .018666666667], atol=1e-10)
    assert np.array_equal(graph.local_poses, local)
    assert graph.last_final_cost < graph.last_initial_cost
    # The registration is a fixed metric measurement, not reprojected maps.
    assert np.allclose(graph.loops[0][2], translation(.018))
    assert np.allclose(graph.noise.sigmas(), [.01] * 3 + [.0001] * 3)


@pytest.mark.parametrize('local_vi_enabled', [False, True])
def test_disconnected_trial_loop_uses_registration_and_preserves_local_priors(
        tmp_path, local_vi_enabled):
    graph = PoseGraph(config(local_vi_enabled=local_vi_enabled))
    first, second = frame(new=True), frame(1, new=True, group=1)
    second.navigation_state.pose = translation(.15)
    second.trial_T_curr = translation(.15)
    graph.add_keyframes([first, second], 0, 2)
    size = graph.graph.size()
    local = [state.copy() for state in graph.states]
    assert graph._add_loop_pair(0, 1, translation(-.002))
    graph.remove_prior_factors([1])
    assert graph.graph.size() == size
    assert graph.factor_kinds.count('gelsight') == 0
    solutions = graph.solve(global_optimization=True)
    assert np.allclose(solutions[1], translation(.002), atol=1e-10)
    assert graph.prior_keys == ([0, 1] if local_vi_enabled else [])
    assert graph.loop_pgo.factor_kinds == ['pose_anchor', 'loop']
    for actual, expected in zip(graph.states, local):
        assert np.allclose(actual.pose, expected.pose, atol=1e-12)
        assert np.allclose(actual.velocity, expected.velocity, atol=1e-12)
    graph.save(tmp_path)
    restored = PoseGraph.load(tmp_path)
    assert np.allclose(restored.map_poses(), graph.map_poses())
    assert np.allclose(restored.loop_pgo.loops[0][2], translation(.002))
    restored.solve()
    assert restored.factor_kinds.count('gelsight') == 0
    assert restored.prior_keys == graph.prior_keys
    assert np.allclose(restored.map_poses()[1], translation(.002), atol=1e-10)


def test_map_state_rotates_world_velocity_and_keeps_sensor_biases():
    graph = PoseGraph(config())
    graph.add_keyframes([frame(new=True)], 0, 1)
    graph.raw_solutions()
    local = graph.states[0]
    local.velocity[:] = [1, 2, 3]
    local.accel_bias[:] = [.1, .2, .3]
    local.gyro_bias[:] = [.01, .02, .03]
    correction = translation(.04)
    correction[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    graph.loop_pgo.corrections[0] = correction
    published = graph.map_state(0)
    assert np.allclose(published.pose, correction @ local.pose)
    assert np.array_equal(published.velocity, [-2, 1, 3])
    assert np.array_equal(published.accel_bias, local.accel_bias)
    assert np.array_equal(published.gyro_bias, local.gyro_bias)
    assert published.timestamp == local.timestamp
    assert np.array_equal(graph.map_T_trial_group(0), correction)
    assert np.array_equal(local.pose, np.eye(4))


def test_pgo_only_direct_solve_and_reload_never_run_local_vi(monkeypatch, tmp_path):
    graph = PoseGraph(config(local_vi_enabled=False))
    frames = [frame(i, new=i == 0, z=.001 * i) for i in range(3)]
    frames[1].navigation_state.velocity[:] = [.1, .2, .3]
    frames[1].navigation_state.accel_bias[:] = [.01, .02, .03]
    frames[1].navigation_state.gyro_bias[:] = [.001, .002, .003]
    graph.add_keyframes(frames, 0, len(frames))

    def unexpected(_self):
        pytest.fail('The config flag must also guard direct and restored solve calls')

    monkeypatch.setattr(PoseGraph, '_solve_local', unexpected)
    graph.solve()
    measured = np.eye(4)
    measured[2, 3] = -.001
    assert graph._add_loop_pair(0, 2, measured)
    graph.solve(global_optimization=True)
    graph.save(tmp_path)
    restored = PoseGraph.load(tmp_path)
    assert not restored.local_vi_enabled
    assert restored.graph.size() == restored.marginal_prior.size() == 0
    assert restored.temporal_links == graph.temporal_links
    np.testing.assert_allclose(restored.map_poses(), graph.map_poses(), atol=1e-12)
    for actual, expected in zip(restored.states, graph.states):
        for field in ('pose', 'velocity', 'accel_bias', 'gyro_bias'):
            np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    restored.solve(global_optimization=True)
    assert restored.loop_pgo.last_final_cost <= restored.loop_pgo.last_initial_cost
    assert restored.graph.size() == 0


@pytest.mark.parametrize('invalid', [np.full((4, 4), np.nan), np.zeros((3, 3)),
                                     np.diag([2., 1., 1., 1.]), np.diag([-1., 1., 1., 1.])])
def test_invalid_registration_cannot_modify_either_graph(invalid):
    graph = PoseGraph(config())
    graph.add_keyframes([frame(new=True), frame(1)], 0, 2)
    count = graph.graph.size()
    assert not graph._add_loop_pair(0, 1, invalid)
    assert graph.graph.size() == count
    assert not graph.loop_pgo.loops
    assert all(kind != 'loop' for _, _, _, kind in graph.edges)


def test_legacy_direct_loop_snapshot_requires_replay(tmp_path):
    graph = PoseGraph(config())
    graph.add_keyframes([frame(new=True), frame(1)], 0, 2)
    graph.save(tmp_path)
    path = tmp_path / 'pose_graph.pkl'
    with path.open('rb') as handle:
        data = pickle.load(handle)
    data['format_version'] = 2
    data.pop('loop_pgo')
    with path.open('wb') as handle:
        pickle.dump(data, handle)
    assert len(PoseGraph.load(tmp_path).states) == 2
    data['edges'].append((0, 1, None, 'loop'))
    with path.open('wb') as handle:
        pickle.dump(data, handle)
    with pytest.raises(ValueError, match='lacks registered loop measurements'):
        PoseGraph.load(tmp_path)
