"""Schedule local VI from new tactile factors while retaining IMU states."""
from dataclasses import replace

import numpy as np
import pytest
from std_msgs.msg import Header

from backend_test_support import config, frame
from gelslam.core.frame import create_keyframe_msg
from gelslam.core.local_mapping import LocalMapping, LoopCandidate
from gelslam.core.shared_map import SharedMap
from gelslam.core.slam_packets import ArrayBridge
from gelslam.utils import Logger


def packet(index, tactile=True, z=0.):
    observation = frame(index, new=index == 0, z=z)
    header = Header()
    nanoseconds = round(observation.navigation_state.timestamp * 1e9)
    header.stamp.sec, header.stamp.nanosec = divmod(nanoseconds, 10**9)
    return create_keyframe_msg(
        ArrayBridge(), observation, index, index, index == 0,
        observation.trial_T_curr, header, observation.navigation_state,
        observation.gravity, observation.imu_samples, tactile_valid=tactile)


def mapper_and_store(marginalization, **overrides):
    cfg = config(marginalization_enabled=marginalization,
                 local_window_size=4, coverage_graph_overlap_threshold=.8, **overrides)
    cfg['device_config'] = {'ppmm': 1.}
    return LocalMapping(cfg, Logger(print_output=False)), SharedMap()


def process(mapper, store, packets=(), candidates=()):
    for item in packets:
        store.append_keyframe(item)
    return mapper.process(store.snapshot_for_optimization(), candidates, store)


def assert_state_equal(actual, expected):
    assert actual.timestamp == expected.timestamp
    for field in ('pose', 'velocity', 'accel_bias', 'gyro_bias'):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))


@pytest.mark.parametrize('marginalization', [False, True])
def test_recovery_and_first_contact_publish_seeds_then_tactile_pair_resumes_vi(
        monkeypatch, marginalization):
    mapper, store = mapper_and_store(marginalization)
    process(mapper, store, [packet(0), packet(1)])
    before = [state.copy() for state in mapper.graph.states]
    old_start, old_prior = mapper.graph.window_start, mapper.graph.marginal_prior
    solve = mapper.graph._solve_local

    def unexpected():
        pytest.fail('A batch without a new tactile factor must not run local VI')

    monkeypatch.setattr(mapper.graph, '_solve_local', unexpected)
    process(mapper, store, [packet(2, tactile=False, z=.002)])
    recovery = mapper.graph.states[2].copy()
    process(mapper, store, [packet(3, z=.002)])
    for actual, expected in zip(mapper.graph.states, before + [recovery]):
        assert_state_equal(actual, expected)
    assert mapper.graph.window_start == old_start
    assert mapper.graph.marginal_prior is old_prior
    assert mapper.graph.factor_kinds.count('imu') == 3
    assert mapper.graph.factor_kinds.count('bias') == 3
    assert mapper.graph.factor_kinds.count('gelsight') == 1
    assert mapper.solutions.size() == 4
    snapshot = store.snapshot_for_tracking()
    assert set(snapshot.estimates) == set(range(4))
    for index in (2, 3):
        assert_state_equal(snapshot.estimates[index].state, mapper.graph.states[index])
        assert snapshot.estimates[index].state.pose[2, 3] == pytest.approx(.002)

    calls = []

    def resume():
        calls.append(tuple(range(len(mapper.graph.states))))
        solve()

    monkeypatch.setattr(mapper.graph, '_solve_local', resume)
    process(mapper, store, [packet(4, z=.004)])
    assert calls == [(0, 1, 2, 3, 4)]
    assert mapper.graph.factor_kinds.count('gelsight') == 2
    assert abs(mapper.graph.states[2].pose[2, 3]) < 1e-6
    assert abs(mapper.graph.states[4].pose[2, 3]) < 1e-6
    assert mapper.solutions.size() == 5


@pytest.mark.parametrize('marginalization', [False, True])
def test_mixed_batch_ending_in_recovery_still_solves_new_tactile_pair(
        monkeypatch, marginalization):
    mapper, store = mapper_and_store(marginalization)
    process(mapper, store, [packet(0)])
    solve, calls = mapper.graph._solve_local, []

    def run():
        calls.append(len(mapper.graph.states))
        solve()

    monkeypatch.setattr(mapper.graph, '_solve_local', run)
    process(mapper, store, [packet(1, z=.002), packet(2, tactile=False, z=.003)])
    assert calls == [3]
    assert mapper.graph.factor_kinds.count('gelsight') == 1
    assert abs(mapper.graph.states[1].pose[2, 3]) < 1e-6
    assert mapper.solutions.size() == 3


def test_valid_tactile_flags_without_shared_points_do_not_trigger_vi(monkeypatch):
    mapper, store = mapper_and_store(False)

    def unexpected():
        pytest.fail('Initialization or failed tactile construction must not run local VI')

    monkeypatch.setattr(mapper.graph, '_solve_local', unexpected)
    process(mapper, store, [packet(0)])
    contact = packet(1)
    process(mapper, store, [replace(contact, contact_mask=np.zeros_like(contact.contact_mask))])
    assert mapper.graph.factor_kinds.count('imu') == 1
    assert 'gelsight' not in mapper.graph.factor_kinds
    assert 'shared surface points' in mapper.graph.tactile_skipped[-1][2]
    assert len(store.snapshot_for_tracking().estimates) == 2


def test_batch_with_new_tactile_and_loop_runs_both_optimizers(monkeypatch):
    mapper, store = mapper_and_store(False)
    process(mapper, store, [packet(0), packet(1)])
    local, loop, calls = mapper.graph._solve_local, mapper.graph.loop_pgo.solve, []

    def run_local():
        calls.append('local')
        local()

    def run_loop():
        calls.append('loop')
        loop()

    monkeypatch.setattr(mapper.graph, '_solve_local', run_local)
    monkeypatch.setattr(mapper.graph.loop_pgo, 'solve', run_loop)
    process(mapper, store, [packet(2)], [LoopCandidate(0, 0, 0, 2, np.eye(4))])
    assert calls == ['local', 'loop']
    assert mapper.graph.factor_kinds.count('gelsight') == 2
    assert mapper.solutions.size() == 3


@pytest.mark.parametrize('with_recovery', [False, True])
def test_late_loop_runs_only_pgo_and_preserves_local_navigation(monkeypatch, with_recovery):
    mapper, store = mapper_and_store(False)
    process(mapper, store, [packet(0), packet(1), packet(2)])
    before = [state.copy() for state in mapper.graph.states]

    def unexpected():
        pytest.fail('A loop without a new tactile factor must not run local VI')

    monkeypatch.setattr(mapper.graph, '_solve_local', unexpected)
    loop_solve, calls = mapper.graph.loop_pgo.solve, []

    def run_loop():
        calls.append(True)
        loop_solve()

    monkeypatch.setattr(mapper.graph.loop_pgo, 'solve', run_loop)
    measured = np.eye(4)
    measured[0, 3] = -.002
    candidate = LoopCandidate(0, 0, 0, 2, measured)
    pending = [packet(3, tactile=False)] if with_recovery else []
    process(mapper, store, pending, [candidate])
    assert calls == [True]
    assert (0, 2) in mapper.accepted_pairs
    for actual, expected in zip(mapper.graph.states, before):
        assert_state_equal(actual, expected)
    assert mapper.graph.loop_pgo.last_final_cost < mapper.graph.loop_pgo.last_initial_cost
    assert store.snapshot_for_tracking().estimates[2].map_pose[0, 3] > .001
    assert mapper.solutions.size() == 3 + int(with_recovery)
    process(mapper, store, candidates=[candidate])
    assert calls == [True]  # A duplicate loop does not trigger either optimizer.


@pytest.mark.parametrize('windowed', [False, True])
@pytest.mark.parametrize('loop_only', [False, True])
def test_pgo_only_skips_local_factors_and_corrects_frontend_seed_odometry(
        monkeypatch, windowed, loop_only):
    mapper, store = mapper_and_store(True, local_vi_enabled=False,
        local_window_optimization=windowed, loop_closure_only_optimization=loop_only)

    def unexpected(*_args):
        pytest.fail('PGO-only mode must not build or solve the local VI graph')

    for name in ('_add_prior', '_add_imu', '_add_tactile', '_solve_local'):
        monkeypatch.setattr(mapper.graph, name, unexpected)
    process(mapper, store, [packet(i, z=.001 * i) for i in range(3)])
    seeds = [state.copy() for state in mapper.graph.states]
    assert mapper.graph.graph.size() == 0
    assert mapper.graph.factor_kinds == []
    assert mapper.graph.temporal_links == [(0, 1), (1, 2)]
    for index, seed in enumerate(seeds):
        assert seed.pose[2, 3] == pytest.approx(.001 * index)
        np.testing.assert_allclose(mapper.solutions[index], seed.pose, atol=1e-10)

    measured = np.eye(4)
    measured[2, 3] = -.001
    candidate = LoopCandidate(0, 0, 0, 2, measured)
    process(mapper, store, candidates=[candidate])
    assert (0, 2) in mapper.accepted_pairs
    assert mapper.graph.loop_pgo.factor_kinds == ['pose_anchor', 'odometry', 'odometry', 'loop']
    assert mapper.graph.loop_pgo.last_final_cost < mapper.graph.loop_pgo.last_initial_cost
    assert .001 < mapper.solutions[2][2, 3] < .002
    for actual, expected in zip(mapper.graph.states, seeds):
        assert_state_equal(actual, expected)
    snapshot = store.snapshot_for_tracking()
    for index in range(3):
        np.testing.assert_array_equal(snapshot.estimates[index].map_pose, mapper.solutions[index])
        np.testing.assert_allclose(snapshot.estimates[index].state.pose,
                                   mapper.solutions[index], atol=1e-10)

    correction = mapper.graph.loop_pgo.corrections[2].copy()
    cost = mapper.graph.loop_pgo.last_final_cost
    process(mapper, store, [packet(3, tactile=False, z=.003)], [candidate])
    assert mapper.graph.loop_pgo.last_final_cost == cost
    assert mapper.graph.temporal_links[-1] == (2, 3)
    np.testing.assert_array_equal(mapper.graph.loop_pgo.corrections[3], correction)
    np.testing.assert_allclose(mapper.solutions[3], correction @ mapper.graph.poses[3], atol=1e-10)
    assert mapper.graph.graph.size() == mapper.graph.marginal_prior.size() == 0
    assert mapper.graph.window_start == 0
    assert mapper.solutions.size() == 4
