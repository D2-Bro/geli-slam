"""Schedule global loop PGO while retaining keyframe navigation seeds."""
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


def mapper_and_store(**overrides):
    cfg = config(coverage_graph_overlap_threshold=.8, **overrides)
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


@pytest.mark.parametrize('with_recovery', [False, True])
def test_late_loop_runs_only_pgo_and_preserves_local_navigation(monkeypatch, with_recovery):
    mapper, store = mapper_and_store()
    process(mapper, store, [packet(0), packet(1), packet(2)])
    before = [state.copy() for state in mapper.graph.states]
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
    assert calls == [True]  # A duplicate loop does not trigger the optimizer.


def test_pgo_corrects_frontend_seed_odometry_without_changing_seeds(monkeypatch):
    mapper, store = mapper_and_store()
    loop_solve, calls = mapper.graph.loop_pgo.solve, []

    def run_loop():
        calls.append(True)
        loop_solve()

    monkeypatch.setattr(mapper.graph.loop_pgo, 'solve', run_loop)
    process(mapper, store, [packet(i, z=.001 * i) for i in range(3)])
    assert calls == []  # Keyframes without a loop do not run an optimizer.
    seeds = [state.copy() for state in mapper.graph.states]
    assert mapper.graph.temporal_links == [(0, 1), (1, 2)]
    for index, seed in enumerate(seeds):
        assert seed.pose[2, 3] == pytest.approx(.001 * index)
        np.testing.assert_allclose(mapper.solutions[index], seed.pose, atol=1e-10)

    measured = np.eye(4)
    measured[2, 3] = -.001
    candidate = LoopCandidate(0, 0, 0, 2, measured)
    process(mapper, store, candidates=[candidate])
    assert (0, 2) in mapper.accepted_pairs
    assert mapper.graph.loop_pgo.factor_kinds == ['pose_anchor', 'velocity_prior', 'bias_prior',
        'imu', 'bias', 'odometry', 'imu', 'bias', 'odometry', 'loop']
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
    assert mapper.solutions.size() == 4
