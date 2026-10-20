"""ORB-style temporal boundary weighting and distinct inside/outside locks."""
import gtsam
import numpy as np
import pytest

from backend_test_support import config, frame
from gelslam.core.backend_factors import state_keys, velocity_key, bias_key
from gelslam.core.gtsam_pose_graph import PoseGraph


def make_graph():
    graph = PoseGraph(config(marginalization_enabled=False))
    graph.add_keyframes([frame(k, new=k == 0) for k in range(6)], 0, 6)
    return graph


def motion(graph, reference, target):
    return next(graph.at(i) for i in range(graph.size())
                if isinstance(graph.at(i), gtsam.ImuFactor)
                and graph.at(i).keys()[0] == reference
                and graph.at(i).keys()[2] == target)


def test_only_crossing_motion_factor_is_weakened_and_robust():
    graph = make_graph()
    window, values, fixed = graph._window_graph(3)
    assert fixed == set(state_keys(2))
    original = motion(graph.graph, 2, 3)
    boundary = motion(window, 2, 3)
    assert isinstance(boundary.noiseModel(), gtsam.noiseModel.Robust)
    interior = motion(window, 3, 4)
    assert not isinstance(interior.noiseModel(), gtsam.noiseModel.Robust)
    np.testing.assert_array_equal(interior.noiseModel().information(),
                                  motion(graph.graph, 3, 4).noiseModel().information())
    np.testing.assert_array_equal(boundary.preintegratedMeasurements().preintMeasCov(),
                                  original.preintegratedMeasurements().preintMeasCov())
    A, b = boundary.linearize(values).jacobian()
    full_A, full_b = original.linearize(values).jacobian()
    np.testing.assert_allclose(A, full_A * .1, rtol=1e-13, atol=1e-12)
    np.testing.assert_allclose(b, full_b * .1, rtol=1e-13, atol=1e-12)
    # Bias random walk across the boundary retains its native information.
    for i in range(window.size()):
        factor = window.at(i)
        if list(factor.keys()) == [bias_key(2), bias_key(3)]:
            original_bias = next(graph.graph.at(j) for j in range(graph.graph.size())
                if list(graph.graph.at(j).keys()) == [bias_key(2), bias_key(3)])
            np.testing.assert_array_equal(factor.noiseModel().information(),
                                          original_bias.noiseModel().information())
            break
    else:
        pytest.fail('boundary bias random-walk factor is missing')


def test_boundary_huber_cost_uses_scaled_nine_dimensional_residual():
    graph = make_graph()
    window, values, _ = graph._window_graph(3)
    original, boundary = motion(graph.graph, 2, 3), motion(window, 2, 3)
    changed = gtsam.Values(values)
    pose = values.atPose3(3).matrix()
    pose[:3, :3] = gtsam.Rot3.Expmap(np.array([.2, -.1, .4])).matrix()
    pose[:3, 3] = [.01, -.02, .03]
    changed.update(3, gtsam.Pose3(pose))
    norm = np.linalg.norm(original.whitenedError(changed)) * .1
    delta = np.sqrt(16.92)
    assert norm > delta
    assert boundary.error(changed) == pytest.approx(delta * (norm - .5 * delta), rel=1e-12)
    assert boundary.error(changed) < original.error(changed) * .01


def test_window_advance_does_not_mutate_or_repeatedly_scale_stored_imu_factors():
    graph = make_graph()
    original = motion(graph.graph, 2, 3)
    information = original.noiseModel().information().copy()
    before = graph.graph.error(graph.graph_init)
    for start in (3, 4, 3):
        window, values, _ = graph._window_graph(start)
        for i in range(window.size()):
            factor = window.at(i)
            if isinstance(factor, gtsam.ImuFactor):
                crossing = factor.keys()[0] == start - 1
                assert isinstance(factor.noiseModel(), gtsam.noiseModel.Robust) == crossing
        native = motion(graph.graph, start - 1, start)
        A, _ = motion(window, start - 1, start).linearize(values).jacobian()
        native_A, _ = native.linearize(values).jacobian()
        np.testing.assert_allclose(A, native_A * .1, rtol=1e-13, atol=1e-12)
    assert not isinstance(original.noiseModel(), gtsam.noiseModel.Robust)
    np.testing.assert_array_equal(original.noiseModel().information(), information)
    assert graph.graph.error(graph.graph_init) == before


def test_only_immediate_temporal_predecessor_has_velocity_and_bias_locked():
    graph = make_graph()
    # A non-temporal tactile endpoint exercises the outside-pose policy;
    # production loop constraints still belong exclusively to loop PGO.
    assert graph._add_tactile(0, 5)
    graph.states[0].velocity[:] = [.1, -.2, .3]
    graph.states[0].accel_bias[:] = [.01, -.02, .03]
    graph.states[0].gyro_bias[:] = [.001, -.002, .003]
    graph._put_state(0, graph.states[0])
    before = graph.states[2].copy()
    window, values, fixed = graph._window_graph(3)
    assert fixed == set(state_keys(2)) | {0}
    assert values.exists(velocity_key(0)) and velocity_key(0) not in fixed
    assert values.exists(bias_key(0)) and bias_key(0) not in fixed
    graph.solve()
    np.testing.assert_allclose(graph.states[0].velocity, 0., atol=1e-8)
    np.testing.assert_allclose(graph.states[0].accel_bias, 0., atol=1e-8)
    np.testing.assert_allclose(graph.states[0].gyro_bias, 0., atol=1e-8)
    for name in ('pose', 'velocity', 'accel_bias', 'gyro_bias'):
        np.testing.assert_array_equal(getattr(graph.states[2], name), getattr(before, name))


def test_trial_start_inside_window_is_pose_only_without_crossing_imu_factor():
    graph = PoseGraph(config(marginalization_enabled=False))
    graph.add_keyframes([frame(k, new=k in (0, 3), group=int(k >= 3))
                         for k in range(6)], 0, 6)
    window, values, fixed = graph._window_graph(3)
    assert fixed == {3}
    assert not set(state_keys(2)).intersection(values.keys())
    assert all(not isinstance(window.at(i).noiseModel(), gtsam.noiseModel.Robust)
               for i in range(window.size()))


def test_marginalized_window_does_not_use_fixed_boundary_noise():
    graph = make_graph()
    graph.pose_graph_config['marginalization_enabled'] = True
    window, _, fixed = graph._window_graph(3)
    assert not fixed
    for i in range(window.size()):
        factor = window.at(i)
        if isinstance(factor, gtsam.ImuFactor):
            assert not isinstance(factor.noiseModel(), gtsam.noiseModel.Robust)
