"""Finite increments and factor derivatives in the ORB split pose chart."""

import gtsam
import numpy as np
import pytest

from gelslam.core.gtsam_vi import (
    GtsamVisualInertialSolver, _imu_factor, _imu_values, _split_imu_factor,
    _navigation_plus, navigation_factor_terms,
)
from gelslam.core.inertial import ImuSample
from gelslam.core.numpy_vi import NavigationState
from gelslam.core.split_pose import (
    optimize_split_pose_graph, split_between_factor,
    split_pose_coordinate_jacobian, split_pose_coordinates, split_pose_plus,
    split_prior_factor,
)


def _pose(rotation, position):
    return gtsam.Pose3(gtsam.Rot3.Expmap(np.asarray(rotation)), np.asarray(position)).matrix()


def test_fixed_pose_velocity_and_bias_are_removed_from_qr_and_remain_exact(monkeypatch):
    from gelslam.core import split_pose
    from gelslam.core.backend_factors import bias_key, velocity_key, variable_type
    values = gtsam.Values()
    for index in (0, 1):
        values.insert(index, gtsam.Pose3(_pose((.1, -.2, .3), (.02, -.01, .03))))
        values.insert(velocity_key(index), np.array([.1, -.2, .3]))
        values.insert(bias_key(index), gtsam.imuBias.ConstantBias(np.ones(3) * .01, np.ones(3) * .001))
    fixed = {0, velocity_key(0), bias_key(0)}
    measurement = _pose((.01, .02, -.03), (.001, -.002, .003))
    velocity_step, bias_step = np.array([.01, -.02, .03]), np.arange(6) * .001

    def difference(first, second, step, read):
        def error(_factor, current, jacobians):
            if jacobians is not None:
                jacobians[0], jacobians[1] = -np.eye(len(step)), np.eye(len(step))
            return read(current, second) - read(current, first) - step
        return gtsam.CustomFactor(gtsam.noiseModel.Unit.Create(len(step)), [first, second], error)

    graph = gtsam.NonlinearFactorGraph()
    graph.add(split_between_factor(0, 1, measurement, gtsam.noiseModel.Unit.Create(6)))
    graph.add(difference(velocity_key(0), velocity_key(1), velocity_step,
                         lambda current, key: current.atVector(key)))
    graph.add(difference(bias_key(0), bias_key(1), bias_step,
                         lambda current, key: current.atConstantBias(key).vector()))
    coupling = gtsam.NonlinearFactorGraph(graph)
    # Strong conflicting fixed-only observations must never move the fixed state.
    strong = gtsam.noiseModel.Isotropic.Sigma
    offset = values.atPose3(0).matrix().copy()
    offset[:3, 3] += 1e-4
    graph.add(split_prior_factor(0, offset, strong(6, 1e-4)))
    graph.add(gtsam.PriorFactorVector(
        velocity_key(0), values.atVector(velocity_key(0)) + 1e-4, strong(3, 1e-4)))
    graph.add(gtsam.PriorFactorConstantBias(bias_key(0), gtsam.imuBias.ConstantBias(
        values.atConstantBias(bias_key(0)).vector()[:3] + 1e-4,
        values.atConstantBias(bias_key(0)).vector()[3:] + 1e-4), strong(6, 1e-4)))
    original_qr = split_pose.solve_gaussian_qr
    calls = []

    def check_qr(gaussian):
        assert not fixed.intersection(gaussian.keys())
        calls.append(True)
        return original_qr(gaussian)

    monkeypatch.setattr(split_pose, 'solve_gaussian_qr', check_qr)
    result = optimize_split_pose_graph(
        graph, values, fixed_keys=fixed,
        variable_types={key: variable_type(key) for key in values.keys()})
    assert calls
    np.testing.assert_array_equal(result.atPose3(0).matrix(), values.atPose3(0).matrix())
    np.testing.assert_array_equal(result.atVector(velocity_key(0)), values.atVector(velocity_key(0)))
    np.testing.assert_array_equal(result.atConstantBias(bias_key(0)).vector(),
                                  values.atConstantBias(bias_key(0)).vector())
    assert coupling.error(values) > 1e-6
    assert coupling.error(result) < 1e-12


def test_conditioning_supports_factors_with_more_than_three_free_keys():
    values = gtsam.Values()
    values.insert(0, np.array([.1, -.2, .3]))
    for key in range(1, 5):
        values.insert(key, np.zeros(3))
    target = np.arange(12) * .01

    def error(_factor, current, jacobians):
        if jacobians is not None:
            jacobians[0] = -np.tile(np.eye(3), (4, 1))
            for slot in range(1, 5):
                jacobians[slot] = np.eye(12)[:, (slot - 1) * 3:slot * 3]
        return (np.concatenate([current.atVector(key) for key in range(1, 5)])
                - np.tile(current.atVector(0), 4) - target)

    graph = gtsam.NonlinearFactorGraph()
    graph.add(gtsam.CustomFactor(gtsam.noiseModel.Unit.Create(12), list(range(5)), error))
    result = optimize_split_pose_graph(graph, values, fixed_keys={0},
                                       variable_types={key: 'vector' for key in range(5)})
    np.testing.assert_array_equal(result.atVector(0), values.atVector(0))
    np.testing.assert_allclose(np.concatenate([result.atVector(key) for key in range(1, 5)]),
                               np.tile(values.atVector(0), 4) + target, atol=1e-8)


def test_all_fixed_problem_skips_qr_and_rejects_unknown_fixed_key(monkeypatch):
    from gelslam.core import split_pose
    values = gtsam.Values()
    values.insert(0, np.ones(3))
    graph = gtsam.NonlinearFactorGraph()
    graph.add(gtsam.PriorFactorVector(0, np.zeros(3), gtsam.noiseModel.Unit.Create(3)))

    def unexpected(_graph):
        raise AssertionError('all-fixed problem must not run QR')

    monkeypatch.setattr(split_pose, 'solve_gaussian_qr', unexpected)
    result = optimize_split_pose_graph(graph, values, fixed_keys={0}, variable_types={0: 'vector'})
    np.testing.assert_array_equal(result.atVector(0), values.atVector(0))
    with pytest.raises(ValueError, match='fixed keys must exist'):
        optimize_split_pose_graph(graph, values, fixed_keys={1}, variable_types={0: 'vector'})


def test_finite_rotation_does_not_redirect_the_position_increment():
    pose = _pose((0.2, -0.1, 0.3), (0.04, -0.02, 0.01))
    rotation = np.array((0., 0., np.pi / 2))
    position = np.array((0.01, 0., 0.))
    result = split_pose_plus(pose, np.r_[rotation, position])
    expected_position = pose[:3, 3] + pose[:3, :3] @ position
    assert np.allclose(result[:3, 3], expected_position)
    assert np.allclose(result[:3, :3], pose[:3, :3] @ gtsam.Rot3.Expmap(rotation).matrix())
    assert np.linalg.norm(result[:3, 3] - gtsam.Pose3(pose).retract(
        np.r_[rotation, position]
    ).translation()) > 0.007
    # The current rotation increment cannot change a fixed position increment.
    other_rotation = split_pose_plus(pose, np.r_[-rotation, position])
    assert np.allclose(result[:3, 3], other_rotation[:3, 3])


def test_position_error_is_independent_of_current_rotation_error():
    anchor = _pose((0.2, -0.1, 0.3), (0.04, -0.02, 0.01))
    first = split_pose_plus(anchor, np.array((0., 0., 1.2, 0.01, -0.02, 0.03)))
    second = first.copy()
    second[:3, :3] = anchor[:3, :3] @ gtsam.Rot3.Expmap(np.array((0.5, 0., 0.))).matrix()
    e1, e2 = split_pose_coordinates(anchor, first), split_pose_coordinates(anchor, second)
    assert np.allclose(e1[3:], (0.01, -0.02, 0.03))
    assert np.allclose(e1[3:], e2[3:])
    assert not np.allclose(e1[:3], e2[:3])
    derivative = split_pose_coordinate_jacobian(anchor, first)
    assert np.all(derivative[:3, 3:] == 0)
    assert np.all(derivative[3:, :3] == 0)


def test_navigation_update_uses_local_position_and_trial_velocity_axes():
    pose = _pose((0.1, -0.2, 0.3), (0.04, 0.02, -0.01))
    state = NavigationState(1., pose, np.array((.1, .2, .3)),
                            np.array((.01, .02, .03)), np.array((.001, .002, .003)))
    step = np.array((0., 0., np.pi / 2, .01, 0., 0.,
                     .03, -.01, .02, .01, 0., 0., 0., .001, 0.))
    result = _navigation_plus(state, step)
    assert np.allclose(result.pose[:3, 3], pose[:3, 3] + pose[:3, :3] @ step[3:6])
    assert np.allclose(result.velocity, state.velocity + step[6:9])
    assert np.allclose(result.accel_bias, state.accel_bias + step[9:12])
    assert np.allclose(result.gyro_bias, state.gyro_bias + step[12:15])
    assert result.timestamp == state.timestamp
    assert np.array_equal(state.pose, pose)


def _check_factor_derivative(factor, poses):
    values = gtsam.Values()
    for key, pose in poses.items():
        values.insert(key, gtsam.Pose3(pose))
    linear = factor.linearize(values)
    A, _ = linear.jacobian()
    numerical = np.zeros_like(A)
    epsilon = 1e-6
    for slot, key in enumerate(linear.keys()):
        for column in range(6):
            increment = np.eye(6)[column] * epsilon
            plus, minus = gtsam.Values(values), gtsam.Values(values)
            plus.update(key, gtsam.Pose3(split_pose_plus(poses[key], increment)))
            minus.update(key, gtsam.Pose3(split_pose_plus(poses[key], -increment)))
            numerical[:, 6 * slot + column] = (
                factor.whitenedError(plus) - factor.whitenedError(minus)
            ) / (2 * epsilon)
    assert np.allclose(A, numerical, rtol=1e-5, atol=1e-6)


def test_backend_prior_and_between_jacobians_match_split_finite_differences():
    noise = gtsam.noiseModel.Diagonal.Sigmas(np.array((.1, .2, .3, .01, .02, .03)))
    anchor = _pose((.2, -.3, .1), (.01, -.02, .04))
    reference = _pose((-.1, .2, .3), (.02, .01, -.01))
    target = _pose((.4, -.2, .1), (.04, -.01, .02))
    measurement = _pose((.1, -.05, .2), (.01, .02, -.02))
    _check_factor_derivative(split_prior_factor(3, anchor, noise), {3: target})
    _check_factor_derivative(split_between_factor(3, 5, measurement, noise),
                             {3: reference, 5: target})


@pytest.mark.parametrize("factor_builder", [_imu_factor, _split_imu_factor])
def test_imu_columns_match_split_navigation_finite_differences(factor_builder):
    previous = NavigationState(0., _pose((.1, -.2, .3), (.04, .02, -.01)),
                               np.array((.1, -.2, .3)), np.array((.01, -.02, .03)),
                               np.array((.001, -.002, .003)))
    samples = [ImuSample(float(t), (.2, -.1, 9.8), (.1, -.2, .3))
               for t in np.linspace(0., .1, 11)]
    extrinsic = _pose((.2, .1, -.1), (.02, -.01, .03))
    solver = GtsamVisualInertialSolver({})
    current, pim = solver.predict_current(previous, samples, extrinsic)
    current = _navigation_plus(current, np.arange(15) * .0001)
    factor = factor_builder(pim)
    Ap, Ac, _ = navigation_factor_terms(factor, previous, current)
    expected = np.column_stack((Ap, Ac))
    numerical = np.zeros_like(expected)
    epsilon = 1e-6
    for index in range(30):
        increment = np.eye(15)[index % 15] * epsilon
        if index < 15:
            plus = _imu_values(_navigation_plus(previous, increment), current)
            minus = _imu_values(_navigation_plus(previous, -increment), current)
        else:
            plus = _imu_values(previous, _navigation_plus(current, increment))
            minus = _imu_values(previous, _navigation_plus(current, -increment))
        numerical[:, index] = (factor.whitenedError(plus) - factor.whitenedError(minus)) / (2 * epsilon)
    assert np.allclose(expected, numerical, rtol=2e-4, atol=2e-4)


def test_motion_errors_match_prediction_in_target_imu_axes():
    previous = NavigationState(0., _pose((.1, -.2, .3), (.04, .02, -.01)),
                               np.array((.1, -.2, .3)), np.array((.01, -.02, .03)),
                               np.array((.001, -.002, .003)))
    interval = [ImuSample(float(t), (.2, -.1, 9.8), (.1, -.2, .3))
                for t in np.linspace(0., .1, 11)]
    solver = GtsamVisualInertialSolver({})
    predicted, pim = solver.predict_current(previous, interval, np.eye(4))
    current = _navigation_plus(predicted, np.array((
        0., 0., 1.2, .01, -.02, .03, .1, -.2, .3, .01, -.02, .03, .001, -.002, .003,
    )))
    factor = _split_imu_factor(pim)
    residual = factor.unwhitenedError(_imu_values(previous, current))
    Rt = current.pose[:3, :3].T
    expected = np.r_[
        gtsam.Rot3.Logmap(gtsam.Rot3(Rt @ predicted.pose[:3, :3])),
        Rt @ (predicted.pose[:3, 3] - current.pose[:3, 3]),
        Rt @ (predicted.velocity - current.velocity),
    ]
    assert np.allclose(residual, expected)
    rotated = current.copy()
    rotated.pose[:3, :3] = predicted.pose[:3, :3] @ gtsam.Rot3.Expmap(np.array((.5, 0., 0.))).matrix()
    rotated_residual = factor.unwhitenedError(_imu_values(previous, rotated))
    transport = rotated.pose[:3, :3].T @ current.pose[:3, :3]
    assert np.allclose(rotated_residual[3:6], transport @ residual[3:6])
    assert np.allclose(rotated_residual[6:9], transport @ residual[6:9])
    assert not np.allclose(rotated_residual[:3], residual[:3])


def test_motion_factor_retains_native_covariance_and_all_motion_cross_terms():
    previous = NavigationState(0., np.eye(4), np.zeros(3), np.zeros(3), np.zeros(3))
    interval = [ImuSample(float(t), (.2, -.1, 9.8), (.1, -.2, .3))
                for t in np.linspace(0., .1, 11)]
    _, pim = GtsamVisualInertialSolver({}).predict_current(previous, interval, np.eye(4))
    native = _imu_factor(pim)
    split = _split_imu_factor(pim)
    expected = native.noiseModel().covariance()
    covariance = split.noiseModel().covariance()
    assert np.allclose(covariance, expected, rtol=1e-8, atol=1e-15)
    assert covariance.shape == (9, 9)
    assert np.linalg.norm(covariance[3:6, 6:9]) > 0
    assert np.all(np.linalg.eigvalsh(covariance) > 0)


@pytest.mark.parametrize("rotation", [(.001, -.002, .003), (.2, -.3, .4), (.7, .1, -.5)])
def test_motion_cost_matches_native_factor_away_from_preintegrated_rotation(rotation):
    previous = NavigationState(0., _pose((.1, -.2, .3), (.04, .02, -.01)),
                               np.array((.1, -.2, .3)), np.array((.01, -.02, .03)),
                               np.array((.001, -.002, .003)))
    samples = [ImuSample(float(t), (.2 + t, -.1, 9.8), (.1, -.2, .3 + t))
               for t in np.linspace(0., .1, 11)]
    target, pim = GtsamVisualInertialSolver({}).predict_current(previous, samples, np.eye(4))
    previous.accel_bias += (.002, -.001, .003)
    previous.gyro_bias += (.0001, -.0002, .0003)
    target = _navigation_plus(target, np.r_[rotation, [.01, -.02, .03],
                                           [.1, -.2, .3], np.zeros(6)])
    values = _imu_values(previous, target)
    native = _imu_factor(pim)
    motion = _split_imu_factor(pim)
    assert motion.error(values) == pytest.approx(native.error(values), rel=1e-12)
    # The linearized objective must also agree, including all motion cross terms.
    native_rows = navigation_factor_terms(native, previous, target)
    motion_rows = navigation_factor_terms(motion, previous, target)
    A = np.column_stack(motion_rows[:2])
    reference_A = np.column_stack(native_rows[:2])
    assert np.allclose(A.T @ A, reference_A.T @ reference_A, rtol=1e-12, atol=1e-12)
    assert np.allclose(A.T @ motion_rows[2], reference_A.T @ native_rows[2],
                       rtol=1e-12, atol=1e-12)


def test_frontend_accepts_a_finite_rotation_and_position_step_in_one_iteration():
    target = _pose((0., 0., np.pi / 2), (.01, 0., 0.))

    class PoseObservation:
        def __len__(self):
            return 10

        def evaluate(self, _reference, pose):
            return (np.tile(split_pose_coordinates(target, pose), 5),
                    np.zeros((30, 6)),
                    np.tile(split_pose_coordinate_jacobian(target, pose), (5, 1)))

    previous = NavigationState(0., np.eye(4), np.zeros(3), np.zeros(3), np.zeros(3))
    current = previous.copy()
    current.timestamp = .1
    interval = [ImuSample(float(t), (0., 0., 9.81), (0., 0., 0.))
                for t in np.linspace(0., .1, 11)]
    solver = GtsamVisualInertialSolver({
        "gtsam_vi_max_iterations": 1, "gtsam_vi_normal_huber_delta": 0.,
        "dense_normalflow_residual_weight": 1e8, "imu_residual_weight": 1e-8,
    })
    result = solver.refine_current(
        PoseObservation(), np.eye(4), previous, current, interval, np.eye(4)
    )
    assert result is not None
    assert np.allclose(result.pose, target, atol=1e-7)
    assert solver.last_optimization_trace[0]["accepted_scale"] == 1.
    assert solver.pending_marginal_prior.matches(result)


def test_backend_solver_applies_split_retraction_at_finite_steps(monkeypatch):
    import gelslam.core.split_pose as module

    noise = gtsam.noiseModel.Isotropic.Sigma(6, .01)
    graph = gtsam.NonlinearFactorGraph()
    anchor = np.eye(4)
    target = _pose((0., 0., np.pi / 2), (.01, 0., 0.))
    graph.add(split_prior_factor(0, anchor, noise))
    graph.add(split_between_factor(0, 1, target, noise))
    initial = gtsam.Values()
    initial.insert(0, gtsam.Pose3())
    initial.insert(1, gtsam.Pose3())
    original_plus = module.split_pose_plus
    steps = []

    def observed_plus(pose, increment):
        steps.append((pose.copy(), increment.copy()))
        return original_plus(pose, increment)

    monkeypatch.setattr(module, "split_pose_plus", observed_plus)
    solved = optimize_split_pose_graph(graph, initial)
    assert np.allclose(solved.atPose3(1).matrix(), target, atol=1e-7)
    assert any(np.linalg.norm(step[:3]) > 1. for _, step in steps)
    assert graph.error(solved) < graph.error(initial) * 1e-10
