"""ORB-SLAM3 sample averaging and unrobustified frontend prior regressions."""

import gtsam
import numpy as np
import pytest

from gelslam.core.gtsam_vi import (
    GtsamVisualInertialSolver, _bias_factor, _imu_values, _navigation_plus,
    _split_imu_factor,
    preintegrate_combined, preintegrate_imu,
)
from gelslam.core.inertial import ImuBuffer, ImuSample
from gelslam.core.marginal_prior import NavigationPrior, marginalize_rows
from gelslam.core.numpy_vi import NavigationState


def state(timestamp):
    return NavigationState(timestamp, np.eye(4), np.zeros(3), np.zeros(3), np.zeros(3))


@pytest.mark.parametrize("preintegrate", [preintegrate_imu, preintegrate_combined])
@pytest.mark.parametrize("start,end", [(0., .3), (.025, .275), (.025, .075)])
def test_linear_acceleration_integrates_between_interpolated_image_times(preintegrate, start, end):
    buffer = ImuBuffer()
    for t in (0., .1, .2, .3):
        buffer.add(ImuSample(t, (2. + 4. * t, 0., 9.81), (0., 0., 0.)))
    samples = buffer.samples_covering(start, end)
    previous = state(start)
    pim = preintegrate(samples, previous, {}, np.eye(4), np.array([0., 0., -9.81]))
    predicted = pim.predict(gtsam.NavState(gtsam.Pose3(), previous.velocity),
                            gtsam.imuBias.ConstantBias())
    # Exact integral of a_x(t)=2+4t. Covers two clipped boundaries and a
    # complete image interval inside one raw IMU period.
    expected_velocity = 2. * (end - start) + 2. * (end**2 - start**2)
    assert pim.deltaTij() == pytest.approx(end - start)
    assert predicted.velocity() == pytest.approx([expected_velocity, 0., 0.], abs=1e-12)


def test_nonuniform_samples_average_angular_velocity_and_preserve_bias_correction():
    previous = state(0.)
    previous.gyro_bias = np.array([0., 0., .2])
    samples = [ImuSample(t, (0., 0., 9.81), (0., 0., .2 + 1. + 2. * t))
               for t in (0., .01, .04, .1)]
    solver = GtsamVisualInertialSolver({})
    target, pim = solver.predict_current(previous, samples, np.eye(4))
    # Integral of the bias-corrected gyro rate 1+2t over [0,.1].
    expected_rotation = gtsam.Rot3.Rz(.11).matrix()
    assert target.timestamp == pytest.approx(.1)
    assert np.allclose(target.pose[:3, :3], expected_rotation, atol=1e-12)
    assert np.allclose(target.velocity, 0., atol=1e-12)
    assert np.linalg.norm(_split_imu_factor(pim).unwhitenedError(
        _imu_values(previous, target))) < 1e-12


@pytest.mark.parametrize("norm", [0., 2., 5., 6., 20.])
def test_prior_matches_native_quadratic_cost_and_normal_equations(norm):
    previous = state(0.)
    prior = NavigationPrior.initial(previous, np.ones(15))
    delta = np.zeros(15)
    # Large prior residuals must retain their full quadratic information.
    delta[6:8] = [norm * .6, norm * .8]
    current = _navigation_plus(previous, delta)
    factor = gtsam.PriorFactorVector(1, np.zeros(15), gtsam.noiseModel.Unit.Create(15))
    values = gtsam.Values()
    values.insert(1, delta)
    native_A, native_rhs = factor.linearize(values).jacobian()
    A, rhs = prior.linearize(current)
    assert prior.error(current) == pytest.approx(factor.error(values))
    assert np.allclose(A.T @ A, native_A.T @ native_A)
    assert np.allclose(A.T @ rhs, native_A.T @ native_rhs)
    assert prior.error(current) == pytest.approx(.5 * norm**2)


def test_prior_gradient_matches_cost_with_correlations_and_retained_rhs():
    rng = np.random.default_rng(37)
    previous = state(0.)
    prior = NavigationPrior(previous, rng.normal(size=(16, 15)) * 8.,
                            rng.normal(size=16))
    current = _navigation_plus(previous, rng.normal(size=15) * .2)
    A, rhs = prior.linearize(current)
    assert np.sqrt(2. * prior.error(current)) > 5.
    gradient = np.empty(15)
    for column in range(15):
        step = np.eye(15)[column] * 1e-6
        gradient[column] = (
            prior.error(_navigation_plus(current, step))
            - prior.error(_navigation_plus(current, -step))
        ) / 2e-6
    assert np.allclose(-A.T @ rhs, gradient, rtol=1e-6, atol=1e-7)


class PoseVisual:
    def __len__(self):
        return 10

    def evaluate(self, _reference, target):
        residual = np.full(30, target[0, 3] - .02)
        J = np.zeros((30, 6))
        J[:, 3] = 1.
        return residual, np.zeros_like(J), J


def test_solver_uses_quadratic_prior_for_gn_and_retention():
    solver = GtsamVisualInertialSolver({
        "gtsam_vi_max_iterations": 3,
        "gtsam_vi_rotation_stop_rad": 1e-12,
        "gtsam_vi_translation_stop_m": 1e-12,
        "gtsam_vi_normal_huber_delta": 0.,
        "dense_normalflow_residual_weight": 1e6,
    })
    previous = state(0.)
    samples = [ImuSample(t, (0., 0., 9.81), (0., 0., 0.))
               for t in np.linspace(0., .1, 11)]
    target, pim = solver.predict_current(previous, samples, np.eye(4))
    # Nonzero retained gradient at the accepted anchor, above the old Huber limit.
    prior = NavigationPrior.initial(previous, solver.prior_sigmas)
    prior.rhs[3] = 20.
    solver.marginal_prior = prior
    factors = (_split_imu_factor(pim), _bias_factor(pim, solver.config))
    raw_rows = solver._joint_rows(factors, prior, previous, target)
    A, rhs = prior.linearize(previous)
    assert np.array_equal(raw_rows[0][:15], A)
    assert np.array_equal(raw_rows[2][:15], rhs)

    calls = []
    original_joint_rows = solver._joint_rows

    def capture_rows(factors, prior, previous, current, **kwargs):
        rows = original_joint_rows(factors, prior, previous, current, **kwargs)
        expected_A, expected_rhs = prior.linearize(previous)
        assert np.array_equal(rows[0][:15], expected_A)
        assert np.array_equal(rows[2][:15], expected_rhs)
        calls.append((rows, current.copy()))
        return rows

    solver._joint_rows = capture_rows
    result = solver.refine_current(PoseVisual(), np.eye(4), previous, target,
                                   samples, np.eye(4), preintegrated=pim)
    assert result is not None
    traces = solver.last_optimization_trace
    assert traces[0]["base_prior_cost"] == pytest.approx(200.)
    assert any(trace["accepted_scale"] is not None for trace in traces)
    for trace in traces:
        assert trace["base_cost"] == pytest.approx(sum(trace[key] for key in (
            "base_normal_cost", "base_imu_cost", "base_bias_cost", "base_prior_cost")))
        if trace["accepted_scale"] is not None:
            assert trace["accepted_scale"] == 1.
            assert trace["update_rule"] == 'full_gauss_newton'
            assert trace["accepted_cost"] == pytest.approx(sum(trace[key] for key in (
                "accepted_normal_cost", "accepted_imu_cost",
                "accepted_bias_cost", "accepted_prior_cost")))
    assert len(calls) == len(traces) + 1
    Ap, Ac, rhs = calls[-1][0]
    current = calls[-1][1]
    normal_b, _, normal_A = PoseVisual().evaluate(np.eye(4), current.pose)
    scale = np.sqrt(solver.visual_weight / (len(normal_b) // 3)) / solver.normal_sigma
    expected_A, expected_rhs = marginalize_rows(
        np.vstack((Ap, np.zeros((30, 15)))),
        np.vstack((Ac, np.pad(normal_A * scale, ((0, 0), (0, 9))))),
        np.r_[rhs, -normal_b * scale],
    )
    retained = solver.pending_marginal_prior
    assert np.allclose(retained.A.T @ retained.A, expected_A.T @ expected_A)
    assert np.array_equal(retained.rhs, np.zeros(15))
    assert retained.error(result) == 0.
    assert solver.marginal_prior is prior  # Publication/acceptance still owns commit.
