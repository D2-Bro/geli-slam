"""Staged frontend IMU coupling: exact row split, schedule, sigmas and reinflation."""

from types import SimpleNamespace

import numpy as np
import pytest

from gelslam.core.gtsam_vi import (
    _bias_factor, _split_imu_factor, navigation_factor_terms, split_rotation_motion_rows,
)
from gelslam.core.imu_coupling import (
    CouplingSchedule, inflate_bias, navigation_sigmas, resolve_gradual_coupling,
)
from gelslam.core.inertial import ImuSample
from gelslam.core.marginal_prior import NavigationPrior
from gelslam.core.numpy_vi import DenseNormalResidual, NavigationState
from gelslam.core.retained_prior import (
    RetainedPrior, VariableLayout, rank_solve, state_blocks, state_values,
)
from gelslam.core.tracker import Tracker
from test_reference_marginalization import FlatVisual, advance, solver, state

RM = dict(frontend_optimization=True)
# Prior/state column groups in the 15D navigation layout.
ROTATION, TRANSLATION, VELOCITY, ACCEL, GYRO = (np.arange(0, 3), np.arange(3, 6),
                                                np.arange(6, 9), np.arange(9, 12),
                                                np.arange(12, 15))


def settings(**overrides):
    return resolve_gradual_coupling(dict(RM, gradual_coupling=dict(enabled=True, **overrides)))


def interval(span=.033, count=4, perturb=True):
    # real_objects_config noise; the shortest interval then has a Diagonal model.
    s = solver(imu_residual_weight=1., accelerometer_noise_density=.002,
               gyroscope_noise_density=.00016, gtsam_integration_sigma_m=.0003)
    previous = NavigationState(1., np.eye(4), np.array([.01, .02, 0.]),
                               np.array([.01, -.02, .03]), np.array([1e-3, 2e-3, -1e-3]))
    samples = tuple(ImuSample(float(t), [.3, -.2, 9.81], [.1, .2, -.3])
                    for t in np.linspace(1., 1. + span, count))
    current, pim = s.predict_current(previous, samples, np.eye(4))
    if perturb:
        current.pose[:3, 3] += 1e-3
        current.velocity += .01
    return s, previous, current, pim


@pytest.mark.parametrize("span,count,model", [(.004, 2, "Diagonal"), (.033, 4, "Gaussian"),
                                              (.1, 11, "Gaussian")])
def test_split_is_an_orthogonal_regrouping_with_gyroscope_only_rotation_rows(span, count, model):
    _, previous, current, pim = interval(span, count)
    factor = _split_imu_factor(pim)
    assert type(factor.noiseModel()).__name__ == model
    Ai, Aj, rhs = navigation_factor_terms(factor, previous, current)
    (mi, mj, mb), (ri, rj, rb) = split_rotation_motion_rows(factor, Ai, Aj, rhs)
    A, split = np.hstack((Ai, Aj)), np.vstack((np.hstack((mi, mj)), np.hstack((ri, rj))))
    b, split_b = rhs, np.concatenate((mb, rb))
    assert mi.shape == (6, 15) and ri.shape == (3, 15)
    np.testing.assert_allclose(split.T @ split, A.T @ A, rtol=0, atol=1e-12 * np.abs(A.T @ A).max())
    np.testing.assert_allclose(split.T @ split_b, A.T @ b, rtol=0,
                               atol=1e-12 * np.abs(A.T @ b).max())
    assert split_b @ split_b == pytest.approx(b @ b, rel=1e-12)
    rotation = np.abs(np.hstack((ri, rj))) / np.abs(np.hstack((ri, rj))).max()
    used = np.flatnonzero(rotation.max(axis=0) > 1e-12)
    # Previous rotation, previous gyroscope bias and current rotation only.
    np.testing.assert_array_equal(used, np.r_[ROTATION, GYRO, 15 + ROTATION])
    # Rotation rows carry exactly the marginal rotation information inv(Sigma_RR).
    # In residual coordinates the rotation rows are L e_R with L^T L = inv(Sigma_RR):
    # rows = L @ (rotation-error Jacobian), the first 3 rows of U^-1 times the whitened rows.
    U = factor.noiseModel().R()
    covariance = np.linalg.inv(U.T @ U)
    errors = np.linalg.solve(U, np.column_stack((A, b)))[:3]
    L = np.column_stack((np.hstack((ri, rj)), rb)) @ np.linalg.pinv(errors)
    expected = np.linalg.inv(covariance[:3, :3])
    np.testing.assert_allclose(L.T @ L, expected, rtol=0, atol=1e-8 * np.abs(expected).max())


def _rows(s, scale):
    s.coupling_schedule = None if scale is None else SimpleNamespace(scale=lambda t: scale)
    previous = state()
    samples = tuple(ImuSample(float(t), [.1, -.2, 9.81], [.01, .02, -.03])
                    for t in np.linspace(1., 1.1, 11))
    current, pim = s.predict_current(previous, samples, np.eye(4))
    prior, layout, values = s._problem(previous, current)
    return s._inertial_rows(prior, layout, values, previous.timestamp, current.timestamp,
                            (_split_imu_factor(pim), _bias_factor(pim, s.config)))


def test_motion_rows_scale_alone_and_zero_drops_them():
    s = solver(imu_residual_weight=1e-4)
    unsplit_rows, unsplit_right = _rows(s, None)
    assert [len(r) for r in unsplit_rows] == [15, 9, 6]
    full_rows, full_right = _rows(s, 1.)
    weak_rows, weak_right = _rows(s, .1)
    zero_rows, zero_right = _rows(s, 0.)
    # [prior, rotation (3), motion (6), bias (6)]; motion is absent at c = 0.
    assert [len(r) for r in full_rows] == [15, 3, 6, 6]
    assert [len(r) for r in zero_rows] == [15, 3, 6]
    for a, b in ((weak_rows[1], full_rows[1]), (weak_right[1], full_right[1]),
                 (zero_rows[1], full_rows[1]), (zero_rows[2], full_rows[3])):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(weak_rows[2], .1 * full_rows[2], rtol=1e-12, atol=0)
    np.testing.assert_allclose(weak_right[2], .1 * full_right[2], rtol=1e-12, atol=0)
    # c = 1 is the unsplit IMU block, up to an orthogonal row transform.
    unsplit = np.vstack(unsplit_rows[1:2])
    split = np.vstack(full_rows[1:3])
    np.testing.assert_allclose(split.T @ split, unsplit.T @ unsplit, rtol=0,
                               atol=1e-12 * np.abs(unsplit.T @ unsplit).max())


class PerturbedVisual(FlatVisual):
    def __init__(self):
        from test_frontend_gauss_newton import frame
        self.image = frame()

    def evaluate(self, ref, tar):
        visual = DenseNormalResidual(self.image, self.image, 1., ref, tar, 1000)
        residual, Jr, Jt = visual.evaluate(ref, tar)
        return residual + .01, Jr, Jt

    def target_z_height_correction(self, ref, tar):
        visual = DenseNormalResidual(self.image, self.image, 1., ref, tar, 1000)
        return visual.target_z_height_correction(ref, tar)


@pytest.mark.parametrize("mode,system", [("eliminate_first", "hessian"), ("joint", "rows")])
def test_unit_motion_scale_reproduces_the_unsplit_solve(mode, system):
    runs = []
    for scale in (None, 1.):
        s = solver(dense_normalflow_residual_weight=1e5, gtsam_vi_height_z_enabled=True,
                   frontend_reference_previous_state=mode, frontend_reference_linear_system=system)
        if scale is not None:
            s.coupling_schedule = CouplingSchedule(settings(attitude_motion_sigma_scale=1.,
                                                            max_attitude_s=None))
            assert s.motion_row_scale(2.) == 1.
        states = [advance(s, state(), PerturbedVisual())]
        states.append(advance(s, states[-1], PerturbedVisual()))
        s.set_reference(states[-1].timestamp, states[-1])
        states.append(advance(s, states[-1], PerturbedVisual()))
        states.append(advance(s, states[-1], PerturbedVisual()))
        runs.append((states, s.marginal_prior))
    (base, base_prior), (split, split_prior) = runs
    for a, b in zip(base, split):
        for name in ('pose', 'velocity', 'accel_bias', 'gyro_bias'):
            np.testing.assert_allclose(getattr(b, name), getattr(a, name), rtol=0., atol=1e-10)
    H, Hs = base_prior.A.T @ base_prior.A, split_prior.A.T @ split_prior.A
    np.testing.assert_allclose(Hs, H, rtol=0., atol=1e-9 * np.abs(H).max())
    g, gs = base_prior.A.T @ base_prior.rhs, split_prior.A.T @ split_prior.rhs
    np.testing.assert_allclose(gs, g, rtol=0., atol=1e-9 * max(np.abs(g).max(), 1.))


def test_rotation_only_propagation_adds_no_velocity_or_accel_bias_information():
    s = solver()
    s.coupling_schedule = SimpleNamespace(scale=lambda t: 0.)
    previous = state()
    samples = tuple(ImuSample(float(t), [.1, -.2, 9.81], [.01, .02, -.03])
                    for t in np.linspace(1., 1.1, 11))
    current, pim = s.predict_current(previous, samples, np.eye(4))
    rotation_only = s.propagate_prior(previous, current, pim)
    s.coupling_schedule = None
    full = s.propagate_prior(previous, current, pim)
    # The target velocity is reachable only through accelerometer motion rows.
    assert navigation_sigmas(rotation_only, current.timestamp)["velocity"] == np.inf
    assert np.isfinite(navigation_sigmas(full, current.timestamp)["velocity"])
    # Accelerometer bias still follows its prior plus random walk.
    assert np.isfinite(navigation_sigmas(rotation_only, current.timestamp)["accel_bias"])


def test_cached_propagation_is_not_reused_after_the_scale_changes(monkeypatch):
    s = solver(frontend_reference_previous_state="eliminate_first")
    scale = {"c": .1}
    s.coupling_schedule = SimpleNamespace(scale=lambda t: scale["c"])
    previous = state()
    samples = tuple(ImuSample(float(t), [0., 0., 9.81], [0., 0., 0.])
                    for t in np.linspace(1., 1.1, 11))
    current, pim = s.predict_current(previous, samples, np.eye(4))
    s.propagate_prior(previous, current, pim)
    calls = []
    original = s._inertial_rows
    monkeypatch.setattr(s, "_inertial_rows", lambda *a: calls.append(1) or original(*a))
    assert s.refine_current(FlatVisual(), np.eye(4), previous, current, samples, np.eye(4),
                            preintegrated=pim) is not None
    assert not calls  # Same prior, interval and scale: rows reused.
    s.discard_pending_prior()
    s.propagate_prior(previous, current, pim)
    scale["c"] = 1.
    assert s.refine_current(FlatVisual(), np.eye(4), previous, current, samples, np.eye(4),
                            preintegrated=pim) is not None
    assert calls


def random_prior(seed=3, timestamp=2.):
    rng = np.random.default_rng(seed)
    layout = VariableLayout(state_blocks(timestamp))
    values = state_values(NavigationState(timestamp, np.eye(4), rng.normal(size=3),
                                          rng.normal(size=3), rng.normal(size=3)))
    return RetainedPrior(layout, values, rng.normal(size=(20, 15)), rng.normal(size=20), timestamp)


def test_navigation_sigmas_are_marginal_and_infinite_without_information():
    prior = random_prior()
    covariance = np.linalg.inv(prior.A.T @ prior.A)
    sigmas = navigation_sigmas(prior, 2.)
    for kind, columns in (("velocity", VELOCITY), ("accel_bias", ACCEL), ("gyro_bias", GYRO)):
        expected = np.sqrt(np.max(np.diag(covariance)[columns]))
        assert sigmas[kind] == pytest.approx(expected, rel=1e-9)
    blind = RetainedPrior(prior.layout, prior.anchors, prior.A.copy(), prior.rhs, 2.)
    blind.A[:, ACCEL[1]] = 0.
    sigmas = navigation_sigmas(blind, 2.)
    assert sigmas["accel_bias"] == np.inf and np.isfinite(sigmas["velocity"])
    navigation = NavigationPrior.initial(NavigationState(2., np.eye(4), np.zeros(3),
                                                         np.zeros(3), np.zeros(3)),
                                         np.repeat([.1, .2, .3, .4, .5], 3))
    assert navigation_sigmas(navigation, 2.) == pytest.approx(
        dict(velocity=.3, accel_bias=.4, gyro_bias=.5), rel=1e-12)


def test_bias_reinflation_adds_variance_only_to_the_bias_block():
    prior = random_prior(seed=11)
    inflated = inflate_bias(prior, .05, .002)
    before = np.linalg.inv(prior.A.T @ prior.A)
    after = np.linalg.inv(inflated.A.T @ inflated.A)
    added = np.zeros(15)
    added[ACCEL], added[GYRO] = .05 ** 2, .002 ** 2
    np.testing.assert_allclose(after, before + np.diag(added), rtol=0,
                               atol=1e-9 * np.abs(before).max())
    # Same mean, anchors and layout.
    np.testing.assert_allclose(rank_solve(inflated.A, inflated.rhs)[0],
                               rank_solve(prior.A, prior.rhs)[0], atol=1e-9)
    assert inflated.layout is prior.layout and inflated.timestamp == prior.timestamp
    for block, value in prior.anchors.items():
        np.testing.assert_array_equal(inflated.anchors[block], value)


def test_schedule_gate_ramp_timeout_and_reset():
    prior = random_prior()
    observed = navigation_sigmas(prior, 2.)
    schedule = CouplingSchedule(settings(attitude_motion_sigma_scale=4., min_attitude_s=.5,
                                         max_attitude_s=2., ramp_s=1.))
    schedule.reset(1.)
    assert schedule.scale(1.2) == .25 and schedule.phase(1.2) == "attitude"
    # No thresholds: the gate opens at min_attitude_s.
    update = schedule.update(random_prior(timestamp=1.4), 1.4)
    assert update.phase == "attitude" and schedule.gate_time is None
    update = schedule.update(random_prior(timestamp=1.5), 1.5)
    assert schedule.gate_time == 1.5 and update.gate_reason == "min_attitude_s"
    assert update.scale == .25  # The committed interval used the attitude scale.
    assert schedule.scale(2.0) == pytest.approx(.625) and schedule.phase(2.0) == "ramp"
    assert schedule.scale(2.5) == 1. and schedule.phase(2.5) == "full"
    schedule.reset(5.)
    assert schedule.gate_time is None and schedule.scale(5.1) == .25

    tight = CouplingSchedule(settings(accel_bias_sigma_mps2=observed["accel_bias"] / 2,
                                      min_attitude_s=0., max_attitude_s=1.))
    tight.reset(1.)
    assert tight.update(random_prior(timestamp=1.5), 1.5).gate_reason is None
    assert tight.update(random_prior(timestamp=2.), 2.).gate_reason == "max_attitude_s"
    loose = CouplingSchedule(settings(accel_bias_sigma_mps2=observed["accel_bias"] * 2,
                                      velocity_sigma_mps=observed["velocity"] * 2,
                                      min_attitude_s=0., max_attitude_s=None))
    loose.reset(2.)
    assert loose.update(prior, 2.).gate_reason == "sigma_thresholds"
    rotation_only = CouplingSchedule(settings(attitude_motion_sigma_scale=None))
    assert rotation_only.scale(0.) == 0.


def test_schedule_requests_one_bias_reinflation_after_full_coupling():
    schedule = CouplingSchedule(settings(min_attitude_s=0., ramp_s=.5, bias_reinflation=dict(
        enabled=True, after_full_s=1.)))
    schedule.reset(2.)
    assert schedule.update(random_prior(timestamp=2.), 2.).gate_reason == "min_attitude_s"
    assert not schedule.update(random_prior(timestamp=3.), 3.).reinflate  # full at 2.5
    assert schedule.update(random_prior(timestamp=3.5), 3.5).reinflate
    assert not schedule.update(random_prior(timestamp=4.), 4.).reinflate


@pytest.mark.parametrize("block,message", [
    (dict(enabled=True, typo=1), "unknown gradual_coupling"),
    (dict(enabled="yes"), "enabled must be a boolean"),
    (dict(enabled=True, attitude_motion_sigma_scale=.5), "attitude_motion_sigma_scale"),
    (dict(enabled=True, min_attitude_s=2., max_attitude_s=1.), "max_attitude_s"),
    (dict(enabled=True, accel_bias_sigma_mps2=0.), "accel_bias_sigma_mps2"),
    (dict(enabled=True, ramp_s=float("nan")), "ramp_s"),
    (dict(enabled=True, bias_reinflation=dict(enabled=True, extra=1)), "bias_reinflation"),
    (dict(enabled=True, bias_reinflation=dict(gyro_bias_sigma_rps=-1.)), "gyro_bias_sigma_rps"),
])
def test_invalid_settings_fail_explicitly(block, message):
    with pytest.raises(ValueError, match=message):
        resolve_gradual_coupling(dict(RM, gradual_coupling=block))


def test_disabled_block_is_inert():
    assert resolve_gradual_coupling({}) is None
    assert resolve_gradual_coupling(dict(gradual_coupling=dict(enabled=False, typo=1))) is None


def test_tracker_commit_hook_logs_gate_and_applies_reinflation():
    schedule = CouplingSchedule(settings(min_attitude_s=0., bias_reinflation=dict(
        enabled=True, after_full_s=0., accel_bias_sigma_mps2=.05, gyro_bias_sigma_rps=.002)))
    schedule.reset(2.)
    messages = []
    tracker = SimpleNamespace(_imu_coupling=schedule, _trial_index=3,
                              logger=SimpleNamespace(info=messages.append, warning=messages.append))
    prior = random_prior()
    rm_solver = Tracker._attach_imu_coupling(
        tracker, SimpleNamespace(reference_marginalization=True, marginal_prior=prior))
    Tracker._after_frontend_commit(tracker, rm_solver, 2., "contact")
    assert rm_solver.marginal_prior is not prior  # Gate and reinflation at the same commit.
    assert any("gate opened" in m for m in messages)
    assert any("bias reinflation" in m for m in messages)
    assert messages[-1].startswith("IMU coupling: t=2.000000 trial=3 source=contact phase=attitude")
    other = SimpleNamespace(coupling_schedule=None, marginal_prior=prior)
    count = len(messages)
    Tracker._after_frontend_commit(tracker, other, 2., "contact")
    assert len(messages) == count
