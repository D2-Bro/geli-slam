"""Configured bias constants must not become optimization variables."""
import importlib.util
from pathlib import Path

import gtsam
import numpy as np
import pytest

import gelslam.core.gtsam_vi as gv
from gelslam.core.inertial import ImuSample
from gelslam.core.numpy_vi import NavigationState


ACCEL_BIAS = np.array([.003, -.002, .004])
GYRO_BIAS = np.array([.0002, -.0001, .00015])


def config(**options):
    return dict(frontend_fix_bias=True,
                initial_accel_bias_mps2=ACCEL_BIAS.tolist(),
                initial_gyro_bias_rps=GYRO_BIAS.tolist(),
                **options)


def inputs(start=0.):
    previous = NavigationState(start, np.eye(4), np.array([.01, -.02, .005]),
                               ACCEL_BIAS.copy(), GYRO_BIAS.copy())
    samples = [ImuSample(t, ACCEL_BIAS + [0., 0., 9.81], GYRO_BIAS)
               for t in np.linspace(start, start + .1, 11)]
    return previous, samples


class PositionVisual:
    def __len__(self):
        return 10

    def evaluate(self, _reference, target):
        residual = np.tile(target[:3, 3] - [.0005, -.0002, .0001], 10)
        J = np.tile(np.c_[np.zeros((3, 3)), target[:3, :3]], (10, 1))
        return residual, np.zeros_like(J), J


def assert_bias(state):
    assert np.array_equal(state.accel_bias, ACCEL_BIAS)
    assert np.array_equal(state.gyro_bias, GYRO_BIAS)


@pytest.mark.parametrize('initialize', [False, True])
def test_prediction_uses_configured_bias_without_mutating_input(initialize):
    previous, samples = inputs()
    solver = gv.GtsamVisualInertialSolver(config(
        frontend_initialize_from_imu_preintegration=initialize))
    expected, _ = gv.GtsamVisualInertialSolver({
        'frontend_initialize_from_imu_preintegration': initialize,
    }).predict_current(previous, samples, np.eye(4))
    previous.accel_bias += .5
    previous.gyro_bias -= .1
    original = previous.copy()
    actual, pim = solver.predict_current(previous, samples, np.eye(4))
    assert_bias(actual)
    assert np.array_equal(actual.pose, expected.pose)
    assert np.array_equal(actual.velocity, expected.velocity)
    assert np.array_equal(previous.accel_bias, original.accel_bias)
    assert np.array_equal(previous.gyro_bias, original.gyro_bias)
    assert pim.biasHat().equals(gtsam.imuBias.ConstantBias(ACCEL_BIAS, GYRO_BIAS), 0.)


@pytest.mark.parametrize('fixed_previous', [False, True])
@pytest.mark.parametrize('marginalization', [False, True])
def test_bias_is_excluded_while_pose_velocity_and_prior_are_updated(
        monkeypatch, fixed_previous, marginalization):
    spec = importlib.util.spec_from_file_location(
        'bias_iteration_observer',
        Path(__file__).resolve().parents[3] / 'scripts/diagnose_five_observations.py')
    diagnostic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostic)
    solver = gv.GtsamVisualInertialSolver(config(
        frontend_fix_previous_state=fixed_previous,
        gtsam_vi_marginalization=marginalization,
        gtsam_vi_max_iterations=5,
        dense_normalflow_residual_weight=1e8))
    original_solve = gv.solve_gaussian_qr
    graphs = []

    def solve(graph):
        keys = {key for i in range(graph.size()) for key in graph.at(i).keys()}
        assert gv.BIAS_I not in keys and gv.BIAS_J not in keys
        graphs.append(keys)
        return original_solve(graph)

    monkeypatch.setattr(gv, 'solve_gaussian_qr', solve)
    previous, samples = inputs()
    current, pim = solver.predict_current(previous, samples, np.eye(4))
    initial = current.copy()
    # Enforce configured values even if a caller supplies a different target.
    current.accel_bias += .02
    current.gyro_bias -= .001
    observer = diagnostic.IterationObserver()
    result = observer.run(solver.refine_current, PositionVisual(), np.eye(4),
                          previous, current, samples, np.eye(4), preintegrated=pim)
    assert result is not None and graphs and observer.iterations
    assert_bias(previous)
    assert_bias(result)
    assert not np.array_equal(result.pose, initial.pose)
    assert not np.array_equal(result.velocity, initial.velocity)
    for iteration in observer.iterations:
        for name in ('previous', 'current', 'after_previous', 'after_current'):
            assert_bias(iteration[name])
        assert np.array_equal(iteration['bias_step'], np.zeros(6))
        if iteration['previous_step'] is not None:
            assert np.array_equal(iteration['previous_step'][9:], np.zeros(6))
    if marginalization and not fixed_previous:
        prior = solver.pending_marginal_prior
        assert prior is not None and prior.A.shape[1] == 15
        assert np.array_equal(prior.A[:, 9:], np.zeros_like(prior.A[:, 9:]))
        assert np.linalg.norm(prior.A[:, :9]) > 0
        solver.commit_marginal_prior(result)
        previous = result
        _, samples = inputs(result.timestamp)
        current, pim = solver.predict_current(previous, samples, np.eye(4))
        next_result = solver.refine_current(PositionVisual(), np.eye(4), previous,
                                            current, samples, np.eye(4), preintegrated=pim)
        assert next_result is not None and solver.last_prior_source == 'retained'
        assert_bias(next_result)
    else:
        assert solver.pending_marginal_prior is None


def test_supplied_preintegration_with_a_different_bias_is_rebuilt(monkeypatch):
    previous, samples = inputs()
    previous.accel_bias += .1
    previous.gyro_bias += .01
    current, wrong_pim = gv.GtsamVisualInertialSolver({}).predict_current(
        previous, samples, np.eye(4))
    solver = gv.GtsamVisualInertialSolver(config())
    original = gv.preintegrate_imu
    rebuilt = []

    def integrate(samples, anchor, *args):
        assert_bias(anchor)
        rebuilt.append(True)
        return original(samples, anchor, *args)

    monkeypatch.setattr(gv, 'preintegrate_imu', integrate)
    result = solver.refine_current(PositionVisual(), np.eye(4), previous, current,
                                    samples, np.eye(4), preintegrated=wrong_pim)
    assert result is not None and rebuilt == [True]
    assert_bias(result)


def test_prediction_only_still_uses_configured_constants(monkeypatch):
    previous, samples = inputs()
    solver = gv.GtsamVisualInertialSolver(config(frontend_imu_prediction_only=True))
    previous.accel_bias += .3
    previous.gyro_bias -= .2
    current, pim = solver.predict_current(previous, samples, np.eye(4))

    def forbidden(*args):
        pytest.fail('prediction-only mode must not solve')

    monkeypatch.setattr(gv, 'solve_gaussian_qr', forbidden)
    result = solver.refine_current(None, np.eye(4), previous, current, samples,
                                   np.eye(4), preintegrated=pim)
    assert_bias(result)
    assert np.array_equal(result.pose, current.pose)
    assert solver.last_optimization_trace == []


@pytest.mark.parametrize('invalid', ['true', 'false', 0, 1, None])
def test_flag_requires_a_boolean(invalid):
    with pytest.raises(ValueError, match='frontend_fix_bias must be a boolean'):
        gv.GtsamVisualInertialSolver({'frontend_fix_bias': invalid})


@pytest.mark.parametrize('field', ['initial_accel_bias_mps2', 'initial_gyro_bias_rps'])
@pytest.mark.parametrize('value', [[1., 2.], [1., 2., float('nan')], 'bad'])
def test_fixed_bias_requires_finite_three_vectors(field, value):
    settings = config()
    settings[field] = value
    with pytest.raises(ValueError, match=field + ' must be a finite three-vector'):
        gv.GtsamVisualInertialSolver(settings)


def test_omitted_flag_keeps_bias_estimation_and_omitted_vectors_use_zero():
    assert not gv.GtsamVisualInertialSolver({}).fix_bias
    assert not gv.GtsamVisualInertialSolver({'frontend_fix_bias': False}).fix_bias
    solver = gv.GtsamVisualInertialSolver({'frontend_fix_bias': True})
    assert all(np.array_equal(value, np.zeros(3)) for value in solver.fixed_bias)
