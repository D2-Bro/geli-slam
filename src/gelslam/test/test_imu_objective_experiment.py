"""Independent physical trajectories for the active frontend IMU factor."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'imu_objective_experiment', Path(__file__).resolve().parents[3]/'scripts/experiment_imu_objective_and_drift.py')
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
gv = experiment.gv


@pytest.mark.parametrize('motion', ['stationary', 'constant_velocity',
                                    'constant_acceleration', 'gravity_axis_rotation'])
def test_analytic_gt_has_zero_motion_residual_and_matches_direct_equations(motion):
    previous, target, samples, gravity = experiment.synthetic_case(motion)
    pim = gv.preintegrate_imu(samples, previous, {}, np.eye(4), gravity)
    factor = gv._split_imu_factor(pim)
    assert np.linalg.norm(factor.unwhitenedError(gv._imu_values(previous, target))) < 1e-11
    perturbed = gv._navigation_plus(target, experiment.PERTURBATION)
    actual = factor.unwhitenedError(gv._imu_values(previous, perturbed))
    expected = experiment.physical_residual(previous, perturbed, pim, gravity)
    assert np.allclose(actual, expected, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize('sign', [-1., 1.])
def test_fixed_gt_anchor_recovers_analytic_motion_from_opposite_perturbations(sign):
    previous, target, samples, gravity = experiment.synthetic_case('constant_acceleration')
    previous_before = experiment.diag.state(previous)
    pim = gv.preintegrate_imu(samples, previous, {}, np.eye(4), gravity)
    initial = gv._navigation_plus(target, experiment.PERTURBATION*sign)
    result, trace = experiment.fixed_anchor_solve(previous, initial, pim)
    error = experiment.difference(result, target)
    assert trace
    assert all(t['after'] < t['before'] for t in trace)
    assert error['position_mm'] < 1e-8
    assert error['rotation_deg'] < 1e-8
    assert error['velocity_mps'] < 1e-10
    assert experiment.diag.state(previous) == previous_before
    assert np.array_equal(result.accel_bias, initial.accel_bias)
    assert np.array_equal(result.gyro_bias, initial.gyro_bias)


def test_motion_jacobians_match_actual_retraction_at_nonzero_residual():
    previous, target, samples, gravity = experiment.synthetic_case('general_rotation_acceleration')
    pim = gv.preintegrate_imu(samples, previous, {}, np.eye(4), gravity)
    current = gv._navigation_plus(target, experiment.PERTURBATION)
    checks = experiment.jacobian_check(gv._split_imu_factor(pim), previous, current)
    assert max(c['relative_frobenius'] for c in checks) < 1e-7


def test_midpoint_rotating_acceleration_error_converges_quadratically():
    errors = []
    for dt in (.001, .0001):
        previous, target, samples, gravity = experiment.synthetic_case('general_rotation_acceleration', dt)
        pim = gv.preintegrate_imu(samples, previous, {}, np.eye(4), gravity)
        residual = gv._split_imu_factor(pim).unwhitenedError(gv._imu_values(previous, target))
        errors.append(np.linalg.norm(residual[3:]))
    # The active preintegrator uses midpoint integration. Reducing dt by
    # ten gives a hundredfold reduction in its second-order global error.
    assert errors[0] < 1e-7
    assert 90. < errors[0]/errors[1] < 110.
