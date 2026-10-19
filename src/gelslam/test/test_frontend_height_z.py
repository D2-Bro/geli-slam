"""Conditional height measurements in the contact frontend and retained prior."""

from types import SimpleNamespace

import numpy as np
import pytest

from gelslam.core.gtsam_vi import GtsamVisualInertialSolver
from gelslam.core.imu_navigation import gelsight_pose, gelsight_pose_jacobian, imu_pose
from gelslam.core.inertial import ImuSample
from gelslam.core.numpy_vi import DenseNormalResidual, NavigationState, so3_exp
from gelslam.core.split_pose import split_pose_plus


def visual_pair(height_m=0., pose=None, samples=200):
    yy, xx = np.mgrid[:40, :40]
    contact = np.ones((40, 40), dtype=bool)
    normals = np.zeros((40, 40, 3), dtype=np.float32)
    normals[:, :, 2] = 1.
    curvature = np.maximum(400 - (xx - 20)**2 - (yy - 20)**2, 0).astype(float)
    reference = SimpleNamespace(N=normals, C=contact, L=curvature,
                                H=np.zeros((40, 40), dtype=np.float32))
    target = SimpleNamespace(N=normals, C=contact.copy(), L=curvature,
                             H=np.full((40, 40), height_m / .001, dtype=np.float32))
    pose = np.eye(4) if pose is None else pose
    # ppmm is mm/pixel, so one height unit here is 1 mm.
    return DenseNormalResidual(reference, target, 1., pose, pose, samples)


def solve(visual, **overrides):
    solver = GtsamVisualInertialSolver({
        'gtsam_vi_height_z_enabled': True,
        'gtsam_vi_height_z_sigma_m': 1e-4,
        'gtsam_vi_max_iterations': 4,
        'gtsam_vi_translation_stop_m': 1e-10,
        'gtsam_vi_rotation_stop_rad': 1e-10,
        'imu_residual_weight': 1e-8,
        **overrides,
    })
    previous = NavigationState(0., np.eye(4), np.zeros(3), np.zeros(3), np.zeros(3))
    samples = [ImuSample(t, [0., 0., 9.81], [0., 0., 0.])
               for t in np.linspace(0., .1, 11)]
    current, pim = solver.predict_current(previous, samples, np.eye(4))
    result = solver.refine_current(visual, np.eye(4), previous, current, samples,
                                    np.eye(4), preintegrated=pim)
    return solver, result


@pytest.mark.parametrize('fixed_normal_z', [False, True])
def test_height_residual_follows_actual_local_z_in_rotated_world(fixed_normal_z):
    pose = np.eye(4)
    pose[:3, :3] = so3_exp(np.array([.7, -.2, .3]))
    pose[:3, 3] = [.1, -.2, .4]
    visual = visual_pair(.0003, pose)
    if fixed_normal_z:
        visual.set_fixed_reference_z(0.)
    original_count = len(visual)
    for dz in (0., -.0002, -.0003):
        target = split_pose_plus(pose, np.array([0., 0., 0., 0., 0., dz]))
        error = visual.target_z_height_correction(pose, target)
        assert np.isclose(error, .0003 + dz, atol=1e-10)
    assert len(visual) == original_count
    eps = 1e-7
    step = np.array([0., 0., 0., 0., 0., eps])
    derivative = (visual.target_z_height_correction(pose, split_pose_plus(pose, step))
                  - visual.target_z_height_correction(pose, split_pose_plus(pose, -step))) / (2 * eps)
    assert np.isclose(derivative, 1., atol=1e-8)


def test_height_linearization_maps_conditional_gel_z_to_offset_imu():
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = so3_exp(np.array([.2, -.4, .1]))
    extrinsic[:3, 3] = [.012, -.008, -.04]
    gel = np.eye(4)
    gel[:3, :3] = so3_exp(np.array([.3, .2, -.5]))
    gel[:3, 3] = [.1, .2, -.3]
    imu = imu_pose(gel, extrinsic)
    visual = visual_pair(.0003, gel)
    solver = GtsamVisualInertialSolver({'gtsam_vi_height_z_enabled': True})
    local_A, b, cost = solver._height_z_linearization(visual, gel, gel, np.eye(6))
    assert np.array_equal(local_A[0, :5], np.zeros(5))
    assert np.isclose(local_A[0, 5], 1. / solver.height_z_sigma)
    A, _, _ = solver._height_z_linearization(
        visual, gel, gel, gelsight_pose_jacobian(extrinsic))
    # Independently differentiate the conditional model: freeze sensor Z in
    # world coordinates, and measure only the displaced GelSight origin.
    eps = 1e-7
    numeric = np.empty(6)
    for column in range(6):
        step = np.eye(6)[column] * eps
        plus = gelsight_pose(split_pose_plus(imu, step), extrinsic)
        minus = gelsight_pose(split_pose_plus(imu, -step), extrinsic)
        numeric[column] = gel[:3, 2] @ (plus[:3, 3] - minus[:3, 3]) / (2 * eps)
    assert np.allclose(A[0] * solver.height_z_sigma, numeric, atol=1e-8)
    assert np.isclose(cost, .5 * float(b @ b))


@pytest.mark.parametrize('fixed_previous', [False, True])
@pytest.mark.parametrize('fixed_bias', [False, True])
def test_enabled_height_changes_target_z_and_is_recomputed(fixed_previous, fixed_bias, monkeypatch):
    visual = visual_pair(.0002)
    seen = []
    evaluate = visual.target_z_height_correction

    def observe(reference, target):
        error = evaluate(reference, target)
        seen.append((target.copy(), error))
        return error

    monkeypatch.setattr(visual, 'target_z_height_correction', observe)
    solver, result = solve(visual, frontend_fix_previous_state=fixed_previous,
                           frontend_fix_bias=fixed_bias)
    assert result is not None
    assert np.isclose(result.pose[2, 3], -.0002, atol=1e-7)
    assert np.allclose(result.pose[:2, 3], 0., atol=1e-9)
    assert np.allclose(result.pose[:3, :3], np.eye(3), atol=1e-9)
    assert abs(seen[-1][1]) < abs(seen[0][1]) * .001
    traces = solver.last_optimization_trace
    # Initial state plus each update. Reuse the updated measurement at the
    # next iteration and when retaining the final state's marginal prior.
    assert len(seen) == len(traces) + 1
    assert all(t['height_z_enabled'] for t in traces)
    for trace in traces:
        assert np.isclose(trace['base_cost'], sum(trace['base_' + name + '_cost']
                          for name in ('normal', 'imu', 'bias', 'prior', 'height')))
        assert np.isclose(trace['accepted_cost'], sum(trace['accepted_' + name + '_cost']
                          for name in ('normal', 'imu', 'bias', 'prior', 'height')))
    if fixed_previous:
        assert solver.pending_marginal_prior is None
    else:
        assert solver.pending_marginal_prior.matches(result)
        solver.commit_marginal_prior(result)
        assert solver.marginal_prior.matches(result)


@pytest.mark.parametrize('fixed_bias', [False, True])
def test_height_information_is_retained_once_without_pixel_count_scaling(fixed_bias):
    expected = np.zeros((15, 15))
    expected[5, 5] = 1e8
    for sample_count in (100, 400):
        off, state_off = solve(visual_pair(samples=sample_count), gtsam_vi_height_z_enabled=False,
                               frontend_fix_bias=fixed_bias)
        before = off.pending_marginal_prior.A.T @ off.pending_marginal_prior.A
        on, state_on = solve(visual_pair(samples=sample_count), frontend_fix_bias=fixed_bias)
        assert np.allclose(state_on.pose, state_off.pose, atol=1e-12)
        after = on.pending_marginal_prior.A.T @ on.pending_marginal_prior.A
        assert np.allclose(after - before, expected, atol=1e-5, rtol=1e-8)


def test_disabled_height_never_evaluates_measurement(monkeypatch):
    visual = visual_pair(.0002)

    def unexpected(*args):
        raise AssertionError('disabled height factor was evaluated')

    monkeypatch.setattr(visual, 'target_z_height_correction', unexpected)
    solver, result = solve(visual, gtsam_vi_height_z_enabled=False)
    assert result is not None
    assert np.allclose(result.pose, np.eye(4), atol=1e-12)
    assert all(t['base_height_cost'] == 0. and t['accepted_height_cost'] == 0.
               for t in solver.last_optimization_trace)
    assert GtsamVisualInertialSolver({}).height_z_enabled is False


def test_insufficient_height_contacts_returns_no_measurement():
    visual = visual_pair()
    visual.tar.C[:] = False
    assert visual.target_z_height_correction(np.eye(4), np.eye(4)) is None


@pytest.mark.parametrize('bad', [None, np.nan, np.inf])
def test_invalid_height_rejects_refinement_without_retaining_prior(bad, monkeypatch):
    visual = visual_pair()
    monkeypatch.setattr(visual, 'target_z_height_correction', lambda *args: bad)
    solver, result = solve(visual)
    assert result is None
    assert solver.last_refinement_failure == 'invalid_height_residual'
    assert solver.pending_marginal_prior is None


@pytest.mark.parametrize('value', ['false', 'true', None, 0, 1])
def test_height_flag_requires_boolean(value):
    with pytest.raises(ValueError, match='gtsam_vi_height_z_enabled'):
        GtsamVisualInertialSolver({'gtsam_vi_height_z_enabled': value})


@pytest.mark.parametrize('value', [0., -1., np.nan, np.inf])
def test_height_sigma_requires_positive_finite_metres(value):
    with pytest.raises(ValueError, match='gtsam_vi_height_z_sigma_m'):
        GtsamVisualInertialSolver({'gtsam_vi_height_z_sigma_m': value})
