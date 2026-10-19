"""Full frontend GN steps with current target-normal derivatives."""

from types import SimpleNamespace

import gtsam
import numpy as np
import pytest

from gelslam.core.gtsam_vi import GtsamVisualInertialSolver
from gelslam.core.imu_navigation import gelsight_pose, gelsight_pose_jacobian
from gelslam.core.inertial import ImuSample
from gelslam.core.numpy_vi import DenseNormalResidual, NavigationState
from gelslam.core.split_pose import split_pose_plus


def frame():
    yy, xx = np.mgrid[:64, :64]
    normals = np.stack((.1 * np.sin(xx / 9), .1 * np.cos(yy / 11),
                        np.ones_like(xx)), axis=2).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)
    curvature = np.zeros((64, 64), dtype=np.float32)
    curvature[8:-8, 8:-8] = np.arange(48 * 48).reshape(48, 48) + 1
    return SimpleNamespace(N=normals, L=curvature,
                           H=np.zeros((64, 64), dtype=np.float32),
                           C=np.ones((64, 64), dtype=bool))


def solve(visual, pose, **config):
    previous = NavigationState(0., np.eye(4), np.zeros(3), np.zeros(3), np.zeros(3))
    current = previous.copy()
    current.timestamp, current.pose = .1, pose.copy()
    samples = [ImuSample(t, [0., 0., 9.81], [0., 0., 0.])
               for t in np.linspace(0., .1, 11)]
    solver = GtsamVisualInertialSolver({
        'gtsam_vi_normal_linearization': 'target_relinearized',
        'gtsam_vi_max_iterations': 1,
        'dense_normalflow_residual_weight': 1e8,
        'imu_residual_weight': 1e-8,
        'gtsam_vi_normal_huber_delta': 0.,
        'gtsam_vi_rotation_stop_rad': 1e-10,
        'gtsam_vi_translation_stop_m': 1e-10,
        **config,
    })
    result = solver.refine_current(visual, np.eye(4), previous, current, samples, np.eye(4))
    return solver, result


@pytest.mark.parametrize('fixed_previous', [False, True])
@pytest.mark.parametrize('fixed_bias', [False, True])
def test_cached_reference_direction_is_unused_from_first_iteration(fixed_previous, fixed_bias):
    image = frame()
    pose = np.eye(4)
    pose[0, 3] = .001
    visual = DenseNormalResidual(image, image, 1., np.eye(4), pose, 1500)
    # The cached reference derivative now points uphill for the actual image.
    visual.candidate_normal_jacobian *= -1
    solver, result = solve(visual, pose, frontend_fix_previous_state=fixed_previous,
                           frontend_fix_bias=fixed_bias)
    first, = solver.last_optimization_trace
    assert first['iteration'] == 0
    assert first['update_rule'] == 'full_gauss_newton'
    assert first['accepted_scale'] == 1.
    assert len(first['attempts']) == 1
    assert first['normal_jacobian_source'] == 'target_normal_map'
    assert first['accepted_cost'] < first['base_cost'] * .001
    assert abs(result.pose[0, 3]) < 1e-5
    if fixed_bias:
        assert np.array_equal(result.accel_bias, np.zeros(3))
        assert np.array_equal(result.gyro_bias, np.zeros(3))
    if not fixed_previous:
        assert solver.pending_marginal_prior is not None


def test_solver_relinearizes_target_derivatives_at_each_updated_pose(monkeypatch):
    image = frame()
    pose = np.eye(4)
    pose[0, 3] = .001
    visual = DenseNormalResidual(image, image, 1., np.eye(4), pose, 1500)
    visual.candidate_normal_jacobian *= -1
    evaluated_poses = []
    evaluate = visual.evaluate_relinearized

    def capture(reference, target):
        evaluated_poses.append(target.copy())
        return evaluate(reference, target)

    monkeypatch.setattr(visual, 'evaluate_relinearized', capture)
    solver, result = solve(visual, pose, gtsam_vi_max_iterations=3)
    assert result is not None
    traces = solver.last_optimization_trace
    assert len(traces) == 3
    assert all(t['normal_jacobian_source'] == 'target_normal_map' for t in traces)
    # One evaluation per GN step, then one at the retained prior's anchor.
    assert len(evaluated_poses) == len(traces) + 1
    assert np.array_equal(evaluated_poses[0], pose)
    assert not np.array_equal(evaluated_poses[0], evaluated_poses[1])
    assert not np.array_equal(evaluated_poses[1], evaluated_poses[2])
    assert np.array_equal(evaluated_poses[-1], result.pose)


def test_invalid_updated_measurement_fails_without_retry(monkeypatch):
    image = frame()
    pose = np.eye(4)
    visual = DenseNormalResidual(image, image, 1., pose, pose, 1500)
    # The full update has no valid contact measurements.
    monkeypatch.setattr(visual, 'residual_on_fixed_rows', lambda *args: None)
    solver, result = solve(visual, pose, gtsam_vi_max_iterations=50)
    trace, = solver.last_optimization_trace
    assert result is None
    assert trace['stop_reason'] == 'invalid_updated_residual'
    assert solver.last_refinement_failure == 'invalid_updated_residual'
    assert trace['accepted_scale'] == 1.
    assert len(trace['attempts']) == 1
    assert solver.pending_marginal_prior is None
    assert np.array_equal(pose, np.eye(4))


@pytest.mark.parametrize('fixed_z', [False, True])
def test_target_jacobian_matches_independent_derivative_with_rotated_mount(fixed_z):
    # A linear normal field has known spatial derivatives. Choose target sample
    # centres exactly on pixels to remove remap coordinate quantization here.
    yy, xx = np.mgrid[:64, :64]
    origin = np.array([.1, -.2, .9])
    dx, dy = np.array([.002, .001, -.0004]), np.array([-.001, .003, .0002])
    normals = (origin + xx[:, :, None] * dx + yy[:, :, None] * dy).astype(np.float32)
    image = frame()
    image.N = normals
    reference, target, extrinsic = np.eye(4), np.eye(4), np.eye(4)
    reference[:3, :3] = gtsam.Rot3.Expmap([.1, -.2, .15]).matrix()
    reference[:3, 3] = [.01, -.02, .03]
    target[:3, :3] = gtsam.Rot3.Expmap([-.12, .07, -.09]).matrix()
    target[:3, 3] = [.011, -.019, .032]
    extrinsic[:3, :3] = gtsam.Rot3.Expmap([.2, -.1, .3]).matrix()
    extrinsic[:3, 3] = [.003, -.002, .005]
    visual = DenseNormalResidual(image, image, .1, reference, target, 1500)
    pixels = np.array([(x, y) for x in (24, 30, 36) for y in (25, 29, 33, 37)])
    points = np.column_stack(((pixels - 31.5) * .0001, np.linspace(.001, .003, 12)))
    relative = np.linalg.inv(target) @ reference
    visual.candidate_points = (points - relative[:3, 3]) @ relative[:3, :3]
    visual.candidate_normals = np.tile([0., 0., 1.], (12, 1))
    if fixed_z:
        visual.fixed_reference_z_m = (np.linalg.inv(reference) @ target)[2, 3]
    residual, _, gel_jacobian = visual.evaluate_relinearized(reference, target)
    imu_jacobian = gel_jacobian @ gelsight_pose_jacobian(extrinsic)
    imu_target = target @ extrinsic

    def exact_residual(pose):
        gel = gelsight_pose(pose, extrinsic)
        if fixed_z:
            ref_to_gel = np.linalg.inv(reference) @ gel
            ref_to_gel[2, 3] = visual.fixed_reference_z_m
            gel = reference @ ref_to_gel
        transform = np.linalg.inv(gel) @ reference
        q = visual.candidate_points @ transform[:3, :3].T + transform[:3, 3]
        uv = q[:, :2] / .0001 + 31.5
        n = origin + uv[:, :1] * dx + uv[:, 1:] * dy
        return (n @ transform[:3, :3] - visual.candidate_normals).ravel()

    assert np.allclose(residual, exact_residual(imu_target), atol=1e-7)
    numerical = np.empty_like(imu_jacobian)
    for column in range(6):
        epsilon = 1e-6 if column < 3 else 1e-8
        step = np.eye(6)[column] * epsilon
        numerical[:, column] = (
            exact_residual(split_pose_plus(imu_target, step))
            - exact_residual(split_pose_plus(imu_target, -step))
        ) / (2 * epsilon)
    assert np.allclose(imu_jacobian, numerical, atol=4e-4, rtol=1e-4)


def test_relinearization_uses_current_target_samples_and_current_contact_rows():
    image = frame()
    target = frame()
    visual = DenseNormalResidual(image, target, 1., np.eye(4), np.eye(4), 1500)
    before = visual.evaluate_relinearized(np.eye(4), np.eye(4))[2]
    shifted = np.eye(4)
    shifted[0, 3] = .003
    after = visual.evaluate_relinearized(np.eye(4), shifted)[2]
    assert not np.allclose(before, after)
    target.C[:, :32] = False
    old_residual = visual.evaluate(np.eye(4), shifted)[0]
    old_rows = visual.last_active_rows.copy()
    new_residual, _, new_jacobian = visual.evaluate_relinearized(np.eye(4), shifted)
    assert np.array_equal(old_rows, visual.last_active_rows)
    assert len(new_residual) < len(after)
    assert np.array_equal(new_residual, old_residual)
    assert new_jacobian.shape == (len(new_residual), 6)
