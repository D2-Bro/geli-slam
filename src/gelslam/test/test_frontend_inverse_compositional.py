"""Reference IC derivatives, navigation coordinates and joint-solver routing."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import lstsq
from scipy.spatial.transform import Rotation

import normalflow.registration as origin
import gelslam.core.numpy_vi as normal
from gelslam.core.gtsam_vi import GtsamVisualInertialSolver
from gelslam.core.imu_navigation import gelsight_pose, gelsight_pose_jacobian
from gelslam.core.inertial import ImuSample
from gelslam.core.numpy_vi import DenseNormalResidual, NavigationState, adjoint
from gelslam.core.split_pose import split_pose_plus


def frame():
    yy, xx = np.mgrid[:64, :64]
    normals = np.stack((.1 * np.sin(xx / 9), .1 * np.cos(yy / 11),
                        np.ones_like(xx)), axis=2).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    curvature = np.zeros((64, 64), dtype=np.float32)
    curvature[8:-8, 8:-8] = np.arange(48 * 48).reshape(48, 48) + 1
    return SimpleNamespace(N=normals, L=curvature, C=np.ones((64, 64), dtype=bool),
                           H=np.zeros((64, 64), dtype=np.float32))


def pose(rotation, translation):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    result[:3, 3] = translation
    return result


def affine_frame():
    image = frame()
    yy, xx = np.mgrid[:64, :64]
    gradient = np.array([[9., -7.], [5., 11.], [-3., 4.]])
    origin_normal = np.array([.1, -.2, .9])
    # ppmm=.1 is mm/pixel. This smooth, unnormalized field isolates derivatives
    # without cv2.remap's subpixel coordinate quantization.
    x, y = (xx - 31.5) * .0001, (yy - 31.5) * .0001
    image.N = (origin_normal + x[..., None] * gradient[:, 0]
               + y[..., None] * gradient[:, 1]).astype(np.float32)
    image.H[:] = 2.
    return image, origin_normal, gradient


def test_ic_rows_residual_and_five_dof_step_match_origin(monkeypatch):
    reference, target = frame(), frame()
    target.N = np.roll(target.N, 1, axis=1)
    # Original NormalFlow passes remap coordinates through without casting;
    # its tracker supplies a float32 initial transform.
    transform = pose([.004, -.003, .01], [.0002, -.0001, .0003]).astype(np.float32)
    visual = DenseNormalResidual(reference, target, 1., np.eye(4), np.linalg.inv(transform), 1500)
    captured = {}

    class FirstStep(Exception):
        pass

    def capture(A, b, **kwargs):
        captured.update(A=A.copy(), b=b.copy(), step=lstsq(A, b, **kwargs)[0])
        raise FirstStep

    monkeypatch.setattr(origin, 'lstsq', capture)
    with pytest.raises(FirstStep):
        origin.normalflow(reference.N, reference.C, reference.H, reference.L,
                          target.N, target.C, target.H, target.L,
                          transform, 1., n_samples=1500)
    residual, reference_j, _ = visual.evaluate_ic(np.eye(4), np.linalg.inv(transform))
    assert np.allclose(residual, captured['b'], atol=1e-7)
    assert np.array_equal(reference_j[:, :5], captured['A'])
    assert np.array_equal(reference_j[:, 5], np.zeros(len(residual)))
    assert np.allclose(lstsq(reference_j[:, :5], residual, lapack_driver='gelsy')[0],
                       captured['step'], atol=1e-7)


def test_cached_j_is_derivative_of_inverse_warped_reference_at_identity():
    image, n0, gradient = affine_frame()
    visual = DenseNormalResidual(image, image, .1, np.eye(4), np.eye(4), 1500)
    points = visual.candidate_points

    def inverse_warped_reference(step):
        rotation = Rotation.from_euler('xyz', step[:3]).as_matrix()
        moved = points @ rotation.T + np.r_[step[3:], 0.]
        return (n0 + moved[:, :2] @ gradient.T) @ rotation

    numerical = np.stack([
        (inverse_warped_reference(np.eye(5)[j] * 1e-7)
         - inverse_warped_reference(-np.eye(5)[j] * 1e-7)) / 2e-7 for j in range(5)
    ], axis=-1)
    expected = visual.candidate_normal_jacobian.transpose(2, 0, 1)
    assert np.allclose(expected, numerical, atol=7e-4, rtol=1e-4)


@pytest.mark.parametrize('fixed_z', [False, True])
def test_ic_surrogate_chain_matches_imu_perturbation_with_rotated_offset_mount(fixed_z):
    image, n0, gradient = affine_frame()
    reference = pose([.1, -.2, .15], [.01, -.02, .03])
    target = reference @ pose([.03, -.02, .04], [.0001, -.0001, .0002])
    mount = pose([.2, -.1, .3], [.003, -.002, .005])
    imu_target = target @ mount
    visual = DenseNormalResidual(image, image, .1, reference, target, 1500)
    if fixed_z:
        visual.set_fixed_reference_z((np.linalg.inv(reference) @ target)[2, 3])
    residual, _, gel_j = visual.evaluate_ic(reference, target)
    expected = gel_j @ gelsight_pose_jacobian(mount)
    rows = visual.last_active_rows.copy()
    points = visual.candidate_points[rows]
    base = np.linalg.inv(visual._normal_target_pose(reference, target)) @ reference

    def surrogate(imu_pose):
        gel = visual._normal_target_pose(reference, gelsight_pose(imu_pose, mount))
        moved_transform = np.linalg.inv(gel) @ reference
        # T_new = T inv(d), hence d = inv(T_new) T. Differentiate this IC
        # surrogate, not the forward residual at a misaligned target pose.
        correction = np.linalg.inv(moved_transform) @ base
        moved = points @ correction[:3, :3].T + correction[:3, 3]
        field = (n0 + moved[:, :2] @ gradient.T) @ correction[:3, :3]
        base_field = n0 + points[:, :2] @ gradient.T
        return residual - (field - base_field).reshape(-1)

    numerical = np.column_stack([
        (surrogate(split_pose_plus(imu_target, np.eye(6)[j] * 1e-7))
         - surrogate(split_pose_plus(imu_target, -np.eye(6)[j] * 1e-7))) / 2e-7
        for j in range(6)
    ])
    assert np.allclose(expected, numerical, atol=8e-4, rtol=2e-4)


def test_navigation_update_matches_origin_inverse_composition_to_first_order():
    reference = pose([.1, -.2, .15], [.01, -.02, .03])
    target = reference @ pose([.03, -.02, .04], [.001, -.002, .003])
    mount = pose([.2, -.1, .3], [.003, -.002, .005])
    transform = np.linalg.inv(target) @ reference
    delta = np.array([.12, -.08, .04, .002, -.001, 0.])
    errors = []
    for scale in (.1, .05):
        step = delta * scale
        correction = np.eye(4)
        correction[:3, :3] = Rotation.from_euler('xyz', step[:3]).as_matrix()
        correction[:3, 3] = step[3:]
        origin_updated = transform @ np.linalg.inv(correction)
        imu_step = np.linalg.solve(gelsight_pose_jacobian(mount), adjoint(transform) @ step)
        updated = gelsight_pose(split_pose_plus(target @ mount, imu_step), mount)
        actual = np.linalg.inv(updated) @ reference
        errors.append(np.linalg.norm(actual - origin_updated))
    # The current split retraction and origin's Euler increments agree at
    # first order; their finite-step difference must shrink quadratically.
    assert errors[1] < .3 * errors[0]
    assert errors[0] < 2e-4


@pytest.mark.parametrize('fixed_previous', [False, True])
def test_ic_reuses_reference_j_but_updates_pose_and_prior(monkeypatch, fixed_previous):
    image = frame()
    initial = np.eye(4)
    initial[0, 3] = .001
    evaluations, jacobian_calls = [], []
    original_get_j = normal.get_J

    def get_j(*args):
        jacobian_calls.append(1)
        return original_get_j(*args)

    monkeypatch.setattr(normal, 'get_J', get_j)
    visual = DenseNormalResidual(image, image, 1., np.eye(4), initial, 1500)
    cached = visual.candidate_normal_jacobian.copy()
    evaluate = visual.evaluate_ic

    def capture(reference, target):
        evaluations.append(target.copy())
        return evaluate(reference, target)

    def unexpected(*args):
        raise AssertionError('IC solve called the target-map derivative')

    monkeypatch.setattr(visual, 'evaluate_ic', capture)
    monkeypatch.setattr(visual, 'evaluate_relinearized', unexpected)
    previous = NavigationState(0., np.eye(4), np.zeros(3), np.zeros(3), np.zeros(3))
    current = previous.copy()
    current.timestamp, current.pose = .1, initial.copy()
    samples = [ImuSample(t, [0., 0., 9.81], [0., 0., 0.]) for t in np.linspace(0., .1, 11)]
    solver = GtsamVisualInertialSolver({
        'gtsam_vi_max_iterations': 3, 'dense_normalflow_residual_weight': 1e8,
        'imu_residual_weight': 1e-8, 'gtsam_vi_normal_huber_delta': 0.,
        'gtsam_vi_rotation_stop_rad': 1e-10, 'gtsam_vi_translation_stop_m': 1e-10,
        'frontend_fix_previous_state': fixed_previous,
    })
    result = solver.refine_current(visual, np.eye(4), previous, current, samples, np.eye(4))
    assert result is not None
    assert abs(result.pose[0, 3]) < 1e-4
    assert len(jacobian_calls) == 1
    assert np.array_equal(visual.candidate_normal_jacobian, cached)
    assert len(evaluations) == len(solver.last_optimization_trace) + (not fixed_previous)
    assert not np.array_equal(evaluations[0], evaluations[1])
    assert all(t['normal_jacobian_source'] == 'reference_normal_map_ic'
               and t['normal_linearization'] == 'inverse_compositional'
               for t in solver.last_optimization_trace)
    if not fixed_previous:
        assert np.array_equal(evaluations[-1], result.pose)
        assert solver.pending_marginal_prior.matches(result)
    else:
        assert solver.pending_marginal_prior is None


def test_ic_reselects_contact_and_maps_fixed_reference_j_at_each_pose():
    reference, target = frame(), frame()
    visual = DenseNormalResidual(reference, target, 1., np.eye(4), np.eye(4), 1500)
    cached = visual.candidate_normal_jacobian.copy()
    before, _, J_before = visual.evaluate_ic(np.eye(4), np.eye(4))
    moved = pose([0., 0., .02], [.001, 0., 0.])
    after, _, J_after = visual.evaluate_ic(np.eye(4), moved)
    assert not np.allclose(before, after)
    assert not np.allclose(J_before, J_after)
    target.C[:, :32] = False
    residual, _, J = visual.evaluate_ic(np.eye(4), moved)
    assert len(residual) < len(after)
    assert J.shape == (len(residual), 6)
    assert np.array_equal(cached, visual.candidate_normal_jacobian)
    # A zero reference-Z column becomes a rotated null direction in target
    # coordinates. It must not be replaced by an arbitrary world/IMU Z column.
    transform = np.linalg.inv(moved)
    null_target = adjoint(transform) @ np.eye(6)[5]
    assert np.allclose(J @ null_target, 0., atol=1e-10)


@pytest.mark.parametrize('value', [None, '', 'ic', 1, True, [], {}])
def test_unknown_normal_linearization_is_rejected(value):
    with pytest.raises(ValueError, match='gtsam_vi_normal_linearization'):
        GtsamVisualInertialSolver({'gtsam_vi_normal_linearization': value})


def test_reference_ic_is_the_default():
    assert GtsamVisualInertialSolver({}).normal_linearization == 'inverse_compositional'
