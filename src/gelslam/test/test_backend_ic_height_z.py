"""Frontend IC/Z models in a two-pose combined backend GelSight factor."""
import gtsam
import numpy as np
import pytest

from backend_test_support import frame
from gelslam.core.backend_factors import TactilePair
from gelslam.core.gtsam_vi import GtsamVisualInertialSolver
from gelslam.core.imu_navigation import gelsight_pose, imu_pose
from gelslam.core.numpy_vi import adjoint


def observation_pair(seed=0, normal_scale=1., height_scale=1.):
    first, second = frame(), frame(1)
    yy, xx = np.mgrid[:40, :40]
    nx, ny = .2 * np.sin(xx / 5), .2 * np.cos(yy / 7)
    first.N = np.dstack((nx, ny, np.sqrt(1 - nx**2 - ny**2))).astype(np.float32)
    second.N = (first.N * normal_scale).astype(np.float32)
    second.H = ((.03 * xx + .01 * yy**2) * height_scale).astype(np.float32)
    second.C[15:18, 9:12] = False
    mount = gtsam.Pose3(gtsam.Rot3.RzRyRx(.2, -.1, .3), [.02, -.01, .03]).matrix()
    rng = np.random.default_rng(seed)
    ref = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([.2, -.3, .1])), [.1, -.2, .3]).matrix()
    relative = gtsam.Pose3(gtsam.Rot3.Expmap(rng.normal(0, .01, 3)),
                           rng.normal(0, .0002, 3)).matrix()
    tar = ref @ relative
    pair = TactilePair(first, second, ref, tar, mount,
                       {'tactile_normal_sigma': .02,
                        'tactile_height_sigma_m': .0001})
    values = gtsam.Values()
    values.insert(0, gtsam.Pose3(imu_pose(ref, mount)))
    values.insert(1, gtsam.Pose3(imu_pose(tar, mount)))
    return pair, values


@pytest.mark.parametrize('seed', range(6))
def test_backend_normal_geometry_matches_frontend_reference_ic_for_both_poses(seed):
    pair, values = observation_pair(seed)
    factor = pair.factor(0, 1)
    A, _ = factor.linearize(values).jacobian()
    projection = pair._projection_cache[2]
    ref, tar = [gelsight_pose(values.atPose3(key).matrix(), pair.extrinsic) for key in (0, 1)]
    _, reference, target = pair.visual.evaluate_ic(ref, tar)
    raw = pair._normal_pose_jacobians(projection)
    np.testing.assert_array_equal(raw[0], reference)
    np.testing.assert_array_equal(raw[1], target)
    rows = np.repeat(projection.shared, 3)
    normal_A = A[:-1][rows]
    scale = pair.normal_sigma * np.sqrt(np.count_nonzero(projection.shared))
    expected = np.column_stack((reference @ pair.gel_jacobian,
                                target @ pair.gel_jacobian)) / scale
    np.testing.assert_allclose(normal_A, expected, rtol=1e-12, atol=1e-12)
    # IC's zero reference-Z column is preserved before chart/extrinsic mapping.
    np.testing.assert_array_equal(raw[0][:, 5], 0.)


def test_target_normal_field_changes_residual_but_not_reference_ic_geometry():
    first, values = observation_pair(normal_scale=1.)
    second, _ = observation_pair(normal_scale=2.)
    f1, f2 = first.factor(0, 1), second.factor(0, 1)
    A1, _ = f1.linearize(values).jacobian()
    A2, _ = f2.linearize(values).jacobian()
    np.testing.assert_array_equal(A1, A2)
    assert not np.allclose(f1.whitenedError(values)[:-1], f2.whitenedError(values)[:-1])


@pytest.mark.parametrize('seed', range(4))
def test_height_target_row_is_frontend_local_z_with_reference_chart_transport(seed):
    pair, values = observation_pair(seed)
    A, _ = pair.factor(0, 1).linearize(values).jacobian()
    projection = pair._projection_cache[2]
    solver = GtsamVisualInertialSolver({'gtsam_vi_height_z_enabled': True,
                                       'gtsam_vi_height_z_sigma_m': pair.height_sigma})
    np.testing.assert_array_equal(A[-1:, 6:], solver._height_z_jacobian(pair.gel_jacobian))
    reference, target = pair._height_pose_jacobians(projection)
    np.testing.assert_array_equal(target, [[0., 0., 0., 0., 0., 1.]])
    np.testing.assert_allclose(reference + target @ adjoint(projection.relative), 0., atol=1e-15)
    np.testing.assert_allclose(A[-1:, :6],
                               (reference / pair.height_sigma) @ pair.gel_jacobian, atol=1e-12)
    # Both endpoint charts must describe the same relative constraint. This
    # reference transport can contain rotation/XY terms even though the target
    # GelSight row is local-Z only. External calibration adds IMU lever arms.
    assert np.linalg.norm(reference[0, :5]) > 0.
    assert np.linalg.norm(A[-1, 6:9]) > 0.


def test_height_image_slope_changes_residual_but_not_conditional_z_row():
    first, values = observation_pair(height_scale=0.)
    second, _ = observation_pair(height_scale=10.)
    f1, f2 = first.factor(0, 1), second.factor(0, 1)
    A1, _ = f1.linearize(values).jacobian()
    A2, _ = f2.linearize(values).jacobian()
    np.testing.assert_array_equal(A1[-1], A2[-1])
    assert not np.isclose(f1.whitenedError(values)[-1], f2.whitenedError(values)[-1])


def test_two_pose_ic_z_rows_have_zero_response_to_a_common_world_rigid_motion():
    pair, values = observation_pair()
    A, _ = pair.factor(0, 1).linearize(values).jacobian()
    reference_motion = adjoint(np.linalg.inv(values.atPose3(0).matrix()))
    target_motion = adjoint(np.linalg.inv(values.atPose3(1).matrix()))
    np.testing.assert_allclose(A[:, :6] @ reference_motion + A[:, 6:] @ target_motion,
                               0., atol=2e-11)


def test_single_factor_with_z_only_height_matches_separate_normal_height_rows():
    pair, values = observation_pair()
    combined = pair.factor(0, 1).linearize(values)
    A, b = combined.jacobian()
    split = gtsam.GaussianFactorGraph()
    for rows in (slice(None, -1), slice(-1, None)):
        rhs = b[rows]
        split.add(gtsam.JacobianFactor(0, A[rows, :6], 1, A[rows, 6:], rhs,
                                       gtsam.noiseModel.Unit.Create(len(rhs))))
    combined_graph = gtsam.GaussianFactorGraph()
    combined_graph.add(combined)
    H1, g1 = combined_graph.hessian()
    H2, g2 = split.hessian()
    np.testing.assert_allclose(H1, H2, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(g1, g2, rtol=1e-12, atol=1e-10)
    delta = gtsam.VectorValues()
    delta.insert(0, np.arange(6) * 1e-5)
    delta.insert(1, -np.arange(6) * 1e-5)
    assert combined_graph.error(delta) == pytest.approx(split.error(delta), rel=1e-14)
