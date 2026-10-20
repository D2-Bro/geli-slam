"""Backend normal/height projection reuse and pose-change invalidation."""
from unittest.mock import patch

import gtsam
import cv2
import numpy as np
import pytest
from normalflow.utils import wide_remap

from backend_test_support import frame
import gelslam.core.backend_factors as backend_factors
from gelslam.core.backend_factors import TactilePair
from gelslam.core.imu_navigation import imu_pose
from gelslam.core.split_pose import split_pose_plus


def make_pair():
    first, second = frame(), frame(1)
    yy, xx = np.mgrid[:40, :40]
    second.N = np.stack((.005 * xx, .003 * yy, np.ones_like(xx)), axis=2)
    second.H = .02 * xx + .01 * yy
    extrinsic = gtsam.Pose3(gtsam.Rot3.RzRyRx(.1, -.2, .3),
                          np.array([.02, -.01, .03])).matrix()
    pair = TactilePair(first, second, np.eye(4), np.eye(4), extrinsic,
                       {'tactile_normal_sigma': .02})
    values = gtsam.Values()
    for key in (0, 1):
        values.insert(key, gtsam.Pose3(imu_pose(np.eye(4), extrinsic)))
    return pair, values


@pytest.mark.parametrize('changed_key', [0, 1])
def test_error_and_linearization_share_projection_and_refresh_for_either_pose(changed_key):
    pair, values = make_pair()
    graph = gtsam.NonlinearFactorGraph()
    for factor in (pair.factor(0, 1),):
        graph.add(factor)
    original = gtsam.Values(values)
    with patch.object(pair, '_compute_projection', wraps=pair._compute_projection) as compute:
        original_error = graph.error(values)
        graph.linearize(values)
        graph.error(gtsam.Values(values))  # Equal matrices, different Values object.
        assert compute.call_count == 1

        # This is far below np.allclose's default tolerance. It must still miss.
        step = np.array([0., 0., 0., 1e-12, 0., 0.])
        values.update(changed_key, gtsam.Pose3(
            split_pose_plus(values.atPose3(changed_key).matrix(), step)))
        graph.error(values)
        graph.linearize(values)
        assert compute.call_count == 2

        # A rejected LM candidate can return to the previous accepted state.
        assert graph.error(original) == original_error
        graph.linearize(original)
        assert compute.call_count == 3


def test_cached_pose_keys_are_copies_and_only_the_latest_projection_is_retained():
    pair, _ = make_pair()
    reference, target = np.eye(4), np.eye(4)
    initial = pair._projection(reference, target)
    reference[0, 3] = .0001
    moved_reference = pair._projection(reference, target)
    assert moved_reference is not initial
    target[1, 3] = .0002
    moved_target = pair._projection(reference, target)
    assert moved_target is not moved_reference
    assert pair._projection(reference.copy(), target.copy()) is moved_target
    assert pair._projection_cache[2] is moved_target
    assert not np.shares_memory(pair._projection_cache[0], reference)
    assert not np.shares_memory(pair._projection_cache[1], target)


@pytest.mark.parametrize('step', [np.zeros(6),
                                np.array([.01, -.02, .03, .0002, -.0003, .0001]),
                                np.array([0., 0., 0., 1., 0., 0.])])
def test_shared_projection_preserves_both_factor_residuals_and_jacobians(step):
    pair, values = make_pair()
    uncached_pair, _ = make_pair()
    uncached_pair._projection = uncached_pair._compute_projection
    values.update(1, gtsam.Pose3(split_pose_plus(values.atPose3(1).matrix(), step)))
    cached_factors = (pair.factor(0, 1),)
    uncached_factors = (uncached_pair.factor(0, 1),)
    for cached, uncached in zip(cached_factors, uncached_factors):
        np.testing.assert_array_equal(cached.whitenedError(values), uncached.whitenedError(values))
        cached_A, cached_b = cached.linearize(values).jacobian()
        uncached_A, uncached_b = uncached.linearize(values).jacobian()
        np.testing.assert_array_equal(cached_A, uncached_A)
        np.testing.assert_array_equal(cached_b, uncached_b)


def test_residual_only_requests_skip_both_pose_jacobians():
    pair, values = make_pair()
    graph = gtsam.NonlinearFactorGraph()
    factors = (pair.factor(0, 1),)
    for factor in factors:
        graph.add(factor)
    residuals = [factor.whitenedError(values) for factor in factors]
    expected_error = sum(.5 * r @ r for r in residuals)
    with patch.object(pair, '_normal_pose_jacobians', side_effect=AssertionError('Jacobian requested')), \
         patch.object(pair, '_height_pose_jacobians', side_effect=AssertionError('Jacobian requested')):
        assert graph.error(values) == pytest.approx(expected_error)
        for factor, expected in zip(factors, residuals):
            np.testing.assert_array_equal(factor.whitenedError(values), expected)
        # Ensure the guard really detects the Jacobian path.
        with pytest.raises(AssertionError, match='Jacobian requested'):
            graph.linearize(values)
    for factor, residual in zip(factors, residuals):
        _, rhs = factor.linearize(values).jacobian()
        np.testing.assert_array_equal(-rhs, residual)


def test_large_normal_residuals_use_mean_squared_cost_and_unweighted_ic_rows():
    pair, values = make_pair()
    factor = pair.factor(0, 1)
    residual = factor.whitenedError(values)
    projection = pair._projection_cache[2]
    mask = projection.shared
    raw = (projection.normals[mask] @ projection.relative[:3, :3]
           - pair.visual.candidate_normals[mask]) / pair.normal_sigma
    assert np.max(np.linalg.norm(raw, axis=1)) > .4
    expected = raw.reshape(-1) / np.sqrt(mask.sum())
    rows = np.repeat(mask, 3)
    np.testing.assert_allclose(residual[:-1][rows], expected, atol=1e-14)
    assert .5 * residual[:-1] @ residual[:-1] == pytest.approx(.5 * np.mean(np.sum(raw**2, axis=1)))
    A, _ = factor.linearize(values).jacobian()
    for slot, J in enumerate(pair._normal_pose_jacobians(projection)):
        np.testing.assert_allclose(A[:-1][rows, 6*slot:6*(slot+1)],
            J @ pair.gel_jacobian / (pair.normal_sigma * np.sqrt(mask.sum())), atol=1e-12)


def test_gelsight_factor_shares_pose_conversion_and_interpolation_coordinates():
    pair, values = make_pair()
    factor = pair.factor(0, 1)
    assert list(factor.keys()) == [0, 1]
    assert factor.dim() == 3 * pair.count + 1
    with patch.object(backend_factors, 'gelsight_pose', wraps=backend_factors.gelsight_pose) as pose, \
         patch.object(pair, '_sample_coordinates', wraps=pair._sample_coordinates) as coordinates, \
         patch.object(pair, '_projection_from_values', wraps=pair._projection_from_values) as prepare:
        factor.whitenedError(values)
        assert prepare.call_count == 1  # One callback prepares both residual blocks.
        factor.linearize(values)
        assert prepare.call_count == 2
        assert pose.call_count == 2  # One conversion for each endpoint, reused.
        assert coordinates.call_count == 1
        moved = split_pose_plus(values.atPose3(0).matrix(), [0., 0., 0., 1e-12, 0., 0.])
        values.update(0, gtsam.Pose3(moved))
        factor.whitenedError(values)
        assert pose.call_count == 4
        assert coordinates.call_count == 2


def test_residual_evaluation_defers_ic_and_z_pose_mapping_until_linearization():
    pair, values = make_pair()
    factor = pair.factor(0, 1)
    with patch.object(backend_factors, 'normal_ic_pose_jacobians',
                      wraps=backend_factors.normal_ic_pose_jacobians) as normal, \
         patch.object(backend_factors, 'adjoint', wraps=backend_factors.adjoint) as height:
        before = factor.whitenedError(values)
        projection = pair._projection_cache[2]
        assert projection.normal_jacobians is None
        assert projection.height_jacobians is None
        assert normal.call_count == height.call_count == 0

        factor.linearize(values)
        assert normal.call_count == height.call_count == 1
        assert projection.normal_jacobians is not None
        assert projection.height_jacobians is not None
        np.testing.assert_array_equal(factor.whitenedError(values), before)
        factor.linearize(values)
        assert normal.call_count == height.call_count == 1

        values.update(1, gtsam.Pose3(split_pose_plus(
            values.atPose3(1).matrix(), [0., 0., 0., 1e-7, 0., 0.])))
        factor.whitenedError(values)
        projection = pair._projection_cache[2]
        assert projection.normal_jacobians is None and projection.height_jacobians is None
        assert normal.call_count == height.call_count == 1
        factor.linearize(values)
        assert normal.call_count == height.call_count == 2


def test_gelsight_blocks_preserve_independent_least_squares_weights():
    pair, values = make_pair()
    factor = pair.factor(0, 1)
    before = factor.whitenedError(values)
    original_A, original_b = factor.linearize(values).jacobian()
    pair.height_sigma *= 13
    updated = factor.whitenedError(values)
    A, b = factor.linearize(values).jacobian()
    np.testing.assert_array_equal(updated[:-1], before[:-1])
    np.testing.assert_array_equal(A[:-1], original_A[:-1])
    np.testing.assert_array_equal(b[:-1], original_b[:-1])
    assert updated[-1] == pytest.approx(before[-1] / 13)

    height_before, height_A, height_b = updated[-1:], A[-1:], b[-1:]
    pair.normal_sigma *= 7
    updated = factor.whitenedError(values)
    A, b = factor.linearize(values).jacobian()
    np.testing.assert_array_equal(updated[-1:], height_before)
    np.testing.assert_array_equal(A[-1:], height_A)
    np.testing.assert_array_equal(b[-1:], height_b)
    np.testing.assert_allclose(updated[:-1], before[:-1] / 7)
    np.testing.assert_allclose(A[:-1], original_A[:-1] / 7)
    np.testing.assert_allclose(b[:-1], original_b[:-1] / 7)


@pytest.mark.parametrize('vector', [False, True])
def test_values_match_origin_for_subpixels_borders_and_chunked_maps(vector):
    rng = np.random.default_rng(917)
    field = rng.normal(size=(24, 32, 3) if vector else (24, 32)).astype(np.float32)
    # More than 32000 samples also verifies wide_remap's chunk boundary.
    x = np.r_[[-1., -.1, 0., 31., 31.9, 32., 2.1234],
              rng.uniform(-1, 33, 33000)].astype(np.float32)
    y = np.r_[[0., .1, 0., 23., 23.9, 24., 1.4567],
              rng.uniform(-1, 25, 33000)].astype(np.float32)
    samples = TactilePair._samples(field, TactilePair._sample_coordinates(field.shape, x, y))
    actual = samples.value()
    np.testing.assert_array_equal(actual, wide_remap(field, x, y)[:, 0])
    assert actual.dtype == np.float32


def test_projection_matches_frontend_origin_remap_and_contact_selection():
    pair, _ = make_pair()
    # Identity includes final-row/column points; shifts exercise subpixels and
    # points immediately outside the image, including nearest-pixel ties.
    for xy in ((0., 0.), (.1234, .4567), (.5, .5), (-.2, -.1), (30.1, 20.2)):
        target = np.eye(4)
        target[:2, 3] = np.array(xy) * pair.visual.ppmm / 1000
        projection = pair._compute_projection(np.eye(4), target, need_jacobians=False)
        xx, yy = pair.visual._pixel_coordinates(projection.points)
        expected_normal = wide_remap(np.asarray(pair.visual.tar.N, dtype=np.float32), xx, yy)[:, 0]
        expected_height = wide_remap(np.asarray(pair.visual.tar.H, dtype=np.float32), xx, yy)[:, 0]
        expected_contact = wide_remap(np.asarray(pair.visual.tar.C, dtype=np.float32),
                                      xx, yy, mode=cv2.INTER_NEAREST)[:, 0] > .5
        valid = (pair.visual._in_image(xx, yy) & expected_contact
                 & np.isfinite(expected_normal).all(axis=1) & np.isfinite(expected_height))
        np.testing.assert_array_equal(projection.normals, expected_normal)
        np.testing.assert_array_equal(projection.height, expected_height.astype(float) * .001)
        np.testing.assert_array_equal(projection.shared, valid)


def test_float32_maps_are_shared_and_only_sampled_pixels_are_promoted():
    first, second = frame(), frame(1)
    pair = TactilePair(first, second, np.eye(4), np.eye(4), np.eye(4), {})
    assert np.shares_memory(pair._normal_map, second.N)
    assert np.shares_memory(pair._height_map, second.H)
    projection = pair._projection(np.eye(4), np.eye(4), need_jacobians=False)
    assert projection.normals.dtype == np.float32
    assert projection.normal_samples.field.dtype == np.float32
    assert projection.height_samples.field.dtype == np.float32
    assert projection.normal_samples.coordinates is projection.height_samples.coordinates
    assert projection.normal_jacobians is None and projection.height_jacobians is None
    normal = pair._normal_pose_jacobians(projection)
    height = pair._height_pose_jacobians(projection)
    assert all(block.dtype == np.float64 for block in normal + height)
    count = np.count_nonzero(projection.shared)
    assert normal[0].shape == normal[1].shape == (count * 3, 6)
    assert height[0].shape == height[1].shape == (1, 6)
