"""Information, gradient and lifecycle checks for frontend GTSAM priors."""

import gtsam
import numpy as np
import pytest

from gelslam.core.gtsam_vi import GtsamVisualInertialSolver
from gelslam.core.inertial import ImuSample
from gelslam.core.marginal_prior import NavigationPrior, marginalize_rows
from gelslam.core.numpy_vi import NavigationState
from gelslam.core.split_pose import split_pose_plus


def state(time):
    return NavigationState(time, np.eye(4), np.zeros(3), np.zeros(3), np.zeros(3))


def test_world_reframe_preserves_dense_prior_error_and_gradient():
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(57)
    anchor = state(1.)
    anchor.pose[:3, :3] = Rotation.from_rotvec([.4, -.2, .3]).as_matrix()
    anchor.pose[:3, 3] = [.3, -.1, .5]
    anchor.velocity = rng.normal(size=3)
    anchor.accel_bias = rng.normal(size=3)
    anchor.gyro_bias = rng.normal(size=3)
    prior = NavigationPrior(anchor, rng.normal(size=(25, 15)), rng.normal(size=25))
    target = anchor.copy()
    target.pose = split_pose_plus(target.pose, rng.normal(size=6) * .03)
    target.velocity += rng.normal(size=3) * .02
    target.accel_bias += rng.normal(size=3) * .01
    target.gyro_bias += rng.normal(size=3) * .001
    rotation = Rotation.from_rotvec([2.7, .5, -.2]).as_matrix()
    origin = np.array([.3, -.1, .5])
    transformed = prior.reframe(rotation, origin)
    transformed_target = target.reframe(rotation, origin)
    assert np.array_equal(transformed.anchor.pose[:3, 3], np.zeros(3))
    assert np.array_equal(transformed.anchor.accel_bias, anchor.accel_bias)
    assert np.array_equal(transformed.anchor.gyro_bias, anchor.gyro_bias)
    assert np.isclose(prior.error(target), transformed.error(transformed_target), atol=1e-12)
    A, rhs = prior.linearize(target)
    new_A, new_rhs = transformed.linearize(transformed_target)
    transport = np.eye(15)
    transport[6:9, 6:9] = rotation.T
    assert np.allclose(new_A, A @ transport, atol=1e-12)
    assert np.allclose(new_rhs, rhs, atol=1e-12)


class FlatVisual:
    def __len__(self):
        return 10

    def evaluate(self, _reference, _pose):
        return np.zeros(30), np.zeros((30, 6)), np.zeros((30, 6))


def samples(start, end):
    return [ImuSample(t, (0, 0, 9.81), (0, 0, 0))
            for t in np.linspace(start, end, 11)]


@pytest.mark.parametrize("rank_deficient", [False, True])
def test_hessian_matches_schur_information_and_gradient(rank_deficient):
    rng = np.random.default_rng(14)
    Ap, Ac = rng.normal(size=(50, 15)), rng.normal(size=(50, 15))
    if rank_deficient:
        Ap[:, -3:] = 0
    rhs = rng.normal(size=50)
    A, b = marginalize_rows(Ap, Ac, rhs)
    cross = Ac.T @ Ap
    inverse = np.linalg.pinv(Ap.T @ Ap)
    expected_H = Ac.T @ Ac - cross @ inverse @ cross.T
    expected_g = Ac.T @ rhs - cross @ inverse @ Ap.T @ rhs
    assert np.allclose(A.T @ A, expected_H, atol=1e-10)
    assert np.allclose(A.T @ b, expected_g, atol=1e-10)
    assert A.shape == (15, 15) and b.shape == (15,)
    dp0 = np.linalg.lstsq(Ap, rhs, rcond=None)[0]
    base_original = 0.5 * float(np.linalg.norm(Ap @ dp0 - rhs)**2)
    base_reduced = 0.5 * float(b @ b)
    for _ in range(4):
        delta = rng.normal(size=15)
        previous_delta = np.linalg.lstsq(Ap, rhs - Ac @ delta, rcond=None)[0]
        original = 0.5 * float(np.linalg.norm(Ap @ previous_delta + Ac @ delta - rhs)**2)
        reduced = 0.5 * float(np.linalg.norm(A @ delta - b)**2)
        assert np.isclose(reduced - base_reduced, original - base_original)


def test_navigation_prior_pose_transport_matches_retraction_derivative():
    rng = np.random.default_rng(15)
    anchor, current = state(0), state(0)
    anchor.pose = gtsam.Pose3.Expmap(rng.normal(size=6) * 0.2).matrix()
    current.pose = split_pose_plus(anchor.pose, rng.normal(size=6) * 0.2)
    prior = NavigationPrior(anchor, rng.normal(size=(16, 15)), rng.normal(size=16))
    A, rhs = prior.linearize(current)
    numerical = np.zeros((16, 6))
    for column in range(6):
        increment = np.eye(6)[column] * 1e-6
        plus, minus = current.copy(), current.copy()
        plus.pose = split_pose_plus(current.pose, increment)
        minus.pose = split_pose_plus(current.pose, -increment)
        numerical[:, column] = prior.A @ (
            prior.coordinates(plus) - prior.coordinates(minus)
        ) / 2e-6
    assert np.allclose(A[:, :6], numerical, atol=1e-7)
    assert np.isclose(prior.error(current), 0.5 * rhs @ rhs)


@pytest.mark.parametrize("sigma", [0.0, -1.0, np.nan, np.inf])
def test_frontend_rejects_invalid_initial_prior_sigma(sigma):
    with pytest.raises(ValueError, match="prior sigmas"):
        GtsamVisualInertialSolver({"gtsam_vi_prior_velocity_sigma_mps": sigma})
