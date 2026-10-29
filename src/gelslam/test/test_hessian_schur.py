"""Absolute Hessian cutoffs, state recovery, and zero-gradient prior semantics."""
import pickle
from types import SimpleNamespace

import gtsam
import numpy as np
import pytest

from gelslam.core.gtsam_vi import GtsamVisualInertialSolver
from gelslam.core.marginal_prior import (
    NavigationPrior, information_rows, marginalize_rows_system, schur_complement,
)
from gelslam.core.numpy_vi import NavigationState
from gelslam.core.tracker import Tracker
import gelslam.core.tracker as tracker_module


def state():
    return NavigationState(2., np.eye(4), np.zeros(3), np.zeros(3), np.zeros(3))


@pytest.mark.parametrize('width', [9, 15])
@pytest.mark.parametrize('rank_deficient', [False, True])
def test_schur_and_previous_recovery_use_same_absolute_svd_cutoff(width, rank_deficient):
    rng = np.random.default_rng(103)
    Q, _ = np.linalg.qr(rng.normal(size=(60, width)))
    V, _ = np.linalg.qr(rng.normal(size=(width, width)))
    spectrum = np.linspace(1., 3., width)
    if rank_deficient:
        spectrum[-3:] = [0., 1e-7, 2e-6]
    Ap = Q @ np.diag(np.sqrt(spectrum)) @ V.T
    Ac = rng.normal(size=(60, width))
    rhs = rng.normal(size=60)
    A = np.column_stack((Ap, Ac))
    H, g = A.T @ A, A.T @ rhs
    # Independent block calculation, with the requested absolute threshold.
    u, singular, vt = np.linalg.svd(H[:width, :width])
    reciprocal = np.array([1. / s if s > 1e-6 else 0. for s in singular])
    inverse = vt.T @ np.diag(reciprocal) @ u.T
    expected_H = H[width:, width:] - H[width:, :width] @ inverse @ H[:width, width:]
    expected_g = g[width:] - H[width:, :width] @ inverse @ g[:width]
    reduced = marginalize_rows_system(Ap, Ac, rhs)
    assert reduced.information.dtype == np.float64
    assert np.allclose(reduced.information, expected_H, rtol=1e-7, atol=1e-8)
    assert np.allclose(reduced.gradient, expected_g, rtol=1e-7, atol=1e-8)
    rows, reduced_rhs = reduced.rows()
    assert np.allclose(rows.T @ reduced_rhs, expected_g, rtol=1e-7, atol=1e-8)
    dc = np.linalg.solve(reduced.information, reduced.gradient)
    assert np.allclose(reduced.recover_previous(dc),
                       inverse @ (g[:width] - H[:width, width:] @ dc),
                       rtol=1e-7, atol=1e-8)
    if not rank_deficient:
        joint = np.r_[reduced.recover_previous(dc), dc]
        assert np.allclose(joint, np.linalg.solve(H, g), atol=1e-10)


def test_absolute_cutoff_boundary_and_no_invented_information():
    H = np.diag([1e-7, 1e-6, 1.0001e-6, 2.])
    reduced = schur_complement(H, np.ones(4), 3)
    assert np.array_equal(np.diag(reduced.previous_inverse)[:2], np.zeros(2))
    assert reduced.previous_inverse[2, 2] == pytest.approx(1. / H[2, 2])
    assert reduced.information[0, 0] == pytest.approx(2., rel=1e-15)
    # All retained directions can be explained by the removed variables.
    reduced = marginalize_rows_system(np.eye(3), np.eye(3), np.ones(3))
    A, rhs = reduced.rows()
    assert np.array_equal(A, np.zeros((3, 3)))
    assert np.array_equal(rhs, np.zeros(3))
    assert np.array_equal(reduced.gradient, np.zeros(3))
    assert np.array_equal(reduced.recover_previous(np.ones(3)), np.zeros(3))
    # Backend components without a separator produce an empty prior.
    assert schur_complement(np.eye(3), np.ones(3), 3).information.shape == (0, 0)
    assert np.array_equal(schur_complement(np.eye(3), np.ones(3), 0).gradient, np.ones(3))


def test_information_eigen_cutoff_keeps_boundary_and_discards_negative_values():
    H = np.diag([-1e-10, 0., 0.999e-12, 1e-12, 2.])
    A, rhs = information_rows(H, np.ones(5))
    assert np.allclose(A.T @ A, np.diag([0., 0., 0., 1e-12, 2.]), rtol=1e-14, atol=0.)
    assert np.allclose(A.T @ rhs, [0., 0., 0., 1., 1.])


def test_dense_information_prior_has_zero_anchor_gradient_and_preserves_reframe_cross_terms():
    rng = np.random.default_rng(41)
    rows = rng.normal(size=(30, 15))
    H = rows.T @ rows
    anchor = state()
    prior = NavigationPrior.from_information(anchor, H)
    assert prior.anchor is not anchor
    assert np.allclose(prior.A.T @ prior.A, H)
    assert prior.error(anchor) == 0.
    A, rhs = prior.linearize(anchor)
    assert np.array_equal(A.T @ rhs, np.zeros(15))
    rotation = gtsam.Rot3.Expmap(np.array([.5, -.2, .3])).matrix()
    transformed = prior.reframe(rotation, np.array([1., 2., 3.]))
    transport = np.eye(15)
    transport[6:9, 6:9] = rotation.T
    assert np.allclose(transformed.A.T @ transformed.A, transport.T @ H @ transport)
    assert np.linalg.norm((transformed.A.T @ transformed.A)[:6, 6:9]) > 1.
    assert transformed.error(transformed.anchor) == pytest.approx(0., abs=1e-25)


def test_tracker_load_converts_legacy_rhs_and_drops_uncommitted_prior(tmp_path, monkeypatch):
    tracker = object.__new__(Tracker)
    tracker.bg_image = np.zeros((2, 2, 3))
    tracker.curr_kid = -1
    tracker.imu_config = {}
    solver = GtsamVisualInertialSolver({})
    tracker.frontend_visual_inertial = solver
    rng = np.random.default_rng(99)
    rows = rng.normal(size=(20, 15))
    solver.marginal_prior = NavigationPrior(state(), rows, np.ones(20))
    solver.pending_marginal_prior = NavigationPrior.initial(state(), np.ones(15))
    with (tmp_path / 'tracker.pkl').open('wb') as file:
        pickle.dump(tracker, file)
    monkeypatch.setattr(tracker_module, 'Reconstructor',
                        lambda _: SimpleNamespace(load_bg=lambda _: None))
    restored = Tracker.load(tmp_path, 'unused', SimpleNamespace(info=lambda _: None))
    prior = restored.frontend_visual_inertial.marginal_prior
    assert prior.matches(solver.marginal_prior.anchor)
    assert np.allclose(prior.A.T @ prior.A, rows.T @ rows)
    assert prior.error(prior.anchor) == 0.
    assert np.array_equal(prior.rhs, np.zeros(15))
    assert restored.frontend_visual_inertial.pending_marginal_prior is None
