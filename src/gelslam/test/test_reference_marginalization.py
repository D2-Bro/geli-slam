"""Numerical and lifecycle contracts for the 21D reference/current prior."""

from types import SimpleNamespace

import gtsam
import numpy as np
import pytest

from gelslam.core.gtsam_vi import create_frontend_solver
from gelslam.core.imu_navigation import gelsight_pose
from gelslam.core.inertial import ImuSample
from gelslam.core.marginal_prior import NavigationPrior
from gelslam.core.numpy_vi import DenseNormalResidual, NavigationState, adjoint
from gelslam.core.retained_prior import (
    RetainedPrior, VariableLayout, eliminate_rows, rank_solve,
    state_blocks, state_values,
)
from gelslam.core.split_pose import split_pose_plus


class FlatVisual:
    def __len__(self):
        return 10

    def evaluate(self, *_):
        return np.zeros(30), np.zeros((30, 6)), np.zeros((30, 6))


def state(time=1.):
    return NavigationState(time, np.eye(4), np.zeros(3), np.zeros(3), np.zeros(3))


def solver(**overrides):
    return create_frontend_solver(dict(
        frontend_state_mode="reference_marginalization", gtsam_vi_max_iterations=3,
        gtsam_vi_min_iterations=1, **overrides))


def advance(s, previous, visual=None, commit=True):
    samples = tuple(ImuSample(float(t), [0., 0., 9.81], [0., 0., 0.])
                    for t in np.linspace(previous.timestamp, previous.timestamp + .1, 11))
    current, pim = s.predict_current(previous, samples, np.eye(4))
    result = s.refine_current(visual or FlatVisual(), np.eye(4), previous, current,
                              samples, np.eye(4), preintegrated=pim)
    assert result is not None, s.last_refinement_failure
    if commit:
        s.commit_marginal_prior(result)
    return result


def test_full_normal_equation_and_schur_solution_agree_with_nonzero_gradient():
    rng = np.random.default_rng(71)
    A, rhs = rng.normal(size=(90, 36)), rng.normal(size=90)
    # Remove the middle 15D previous state; retain reference then target.
    m, r = np.arange(6, 21), np.r_[0:6, 21:36]
    Ar, br = eliminate_rows(A, rhs, m, r)
    H, b = A.T @ A, A.T @ rhs
    Hschur = H[np.ix_(r, r)] - H[np.ix_(r, m)] @ np.linalg.solve(H[np.ix_(m, m)], H[np.ix_(m, r)])
    bschur = b[r] - H[np.ix_(r, m)] @ np.linalg.solve(H[np.ix_(m, m)], b[m])
    np.testing.assert_allclose(Ar.T @ Ar, Hschur, atol=2e-12)
    np.testing.assert_allclose(Ar.T @ br, bschur, atol=2e-12)
    full = np.linalg.solve(H, b)[r]
    reduced = np.linalg.solve(Ar.T @ Ar, Ar.T @ br)
    np.testing.assert_allclose(full, reduced, atol=1e-13)
    # User cost convention is +g.T dx, so g = -A.T rhs.
    np.testing.assert_allclose(-Ar.T @ br, -bschur, atol=2e-12)


def test_rank_deficiency_preserves_zero_information_and_observable_solution():
    rng = np.random.default_rng(43)
    A = rng.normal(size=(60, 36))
    A[:, 5] = 0.  # Unobserved retained translation axis.
    A[:, 7] = A[:, 8]  # Singular eliminated block.
    rhs = rng.normal(size=60)
    m, r = np.arange(6, 21), np.r_[0:6, 21:36]
    Ar, br = eliminate_rows(A, rhs, m, r)
    assert np.max(np.abs(Ar[:, 5])) < 1e-12
    full = rank_solve(A, rhs)[0][r]
    reduced = rank_solve(Ar, br)[0]
    np.testing.assert_allclose(full, reduced, atol=1e-12)
    assert rank_solve(Ar, br)[1] == 20


def test_frozen_prior_jacobian_and_cost_gradient_away_from_pose_anchors():
    rng = np.random.default_rng(19)
    first, current = state(1.), state(2.)
    first.pose = split_pose_plus(first.pose, np.array([.2, -.3, .1, .2, .1, -.1]))
    layout = VariableLayout((state_blocks(1.)[0],) + state_blocks(2.))
    values = {state_blocks(1.)[0]: first.pose, **state_values(current)}
    prior = RetainedPrior(layout, values, rng.normal(size=(21, 21)), rng.normal(size=21), 2.)
    shifted = layout.plus(values, rng.normal(size=21) * .15)
    J, rhs = prior.linearize_values(shifted)
    numeric = np.zeros_like(J)
    gradient = np.zeros(21)
    eps = 1e-7
    for col in range(21):
        step = np.eye(21)[col] * eps
        plus, minus = layout.plus(shifted, step), layout.plus(shifted, -step)
        numeric[:, col] = prior.A @ (prior.coordinates(plus) - prior.coordinates(minus)) / (2 * eps)
        gradient[col] = (prior.error_values(plus) - prior.error_values(minus)) / (2 * eps)
    np.testing.assert_allclose(J, numeric, atol=2e-8)
    np.testing.assert_allclose(-J.T @ rhs, gradient, atol=2e-7)
    assert np.linalg.norm(prior.information[:6, 6:]) > 1.


def test_bootstrap_generic_and_consecutive_reference_changes_have_unique_blocks():
    s, previous = solver(), state()
    s.set_reference(previous.timestamp, previous)
    assert s.marginal_prior.layout.dimension == 9
    for index in range(8):
        previous = advance(s, previous)
        assert s.last_problem['solve_dimension'] == [24, 30, 30, 30, 30, 30, 36, 36][index]
        assert s.marginal_prior.layout.dimension == (15 if index <= 2 else 21)
        assert len(s.last_problem['blocks']) == len(set(s.last_problem['blocks']))
        assert s.last_problem['symmetry_error'] == 0.
        assert np.isfinite(s.marginal_prior.information).all()
        assert s.last_problem['numerical_rank'] == s.last_problem['solve_dimension']
        if index in (2, 3, 4):
            old_reference = s.reference_id
            s.set_reference(previous.timestamp, previous)
            assert old_reference not in s.marginal_prior.layout.blocks
            assert s.marginal_prior.layout.dimension == 15
        assert s.marginal_prior.absorbed_intervals == index + 1


def test_initialization_reuses_correlated_navigation_prior_and_nonzero_rhs():
    s, previous = solver(), state()
    rng = np.random.default_rng(5)
    A = rng.normal(size=(15, 15))
    rhs = rng.normal(size=15) * .01
    s.marginal_prior = NavigationPrior(previous.copy(), A, rhs)
    s.set_reference(1., previous)
    np.testing.assert_array_equal(s.marginal_prior.A, A[:, 6:])
    np.testing.assert_array_equal(s.marginal_prior.rhs, rhs)
    assert s.marginal_prior.layout.dimension == 9


def test_rejected_candidate_is_transactional_and_measurement_intervals_are_consumed_once():
    s, previous = solver(), state()
    current = advance(s, previous)
    prior = s.marginal_prior
    candidate = advance(s, current, commit=False)
    s.discard_pending_prior()
    assert s.marginal_prior is prior
    assert s.marginal_prior.timestamp == current.timestamp
    # Recompute the rejected candidate from the accepted prior. No rejected
    # NormalFlow or IMU rows are carried forward.
    candidate = advance(s, current)
    assert s.marginal_prior.absorbed_intervals == 2
    assert s.last_problem['active_measurement_factors_after_commit'] == 0
    assert s.last_problem['factor_ids'][2] == ('imu', current.timestamp, candidate.timestamp)
    with pytest.raises(ValueError, match='timestamp mismatch'):
        s._problem(current, candidate)


def test_exact_pose_conditioning_keeps_cross_terms_and_shifted_gradient():
    rng = np.random.default_rng(132)
    anchor = state()
    layout = VariableLayout(state_blocks(anchor.timestamp))
    A, rhs = rng.normal(size=(25, 15)), rng.normal(size=25)
    prior = RetainedPrior(layout, state_values(anchor), A, rhs, anchor.timestamp)
    pose_id = state_blocks(anchor.timestamp)[0]
    fixed_pose = split_pose_plus(anchor.pose, np.array([.1, -.05, .02, .003, -.002, .004]))
    conditioned = prior.condition_fixed({pose_id: fixed_pose})
    assert conditioned.layout.dimension == 9
    np.testing.assert_array_equal(conditioned.A, A[:, 6:])
    point = state_values(anchor)
    point[pose_id] = fixed_pose.copy()
    for block in state_blocks(anchor.timestamp)[1:]:
        point[block] += rng.normal(size=3) * .1
    assert prior.error_values(point) == pytest.approx(conditioned.error_values(point), abs=1e-12)
    np.testing.assert_allclose(conditioned.information, (A.T @ A)[6:, 6:], atol=1e-12)
    J, b = conditioned.linearize_values(point)
    Jfull, bfull = prior.linearize_values(point)
    np.testing.assert_allclose(J.T @ b, Jfull[:, 6:].T @ bfull, atol=1e-12)
    # Hard conditioning is not Schur integration of an uncertain pose.
    marginalized = prior.retain(state_blocks(anchor.timestamp)[1:])
    assert np.linalg.norm(conditioned.information - marginalized.information) > 1.


def test_first_pose_is_copied_exactly_during_optimization_and_imu_propagation():
    s, previous = solver(), state()
    previous.pose = split_pose_plus(previous.pose, np.array([.1, -.2, .3, .01, -.02, .03]))
    extrinsic = split_pose_plus(np.eye(4), np.array([.2, .1, -.1, .002, -.003, .004]))
    frozen_pose = previous.pose.copy()
    reference = gelsight_pose(frozen_pose, extrinsic)
    evaluated = []

    class ShiftVisual(FlatVisual):
        def evaluate(self, ref, target):
            evaluated.append(ref.copy())
            relative = np.linalg.inv(ref) @ target
            Jt = np.zeros((30, 6))
            Jt[:, 3:] = relative[:3, :3][0]
            Jr = -Jt @ adjoint(np.linalg.inv(target) @ ref)
            return np.full(30, relative[0, 3] - .001), Jr, Jt

    samples = [ImuSample(t, frozen_pose[:3, :3].T @ np.array([0., 0., 9.81]), [0., 0., 0.])
               for t in np.linspace(previous.timestamp, previous.timestamp + .1, 11)]
    current, pim = s.predict_current(previous, samples, extrinsic)
    predicted_prior = s.propagate_prior(previous, current, pim)
    original = s.marginal_prior
    np.testing.assert_array_equal(predicted_prior.fixed_values[s.first_pose_id], frozen_pose)
    result = s.refine_current(ShiftVisual(), reference, previous, current, samples, extrinsic, pim)
    assert result is not None, s.last_refinement_failure
    assert all(np.array_equal(pose, reference) for pose in evaluated)
    np.testing.assert_array_equal(s.last_reference_pose, frozen_pose)
    np.testing.assert_array_equal(s.last_reference_state.pose, frozen_pose)
    assert np.linalg.norm(result.pose - current.pose) > 1e-10
    assert np.linalg.norm(s.last_reference_state.velocity - previous.velocity) > 1e-10
    s.discard_pending_prior()
    assert s.marginal_prior is original
    np.testing.assert_array_equal(original.fixed_values[s.first_pose_id], frozen_pose)


def test_reference_change_eliminates_old_pose_preserving_gradient_and_cross_information():
    s, previous = solver(), state()
    current = advance(s, previous)
    old = s.marginal_prior
    expected = old.current_prior()
    s.set_reference(current.timestamp, current)
    np.testing.assert_allclose(s.marginal_prior.information, expected.A.T @ expected.A, atol=1e-9)
    np.testing.assert_allclose(s.marginal_prior.A.T @ s.marginal_prior.rhs,
                               expected.A.T @ expected.rhs, atol=1e-12)
    with pytest.raises(ValueError, match='already marginalized'):
        s.set_reference(previous.timestamp, current)


def test_reference_is_free_and_dense_normalflow_model_retains_five_dof():
    from test_frontend_gauss_newton import frame
    image = frame()
    s = solver(dense_normalflow_residual_weight=1e5)
    previous = state()
    current = advance(s, previous)
    # Only the first pose is fixed. A later reference remains a free pose.
    s.set_reference(current.timestamp, current)
    current = advance(s, current)
    reference_before = s.marginal_prior.anchors[s.reference_id].copy()

    class PerturbedVisual(FlatVisual):
        def evaluate(self, ref, tar):
            visual = DenseNormalResidual(image, image, 1., ref, tar, 1000)
            residual, Jr, Jt = visual.evaluate(ref, tar)
            return residual + .01, Jr, Jt

    current = advance(s, current, PerturbedVisual())
    assert s.last_problem['solve_dimension'] == 36
    assert np.linalg.norm(s.last_reference_pose - reference_before) > 1e-10
    ref = np.eye(4)
    tar = split_pose_plus(ref, np.array([.1, -.1, .05, .001, .002, .003]))
    visual = DenseNormalResidual(image, image, 1., ref, tar, 1000)
    _, Jr, Jt = visual.evaluate(ref, tar)
    assert np.array_equal(Jr[:, 5], np.zeros(len(Jr)))
    assert np.linalg.matrix_rank(Jr) <= 5


def test_eliminate_first_matches_joint_solve_with_smaller_system():
    from test_frontend_gauss_newton import frame
    image = frame()

    class PerturbedVisual(FlatVisual):
        def evaluate(self, ref, tar):
            visual = DenseNormalResidual(image, image, 1., ref, tar, 1000)
            residual, Jr, Jt = visual.evaluate(ref, tar)
            return residual + .01, Jr, Jt

    runs = {}
    for mode in ('joint', 'eliminate_first'):
        s = solver(dense_normalflow_residual_weight=1e5, frontend_reference_previous_state=mode)
        current = advance(s, state())
        s.set_reference(current.timestamp, current)
        # Reference pose is the previous pose; its state is published.
        aliased = advance(s, current, PerturbedVisual())
        aliased_problem, reference_state = s.last_problem, s.last_reference_state
        # Distinct reference, previous and current states.
        generic = advance(s, aliased, PerturbedVisual())
        runs[mode] = (aliased, reference_state, generic, aliased_problem, s.last_problem)
    joint, eliminated = runs['joint'], runs['eliminate_first']
    assert [p['solve_dimension'] for p in joint[3:]] == [30, 36]
    assert [p['solve_dimension'] for p in eliminated[3:]] == [21, 21]
    assert all(p['numerical_rank'] == p['solve_dimension'] for p in eliminated[3:])
    # Only the IMU linearization point of the previous state differs
    # (observed <= 1.3e-7 here; velocities are about 1e-2 m/s).
    for a, b in zip(joint[:3], eliminated[:3]):
        for field in ('pose', 'velocity', 'accel_bias', 'gyro_bias'):
            np.testing.assert_allclose(getattr(b, field), getattr(a, field), atol=1e-6)


@pytest.mark.parametrize('mode', ['inverse_compositional', 'target_relinearized'])
def test_projected_normal_rows_do_not_observe_common_world_motion(mode):
    from test_frontend_gauss_newton import frame
    image = frame()
    s = solver(gtsam_vi_normal_linearization=mode)
    ref = split_pose_plus(np.eye(4), np.array([.01, -.02, .03, .001, 0., 0.]))
    tar = split_pose_plus(ref, np.array([.01, -.01, .02, .0001, .0002, .0003]))
    visual = DenseNormalResidual(image, image, 1., ref, tar, 1000)
    visual.set_fixed_reference_z((np.linalg.inv(ref) @ tar)[2, 3])
    evaluate = visual.evaluate_ic if mode == 'inverse_compositional' else visual.evaluate_relinearized
    _, Jr, Jt = evaluate(ref, tar)
    Jr, Jt = s._normal_pose_rows(visual, ref, tar, Jr, Jt)
    common_world_motion = Jr @ adjoint(np.linalg.inv(ref)) + Jt @ adjoint(np.linalg.inv(tar))
    np.testing.assert_allclose(common_world_motion, 0., atol=1e-11)
    assert np.linalg.matrix_rank(np.column_stack((Jr, Jt)), tol=1e-8) <= 5


def test_reframing_transports_both_poses_velocity_and_full_prior():
    s, previous = solver(), state()
    advance(s, previous)
    prior = s.marginal_prior
    rng = np.random.default_rng(32)
    values = prior.layout.plus(prior.values, rng.normal(size=prior.layout.dimension) * .01)
    rotation = gtsam.Rot3.Expmap([.2, -.1, .3]).matrix()
    origin = np.array([.3, -.2, .1])
    moved = prior.reframe(rotation, origin)
    # Reframe arbitrary evaluation values with the same block rules.
    point = RetainedPrior(prior.layout, values, prior.A, prior.rhs, prior.timestamp,
                          fixed_values=prior.fixed_values).reframe(rotation, origin)
    assert prior.error_values(values) == pytest.approx(moved.error_values(point.anchors), abs=1e-10)


def test_prediction_candidate_does_not_add_its_measurements_to_visual_objective():
    s, previous = solver(), state()
    current = advance(s, previous)
    accepted_prior = s.marginal_prior
    samples = [ImuSample(t, [0, 0, 9.81], [0, 0, 0])
               for t in np.linspace(current.timestamp, current.timestamp + .1, 11)]
    predicted, pim = s.predict_current(current, samples, np.eye(4))
    propagated = s.propagate_prior(current, predicted, pim)
    assert s.marginal_prior is accepted_prior
    assert propagated.absorbed_intervals == 2
    result = s.refine_current(FlatVisual(), np.eye(4), current, predicted, samples, np.eye(4), pim)
    s.commit_marginal_prior(result)
    np.testing.assert_allclose(s.marginal_prior.information, propagated.information, rtol=1e-8, atol=1e-5)
    assert s.marginal_prior.absorbed_intervals == 2


def test_missing_initial_pose_information_uses_an_exact_datum_without_soft_pose_rows():
    s, previous = solver(), state()
    A = np.eye(15)
    A[:6] = 0.
    s.marginal_prior = NavigationPrior(previous.copy(), A, np.zeros(15))
    current = advance(s, previous)
    assert s.last_problem['numerical_rank'] == s.last_problem['solve_dimension'] == 24
    prior = s.marginal_prior
    assert s.first_pose_id not in prior.layout.blocks
    np.testing.assert_array_equal(prior.fixed_values[s.first_pose_id], previous.pose)
    assert prior.layout.dimension == 15
    current = advance(s, current)
    assert s.last_problem['numerical_rank'] == s.last_problem['solve_dimension'] == 30
    assert s.marginal_prior.layout.dimension == 15


@pytest.mark.parametrize('bad', [np.nan, np.inf])
def test_nonfinite_failure_keeps_accepted_prior(bad):
    s, previous = solver(), state()
    previous = advance(s, previous)
    prior = s.marginal_prior

    class BadVisual(FlatVisual):
        def evaluate(self, *args):
            r, Jr, Jt = super().evaluate(*args)
            r[0] = bad
            return r, Jr, Jt

    samples = [ImuSample(t, [0, 0, 9.81], [0, 0, 0])
               for t in np.linspace(previous.timestamp, previous.timestamp + .1, 11)]
    current, pim = s.predict_current(previous, samples, np.eye(4))
    assert s.refine_current(BadVisual(), np.eye(4), previous, current, samples, np.eye(4), pim) is None
    assert s.marginal_prior is prior and s.pending_marginal_prior is None


@pytest.mark.parametrize('options', [
    {'frontend_fix_previous_state': True},
    {'frontend_imu_prediction_only': True}, {'gtsam_vi_marginalization': False},
    {'frontend_gt_previous_state_test': True}, {'frontend_state_mode': 'unknown'},
    {'frontend_reference_previous_state': 'unknown'},
])
def test_incompatible_modes_fail_explicitly(options):
    with pytest.raises(ValueError):
        create_frontend_solver(dict(frontend_state_mode='reference_marginalization', **options)
                               if 'frontend_state_mode' not in options else options)


def test_tracker_contact_keyframe_recovery_and_backend_map_only(monkeypatch):
    import gelslam.core.tracker as tm
    from test_contact_recovery_policy import tracker_with_imu
    from test_tracker_inertial import header
    tracker, initial = tracker_with_imu(monkeypatch)
    s = solver()
    tracker.frontend_visual_inertial = s
    s.set_reference(initial.timestamp, initial)
    monkeypatch.setattr(tm, 'normalflow_quality', lambda *args: (.9, .9))
    image = np.zeros((1, 1, 3), np.uint8)
    for time in (1.04, 1.08, 1.12):
        assert tracker.track(image, header(time))[0]
        np.testing.assert_array_equal(tracker.trial_T_ref, initial.pose)
        np.testing.assert_array_equal(s.marginal_prior.fixed_values[s.first_pose_id], initial.pose)
    assert s.last_problem['solve_dimension'] == 30
    assert s.marginal_prior.layout.dimension == 15
    monkeypatch.setattr(tm, 'normalflow_quality', lambda *args: (.1, .9))
    for time in (1.16, 1.20, 1.24):
        ok, (_, packets) = tracker.track(image, header(time))
        assert ok and len(packets) == 1
        assert s.marginal_prior.layout.dimension == 15
    assert s.last_problem['solve_dimension'] == 30
    previous, prior, ref = tracker._frontend_vi_state, s.marginal_prior, tracker.trial_T_ref.copy()
    snapshot = SimpleNamespace()  # No map data is allowed to touch solve state.
    assert not tracker.apply_map_snapshot(snapshot, header(1.25))
    assert tracker._frontend_vi_state is previous and s.marginal_prior is prior
    np.testing.assert_array_equal(tracker.trial_T_ref, ref)
    tracker.prev_in_contact_flag = False

    class EmptyVisual(FlatVisual):
        def __len__(self):
            return 0

    monkeypatch.setattr(tm, 'DenseNormalResidual', lambda *a, **k: EmptyVisual())
    ok, (_, packets) = tracker.track(image, header(1.28))
    assert ok and not packets[0]['tactile_valid']
    assert s.marginal_prior.layout.dimension == 15
    monkeypatch.setattr(tm, 'DenseNormalResidual', lambda *a, **k: FlatVisual())
    assert tracker.track(image, header(1.32))[0]
    assert s.marginal_prior.matches(tracker._frontend_vi_state)


def test_no_contact_restart_transfers_information_and_replaces_both_references(monkeypatch):
    import gelslam.core.tracker as tm
    from test_contact_recovery_policy import tracker_with_imu
    from test_tracker_inertial import header
    tracker, initial = tracker_with_imu(monkeypatch)
    tracker.imu_config['frontend_state_mode'] = 'reference_marginalization'
    tracker.frontend_visual_inertial = solver()
    tracker.frontend_visual_inertial.set_reference(initial.timestamp, initial)
    contact, trial = tracker.prev_frame, tracker._trial_index
    image = np.zeros((1, 1, 3), np.uint8)
    monkeypatch.setattr(tm, 'Frame', lambda *a, **k: SimpleNamespace(is_contacted=False))
    tracker.track(image, header(1.10))
    tracker.track(image, header(2.12))
    assert tracker._restart_pending
    monkeypatch.setattr(tm, 'Frame', lambda *a, **k: contact)
    monkeypatch.setattr(tm, 'normalflow_quality', lambda *a: (.2, .9))
    assert tracker.track(image, header(2.14)) == (False, None)
    ok, (_, packets) = tracker.track(image, header(2.18))
    assert ok and len(packets) == 2
    assert tracker._trial_index == trial + 1
    s = tracker.frontend_visual_inertial
    assert s.marginal_prior.layout.dimension == 15
    assert s.marginal_prior.matches(tracker._frontend_vi_state)
    assert s.marginal_prior.absorbed_intervals == 2
    assert s.first_pose_id.timestamp == pytest.approx(2.14)
    np.testing.assert_allclose(s.marginal_prior.fixed_values[s.first_pose_id],
                               packets[0]['trial_T_curr'], atol=1e-14)


def test_replacing_unpublished_restart_reference_sets_only_the_new_first_pose(monkeypatch):
    import gelslam.core.tracker as tm
    from test_contact_recovery_policy import tracker_with_imu
    from test_tracker_inertial import header
    tracker, initial = tracker_with_imu(monkeypatch)
    tracker.imu_config['frontend_state_mode'] = 'reference_marginalization'
    old_solver = solver()
    tracker.frontend_visual_inertial = old_solver
    old_solver.set_reference(initial.timestamp, initial)
    old_prior = old_solver.marginal_prior
    contact = tracker.prev_frame
    image = np.zeros((1, 1, 3), np.uint8)
    monkeypatch.setattr(tm, 'Frame', lambda *a, **k: SimpleNamespace(is_contacted=False))
    tracker.track(image, header(1.10))
    tracker.track(image, header(2.12))
    monkeypatch.setattr(tm, 'Frame', lambda *a, **k: contact)
    monkeypatch.setattr(tm, 'normalflow_quality', lambda *a: (.9, .1))
    assert tracker.track(image, header(2.14)) == (False, None)
    assert tracker.track(image, header(2.70)) == (False, None)
    staged = tracker._pending_segment_contact[-1]
    assert staged.first_pose_id.timestamp == pytest.approx(2.70)
    assert list(staged.marginal_prior.fixed_values) == [staged.first_pose_id]
    assert old_solver.marginal_prior is old_prior
    monkeypatch.setattr(tm, 'normalflow_quality', lambda *a: (.9, .9))
    ok, (_, packets) = tracker.track(image, header(2.74))
    assert ok and len(packets) == 2
    assert tracker.frontend_visual_inertial.first_pose_id.timestamp == pytest.approx(2.70)
    assert tracker.frontend_visual_inertial.marginal_prior.absorbed_intervals == 2


def test_imu_callback_during_solve_only_buffers_measurements(monkeypatch):
    import threading
    from gelslam.nodes.tracker_node import TrackerNode
    from test_tracker_image_imu_sync import SyncHarness, message_at
    from sensor_msgs.msg import Imu
    s, previous = solver(), state()
    previous = advance(s, previous)
    prior = s.marginal_prior
    entered, received = threading.Event(), threading.Event()

    class ReceiveWhileEvaluating(FlatVisual):
        def evaluate(self, *args):
            entered.set()
            assert received.wait(2.)
            assert s.marginal_prior is prior
            return super().evaluate(*args)

    node = SyncHarness()
    node.tracker.frontend_visual_inertial = s
    node.tracker._frontend_vi_state = previous

    def callback():
        assert entered.wait(2.)
        TrackerNode.imu_callback(node, message_at(Imu, 1.25))
        received.set()

    worker = threading.Thread(target=callback)
    worker.start()
    try:
        advance(s, previous, ReceiveWhileEvaluating(), commit=False)
    finally:
        worker.join(2.)
    assert not worker.is_alive()
    assert s.marginal_prior is prior
    assert node.tracker._frontend_vi_state is previous
    assert node.tracker.imu_buffer.latest_timestamp() == 1.25
