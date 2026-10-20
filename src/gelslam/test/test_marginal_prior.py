"""Information, gradient, lifecycle, and window checks for active GTSAM priors."""

from types import SimpleNamespace

import gtsam
import numpy as np
import pytest

from gelslam.core.gtsam_pose_graph import PoseGraph
from gelslam.core.gtsam_vi import GtsamVisualInertialSolver
from gelslam.core.inertial import ImuSample
from gelslam.core.marginal_prior import NavigationPrior, marginalize_rows, marginalize_rows_system
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


def test_frontend_propagates_full_correlated_prior_only_after_acceptance():
    solver = GtsamVisualInertialSolver({})
    previous = state(0)
    covariance_trace = []
    for target_time in (0.1, 0.2, 0.3):
        interval = samples(previous.timestamp, target_time)
        prediction, pim = solver.predict_current(previous, interval, np.eye(4))
        result = solver.refine_current(
            FlatVisual(), np.eye(4), previous, prediction, interval, np.eye(4), pim
        )
        assert result is not None
        prior = solver.pending_marginal_prior
        assert prior.matches(result)
        if solver.marginal_prior is not None:
            assert solver.marginal_prior.anchor.timestamp == previous.timestamp
        H = prior.A.T @ prior.A
        assert np.all(np.linalg.eigvalsh(H) > 0)
        assert np.linalg.norm(H[:6, 6:]) > 0  # Pose/velocity/bias correlations.
        covariance_trace.append(np.trace(np.linalg.inv(H)))
        solver.commit_marginal_prior(result)
        assert solver.pending_marginal_prior is None
        previous = result
    assert np.all(np.diff(covariance_trace) > 0)  # No fictitious fixed previous state.


def test_frontend_rejected_target_does_not_replace_previous_prior():
    solver = GtsamVisualInertialSolver({})
    solver.marginal_prior = NavigationPrior.initial(state(0), solver.prior_sigmas)
    accepted = solver.marginal_prior
    result = solver.refine_current(
        FlatVisual(), np.eye(4), state(0), state(0.1), samples(0, 0.1), np.eye(4)
    )
    assert result is not None and solver.pending_marginal_prior is not None
    solver.discard_pending_prior()
    assert solver.marginal_prior is accepted
    assert solver.pending_marginal_prior is None


def test_frontend_uses_gradient_in_step_but_stores_zero_gradient_at_target():
    class NonlinearVisual(FlatVisual):
        def evaluate(self, _reference, pose):
            x = pose[0, 3]
            residual = np.tile([x * x - .0001, 0., 0.], 10)
            jacobian = np.zeros((30, 6))
            jacobian[::3, 3] = 2 * x
            return residual, np.zeros_like(jacobian), jacobian

    solver = GtsamVisualInertialSolver({"dense_normalflow_residual_weight": 1e8,
                                        "gtsam_vi_normal_huber_delta": 0.,
                                        "gtsam_vi_max_iterations": 1})
    previous, current = state(0), state(0.1)
    current.pose[0, 3] = 0.1
    interval = samples(0, 0.1)
    _, pim = solver.predict_current(previous, interval, np.eye(4))
    calls = []
    original = solver._joint_rows
    def capture(*args, **kwargs):
        rows = original(*args, **kwargs)
        calls.append((rows, args[-1].copy()))
        return rows
    solver._joint_rows = capture
    result = solver.refine_current(
        NonlinearVisual(), np.eye(4), previous, current, interval, np.eye(4), pim
    )
    prior = solver.pending_marginal_prior
    assert result is not None
    assert result.pose[0, 3] < current.pose[0, 3]
    (Ap, Ac, rhs), target = calls[-1]
    residual, _, J = NonlinearVisual().evaluate(np.eye(4), target.pose)
    scale = np.sqrt(solver.visual_weight / 10) / solver.normal_sigma
    reduced = marginalize_rows_system(
        np.vstack((Ap, np.zeros((30, 15)))),
        np.vstack((Ac, np.pad(J * scale, ((0, 0), (0, 9))))),
        np.r_[rhs, -residual * scale],
    )
    assert np.linalg.norm(reduced.gradient) > 1e-3
    assert np.allclose(prior.A.T @ prior.A, reduced.information)
    assert np.array_equal(prior.rhs, np.zeros(15))
    assert prior.matches(result) and prior.error(result) == 0.
    assert solver.last_optimization_trace[0]['accepted_scale'] == 1.


def test_frontend_discontinuity_initializes_at_the_new_accepted_time():
    solver = GtsamVisualInertialSolver({})
    solver.marginal_prior = NavigationPrior.initial(state(0), solver.prior_sigmas * 100)
    result = solver.refine_current(
        FlatVisual(), np.eye(4), state(1), state(1.1), samples(1, 1.1), np.eye(4)
    )
    fresh_solver = GtsamVisualInertialSolver({})
    fresh_solver.refine_current(
        FlatVisual(), np.eye(4), state(1), state(1.1), samples(1, 1.1), np.eye(4)
    )
    assert result is not None
    assert np.allclose(solver.pending_marginal_prior.A.T @ solver.pending_marginal_prior.A,
                       fresh_solver.pending_marginal_prior.A.T @ fresh_solver.pending_marginal_prior.A)


@pytest.mark.parametrize("sigma", [0.0, -1.0, np.nan, np.inf])
def test_frontend_rejects_invalid_initial_prior_sigma(sigma):
    with pytest.raises(ValueError, match="prior sigmas"):
        GtsamVisualInertialSolver({"gtsam_vi_prior_velocity_sigma_mps": sigma})


# Backend state/window tests use the 15D graph; frontend prior tests above
# and below retain their original independent navigation-state coverage.
from backend_test_support import config as backend_config, frame as backend_frame
from gelslam.core.backend_factors import state_keys


def test_backend_marginal_prior_keeps_full_navigation_covariance():
    instance = PoseGraph(backend_config())
    frames = [backend_frame(i, new=i == 0) for i in range(6)]
    instance.add_keyframes(frames, 0, len(frames))
    window, values, _ = instance._window_graph(3)
    full, local = gtsam.Marginals(instance.graph, instance.graph_init), gtsam.Marginals(window, values)
    for key in state_keys(3):
        assert np.allclose(local.marginalCovariance(key), full.marginalCovariance(key), rtol=1e-4, atol=1e-9)
    A, _ = instance.marginal_prior.linearize(values).jacobian()
    instance._window_graph(3)
    again, _ = instance.marginal_prior.linearize(values).jacobian()
    assert np.allclose(A.T @ A, again.T @ again)


def test_separate_loop_pgo_preserves_local_prior_and_reload(tmp_path):
    instance = PoseGraph(backend_config())
    frames = [backend_frame(i, new=i == 0) for i in range(6)]
    instance.add_keyframes(frames, 0, len(frames))
    instance.solve()
    assert instance._add_loop_pair(0, 5, np.eye(4))
    instance.solve(global_optimization=True)
    assert instance.marginal_prior.size() > 0 and instance.window_start == 3
    assert instance.loop_pgo.factor_kinds.count('loop') == 1
    instance.solve()
    assert instance.marginal_prior.size() > 0
    instance.save(tmp_path)
    restored = PoseGraph.load(tmp_path)
    assert np.allclose(restored.poses, instance.poses)
    assert np.all(np.isfinite(restored.solve()[5]))


def test_backend_trial_merge_removes_pgo_anchor_but_retains_local_priors():
    instance = PoseGraph(backend_config())
    frames = [backend_frame(0, new=True), backend_frame(1),
              backend_frame(2, new=True, group=1), backend_frame(3, group=1), backend_frame(4, group=1)]
    instance.add_keyframes(frames, 0, 5)
    instance.solve()
    assert instance._add_loop_pair(1, 3, np.eye(4))
    instance.remove_prior_factors([1])
    assert instance.prior_keys == [0, 2]
    assert instance.loop_pgo.removed_groups == {1}
    assert all(instance.graph.at(index) is not None for index in instance.navigation_prior_idxs[2])
    instance.solve(global_optimization=True)
    assert np.allclose(instance.poses, np.tile(np.eye(4), (5, 1, 1)), atol=1e-7)


def test_disabled_marginalization_fixes_predecessor_and_optimizes_first_window_state():
    instance = PoseGraph(backend_config(marginalization_enabled=False))
    frames = [backend_frame(i, new=i == 0) for i in range(5)]
    instance.add_keyframes(frames, 0, 5)
    pose = instance.states[2].pose.copy()
    pose[2, 3] = .004
    instance.graph_init.update(2, gtsam.Pose3(pose))
    predecessor = instance.states[1].pose.copy()
    instance.solve()
    np.testing.assert_array_equal(instance.states[1].pose, predecessor)
    assert abs(instance.poses[2][2, 3]) < 1e-7
    assert instance.marginal_prior.size() == 0


def test_backend_local_prior_does_not_absorb_loop_pose_constraints():
    instance = PoseGraph(backend_config())
    frames = [backend_frame(i, new=i == 0) for i in range(6)]
    instance.add_keyframes(frames, 0, 6)
    assert instance._add_loop_pair(0, 5, np.eye(4))
    window, values, _ = instance._window_graph(3)
    prior_keys = {key for i in range(instance.marginal_prior.size()) for key in instance.marginal_prior.at(i).keys()}
    assert set(state_keys(3)) == prior_keys
    assert instance.loop_pgo.loops[0][:2] == (0, 5)
    ordering = gtsam.Ordering()
    for key in sorted(prior_keys):
        ordering.push_back(key)
    H, g = instance.marginal_prior.linearize(values).hessian(ordering)
    assert np.linalg.norm(H - np.diag(np.diag(H))) > 1.
    assert np.allclose(g, 0., atol=1e-12)


def test_backend_discards_expired_component_without_separator():
    instance = PoseGraph(backend_config())
    frames = [backend_frame(0, new=True), backend_frame(1, new=True, group=1)]
    instance.add_keyframes(frames, 0, 2)
    window, values, _ = instance._window_graph(1)
    assert instance.marginal_prior.size() == 0
    assert instance._marginalized_factor_idxs == {0, 1, 2}
    assert window.size() == 3 and instance.graph.size() == 6
    assert window.error(values) == 0.


def test_frontend_option_can_retain_fixed_previous_state_mode():
    solver = GtsamVisualInertialSolver({"gtsam_vi_marginalization": False})
    result = solver.refine_current(
        FlatVisual(), np.eye(4), state(0), state(0.1), samples(0, 0.1), np.eye(4)
    )
    assert result is not None
    assert solver.pending_marginal_prior is None
    solver.commit_marginal_prior(result)
    assert solver.marginal_prior is None


def test_frontend_solves_strong_pose_rows_with_weak_navigation_information():
    rng = np.random.default_rng(21)

    class StrongPoseVisual:
        def __len__(self):
            return 100

        def evaluate(self, _reference, _pose):
            return np.zeros(300), np.zeros((300, 6)), J

    J = rng.normal(size=(300, 6)) * 1e5
    solver = GtsamVisualInertialSolver({
        "imu_residual_weight": 1e-6, "gtsam_vi_damping": 0.0,
        "gtsam_vi_prior_velocity_sigma_mps": 100.0,
        "gtsam_vi_prior_accel_bias_sigma_mps2": 100.0,
        "gtsam_vi_prior_gyro_bias_sigma_rps": 100.0,
    })
    result = solver.refine_current(
        StrongPoseVisual(), np.eye(4), state(0), state(0.1), samples(0, 0.1), np.eye(4)
    )
    assert result is not None
    assert np.allclose(result.pose, np.eye(4))
    assert np.allclose(result.velocity, 0.0)
    assert solver.pending_marginal_prior is not None
