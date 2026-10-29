"""PGO feedback changes the frame of the retained frontend posterior."""
from types import SimpleNamespace

import numpy as np
import pytest

from gelslam.core.backend_feedback import FeedbackReceiver, feedback_message
from gelslam.core.gtsam_vi import GtsamVisualInertialSolver
from gelslam.core.imu_navigation import gelsight_pose
from gelslam.core.inertial import ImuBuffer, ImuSample
from gelslam.core.marginal_prior import NavigationPrior
from gelslam.core.numpy_vi import NavigationState
from gelslam.core.shared_map import MapEstimate, SharedMap
from gelslam.core.tracker import Tracker, TrackingMode
import gelslam.core.tracker as tm
from test_tracker_inertial import header


def tracker_and_feedback(with_prior=True, extrinsic=False):
    tracker = Tracker.__new__(Tracker)
    tracker.curr_kid, tracker._trial_index = 0, 0
    tracker.gelsight_T_imu = np.eye(4)
    if extrinsic:
        tracker.gelsight_T_imu[:3, :3] = np.diag([1., -1., -1.])
        tracker.gelsight_T_imu[:3, 3] = [.01, -.02, .04]
    tracker.imu_config = dict(gtsam_vi_marginalization=True, gtsam_vi_max_iterations=2)
    tracker.frontend_visual_inertial = GtsamVisualInertialSolver(tracker.imu_config)
    seed = NavigationState(1., np.eye(4), np.array([.02, .01, 0.]),
                           np.zeros(3), np.zeros(3))
    accepted = seed.copy()
    accepted.timestamp = 1.05
    accepted.pose[:3, 3] = [.001, .0005, 0.]
    accepted.accel_bias[:] = [.001, -.002, .003]
    accepted.gyro_bias[:] = [.0001, -.0002, .0003]
    tracker._frontend_vi_state = accepted
    tracker._keyframe_seed_states = {0: seed}
    tracker.trial_T_ref = gelsight_pose(seed.pose, tracker.gelsight_T_imu)
    tracker.ref_T_prev = (np.linalg.inv(tracker.trial_T_ref)
                         @ gelsight_pose(accepted.pose, tracker.gelsight_T_imu)).astype(np.float32)
    tracker._last_tactile_trial_pose = gelsight_pose(accepted.pose, tracker.gelsight_T_imu)
    tracker.ref_header, tracker.prev_header = header(1.), header(1.05)
    tracker.ref_frame = object()
    tracker._imu_coasting_ready = lambda: True
    tracker.imu_buffer, tracker._backend_imu_buffer = ImuBuffer(), ImuBuffer()
    for time in np.linspace(1., 1.1, 11):
        tracker.imu_buffer.add(ImuSample(time, [0., 0., 9.81], [0., 0., 0.]))
    tracker._frontend_imu_preintegration = tracker._frontend_coast_state = object()
    tracker.ppmm = 1.
    tracker.track_config = dict(scr_threshold=.3, ccs_threshold=.5)
    tracker._record_normal_correspondences = lambda *a, **k: None
    if with_prior:
        A = np.random.default_rng(3).normal(size=(20, 15))
        tracker.frontend_visual_inertial.marginal_prior = NavigationPrior(
            accepted.copy(), A, np.arange(20) * .001)
    store = SharedMap()
    store.append_keyframe(SimpleNamespace(kid=0, trial_id=0))
    receiver = FeedbackReceiver('tracking', tracker.gelsight_T_imu)
    return tracker, store, receiver


def correction(tracker, store, receiver, transform):
    anchor = tracker._keyframe_seed_states[0].copy()
    anchor.pose = transform @ anchor.pose
    anchor.velocity = transform[:3, :3] @ anchor.velocity
    pose = gelsight_pose(anchor.pose, tracker.gelsight_T_imu)
    store.commit_solution(0, {0: MapEstimate(anchor, pose, pose, 0, 0)})
    assert receiver.receive(feedback_message(store.snapshot_for_tracking(), 'tracking'))
    return receiver.snapshot()


def yaw_shift():
    angle = .3
    result = np.eye(4)
    result[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    result[:3, 3] = [.004, -.003, .002]
    return result


@pytest.mark.parametrize('change', ['velocity', 'bias', 'tilt'])
@pytest.mark.parametrize('imu_reference', [False, True])
def test_global_15d_feedback_uses_corrected_navigation_anchor(change, imu_reference):
    tracker, store, receiver = tracker_and_feedback()
    tracker._reference_is_imu_only = imu_reference
    tracker._backend_imu_buffer = tracker.imu_buffer
    accepted = tracker._frontend_vi_state.copy()
    reference = tracker.trial_T_ref.copy()
    anchor = tracker._keyframe_seed_states[0].copy()
    if change == 'velocity':
        anchor.velocity += [.01, -.02, .03]
    elif change == 'bias':
        anchor.accel_bias += [.001, -.002, .003]
        anchor.gyro_bias += [.0001, -.0002, .0003]
    else:
        import gtsam
        anchor.pose[:3, :3] = gtsam.Rot3.Expmap(np.array([.01, -.02, 0.])).matrix()
    pose = gelsight_pose(anchor.pose, tracker.gelsight_T_imu)
    store.commit_solution(0, {0: MapEstimate(anchor, pose, pose, 0, 0)})
    assert receiver.receive(feedback_message(store.snapshot_for_tracking(), 'tracking'))
    assert tracker.apply_map_snapshot(receiver.snapshot(), header(1.1))
    assert tracker._backend_correction_is_global_navigation
    shift = pose @ np.linalg.inv(reference)
    actual = tracker._frontend_vi_state
    np.testing.assert_allclose(actual.pose, shift @ accepted.pose)
    np.testing.assert_allclose(actual.velocity, shift[:3, :3] @ accepted.velocity)
    assert actual.timestamp == accepted.timestamp
    for field in ('accel_bias', 'gyro_bias'):
        np.testing.assert_array_equal(getattr(actual, field), getattr(accepted, field))
    corrected, buffer, fixed = tracker._recovery_prediction_anchor()
    assert fixed and buffer is tracker._backend_imu_buffer
    for field in ('pose', 'velocity', 'accel_bias', 'gyro_bias'):
        np.testing.assert_allclose(getattr(corrected, field), getattr(anchor, field))


@pytest.mark.parametrize('imu_reference', [False, True])
def test_next_target_uses_global_velocity_and_bias_corrections(monkeypatch, imu_reference):
    tracker, store, receiver = tracker_and_feedback()
    tracker._reference_is_imu_only = imu_reference
    tracker._backend_imu_buffer = tracker.imu_buffer
    solver = tracker.frontend_visual_inertial
    anchor = tracker._keyframe_seed_states[0].copy()
    anchor.velocity += [.003, -.002, .001]
    anchor.accel_bias += [.001, -.002, .003]
    anchor.gyro_bias += [.0001, -.0002, .0003]
    if imu_reference:
        tracker._frontend_vi_state.timestamp = 1.
    solver.marginal_prior = NavigationPrior.initial(tracker._frontend_vi_state, np.ones(15))
    pose = gelsight_pose(anchor.pose, tracker.gelsight_T_imu)
    store.commit_solution(0, {0: MapEstimate(anchor, pose, pose, 0, 0)})
    assert receiver.receive(feedback_message(store.snapshot_for_tracking(), 'tracking'))
    assert tracker.apply_map_snapshot(receiver.snapshot(), header(1.1))
    seen = []
    original = solver.refine_from_keyframe

    def capture(*args, **kwargs):
        seen.append(args[2].copy())
        return original(*args, **kwargs)

    class FlatVisual:
        def __len__(self):
            return 10

        def evaluate(self, *_args):
            return np.zeros(30), np.zeros((30, 6)), np.zeros((30, 6))

    monkeypatch.setattr(solver, 'refine_from_keyframe', capture)
    monkeypatch.setattr(tm, 'DenseNormalResidual', lambda *a, **k: FlatVisual())
    monkeypatch.setattr(tm, 'normalflow_quality', lambda *a: (1., 1.))
    tracker._refine_normalflow_with_imu(np.eye(4), header(1.1), tracker.ref_frame, object())
    assert len(seen) == 1
    for field in ('pose', 'velocity', 'accel_bias', 'gyro_bias'):
        np.testing.assert_allclose(getattr(seen[0], field), getattr(anchor, field))
    assert solver.last_prior_source == 'backend_keyframe'
    assert tracker._frontend_vi_state.timestamp == pytest.approx(1.1)
    assert solver.marginal_prior.matches(tracker._frontend_vi_state)
    assert tracker._backend_correction_anchor is None


@pytest.mark.parametrize('with_prior', [False, True])
@pytest.mark.parametrize('extrinsic', [False, True])
def test_pgo_feedback_transports_state_and_full_prior_without_keyframe_imu_history(
        with_prior, extrinsic):
    tracker, store, receiver = tracker_and_feedback(with_prior, extrinsic)
    original = tracker._frontend_vi_state.copy()
    prior = tracker.frontend_visual_inertial.marginal_prior
    tactile = tracker._last_tactile_trial_pose.copy()
    relative = tracker.ref_T_prev.copy()
    step, total = yaw_shift(), np.eye(4)
    for _ in range(2):
        snapshot = correction(tracker, store, receiver, step)
        total = step @ total
        assert tracker.apply_map_snapshot(snapshot, header(1.1))
        actual = tracker._frontend_vi_state
        np.testing.assert_allclose(actual.pose, total @ original.pose, atol=1e-12)
        np.testing.assert_allclose(actual.velocity, total[:3, :3] @ original.velocity, atol=1e-12)
        assert actual.timestamp == original.timestamp
        for field in ('accel_bias', 'gyro_bias'):
            np.testing.assert_array_equal(getattr(actual, field), getattr(original, field))
        np.testing.assert_allclose(tracker.ref_T_prev, relative, atol=1e-8)
        np.testing.assert_allclose(tracker._last_tactile_trial_pose, total @ tactile, atol=1e-12)
        assert tracker._backend_correction_anchor is None
        assert tracker._recovery_prediction_anchor()[0] is actual
        assert tracker._recovery_prediction_anchor()[2] is False
        current_prior = tracker.frontend_visual_inertial.marginal_prior
        if prior is None:
            assert current_prior is None
        else:
            expected = prior.reframe(total[:3, :3], -total[:3, :3].T @ total[:3, 3])
            np.testing.assert_allclose(current_prior.A, expected.A, atol=1e-12)
            np.testing.assert_allclose(current_prior.anchor.pose, actual.pose, atol=1e-12)
            np.testing.assert_allclose(current_prior.anchor.velocity, actual.velocity, atol=1e-12)
            np.testing.assert_array_equal(current_prior.rhs, prior.rhs)
            probe = original.copy()
            probe.velocity += [.01, -.02, .03]
            moved = probe.reframe(total[:3, :3], -total[:3, :3].T @ total[:3, 3])
            assert current_prior.error(moved) == pytest.approx(prior.error(probe), abs=1e-12)
        assert not tracker.apply_map_snapshot(snapshot, header(1.1))
        assert tracker._frontend_vi_state is actual
    assert tracker._frontend_tracking_counts['backend_feedback_reframed'] == 2


def test_next_contact_uses_last_accepted_state_and_retained_prior(monkeypatch):
    tracker, store, receiver = tracker_and_feedback()
    solver = tracker.frontend_visual_inertial
    solver.marginal_prior = NavigationPrior.initial(tracker._frontend_vi_state, np.ones(15))
    assert tracker.apply_map_snapshot(correction(tracker, store, receiver, yaw_shift()), header(1.1))
    previous = tracker._frontend_vi_state.copy()

    class FlatVisual:
        def __len__(self):
            return 10

        def evaluate(self, *_args):
            return np.zeros(30), np.zeros((30, 6)), np.zeros((30, 6))

    def unexpected(*_args, **_kwargs):
        pytest.fail('PGO-only feedback must not condition on a fixed old keyframe')

    monkeypatch.setattr(solver, 'refine_from_keyframe', unexpected)
    monkeypatch.setattr(tm, 'DenseNormalResidual', lambda *a, **k: FlatVisual())
    monkeypatch.setattr(tm, 'normalflow_quality', lambda *a: (1., 1.))
    tracker._refine_normalflow_with_imu(tracker.ref_T_prev, header(1.1), tracker.ref_frame, object())
    assert solver.last_prior_source == 'retained'
    assert tracker._frontend_vi_state.timestamp == pytest.approx(1.1)
    for field in ('accel_bias', 'gyro_bias'):
        np.testing.assert_allclose(getattr(tracker._frontend_vi_state, field), getattr(previous, field), atol=1e-10)
    assert solver.marginal_prior.matches(tracker._frontend_vi_state)
    assert np.max(np.diag(solver.marginal_prior.A.T @ solver.marginal_prior.A)[9:]) < 10.


def test_no_contact_prediction_uses_rotated_velocity_and_keeps_accepted_posterior():
    tracker, store, receiver = tracker_and_feedback()
    original = tracker._frontend_vi_state.copy()
    shift = yaw_shift()
    solver = tracker.frontend_visual_inertial
    samples = tracker.imu_buffer.samples_covering(original.timestamp, 1.1)
    expected, _ = solver.predict_current(original, samples, tracker.gelsight_T_imu, force=True)
    assert tracker.apply_map_snapshot(correction(tracker, store, receiver, shift), header(1.1))
    accepted, prior = tracker._frontend_vi_state, solver.marginal_prior
    assert tracker._initialize_no_contact_coast()
    assert tracker._coast_to_header(header(1.1), TrackingMode.NO_CONTACT_COASTING)
    np.testing.assert_allclose(tracker._frontend_coast_state.pose, shift @ expected.pose, atol=1e-10)
    np.testing.assert_allclose(tracker._frontend_coast_state.velocity, shift[:3, :3] @ expected.velocity, atol=1e-10)
    assert tracker._frontend_vi_state is accepted
    assert solver.marginal_prior is prior
