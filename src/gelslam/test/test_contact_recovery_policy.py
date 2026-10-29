"""Contact decisions, uncertainty, and segment boundaries with real IMU factors."""
from types import SimpleNamespace

import numpy as np
import pytest

import gelslam.core.tracker as tm
from gelslam.core.gtsam_vi import (
    GtsamVisualInertialSolver, _bias_factor, _imu_factor, create_frontend_solver,
    navigation_factor_terms,
)
from gelslam.core.inertial import ImuSample
from gelslam.core.marginal_prior import NavigationPrior
from gelslam.core.numpy_vi import NavigationState
from gelslam.core.tracker import TrackingMode
from test_tracker_inertial import make_failing_contact_tracker, header


class FlatVisual:
    def __len__(self):
        return 10

    def evaluate(self, *_args):
        return np.zeros(30), np.zeros((30, 6)), np.zeros((30, 6))


def tracker_with_imu(monkeypatch):
    tracker, initial = make_failing_contact_tracker(monkeypatch, already_keyframe=True)
    tracker.track_config.update(contact_failure_keyframe_interval_s=.5,
                                keyframe_max_interval_s=.5)
    tracker.imu_config["max_sample_gap_s"] = .05
    tracker._trial_index = 7
    tracker.logger = SimpleNamespace(info=lambda _: None, warning=lambda _: None)
    for time in np.arange(1.62, 4.02, .02):
        tracker.imu_buffer.add(ImuSample(
            float(time), np.array([0, 0, 9.81]) + initial.accel_bias,
            np.array([0, 0, .4]) + initial.gyro_bias))
    monkeypatch.setattr(tm, "DenseNormalResidual", lambda *a, **k: FlatVisual())
    return tracker, initial


@pytest.mark.parametrize("scr,ccs,accepted,keyframes", [
    (.29, .70, True, 1), (.30, .70, True, 0),
    (.29, .69, False, 0), (.31, .69, False, 0),
    (float("nan"), .9, False, 0), (.1, float("nan"), False, 0),
])
def test_current_image_quality_decides_commit_and_immediate_keyframe(
        monkeypatch, scr, ccs, accepted, keyframes):
    tracker, anchor = tracker_with_imu(monkeypatch)
    solver = tracker.frontend_visual_inertial
    prior = solver.marginal_prior
    monkeypatch.setattr(tm, "normalflow_quality", lambda *a: (scr, ccs))
    ret, result = tracker.track(np.zeros((1, 1, 3), np.uint8), header(1.04))
    assert ret is accepted
    if accepted:
        message, packets = result
        assert len(packets) == keyframes
        assert tracker._frontend_vi_state.timestamp == pytest.approx(1.04)
        assert solver.marginal_prior.matches(tracker._frontend_vi_state)
        assert tracker.tracking_mode is TrackingMode.CONTACT_TRACKING
        assert tracker.last_tactile_header is message["header"]
        if keyframes:
            assert packets[0]["fid"] == message["fid"]
            assert packets[0]["tactile_valid"]
            np.testing.assert_allclose(packets[0]["trial_T_curr"], tracker._frontend_vi_state.pose)
    else:
        assert result is None
        assert tracker._frontend_vi_state is anchor
        assert solver.marginal_prior is prior
        assert solver.pending_marginal_prior is None


def reject_contact(*_args, **_kwargs):
    raise tm.QualityGateError(np.eye(4), scr=.5, ccs=.1)


@pytest.mark.parametrize("initialize_from_imu", [False, True])
@pytest.mark.parametrize("state_mode", ["pairwise", "reference_marginalization"])
@pytest.mark.parametrize("scr,target_time,keyframes", [
    (.9, 1.24, 0), (.29, 1.24, 1), (.9, 1.70, 1),
])
def test_recontact_without_shared_points_recovers_immediately_then_tracks(
        monkeypatch, initialize_from_imu, state_mode, scr, target_time, keyframes):
    tracker, anchor = tracker_with_imu(monkeypatch)
    tracker.imu_config["frontend_state_mode"] = state_mode
    solver = create_frontend_solver(dict(
        frontend_state_mode=state_mode,
        frontend_initialize_from_imu_preintegration=initialize_from_imu,
    ))
    tracker.frontend_visual_inertial = solver
    solver.marginal_prior = NavigationPrior.initial(anchor, solver.prior_sigmas)
    tracker._sync_reference_prior()
    contact = tracker.prev_frame
    tracker.last_tactile_header = tactile_header = header(1.)
    tracker.last_tactile_ref_T = np.eye(4)
    image = np.zeros((1, 1, 3), np.uint8)
    monkeypatch.setattr(tm, "Frame", lambda *a, **k: SimpleNamespace(is_contacted=False))
    assert tracker.track(image, header(1.10))[0]
    assert tracker._frontend_vi_state is anchor
    monkeypatch.setattr(tm, "Frame", type(contact))

    class NoSharedPoints:
        def __len__(self):
            return 0

    monkeypatch.setattr(tm, "DenseNormalResidual", lambda *a, **k: NoSharedPoints())
    expected, _ = solver.predict_current(
        anchor, tracker.imu_buffer.samples_covering(1., 1.20), np.eye(4), force=True)
    recovery_header = header(1.20)
    ok, (message, packets) = tracker.track(image, recovery_header)
    assert ok and len(packets) == 1
    assert message["tracking_status"] == "imu_only"
    assert not packets[0]["tactile_valid"] and not packets[0]["is_new_trial"]
    assert tracker._trial_index == 7
    assert tracker._frontend_vi_state.timestamp == pytest.approx(1.20)
    np.testing.assert_allclose(tracker._frontend_vi_state.pose, expected.pose)
    assert solver.marginal_prior.matches(tracker._frontend_vi_state)
    assert solver.pending_marginal_prior is None
    assert tracker.last_tactile_header is tactile_header
    assert tracker._contact_failure_since is None
    recovery_kid = tracker.curr_kid

    monkeypatch.setattr(tm, "DenseNormalResidual", lambda *a, **k: FlatVisual())
    monkeypatch.setattr(tm, "normalflow_quality", lambda *a: (scr, .9))
    ok, (message, packets) = tracker.track(image, header(target_time))
    assert ok and len(packets) == keyframes
    assert tracker._frontend_vi_state.timestamp == pytest.approx(target_time)
    assert tracker.tracking_mode is TrackingMode.CONTACT_TRACKING
    assert solver.marginal_prior.matches(tracker._frontend_vi_state)
    assert not tracker._reference_is_imu_only
    assert tracker.frontend_tracking_statistics()["recovered_contact_keyframes"] == 0
    if keyframes:
        assert packets[0]["tactile_valid"]
        assert tracker.curr_kid == recovery_kid + 1
        assert tracker.frontend_tracking_statistics()["low_scr_keyframes"] == (scr < .3)
    else:
        assert tracker.curr_kid == message["ref_kid"] == recovery_kid
        assert tracker.ref_header is recovery_header
        assert not np.allclose(message["ref_T_curr"], np.eye(4))
        expected_reference = solver.last_reference_state.pose.copy()
        np.testing.assert_allclose(tracker.trial_T_ref, expected_reference)
        ok, (message, packets) = tracker.track(image, header(1.28))
        assert ok and not packets
        assert tracker.curr_kid == message["ref_kid"] == recovery_kid
        assert tracker.ref_header is recovery_header
        assert solver.marginal_prior.matches(tracker._frontend_vi_state)
        if state_mode == "reference_marginalization":
            assert solver.reference_id.timestamp == pytest.approx(1.20)
            assert solver.last_problem["solve_dimension"] == 36
            assert solver.marginal_prior.layout.dimension == 21


def test_recontact_quality_rejection_does_not_start_a_new_half_second_wait(monkeypatch):
    tracker, _ = tracker_with_imu(monkeypatch)
    tracker.prev_in_contact_flag = False
    monkeypatch.setattr(tracker, "_track_contact", reject_contact)
    ok, (message, packets) = tracker.track(np.zeros((1, 1, 3), np.uint8), header(1.04))
    assert ok and message["tracking_status"] == "imu_only"
    assert len(packets) == 1 and not packets[0]["tactile_valid"]
    assert tracker._frontend_vi_state.timestamp == pytest.approx(1.04)
    assert tracker._contact_failure_since is None


def test_only_continuous_quality_failures_accumulate_the_half_second_wait(monkeypatch):
    tracker, _ = tracker_with_imu(monkeypatch)
    image = np.zeros((1, 1, 3), np.uint8)
    monkeypatch.setattr(tracker, "_track_contact", reject_contact)
    assert tracker.track(image, header(1.04)) == (False, None)
    assert tracker._contact_failure_since == pytest.approx(1.04)

    def reject_registration(*args, **kwargs):
        raise tm.LoseTrackError("initial_shared_points_lt_10")

    monkeypatch.setattr(tracker, "_track_contact", reject_registration)
    ok, (message, _) = tracker.track(image, header(1.20))
    assert ok and message["tracking_status"] == "imu_only"
    assert tracker._contact_failure_since is None

    monkeypatch.setattr(tracker, "_track_contact", reject_contact)
    assert tracker.track(image, header(1.24)) == (False, None)
    assert tracker._contact_failure_since == pytest.approx(1.24)
    assert tracker.track(image, header(1.73)) == (False, None)
    ok, (message, _) = tracker.track(image, header(1.74))
    assert ok and message["tracking_status"] == "imu_only"


def test_recontact_without_valid_imu_retains_the_accepted_state(monkeypatch):
    from gelslam.core.inertial import ImuBuffer
    tracker, anchor = tracker_with_imu(monkeypatch)
    solver = tracker.frontend_visual_inertial
    prior = solver.marginal_prior
    tracker.imu_buffer = ImuBuffer()
    tracker.prev_in_contact_flag = False
    monkeypatch.setattr(tracker, "_track_contact", reject_contact)
    assert tracker.track(np.zeros((1, 1, 3), np.uint8), header(1.20)) == (False, None)
    assert tracker._frontend_vi_state is anchor
    assert solver.marginal_prior is prior and solver.pending_marginal_prior is None
    assert tracker._trial_index == 7 and tracker.curr_kid == 2
    assert tracker._contact_failure_since is None


def test_half_second_and_repeated_contact_failures_keep_trial_and_prior(monkeypatch):
    tracker, anchor = tracker_with_imu(monkeypatch)
    tracker.last_tactile_header = tactile_header = header(1.0)
    tracker.last_tactile_ref_T = np.eye(4)
    monkeypatch.setattr(tracker, "_refine_normalflow_with_imu", reject_contact)
    image = np.zeros((1, 1, 3), np.uint8)
    assert tracker.track(image, header(1.04)) == (False, None)
    assert tracker.track(image, header(1.53)) == (False, None)
    for time in (1.54, 2.04, 2.54):
        ret, (message, packets) = tracker.track(image, header(time))
        assert ret and len(packets) == 1
        assert not packets[0]["is_new_trial"]
        assert not packets[0]["tactile_valid"]
        assert message["tracking_status"] == "imu_only"
        assert tracker._trial_index == 7
        assert tracker.last_tactile_header is tactile_header
        assert tracker.tracking_mode is TrackingMode.IMU_ONLY
        assert tracker.frontend_visual_inertial.marginal_prior.matches(tracker._frontend_vi_state)
        from gelslam.core.state_velocity_evaluation import capture_state_velocity
        assert capture_state_velocity(tracker, SimpleNamespace(**message))["stage"] == "imu_recovery_prediction"
    assert tracker._frontend_vi_state.timestamp > anchor.timestamp + 1


def test_success_at_failure_deadline_emits_only_one_tactile_keyframe(monkeypatch):
    tracker, _ = tracker_with_imu(monkeypatch)
    tracker._contact_failure_since = 1.04
    monkeypatch.setattr(tm, "normalflow_quality", lambda *a: (.2, .9))
    ret, (message, packets) = tracker.track(np.zeros((1, 1, 3), np.uint8), header(1.54))
    assert ret and len(packets) == 1 and packets[0]["tactile_valid"]
    assert tracker._contact_failure_since is None
    assert tracker.tracking_mode is TrackingMode.CONTACT_TRACKING


def test_no_contact_timeout_stages_moving_restart_without_empty_trials(monkeypatch):
    tracker, initial = tracker_with_imu(monkeypatch)
    contact = tracker.prev_frame
    image = np.zeros((1, 1, 3), np.uint8)
    monkeypatch.setattr(tm, "Frame", lambda *a, **k: SimpleNamespace(is_contacted=False))
    tracker.track(image, header(1.10))
    tracker.track(image, header(2.12))
    assert tracker._restart_pending
    assert tracker._trial_index == 7
    monkeypatch.setattr(tm, "Frame", lambda *a, **k: contact)
    monkeypatch.setattr(tm, "normalflow_quality", lambda *a: (.2, .9))
    assert tracker.track(image, header(2.14)) == (False, None)
    assert tracker._trial_index == 7
    assert tracker._frontend_vi_state is initial
    ret, (message, packets) = tracker.track(image, header(2.18))
    assert ret and len(packets) == 2
    assert packets[0]["is_new_trial"] and not packets[1]["is_new_trial"]
    assert tracker._trial_index == 8
    assert tracker._frontend_vi_state.timestamp == pytest.approx(2.18)
    assert np.linalg.norm(tracker._frontend_vi_state.velocity) > .01
    assert not tracker._restart_pending


def test_real_imu_gap_requests_new_stationary_window_not_startup_interval(monkeypatch):
    tracker, initial = tracker_with_imu(monkeypatch)
    from gelslam.core.inertial import ImuBuffer
    tracker.imu_buffer = ImuBuffer()
    for time in (1., 1.2, 1.22):
        tracker.imu_buffer.add(ImuSample(time, (0, 0, 9.81), (0, 0, 0)))
    prior = tracker.frontend_visual_inertial.marginal_prior
    tracker._static_imu_settings = {"mode": "fixed_time", "start_time_s": 1., "end_time_s": 2.9}
    tracker.track(np.zeros((1, 1, 3), np.uint8), header(1.22))
    assert tracker._restart_pending and tracker._restart_needs_static
    assert tracker._restart_initializer.settings["mode"] == "stationary_detection"
    assert tracker._restart_initializer.settings["start_time_s"] is None
    assert tracker._frontend_vi_state is initial
    assert tracker.frontend_visual_inertial.marginal_prior is prior


def test_propagated_prior_matches_full_joint_gaussian_mean_and_covariance():
    solver = GtsamVisualInertialSolver({
        "accelerometer_noise_density": .8, "gyroscope_noise_density": .6,
        "accelerometer_bias_random_walk": .5, "gyroscope_bias_random_walk": .4,
        "gtsam_integration_sigma_m": .1,
    })
    previous = NavigationState(1., np.eye(4), np.zeros(3), np.zeros(3), np.zeros(3))
    rng = np.random.default_rng(25)
    A = np.eye(15) * 3 + rng.normal(0, .1, (15, 15))
    rhs = rng.normal(0, .02, 15)
    solver.marginal_prior = original = NavigationPrior(previous.copy(), A, rhs)
    samples = [ImuSample(t, (0, 0, 9.81), (0, 0, 0)) for t in (1., 1.1)]
    current, pim = solver.predict_current(previous, samples, np.eye(4), force=True)
    propagated = solver.propagate_prior(previous, current, pim)
    factors = [_imu_factor(pim), _bias_factor(pim, solver.config)]
    rows = [navigation_factor_terms(f, previous, current) for f in factors]
    joint = np.vstack([np.column_stack((A, np.zeros_like(A)))] +
                      [np.column_stack((left, right)) for left, right, _ in rows])
    target = np.concatenate([rhs] + [row[2] for row in rows])
    joint_mean = np.linalg.lstsq(joint, target, rcond=None)[0]
    joint_covariance = np.linalg.inv(joint.T @ joint)
    np.testing.assert_allclose(
        np.linalg.solve(propagated.A.T @ propagated.A, propagated.A.T @ propagated.rhs),
        joint_mean[15:], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(np.linalg.inv(propagated.A.T @ propagated.A),
                               joint_covariance[15:, 15:], rtol=1e-7, atol=1e-9)
    assert solver.marginal_prior is original
    assert np.linalg.norm(propagated.rhs) > 0


def test_frame_status_survives_ros_serialization():
    from cv_bridge import CvBridge
    from rclpy.serialization import serialize_message, deserialize_message
    from std_msgs.msg import Header
    from gelslam_msgs.msg import FrameMsg
    from gelslam.core.frame import create_frame_msg
    from gelslam.core.slam_packets import ArrayBridge, to_ros
    f = SimpleNamespace(H=np.zeros((2, 2), np.float32), C=np.zeros((2, 2), np.uint8),
                        N=np.zeros((2, 2, 3), np.float32), L=np.zeros((2, 2), np.float32))
    h = Header()
    h.stamp.sec = 2
    packet = create_frame_msg(ArrayBridge(), f, 3, 1, np.eye(4), h,
                              tracking_status="imu_only", pose_valid=False, pose_timestamp=1.99)
    restored = deserialize_message(serialize_message(to_ros(packet, CvBridge())), FrameMsg)
    assert restored.tracking_status == "imu_only" and not restored.pose_valid
    assert restored.pose_timestamp == 1.99 and restored.header.stamp.sec == 2


def test_feedback_transports_imu_reference_prior_and_keeps_reference_variable(monkeypatch):
    from time import monotonic
    from scipy.spatial.transform import Rotation
    tracker, initial = tracker_with_imu(monkeypatch)
    solver = tracker.frontend_visual_inertial
    tracker._reference_is_imu_only = True
    tracker._last_tactile_trial_pose = initial.pose.copy()
    tracker._backend_imu_buffer = tracker.imu_buffer
    tracker._keyframe_seed_states = {tracker.curr_kid: initial.copy()}
    prior = solver.marginal_prior
    prior.rhs[:] = np.linspace(-.01, .01, len(prior.rhs))
    correction = np.eye(4)
    correction[:3, :3] = Rotation.from_euler("z", .1).as_matrix()
    correction[:3, 3] = (.01, .02, 0)
    corrected = initial.reframe(correction[:3, :3], -correction[:3, :3].T @ correction[:3, 3])
    packet = SimpleNamespace(kid=tracker.curr_kid, trial_id=7)
    estimate = SimpleNamespace(state=corrected, revision=1, trial_pose=corrected.pose,
                               committed_at=monotonic())
    snapshot = SimpleNamespace(epoch=0, observations=[packet], estimates={packet.kid: estimate}, revision=1)
    assert tracker.apply_map_snapshot(snapshot, header(1.04))
    assert solver.marginal_prior.matches(tracker._frontend_vi_state)
    np.testing.assert_allclose(tracker._last_tactile_trial_pose, corrected.pose)
    np.testing.assert_array_equal(solver.marginal_prior.rhs, prior.rhs)
    np.testing.assert_allclose(tracker._frontend_vi_state.velocity, corrected.velocity)
    monkeypatch.setattr(tm, "normalflow_quality", lambda *a: (.9, .9))
    tracker._refine_normalflow_with_imu(np.eye(4), header(1.04), tracker.ref_frame, tracker.prev_frame)
    assert solver.last_prior_source == "propagated_recovery_reference"
    assert tracker._backend_correction_anchor is None
    # The next packet must describe the reference endpoint used to validate
    # this pair, including its optimized velocity and biases.
    basis = tracker._keyframe_seed_states[tracker.curr_kid]
    for field in ("pose", "velocity", "accel_bias", "gyro_bias"):
        np.testing.assert_array_equal(getattr(basis, field),
                                      getattr(solver.last_reference_state, field))
    assert basis is not solver.last_reference_state


def test_imu_recovery_uses_pending_backend_anchor_without_stale_visual_prior(monkeypatch):
    tracker, initial = tracker_with_imu(monkeypatch)
    solver = tracker.frontend_visual_inertial
    samples = tracker.imu_buffer.samples_covering(1., 1.1)
    accepted, pim = solver.predict_current(initial, samples, np.eye(4))
    solver.marginal_prior = solver.propagate_prior(initial, accepted, pim)
    tracker._frontend_vi_state = accepted
    anchor = initial.copy()
    anchor.pose[0, 3] += .02
    tracker._backend_correction_anchor = anchor
    tracker._backend_correction_trial = 7
    tracker._backend_imu_buffer = tracker.imu_buffer
    samples = tracker.imu_buffer.samples_covering(1., 1.54)
    expected, _ = solver.predict_current(anchor, samples, np.eye(4), force=True)
    assert tracker._recover_current_frame_with_imu(tracker.prev_frame, header(1.54), "test")[0]
    np.testing.assert_allclose(tracker._frontend_vi_state.pose, expected.pose)
    assert solver.marginal_prior.matches(tracker._frontend_vi_state)
    assert tracker._backend_correction_anchor is None


def test_backend_trajectory_uses_actual_pose_timestamp_and_skips_invalid_pose():
    from threading import Lock
    from gelslam.nodes.pose_graph_node import PoseGraphNode
    node = SimpleNamespace(_trajectory_frames=[], trajectory_lock=Lock(),
                           _header_timestamp=lambda _: 20.)
    msg = SimpleNamespace(header=None, pose_valid=True, pose_timestamp=19.9,
                          ref_t_curr=np.eye(4).ravel(), fid=4, ref_kid=2)
    PoseGraphNode.trajectory_frame_callback(node, msg)
    assert node._trajectory_frames[0][0] == 19.9
    msg.pose_valid = False
    PoseGraphNode.trajectory_frame_callback(node, msg)
    assert len(node._trajectory_frames) == 1
