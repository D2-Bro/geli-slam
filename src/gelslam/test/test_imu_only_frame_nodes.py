"""Retain image-time IMU predictions without changing accepted tracking states."""

from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest
from std_msgs.msg import Header

import gelslam.core.tracker as tm
from gelslam.core.frame import create_frame_msg
from gelslam.core.imu_navigation import gelsight_pose
from gelslam.core.inertial import ImuBuffer, ImuSample
from gelslam.core.marginal_prior import NavigationPrior
from gelslam.core.slam_packets import ArrayBridge
from gelslam.core.state_velocity_evaluation import capture_state_velocity
from gelslam.core.tracker import Tracker, TrackingMode
from gelslam.nodes.pose_graph_node import PoseGraphNode
from gelslam.nodes.tracker_node import TrackerNode
from test_tracker_inertial import make_failing_contact_tracker, header


IMAGE = np.zeros((1, 1, 3), np.uint8)


def prediction_tracker(monkeypatch, *, enabled=True, already_keyframe=True):
    tracker, accepted = make_failing_contact_tracker(monkeypatch, already_keyframe)
    tracker.track_config.update(keep_imu_only_frames=enabled,
                                contact_failure_keyframe_interval_s=.5)
    tracker.imu_config['max_sample_gap_s'] = .05
    tracker._trial_index = 3
    if not already_keyframe:
        tracker.ref_T_prev[:3, 3] = [.003, -.002, .001]
        accepted.pose = tracker.trial_T_ref @ tracker.ref_T_prev
        solver = tracker.frontend_visual_inertial
        solver.marginal_prior = NavigationPrior.initial(accepted, solver.prior_sigmas)
        tracker._sync_reference_prior()

    def reject(*_args):
        raise tm.QualityGateError(np.eye(4))

    monkeypatch.setattr(tracker, '_track_contact', reject)
    monkeypatch.setattr(tm, 'create_frame_msg',
                        lambda _bridge, frame, **kwargs: dict(kwargs, source_frame=frame))
    return tracker, accepted


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('already_keyframe', [False, True])
def test_quality_wait_keeps_each_image_only_with_flag_and_preserves_anchor(
        monkeypatch, enabled, already_keyframe):
    tracker, anchor = prediction_tracker(monkeypatch, enabled=enabled,
                                         already_keyframe=already_keyframe)
    solver = tracker.frontend_visual_inertial
    prior = solver.marginal_prior
    tactile_frame, tactile_header = tracker.prev_frame, tracker.prev_header
    first_fid, first_kid = tracker.curr_fid, tracker.curr_kid
    messages, keyframes = [], []
    for timestamp in (1.04, 1.08, 1.12):
        expected, _ = solver.predict_current(
            anchor, tracker.imu_buffer.samples_covering(anchor.timestamp, timestamp),
            tracker.gelsight_T_imu, force=True)
        ret, result = tracker.track(IMAGE, header(timestamp))
        if ret:
            message, packets = result
            keyframes.extend(packets)
            if message is not None:
                messages.append(message)
                assert message['tracking_status'] == 'imu_prediction'
                assert message['pose_timestamp'] == pytest.approx(timestamp)
                assert message['ref_kid'] == tracker.curr_kid
                assert message['source_frame'] is tracker.latest_observation_frame
                np.testing.assert_allclose(
                    tracker.trial_T_ref @ message['ref_T_curr'],
                    gelsight_pose(expected.pose, tracker.gelsight_T_imu), atol=1e-7)
                record = capture_state_velocity(tracker, SimpleNamespace(**message))
                assert record['stage'] == 'imu_frame_prediction'
                assert record['state_timestamp_s'] == pytest.approx(timestamp)
                np.testing.assert_allclose(record['velocity'], expected.velocity)
        assert tracker._frontend_vi_state is anchor
        assert solver.marginal_prior.matches(anchor)
        if timestamp == 1.04 and not already_keyframe:
            # Promotion of the accepted frame retains the prior on the new
            # reference. Later predictions must not change it.
            prior = solver.marginal_prior
        assert solver.marginal_prior is prior
        assert solver.pending_marginal_prior is None
        assert getattr(tracker, '_frontend_coast_state', None) is None
        assert tracker.prev_frame is tactile_frame and tracker.prev_header is tactile_header
        assert tracker.tracking_mode is TrackingMode.TACTILE_HOLD
        assert tracker._contact_failure_since == pytest.approx(1.04)
    assert len(keyframes) == int(not already_keyframe)
    assert tracker.curr_kid == first_kid + int(not already_keyframe)
    assert [message['fid'] for message in messages] == (
        [first_fid + 1, first_fid + 2, first_fid + 3] if enabled else [])
    assert tracker.curr_fid == first_fid + (3 if enabled else 0)


@pytest.mark.parametrize('enabled', [False, True])
def test_frame_retention_does_not_move_imu_recovery_deadline(monkeypatch, enabled):
    tracker, anchor = prediction_tracker(monkeypatch, enabled=enabled)
    tracker.track(IMAGE, header(1.04))
    tracker.track(IMAGE, header(1.40))
    ret, (message, packets) = tracker.track(IMAGE, header(1.54))
    assert ret and len(packets) == 1
    assert not packets[0]['tactile_valid']
    assert message['tracking_status'] == 'imu_only'
    assert tracker._frontend_vi_state is not anchor
    assert tracker._frontend_vi_state.timestamp == pytest.approx(1.54)
    assert tracker._frontend_frame_prediction is None
    assert capture_state_velocity(tracker, SimpleNamespace(**message))['stage'] == 'imu_recovery_prediction'


@pytest.mark.parametrize('problem', ['missing_endpoint', 'gap', 'uninitialized', 'restart'])
def test_invalid_imu_interval_does_not_create_an_evaluation_frame(monkeypatch, problem):
    tracker, anchor = prediction_tracker(monkeypatch)
    if problem == 'missing_endpoint':
        tracker.imu_buffer = ImuBuffer()
        times = (1., 1.02)
    elif problem == 'gap':
        tracker.imu_buffer = ImuBuffer()
        times = (1., 1.04, 1.12)
    else:
        times = ()
    for timestamp in times:
        tracker.imu_buffer.add(ImuSample(timestamp, (0, 0, 9.81), (0, 0, .4)))
    if problem == 'uninitialized':
        tracker._frontend_vi_state = None
    elif problem == 'restart':
        tracker._restart_pending = True
    assert tracker._imu_prediction_frame_msg(tracker.prev_frame, header(1.12), 5) is None
    assert tracker._frontend_vi_state is (None if problem == 'uninitialized' else anchor)


@pytest.mark.parametrize('already_keyframe', [False, True])
def test_prediction_and_later_recovery_keep_coast_shift_through_reference_promotion(
        monkeypatch, already_keyframe):
    from gelslam.core.contact_velocity import CoastVelocitySeed, ContactVelocityHistory
    from gelslam.core.gtsam_vi import create_frontend_solver
    tracker, anchor = prediction_tracker(monkeypatch, already_keyframe=already_keyframe)
    solver = create_frontend_solver({})
    tracker.frontend_visual_inertial = solver
    solver.marginal_prior = NavigationPrior.initial(anchor, solver.prior_sigmas)
    tracker._sync_reference_prior()
    prior = solver.marginal_prior
    for timestamp in (1.62, 1.64):
        tracker.imu_buffer.add(ImuSample(timestamp, np.array([0, 0, 9.81])+anchor.accel_bias,
                                       np.array([0, 0, .4])+anchor.gyro_bias))
    tracker.last_tactile_ref_T = tracker.ref_T_prev.copy()
    tracker.last_tactile_header = tracker.prev_header
    tracker._contact_velocity_history = ContactVelocityHistory.from_config(dict(
        coast_velocity_fit=dict(enabled=True)))
    seed = CoastVelocitySeed(anchor.timestamp, np.array([.004, -.003, .001]), 0., 8, .35)
    tracker._coast_velocity_seed = seed
    tracker._coast_velocity_anchor_timestamp = anchor.timestamp
    assert tracker._initialize_no_contact_coast()
    assert tracker._coast_to_header(header(1.10), TrackingMode.NO_CONTACT_COASTING)
    np.testing.assert_allclose(tracker._frontend_coast_state.velocity,
                               seed.apply(anchor).velocity)

    # Quality rejection waits before accepting an IMU recovery State. During
    # this wait publication must use the same corrected prediction as coast.
    for timestamp in (1.12, 1.16):
        raw, _ = solver.predict_current(anchor, tracker.imu_buffer.samples_covering(
            anchor.timestamp, timestamp), tracker.gelsight_T_imu, force=True)
        expected = seed.apply(raw)
        ok, (message, packets) = tracker.track(IMAGE, header(timestamp))
        assert ok and len(packets) == (not already_keyframe and timestamp == 1.12)
        # Consumers reconstruct F_j from the published K_k observation, not
        # the live reference updated during reference promotion.
        published_ref = getattr(tracker, '_published_keyframe_poses', {}).get(
            message['ref_kid'], tracker.trial_T_ref)
        np.testing.assert_allclose(published_ref @ message['ref_T_curr'],
                                   gelsight_pose(expected.pose, tracker.gelsight_T_imu), atol=1e-7)
        np.testing.assert_allclose(tracker._frontend_frame_prediction.velocity, expected.velocity)
        assert tracker._frontend_vi_state is anchor
        assert solver.marginal_prior is prior
        assert tracker._coast_velocity_seed is seed
        assert not tracker._contact_velocity_history.samples

    raw, _ = solver.predict_current(anchor, tracker.imu_buffer.samples_covering(
        anchor.timestamp, 1.62), tracker.gelsight_T_imu, force=True)
    expected = seed.apply(raw)
    ok, (message, packets) = tracker.track(IMAGE, header(1.62))
    assert ok and len(packets) == 1 and message['tracking_status'] == 'imu_only'
    np.testing.assert_allclose(tracker._frontend_vi_state.pose, expected.pose, atol=1e-12)
    np.testing.assert_allclose(tracker._frontend_vi_state.velocity, expected.velocity, atol=1e-12)
    assert solver.marginal_prior.matches(tracker._frontend_vi_state)
    assert tracker._coast_velocity_seed is None


def test_prediction_does_not_apply_coast_shift_to_another_anchor(monkeypatch):
    from gelslam.core.contact_velocity import CoastVelocitySeed
    tracker, accepted = prediction_tracker(monkeypatch)
    tracker._coast_velocity_seed = CoastVelocitySeed(
        accepted.timestamp-.1, np.array([.004, -.003, .001]), 0., 8, .35)
    anchor = accepted
    expected, _ = tracker.frontend_visual_inertial.predict_current(
        anchor, tracker.imu_buffer.samples_covering(anchor.timestamp, 1.12),
        tracker.gelsight_T_imu, force=True)
    message = tracker._imu_prediction_frame_msg(tracker.prev_frame, header(1.12), 5)
    np.testing.assert_allclose(tracker._frontend_frame_prediction.pose, expected.pose)
    np.testing.assert_allclose(tracker._frontend_frame_prediction.velocity, expected.velocity)
    assert tracker._frontend_vi_state is accepted
    assert message['tracking_status'] == 'imu_prediction'


@pytest.mark.parametrize('enabled', [False, True])
def test_no_contact_evaluation_preserves_existing_surface_and_prediction(monkeypatch, enabled):
    tracker, anchor = prediction_tracker(monkeypatch, enabled=enabled)
    tactile_frame = tracker.prev_frame
    observation = SimpleNamespace(is_contacted=False)
    monkeypatch.setattr(tm, 'Frame', lambda *a, **k: observation)
    ret, (message, packets) = tracker.track(IMAGE, header(1.10))
    assert ret and not packets
    assert message['tracking_status'] == 'no_contact_coasting'
    assert message['pose_valid']
    assert message['source_frame'] is tactile_frame
    assert capture_state_velocity(tracker, SimpleNamespace(**message))['stage'] == 'no_contact_prediction'
    assert tracker._frontend_vi_state is anchor


def test_prediction_frame_reaches_ros_and_final_backend_trajectory(monkeypatch, tmp_path):
    tracker, _ = prediction_tracker(monkeypatch)
    monkeypatch.setattr(tm, 'create_frame_msg', create_frame_msg)
    monkeypatch.setattr(tm, 'Frame', lambda *a, **k: SimpleNamespace(
        is_contacted=True, H=np.zeros((1, 1), np.float32),
        C=np.ones((1, 1), np.uint8), N=np.zeros((1, 1, 3), np.float32),
        L=np.zeros((1, 1), np.float32)))
    tracker.bridge = ArrayBridge()
    backend = SimpleNamespace(_trajectory_frames=[], trajectory_lock=Lock(),
                              _header_timestamp=PoseGraphNode._header_timestamp)
    ros_frames = []
    publisher = SimpleNamespace(
        bridge=tm.CvBridge(), frame_pub=SimpleNamespace(publish=ros_frames.append),
        frame_pose_pub=SimpleNamespace(
            publish=lambda msg: PoseGraphNode.trajectory_frame_callback(backend, msg)))
    for timestamp in (1.04, 1.08, 1.12):
        image_header = Header()
        image_header.stamp.sec = 1
        image_header.stamp.nanosec = round((timestamp - 1) * 1e9)
        ret, (message, packets) = tracker.track(IMAGE, image_header)
        assert ret and not packets
        TrackerNode._publish_tracking_result(publisher, message, packets)
    assert [message.tracking_status for message in ros_frames] == ['imu_prediction'] * 3
    assert [record[0] for record in backend._trajectory_frames] == pytest.approx([1.04, 1.08, 1.12])
    final_pose = np.eye(4)
    final_pose[:3, 3] = [1, 2, 3]

    class FinalPoses:
        def size(self):
            return 1

        def __getitem__(self, _index):
            return final_pose

    backend.trajectory_evaluation_enabled = True
    backend.trajectory_evaluation_config = {}
    backend._trajectory_ground_truth = []
    backend.add_keyframe_lock = Lock()
    backend.keyframedb = SimpleNamespace(get_kid2kidx=lambda: {2: 0},
                                         keyframes=[SimpleNamespace(kid=2)])
    backend.pose_graph_solutions = FinalPoses()
    backend.parent_groups_info = SimpleNamespace(get_largest_parent_group=lambda: 0,
                                                get_parent_group=lambda *a: 0)
    backend.trajectory_output_dir = str(tmp_path)
    backend.logger = SimpleNamespace(info=lambda _: None, warning=lambda _: None,
                                      error=lambda text: pytest.fail(text))
    PoseGraphNode.save_final_trajectory_evaluation(backend)
    rows = np.genfromtxt(tmp_path / 'trajectory_final_raw.csv', delimiter=',', skip_header=1)
    assert rows[:, 0] == pytest.approx([1.04, 1.08, 1.12])
    for row, record in zip(rows, backend._trajectory_frames):
        np.testing.assert_allclose(row[3:6], (final_pose @ record[3])[:3, 3])


@pytest.mark.parametrize('value', ['true', 1, None])
def test_keep_imu_only_frames_requires_a_yaml_boolean(value):
    config = dict(device_config={'ppmm': 1.}, contact_mask_config={},
                  gelslam_config={'track_config': {'keep_imu_only_frames': value}})
    with pytest.raises(ValueError, match='keep_imu_only_frames must be a boolean'):
        Tracker('unused.pth', config, True, SimpleNamespace())
