"""Published frames must use the same reference pose as their keyframe packet."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import gelslam.core.tracker as tm
from gelslam.core.frame import create_frame_msg, create_keyframe_msg
from gelslam.core.imu_navigation import gelsight_pose
from gelslam.core.slam_packets import ArrayBridge, to_ros
from test_tracker_inertial import make_failing_contact_tracker, header


def published_reference(monkeypatch):
    tracker, accepted = make_failing_contact_tracker(monkeypatch, already_keyframe=True)
    monkeypatch.setattr(tm, 'create_frame_msg', create_frame_msg)
    monkeypatch.setattr(tm, 'create_keyframe_msg', create_keyframe_msg)
    tracker.bridge = ArrayBridge()
    tracker.frontend_visual_inertial.reference_marginalization = True
    monkeypatch.setattr(tracker.frontend_visual_inertial, 'set_reference',
                        lambda *_: None, raising=False)
    tracker.prev_frame.H = tracker.prev_frame.L = np.zeros((1, 1), np.float32)
    tracker.prev_frame.N = np.zeros((1, 1, 3), np.float32)
    tracker.gelsight_T_imu = np.diag((1., -1., -1., 1.))
    tracker.gelsight_T_imu[2, 3] = -.04
    accepted.pose = tracker.trial_T_ref @ tracker.gelsight_T_imu
    packet = tracker._make_keyframe_msg(
        tracker.ref_frame, tracker.curr_kid, tracker.prev_fid, True,
        tracker.trial_T_ref, tracker.ref_header)
    frozen = gelsight_pose(np.asarray(packet.navigation_pose).reshape(4, 4),
                           tracker.gelsight_T_imu)
    return tracker, accepted, packet, frozen


def moved_reference_and_target(frozen):
    shift = np.eye(4)
    shift[:3, :3] = Rotation.from_euler('xyz', (.01, -.02, .03)).as_matrix()
    shift[:3, 3] = (.002, -.003, .001)
    reference = shift @ frozen
    relative = np.eye(4)
    relative[:3, 3] = (.001, .002, -.0005)
    return reference, reference @ relative


def test_contact_frame_reconstructs_accepted_state_after_reference_pose_moves(monkeypatch):
    tracker, accepted, keyframe, frozen = published_reference(monkeypatch)
    reference, target = moved_reference_and_target(frozen)
    tracker.track_config['keyframe_max_interval_s'] = .5
    prior = tracker.frontend_visual_inertial.marginal_prior
    current = accepted.copy()
    current.timestamp = 1.1
    current.pose = target @ tracker.gelsight_T_imu

    def refine(*args, **kwargs):
        tracker.trial_T_ref = reference.copy()
        tracker._frontend_vi_state = current
        return np.linalg.inv(reference) @ target

    monkeypatch.setattr(tracker, '_refine_normalflow_with_imu', refine)
    ok, (message, new_keyframes) = tracker.track(
        np.zeros((1, 1, 3), np.uint8), header(1.1))
    assert ok and new_keyframes == []
    assert message.ref_kid == keyframe.kid
    np.testing.assert_allclose(
        frozen @ np.asarray(message.ref_t_curr).reshape(4, 4), target, atol=1e-9)
    np.testing.assert_array_equal(tracker.trial_T_ref, reference)
    np.testing.assert_array_equal(tracker._frontend_vi_state.pose, current.pose)
    assert tracker.frontend_visual_inertial.marginal_prior is prior


@pytest.mark.parametrize('mode', ('no_contact_coasting', 'imu_prediction', 'imu_only'))
def test_prediction_packets_keep_the_published_reference_through_ros_roundtrip(monkeypatch, mode):
    from cv_bridge import CvBridge
    from rclpy.serialization import deserialize_message, serialize_message
    from gelslam_msgs.msg import FrameMsg
    from std_msgs.msg import Header

    tracker, accepted, _, frozen = published_reference(monkeypatch)
    reference, target = moved_reference_and_target(frozen)
    tracker.trial_T_ref = reference
    # Recovery can update mutable navigation seeds after publication.
    tracker._keyframe_seed_states[tracker.curr_kid] = accepted.copy()
    tracker._keyframe_seed_states[tracker.curr_kid].pose[:3, 3] += 1.
    stamp = Header()
    stamp.stamp.sec, stamp.stamp.nanosec = 1, 100000000
    message = tracker._make_frame_msg(
        tracker.prev_frame, 5, tracker.curr_kid, np.linalg.inv(reference) @ target,
        header=stamp, tracking_status=mode, pose_timestamp=1.1)
    # Use actual float32 image arrays for the DDS transport.
    from dataclasses import replace
    message = replace(message, height_map=message.height_map.astype(np.float32),
                      normal_map=message.normal_map.astype(np.float32),
                      laplacian_map=message.laplacian_map.astype(np.float32))
    encoded = to_ros(message, CvBridge())
    decoded = deserialize_message(serialize_message(encoded), FrameMsg)
    assert decoded.tracking_status == mode and decoded.pose_valid
    assert decoded.pose_timestamp == 1.1
    np.testing.assert_allclose(
        frozen @ np.asarray(decoded.ref_t_curr).reshape(4, 4), target, atol=2e-8)


@pytest.mark.parametrize('force', (False, True), ids=('timed', 'low_scr'))
def test_keyframe_promotion_keeps_one_publication_basis_for_both_endpoints(monkeypatch, force):
    tracker, accepted, old_keyframe, frozen = published_reference(monkeypatch)
    reference, previous_target = moved_reference_and_target(frozen)
    tracker.trial_T_ref = reference.copy()
    before = tracker._make_frame_msg(
        tracker.prev_frame, 5, old_keyframe.kid,
        np.linalg.inv(reference) @ previous_target, header=header(1.45))

    current_target = previous_target.copy()
    current_target[:3, 3] += (.0002, -.0001, .00005)
    current = accepted.copy()
    current.timestamp = 1.5
    current.pose = current_target @ tracker.gelsight_T_imu
    tracker._frontend_vi_state = current
    tracker.track_config['keyframe_max_interval_s'] = .5
    tracker.curr_fid = 6
    tracker.ref_T_prev = np.linalg.inv(reference) @ current_target
    prior = tracker.frontend_visual_inertial.marginal_prior

    new_keyframe = tracker._maybe_add_timed_keyframe(
        tracker.prev_frame, header(1.5), 'test', force=force)
    assert new_keyframe is not None
    after = tracker._make_frame_msg(
        tracker.prev_frame, 6, new_keyframe.kid,
        tracker.ref_T_prev, header=header(1.5))
    published_new = gelsight_pose(
        np.asarray(new_keyframe.navigation_pose).reshape(4, 4), tracker.gelsight_T_imu)

    # A common backend transform must preserve the true small inter-frame step.
    backend_transform = np.eye(4)
    backend_transform[:3, :3] = Rotation.from_euler('z', .2).as_matrix()
    backend_transform[:3, 3] = (.01, -.02, .03)
    reconstructed_before = (backend_transform @ frozen
                            @ np.asarray(before.ref_t_curr).reshape(4, 4))
    reconstructed_after = (backend_transform @ published_new
                           @ np.asarray(after.ref_t_curr).reshape(4, 4))
    np.testing.assert_allclose(
        reconstructed_before, backend_transform @ previous_target, atol=1e-9)
    np.testing.assert_allclose(
        reconstructed_after, backend_transform @ current_target, atol=1e-9)
    np.testing.assert_allclose(
        np.linalg.inv(reconstructed_before) @ reconstructed_after,
        np.linalg.inv(previous_target) @ current_target, atol=1e-9)
    np.testing.assert_array_equal(tracker._published_keyframe_poses[old_keyframe.kid], frozen)
    assert tracker._frontend_vi_state is current
    assert tracker.frontend_visual_inertial.marginal_prior is prior
    for field in ('pose', 'velocity', 'accel_bias', 'gyro_bias'):
        expected = current_target @ tracker.gelsight_T_imu if field == 'pose' else getattr(accepted, field)
        np.testing.assert_array_equal(getattr(current, field), expected)
