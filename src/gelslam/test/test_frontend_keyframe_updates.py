"""Frontend keyframe re-estimates reach the backend seed of that keyframe."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from std_msgs.msg import Header

import gelslam.core.tracker as tracker_module
from gelslam.core.frame import create_keyframe_msg
from gelslam.core.imu_navigation import gelsight_pose
from gelslam.core.local_mapping import LocalMapping
from gelslam.core.shared_map import SharedMap, frozen_state
from gelslam.core.slam_packets import (
    ArrayBridge, KEYFRAME_UPDATE_SIZE, apply_keyframe_update, keyframe_update_row)
from gelslam.utils import Logger
from backend_test_support import config, frame
from test_ros_backend_transport import packet, roundtrip
from test_tracker_inertial import make_failing_contact_tracker


def rigid(rotvec, translation):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = translation
    return T


def moved(state, T=None, velocity=(0., 0., 0.), accel_bias=(0., 0., 0.), gyro_bias=(0., 0., 0.)):
    result = state.copy()
    if T is not None:
        result.pose = state.pose @ T
    result.velocity = state.velocity + np.asarray(velocity)
    result.accel_bias = state.accel_bias + np.asarray(accel_bias)
    result.gyro_bias = state.gyro_bias + np.asarray(gyro_bias)
    return result


def test_update_row_moves_a_seed_in_other_axes_like_the_frontend_estimate():
    basis = frame(1).navigation_state
    basis.velocity[:] = (.01, -.02, .003)
    estimate = moved(basis, rigid((0., .002, -.001), (.001, -.0005, .0002)),
                     velocity=(.002, .001, -.001), accel_bias=(1e-3, 0., -2e-3),
                     gyro_bias=(0., 1e-4, 0.))
    row = keyframe_update_row(4, basis, estimate)
    assert len(row) == KEYFRAME_UPDATE_SIZE and row[0] == 4 and row[1] == basis.timestamp
    # The backend stores the same keyframe in other world axes: seed = S basis.
    S = rigid((.1, -.2, .3), (.01, .02, -.03))
    seed = basis.copy()
    seed.pose = S @ basis.pose
    seed.velocity = S[:3, :3] @ basis.velocity
    updated = apply_keyframe_update(seed, row)
    np.testing.assert_allclose(updated.pose, S @ estimate.pose, atol=1e-12)
    np.testing.assert_allclose(updated.velocity, S[:3, :3] @ estimate.velocity, atol=1e-12)
    np.testing.assert_allclose(updated.accel_bias, estimate.accel_bias, atol=1e-15)
    np.testing.assert_allclose(updated.gyro_bias, estimate.gyro_bias, atol=1e-15)


def test_update_row_is_empty_without_change_or_at_another_time():
    basis = frame(1).navigation_state
    assert keyframe_update_row(1, basis, basis.copy()) == []
    assert keyframe_update_row(1, basis, None) == []
    later = moved(basis, rigid((0., 0., 0.), (.001, 0., 0.)))
    later.timestamp += .05
    assert keyframe_update_row(1, basis, later) == []
    with pytest.raises(ValueError):
        apply_keyframe_update(basis, [1., 2.])


def test_ros_roundtrip_keeps_the_keyframe_update():
    basis = frame(1).navigation_state
    row = keyframe_update_row(1, basis, moved(basis, rigid((0., 0., .001), (.001, 0., 0.))))
    restored = roundtrip(replace(packet(2), anchor_kid=1, anchor_state=frozen_state(basis),
                                 keyframe_update=np.asarray(row)))
    np.testing.assert_array_equal(np.asarray(restored.keyframe_update), row)
    assert len(roundtrip(packet(2)).keyframe_update) == 0


def updated_sequence(batches=((0, 1, 2),), enabled=True):
    """K_0 root, K_1, and K_2 carrying the frontend's re-estimate of K_1."""
    cfg = config(coverage_graph_overlap_threshold=.8)
    cfg['device_config'] = {'ppmm': 1.}
    store, mapper = SharedMap(), LocalMapping(cfg, Logger(print_output=False), enabled)
    first, second, third = packet(0), packet(1), packet(2)
    sent = frame(1).navigation_state                 # K_1 as published
    estimate = moved(sent, rigid((0., .001, 0.), (.001, -.0004, 0.)), velocity=(.003, 0., 0.))
    # K_2 was tracked against the re-estimated K_1: keep the published relative motion.
    sent2 = frame(2).navigation_state
    pose2 = estimate.pose @ np.linalg.solve(sent.pose, sent2.pose)
    third = replace(third, navigation_pose=pose2.reshape(-1),
                    trial_t_curr=gelsight_pose(pose2, mapper.graph.gelsight_T_imu).reshape(-1),
                    anchor_kid=1, anchor_state=frozen_state(sent),
                    keyframe_update=np.asarray(keyframe_update_row(1, sent, estimate)))
    packets = (first, second, third)
    for batch in batches:
        for index in batch:
            store.append_keyframe(packets[index])
        mapper.process(store.snapshot_for_optimization(), [], store)
    return mapper, store, sent, estimate, pose2


@pytest.mark.parametrize('batches', [((0, 1, 2),), ((0, 1), (2,))])
def test_backend_applies_the_update_and_odometry_uses_the_reestimated_keyframe(batches):
    mapper, store, sent, estimate, pose2 = updated_sequence(batches)
    np.testing.assert_allclose(mapper.db[1].navigation_state.pose, estimate.pose, atol=1e-12)
    np.testing.assert_allclose(mapper.db[1].navigation_state.velocity, estimate.velocity, atol=1e-12)
    np.testing.assert_allclose(mapper.graph.states[1].pose, estimate.pose, atol=1e-12)
    np.testing.assert_allclose(mapper.graph.states[2].pose, pose2, atol=1e-12)
    T = mapper.graph.gelsight_T_imu
    np.testing.assert_allclose(mapper.graph.poses[1], gelsight_pose(estimate.pose, T), atol=1e-12)
    # Contact odometry K_1 -> K_2 is now the frontend's relative motion from its K_1 estimate.
    odometry = np.linalg.solve(mapper.graph.poses[1], mapper.graph.poses[2])
    np.testing.assert_allclose(
        odometry, np.linalg.solve(gelsight_pose(sent.pose, T), gelsight_pose(frame(2).navigation_state.pose, T)),
        atol=1e-12)
    # Frames sent against K_1 stay resolved at its published pose.
    published = gelsight_pose(sent.pose, T)
    np.testing.assert_allclose(mapper.db[1].trial_T_curr @ np.linalg.inv(mapper.db[1].published_T_current),
                               published, atol=1e-12)
    np.testing.assert_allclose(store.snapshot_for_tracking().estimates[1].trial_pose,
                               gelsight_pose(estimate.pose, T), atol=1e-12)


def test_update_of_a_group_root_is_not_applied():
    cfg = config(coverage_graph_overlap_threshold=.8)
    cfg['device_config'] = {'ppmm': 1.}
    store, mapper = SharedMap(), LocalMapping(cfg, Logger(print_output=False))
    root = frame(0).navigation_state
    second = replace(packet(1), anchor_kid=0, anchor_state=frozen_state(root),
                     keyframe_update=np.asarray(keyframe_update_row(
                         0, root, moved(root, rigid((0., 0., 0.), (.001, 0., 0.))))))
    for p in (packet(0), second):
        store.append_keyframe(p)
    mapper.process(store.snapshot_for_optimization(), [], store)
    np.testing.assert_array_equal(mapper.db[0].navigation_state.pose, root.pose)


@pytest.mark.parametrize('reestimated', [True, False])
def test_next_keyframe_packet_sends_the_change_of_the_previous_reference(monkeypatch, reestimated):
    tracker, accepted = make_failing_contact_tracker(monkeypatch)
    monkeypatch.setattr(tracker_module, 'create_keyframe_msg', create_keyframe_msg)
    tracker.bridge = ArrayBridge()
    tracker.prev_header = tracker.ref_header = Header()
    tracker.prev_header.stamp.sec = 1
    surface = frame()
    for name in ('N', 'C', 'H', 'L'):
        setattr(tracker.prev_frame, name, getattr(surface, name))
    tracker.prev_frame.L = tracker.prev_frame.L.astype(np.float32)
    tracker._backend_imu_buffer = tracker.imu_buffer
    tracker._backend_keyframe_timestamp = .8
    tracker._trial_index = 0
    tracker._ros_session_id = 'update-test'
    basis = accepted.copy()
    basis.timestamp = .8
    estimate = moved(basis, rigid((0., .001, 0.), (.0007, 0., 0.)), velocity=(.001, 0., 0.))
    tracker._keyframe_seed_states = {2: frozen_state(basis)}
    tracker._keyframe_reference_estimates = {2: frozen_state(estimate)} if reestimated else {}
    emitted = tracker._maybe_add_timed_keyframe(
        tracker.prev_frame, tracker.prev_header, 'test', force=True)
    assert emitted.kid == 3 and emitted.anchor_kid == 2
    # The anchor basis is unchanged; the change to K_2's estimate travels separately.
    np.testing.assert_array_equal(emitted.anchor_state.pose, basis.pose)
    assert not emitted.keyframe_update.flags.writeable
    assert tracker._keyframe_reference_estimates == {}
    if reestimated:
        np.testing.assert_array_equal(emitted.keyframe_update, keyframe_update_row(2, basis, estimate))
    else:
        assert emitted.keyframe_update.size == 0


@pytest.mark.parametrize('batches', [((0, 1, 2),), ((0, 1), (2,))])
def test_backend_without_optimization_applies_the_update_to_raw_poses(batches):
    """backend:=false keeps no IMU/odometry factors but still follows keyframe updates."""
    mapper, _, sent, estimate, pose2 = updated_sequence(batches, enabled=False)
    T = mapper.graph.gelsight_T_imu
    assert mapper.graph.states == [None, None, None]
    np.testing.assert_allclose(mapper.graph.poses[1], gelsight_pose(estimate.pose, T), atol=1e-12)
    np.testing.assert_allclose(mapper.solutions[1], gelsight_pose(estimate.pose, T), atol=1e-6)
    np.testing.assert_allclose(mapper.solutions[2], gelsight_pose(pose2, T), atol=1e-6)
    np.testing.assert_allclose(mapper.db[1].published_T_current,
                               np.linalg.solve(gelsight_pose(sent.pose, T), gelsight_pose(estimate.pose, T)),
                               atol=1e-12)


def evaluation_node(tmp_path, frames, final_poses, changes):
    from threading import Lock
    from types import SimpleNamespace
    from gelslam.nodes.pose_graph_node import PoseGraphNode

    class Solutions:
        def size(self):
            return len(final_poses)

        def __getitem__(self, index):
            return final_poses[index]

    node = SimpleNamespace(
        trajectory_evaluation_enabled=True, trajectory_evaluation_config={}, trajectory_lock=Lock(),
        _trajectory_frames=[], _trajectory_ground_truth=[], add_keyframe_lock=Lock(),
        keyframedb=SimpleNamespace(
            get_kid2kidx=lambda: {kid: kid for kid in range(len(final_poses))},
            keyframes=[SimpleNamespace(kid=kid, published_T_current=change) for kid, change in enumerate(changes)]),
        pose_graph_solutions=Solutions(),
        parent_groups_info=SimpleNamespace(get_largest_parent_group=lambda: 0, get_parent_group=lambda *a: 0),
        trajectory_output_dir=str(tmp_path), keep_imu_only_frames=True, excluded_imu_only_frames=0,
        _header_timestamp=lambda _: None, logger=SimpleNamespace(
            info=lambda _: None, warning=lambda _: None, error=lambda text: pytest.fail(text)))
    for index, (ref_kid, ref_T, estimate) in enumerate(frames):
        message = SimpleNamespace(header=None, pose_valid=True, pose_timestamp=1. + .05 * index,
                                  ref_t_curr=ref_T.ravel(), fid=index, ref_kid=ref_kid,
                                  tracking_status='contact_tracking',
                                  published_t_ref_estimate=[] if estimate is None else estimate.ravel())
        PoseGraphNode.trajectory_frame_callback(node, message)
    PoseGraphNode.save_final_trajectory_evaluation(node)
    import json
    rows = np.genfromtxt(tmp_path / 'trajectory_final_raw.csv', delimiter=',', skip_header=1)
    return rows[:, 3:6], json.loads((tmp_path / 'trajectory_metrics.json').read_text())


def test_final_evaluation_moves_frames_tracked_against_an_earlier_keyframe_estimate(tmp_path):
    """K_1 was published at P_1, re-estimated to P_1 D (sent as keyframe_update), and
    K_2 was re-estimated but never updated (no later keyframe packet)."""
    P1, P2 = rigid((0., 0., .2), (.01, 0., 0.)), rigid((0., .1, 0.), (.02, .01, 0.))
    D = rigid((0., .002, 0.), (.0009, -.0003, .0001))
    E2 = rigid((0., 0., .001), (.0004, 0., 0.))
    ref_a, ref_b, ref_c = (rigid((0., 0., 0.), xyz) for xyz in ((0., 0., 0.), (.0006, 0., 0.), (0., .0005, 0.)))
    frames = [(1, ref_a, None),   # the recovery frame itself: tracked against the published K_1
              (1, ref_b, D),      # tracked against the re-estimated K_1
              (2, ref_c, E2)]     # K_2 re-estimate that never reached the backend
    final_poses = [np.eye(4), P1 @ D, P2]   # no optimization: K_1 raw pose = P_1 D
    changes = [np.eye(4), D, np.eye(4)]
    positions, metrics = evaluation_node(tmp_path, frames, final_poses, changes)
    np.testing.assert_allclose(positions[0], (P1 @ D @ ref_a)[:3, 3], atol=1e-9)   # moves with K_1
    np.testing.assert_allclose(positions[1], (P1 @ ref_b)[:3, 3], atol=1e-9)       # as published
    np.testing.assert_allclose(positions[2], (P2 @ ref_c)[:3, 3], atol=1e-9)       # as published
    assert metrics['keyframes_with_frontend_update'] == 1
    assert metrics['frames_moved_with_keyframe_update'] == 1


def test_frame_message_carries_the_reference_estimate():
    from types import SimpleNamespace
    from cv_bridge import CvBridge
    from rclpy.serialization import deserialize_message, serialize_message
    from gelslam_msgs.msg import FrameMsg
    from gelslam.core.slam_packets import to_ros
    published = rigid((0., 0., .3), (.01, .02, 0.))
    estimate = published @ rigid((0., .001, 0.), (.0005, 0., 0.))
    tracker = SimpleNamespace(
        frontend_visual_inertial=SimpleNamespace(reference_marginalization=True),
        _published_keyframe_poses={4: published}, trial_T_ref=estimate, bridge=ArrayBridge())
    surface = frame()
    surface.L = surface.L.astype(np.float32)
    ref_T_curr = rigid((0., 0., 0.), (.0003, 0., 0.))
    packet = tracker_module.Tracker._make_frame_msg(tracker, surface, 9, 4, ref_T_curr, Header())
    np.testing.assert_allclose(packet.published_t_ref_estimate.reshape(4, 4),
                               np.linalg.solve(published, estimate), atol=1e-12)
    np.testing.assert_allclose(published @ packet.ref_t_curr.reshape(4, 4), estimate @ ref_T_curr, atol=1e-7)
    message = deserialize_message(serialize_message(to_ros(packet, CvBridge())), FrameMsg)
    np.testing.assert_allclose(np.reshape(message.published_t_ref_estimate, (4, 4)),
                               np.linalg.solve(published, estimate), atol=1e-15)
    tracker.trial_T_ref = published
    unchanged = tracker_module.Tracker._make_frame_msg(tracker, surface, 9, 4, ref_T_curr, Header())
    assert unchanged.published_t_ref_estimate.size == 0


def test_reference_estimate_reaches_the_backend_record_through_frame_pose():
    """The backend records frames from FramePoseMsg (topic frame_pose), not FrameMsg."""
    from threading import Lock
    from types import SimpleNamespace
    from cv_bridge import CvBridge
    from rclpy.serialization import deserialize_message, serialize_message
    from gelslam_msgs.msg import FramePoseMsg
    from gelslam.core.frame import create_frame_msg
    from gelslam.nodes.pose_graph_node import PoseGraphNode
    from gelslam.nodes.tracker_node import TrackerNode
    surface = frame()
    surface.L = surface.L.astype(np.float32)
    estimate = rigid((0., .001, 0.), (.0009, 0., 0.))
    header = Header()
    header.stamp.sec = 2
    packet = create_frame_msg(ArrayBridge(), surface, 5, 3, np.eye(4), header,
                              published_T_ref_estimate=estimate)
    sent = []
    publisher = SimpleNamespace(bridge=CvBridge(), frame_pub=SimpleNamespace(publish=lambda _: None),
                                frame_pose_pub=SimpleNamespace(publish=sent.append))
    TrackerNode._publish_tracking_result(publisher, packet, [])
    message = deserialize_message(serialize_message(sent[0]), FramePoseMsg)
    backend = SimpleNamespace(_trajectory_frames=[], trajectory_lock=Lock(),
                              _header_timestamp=PoseGraphNode._header_timestamp)
    PoseGraphNode.trajectory_frame_callback(backend, message)
    np.testing.assert_allclose(backend._trajectory_frames[0][4], estimate, atol=1e-15)
