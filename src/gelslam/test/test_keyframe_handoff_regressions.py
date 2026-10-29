"""Keyframe anchor and loop-search coverage across asynchronous handoffs."""
from dataclasses import replace

import numpy as np
import pytest
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message, serialize_message
from gelslam_msgs.msg import KeyFrameMsg
from std_msgs.msg import Header

import gelslam.core.tracker as tracker_module
from gelslam.core.frame import create_keyframe_msg
from gelslam.core.gtsam_pose_graph import PoseGraph
from gelslam.core.local_mapping import LoopSearch
from gelslam.core.shared_map import SharedMap, frozen_state
from gelslam.core.slam_packets import ArrayBridge, keyframe_from_ros, to_ros
from gelslam.core.topic_backend import TopicBackend
from gelslam.utils import Logger
from backend_test_support import config, frame
from test_ros_backend_transport import packet
from test_tracker_inertial import make_failing_contact_tracker


@pytest.mark.parametrize('feedback_anchor', [False, True])
@pytest.mark.parametrize('promotion', ['previous', 'timed'])
def test_promoted_packet_keeps_previous_keyframe_basis_and_accepted_state(
        monkeypatch, feedback_anchor, promotion):
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
    tracker._ros_session_id = 'promotion-test'
    # K_2 at t=.8 is the frontend basis. F_4 at t=1 is accepted.
    # A failed later image must not replace F_4's state or timestamp.
    basis = accepted.copy()
    basis.timestamp = .8
    basis.pose[0, 3] = .03
    basis.velocity[:] = [.3, .2, .1]
    tracker._keyframe_seed_states = {2: frozen_state(basis)}
    tracker._tracking_anchor_kid = 1 if feedback_anchor else -1
    tracker._tracking_anchor_state = basis.copy() if feedback_anchor else None
    before = accepted.copy()
    prior = tracker.frontend_visual_inertial.marginal_prior
    if promotion == 'previous':
        tracker.curr_fid = 5
        success, (_, messages) = tracker._promote_previous_frame('test', 'failed F_5')
        assert success
        emitted = messages[0]
    else:
        emitted = tracker._maybe_add_timed_keyframe(
            tracker.prev_frame, tracker.prev_header, 'test', force=True)

    assert emitted.kid == tracker.curr_kid == 3
    assert emitted.fid == 4
    assert emitted.anchor_kid == 2
    assert emitted.anchor_state is not None
    assert emitted.anchor_state.timestamp == .8
    assert emitted.navigation_valid and emitted.navigation_timestamp == 1.
    assert tracker._frontend_vi_state is accepted
    assert tracker.frontend_visual_inertial.marginal_prior is prior
    assert tracker.ref_header.stamp.sec == 1
    for field in ('pose', 'velocity', 'accel_bias', 'gyro_bias'):
        np.testing.assert_array_equal(getattr(accepted, field), getattr(before, field))
        np.testing.assert_array_equal(getattr(emitted.anchor_state, field), getattr(basis, field))
        np.testing.assert_array_equal(getattr(tracker._keyframe_seed_states[3], field),
                                      getattr(before, field))
    # The two-process path must preserve the same anchor contract over DDS.
    message = to_ros(emitted, CvBridge())
    decoded = keyframe_from_ros(
        deserialize_message(serialize_message(message), KeyFrameMsg), CvBridge())
    assert decoded.anchor_kid == 2 and decoded.anchor_state.timestamp == .8
    np.testing.assert_array_equal(decoded.anchor_state.pose, basis.pose)
    assert not emitted.anchor_state.pose.flags.writeable
    saved = emitted.anchor_state.pose.copy()
    basis.pose[0, 3] += 1.
    np.testing.assert_array_equal(emitted.anchor_state.pose, saved)


def search_config():
    cfg = config(coverage_graph_overlap_threshold=.8)
    cfg['device_config'] = {'ppmm': 1.}
    return cfg


@pytest.mark.parametrize('batch_sizes', [(5,), (1, 4), (2, 1, 2), (1, 1, 1, 1, 1)])
def test_all_new_loop_targets_are_searched_once_in_order(monkeypatch, batch_sizes):
    search = LoopSearch(search_config(), Logger(print_output=False))
    store = SharedMap()
    targets, found = [], []
    transform = np.eye(4)
    transform[0, 3] = -.002
    def detect(db, target, cfg):
        targets.append(target)
        return [(0, target, transform)] if target == 2 else []
    monkeypatch.setattr(PoseGraph, 'detect_loop_candidates', staticmethod(detect))
    # IDs are distinct from DB indices; the emitted candidate must use IDs.
    packets = [replace(packet(i), kid=10 + 7 * i) for i in range(5)]
    offset = 0
    assert search.detect(store.snapshot_for_optimization()) == []
    for count in batch_sizes:
        for item in packets[offset:offset + count]:
            store.append_keyframe(item)
        offset += count
        snapshot = store.snapshot_for_optimization()
        found.extend(search.detect(snapshot))
        assert search.detect(replace(snapshot, revision=snapshot.revision + 1)) == []
    assert targets == [1, 2, 3, 4]
    assert len(found) == 1
    candidate = found[0]
    assert (candidate.reference_kid, candidate.target_kid) == (10, 24)
    assert candidate.epoch == 0 and candidate.revision == 0
    np.testing.assert_array_equal(candidate.target_T_reference, transform)
    assert not candidate.target_T_reference.flags.writeable
    transform[0, 3] = 1.
    assert candidate.target_T_reference[0, 3] == -.002


def test_loop_search_cursor_resets_with_map_epoch(monkeypatch):
    search = LoopSearch(search_config(), Logger(print_output=False))
    store, targets = SharedMap(), []
    def detect(db, target, cfg):
        targets.append(target)
        return [(0, target, np.eye(4))] if target == 2 else []
    monkeypatch.setattr(PoseGraph, 'detect_loop_candidates', staticmethod(detect))
    for epoch in range(2):
        for i in range(3):
            store.append_keyframe(packet(i))
        found = search.detect(store.snapshot_for_optimization())
        assert len(found) == 1 and found[0].epoch == epoch
        assert search.db.size() == 3
        store.reset()
    assert targets == [1, 2, 1, 2]


def test_failed_search_does_not_commit_a_partially_processed_batch(monkeypatch):
    search = LoopSearch(search_config(), Logger(print_output=False))
    store = SharedMap()
    for i in range(4):
        store.append_keyframe(packet(i))
    fail = True
    def detect(db, target, cfg):
        if fail and target == 3:
            raise RuntimeError('search failure')
        return [(0, target, np.eye(4))] if target == 2 else []
    monkeypatch.setattr(PoseGraph, 'detect_loop_candidates', staticmethod(detect))
    with pytest.raises(RuntimeError, match='search failure'):
        search.detect(store.snapshot_for_optimization())
    fail = False
    found = search.detect(store.snapshot_for_optimization())
    assert [(p.reference_kid, p.target_kid) for p in found] == [(0, 2)]
    assert search.detect(store.snapshot_for_optimization()) == []


@pytest.mark.parametrize('batches', [(5,), (1, 4), (1, 1, 1, 1, 1)])
def test_topic_backend_accepts_loop_from_middle_of_batch(monkeypatch, batches):
    service = TopicBackend(search_config(), Logger(print_output=False))
    def detect(db, target, cfg):
        return [(0, 2, np.eye(4))] if target == 2 else []
    monkeypatch.setattr(PoseGraph, 'detect_loop_candidates', staticmethod(detect))
    offset = 0
    for count in batches:
        assert service.process([packet(i) for i in range(offset, offset + count)])
        offset += count
    assert service.mapper.accepted_pairs == {(0, 2)}
    assert len(service.mapper.graph.loop_pgo.loops) == 1
    # A new tracking session resets both feature DB and search progress.
    assert service.process([packet(i, session='next') for i in range(3)])
    assert service.mapper.accepted_pairs == {(0, 2)}
    assert service.search.db.size() == 3
