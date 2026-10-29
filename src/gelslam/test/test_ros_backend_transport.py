"""ROS serialization, correction freshness and backend process ownership."""
from collections import deque
from dataclasses import replace
from threading import Condition
from types import SimpleNamespace

import numpy as np
import pytest
from cv_bridge import CvBridge
from rclpy.serialization import serialize_message, deserialize_message
from std_msgs.msg import Header
from gelslam_msgs.msg import KeyFrameMsg, BackendFeedbackMsg

from backend_test_support import config, frame
from gelslam.core.backend_feedback import FeedbackReceiver, feedback_message
from gelslam.core.frame import create_keyframe_msg
from gelslam.core.keyframe import keyframe_msg2keyframe
from gelslam.core.shared_map import SharedMap, MapEstimate
from gelslam.core.slam_packets import ArrayBridge, to_ros, keyframe_from_ros
from gelslam.core.topic_backend import TopicBackend
from gelslam.nodes.pose_graph_node import PoseGraphNode
from gelslam.utils import Logger


def packet(index=0, session='tracking-a'):
    f = frame(index, new=index == 0)
    f.L = f.L.astype(np.float32)  # Production Laplacian maps use ROS 32FC1.
    h = Header()
    h.stamp.sec = int(f.navigation_state.timestamp)
    h.stamp.nanosec = int(round((f.navigation_state.timestamp - h.stamp.sec) * 1e9))
    return replace(create_keyframe_msg(ArrayBridge(), f, index, index, index == 0,
        f.trial_T_curr, h, f.navigation_state, f.gravity, f.imu_samples),
        trial_id=0, session_id=session)


def roundtrip(p):
    message = to_ros(p, CvBridge())
    return deserialize_message(serialize_message(message), KeyFrameMsg)


def test_dds_roundtrip_preserves_full_precision_seed_anchor_and_immutable_maps():
    p = packet()
    pose = np.eye(4)
    pose[0, 3] = .123456789012345
    anchor = frame().navigation_state
    anchor.pose = pose.copy()
    p = replace(p, trial_t_curr=pose.ravel(), navigation_pose=pose.ravel(),
                anchor_kid=0, anchor_state=anchor, solution_revision=7)
    message = roundtrip(p)
    restored = keyframe_from_ros(message, CvBridge())
    np.testing.assert_array_equal(restored.navigation_pose.reshape(4, 4), pose)
    np.testing.assert_array_equal(restored.anchor_state.pose, pose)
    assert restored.session_id == p.session_id and restored.solution_revision == 7
    assert restored.anchor_kid == 0 and restored.trial_id == 0
    np.testing.assert_array_equal(restored.normal_map, p.normal_map)
    with pytest.raises(ValueError):
        restored.normal_map[0, 0] = 0
    with pytest.raises(ValueError):
        restored.anchor_state.pose[0, 3] = 0
    k = keyframe_msg2keyframe(CvBridge(), message, 0, True, 1.,
                             gelsight_T_imu=np.eye(4), compute_features=False)
    np.testing.assert_array_equal(k.navigation_state.pose, pose)
    np.testing.assert_array_equal(k.trial_T_curr, pose)


def feedback():
    store = SharedMap()
    p = packet()
    store.append_keyframe(p)
    state = frame().navigation_state
    state.pose[0, 3] = .01
    store.commit_solution(0, {0: MapEstimate(state, state.pose, state.pose, 0, 0)})
    msg = feedback_message(store.snapshot_for_tracking(), p.session_id)
    return deserialize_message(serialize_message(msg), BackendFeedbackMsg), state


def test_feedback_roundtrip_is_immutable_and_rejects_stale_or_other_session():
    message, expected = feedback()
    receiver = FeedbackReceiver('tracking-a', np.eye(4))
    assert receiver.receive(message)
    snapshot = receiver.snapshot()
    np.testing.assert_array_equal(snapshot.estimates[0].state.pose, expected.pose)
    assert snapshot.observations[0].trial_id == 0
    with pytest.raises(ValueError):
        snapshot.estimates[0].state.pose[0, 3] = 0
    assert not receiver.receive(message)
    message.session_id = 'old-tracking'
    message.revision += 100
    assert not receiver.receive(message)
    assert receiver.snapshot() is snapshot


@pytest.mark.parametrize('damage', ['length', 'nan', 'revision', 'duplicate'])
def test_invalid_feedback_does_not_replace_last_valid_snapshot(damage):
    message, _ = feedback()
    receiver = FeedbackReceiver('tracking-a', np.eye(4))
    assert receiver.receive(message)
    old = receiver.snapshot()
    message.revision += 1
    if damage == 'length':message.states.pop()
    if damage == 'nan':message.states[-1] = float('nan')
    if damage == 'revision':message.state_revisions[0] = message.revision + 1
    if damage == 'duplicate':message.kids.append(message.kids[0])
    with pytest.raises(ValueError):receiver.receive(message)
    assert receiver.snapshot() is old


def test_topic_backend_keeps_15d_seeds_and_resets_only_on_new_session_start():
    cfg = config(coverage_graph_overlap_threshold=.8, min_sift_inliers=8, scr_threshold=.3, ccs_threshold=.7)
    cfg['device_config'] = {'ppmm': 1.}
    service = TopicBackend(cfg, Logger(print_output=False))
    assert service.process([keyframe_from_ros(roundtrip(packet(i)), CvBridge()) for i in range(2)])
    assert service.mapper.graph.temporal_links == [(0, 1)]
    assert all(edge[2] is None for edge in service.mapper.graph.edges)
    assert not service.process([replace(packet(2), session_id='foreign')])
    assert service.mapper.db.size() == 2
    assert service.process([packet(0, 'tracking-b')])
    assert service.mapper.db.size() == 1 and service.session_id == 'tracking-b'


def test_backend_off_keeps_raw_data_without_optimization_or_feedback_states():
    cfg = config(coverage_graph_overlap_threshold=.8)
    cfg['device_config'] = {'ppmm': 1.}
    service = TopicBackend(cfg, Logger(print_output=False), enabled=False)
    assert service.process([packet(i) for i in range(3)])
    assert service.mapper.graph.states == [None] * 3 and service.search is None
    assert not feedback_message(service.store.snapshot_for_tracking(), 'tracking-a').kids


def test_end_marker_waits_for_the_last_keyframe_callback():
    node = SimpleNamespace(_work=Condition(), _backend=SimpleNamespace(session_id='tracking-a'),
                           _last_received={'tracking-a': 1}, _stream_ends={'tracking-a': 2})
    assert not PoseGraphNode.input_complete(node)
    node._last_received['tracking-a'] = 2
    assert PoseGraphNode.input_complete(node)


def test_backend_worker_failure_is_reported_to_coordinator():
    def fail(_packets):raise ValueError('backend sentinel')
    node = SimpleNamespace(_work=Condition(), running=True, _failure=None,
        _pending=deque([roundtrip(packet())]), _bridge=CvBridge(),
        backend_optimization_enabled=True, _backend=SimpleNamespace(process=fail))
    PoseGraphNode.loop_closure(node)
    assert not node.running
    with pytest.raises(RuntimeError, match='backend sentinel'):
        PoseGraphNode.raise_if_failed(node)


@pytest.mark.parametrize('enabled', [False, True])
def test_feedback_flag_preserves_backend_solve_and_map_publication(enabled):
    cfg = config(coverage_graph_overlap_threshold=.8, min_sift_inliers=8,
                 scr_threshold=.3, ccs_threshold=.7)
    cfg['device_config'] = {'ppmm': 1.}
    service = TopicBackend(cfg, Logger(print_output=False))
    published, trajectories = [], []
    node = SimpleNamespace(_work=Condition(), running=True, _failure=None,
        _pending=deque([roundtrip(packet(i)) for i in range(2)]), _bridge=CvBridge(),
        backend_optimization_enabled=True, backend_feedback_enabled=enabled,
        feedback_messages_published=0, rendering=False, _backend=service,
        feedback_pub=SimpleNamespace(publish=published.append),
        publish_gelsight_keyframe_trajectory=lambda: trajectories.append(True),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: Header().stamp)))
    process = service.process
    def process_once(packets):
        node.running = False
        return process(packets)
    service.process = process_once
    node._bind_backend = lambda: PoseGraphNode._bind_backend(node)
    PoseGraphNode.loop_closure(node)
    PoseGraphNode.raise_if_failed(node)
    assert node.processed_keyframe_count == 2
    assert node.pose_graph.temporal_links == [(0, 1)]
    assert service.store.snapshot_for_tracking().revision == 1
    assert trajectories == [True]
    assert len(published) == node.feedback_messages_published == int(enabled)
    if enabled:
        assert list(published[0].kids) == [0, 1]


@pytest.mark.parametrize('filename', ['bag_replay_launch.py', 'live_demo_launch.py',
                                     'live_replay_demo_launch.py'])
def test_standard_launch_uses_two_core_processes(filename):
    import importlib.util
    from pathlib import Path
    from launch import LaunchContext
    from launch.utilities import perform_substitutions, normalize_to_list_of_substitutions
    from launch_ros.actions import Node
    path = Path(__file__).resolve().parents[1] / 'launch' / filename
    spec = importlib.util.spec_from_file_location('process_launch_check', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    executables = [perform_substitutions(LaunchContext(),
                   normalize_to_list_of_substitutions(entity.node_executable))
                   for entity in module.generate_launch_description().entities
                   if isinstance(entity, Node)]
    assert 'tracker_node' in executables and 'pose_graph_node' in executables
    assert 'gelslam_node' not in executables
