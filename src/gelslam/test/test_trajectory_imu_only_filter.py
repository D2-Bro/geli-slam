"""Final trajectory evaluation of IMU-only frames.

IMU recovery keyframe frames (imu_only) are always evaluated. keep_imu_only_frames
false leaves the coast and imu_prediction frames out; true evaluates them, coast
frames at their IMU prediction time.
"""
import json
from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest

from gelslam.nodes.pose_graph_node import (
    IMU_ONLY_TRACKING_STATUSES, OPTIONAL_IMU_ONLY_STATUSES, PoseGraphNode)

STATUSES = ("contact_tracking", "no_contact_coasting", "imu_only", "imu_prediction",
            "tactile_hold", "reacquire")


def backend(keep):
    return SimpleNamespace(_trajectory_frames=[], trajectory_lock=Lock(),
                           _header_timestamp=lambda _: None, keep_imu_only_frames=keep,
                           excluded_imu_only_frames=0, invalid_pose_frames=0)


def frame(index, status, pose_valid=True, pose_timestamp=None):
    return SimpleNamespace(header=None, pose_valid=pose_valid,
                           pose_timestamp=1. + .05 * index if pose_timestamp is None
                           else pose_timestamp,
                           ref_t_curr=np.eye(4).ravel(), fid=index, ref_kid=0,
                           tracking_status=status)


@pytest.mark.parametrize('keep', [False, True])
def test_recovery_frames_stay_and_other_imu_only_frames_follow_the_setting(keep):
    node = backend(keep)
    for index, status in enumerate(STATUSES):
        PoseGraphNode.trajectory_frame_callback(node, frame(index, status))
    kept = [STATUSES[record[1]] for record in node._trajectory_frames]
    if keep:
        assert kept == list(STATUSES) and node.excluded_imu_only_frames == 0
    else:
        assert kept == ['contact_tracking', 'imu_only', 'tactile_hold', 'reacquire']
        assert node.excluded_imu_only_frames == 2
    assert [record[5] for record in node._trajectory_frames] == kept
    assert {'no_contact_coasting', 'imu_only', 'imu_prediction'} <= IMU_ONLY_TRACKING_STATUSES
    assert OPTIONAL_IMU_ONLY_STATUSES == IMU_ONLY_TRACKING_STATUSES - {'imu_only'}


def test_kept_coast_frames_use_their_prediction_time():
    # The tracker marks a coast frame pose_valid false: its prediction ends at the
    # last IMU sample (pose_timestamp) before the image time.
    node = backend(True)
    PoseGraphNode.trajectory_frame_callback(node, frame(0, 'contact_tracking'))
    PoseGraphNode.trajectory_frame_callback(
        node, frame(1, 'no_contact_coasting', pose_valid=False, pose_timestamp=1.047))
    # No new IMU sample: the prediction did not advance, so the pose is stale.
    PoseGraphNode.trajectory_frame_callback(
        node, frame(2, 'no_contact_coasting', pose_valid=False, pose_timestamp=1.047))
    PoseGraphNode.trajectory_frame_callback(
        node, frame(3, 'no_contact_coasting', pose_valid=False, pose_timestamp=0.))
    # Only coast frames carry a prediction time for an invalid image-time pose.
    PoseGraphNode.trajectory_frame_callback(
        node, frame(4, 'contact_tracking', pose_valid=False, pose_timestamp=1.3))
    assert [(record[0], record[1]) for record in node._trajectory_frames] == [(1., 0), (1.047, 1)]
    assert node.invalid_pose_frames == 3 and node.excluded_imu_only_frames == 0


def test_left_out_coast_frames_are_counted_whatever_their_pose_validity():
    node = backend(False)
    PoseGraphNode.trajectory_frame_callback(
        node, frame(0, 'no_contact_coasting', pose_valid=False, pose_timestamp=0.99))
    PoseGraphNode.trajectory_frame_callback(node, frame(1, 'imu_only'))
    assert [record[1] for record in node._trajectory_frames] == [1]
    assert node.excluded_imu_only_frames == 1 and node.invalid_pose_frames == 0


def test_final_metrics_report_the_left_out_frames(tmp_path):
    node = backend(False)
    for index, status in enumerate(STATUSES):
        PoseGraphNode.trajectory_frame_callback(node, frame(index, status))

    class FinalPoses:
        def size(self):
            return 1

        def __getitem__(self, _index):
            return np.eye(4)

    node.trajectory_evaluation_enabled = True
    node.trajectory_evaluation_config = {}
    node._trajectory_ground_truth = []
    node.add_keyframe_lock = Lock()
    node.keyframedb = SimpleNamespace(get_kid2kidx=lambda: {0: 0}, keyframes=[SimpleNamespace(kid=0)])
    node.pose_graph_solutions = FinalPoses()
    node.parent_groups_info = SimpleNamespace(get_largest_parent_group=lambda: 0,
                                             get_parent_group=lambda *a: 0)
    node.trajectory_output_dir = str(tmp_path)
    lines = []
    node.logger = SimpleNamespace(info=lines.append, warning=lines.append,
                                  error=lambda text: pytest.fail(text))
    PoseGraphNode.save_final_trajectory_evaluation(node)
    metrics = json.loads((tmp_path / 'trajectory_metrics.json').read_text())
    assert metrics['excluded_imu_only_frames'] == 2 and metrics['keep_imu_only_frames'] is False
    assert metrics['invalid_pose_frames'] == 0
    assert metrics['final_frames_by_status'] == {
        'contact_tracking': 1, 'imu_only': 1, 'reacquire': 1, 'tactile_hold': 1}
    rows = np.genfromtxt(tmp_path / 'trajectory_final_raw.csv', delimiter=',', skip_header=1)
    assert len(rows) == 4
    assert any('left out 2 IMU-only frames' in line for line in lines)
    assert any('by tracking status: contact_tracking 1, imu_only 1' in line for line in lines)
