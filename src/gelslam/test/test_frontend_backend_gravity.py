"""Check the frontend gravity publication path."""

from types import SimpleNamespace

import numpy as np
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

from gelslam.core.numpy_vi import NavigationState
from gelslam.core.tracker import Tracker
from gelslam.nodes.tracker_node import TrackerNode


def test_frontend_publishes_initialized_gravity_once():
    tracker = object.__new__(Tracker)
    tracker.frontend_visual_inertial = SimpleNamespace(
        gravity=np.array((1.0, -2.0, -np.sqrt(9.81**2 - 5.0)))
    )
    tracker._frontend_vi_state = None
    published = []
    frontend = SimpleNamespace(
        frontend_gravity_pub=SimpleNamespace(publish=published.append),
        _frontend_gravity_published=False,
        tracker=tracker,
        get_logger=lambda: SimpleNamespace(info=lambda _message: None),
    )
    header = Header()
    header.stamp.sec = 10

    TrackerNode._publish_frontend_gravity(frontend, header)
    assert published == []

    tracker._frontend_vi_state = NavigationState(
        10.0, np.eye(4), np.zeros(3), np.zeros(3), np.zeros(3)
    )
    TrackerNode._publish_frontend_gravity(frontend, header)
    TrackerNode._publish_frontend_gravity(frontend, header)
    assert len(published) == 1
    assert published[0].header.frame_id == "gelslam_first_trial"

    assert np.allclose(
        (published[0].vector.x, published[0].vector.y, published[0].vector.z),
        tracker.frontend_visual_inertial.gravity,
    )


def test_gt_pose_callback_keeps_gelsight_attitude_for_gravity_initialization():
    received = []
    frontend = SimpleNamespace(
        tracker=SimpleNamespace(add_recent_ground_truth_gel_rotation=lambda *args: received.append(args)),
        get_logger=lambda: SimpleNamespace(warning=lambda _message: None),
    )
    message = PoseStamped()
    message.header.stamp.sec = 2
    message.pose.orientation.w = 1.0
    message.pose.position.x = 0.01
    message.pose.position.y = -0.02
    message.pose.position.z = 0.03

    TrackerNode.ground_truth_gel_pose_callback(frontend, message)

    assert len(received) == 1
    timestamp, rotation = received[0]
    assert timestamp == 2.0
    assert np.allclose(rotation, np.eye(3))


def test_gt_imu_pose_callback_forwards_imu_position_for_velocity_initialization():
    received = []
    frontend = SimpleNamespace(
        tracker=SimpleNamespace(
            add_ground_truth_imu_position=lambda *args: received.append(args) or True,
        ),
        get_logger=lambda: SimpleNamespace(warning=lambda _message: None),
    )
    message = PoseStamped()
    message.header.stamp.sec = 2
    message.header.stamp.nanosec = 5_000_000
    message.pose.position.x = .04
    message.pose.position.y = -.03
    message.pose.position.z = .02

    TrackerNode.ground_truth_imu_pose_callback(frontend, message)

    assert received == [(2.005, (.04, -.03, .02))]


def test_initialization_gt_subscription_waits_for_completed_anchor():
    removed = []
    tracker = SimpleNamespace(
        imu_config={}, _frontend_vi_state=None, _trial_index=0,
        _pending_frontend_initialization=object(), _pending_trial_contact=object(),
        _ground_truth_imu_positions=[(1., np.zeros(3))],
    )
    subscription = object()
    node = SimpleNamespace(
        tracker=tracker, ground_truth_imu_pose_sub=subscription,
        destroy_subscription=lambda sub: removed.append(sub) or True,
        get_logger=lambda: SimpleNamespace(info=lambda _: None),
    )
    node._initialization_imu_gt_release_ready = lambda: (
        TrackerNode._initialization_imu_gt_release_ready(node))
    TrackerNode._release_initialization_imu_gt(node)
    assert removed == []
    tracker._frontend_vi_state = object()
    TrackerNode._release_initialization_imu_gt(node)
    assert removed == []
    tracker._pending_frontend_initialization = None
    TrackerNode._release_initialization_imu_gt(node)
    assert removed == []
    tracker._pending_trial_contact = None
    TrackerNode._release_initialization_imu_gt(node)
    assert removed == [subscription]
    assert node.ground_truth_imu_pose_sub is None
    assert tracker._ground_truth_imu_positions == []
    TrackerNode._release_initialization_imu_gt(node)
    assert removed == [subscription]


def test_gt_state_diagnostic_keeps_its_continuous_imu_gt_subscription():
    tracker = SimpleNamespace(
        imu_config={'frontend_gt_previous_state_test': True},
        _frontend_vi_state=object(), _trial_index=0,
        _pending_frontend_initialization=None, _pending_trial_contact=None,
    )
    removed = []
    subscription = object()
    node = SimpleNamespace(tracker=tracker, ground_truth_imu_pose_sub=subscription,
                           destroy_subscription=removed.append)
    node._initialization_imu_gt_release_ready = lambda: (
        TrackerNode._initialization_imu_gt_release_ready(node))
    TrackerNode._release_initialization_imu_gt(node)
    assert removed == []
    assert node.ground_truth_imu_pose_sub is subscription
