"""Disabling debug work must preserve the frontend and core publication path."""
from types import SimpleNamespace

import numpy as np
import pytest
from std_msgs.msg import Header
from gelslam.nodes.tracker_node import TrackerNode
import gelslam.nodes.tracker_node as tracker_node_module


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("accepted", [False, True])
def test_debug_gate_skips_construction_without_skipping_tracking(monkeypatch, enabled, accepted):
    calls = []
    header = Header()
    header.stamp.sec = 2
    frame, keyframe = object(), object()
    observation = SimpleNamespace(N=np.ones((2, 2, 3), np.float32))

    def record(name, result=None):
        def function(*args, **kwargs):
            calls.append(name)
            return result
        return function

    def rgb(normals):
        assert enabled, "disabled debug must not convert normal RGB"
        calls.append("rgb")
        return np.zeros((2, 2, 3), np.uint8)

    monkeypatch.setattr(tracker_node_module, "normal_map_to_rgb8", rgb)
    normal_message = SimpleNamespace(header=None)
    node = SimpleNamespace(
        recorded_source_clock=None,
        debug_visualization_enabled=enabled,
        bridge=SimpleNamespace(
            imgmsg_to_cv2=record("input_conversion", np.zeros((2, 2, 3), np.uint8)),
            cv2_to_imgmsg=record("debug_image_message", normal_message),
        ),
        tracker=SimpleNamespace(
            track=record("tracking", (accepted, (frame, [keyframe]) if accepted else None)),
            latest_observation_frame=observation,
        ),
        _publish_frontend_gravity=record("gravity"),
        _publish_tracking_result=lambda f, k: calls.append(("core_output", f, k)),
        _log_tracking_pose=record("pose_log"),
        state_velocity_evaluation=None,
        _contact_surface_cloud=record("contact_cloud_build", object()),
        _publish_normal_correspondences=record("correspondence_build_publish"),
        normal_map_pub=SimpleNamespace(publish=record("normal_publish")),
        contact_surface_pub=SimpleNamespace(publish=record("cloud_publish")),
    )
    TrackerNode.image_callback(node, SimpleNamespace(header=header))
    assert calls[:3] == ["input_conversion", "tracking", "gravity"]
    outputs = [call for call in calls if isinstance(call, tuple)]
    assert outputs == ([("core_output", frame, [keyframe])] if accepted else [])
    debug_calls = {"rgb", "debug_image_message", "normal_publish", "contact_cloud_build",
                   "cloud_publish", "correspondence_build_publish"}
    assert set(call for call in calls if isinstance(call, str)) & debug_calls == (
        debug_calls if enabled else set())
    if enabled:
        assert normal_message.header is header
