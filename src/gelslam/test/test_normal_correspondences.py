"""Checks for frontend normal correspondence coordinates and publication."""

from types import SimpleNamespace

import numpy as np
import pytest
from cv_bridge import CvBridge
from std_msgs.msg import Header

from gelslam.core.numpy_vi import DenseNormalResidual
from gelslam.core.tracker import QualityGateError, Tracker
from gelslam.nodes.tracker_node import TrackerNode
from gelslam.visualization.normal_correspondences import (
    normal_map_to_rgb8,
    project_normal_correspondences,
    render_normal_correspondences,
)
import gelslam.core.tracker as tracker_module


def surface(h=24, w=32):
    y, x = np.mgrid[:h, :w]
    normals = np.dstack((x / w, y / h, np.ones((h, w)))).astype(np.float32)
    return SimpleNamespace(
        N=normals, H=np.zeros((h, w), np.float32),
        C=np.ones((h, w), bool), L=(1 + x + y * w).astype(np.float32),
        is_contacted=True,
    )


def test_projection_has_correct_transform_direction_and_pixel_centers():
    ref, current = surface(), surface()
    # Current->reference +0.25 mm means reference->current -0.25 pixels.
    ref_T_current = np.eye(4)
    ref_T_current[0, 3] = 0.00025
    reference_xy, current_xy = project_normal_correspondences(
        ref, current, ref_T_current, 1.0, n_samples=100
    )
    assert len(reference_xy) > 10
    assert np.allclose(current_xy - reference_xy, (-0.25, 0), atol=1e-5)
    assert np.all(reference_xy == np.rint(reference_xy))


def test_projection_filters_target_contact_and_bounds_but_not_reference_contact():
    ref, current = surface(), surface()
    ref.C[:] = False  # Normal-flow row selection is curvature-based.
    current.C[:, 10:] = False
    pose = np.eye(4)
    pose[0, 3] = 0.002
    reference_xy, current_xy = project_normal_correspondences(ref, current, pose, 1.0)
    assert len(reference_xy) > 10
    assert np.all((current_xy[:, 0] >= 0) & (current_xy[:, 0] < 10))
    assert np.allclose(current_xy[:, 0], reference_xy[:, 0] - 2)
    current.C[:] = False
    assert len(project_normal_correspondences(ref, current, pose, 1.0)[0]) == 0


def test_dense_pairs_use_solver_candidates_and_active_target_rows():
    ref, current = surface(), surface()
    current.C[:, 20:] = False
    pose = np.eye(4)
    pose[0, 3] = 0.001
    visual = DenseNormalResidual(ref, current, 1.0, np.eye(4), pose, max_samples=100)
    visual.evaluate(np.eye(4), pose)
    active_rows = visual.last_active_rows.copy()
    reference_xy, current_xy = visual.correspondences(np.eye(4), pose)
    assert len(reference_xy) == np.count_nonzero(active_rows)
    assert np.allclose(current_xy - reference_xy, (-1, 0), atol=1e-5)
    assert np.all(current_xy[:, 0] < 20)
    assert np.array_equal(visual.last_active_rows, active_rows)
    current.C[:] = False
    assert len(visual.correspondences(np.eye(4), pose)[0]) == 0


def test_rgb_panels_and_solid_green_line_endpoints():
    ref, current = surface(), surface()
    current.N[:] = (-1, 0, 1)
    canvas = render_normal_correspondences(
        ref, current, [[5, 4]], [[10, 12]], "Reference", "Current", "Accepted"
    )
    assert canvas.shape == (72, 80, 3)
    assert np.array_equal(canvas[48 + 4, 5], (0, 255, 0))
    assert np.array_equal(canvas[48 + 12, 32 + 16 + 10], (0, 255, 0))
    assert np.array_equal(canvas[48, 0], normal_map_to_rgb8(ref.N)[0, 0])
    assert np.array_equal(canvas[48, 32 + 16], normal_map_to_rgb8(current.N)[0, 0])
    # Every column between endpoints contains the continuous green line.
    for x in range(5, 59):
        assert np.any(np.all(canvas[48:, x] == (0, 255, 0), axis=1))


def test_display_limit_and_invalid_pairs_do_not_modify_inputs():
    ref = surface()
    xy = np.array([[4, 4], [6, 6], [8, 8], [10, 10]], dtype=float)
    before = xy.copy()
    limited = render_normal_correspondences(ref, ref, xy, xy, "", "", "", max_lines=1)
    assert np.array_equal(xy, before)
    assert np.array_equal(limited[48 + 4, 4], (0, 255, 0))
    assert np.array_equal(limited[48 + 10, 10], normal_map_to_rgb8(ref.N)[10, 10])
    render_normal_correspondences(ref, ref, [[np.nan, 0]], [[0, 0]], "", "", "")


def test_rejected_pose_snapshot_retains_compared_reference_after_promotion(monkeypatch):
    tracker = object.__new__(Tracker)
    reference, current = surface(), surface()
    tracker.ppmm = 1.0
    tracker.track_config = {"scr_threshold": 0.3, "ccs_threshold": 0.5}
    tracker.frontend_visual_inertial = None
    tracker.trial_T_ref = np.eye(4)
    tracker.imu_buffer = None
    tracker.ref_frame = reference
    tracker.ref_header = Header()
    tracker.ref_header.stamp.sec = 1
    tracker.logger = SimpleNamespace(info=lambda _message: None)
    current_header = Header()
    current_header.stamp.sec = 2
    monkeypatch.setattr(tracker_module, "normalflow", lambda *a, **k: (
        np.eye(4, dtype=np.float32), {"scr": 0.2, "ccs": 0.9}
    ))
    with pytest.raises(QualityGateError):
        tracker._refine_normalflow_with_imu(np.eye(4), current_header, reference, current)
    tracker.ref_frame = surface()  # A subsequent keyframe promotion changes this.
    snapshot = tracker.latest_normal_correspondences
    assert snapshot["reference"] is reference
    assert snapshot["current"] is current
    assert snapshot["reference_stamp"] == 1.0
    assert snapshot["current_stamp"] == 2.0
    assert snapshot["status"] == "NormalFlow final pose: quality rejected"
    assert len(snapshot["reference_xy"]) > 10


def test_ros_image_uses_current_header_and_saved_reference():
    reference, current = surface(), surface()
    messages = []
    source_header = Header()
    source_header.stamp.sec = 3
    node = SimpleNamespace(
        bridge=CvBridge(), correspondence_max_lines=0,
        normal_correspondences_pub=SimpleNamespace(publish=messages.append),
        tracker=SimpleNamespace(latest_normal_correspondences={
            "reference": reference, "current": current,
            "reference_stamp": 1.0, "current_stamp": 3.0,
            "reference_xy": np.array([[5, 4]]), "current_xy": np.array([[10, 12]]),
            "status": "VI final pose: accepted",
        }),
    )
    TrackerNode._publish_normal_correspondences(node, current, source_header)
    assert len(messages) == 1
    assert messages[0].encoding == "rgb8"
    assert messages[0].header == source_header
    canvas = node.bridge.imgmsg_to_cv2(messages[0], desired_encoding="rgb8")
    assert np.array_equal(canvas[48 + 4, 5], (0, 255, 0))
    assert np.array_equal(canvas[48 + 12, 48 + 10], (0, 255, 0))


def test_no_contact_clears_previous_correspondence_snapshot():
    tracker = object.__new__(Tracker)
    tracker.bg_images = [None] * 10
    tracker.ppmm = 1.0
    tracker.contact_mask_config = {"color_dist_threshold": 15, "height_threshold": 0.2}
    tracker.track_config = {"contact_threshold": 500}
    tracker.curr_fid = 10
    tracker.recon = SimpleNamespace(get_surface_info=lambda *a, **k: (
        np.zeros((24, 32, 2), np.float32), np.zeros((24, 32), np.float32),
        np.zeros((24, 32), bool),
    ))
    tracker.latest_normal_correspondences = {"status": "Previous frame"}
    tracker.logger = SimpleNamespace(info=lambda _message: None)
    tracker._initialize_no_contact_coast = lambda: False
    assert tracker.track(np.zeros((24, 32, 3), np.uint8)) == (False, None)
    assert tracker.latest_normal_correspondences is None
    assert not tracker.latest_observation_frame.is_contacted
