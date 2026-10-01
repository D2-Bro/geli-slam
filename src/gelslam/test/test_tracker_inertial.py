"""Tracker-level tests for replacing a failed tactile update with IMU coasting."""

from types import SimpleNamespace

import numpy as np

from gelslam.core.inertial import ImuBuffer, ImuSample
from gelslam.core.tracker import Tracker, TrackingMode
import gelslam.core.tracker as tracker_module


def header(timestamp):
    sec = int(timestamp)
    return SimpleNamespace(
        stamp=SimpleNamespace(sec=sec, nanosec=int((timestamp - sec) * 1e9))
    )


def make_coasting_tracker():
    """Build only the inertial tracker state; no hardware reconstructor is needed."""
    tracker = object.__new__(Tracker)
    tracker.gelsight_T_imu = np.eye(4)
    tracker.imu_buffer = ImuBuffer()
    tracker.imu_buffer.add(
        ImuSample(1.0, (0.0, 0.0, 9.81), (0.0, 0.0, np.pi / 2.0))
    )
    tracker.imu_buffer.add(
        ImuSample(2.0, (0.0, 0.0, 9.81), (0.0, 0.0, 0.0))
    )
    tracker.imu_config = {"gravity_mps2": 9.81}
    tracker.inertial_state = None
    tracker.inertial_propagator = None
    tracker.ref_T_prev = np.eye(4, dtype=np.float32)
    return tracker


def test_successful_tactile_anchor_can_coast_to_a_later_image_timestamp():
    tracker = make_coasting_tracker()

    assert tracker._anchor_inertial_state(np.eye(4), header(1.0), reset_reference=True)
    assert tracker._coast_to_header(header(2.0), TrackingMode.IMU_COASTING)

    assert tracker.tracking_mode is TrackingMode.IMU_COASTING
    assert np.allclose(
        tracker.ref_T_prev[:3, :3],
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        atol=1e-6,
    )


def test_failed_tactile_update_keeps_the_existing_trial_when_imu_can_coast(
    monkeypatch,
):
    tracker = make_coasting_tracker()
    assert tracker._anchor_inertial_state(np.eye(4), header(1.0), reset_reference=True)

    class FakeFrame:
        def __init__(self, *_, **__):
            self.is_contacted = True
            self.N = self.C = self.H = self.L = np.zeros((1, 1))

    tracker.bg_images = [object()] * 10
    tracker.recon = SimpleNamespace(
        get_surface_info=lambda *_args, **_kwargs: (None, None, None)
    )
    tracker.contact_mask_config = {"color_dist_threshold": 1, "height_threshold": 1}
    tracker.track_config = {
        "contact_threshold": 1,
        "scr_threshold": 1,
        "ccs_threshold": 1,
    }
    tracker.ppmm = 1.0
    tracker.prev_in_contact_flag = True
    tracker.prev_is_keyframe_flag = True
    tracker.prev_frame = FakeFrame()
    tracker.ref_frame = tracker.prev_frame
    tracker.prev_header = header(1.0)
    tracker.ref_header = header(1.0)
    tracker.curr_fid = 0
    tracker.curr_kid = 4
    tracker.bridge = object()
    tracker.logger = SimpleNamespace(info=lambda _message: None)

    monkeypatch.setattr(tracker_module, "Frame", FakeFrame)
    monkeypatch.setattr(
        tracker_module,
        "normalflow",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            tracker_module.LoseTrackError()
        ),
    )
    monkeypatch.setattr(
        tracker_module,
        "create_frame_msg",
        lambda _bridge, _frame, **kwargs: kwargs,
    )

    ret, (frame_msg, keyframes) = tracker.track(
        np.zeros((1, 1, 3), dtype=np.uint8), header(2.0)
    )

    assert ret
    assert keyframes == []
    assert frame_msg["ref_kid"] == 4
    assert tracker.curr_kid == 4
    assert tracker.tracking_mode is TrackingMode.IMU_COASTING
