"""keyframe_rule kernel_overlap: no Timed keyframes; key t when O_K(t) < keyframe_overlap_threshold."""
from types import SimpleNamespace

import numpy as np
import pytest

import gelslam.core.tracker as tm
from gelslam.core.keyframe_selection import (
    KernelMoments, kernel_moments, reference_kernel_overlap, resolve_kernel_settings)
from test_contact_recovery_policy import tracker_with_imu
from test_timed_keyframes import header, make_tracker

KERNEL = dict(kernel_scale=4.0, covariance_epsilon_px2=0.001, kernel_ellipse_vertices=32)
MM = 1.0   # mm per pixel


class Disk:
    """Contact frame: a disk of contact pixels, flat height map (uniform weights)."""

    def __init__(self, radius=6, centre=(20, 20), shape=(41, 41)):
        v, u = np.mgrid[:shape[0], :shape[1]]
        self.C = (u - centre[1]) ** 2 + (v - centre[0]) ** 2 <= radius ** 2
        self.H = np.zeros(shape)
        self.N = self.L = self.H
        self.is_contacted = True


def shifted(dx_mm):
    pose = np.eye(4)
    pose[0, 3] = dx_mm / 1000.0
    return pose


def test_reference_overlap_is_one_for_the_same_contact_and_pose():
    disk = kernel_moments(Disk(), MM)
    assert reference_kernel_overlap(disk, np.eye(4), disk, np.eye(4), KERNEL, MM) == pytest.approx(1.0)


def test_reference_overlap_falls_with_the_motion_between_k_and_t():
    disk = kernel_moments(Disk(), MM)
    values = [reference_kernel_overlap(disk, np.eye(4), disk, shifted(dx), KERNEL, MM)
              for dx in (0.0, 2.0, 4.0, 8.0, 20.0)]
    assert all(a > b for a, b in zip(values, values[1:-1]))
    assert values[-1] == 0.0


def test_reference_overlap_is_normalized_by_the_current_ellipse():
    small, large = kernel_moments(Disk(radius=3), MM), kernel_moments(Disk(radius=6), MM)
    # K small inside a large t: covers about (3/6)^2 of E_t. K large: covers all of a small t.
    ratio = reference_kernel_overlap(small, np.eye(4), large, np.eye(4), KERNEL, MM)
    assert ratio == pytest.approx(small.cov3d[0, 0] / large.cov3d[0, 0], rel=0.02)
    assert reference_kernel_overlap(large, np.eye(4), small, np.eye(4), KERNEL, MM) == pytest.approx(1.0)


def test_reference_overlap_is_zero_without_contact_pixels():
    empty = KernelMoments(0.0, None, None, (41, 41))
    disk = kernel_moments(Disk(), MM)
    assert reference_kernel_overlap(empty, np.eye(4), disk, np.eye(4), KERNEL, MM) == 0.0
    assert reference_kernel_overlap(disk, np.eye(4), empty, np.eye(4), KERNEL, MM) == 0.0


def test_kernel_settings_are_read_for_any_selection_method():
    block = dict(method="current", window_size=8, kernel_score_alpha=0.5, **KERNEL)
    assert resolve_kernel_settings(dict(keyframe_selection=block)) == KERNEL
    with pytest.raises(ValueError, match="missing"):
        resolve_kernel_settings(dict(keyframe_selection=dict(method="current", window_size=8,
                                                             kernel_score_alpha=0.5)))
    with pytest.raises(ValueError, match="kernel_ellipse_vertices"):
        resolve_kernel_settings(dict(keyframe_selection=dict(block, kernel_ellipse_vertices=4)))
    with pytest.raises(ValueError, match="mapping"):
        resolve_kernel_settings({})


@pytest.mark.parametrize("value", [None, -0.1, 1.5, True, "0.5"])
def test_overlap_threshold_must_be_a_fraction(value):
    t = object.__new__(tm.Tracker)
    t.track_config = dict(keyframe_overlap_threshold=value)
    with pytest.raises(ValueError, match="keyframe_overlap_threshold"):
        t._overlap_threshold()


def overlap_tracker(monkeypatch, threshold):
    t = make_tracker(monkeypatch)
    t._keyframe_rule = "kernel_overlap"
    t._keyframe_kernel = dict(KERNEL)
    t._keyframe_overlap_threshold = threshold
    t.ref_frame = t.prev_frame = Disk()
    infos = []
    t.logger = SimpleNamespace(info=infos.append)
    return t, infos


def test_kernel_overlap_rule_makes_no_timed_keyframe(monkeypatch):
    t, _ = overlap_tracker(monkeypatch, 0.5)
    t.ref_T_prev = shifted(1.0).astype(np.float32)
    # Far beyond keyframe_max_interval_s, and force (Low SCR) is not used.
    assert t._maybe_add_timed_keyframe(Disk(), header(99.), "Test") is None
    assert t._maybe_add_timed_keyframe(Disk(), header(99.1), "Test", force=True) is None
    assert t.latest_reference_kernel_overlap > 0.5
    statistics = t.frontend_tracking_statistics()
    assert statistics["low_overlap_keyframes"] == statistics["low_scr_keyframes"] == 0


def test_kernel_overlap_rule_keys_t_below_the_threshold(monkeypatch):
    t, infos = overlap_tracker(monkeypatch, 0.5)
    reference_pose = t.trial_T_ref.copy()
    t.ref_T_prev = shifted(8.0).astype(np.float32)
    current = Disk()
    keyframe = t._maybe_add_timed_keyframe(current, header(1.1), "Test")
    assert keyframe is not None and keyframe["surface"] is current
    assert t.curr_kid == 1 and t.ref_frame is current
    np.testing.assert_allclose(t.trial_T_ref, reference_pose @ shifted(8.0), atol=1e-7)
    assert t.latest_reference_kernel_overlap < 0.5
    assert t.frontend_tracking_statistics()["low_overlap_keyframes"] == 1
    assert "Low overlap Keyframe" in infos[-1] and "kernel overlap: " in infos[-1]
    # The next frame compares against the new reference K = current.
    t.ref_T_prev = np.eye(4, dtype=np.float32)
    assert t._maybe_add_timed_keyframe(Disk(), header(1.2), "Test") is None
    assert t.latest_reference_kernel_overlap == pytest.approx(1.0)


def test_reference_moments_are_cached_per_reference_frame(monkeypatch):
    t, _ = overlap_tracker(monkeypatch, 0.5)
    calls = []
    original = tm.kernel_moments
    monkeypatch.setattr(tm, "kernel_moments", lambda frame, mm: calls.append(frame) or original(frame, mm))
    for time in (1.1, 1.2, 1.3):
        t._maybe_add_timed_keyframe(Disk(), header(time), "Test")
    assert sum(frame is t.ref_frame for frame in calls) == 1


@pytest.mark.parametrize("overlap,keyframes", [(0.9, 0), (0.1, 1)])
def test_low_scr_makes_no_keyframe_under_kernel_overlap(monkeypatch, overlap, keyframes):
    tracker, _ = tracker_with_imu(monkeypatch)
    tracker._keyframe_rule = "kernel_overlap"
    tracker._keyframe_kernel = dict(KERNEL)
    tracker._keyframe_overlap_threshold = 0.5
    monkeypatch.setattr(tm, "normalflow_quality", lambda *a: (0.1, 0.9))   # SCR below 0.3, CCS passes
    monkeypatch.setattr(tm, "kernel_moments", lambda frame, mm: None)
    monkeypatch.setattr(tm, "reference_kernel_overlap", lambda *a: overlap)
    ok, (message, packets) = tracker.track(np.zeros((1, 1, 3), np.uint8), header(1.04))
    assert ok and len(packets) == keyframes
    statistics = tracker.frontend_tracking_statistics()
    assert statistics["low_scr_keyframes"] == 0
    assert statistics["low_overlap_keyframes"] == keyframes
