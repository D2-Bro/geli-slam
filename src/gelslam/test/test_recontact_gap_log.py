"""Recontact log: gyro rotation over the no-contact gap."""
from types import SimpleNamespace

import numpy as np
import pytest
from std_msgs.msg import Header

from gelslam.core.inertial import ImuBuffer, ImuSample
from gelslam.core.tracker import Tracker


def header(t):
    h = Header()
    h.stamp.sec, h.stamp.nanosec = divmod(round(t * 1e9), 10 ** 9)
    return h


def tracker_with_gyro(rate, bias=(0.0, 0.0, 0.0)):
    tracker = Tracker.__new__(Tracker)
    tracker.image_to_imu_time_offset_s = 0.0
    tracker.imu_buffer = ImuBuffer(1000)
    for t in np.arange(1.0, 2.0001, 0.005):
        tracker.imu_buffer.add(ImuSample(float(t), (0.0, 0.0, 9.81), np.asarray(rate) + np.asarray(bias)))
    tracker._frontend_vi_state = SimpleNamespace(gyro_bias=np.asarray(bias, dtype=float))
    return tracker


def test_gap_rotation_integrates_bias_corrected_gyro():
    tracker = tracker_with_gyro((0.0, 0.0, 0.5), bias=(0.01, -0.02, 0.0))
    angle = tracker._gap_rotation_deg(header(1.2), header(1.6))
    assert angle == pytest.approx(np.degrees(0.5 * 0.4), abs=1e-6)


def test_gap_rotation_is_none_without_imu_interval():
    tracker = tracker_with_gyro((0.0, 0.0, 0.5))
    assert tracker._gap_rotation_deg(None, header(1.6)) is None
    assert tracker._gap_rotation_deg(header(1.6), header(1.2)) is None
    tracker.imu_buffer = None
    assert tracker._gap_rotation_deg(header(1.2), header(1.6)) is None
