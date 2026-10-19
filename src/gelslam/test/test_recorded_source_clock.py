"""Check deterministic source-clock pairing for a recorded ROS bag."""

import json
import sqlite3

import numpy as np
import pytest
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import String, UInt32

from gelslam.core.recorded_source_clock import RecordedSourceClock, _unwrap_device_ms
from gelslam.core.tracker import Tracker


def _set_stamp(message, nanoseconds):
    message.header.stamp.sec = nanoseconds // 1_000_000_000
    message.header.stamp.nanosec = nanoseconds % 1_000_000_000
    return message


def _bag(tmp_path, mismatch=False):
    path = tmp_path / "paired.db3"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT)")
    connection.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER, data BLOB)")
    names = ("/gelsight/image_raw", "/gelsight/timing",
             "/imu/data_raw", "/imu/device_time_ms")
    for identifier, name in enumerate(names, 1):
        connection.execute("INSERT INTO topics VALUES (?, ?)", (identifier, name))
    images, imus = [], []
    identifier = 0
    for index in range(3):
        image_ns = 100_000_000_000 + index * 100_000_000
        image = _set_stamp(Image(), image_ns)
        timing = String(data=json.dumps({
            "host_wall_ns": image_ns,
            "source_pts_ns": 10_000_000_000 + index * 100_000_000,
        }))
        imu = _set_stamp(Imu(), 99_950_000_000 + index * 10_000_000)
        imu.linear_acceleration.x = float(index)
        device = UInt32(data=1000 + index * 10)
        images.append(image)
        imus.append(imu)
        for topic_id, timestamp, message in (
            (1, image_ns, image),
            (2, image_ns + (1 if mismatch and index == 1 else 0), timing),
            (3, 99_950_000_000 + index * 10_000_000, imu),
            (4, 99_950_000_000 + index * 10_000_000, device),
        ):
            identifier += 1
            connection.execute("INSERT INTO messages VALUES (?, ?, ?, ?)",
                               (identifier, topic_id, timestamp, serialize_message(message)))
    connection.commit()
    connection.close()
    return path, images, imus


def test_recorded_source_clock_uses_paired_topics_and_preserves_intervals(tmp_path):
    path, images, imus = _bag(tmp_path)
    source = RecordedSourceClock(path)
    image_times = [source.image_timestamp(image.header) for image in images]
    imu_times = [source.imu_timestamp(imu) for imu in imus]

    assert np.allclose(np.diff(image_times), 0.1)
    assert np.allclose(np.diff(imu_times), 0.01)
    assert source.image_timestamp(images[0].header) is None
    assert source.imu_timestamp(imus[0]) is None

    tracker = object.__new__(Tracker)
    tracker.timestamp_source = "recorded_source"
    tracker.image_to_imu_time_offset_s = 0.025
    tracker._recorded_image_times = {}
    tracker.set_recorded_image_timestamp(images[0].header, image_times[0])
    assert tracker._imu_timestamp(images[0].header) == pytest.approx(image_times[0] + 0.025)
    assert tracker._imu_timestamp(images[1].header) is None


def test_recorded_source_clock_rejects_unpaired_timing(tmp_path):
    path, _, _ = _bag(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="bag timestamps differ"):
        RecordedSourceClock(path)


def test_device_millisecond_wrap_keeps_time_monotonic():
    assert _unwrap_device_ms([2**32 - 5, 0, 5]).tolist() == [2**32 - 5, 2**32, 2**32 + 5]
