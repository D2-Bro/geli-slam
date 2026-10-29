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


def _bag(tmp_path, mismatch=False, device_ms=(1000, 1010, 1020),
         duplicate_imu_header=False):
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
        imu_ns = 99_950_000_000 + (
            0 if duplicate_imu_header and index == 1 else index
        ) * 10_000_000
        imu = _set_stamp(Imu(), imu_ns)
        imu.linear_acceleration.x = float(index)
        device = UInt32(data=device_ms[index])
        images.append(image)
        imus.append(imu)
        for topic_id, timestamp, message in (
            (1, image_ns, image),
            (2, image_ns + (1 if mismatch and index == 1 else 0), timing),
            (3, imu_ns, imu),
            (4, imu_ns, device),
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


@pytest.mark.parametrize('raw, expected', [
    ([59997, 2, 7], [59997, 60002, 60007]),
    ([59999, 0, 1], [59999, 60000, 60001]),
    ([59990, 59995, 0, 5, 59995, 0, 5],
     [59990, 59995, 60000, 60005, 119995, 120000, 120005]),
    ([59990, 60000, 60010], [59990, 60000, 60010]),
])
def test_device_minute_wrap_preserves_elapsed_milliseconds(raw, expected):
    assert _unwrap_device_ms(raw).tolist() == expected


@pytest.mark.parametrize('raw', [
    [59997, 59997],
    [59997, 59996],
    [20000, 0],
    [100000, 59997, 2],
    [59997, 2, 70000],
    [2**32 - 5, 0, 59997, 2],
])
def test_device_wrap_does_not_hide_duplicates_or_clock_resets(raw):
    with pytest.raises(ValueError, match='not increasing'):
        _unwrap_device_ms(raw)


@pytest.mark.parametrize('duplicate_imu_header', [False, True])
def test_recorded_minute_wrap_keeps_all_imu_samples(tmp_path, duplicate_imu_header):
    path, _, imus = _bag(tmp_path, device_ms=(59990, 0, 10),
                        duplicate_imu_header=duplicate_imu_header)
    source = RecordedSourceClock(path)
    times = [source.imu_timestamp(imu) for imu in imus]
    assert all(time is not None for time in times)
    assert np.allclose(np.diff(times), 0.01)
    assert source.imu_count == 3
