"""Timestamp lookup for paired camera and IMU topics in a recorded ROS bag.

The bag database gives the two headerless timing topics an unambiguous partner:
each timing message has the same bag timestamp as its sensor message.  This
lookup is for replay of that same bag, not for live independent ROS topics.
"""

from collections import defaultdict, deque
import json
import sqlite3

import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import String, UInt32


def _stamp_ns(header):
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def _imu_key(message):
    acceleration = message.linear_acceleration
    angular_velocity = message.angular_velocity
    return (
        _stamp_ns(message.header),
        float(acceleration.x), float(acceleration.y), float(acceleration.z),
        float(angular_velocity.x), float(angular_velocity.y), float(angular_velocity.z),
    )


def _paired_rows(connection, topics, sensor_topic, timing_topic):
    if sensor_topic not in topics or timing_topic not in topics:
        raise ValueError("recorded source clock requires %s and %s" %
                         (sensor_topic, timing_topic))
    query = "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp, id"
    sensor_count = connection.execute(
        "SELECT COUNT(*) FROM messages WHERE topic_id=?", (topics[sensor_topic],)
    ).fetchone()[0]
    timing_count = connection.execute(
        "SELECT COUNT(*) FROM messages WHERE topic_id=?", (topics[timing_topic],)
    ).fetchone()[0]
    if not sensor_count or sensor_count != timing_count:
        raise ValueError("recorded sensor and timing topic counts differ")
    sensor_rows = connection.execute(query, (topics[sensor_topic],))
    timing_rows = connection.execute(query, (topics[timing_topic],))
    for sensor, timing in zip(sensor_rows, timing_rows):
        if sensor[0] != timing[0]:
            raise ValueError("recorded sensor and timing bag timestamps differ")
        yield sensor[1], timing[1]


def _unwrap_device_ms(raw_values):
    """Keep uint32 millisecond wraparound from reversing the IMU time axis."""
    values = []
    wraps = 0
    previous = None
    for raw in raw_values:
        raw = int(raw)
        if previous is not None and raw < previous and previous - raw > 2**31:
            wraps += 1
        value = raw + wraps * 2**32
        if values and value <= values[-1]:
            raise ValueError("recorded IMU device times are not increasing")
        values.append(value)
        previous = raw
    return np.asarray(values, dtype=np.float64)


def _fit_to_host(source_seconds, host_seconds):
    """Fit clock rate and receipt-time origin while limiting receipt outliers."""
    source = np.asarray(source_seconds, dtype=np.float64)
    host = np.asarray(host_seconds, dtype=np.float64)
    if len(source) < 2 or len(source) != len(host):
        raise ValueError("at least two paired clock samples are required")
    if np.any(np.diff(source) <= 0):
        raise ValueError("recorded source clock must increase")
    anchor = float(source[0])
    x = source - anchor
    y = host - host[0]
    keep = np.ones(len(x), dtype=bool)
    for _ in range(4):
        slope, relative_origin = np.linalg.lstsq(
            np.column_stack((x[keep], np.ones(int(keep.sum())))), y[keep], rcond=None
        )[0]
        residual = y - (slope * x + relative_origin)
        cutoff = max(float(np.quantile(np.abs(residual), 0.9)), 1e-6)
        keep = np.abs(residual) <= cutoff
    if not 0.95 <= slope <= 1.05:
        raise ValueError("recorded source and host clock rates disagree")
    residual = y - (slope * x + relative_origin)
    return (anchor, float(host[0] + relative_origin), float(slope),
            float(np.median(np.abs(residual))))


class RecordedSourceClock:
    """Map replayed sensor messages to source-clock times on one host axis."""

    def __init__(self, bag_database, image_topic="/gelsight/image_raw",
                 camera_timing_topic="/gelsight/timing", imu_topic="/imu/data_raw",
                 imu_device_time_topic="/imu/device_time_ms"):
        self.image_times = defaultdict(deque)
        self.imu_times = defaultdict(deque)
        connection = sqlite3.connect(str(bag_database))
        try:
            topics = dict(connection.execute("SELECT name, id FROM topics"))
            camera_records = []
            for image_blob, timing_blob in _paired_rows(
                connection, topics, image_topic, camera_timing_topic
            ):
                image = deserialize_message(bytes(image_blob), Image)
                timing = json.loads(deserialize_message(bytes(timing_blob), String).data)
                if _stamp_ns(image.header) != int(timing["host_wall_ns"]):
                    raise ValueError("camera timing host_wall_ns does not match image header")
                camera_records.append((_stamp_ns(image.header), int(timing["source_pts_ns"]),
                                       int(timing["host_wall_ns"])))

            imu_records = []
            for imu_blob, device_blob in _paired_rows(
                connection, topics, imu_topic, imu_device_time_topic
            ):
                imu = deserialize_message(bytes(imu_blob), Imu)
                device = deserialize_message(bytes(device_blob), UInt32)
                imu_records.append((_imu_key(imu), int(device.data)))
        finally:
            connection.close()

        camera_source = np.asarray([item[1] for item in camera_records], dtype=np.float64) * 1e-9
        camera_host = np.asarray([item[2] for item in camera_records], dtype=np.float64) * 1e-9
        imu_source = _unwrap_device_ms([item[1] for item in imu_records]) * 1e-3
        imu_host = np.asarray([item[0][0] for item in imu_records], dtype=np.float64) * 1e-9
        self.camera_fit = _fit_to_host(camera_source, camera_host)
        self.imu_fit = _fit_to_host(imu_source, imu_host)
        for record, source in zip(camera_records, camera_source):
            anchor, origin, slope, _ = self.camera_fit
            self.image_times[record[0]].append(origin + slope * (source - anchor))
        for record, source in zip(imu_records, imu_source):
            anchor, origin, slope, _ = self.imu_fit
            self.imu_times[record[0]].append(origin + slope * (source - anchor))
        self.image_count = len(camera_records)
        self.imu_count = len(imu_records)

    def image_timestamp(self, header):
        values = self.image_times.get(_stamp_ns(header))
        return values.popleft() if values else None

    def imu_timestamp(self, message):
        values = self.imu_times.get(_imu_key(message))
        return values.popleft() if values else None
