"""Source-topic pairing, frozen time mapping, and tracker callback ownership."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import String, UInt32

from gelslam.core.streaming_source_clock import StreamingSourceClock
from gelslam.nodes.tracker_node import TrackerNode
from test_tracker_image_imu_sync import Guard, SyncHarness, message_at


def camera_timing(image, source):
    stamp = image.header.stamp
    return String(data=json.dumps(dict(
        host_wall_ns=stamp.sec * 10**9 + stamp.nanosec,
        source_pts_ns=round(source * 10**9))))


def packets():
    images = [message_at(Image, 100. + i*.01) for i in range(3)]
    imus = [message_at(Imu, 100. + i*.01) for i in range(3)]
    camera = [camera_timing(image, 10. + i*.01) for i, image in enumerate(images)]
    device = [UInt32(data=t) for t in (59990, 0, 10)]
    return images, imus, camera, device


@pytest.mark.parametrize('timing_first', [False, True])
@pytest.mark.parametrize('batch_topics', [False, True])
def test_stream_pairs_topics_on_either_side_of_sensor_callbacks(timing_first, batch_topics):
    clock = StreamingSourceClock(warmup_s=.02)
    images, imus, camera, device = packets()
    groups = [(clock.add_image, images), (clock.add_imu, imus),
              (clock.add_camera_timing, camera), (clock.add_imu_timing, device)]
    if timing_first:
        groups = groups[2:] + groups[:2]
    if batch_topics:
        for callback, messages in groups:
            for message in messages:
                callback(message)
    else:
        for i in range(3):
            for callback, messages in groups:
                callback(messages[i])
    assert clock.ready
    mapped_images, mapped_imus = clock.take_images(), clock.take_imus()
    assert [m for m, _ in mapped_images] == images
    assert [m for m, _ in mapped_imus] == imus
    np.testing.assert_allclose([t for _, t in mapped_images], [100., 100.01, 100.02])
    np.testing.assert_allclose([t for _, t in mapped_imus], [100., 100.01, 100.02])
    assert not clock.take_images() and not clock.take_imus()


def test_duplicate_imu_headers_keep_distinct_source_times():
    clock = StreamingSourceClock(warmup_s=.02)
    images, imus, camera, device = packets()
    for i in range(3):
        imus[i].header = imus[0].header
        clock.add_imu(imus[i])
        clock.add_imu_timing(device[i])
        clock.add_image(images[i])
        clock.add_camera_timing(camera[i])
    times = [t for _, t in clock.take_imus()]
    np.testing.assert_allclose(np.diff(times), [.01, .01])
    assert len(times) == 3


def test_warmup_does_not_publish_a_header_fallback():
    clock = StreamingSourceClock(warmup_s=.02)
    images, imus, _, device = packets()
    for imu, timing in zip(imus, device):
        clock.add_imu(imu)
        clock.add_imu_timing(timing)
    for image in images:
        clock.add_image(image)
    assert not clock.ready
    assert clock.take_images() == clock.take_imus() == []


def test_camera_drops_do_not_shift_timing_to_the_next_image():
    clock = StreamingSourceClock(warmup_s=.02)
    images, imus, camera, device = packets()
    for imu, timing in zip(imus, device):
        clock.add_imu(imu)
        clock.add_imu_timing(timing)
    for i in range(6):
        image = message_at(Image, 100. + i*.01)
        clock.add_camera_timing(camera_timing(image, 10. + i*.01))
        if i in (1, 3, 5):
            clock.add_image(image)
    mapped = clock.take_images()
    np.testing.assert_allclose([t for _, t in mapped], [100.01, 100.03, 100.05])


def test_mapping_is_frozen_after_warmup_despite_receipt_jitter():
    clock = StreamingSourceClock(warmup_s=.02)
    images, imus, camera, device = packets()
    for i in range(3):
        clock.add_image(images[i])
        clock.add_camera_timing(camera[i])
        clock.add_imu(imus[i])
        clock.add_imu_timing(device[i])
    old_imu_times = [t for _, t in clock.take_imus()]
    clock.take_images()
    # Later host receipt is 0.5 s late. The integration clock keeps its 10 ms step.
    late = message_at(Imu, 100.53)
    clock.add_imu(late)
    clock.add_imu_timing(UInt32(data=20))
    new_time = clock.take_imus()[0][1]
    assert new_time-old_imu_times[-1] == pytest.approx(.01)


@pytest.mark.parametrize('value', [10, 9])
def test_invalid_device_clock_stops_source_delivery(value):
    clock = StreamingSourceClock(warmup_s=.02)
    clock.add_imu(message_at(Imu, 100.))
    clock.add_imu_timing(UInt32(data=10))
    clock.add_imu(message_at(Imu, 100.01))
    with pytest.raises(ValueError, match='not increasing'):
        clock.add_imu_timing(UInt32(data=value))
    assert clock.error and not clock.ready
    assert not clock.take_imus()
    with pytest.raises(ValueError):
        clock.add_imu(message_at(Imu, 100.02))


def test_unpaired_source_queues_are_bounded():
    clock = StreamingSourceClock(max_pending=3)
    for i in range(3):
        clock.add_imu(message_at(Imu, 100.+i*.01))
    with pytest.raises(ValueError, match='queue overflow'):
        clock.add_imu(message_at(Imu, 100.03))
    assert not clock.ready


def test_source_callbacks_preserve_imu_and_image_owners():
    node = SyncHarness()
    node.imu_callback_group = object()
    node.tracker.timestamp_source = 'recorded_source'
    node.streaming_source_clock = StreamingSourceClock(warmup_s=.02)
    node._source_clock_imu_guard = Guard(
        lambda: TrackerNode._drain_source_imus(node), node.imu_callback_group)
    images, imus, camera, device = packets()
    for i in range(3):
        node.image_callback(images[i])
        TrackerNode.source_camera_timing_callback(node, camera[i])
        node.imu_callback(imus[i])
        TrackerNode.source_imu_timing_callback(node, device[i])
    assert node.tracked == []  # No image tracking in IMU reception.
    node._source_clock_imu_guard.dispatch()
    assert node.tracked == []
    node._imu_image_sync_guard.dispatch()
    assert node.tracked == images
    assert len(node.tracker.imu_buffer.samples_between(99., 101.)) == 3
    assert node._imu_dropped_samples == 0
    assert node.tracker._imu_timestamp(images[-1].header) == pytest.approx(100.02)


@pytest.mark.parametrize('custom_topics', [False, True])
def test_node_subscribes_to_source_topics_without_a_bag(tmp_path, monkeypatch, custom_topics):
    import rclpy
    from rclpy.node import Node
    import yaml
    import gelslam.nodes.tracker_node as module
    from gelslam.core.inertial import ImuBuffer

    monkeypatch.setenv('ROS_LOG_DIR', str(tmp_path/'ros_log'))
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    monkeypatch.setenv('ROS_DOMAIN_ID', '219')
    imu_config = dict(enabled=True, timestamp_source='recorded_source', topic='/test/imu')
    camera_topic, device_topic = '/gelsight/timing', '/imu/device_time_ms'
    if custom_topics:
        camera_topic, device_topic = '/test/camera_time', '/test/imu_time'
        imu_config.update(camera_timing_topic=camera_topic, imu_device_time_topic=device_topic)
    config = dict(device_config=dict(calibration_model_path='unused', image_topic='/test/image'),
                  gelslam_config=dict(track_config=dict(debug_visualization_enabled=False)),
                  imu_config=imu_config)
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(config))
    tracker = SimpleNamespace(imu_enabled=True, timestamp_source='recorded_source',
        frontend_visual_inertial=None, imu_config=imu_config, imu_buffer=ImuBuffer(),
        image_to_imu_time_offset_s=0.)
    monkeypatch.setattr(module, 'Tracker', lambda *args, **kwargs: tracker)
    rclpy.init(args=['--ros-args', '-p', 'config_path:='+str(path),
                    '-p', 'data_dir:='+str(tmp_path), '-p', 'backend_feedback:=false'])
    node = None
    try:
        node = TrackerNode()
        topics = {subscription.topic_name for subscription in node.subscriptions}
        assert {camera_topic, device_topic, '/test/image', '/test/imu'} <= topics
        assert isinstance(node.streaming_source_clock, StreamingSourceClock)
    finally:
        if node is not None:
            Node.destroy_node(node)
        rclpy.shutdown()
