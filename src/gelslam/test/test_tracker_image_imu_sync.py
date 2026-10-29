"""IMU-ready image scheduling, selected clocks, and callback ownership."""
from collections import deque
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from sensor_msgs.msg import Image, Imu

from gelslam.core.inertial import ImuBuffer, ImuSample
from gelslam.core.tracker import Tracker
from gelslam.nodes.tracker_node import TrackerNode


def message_at(kind, timestamp):
    message = kind()
    message.header.stamp.sec = int(timestamp)
    message.header.stamp.nanosec = round((timestamp - int(timestamp)) * 1e9)
    message.header.frame_id = "imu_frame" if kind is Imu else "gel_frame"
    if kind is Imu:
        message.linear_acceleration.z = 9.81
    return message


class Guard:
    def __init__(self, callback, group):
        self.callback, self.group = callback, group
        self.triggers = 0
        self.ready = False

    def trigger(self):
        self.triggers += 1
        self.ready = True

    def dispatch(self):
        if self.ready:
            self.ready = False
            self.callback()


class SyncHarness:
    _initialize_image_imu_sync = TrackerNode._initialize_image_imu_sync
    _process_pending_image = TrackerNode._process_pending_image
    _resume_image_after_imu = TrackerNode._resume_image_after_imu
    image_callback = TrackerNode.image_callback
    imu_callback = TrackerNode.imu_callback

    def __init__(self, enabled=True, offset=0.):
        self.tracker = object.__new__(Tracker)
        self.tracker.imu_enabled = enabled
        self.tracker.imu_buffer = ImuBuffer()
        self.tracker.imu_config = {}
        self.tracker.timestamp_source = "header"
        self.tracker.image_to_imu_time_offset_s = offset
        self.recorded_source_clock = None
        self._unmatched_source_images = self._unmatched_source_imus = 0
        self._imu_dropped_samples = self._imu_invalid_gyro_samples = 0
        self.imu_frame_id = "imu_frame"
        self.gyro_input_unit = "rad_s"
        self.gyro_max_rate_rad_s = 5.
        self.image_callback_group = object()
        self.logs, self.tracked = [], []
        self.logger = SimpleNamespace(info=self.logs.append, warning=self.logs.append,
                                      error=self.logs.append)
        self._initialize_image_imu_sync()

    def get_logger(self):
        return self.logger

    def create_guard_condition(self, callback, callback_group):
        return Guard(callback, callback_group)

    def _process_image(self, message):
        self.tracked.append(message)


def test_pending_image_resumes_with_real_bracketing_imu_measurements():
    node = SyncHarness()
    before = message_at(Imu, 1.04)
    before.linear_acceleration.x = 1.
    node.imu_callback(before)
    image = message_at(Image, 1.05)
    node.image_callback(image)
    assert node.tracked == [] and node._pending_image is image
    assert node.tracker.imu_buffer.sample_at(1.05) is None
    node._imu_image_sync_guard.dispatch()
    assert node.tracked == [] and node._image_imu_waits == 1

    after = message_at(Imu, 1.06)
    after.linear_acceleration.x = 3.
    node.imu_callback(after)
    assert node.tracked == []  # IMU reception cannot execute image tracking.
    node._imu_image_sync_guard.dispatch()
    assert node.tracked == [image] and node._pending_image is None
    endpoint = node.tracker.imu_buffer.sample_at(1.05)
    np.testing.assert_allclose(endpoint.acceleration, [2., 0., 9.81])
    assert node.tracker.imu_buffer.latest_timestamp() == 1.06
    node._imu_image_sync_guard.trigger()
    node._imu_image_sync_guard.dispatch()
    assert node.tracked == [image]  # A ready image is processed exactly once.


@pytest.mark.parametrize("latest", [1.05, 1.06])
def test_covered_image_runs_immediately_without_waiting(latest):
    node = SyncHarness()
    node.imu_callback(message_at(Imu, latest))
    image = message_at(Image, 1.05)
    node.image_callback(image)
    assert node.tracked == [image]
    assert node._image_imu_waits == node._image_imu_replacements == 0


def test_new_image_replaces_pending_image_and_waits_for_its_own_time():
    node = SyncHarness()
    first, latest = message_at(Image, 1.05), message_at(Image, 1.07)
    node.image_callback(first)
    node.image_callback(latest)
    node.imu_callback(message_at(Imu, 1.06))
    node._imu_image_sync_guard.dispatch()
    assert node.tracked == [] and node._pending_image is latest
    node.imu_callback(message_at(Imu, 1.08))
    node._imu_image_sync_guard.dispatch()
    assert node.tracked == [latest]
    assert node._image_imu_waits == 2 and node._image_imu_replacements == 1


def test_wait_uses_recorded_camera_and_imu_clocks_with_image_offset():
    node = SyncHarness(offset=.02)
    node.tracker.timestamp_source = "recorded_source"
    image_times, imu_times = deque([2.05]), deque([2.06, 2.08])
    node.recorded_source_clock = SimpleNamespace(
        image_timestamp=lambda _: image_times.popleft(),
        imu_timestamp=lambda _: imu_times.popleft())
    image = message_at(Image, 100.)
    node.image_callback(image)
    node.imu_callback(message_at(Imu, 101.))
    node._imu_image_sync_guard.dispatch()
    assert node.tracked == []  # Selected target is 2.05 + 0.02, not host 100.
    node.imu_callback(message_at(Imu, 102.))
    node._imu_image_sync_guard.dispatch()
    assert node.tracked == [image] and not image_times and not imu_times
    assert node.tracker._imu_timestamp(image.header) == pytest.approx(2.07)


def test_rejected_imu_does_not_advance_or_release_pending_image():
    node = SyncHarness()
    image = message_at(Image, 1.05)
    node.image_callback(image)
    invalid = message_at(Imu, 1.06)
    invalid.header.frame_id = "wrong_frame"
    node.imu_callback(invalid)
    assert node.tracker.imu_buffer.latest_timestamp() is None
    assert node._imu_image_sync_guard.triggers == 0 and not node.tracked
    node.imu_callback(message_at(Imu, 1.06))
    node._imu_image_sync_guard.dispatch()
    assert node.tracked == [image]
    node.imu_callback(message_at(Imu, 1.06))  # Duplicate stamp cannot wake the gate.
    assert node._imu_image_sync_guard.triggers == 1


def test_image_waits_for_complete_imu_ingestion_before_tracking():
    node = SyncHarness()
    appended, release = threading.Event(), threading.Event()
    def delayed_ingestion(sample):
        node.tracker.imu_buffer.add(sample)
        appended.set()
        assert release.wait(2.)
        return True
    node.tracker.add_imu_sample = delayed_ingestion
    producer = threading.Thread(target=node.imu_callback, args=(message_at(Imu, 1.06),))
    producer.start()
    image = message_at(Image, 1.05)
    try:
        assert appended.wait(2.)
        assert node.tracker.imu_buffer.latest_timestamp() == 1.06
        node.image_callback(image)
        assert node.tracked == []  # One buffer append is not yet accepted input.
    finally:
        release.set()
        producer.join(timeout=2.)
    assert not producer.is_alive()
    node._imu_image_sync_guard.dispatch()
    assert node.tracked == [image]


def test_imu_disabled_images_run_without_a_guard_or_measurements():
    node = SyncHarness(enabled=False)
    image = message_at(Image, 1.)
    node.image_callback(image)
    assert node.tracked == [image] and node._imu_image_sync_guard is None


def test_missing_old_coverage_does_not_wait_for_an_unrecoverable_timestamp():
    node = SyncHarness()
    node.imu_callback(message_at(Imu, 2.))
    image = message_at(Image, 1.)
    node.image_callback(image)
    assert node.tracked == [image]  # Existing tracking handles the missing anchor.
    assert node.tracker.imu_buffer.sample_at(1.) is None


def test_latest_timestamp_follows_bounded_buffer_and_restored_state():
    buffer = ImuBuffer(max_samples=2)
    assert buffer.latest_timestamp() is None
    for timestamp in (1., 2., 3.):
        buffer.add(ImuSample(timestamp, (0., 0., 9.81), (0., 0., 0.)))
    assert buffer.latest_timestamp() == 3.
    restored = object.__new__(ImuBuffer)
    restored.__setstate__(buffer.__getstate__())
    assert restored.latest_timestamp() == 3.


def test_ros_guard_keeps_tracking_serial_and_allows_imu_during_tracking(tmp_path, monkeypatch):
    """Exercise real ROS guard scheduling without DDS sensor publishers."""
    import rclpy
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
    from rclpy.context import Context
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node

    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path / "ros_log"))
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    context = Context()
    rclpy.init(context=context)
    started, release, received = threading.Event(), threading.Event(), threading.Event()
    finished = threading.Event()

    class SyncNode(TrackerNode):
        def __init__(self):
            Node.__init__(self, "gelslam_image_imu_sync_test", context=context,
                          enable_rosout=False, start_parameter_services=False)
            harness = SyncHarness()
            for name in ("tracker", "recorded_source_clock", "_unmatched_source_images",
                         "_unmatched_source_imus", "_imu_dropped_samples",
                         "_imu_invalid_gyro_samples", "imu_frame_id", "gyro_input_unit",
                         "gyro_max_rate_rad_s", "logger"):
                setattr(self, name, getattr(harness, name))
            self.image_callback_group = MutuallyExclusiveCallbackGroup()
            self.imu_callback_group = MutuallyExclusiveCallbackGroup()
            self.tracked, self.active, self.max_active = [], 0, 0
            self._initialize_image_imu_sync()

        def _process_image(self, message):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.tracked.append(message)
            try:
                if len(self.tracked) == 1:
                    started.set()
                    assert release.wait(3.), "test did not release image tracking"
                else:
                    finished.set()
            finally:
                self.active -= 1

    node = SyncNode()
    executor = MultiThreadedExecutor(num_threads=2, context=context)
    executor.add_node(node)
    first, second = message_at(Image, 1.05), message_at(Image, 1.07)
    node.image_callback(first)
    imu_messages = deque([message_at(Imu, 1.06), message_at(Imu, 1.08)])
    def accept_imu():
        node.imu_callback(imu_messages.popleft())
        if not imu_messages:
            received.set()
    imu_guard = node.create_guard_condition(accept_imu, callback_group=node.imu_callback_group)
    image_guard = node.create_guard_condition(
        lambda: node.image_callback(second), callback_group=node.image_callback_group)
    runner = threading.Thread(target=executor.spin)
    runner.start()
    try:
        imu_guard.trigger()
        assert started.wait(3.), "IMU did not resume the waiting image"
        imu_guard.trigger()
        image_guard.trigger()
        assert received.wait(3.), "image tracking blocked IMU ingestion"
        assert node.tracker.imu_buffer.latest_timestamp() == 1.08
        assert node.tracked == [first]  # Image group remains exclusive.
        release.set()
        assert finished.wait(3.), "queued image did not run after tracking finished"
    finally:
        release.set()
        executor.shutdown(timeout_sec=3.)
        runner.join(timeout=3.)
        Node.destroy_node(node)
        context.try_shutdown()
    assert not runner.is_alive()
    assert node.tracked == [first, second]
    assert node.max_active == 1


def test_pending_image_does_not_block_executor_shutdown(tmp_path, monkeypatch):
    import rclpy
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
    from rclpy.context import Context
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node

    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path / "ros_log"))
    context = Context()
    rclpy.init(context=context)
    node = Node("gelslam_pending_image_shutdown_test", context=context,
                enable_rosout=False, start_parameter_services=False)
    harness = SyncHarness()
    harness.image_callback_group = MutuallyExclusiveCallbackGroup()
    harness.create_guard_condition = node.create_guard_condition
    harness._initialize_image_imu_sync()
    executor = MultiThreadedExecutor(num_threads=2, context=context)
    executor.add_node(node)
    try:
        harness.image_callback(message_at(Image, 1.05))
        assert not harness.tracked and harness._pending_image is not None
        assert executor.shutdown(timeout_sec=.5)
    finally:
        Node.destroy_node(node)
        context.try_shutdown()


def test_single_process_adapter_resumes_only_on_its_tracking_worker():
    from gelslam.core.slam_system import SlamSystem
    from gelslam.nodes.gelslam_node import GelSlamNode

    waiting, ingested, processed = threading.Event(), threading.Event(), threading.Event()
    access_threads = []

    class WorkerHarness(SyncHarness):
        image_callback = GelSlamNode.image_callback
        _resume_image_after_imu = GelSlamNode._resume_image_after_imu

        def _process_pending_image(self):
            access_threads.append(threading.get_ident())
            TrackerNode._process_pending_image(self)
            if self._pending_image is not None:
                waiting.set()

        def _process_image(self, message):
            super()._process_image(message)
            processed.set()

    class Mapper:
        def process(self, *args):
            return False

    node = WorkerHarness()
    node.system = SlamSystem(lambda message: TrackerNode.image_callback(node, message), Mapper())
    node.system.start()
    image = message_at(Image, 1.05)
    try:
        node.image_callback(image)
        assert waiting.wait(2.) and node.tracked == []
        node.system.submit_input(TrackerNode.imu_callback, node, message_at(Imu, 1.06))
        node.system.submit_input(ingested.set)
        assert ingested.wait(2.)
        assert node.tracked == [] and node._imu_image_sync_guard.ready
        node._imu_image_sync_guard.dispatch()  # Simulate the ROS executor thread.
        assert processed.wait(2.) and node.tracked == [image]
    finally:
        node.system.stop()
    tracking_worker = next(t for t in node.system._threads if t.name == "gelslam-tracking")
    assert access_threads and set(access_threads) == {tracking_worker.ident}
    assert tracking_worker.ident != threading.get_ident()
