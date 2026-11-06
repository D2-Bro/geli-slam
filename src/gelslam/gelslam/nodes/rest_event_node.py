"""Operator rest events for live ``recorded_rest_event`` initialization.

``launch/live_vi_launch.py`` starts this node (``rest_events:=true``). It opens
the window "GelSLAM rest events". Press SPACE in that window after the tracker
background is confirmed. The node publishes ``recorded_rest_started``, shows a
countdown of ``rest_duration_s``, checks the IMU coverage of the rest interval,
then publishes ``recorded_rest_completed`` and ``trial_ready`` (GO). Events are
std_msgs/String JSON on ``imu_config.static_initialization.events_topic`` with
``host_wall_ns`` from the host wall clock, the same axis as the live IMU and
image header stamps. If the IMU coverage check fails, the node does not publish
GO. SPACE repeats the rest; a new ``recorded_rest_started`` replaces the earlier
sequence in the tracker before the first contact.
"""

import json
import os
import threading
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Imu
from std_msgs.msg import String

from gelslam.core.static_imu_initialization import resolve_static_imu_config
from gelslam.core.imu_mount_calibration import load_config

WINDOW = "GelSLAM rest events"
COLORS = {"idle": (90, 90, 90), "rest": (0, 170, 230), "go": (40, 160, 40), "failed": (40, 40, 200)}


def rest_interval_check(samples, start, end, min_samples, max_gap_s):
    """Return an error string when IMU samples do not cover [start, end]."""
    times = np.array([sample[0] for sample in samples if start - max_gap_s <= sample[0] <= end + max_gap_s])
    if len(times) == 0:
        return "no IMU samples in the rest interval"
    if times[0] > start + max_gap_s or times[-1] < end - max_gap_s:
        return "IMU samples cover %.3f -> %.3f s of the %.3f s rest" % (
            max(times[0] - start, 0.0), min(times[-1] - start, end - start), end - start)
    inside = np.count_nonzero((times >= start) & (times <= end))
    if inside < min_samples:
        return "%d IMU samples in the rest; static_initialization.min_samples is %d" % (inside, min_samples)
    gap = float(np.max(np.diff(times), initial=0.0))
    if gap > max_gap_s + 1e-9:
        return "IMU sample gap %.3f s exceeds max_sample_gap_s %.3f s" % (gap, max_gap_s)
    return None


def rest_statistics(samples, start, end):
    """(sample rate, accel std, gyro std) with the static initializer's norms."""
    inside = [sample for sample in samples if start <= sample[0] <= end]
    accel = np.array([sample[1] for sample in inside])
    gyro = np.array([sample[2] for sample in inside])
    rate = (len(inside) - 1) / (inside[-1][0] - inside[0][0])
    return (rate, float(np.sqrt(np.sum(np.var(accel, axis=0, ddof=1)))),
            float(np.sqrt(np.sum(np.var(gyro, axis=0, ddof=1)))))


class RestEventNode(Node):
    def __init__(self):
        super().__init__("rest_event_node")
        default_config = os.path.join(
            get_package_share_directory("gelslam"), "config", "live_thick_far_config.yaml")
        self.declare_parameter("config_path", default_config)
        # recorded_rest_duration_s of the real-objects recordings.
        self.declare_parameter("rest_duration_s", 5.0)
        # IMU mounting slot of imu_config.gelsight_T_imu_calibration; "" keeps its slot.
        self.declare_parameter("imu_slot", "")
        config_path = os.path.abspath(os.path.expanduser(
            self.get_parameter("config_path").get_parameter_value().string_value))
        self.rest_duration_s = self.get_parameter("rest_duration_s").get_parameter_value().double_value
        if not np.isfinite(self.rest_duration_s) or self.rest_duration_s <= 0:
            raise ValueError("rest_duration_s must be positive and finite")
        imu_config = load_config(config_path, self.get_parameter("imu_slot").value)["imu_config"]
        _, settings, _ = resolve_static_imu_config(
            imu_config, np.asarray(imu_config["gelsight_T_imu"], dtype=float))
        if settings is None or settings["mode"] != "recorded_rest_event":
            raise ValueError("%s: imu_config.static_initialization must be enabled with "
                             "mode: recorded_rest_event" % config_path)
        self.settings = settings
        self.buffer_max_samples = int(imu_config["buffer_max_samples"])
        self.events_topic = settings["events_topic"]
        self.imu_topic = imu_config["topic"]
        self.publisher = self.create_publisher(String, self.events_topic, 10)
        self._lock = threading.Lock()
        self._imu = deque(maxlen=self.buffer_max_samples)
        self.create_subscription(Imu, self.imu_topic, self._imu_callback, 1000)
        self.state, self.warning, self.failure = "idle", None, None
        self.rest_start_ns = self.go_ns = None
        self.buffer_span_s = None
        self.get_logger().info("Config %s; publishing rest events on %s; IMU check from %s; "
                               "press SPACE in the '%s' window to start the rest"
                               % (config_path, self.events_topic, self.imu_topic, WINDOW))

    def _imu_callback(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        a, w = msg.linear_acceleration, msg.angular_velocity
        with self._lock:
            self._imu.append((stamp, (a.x, a.y, a.z), (w.x, w.y, w.z)))

    def _imu_samples(self):
        with self._lock:
            return list(self._imu)

    def _publish(self, name, wall_ns, **fields):
        event = dict(event=name, host_wall_ns=int(wall_ns), host_monotonic_ns=time.monotonic_ns(),
                     source="rest_event_node", **fields)
        self.publisher.publish(String(data=json.dumps(event)))

    def start_rest(self):
        """Publish recorded_rest_started; ignored while a rest is running."""
        if self.state == "rest":
            return
        if self.count_subscribers(self.events_topic) == 0:
            self._fail("no subscriber on %s; tracker_node is not running" % self.events_topic)
            return
        self.rest_start_ns = time.time_ns()
        self._publish("recorded_rest_started", self.rest_start_ns,
                      duration_required_s=self.rest_duration_s)
        self.state, self.warning = "rest", None
        self.get_logger().info("REST: hands off, sensor supported, gel not in contact, do not move "
                               "for %.1f s" % self.rest_duration_s)

    def update(self):
        """Finish the rest after its end plus one max sample gap of IMU delivery."""
        if self.state != "rest":
            return
        end_ns = self.rest_start_ns + int(round(self.rest_duration_s * 1e9))
        if time.time_ns() < end_ns + int(self.settings["max_sample_gap_s"] * 1e9):
            return
        start, end = self.rest_start_ns * 1e-9, end_ns * 1e-9
        samples = self._imu_samples()
        error = rest_interval_check(samples, start, end, int(self.settings["min_samples"]),
                                    self.settings["max_sample_gap_s"])
        if error is not None:
            self._fail(error)
            return
        rate, accel_std, gyro_std = rest_statistics(samples, start, end)
        self.get_logger().info("Rest IMU: %.0f Hz, accel std %.4f m/s^2, gyro std %.5f rad/s"
                               % (rate, accel_std, gyro_std))
        if (accel_std > self.settings["accel_std_threshold_mps2"]
                or gyro_std > self.settings["gyro_std_threshold_rps"]):
            self.warning = "rest IMU std above stationary thresholds; repeat if the sensor moved"
            self.get_logger().warning(self.warning)
        self._publish("recorded_rest_completed", end_ns, completed=True,
                      duration_required_s=self.rest_duration_s, elapsed_s=end - start)
        self.go_ns = time.time_ns()
        self._publish("trial_ready", self.go_ns, recorded_supported_rest_s=end - start)
        self.buffer_span_s = self.buffer_max_samples / rate
        self.state = "go"
        self.get_logger().info(
            "GO: make the first contact within %.1f s %s and within %.1f s of GO (IMU buffer)"
            % (self.settings["max_propagation_s"], self._anchor_text(), self.buffer_span_s))

    def _anchor_text(self):
        return "of the first motion" if self.settings["motion_check"] else "of GO"

    def _fail(self, reason):
        self.state, self.failure = "failed", reason
        self.get_logger().error("REST FAILED: %s; GO not sent; press SPACE to repeat" % reason)

    def render(self):
        """Window image of the current operator instruction."""
        if self.state == "idle":
            title = "PRESS SPACE"
            lines = ["Start the %.0f s rest after the tracker background is confirmed."
                     % self.rest_duration_s, "Sensor supported, gel not in contact."]
        elif self.state == "rest":
            left = self.rest_duration_s - (time.time_ns() - self.rest_start_ns) * 1e-9
            title = "HOLD STILL  %d" % max(1, int(np.ceil(left)))
            lines = ["No contact. Hands off. Do not move."]
        elif self.state == "go":
            title = "GO  +%.1f s" % ((time.time_ns() - self.go_ns) * 1e-9)
            lines = ["Approach and touch: within %.0f s %s, %.0f s of GO."
                     % (self.settings["max_propagation_s"], self._anchor_text(), self.buffer_span_s),
                     "SPACE before the first contact: remove contact, repeat the rest."]
            if self.warning:
                lines.append("WARNING: " + self.warning)
        else:
            title = "REST FAILED"
            lines = [self.failure, "GO not sent. Press SPACE to repeat."]
        image = np.full((260, 900, 3), COLORS[self.state], dtype=np.uint8)
        cv2.putText(image, title, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (255, 255, 255), 5, cv2.LINE_AA)
        for row, line in enumerate(lines):
            cv2.putText(image, line[:80], (30, 160 + 34 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
        return image


def main(args=None):
    # SIGINT must raise KeyboardInterrupt in the main (window) thread.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = RestEventNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    def spin():
        try:
            executor.spin()
        except ExternalShutdownException:
            pass

    spinner = threading.Thread(target=spin, daemon=True)
    spinner.start()
    try:
        while rclpy.ok():
            node.update()
            cv2.imshow(WINDOW, node.render())
            if cv2.waitKey(30) & 0xFF == ord(" "):
                node.start_rest()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        # Stop the spin thread before the node and context are destroyed.
        rclpy.shutdown()
        spinner.join()
        executor.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
