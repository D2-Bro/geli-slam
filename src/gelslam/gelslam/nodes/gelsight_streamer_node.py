import os
import re
import signal
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
import yaml
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
from gs_sdk.gs_device import Camera, FastCamera, resize_crop
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image


def _time_msg(wall_ns):
    """Convert wall-clock nanoseconds into a ROS time message."""
    stamp = Time()
    stamp.sec, stamp.nanosec = divmod(int(wall_ns), 1_000_000_000)
    return stamp


def _video_index(device_path):
    """Return N for a path that resolves to /dev/videoN."""
    real_path = os.path.realpath(os.path.expanduser(device_path))
    match = re.fullmatch(r"/dev/video(\d+)", real_path)
    if match is None:
        raise ValueError(
            "device_path %s resolves to %s, not /dev/videoN" % (device_path, real_path)
        )
    return int(match.group(1)), real_path


class CheckedFastCamera(FastCamera):
    """gs_sdk FastCamera with a corrected ffmpeg command.

    gs_sdk passes ``video_size="WxH"`` to ffmpeg-python. ffmpeg-python 0.2.0
    formats an input video_size as ``video_size[0] x video_size[1]``, so the
    string becomes ``3x2``, and the camera stream then fails with an
    input/output error. Its global flags are also placed after the output,
    where ffmpeg ignores them, and the constant-rate output duplicates frames.
    This class keeps the gs_sdk frame format (bgr24 rawvideo pipe at the raw
    size) and get_image()/release(), and changes only connect(): correct size,
    explicit MJPG input, options before the input, no duplicated frames,
    ffmpeg stderr written to a log file, and a check for end of stream during
    warm-up.
    """

    warm_up_frames = 100

    def connect(self, verbose=True, stderr_path=None):
        self.ffmpeg_command = [
            "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "warning",
            "-fflags", "nobuffer", "-flags", "low_delay",
            "-f", "v4l2", "-input_format", "mjpeg",
            "-framerate", str(self.framerate),
            "-video_size", "%dx%d" % (self.raw_imgw, self.raw_imgh),
            "-i", self.device,
            # The Mini delivers about 18.8 fps at a 25 fps setting. Without
            # passthrough, rawvideo output duplicates frames to a constant 25 fps.
            "-vsync", "passthrough",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
        ]
        self.stderr_path = stderr_path
        self._stderr_file = open(stderr_path, "ab") if stderr_path else subprocess.DEVNULL
        self.process = subprocess.Popen(
            self.ffmpeg_command, stdout=subprocess.PIPE, stderr=self._stderr_file
        )
        for index in range(self.warm_up_frames):
            frame = self.process.stdout.read(self.raw_size)
            if len(frame) != self.raw_size:
                self.process.wait(timeout=5.0)
                raise RuntimeError(
                    "ffmpeg stopped after %d warm-up frames (exit code %s). %s"
                    % (index, self.process.returncode, self.stderr_tail())
                )

    def stderr_tail(self, lines=5):
        """Return the last ffmpeg stderr lines for an error message."""
        if not self.stderr_path:
            return ""
        try:
            with open(self.stderr_path, "rb") as handle:
                text = handle.read().decode(errors="replace").strip().splitlines()
        except OSError:
            return ""
        return "ffmpeg: " + " | ".join(text[-lines:]) if text else ""

    def release(self):
        super().release()
        if self._stderr_file is not subprocess.DEVNULL:
            self._stderr_file.close()


class GelSightStreamerNode(Node):
    """
    ROS2 node for streaming images from a GelSight sensor through gs_sdk.

    A capture thread only reads frames, so the ffmpeg pipe is drained at the
    camera rate. It stamps each frame when its raw read completes (before the
    crop), then keeps only the newest frame. A publish thread sends that frame
    and optionally writes the video file.
    """

    def __init__(self):
        super().__init__("gelsight_streamer_node")
        self.bridge = CvBridge()
        # Get the configuration path and load the configuration
        self.declare_parameter("config_path", "")
        config_path = (
            self.get_parameter("config_path").get_parameter_value().string_value
        )
        config_path = os.path.abspath(os.path.expanduser(config_path))
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            device_config = config["device_config"]
            device_name = device_config["device_name"]
            imgh = device_config["imgh"]
            imgw = device_config["imgw"]
            raw_imgh = device_config["raw_imgh"]
            raw_imgw = device_config["raw_imgw"]
            framerate = device_config["framerate"]
        self.imgh, self.imgw = imgh, imgw
        # Get the data directory to save the video file
        self.declare_parameter("data_dir", "")
        self.data_dir = (
            self.get_parameter("data_dir").get_parameter_value().string_value
        )
        self.data_dir = os.path.abspath(os.path.expanduser(self.data_dir))
        # Get the streamer
        self.declare_parameter("streamer", "ffmpeg")
        self.streamer = (
            self.get_parameter("streamer").get_parameter_value().string_value
        )
        # Optional fixed device path, e.g. /dev/v4l/by-id/...-video-index0.
        # It replaces the gs_sdk name search, which can match a metadata node.
        self.declare_parameter("device_path", "")
        device_path = self.get_parameter("device_path").get_parameter_value().string_value
        self.declare_parameter("image_topic", "image")
        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.declare_parameter("frame_id", "gelsight")
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self.declare_parameter("save_video", True)
        self.save_video = self.get_parameter("save_video").get_parameter_value().bool_value
        self.declare_parameter("publish_encoding", "bgr8")
        self.publish_encoding = (
            self.get_parameter("publish_encoding").get_parameter_value().string_value
        )
        if self.publish_encoding not in ("bgr8", "rgb8"):
            raise ValueError("publish_encoding must be bgr8 or rgb8")
        self.declare_parameter("stats_period_s", 5.0)
        stats_period_s = self.get_parameter("stats_period_s").get_parameter_value().double_value

        # Create the directory for other nodes to save information
        self.save_dir = os.path.join(self.data_dir, "gelslam_online")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # Initialize and connect to the device
        if self.streamer == "ffmpeg":
            self.device = CheckedFastCamera(
                device_name, imgh, imgw, raw_imgh, raw_imgw, framerate, verbose=False
            )
            if device_path:
                _, self.device.device = _video_index(device_path)
            stderr_path = os.path.join(self.save_dir, "ffmpeg_camera.log")
            self.get_logger().info(
                "Connecting to %s with ffmpeg (MJPG %dx%d @ %s fps); warm-up discards %d "
                "frames; ffmpeg log %s"
                % (self.device.device, raw_imgw, raw_imgh, framerate,
                   CheckedFastCamera.warm_up_frames, stderr_path)
            )
            self.device.connect(verbose=False, stderr_path=stderr_path)
        elif self.streamer == "cv2":
            self.device = Camera(device_name, imgh, imgw)
            if device_path:
                self.device.dev_id, _ = _video_index(device_path)
            self.device.connect()
        else:
            raise ValueError("Invalid streamer: {}".format(self.streamer))
        # The video writer
        self.video_writer = None
        if self.save_video:
            self.video_writer = cv2.VideoWriter(
                os.path.join(self.data_dir, "gelsight.avi"),
                cv2.VideoWriter_fourcc(*"FFV1"),
                framerate,
                (imgw, imgh),
            )
        # Newest-frame slot shared by the capture and publish threads.
        self._slot_lock = threading.Condition()
        self._slot = None
        self._captured = 0
        self._published = 0
        self._overwritten = 0
        self._last_stats = (time.monotonic(), 0, 0, 0)
        # Flag to control the streaming threads
        self.running = True
        # Set when the camera stream fails; main() then exits with code 1.
        self.stream_failed = threading.Event()
        # Initialize publishers before starting the streaming threads.
        self.image_pub = self.create_publisher(Image, image_topic, 10)
        self.get_logger().info(
            "Publishing %s (%s, frame_id %s, save_video %s)"
            % (image_topic, self.publish_encoding, self.frame_id, self.save_video)
        )
        if stats_period_s > 0.0:
            self.create_timer(stats_period_s, self.log_stats)
        # Start the image streaming threads
        self.stream_thread = threading.Thread(target=self.stream_images)
        self.publish_thread = threading.Thread(target=self.publish_images)
        self.stream_thread.start()
        self.publish_thread.start()

    def read_frame(self):
        """Return (stamp_wall_ns, BGR image) or (None, None) at end of stream."""
        if self.streamer == "ffmpeg":
            device = self.device
            raw_frame = device.process.stdout.read(device.raw_size)
            # Stamp when the raw frame is fully read, before the crop.
            stamp_ns = time.time_ns()
            if raw_frame is None or len(raw_frame) != device.raw_size:
                return None, None
            frame = np.frombuffer(raw_frame, np.uint8).reshape(
                (device.raw_imgh, device.raw_imgw, 3)
            )
            # Same processing as gs_sdk FastCamera.get_image().
            return stamp_ns, resize_crop(frame, self.imgw, self.imgh)
        image = self.device.get_image()
        return time.time_ns(), image

    def stream_images(self):
        """
        Continuously read frames and keep only the newest one.
        """
        while self.running:
            try:
                stamp_ns, image = self.read_frame()
            except (OSError, ValueError) as error:
                if self.running:
                    self.get_logger().error("Camera read failed: %s" % error)
                    self.stream_failed.set()
                break
            if image is None:
                if self.running:
                    detail = ""
                    if self.streamer == "ffmpeg":
                        detail = self.device.stderr_tail()
                    self.get_logger().error("Camera stream ended; stopping capture. " + detail)
                    self.stream_failed.set()
                break
            with self._slot_lock:
                if self._slot is not None:
                    self._overwritten += 1
                self._slot = (stamp_ns, image)
                self._captured += 1
                self._slot_lock.notify()
        with self._slot_lock:
            self._slot_lock.notify_all()

    def publish_images(self):
        """
        Publish the newest frame and optionally save it.
        """
        while True:
            with self._slot_lock:
                while self._slot is None and self.running and self.stream_thread.is_alive():
                    self._slot_lock.wait(timeout=0.5)
                if self._slot is None:
                    if not self.running or not self.stream_thread.is_alive():
                        return
                    continue
                stamp_ns, image = self._slot
                self._slot = None
            if self.video_writer is not None:
                self.video_writer.write(image)
            if self.publish_encoding == "rgb8":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_msg = self.bridge.cv2_to_imgmsg(image, encoding=self.publish_encoding)
            image_msg.header.stamp = _time_msg(stamp_ns)
            image_msg.header.frame_id = self.frame_id
            if self.context.ok():
                self.image_pub.publish(image_msg)
                self._published += 1

    def log_stats(self):
        """Report capture and publish rates since the last report."""
        now = time.monotonic()
        last_time, last_captured, last_published, last_overwritten = self._last_stats
        elapsed = max(now - last_time, 1e-9)
        captured, published, overwritten = self._captured, self._published, self._overwritten
        message = "capture %.2f Hz, publish %.2f Hz, overwritten %d" % (
            (captured - last_captured) / elapsed,
            (published - last_published) / elapsed,
            overwritten - last_overwritten,
        )
        if captured == last_captured:
            self.get_logger().warning("No camera frame in the last period: " + message)
        else:
            self.get_logger().info(message)
        self._last_stats = (now, captured, published, overwritten)

    def destroy_node(self):
        # Cleanup
        self.running = False
        with self._slot_lock:
            self._slot_lock.notify_all()
        super().destroy_node()
        # Stop ffmpeg first so that a blocked pipe read returns at end of file.
        if self.streamer == "ffmpeg":
            self.device.process.terminate()
        # Stop the streaming threads
        self.stream_thread.join(timeout=5.0)
        self.publish_thread.join(timeout=5.0)
        self.device.release()
        # Save the tactile video
        if self.video_writer is not None:
            self.video_writer.release()


def main(args=None):
    """
    Main function to initialize and run the node.
    """
    rclpy.init(args=args)
    try:
        node = GelSightStreamerNode()
    except (RuntimeError, ValueError) as error:
        print("gelsight_streamer_node error: %s" % error, file=sys.stderr)
        if rclpy.ok():
            rclpy.shutdown()
        return 1
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        while rclpy.ok() and not node.stream_failed.is_set():
            executor.spin_once(timeout_sec=0.2)
    except (KeyboardInterrupt, ExternalShutdownException):
        # SIGINT from a terminal or from ros2 launch ends the node normally.
        pass
    finally:
        # A repeated SIGINT from the launch system must not interrupt cleanup.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    # A nonzero exit lets the live launch stop all nodes and save the outputs.
    return 1 if node.stream_failed.is_set() else 0


if __name__ == "__main__":
    sys.exit(main())
