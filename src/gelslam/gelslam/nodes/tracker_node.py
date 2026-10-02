import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time

import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, Imu
from tf2_msgs.msg import TFMessage

from gelslam.core.inertial import ImuSample
from gelslam.core.tracker import Tracker
from gelslam.utils import Logger
from gelslam_msgs.msg import FrameMsg, KeyFrameMsg

"""
Single thread:
1. Track thread:
    - Image callback: Tracks the frame and publishes keyframes/frames.
"""


class TrackerNode(Node):
    """
    ROS2 node for real-time tracking.
    """

    def __init__(self):
        super().__init__("tracker_node")
        self.bridge = CvBridge()
        # Load configuration path
        self.declare_parameter("config_path", "")
        config_path = (
            self.get_parameter("config_path").get_parameter_value().string_value
        )
        config_path = os.path.abspath(os.path.expanduser(config_path))
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        # Get the calibration model path
        calib_model_path = config["device_config"]["calibration_model_path"]
        if not os.path.isabs(calib_model_path):
            calib_model_path = os.path.join(
                os.path.dirname(config_path), calib_model_path
            )
        # Get the directory to save the tracker states
        self.declare_parameter("data_dir", "")
        data_dir = self.get_parameter("data_dir").get_parameter_value().string_value
        data_dir = os.path.abspath(os.path.expanduser(data_dir))
        self.save_dir = os.path.join(data_dir, "gelslam_online")
        # Get the skip background check flag
        self.declare_parameter("skip_background_check", False)
        self.skip_background_check = (
            self.get_parameter("skip_background_check").get_parameter_value().bool_value
        )
        # Get the save gelslam state flag
        self.declare_parameter("save_gelslam_state", False)
        self.save_gelslam_state = (
            self.get_parameter("save_gelslam_state").get_parameter_value().bool_value
        )

        # The logger
        self.logger = Logger(ros_logger=self.get_logger())

        # Initialize tracker
        self.tracker = Tracker(
            calib_model_path, config, self.skip_background_check, logger=self.logger
        )

        # Initialize subscribers
        self.create_subscription(Image, "image", self.image_callback, 1)
        self.imu_sub = None
        self._imu_dropped_samples = 0
        if self.tracker.imu_enabled:
            imu_config = config["imu_config"]
            self.imu_frame_id = imu_config.get("frame_id", "")
            self.imu_sub = self.create_subscription(
                Imu,
                imu_config.get("topic", "/imu/data"),
                self.imu_callback,
                1000,
            )
            self.get_logger().info(
                "Buffering IMU samples from %s (frame_id=%s)"
                % (imu_config.get("topic", "/imu/data"), self.imu_frame_id)
            )
            if self.tracker.static_tf_validation_enabled:
                self._static_tf_validated_logged = False
                self._static_tf_invalid_logged = False
                tf_topic = self.tracker.static_tf_validation_config.get(
                    "topic", "/tf_static"
                )
                # A live static-transform broadcaster is transient-local,
                # while rosbag playback can offer the topic as volatile. Keep
                # both compatible subscriptions alive so either source is
                # validated before IMU propagation begins.
                self.static_tf_subs = [
                    self.create_subscription(
                        TFMessage,
                        tf_topic,
                        self.static_tf_callback,
                        QoSProfile(
                            depth=1,
                            durability=durability,
                            reliability=ReliabilityPolicy.RELIABLE,
                        ),
                    )
                    for durability in (
                        DurabilityPolicy.TRANSIENT_LOCAL,
                        DurabilityPolicy.VOLATILE,
                    )
                ]
                self.get_logger().info(
                    "Waiting to validate IMU extrinsic on %s" % tf_topic
                )

        # Initialize publishers
        self.keyframe_pub = self.create_publisher(KeyFrameMsg, "keyframe", 10)
        self.frame_pub = self.create_publisher(FrameMsg, "frame", 10)

    def image_callback(self, msg):
        """
        Callback function for the image subscriber.
        Tracks the frame and publishes keyframes/frames.

        :param msg: Image; The image message.
        """
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        ret, track_result = self.tracker.track(
            image, header=msg.header, log_prefix="Track Thread"
        )
        if ret:
            frame_msg, keyframe_msgs = track_result
            for keyframe_msg in keyframe_msgs:
                self.keyframe_pub.publish(keyframe_msg)
                time.sleep(0.001)
            self.frame_pub.publish(frame_msg)
            time.sleep(0.001)

    def imu_callback(self, msg):
        """Store calibrated gyro/accel measurements for future preintegration."""
        stamp = msg.header.stamp
        timestamp = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if timestamp <= 0.0:
            self._imu_dropped_samples += 1
            if self._imu_dropped_samples == 1:
                self.get_logger().warning("Dropping IMU sample with an unset header stamp")
            return
        if self.imu_frame_id and msg.header.frame_id != self.imu_frame_id:
            self._imu_dropped_samples += 1
            if self._imu_dropped_samples == 1:
                self.get_logger().warning(
                    "Dropping IMU sample from frame_id=%s; expected %s"
                    % (msg.header.frame_id, self.imu_frame_id)
                )
            return
        sample = ImuSample(
            timestamp=timestamp,
            acceleration=(
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ),
            angular_velocity=(
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ),
        )
        if not self.tracker.add_imu_sample(sample):
            self._imu_dropped_samples += 1
            if self._imu_dropped_samples == 1:
                self.get_logger().warning(
                    "Dropping duplicate or out-of-order IMU timestamp"
                )

    def static_tf_callback(self, msg):
        """Accept IMU coasting only after the configured static TF matches."""
        for transform in msg.transforms:
            result = self.tracker.validate_static_transform(
                transform.header.frame_id,
                transform.child_frame_id,
                (
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                ),
                (
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w,
                ),
            )
            if result is True and not self._static_tf_validated_logged:
                self.get_logger().info("Validated configured GelSight-to-IMU static TF")
                self._static_tf_validated_logged = True
            elif result is False and not self._static_tf_invalid_logged:
                self.get_logger().error(
                    "Configured GelSight-to-IMU transform disagrees with /tf_static; "
                    "IMU coasting is disabled"
                )
                self._static_tf_invalid_logged = True

    def destroy_node(self):
        """
        Clean up node and save tracker state.
        """
        # Clean up node
        super().destroy_node()
        # Save tracker state
        if self.save_gelslam_state:
            self.tracker.save(self.save_dir)


def main(args=None):
    """
    Main function to initialize and run the node.
    """
    rclpy.init(args=args)
    node = TrackerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        node.destroy_node()


if __name__ == "__main__":
    main()
