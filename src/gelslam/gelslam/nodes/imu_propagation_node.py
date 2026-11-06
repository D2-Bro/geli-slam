"""Publish raw IMU dead reckoning for inspecting a rosbag in RViz.

This node intentionally contains no tactile, visual, frontend, or backend
update.  The first accepted IMU sample defines the world origin and initial
attitude.  Each later sample advances that state once using the same raw
propagator used by GelSLAM no-contact coasting.
"""

import os

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster

from gelslam.core.inertial import (
    ImuSample,
    InertialPropagator,
    InertialState,
    gyro_to_rad_per_sec,
)


WORLD_FRAME = "imu_dead_reckoning_world"
BODY_FRAME = "imu_dead_reckoning"


def quaternion_from_rotation(rotation):
    """Return an xyzw quaternion for a proper 3x3 rotation matrix."""
    rotation = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(rotation)
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        return np.array(((rotation[2, 1] - rotation[1, 2]) / scale,
                         (rotation[0, 2] - rotation[2, 0]) / scale,
                         (rotation[1, 0] - rotation[0, 1]) / scale,
                         0.25 * scale))
    axis = int(np.argmax(np.diag(rotation)))
    if axis == 0:
        scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        return np.array((0.25 * scale, (rotation[0, 1] + rotation[1, 0]) / scale,
                         (rotation[0, 2] + rotation[2, 0]) / scale,
                         (rotation[2, 1] - rotation[1, 2]) / scale))
    if axis == 1:
        scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        return np.array(((rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale,
                         (rotation[1, 2] + rotation[2, 1]) / scale,
                         (rotation[0, 2] - rotation[2, 0]) / scale))
    scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
    return np.array(((rotation[0, 2] + rotation[2, 0]) / scale,
                     (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale,
                     (rotation[1, 0] - rotation[0, 1]) / scale))


class ImuPropagationNode(Node):
    """ROS 2 wrapper for raw IMU-only dead reckoning."""

    def __init__(self):
        super().__init__("imu_propagation_node")
        self.declare_parameter("config_path", "")
        config_path = os.path.abspath(os.path.expanduser(
            self.get_parameter("config_path").value
        ))
        with open(config_path, encoding="utf-8") as stream:
            imu_config = yaml.safe_load(stream).get("imu_config", {})

        self.declare_parameter("imu_topic", imu_config.get("topic", "/imu/data"))
        self.declare_parameter("world_frame", WORLD_FRAME)
        self.declare_parameter("body_frame", BODY_FRAME)
        self.declare_parameter("path_max_poses", 10000)
        self.imu_topic = self.get_parameter("imu_topic").value
        self.world_frame = self.get_parameter("world_frame").value
        self.body_frame = self.get_parameter("body_frame").value
        self.path_max_poses = max(1, int(self.get_parameter("path_max_poses").value))
        self.frame_id = imu_config.get("frame_id", "")
        self.gyro_limit = imu_config.get("gyro_max_rate_rad_s", 5.0)
        self.accel_bias = np.asarray(
            imu_config.get("initial_accel_bias_mps2", (0.0, 0.0, 0.0)), dtype=np.float64
        )
        self.gyro_bias = np.asarray(
            imu_config.get("initial_gyro_bias_rps", (0.0, 0.0, 0.0)), dtype=np.float64
        )
        self.gravity_magnitude = float(imu_config.get("gravity_mps2", 9.81))
        self.declare_parameter("gravity_initialization", "measured_acceleration")
        self.gravity_initialization = self.get_parameter("gravity_initialization").value
        if self.gravity_initialization not in {"measured_acceleration", "fixed_world_z"}:
            raise ValueError(
                "gravity_initialization must be 'measured_acceleration' or 'fixed_world_z'"
            )
        self.propagator = None
        self.state = None
        self.previous_sample = None
        self.path = Path()
        self.path.header.frame_id = self.world_frame
        self.pose_pub = self.create_publisher(PoseStamped, "/gelslam/imu_only/pose", 10)
        self.path_pub = self.create_publisher(Path, "/gelslam/imu_only/path", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(Imu, self.imu_topic, self.imu_callback, 1000)
        self.get_logger().info(
            "IMU-only propagation: topic=%s, world=%s, body=%s" %
            (self.imu_topic, self.world_frame, self.body_frame)
        )

    def imu_callback(self, message):
        timestamp = float(message.header.stamp.sec) + message.header.stamp.nanosec * 1e-9
        if timestamp <= 0.0 or (self.frame_id and message.header.frame_id != self.frame_id):
            return
        try:
            sample = ImuSample(
                timestamp,
                (message.linear_acceleration.x, message.linear_acceleration.y,
                 message.linear_acceleration.z),
                gyro_to_rad_per_sec(
                    (message.angular_velocity.x, message.angular_velocity.y,
                     message.angular_velocity.z), max_rate_rad_s=self.gyro_limit
                ),
            )
        except ValueError as error:
            self.get_logger().warning("Dropping IMU sample: %s" % error)
            return
        if self.previous_sample is not None and sample.timestamp <= self.previous_sample.timestamp:
            self.get_logger().warning("Dropping non-monotonic IMU timestamp")
            return
        if self.state is None:
            if self.gravity_initialization == "fixed_world_z":
                gravity = np.array((0.0, 0.0, -self.gravity_magnitude))
            else:
                # The visualization frame is the first IMU frame.  Derive
                # gravity in that frame from a stationary first sample, so
                # the bag need not begin level.  A moving start will expose
                # itself immediately as propagation error, as intended.
                measured_gravity = sample.acceleration - self.accel_bias
                magnitude = np.linalg.norm(measured_gravity)
                if magnitude <= 1e-9:
                    self.get_logger().warning("Dropping zero-norm initial acceleration")
                    return
                gravity = -self.gravity_magnitude * measured_gravity / magnitude
            self.propagator = InertialPropagator(gravity)
            self.state = InertialState(
                sample.timestamp, np.eye(3), np.zeros(3), np.zeros(3),
                self.accel_bias, self.gyro_bias,
            )
            self.previous_sample = sample
            self.get_logger().info(
                "Initialized at first IMU stamp %.9f; gravity=%s" %
                (timestamp, np.array2string(gravity, precision=4))
            )
        else:
            self.state = self.propagator.propagate(self.state, (self.previous_sample, sample))
            self.previous_sample = sample
        self.publish_state(message.header.stamp)

    def publish_state(self, stamp):
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.world_frame
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = self.state.position
        quaternion = quaternion_from_rotation(self.state.rotation)
        pose.pose.orientation.x, pose.pose.orientation.y = quaternion[:2]
        pose.pose.orientation.z, pose.pose.orientation.w = quaternion[2:]
        self.pose_pub.publish(pose)
        self.path.header.stamp = stamp
        self.path.poses.append(pose)
        if len(self.path.poses) > self.path_max_poses:
            del self.path.poses[:-self.path_max_poses]
        self.path_pub.publish(self.path)
        transform = TransformStamped()
        transform.header = pose.header
        transform.child_frame_id = self.body_frame
        transform.transform.translation.x = pose.pose.position.x
        transform.transform.translation.y = pose.pose.position.y
        transform.transform.translation.z = pose.pose.position.z
        transform.transform.rotation = pose.pose.orientation
        self.tf_broadcaster.sendTransform(transform)


def main(args=None):
    rclpy.init(args=args)
    node = ImuPropagationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
