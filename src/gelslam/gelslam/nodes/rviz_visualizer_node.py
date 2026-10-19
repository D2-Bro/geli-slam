"""Publish GelSLAM point clouds and estimated GelSight/IMU poses for RViz."""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import TransformStamped, Vector3Stamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from gelslam.core.frame import pose_of_frame_msg
from gelslam.core.inertial import rotation_aligning_gravity_to_negative_z
from gelslam.core.keyframe import KeyFrameDB
from gelslam.utils import Logger
from gelslam.visualization.visible_coverage_meshes import (
    VisibleCoverageMeshes,
    compute_frontend_tracking_meshes,
    compute_visible_coverage_meshes,
)
from gelslam_msgs.msg import FrameMsg, KeyFrameMsg, VisibleCoverageGraphMsg


RVIZ_FRAME = "gelslam_map"
GRAVITY_ALIGNED_FRAME = "gelslam_gravity_aligned"
GELSIGHT_FRAME = "gelsight_estimate"
IMU_FRAME = "imu_estimate"


def rotation_matrix_to_quaternion(rotation):
    """Convert a proper 3x3 rotation matrix to an xyzw quaternion."""
    rotation = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(rotation)
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        return np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    index = int(np.argmax(np.diag(rotation)))
    if index == 0:
        scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        return np.array(
            [
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
            ]
        )
    if index == 1:
        scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        return np.array(
            [
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
            ]
        )
    scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
    return np.array(
        [
            (rotation[0, 2] + rotation[2, 0]) / scale,
            (rotation[1, 2] + rotation[2, 1]) / scale,
            0.25 * scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        ]
    )


def vertices_to_pointcloud(vertices, stamp):
    """Create a compact XYZ point cloud that RViz renders as a smooth surface."""
    vertices = np.ascontiguousarray(np.asarray(vertices, dtype=np.float32).reshape(-1, 3))
    cloud = PointCloud2()
    cloud.header.frame_id = RVIZ_FRAME
    cloud.header.stamp = stamp
    cloud.height = 1
    cloud.width = len(vertices)
    cloud.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    cloud.is_bigendian = False
    cloud.point_step = 12
    cloud.row_step = cloud.point_step * cloud.width
    cloud.is_dense = True
    cloud.data = vertices.tobytes()
    return cloud


class RvizVisualizerNode(Node):
    """Bridge GelSLAM's custom messages to RViz point clouds and TF."""

    def __init__(self):
        super().__init__("rviz_visualizer_node")
        self.declare_parameter("config_path", "")
        config_path = self.get_parameter("config_path").value
        config_path = os.path.abspath(os.path.expanduser(config_path))
        with open(config_path, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        self.ppmm = config["device_config"]["ppmm"]
        imu_config = config.get("imu_config", {})
        transform = imu_config.get("gelsight_T_imu")
        self.gelsight_T_imu = (
            np.asarray(transform, dtype=np.float64) if transform is not None else None
        )
        if self.gelsight_T_imu is not None and self.gelsight_T_imu.shape != (4, 4):
            raise ValueError("imu_config.gelsight_T_imu must be a 4x4 matrix")

        self.declare_parameter("use_backend_poses", True)
        self.use_backend_poses = bool(
            self.get_parameter("use_backend_poses").value
        )
        self.keyframedb = KeyFrameDB(self.ppmm, Logger())
        self.visible_coverage_meshes = VisibleCoverageMeshes()
        self.pending_frame_msg = None
        self.latest_map_T_gelsight = np.eye(4, dtype=np.float64)
        self.latest_contact_local_cloud = None

        self.current_cloud_pub = self.create_publisher(
            PointCloud2, "/gelslam/rviz/current_surface", 1
        )
        self.global_cloud_pub = self.create_publisher(
            PointCloud2, "/gelslam/rviz/global_surface", 1
        )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.gravity_alignment = None
        self.publish_static_imu_transform()
        frontend_vi_enabled = (
            imu_config.get("enabled", False)
            and imu_config.get("frontend_optimization", False)
            and self.gelsight_T_imu is not None
        )
        if frontend_vi_enabled:
            self.create_subscription(
                Vector3Stamped,
                imu_config.get("frontend_gravity_topic", "/gelslam/frontend/gravity"),
                self.frontend_gravity_callback,
                QoSProfile(
                    depth=1,
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                ),
            )
        else:
            gravity = (
                imu_config.get(
                    "frontend_gravity_mps2",
                    (0.0, 0.0, -float(imu_config.get("gravity_mps2", 9.81))),
                )
                if frontend_vi_enabled
                else (0.0, 0.0, -1.0)
            )
            self.set_gravity_alignment(gravity)

        self.create_subscription(KeyFrameMsg, "keyframe", self.keyframe_callback, 10)
        self.create_subscription(FrameMsg, "frame", self.frame_callback, 1)
        self.create_subscription(
            PointCloud2, "/gelslam/current_contact_local",
            self.current_contact_callback, 1,
        )
        self.create_subscription(
            VisibleCoverageGraphMsg,
            "visible_coverage_graph",
            self.visible_coverage_graph_callback,
            1,
        )

    def frontend_gravity_callback(self, message):
        """Set the RViz map rotation from first-trial frontend gravity."""
        if self.gravity_alignment is not None:
            return
        if message.header.frame_id != "gelslam_first_trial":
            self.get_logger().warning("Ignored gravity in an unexpected frame")
            return
        self.set_gravity_alignment(
            (message.vector.x, message.vector.y, message.vector.z)
        )

    def set_gravity_alignment(self, gravity):
        """Align a legacy map, or use identity for an already aligned VI world."""
        try:
            rotation = rotation_aligning_gravity_to_negative_z(gravity)
        except ValueError as error:
            self.get_logger().warning("Ignored invalid frontend gravity: %s" % error)
            return
        self.gravity_alignment = np.eye(4)
        self.gravity_alignment[:3, :3] = rotation
        self.publish_static_imu_transform()
        self.get_logger().info("RViz map aligned to first-trial frontend gravity")

    def keyframe_callback(self, message):
        """Store a keyframe and update the tracker-only surface when selected."""
        self.keyframedb.insert(message)
        if not self.use_backend_poses:
            self.visible_coverage_meshes = compute_frontend_tracking_meshes(
                self.keyframedb, create_meshes=False
            )
            self._publish_global_surface()
        if self.pending_frame_msg is not None:
            pending = self.pending_frame_msg
            if self.keyframedb.find_kidx_from_kid(
                pending.ref_kid, self.keyframedb.size()
            ) is not None:
                self.frame_callback(pending)

    def visible_coverage_graph_callback(self, message):
        """Rebuild global point clouds when pose-graph coverage changes."""
        if not self.use_backend_poses:
            return
        kid2kidx = self.keyframedb.get_kid2kidx()
        self.visible_coverage_meshes = compute_visible_coverage_meshes(
            message, self.keyframedb, kid2kidx, create_meshes=False
        )
        self._publish_global_surface()

    def _publish_global_surface(self):
        """Publish XYZ data directly, including an empty cloud to clear RViz."""
        pointclouds = self.visible_coverage_meshes.visible_coverage_pointclouds
        vertices = np.vstack(pointclouds) if pointclouds else np.empty((0, 3))
        self.global_cloud_pub.publish(
            vertices_to_pointcloud(vertices, self.get_clock().now().to_msg())
        )

    def frame_callback(self, message):
        """Publish the current tactile patch and dynamic GelSight pose TF."""
        ref_kidx = self.keyframedb.find_kidx_from_kid(
            message.ref_kid, self.keyframedb.size()
        )
        if ref_kidx is None:
            # Keyframe and frame use different topics. Their callbacks can
            # arrive in either order, so keep the newest frame until its
            # reference keyframe is in the local database.
            self.pending_frame_msg = message
            return
        self.pending_frame_msg = None
        _, map_T_ref = self.visible_coverage_meshes.get_keyframe_pose(
            ref_kidx, self.keyframedb
        )
        map_T_gelsight = map_T_ref @ pose_of_frame_msg(message)
        self.latest_map_T_gelsight = map_T_gelsight
        self.tf_broadcaster.sendTransform(
            self.matrix_to_transform(
                map_T_gelsight,
                message.header.stamp,
                RVIZ_FRAME,
                GELSIGHT_FRAME,
            )
        )
        local_cloud = self.latest_contact_local_cloud
        if local_cloud is not None and local_cloud.header.stamp == message.header.stamp:
            self._publish_current_contact_cloud(local_cloud)

    def current_contact_callback(self, message):
        """Update the red surface from the latest reconstructed contact mask."""
        self.latest_contact_local_cloud = message
        self._publish_current_contact_cloud(message)

    def _publish_current_contact_cloud(self, message):
        if message.point_step != 12 or len(message.data) != message.width * 12:
            self.get_logger().warning("Ignored invalid current contact cloud")
            return
        local_points = np.frombuffer(message.data, dtype=np.float32).reshape(-1, 3)
        map_T_gelsight = self.latest_map_T_gelsight
        map_points = (
            local_points @ map_T_gelsight[:3, :3].T
            + map_T_gelsight[:3, 3]
        )
        self.current_cloud_pub.publish(
            vertices_to_pointcloud(map_points, message.header.stamp)
        )

    def publish_static_imu_transform(self):
        """Publish all static transforms together for late TF subscribers."""
        stamp = self.get_clock().now().to_msg()
        transforms = []
        if self.gelsight_T_imu is not None:
            transforms.append(
                self.matrix_to_transform(
                    self.gelsight_T_imu, stamp, GELSIGHT_FRAME, IMU_FRAME
                )
            )
        if self.gravity_alignment is not None:
            transforms.append(
                self.matrix_to_transform(
                    self.gravity_alignment, stamp, GRAVITY_ALIGNED_FRAME, RVIZ_FRAME
                )
            )
        if transforms:
            self.static_tf_broadcaster.sendTransform(transforms)

    @staticmethod
    def matrix_to_transform(matrix, stamp, parent_frame, child_frame):
        """Build a TransformStamped from a homogeneous transform."""
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame
        transform.transform.translation.x = float(matrix[0, 3])
        transform.transform.translation.y = float(matrix[1, 3])
        transform.transform.translation.z = float(matrix[2, 3])
        quaternion = rotation_matrix_to_quaternion(matrix[:3, :3])
        transform.transform.rotation.x = float(quaternion[0])
        transform.transform.rotation.y = float(quaternion[1])
        transform.transform.rotation.z = float(quaternion[2])
        transform.transform.rotation.w = float(quaternion[3])
        return transform


def main(args=None):
    """Run the RViz bridge node."""
    rclpy.init(args=args)
    node = RvizVisualizerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
