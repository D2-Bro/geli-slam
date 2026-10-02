"""Publish GelSLAM meshes and estimated GelSight/IMU poses for RViz."""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from gelslam.core.frame import frame_msg2mesh, pose_of_frame_msg
from gelslam.core.keyframe import KeyFrameDB
from gelslam.utils import Logger
from gelslam.visualization.visible_coverage_meshes import (
    VisibleCoverageMeshes,
    compute_visible_coverage_meshes,
)
from gelslam_msgs.msg import FrameMsg, KeyFrameMsg, VisibleCoverageGraphMsg


RVIZ_FRAME = "gelslam_map"
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


def mesh_to_marker(mesh, stamp, namespace, marker_id, color, triangle_stride):
    """Convert an Open3D triangle mesh into a bounded RViz TRIANGLE_LIST."""
    marker = Marker()
    marker.header.frame_id = RVIZ_FRAME
    marker.header.stamp = stamp
    marker.ns = namespace
    marker.id = marker_id
    marker.type = Marker.TRIANGLE_LIST
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = marker.scale.y = marker.scale.z = 1.0
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = color

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    for triangle in triangles[:: max(1, triangle_stride)]:
        for vertex_index in triangle:
            vertex = vertices[vertex_index]
            marker.points.append(
                Point(x=float(vertex[0]), y=float(vertex[1]), z=float(vertex[2]))
            )
    return marker


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
    """Bridge GelSLAM's custom messages to RViz markers and TF."""

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

        self.declare_parameter("current_triangle_stride", 2)
        self.declare_parameter("global_triangle_stride", 8)
        self.current_triangle_stride = int(
            self.get_parameter("current_triangle_stride").value
        )
        self.global_triangle_stride = int(
            self.get_parameter("global_triangle_stride").value
        )
        self.bridge = CvBridge()
        self.keyframedb = KeyFrameDB(self.ppmm, Logger())
        self.visible_coverage_meshes = VisibleCoverageMeshes()
        self.global_marker_count = 0

        self.current_marker_pub = self.create_publisher(
            Marker, "/gelslam/rviz/current_mesh", 1
        )
        self.global_marker_pub = self.create_publisher(
            MarkerArray, "/gelslam/rviz/global_mesh", 1
        )
        self.current_cloud_pub = self.create_publisher(
            PointCloud2, "/gelslam/rviz/current_surface", 1
        )
        self.global_cloud_pub = self.create_publisher(
            PointCloud2, "/gelslam/rviz/global_surface", 1
        )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        self.create_subscription(KeyFrameMsg, "keyframe", self.keyframe_callback, 10)
        self.create_subscription(FrameMsg, "frame", self.frame_callback, 1)
        self.create_subscription(
            VisibleCoverageGraphMsg,
            "visible_coverage_graph",
            self.visible_coverage_graph_callback,
            1,
        )
        self.publish_static_imu_transform()

    def keyframe_callback(self, message):
        """Store keyframes needed to place current and global meshes."""
        self.keyframedb.insert(message)

    def visible_coverage_graph_callback(self, message):
        """Rebuild global mesh markers only when pose-graph coverage changes."""
        kid2kidx = self.keyframedb.get_kid2kidx()
        self.visible_coverage_meshes = compute_visible_coverage_meshes(
            message, self.keyframedb, kid2kidx
        )
        markers = MarkerArray()
        meshes = self.visible_coverage_meshes.visible_coverage_meshes
        for marker_id, mesh in enumerate(meshes):
            markers.markers.append(
                mesh_to_marker(
                    mesh,
                    self.get_clock().now().to_msg(),
                    "gelslam_global_mesh",
                    marker_id,
                    (0.65, 0.70, 0.75, 0.70),
                    self.global_triangle_stride,
                )
            )
        for marker_id in range(len(meshes), self.global_marker_count):
            marker = Marker()
            marker.header.frame_id = RVIZ_FRAME
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "gelslam_global_mesh"
            marker.id = marker_id
            marker.action = Marker.DELETE
            markers.markers.append(marker)
        self.global_marker_count = len(meshes)
        self.global_marker_pub.publish(markers)
        if meshes:
            vertices = np.vstack([np.asarray(mesh.vertices) for mesh in meshes])
            self.global_cloud_pub.publish(
                vertices_to_pointcloud(vertices, self.get_clock().now().to_msg())
            )

    def frame_callback(self, message):
        """Publish the current tactile patch and dynamic GelSight pose TF."""
        ref_kidx = self.keyframedb.find_kidx_from_kid(
            message.ref_kid, self.keyframedb.size()
        )
        if ref_kidx is None:
            return
        _, map_T_ref = self.visible_coverage_meshes.get_keyframe_pose(
            ref_kidx, self.keyframedb
        )
        map_T_gelsight = map_T_ref @ pose_of_frame_msg(message)
        current_mesh = frame_msg2mesh(self.bridge, message, map_T_ref, self.ppmm)
        self.current_cloud_pub.publish(
            vertices_to_pointcloud(current_mesh.vertices, message.header.stamp)
        )
        self.current_marker_pub.publish(
            mesh_to_marker(
                current_mesh,
                message.header.stamp,
                "gelslam_current_mesh",
                0,
                (0.95, 0.15, 0.15, 0.85),
                self.current_triangle_stride,
            )
        )
        self.tf_broadcaster.sendTransform(
            self.matrix_to_transform(
                map_T_gelsight,
                message.header.stamp,
                RVIZ_FRAME,
                GELSIGHT_FRAME,
            )
        )

    def publish_static_imu_transform(self):
        """Publish the configured GelSight-to-IMU relation under estimate frames."""
        if self.gelsight_T_imu is None:
            return
        self.static_tf_broadcaster.sendTransform(
            self.matrix_to_transform(
                self.gelsight_T_imu,
                self.get_clock().now().to_msg(),
                GELSIGHT_FRAME,
                IMU_FRAME,
            )
        )

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
