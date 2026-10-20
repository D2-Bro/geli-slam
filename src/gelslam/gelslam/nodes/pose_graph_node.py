import os
import signal
import json
from collections import deque
from traceback import format_exc

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import threading
import time

import numpy as np
import open3d as o3d
import rclpy
import yaml
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from cv_bridge import CvBridge
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from gelslam.core.coverage_graph import CoverageGraph, create_visible_coverage_graph_msg
from gelslam.core.keyframe import KeyFrameDB, compute_adjusted_pointcloud
from gelslam.core.parent_groups_info import ParentGroupsInfo
from gelslam.core.gtsam_pose_graph import PoseGraph, PoseGraphSolutions
from gelslam.core.topic_backend import TopicBackend
from gelslam.core.backend_feedback import feedback_message
from gelslam.core.slam_packets import keyframe_from_ros
from gelslam.core.trajectory_evaluation import (
    save_aligned_trajectory,
    trajectory_output_dir,
)
from gelslam.utils import Logger, pointcloud2mesh
from gelslam_msgs.msg import FrameMsg, KeyFrameMsg, VisibleCoverageGraphMsg, BackendFeedbackMsg, FramePoseMsg

"""
ROS callbacks enqueue keyframes and record lightweight frame poses.
A backend worker owns decoding, feature search, graph mutation, optimization,
coverage updates and publication. Tracking executes in another ROS process.
"""


class PoseGraphNode(Node):
    """
    ROS2 node for backend pose graph optimization.
    """

    def __init__(self):
        super().__init__("pose_graph_node")
        # Load the configuration
        self.declare_parameter("config_path", "")
        config_path = (
            self.get_parameter("config_path").get_parameter_value().string_value
        )
        config_path = os.path.abspath(os.path.expanduser(config_path))
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        # Get the directory to save the pose graph states
        self.declare_parameter("data_dir", "")
        data_dir_parameter = self.get_parameter("data_dir").get_parameter_value().string_value
        data_dir = os.path.abspath(os.path.expanduser(data_dir_parameter))
        self.save_dir = os.path.join(data_dir, "gelslam_online")
        self.declare_parameter("sequence_name", "")
        sequence_name = self.get_parameter("sequence_name").get_parameter_value().string_value
        self.trajectory_output_dir = trajectory_output_dir(
            self.save_dir, sequence_name, data_dir_parameter
        )
        self.declare_parameter("backend_optimization", True)
        self.backend_optimization_enabled = bool(
            self.get_parameter("backend_optimization").get_parameter_value().bool_value
        )
        self.declare_parameter("backend_feedback", True)
        self.backend_feedback_enabled = bool(
            self.get_parameter("backend_feedback").value and self.backend_optimization_enabled)
        self.feedback_messages_published = 0
        # Get the rendering flag
        self.declare_parameter("rendering", True)
        self.rendering = (
            self.get_parameter("rendering").get_parameter_value().bool_value
        )
        # Get the save gelslam states flag
        self.declare_parameter("save_gelslam_states", False)
        self.save_gelslam_states = (
            self.get_parameter("save_gelslam_states").get_parameter_value().bool_value
        )

        # The overall running flag
        self.running = True

        # The logger
        self.logger = Logger(ros_logger=self.get_logger())
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, 'backend_process_config_supplied.yaml'), 'w') as handle:
            yaml.safe_dump(dict(config=self.config, backend_optimization=self.backend_optimization_enabled,
                                backend_feedback=self.backend_feedback_enabled,
                                rendering=self.rendering, pid=os.getpid()), handle, sort_keys=False)

        self._backend = TopicBackend(self.config, self.logger, self.backend_optimization_enabled)
        self._bind_backend()
        self.add_keyframe_lock = threading.Lock()
        self._work = threading.Condition()
        self._pending = deque()
        self._failure = None
        self._closed = False
        self._last_received = {}
        self._stream_ends = {}
        self._bridge = CvBridge()
        self.processed_keyframe_count = 0
        self.gelsight_trajectory_pub = self.create_publisher(Path, "/gelslam/rviz/gelsight_trajectory", 1)
        self.feedback_pub = self.create_publisher(BackendFeedbackMsg, "backend_feedback", 1)
        self.create_subscription(String, "backend_stream_end", self.stream_end_callback, 10)
        if self.backend_optimization_enabled:
            if self.pose_graph.local_vi_enabled:
                self.logger.info("ROS process: backend; local 15D GelSight/IMU/bias graph; separate 4DoF yaw/XYZ loop PGO")
            else:
                self.logger.info("ROS process: backend; local VI disabled; 4DoF yaw/XYZ loop PGO with frontend seed odometry")
        else:
            self.logger.info("ROS process: backend raw storage; optimization and feedback disabled")
        self.logger.info("Backend feedback publication %s" % (
            "enabled" if self.backend_feedback_enabled else "disabled"))
        self.trajectory_evaluation_config = self.config.get("trajectory_evaluation", {})
        self.trajectory_evaluation_enabled = bool(
            self.trajectory_evaluation_config.get("enabled", False)
        )
        self.trajectory_lock = threading.Lock()
        self._trajectory_frames = []
        self._trajectory_ground_truth = []
        if self.rendering:
            self.visible_coverage_graph_pub = self.create_publisher(
                VisibleCoverageGraphMsg, "visible_coverage_graph", 1
            )

        # Initialize KeyFrame subscriber (K thread)
        self.create_subscription(
            KeyFrameMsg,
            "keyframe",
            self.keyframe_callback,
            1000,
        )
        if self.trajectory_evaluation_enabled:
            self.create_subscription(FramePoseMsg, "frame_pose", self.trajectory_frame_callback, 1000)
            self.create_subscription(
                PoseStamped,
                self.trajectory_evaluation_config.get(
                    "ground_truth_pose_topic", "/ground_truth/gel_pose"
                ),
                self.trajectory_ground_truth_callback,
                1000,
            )
            self.logger.info("Recording frame and ground-truth poses for final SE(3) evaluation")

        # Start the L thread: loop detection and pose graph optimization thread
        self.loop_closure_thread = threading.Thread(target=self.loop_closure,
                                                    name='gelslam-backend', daemon=False)
        self.loop_closure_thread.start()

    def _bind_backend(self):
        mapper = self._backend.mapper
        self.keyframedb, self.pose_graph = mapper.db, mapper.graph
        self.parent_groups_info, self.coverage_graph = mapper.groups, mapper.coverage
        self.pose_graph_solutions = mapper.solutions

    def keyframe_callback(self, keyframe_msg):
        # Decode and derived-feature work belong to the backend worker.
        with self._work:
            if not self.running:
                return
            self._last_received[keyframe_msg.session_id] = keyframe_msg.kid
            self._pending.append(keyframe_msg)
            self._work.notify_all()

    def stream_end_callback(self, message):
        try:
            record = json.loads(message.data)
            session, last = str(record['session_id']), int(record['last_kid'])
        except (ValueError, KeyError, TypeError):
            self.logger.warning('Ignoring invalid tracking end marker')
            return
        with self._work:
            self._stream_ends[session] = last

    def input_complete(self):
        with self._work:
            session = self._backend.session_id
            if session is None:
                return any(last < 0 for last in self._stream_ends.values())
            return (session in self._stream_ends and
                    self._last_received.get(session, -1) >= self._stream_ends[session])

    def raise_if_failed(self):
        with self._work:
            failure = self._failure
        if failure is not None:
            raise RuntimeError('Backend worker failed:\n' + failure[1]) from failure[0]

    def publish_gelsight_keyframe_trajectory(self):
        """Publish the solved poses of the largest backend keyframe group."""
        trajectory = Path()
        trajectory.header.frame_id = "gelslam_map"
        trajectory.header.stamp = self.get_clock().now().to_msg()
        with self.add_keyframe_lock:
            count = min(self.keyframedb.size(), self.pose_graph_solutions.size())
            if count and self.parent_groups_info.parent_group_sizes:
                active_group = self.parent_groups_info.get_largest_parent_group()
                for kidx in range(count):
                    if (
                        self.parent_groups_info.get_parent_group(kidx, self.keyframedb)
                        != active_group
                    ):
                        continue
                    pose = self.pose_graph_solutions[kidx]
                    stamped_pose = PoseStamped()
                    stamped_pose.header.frame_id = trajectory.header.frame_id
                    stamped_pose.header.stamp = trajectory.header.stamp
                    stamped_pose.pose.position.x = float(pose[0, 3])
                    stamped_pose.pose.position.y = float(pose[1, 3])
                    stamped_pose.pose.position.z = float(pose[2, 3])
                    # A Path display draws position segments. Keep a valid
                    # orientation without adding a second rotation conversion.
                    stamped_pose.pose.orientation.w = 1.0
                    trajectory.poses.append(stamped_pose)
        self.gelsight_trajectory_pub.publish(trajectory)


    @staticmethod
    def _header_timestamp(header):
        timestamp = float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9
        return timestamp if timestamp > 0.0 else None

    @staticmethod
    def _pose_from_message(message):
        quaternion = np.asarray(
            (message.pose.orientation.x, message.pose.orientation.y,
             message.pose.orientation.z, message.pose.orientation.w), dtype=np.float64,
        )
        norm = np.linalg.norm(quaternion)
        if not np.isfinite(norm) or norm <= 1e-12:
            return None
        x, y, z, w = quaternion / norm
        pose = np.eye(4)
        pose[:3, :3] = np.array(
            [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
             [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
             [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]],
            dtype=np.float64,
        )
        pose[:3, 3] = (message.pose.position.x, message.pose.position.y, message.pose.position.z)
        return pose if np.all(np.isfinite(pose)) else None

    def trajectory_frame_callback(self, frame_msg):
        timestamp = self._header_timestamp(frame_msg.header)
        pose_values = np.asarray(frame_msg.ref_t_curr, dtype=np.float64)
        if timestamp is None or pose_values.size != 16 or not np.all(np.isfinite(pose_values)):
            return
        with self.trajectory_lock:
            self._trajectory_frames.append(
                (timestamp, int(frame_msg.fid), int(frame_msg.ref_kid), pose_values.reshape(4, 4))
            )

    def trajectory_ground_truth_callback(self, pose_msg):
        timestamp = self._header_timestamp(pose_msg.header)
        pose = self._pose_from_message(pose_msg)
        if timestamp is None or pose is None:
            return
        with self.trajectory_lock:
            self._trajectory_ground_truth.append((timestamp, -1, -1, pose))

    def save_final_trajectory_evaluation(self):
        """Resolve stored frame poses with the final graph, then align to GT."""
        if not self.trajectory_evaluation_enabled:
            return
        with self.trajectory_lock:
            frames = list(self._trajectory_frames)
            ground_truth = list(self._trajectory_ground_truth)
        with self.add_keyframe_lock:
            kid_to_kidx = self.keyframedb.get_kid2kidx()
            if self.pose_graph_solutions.size() == 0 or not frames:
                self.logger.warning("No final trajectory was saved: missing frames or pose-graph solution")
                return
            active_group = self.parent_groups_info.get_largest_parent_group()
            final_poses = [self.pose_graph_solutions[index].copy()
                           for index in range(self.pose_graph_solutions.size())]
            active_kids = {
                keyframe.kid
                for index, keyframe in enumerate(self.keyframedb.keyframes)
                if index < len(final_poses)
                and self.parent_groups_info.get_parent_group(index, self.keyframedb) == active_group
            }
        final_frames = []
        for timestamp, fid, ref_kid, ref_T_frame in frames:
            kidx = kid_to_kidx.get(ref_kid)
            if kidx is None or ref_kid not in active_kids or kidx >= len(final_poses):
                continue
            final_frames.append((timestamp, fid, ref_kid, final_poses[kidx] @ ref_T_frame))
        if not final_frames:
            self.logger.warning("No recorded frames belong to the final active pose-graph component")
            return
        output_dir = self.trajectory_output_dir
        try:
            metrics = save_aligned_trajectory(
                output_dir,
                final_frames,
                ground_truth,
                float(self.trajectory_evaluation_config.get("timestamp_tolerance_s", 0.03)),
            )
        except (OSError, ValueError, ImportError) as error:
            self.logger.error("Failed to save final trajectory evaluation: %s" % error)
            return
        if metrics["status"] == "ok":
            self.logger.info(
                "Saved %d SE(3)-aligned poses: translation RMSE %.6f m, rotation RMSE %.3f deg"
                % (metrics["matched_poses"], metrics["translation_rmse_m"], metrics["rotation_rmse_deg"])
            )
        else:
            self.logger.warning(
                "Saved raw final trajectory, but only %d GT matches were available"
                % metrics["matched_poses"]
            )

    def loop_closure(self):
        """Drain ordered keyframes; search, optimize and publish in one owner."""
        try:
            while True:
                with self._work:
                    self._work.wait_for(lambda: self._pending or not self.running)
                    if not self._pending and not self.running:
                        return
                    pending, self._pending = self._pending, deque()
                packets = [keyframe_from_ros(message, self._bridge) for message in pending]
                if self.backend_optimization_enabled and any(not p.navigation_valid for p in packets):
                    raise ValueError('15D backend requires synchronized navigation keyframe seeds')
                changed = self._backend.process(packets)
                self._bind_backend()
                self.processed_keyframe_count = self.keyframedb.size()
                if not changed:
                    continue
                if self.backend_feedback_enabled:
                    feedback = feedback_message(self._backend.store.snapshot_for_tracking(),
                                                self._backend.session_id)
                    feedback.header.stamp = self.get_clock().now().to_msg()
                    self.feedback_pub.publish(feedback)
                    self.feedback_messages_published += 1
                self.publish_gelsight_keyframe_trajectory()
                if self.rendering and self.backend_optimization_enabled and self.keyframedb.size():
                    self.visible_coverage_graph_pub.publish(create_visible_coverage_graph_msg(
                        self.keyframedb.size() - 1, self.keyframedb, self.pose_graph_solutions,
                        self.parent_groups_info, self.coverage_graph))
        except BaseException as error:
            with self._work:
                self._failure = (error, format_exc())
                self.running = False
                self._work.notify_all()

    def merge_and_save_mesh(self):
        """
        Find the keyframes from the largest connected group of keyframes.
        Merging the meshes and save the reconstructed mesh.
        """
        updated_size = self.pose_graph_solutions.size()
        if not updated_size:
            self.logger.info('No backend mesh to export: no keyframes')
            return
        active_parent_group = self.parent_groups_info.get_largest_parent_group()
        active_kidxs = []
        for kidx in range(updated_size):
            parent_group = self.parent_groups_info.get_parent_group(
                kidx, self.keyframedb
            )
            if parent_group == active_parent_group:
                active_kidxs.append(kidx)
        # Construct a new and clean coverage graph
        coverage_graph = CoverageGraph(self.config)
        coverage_graph.add_new_coverage_nodes(0, updated_size)
        coverage_graph.update_wrt_new_keyframes(
            active_kidxs,
            self.keyframedb,
            self.pose_graph_solutions,
            self.parent_groups_info,
        )
        # Merging the meshes
        merged_mesh = o3d.geometry.TriangleMesh()
        for kidx in active_kidxs:
            keyframe = self.keyframedb[kidx]
            if not coverage_graph[kidx].is_active:
                continue
            adjusted_pointcloud = compute_adjusted_pointcloud(
                kidx,
                coverage_graph[kidx].neighbor_kidxs,
                self.keyframedb,
                self.pose_graph_solutions,
            )
            keyframe_mesh = pointcloud2mesh(adjusted_pointcloud, keyframe.C)
            merged_mesh += keyframe_mesh
        save_path = os.path.join(self.save_dir, "reconstructed_mesh.ply")
        o3d.io.write_triangle_mesh(
            save_path,
            merged_mesh,
            write_ascii=False,
            compressed=False,
        )
        self.logger.info("Merging and saving mesh...")
        self.logger.info("Reconstructed mesh saved in %s" % (save_path))

    def destroy_node(self):
        """
        Cleanup node and stop the loop closure thread.
        Merge the meshes for reconstruction.
        Save pose graph states.
        """
        if self._closed:
            return
        self._closed = True
        with self._work:
            self.running = False
            self._work.notify_all()
        self.loop_closure_thread.join()
        self.raise_if_failed()
        with open(os.path.join(self.save_dir, 'backend_process_summary.json'), 'w') as handle:
            json.dump(dict(pid=os.getpid(), session_id=self._backend.session_id,
                keyframes=self.keyframedb.size(), last_received=self._last_received,
                stream_ends=self._stream_ends, worker_joined=not self.loop_closure_thread.is_alive(),
                map_revision=self._backend.store.snapshot_for_tracking().revision,
                backend_feedback=self.backend_feedback_enabled,
                feedback_messages_published=self.feedback_messages_published,
                local_vi_enabled=self.backend_optimization_enabled and self.pose_graph.local_vi_enabled,
                factor_kinds={kind: self.pose_graph.factor_kinds.count(kind)
                              for kind in set(self.pose_graph.factor_kinds)},
                loop_pgo_factor_kinds={kind: self.pose_graph.loop_pgo.factor_kinds.count(kind)
                                      for kind in set(self.pose_graph.loop_pgo.factor_kinds)},
                loop_pgo_degrees_of_freedom=self.pose_graph.loop_pgo.degrees_of_freedom,
                loop_pairs=len(self._backend.mapper.accepted_pairs),
                loop_pgo_initial_cost=self.pose_graph.loop_pgo.last_initial_cost,
                loop_pgo_final_cost=self.pose_graph.loop_pgo.last_final_cost), handle, indent=2)

        # The graph solution is final only after the loop-closure thread has
        # stopped; resolve all lightweight FrameMsg records at this point.
        self.save_final_trajectory_evaluation()

        # A backend-off run keeps raw keyframe poses for evaluation only. It
        # does not create a separate backend mesh that could overwrite a
        # tracker-only visualization result.
        if self.backend_optimization_enabled:
            self.merge_and_save_mesh()

        # Save pose graph states
        if self.save_gelslam_states:
            self.pose_graph.save(self.save_dir)
            self.keyframedb.save(self.save_dir)
            self.parent_groups_info.save(self.save_dir)
            self.pose_graph_solutions.save(self.save_dir)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = PoseGraphNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    def stop(_signum, _frame):
        raise KeyboardInterrupt
    for signum in previous:
        signal.signal(signum, stop)
    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=.1)
            node.raise_if_failed()
    except (KeyboardInterrupt, SystemExit):
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        # Tracking finishes its current image and sends an ordered stream end.
        # Continue ROS reception until its last keyframe has reached our queue.
        deadline = time.monotonic() + 5.
        while rclpy.ok() and not node.input_complete() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=.05)
            node.raise_if_failed()
    finally:
        try:
            executor.shutdown()
            node.destroy_node()
        finally:
            rclpy.try_shutdown()
            for signum, handler in previous.items():
                signal.signal(signum, handler)


if __name__ == "__main__":
    main()
