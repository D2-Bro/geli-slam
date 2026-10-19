import os
import signal

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
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from gelslam.core.coverage_graph import CoverageGraph, create_visible_coverage_graph_msg
from gelslam.core.keyframe import KeyFrameDB, compute_adjusted_pointcloud
from gelslam.core.parent_groups_info import ParentGroupsInfo
from gelslam.core.gtsam_pose_graph import PoseGraph, PoseGraphSolutions
from gelslam.core.trajectory_evaluation import (
    save_aligned_trajectory,
    trajectory_output_dir,
)
from gelslam.utils import Logger, pointcloud2mesh
from gelslam_msgs.msg import FrameMsg, KeyFrameMsg, VisibleCoverageGraphMsg

"""
Two threads:
1. K thread (Keyframe callback):
    - Adds keyframe to the KeyFrameDB.
2. L thread (Loop closure):
    - Detects loops.
    - Performs Loop Closure: updates parent groups, solves poses, updates coverage graph.
    - Publishes the visible coverage graph message.
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

        # KeyframeDB (keyframe database) is never modified except appending.
        # K thread writes (appending), L thread reads
        self.add_keyframe_lock = threading.Lock()
        self.keyframedb = KeyFrameDB(
            ppmm=self.config["device_config"]["ppmm"], logger=self.logger,
            gelsight_T_imu=self.config.get("imu_config", {}).get("gelsight_T_imu"),
            imu_max_sample_gap_s=self.config["gelslam_config"]["pose_graph_config"].get("imu_max_sample_gap_s", 0.05),
        )

        # Parent groups, pose graph, pose graph solutions, and coverage graph are updated in L thread.
        self.parent_groups_info = ParentGroupsInfo()
        self.pose_graph = PoseGraph(self.config)
        if self.backend_optimization_enabled:
            if not self.pose_graph.mounting_calibrated:
                raise ValueError("15D backend requires calibrated imu_config.gelsight_T_imu; use the calibrated real/simulation configuration")
            self.logger.info("Backend: 15D keyframe states; direct normal/height, IMU and bias factors; no relative-pose factors")
        if (
            self.backend_optimization_enabled
            and self.pose_graph.pose_graph_config.get("loop_closure_only_optimization", False)
        ):
            self.logger.info("Backend optimization will run only after a loop edge is added")
        self.pose_graph_solutions = PoseGraphSolutions()
        self.processed_keyframe_count = 0
        self.coverage_graph = CoverageGraph(self.config)
        # This is independent of mesh rendering.  RViz can therefore show
        # backend keyframe poses when Open3D rendering is disabled.
        self.gelsight_trajectory_pub = self.create_publisher(
            Path, "/gelslam/rviz/gelsight_trajectory", 1
        )
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
            10,
        )
        if self.trajectory_evaluation_enabled:
            self.create_subscription(FrameMsg, "frame", self.trajectory_frame_callback, 1000)
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
        self.loop_closure_thread = threading.Thread(target=self.loop_closure)
        self.loop_closure_thread.start()

    def keyframe_callback(self, keyframe_msg):
        """
        Adds received keyframe to KeyFrameDB (K thread).

        :param keyframe_msg: KeyFrameMsg; The received keyframe message.
        """
        if self.backend_optimization_enabled and not keyframe_msg.navigation_valid:
            self.logger.error("Rejecting keyframe %d: 15D backend needs a navigation seed at its image time" % keyframe_msg.kid)
            return
        with self.add_keyframe_lock:
            self.keyframedb.insert(keyframe_msg, log_prefix="Keyframe Insert Thread")

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
        """
        Loop closure thread (L thread).
        Detects loops, merges parent groups, solves poses, and updates coverage graph.

        Steps:
        1. Detect loops, add loop closing factors to the pose graph. (Expensive)
        2. Perform loop closure updates. (Faster)
        """
        while self.running:
            # Get the targeted number of keyframes
            with self.add_keyframe_lock:
                targeted_size = self.keyframedb.size()
            # Skip if no new keyframes since last loop closure
            updated_size = self.processed_keyframe_count
            if targeted_size == 0 or targeted_size == updated_size:
                time.sleep(0.01)
                continue

            self.logger.info(
                "Loop Closure Thread -- Begin: current keyframe database size: %d"
                % targeted_size
            )
            # Tracker estimates initialize states; observation factors use maps/IMU.
            self.pose_graph.add_keyframes(
                self.keyframedb, updated_size, targeted_size,
                build_factors=self.backend_optimization_enabled,
            )
            for reference, target, reason in self.pose_graph.tactile_skipped:
                self.logger.warning("No tactile factor for keyframes %d -> %d: %s" % (reference, target, reason))
            self.pose_graph.tactile_skipped.clear()

            # Always preserve the keyframe graph and its tracker odometry
            # chain. With optimization disabled, do not create loop factors or
            # solve the graph; expose the raw chain for trajectory evaluation.
            if not self.backend_optimization_enabled:
                self.parent_groups_info.update_wrt_new_keyframes(
                    self.keyframedb, updated_size, targeted_size
                )
                self.pose_graph_solutions = self.pose_graph.raw_solutions()
                self.processed_keyframe_count = targeted_size
                self.logger.info(
                    "Loop Closure Thread -- Added keyframes without backend optimization: %d"
                    % targeted_size
                )
                continue

            # Detect loops
            tar_kidx = targeted_size - 1
            matched_kidxs = self.pose_graph.detect_and_add_loops(
                self.keyframedb, tar_kidx, self.coverage_graph
            )
            self.logger.info(
                "Loop Closure Thread -- Loop detected: keyframe index %d matched with keyframe indices %s"
                % (tar_kidx, str(matched_kidxs))
            )

            # Update parent groups due to newly introduced keyframes since last loop closure
            self.parent_groups_info.update_wrt_new_keyframes(
                self.keyframedb, updated_size, targeted_size
            )
            # Update parent groups due to newly detected loops
            (
                original_member_kidxs,
                new_member_kidxs,
                other_kidxs,
                removed_trial_groups,
            ) = self.parent_groups_info.update_wrt_loop_closure(
                self.keyframedb,
                tar_kidx,
                matched_kidxs,
                updated_size,
            )
            # Remove prior factors from pose graph due to trial fusion
            self.pose_graph.remove_prior_factors(removed_trial_groups)
            # A new loop constrains old and new states, so it needs a full
            # solve. Without a loop, the graph either keeps tracker odometry
            # or runs a local solve, according to the configuration.
            self.pose_graph_solutions = self.pose_graph.solve(
                global_optimization=bool(matched_kidxs)
            )
            self.processed_keyframe_count = targeted_size
            self.publish_gelsight_keyframe_trajectory()

            # Add new coverage node to coverage graph that represents new keyframes
            self.coverage_graph.add_new_coverage_nodes(updated_size, targeted_size)
            # Update the coverage graph based on the new keyframes from other trials
            self.coverage_graph.update_wrt_new_keyframes(
                other_kidxs,
                self.keyframedb,
                self.pose_graph_solutions,
                self.parent_groups_info,
            )
            # Update the coverage graph based on keyframes in the merged trials
            self.coverage_graph.update_wrt_loop_closure(
                original_member_kidxs,
                new_member_kidxs,
                self.keyframedb,
                self.pose_graph_solutions,
            )
            # Publish the visible coverage graph message for rendering
            if self.rendering:
                self.visible_coverage_graph_pub.publish(
                    create_visible_coverage_graph_msg(
                        tar_kidx,
                        self.keyframedb,
                        self.pose_graph_solutions,
                        self.parent_groups_info,
                        self.coverage_graph,
                    )
                )

            # Log the connectability information
            n_disjoint_trials = np.unique(self.parent_groups_info.parent_groups).size

            self.logger.info(
                "Loop Closure Thread -- End: disjoint trials: %d, largest trial keyframe portions: %d/%d"
                % (
                    np.unique(self.parent_groups_info.parent_groups).size,
                    np.max(self.parent_groups_info.parent_group_sizes),
                    np.sum(self.parent_groups_info.parent_group_sizes),
                )
            )

    def merge_and_save_mesh(self):
        """
        Find the keyframes from the largest connected group of keyframes.
        Merging the meshes and save the reconstructed mesh.
        """
        updated_size = self.pose_graph_solutions.size()
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
        self.running = False
        super().destroy_node()
        self.loop_closure_thread.join()

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


def main(args=None):
    """
    Main function to initialize and run the node.
    """
    rclpy.init(args=args)
    node = PoseGraphNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, SystemExit):
        # ros2 launch can send a second SIGINT while the node writes the final
        # trajectory evaluation.  Ignore it so cleanup can complete.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    finally:
        node.destroy_node()
        executor.shutdown()


if __name__ == "__main__":
    main()
