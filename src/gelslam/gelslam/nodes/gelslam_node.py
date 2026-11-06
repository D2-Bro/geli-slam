"""ROS adapter for the single-process GelSLAM system."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import json
import signal
import threading
import yaml
import rclpy
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from gelslam_msgs.msg import VisibleCoverageGraphMsg
from gelslam.nodes.tracker_node import TrackerNode
from gelslam.nodes.tracker_executor import TrackerExecutor
from gelslam.nodes.pose_graph_node import PoseGraphNode
from gelslam.core.local_mapping import LocalMapping, LoopSearch
from gelslam.core.slam_system import SlamSystem
from gelslam.core.slam_packets import ArrayBridge, to_ros
from gelslam.core.coverage_graph import create_visible_coverage_graph_msg
from gelslam.core.trajectory_evaluation import trajectory_output_dir


class GelSlamNode(TrackerNode):
    def __init__(self):
        super().__init__(node_name='gelslam_node')
        self.declare_parameter('backend_optimization', True)
        self.declare_parameter('rendering', True)
        self.declare_parameter('save_gelslam_states', False)
        self.backend_optimization_enabled = bool(self.get_parameter('backend_optimization').value)
        self.rendering = bool(self.get_parameter('rendering').value)
        self.save_gelslam_states = bool(self.get_parameter('save_gelslam_states').value)
        self.save_gelslam_state = self.save_gelslam_state or self.save_gelslam_states
        self.backend = LocalMapping(self.config, self.logger, self.backend_optimization_enabled)
        self.keyframedb = self.backend.db
        self.pose_graph = self.backend.graph
        self.parent_groups_info = self.backend.groups
        self.coverage_graph = self.backend.coverage
        self.pose_graph_solutions = self.backend.solutions
        self.add_keyframe_lock = threading.Lock()  # Publication/finalization only.
        self.trajectory_lock = threading.Lock()
        self._trajectory_frames, self._trajectory_ground_truth = [], []
        # Same IMU-only frame selection as pose_graph_node (the tracker checks the type).
        self.keep_imu_only_frames = self.config.get('gelslam_config', {}).get(
            'track_config', {}).get('keep_imu_only_frames', False)
        self.trajectory_evaluation_config = self.config.get('trajectory_evaluation', {})
        self.trajectory_evaluation_enabled = bool(self.trajectory_evaluation_config.get('enabled', False))
        self.trajectory_output_dir = trajectory_output_dir(
            self.save_dir, self.get_parameter('sequence_name').value,
            self.get_parameter('data_dir').value)
        self.gelsight_trajectory_pub = self.create_publisher(Path, '/gelslam/rviz/gelsight_trajectory', 1)
        if self.rendering:
            self.visible_coverage_graph_pub = self.create_publisher(VisibleCoverageGraphMsg, 'visible_coverage_graph', 1)
        if self.trajectory_evaluation_enabled:
            self.create_subscription(PoseStamped, self.trajectory_evaluation_config.get(
                'ground_truth_pose_topic', '/ground_truth/gel_pose'),
                self.trajectory_ground_truth_callback, 1000)
        self.tracker.bridge = ArrayBridge()
        self.system = SlamSystem(
            self._track_image, self.backend,
            LoopSearch(self.config, self.logger) if self.backend_optimization_enabled else None,
            self._on_solution,
            max_pending_inputs=int(self.config.get('slam_system', {}).get('max_pending_inputs', 100000)))
        self.tracker._map_snapshot_provider = (self.system.shared_map.snapshot_for_tracking
                                              if self.backend_optimization_enabled else None)
        self._closed = False
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, 'slam_config_supplied.yaml'), 'w') as handle:
            yaml.safe_dump(dict(config=self.config, backend_optimization=self.backend_optimization_enabled,
                                rendering=self.rendering), handle, sort_keys=False)
        self.system.start()
        self.get_logger().info(
            'Single process: tracking, Local BA and loop-search workers; shared 15D map'
            if self.backend_optimization_enabled else
            'Single process: tracking and raw keyframe storage; backend/loop/feedback disabled')

    def _track_image(self, message):
        TrackerNode.image_callback(self, message)

    def _resume_image_after_imu(self):
        # The ROS guard only enqueues work. The tracking worker owns pending
        # images, clock mappings, tracking, and publication in this adapter.
        return self.system.submit_input(TrackerNode._resume_image_after_imu, self)

    def _resume_source_imu(self):
        return self.system.submit_input(TrackerNode._drain_source_imus, self)

    def source_camera_timing_callback(self, message):
        return self.system.submit_input(TrackerNode.source_camera_timing_callback, self, message)

    def source_imu_timing_callback(self, message):
        return self.system.submit_input(TrackerNode.source_imu_timing_callback, self, message)

    # Executor callbacks only enqueue data. The tracking worker owns all tracker mutations.
    def image_callback(self, message):
        return self.system.submit_image(message)

    def imu_callback(self, message):
        return self.system.submit_input(TrackerNode.imu_callback, self, message)

    def ground_truth_gel_pose_callback(self, message):
        return self.system.submit_input(TrackerNode.ground_truth_gel_pose_callback, self, message)

    def ground_truth_imu_pose_callback(self, message):
        return self.system.submit_input(TrackerNode.ground_truth_imu_pose_callback, self, message)

    def _release_initialization_imu_gt(self):
        """After ROS dispatch stops, retire GT behind the tracking worker's queued inputs."""
        done, outcome = threading.Event(), {}

        def release():
            try:
                outcome['result'] = TrackerNode._release_initialization_imu_gt(self)
            except BaseException as error:
                outcome['error'] = error
            finally:
                done.set()

        if not self.system.submit_input(release):
            self.system.raise_if_failed()
            raise RuntimeError('Tracking worker stopped before initialization GT cleanup')
        while not done.wait(.1):
            self.system.raise_if_failed()
        if 'error' in outcome:
            raise outcome['error']
        return outcome['result']

    def _state_velocity_gt_callback(self, message):
        return self.system.submit_input(TrackerNode._state_velocity_gt_callback, self, message)

    def _publish_tracking_result(self, frame, keyframes):
        self.system.accept_keyframes(keyframes)
        # External consumers still receive the existing ROS schemas and order.
        for packet in keyframes:
            self.keyframe_pub.publish(to_ros(packet, self.bridge))
        if frame is not None:
            self.frame_pub.publish(to_ros(frame, self.bridge))
            self.trajectory_frame_callback(frame)

    def _on_solution(self, backend):
        self.keyframedb, self.pose_graph = backend.db, backend.graph
        self.parent_groups_info, self.coverage_graph = backend.groups, backend.coverage
        self.pose_graph_solutions = backend.solutions
        PoseGraphNode.publish_gelsight_keyframe_trajectory(self)
        if self.rendering and self.backend_optimization_enabled and backend.db.size():
            self.visible_coverage_graph_pub.publish(create_visible_coverage_graph_msg(
                backend.db.size() - 1, backend.db, backend.solutions,
                backend.groups, backend.coverage))

    _header_timestamp = staticmethod(PoseGraphNode._header_timestamp)
    _pose_from_message = staticmethod(PoseGraphNode._pose_from_message)
    trajectory_frame_callback = PoseGraphNode.trajectory_frame_callback
    trajectory_ground_truth_callback = PoseGraphNode.trajectory_ground_truth_callback

    def destroy_node(self):
        if self._closed:
            return
        self._closed = True
        error = None
        try:
            self.system.stop()
        except RuntimeError as failure:
            error = failure
            self.get_logger().error(str(failure))
        # All estimator workers are joined before evaluation and destructive saves.
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, 'slam_thread_metrics.json'), 'w') as handle:
            json.dump(self.system.metrics, handle, indent=2)
        if error is None:
            PoseGraphNode.save_final_trajectory_evaluation(self)
            if self.backend_optimization_enabled and self.backend.db.size():
                PoseGraphNode.merge_and_save_mesh(self)
            if self.save_gelslam_states or self.save_gelslam_state:
                self.backend.save(self.save_dir)
        else:
            with open(os.path.join(self.save_dir, 'slam_worker_failure.txt'), 'w') as handle:
                handle.write(str(error))
            self.save_gelslam_state = False
        result = super().destroy_node()
        if error is not None:
            raise error
        return result


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = GelSlamNode()
    executor = TrackerExecutor(num_threads=2)
    executor.add_node(node)
    handlers = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}

    def stop(_number, _frame):
        raise KeyboardInterrupt

    for number in handlers:
        signal.signal(number, stop)
    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=.1)
            node.system.raise_if_failed()
            if node._initialization_imu_gt_release_ready():
                executor.shutdown()
                node._release_initialization_imu_gt()
                executor = TrackerExecutor(num_threads=2)
                executor.add_node(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        for number in handlers:
            signal.signal(number, signal.SIG_IGN)
        try:
            try:
                executor.shutdown()
            finally:
                node.destroy_node()
        finally:
            rclpy.try_shutdown()
            for number, handler in handlers.items():
                signal.signal(number, handler)


if __name__ == '__main__':
    main()
