import os
import signal
import json
import uuid

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import rclpy
import yaml
import numpy as np
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, PointCloud2, PointField
from std_msgs.msg import String
from normalflow.utils import height2pointcloud

from gelslam.core.inertial import ImuSample, gyro_to_rad_per_sec
from gelslam.core.recorded_source_clock import RecordedSourceClock
from gelslam.core.tracker import Tracker, TrackingMode
from gelslam.core.backend_feedback import FeedbackReceiver
from gelslam.core.slam_packets import ArrayBridge, FramePacket, KeyFramePacket, to_ros
from gelslam.core.gt_state_diagnostic import gt_state_test_needs_gt, resolve_gt_state_test
from gelslam.core.state_velocity_evaluation import StateVelocityEvaluation, capture_state_velocity
from gelslam.core.trajectory_evaluation import trajectory_output_dir
from gelslam.utils import Logger
from gelslam.visualization.normal_correspondences import (
    DEFAULT_CORRESPONDENCE_MAX_LINES,
    normal_map_to_rgb8,
    render_normal_correspondences,
)
from gelslam_msgs.msg import FrameMsg, KeyFrameMsg, BackendFeedbackMsg, FramePoseMsg


"""
Single thread:
1. Track thread:
    - Image callback: Tracks the frame and publishes keyframes/frames.
"""


class TrackerNode(Node):
    """
    ROS2 node for real-time tracking.
    """

    def __init__(self, node_name="tracker_node"):
        super().__init__(node_name)
        self.bridge = CvBridge()
        # Load configuration path
        self.declare_parameter("config_path", "")
        config_path = (
            self.get_parameter("config_path").get_parameter_value().string_value
        )
        config_path = os.path.abspath(os.path.expanduser(config_path))
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.config = config
        self.debug_visualization_enabled = config["gelslam_config"]["track_config"].get(
            "debug_visualization_enabled", True
        )
        if not isinstance(self.debug_visualization_enabled, bool):
            raise ValueError("gelslam_config.track_config.debug_visualization_enabled must be a boolean")
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
        self.logger.info("Frontend debug visualization: %s" % (
            "enabled" if self.debug_visualization_enabled else "disabled"
        ))

        # Initialize tracker
        self.tracker = Tracker(
            calib_model_path, config, self.skip_background_check, logger=self.logger
        )
        self.session_id = uuid.uuid4().hex
        self.tracker._ros_session_id = self.session_id
        self.tracker.bridge = ArrayBridge()
        self._last_published_kid = -1
        self.declare_parameter("backend_feedback", True)
        self.backend_feedback_enabled = bool(self.get_parameter("backend_feedback").value
            and self.tracker.frontend_visual_inertial is not None)
        self.feedback_receiver = None
        if self.backend_feedback_enabled:
            self.feedback_receiver = FeedbackReceiver(self.session_id, self.tracker.gelsight_T_imu)
            self.tracker._map_snapshot_provider = self.feedback_receiver.snapshot
            self.create_subscription(BackendFeedbackMsg, "backend_feedback",
                                     self.backend_feedback_callback, 1)
        self.stream_end_pub = self.create_publisher(String, "backend_stream_end", 10)
        self.logger.info("ROS process: tracking; backend feedback %s; session %s" % (
            "enabled" if self.backend_feedback_enabled else "disabled", self.session_id))
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, 'tracker_process_config_supplied.yaml'), 'w') as handle:
            yaml.safe_dump(dict(config=config, backend_feedback=self.backend_feedback_enabled,
                                pid=os.getpid(), session_id=self.session_id), handle, sort_keys=False)
        self.declare_parameter("sequence_name", "")
        self.state_velocity_evaluation = None
        velocity_settings = dict(config.get("trajectory_evaluation", {}).get("state_velocity", {}))
        if velocity_settings.get("enabled", False):
            if self.tracker.frontend_visual_inertial is None:
                self.get_logger().warning("State.velocity evaluation requires the VI frontend; inactive")
            else:
                path = os.path.expanduser(velocity_settings["truth_path"])
                if not os.path.isabs(path):
                    path = os.path.join(os.path.dirname(config_path), path)
                velocity_settings["truth_path"] = path
                output_dir = trajectory_output_dir(
                    self.save_dir, self.get_parameter("sequence_name").value, data_dir
                )
                self.state_velocity_evaluation = StateVelocityEvaluation(velocity_settings, output_dir)
                self.create_subscription(
                    PoseStamped, velocity_settings.get("ground_truth_pose_topic", "/ground_truth/imu_pose"),
                    self._state_velocity_gt_callback,
                    QoSProfile(depth=1000, reliability=ReliabilityPolicy.BEST_EFFORT),
                )
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, "state_velocity_config_supplied.yaml"), "w") as handle:
                    yaml.safe_dump(config, handle, sort_keys=False)
                self.get_logger().info("Recording actual State.velocity; native IMU-origin GT, evaluation only: %s" % output_dir)
        self.recorded_source_clock = None
        self._unmatched_source_images = 0
        self._unmatched_source_imus = 0
        if self.tracker.imu_enabled and self.tracker.timestamp_source == "recorded_source":
            bag_path = config["imu_config"].get("recorded_source_bag_path")
            if not bag_path:
                raise ValueError("imu_config.recorded_source_bag_path is required")
            if not os.path.isabs(bag_path):
                bag_path = os.path.join(os.path.dirname(config_path), bag_path)
            if not os.path.isfile(bag_path):
                raise ValueError("recorded source bag database does not exist: %s" % bag_path)
            self.recorded_source_clock = RecordedSourceClock(
                bag_path,
                image_topic=config["device_config"].get("image_topic", "image"),
                camera_timing_topic=config["imu_config"].get(
                    "camera_timing_topic", "/gelsight/timing"
                ),
                imu_topic=config["imu_config"].get("topic", "/imu/data"),
                imu_device_time_topic=config["imu_config"].get(
                    "imu_device_time_topic", "/imu/device_time_ms"
                ),
            )
            self.get_logger().info(
                "Using recorded source clocks from %s: %d images, %d IMU samples; "
                "camera/IMU host-fit median residuals %.3f/%.3f ms; "
                "image_to_imu_time_offset_s=%.6f"
                % (
                    bag_path, self.recorded_source_clock.image_count,
                    self.recorded_source_clock.imu_count,
                    self.recorded_source_clock.camera_fit[3] * 1000,
                    self.recorded_source_clock.imu_fit[3] * 1000,
                    self.tracker.image_to_imu_time_offset_s,
                )
            )

        # Initialize subscribers
        # Normal-flow image processing can take longer than one IMU period.
        # Keep IMU ingestion independent of image tracking, but serialize IMU
        # callbacks themselves: the subscription is timestamp ordered and the
        # buffer is a single ordered producer.
        self.image_callback_group = MutuallyExclusiveCallbackGroup()
        self.imu_callback_group = MutuallyExclusiveCallbackGroup()
        image_queue_size = int(
            config.get("gelslam_config", {})
            .get("track_config", {})
            .get("image_queue_size", 1)
        )
        if image_queue_size < 1:
            raise ValueError("gelslam_config.track_config.image_queue_size must be positive")
        image_topic = config["device_config"].get("image_topic", "image")
        self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            image_queue_size,
            callback_group=self.image_callback_group,
        )
        self.get_logger().info(
            "Subscribing to tactile images from %s (queue depth: %d)"
            % (image_topic, image_queue_size)
        )
        self.imu_sub = None
        self.ground_truth_gel_pose_sub = None
        self.ground_truth_imu_pose_sub = None
        self._gt_imu_cleanup_timer = None
        self._imu_dropped_samples = 0
        self._imu_invalid_gyro_samples = 0
        if self.tracker.imu_enabled:
            # Use resolved initialization settings. Static mode suppresses GT
            # initialization even when a simulation preset still contains it.
            imu_config = self.tracker.imu_config
            self.imu_frame_id = imu_config.get("frame_id", "")
            self.gyro_input_unit = imu_config.get("gyro_input_unit", "rad_s")
            self.gyro_max_rate_rad_s = imu_config.get("gyro_max_rate_rad_s", 5.0)
            # Validate the declared unit and guard before accepting any IMU
            # sample, rather than allowing an invalid config to fail later in
            # the integration path.
            gyro_to_rad_per_sec(
                (0.0, 0.0, 0.0), self.gyro_input_unit, self.gyro_max_rate_rad_s
            )
            self.imu_sub = self.create_subscription(
                Imu,
                imu_config.get("topic", "/imu/data"),
                self.imu_callback,
                1000,
                callback_group=self.imu_callback_group,
            )
            self.get_logger().info(
                "Buffering IMU samples from %s (frame_id=%s)"
                % (imu_config.get("topic", "/imu/data"), self.imu_frame_id)
            )
            self.get_logger().info(
                "IMU gyro unit=%s; rejecting rates above %s rad/s after conversion"
                % (self.gyro_input_unit, self.gyro_max_rate_rad_s)
            )
            if (
                imu_config.get("gravity_initialization") in {
                    "ground_truth_initial_pose", "ground_truth_gel_pose"
                }
                or imu_config.get("frontend_gravity_initialization")
                == "ground_truth_initial_pose"
                or imu_config.get("frontend_first_trial_velocity_from_gt", False)
            ):
                ground_truth_topic = imu_config.get(
                    "ground_truth_gel_pose_topic", "/ground_truth/gel_pose"
                )
                self.ground_truth_gel_pose_sub = self.create_subscription(
                    PoseStamped,
                    ground_truth_topic,
                    self.ground_truth_gel_pose_callback,
                    1000,
                )
                self.get_logger().warning(
                    "Using simulation GT GelSight poses for per-trial gravity "
                    "and trial-axis initialization: %s"
                    % ground_truth_topic
                )
            diagnostic_gt = gt_state_test_needs_gt(resolve_gt_state_test(imu_config))
            if (imu_config.get("frontend_first_trial_velocity_from_gt", False) or diagnostic_gt):
                ground_truth_imu_topic = imu_config.get(
                    "ground_truth_imu_pose_topic", "/ground_truth/imu_pose"
                )
                self.ground_truth_imu_pose_sub = self.create_subscription(
                    PoseStamped, ground_truth_imu_topic,
                    self.ground_truth_imu_pose_callback, 1000,
                )
                self.get_logger().warning(
                    "Using simulation GT IMU poses for %s: %s"
                    % ("per-frame previous-state diagnostic"
                       if diagnostic_gt
                       else "first-trial velocity initialization", ground_truth_imu_topic)
                )
                if not diagnostic_gt:
                    # Run in the GT subscriptions' default mutually-exclusive
                    # group, so destruction cannot race a GT callback.
                    self._gt_imu_cleanup_timer = self.create_timer(
                        0.1, self._release_initialization_imu_gt,
                        clock=Clock(clock_type=ClockType.STEADY_TIME),
                    )

        # Initialize publishers
        self.keyframe_pub = self.create_publisher(KeyFrameMsg, "keyframe", 1000)
        self.frame_pub = self.create_publisher(FrameMsg, "frame", 10)
        self.frame_pose_pub = self.create_publisher(FramePoseMsg, "frame_pose", 1000)
        self.contact_surface_pub = self.create_publisher(
            PointCloud2, "/gelslam/current_contact_local", 1
        )
        self.normal_map_pub = self.create_publisher(
            Image, "/gelslam/rviz/normal_map", 1
        )
        self.normal_correspondences_pub = self.create_publisher(
            Image, "/gelslam/rviz/normal_correspondences", 1
        )
        self.correspondence_max_lines = int(
            config["gelslam_config"]["track_config"].get(
                "correspondence_max_lines", DEFAULT_CORRESPONDENCE_MAX_LINES
            )
        )
        if self.correspondence_max_lines < 0:
            raise ValueError("track_config.correspondence_max_lines must be nonnegative")
        self.frontend_gravity_pub = None
        self._frontend_gravity_published = False
        if self.tracker.frontend_visual_inertial is not None:
            self.frontend_gravity_pub = self.create_publisher(
                Vector3Stamped,
                config.get("imu_config", {}).get(
                    "frontend_gravity_topic", "/gelslam/frontend/gravity"
                ),
                QoSProfile(
                    depth=1,
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                ),
            )

    def image_callback(self, msg):
        """
        Callback function for the image subscriber.
        Tracks the frame and publishes keyframes/frames.

        :param msg: Image; The image message.
        """
        if self.recorded_source_clock is not None:
            source_time = self.recorded_source_clock.image_timestamp(msg.header)
            if source_time is None:
                self._unmatched_source_images += 1
                if self._unmatched_source_images == 1:
                    self.get_logger().error(
                        "Dropping image without matching recorded camera timing"
                    )
                return
            self.tracker.set_recorded_image_timestamp(msg.header, source_time)
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        ret, track_result = self.tracker.track(
            image, header=msg.header, log_prefix="Track Thread"
        )
        self._publish_frontend_gravity(msg.header)
        if ret:
            frame_msg, keyframe_msgs = track_result
            self._publish_tracking_result(frame_msg, keyframe_msgs)
            if frame_msg is not None:
                evaluation = getattr(self, "state_velocity_evaluation", None)
                if evaluation is not None:
                    evaluation.add_state(capture_state_velocity(self.tracker, frame_msg))
                self._log_tracking_pose(frame_msg)
        observation = self.tracker.latest_observation_frame
        if self.debug_visualization_enabled and observation is not None:
            normal_image = self.bridge.cv2_to_imgmsg(
                normal_map_to_rgb8(observation.N), encoding="rgb8"
            )
            normal_image.header = msg.header
            self.normal_map_pub.publish(normal_image)
            self.contact_surface_pub.publish(
                self._contact_surface_cloud(observation, msg.header.stamp)
            )
            self._publish_normal_correspondences(observation, msg.header)

    def _publish_tracking_result(self, frame_msg, keyframe_msgs):
        for keyframe_msg in keyframe_msgs:
            message = to_ros(keyframe_msg, self.bridge) if isinstance(keyframe_msg, KeyFramePacket) else keyframe_msg
            self.keyframe_pub.publish(message)
            self._last_published_kid = message.kid
        if frame_msg is not None:
            pose_message = FramePoseMsg()
            pose_message.header = frame_msg.header
            pose_message.fid, pose_message.ref_kid = frame_msg.fid, frame_msg.ref_kid
            pose_message.ref_t_curr = np.asarray(frame_msg.ref_t_curr).ravel().tolist()
            self.frame_pose_pub.publish(pose_message)
            message = to_ros(frame_msg, self.bridge) if isinstance(frame_msg, FramePacket) else frame_msg
            self.frame_pub.publish(message)

    def backend_feedback_callback(self, message):
        try:
            self.feedback_receiver.receive(message)
        except ValueError as error:
            self.get_logger().warning("Ignoring invalid backend feedback: %s" % error)

    def _publish_stream_end(self):
        if getattr(self, 'stream_end_pub', None) is None:
            return
        message = String()
        message.data = json.dumps(dict(session_id=self.session_id, last_kid=self._last_published_kid))
        from rclpy.duration import Duration
        self.keyframe_pub.wait_for_all_acked(Duration(seconds=2.))
        self.frame_pose_pub.wait_for_all_acked(Duration(seconds=2.))
        self.stream_end_pub.publish(message)
        self.stream_end_pub.wait_for_all_acked(Duration(seconds=1.))

    def _state_velocity_gt_callback(self, message):
        from scipy.spatial.transform import Rotation
        stamp = message.header.stamp
        timestamp = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        p, q = message.pose.position, message.pose.orientation
        quaternion = np.asarray((q.x, q.y, q.z, q.w))
        if not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) <= 1e-12:
            return
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
        pose[:3, 3] = (p.x, p.y, p.z)
        self.state_velocity_evaluation.add_gt_pose(timestamp, pose)

    def _publish_normal_correspondences(self, observation, header):
        """Publish the current image's compared normals, including rejected solves."""
        snapshot = self.tracker.latest_normal_correspondences
        if snapshot is None:
            reference = self.tracker.ref_frame
            reference_stamp = self.tracker._header_timestamp(self.tracker.ref_header)
            current_stamp = self.tracker._header_timestamp(header)
            reference_xy = current_xy = np.empty((0, 2))
            status = ("Reference initialization: no comparison"
                      if observation.is_contacted else "No contact: no comparison")
        else:
            reference = snapshot["reference"]
            reference_stamp = snapshot["reference_stamp"]
            current_stamp = snapshot["current_stamp"]
            reference_xy, current_xy = snapshot["reference_xy"], snapshot["current_xy"]
            status = snapshot["status"]
        def label(name, stamp):
            return name if stamp is None else f"{name} t={stamp:.6f}"
        canvas = render_normal_correspondences(
            reference, observation, reference_xy, current_xy,
            label("Reference normal", reference_stamp),
            label("Current normal", current_stamp), status,
            max_lines=self.correspondence_max_lines,
        )
        message = self.bridge.cv2_to_imgmsg(canvas, encoding="rgb8")
        message.header = header
        self.normal_correspondences_pub.publish(message)

    def _contact_surface_cloud(self, frame, stamp):
        """Publish the current contact pixels, independent of pose acceptance."""
        points = np.ascontiguousarray(
            height2pointcloud(frame.H, frame.C, self.tracker.ppmm),
            dtype=np.float32,
        )
        cloud = PointCloud2()
        cloud.header.stamp = stamp
        cloud.header.frame_id = "gelsight_estimate"
        cloud.height = 1
        cloud.width = len(points)
        cloud.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.point_step = 12
        cloud.row_step = 12 * cloud.width
        cloud.is_dense = True
        cloud.data = points.tobytes()
        return cloud

    def _publish_frontend_gravity(self, header):
        """Publish the first trial's frontend gravity after VI initialization."""
        if self.frontend_gravity_pub is None or self._frontend_gravity_published:
            return
        gravity = self.tracker.frontend_gravity_for_backend()
        if gravity is None:
            return
        message = Vector3Stamped()
        message.header.stamp = header.stamp
        message.header.frame_id = "gelslam_first_trial"
        message.vector.x, message.vector.y, message.vector.z = map(float, gravity)
        self.frontend_gravity_pub.publish(message)
        self._frontend_gravity_published = True
        self.get_logger().info(
            "Published first-trial gravity in tracker trial coordinates: %s"
            % np.array2string(gravity, precision=6)
        )

    def _log_tracking_pose(self, frame_msg):
        """Log the relative pose for each successful tactile tracking frame."""
        if self.tracker.tracking_mode is not TrackingMode.CONTACT_TRACKING:
            return
        ref_T_curr = np.asarray(frame_msg.ref_t_curr, dtype=np.float64).reshape(4, 4)
        stamp = frame_msg.header.stamp
        self.get_logger().info(
            "Tracking success: frame ID: %d, reference keyframe ID: %d, "
            "stamp: %d.%09d, ref_T_curr (current frame -> reference keyframe):\n%s"
            % (
                frame_msg.fid,
                frame_msg.ref_kid,
                stamp.sec,
                stamp.nanosec,
                np.array2string(
                    ref_T_curr,
                    precision=7,
                    suppress_small=False,
                    max_line_width=120,
                ),
            )
        )

    def imu_callback(self, msg):
        """Store calibrated gyro/accel measurements for future preintegration."""
        stamp = msg.header.stamp
        timestamp = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if self.recorded_source_clock is not None:
            timestamp = self.recorded_source_clock.imu_timestamp(msg)
            if timestamp is None:
                self._unmatched_source_imus += 1
                if self._unmatched_source_imus == 1:
                    self.get_logger().error(
                        "Dropping IMU without matching recorded device time"
                    )
                return
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
        try:
            angular_velocity = gyro_to_rad_per_sec(
                (
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z,
                ),
                self.gyro_input_unit,
                self.gyro_max_rate_rad_s,
            )
        except ValueError as error:
            self._imu_dropped_samples += 1
            self._imu_invalid_gyro_samples += 1
            if self._imu_invalid_gyro_samples == 1:
                self.get_logger().error("Dropping IMU sample: %s" % error)
            return
        sample = ImuSample(
            timestamp=timestamp,
            acceleration=(
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ),
            angular_velocity=angular_velocity,
        )
        if not self.tracker.add_imu_sample(sample):
            self._imu_dropped_samples += 1
            if self._imu_dropped_samples == 1:
                self.get_logger().warning(
                    "Dropping IMU: duplicate/out-of-order timestamp or no simulation truth support"
                )

    def ground_truth_gel_pose_callback(self, msg):
        """Buffer GT GelSight pose for explicit simulation initialization."""
        timestamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        quaternion = np.array(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
            dtype=np.float64,
        )
        norm = np.linalg.norm(quaternion)
        if timestamp <= 0.0 or not np.isfinite(norm) or norm <= 1e-12:
            self.get_logger().warning("Dropping invalid ground-truth GelSight pose")
            return
        x, y, z, w = quaternion / norm
        world_R_gel = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        self.tracker.add_recent_ground_truth_gel_rotation(timestamp, world_R_gel)

    def _release_initialization_imu_gt(self):
        """Stop initialization-only GT after the first trial state is ready."""
        tracker = self.tracker
        if gt_state_test_needs_gt(resolve_gt_state_test(tracker.imu_config)):
            return
        if (self.ground_truth_imu_pose_sub is None
                or getattr(tracker, "_frontend_vi_state", None) is None
                or getattr(tracker, "_pending_frontend_initialization", None) is not None
                or getattr(tracker, "_pending_trial_contact", None) is not None
                or getattr(tracker, "_trial_index", -1) < 0):
            return
        if self.destroy_subscription(self.ground_truth_imu_pose_sub):
            self.ground_truth_imu_pose_sub = None
            tracker._ground_truth_imu_positions.clear()
            self._gt_imu_cleanup_timer.cancel()
            self.get_logger().info(
                "First-trial initialization complete; stopped GT IMU subscription"
            )

    def ground_truth_imu_pose_callback(self, msg):
        """Use the GT IMU position directly; gravity still uses GelSight attitude."""
        timestamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        position = msg.pose.position
        if gt_state_test_needs_gt(resolve_gt_state_test(getattr(self.tracker, "imu_config", {}))):
            q = msg.pose.orientation
            quaternion = np.array([q.x, q.y, q.z, q.w])
            norm = np.linalg.norm(quaternion)
            if timestamp <= 0 or not np.isfinite(norm) or norm <= 1e-12:
                self.get_logger().warning("Dropping invalid GT IMU diagnostic pose")
                return
            from scipy.spatial.transform import Rotation
            if not self.tracker.add_ground_truth_imu_pose(
                timestamp, Rotation.from_quat(quaternion / norm).as_matrix(),
                (position.x, position.y, position.z),
            ):
                self.get_logger().warning("Dropping invalid GT IMU diagnostic pose")
            return
        if timestamp <= 0.0 or not self.tracker.add_ground_truth_imu_position(
            timestamp, (position.x, position.y, position.z),
        ):
            self.get_logger().warning("Dropping invalid GT IMU position")

    def destroy_node(self):
        """
        Clean up node and save tracker state.
        """
        self._publish_stream_end()
        self.tracker.log_frontend_tracking_summary()
        if getattr(self, 'session_id', None) is not None:
            with open(os.path.join(self.save_dir, 'tracker_process_summary.json'), 'w') as handle:
                json.dump(dict(pid=os.getpid(), session_id=self.session_id,
                    last_published_kid=self._last_published_kid,
                    backend_feedback_enabled=self.backend_feedback_enabled,
                    tracking_counts=dict(self.tracker._frontend_tracking_counts)), handle, indent=2)
        if self.state_velocity_evaluation is not None:
            try:
                metrics = self.state_velocity_evaluation.save()
                self.get_logger().info("State.velocity evaluation: %s; %d matched states"
                                       % (metrics["status"], metrics.get("matched_states", 0)))
            except (OSError, ValueError, ImportError) as error:
                self.get_logger().error("State.velocity evaluation failed: %s" % error)
        # Clean up node
        super().destroy_node()
        # Save tracker state
        if self.save_gelslam_state:
            self.tracker.save(self.save_dir)


def main(args=None):
    """
    Main function to initialize and run the node.
    """
    # Keep ROS logging available until the final summary has been written.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = TrackerNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def stop_tracking(_signum, _frame):
        raise KeyboardInterrupt

    for signum in previous_handlers:
        signal.signal(signum, stop_tracking)
    try:
        # A bounded wait lets Python handle stop signals even with no inputs.
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
    except (KeyboardInterrupt, SystemExit):
        # ros2 launch can send a second SIGINT while the node is saving state.
        # Do not let that interrupt the cleanup and output write.
        for signum in previous_handlers:
            signal.signal(signum, signal.SIG_IGN)
    finally:
        # Finish any image callback before taking the final counter snapshot.
        try:
            try:
                executor.shutdown()
            finally:
                node.destroy_node()
        finally:
            rclpy.try_shutdown()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


if __name__ == "__main__":
    main()
