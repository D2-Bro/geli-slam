import os
import pickle
from bisect import bisect_left
from collections import Counter
from enum import Enum
from time import monotonic

import cv2
import numpy as np
import torch
from cv_bridge import CvBridge
from gs_sdk.gs_reconstruct import Reconstructor
from normalflow.registration import LoseTrackError
from normalflow.utils import Frame

from gelslam.core.frame import create_frame_msg, create_keyframe_msg
from gelslam.core.inertial import (
    ImuBuffer,
    ImuSample,
    InertialPropagator,
    InertialState,
    gravity_from_initial_world_R_gel,
    gravity_from_world_R_imu,
    rotation_aligning_gravity_to_negative_z,
)
from gelslam.core.numpy_vi import (
    DenseNormalResidual,
    NavigationState,
    normalflow_quality,
)
from gelslam.core.gtsam_vi import create_frontend_solver, ImuPreintegrationAccumulator
from gelslam.core.gt_state_diagnostic import (
    GroundTruthImuStates, gt_state_test_needs_gt, resolve_gt_state_test,
)
from gelslam.core.imu_navigation import gelsight_pose, imu_pose
from gelslam.core.marginal_prior import NavigationPrior
from gelslam.core.static_imu_initialization import (
    StaticImuInitializer, resolve_static_imu_config,
)
from gelslam.core.contact_velocity import ContactVelocityHistory
from gelslam.visualization.normal_correspondences import project_normal_correspondences


class TrackingMode(Enum):
    """Tracker lifecycle and published frame estimation modes."""

    CONTACT_TRACKING = "contact_tracking"
    NO_CONTACT_COASTING = "no_contact_coasting"
    IMU_COASTING = "imu_coasting"
    IMU_ONLY = "imu_only"
    IMU_PREDICTION = "imu_prediction"
    REACQUIRE = "reacquire"
    TACTILE_HOLD = "tactile_hold"


class QualityGateError(LoseTrackError):
    """A candidate failed CCS, finite-state, or IMU-consistency checks."""

    def __init__(self, trial_T_curr, vi_state=None, *, scr=None, ccs=None,
                 reason="ccs_below_threshold"):
        super().__init__("normal-flow SCR/CCS quality failed")
        self.trial_T_curr = np.asarray(trial_T_curr, dtype=np.float64).copy()
        self.vi_state = vi_state
        self.scr, self.ccs, self.reason = scr, ccs, reason


class Tracker:
    """
    Handles frame tracking.
    """

    def __init__(self, calib_model_path, config, skip_background_check, logger):
        """
        Initialize the Tracker.

        :param calib_model_path: str; The path to the calibration model.
        :param config: dict; The configuration.
        :param skip_background_check: bool; Whether to skip the background check step.
        :param logger: Logger; The logger object.
        """
        self.logger = logger
        self._frontend_tracking_counts = Counter()
        self.ppmm = config["device_config"]["ppmm"]
        self.torch_num_threads = config["device_config"].get("torch_num_threads", 4)
        self.contact_mask_config = config["contact_mask_config"]
        self.track_config = config["gelslam_config"]["track_config"]
        if not isinstance(self.track_config.get("keep_imu_only_frames", False), bool):
            raise ValueError("gelslam_config.track_config.keep_imu_only_frames must be a boolean")
        self._keyframe_interval_s()  # Validate the common selection condition.
        self._last_keyframe_timestamp = None
        self.imu_config = config.get("imu_config", {})
        self._contact_velocity_history = ContactVelocityHistory.from_config(self.imu_config)
        self._coast_velocity_seed = None
        self._coast_velocity_anchor_timestamp = None
        self.latest_coast_velocity_fit = None
        self._pending_trial_contact = None
        self._trial_gravity_rotation = None
        self.imu_enabled = bool(self.imu_config.get("enabled", False))
        self.timestamp_source = self.imu_config.get("timestamp_source", "header")
        if self.timestamp_source not in {"header", "recorded_source"}:
            raise ValueError("imu_config.timestamp_source must be 'header' or 'recorded_source'")
        self._recorded_image_times = {}
        self.image_to_imu_time_offset_s = float(
            self.imu_config.get("image_to_imu_time_offset_s", 0.0)
        )
        if not np.isfinite(self.image_to_imu_time_offset_s):
            raise ValueError("imu_config.image_to_imu_time_offset_s must be finite")
        self.imu_buffer = (
            ImuBuffer(
                max_samples=int(self.imu_config.get("buffer_max_samples", 6000))
            )
            if self.imu_enabled
            else None
        )
        self.gelsight_T_imu = self._parse_gelsight_T_imu(
            self.imu_config.get("gelsight_T_imu")
        )
        self.imu_config, self._static_imu_settings, overrides = resolve_static_imu_config(
            self.imu_config, self.gelsight_T_imu
        )
        self._static_imu_initializer = (
            StaticImuInitializer(self._static_imu_settings, self.imu_config)
            if self._static_imu_settings is not None else None
        )
        self._static_imu_collecting = True
        self._static_imu_collection_start = 0.0
        self._imu_stream_start_time = None
        self._static_trial_seed = None
        self._static_imu_status_logged = None
        self._static_imu_contact_cutoff = None
        self._recorded_rest_events = None
        if self._static_imu_settings is not None:
            self.logger.info(
                "Static IMU initialization enabled; owns trial gravity, velocity and "
                "gyro bias; superseded settings: %s" % (", ".join(overrides) or "none")
            )
            if self._static_imu_settings["mode"] == "fixed_time":
                self.logger.info(
                    "Fixed IMU initialization interval: %.9f -> %.9f s in the selected "
                    "absolute IMU clock; stationary detection disabled"
                    % (self._static_imu_settings["start_time_s"], self._static_imu_settings["end_time_s"])
                )
            elif self._static_imu_settings["mode"] == "recorded_rest_event":
                self.logger.info(
                    "IMU initialization averages recorded_rest_started -> recorded_rest_completed "
                    "from %s and propagates from trial_ready with zero velocity; motion check %s"
                    % (self._static_imu_settings["events_topic"],
                       "enabled" if self._static_imu_settings["motion_check"] else "disabled")
                )
            elif self._static_imu_settings["mode"] == "startup_duration":
                self.logger.info(
                    "IMU initialization uses the first %.3f s after the first accepted IMU measurement"
                    % self._static_imu_settings["duration_s"]
                )
        # Simulation-only GT samples can initialize each trial's gravity and,
        # when requested, the first trial's velocity.  GT poses are not factors.
        self.initial_ground_truth_world_R_gel = None
        self._ground_truth_gel_rotations = []
        self._ground_truth_gel_positions = []
        self._ground_truth_imu_positions = []
        controls = resolve_gt_state_test(self.imu_config)
        gt_test = controls['enabled']
        if gt_test and not (self.imu_enabled
                            and self.imu_config.get("frontend_optimization", False)
                            and self.gelsight_T_imu is not None):
            raise ValueError("GT previous-state test requires calibrated frontend VI")
        self._gt_test_states = GroundTruthImuStates() if gt_state_test_needs_gt(controls) else None
        self._gt_test_initial_state = None
        self._gt_test_navigation_T_world = None
        self._gt_test_last_image_timestamp = None
        self._gt_test_previous_image_timestamp = None
        self.latest_gt_state_test = None
        if gt_test:
            self.logger.warning(
                "GT previous-state diagnostic controls: %s; bias remains estimated" % controls
            )
        self._frontend_trial_start_timestamp = None
        self._trial_start_world_R_gel = None
        self.inertial_state = None
        self.inertial_propagator = None
        self._imu_integration_sample = None
        self._coast_start_timestamp = None
        self.frontend_visual_inertial = self._make_frontend_solver(self.imu_config)
        if self.frontend_visual_inertial is None:
            self.logger.info("World gravity alignment requires the calibrated VI frontend; inactive")
        # The frontend VI state and published poses use the same trial axes.
        self._frontend_vi_state = None
        self._frontend_state_frame = "imu"
        self._frontend_coast_state = None
        self._frontend_frame_prediction = None
        self._frontend_imu_preintegration = None
        self._pending_frontend_initialization = None
        self.bridge = CvBridge()
        # Initialize reconstructor
        self.recon = self._make_reconstructor(calib_model_path)
        # Skip background check flag
        self.skip_background_check = skip_background_check

        # Store initial images as background
        self.bg_images = []
        # Tracking state variables
        self.prev_in_contact_flag = False
        self.prev_is_keyframe_flag = False
        self._consecutive_tracking_failures = 0
        self._contact_failure_since = None
        self._no_contact_since = None
        self._reference_is_imu_only = False
        self.ref_T_prev = np.eye(4, dtype=np.float32)
        self.trial_T_ref = np.eye(4, dtype=np.float64)
        self._trial_index = -1
        self.prev_frame = None
        self.prev_header = None
        self.prev_fid = None
        self.latest_observation_frame = None
        self.latest_normal_correspondences = None
        self.curr_kid = -1
        self.ref_frame = None
        self.ref_header = None
        # Pose of ``prev_frame`` in the current reference-keyframe coordinates.
        # Unlike ``ref_T_prev``, this is deliberately not advanced while the
        # sensor is out of contact.  It lets reacquisition match the first new
        # tactile frame against the last real tactile surface as a fallback.
        self.last_tactile_ref_T = None
        self.last_tactile_header = None
        self.previous_tactile_ref_T = None
        self.previous_tactile_header = None
        self.curr_fid = -1
        # IMU propagation will use this lifecycle after extrinsic/time calibration.
        self.tracking_mode = TrackingMode.NO_CONTACT_COASTING

    def _observe_static_imu(self, header, at_contact=False):
        """Collect stationary evidence only during background/no-contact images."""
        settings = getattr(self, "_static_imu_settings", None)
        timestamp = self._imu_timestamp(header)
        if settings is None or timestamp is None:
            return
        if at_contact:
            if self._static_imu_contact_cutoff is None:
                self._static_imu_contact_cutoff = timestamp
            timestamp = min(timestamp, self._static_imu_contact_cutoff)
        else:
            self._static_imu_contact_cutoff = None
        if not self._static_imu_collecting:
            self._static_imu_initializer = StaticImuInitializer(settings, self.imu_config)
            self._static_imu_collecting = True
            self._static_imu_collection_start = timestamp
            self._static_imu_status_logged = None
        initializer = self._static_imu_initializer
        # Recover from a timed-out approach only after a new stationary window.
        if (settings["mode"] == "stationary_detection" and initializer.stationary is not None
                and timestamp - initializer.stationary[0] > settings["max_propagation_s"]):
            self._static_imu_initializer = StaticImuInitializer(settings, self.imu_config)
            self._static_imu_collection_start = timestamp
            initializer = self._static_imu_initializer
        initializer.observe(self.imu_buffer.samples_between(
            self._static_imu_collection_start, timestamp
        ), stream_start_time=getattr(self, "_imu_stream_start_time", None),
            rest_events=getattr(self, "_recorded_rest_events", None))

    def add_rest_event(self, name, timestamp):
        """Record one operator rest event time in the selected IMU clock.

        A new ``recorded_rest_started`` discards the previous rest sequence.
        """
        if not np.isfinite(timestamp) or timestamp <= 0.0:
            raise ValueError("rest event timestamp must be positive and finite")
        events = getattr(self, "_recorded_rest_events", None)
        if events is None or name == "recorded_rest_started":
            events = self._recorded_rest_events = {}
        events[name] = float(timestamp)

    def _prepare_static_trial(self, header):
        """Hold node registration until a bounded first-contact state exists."""
        if not self._static_imu_collecting:
            status = "new trial requires a stationary no-contact interval"
            seed = None
        else:
            self._observe_static_imu(header, at_contact=True)
            timestamp = self._imu_timestamp(header)
            initializer = self._static_imu_initializer
            seed = (initializer.predict_trial(
                self.imu_buffer, timestamp, self.gelsight_T_imu,
                self.frontend_visual_inertial.prior_sigmas,
            ) if timestamp is not None else None)
            status = initializer.status if timestamp is not None else "waiting for a valid image timestamp"
        if seed is None:
            if status != self._static_imu_status_logged:
                self.logger.info("Static IMU initialization: " + status)
                self._static_imu_status_logged = status
            return False
        self._static_trial_seed = seed
        return True

    def _make_frontend_solver(self, config):
        """Create the GTSAM frontend only when calibration is valid."""
        if not (
            bool(config.get("frontend_optimization", False))
            and self.gelsight_T_imu is not None
        ):
            return None
        solver = create_frontend_solver(config)
        if getattr(solver, "reference_marginalization", False):
            self.logger.info("Frontend reference marginalization: retained 21D, solve 36D; "
                             "first pose fixed: bootstrap 9D -> 24D solve -> 15D retained; "
                             "later shared reference 30D solve; backend map only")
        return solver

    def frontend_gravity_for_backend(self):
        """Return gravity in the public trial axes after VI initialization."""
        solver = self.frontend_visual_inertial
        if solver is None or self._frontend_vi_state is None:
            return None
        gravity = np.asarray(solver.gravity, dtype=np.float64)
        if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
            return None
        return gravity.copy()

    @staticmethod
    def _parse_gelsight_T_imu(transform):
        """Validate the calibrated transform that maps IMU coordinates into GelSight."""
        if transform is None:
            return None
        matrix = np.asarray(transform, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("imu_config.gelsight_T_imu must be a finite 4x4 matrix")
        if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0)):
            raise ValueError("imu_config.gelsight_T_imu must be homogeneous")
        return matrix

    @staticmethod
    def _header_timestamp(header):
        """Extract a positive ROS header timestamp, or return ``None``."""
        if header is None:
            return None
        timestamp = float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9
        return timestamp if timestamp > 0.0 else None

    def _camera_timestamp(self, header):
        """Return the selected camera acquisition time on the host time axis."""
        timestamp = self._header_timestamp(header)
        if timestamp is None:
            return None
        if getattr(self, "timestamp_source", "header") == "recorded_source":
            key = int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)
            timestamp = getattr(self, "_recorded_image_times", {}).get(key)
        return timestamp

    def _imu_timestamp(self, header):
        """Map the selected camera time to the synchronized IMU clock."""
        timestamp = self._camera_timestamp(header)
        if timestamp is None:
            return None
        offset = float(getattr(self, "image_to_imu_time_offset_s", 0.0))
        synchronized = timestamp + offset
        return synchronized if synchronized > 0.0 else None

    def set_recorded_image_timestamp(self, header, timestamp):
        """Keep source time for delayed use of a prior tactile frame header."""
        if not np.isfinite(timestamp) or timestamp <= 0.0:
            raise ValueError("recorded image timestamp must be positive and finite")
        key = int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)
        if not hasattr(self, "_recorded_image_times"):
            self._recorded_image_times = {}
        self._recorded_image_times[key] = float(timestamp)
        # Delayed keyframe promotion and coast anchoring need recent history.
        while len(self._recorded_image_times) > 6000:
            self._recorded_image_times.pop(next(iter(self._recorded_image_times)))

    def add_ground_truth_gel_rotation(self, timestamp, world_R_gel):
        """Buffer simulation GT attitudes for trial-start initialization."""
        rotation = np.asarray(world_R_gel, dtype=np.float64)
        if (
            not np.isfinite(timestamp)
            or rotation.shape != (3, 3)
            or not np.all(np.isfinite(rotation))
        ):
            return False
        samples = getattr(self, "_ground_truth_gel_rotations", None)
        if samples is None:
            samples = []
            self._ground_truth_gel_rotations = samples
        timestamps = [item[0] for item in samples]
        index = bisect_left(timestamps, float(timestamp))
        if index < len(samples) and samples[index][0] == float(timestamp):
            samples[index] = (float(timestamp), rotation.copy())
        else:
            samples.insert(index, (float(timestamp), rotation.copy()))
        # A bag's GT stream is normally ordered; keep enough history to select
        # the first-contact sample even if ROS callbacks arrive slightly late.
        if len(samples) > 10000:
            del samples[:-10000]
        return True

    def add_ground_truth_gel_pose(self, timestamp, world_R_gel, world_p_gel):
        """Buffer a simulation GelSight pose for first-trial initialization."""
        position = np.asarray(world_p_gel, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            return False
        if not self.add_ground_truth_gel_rotation(timestamp, world_R_gel):
            return False
        samples = getattr(self, "_ground_truth_gel_positions", None)
        if samples is None:
            samples = []
            self._ground_truth_gel_positions = samples
        timestamps = [item[0] for item in samples]
        index = bisect_left(timestamps, float(timestamp))
        if index < len(samples) and samples[index][0] == float(timestamp):
            samples[index] = (float(timestamp), position.copy())
        else:
            samples.insert(index, (float(timestamp), position.copy()))
        if len(samples) > 10000:
            del samples[:-10000]
        pending = getattr(self, "_pending_frontend_initialization", None)
        if pending is not None and getattr(self, "_frontend_vi_state", None) is None:
            self._reset_frontend_visual_inertial(
                pending["ref_T_gelsight"], pending["header"]
            )
        return True

    def add_recent_ground_truth_gel_rotation(self, timestamp, world_R_gel):
        """Keep one second of online trial attitudes, without GT positions.

        Retain the two samples bracketing an unresolved trial anchor as well.
        Offline callers can still preload GT through the existing methods.
        """
        if not self.add_ground_truth_gel_rotation(timestamp, world_R_gel):
            return False
        samples = self._ground_truth_gel_rotations
        times = [sample[0] for sample in samples]
        max_age = float(self.imu_config.get("gt_trial_attitude_max_age_s", 0.05))
        cutoff = times[-1] - max(1.0, 2.0 * max_age)
        first_recent = bisect_left(times, cutoff)
        retained_anchor = []
        anchor = getattr(self, "_frontend_trial_start_timestamp", None)
        if anchor is not None and getattr(self, "_trial_start_world_R_gel", None) is None:
            right = bisect_left(times, anchor)
            for index in (right - 1, right):
                if 0 <= index < first_recent and abs(times[index] - anchor) <= max_age:
                    retained_anchor.append(samples[index])
        self._ground_truth_gel_rotations = retained_anchor + samples[first_recent:]
        pending = getattr(self, "_pending_frontend_initialization", None)
        if pending is not None and getattr(self, "_frontend_vi_state", None) is None:
            self._reset_frontend_visual_inertial(
                pending["ref_T_gelsight"], pending["header"]
            )
        return True

    def add_ground_truth_imu_position(self, timestamp, world_p_imu):
        """Buffer the GT IMU origin for first-trial velocity initialization."""
        position = np.asarray(world_p_imu, dtype=np.float64)
        if (not np.isfinite(timestamp) or position.shape != (3,)
                or not np.all(np.isfinite(position))):
            return False
        samples = getattr(self, "_ground_truth_imu_positions", None)
        if samples is None:
            samples = []
            self._ground_truth_imu_positions = samples
        timestamps = [item[0] for item in samples]
        index = bisect_left(timestamps, float(timestamp))
        if index < len(samples) and samples[index][0] == float(timestamp):
            samples[index] = (float(timestamp), position.copy())
        else:
            samples.insert(index, (float(timestamp), position.copy()))
        if len(samples) > 10000:
            del samples[:-10000]
        pending = getattr(self, "_pending_frontend_initialization", None)
        if pending is not None and getattr(self, "_frontend_vi_state", None) is None:
            self._reset_frontend_visual_inertial(
                pending["ref_T_gelsight"], pending["header"]
            )
        return True

    def _first_trial_velocity_from_gt(self, timestamp):
        """Interpolate GT IMU-origin central-difference velocity into trial axes."""
        window = float(self.imu_config.get("frontend_gt_velocity_window_s", 0.05))
        max_age = float(self.imu_config.get("frontend_gt_velocity_max_age_s", 0.05))
        if not all(np.isfinite(value) and value > 0.0 for value in (window, max_age)):
            raise ValueError("GT velocity stencil span and maximum age must be positive")
        samples = tuple(getattr(self, "_ground_truth_imu_positions", ()))
        if timestamp is None or self.initial_ground_truth_world_R_gel is None or not samples:
            return None
        timestamps = [item[0] for item in samples]
        right = bisect_left(timestamps, timestamp)
        if right == len(samples):
            return None
        centers = (right,) if timestamps[right] == timestamp else (right - 1, right)
        derivatives = []
        for center in centers:
            if center < 1 or center + 1 >= len(samples):
                return None  # Central differences require both time directions.
            before, current, after = samples[center - 1:center + 2]
            if (
                after[0] - before[0] > window + 1e-9
                or max(abs(before[0] - timestamp), abs(after[0] - timestamp)) > max_age + 1e-9
            ):
                return None
            left_dt, right_dt = current[0] - before[0], after[0] - current[0]
            left_slope = (current[1] - before[1]) / left_dt
            right_slope = (after[1] - current[1]) / right_dt
            derivatives.append(
                (right_dt * left_slope + left_dt * right_slope) / (left_dt + right_dt)
            )
        velocity_world = derivatives[0]
        if len(derivatives) == 2:
            fraction = (timestamp - timestamps[right - 1]) / (timestamps[right] - timestamps[right - 1])
            velocity_world = (1 - fraction) * derivatives[0] + fraction * derivatives[1]
        return self.initial_ground_truth_world_R_gel.T @ velocity_world

    def add_ground_truth_imu_pose(self, timestamp, world_R_imu, world_p_imu):
        """Supply full GT IMU poses to the optional per-frame diagnostic."""
        states = getattr(self, "_gt_test_states", None)
        if states is not None and not states.add(timestamp, world_R_imu, world_p_imu):
            return False
        return self.add_ground_truth_imu_position(timestamp, world_p_imu)

    def _set_gt_test_anchor(self):
        if resolve_gt_state_test(self.imu_config)['enabled']:
            self._gt_test_initial_state = self._frontend_vi_state.copy()
            self._gt_test_navigation_T_world = None

    def _gt_test_inputs(self, previous, previous_timestamp, reference_timestamp, reference_pose=None):
        """Correct inputs in the fixed trial world, without using target GT."""
        controls = resolve_gt_state_test(self.imu_config)
        needs_previous = controls['previous_pose_from_gt'] or controls['previous_velocity_from_gt']
        corrected = previous.copy()
        if not gt_state_test_needs_gt(controls):
            self.latest_gt_state_test = dict(
                controls=controls, previous_timestamp=previous.timestamp,
                reference_timestamp=reference_timestamp, prior_state_timestamp=previous.timestamp,
                previous_pose=previous.pose.tolist(), previous_velocity=previous.velocity.tolist(),
                visual_reference_pose=reference_pose.tolist(),
            )
            return corrected, reference_pose
        states = getattr(self, "_gt_test_states", None)
        anchor = getattr(self, "_gt_test_initial_state", None)
        if states is None or anchor is None or (controls['reference_pose_from_gt'] and reference_timestamp is None):
            raise RuntimeError("GT previous-state test lacks its trial anchor or visual reference time")
        # The anchor can leave the bounded GT buffer during a long trial.
        # Once fixed, the trial transform does not need the anchor samples again.
        transform = self._gt_test_navigation_T_world
        anchor_gt = states.at(anchor.timestamp) if transform is None else None
        previous_gt = states.at(previous_timestamp) if needs_previous else None
        reference_gt = states.at(reference_timestamp) if controls['reference_pose_from_gt'] else None
        missing = []
        if transform is None and anchor_gt is None:
            missing.append('anchor %s' % anchor.timestamp)
        if needs_previous and previous_gt is None:
            missing.append('previous %s' % previous_timestamp)
        if controls['reference_pose_from_gt'] and reference_gt is None:
            missing.append('visual reference %s' % reference_timestamp)
        if missing:
            raise RuntimeError(
                "GT previous-state test needs bracketing /ground_truth/imu_pose samples "
                "for %s" % ', '.join(missing)
            )
        if transform is None:
            transform = anchor.pose @ np.linalg.inv(anchor_gt[0])
            self._gt_test_navigation_T_world = transform
        if needs_previous:
            if (abs(previous_timestamp - previous.timestamp) > 1e-9
                    and not (controls['previous_pose_from_gt'] and controls['previous_velocity_from_gt'])):
                raise RuntimeError(
                    'Changing the previous IMU timestamp requires both GT pose and velocity; '
                    'use previous_state_time: accepted_state for independent state ablations'
                )
            corrected.timestamp = previous_timestamp
        if controls['previous_pose_from_gt']:
            corrected.pose = transform @ previous_gt[0]
        if controls['previous_velocity_from_gt']:
            corrected.velocity = transform[:3, :3] @ previous_gt[1]
        if controls['reference_pose_from_gt']:
            reference_pose = gelsight_pose(transform @ reference_gt[0], self.gelsight_T_imu)
        self.latest_gt_state_test = dict(
            controls=controls, previous_timestamp=corrected.timestamp, reference_timestamp=reference_timestamp,
            prior_state_timestamp=previous.timestamp,
            previous_pose=corrected.pose.tolist(), previous_velocity=corrected.velocity.tolist(),
            visual_reference_pose=reference_pose.tolist(),
            accel_bias=corrected.accel_bias.tolist(), gyro_bias=corrected.gyro_bias.tolist(),
        )
        return corrected, reference_pose

    def _lock_gravity_to_first_trial(self, timestamp):
        """Freeze the GT attitude closest to the first trial anchor time."""
        if getattr(self, "initial_ground_truth_world_R_gel", None) is not None:
            return True
        samples = tuple(getattr(self, "_ground_truth_gel_rotations", ()))
        if timestamp is None or not samples:
            return False
        timestamps = [item[0] for item in samples]
        index = bisect_left(timestamps, float(timestamp))
        candidates = []
        if index < len(samples):
            candidates.append(samples[index])
        if index:
            candidates.append(samples[index - 1])
        if not candidates:
            return False
        _, rotation = min(candidates, key=lambda item: abs(item[0] - timestamp))
        self.initial_ground_truth_world_R_gel = rotation.copy()
        gravity_imu = gravity_from_world_R_imu(
            rotation, self.gelsight_T_imu,
            self.imu_config.get("gravity_mps2", 9.81),
        )
        logger = getattr(self, "logger", None)
        if logger is not None:
            logger.info(
                "Locked simulation gravity to first trial at %.9f; "
                "IMU-frame gravity=%s" % (timestamp, np.array2string(gravity_imu, precision=5))
            )
        return True

    def _ground_truth_rotation_near(self, timestamp):
        """Get a bounded GT attitude at the start of a tactile trial."""
        samples = tuple(getattr(self, "_ground_truth_gel_rotations", ()))
        if timestamp is None or not samples:
            return None
        max_age = float(self.imu_config.get("gt_trial_attitude_max_age_s", 0.05))
        if not np.isfinite(max_age) or max_age <= 0.0:
            raise ValueError("gt_trial_attitude_max_age_s must be positive")
        timestamps = [item[0] for item in samples]
        index = bisect_left(timestamps, float(timestamp))
        candidates = samples[max(0, index - 1):min(len(samples), index + 1)]
        nearest_timestamp, rotation = min(
            candidates, key=lambda item: abs(item[0] - timestamp)
        )
        if abs(nearest_timestamp - timestamp) > max_age:
            return None
        return rotation.copy()

    def _trial_initial_gt_rotation(self, timestamp):
        """Use one nearby GT attitude at the start of the active trial."""
        locked = getattr(self, "_trial_start_world_R_gel", None)
        if locked is not None:
            return locked.copy()
        if getattr(self, "_trial_index", 0) > 0:
            rotation = self._ground_truth_rotation_near(timestamp)
        else:
            rotation = getattr(self, "initial_ground_truth_world_R_gel", None)
            if rotation is None and self._ground_truth_rotation_near(timestamp) is not None:
                self._lock_gravity_to_first_trial(timestamp)
                rotation = self.initial_ground_truth_world_R_gel
        if rotation is None:
            return None
        self._trial_start_world_R_gel = rotation.copy()
        return rotation.copy()

    def _imu_coasting_ready(self):
        return (
            self.gelsight_T_imu is not None
            and self.imu_buffer is not None
        )

    def _record_tactile_pose(self, ref_T_gelsight, header, reference_changed=False,
                             preserve_frontend_state=False):
        """Record an accepted contact surface pose and its trial-frame VI state."""
        pose = np.asarray(ref_T_gelsight, dtype=np.float64).copy()
        if reference_changed:
            self.previous_tactile_ref_T = None
            self.previous_tactile_header = None
        elif getattr(self, "last_tactile_ref_T", None) is not None:
            self.previous_tactile_ref_T = self.last_tactile_ref_T.copy()
            self.previous_tactile_header = getattr(self, "last_tactile_header", None)
        self.last_tactile_ref_T = pose
        self.last_tactile_header = header
        self._last_tactile_frame = getattr(self, "prev_frame", None)
        self._last_tactile_trial_pose = self.trial_T_ref @ pose
        self._contact_failure_since = None
        self._reference_is_imu_only = False
        # Contact acceptance clears the separate no-contact prediction. The
        # next gap starts from the newly accepted frontend trial-frame state.
        self._frontend_coast_state = None
        if not preserve_frontend_state:
            self._frontend_imu_preintegration = None
        self.inertial_state = None
        self.inertial_propagator = None
        self._imu_integration_sample = None
        self._coast_start_timestamp = None
        if reference_changed:
            if not preserve_frontend_state:
                self._reset_frontend_visual_inertial(ref_T_gelsight, header)
        else:
            timestamp = self._imu_timestamp(header)
            if (
                self.frontend_visual_inertial is not None
                and timestamp is not None
            ):
                state = self._frontend_vi_state
                if (
                    state is None
                    and (
                        self.imu_config.get("frontend_gravity_initialization")
                        == "ground_truth_initial_pose"
                        or self.imu_config.get("frontend_first_trial_velocity_from_gt", False)
                    )
                ):
                    # A trial may arrive before its GT attitude.
                    # Retry the frontend initialization on a later contact
                    # frame; do not create a VI state with fallback -Z gravity.
                    self._reset_frontend_visual_inertial(pose, header)
                    state = self._frontend_vi_state
                    if state is None:
                        return
                if state is None or timestamp != state.timestamp:
                    # Keep the complete 15D state across contact frames.  The
                    # direct VIO objective estimates velocity and biases jointly;
                    # resetting them here turns it into independent one-step fits.
                    velocity = np.zeros(3) if state is None else state.velocity.copy()
                    accel_bias = (
                        np.asarray(
                            self.imu_config.get("initial_accel_bias_mps2", (0, 0, 0)),
                            dtype=np.float64,
                        )
                        if state is None else state.accel_bias.copy()
                    )
                    gyro_bias = (
                        np.asarray(
                            self.imu_config.get("initial_gyro_bias_rps", (0, 0, 0)),
                            dtype=np.float64,
                        )
                        if state is None else state.gyro_bias.copy()
                    )
                    navigation_pose = imu_pose(self.trial_T_ref @ pose, self.gelsight_T_imu)
                    self._frontend_vi_state = NavigationState(
                        timestamp, navigation_pose, velocity, accel_bias, gyro_bias,
                    )
        self._sync_reference_prior()
        history = getattr(self, "_contact_velocity_history", None)
        if (history is not None and self._frontend_vi_state is not None
                and not getattr(self.frontend_visual_inertial, "imu_prediction_only", False)):
            samples = ()
            if (history.settings['model'] == 'inertial_quadratic' and history.samples
                    and self.imu_buffer is not None
                    and self._frontend_vi_state.timestamp > history.samples[-1][0]):
                samples = self.imu_buffer.samples_covering(
                    history.samples[-1][0], self._frontend_vi_state.timestamp)
            history.observe(self._frontend_vi_state, self._trial_index, samples)
        elif history is not None:
            history.clear()
        self._coast_velocity_seed = None
        self._coast_velocity_anchor_timestamp = None
        if getattr(self, "_static_imu_settings", None) is not None:
            # A future disconnected trial must not reuse stationary evidence
            # from an accepted tactile contact (which may include sliding).
            self._static_imu_collecting = False

    def _sync_reference_prior(self):
        """Change visual reference only at a serialized accepted-state boundary."""
        solver = self.frontend_visual_inertial
        state = getattr(self, "_frontend_vi_state", None)
        if not getattr(solver, "reference_marginalization", False) or state is None:
            return
        timestamp = self._imu_timestamp(getattr(self, "ref_header", None))
        if timestamp is not None:
            solver.set_reference(timestamp, state)

    def _reset_frontend_visual_inertial(self, ref_T_gelsight, header):
        """Start the tracker-local VI state at a new tactile reference frame."""
        self._clear_contact_velocity_history()
        self._frontend_imu_preintegration = None
        if getattr(self, "_static_imu_settings", None) is not None:
            seed = self._static_trial_seed
            if seed is None:
                raise RuntimeError("static IMU trial must be prepared before node registration")
            timestamp = self._imu_timestamp(header)
            if timestamp is None or abs(seed.state.timestamp - timestamp) > 1e-9:
                raise ValueError("static IMU seed timestamp must match the contact image")
            if not np.allclose(self.trial_T_ref @ ref_T_gelsight, np.eye(4)):
                raise ValueError("static IMU initialization requires the new trial origin")
            # Static startup already predicts in gravity-aligned trial axes.
            self.trial_T_ref = gelsight_pose(seed.state.pose, self.gelsight_T_imu)
            self.trial_T_ref[:3, 3] = 0.0
            self._trial_gravity_rotation = self.trial_T_ref[:3, :3].copy()
            local_config = dict(self.imu_config, frontend_gravity_mps2=seed.gravity.tolist())
            self.frontend_visual_inertial = self._make_frontend_solver(local_config)
            self.frontend_visual_inertial.marginal_prior = seed.prior
            self._frontend_vi_state = seed.state.copy()
            self._pending_frontend_initialization = None
            self._static_imu_collecting = False
            self._static_trial_seed = None
            self._set_gt_test_anchor()
            self.logger.info(
                "Static IMU trial %d: stationary anchor %.9f -> contact %.9f; "
                "gravity in trial axes=%s; IMU-origin velocity=%s m/s; gyro bias=%s rad/s"
                % (self._trial_index, seed.stationary_timestamp, seed.state.timestamp,
                   np.array2string(seed.gravity, precision=6),
                   np.array2string(seed.state.velocity, precision=6),
                   np.array2string(seed.state.gyro_bias, precision=6))
            )
            return
        timestamp = self._imu_timestamp(header)
        camera_timestamp = self._camera_timestamp(header)
        anchor_timestamp = (
            getattr(self, "_frontend_trial_start_timestamp", None)
            or camera_timestamp
        )
        if not (
            self.frontend_visual_inertial is not None
            and timestamp is not None
        ):
            self._frontend_vi_state = None
            self._pending_frontend_initialization = None
            return
        gt_velocity_requested = (
            self.imu_config.get("frontend_first_trial_velocity_from_gt", False)
            and getattr(self, "_trial_index", 0) == 0
        )
        if gt_velocity_requested:
            pending = getattr(self, "_pending_frontend_initialization", None)
            if pending is None or pending["anchor_timestamp"] != anchor_timestamp:
                pending = dict(
                    anchor_timestamp=anchor_timestamp, timestamp=timestamp,
                    camera_timestamp=camera_timestamp,
                    pose=self.trial_T_ref @ np.asarray(ref_T_gelsight, dtype=np.float64),
                    ref_T_gelsight=np.asarray(ref_T_gelsight, dtype=np.float64).copy(),
                    header=header,
                )
                self._pending_frontend_initialization = pending
            # A future GT callback completes the original anchor state. A later
            # image must not replace its timestamp or pose while GT is pending.
            timestamp = pending["timestamp"]
            camera_timestamp = pending["camera_timestamp"]
        else:
            self._pending_frontend_initialization = None
        gravity_initialization = str(
            self.imu_config.get(
                "frontend_gravity_initialization", "fixed_world_z"
            )
        )
        if gravity_initialization == "ground_truth_initial_pose":
            world_R_gel = self._trial_initial_gt_rotation(anchor_timestamp)
            if world_R_gel is None:
                # Wait for the trial's GT attitude.  Do not reuse a previous
                # trial's gravity or inject a fallback -Z vector.
                self._frontend_vi_state = None
                return
            trial_gravity = gravity_from_initial_world_R_gel(
                world_R_gel,
                self.imu_config.get("gravity_mps2", 9.81),
            )
        elif gravity_initialization == "fixed_world_z":
            trial_gravity = np.asarray(
                self.imu_config.get(
                    "frontend_gravity_mps2",
                    (0.0, 0.0, -float(self.imu_config.get("gravity_mps2", 9.81))),
                ),
                dtype=np.float64,
            )
        else:
            raise ValueError(
                "imu_config.frontend_gravity_initialization must be "
                "'ground_truth_initial_pose' or 'fixed_world_z'"
            )
        pose = self.trial_T_ref @ np.asarray(
            ref_T_gelsight, dtype=np.float64
        )
        velocity = np.zeros(3)
        if gt_velocity_requested:
            pose = pending["pose"].copy()
            velocity = (
                self._first_trial_velocity_from_gt(timestamp)
                if self._lock_gravity_to_first_trial(anchor_timestamp)
                else None
            )
            if velocity is None:
                # Wait for GT samples on both sides of the original anchor.
                # Never substitute a past-only fit or zero velocity.
                self._frontend_vi_state = None
                return
            self.logger.info(
                "Initialized first-trial IMU-origin velocity from GT IMU poses: %s m/s"
                % np.array2string(velocity, precision=5)
            )
        gyro_bias = np.asarray(self.imu_config.get("initial_gyro_bias_rps", (0, 0, 0)), dtype=np.float64)
        navigation_pose = imu_pose(pose, self.gelsight_T_imu)
        self._frontend_vi_state = NavigationState(
            timestamp, navigation_pose, velocity,
            np.asarray(self.imu_config.get("initial_accel_bias_mps2", (0, 0, 0)), dtype=np.float64),
            gyro_bias,
        )
        self._frontend_vi_state, trial_gravity, _ = self._align_initial_navigation(
            self._frontend_vi_state, trial_gravity,
        )
        local_config = dict(self.imu_config, frontend_gravity_mps2=trial_gravity.tolist())
        self.frontend_visual_inertial = self._make_frontend_solver(local_config)
        self._set_gt_test_anchor()
        self._pending_frontend_initialization = None
        logger = getattr(self, "logger", None)
        if logger is not None:
            logger.info(
                "Frontend VI trial %d: gravity in trial frame=%s"
                % (
                    getattr(self, "_trial_index", 0),
                    np.array2string(trial_gravity, precision=5),
                )
            )

    def _align_initial_navigation(self, state, gravity, prior=None):
        """Initialize a gravity-aligned world at the first contact GelSight origin."""
        rotation = rotation_aligning_gravity_to_negative_z(gravity)
        origin = gelsight_pose(state.pose, self.gelsight_T_imu)[:3, 3].copy()
        aligned = state.reframe(rotation, origin)
        if prior is not None:
            prior = prior.reframe(rotation, origin)
        self.trial_T_ref = gelsight_pose(aligned.pose, self.gelsight_T_imu)
        self.trial_T_ref[:3, 3] = 0.0
        self._trial_gravity_rotation = rotation
        # Use exact -Z after the coordinate transform, without changing magnitude.
        gravity = np.array([0., 0., -np.linalg.norm(gravity)])
        self.logger.info(
            "Gravity-aligned trial world: first contact position=(0,0,0); "
            "gravity=%s; first contact rotation=%s"
            % (np.array2string(gravity, precision=6),
               np.array2string(self.trial_T_ref[:3, :3], precision=6))
        )
        return aligned, gravity, prior

    def _start_trial_contact(self, frame, header):
        """Initialize before publishing; retain the anchor image while GT is pending."""
        align = self.frontend_visual_inertial is not None
        pending = getattr(self, "_pending_trial_contact", None) if align else None
        if pending is None:
            self._trial_index = getattr(self, "_trial_index", -1) + 1
            self._frontend_trial_start_timestamp = self._camera_timestamp(header)
            self._trial_start_world_R_gel = None
            self._trial_gravity_rotation = None
            self.trial_T_ref = np.eye(4, dtype=np.float64)
            if align:
                self._pending_trial_contact = (frame, header, self.curr_fid)
                self._frontend_vi_state = None
        else:
            frame, header, _ = pending
        if align and self._frontend_vi_state is None:
            self._reset_frontend_visual_inertial(np.eye(4), header)
        if align and self._frontend_vi_state is None:
            return None
        self._pending_trial_contact = None
        return frame, header, align

    def _make_keyframe_msg(self, frame, kid, fid, is_new_trial, trial_T_curr, header):
        """Emit tactile observations and a synchronized backend navigation seed."""
        timestamp = self._camera_timestamp(header)
        imu_timestamp = self._imu_timestamp(header)
        state = getattr(self, "_backend_keyframe_state_override", None)
        if state is None:
            state = getattr(self, "_frontend_vi_state", None)
        solver = getattr(self, "frontend_visual_inertial", None)
        navigation = {}
        if (state is not None and imu_timestamp is not None and solver is not None
                and abs(state.timestamp - imu_timestamp) < 1e-8):
            tactile_valid = not getattr(self, "_backend_recovery_keyframe", False)
            previous = getattr(self, "_backend_keyframe_timestamp", None)
            buffer = getattr(self, "_backend_imu_buffer", None)
            samples = ()
            if not is_new_trial and previous is not None and buffer is not None:
                samples = buffer.samples_covering(previous, state.timestamp)
                # Match the frontend's bounded hold at an IMU recovery endpoint.
                if not samples and not tactile_valid:
                    received = buffer.samples_between(previous, state.timestamp)
                    start = buffer.sample_at(previous)
                    if start is not None and len(received) >= 2:
                        last = received[-1]
                        period = last.timestamp - received[-2].timestamp
                        if 0 < state.timestamp - last.timestamp <= period + 1e-9:
                            samples = (start, *(s for s in received if s.timestamp > previous),
                                       ImuSample(state.timestamp, last.acceleration, last.angular_velocity))
            navigation = dict(navigation_state=state, gravity=solver.gravity,
                              imu_samples=samples, tactile_valid=tactile_valid)
            self._backend_keyframe_timestamp = state.timestamp
            # Retain raw IMU for delayed K_k -> F_j backend feedback.
            # ImuBuffer.max_samples remains the explicit retention bound.
        message = create_keyframe_msg(
            self.bridge, frame, kid=kid, fid=fid, is_new_trial=is_new_trial,
            trial_T_curr=trial_T_curr, header=header, **navigation,
        )
        if getattr(self.bridge, "internal_packets", False):
            from dataclasses import replace
            snapshot = getattr(self, "_tracking_map_snapshot", None)
            anchor_kid = getattr(self, "_tracking_anchor_kid", -1)
            anchor_state = getattr(self, "_tracking_anchor_state", None)
            from gelslam.core.shared_map import frozen_state
            seeds = getattr(self, "_keyframe_seed_states", {})
            # Feedback can be disabled while local VI still changes its map.
            # Always transmit the frontend basis of the previous keyframe.
            if not is_new_trial and self.curr_kid in seeds:
                anchor_kid = self.curr_kid
                anchor_state = seeds[anchor_kid]
            if state is not None:
                seeds[kid] = frozen_state(state)
            self._keyframe_seed_states = seeds
            message = replace(message, trial_id=getattr(self, "_trial_index", -1),
                              session_id=getattr(self, "_ros_session_id", ""),
                              anchor_kid=anchor_kid, anchor_state=anchor_state,
                              solution_revision=0 if snapshot is None else snapshot.revision)
        if timestamp is not None:
            self._last_keyframe_timestamp = timestamp
        # Keep the actual published observation basis independent of mutable
        # frontend references and navigation seeds used for backend rebasing.
        published_pose = (gelsight_pose(navigation['navigation_state'].pose, self.gelsight_T_imu)
                          if 'navigation_state' in navigation else np.asarray(trial_T_curr))
        published_pose = np.array(published_pose, dtype=np.float64, copy=True)
        published_pose.setflags(write=False)
        published = getattr(self, '_published_keyframe_poses', {})
        published[kid] = published_pose
        self._published_keyframe_poses = published
        return message

    def _make_frame_msg(self, frame, fid, ref_kid, ref_T_curr, header=None, **metadata):
        """Publish K_k -> F_j against K_k's transmitted, immutable pose.

        Reference marginalization can move the live frontend K_k estimate.
        Consumers still hold its original observation basis, so transport the
        relative pose without changing the accepted State or retained prior.
        """
        if getattr(self.frontend_visual_inertial, 'reference_marginalization', False):
            published = getattr(self, '_published_keyframe_poses', {}).get(ref_kid)
            if published is not None and not np.array_equal(published, self.trial_T_ref):
                ref_T_curr = np.linalg.solve(
                    published, self.trial_T_ref @ np.asarray(ref_T_curr))
        return create_frame_msg(self.bridge, frame, fid=fid, ref_kid=ref_kid,
                                ref_T_curr=ref_T_curr, header=header, **metadata)

    def _keyframe_interval_s(self):
        value = self.track_config.get("keyframe_max_interval_s", 0.5)
        try:
            interval = float(value)
        except (TypeError, ValueError):
            raise ValueError("track_config.keyframe_max_interval_s must be finite and nonnegative") from None
        if isinstance(value, bool) or not np.isfinite(interval) or interval < 0:
            raise ValueError("track_config.keyframe_max_interval_s must be finite and nonnegative")
        return interval

    def _maybe_add_timed_keyframe(self, frame, header, log_prefix, *, force=False):
        """Promote a valid contact target after the common keyframe deadline."""
        if not frame.is_contacted:
            return None
        interval = self._keyframe_interval_s()
        timestamp = self._camera_timestamp(header)
        last = getattr(self, "_last_keyframe_timestamp", None)
        if last is None:
            # Saved trackers predating the timer still have a reference header.
            last = self._camera_timestamp(getattr(self, "ref_header", None))
        if (timestamp is None or (not force and (
                interval == 0 or last is None
                or timestamp - last < interval - 1e-9))):
            return None
        trial_T_current = self.trial_T_ref @ self.ref_T_prev
        if not np.all(np.isfinite(trial_T_current)):
            return None
        keyframe = self._make_keyframe_msg(
            frame, self.curr_kid + 1, self.curr_fid, False,
            trial_T_current, header,
        )
        self.curr_kid += 1
        self.trial_T_ref = trial_T_current
        self.ref_T_prev = np.eye(4, dtype=np.float32)
        self.ref_frame = self.prev_frame = frame
        self.ref_header = self.prev_header = header
        self.prev_is_keyframe_flag = True
        # Change only the visual reference. Keep the accepted navigation State,
        # its velocity/biases and the correlated marginal prior unchanged.
        self._record_tactile_pose(
            self.ref_T_prev, header, reference_changed=True,
            preserve_frontend_state=True,
        )
        reason = "Low SCR" if force else "Timed"
        if force:
            self._count_frontend_event("low_scr_keyframes")
        self.logger.info(
            "%s -- %s Keyframe: frame ID: %d, interval: %.6f s"
            % (log_prefix, reason, self.curr_fid,
               0.0 if last is None else timestamp - last)
        )
        return keyframe

    def _promote_previous_frame(self, log_prefix, reason):
        """Promote the last accepted tactile frame; reject the failed estimate.

        Keep the accepted VI state and its timestamp. The next frontend solve
        must preintegrate from that state across the rejected image timestamp.
        """
        if (
            self.prev_is_keyframe_flag
            or self.prev_frame is None
            or self.prev_header is None
        ):
            self.curr_fid -= 1
            self.tracking_mode = TrackingMode.TACTILE_HOLD
            self.logger.info(
                "%s -- %s; prior frame unavailable or already a keyframe; "
                "rejected tactile estimate"
                % (log_prefix, reason)
            )
            return False, None

        previous_frame = self.prev_frame
        previous_header = self.prev_header
        previous_fid = getattr(self, "prev_fid", None)
        if previous_fid is None:
            previous_fid = self.curr_fid - 1
        previous_ref_T = self.ref_T_prev
        if not self.prev_in_contact_flag:
            previous_ref_T = getattr(self, "last_tactile_ref_T", None)
            if previous_ref_T is None:
                previous_ref_T = self.ref_T_prev
        trial_T_curr = self.trial_T_ref @ previous_ref_T
        # Keep K_k as the seed anchor while constructing promoted K_{k+1}.
        keyframe_msg = self._make_keyframe_msg(
            previous_frame, self.curr_kid + 1, previous_fid, False,
            trial_T_curr, previous_header,
        )
        self.curr_kid += 1
        self.trial_T_ref = trial_T_curr
        self.ref_frame = previous_frame
        self.ref_header = previous_header
        self.ref_T_prev = np.eye(4, dtype=np.float32)
        self.prev_in_contact_flag = True
        self.prev_is_keyframe_flag = True
        coast_seed = getattr(self, "_coast_velocity_seed", None)
        coast_anchor = getattr(self, "_coast_velocity_anchor_timestamp", None)
        self._record_tactile_pose(
            self.ref_T_prev, previous_header, reference_changed=True,
            preserve_frontend_state=True,
        )
        # Promoting the accepted image changes the reference, not its IMU
        # state or time. It is not a new contact measurement. Keep the frozen
        # coast correction for prediction-only frames and later IMU recovery.
        accepted = getattr(self, "_frontend_vi_state", None)
        if (coast_seed is not None and accepted is not None
                and abs(coast_seed.timestamp - accepted.timestamp) < 1e-9):
            self._coast_velocity_seed = coast_seed
            self._coast_velocity_anchor_timestamp = coast_anchor
            history = getattr(self, "_contact_velocity_history", None)
            if history is not None:
                history.clear()
        self.curr_fid -= 1
        self.tracking_mode = TrackingMode.TACTILE_HOLD
        self.logger.info(
            "%s -- New Keyframe: promoted frame ID: %d after %s; "
            "rejected tactile estimate" % (log_prefix, previous_fid, reason)
        )
        return True, (None, [keyframe_msg])

    def _handle_tracking_failure(self, frame, header, log_prefix, reason,
                                 promote_previous=True, quality_failure=False):
        """Wait only for quality rejections during continuous contact."""
        self._consecutive_tracking_failures = (
            getattr(self, "_consecutive_tracking_failures", 0) + 1
        )
        solver = self.frontend_visual_inertial
        if solver is not None and hasattr(solver, "discard_pending_prior"):
            solver.discard_pending_prior()
        timestamp = self._imu_timestamp(header)
        start = None
        recover_now = not quality_failure
        if quality_failure:
            interval = float(self.track_config.get("contact_failure_keyframe_interval_s", 0.5))
            if not np.isfinite(interval) or interval <= 0:
                raise ValueError("contact_failure_keyframe_interval_s must be positive")
            start = getattr(self, "_contact_failure_since", None)
            if start is None:
                self._contact_failure_since = start = timestamp
            recover_now = (timestamp is not None and start is not None
                           and timestamp - start >= interval - 1e-9)
        else:
            # Recontact and missing correspondences do not start or extend a
            # continuous-contact quality rejection interval.
            self._contact_failure_since = None
        if recover_now:
            recovered = self._recover_current_frame_with_imu(frame, header, log_prefix)
            if recovered is not None:
                if not quality_failure:
                    self._contact_failure_since = None
                return recovered
            self.logger.info(
                "%s -- IMU recovery unavailable; retained last accepted state"
                % log_prefix
            )
        failed_fid = self.curr_fid
        if promote_previous and self._consecutive_tracking_failures == 1:
            # Reference promotion must not reset the failure interval.
            result = self._promote_previous_frame(log_prefix, reason)
            self._contact_failure_since = start
        else:
            self.curr_fid -= 1
            self.tracking_mode = TrackingMode.TACTILE_HOLD
            self.logger.info("%s -- %s; rejected tactile estimate" % (log_prefix, reason))
            result = False, None
        frame_msg = self._imu_prediction_frame_msg(frame, header, failed_fid)
        if frame_msg is not None:
            self.curr_fid = failed_fid
            keyframe_msgs = result[1][1] if result[0] else []
            self.logger.info("%s -- IMU prediction frame: frame ID: %d" % (log_prefix, failed_fid))
            return True, (frame_msg, keyframe_msgs)
        return result

    def _imu_prediction_frame_msg(self, frame, header, fid):
        """Retain a rejected image at its predicted pose without accepting a State.

        Predict from the recovery anchor at t_a to this image's IMU time t_j.
        The frame uses the current keyframe basis, including any promotion.
        Do not change the accepted State, prior, coast cursor or tactile history.
        """
        if (not self.track_config.get("keep_imu_only_frames", False)
                or self.curr_kid < 0 or getattr(self, "_restart_pending", False)):
            return None
        previous, buffer, fixed_anchor = self._recovery_prediction_anchor()
        solver = self.frontend_visual_inertial
        timestamp = self._imu_timestamp(header)
        if (previous is None or solver is None or not self._imu_coasting_ready()
                or timestamp is None or timestamp <= previous.timestamp):
            return None
        samples = buffer.samples_covering(previous.timestamp, timestamp)
        if not self._continuous_imu_samples(samples):
            return None
        prediction = solver.predict_current(previous, samples, self.gelsight_T_imu, force=True)
        if prediction is None:
            return None
        current, _ = prediction
        seed = getattr(self, "_coast_velocity_seed", None)
        if (seed is not None and not fixed_anchor
                and abs(seed.timestamp - previous.timestamp) < 1e-9):
            current = seed.apply(current)
        ref_T_current = (np.linalg.inv(self.trial_T_ref)
                         @ gelsight_pose(current.pose, self.gelsight_T_imu)).astype(np.float32)
        if (abs(current.timestamp - timestamp) > 1e-8
                or not all(np.isfinite(value).all() for value in (
                    ref_T_current, current.velocity, current.accel_bias, current.gyro_bias))):
            return None
        self._frontend_frame_prediction = current
        return self._make_frame_msg(
            frame, fid=fid, ref_kid=self.curr_kid,
            ref_T_curr=ref_T_current, header=header,
            tracking_status=TrackingMode.IMU_PREDICTION.value,
            pose_timestamp=self._camera_timestamp(header),
        )

    def _recover_current_frame_with_imu(self, frame, header, log_prefix):
        """Make an IMU-predicted keyframe, inheriting any matching coast correction."""
        previous, buffer, fixed_anchor = self._recovery_prediction_anchor()
        solver = self.frontend_visual_inertial
        timestamp = self._imu_timestamp(header)
        if not (
            previous is not None and solver is not None
            and self._imu_coasting_ready() and timestamp is not None
            and timestamp > previous.timestamp
        ):
            return None
        samples = buffer.samples_covering(previous.timestamp, timestamp)
        if len(samples) < 2:
            # An image callback can precede its next IMU sample. Hold the last
            # measured acceleration/gyro for at most one received IMU period.
            # Do not put this synthetic endpoint into the measurement buffer.
            start = buffer.sample_at(previous.timestamp)
            received = buffer.samples_between(previous.timestamp, timestamp)
            if start is None:
                self._arm_restart(timestamp, missing_imu=True)
                return None
            if len(received) < 2:
                return None
            last = received[-1]
            period = last.timestamp - received[-2].timestamp
            if timestamp - last.timestamp > period + 1e-9:
                return None
            samples = (
                start, *(sample for sample in received if sample.timestamp > start.timestamp),
                ImuSample(timestamp, last.acceleration, last.angular_velocity),
            )
            self.logger.info(
                "%s -- IMU recovery target boundary: held latest measurement for %.6f s"
                % (log_prefix, timestamp - last.timestamp)
            )
        prediction = self._predict_frontend_imu(previous, samples, force=True)
        if prediction is None:
            return None
        if not self._continuous_imu_samples(samples):
            self._arm_restart(timestamp, missing_imu=True)
            return None
        current, pim = prediction
        predicted_prior = solver.propagate_prior(previous, current, pim, fixed_anchor=fixed_anchor)
        seed = getattr(self, "_coast_velocity_seed", None)
        if (seed is not None and not fixed_anchor
                and abs(seed.timestamp - previous.timestamp) < 1e-9):
            # Propagate the original IMU distribution first, then push both
            # its target mean and correlated prior through the coast shift.
            # Applying the shift only to State leaves a prior that undoes it.
            current = seed.apply(current)
            predicted_prior = seed.apply_prior(predicted_prior)
            self.logger.info(
                "%s -- IMU recovery coast velocity handoff: anchor=%.9f s, target=%.9f s, "
                "position delta=%s m, velocity delta=%s m/s"
                % (log_prefix, seed.timestamp, current.timestamp,
                   seed.velocity_delta * (current.timestamp - seed.timestamp), seed.velocity_delta)
            )
        trial_T_curr = gelsight_pose(current.pose, self.gelsight_T_imu)
        self._backend_keyframe_state_override = current
        self._backend_recovery_keyframe = True
        try:
            keyframe_msg = self._make_keyframe_msg(
                frame, self.curr_kid + 1, self.curr_fid, False, trial_T_curr, header,
            )
        finally:
            self._backend_keyframe_state_override = None
            self._backend_recovery_keyframe = False
        # Commit prediction uncertainty, never a rejected visual posterior.
        solver.discard_pending_prior()
        solver.marginal_prior = predicted_prior
        self._backend_correction_anchor = None
        self._frontend_vi_state = current
        self._frontend_imu_preintegration = None
        self.curr_kid += 1
        old_reference = self.trial_T_ref.copy()
        self.trial_T_ref = trial_T_curr
        self.ref_T_prev = np.eye(4, dtype=np.float32)
        self.prev_frame = self.ref_frame = frame
        self.prev_header = self.ref_header = header
        self.prev_fid = self.curr_fid
        self.prev_in_contact_flag = self.prev_is_keyframe_flag = True
        if getattr(self, "last_tactile_ref_T", None) is not None:
            self.last_tactile_ref_T = (
                np.linalg.inv(trial_T_curr) @ old_reference @ self.last_tactile_ref_T)
        self._frontend_coast_state = None
        self.inertial_state = self.inertial_propagator = None
        self._imu_integration_sample = None
        self._coast_start_timestamp = None
        self._reference_is_imu_only = True
        self._clear_contact_velocity_history()
        self._sync_reference_prior()
        self._contact_failure_since = timestamp
        self._consecutive_tracking_failures = 0
        self.tracking_mode = TrackingMode.IMU_ONLY
        self._count_frontend_event("imu_recovery_keyframes")
        snapshot = getattr(self, "latest_normal_correspondences", None)
        if snapshot is not None:
            self.latest_normal_correspondences = dict(
                snapshot, reference_xy=np.empty((0, 2)), current_xy=np.empty((0, 2)),
                status="IMU prediction: normal-flow rejected",
            )
        frame_msg = self._make_frame_msg(
            frame, fid=self.curr_fid, ref_kid=self.curr_kid,
            ref_T_curr=self.ref_T_prev, header=header,
            tracking_status=TrackingMode.IMU_ONLY.value,
            pose_timestamp=self._camera_timestamp(header),
        )
        self.logger.info(
            "%s -- IMU Recovery Keyframe: frame ID: %d, IMU interval: %.9f -> %.9f"
            % (log_prefix, self.curr_fid, previous.timestamp, current.timestamp)
        )
        return True, (frame_msg, [keyframe_msg])

    def _record_normal_correspondences(self, reference_frame, current_frame,
                                      ref_T_curr, status, visual=None,
                                      reference_pose=None, target_pose=None):
        """Retain the compared frames before a rejected image promotes a keyframe."""
        if not getattr(self, "track_config", {}).get("debug_visualization_enabled", True):
            # Pixel pairs feed only the debug correspondence image.
            reference_xy = current_xy = np.empty((0, 2))
        elif visual is None:
            reference_xy, current_xy = project_normal_correspondences(
                reference_frame, current_frame, ref_T_curr, self.ppmm
            )
        elif hasattr(visual, "correspondences"):
            reference_xy, current_xy = visual.correspondences(reference_pose, target_pose)
        else:
            # Other visual-factor implementations can omit pixel diagnostics.
            reference_xy = current_xy = np.empty((0, 2))
        snapshot = getattr(self, "latest_normal_correspondences", None)
        if snapshot is None:
            snapshot = {}
        self.latest_normal_correspondences = dict(
            snapshot, reference=reference_frame, current=current_frame,
            reference_xy=reference_xy, current_xy=current_xy, status=status,
        )

    def _count_frontend_event(self, name):
        # Also support trackers restored from an older saved state.
        if not hasattr(self, "_frontend_tracking_counts"):
            self._frontend_tracking_counts = Counter()
        self._frontend_tracking_counts[name] += 1

    def frontend_tracking_statistics(self):
        """Return this run's call counts, including separate reacquisition retries."""
        counts = getattr(self, "_frontend_tracking_counts", {})
        names = (
            "refine_calls", "frontend_failed_calls", "vi_solve_calls", "vi_accepted", "vi_quality_rejected",
            "previous_state_retry_calls", "previous_state_retry_accepted",
            "low_scr_keyframes", "imu_recovery_keyframes",
            # Kept for log consumers; recovery alone no longer promotes a target.
            "recovered_contact_keyframes",
            "imu_prediction_only_calls", "imu_prediction_only_accepted",
            "imu_prediction_only_quality_rejected",
            # Legacy summary keys remain for log consumers; no path now
            # increments the removed tactile-only fallback counters.
            "normalflow_only_calls", "initial_shared_points_lt_10",
            "vi_unavailable", "imu_prediction_unavailable", "vi_refinement_failed_other",
            "normalflow_only_accepted", "normalflow_only_quality_rejected",
            "normalflow_only_registration_failed",
        )
        return {name: counts.get(name, 0) for name in names}

    def log_frontend_tracking_summary(self):
        """Log totals once image processing has stopped, before node destruction."""
        self.logger.info(
            "Frontend tracking summary (calls; retries counted separately): "
            + ", ".join(
                "%s=%d" % item for item in self.frontend_tracking_statistics().items()
            )
        )

    def _fail_frontend(self, reason):
        """Reject the target without a tactile-only registration or State update."""
        self._count_frontend_event("frontend_failed_calls")
        self._count_frontend_event(reason)
        solver = self.frontend_visual_inertial
        if solver is not None and hasattr(solver, "discard_pending_prior"):
            solver.discard_pending_prior()
        self.latest_normal_correspondences["status"] = "Frontend failed: " + reason
        self.logger.info("Frontend tracking failed: " + reason)
        raise LoseTrackError("frontend tracking failed: " + reason)

    def apply_map_snapshot(self, snapshot, header):
        """At the F_j boundary, read corrected K_k in this trial's axes.

        Pure gravity-preserving frame corrections transport the retained prior.
        Navigation updates require t_k -> t_j IMU for a fixed-K_k target solve.
        """
        self._tracking_map_snapshot = snapshot
        if getattr(self.frontend_visual_inertial, "reference_marginalization", False):
            # Backend map estimates remain available to publication and seed
            # rebasing. They are correlated with these frontend measurements,
            # so they are not a second prior or an overwrite of a retained pose.
            self._tracking_anchor_kid = -1
            self._tracking_anchor_state = None
            self._backend_correction_anchor = None
            return False
        trial = getattr(self, "_trial_index", -1)
        candidates = [packet for packet in snapshot.observations
                      if packet.trial_id == trial and packet.kid <= self.curr_kid
                      and packet.kid in snapshot.estimates
                      and snapshot.estimates[packet.kid].state is not None]
        self._tracking_anchor_kid = -1
        self._tracking_anchor_state = None
        if not candidates or self.frontend_visual_inertial is None:
            return False
        packet = candidates[-1]
        estimate = snapshot.estimates[packet.kid]
        self._tracking_anchor_kid = packet.kid
        # This is the basis used by new seeds, not an additional measurement.
        self._tracking_anchor_state = getattr(self, "_keyframe_seed_states", {}).get(packet.kid)
        token = (snapshot.epoch, packet.kid, estimate.revision)
        if token == getattr(self, "_backend_feedback_token", None):
            return False
        anchor = estimate.state
        timestamp = self._imu_timestamp(header)
        if timestamp is None or timestamp <= anchor.timestamp:
            return False
        basis = getattr(self, "_keyframe_seed_states", {}).get(packet.kid)
        rigid_update = False
        if basis is not None:
            rotation = (anchor.pose @ np.linalg.inv(basis.pose))[:3, :3]
            gravity = self.frontend_visual_inertial.gravity
            rigid_update = (
                np.allclose(anchor.velocity, rotation @ basis.velocity, rtol=0, atol=1e-10)
                and np.allclose(anchor.accel_bias, basis.accel_bias, rtol=0, atol=1e-10)
                and np.allclose(anchor.gyro_bias, basis.gyro_bias, rtol=0, atol=1e-10)
                and np.allclose(rotation @ gravity, gravity, rtol=0, atol=1e-10))
        global_navigation_update = not rigid_update
        pgo_only = rigid_update
        previous = getattr(self, "_frontend_vi_state", None)
        if pgo_only:
            if previous is None or previous.timestamp < anchor.timestamp:
                return False
        else:
            buffer = getattr(self, "_backend_imu_buffer", None)
            if buffer is None:
                return False
            samples = buffer.samples_covering(anchor.timestamp, timestamp)
            max_gap = float(self.imu_config.get("backend_feedback_max_sample_gap_s", .05))
            if len(samples) < 2 or np.any(np.diff([s.timestamp for s in samples]) > max_gap):
                self._count_frontend_event("backend_feedback_interval_unavailable")
                return False
        ref = snapshot.estimates.get(self.curr_kid)
        if ref is None:
            return False
        # An initial backend seed identical to the tracking anchor needs no reset.
        old_anchor = getattr(self, "_backend_feedback_anchor", None)
        basis = self._tracking_anchor_state
        if basis is None and old_anchor is not None and old_anchor[0] == packet.kid:
            basis = old_anchor[1]
        # Compare with the seed actually sent for K_k, including its first
        # feedback. Comparing only with a previous feedback treats an identity
        # result as a correction and discards later tactile updates at F_i.
        # Use the seed's calibrated pose, not a possibly float32 display pose.
        unchanged = (basis is not None and abs(basis.timestamp - anchor.timestamp) < 1e-9
                     and np.allclose(ref.trial_pose, gelsight_pose(basis.pose, self.gelsight_T_imu),
                                     rtol=0, atol=1e-10)
                     and all(np.allclose(getattr(basis, field), getattr(anchor, field),
                                         rtol=0, atol=1e-10)
                             for field in ('pose', 'velocity', 'accel_bias', 'gyro_bias')))
        self._backend_feedback_token = token
        self._backend_feedback_anchor = (packet.kid, anchor)
        if unchanged:
            self._count_frontend_event("backend_feedback_unchanged")
            return False
        reference_shift = ref.trial_pose @ np.linalg.inv(self.trial_T_ref)
        last_tactile_pose = getattr(self, "_last_tactile_trial_pose", None)
        if last_tactile_pose is not None:
            self._last_tactile_trial_pose = reference_shift @ last_tactile_pose
        self._tracking_anchor_state = anchor
        self._keyframe_seed_states[packet.kid] = anchor
        self.trial_T_ref = ref.trial_pose.copy()
        self._backend_correction_anchor = None if pgo_only else anchor.copy()
        self._backend_correction_is_global_navigation = global_navigation_update
        self._backend_correction_trial = trial
        self._clear_contact_velocity_history()
        self._frontend_imu_preintegration = None
        self._frontend_coast_state = None
        # Preserve the accepted K_k -> F_i relative pose when K_k changes.
        # This is a pose update, not IMU propagation of the accepted state.
        # Pose and world velocity follow the correction; sensor biases stay in IMU axes.
        # Pure frame feedback also transports F_i's retained uncertainty.
        # Navigation feedback retains corrected K_k for the next fixed-anchor
        # solve; its velocity/bias changes must not become a frame-only update.
        if previous is not None and previous.timestamp >= anchor.timestamp:
            corrected = previous.copy()
            corrected.pose = reference_shift @ previous.pose
            corrected.velocity = reference_shift[:3, :3] @ previous.velocity
            if pgo_only or getattr(self, "_reference_is_imu_only", False):
                # Transport retained covariance cross terms as well, including
                # uncertain IMU-only references.
                rotation, translation = reference_shift[:3, :3], reference_shift[:3, 3]
                prior = self.frontend_visual_inertial.marginal_prior
                if prior is not None and (pgo_only or prior.matches(previous)):
                    self.frontend_visual_inertial.marginal_prior = prior.reframe(
                        rotation, -rotation.T @ translation)
            if pgo_only:
                self.frontend_visual_inertial.discard_pending_prior()
                self._count_frontend_event("backend_feedback_reframed")
            self._frontend_vi_state = corrected
            self.inertial_state = self.inertial_propagator = None
            self._imu_integration_sample = None
            self._coast_start_timestamp = None
            self.ref_T_prev = (np.linalg.inv(self.trial_T_ref)
                              @ gelsight_pose(corrected.pose, self.gelsight_T_imu)).astype(np.float32)
        self.last_backend_feedback_delay_s = max(0., monotonic() - estimate.committed_at)
        self._count_frontend_event("backend_feedback_applied")
        return True

    def _recovery_prediction_anchor(self):
        anchor = getattr(self, "_backend_correction_anchor", None)
        if (anchor is not None
                and getattr(self, "_backend_correction_trial", None)
                == getattr(self, "_trial_index", -1)
                and (not getattr(self, "_reference_is_imu_only", False)
                     or getattr(self, "_backend_correction_is_global_navigation", False))):
            return anchor, self._backend_imu_buffer, True
        return getattr(self, "_frontend_vi_state", None), self.imu_buffer, False

    def _continuous_imu_samples(self, samples):
        limit = float(self.imu_config.get("max_sample_gap_s", np.inf))
        return (len(samples) >= 2 and
                np.all(np.diff([sample.timestamp for sample in samples]) <= limit + 1e-9))

    def _arm_restart(self, timestamp, *, missing_imu):
        if self.frontend_visual_inertial is None:
            return
        if not getattr(self, "_restart_pending", False):
            self._restart_pending = True
            self._restart_since = timestamp
            self._pending_segment_contact = None
            self._restart_needs_static = False
            if hasattr(self.logger, "info"):
                self.logger.info("Segment restart pending at %.9f" % timestamp)
        if missing_imu and not self._restart_needs_static:
            from gelslam.core.static_imu_initialization import DEFAULTS
            settings = dict(DEFAULTS, **(getattr(self, "_static_imu_settings", None) or {}))
            settings.update(mode="stationary_detection", start_time_s=None, end_time_s=None, duration_s=None,
                            motion_check=False)
            self._restart_initializer = StaticImuInitializer(settings, self.imu_config)
            self._restart_needs_static = True
            self._restart_since = timestamp
            self._pending_segment_contact = None

    def _restart_contact(self, frame, header):
        """Stage a new segment and publish it only after one accepted image pair."""
        timestamp = self._imu_timestamp(header)
        if timestamp is None:
            return False, None
        pending = getattr(self, "_pending_segment_contact", None)
        solver = self.frontend_visual_inertial
        if pending is None:
            if self._restart_needs_static:
                initializer = self._restart_initializer
                initializer.observe(self.imu_buffer.samples_between(self._restart_since, timestamp))
                seed = initializer.predict_trial(
                    self.imu_buffer, timestamp, self.gelsight_T_imu, solver.prior_sigmas)
                if seed is None:
                    self.tracking_mode = TrackingMode.TACTILE_HOLD
                    return False, None
                state, prior, gravity = seed.state, seed.prior, seed.gravity
            else:
                previous, buffer, fixed_anchor = self._recovery_prediction_anchor()
                samples = buffer.samples_covering(previous.timestamp, timestamp)
                if not self._continuous_imu_samples(samples):
                    # A missing target endpoint can still arrive in the next callback.
                    received = buffer.samples_between(previous.timestamp, timestamp)
                    if (buffer.sample_at(previous.timestamp) is None or
                            (len(received) >= 2 and not self._continuous_imu_samples(received))):
                        self._arm_restart(timestamp, missing_imu=True)
                    return False, None
                predicted = solver.predict_current(previous, samples, self.gelsight_T_imu, force=True)
                if predicted is None:
                    return False, None
                state, pim = predicted
                prior = solver.propagate_prior(previous, state, pim, fixed_anchor=fixed_anchor)
                gravity = solver.gravity.copy()
                origin = gelsight_pose(state.pose, self.gelsight_T_imu)[:3, 3]
                state = state.reframe(np.eye(3), origin)
                prior = prior.reframe(np.eye(3), origin)
            staged_solver = create_frontend_solver(dict(
                self.imu_config, frontend_gravity_mps2=gravity.tolist()))
            staged_solver.marginal_prior = prior
            if getattr(staged_solver, "reference_marginalization", False):
                staged_solver.set_reference(state.timestamp, state)
            self._pending_segment_contact = (frame, header, self.curr_fid, state, staged_solver)
            self.tracking_mode = TrackingMode.REACQUIRE
            return False, None
        reference, ref_header, ref_fid, previous, staged_solver = pending
        samples = self.imu_buffer.samples_covering(previous.timestamp, timestamp)
        if not self._continuous_imu_samples(samples):
            if len(samples) >= 2:
                self._arm_restart(timestamp, missing_imu=True)
            return False, None
        predicted = staged_solver.predict_current(previous, samples, self.gelsight_T_imu, force=True)
        if predicted is None:
            return False, None
        current, pim = predicted
        predicted_prior = staged_solver.propagate_prior(previous, current, pim)
        ref_pose = gelsight_pose(previous.pose, self.gelsight_T_imu)
        visual = DenseNormalResidual(
            reference, frame, self.ppmm, ref_pose, gelsight_pose(current.pose, self.gelsight_T_imu),
            max_samples=int(self.imu_config.get("dense_normalflow_samples", 3000)))
        refined = staged_solver.refine_recovery_reference(
            visual, ref_pose, previous, current, samples, self.gelsight_T_imu, preintegrated=pim)
        accepted = False
        if refined is not None:
            ref_pose = gelsight_pose(staged_solver.last_reference_state.pose, self.gelsight_T_imu)
            pose = gelsight_pose(refined.pose, self.gelsight_T_imu)
            scr, ccs = normalflow_quality(reference, frame, np.linalg.inv(pose) @ ref_pose,
                self.ppmm, int(self.imu_config.get("dense_normalflow_quality_samples", 3000)))
            distance = staged_solver.prediction_pose_distance(current, refined, predicted_prior)
            accepted = (np.isfinite([scr, ccs, distance]).all()
                        and ccs >= float(self.track_config["ccs_threshold"]) and distance <= 22.46)
        if not accepted:
            staged_solver.discard_pending_prior()
            if timestamp - previous.timestamp >= float(self.track_config.get(
                    "contact_failure_keyframe_interval_s", 0.5)) - 1e-9:
                if getattr(staged_solver, "reference_marginalization", False):
                    # This reference has never been published as a trial's
                    # first frame. Rebuild from the accepted old-trial prior
                    # before choosing a different first-pose datum. Reusing
                    # its already-conditioned candidate would accumulate hard
                    # pose conditions from discarded bootstrap images.
                    self._pending_segment_contact = None
                    return self._restart_contact(frame, header)
                staged_solver.marginal_prior = predicted_prior
                self._pending_segment_contact = (frame, header, self.curr_fid, current, staged_solver)
            return False, None
        # Commit the new trial only now. No old-trial IMU or tactile edge crosses it.
        staged_solver.commit_marginal_prior(refined)
        # Keep the confirmed first contact at the new trial origin.
        origin = ref_pose[:3, 3].copy()
        staged_solver.last_reference_state = staged_solver.last_reference_state.reframe(np.eye(3), origin)
        refined = refined.reframe(np.eye(3), origin)
        staged_solver.marginal_prior = staged_solver.marginal_prior.reframe(np.eye(3), origin)
        ref_pose = gelsight_pose(staged_solver.last_reference_state.pose, self.gelsight_T_imu)
        pose = gelsight_pose(refined.pose, self.gelsight_T_imu)
        self.frontend_visual_inertial = staged_solver
        self._trial_index = getattr(self, "_trial_index", -1) + 1
        self._backend_correction_anchor = None
        self._backend_feedback_token = None
        self._tracking_anchor_kid, self._tracking_anchor_state = -1, None
        self._frontend_trial_start_timestamp = self._camera_timestamp(ref_header)
        self._frontend_vi_state = staged_solver.last_reference_state
        self.trial_T_ref = ref_pose
        first = self._make_keyframe_msg(reference, self.curr_kid + 1, ref_fid, True, ref_pose, ref_header)
        self.curr_kid += 1
        self._frontend_vi_state = refined
        second = self._make_keyframe_msg(frame, self.curr_kid + 1, self.curr_fid, False, pose, header)
        self.curr_kid += 1
        self.trial_T_ref, self.ref_T_prev = pose, np.eye(4, dtype=np.float32)
        self.prev_frame = self.ref_frame = frame
        self.prev_header = self.ref_header = header
        self.prev_fid = self.curr_fid
        self.prev_is_keyframe_flag = self.prev_in_contact_flag = True
        self._record_tactile_pose(self.ref_T_prev, header)
        self._restart_pending = False
        self._pending_segment_contact = None
        self._consecutive_tracking_failures = 0
        self.tracking_mode = TrackingMode.CONTACT_TRACKING
        self.logger.info("Segment restart accepted: trial %d, frame %d" % (self._trial_index, self.curr_fid))
        message = self._make_frame_msg(frame, self.curr_fid, self.curr_kid,
                                   self.ref_T_prev, header=header,
                                   pose_timestamp=self._camera_timestamp(header))
        return True, (message, [first, second])

    def _track_contact(self, frame, header):
        """Try each reference, including its seed retry; retain rejected anchors."""
        try:
            return self._refine_normalflow_with_imu(
                self.ref_T_prev, header, self.ref_frame, frame)
        except QualityGateError:
            raise
        except LoseTrackError as error:
            if getattr(self.frontend_visual_inertial, "reference_marginalization", False):
                # An alternate old tactile pose may no longer be retained.
                # Use the normal IMU recovery/keyframe policy at this boundary.
                raise
            if any(reason in str(error) for reason in (
                    "vi_unavailable", "imu_prediction_unavailable", "vi_refinement_failed_other")):
                raise
            reference = getattr(self, "_last_tactile_frame", self.prev_frame)
            pose = getattr(self, "_last_tactile_trial_pose", None)
            if pose is None and getattr(self, "last_tactile_ref_T", None) is not None:
                pose = self.trial_T_ref @ self.last_tactile_ref_T
            if reference is None or reference is self.ref_frame or pose is None:
                raise
            relative = self._refine_normalflow_with_imu(
                (np.linalg.inv(pose) @ self.trial_T_ref @ self.ref_T_prev).astype(np.float32),
                header, reference, frame, reference_pose=pose)
            return (np.linalg.inv(self.trial_T_ref) @ pose @ relative).astype(np.float32)

    def _refine_normalflow_with_imu(self, ref_T_curr, header, reference_frame=None,
                                    current_frame=None, reference_pose=None):
        """Solve NormalFlow rows and native GTSAM IMU residuals.

        Pairwise mode fixes the visual reference. Reference marginalization
        also optimizes it, aliasing its pose when it shares the previous time.
        Prediction-only mode bypasses both the solve and SCR/CCS gate.
        Only the VI optimization path uses visual quality for acceptance.
        """
        self._count_frontend_event("refine_calls")
        self._quality_force_keyframe = False
        reference_header = (
            getattr(self, "ref_header", None)
            if reference_frame is getattr(self, "ref_frame", None)
            else getattr(self, "prev_header", None)
        )
        if reference_frame is not None and reference_frame is getattr(self, "_last_tactile_frame", None):
            reference_header = self.last_tactile_header
        # Preserve this pair even if tracking later changes ref_frame or drops
        # the target. Do not draw a previous image's successful correspondences.
        self.latest_normal_correspondences = {
            "reference": reference_frame, "current": current_frame,
            "reference_stamp": self._header_timestamp(reference_header),
            "current_stamp": self._header_timestamp(header),
            "reference_xy": np.empty((0, 2)), "current_xy": np.empty((0, 2)),
            "status": "Solve failed: no final pose",
        }
        reference_pose = (
            self.trial_T_ref if reference_pose is None
            else np.asarray(reference_pose, dtype=np.float64)
        )
        relative_pose = np.asarray(ref_T_curr, dtype=np.float64)
        if reference_pose.shape != (4, 4) or relative_pose.shape != (4, 4):
            raise ValueError("reference_pose and ref_T_curr must be 4x4 matrices")
        navigation_reference_pose = reference_pose
        solver = self.frontend_visual_inertial
        timestamp = self._imu_timestamp(header)
        previous = getattr(self, "_frontend_vi_state", None)
        reference_marginalization = getattr(solver, "reference_marginalization", False)
        if reference_marginalization and previous is not None:
            # The fallback reference is the last accepted state, so it is
            # still present. Remove the old reference instead of relabeling it.
            solver.set_reference(self._imu_timestamp(reference_header), previous)
        correction_anchor = getattr(self, "_backend_correction_anchor", None)
        if getattr(self, "_backend_correction_trial", None) != getattr(self, "_trial_index", -1):
            correction_anchor = None
        if (getattr(self, "_reference_is_imu_only", False)
                and reference_frame is self.ref_frame and previous is not None
                and self._imu_timestamp(reference_header) == previous.timestamp
                and not getattr(self, "_backend_correction_is_global_navigation", False)):
            correction_anchor = None
        if correction_anchor is not None:
            previous = correction_anchor
        if not (
            solver is not None
            and previous is not None
            and self.imu_buffer is not None
            and self._imu_coasting_ready()
            and timestamp is not None
            and timestamp > previous.timestamp
            and reference_frame is not None
            and current_frame is not None
        ):
            self._fail_frontend("vi_unavailable")
        controls = resolve_gt_state_test(self.imu_config)
        gt_test = controls['enabled']
        if gt_test:
            previous_timestamp = previous.timestamp
            if (controls['previous_state_time'] == 'previous_image'
                    and (controls['previous_pose_from_gt'] or controls['previous_velocity_from_gt'])):
                previous_timestamp = getattr(self, "_gt_test_previous_image_timestamp", None) or previous.timestamp
            previous, navigation_reference_pose = self._gt_test_inputs(
                previous, previous_timestamp, self._imu_timestamp(reference_header), navigation_reference_pose,
            )
            if previous.timestamp >= timestamp:
                raise RuntimeError("GT previous-state test requires an earlier input image")
            self._frontend_vi_state = previous
            if controls['reset_marginal_prior']:
                solver.marginal_prior = None
                solver.discard_pending_prior()
            self._count_frontend_event("gt_previous_state_resets")
        interval_buffer = self._backend_imu_buffer if correction_anchor is not None else self.imu_buffer
        samples = interval_buffer.samples_covering(previous.timestamp, timestamp)
        if len(samples) >= 2 and not self._continuous_imu_samples(samples):
            self._arm_restart(timestamp, missing_imu=True)
            self._fail_frontend("imu_prediction_unavailable")
        prediction = (
            self._predict_frontend_imu(previous, samples)
            if len(samples) >= 2 else None
        )
        if prediction is None:
            self._fail_frontend("imu_prediction_unavailable")
        current, preintegrated = prediction
        # The initialization flag controls the optimizer's starting value only.
        # Consistency is always measured against the physical IMU prediction.
        imu_predicted = (solver.predict_preintegrated(
            previous, preintegrated, timestamp, force=True)[0]
            if hasattr(solver, "predict_preintegrated") else current)
        predicted_prior = (solver.propagate_prior(
            previous, imu_predicted, preintegrated, fixed_anchor=correction_anchor is not None)
                           if hasattr(solver, "propagate_prior") else None)
        if gt_test:
            self.latest_gt_state_test.update(
                target_timestamp=timestamp, predicted_pose=current.pose.tolist(),
                predicted_velocity=current.velocity.tolist(),
            )
        prediction_only = getattr(solver, "imu_prediction_only", False)
        visual = (None if prediction_only else DenseNormalResidual(
            reference_frame, current_frame, self.ppmm,
            navigation_reference_pose, gelsight_pose(current.pose, self.gelsight_T_imu),
            max_samples=int(self.imu_config.get("dense_normalflow_samples", 3000)),
        ))
        self._count_frontend_event(
            "imu_prediction_only_calls" if prediction_only else "vi_solve_calls"
        )
        refine = solver.refine_from_keyframe if correction_anchor is not None else solver.refine_current
        variable_reference = (
            getattr(self, "_reference_is_imu_only", False)
            and reference_frame is self.ref_frame
            and self._imu_timestamp(reference_header) == previous.timestamp
            and correction_anchor is None)
        if variable_reference and not reference_marginalization:
            refine = solver.refine_recovery_reference
        refined = refine(
            visual, navigation_reference_pose, previous, current, samples,
            self.gelsight_T_imu, preintegrated=preintegrated,
        )
        previous_state_retry = (
            not prediction_only
            and getattr(solver, "initialize_from_imu_preintegration", True)
            and refined is None
            and getattr(solver, "last_refinement_failure", None)
            == "initial_shared_points_lt_10"
        )
        if previous_state_retry:
            # Copy the anchor's values into a target-time initial guess. Keep
            # the physical IMU prediction, interval, and prior for the solve
            # and its later consistency check.
            initial_shared_points = len(visual)
            current = previous.copy()
            current.timestamp = timestamp
            visual = DenseNormalResidual(
                reference_frame, current_frame, self.ppmm,
                navigation_reference_pose, gelsight_pose(current.pose, self.gelsight_T_imu),
                max_samples=int(self.imu_config.get("dense_normalflow_samples", 3000)),
            )
            self._count_frontend_event("previous_state_retry_calls")
            self._count_frontend_event("vi_solve_calls")
            self.logger.info(
                "Frontend VI retry: initial_shared_points_lt_10; "
                "initialize target from previous State; "
                "anchor=%.9f s, target=%.9f s, shared points=%d -> %d"
                % (previous.timestamp, timestamp, initial_shared_points, len(visual))
            )
            refined = refine(
                visual, navigation_reference_pose, previous, current, samples,
                self.gelsight_T_imu, preintegrated=preintegrated,
            )
        if gt_test:
            self.latest_gt_state_test['prior_source'] = getattr(solver, 'last_prior_source', None)
        if refined is None:
            reason = (
                "initial_shared_points_lt_10"
                if getattr(solver, "last_refinement_failure", None)
                == "initial_shared_points_lt_10"
                else "vi_refinement_failed_other"
            )
            self._fail_frontend(reason)
        if not all(np.isfinite(value).all() for value in (
                refined.pose, refined.velocity, refined.accel_bias, refined.gyro_bias)):
            self._fail_frontend("vi_refinement_failed_other")
        refined_pose = gelsight_pose(refined.pose, self.gelsight_T_imu)
        if variable_reference or reference_marginalization:
            navigation_reference_pose = gelsight_pose(
                solver.last_reference_pose if reference_marginalization
                else solver.last_reference_state.pose, self.gelsight_T_imu)
        if gt_test:
            self.latest_gt_state_test.update(
                optimized_pose=refined.pose.tolist(), optimized_velocity=refined.velocity.tolist(),
            )
        if prediction_only:
            # This diagnostic mode measures the uncorrected IMU trajectory.
            # Do not let visual quality or overlap change State acceptance.
            self.latest_normal_correspondences["status"] = (
                "IMU prediction pose: accepted; SCR/CCS skipped"
            )
        else:
            # The quality gate projects reference pixels into the current image.
            scr, ccs = normalflow_quality(
                reference_frame,
                current_frame,
                np.linalg.inv(refined_pose) @ navigation_reference_pose,
                self.ppmm,
                int(self.imu_config.get("dense_normalflow_quality_samples", 3000)),
            )
            finite = all(np.all(np.isfinite(value)) for value in (
                refined.pose, refined.velocity, refined.accel_bias, refined.gyro_bias,
                [scr, ccs]))
            distance = (solver.prediction_pose_distance(imu_predicted, refined, predicted_prior)
                        if finite and predicted_prior is not None else 0.0)
            consistent = np.isfinite(distance) and distance <= 22.46
            quality_failed = (not finite or not consistent
                              or ccs < float(self.track_config["ccs_threshold"]))
            self._quality_force_keyframe = (
                not quality_failed and scr < float(self.track_config["scr_threshold"]))
            self._record_normal_correspondences(
                reference_frame, current_frame, np.linalg.inv(navigation_reference_pose) @ refined_pose,
                "VI final pose: "
                + ("quality rejected" if quality_failed else "accepted"),
                visual=visual, reference_pose=navigation_reference_pose,
                target_pose=refined_pose,
            )
            if quality_failed:
                self._count_frontend_event("vi_quality_rejected")
                self.logger.info(
                    "Dense normal-flow quality failed: scr=%.3f (min %.3f), "
                    "ccs=%.3f (min %.3f), IMU pose distance=%.6f (max 22.46)"
                    % (
                        scr,
                        float(self.track_config["scr_threshold"]),
                        ccs,
                        float(self.track_config["ccs_threshold"]),
                        distance,
                    )
                )
                trial_T_curr = refined_pose
                if hasattr(solver, "discard_pending_prior"):
                    solver.discard_pending_prior()
                if not np.all(np.isfinite(trial_T_curr)):
                    raise LoseTrackError("dense normal-flow pose is not finite")
                raise QualityGateError(
                    trial_T_curr, refined, scr=scr, ccs=ccs,
                    reason="nonfinite" if not finite else
                    "imu_inconsistent" if not consistent else "ccs_below_threshold")
        if hasattr(solver, "commit_marginal_prior"):
            solver.commit_marginal_prior(refined)
        if reference_marginalization and reference_frame is self.ref_frame:
            # Preserve even the original GelSight matrix's origin roundoff
            # while the exact first-contact datum is the visual reference.
            if solver.reference_id != solver.first_pose_id:
                self.trial_T_ref = navigation_reference_pose.copy()
            reference_pose = self.trial_T_ref
        elif variable_reference:
            # Recovery acceptance can keep this keyframe as the visual reference.
            # Use the endpoint that passed quality checks for subsequent tracking.
            self.trial_T_ref = navigation_reference_pose.copy()
            reference_pose = self.trial_T_ref
        if variable_reference:
            # The pair was validated at the optimized reference endpoint.
            # Send that basis when initializing its successor in local VI.
            from gelslam.core.shared_map import frozen_state
            seeds = getattr(self, "_keyframe_seed_states", {})
            seeds[self.curr_kid] = frozen_state(solver.last_reference_state)
            self._keyframe_seed_states = seeds
        self._count_frontend_event(
            "imu_prediction_only_accepted" if prediction_only else "vi_accepted"
        )
        if previous_state_retry:
            self._count_frontend_event("previous_state_retry_accepted")
        if correction_anchor is not None or variable_reference:
            self._backend_correction_anchor = None
            self._count_frontend_event("backend_feedback_target_accepted")
        self._frontend_vi_state = refined
        self._frontend_imu_preintegration = None
        return (
            np.linalg.inv(reference_pose) @ refined_pose
        ).astype(np.float32)

    def _predict_frontend_imu(self, previous, samples, *, force=False):
        """Reuse a contact-gap interval while preserving its accepted anchor."""
        solver = self.frontend_visual_inertial
        cache = getattr(self, "_frontend_imu_preintegration", None)
        if cache is not None:
            if cache.matches(previous, solver.config, solver.gravity):
                pim = cache.snapshot(samples)
                if pim is not None:
                    self._imu_integration_sample = cache.last_sample
                    return solver.predict_preintegrated(
                        previous, pim, samples[-1].timestamp, force=force,
                    )
            # A GT intervention, changed bias/gravity, or unavailable prefix
            # invalidates the cached interval. The full raw interval is kept.
            self._frontend_imu_preintegration = None
        if force:
            return solver.predict_current(previous, samples, self.gelsight_T_imu, force=True)
        return solver.predict_current(previous, samples, self.gelsight_T_imu)

    def _initialize_no_contact_coast(self):
        """Keep an accepted-anchor preintegration throughout a contact gap."""
        if self.frontend_visual_inertial is not None:
            accepted = getattr(self, "_frontend_vi_state", None)
            if accepted is None or not self._imu_coasting_ready():
                # Do not bypass pending frontend gravity/velocity initialization
                # with a separate reference-frame state and zero velocity.
                return False
            solver = self.frontend_visual_inertial
            cache = getattr(self, "_frontend_imu_preintegration", None)
            if cache is None or not cache.matches(accepted, solver.config, solver.gravity):
                sample = self.imu_buffer.sample_at(accepted.timestamp)
                if sample is None:
                    return False
                cache = ImuPreintegrationAccumulator(
                    accepted, sample, solver.config, solver.gravity,
                )
                self._frontend_imu_preintegration = cache
                self.imu_buffer.begin_online_interval(accepted.timestamp)
            # Pose and velocity belong to the IMU origin, in fixed trial axes.
            # Published tactile poses are converted back to the GelSight origin.
            self._frontend_coast_state = accepted.copy()
            self._prepare_coast_velocity_seed(accepted)
            self._frontend_coast_state = self._apply_coast_velocity_seed(self._frontend_coast_state)
            if cache.last_sample.timestamp > accepted.timestamp:
                prediction = solver.predict_preintegrated(
                    accepted, cache.pim, cache.last_sample.timestamp, force=True,
                )
                if prediction is None:
                    return False
                self._frontend_coast_state = self._apply_coast_velocity_seed(prediction[0])
                self.ref_T_prev = (
                    np.linalg.inv(self.trial_T_ref)
                    @ gelsight_pose(self._frontend_coast_state.pose, self.gelsight_T_imu)
                ).astype(np.float32)
            self._imu_integration_sample = cache.last_sample
            self._coast_start_timestamp = (
                getattr(self, "_no_contact_since", None) or accepted.timestamp)
            return True
        # Preserve the tactile-only frontend's existing IMU coasting path.
        if self.inertial_state is not None and self.inertial_propagator is not None:
            return True
        tactile_pose = getattr(self, "last_tactile_ref_T", None)
        tactile_header = getattr(self, "last_tactile_header", None)
        if tactile_pose is None or tactile_header is None:
            return False
        if not self._anchor_inertial_state(tactile_pose, tactile_header):
            return False
        self._coast_start_timestamp = self.inertial_state.timestamp

        previous_pose = getattr(self, "previous_tactile_ref_T", None)
        previous_header = getattr(self, "previous_tactile_header", None)
        previous_time = self._camera_timestamp(previous_header)
        current_time = self._camera_timestamp(tactile_header)
        if (
            previous_pose is not None
            and previous_time is not None
            and current_time is not None
            and current_time > previous_time
        ):
            previous_imu_position = (
                np.asarray(previous_pose, dtype=np.float64) @ self.gelsight_T_imu
            )[:3, 3]
            velocity = (
                self.inertial_state.position - previous_imu_position
            ) / (current_time - previous_time)
            self.inertial_state = InertialState(
                timestamp=self.inertial_state.timestamp,
                rotation=self.inertial_state.rotation,
                position=self.inertial_state.position,
                velocity=velocity,
                accel_bias=self.inertial_state.accel_bias,
                gyro_bias=self.inertial_state.gyro_bias,
            )
        return True

    def _clear_contact_velocity_history(self):
        history = getattr(self, "_contact_velocity_history", None)
        if history is not None:
            history.clear()
        self._coast_velocity_seed = None
        self._coast_velocity_anchor_timestamp = None

    def _prepare_coast_velocity_seed(self, accepted):
        """Fit once at the accepted contact anchor; freeze it throughout the gap."""
        history = getattr(self, "_contact_velocity_history", None)
        if history is None or getattr(self, "_coast_velocity_anchor_timestamp", None) == accepted.timestamp:
            return
        seed = history.seed(accepted, self.frontend_visual_inertial.gravity)
        self._coast_velocity_seed = seed
        self._coast_velocity_anchor_timestamp = accepted.timestamp
        self.latest_coast_velocity_fit = dict(anchor_timestamp_s=accepted.timestamp,
            status=history.last_status,
            velocity_delta_mps=None if seed is None else seed.velocity_delta.tolist(),
            fit_rmse_m=None if seed is None else seed.fit_rmse_m,
            samples=None if seed is None else seed.samples,
            span_s=None if seed is None else seed.span_s)
        # A later contact history must not span any no-contact image.
        history.clear()
        if seed is not None:
            self.logger.info("Coast velocity fit at %.9f s: delta=%s m/s, rmse=%.9f m, samples=%d"
                             % (accepted.timestamp, seed.velocity_delta, seed.fit_rmse_m, seed.samples))

    def _apply_coast_velocity_seed(self, predicted):
        seed = getattr(self, "_coast_velocity_seed", None)
        return predicted if seed is None else seed.apply(predicted)

    @staticmethod
    def _config_vector(value, name):
        """Return an optional finite three-vector from the IMU configuration."""
        if value is None:
            return None
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"imu_config.{name} must be a finite three-vector")
        return vector

    def _anchor_inertial_state(self, ref_T_gelsight, header):
        """Start tactile-only IMU propagation at a successful tactile pose.

        The tracker reference keyframe is the local navigation world.  The
        supplied transform therefore maps GelSight coordinates into that world.
        The anchor takes the tactile rotation and position with zero velocity,
        then initializes gravity and the online IMU interval.  IMU propagation
        is disabled until a calibrated mounting transform and a bracketing IMU
        sample are available.
        """
        timestamp = self._imu_timestamp(header)
        camera_timestamp = self._camera_timestamp(header)
        if not self._imu_coasting_ready() or timestamp is None:
            return False
        sample = self.imu_buffer.sample_at(timestamp)
        if sample is None:
            # ROS dispatch can deliver an image before the IMU sample with the
            # same sensor timestamp.  Use the latest already received sample
            # rather than abandoning the active track for that one callback.
            sample = self.imu_buffer.latest_at_or_before(timestamp)
        if sample is None:
            return False

        ref_T_gelsight = np.asarray(ref_T_gelsight, dtype=np.float64)
        ref_T_imu = ref_T_gelsight @ self.gelsight_T_imu
        # Biases use IMU axes in both contact and no-contact states.
        # Preserve the accepted frontend bias at this tactile anchor.
        bias_state = self.inertial_state
        frontend_state = getattr(self, "_frontend_vi_state", None)
        if (
            bias_state is None
            and frontend_state is not None
            and abs(frontend_state.timestamp - timestamp) <= 1e-9
        ):
            bias_state = frontend_state
        accel_bias = (
            bias_state.accel_bias
            if bias_state is not None
            else self._config_vector(
                self.imu_config.get("initial_accel_bias_mps2"),
                "initial_accel_bias_mps2",
            )
        )
        gyro_bias = (
            bias_state.gyro_bias
            if bias_state is not None
            else self._config_vector(
                self.imu_config.get("initial_gyro_bias_rps"),
                "initial_gyro_bias_rps",
            )
        )
        if accel_bias is None:
            accel_bias = np.zeros(3, dtype=np.float64)
        if gyro_bias is None:
            gyro_bias = np.zeros(3, dtype=np.float64)
        self.inertial_state = InertialState(
            timestamp=sample.timestamp,
            rotation=ref_T_imu[:3, :3],
            position=ref_T_imu[:3, 3],
            velocity=np.zeros(3, dtype=np.float64),
            accel_bias=accel_bias,
            gyro_bias=gyro_bias,
        )
        self._imu_integration_sample = sample
        gravity_magnitude = float(self.imu_config.get("gravity_mps2", 9.81))
        gravity_initialization = str(
            self.imu_config.get("gravity_initialization", "measured_acceleration")
        )
        if gravity_initialization == "fixed_world_z":
            gravity = self.trial_T_ref[:3, :3].T @ np.array(
                [0.0, 0.0, -gravity_magnitude]
            )
        elif gravity_initialization in {
            "ground_truth_initial_pose", "ground_truth_gel_pose"
        }:
            anchor_timestamp = (
                getattr(self, "_frontend_trial_start_timestamp", None)
                or camera_timestamp
            )
            world_R_gel = self._trial_initial_gt_rotation(anchor_timestamp)
            if world_R_gel is None:
                self.inertial_state = None
                return False
            # Propagation uses the current reference keyframe as its local
            # world.  Convert trial gravity into those reference axes.
            trial_gravity = gravity_from_initial_world_R_gel(
                world_R_gel, gravity_magnitude
            )
            gravity = self.trial_T_ref[:3, :3].T @ trial_gravity
        elif gravity_initialization == "measured_acceleration":
            gravity_window_s = float(
                self.imu_config.get("gravity_initialization_window_s", 0.0)
            )
            samples = ()
            if gravity_window_s > 0.0 and len(self.imu_buffer):
                buffered_samples = self.imu_buffer.samples_between(0.0, timestamp)
                first_timestamp = buffered_samples[0].timestamp
                samples = self.imu_buffer.samples_between(
                    first_timestamp,
                    min(first_timestamp + gravity_window_s, timestamp),
                )
            measured_acceleration = (
                np.mean([item.acceleration for item in samples], axis=0)
                if samples
                else sample.acceleration
            )
            measured_gravity = ref_T_imu[:3, :3] @ (
                measured_acceleration - accel_bias
            )
            norm = np.linalg.norm(measured_gravity)
            if norm <= 1e-9:
                self.inertial_state = None
                return False
            gravity = -gravity_magnitude * measured_gravity / norm
        else:
            raise ValueError(
                "imu_config.gravity_initialization must be "
                "'measured_acceleration', 'fixed_world_z', or "
                "'ground_truth_initial_pose'"
            )
        self.inertial_propagator = InertialPropagator(gravity)
        # From here the image callback consumes each IMU sample once.  The
        # buffer keeps only the boundary sample and future measurements,
        # rather than being searched from its oldest retained entry for
        # every image timestamp.
        self.imu_buffer.begin_online_interval(self.inertial_state.timestamp)
        return True

    def _coast_to_header(self, header, mode):
        """Predict a contact-gap pose; keep the accepted visual state unchanged."""
        timestamp = self._imu_timestamp(header)
        frontend_coast = getattr(self, "_frontend_coast_state", None)
        state = frontend_coast if frontend_coast is not None else self.inertial_state
        if (
            not self._imu_coasting_ready()
            or state is None
            or (frontend_coast is None and self.inertial_propagator is None)
            or timestamp is None
        ):
            return False
        max_coast = float(self.imu_config.get("max_coast_duration_s", 1.0))
        if not np.isfinite(max_coast) or max_coast <= 0.0:
            raise ValueError("imu_config.max_coast_duration_s must be positive")
        coast_start = getattr(self, "_coast_start_timestamp", None)
        if coast_start is None:
            coast_start = state.timestamp
            self._coast_start_timestamp = coast_start
        if timestamp - coast_start > max_coast:
            self._arm_restart(timestamp, missing_imu=False)
            self.logger.warning(
                "Stopping IMU coast after %.3f s without tactile contact" % max_coast
            )
            return False
        start_sample = getattr(self, "_imu_integration_sample", None)
        if start_sample is None:
            # Compatibility for trackers loaded from states written before the
            # online-consumption cursor was introduced.
            start_sample = self.imu_buffer.sample_at(state.timestamp)
            if start_sample is None:
                start_sample = self.imu_buffer.latest_at_or_before(
                    state.timestamp
                )
            if start_sample is None:
                return False
            self.imu_buffer.begin_online_interval(start_sample.timestamp)
            self._imu_integration_sample = start_sample
        frontend_state = getattr(self, "_frontend_vi_state", None)
        samples = self.imu_buffer.consume_online_interval(
            start_sample, timestamp,
            retain_from_timestamp=(
                None if frontend_state is None else frontend_state.timestamp
            ),
        )
        if len(samples) < 2:
            # Keep the latest predicted pose until the next IMU callback
            # arrives.  This avoids opening a new tactile trial solely because
            # DDS dispatched the image before its contemporaneous IMU sample.
            self.tracking_mode = mode
            return True
        if frontend_coast is not None:
            solver = self.frontend_visual_inertial
            cache = self._frontend_imu_preintegration
            if not cache.append(samples):
                return False
            # Always predict from the accepted anchor with the full interval.
            # Reading the prediction does not alter the active marginal prior.
            prediction = solver.predict_preintegrated(
                cache.anchor, cache.pim, cache.last_sample.timestamp, force=True,
            )
            if prediction is None:
                return False
            propagated, _ = prediction
            propagated = self._apply_coast_velocity_seed(propagated)
            self._frontend_coast_state = propagated
            self.ref_T_prev = (
                np.linalg.inv(self.trial_T_ref)
                @ gelsight_pose(propagated.pose, self.gelsight_T_imu)
            ).astype(np.float32)
        else:
            propagated = self.inertial_propagator.propagate(state, samples)
            if propagated is None:
                return False
            self.inertial_state = propagated
            ref_T_imu = np.eye(4, dtype=np.float64)
            ref_T_imu[:3, :3] = propagated.rotation
            ref_T_imu[:3, 3] = propagated.position
            self.ref_T_prev = (ref_T_imu @ np.linalg.inv(self.gelsight_T_imu)).astype(
                np.float32
            )
        self._imu_integration_sample = samples[-1]
        self.tracking_mode = mode
        return True

    def _coasting_frame_result(self, fid, header):
        """Publish the last reliable tactile surface at its IMU-predicted pose."""
        if self.prev_frame is None or self.curr_kid < 0:
            return False, None
        state = getattr(self, "_frontend_coast_state", None)
        if state is None:
            state = self.inertial_state
        target = self._imu_timestamp(header)
        pose_time = state.timestamp if state is not None else target
        valid = pose_time is not None and target is not None and abs(pose_time - target) < 1e-8
        frame_msg = self._make_frame_msg(
            self.prev_frame,
            fid=fid,
            ref_kid=self.curr_kid,
            ref_T_curr=self.ref_T_prev,
            header=header,
            tracking_status=self.tracking_mode.value, pose_valid=valid,
            pose_timestamp=(self._camera_timestamp(header) - (target - pose_time)
                            if target is not None and pose_time is not None else 0.0),
        )
        return True, (frame_msg, [])

    def add_imu_sample(self, sample):
        """Buffer one calibrated IMU sample received by the ROS tracker node.

        Samples remain independent of tactile tracking until camera/IMU timing
        and the IMU-to-GelSight transform are calibrated. ``False`` means
        IMU input is disabled or its timestamp is not increasing.
        """
        if not self.imu_enabled or self.imu_buffer is None:
            return False
        if not isinstance(sample, ImuSample):
            raise TypeError("sample must be an ImuSample")
        try:
            self.imu_buffer.add(sample)
        except ValueError:
            return False
        if getattr(self, "_imu_stream_start_time", None) is None:
            self._imu_stream_start_time = sample.timestamp
        # Frontend coasting prunes its own buffer. Preserve the independent
        # keyframe-to-keyframe interval until it has been serialized.
        if getattr(self, "_backend_imu_buffer", None) is None:
            self._backend_imu_buffer = ImuBuffer(
                max_samples=int(self.imu_config.get("buffer_max_samples", 10000))
            )
        self._backend_imu_buffer.add(sample)
        return True

    def _make_reconstructor(self, calib_model_path):
        """Set process-wide Torch CPU threads before model initialization."""
        threads = getattr(self, "torch_num_threads", 4)
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError("device_config.torch_num_threads must be a positive integer")
        self.torch_num_threads = threads
        torch.set_num_threads(threads)
        self.logger.info(
            "Surface reconstruction Torch CPU intra-op threads: %d"
            % torch.get_num_threads()
        )
        return Reconstructor(calib_model_path)

    def save(self, save_dir):
        """
        Save the tracker state to a directory.

        :param save_dir: str; The directory to save to.
        """
        # Exclude non-picklable objects
        self._map_snapshot_provider = None
        self._tracking_map_snapshot = None
        self._tracking_anchor_state = None
        self._keyframe_seed_states = {}
        self._backend_correction_anchor = None
        self._backend_feedback_anchor = None
        self.logger = None
        self.bridge = None
        self.recon = None
        # Recreate the small direct solver from configuration when loading.
        self.frontend_visual_inertial = None
        self._frontend_vi_state = None
        self._frontend_coast_state = None
        self._frontend_frame_prediction = None
        self._frontend_imu_preintegration = None
        self._backend_imu_buffer = None
        self._backend_keyframe_timestamp = None
        # Pickle remaining state
        with open(os.path.join(save_dir, "tracker.pkl"), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, load_dir, calib_model_path, logger):
        """
        Load the tracker state from a directory.

        :param load_dir: str; The directory to load from.
        :param calib_model_path: str; The path to the calibration model.
        :param logger: Logger; The logger object.
        :return: Tracker; The loaded tracker.
        """
        # Load pickled state
        with open(os.path.join(load_dir, "tracker.pkl"), "rb") as f:
            instance = pickle.load(f)
        if getattr(instance, "_frontend_state_frame", None) != "imu":
            if (getattr(instance, "_frontend_vi_state", None) is not None
                    or getattr(instance, "_frontend_coast_state", None) is not None):
                raise ValueError("saved tracker uses GelSight navigation states; start a new trial with IMU states")
            instance._frontend_state_frame = "imu"
        # Reconstruct non-picklable objects
        instance.logger = logger
        # Counters describe this run, not the run that wrote the saved state.
        instance._frontend_tracking_counts = Counter()
        instance.bridge = CvBridge()
        instance.recon = instance._make_reconstructor(calib_model_path)
        instance.recon.load_bg(instance.bg_image)
        # Preserve compatibility with tracker states saved before timestamping.
        if not hasattr(instance, "prev_header"):
            instance.prev_header = None
        if not hasattr(instance, "prev_fid"):
            instance.prev_fid = None
        if not hasattr(instance, "_consecutive_tracking_failures"):
            instance._consecutive_tracking_failures = 0
        if not hasattr(instance, "ref_header"):
            instance.ref_header = None
        if not hasattr(instance, "last_tactile_ref_T"):
            instance.last_tactile_ref_T = None
        if not hasattr(instance, "last_tactile_header"):
            instance.last_tactile_header = instance.prev_header
        if not hasattr(instance, "previous_tactile_ref_T"):
            instance.previous_tactile_ref_T = None
        if not hasattr(instance, "previous_tactile_header"):
            instance.previous_tactile_header = None
        if not hasattr(instance, "tracking_mode"):
            instance.tracking_mode = TrackingMode.NO_CONTACT_COASTING
        if not hasattr(instance, "imu_config"):
            instance.imu_config = {}
        if not hasattr(instance, "timestamp_source"):
            instance.timestamp_source = instance.imu_config.get("timestamp_source", "header")
        if not hasattr(instance, "_recorded_image_times"):
            instance._recorded_image_times = {}
        if not hasattr(instance, "imu_enabled"):
            instance.imu_enabled = False
        if not hasattr(instance, "imu_buffer"):
            instance.imu_buffer = (
                ImuBuffer(
                    max_samples=int(instance.imu_config.get("buffer_max_samples", 6000))
                )
                if instance.imu_enabled
                else None
            )
        if not hasattr(instance, "gelsight_T_imu"):
            instance.gelsight_T_imu = cls._parse_gelsight_T_imu(
                instance.imu_config.get("gelsight_T_imu")
            )
        instance.imu_config, instance._static_imu_settings, _ = resolve_static_imu_config(
            instance.imu_config, instance.gelsight_T_imu
        )
        instance._static_imu_initializer = (
            StaticImuInitializer(instance._static_imu_settings, instance.imu_config)
            if instance._static_imu_settings is not None else None
        )
        instance._static_imu_collecting = instance.curr_kid < 0
        instance._static_imu_collection_start = 0.0
        if not hasattr(instance, "_imu_stream_start_time"):
            instance._imu_stream_start_time = None
        if not hasattr(instance, "_recorded_rest_events"):
            instance._recorded_rest_events = None
        instance._static_trial_seed = None
        instance._static_imu_status_logged = None
        instance._static_imu_contact_cutoff = None
        instance._contact_velocity_history = ContactVelocityHistory.from_config(instance.imu_config)
        instance._coast_velocity_seed = None
        instance._coast_velocity_anchor_timestamp = None
        instance.latest_coast_velocity_fit = None
        # Pending input images are not resumed across saved-state loading.
        instance._pending_trial_contact = None
        if not hasattr(instance, "_trial_gravity_rotation"):
            instance._trial_gravity_rotation = None
        if not hasattr(instance, "trial_T_ref"):
            if instance.curr_kid >= 0:
                raise ValueError(
                    "saved tracker lacks the trial-frame keyframe pose; "
                    "start a new trial with the current state format"
                )
            instance.trial_T_ref = np.eye(4, dtype=np.float64)
        if not hasattr(instance, "_trial_index"):
            instance._trial_index = 0 if instance.curr_kid >= 0 else -1
        if not hasattr(instance, "inertial_state"):
            instance.inertial_state = None
        if not hasattr(instance, "initial_ground_truth_world_R_gel"):
            instance.initial_ground_truth_world_R_gel = None
        if not hasattr(instance, "_ground_truth_gel_positions"):
            instance._ground_truth_gel_positions = []
        if not hasattr(instance, "_ground_truth_imu_positions"):
            instance._ground_truth_imu_positions = []
        if not hasattr(instance, "_frontend_trial_start_timestamp"):
            instance._frontend_trial_start_timestamp = None
        if not hasattr(instance, "_trial_start_world_R_gel"):
            instance._trial_start_world_R_gel = None
        if not hasattr(instance, "inertial_propagator"):
            instance.inertial_propagator = None
        if not hasattr(instance, "_imu_integration_sample"):
            instance._imu_integration_sample = None
        if getattr(instance, "frontend_visual_inertial", None) is None:
            instance.frontend_visual_inertial = instance._make_frontend_solver(
                instance.imu_config
            )
        solver = instance.frontend_visual_inertial
        if solver is not None:
            # Legacy files retain rows/RHS. Keep their anchor and information,
            # but apply the current zero-gradient stored-prior convention.
            prior = getattr(solver, "marginal_prior", None)
            if prior is not None and not getattr(solver, "reference_marginalization", False):
                solver.marginal_prior = NavigationPrior.from_information(
                    prior.anchor, prior.A.T @ prior.A
                )
            solver.pending_marginal_prior = None
        if not hasattr(instance, "_frontend_vi_state"):
            instance._frontend_vi_state = None
        if not hasattr(instance, "_frontend_coast_state"):
            instance._frontend_coast_state = None
        instance._frontend_frame_prediction = None
        instance._frontend_imu_preintegration = None
        if hasattr(instance, "_frontend_nav_T_trial"):
            if instance._frontend_vi_state is not None:
                raise ValueError(
                    "saved tracker uses aligned VI axes; start a new trial "
                    "with trial-frame VI state"
                )
            del instance._frontend_nav_T_trial
        return instance

    def track(self, image, header=None, log_prefix="Track"):
        """
        Tracks the tactile image.

        :param image: np.ndarray; The tactile image.
        :param header: std_msgs.msg.Header; Source image header, if available.
        :param log_prefix: str; The prefix for logging.
        :return: (bool, tuple or None);
            - bool: Validity of the result. Returns False if collecting background or not in contact.
            - tuple or None: (frame_msg, keyframe_msgs) if valid, else None.
                - frame_msg (FrameMsg or None): None when a failed image is
                  dropped after the prior frame becomes a keyframe.
                - keyframe_msgs (list[KeyFrameMsg]): List of new keyframes.
        """
        snapshot_provider = getattr(self, "_map_snapshot_provider", None)
        if snapshot_provider is not None:
            self.apply_map_snapshot(snapshot_provider(), header)

        self.latest_normal_correspondences = None
        self._frontend_frame_prediction = None
        if resolve_gt_state_test(self.imu_config)['enabled']:
            timestamp = self._imu_timestamp(header)
            self._gt_test_previous_image_timestamp = self._gt_test_last_image_timestamp
            self._gt_test_last_image_timestamp = timestamp
            self.latest_gt_state_test = None
        if len(self.bg_images) < 10:
            self._observe_static_imu(header)
            # Collect background images
            self.bg_images.append(image)
            if len(self.bg_images) == 10:
                self.logger.info("%s -- Background images collected" % log_prefix)

                self.bg_image = np.mean(self.bg_images, axis=0)
                # Background check by user
                if not self.skip_background_check:
                    display_image = self.bg_image.copy().astype(np.uint8)
                    display_image = cv2.resize(
                        display_image,
                        (display_image.shape[1] * 3, display_image.shape[0] * 3),
                    )
                    cv2.putText(
                        display_image,
                        "Ensure no contact. Press 'y' to confirm, 'n' to reject.",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("Background image verification", display_image)
                    while True:
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord("n"):
                            raise SystemExit(
                                "User aborted: contact detected in background image."
                            )
                        elif key == ord("y"):
                            break
                    cv2.destroyWindow("Background image verification")
                # Load the confirmed background
                self.recon.load_bg(self.bg_image)
            return False, None
        else:
            # Get surface information
            G, H, C = self.recon.get_surface_info(
                image,
                self.ppmm,
                color_dist_threshold=self.contact_mask_config["color_dist_threshold"],
                height_threshold=self.contact_mask_config["height_threshold"],
            )
            frame = Frame(
                G,
                H,
                C,
                contact_threshold=self.track_config["contact_threshold"],
            )
            # Keep the actual image surface for RViz even if pose tracking
            # rejects this image or publishes a coasted prior surface.
            self.latest_observation_frame = frame
            # Skip if not in contact
            if not frame.is_contacted:
                self._observe_static_imu(header)
                self._consecutive_tracking_failures = 0
                self._contact_failure_since = None
                if getattr(self, "_no_contact_since", None) is None:
                    self._no_contact_since = self._imu_timestamp(header)
                self.logger.info("%s -- Skip Frame: Not in contact" % log_prefix)
                self.prev_in_contact_flag = False
                self.curr_fid += 1
                if (
                    self._initialize_no_contact_coast()
                    and self._coast_to_header(header, TrackingMode.NO_CONTACT_COASTING)
                ):
                    self.logger.info(
                        "%s -- IMU Coast: no tactile contact, frame ID: %d"
                        % (log_prefix, self.curr_fid)
                    )
                    return self._coasting_frame_result(self.curr_fid, header)
                self.curr_fid -= 1
                self.tracking_mode = TrackingMode.NO_CONTACT_COASTING
                return False, None
            else:
                self._no_contact_since = None
                self.curr_fid += 1
                if getattr(self, "_restart_pending", False):
                    return self._restart_contact(frame, header)
                keyframe_msgs = []
                if not self.prev_in_contact_flag:
                    self.tracking_mode = TrackingMode.REACQUIRE
                    # Contact tracking keeps a tactile reference; the VI
                    # solver predicts this frame from IMU before refinement.
                    # The separate no-contact coast state is not advanced here.
                    if self.ref_frame is not None and (
                        self.inertial_state is not None
                        or getattr(self, "_frontend_coast_state", None) is not None
                        # Backend feedback invalidates the temporary coast
                        # prediction, but retains the accepted VI state.
                        # That state still belongs to the initialized trial.
                        or (self.frontend_visual_inertial is not None
                            and getattr(self, "_frontend_vi_state", None) is not None)
                    ):
                        try:
                            self.ref_T_prev = self._track_contact(frame, header)
                            self.prev_in_contact_flag = True
                            self.prev_is_keyframe_flag = False
                            self.prev_frame, self.prev_header = frame, header
                            self._record_tactile_pose(self.ref_T_prev, header)
                            self.tracking_mode = TrackingMode.CONTACT_TRACKING
                        except LoseTrackError as error:
                            # Recontact does not start a new quality-rejection
                            # wait, even if this attempt reached the quality gate.
                            return self._handle_tracking_failure(
                                frame, header, log_prefix, str(error), promote_previous=False)
                    if not self.prev_in_contact_flag:
                        if (getattr(self, "_static_imu_settings", None) is not None
                                and not self._prepare_static_trial(header)):
                            # No tactile anchor or keyframe may precede the IMU seed.
                            self.curr_fid -= 1
                            self.tracking_mode = TrackingMode.NO_CONTACT_COASTING
                            return False, None
                        initialized = self._start_trial_contact(frame, header)
                        if initialized is None:
                            self.curr_fid -= 1
                            self.tracking_mode = TrackingMode.NO_CONTACT_COASTING
                            return False, None
                        frame, header, initialized_frontend = initialized
                        self.logger.info(
                            "%s -- New Trial: New contact initiated, keyframe ID: %d"
                            % (log_prefix, self.curr_kid + 1)
                        )
                        # New contact detected, start a new trial.
                        self.curr_kid += 1
                        keyframe_msg = self._make_keyframe_msg(
                            frame, self.curr_kid, self.curr_fid, True,
                            self.trial_T_ref.copy(), header,
                        )
                        keyframe_msgs.append(keyframe_msg)
                        self.prev_in_contact_flag = True
                        self.prev_is_keyframe_flag = True
                        self.ref_T_prev = np.eye(4, dtype=np.float32)
                        self.prev_frame = frame
                        self.prev_header = header
                        self.ref_frame = frame
                        self.ref_header = header
                        self._record_tactile_pose(
                            self.ref_T_prev, header, reference_changed=True,
                            preserve_frontend_state=initialized_frontend,
                        )
                        self.tracking_mode = TrackingMode.CONTACT_TRACKING
                else:
                    try:
                        # Current frame is tracked successfully (not a keyframe)
                        self.prev_in_contact_flag = True
                        self.ref_T_prev = self._track_contact(frame, header)
                        self.prev_is_keyframe_flag = False
                        self.prev_frame = frame
                        self.prev_header = header
                        self._record_tactile_pose(self.ref_T_prev, header)
                        self.tracking_mode = TrackingMode.CONTACT_TRACKING
                        self.logger.info(
                            "%s -- Regular Frame: frame ID: %d"
                            % (log_prefix, self.curr_fid)
                        )

                    except QualityGateError:
                        return self._handle_tracking_failure(
                            frame, header, log_prefix, "SCR/CCS failure",
                            quality_failure=True,
                        )
                    except LoseTrackError:
                        # Without a usable tactile solve, advance from the
                        # accepted state using IMU when its interval is valid.
                        return self._handle_tracking_failure(
                            frame, header, log_prefix, "tracking failure"
                        )

                # Create current frame message
                if not keyframe_msgs:
                    timed_keyframe = self._maybe_add_timed_keyframe(
                        frame, header, log_prefix,
                        force=getattr(self, "_quality_force_keyframe", False))
                    if timed_keyframe is not None:
                        keyframe_msgs.append(timed_keyframe)
                self._consecutive_tracking_failures = 0
                self.prev_fid = self.curr_fid
                frame_msg = self._make_frame_msg(
                    frame,
                    fid=self.curr_fid,
                    ref_kid=self.curr_kid,
                    ref_T_curr=self.ref_T_prev,
                    header=header,
                    pose_timestamp=self._camera_timestamp(header),
                )

                return True, (frame_msg, keyframe_msgs)
