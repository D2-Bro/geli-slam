import os
import pickle
from enum import Enum

import cv2
import numpy as np
from cv_bridge import CvBridge
from gs_sdk.gs_reconstruct import Reconstructor
from normalflow.registration import LoseTrackError, normalflow
from normalflow.utils import Frame

from gelslam.core.frame import create_frame_msg, create_keyframe_msg
from gelslam.core.inertial import (
    ImuBuffer,
    ImuSample,
    InertialPropagator,
    InertialState,
)


class TrackingMode(Enum):
    """Tactile/IMU lifecycle states for the tracker."""

    CONTACT_TRACKING = "contact_tracking"
    NO_CONTACT_COASTING = "no_contact_coasting"
    IMU_COASTING = "imu_coasting"
    REACQUIRE = "reacquire"


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
        self.ppmm = config["device_config"]["ppmm"]
        self.contact_mask_config = config["contact_mask_config"]
        self.track_config = config["gelslam_config"]["track_config"]
        self.imu_config = config.get("imu_config", {})
        self.imu_enabled = bool(self.imu_config.get("enabled", False))
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
        self.inertial_state = None
        self.inertial_propagator = None
        self.bridge = CvBridge()
        # Initialize reconstructor
        self.recon = Reconstructor(calib_model_path)
        # Skip background check flag
        self.skip_background_check = skip_background_check

        # Store initial images as background
        self.bg_images = []
        # Tracking state variables
        self.prev_in_contact_flag = False
        self.prev_is_keyframe_flag = False
        self.ref_T_prev = np.eye(4, dtype=np.float32)
        self.prev_frame = None
        self.prev_header = None
        self.curr_kid = -1
        self.ref_frame = None
        self.ref_header = None
        self.curr_fid = -1
        # IMU propagation will use this lifecycle after extrinsic/time calibration.
        self.tracking_mode = TrackingMode.NO_CONTACT_COASTING

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

    def _imu_coasting_ready(self):
        return self.gelsight_T_imu is not None and self.imu_buffer is not None

    def _anchor_inertial_state(self, ref_T_gelsight, header, reset_reference=False):
        """Correct the navigation state with a successful tactile pose estimate.

        The tracker reference keyframe is the local navigation world.  The
        supplied transform therefore maps GelSight coordinates into that world.
        IMU propagation is disabled until a calibrated mounting transform and a
        bracketing IMU sample are available.
        """
        timestamp = self._header_timestamp(header)
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
        previous_state = None if reset_reference else self.inertial_state
        velocity = np.zeros(3, dtype=np.float64)
        if previous_state is not None and timestamp > previous_state.timestamp:
            velocity = (ref_T_imu[:3, 3] - previous_state.position) / (
                timestamp - previous_state.timestamp
            )
        self.inertial_state = InertialState(
            timestamp=sample.timestamp,
            rotation=ref_T_imu[:3, :3],
            position=ref_T_imu[:3, 3],
            velocity=velocity,
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
        )
        if self.inertial_propagator is None or reset_reference:
            gravity_magnitude = float(self.imu_config.get("gravity_mps2", 9.81))
            measured_gravity = ref_T_imu[:3, :3] @ sample.acceleration
            norm = np.linalg.norm(measured_gravity)
            if norm <= 1e-9:
                self.inertial_state = None
                return False
            self.inertial_propagator = InertialPropagator(
                -gravity_magnitude * measured_gravity / norm
            )
        return True

    def _coast_to_header(self, header, mode):
        """Propagate the last tactile-anchored state to an image timestamp."""
        timestamp = self._header_timestamp(header)
        if (
            not self._imu_coasting_ready()
            or self.inertial_state is None
            or self.inertial_propagator is None
            or timestamp is None
        ):
            return False
        target_sample = self.imu_buffer.sample_at(timestamp)
        if target_sample is None:
            target_sample = self.imu_buffer.latest_at_or_before(timestamp)
        if target_sample is None or target_sample.timestamp <= self.inertial_state.timestamp:
            # Keep the latest predicted pose until the next IMU callback
            # arrives.  This avoids opening a new tactile trial solely because
            # DDS dispatched the image before its contemporaneous IMU sample.
            self.tracking_mode = mode
            return True
        samples = self.imu_buffer.samples_covering(
            self.inertial_state.timestamp, target_sample.timestamp
        )
        propagated = self.inertial_propagator.propagate(self.inertial_state, samples)
        if propagated is None:
            return False
        self.inertial_state = propagated
        ref_T_imu = np.eye(4, dtype=np.float64)
        ref_T_imu[:3, :3] = propagated.rotation
        ref_T_imu[:3, 3] = propagated.position
        self.ref_T_prev = (ref_T_imu @ np.linalg.inv(self.gelsight_T_imu)).astype(
            np.float32
        )
        self.tracking_mode = mode
        return True

    def _coasting_frame_result(self, fid, header):
        """Publish the last reliable tactile surface at its IMU-predicted pose."""
        if self.prev_frame is None or self.curr_kid < 0:
            return False, None
        frame_msg = create_frame_msg(
            self.bridge,
            self.prev_frame,
            fid=fid,
            ref_kid=self.curr_kid,
            ref_T_curr=self.ref_T_prev,
            header=header,
        )
        return True, (frame_msg, [])

    def add_imu_sample(self, sample):
        """Buffer one calibrated IMU sample received by the ROS tracker node.

        Samples remain independent of tactile tracking until camera/IMU timing
        and the IMU-to-GelSight transform are calibrated.  ``False`` means the
        timestamp was duplicate or out of order and the caller safely dropped it.
        """
        if not self.imu_enabled or self.imu_buffer is None:
            return False
        if not isinstance(sample, ImuSample):
            raise TypeError("sample must be an ImuSample")
        try:
            self.imu_buffer.add(sample)
        except ValueError:
            return False
        return True

    def save(self, save_dir):
        """
        Save the tracker state to a directory.

        :param save_dir: str; The directory to save to.
        """
        # Exclude non-picklable objects
        self.logger = None
        self.bridge = None
        self.recon = None
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
        # Reconstruct non-picklable objects
        instance.logger = logger
        instance.bridge = CvBridge()
        instance.recon = Reconstructor(calib_model_path)
        instance.recon.load_bg(instance.bg_image)
        # Preserve compatibility with tracker states saved before timestamping.
        if not hasattr(instance, "prev_header"):
            instance.prev_header = None
        if not hasattr(instance, "ref_header"):
            instance.ref_header = None
        if not hasattr(instance, "tracking_mode"):
            instance.tracking_mode = TrackingMode.NO_CONTACT_COASTING
        if not hasattr(instance, "imu_config"):
            instance.imu_config = {}
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
        if not hasattr(instance, "inertial_state"):
            instance.inertial_state = None
        if not hasattr(instance, "inertial_propagator"):
            instance.inertial_propagator = None
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
                - frame_msg (FrameMsg): Current frame information.
                - keyframe_msgs (list[KeyFrameMsg]): List of new keyframes.
        """
        if len(self.bg_images) < 10:
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
            # Skip if not in contact
            if not frame.is_contacted:
                self.logger.info("%s -- Skip Frame: Not in contact" % log_prefix)
                self.prev_in_contact_flag = False
                self.curr_fid += 1
                if self._coast_to_header(header, TrackingMode.NO_CONTACT_COASTING):
                    self.logger.info(
                        "%s -- IMU Coast: no tactile contact, frame ID: %d"
                        % (log_prefix, self.curr_fid)
                    )
                    return self._coasting_frame_result(self.curr_fid, header)
                self.curr_fid -= 1
                self.tracking_mode = TrackingMode.NO_CONTACT_COASTING
                return False, None
            else:
                self.curr_fid += 1
                keyframe_msgs = []
                if not self.prev_in_contact_flag:
                    self.tracking_mode = TrackingMode.REACQUIRE
                    if self.ref_frame is not None and self.inertial_state is not None:
                        try:
                            curr_T_ref = normalflow(
                                self.ref_frame.N,
                                self.ref_frame.C,
                                self.ref_frame.H,
                                self.ref_frame.L,
                                frame.N,
                                frame.C,
                                frame.H,
                                frame.L,
                                np.linalg.inv(self.ref_T_prev),
                                self.ppmm,
                                scr_threshold=self.track_config["scr_threshold"],
                                ccs_threshold=self.track_config["ccs_threshold"],
                            )
                            self.prev_in_contact_flag = True
                            self.prev_is_keyframe_flag = False
                            self.ref_T_prev = np.linalg.inv(curr_T_ref)
                            self.prev_frame = frame
                            self.prev_header = header
                            self._anchor_inertial_state(self.ref_T_prev, header)
                            self.tracking_mode = TrackingMode.CONTACT_TRACKING
                            self.logger.info(
                                "%s -- IMU Reacquire: frame ID: %d"
                                % (log_prefix, self.curr_fid)
                            )
                        except LoseTrackError:
                            if self._coast_to_header(header, TrackingMode.IMU_COASTING):
                                self.logger.info(
                                    "%s -- IMU Coast: tactile reacquire failed, frame ID: %d"
                                    % (log_prefix, self.curr_fid)
                                )
                                return self._coasting_frame_result(self.curr_fid, header)
                            self.ref_frame = None
                    if not self.prev_in_contact_flag:
                        self.logger.info(
                            "%s -- New Trial: New contact initiated, keyframe ID: %d"
                            % (log_prefix, self.curr_kid + 1)
                        )
                        # New contact detected, start a new trial.
                        self.curr_kid += 1
                        keyframe_msg = create_keyframe_msg(
                            self.bridge,
                            frame,
                            kid=self.curr_kid,
                            fid=self.curr_fid,
                            is_new_trial=True,
                            ref_T_curr=np.eye(4, dtype=np.float32),
                            header=header,
                        )
                        keyframe_msgs.append(keyframe_msg)
                        self.prev_in_contact_flag = True
                        self.prev_is_keyframe_flag = True
                        self.ref_T_prev = np.eye(4, dtype=np.float32)
                        self.prev_frame = frame
                        self.prev_header = header
                        self.ref_frame = frame
                        self.ref_header = header
                        self._anchor_inertial_state(
                            self.ref_T_prev, header, reset_reference=True
                        )
                        self.tracking_mode = TrackingMode.CONTACT_TRACKING
                else:
                    try:
                        curr_T_ref = normalflow(
                            self.ref_frame.N,
                            self.ref_frame.C,
                            self.ref_frame.H,
                            self.ref_frame.L,
                            frame.N,
                            frame.C,
                            frame.H,
                            frame.L,
                            np.linalg.inv(self.ref_T_prev),
                            self.ppmm,
                            scr_threshold=self.track_config["scr_threshold"],
                            ccs_threshold=self.track_config["ccs_threshold"],
                        )
                        # Current frame is tracked successfully (not a keyframe)
                        self.prev_in_contact_flag = True
                        self.prev_is_keyframe_flag = False
                        self.ref_T_prev = np.linalg.inv(curr_T_ref)
                        self.prev_frame = frame
                        self.prev_header = header
                        self._anchor_inertial_state(self.ref_T_prev, header)
                        self.tracking_mode = TrackingMode.CONTACT_TRACKING
                        self.logger.info(
                            "%s -- Regular Frame: frame ID: %d"
                            % (log_prefix, self.curr_fid)
                        )

                    except LoseTrackError:
                        if self._coast_to_header(header, TrackingMode.IMU_COASTING):
                            self.logger.info(
                                "%s -- IMU Coast: tactile tracking failed, frame ID: %d"
                                % (log_prefix, self.curr_fid)
                            )
                            return self._coasting_frame_result(self.curr_fid, header)
                        if self.prev_is_keyframe_flag:
                            self.logger.info(
                                "%s -- New Trial: Failed to track, keyframe ID: %d"
                                % (log_prefix, self.curr_kid + 1)
                            )
                            # Tracking failed. Set current frame as keyframe and start a new trial.
                            self.curr_kid += 1
                            keyframe_msg = create_keyframe_msg(
                                self.bridge,
                                frame,
                                kid=self.curr_kid,
                                fid=self.curr_fid,
                                is_new_trial=True,
                                ref_T_curr=np.eye(4, dtype=np.float32),
                                header=header,
                            )
                            keyframe_msgs.append(keyframe_msg)
                            # Update tracker state
                            self.prev_in_contact_flag = True
                            self.prev_is_keyframe_flag = True
                            self.ref_T_prev = np.eye(4, dtype=np.float32)
                            self.prev_frame = frame
                            self.prev_header = header
                            self.ref_frame = frame
                            self.ref_header = header
                            self._anchor_inertial_state(
                                self.ref_T_prev, header, reset_reference=True
                            )
                            self.tracking_mode = TrackingMode.CONTACT_TRACKING
                        else:
                            self.logger.info(
                                "%s -- New Keyframe, keyframe ID: %d"
                                % (log_prefix, self.curr_kid + 1)
                            )
                            # Set the previous frame as keyframe
                            self.curr_kid += 1
                            keyframe_msg = create_keyframe_msg(
                                self.bridge,
                                self.prev_frame,
                                kid=self.curr_kid,
                                fid=self.curr_fid - 1,
                                is_new_trial=False,
                                ref_T_curr=self.ref_T_prev,
                                header=self.prev_header,
                            )
                            keyframe_msgs.append(keyframe_msg)
                            # Update tracker state
                            self.prev_in_contact_flag = True
                            self.prev_is_keyframe_flag = True
                            self.ref_T_prev = np.eye(4, dtype=np.float32)
                            self.ref_frame = self.prev_frame
                            self.ref_header = self.prev_header
                            # The reference frame changed, so the previous IMU
                            # navigation state is no longer expressed in this
                            # local world.  A later tactile anchor reinitializes it.
                            self.inertial_state = None
                            self.inertial_propagator = None
                            # Check if the current frame can be tracked relative to the new keyframe
                            try:
                                curr_T_ref = normalflow(
                                    self.ref_frame.N,
                                    self.ref_frame.C,
                                    self.ref_frame.H,
                                    self.ref_frame.L,
                                    frame.N,
                                    frame.C,
                                    frame.H,
                                    frame.L,
                                    np.linalg.inv(self.ref_T_prev),
                                    self.ppmm,
                                    scr_threshold=self.track_config["scr_threshold"],
                                    ccs_threshold=self.track_config["ccs_threshold"],
                                )
                                # Current frame is tracked successfully (not a keyframe)
                                self.prev_in_contact_flag = True
                                self.prev_is_keyframe_flag = False
                                self.ref_T_prev = np.linalg.inv(curr_T_ref)
                                self.prev_frame = frame
                                self.prev_header = header
                                self._anchor_inertial_state(
                                    self.ref_T_prev, header, reset_reference=True
                                )
                                self.tracking_mode = TrackingMode.CONTACT_TRACKING
                                self.logger.info(
                                    "%s -- Regular Frame: frame ID: %d"
                                    % (log_prefix, self.curr_fid)
                                )

                            except LoseTrackError:
                                if self._coast_to_header(
                                    header, TrackingMode.IMU_COASTING
                                ):
                                    self.logger.info(
                                        "%s -- IMU Coast: tactile tracking failed, frame ID: %d"
                                        % (log_prefix, self.curr_fid)
                                    )
                                    return self._coasting_frame_result(
                                        self.curr_fid, header
                                    )
                                # Tracking failed. Set current frame as a keyframe and start a new trial.
                                self.logger.info(
                                    "%s -- New Trial: Failed to track, keyframe ID: %d"
                                    % (log_prefix, self.curr_kid + 1)
                                )
                                self.curr_kid += 1
                                keyframe_msg = create_keyframe_msg(
                                    self.bridge,
                                    frame,
                                    kid=self.curr_kid,
                                    fid=self.curr_fid,
                                    is_new_trial=True,
                                    ref_T_curr=np.eye(4, dtype=np.float32),
                                    header=header,
                                )
                                keyframe_msgs.append(keyframe_msg)
                                # Update tracker state
                                self.prev_in_contact_flag = True
                                self.prev_is_keyframe_flag = True
                                self.ref_T_prev = np.eye(4, dtype=np.float32)
                                self.prev_frame = frame
                                self.prev_header = header
                                self.ref_frame = frame
                                self.ref_header = header
                                self._anchor_inertial_state(
                                    self.ref_T_prev, header, reset_reference=True
                                )
                                self.tracking_mode = TrackingMode.CONTACT_TRACKING

                # Create current frame message
                frame_msg = create_frame_msg(
                    self.bridge,
                    frame,
                    fid=self.curr_fid,
                    ref_kid=self.curr_kid,
                    ref_T_curr=self.ref_T_prev,
                    header=header,
                )

                return True, (frame_msg, keyframe_msgs)
