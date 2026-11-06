"""Frontend solve with separate 9D IMU and 6D bias random-walk factors."""

import copy

import gtsam
import numpy as np

from gelslam.core.numpy_vi import NavigationState, gelsight_z_pose_jacobian


POSE_I, VELOCITY_I, POSE_J, VELOCITY_J, BIAS_I, BIAS_J = range(1, 7)
_WIDTHS = (6, 3, 6, 3, 6, 6)


def create_frontend_solver(config, gravity=None):
    """Create the reference/current marginalization frontend.

    gravity: trial-axes gravity vector from the tracker's initialization;
    None uses (0, 0, -gravity_mps2).
    """
    from gelslam.core.reference_tracking import ReferenceMarginalizationSolver
    return ReferenceMarginalizationSolver(config, gravity)


def _normal_dim(visual):
    """Residual rows per normal point: 3 (vector) or 2 (tangent)."""
    return int(getattr(visual, "residual_dim", 3))


def _normal_huber_cost_from_norms(norms, delta):
    """Use the same scaled norms for the cost and IRLS weights."""
    if len(norms) == 0:
        return np.inf
    if delta == 0.0:
        return 0.5 * float(norms @ norms) / len(norms)
    return float(np.mean(np.where(
        norms <= delta, 0.5 * norms**2,
        delta * (norms - 0.5 * delta),
    )))


def _normal_huber_linearization(residual, jacobian, scale, delta, dim=3):
    """IRLS rows scaled by 1/sqrt(N) to match the mean Huber cost."""
    scaled = np.asarray(residual) * scale
    norms = np.linalg.norm(scaled.reshape(-1, dim), axis=1)
    if len(norms) == 0:
        raise ValueError("normal linearization requires valid contact points")
    block_weights = np.ones(len(norms), dtype=np.float64)
    if delta > 0.0:
        outliers = norms > delta
        block_weights[outliers] = delta / norms[outliers]
    else:
        outliers = np.zeros(len(norms), dtype=bool)
    row_weights = np.repeat(np.sqrt(block_weights / len(norms)), dim)
    return (
        np.asarray(jacobian) * (scale * row_weights[:, None]),
        scaled * row_weights,
        _normal_huber_cost_from_norms(norms, delta),
        int(np.count_nonzero(outliers)),
    )


def _bias(state):
    return gtsam.imuBias.ConstantBias(state.accel_bias, state.gyro_bias)


def _imu_values(previous, current):
    values = gtsam.Values()
    values.insert(POSE_I, gtsam.Pose3(previous.pose))
    values.insert(VELOCITY_I, previous.velocity)
    values.insert(POSE_J, gtsam.Pose3(current.pose))
    values.insert(VELOCITY_J, current.velocity)
    values.insert(BIAS_I, _bias(previous))
    values.insert(BIAS_J, _bias(current))
    return values


def _combined_imu_factor(pim):
    """Legacy combined factor for comparison diagnostics, not the active solve."""
    return gtsam.CombinedImuFactor(
        POSE_I, VELOCITY_I, POSE_J, VELOCITY_J, BIAS_I, BIAS_J, pim
    )


def _imu_factor(pim):
    """Native 9D factor; only the bias at the interval anchor is a key."""
    return gtsam.ImuFactor(POSE_I, VELOCITY_I, POSE_J, VELOCITY_J, BIAS_I, pim)


def _bias_factor(pim, config):
    """6D residual b_j - b_i, ordered [accelerometer, gyroscope]."""
    dt = float(pim.deltaTij())
    densities = np.repeat([
        float(config.get("accelerometer_bias_random_walk", 0.0002)),
        float(config.get("gyroscope_bias_random_walk", 0.00002)),
    ], 3)
    if (not np.isfinite(dt) or dt <= 0 or
            not np.all(np.isfinite(densities)) or np.any(densities <= 0)):
        raise ValueError("bias random-walk interval and noise must be positive and finite")
    return gtsam.BetweenFactorConstantBias(
        BIAS_I, BIAS_J, gtsam.imuBias.ConstantBias(),
        gtsam.noiseModel.Diagonal.Sigmas(densities * np.sqrt(dt)),
    )


def _split_imu_factor(pim):
    """Use native motion residuals and their matching preintegration covariance.

    Keep the historical helper name for the separate motion/bias interface.
    Position and velocity errors use the target IMU axes. Do not rotate the
    residual with the optimized relative rotation while freezing covariance
    transport at the nominal preintegrated rotation: that changes the cost.
    Native Jacobians map to the split state update in navigation_factor_terms.
    Bias random walk remains a separate 6D factor.
    """
    return _imu_factor(pim)


def configured_integration_method(config):
    """Validate the shared frontend/backend native IMU integration method."""
    method = config.get("integration_method", "midpoint")
    if method != "midpoint":
        raise ValueError("imu_config.integration_method must be midpoint")
    return method


def _new_preintegration(previous, config, gravity, combined=False):
    """Create an empty native IMU preintegration at the anchor bias."""
    params_type = gtsam.PreintegrationCombinedParams if combined else gtsam.PreintegrationParams
    params = params_type(np.asarray(gravity, dtype=np.float64))
    # Use component-wise native errors with their matching covariance.
    if not hasattr(params, "setImuFactorErrorMode"):
        raise RuntimeError("Separated IMU errors require GTSAM ImuFactorErrorMode support")
    params.setImuFactorErrorMode(gtsam.ImuFactorErrorMode.ComponentWise)
    configured_integration_method(config)
    if not hasattr(params, "setUseMidpointIntegration"):
        raise RuntimeError("IMU midpoint integration requires the patched GTSAM API")
    params.setUseMidpointIntegration(True)
    accel = float(config.get("accelerometer_noise_density", 0.02))
    gyro = float(config.get("gyroscope_noise_density", 0.002))
    accel_walk = float(config.get("accelerometer_bias_random_walk", 0.0002))
    gyro_walk = float(config.get("gyroscope_bias_random_walk", 0.00002))
    integration = float(config.get("gtsam_integration_sigma_m", 1e-4))
    if not all(np.isfinite(value) and value > 0 for value in
               (accel, gyro, accel_walk, gyro_walk, integration)):
        raise ValueError("GTSAM IMU noise values must be positive and finite")
    params.setAccelerometerCovariance(np.eye(3) * accel**2)
    params.setGyroscopeCovariance(np.eye(3) * gyro**2)
    if combined:
        params.setBiasAccCovariance(np.eye(3) * accel_walk**2)
        params.setBiasOmegaCovariance(np.eye(3) * gyro_walk**2)
    params.setIntegrationCovariance(np.eye(3) * integration**2)
    pim_type = (gtsam.PreintegratedCombinedMeasurements if combined
                else gtsam.PreintegratedImuMeasurements)
    return pim_type(params, _bias(previous))


def _integrate_measurements(pim, samples):
    """Append adjacent sample intervals, validating all times before mutation."""
    if any(not np.isfinite(b.timestamp - a.timestamp) or b.timestamp <= a.timestamp
           for a, b in zip(samples, samples[1:])):
        return None
    for current, following in zip(samples, samples[1:]):
        dt = float(following.timestamp - current.timestamp)
        # samples_covering() already interpolates both image-time endpoints.
        # Native Tangent preintegration uses the configured start/half-step
        # attitude with matching bias/noise Jacobians.
        acceleration = 0.5 * (current.acceleration + following.acceleration)
        angular_velocity = 0.5 * (current.angular_velocity + following.angular_velocity)
        pim.integrateMeasurement(acceleration, angular_velocity, dt)
    return pim


def _preintegrate(samples, previous, config, gravity, combined):
    """Integrate native IMU measurements for IMU-origin navigation states.

    The extrinsic is used by tactile factors, not by IMU preintegration.
    """
    samples = tuple(samples)
    if len(samples) < 2:
        return None
    return _integrate_measurements(
        _new_preintegration(previous, config, gravity, combined), samples,
    )


class ImuPreintegrationAccumulator:
    """Retain one accepted-anchor interval, ending only at received IMU samples.

    A contact image endpoint may be interpolated. Complete that final interval
    on a copy so a rejected contact cannot split a later raw IMU interval.
    """

    _NOISE_DEFAULTS = (
        ("accelerometer_noise_density", 0.02),
        ("gyroscope_noise_density", 0.002),
        ("gtsam_integration_sigma_m", 1e-4),
    )

    def __init__(self, anchor, sample, config, gravity):
        if abs(anchor.timestamp - sample.timestamp) > 1e-9:
            raise ValueError("preintegration sample must match the accepted anchor")
        self.anchor = anchor.copy()
        self.gravity = np.asarray(gravity, dtype=np.float64).copy()
        self.noise = self._noise(config)
        self.last_sample = sample
        self.pim = _new_preintegration(anchor, config, gravity)

    @classmethod
    def _noise(cls, config):
        return tuple(float(config.get(key, default)) for key, default in cls._NOISE_DEFAULTS)

    def matches(self, anchor, config, gravity):
        return (
            anchor.timestamp == self.anchor.timestamp
            and all(np.array_equal(getattr(anchor, key), getattr(self.anchor, key))
                    for key in ("pose", "velocity", "accel_bias", "gyro_bias"))
            and np.array_equal(gravity, self.gravity)
            and self._noise(config) == self.noise
        )

    def append(self, samples):
        """Append received samples sharing the current integration boundary."""
        samples = tuple(samples)
        if not samples:
            return False
        first = samples[0]
        if (first.timestamp != self.last_sample.timestamp
                or not np.array_equal(first.acceleration, self.last_sample.acceleration)
                or not np.array_equal(first.angular_velocity, self.last_sample.angular_velocity)):
            return False
        if _integrate_measurements(self.pim, samples) is None:
            return False
        self.last_sample = samples[-1]
        return True

    def snapshot(self, samples):
        """Complete an anchor-to-image interval without retaining its endpoint."""
        samples = tuple(samples)
        if (len(samples) < 2 or samples[0].timestamp != self.anchor.timestamp
                or samples[-1].timestamp < self.last_sample.timestamp):
            return None
        cursor = next((i for i, sample in enumerate(samples)
                       if sample.timestamp == self.last_sample.timestamp), None)
        if cursor is None:
            return None
        # Interior samples are received measurements. Only the last endpoint
        # can be interpolated; never retain it after a failed contact attempt.
        if cursor < len(samples) - 1 and not self.append(samples[cursor:-1]):
            return None
        pim = copy.copy(self.pim)
        if samples[-1].timestamp > self.last_sample.timestamp:
            return _integrate_measurements(pim, (self.last_sample, samples[-1]))
        return pim


def preintegrate_imu(samples, previous, config, gelsight_T_imu, gravity):
    """Active 9D preintegration; bias diffusion belongs to _bias_factor()."""
    return _preintegrate(samples, previous, config, gravity, False)


def preintegrate_combined(samples, previous, config, gelsight_T_imu, gravity):
    """Legacy 15D preintegration, retained for comparison diagnostics only."""
    return _preintegrate(samples, previous, config, gravity, True)


def navigation_factor_terms(factor, previous, current, *, values=None):
    """Whitened rows in [previous 15D, target 15D] navigation columns."""
    if values is None:
        values = _imu_values(previous, current)
    linearized = factor.linearize(values)
    A, rhs = linearized.jacobian()
    previous_A = np.zeros((len(rhs), 15))
    current_A = np.zeros((len(rhs), 15))
    # Native Pose3 and split body-local retractions have the same derivative
    # at zero increment. Map by key, not the factor's concatenated ordering.
    columns = {POSE_I: (previous_A, 0, 6), VELOCITY_I: (previous_A, 6, 3),
               BIAS_I: (previous_A, 9, 6), POSE_J: (current_A, 0, 6),
               VELOCITY_J: (current_A, 6, 3), BIAS_J: (current_A, 9, 6)}
    offset = 0
    for key in linearized.keys():
        destination, start, width = columns[key]
        destination[:, start:start + width] = A[:, offset:offset + width]
        offset += width
    if offset != A.shape[1]:
        raise RuntimeError("unexpected navigation factor dimensions")
    return previous_A, current_A, rhs


# ImuFactor residual order is [rotation 0:3, position 3:6, velocity 6:9].
_MOTION_FIRST = np.r_[3:9, 0:3]


def split_rotation_motion_rows(factor, previous_A, current_A, rhs):
    """Split whitened ImuFactor rows into (motion, rotation) row groups.

    The factor whitens e = [e_R, e_p, e_v] with the upper-triangular U of its
    noise model (U^T U = Sigma^-1). With Q0 R0 = U[:, (p, v, R)], Q0^T is
    orthogonal and Q0^T U e = R0 [e_p; e_v; e_R]. Its last three rows are the
    rotation marginal (gyroscope only: R_i, R_j and gyroscope bias). The first
    six are position/velocity conditioned on rotation. Both groups together
    have the same squared norm, so scaling only the motion rows keeps the
    rotation information exact.
    """
    U = np.asarray(factor.noiseModel().R(), dtype=np.float64)
    if U.shape != (9, 9) or len(rhs) != 9:
        raise ValueError("rotation/motion split requires a 9D ImuFactor")
    orthogonal, _ = np.linalg.qr(U[:, _MOTION_FIRST])
    rows = orthogonal.T @ np.column_stack((previous_A, current_A, rhs))
    width = previous_A.shape[1]
    groups = rows[:6], rows[6:]
    return tuple((g[:, :width], g[:, width:-1], g[:, -1]) for g in groups)


def inertial_current_terms(pim, previous, current, config):
    """Stack separate IMU/bias rows for fixed-anchor diagnostics."""
    factors = (_split_imu_factor(pim), _bias_factor(pim, config))
    rows = [navigation_factor_terms(factor, previous, current) for factor in factors]
    A = np.vstack([row[1] for row in rows])
    rhs = np.concatenate([row[2] for row in rows])
    return {POSE_J: A[:, :6], VELOCITY_J: A[:, 6:9], BIAS_J: A[:, 9:15]}, rhs


def combined_imu_current_terms(pim, previous, current, factor=None):
    """Legacy combined-factor columns for comparison diagnostics only."""
    linearized = (factor or _combined_imu_factor(pim)).linearize(
        _imu_values(previous, current)
    )
    if list(linearized.keys()) != [
        POSE_I, VELOCITY_I, POSE_J, VELOCITY_J, BIAS_I, BIAS_J
    ]:
        raise RuntimeError("unexpected GTSAM CombinedImuFactor key order")
    jacobian, rhs = linearized.jacobian()
    if jacobian.shape != (15, 30) or rhs.shape != (15,):
        raise RuntimeError("unexpected GTSAM CombinedImuFactor dimensions")
    offsets = np.cumsum((0,) + _WIDTHS)
    blocks = {
        POSE_J: jacobian[:, offsets[2]:offsets[3]],
        VELOCITY_J: jacobian[:, offsets[3]:offsets[4]],
        BIAS_J: jacobian[:, offsets[5]:offsets[6]],
    }
    return blocks, rhs


def _navigation_step_norms(step):
    """Unscaled navigation increments, with a separate norm for each SI unit."""
    return {name: float(np.linalg.norm(step[start:start + 3]))
            for name, start in (('rotation_rad', 0), ('translation_m', 3),
                                ('velocity_mps', 6), ('accel_bias_mps2', 9),
                                ('gyro_bias_rps', 12))}


class GtsamVisualInertialSolver:
    """Solve normal/IMU rows with 9D fixed-bias or 15D navigation updates."""

    def __init__(self, config, gravity=None):
        self.config = dict(config or {})
        configured_integration_method(self.config)
        if gravity is None:
            gravity = (0.0, 0.0, -float(self.config.get("gravity_mps2", 9.81)))
        self.gravity = np.asarray(gravity, dtype=np.float64)
        self.max_iterations = int(self.config.get("gtsam_vi_max_iterations", 10))
        self.min_iterations = self.config.get("gtsam_vi_min_iterations", 7)
        self.normal_huber_iterations = self.config.get("gtsam_vi_normal_huber_iterations", 5)
        for name, value, minimum in (
                ("gtsam_vi_min_iterations", self.min_iterations, 1),
                ("gtsam_vi_normal_huber_iterations", self.normal_huber_iterations, 0)):
            if type(value) is not int or value < minimum:
                raise ValueError(name + " must be an integer >= " + str(minimum))
        self.visual_weight = float(self.config.get("dense_normalflow_residual_weight", 1.0))
        self.normal_huber_delta = float(self.config.get("gtsam_vi_normal_huber_delta", 0.4))
        self.normal_linearization = self.config.get(
            "gtsam_vi_normal_linearization", "inverse_compositional"
        )
        if (not isinstance(self.normal_linearization, str)
                or self.normal_linearization not in (
                    "inverse_compositional", "target_relinearized")):
            raise ValueError(
                "gtsam_vi_normal_linearization must be inverse_compositional "
                "or target_relinearized"
            )
        self.imu_weight = float(self.config.get("imu_residual_weight", 1.0))
        # Bias random-walk factor weight. Null or absent: use imu_residual_weight.
        # Shipped configs set it explicitly, so a change of imu_residual_weight
        # does not change the bias stiffness.
        bias_weight = self.config.get("bias_residual_weight")
        self.bias_weight = self.imu_weight if bias_weight is None else float(bias_weight)
        if not np.isfinite(self.bias_weight) or self.bias_weight <= 0:
            raise ValueError("bias_residual_weight must be positive and finite or null")
        self.height_z_enabled = self.config.get("gtsam_vi_height_z_enabled", False)
        if not isinstance(self.height_z_enabled, bool):
            raise ValueError("gtsam_vi_height_z_enabled must be a boolean")
        # Uncertainty of the mean height measurement, not of independent pixels.
        self.height_z_sigma = float(self.config.get("gtsam_vi_height_z_sigma_m", 1e-4))
        if not np.isfinite(self.height_z_sigma) or self.height_z_sigma <= 0:
            raise ValueError("gtsam_vi_height_z_sigma_m must be positive and finite")
        # Keep a zero-valued diagnostic attribute for older audit scripts.
        self.damping = 0.0
        self.rotation_stop = float(
            self.config.get("gtsam_vi_rotation_stop_rad", 1e-4)
        )
        self.translation_stop = float(
            self.config.get("gtsam_vi_translation_stop_m", 1e-5)
        )
        if not all(np.isfinite(value) and value > 0
                   for value in (self.rotation_stop, self.translation_stop)):
            raise ValueError('NormalFlow stop tolerances must be positive and finite')
        self.initialize_from_imu_preintegration = self.config.get(
            "frontend_initialize_from_imu_preintegration", True
        )
        self.normal_reference_z_from_imu = bool(self.config.get(
            "gtsam_vi_normal_reference_z_from_imu", False
        ))
        self.last_optimization_trace = []
        self.last_fixed_normal_reference_z_m = None
        self.last_refinement_failure = None
        self.prior_sigmas = np.repeat([
            float(self.config.get("gtsam_vi_prior_rotation_sigma_rad", 0.01)),
            float(self.config.get("gtsam_vi_prior_translation_sigma_m", 0.0001)),
            float(self.config.get("gtsam_vi_prior_velocity_sigma_mps", 0.1)),
            float(self.config.get("gtsam_vi_prior_accel_bias_sigma_mps2", 0.1)),
            float(self.config.get("gtsam_vi_prior_gyro_bias_sigma_rps", 0.01)),
        ], 3)
        self.marginal_prior = None
        self.pending_marginal_prior = None
        if not np.all(np.isfinite(self.prior_sigmas)) or np.any(self.prior_sigmas <= 0):
            raise ValueError("GTSAM navigation prior sigmas must be positive and finite")
        if (self.gravity.shape != (3,) or not np.all(np.isfinite(self.gravity)) or
                self.max_iterations < 1 or
                not isinstance(self.initialize_from_imu_preintegration, bool) or
                self.visual_weight <= 0 or
                not np.isfinite(self.normal_huber_delta) or self.normal_huber_delta < 0 or
                self.imu_weight <= 0 or
                self.rotation_stop <= 0 or self.translation_stop <= 0):
            raise ValueError("invalid GTSAM visual-inertial solver configuration")

    def commit_marginal_prior(self, accepted_state):
        """Commit only after the tracker accepts the target frame's quality."""
        if self.pending_marginal_prior is not None:
            if not self.pending_marginal_prior.matches(accepted_state):
                raise ValueError("accepted state does not match the pending marginal prior")
            self.marginal_prior = self.pending_marginal_prior
        self.pending_marginal_prior = None

    def discard_pending_prior(self):
        self.pending_marginal_prior = None

    def inertial_scales(self):
        """Row scales for the (IMU, bias random-walk) factor pair, in that order."""
        return np.sqrt(self.imu_weight), np.sqrt(self.bias_weight)

    def prediction_pose_distance(self, predicted, candidate, prior):
        """Squared pose distance after marginalizing velocity and both biases."""
        from gelslam.core.split_pose import split_pose_coordinates
        # QR/SVD of the square-root rows avoids squaring their condition number.
        _, singular, vt = np.linalg.svd(prior.A, full_matrices=False)
        active = singular > 1e-12
        covariance = (vt[active].T / singular[active]**2) @ vt[active]
        delta = split_pose_coordinates(predicted.pose, candidate.pose)
        information = np.linalg.pinv(covariance[:6, :6], rcond=1e-12)
        return float(delta @ information @ delta)

    def normal_huber_delta_at(self, iteration):
        """Use Huber only on normal rows in the first configured GN iterations.

        iteration is zero based; the default five-step phase is indices 0..4.
        """
        return self.normal_huber_delta if iteration < self.normal_huber_iterations else 0.0

    def _height_z_jacobian(self, gel_from_imu_increment):
        return gelsight_z_pose_jacobian(gel_from_imu_increment, self.height_z_sigma)

    def _height_z_linearization(self, visual, reference_pose, target_pose,
                                gel_from_imu_increment, *, jacobian=None):
        """Conditional mean-height row with only a GelSight local-Z derivative.

        Resample at the current pose each iteration, but hold rotation, XY and
        correspondences fixed in this row's linear model. This is intentionally
        not the full derivative of the nonlinear height residual. Transform the
        row to IMU increments only after restricting it in GelSight coordinates.
        """
        if not self.height_z_enabled:
            return np.empty((0, 6)), np.empty(0), 0.0
        error = visual.target_z_height_correction(reference_pose, target_pose)
        if error is None or not np.isfinite(error):
            return None
        if jacobian is None:
            jacobian = self._height_z_jacobian(gel_from_imu_increment)
        residual = np.array([error / self.height_z_sigma])
        return (jacobian, residual,
                0.5 * float(residual @ residual))

    def predict_current(self, previous, samples, gelsight_T_imu, *, force=False):
        """Predict the target IMU pose and IMU-origin world velocity."""
        samples = tuple(samples)
        if (len(samples) < 2 or
                abs(samples[0].timestamp - previous.timestamp) > 1e-9):
            return None
        pim = preintegrate_imu(
            samples, previous, self.config, gelsight_T_imu, self.gravity
        )
        if pim is None:
            return None
        return self.predict_preintegrated(previous, pim, samples[-1].timestamp, force=force)

    def predict_preintegrated(self, previous, pim, timestamp, *, force=False):
        """Apply an existing anchor-to-target preintegration without reintegration."""
        if (timestamp <= previous.timestamp or not np.isclose(
                pim.deltaTij(), timestamp - previous.timestamp, rtol=0., atol=1e-9)):
            return None
        if not force and not self.initialize_from_imu_preintegration:
            current = previous.copy()
            current.timestamp = timestamp
            return current, pim
        try:
            predicted = pim.predict(
                gtsam.NavState(gtsam.Pose3(previous.pose), previous.velocity),
                _bias(previous),
            )
            pose = np.asarray(predicted.pose().matrix(), dtype=np.float64)
            velocity = np.asarray(predicted.velocity(), dtype=np.float64)
        except (RuntimeError, ValueError):
            return None
        if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(velocity)):
            return None
        return (
            NavigationState(
                timestamp, pose, velocity,
                previous.accel_bias.copy(), previous.gyro_bias.copy(),
            ),
            pim,
        )
