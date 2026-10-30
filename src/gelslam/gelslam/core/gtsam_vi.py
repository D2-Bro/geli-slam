"""Frontend solve with separate 9D IMU and 6D bias random-walk factors."""

import copy
from contextlib import nullcontext

import gtsam
import numpy as np

from gelslam.core.numpy_vi import NavigationState, adjoint, gelsight_z_pose_jacobian
from gelslam.core.marginal_prior import (
    NavigationPrior, marginalize_rows, marginalize_rows_system,
)
from gelslam.core.split_pose import solve_gaussian_qr, split_pose_plus
from gelslam.core.imu_navigation import gelsight_pose, gelsight_pose_jacobian
from gelslam.core.gt_state_diagnostic import resolve_gt_state_test


POSE_I, VELOCITY_I, POSE_J, VELOCITY_J, BIAS_I, BIAS_J = range(1, 7)
_WIDTHS = (6, 3, 6, 3, 6, 6)


def create_frontend_solver(config):
    """Select the legacy pairwise or correlated reference/current frontend."""
    mode = config.get("frontend_state_mode", "pairwise")
    if mode == "reference_marginalization":
        from gelslam.core.reference_tracking import ReferenceMarginalizationSolver
        return ReferenceMarginalizationSolver(config)
    if mode != "pairwise":
        raise ValueError("frontend_state_mode must be pairwise or reference_marginalization")
    return GtsamVisualInertialSolver(config)


def _normal_huber_cost(residual, scale, delta):
    """Mean Huber cost per valid three-component normal residual."""
    norms = np.linalg.norm(np.asarray(residual).reshape(-1, 3) * scale, axis=1)
    return _normal_huber_cost_from_norms(norms, delta)


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


def _normal_huber_linearization(residual, jacobian, scale, delta):
    """IRLS rows scaled by 1/sqrt(N) to match the mean Huber cost."""
    scaled = np.asarray(residual) * scale
    norms = np.linalg.norm(scaled.reshape(-1, 3), axis=1)
    if len(norms) == 0:
        raise ValueError("normal linearization requires valid contact points")
    block_weights = np.ones(len(norms), dtype=np.float64)
    if delta > 0.0:
        outliers = norms > delta
        block_weights[outliers] = delta / norms[outliers]
    else:
        outliers = np.zeros(len(norms), dtype=bool)
    row_weights = np.repeat(np.sqrt(block_weights / len(norms)), 3)
    return (
        np.asarray(jacobian) * (scale * row_weights[:, None]),
        scaled * row_weights,
        _normal_huber_cost_from_norms(norms, delta),
        int(np.count_nonzero(outliers)),
    )


def _normalflow_step_norms(reference_pose, target_pose, gel_from_imu_increment, pose_step,
                           *, reference_inverse=None):
    """Measure the target update in the reference GelSight IC coordinates.

    For T = target_T_reference, T_new = T inv(dT). The IC correction is
    dp = Ad(inv(T)) eta_gel to first order. Its XY components, not IMU-local
    or world XY, are the planar update tested by original NormalFlow.
    """
    if reference_inverse is None:
        reference_inverse = np.linalg.inv(reference_pose)
    reference_T_target = reference_inverse @ target_pose
    correction = adjoint(reference_T_target) @ gel_from_imu_increment @ pose_step
    return dict(rotation_rad=float(np.linalg.norm(correction[:3])),
                translation_xy_m=float(np.linalg.norm(correction[3:5])))


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


def _navigation_plus(state, step):
    result = NavigationState(
        state.timestamp, split_pose_plus(state.pose, step[:6]),
        state.velocity.copy(), state.accel_bias.copy(), state.gyro_bias.copy(),
    )
    result.velocity += step[6:9]
    result.accel_bias += step[9:12]
    result.gyro_bias += step[12:15]
    return result


def _navigation_step_norms(step):
    """Unscaled navigation increments, with a separate norm for each SI unit."""
    return {name: float(np.linalg.norm(step[start:start + 3]))
            for name, start in (('rotation_rad', 0), ('translation_m', 3),
                                ('velocity_mps', 6), ('accel_bias_mps2', 9),
                                ('gyro_bias_rps', 12))}


class GtsamVisualInertialSolver:
    """Solve normal/IMU rows with 9D fixed-bias or 15D navigation updates."""

    def __init__(self, config):
        self.config = dict(config or {})
        configured_integration_method(self.config)
        configured_gravity = self.config.get("frontend_gravity_mps2")
        if configured_gravity is None:
            configured_gravity = (0.0, 0.0, -float(self.config.get("gravity_mps2", 9.81)))
        self.gravity = np.asarray(configured_gravity, dtype=np.float64)
        self.max_iterations = int(self.config.get("gtsam_vi_max_iterations", 10))
        self.min_iterations = self.config.get("gtsam_vi_min_iterations", 7)
        self.normal_huber_iterations = self.config.get("gtsam_vi_normal_huber_iterations", 5)
        for name, value, minimum in (
                ("gtsam_vi_min_iterations", self.min_iterations, 1),
                ("gtsam_vi_normal_huber_iterations", self.normal_huber_iterations, 0)):
            if type(value) is not int or value < minimum:
                raise ValueError(name + " must be an integer >= " + str(minimum))
        self.normal_sigma = float(self.config.get("dense_normalflow_normal_sigma", 1.0))
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
        self.height_z_enabled = self.config.get("gtsam_vi_height_z_enabled", False)
        if not isinstance(self.height_z_enabled, bool):
            raise ValueError("gtsam_vi_height_z_enabled must be a boolean")
        # Uncertainty of the mean height measurement, not of independent pixels.
        self.height_z_sigma = float(self.config.get("gtsam_vi_height_z_sigma_m", 1e-4))
        if not np.isfinite(self.height_z_sigma) or self.height_z_sigma <= 0:
            raise ValueError("gtsam_vi_height_z_sigma_m must be positive and finite")
        # Keep a zero-valued diagnostic attribute for older audit scripts.
        self.damping = 0.0
        if float(self.config.get("gtsam_vi_damping", 0.0)) != 0.0:
            raise ValueError("frontend Gauss-Newton does not support gtsam_vi_damping")
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
        # With initialize_from_imu_preintegration false: start the solve at the IMU
        # rotation but keep the previous GelSight origin (zero relative translation).
        self.initialize_rotation_from_imu = self.config.get("frontend_initialize_rotation_from_imu", False)
        if not isinstance(self.initialize_rotation_from_imu, bool):
            raise ValueError("frontend_initialize_rotation_from_imu must be a boolean")
        self.fix_previous_state = self.config.get("frontend_fix_previous_state", False)
        if not isinstance(self.fix_previous_state, bool):
            raise ValueError("frontend_fix_previous_state must be a boolean")
        self.imu_prediction_only = self.config.get("frontend_imu_prediction_only", False)
        if not isinstance(self.imu_prediction_only, bool):
            raise ValueError("frontend_imu_prediction_only must be a boolean")
        self.normal_reference_z_from_imu = bool(self.config.get(
            "gtsam_vi_normal_reference_z_from_imu", False
        ))
        self.normal_zero_pose_z_jacobian = bool(self.config.get(
            "gtsam_vi_normal_zero_pose_z_jacobian", False
        ))
        self.last_optimization_trace = []
        self.last_fixed_normal_reference_z_m = None
        self.last_refinement_failure = None
        self.marginalization_enabled = bool(self.config.get(
            "gtsam_vi_marginalization", True
        ))
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
                self.normal_sigma <= 0 or self.visual_weight <= 0 or
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

    def propagate_prior(self, previous, current, pim, *, fixed_anchor=False):
        """Transport the accepted 15D uncertainty through one IMU interval.

        This is a candidate prior. The caller commits it only when it accepts
        the corresponding prediction. Rejected visual rows never enter it.
        """
        factors = (_split_imu_factor(pim), _bias_factor(pim, self.config))
        rows = [navigation_factor_terms(factor, previous, current) for factor in factors]
        scale = np.sqrt(self.imu_weight)
        if fixed_anchor:
            return NavigationPrior(
                current.copy(), np.vstack([row[1] * scale for row in rows]),
                np.concatenate([row[2] * scale for row in rows]))
        prior = self.marginal_prior
        if prior is None:
            prior = NavigationPrior.initial(previous, self.prior_sigmas)
        elif abs(prior.anchor.timestamp - previous.timestamp) > 1e-9:
            raise ValueError("IMU propagation prior belongs to a different timestamp")
        Ap, bp = prior.linearize(previous)
        reduced = marginalize_rows_system(
            np.vstack([Ap] + [row[0] * scale for row in rows]),
            np.vstack([np.zeros_like(Ap)] + [row[1] * scale for row in rows]),
            np.concatenate([bp] + [row[2] * scale for row in rows]),
        )
        A, rhs = reduced.rows()
        return NavigationPrior(current.copy(), A, rhs)

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

    def refine_recovery_reference(self, visual, reference_pose, previous, current,
                                  samples, gelsight_T_imu, preintegrated=None):
        """Optimize both endpoints when the reference is an IMU-only keyframe."""
        self.discard_pending_prior()
        self.last_refinement_failure = None
        self.last_optimization_trace = []
        self.last_prior_source = "propagated_recovery_reference"
        if visual is None or len(visual) < 10:
            self.last_refinement_failure = "initial_shared_points_lt_10"
            return None
        prior = self.marginal_prior
        if prior is None or not prior.matches(previous):
            self.last_refinement_failure = "recovery_prior_mismatch"
            return None
        previous, current = previous.copy(), current.copy()
        pim = preintegrated or preintegrate_imu(
            samples, previous, self.config, gelsight_T_imu, self.gravity)
        if pim is None:
            return None
        factors = (_split_imu_factor(pim), _bias_factor(pim, self.config))
        gel_J = gelsight_pose_jacobian(gelsight_T_imu)
        method = ("evaluate_ic" if self.normal_linearization == "inverse_compositional"
                  else "evaluate_relinearized")
        evaluate = getattr(visual, method, visual.evaluate)
        width = 15

        def rows_at(left, right, delta):
            ref = gelsight_pose(left.pose, gelsight_T_imu)
            tar = gelsight_pose(right.pose, gelsight_T_imu)
            evaluated = evaluate(ref, tar)
            if evaluated is None:
                return None
            residual, Jr, Jt = evaluated
            if len(residual) < 30:
                return None
            A, b, _, _ = _normal_huber_linearization(
                residual, np.column_stack((Jr @ gel_J, Jt @ gel_J)),
                np.sqrt(self.visual_weight), delta)
            Ap, Ac, rhs = self._joint_rows(factors, prior, left, right)
            normal_p, normal_c = np.zeros((len(b), width)), np.zeros((len(b), width))
            normal_p[:, :6], normal_c[:, :6] = A[:, :6], A[:, 6:]
            Ap, Ac, rhs = np.vstack((Ap, normal_p)), np.vstack((Ac, normal_c)), np.r_[rhs, -b]
            height = self._height_z_linearization(visual, ref, tar, gel_J)
            if height is None:
                return None
            if self.height_z_enabled:
                Ht, hb, _ = height
                # Relative target-local Z row, transported to the reference chart.
                Hr = -self._height_z_jacobian(np.eye(6)) @ adjoint(np.linalg.inv(tar) @ ref) @ gel_J
                Hp, Hc = np.zeros((1, width)), np.zeros((1, width))
                Hp[:, :6], Hc[:, :6] = Hr, Ht
                Ap, Ac, rhs = np.vstack((Ap, Hp)), np.vstack((Ac, Hc)), np.r_[rhs, -hb]
            if not all(np.isfinite(v).all() for v in (Ap, Ac, rhs)):
                return None
            return Ap, Ac, rhs

        for iteration in range(self.max_iterations):
            delta = self.normal_huber_delta_at(iteration)
            rows = rows_at(previous, current, delta)
            if rows is None:
                self.last_refinement_failure = "invalid_recovery_residual"
                return None
            reduced = marginalize_rows_system(*rows)
            A, rhs = reduced.rows()
            step, _, rank, _ = np.linalg.lstsq(A, rhs, rcond=1e-12)
            if rank < width or not np.isfinite(step).all():
                self.last_refinement_failure = "singular_recovery_system"
                return None
            previous_step = reduced.recover_previous(step)
            previous = _navigation_plus(previous, previous_step)
            current = _navigation_plus(current, step)
            self.last_optimization_trace.append(dict(
                iteration=iteration, variable_reference=True,
                active_rows=getattr(visual, "last_active_rows", np.ones(10)).sum(),
                base_cost=float(rows[2] @ rows[2] / 2)))
            if (iteration + 1 >= self.min_iterations
                    and max(np.linalg.norm(step[:3]), np.linalg.norm(previous_step[:3])) < self.rotation_stop
                    and max(np.linalg.norm(step[3:6]), np.linalg.norm(previous_step[3:6])) < self.translation_stop):
                break
        rows = rows_at(previous, current, delta)
        if rows is None:
            return None
        A, rhs = marginalize_rows_system(*rows).rows()
        self.pending_marginal_prior = NavigationPrior(current.copy(), A, rhs)
        self.last_reference_state = previous
        return current

    def normal_huber_delta_at(self, iteration):
        """Use Huber only on normal rows in the first configured GN iterations.

        iteration is zero based; the default five-step phase is indices 0..4.
        """
        return self.normal_huber_delta if iteration < self.normal_huber_iterations else 0.0

    def _joint_rows(self, factors, prior, previous, current, *, values=None):
        # Huber is restricted to the normal measurements. IMU, bias and prior
        # retain their ordinary whitened least-squares rows in every iteration.
        Ap, bp = (np.empty((0, 15)), np.empty(0)) if prior is None else prior.linearize(previous)
        if values is None:
            values = _imu_values(previous, current)
        rows = [navigation_factor_terms(factor, previous, current, values=values)
                for factor in factors]
        scale = np.sqrt(self.imu_weight)
        previous_A = np.vstack([Ap] + [row[0] * scale for row in rows])
        current_A = np.vstack([np.zeros((len(bp), 15))]
                              + [row[1] * scale for row in rows])
        return (previous_A, current_A,
                np.concatenate([bp] + [row[2] * scale for row in rows]))

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
        if (not force and not self.initialize_from_imu_preintegration
                and not self.imu_prediction_only):
            current = previous.copy()
            current.timestamp = timestamp
            if self.initialize_rotation_from_imu:
                try:
                    rotation = np.asarray(pim.predict(
                        gtsam.NavState(gtsam.Pose3(previous.pose), previous.velocity),
                        _bias(previous)).pose().rotation().matrix(), dtype=np.float64)
                except (RuntimeError, ValueError):
                    return current, pim
                transform = np.asarray(self.config.get("gelsight_T_imu", np.eye(4)), dtype=np.float64)
                gel_in_imu = -transform[:3, :3].T @ transform[:3, 3]
                gel_origin = previous.pose[:3, :3] @ gel_in_imu + previous.pose[:3, 3]
                current.pose[:3, :3] = rotation
                current.pose[:3, 3] = gel_origin - rotation @ gel_in_imu
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

    def refine_current(self, visual, reference_pose, previous, current, samples,
                       gelsight_T_imu, preintegrated=None, *, fixed_anchor=False):
        """Optimize IMU states; evaluate tactile observations with GelSight poses.

        reference_pose is the fixed GelSight reference pose in the navigation world.
        Each iteration builds the selected normal model and relinearizes the
        inertial/prior terms, then applies the full undamped Gauss-Newton step.
        Normal rows use Huber for the first five steps by default, then ordinary
        least squares. Only the target's reference-IC rotation and XY correction
        control convergence, starting at the seventh step. The maximum iteration
        limit and invalid measurements can terminate the solve earlier.
        Increasing finite cost does not reject a step or trigger a retry.
        Prediction-only mode returns the supplied IMU prediction without a solve.
        """
        cache = getattr(visual, "cache_projections", None)
        with cache() if cache is not None else nullcontext():
            return self._refine_current(
                visual, reference_pose, previous, current, samples,
                gelsight_T_imu, preintegrated=preintegrated, fixed_anchor=fixed_anchor)

    def refine_from_keyframe(self, *args, **kwargs):
        """Condition on corrected K_k and form a fresh prior for target F_j."""
        return self.refine_current(*args, **kwargs, fixed_anchor=True)

    def _refine_current(self, visual, reference_pose, previous, current, samples,
                        gelsight_T_imu, preintegrated=None, fixed_anchor=False):
        """Run the solve while its image-pair projection cache is active."""
        self.last_optimization_trace = []
        self.last_fixed_normal_reference_z_m = None
        self.last_refinement_failure = None
        self.pending_marginal_prior = None
        if self.imu_prediction_only:
            # The tracker supplies predict_current()'s State. Keep the complete
            # prediction. The tracker also bypasses SCR/CCS in this mode.
            self.last_prior_source = 'imu_prediction_only'
            self.marginal_prior = None
            return current.copy()
        fixed_previous = self.fix_previous_state or fixed_anchor
        self.last_prior_source = ('backend_keyframe' if fixed_anchor else
                                  'fixed_previous' if fixed_previous else 'none')
        if fixed_previous and not fixed_anchor:
            # The anchor is a constant, not an uncertain state to eliminate.
            # Do not retain a posterior from a prior joint solve or startup seed.
            self.marginal_prior = None
        if visual is None or len(samples) < 2 or gelsight_T_imu is None:
            return None
        pim = preintegrated
        if pim is None:
            pim = preintegrate_imu(
                samples, previous, self.config, gelsight_T_imu, self.gravity
            )
        if pim is None:
            return None
        reference_inverse = None
        if self.normal_reference_z_from_imu:
            predicted = pim.predict(
                gtsam.NavState(gtsam.Pose3(previous.pose), previous.velocity),
                _bias(previous),
            )
            reference_inverse = np.linalg.inv(reference_pose)
            relative = (reference_inverse
                        @ gelsight_pose(predicted.pose().matrix(), gelsight_T_imu))
            self.last_fixed_normal_reference_z_m = float(relative[2, 3])
            visual.set_fixed_reference_z(self.last_fixed_normal_reference_z_m)
        if len(visual) < 10:
            self.last_refinement_failure = "initial_shared_points_lt_10"
            return None
        imu_T_gelsight = np.linalg.inv(gelsight_T_imu)
        if reference_inverse is None:
            reference_inverse = np.linalg.inv(reference_pose)
        nonlinear_imu_factor = _split_imu_factor(pim)
        nonlinear_bias_factor = _bias_factor(pim, self.config)
        inertial_factors = (nonlinear_imu_factor, nonlinear_bias_factor)
        current = current.copy()
        previous = previous.copy()
        prior = None
        if self.marginalization_enabled and not fixed_previous:
            prior = self.marginal_prior
            # A new trial or a tactile-only fallback has no continuous VI
            # posterior. Initialize at that accepted state, not an old time.
            controls = resolve_gt_state_test(self.config)
            keep_prior = controls['enabled'] and not controls['reset_marginal_prior']
            different_time = (prior is not None
                              and abs(prior.anchor.timestamp - previous.timestamp) > 1e-9)
            if (different_time and keep_prior
                    and controls['previous_state_time'] == 'previous_image'):
                raise RuntimeError(
                    'Cannot retain a marginal prior at a different IMU timestamp; '
                    'use previous_state_time: accepted_state or reset_marginal_prior: true'
                )
            if prior is None or different_time or (not keep_prior and not prior.matches(previous)):
                prior = NavigationPrior.initial(previous, self.prior_sigmas)
                self.last_prior_source = 'initial_timestamp_change' if different_time else 'initial'
            else:
                # Its frozen mean/gradient remain unchanged. Only linearize at
                # the diagnostic starting value; do not silently recenter it.
                self.last_prior_source = 'retained'
        normal_scale = np.sqrt(self.visual_weight)
        imu_scale = np.sqrt(self.imu_weight)
        gel_from_imu_increment = gelsight_pose_jacobian(gelsight_T_imu)
        height_jacobian = (self._height_z_jacobian(gel_from_imu_increment)
                           if self.height_z_enabled else None)
        # Select once so the solve and retained marginal prior use the same
        # normal model. Generic residual providers continue to supply their J.
        method, source = (
            ("evaluate_ic", "reference_normal_map_ic")
            if self.normal_linearization == "inverse_compositional" else
            ("evaluate_relinearized", "target_normal_map")
        )
        normal_evaluator = getattr(visual, method, None)
        normal_jacobian_source = source if normal_evaluator is not None else "residual_model"
        if normal_evaluator is None:
            normal_evaluator = visual.evaluate
        values = None
        height_terms = None
        inertial_costs = None
        target_gelsight_pose = current.pose @ imu_T_gelsight
        for iteration in range(self.max_iterations):
            normal_huber_delta = self.normal_huber_delta_at(iteration)
            evaluated = normal_evaluator(
                reference_pose, target_gelsight_pose)
            if evaluated is None:
                return None
            residual, _, normal_jacobian = evaluated
            if (len(residual) < 30 or not np.all(np.isfinite(residual))
                    or not np.all(np.isfinite(normal_jacobian))):
                return None
            if self.normal_zero_pose_z_jacobian:
                normal_jacobian = normal_jacobian.copy()
                normal_jacobian[:, 5] = 0.0
            normal_jacobian = normal_jacobian @ gel_from_imu_increment
            normal_A, normal_b, normal_cost, downweighted = (
                _normal_huber_linearization(
                    residual, normal_jacobian, normal_scale, normal_huber_delta
                )
            )
            normal_factor = gtsam.JacobianFactor(
                POSE_J, normal_A, -normal_b,
                gtsam.noiseModel.Isotropic.Sigma(len(residual), 1.0),
            )
            if height_terms is None:
                height_terms = self._height_z_linearization(
                    visual, reference_pose, target_gelsight_pose,
                    gel_from_imu_increment, jacobian=height_jacobian,
                )
            if height_terms is None:
                self.last_refinement_failure = "invalid_height_residual"
                return None
            height_A, height_b, height_cost = height_terms
            if values is None:
                values = _imu_values(previous, current)
            joint_rows = None
            if prior is not None:
                joint_rows = self._joint_rows(
                    inertial_factors, prior, previous, current, values=values)
                Ap, Ac, rhs = joint_rows
                reduced_system = marginalize_rows_system(Ap, Ac, rhs)
                reduced_A, imu_rhs = reduced_system.rows()
                imu_blocks = {POSE_J: reduced_A[:, :6],
                              VELOCITY_J: reduced_A[:, 6:9],
                              BIAS_J: reduced_A[:, 9:15]}
                if imu_scale != 1.0:
                    # Preserve the existing rounding for non-unit weights.
                    imu_blocks = {key: (block / imu_scale) * imu_scale
                                  for key, block in imu_blocks.items()}
                    imu_rhs = (imu_rhs / imu_scale) * imu_scale
            if inertial_costs is None:
                imu_cost = self.imu_weight * float(
                    nonlinear_imu_factor.error(values)
                )
                bias_cost = self.imu_weight * float(
                    nonlinear_bias_factor.error(values)
                )
                prior_cost = 0.0 if prior is None else prior.error(previous)
            else:
                imu_cost, bias_cost, prior_cost = inertial_costs
            base_cost = normal_cost + imu_cost + bias_cost + prior_cost + height_cost
            if not np.isfinite(base_cost):
                return None
            graph = gtsam.GaussianFactorGraph()
            graph.add(normal_factor)
            if self.height_z_enabled:
                graph.add(gtsam.JacobianFactor(
                    POSE_J, height_A, -height_b, gtsam.noiseModel.Unit.Create(1),
                ))
            if prior is not None:
                # Elimination couples the target pose, velocity and bias.
                graph.add(gtsam.JacobianFactor(
                    POSE_J, imu_blocks[POSE_J],
                    VELOCITY_J, imu_blocks[VELOCITY_J],
                    BIAS_J, imu_blocks[BIAS_J],
                    imu_rhs,
                    gtsam.noiseModel.Isotropic.Sigma(len(imu_rhs), 1.0),
                ))
            else:
                # Fixed previous state: motion has no target-bias column.
                _, motion_A, motion_rhs = navigation_factor_terms(
                    nonlinear_imu_factor, previous, current, values=values
                )
                _, bias_A, bias_rhs = navigation_factor_terms(
                    nonlinear_bias_factor, previous, current, values=values
                )
                graph.add(gtsam.JacobianFactor(
                    POSE_J, motion_A[:, :6] * imu_scale,
                    VELOCITY_J, motion_A[:, 6:9] * imu_scale,
                    motion_rhs * imu_scale, gtsam.noiseModel.Unit.Create(9),
                ))
                graph.add(gtsam.JacobianFactor(
                    BIAS_J, bias_A[:, 9:15] * imu_scale,
                    bias_rhs * imu_scale, gtsam.noiseModel.Unit.Create(6),
                ))
            try:
                solution = solve_gaussian_qr(graph)
                pose_step = np.asarray(solution.at(POSE_J))
                velocity_step = np.asarray(solution.at(VELOCITY_J))
                bias_step = np.asarray(solution.at(BIAS_J))
            except (RuntimeError, ValueError):
                return None
            if not all(np.all(np.isfinite(step)) for step in
                       (pose_step, velocity_step, bias_step)):
                return None
            current_step = np.concatenate((pose_step, velocity_step, bias_step))
            previous_step = None
            if joint_rows is not None:
                previous_step = reduced_system.recover_previous(current_step)
                if not np.all(np.isfinite(previous_step)):
                    return None
            active_rows = getattr(visual, "last_active_rows", None)
            raw_step_norms = {
                'target': _navigation_step_norms(current_step),
                'previous': (None if previous_step is None else
                             _navigation_step_norms(previous_step)),
            }
            # Match original NormalFlow's target rotation/planar check. The
            # other state updates remain optimized and reported, but cannot
            # delay convergence after the minimum number of full steps.
            convergence_step_norms = _normalflow_step_norms(
                reference_pose, target_gelsight_pose, gel_from_imu_increment, pose_step,
                reference_inverse=reference_inverse)
            step_converged = (
                iteration + 1 >= self.min_iterations
                and convergence_step_norms['rotation_rad'] < self.rotation_stop
                and convergence_step_norms['translation_xy_m'] < self.translation_stop)
            trace = {
                "iteration": iteration,
                "update_rule": "full_gauss_newton",
                "normal_linearization": self.normal_linearization,
                "normal_jacobian_source": normal_jacobian_source,
                "normal_huber_delta": normal_huber_delta,
                "normal_huber_active": normal_huber_delta > 0.0,
                "active_rows": len(residual) // 3,
                "base_cost": base_cost,
                "base_normal_cost": normal_cost,
                "height_z_enabled": self.height_z_enabled,
                "base_height_cost": height_cost,
                "downweighted_normal_points": downweighted,
                "base_imu_cost": imu_cost,
                "base_bias_cost": bias_cost,
                "base_prior_cost": prior_cost,
                # Keep these report fields for readers of older diagnostics.
                # There is exactly one full step; cost never selects its scale.
                "attempts": [],
                "accepted_scale": None,
                "accepted_cost": None,
                "raw_step_norms": raw_step_norms,
                "stopping_rule": "reference_ic_rotation_xy",
                "minimum_iterations": self.min_iterations,
                "convergence_step_norms": convergence_step_norms,
                "step_converged": step_converged,
                "stop_reason": None,
            }
            self.last_optimization_trace.append(trace)
            scale = 1.0
            candidate = _navigation_plus(current, current_step)
            candidate_previous = (previous if previous_step is None else
                                  _navigation_plus(previous, previous_step))
            if not all(np.all(np.isfinite(value)) for state in
                       (candidate_previous, candidate) for value in
                       (state.pose, state.velocity, state.accel_bias, state.gyro_bias)):
                self.last_refinement_failure = "nonfinite_gn_state"
                trace["stop_reason"] = self.last_refinement_failure
                return None
            # Post-step costs are diagnostics, not an acceptance or line-search test.
            candidate_gelsight_pose = candidate.pose @ imu_T_gelsight
            if hasattr(visual, "residual_on_fixed_rows"):
                candidate_residual = visual.residual_on_fixed_rows(
                    reference_pose, candidate_gelsight_pose, active_rows
                )
            else:
                candidate_result = normal_evaluator(
                    reference_pose, candidate_gelsight_pose)
                candidate_residual = None if candidate_result is None else candidate_result[0]
            valid_normal = (
                candidate_residual is not None and len(candidate_residual) > 0
                and len(candidate_residual) % 3 == 0
                and np.all(np.isfinite(candidate_residual))
            )
            candidate_normal_cost = (_normal_huber_cost(
                candidate_residual, normal_scale, normal_huber_delta
            ) if valid_normal else None)
            candidate_values = _imu_values(candidate_previous, candidate)
            candidate_imu_cost = self.imu_weight * float(
                nonlinear_imu_factor.error(candidate_values)
            )
            candidate_bias_cost = self.imu_weight * float(
                nonlinear_bias_factor.error(candidate_values)
            )
            candidate_prior_cost = (0.0 if prior is None else
                                    prior.error(candidate_previous))
            candidate_height = self._height_z_linearization(
                visual, reference_pose, candidate_gelsight_pose,
                gel_from_imu_increment, jacobian=height_jacobian,
            )
            candidate_height_cost = None if candidate_height is None else candidate_height[2]
            valid_costs = valid_normal and candidate_height is not None and all(
                np.isfinite(cost) for cost in (candidate_normal_cost, candidate_imu_cost,
                    candidate_bias_cost, candidate_prior_cost, candidate_height_cost))
            candidate_cost = (candidate_normal_cost + candidate_imu_cost
                              + candidate_bias_cost + candidate_prior_cost
                              + candidate_height_cost) if valid_costs else None
            trace["attempts"].append({
                "scale": scale, "cost": candidate_cost, "reason": "full_gn_step",
                "imu_cost": candidate_imu_cost if np.isfinite(candidate_imu_cost) else None,
                "bias_cost": candidate_bias_cost if np.isfinite(candidate_bias_cost) else None,
                "height_cost": candidate_height_cost,
            })
            current = candidate
            previous = candidate_previous
            # Reuse this exact state pair for the next iteration and for the
            # final prior. Build another Values only after a new state update.
            values = candidate_values
            target_gelsight_pose = candidate_gelsight_pose
            # Height uses this state and fixed image pair, independent of the
            # next iteration's normal rows and Huber phase.
            height_terms = candidate_height
            inertial_costs = (candidate_imu_cost, candidate_bias_cost,
                              candidate_prior_cost)
            trace["accepted_scale"] = 1.0
            trace["accepted_cost"] = candidate_cost
            for term, cost in (("normal", candidate_normal_cost), ("imu", candidate_imu_cost),
                               ("bias", candidate_bias_cost), ("prior", candidate_prior_cost),
                               ("height", candidate_height_cost)):
                trace["accepted_" + term + "_cost"] = (
                    float(cost) if cost is not None and np.isfinite(cost) else None
                )
            trace["stop_reason"] = (
                "invalid_updated_residual" if not valid_costs else
                "step_converged" if step_converged else
                "max_iterations" if iteration + 1 == self.max_iterations else None
            )
            if not valid_costs:
                self.last_refinement_failure = trace["stop_reason"]
                return None
            if step_converged:
                break
        if prior is not None or (fixed_anchor and self.marginalization_enabled):
            evaluated = normal_evaluator(
                reference_pose, target_gelsight_pose)
            if evaluated is None:
                return None
            residual, _, J = evaluated
            if len(residual) < 30 or not np.all(np.isfinite(residual)):
                return None
            if self.normal_zero_pose_z_jacobian:
                J = J.copy()
                J[:, 5] = 0.0
            J = J @ gel_from_imu_increment
            normal_A, normal_b, _, _ = _normal_huber_linearization(
                residual, J, normal_scale, normal_huber_delta
            )
            # Use the last solve's normal loss for retention (ordinary least
            # squares after step five). Prior rows are always unrobustified.
            joint_Ap, joint_Ac, joint_rhs = self._joint_rows(
                inertial_factors, prior, previous, current, values=values)
            if height_terms is None:
                self.last_refinement_failure = "invalid_height_residual"
                return None
            height_A, height_b, _ = height_terms
            # Allocate the retained rows once, including the optional height row.
            # Numerical damping is excluded from the retained information.
            joint_end = len(joint_rhs)
            normal_end = joint_end + len(normal_b)
            row_count = normal_end + int(self.height_z_enabled)
            Ap = np.zeros((row_count, 15))
            Ac = np.zeros((row_count, 15))
            rhs = np.empty(row_count)
            Ap[:joint_end] = joint_Ap
            Ac[:joint_end] = joint_Ac
            rhs[:joint_end] = joint_rhs
            Ac[joint_end:normal_end, :6] = normal_A
            rhs[joint_end:normal_end] = -normal_b
            if self.height_z_enabled:
                Ac[normal_end:, :6] = height_A
                rhs[normal_end:] = -height_b
            information = (Ac.T @ Ac if fixed_anchor else
                           marginalize_rows_system(Ap, Ac, rhs).information)
            self.pending_marginal_prior = NavigationPrior.from_information(current, information)
        return current
