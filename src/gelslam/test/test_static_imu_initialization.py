"""Stationary startup, first-contact handoff, and configuration precedence."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from gelslam.core.inertial import (
    ImuBuffer, ImuSample, rotation_aligning_gravity_to_negative_z,
)
from gelslam.core.numpy_vi import so3_exp
from gelslam.core.static_imu_initialization import (
    StaticImuInitializer, resolve_static_imu_config,
)
from gelslam.core.tracker import Tracker
import gelslam.core.tracker as tracker_module
import gelslam.core.static_imu_initialization as initialization_module


def config(**updates):
    return dict({
        "enabled": True, "frontend_optimization": True,
        "gtsam_vi_marginalization": True, "gravity_mps2": 9.81,
        "gelsight_T_imu": np.eye(4).tolist(),
        "static_initialization": {"enabled": True},
    }, **updates)


def initializer(configuration=None, extrinsic=None):
    configuration = configuration or config()
    extrinsic = np.asarray(configuration["gelsight_T_imu"] if extrinsic is None else extrinsic)
    effective, settings, _ = resolve_static_imu_config(configuration, extrinsic)
    return StaticImuInitializer(settings, effective), ImuBuffer(), extrinsic


def feed(detector, buffer, samples):
    for sample in samples:
        buffer.add(sample)
    detector.observe(buffer.samples_between(0.0, samples[-1].timestamp))


def stationary_samples(end=2.0, acceleration=(0, 0, 9.81), gyro=(0, 0, 0), start=1.0):
    return [ImuSample(round(float(t), 9), acceleration, gyro)
            for t in np.arange(start, end + 0.0025, 0.005)]


def midpoint_approach_reference(omega, sensor_offset, world_force):
    """Closed-form reference for the rotating sample fixtures below.

    At 2.000 the sensor is stationary; at 2.005 its gyro is omega and
    its measured attitude is identity. Pair averaging adds a half gyro step
    before the remaining 0.495 s. The first interval rotates its mean force by
    a quarter step; subsequent midpoints retain the half-step attitude offset.
    Averaging two rotated force vectors
    scales the component perpendicular to omega by cos(|omega|*dt/2).
    The IMU-origin state integrates the measured centripetal force directly.
    No correction to the GelSight origin is applied to the measurements.
    """
    dt = .005
    axis = omega / np.linalg.norm(omega)
    force = np.asarray(world_force)
    parallel = axis * (axis @ force)
    averaged_force = parallel + np.cos(np.linalg.norm(omega) * dt / 2) * (force - parallel)
    gravity = np.array([0., 0., -9.81])
    centripetal = np.cross(omega, np.cross(omega, sensor_offset))
    first_force = .5 * (-gravity + force + centripetal)
    velocity = dt * (so3_exp(omega * (dt / 4)) @ first_force + gravity)
    velocity += .495 * (so3_exp(omega * (dt / 2)) @ averaged_force + gravity)
    velocity += dt * sum(so3_exp(omega * (dt + k * dt)) @ centripetal
                         for k in range(99))
    return so3_exp(omega * .4975), velocity


def predict(detector, buffer, extrinsic, timestamp):
    return detector.predict_trial(buffer, timestamp, extrinsic,
                                  np.repeat([0.01, 0.0001, 0.1, 0.1, 0.01], 3))


def header(timestamp):
    sec = int(timestamp)
    return SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=round((timestamp-sec)*1e9)))


@pytest.mark.parametrize("preset", ["config.yaml", "real_config.yaml", "bag_simulation_config.yaml"])
def test_all_presets_are_unchanged_when_off_and_resolve_gt_when_on(preset):
    path = Path(__file__).parents[1] / "config" / preset
    original = yaml.safe_load(path.read_text())["imu_config"]
    snapshot = yaml.safe_dump(original)
    disabled = dict(original, static_initialization=dict(original["static_initialization"], enabled=False))
    effective, settings, overrides = resolve_static_imu_config(disabled, np.eye(4))
    assert effective == disabled and settings is None and overrides == ()
    enabled = dict(original, static_initialization=dict(original["static_initialization"], enabled=True))
    effective, settings, _ = resolve_static_imu_config(enabled, np.eye(4))
    assert effective["frontend_gravity_initialization"] == "static_imu"
    assert not effective["frontend_first_trial_velocity_from_gt"]
    assert effective["gravity_initialization"] == "measured_acceleration"
    assert "frontend_gravity_mps2" not in effective
    assert settings["enabled"]
    assert yaml.safe_dump(original) == snapshot


@pytest.mark.parametrize("update,match", [
    ({"enabled": False}, "requires.*enabled"),
    ({"frontend_optimization": False}, "requires.*frontend_optimization"),
    ({"gtsam_vi_marginalization": False}, "requires.*gtsam_vi_marginalization"),
    ({"buffer_max_samples": 50}, "buffer_max_samples"),
    ({"gravity_mps2": 0}, "gravity_mps2"),
    ({"accelerometer_bias_random_walk": 0}, "accelerometer_bias_random_walk"),
    ({"initial_accel_bias_mps2": [0, np.nan, 0]}, "initial_accel_bias"),
    ({"static_initialization": {"enabled": "false"}}, "boolean"),
    ({"static_initialization": {"enabled": True, "window_s": 0}}, "window_s"),
    ({"static_initialization": {"enabled": True, "window_s": np.nan}}, "window_s"),
    ({"static_initialization": {"enabled": True, "min_samples": 4.2}}, "min_samples"),
    ({"static_initialization": {"enabled": True, "max_propagation_s": 0.1}}, "half window_s"),
    ({"static_initialization": {"enabled": True, "windwo_s": 1}}, "unknown"),
])
def test_invalid_combinations_fail_before_tracker_start(update, match):
    with pytest.raises(ValueError, match=match):
        resolve_static_imu_config(config(**update), np.eye(4))


def test_requires_calibration_and_rotation_not_scale():
    with pytest.raises(ValueError, match="calibrated"):
        resolve_static_imu_config(config(), None)
    with pytest.raises(ValueError, match="orthonormal"):
        resolve_static_imu_config(config(), np.diag([2, 1, 1, 1]))


def fixed_config(start=1.0, end=2.0, **options):
    return config(static_initialization=dict(
        enabled=True, mode="fixed_time", start_time_s=start, end_time_s=end, **options))


def startup_config(duration=1.0, **options):
    return config(static_initialization=dict(
        enabled=True, mode="startup_duration", duration_s=duration, **options))


@pytest.mark.parametrize("duration", [None, False, 0, -1, np.nan, np.inf])
def test_startup_duration_requires_positive_finite_seconds(duration):
    with pytest.raises(ValueError, match="duration_s"):
        initializer(startup_config(duration))


@pytest.mark.parametrize("options", [
    {"mode": "fixed_time", "start_time_s": 1., "end_time_s": 2., "duration_s": 1.},
    {"mode": "stationary_detection", "duration_s": 1.},
    {"mode": "startup_duration", "duration_s": 1., "start_time_s": 1., "end_time_s": 2.},
])
def test_startup_duration_cannot_mix_relative_and_absolute_settings(options):
    with pytest.raises(ValueError, match="require"):
        initializer(config(static_initialization=dict(enabled=True, **options)))


@pytest.mark.parametrize("start", [0., 41., 1792046893.])
def test_startup_duration_uses_first_imu_time_and_freezes_before_approach(monkeypatch, start):
    detector, buffer, extrinsic = initializer(startup_config())
    monkeypatch.setattr(detector, "_statistics", lambda _: pytest.fail("stationary test must not run"))
    detector.observe(())
    rest = [ImuSample(start + i * .005, [0, 0, 9.81], [0, 0, 0]) for i in range(201)]
    feed(detector, buffer, rest[:101])
    assert detector.stationary is None
    feed(detector, buffer, rest[101:])
    assert detector.stationary[0] == start + 1.
    feed(detector, buffer, [ImuSample(start + i * .005, [0.2, 0, 9.81], [0, 0, 0])
                            for i in range(201, 302)])
    seed = predict(detector, buffer, extrinsic, start + 1.5)
    assert seed is not None
    assert seed.stationary_timestamp == start + 1.
    assert seed.state.velocity[0] == pytest.approx(.0995, abs=1e-6)
    np.testing.assert_allclose(detector.stationary[1], [0, 0, 9.81], rtol=0, atol=1e-12)


def test_startup_duration_does_not_shift_start_when_initial_measurements_are_missing():
    detector, _, _ = initializer(startup_config())
    detector.observe(stationary_samples(start=1.1, end=2.1), stream_start_time=1.)
    assert detector.stationary is None
    assert "no start coverage" in detector.status


def test_real_far_preset_uses_first_five_imu_seconds():
    cfg = yaml.safe_load((Path(__file__).parents[1] / "config" / "real_far_config.yaml").read_text())
    _, settings, _ = resolve_static_imu_config(cfg["imu_config"], np.array(cfg["imu_config"]["gelsight_T_imu"]))
    assert settings["enabled"]
    assert settings["mode"] == "startup_duration" and settings["duration_s"] == 5.
    assert settings["start_time_s"] is None and settings["end_time_s"] is None


@pytest.mark.parametrize("options,match", [
    ({"mode": "unknown"}, "mode"),
    ({"mode": "fixed_time"}, "start_time_s"),
    ({"mode": "fixed_time", "start_time_s": 1}, "end_time_s"),
    ({"mode": "fixed_time", "start_time_s": -1, "end_time_s": 2}, "start_time_s"),
    ({"mode": "fixed_time", "start_time_s": 1, "end_time_s": np.nan}, "end_time_s"),
    ({"mode": "fixed_time", "start_time_s": 2, "end_time_s": 2}, "after start_time_s"),
    ({"start_time_s": 1, "end_time_s": 2}, "require mode"),
])
def test_fixed_interval_config_guards(options, match):
    with pytest.raises(ValueError, match=match):
        resolve_static_imu_config(config(static_initialization=dict(enabled=True, **options)), np.eye(4))


def test_fixed_interval_mean_and_anchor_never_follow_approach_motion(monkeypatch):
    detector, buffer, extrinsic = initializer(fixed_config())
    monkeypatch.setattr(detector, "_statistics", lambda _: pytest.fail("stationary test must not run"))
    feed(detector, buffer, stationary_samples(gyro=(0.001, -0.002, 0.003)))
    original = tuple(np.copy(value) for value in detector.stationary[1:])
    assert detector.stationary[0] == 2.0
    approach = [ImuSample(round(float(t), 9), (0.2, 0, 9.81), (0.001, -0.002, 0.003))
                for t in np.arange(2.005, 2.5025, 0.005)]
    feed(detector, buffer, approach)
    seed = predict(detector, buffer, extrinsic, 2.5)
    assert seed.stationary_timestamp == 2.0
    # First pair: .1 m/s^2 for .005 s; remaining .495 s: .2 m/s^2.
    assert np.allclose(seed.state.velocity, [0.0995, 0, 0], atol=1e-8)
    assert all(np.array_equal(before, after) for before, after in zip(original, detector.stationary[1:]))
    assert not detector.motion_detected


def test_fixed_interval_bypasses_variance_and_motion_thresholds():
    detector, buffer, _ = initializer(fixed_config(
        window_s=0, accel_std_threshold_mps2=0, gyro_mean_threshold_rps=0))
    samples = [ImuSample(round(float(t), 9), (0.3 * (-1)**i, 0, 9.81), (0, 0, 0.2))
               for i, t in enumerate(np.arange(1, 2.005, 0.005))]
    feed(detector, buffer, samples)
    assert detector.stationary is not None and detector.stationary[0] == 2.0
    assert np.allclose(detector.stationary[2], [0, 0, 0.2])


def test_fixed_interval_requires_endpoint_data_and_interpolates_boundaries():
    detector, buffer, _ = initializer(fixed_config(start=1.0025, end=1.9975))
    feed(detector, buffer, stationary_samples(end=1.995))
    assert detector.stationary is None
    feed(detector, buffer, [ImuSample(2.0, (0, 0, 9.81), (0, 0, 0))])
    assert detector.stationary[0] == 1.9975
    assert np.allclose(detector.stationary[1], [0, 0, 9.81])


@pytest.mark.parametrize("start,gap,reason", [
    (1.1, False, "no start coverage"),
    (1.0, True, "raw sample gap"),
])
def test_fixed_interval_rejects_missing_start_or_raw_gaps(start, gap, reason):
    detector, buffer, _ = initializer(fixed_config())
    samples = stationary_samples(start=start)
    if gap:
        samples = [sample for sample in samples if not 1.4 < sample.timestamp < 1.6]
    feed(detector, buffer, samples)
    assert detector.stationary is None and reason in detector.status


@pytest.mark.parametrize("contact_at_anchor", [False, True])
def test_stationary_tilt_bias_and_radial_accel_bias(monkeypatch, contact_at_anchor):
    direction = np.array([0.3, -0.4, np.sqrt(0.75)])
    gyro = np.array([0.001, -0.002, 0.003])
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = so3_exp(np.array([0.2, -0.1, 0.3]))
    extrinsic[:3, 3] = [.04, -.02, .03]
    detector, buffer, extrinsic = initializer(extrinsic=extrinsic)
    feed(detector, buffer, stationary_samples(acceleration=direction * 9.91, gyro=gyro))
    integrate = initialization_module.preintegrate_imu
    integration_inputs = []

    def check_aligned_start(samples, previous, config, transform, gravity):
        integration_inputs.append(previous.copy())
        assert np.array_equal(gravity, [0., 0., -9.81])
        assert np.allclose(previous.pose[:3, :3] @ (direction * 9.81) + gravity, 0.)
        gel_pose = previous.pose @ np.linalg.inv(transform)
        assert np.allclose(gel_pose[:3, 3], 0.)
        return integrate(samples, previous, config, transform, gravity)

    monkeypatch.setattr(initialization_module, "preintegrate_imu", check_aligned_start)
    reframe = initialization_module.NavigationPrior.reframe

    def check_contact_heading(prior, rotation, origin):
        assert np.allclose(rotation @ [0., 0., -9.81], [0., 0., -9.81])
        return reframe(prior, rotation, origin)

    monkeypatch.setattr(initialization_module.NavigationPrior, "reframe", check_contact_heading)
    timestamp = detector.stationary[0] if contact_at_anchor else 1.9975
    seed = predict(detector, buffer, extrinsic, timestamp)
    assert seed is not None
    assert len(integration_inputs) == (0 if contact_at_anchor else 1)
    contact_rotation = rotation_aligning_gravity_to_negative_z(
        -extrinsic[:3, :3] @ direction
    )
    expected_pose = np.eye(4)
    expected_pose[:3, :3] = contact_rotation
    assert np.allclose(seed.state.pose, expected_pose @ extrinsic)
    assert np.linalg.norm(seed.state.velocity) < 1e-10
    assert np.array_equal(seed.gravity, [0., 0., -9.81])
    assert np.allclose(seed.state.gyro_bias, gyro)
    assert np.allclose(seed.state.accel_bias, 0.1 * direction)
    assert seed.prior.matches(seed.state)
    assert np.array_equal(seed.prior.rhs, np.zeros(15))
    assert seed.prior.error(seed.state) == pytest.approx(0., abs=1e-20)


@pytest.mark.parametrize("prediction_enabled", [False, True])
def test_approach_velocity_is_nonzero_even_with_contact_prediction_off(prediction_enabled):
    detector, buffer, extrinsic = initializer(config(
        frontend_initialize_from_imu_preintegration=prediction_enabled))
    samples = stationary_samples()
    samples += [ImuSample(round(float(t), 9), (0.2, 0, 9.81), (0, 0, 0))
                for t in np.arange(2.005, 2.5025, 0.005)]
    feed(detector, buffer, samples)
    seed = predict(detector, buffer, extrinsic, 2.5)
    assert seed is not None and detector.motion_detected
    assert np.allclose(seed.state.velocity, [0.0995, 0, 0], atol=1e-8)
    assert np.allclose(seed.state.pose, np.eye(4))
    assert np.allclose(seed.gravity, [0, 0, -9.81])
    covariance = np.linalg.inv(seed.prior.A.T @ seed.prior.A)
    assert covariance[6, 6] > detector.settings["velocity_sigma_mps"]**2
    assert abs(covariance[6, 9]) > 1e-5  # Velocity / accel-bias correlation survives.


def test_rotating_offset_sensor_rebases_imu_velocity_and_gravity():
    extrinsic = np.eye(4)
    extrinsic[:3, 3] = [0.1, -0.05, 0.02]
    detector, buffer, extrinsic = initializer(extrinsic=extrinsic)
    samples = stationary_samples()
    omega = np.array([0.0, 0.0, 0.5])
    for time in np.arange(2.005, 2.5025, 0.005):
        # Gel origin translates at a_G=(0.2,0,0); IMU has centripetal force.
        rotation = so3_exp(omega * (time - 2.005))
        acceleration = rotation.T @ np.array([0.2, 0, 9.81])
        acceleration += np.cross(omega, np.cross(omega, extrinsic[:3, 3]))
        samples.append(ImuSample(round(float(time), 9), acceleration, omega))
    feed(detector, buffer, samples)
    seed = predict(detector, buffer, extrinsic, 2.5)
    rotation, velocity = midpoint_approach_reference(
        omega, extrinsic[:3, 3], [0.2, 0, 9.81]
    )
    assert np.allclose(seed.state.velocity, rotation.T @ velocity, atol=1e-8)
    assert np.allclose(seed.gravity, rotation.T @ [0, 0, -9.81], atol=1e-8)
    assert np.allclose(seed.state.pose, extrinsic)


def test_tilt_during_approach_with_rotated_sensor_mount():
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = so3_exp(np.array([0.2, -0.4, 0.5]))
    extrinsic[:3, 3] = [0.1, -0.05, 0.02]
    detector, buffer, extrinsic = initializer(extrinsic=extrinsic)
    sensor_rotation = extrinsic[:3, :3]
    samples = stationary_samples(acceleration=sensor_rotation.T @ [0, 0, 9.81])
    omega = np.array([0.5, 0, 0])
    for time in np.arange(2.005, 2.5025, 0.005):
        rotation = so3_exp(omega * (time - 2.005))
        force_gel = rotation.T @ [0, 0.2, 9.81]
        force_gel += np.cross(omega, np.cross(omega, extrinsic[:3, 3]))
        samples.append(ImuSample(round(float(time), 9), sensor_rotation.T @ force_gel,
                                 sensor_rotation.T @ omega))
    feed(detector, buffer, samples)
    seed = predict(detector, buffer, extrinsic, 2.5)
    rotation, velocity = midpoint_approach_reference(
        omega, extrinsic[:3, 3], [0, 0.2, 9.81]
    )
    contact_rotation = rotation_aligning_gravity_to_negative_z(rotation.T @ [0, 0, -9.81])
    assert np.array_equal(seed.gravity, [0., 0., -9.81])
    assert np.allclose(seed.state.velocity, contact_rotation @ rotation.T @ velocity, atol=1e-8)
    expected_pose = np.eye(4)
    expected_pose[:3, :3] = contact_rotation
    assert np.allclose(seed.state.pose, expected_pose @ extrinsic)
    covariance = np.linalg.inv(seed.prior.A.T @ seed.prior.A)
    assert np.min(np.linalg.eigvalsh(covariance)) > 0
    assert abs(covariance[7, 8]) > 1e-7


@pytest.mark.parametrize("acceleration,gyro", [
    ((0, 0, 0), (0, 0, 0)),  # Gravity-removed input is invalid.
    ((0, 0, 9.81), (0, 0, 0.2)),  # Constant rotation is not rest.
])
def test_nonstationary_or_gravity_removed_data_do_not_initialize(acceleration, gyro):
    detector, buffer, extrinsic = initializer()
    feed(detector, buffer, stationary_samples(acceleration=acceleration, gyro=gyro))
    assert predict(detector, buffer, extrinsic, 2.0) is None


def test_vibration_and_insufficient_window_do_not_initialize():
    detector, buffer, extrinsic = initializer()
    samples = [ImuSample(round(float(t), 9), (0.3 * (-1)**i, 0, 9.81), (0, 0, 0))
               for i, t in enumerate(np.arange(1, 2.005, 0.005))]
    feed(detector, buffer, samples)
    assert predict(detector, buffer, extrinsic, 2.0) is None
    detector, buffer, extrinsic = initializer()
    feed(detector, buffer, stationary_samples(end=1.5))
    assert predict(detector, buffer, extrinsic, 1.5) is None


def test_missing_future_imu_and_large_raw_gap_do_not_extrapolate():
    detector, buffer, extrinsic = initializer()
    feed(detector, buffer, stationary_samples())
    assert predict(detector, buffer, extrinsic, 2.1) is None
    # The interpolated contact endpoint alone must not hide this 1 s raw gap.
    buffer.add(ImuSample(3.0, (0, 0, 9.81), (0, 0, 0)))
    assert predict(detector, buffer, extrinsic, 2.01) is None


def test_approach_timeout_and_sparse_window_are_rejected():
    detector, buffer, extrinsic = initializer()
    samples = stationary_samples()
    samples += [ImuSample(round(float(t), 9), (0.2, 0, 9.81), (0, 0, 0))
                for t in np.arange(2.005, 7.005, 0.005)]
    feed(detector, buffer, samples)
    assert predict(detector, buffer, extrinsic, 7.0) is None
    assert "max_propagation_s" in detector.status
    detector, buffer, extrinsic = initializer()
    samples = stationary_samples(end=1.3) + stationary_samples(start=1.6, end=2.0)
    feed(detector, buffer, samples)
    assert predict(detector, buffer, extrinsic, 2.0) is None


def make_tracker(monkeypatch, enabled=True):
    logs = []
    logger = SimpleNamespace(info=logs.append, warning=logs.append)
    monkeypatch.setattr(tracker_module, "Reconstructor", lambda _: SimpleNamespace(
        get_surface_info=lambda *a, **k: (None, None, None), load_bg=lambda _: None))
    contacted = [True]
    monkeypatch.setattr(tracker_module, "Frame", lambda *a, **k: SimpleNamespace(
        is_contacted=contacted[0], N=None, C=None, H=None, L=None))
    monkeypatch.setattr(tracker_module, "create_keyframe_msg", lambda *a, **k: k)
    monkeypatch.setattr(tracker_module, "create_frame_msg", lambda *a, **k: k)
    cfg = {"device_config": {"ppmm": 1.0},
           "contact_mask_config": {"color_dist_threshold": 1, "height_threshold": 1},
           "gelslam_config": {"track_config": {"contact_threshold": 1}},
           "imu_config": config(static_initialization={"enabled": enabled})}
    tracker = Tracker("unused", cfg, True, logger)
    tracker.bg_images = [object()] * 10
    return tracker, contacted, logs


def test_first_gelsight_contact_is_at_origin_with_an_offset_imu_state(monkeypatch):
    tracker, contacted, _ = make_tracker(monkeypatch)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = so3_exp(np.array([.2, -.3, .4]))
    extrinsic[:3, 3] = [.04, -.02, .03]
    tracker.gelsight_T_imu = extrinsic
    for sample in stationary_samples(end=2.3):
        tracker.add_imu_sample(sample)
    valid, (frame, keyframes) = tracker.track(np.zeros((2, 2, 3)), header(2.0))
    assert valid
    gel_pose = keyframes[0]["trial_T_curr"]
    assert np.allclose(gel_pose[:3, 3], 0., atol=1e-12)
    assert np.allclose(frame["ref_T_curr"], np.eye(4))
    assert np.allclose(tracker._frontend_vi_state.pose, gel_pose @ extrinsic, atol=1e-12)
    assert np.linalg.norm(tracker._frontend_vi_state.pose[:3, 3]) > .05
    assert tracker.frontend_visual_inertial.marginal_prior.matches(tracker._frontend_vi_state)
    assert np.allclose(tracker.frontend_gravity_for_backend(), [0, 0, -9.81])
    assert not np.allclose(gel_pose[:3, :3], np.eye(3))
    contacted[0] = False
    assert tracker._initialize_no_contact_coast()
    assert tracker._coast_to_header(header(2.2), tracker_module.TrackingMode.NO_CONTACT_COASTING)
    assert np.allclose(tracker.trial_T_ref @ tracker.ref_T_prev, gel_pose, atol=1e-7)


def test_aligned_contact_state_prior_and_coasting_use_one_world(monkeypatch):
    direction = np.array([.3, -.4, np.sqrt(.75)])
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    tracker, contact, _ = make_tracker(monkeypatch)
    contact[0] = False
    for sample in stationary_samples(acceleration=direction * 9.81):
        tracker.add_imu_sample(sample)
    tracker.track(image, header(2.0))
    for t in np.arange(2.005, 2.5025, .005):
        tracker.add_imu_sample(ImuSample(round(float(t), 9),
            direction * 9.81 + [.2, 0., 0.], [.08, -.02, .1]))

    # The tracker must use the initializer's aligned state and complete prior.
    captured = {}
    predict_trial = tracker._static_imu_initializer.predict_trial

    def capture_seed(*args):
        seed = predict_trial(*args)
        captured["state"] = seed.state.copy()
        captured["A"] = seed.prior.A.copy()
        captured["rhs"] = seed.prior.rhs.copy()
        return seed

    monkeypatch.setattr(tracker._static_imu_initializer, "predict_trial", capture_seed)
    monkeypatch.setattr(tracker, "_align_initial_navigation",
                        lambda *args: pytest.fail("static startup must already be aligned"))
    contact[0] = True
    ok, (_, nodes) = tracker.track(image, header(2.5))
    assert ok
    state = tracker._frontend_vi_state
    expected = captured["state"]
    assert np.array_equal(state.pose[:3, 3], np.zeros(3))
    assert not np.allclose(state.pose[:3, :3], np.eye(3))
    assert np.allclose(nodes[0]["trial_T_curr"], expected.pose)
    assert np.array_equal(state.pose, expected.pose)
    assert np.array_equal(state.velocity, expected.velocity)
    assert np.array_equal(state.accel_bias, expected.accel_bias)
    assert np.array_equal(state.gyro_bias, expected.gyro_bias)
    assert np.array_equal(tracker.frontend_gravity_for_backend(), [0., 0., -9.81])
    prior = tracker.frontend_visual_inertial.marginal_prior
    assert prior.matches(state)
    assert np.array_equal(prior.A, captured["A"])
    assert np.array_equal(prior.rhs, captured["rhs"])
    covariance = np.linalg.inv(prior.A.T @ prior.A)
    assert np.linalg.norm(covariance[6:9, 9:12]) > 1e-5

    assert tracker._initialize_no_contact_coast()
    for t in np.arange(2.505, 2.6025, .005):
        tracker.add_imu_sample(ImuSample(round(float(t), 9), direction * 9.81, [.08, -.02, .1]))
    samples = tracker.imu_buffer.samples_covering(2.5, 2.6)
    predicted, _ = tracker.frontend_visual_inertial.predict_current(
        expected, samples, tracker.gelsight_T_imu)
    assert tracker._coast_to_header(header(2.6), tracker_module.TrackingMode.NO_CONTACT_COASTING)
    assert np.allclose(tracker._frontend_coast_state.pose, predicted.pose, atol=1e-10)
    assert np.allclose(tracker._frontend_coast_state.velocity, predicted.velocity, atol=1e-10)


def test_aligned_gt_initialization_waits_without_publishing_identity(monkeypatch):
    tracker, _, _ = make_tracker(monkeypatch, enabled=False)
    tracker.imu_config.update(frontend_gravity_initialization="ground_truth_initial_pose",
        frontend_first_trial_velocity_from_gt=True)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    assert tracker.track(image, header(1.0)) == (False, None)
    assert tracker.curr_kid == -1
    rotation = so3_exp(np.array([.4, -.3, .2]))
    world_velocity = np.array([.01, -.02, .03])
    for t in (.995, 1., 1.005):
        tracker.add_ground_truth_gel_pose(t, rotation, world_velocity * t)
        tracker.add_ground_truth_imu_position(t, world_velocity * t)
    assert tracker._frontend_vi_state.timestamp == 1.0
    ok, (frame, nodes) = tracker.track(image, header(1.05))
    assert ok and len(nodes) == 1 and tracker._trial_index == 0
    assert nodes[0]["header"].stamp.sec == 1
    assert nodes[0]["header"].stamp.nanosec == 0
    assert frame["header"].stamp.nanosec == 0
    assert np.allclose(nodes[0]["trial_T_curr"], tracker._frontend_vi_state.pose)
    assert np.allclose(tracker._frontend_vi_state.velocity,
        tracker._trial_gravity_rotation @ rotation.T @ world_velocity)
    assert np.allclose(tracker.frontend_gravity_for_backend(), [0, 0, -9.81])


def test_first_node_waits_for_rest_then_transfers_prior_and_starts_coasting(monkeypatch):
    tracker, contacted, logs = make_tracker(monkeypatch)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    for sample in stationary_samples(end=1.5):
        tracker.add_imu_sample(sample)
    assert tracker.track(image, header(1.5)) == (False, None)
    assert tracker.curr_kid == -1 and tracker.ref_frame is None
    # Return to no contact before acquiring a complete stationary window.
    contacted[0] = False
    for sample in stationary_samples(start=1.505):
        tracker.add_imu_sample(sample)
    assert tracker.track(image, header(2.0)) == (False, None)
    for t in np.arange(2.005, 2.5025, 0.005):
        tracker.add_imu_sample(ImuSample(round(float(t), 9), (0.2, 0, 9.81), (0, 0, 0)))
    contacted[0] = True
    valid, (_, keyframes) = tracker.track(image, header(2.5))
    assert valid and len(keyframes) == 1 and tracker.curr_kid == 0
    assert keyframes[0]["is_new_trial"]
    assert np.allclose(keyframes[0]["trial_T_curr"], np.eye(4))
    assert np.isclose(tracker._frontend_vi_state.velocity[0], 0.0995)
    assert tracker.frontend_visual_inertial.marginal_prior.matches(tracker._frontend_vi_state)
    assert np.allclose(tracker.frontend_gravity_for_backend(), [0, 0, -9.81])
    # The next gap copies exactly this accepted GelSight state.
    assert tracker._initialize_no_contact_coast()
    assert np.allclose(tracker._frontend_coast_state.velocity, [0.0995, 0, 0])
    assert not tracker._prepare_static_trial(header(2.5))  # Never reuse the consumed anchor.
    assert any("stationary anchor" in message for message in logs)


def test_contact_only_stationary_data_never_complete_failed_initialization(monkeypatch):
    tracker, _, _ = make_tracker(monkeypatch)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    for sample in stationary_samples(end=1.5):
        tracker.add_imu_sample(sample)
    assert tracker.track(image, header(1.5)) == (False, None)
    for sample in stationary_samples(start=1.505):
        tracker.add_imu_sample(sample)
    assert tracker.track(image, header(2.0)) == (False, None)
    assert tracker.curr_kid == -1


def test_new_trial_requires_new_no_contact_window(monkeypatch):
    tracker, _, _ = make_tracker(monkeypatch)
    for sample in stationary_samples():
        tracker.add_imu_sample(sample)
    assert tracker.track(np.zeros((2, 2, 3)), header(2.0))[0]
    accepted = tracker._frontend_vi_state.copy()
    assert not tracker._prepare_static_trial(header(2.0))
    for sample in stationary_samples(start=2.005, end=3.2):
        tracker.add_imu_sample(sample)
    tracker._observe_static_imu(header(2.1))
    assert tracker._static_imu_initializer.stationary is None  # Old contact evidence excluded.
    tracker._observe_static_imu(header(3.2))
    assert tracker._prepare_static_trial(header(3.2))
    tracker._trial_index = 1
    tracker._reset_frontend_visual_inertial(np.eye(4), header(3.2))
    assert tracker._frontend_vi_state.timestamp == 3.2
    assert accepted.timestamp == 2.0
    assert tracker.frontend_visual_inertial.marginal_prior.matches(tracker._frontend_vi_state)


def test_fixed_clock_offset_handoff_and_timeout_do_not_reset_interval(monkeypatch):
    tracker, _, _ = make_tracker(monkeypatch)
    tracker.imu_config, tracker._static_imu_settings, _ = resolve_static_imu_config(fixed_config(), np.eye(4))
    tracker._static_imu_initializer = StaticImuInitializer(tracker._static_imu_settings, tracker.imu_config)
    tracker.image_to_imu_time_offset_s = 0.025
    tracker.timestamp_source = "recorded_source"
    for sample in stationary_samples():
        tracker.add_imu_sample(sample)
    h = header(20.0)
    tracker.set_recorded_image_timestamp(h, 1.975)
    tracker._observe_static_imu(h)
    assert tracker._static_imu_initializer.stationary[0] == 2.0
    assert tracker._prepare_static_trial(h)
    assert tracker._static_trial_seed.state.timestamp == 2.0
    # Keep the same absolute interval even after its approach deadline expires.
    later = header(30.0)
    tracker.set_recorded_image_timestamp(later, 7.975)
    tracker._observe_static_imu(later)
    assert tracker._static_imu_initializer.stationary[0] == 2.0
    assert not tracker._prepare_static_trial(later)
    assert "no later rest window" in tracker._static_imu_initializer.status


def test_startup_duration_uses_accepted_imu_origin_before_first_image(monkeypatch):
    tracker, _, _ = make_tracker(monkeypatch)
    tracker.imu_config, tracker._static_imu_settings, _ = resolve_static_imu_config(startup_config(), np.eye(4))
    tracker._static_imu_initializer = StaticImuInitializer(tracker._static_imu_settings, tracker.imu_config)
    for sample in stationary_samples(end=2.1):
        assert tracker.add_imu_sample(sample)
    assert not tracker.add_imu_sample(ImuSample(.5, [0, 0, 9.81], [0, 0, 0]))
    assert tracker._imu_stream_start_time == 1.
    tracker.timestamp_source = "recorded_source"
    tracker.image_to_imu_time_offset_s = .025
    h = header(100.)
    tracker.set_recorded_image_timestamp(h, 1.475)
    tracker._observe_static_imu(h)
    assert tracker._static_imu_initializer.stationary is None  # Future IMU must not complete the interval.
    tracker.set_recorded_image_timestamp(h, 1.975)
    tracker._observe_static_imu(h)
    assert tracker._static_imu_initializer.stationary[0] == 2.
    assert tracker._prepare_static_trial(h)
    assert tracker._static_trial_seed.state.timestamp == 2.
    # The same startup interval remains fixed after its propagation deadline.
    tracker.set_recorded_image_timestamp(h, 7.975)
    tracker._observe_static_imu(h)
    assert tracker._static_imu_initializer.stationary[0] == 2.
    assert not tracker._prepare_static_trial(h)
    assert "no later rest window" in tracker._static_imu_initializer.status
    # A later disconnected trial must not treat its first image as a new origin.
    tracker._static_imu_collecting = False
    for sample in stationary_samples(start=8., end=9.1):
        tracker.add_imu_sample(sample)
    tracker.set_recorded_image_timestamp(h, 8.975)
    tracker._observe_static_imu(h)
    assert not tracker._prepare_static_trial(h)
    assert "no start coverage" in tracker._static_imu_initializer.status


def test_timed_out_approach_can_recover_after_new_no_contact_rest(monkeypatch):
    tracker, _, _ = make_tracker(monkeypatch)
    for sample in stationary_samples():
        tracker.add_imu_sample(sample)
    tracker._observe_static_imu(header(2.0))
    for t in np.arange(2.005, 7.005, 0.005):
        tracker.add_imu_sample(ImuSample(round(float(t), 9), (0.2, 0, 9.81), (0, 0, 0)))
    tracker._observe_static_imu(header(3.0))  # Freeze the stationary anchor before motion.
    tracker._observe_static_imu(header(7.0))  # Expire it without using moving data as rest.
    assert tracker._static_imu_initializer.stationary is None
    for sample in stationary_samples(start=7.005, end=8.1):
        tracker.add_imu_sample(sample)
    tracker._observe_static_imu(header(8.1))
    assert tracker._prepare_static_trial(header(8.1))
    assert tracker._static_trial_seed.stationary_timestamp > 7.0


def test_off_registers_first_contact_without_imu_rest(monkeypatch):
    tracker, _, _ = make_tracker(monkeypatch, enabled=False)
    valid, (_, keyframes) = tracker.track(np.zeros((2, 2, 3)), header(1.0))
    assert valid and len(keyframes) == 1
    assert np.allclose(tracker._frontend_vi_state.velocity, np.zeros(3))


def test_image_time_offset_and_no_future_stationary_evidence(monkeypatch):
    tracker, contacted, _ = make_tracker(monkeypatch)
    tracker.image_to_imu_time_offset_s = 0.025
    tracker.timestamp_source = "recorded_source"
    # IMU callbacks are ahead, but future rest data must not initialize t=1.5.
    for sample in stationary_samples():
        tracker.add_imu_sample(sample)
    early = header(10.0)
    tracker.set_recorded_image_timestamp(early, 1.475)
    image = np.zeros((2, 2, 3))
    assert tracker.track(image, early) == (False, None)
    contacted[0] = False
    later = header(11.0)
    tracker.set_recorded_image_timestamp(later, 1.975)
    tracker.track(image, later)
    contacted[0] = True
    valid, _ = tracker.track(image, later)
    assert valid
    assert np.isclose(tracker._frontend_vi_state.timestamp, 2.0)


def event_config(**options):
    return config(static_initialization=dict(enabled=True, mode="recorded_rest_event", **options))


REST = {"recorded_rest_started": 1.0, "recorded_rest_completed": 2.0, "trial_ready": 2.2}


def segment(start, end, acceleration=(0, 0, 9.81), gyro=(0, 0, 0)):
    return [ImuSample(round(float(t), 9), acceleration, gyro)
            for t in np.arange(start, end - 0.0025, 0.005)]


@pytest.mark.parametrize("cfg,match", [
    (config(static_initialization=dict(enabled=True, mode="fixed_time", start_time_s=1.,
                                       end_time_s=2., motion_check=True)), "motion_check requires"),
    (event_config(motion_check=1), "motion_check must be a boolean"),
    (event_config(motion_check=True, motion_pre_peak_margin_s=0), "motion_pre_peak_margin_s"),
    (event_config(events_topic=""), "events_topic"),
])
def test_recorded_rest_event_config_guards(cfg, match):
    with pytest.raises(ValueError, match=match):
        resolve_static_imu_config(cfg, np.eye(4))


def test_recorded_rest_event_waits_for_events_and_propagates_from_go():
    detector, buffer, extrinsic = initializer(event_config())
    samples = segment(0.5, 2.6) + segment(2.6, 3.005, acceleration=(0.2, 0, 9.81))
    for sample in samples:
        buffer.add(sample)
    detector.observe(buffer.samples_between(0., 3.0), rest_events={"trial_ready": 2.2})
    assert detector.stationary is None and "recorded_rest_started" in detector.status
    detector.observe(buffer.samples_between(0., 3.0), rest_events=dict(REST, trial_ready=1.5))
    assert detector.stationary is None and "not ordered" in detector.status
    detector.observe(buffer.samples_between(0., 3.0), rest_events=REST)
    assert detector.stationary[0] == 2.2 and "GO=2.2" in detector.status
    np.testing.assert_allclose(detector.stationary[1], [0, 0, 9.81], atol=1e-12)
    seed = predict(detector, buffer, extrinsic, 3.0)
    assert seed.stationary_timestamp == 2.2
    # Zero velocity at GO; the first motion pair contributes half a step.
    assert seed.state.velocity[0] == pytest.approx(.1 * .005 + .2 * .4, abs=1e-9)


def test_recorded_rest_event_excludes_samples_outside_the_rest_interval():
    detector, buffer, _ = initializer(event_config())
    samples = (segment(0.5, 1.0, acceleration=(1, 0, 9.81)) + segment(1.0, 2.005)
               + segment(2.005, 2.3, acceleration=(1, 0, 9.81)))
    for sample in samples:
        buffer.add(sample)
    detector.observe(buffer.samples_between(0., 2.3), rest_events=REST)
    # Interpolated endpoints include half of each neighbouring sample's offset.
    assert abs(detector.stationary[1][0]) < 1. * .005 / 1.0 + 1e-9


# After the peak at 2.6: the 2.595/2.6 pair averages .15 m/s^2, then .25 m/s^2.
MOTION_VELOCITY = .15 * .005 + .25 * .4


@pytest.mark.parametrize("motion_check,anchor,velocity", [
    (False, 2.2, .05 * .395 + MOTION_VELOCITY),  # Sub-threshold offset integrates from GO.
    (True, 2.5, .05 * .095 + MOTION_VELOCITY),   # Zero velocity restated 0.1 s before the peak.
])
def test_motion_check_restates_zero_velocity_before_first_peak(motion_check, anchor, velocity):
    detector, buffer, extrinsic = initializer(event_config(motion_check=motion_check))
    samples = (segment(0.5, 2.2) + segment(2.2, 2.6, acceleration=(0.05, 0, 9.81))
               + segment(2.6, 3.005, acceleration=(0.25, 0, 9.81)))
    for sample in samples:
        buffer.add(sample)
    detector.observe(buffer.samples_between(0., 3.0), rest_events=REST)
    seed = predict(detector, buffer, extrinsic, 3.0)
    assert seed.stationary_timestamp == pytest.approx(anchor, abs=1e-9)
    assert seed.state.velocity[0] == pytest.approx(velocity, abs=1e-9)
    assert ("pre-motion" in detector.status) == motion_check


def test_motion_check_carries_go_attitude_through_slow_rotation(monkeypatch):
    rate = np.array([0.008, 0, 0])  # Below the 0.01 rad/s peak threshold.
    detector, buffer, extrinsic = initializer(event_config(motion_check=True))
    samples = (segment(0.5, 2.2) + segment(2.2, 2.6, gyro=rate)
               + segment(2.6, 3.005, gyro=(0, 0.5, 0)))
    for sample in samples:
        buffer.add(sample)
    detector.observe(buffer.samples_between(0., 3.0), rest_events=REST)
    starts = []
    original = initialization_module.preintegrate_imu

    def capture_start(samples, previous, *args):
        starts.append(previous.copy())
        return original(samples, previous, *args)

    monkeypatch.setattr(initialization_module, "preintegrate_imu", capture_start)
    seed = predict(detector, buffer, extrinsic, 3.0)
    assert seed.stationary_timestamp == pytest.approx(2.5)
    go, zero_velocity = starts
    assert go.timestamp == 2.2 and zero_velocity.timestamp == pytest.approx(2.5)
    # The pre-peak attitude includes 0.3 s of slow rotation; velocity is reset.
    np.testing.assert_allclose(zero_velocity.pose[:3, :3], go.pose[:3, :3] @ so3_exp(rate * .3), atol=1e-9)
    np.testing.assert_array_equal(zero_velocity.velocity, np.zeros(3))


def test_motion_check_without_peak_keeps_go_and_timeout_uses_moved_anchor():
    detector, buffer, extrinsic = initializer(event_config(motion_check=True, max_propagation_s=1.0))
    samples = segment(0.5, 4.0) + segment(4.0, 4.505, acceleration=(0.3, 0, 9.81))
    for sample in samples:
        buffer.add(sample)
    detector.observe(buffer.samples_between(0., 4.5), rest_events=REST)
    assert predict(detector, buffer, extrinsic, 3.0).stationary_timestamp == 2.2
    # GO is 2.3 s old, but the zero-velocity anchor 3.9 is within 1.0 s.
    seed = predict(detector, buffer, extrinsic, 4.5)
    assert seed.stationary_timestamp == pytest.approx(3.9)


def test_tracker_forwards_rest_events_and_restart_clears_sequence(monkeypatch):
    tracker, _, _ = make_tracker(monkeypatch)
    tracker.imu_config, tracker._static_imu_settings, _ = resolve_static_imu_config(event_config(), np.eye(4))
    tracker._static_imu_initializer = StaticImuInitializer(tracker._static_imu_settings, tracker.imu_config)
    for sample in segment(0.5, 2.5):
        tracker.add_imu_sample(sample)
    tracker.add_rest_event("trial_ready", 9.0)
    tracker.add_rest_event("recorded_rest_started", 1.0)
    assert tracker._recorded_rest_events == {"recorded_rest_started": 1.0}
    tracker.add_rest_event("recorded_rest_completed", 2.0)
    tracker._observe_static_imu(header(2.2))
    assert tracker._static_imu_initializer.stationary is None
    tracker.add_rest_event("trial_ready", 2.2)
    tracker._observe_static_imu(header(2.4))
    assert tracker._static_imu_initializer.stationary[0] == 2.2
    assert tracker._prepare_static_trial(header(2.4))
    assert tracker._static_trial_seed.stationary_timestamp == 2.2
    with pytest.raises(ValueError):
        tracker.add_rest_event("trial_ready", np.nan)


def test_real_objects_preset_uses_rest_events_source_time_and_previous_state_start():
    from gelslam.core.contact_velocity import ContactVelocityHistory
    cfg = yaml.safe_load((Path(__file__).parents[1] / "config" / "real_objects_config.yaml").read_text())
    imu = cfg["imu_config"]
    effective, settings, _ = resolve_static_imu_config(imu, np.array(imu["gelsight_T_imu"]))
    assert settings["mode"] == "recorded_rest_event" and settings["max_propagation_s"] == 15.
    assert imu["frontend_initialize_from_imu_preintegration"] is False
    assert imu["image_to_imu_time_offset_s"] == pytest.approx(-0.0615)
    assert settings["contact_velocity_sigma_mps"] == pytest.approx(.3)
    assert imu["timestamp_source"] == "recorded_source"
    assert imu["gtsam_integration_sigma_m"] == pytest.approx(1e-3)
    assert ContactVelocityHistory.from_config(effective) is None


def test_contact_velocity_sigma_adds_variance_and_keeps_mean():
    samples = segment(0.5, 2.6) + segment(2.6, 3.005, acceleration=(0.2, 0, 9.81))
    seeds = {}
    for extra in (None, 0.2):
        detector, buffer, extrinsic = initializer(event_config(contact_velocity_sigma_mps=extra))
        for sample in samples:
            buffer.add(sample)
        detector.observe(buffer.samples_between(0., 3.0), rest_events=REST)
        seeds[extra] = predict(detector, buffer, extrinsic, 3.0)
    base, inflated = (np.linalg.inv(seeds[k].prior.A.T @ seeds[k].prior.A) for k in (None, 0.2))
    np.testing.assert_allclose(seeds[0.2].state.velocity, seeds[None].state.velocity)
    np.testing.assert_allclose(inflated[6:9, 6:9] - base[6:9, 6:9], 0.04 * np.eye(3), atol=1e-9)
    np.testing.assert_allclose(np.delete(np.delete(inflated, slice(6, 9), 0), slice(6, 9), 1),
                               np.delete(np.delete(base, slice(6, 9), 0), slice(6, 9), 1), atol=1e-9)
    with pytest.raises(ValueError, match="contact_velocity_sigma_mps"):
        resolve_static_imu_config(event_config(contact_velocity_sigma_mps=0), np.eye(4))
