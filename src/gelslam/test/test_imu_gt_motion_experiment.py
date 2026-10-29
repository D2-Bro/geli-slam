"""Independent checks for the IMU-only GT motion experiment."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gelslam.core.inertial import ImuSample

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/evaluate_imu_gt_motion_intervals.py"
spec = importlib.util.spec_from_file_location("imu_gt_motion_experiment", SCRIPT)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)

GRAVITY = np.array([0., 0., -9.81])


def samples(start, end, acceleration, angular_velocity, count=21):
    return [ImuSample(t, np.asarray(acceleration, dtype=float), np.asarray(angular_velocity, dtype=float))
            for t in np.linspace(start, end, count)]


def test_second_difference_removes_constant_velocity_with_unequal_steps():
    dt1, dt2 = .05, .07
    p0 = np.array([.1, -.2, .3])
    velocity = np.array([.4, -.1, .2])
    acceleration = np.array([1., -2., .5])

    def position(t):
        return p0 + velocity * t + .5 * acceleration * t**2

    s = experiment.second_difference(position(0), position(dt1), position(dt1 + dt2), dt1, dt2)
    assert np.allclose(s, .5 * acceleration * dt2 * (dt1 + dt2), atol=1e-15)
    linear = experiment.second_difference(p0, p0 + velocity * dt1, p0 + velocity * (dt1 + dt2), dt1, dt2)
    assert np.allclose(linear, 0, atol=1e-15)


def test_imu_equals_constant_velocity_baseline_without_specific_force_change():
    reference = np.eye(4)
    reference[:3, 3] = [.2, -.1, .3]
    velocity = np.array([.1, -.02, .03])
    pose, predicted_velocity = experiment.integrate_imu_pose(
        reference, velocity, 0., .1, samples(0., .1, -GRAVITY, np.zeros(3)), GRAVITY, {},
    )
    baseline = experiment.constant_velocity_pose(reference, velocity, .1)
    assert np.allclose(pose, baseline, atol=1e-12)
    assert np.allclose(predicted_velocity, velocity, atol=1e-12)


def test_imu_second_difference_does_not_depend_on_initial_velocity():
    reference = np.eye(4)
    reference[:3, :3] = Rotation.from_euler("xyz", [.2, -.1, .4]).as_matrix()
    reference[:3, 3] = [.2, -.1, .3]
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = Rotation.from_euler("xyz", [.3, -.2, .1]).as_matrix()
    extrinsic[:3, 3] = [.02, -.01, .03]
    measured = dict(acceleration=[.5, -.3, 9.6], angular_velocity=[.2, -.4, .3])
    dt1, dt2 = .05, .06

    def gel_positions(initial_velocity):
        pose1, velocity1 = experiment.integrate_imu_pose(
            reference, initial_velocity, 0., dt1, samples(0., dt1, **measured), GRAVITY, {})
        pose2, _ = experiment.integrate_imu_pose(
            pose1, velocity1, dt1, dt1 + dt2, samples(dt1, dt1 + dt2, **measured), GRAVITY, {})
        return [(pose @ np.linalg.inv(extrinsic))[:3, 3] for pose in (reference, pose1, pose2)]

    moving = experiment.second_difference(*gel_positions(np.array([.3, -.2, .1])), dt1, dt2)
    still = experiment.second_difference(*gel_positions(np.zeros(3)), dt1, dt2)
    assert np.linalg.norm(moving) > 1e-4
    assert np.allclose(moving, still, atol=1e-12)


def test_gt_support_gap_covers_interpolation_and_start_velocity_samples():
    times = np.array([0., .01, .02, .03, .07, .08, .09, .10, .11])
    series = experiment.GroundTruthSeries(times, np.tile(np.eye(4), (len(times), 1, 1)))
    # Start velocity at .012 interpolates central differences at .01 and .02.
    assert series.support_gap(.012, .018) == pytest.approx(.01)
    # Start velocity at .025 uses the central difference at .03, which spans .02-.07.
    assert series.support_gap(.025, .029) == pytest.approx(.04)
    # The end pose at .035 is interpolated across the gap.
    assert series.support_gap(.012, .035) == pytest.approx(.04)
    assert series.support_gap(.095, .105) == pytest.approx(.01)


def test_rotation_error_is_the_same_at_both_rigidly_attached_poses():
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = Rotation.from_euler("xyz", [.3, -.2, .1]).as_matrix()
    extrinsic[:3, 3] = [.02, -.01, .03]
    target = np.eye(4)
    target[:3, :3] = Rotation.from_euler("xyz", [.1, .2, -.3]).as_matrix()
    estimate = target.copy()
    estimate[:3, :3] = target[:3, :3] @ Rotation.from_euler("y", .02).as_matrix()
    imu = experiment.pose_error(estimate, target)
    gel = experiment.pose_error(estimate @ extrinsic, target @ extrinsic)
    assert gel[1] == pytest.approx(imu[1])
    assert imu[0] == pytest.approx(0.)
    assert gel[0] > 0


def test_explained_fraction_and_pose_error():
    assert experiment.explained_fraction(1., 4.) == pytest.approx(.75)
    assert experiment.explained_fraction(8., 4.) == pytest.approx(-1.)
    reference = np.eye(4)
    estimate = np.eye(4)
    estimate[:3, :3] = Rotation.from_euler("z", 2., degrees=True).as_matrix()
    estimate[:3, 3] = [.003, .004, 0.]
    translation, rotation = experiment.pose_error(estimate, reference)
    assert translation == pytest.approx(5.)
    assert rotation == pytest.approx(2.)
