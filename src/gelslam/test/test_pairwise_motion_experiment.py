"""Independent checks for the GT-initialized consecutive-image experiment."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gelslam.core.inertial import ImuSample

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/compare_normalflow_imu_gt_intervals.py"
spec = importlib.util.spec_from_file_location("pairwise_motion_experiment", SCRIPT)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_central_difference_velocity_on_nonuniform_gt_times():
    times = np.array([0., .07, .19, .31, .55])
    poses = np.tile(np.eye(4), (len(times), 1, 1))
    poses[:, 0, 3] = .3 * times**2 + .1 * times + .4
    series = experiment.GroundTruthSeries(times, poses)
    for time in (.05, .12, .28, .4):
        pose, velocity = series.at(time)
        assert velocity[0] == pytest.approx(.6 * time + .1, abs=1e-12)
        assert np.allclose(pose[:3, :3], np.eye(3))
    with pytest.raises(ValueError):
        series.at(-.01)


def test_imu_origin_integration_and_gel_conversion_with_rotation_and_velocity():
    reference_imu = np.eye(4)
    reference_imu[:3, :3] = Rotation.from_euler("z", .4).as_matrix()
    reference_imu[:3, 3] = [.2, -.1, .3]
    velocity = np.array([.1, -.02, .03])
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = Rotation.from_euler("xyz", [.3, -.2, .1]).as_matrix()
    extrinsic[:3, 3] = [.02, -.01, .03]
    accel_bias = np.array([.01, -.02, .03])
    gyro_bias = np.array([.001, -.002, .003])
    config = dict(initial_accel_bias_mps2=accel_bias, initial_gyro_bias_rps=gyro_bias)
    samples = [ImuSample(t, np.array([0., 0., 9.81]) + accel_bias,
                         np.array([0., 0., .4]) + gyro_bias)
               for t in np.linspace(0., .1, 21)]
    relative, predicted_velocity, _ = experiment.integrate_gelsight_motion(
        reference_imu, velocity, 0., .1, samples, extrinsic, np.array([0., 0., -9.81]), config,
    )
    target_imu = reference_imu.copy()
    target_imu[:3, :3] = reference_imu[:3, :3] @ Rotation.from_euler("z", .04).as_matrix()
    target_imu[:3, 3] += velocity * .1
    reference_gel = reference_imu @ np.linalg.inv(extrinsic)
    target_gel = target_imu @ np.linalg.inv(extrinsic)
    expected = np.linalg.inv(reference_gel) @ target_gel
    assert np.allclose(relative, expected, atol=1e-10)
    assert np.allclose(predicted_velocity, velocity, atol=1e-10)
    assert np.allclose(reference_gel @ relative, target_gel, atol=1e-10)


def test_error_norms_are_equal_after_using_the_same_gt_reference_pose():
    gt_reference = np.eye(4)
    gt_reference[:3, :3] = Rotation.from_euler("xyz", [.4, -.3, .2]).as_matrix()
    gt_reference[:3, 3] = [1., 2., 3.]
    gt_relative = np.eye(4)
    gt_relative[:3, 3] = [.001, .002, -.001]
    estimate = gt_relative.copy()
    estimate[:3, :3] = Rotation.from_euler("z", .01).as_matrix()
    estimate[:3, 3] += [.0001, -.0002, .0003]
    local = experiment.compare_transforms(estimate, gt_relative)
    world = experiment.compare_transforms(gt_reference @ estimate, gt_reference @ gt_relative)
    assert local["translation_error_mm"] == pytest.approx(world["translation_error_mm"])
    assert local["rotation_error_deg"] == pytest.approx(world["rotation_error_deg"])
    opposite = gt_relative.copy()
    opposite[:3, 3] *= -1
    assert experiment.compare_transforms(opposite, gt_relative)["translation_direction_deg"] == pytest.approx(180.)
