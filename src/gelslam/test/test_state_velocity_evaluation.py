"""Raw State.velocity, frame conversion and native GT regression checks."""

from types import SimpleNamespace
import csv

import numpy as np
from scipy.spatial.transform import Rotation

from gelslam.core.state_velocity_evaluation import (
    NativeImuVelocityTruth, StateVelocityEvaluation, capture_state_velocity,
)


def native_truth(tmp_path):
    path = tmp_path / 'truth.npz'
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    sensors = poses.copy()
    sensors[:, 0, 3] = -1
    np.savez(path, t_s=np.arange(3.), T_WI=poses, T_WS=sensors,
             ideal_gyro_rad_s=np.tile((0., 0., 2.), (3, 1)),
             velocity_WS_m_s=np.tile((1., 2., 3.), (3, 1)))
    return path, poses


def test_native_gt_uses_same_angular_lever_arm_term_as_previous_experiments(tmp_path):
    path, _ = native_truth(tmp_path)
    truth = NativeImuVelocityTruth(str(path), 1.)
    assert np.allclose(truth.at(1.)[1], (1., 4., 3.))
    assert np.allclose(truth.at(1.5)[1], (1., 4., 3.))
    assert truth.at(0.) is None


def test_raw_velocity_drives_error_even_with_stationary_estimated_positions(tmp_path):
    path, poses = native_truth(tmp_path)
    evaluation = StateVelocityEvaluation(dict(truth_path=str(path)), str(tmp_path / 'out'))
    state_to_gt = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    state_pose = np.eye(4)
    state_pose[:3, :3] = state_to_gt.T
    raw_velocity = state_to_gt.T @ np.array((1.1, 4.2, 3.))
    for index in range(3):
        timestamp = float(index + 1)
        evaluation.add_gt_pose(timestamp, poses[index])
        evaluation.add_state(dict(state_timestamp_s=timestamp, image_timestamp_s=timestamp - 0.05,
                                  fid=index, ref_kid=0, trial_id=0, stage='accepted_contact_state',
                                  pose=state_pose.tolist(), velocity=raw_velocity.tolist()))
    result = evaluation.save()
    assert result['status'] == 'ok'
    assert np.isclose(result['velocity_rmse_mps'], np.sqrt(0.05))
    assert result['native_gt_pose_mismatches'] == 0
    assert (tmp_path / 'out/trajectory_velocity_error.png').is_file()
    assert (tmp_path / 'out/trajectory_velocity_error_components.png').is_file()
    with (tmp_path / 'out/trajectory_velocity_errors.csv').open() as handle:
        row = next(csv.DictReader(handle))
    assert np.isclose(float(row['state_vx_mps']), raw_velocity[0])
    assert float(row['state_timestamp_s']) == 1.
    assert float(row['image_timestamp_s']) == 0.95


def test_mismatched_bag_gt_is_rejected(tmp_path):
    path, poses = native_truth(tmp_path)
    evaluation = StateVelocityEvaluation(dict(truth_path=str(path)), str(tmp_path / 'out'))
    poses[0, 0, 3] = 5
    evaluation.add_gt_pose(1., poses[0])
    assert evaluation.save()['status'] == 'native_gt_bag_mismatch'
    assert not (tmp_path / 'out/trajectory_velocity_error.png').exists()


def test_capture_uses_state_timestamp_and_never_stale_or_mutated_velocity():
    state = SimpleNamespace(timestamp=1., pose=np.eye(4), velocity=np.array((1., 2., 3.)))
    tracker = SimpleNamespace(frontend_visual_inertial=object(), tracking_mode=SimpleNamespace(value='contact_tracking'),
                              _frontend_vi_state=state, _trial_index=2, _imu_timestamp=lambda header: 1.)
    message = SimpleNamespace(fid=5, ref_kid=3, header=SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=950000000)))
    record = capture_state_velocity(tracker, message)
    state.velocity[:] = 9
    assert record['velocity'] == [1., 2., 3.]
    assert record['state_timestamp_s'] == 1.
    assert np.isclose(record['image_timestamp_s'], 0.95)
    tracker._imu_timestamp = lambda header: 2.
    assert capture_state_velocity(tracker, message) is None
    tracker._imu_timestamp = lambda header: 1.
    tracker.tracking_mode.value = 'tactile_hold'
    assert capture_state_velocity(tracker, message) is None


def test_capture_diagnostics_use_published_coast_bias_and_world_axes():
    from gelslam.core.inertial import ImuBuffer, ImuSample
    rotation = Rotation.from_euler('x', 90, degrees=True).as_matrix()
    pose = np.eye(4)
    pose[:3, :3] = rotation
    state = SimpleNamespace(timestamp=2., pose=pose, velocity=np.array((1., 2., 3.)),
                            accel_bias=np.array((.1, .2, .3)), gyro_bias=np.array((.01, .02, .03)))
    buffer = ImuBuffer()
    buffer.add(ImuSample(1.9, (1., 2., 3.), (.2, .3, .4)))
    buffer.add(ImuSample(2.1, (3., 4., 5.), (.4, .5, .6)))
    gravity = np.array((0., 0., -9.81))
    tracker = SimpleNamespace(frontend_visual_inertial=SimpleNamespace(gravity=gravity),
        tracking_mode=SimpleNamespace(value='no_contact_coasting'),
        _frontend_coast_state=state, _frontend_vi_state=SimpleNamespace(timestamp=1.),
        _trial_index=2, _imu_timestamp=lambda _: 2., imu_buffer=buffer)
    message = SimpleNamespace(fid=5, ref_kid=3, tracking_status='no_contact_coasting',
        header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=950000000)))
    record = capture_state_velocity(tracker, message)
    assert record['stage'] == 'no_contact_prediction'
    assert record['imu_sample_timestamp_s'] == 2.
    np.testing.assert_allclose(record['linear_acceleration_world_mps2'],
                               rotation @ (np.array((2., 3., 4.)) - state.accel_bias) + gravity)
    assert record['acceleration_sampling'] == 'measurement_at_state_timestamp'
    state.accel_bias[:] = 9.
    state.gyro_bias[:] = 8.
    gravity[:] = 7.
    assert record['accel_bias'] == [.1, .2, .3]
    assert record['gyro_bias'] == [.01, .02, .03]
    assert record['gravity_world_mps2'] == [0., 0., -9.81]
