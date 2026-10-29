"""Evaluate recorded frontend State.velocity against native IMU-origin GT."""

import csv
import json
import os
import threading

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from gelslam.core.trajectory_error_plot import write_error_colormap, write_velocity_component_colormaps


class NativeImuVelocityTruth:
    """Same native kinematics as the first10/last_run State error experiments."""

    def __init__(self, path, time_offset_s=1.0):
        self.path = os.path.abspath(os.path.expanduser(path))
        with np.load(self.path, allow_pickle=False) as data:
            self.times = np.asarray(data['t_s'], dtype=float) + float(time_offset_s)
            self.poses = np.asarray(data['T_WI'], dtype=float)
            sensor_poses = np.asarray(data['T_WS'], dtype=float)
            omega_world = np.einsum('nij,nj->ni', self.poses[:, :3, :3], data['ideal_gyro_rad_s'])
            self.velocities = np.asarray(data['velocity_WS_m_s'], dtype=float) + np.cross(
                omega_world, self.poses[:, :3, 3] - sensor_poses[:, :3, 3])
        if (len(self.times) < 2 or np.any(np.diff(self.times) <= 0)
                or self.poses.shape != (len(self.times), 4, 4)
                or self.velocities.shape != (len(self.times), 3)
                or not all(np.isfinite(a).all() for a in (self.times, self.poses, self.velocities))):
            raise ValueError('Invalid native IMU ground truth')
        self.rotations = Slerp(self.times, Rotation.from_matrix(self.poses[:, :3, :3]))

    def at(self, timestamp):
        index = int(np.searchsorted(self.times, timestamp))
        for candidate in (index, index - 1):
            if 0 <= candidate < len(self.times) and abs(self.times[candidate] - timestamp) <= 1e-8:
                return self.poses[candidate].copy(), self.velocities[candidate].copy()
        if index == 0 or index == len(self.times):
            return None
        fraction = (timestamp - self.times[index - 1]) / (self.times[index] - self.times[index - 1])
        pose = np.eye(4)
        pose[:3, :3] = self.rotations([timestamp]).as_matrix()[0]
        pose[:3, 3] = ((1 - fraction) * self.poses[index - 1, :3, 3]
                      + fraction * self.poses[index, :3, 3])
        velocity = (1 - fraction) * self.velocities[index - 1] + fraction * self.velocities[index]
        return pose, velocity


def capture_state_velocity(tracker, frame_msg):
    """Snapshot the exact published contact/coast State, never a stale anchor."""
    if not getattr(frame_msg, 'pose_valid', True):
        return None
    if getattr(tracker, 'frontend_visual_inertial', None) is None:
        return None
    mode = getattr(frame_msg, 'tracking_status', tracker.tracking_mode.value)
    if mode in {'tactile_hold', 'reacquire'}:
        return None
    coasting = mode in {'no_contact_coasting', 'imu_coasting'}
    state_attribute = ('_frontend_frame_prediction' if mode == 'imu_prediction' else
                       '_frontend_coast_state' if coasting else '_frontend_vi_state')
    state = getattr(tracker, state_attribute, None)
    target = tracker._imu_timestamp(frame_msg.header)
    if state is None or target is None or abs(state.timestamp - target) > 1e-7:
        return None
    pose, velocity = np.asarray(state.pose), np.asarray(state.velocity)
    if pose.shape != (4, 4) or velocity.shape != (3,) or not np.isfinite(pose).all() or not np.isfinite(velocity).all():
        return None
    stamp = frame_msg.header.stamp
    anchor_time = getattr(tracker, '_frontend_trial_start_timestamp', None)
    initialization = anchor_time is not None and abs(state.timestamp - anchor_time) <= 1e-7
    stage = ('imu_frame_prediction' if mode == 'imu_prediction' else
             'imu_recovery_prediction' if mode == 'imu_only' else
             'no_contact_prediction' if coasting else
             'initialization' if initialization else 'accepted_contact_state')
    record = dict(state_timestamp_s=float(state.timestamp),
                image_timestamp_s=float(stamp.sec) + float(stamp.nanosec) * 1e-9,
                fid=int(frame_msg.fid), ref_kid=int(frame_msg.ref_kid),
                trial_id=int(tracker._trial_index),
                stage=stage,
                tracking_mode=mode, pose=pose.tolist(), velocity=velocity.tolist())
    # Diagnostics use the State that produced this frame, including coast and
    # uncommitted image predictions. Keep sensor axes separate from world axes.
    for name in ('accel_bias', 'gyro_bias'):
        value = getattr(state, name, None)
        if value is not None and np.asarray(value).shape == (3,) and np.isfinite(value).all():
            record[name] = np.asarray(value).tolist()
    gravity = getattr(tracker.frontend_visual_inertial, 'gravity', None)
    if gravity is not None and np.asarray(gravity).shape == (3,) and np.isfinite(gravity).all():
        record['gravity_world_mps2'] = np.asarray(gravity).tolist()
    buffer = getattr(tracker, 'imu_buffer', None)
    sample = None if buffer is None else buffer.sample_at(state.timestamp)
    if sample is not None:
        record.update(imu_sample_timestamp_s=float(sample.timestamp),
                      specific_force_imu_mps2=sample.acceleration.tolist(),
                      angular_velocity_imu_rps=sample.angular_velocity.tolist())
        if 'accel_bias' in record and 'gravity_world_mps2' in record:
            record['linear_acceleration_world_mps2'] = (
                pose[:3, :3] @ (sample.acceleration - state.accel_bias) + gravity).tolist()
            # This is an endpoint diagnostic. Native preintegration averages
            # adjacent measurements and uses its configured interval attitude.
            record['acceleration_sampling'] = 'measurement_at_state_timestamp'
    return record


class StateVelocityEvaluation:
    def __init__(self, settings, output_dir):
        self.settings = dict(settings)
        self.output_dir = output_dir
        # Without a native truth file (real data), save states and GT poses only.
        self.truth = (None if settings.get('truth_path') is None else
                      NativeImuVelocityTruth(settings['truth_path'], settings.get('time_offset_s', 1.0)))
        self.states, self.gt_poses = [], []
        self.lock = threading.Lock()

    def add_state(self, record):
        if record is not None:
            with self.lock:
                self.states.append(record)

    def add_gt_pose(self, timestamp, pose):
        if np.isfinite(timestamp) and np.asarray(pose).shape == (4, 4) and np.isfinite(pose).all():
            with self.lock:
                self.gt_poses.append((float(timestamp), np.asarray(pose).copy()))

    def save(self):
        """Use one fixed GT attitude/position anchor per trial, no velocity fitting."""
        os.makedirs(self.output_dir, exist_ok=True)
        with self.lock:
            states = sorted(self.states, key=lambda row: row['state_timestamp_s'])
            gt_poses = sorted(self.gt_poses, key=lambda row: row[0])
        with open(os.path.join(self.output_dir, 'state_velocity_states.jsonl'), 'w') as handle:
            for record in states:
                handle.write(json.dumps(record, allow_nan=False) + '\n')
        with open(os.path.join(self.output_dir, 'state_velocity_gt_poses.jsonl'), 'w') as handle:
            for timestamp, pose in gt_poses:
                handle.write(json.dumps(dict(timestamp_s=timestamp, pose=pose.tolist())) + '\n')
        metrics_path = os.path.join(self.output_dir, 'trajectory_velocity_metrics.json')
        plot_path = os.path.join(self.output_dir, 'trajectory_velocity_error.png')
        for name in (plot_path, os.path.join(self.output_dir, 'trajectory_velocity_error_components.png'),
                     os.path.join(self.output_dir, 'trajectory_velocity_errors.csv')):
            if os.path.isfile(name):
                os.remove(name)
        result = dict(velocity_source='frontend State.velocity (IMU origin, raw state values retained)',
                      ground_truth_source=None if self.truth is None else self.truth.path,
                      ground_truth_pose_topic=self.settings.get('ground_truth_pose_topic', '/ground_truth/imu_pose'),
                      ground_truth_method='velocity_WS_m_s + cross(R_WI @ ideal_gyro_rad_s, p_WI - p_WS)',
                      method_reference='first10_contact_state_errors_20261018 and last_run_state_errors_20261016',
                      alignment='one fixed IMU pose anchor per trial; no trajectory/velocity fit or scaling',
                      timestamp_source='State.timestamp; image header is recorded separately',
                      scope='published accepted frontend contact states and no-contact prediction states; all trials',
                      recorded_states=len(states), received_gt_poses=len(gt_poses))

        def finish(status):
            result['status'] = status
            with open(metrics_path, 'w') as handle:
                json.dump(result, handle, indent=2, allow_nan=False)
            return result

        if self.truth is None:
            return finish('states_only_no_truth_file')
        # Reject an unrelated sidecar rather than evaluating against arbitrary motion.
        mismatches = 0
        for timestamp, pose in gt_poses:
            native = self.truth.at(timestamp)
            if native is None or not np.allclose(native[0], pose, atol=1e-8, rtol=0):
                mismatches += 1
        result['native_gt_pose_mismatches'] = mismatches
        if not gt_poses or mismatches:
            return finish('missing_gt_poses' if not gt_poses else 'native_gt_bag_mismatch')
        gt_times = np.asarray([time for time, _ in gt_poses])
        tolerance = float(self.settings.get('timestamp_tolerance_s', 0.03))
        transforms, rows, estimated, matched = {}, [], [], []
        for record in states:
            timestamp = record['state_timestamp_s']
            index = int(np.searchsorted(gt_times, timestamp))
            if timestamp < gt_times[0] or timestamp > gt_times[-1]:
                continue
            closest = min(abs(gt_times[min(index, len(gt_times) - 1)] - timestamp),
                          abs(gt_times[max(0, index - 1)] - timestamp))
            if closest > tolerance:
                continue
            if index and index < len(gt_times) and gt_times[index] != timestamp and gt_times[index] - gt_times[index - 1] > 2 * tolerance:
                continue
            native = self.truth.at(timestamp)
            if native is None:
                continue
            gt_pose, gt_velocity = native
            state_pose = np.asarray(record['pose'])
            raw_velocity = np.asarray(record['velocity'])
            trial = record['trial_id']
            if trial not in transforms:
                transform = np.eye(4)
                transform[:3, :3] = gt_pose[:3, :3] @ state_pose[:3, :3].T
                transform[:3, 3] = gt_pose[:3, 3] - transform[:3, :3] @ state_pose[:3, 3]
                transforms[trial] = transform
            transform = transforms[trial]
            world_velocity = transform[:3, :3] @ raw_velocity
            error = world_velocity - gt_velocity
            row = dict(record, velocity_world_mps=world_velocity.tolist(),
                       gt_velocity_world_mps=gt_velocity.tolist(), error_world_mps=error.tolist(),
                       velocity_error_mps=float(np.linalg.norm(error)))
            rows.append(row)
            estimated.append((timestamp, record['fid'], record['ref_kid'], transform @ state_pose))
            matched.append((timestamp, -1, -1, gt_pose))
        with open(os.path.join(self.output_dir, 'trajectory_velocity_errors.csv'), 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(('state_timestamp_s', 'image_timestamp_s', 'fid', 'ref_kid', 'trial_id', 'stage',
                             'state_vx_mps', 'state_vy_mps', 'state_vz_mps',
                             'estimated_world_vx_mps', 'estimated_world_vy_mps', 'estimated_world_vz_mps',
                             'gt_world_vx_mps', 'gt_world_vy_mps', 'gt_world_vz_mps',
                             'error_world_vx_mps', 'error_world_vy_mps', 'error_world_vz_mps', 'velocity_error_mps'))
            for row in rows:
                writer.writerow((*(row[key] for key in ('state_timestamp_s', 'image_timestamp_s', 'fid', 'ref_kid', 'trial_id', 'stage')),
                                 *row['velocity'], *row['velocity_world_mps'], *row['gt_velocity_world_mps'],
                                 *row['error_world_mps'], row['velocity_error_mps']))
        errors = np.asarray([row['velocity_error_mps'] for row in rows])
        result.update(matched_states=len(rows), unmatched_states=len(states) - len(rows),
                      trial_transforms_state_to_gt={str(key): value.tolist() for key, value in transforms.items()},
                      velocity_rmse_mps=float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
                      velocity_mean_mps=float(np.mean(errors)) if len(errors) else None,
                      velocity_max_mps=float(np.max(errors)) if len(errors) else None)
        if len(rows) >= 2:
            write_velocity_component_colormaps(
                self.output_dir, estimated, matched,
                np.asarray([row['error_world_mps'] for row in rows]),
                groups=[row['trial_id'] for row in rows],
            )
            write_error_colormap(self.output_dir, estimated, matched, errors * 1000,
                                 'trajectory_velocity_error.png',
                                 'GelSLAM: frontend State.velocity (IMU origin)',
                                 'State.velocity error', 'mm/s', groups=[row['trial_id'] for row in rows])
        return finish('ok' if len(rows) >= 2 else 'insufficient_states')


def main():
    """Re-evaluate saved actual states; never reconstruct velocities from positions."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir')
    parser.add_argument('--truth-path', required=True)
    parser.add_argument('--time-offset-s', type=float, default=1.0)
    args = parser.parse_args()
    output = os.path.abspath(os.path.expanduser(args.output_dir))
    # Read both files before save rewrites them.
    with open(os.path.join(output, 'state_velocity_states.jsonl')) as handle:
        states = [json.loads(line) for line in handle if line.strip()]
    with open(os.path.join(output, 'state_velocity_gt_poses.jsonl')) as handle:
        poses = [json.loads(line) for line in handle if line.strip()]
    evaluation = StateVelocityEvaluation(dict(truth_path=args.truth_path, time_offset_s=args.time_offset_s), output)
    for state in states:
        evaluation.add_state(state)
    for pose in poses:
        evaluation.add_gt_pose(pose['timestamp_s'], np.asarray(pose['pose']))
    print(json.dumps(evaluation.save(), indent=2))


if __name__ == '__main__':
    main()
