"""Global 15D navigation optimization with registered GelSight loop poses.

Each keyframe has an IMU Pose3, world velocity, and accelerometer/gyro bias.
Contact odometry and loop measurements observe GelSight poses in metres.
Raw IMU and bias diffusion couple the remaining navigation variables. The
global solution stays separate from local VI states and marginal priors.
"""
import gtsam
import numpy as np

from gelslam.core.backend_factors import bias_key, velocity_key, variable_type
from gelslam.core.gravity_pose import align_directions, up_direction, yaw_angle
from gelslam.core.gtsam_vi import preintegrate_imu
from gelslam.core.split_pose import (
    optimize_split_pose_graph, split_between_factor, split_prior_factor,
)


class LoopPoseGraph:
    degrees_of_freedom = 15

    def __init__(self, config, imu_config=None):
        self.config = dict(config)
        self.imu_config = dict(imu_config or {})
        # Optional backend-only IMU noise (default: the frontend imu_config values).
        # A stiff frontend integration sigma (0.001) made each first global solve
        # move whole trajectories to absorb stale keyframe velocities.
        for key, name in (('imu_integration_sigma_m', 'gtsam_integration_sigma_m'),
                          ('imu_accel_bias_random_walk', 'accelerometer_bias_random_walk')):
            if self.config.get(key) is not None:
                self.imu_config[name] = self._positive(key, 1.)
        mounting = self.imu_config.get('gelsight_T_imu')
        self.extrinsic = np.asarray(np.eye(4) if mounting is None else mounting, dtype=float)
        self.imu_T_gel = np.linalg.inv(self.extrinsic)
        self.noise = gtsam.noiseModel.Diagonal.Sigmas(np.repeat([
            self._positive('loop_pgo_rotation_sigma_rad', .01),
            self._positive('loop_pgo_translation_sigma_m', .0001),
        ], 3))
        # Loop registrations may use their own sigmas (default: the odometry sigmas)
        # and an optional robust kernel. Registered loop rotations were measured
        # 1-5 deg (up to 15 deg) off GT on real-objects, far above 0.01 rad.
        loop_sigmas = np.repeat([
            self._positive('loop_pgo_loop_rotation_sigma_rad', self._positive('loop_pgo_rotation_sigma_rad', .01)),
            self._positive('loop_pgo_loop_translation_sigma_m', self._positive('loop_pgo_translation_sigma_m', .0001)),
        ], 3)
        self.loop_noise = gtsam.noiseModel.Diagonal.Sigmas(loop_sigmas)
        robust = self.config.get('loop_pgo_loop_robust')
        if robust is not None:
            width = self._positive('loop_pgo_loop_robust_width', 1.345)
            kernels = {'huber': gtsam.noiseModel.mEstimator.Huber, 'cauchy': gtsam.noiseModel.mEstimator.Cauchy}
            if robust not in kernels:
                raise ValueError("pose_graph_config.loop_pgo_loop_robust must be null, 'huber' or 'cauchy'")
            self.loop_noise = gtsam.noiseModel.Robust.Create(kernels[robust].Create(width), self.loop_noise)
        # IMU-only links (target keyframe not tactile, i.e. across a no-contact gap):
        # optionally inflate accelerometer and integration noise and the accel bias
        # walk; gyro noise is unchanged. Global PGO runs only after a loop is accepted.
        self.imu_only_accel_scale = self._positive('imu_only_edge_accel_noise_scale', 1.)
        self.imu_only_walk_scale = self._positive('imu_only_edge_accel_bias_walk_scale', 1.)
        error = self.config.get('imu_only_edge_accel_error_mps2')
        self.imu_only_accel_error = None if error is None else self._positive('imu_only_edge_accel_error_mps2', 1.)
        self.anchor_noise = gtsam.noiseModel.Isotropic.Sigma(
            6, self._positive('anchor_sigma', 1e-6))
        self.graph = gtsam.NonlinearFactorGraph()
        self.graph_init = gtsam.Values()
        self.graph_result = None
        self.factor_kinds, self.loops = [], []
        self.corrections, self.velocity_corrections, self.bias_corrections = [], [], []
        self.roots, self.anchor_poses, self.anchor_states = {}, {}, {}
        self.removed_groups = set()
        self.local_poses, self.local_states, self.records = [], [], []
        self.up = np.array([0., 0., 1.])
        self.local_up, self.temporal_links = [], []
        self.last_initial_cost = self.last_final_cost = None

    def _positive(self, name, default):
        value = float(self.config.get(name, default))
        if not np.isfinite(value) or value <= 0:
            raise ValueError('pose_graph_config.' + name + ' must be positive and finite')
        return value

    @staticmethod
    def transform(value):
        value = np.asarray(value, dtype=float)
        if (value.shape != (4, 4) or not np.isfinite(value).all()
                or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
                or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-5, rtol=0)
                or not np.isclose(np.linalg.det(value[:3, :3]), 1., atol=1e-5)):
            raise ValueError('loop measurement must be a finite rigid 4x4 transform')
        return value.copy()

    def synchronize(self, local_poses, records, temporal_links, navigation_states=None):
        """Refresh local seeds while retaining global pose/velocity/bias offsets."""
        states = ([getattr(record, 'navigation_state', None) for record in records]
                  if navigation_states is None else navigation_states)
        if len(states) != len(local_poses) or len(records) != len(local_poses):
            raise ValueError('15D global PGO requires one navigation state per keyframe')
        shapes = (('pose', (4, 4)), ('velocity', (3,)),
                  ('accel_bias', (3,)), ('gyro_bias', (3,)))
        for state in states:
            if (state is None or not np.isfinite(state.timestamp)
                    or any(np.asarray(getattr(state, name)).shape != shape
                           or not np.isfinite(getattr(state, name)).all()
                           for name, shape in shapes)):
                raise ValueError('15D global PGO requires finite navigation states')
        self.local_poses = [np.asarray(pose).copy() for pose in local_poses]
        self.local_states = [state.copy() for state in states]
        self.records = list(records)
        self.local_up = [up_direction(record.gravity) for record in records]
        if self.local_up:
            self.up = self.local_up[0].copy()
        self.temporal_links = list(temporal_links)
        for key in range(len(self.corrections), len(local_poses)):
            record = records[key]
            correction, dv, db = np.eye(4), np.zeros(3), np.zeros(6)
            if record.new_trial_flag:
                correction[:3, :3] = align_directions(self.local_up[key], self.up)
            else:
                correction = self.corrections[key - 1].copy()
                dv = self.velocity_corrections[key - 1].copy()
                db = self.bias_corrections[key - 1].copy()
            self.corrections.append(correction)
            self.velocity_corrections.append(dv)
            self.bias_corrections.append(db)
            if record.new_trial_flag:
                self.roots[record.trial_group] = key
                self.anchor_poses[record.trial_group] = correction @ local_poses[key]
        # Older snapshots retain pose corrections but have no navigation offsets.
        for group, key in self.roots.items():
            if group not in self.anchor_states:
                self.anchor_states[group] = self.map_state(key)

    def map_poses(self):
        return [correction @ pose for correction, pose in
                zip(self.corrections, self.local_poses)]

    def map_state(self, key):
        state = self.local_states[key].copy()
        correction = self.corrections[key]
        state.pose = correction @ state.pose
        state.velocity = correction[:3, :3] @ state.velocity + self.velocity_corrections[key]
        state.accel_bias += self.bias_corrections[key][:3]
        state.gyro_bias += self.bias_corrections[key][3:]
        return state

    def trial_frame_correction(self, group):
        """Remove a gravity-preserving frame gauge without removing tilt updates."""
        key = self.roots[group]
        alignment = align_directions(self.local_up[key], self.up)
        correction = self.corrections[key].copy()
        if np.allclose(correction[:3, :3] @ self.local_up[key], self.up, rtol=0, atol=1e-12):
            return correction
        yaw = yaw_angle(correction[:3, :3] @ alignment.T, self.up)
        correction[:3, :3] = gtsam.Rot3.Expmap(self.up * yaw).matrix() @ alignment
        return correction

    def component(self, key):
        connected = {key}
        links = self.temporal_links + [(r, t) for r, t, _ in self.loops]
        while True:
            previous = len(connected)
            for reference, target in links:
                if reference in connected or target in connected:
                    connected.update((reference, target))
            if len(connected) == previous:
                return connected

    def add_loop(self, reference, target, target_T_reference):
        if any(r == reference and t == target for r, t, _ in self.loops):
            return False
        measurement = np.linalg.inv(self.transform(target_T_reference))
        component = self.component(target)
        if reference not in component:
            poses = self.map_poses()
            shift = poses[reference] @ measurement @ np.linalg.inv(poses[target])
            for key in component:
                self.corrections[key] = shift @ self.corrections[key]
                self.velocity_corrections[key] = shift[:3, :3] @ self.velocity_corrections[key]
            for group, key in self.roots.items():
                if key in component:
                    self.anchor_poses[group] = shift @ self.anchor_poses[group]
                    state = self.anchor_states[group].copy()
                    state.pose = shift @ state.pose
                    state.velocity = shift[:3, :3] @ state.velocity
                    self.anchor_states[group] = state
        self.loops.append((reference, target, measurement))
        return True

    def remove_prior_factors(self, groups):
        # Remove merged pose gauges. Initial velocity/bias observations remain.
        self.removed_groups.update(groups)

    def _imu_factors(self, reference, target, previous):
        record = self.records[target]
        samples = record.imu_samples
        start, end = self.local_states[reference].timestamp, self.local_states[target].timestamp
        max_gap = self._positive('imu_max_sample_gap_s', .05)
        if (len(samples) < 2 or end <= start
                or abs(samples[0].timestamp - start) > 1e-8
                or abs(samples[-1].timestamp - end) > 1e-8
                or any(b.timestamp <= a.timestamp or b.timestamp - a.timestamp > max_gap
                       for a, b in zip(samples, samples[1:]))):
            raise ValueError('global PGO IMU interval must cover both keyframe times with no gaps')
        gravity = -np.linalg.norm(record.gravity) * self.up
        imu_config, walk_scale = self.imu_config, 1.
        if not record.tactile_valid:
            imu_config, walk_scale = self._imu_only_noise(end - start)
        pim = preintegrate_imu(samples, previous, imu_config, self.extrinsic, gravity)
        if pim is None:
            raise ValueError('global PGO IMU preintegration unavailable')
        imu = gtsam.ImuFactor(reference, velocity_key(reference), target,
                              velocity_key(target), bias_key(reference), pim)
        densities = np.repeat([
            float(self.imu_config.get('accelerometer_bias_random_walk', .0002)) * walk_scale,
            float(self.imu_config.get('gyroscope_bias_random_walk', .00002)),
        ], 3)
        if not np.isfinite(densities).all() or np.any(densities <= 0):
            raise ValueError('global PGO bias random walks must be positive and finite')
        bias = gtsam.BetweenFactorConstantBias(
            bias_key(reference), bias_key(target), gtsam.imuBias.ConstantBias(),
            gtsam.noiseModel.Diagonal.Sigmas(densities * np.sqrt(pim.deltaTij())))
        return imu, bias

    def _imu_only_noise(self, duration):
        """Return the preintegration config and accel bias walk scale of one IMU-only link.

        Constant mode multiplies accel/integration sigmas by a fixed scale. The
        error model instead matches a constant acceleration error a over the
        link: white accel density sqrt(3)/2 * a * sqrt(dt) gives position sigma
        a * dt^2 / 2, and the bias may change by a over the link.
        """
        accel = float(self.imu_config.get('accelerometer_noise_density', .02))
        integration = float(self.imu_config.get('gtsam_integration_sigma_m', 1e-4))
        walk = float(self.imu_config.get('accelerometer_bias_random_walk', .0002))
        accel_scale, walk_scale = self.imu_only_accel_scale, self.imu_only_walk_scale
        if self.imu_only_accel_error is not None:
            accel_scale = max(accel_scale, .5 * np.sqrt(3.) * self.imu_only_accel_error * np.sqrt(duration) / accel)
            walk_scale = max(walk_scale, self.imu_only_accel_error / np.sqrt(duration) / walk)
        if accel_scale == 1. and walk_scale == 1.:
            return self.imu_config, 1.
        config = dict(self.imu_config, accelerometer_noise_density=accel * accel_scale,
                      gtsam_integration_sigma_m=integration * accel_scale)
        return config, walk_scale

    def solve(self):
        """Jointly optimize full rotations, positions, velocities and both biases."""
        graph, values, kinds, fixed = gtsam.NonlinearFactorGraph(), gtsam.Values(), [], set()
        states = [self.map_state(key) for key in range(len(self.local_states))]

        def add(factor, kind):
            graph.add(factor)
            kinds.append(kind)

        for key, state in enumerate(states):
            values.insert(key, gtsam.Pose3(state.pose))
            values.insert(velocity_key(key), state.velocity)
            values.insert(bias_key(key), gtsam.imuBias.ConstantBias(state.accel_bias, state.gyro_bias))
        for group, key in self.roots.items():
            if group not in self.removed_groups:
                anchor = self.anchor_poses[group]
                values.update(key, gtsam.Pose3(anchor @ self.extrinsic))
                add(split_prior_factor(key, anchor, self.anchor_noise, self.imu_T_gel), 'pose_anchor')
                fixed.add(key)
            anchor = self.anchor_states[group]
            add(gtsam.PriorFactorVector(velocity_key(key), anchor.velocity,
                gtsam.noiseModel.Isotropic.Sigma(3, self._positive('initial_velocity_sigma_mps', .1))), 'velocity_prior')
            sigmas = np.repeat([self._positive('initial_accel_bias_sigma_mps2', .1),
                                self._positive('initial_gyro_bias_sigma_rps', .01)], 3)
            add(gtsam.PriorFactorConstantBias(bias_key(key),
                gtsam.imuBias.ConstantBias(anchor.accel_bias, anchor.gyro_bias),
                gtsam.noiseModel.Diagonal.Sigmas(sigmas)), 'bias_prior')
        for reference, target in self.temporal_links:
            imu, bias = self._imu_factors(reference, target, states[reference])
            add(imu, 'imu')
            add(bias, 'bias')
            if self.records[target].tactile_valid:
                measurement = np.linalg.inv(self.local_poses[reference]) @ self.local_poses[target]
                add(split_between_factor(reference, target, measurement, self.noise,
                                         self.imu_T_gel), 'odometry')
        for reference, target, measurement in self.loops:
            add(split_between_factor(reference, target, measurement, self.loop_noise,
                                     self.imu_T_gel), 'loop')
        initial_cost = float(graph.error(values))
        result = optimize_split_pose_graph(graph, values,
            max_iterations=int(self.config.get('max_iterations', 10)),
            variable_types={key: variable_type(key) for key in values.keys()}, fixed_keys=fixed)
        final_cost = float(graph.error(result))
        if not np.isfinite(final_cost) or final_cost > initial_cost + 1e-10 * max(initial_cost, 1.):
            raise ValueError('loop PGO did not produce a finite non-increasing cost')
        corrections, velocities, biases = [], [], []
        for key, local in enumerate(self.local_states):
            pose = result.atPose3(key).matrix()
            correction = pose @ np.linalg.inv(local.pose)
            corrections.append(correction)
            velocities.append(result.atVector(velocity_key(key)) - correction[:3, :3] @ local.velocity)
            biases.append(result.atConstantBias(bias_key(key)).vector() - np.r_[local.accel_bias, local.gyro_bias])
        if not all(np.isfinite(value).all() for value in corrections + velocities + biases):
            raise ValueError('loop PGO produced a non-finite navigation correction')
        self.graph, self.graph_init, self.graph_result = graph, values, result
        self.factor_kinds, self.corrections = kinds, corrections
        self.velocity_corrections, self.bias_corrections = velocities, biases
        self.last_initial_cost, self.last_final_cost = initial_cost, final_cost
        if self.config.get('debug_dump_dir'):
            self._dump_solve(graph, values, result, kinds)

    def _dump_solve(self, graph, values, result, kinds):
        """Diagnostics only: one JSON per solve with loops, poses and per-factor chi2."""
        import json
        import os
        os.makedirs(self.config['debug_dump_dir'], exist_ok=True)
        factors = []
        for index, kind in enumerate(kinds):
            factor = graph.at(index)
            factors.append(dict(kind=kind, keys=[int(k) for k in factor.keys()],
                                chi2_initial=2 * float(factor.error(values)),
                                chi2_final=2 * float(factor.error(result))))
        gel = lambda pose: (np.asarray(pose) @ self.imu_T_gel).tolist()
        data = dict(
            timestamps=[float(s.timestamp) for s in self.local_states],
            tactile_valid=[bool(r.tactile_valid) for r in self.records],
            temporal_links=[[int(r), int(t)] for r, t in self.temporal_links],
            loops=[dict(reference=int(r), target=int(t), measurement_gel=np.asarray(m).tolist())
                   for r, t, m in self.loops],
            map_gel_before=[gel(values.atPose3(k).matrix()) for k in range(len(self.local_states))],
            map_gel_after=[gel(result.atPose3(k).matrix()) for k in range(len(self.local_states))],
            factors=factors, initial_cost=self.last_initial_cost, final_cost=self.last_final_cost)
        name = 'solve_%04d_kf%04d.json' % (len(os.listdir(self.config['debug_dump_dir'])), len(self.local_states))
        with open(os.path.join(self.config['debug_dump_dir'], name), 'w') as handle:
            json.dump(data, handle)

    def snapshot(self):
        return dict(degrees_of_freedom=self.degrees_of_freedom,
                    loops=self.loops, corrections=self.corrections, roots=self.roots,
                    velocity_corrections=self.velocity_corrections,
                    bias_corrections=self.bias_corrections, anchor_states=self.anchor_states,
                    anchor_poses=self.anchor_poses, removed_groups=self.removed_groups)

    def restore(self, data):
        self.loops = [(r, t, self.transform(measurement)) for r, t, measurement in data['loops']]
        self.corrections = [self.transform(value) for value in data['corrections']]
        self.velocity_corrections = [np.array(value, copy=True) for value in data.get(
            'velocity_corrections', [np.zeros(3) for _ in self.corrections])]
        self.bias_corrections = [np.array(value, copy=True) for value in data.get(
            'bias_corrections', [np.zeros(6) for _ in self.corrections])]
        if (len(self.velocity_corrections) != len(self.corrections)
                or len(self.bias_corrections) != len(self.corrections)
                or any(v.shape != (3,) or not np.isfinite(v).all() for v in self.velocity_corrections)
                or any(v.shape != (6,) or not np.isfinite(v).all() for v in self.bias_corrections)):
            raise ValueError('invalid saved global navigation corrections')
        self.roots, self.anchor_poses = dict(data['roots']), dict(data['anchor_poses'])
        self.anchor_states = {group: state.copy() for group, state in data.get('anchor_states', {}).items()}
        self.removed_groups = set(data['removed_groups'])
        for group, key in self.roots.items():
            if group not in self.anchor_states and key < len(self.local_states):
                self.anchor_states[group] = self.map_state(key)
