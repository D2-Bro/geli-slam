"""15D keyframe navigation graph with direct tactile, IMU and bias factors."""

import os
import pickle
from types import SimpleNamespace

import cv2
import gtsam
import numpy as np
from normalflow.registration import LoseTrackError, normalflow

from gelslam.utils import M2T
from gelslam.core.imu_navigation import gelsight_pose, imu_pose
from gelslam.core.numpy_vi import NavigationState
from gelslam.core.gtsam_vi import preintegrate_imu
from gelslam.core.marginal_prior import information_rows, schur_complement
from gelslam.core.split_pose import optimize_split_pose_graph, split_prior_factor
from gelslam.core.backend_factors import (
    TactilePair, bias_key, velocity_key, state_index, state_keys,
    variable_type, variable_width, frozen_navigation_factor,
)


class PoseGraph:
    """Public backend API; each keyframe has Pose3 + velocity + ConstantBias.

    Tracker poses initialize variables only. There are no relative-pose
    measurement factors, including at loop closure. Initial-state and window
    priors are separate from tactile/IMU/bias observation factors.
    """

    def __init__(self, config):
        self.config = config
        self.pose_graph_config = dict(config['gelslam_config']['pose_graph_config'])
        self.imu_config = dict(config.get('imu_config', {}))
        transform = self.imu_config.get('gelsight_T_imu', np.eye(4))
        self.mounting_calibrated = transform is not None
        self.gelsight_T_imu = np.asarray(np.eye(4) if transform is None else transform, dtype=float)
        self.imu_T_gelsight = np.linalg.inv(self.gelsight_T_imu)
        self.anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.full(6, self._positive('anchor_sigma', 1e-6)))
        self.graph = gtsam.NonlinearFactorGraph()
        self.graph_init = gtsam.Values()
        self.graph_result = None
        self.poses, self.states, self.keyframes = [], [], []
        self.edges, self.temporal_links = [], []
        self.prior_factor_idxs, self.prior_keys, self.prior_poses = [], [], []
        self.navigation_prior_idxs = {}
        self.initial_velocities = {}
        self.factor_kinds = []
        self.trial_transforms = {}
        self.keyframe_count = 0
        self.window_start = 0
        self.marginal_prior = gtsam.NonlinearFactorGraph()
        self._marginalized_factor_idxs = set()
        self.tactile_skipped = []

    def _positive(self, name, default):
        value = float(self.pose_graph_config.get(name, default))
        if not np.isfinite(value) or value <= 0:
            raise ValueError('pose_graph_config.' + name + ' must be positive and finite')
        return value

    @staticmethod
    def _pose(matrix):
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError('pose must be a finite 4x4 matrix')
        return gtsam.Pose3(matrix)

    def _add_factor(self, factor, kind):
        index = self.graph.size()
        self.graph.add(factor)
        self.factor_kinds.append(kind)
        return index

    def _put_state(self, key, state, insert=False):
        operation = self.graph_init.insert if insert else self.graph_init.update
        operation(key, self._pose(state.pose))
        operation(velocity_key(key), np.asarray(state.velocity))
        operation(bias_key(key), gtsam.imuBias.ConstantBias(state.accel_bias, state.gyro_bias))

    def _add_prior(self, trial_group, key, pose=None):
        pose = self.poses[key] if pose is None else np.asarray(pose).copy()
        while len(self.prior_factor_idxs) <= trial_group:
            self.prior_factor_idxs.append(-1)
            self.prior_keys.append(-1)
            self.prior_poses.append(None)
        self.prior_factor_idxs[trial_group] = self._add_factor(split_prior_factor(
            key, pose, self.anchor_noise, self.imu_T_gelsight), 'pose_anchor')
        self.prior_keys[trial_group], self.prior_poses[trial_group] = key, pose.copy()
        state = self.states[key]
        self.initial_velocities[key] = state.velocity.copy()
        indices = [self._add_factor(gtsam.PriorFactorVector(
            velocity_key(key), state.velocity,
            gtsam.noiseModel.Isotropic.Sigma(3, self._positive('initial_velocity_sigma_mps', .1))),
            'velocity_prior')]
        sigmas = np.repeat([self._positive('initial_accel_bias_sigma_mps2', .1),
                            self._positive('initial_gyro_bias_sigma_rps', .01)], 3)
        indices.append(self._add_factor(gtsam.PriorFactorConstantBias(
            bias_key(key), gtsam.imuBias.ConstantBias(state.accel_bias, state.gyro_bias),
            gtsam.noiseModel.Diagonal.Sigmas(sigmas)), 'bias_prior'))
        self.navigation_prior_idxs[key] = indices

    @staticmethod
    def _record(keyframe):
        state = getattr(keyframe, 'navigation_state', None)
        return SimpleNamespace(
            H=np.asarray(keyframe.H), C=np.asarray(keyframe.C),
            N=np.asarray(keyframe.N), L=np.asarray(keyframe.L), ppmm=keyframe.ppmm,
            trial_T_curr=np.asarray(keyframe.trial_T_curr).copy(),
            trial_group=int(keyframe.trial_group), new_trial_flag=bool(keyframe.is_new_trial()),
            navigation_state=None if state is None else state.copy(),
            gravity=np.asarray(getattr(keyframe, 'gravity', None), dtype=float),
            imu_samples=tuple(getattr(keyframe, 'imu_samples', ())),
            tactile_valid=bool(getattr(keyframe, 'tactile_valid', True)),
        )

    def _validate_imu_interval(self, samples, previous_time, timestamp):
        max_gap = self._positive('imu_max_sample_gap_s', .05)
        if (len(samples) < 2 or abs(samples[0].timestamp - previous_time) > 1e-8
                or abs(samples[-1].timestamp - timestamp) > 1e-8
                or timestamp <= previous_time
                or any(b.timestamp <= a.timestamp or b.timestamp - a.timestamp > max_gap
                       for a, b in zip(samples, samples[1:]))):
            raise ValueError('backend IMU interval must cover both keyframe times with no gaps')

    def _add_imu(self, reference, target):
        packet, previous = self.keyframes[target], self.states[reference]
        samples = packet.imu_samples
        self._validate_imu_interval(samples, previous.timestamp, self.states[target].timestamp)
        pim = preintegrate_imu(samples, previous, self.imu_config,
                              self.gelsight_T_imu, packet.gravity)
        self._add_factor(gtsam.ImuFactor(
            reference, velocity_key(reference), target, velocity_key(target), bias_key(reference), pim), 'imu')
        densities = np.repeat([float(self.imu_config.get('accelerometer_bias_random_walk', .0002)),
                               float(self.imu_config.get('gyroscope_bias_random_walk', .00002))], 3)
        self._add_factor(gtsam.BetweenFactorConstantBias(
            bias_key(reference), bias_key(target), gtsam.imuBias.ConstantBias(),
            gtsam.noiseModel.Diagonal.Sigmas(densities * np.sqrt(pim.deltaTij()))), 'bias')
        self.temporal_links.append((reference, target))

    def _add_tactile(self, reference, target, kind='tactile', initial_target=None):
        if not (self.keyframes[reference].tactile_valid and self.keyframes[target].tactile_valid):
            self.tactile_skipped.append((reference, target, 'IMU-only recovery'))
            return False
        try:
            pair = TactilePair(self.keyframes[reference], self.keyframes[target],
                               self.poses[reference], self.poses[target] if initial_target is None else initial_target,
                               self.gelsight_T_imu, self.pose_graph_config)
        except ValueError as error:
            if 'shared surface points' not in str(error):
                raise
            self.tactile_skipped.append((reference, target, str(error)))
            return False
        for factor, name in zip(pair.factors(reference, target), ('normal', 'height')):
            self._add_factor(factor, name)
        # Metadata only: deliberately no relative-pose measurement is stored.
        self.edges.append((reference, target, None, kind))
        return True

    def add_keyframes(self, keyframedb, updated_size, targeted_size, build_factors=True):
        if build_factors and not self.mounting_calibrated:
            raise ValueError('15D backend requires calibrated imu_config.gelsight_T_imu')
        for key in range(updated_size, targeted_size):
            if key != len(self.poses):
                raise RuntimeError('keyframes must be appended in index order')
            source = keyframedb[key]
            if not build_factors:
                self.poses.append(np.asarray(source.trial_T_curr).copy())
                self.states.append(None)
                continue
            state = getattr(source, 'navigation_state', None)
            gravity = np.asarray(getattr(source, 'gravity', None), dtype=float)
            if state is None or gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
                raise ValueError('15D backend requires synchronized navigation seed and gravity')
            if (not np.isfinite(state.timestamp)
                    or any(np.asarray(v).shape != (3,) or not np.all(np.isfinite(v))
                           for v in (state.velocity, state.accel_bias, state.gyro_bias))):
                raise ValueError('backend navigation seed must be finite')
            self._pose(state.pose)
            if not source.is_new_trial():
                if key == 0:
                    raise ValueError('first keyframe must start a trial')
                self._validate_imu_interval(getattr(source, 'imu_samples', ()),
                                            self.states[key - 1].timestamp, state.timestamp)
            packet = self._record(source)
            transform = self.trial_transforms.setdefault(packet.trial_group, np.eye(4))
            state = state.copy()
            state.pose = transform @ state.pose
            state.velocity = transform[:3, :3] @ state.velocity
            self.states.append(state)
            self.poses.append(gelsight_pose(state.pose, self.gelsight_T_imu))
            self.keyframes.append(packet)
            self._put_state(key, state, insert=True)
            if packet.new_trial_flag:
                self._add_prior(packet.trial_group, key)
            else:
                if key == 0:
                    raise ValueError('first keyframe must start a trial')
                self._add_imu(key - 1, key)
                self._add_tactile(key - 1, key)
        self.keyframe_count = targeted_size

    def _solutions(self):
        self.graph_result = gtsam.Values(self.graph_init)
        return PoseGraphSolutions([pose.astype(np.float32) for pose in self.poses])

    def raw_solutions(self):
        return self._solutions()

    def _component(self, key):
        connected = {key}
        pairs = self.temporal_links + [(r, t) for r, t, _, _ in self.edges]
        changed = True
        while changed:
            changed = False
            for a, b in pairs:
                if (a in connected) != (b in connected):
                    connected.update((a, b))
                    changed = True
        return connected

    def _add_loop_pair(self, reference, target, target_T_reference):
        proposed = self.poses[reference] @ np.linalg.inv(target_T_reference)
        if not self._add_tactile(reference, target, 'loop', initial_target=proposed):
            return False
        # Seed a disconnected trial using registration, never a pose factor.
        # Exclude the just-added loop when finding the former target component.
        edge = self.edges.pop()
        component = self._component(target)
        self.edges.append(edge)
        if reference not in component:
            delta = proposed @ np.linalg.inv(self.poses[target])
            groups = {self.keyframes[key].trial_group for key in component}
            for group in groups:
                self.trial_transforms[group] = delta @ self.trial_transforms[group]
            for key in component:
                state = self.states[key]
                state.pose = delta @ state.pose
                state.velocity = delta[:3, :3] @ state.velocity
                self.poses[key] = gelsight_pose(state.pose, self.gelsight_T_imu)
                self._put_state(key, state)
                # Velocity prior is expressed in the newly aligned world axes.
                if key in self.navigation_prior_idxs:
                    index = self.navigation_prior_idxs[key][0]
                    self.graph.remove(index)
                    self.initial_velocities[key] = delta[:3, :3] @ self.initial_velocities[key]
                    self.navigation_prior_idxs[key][0] = self._add_factor(gtsam.PriorFactorVector(
                        velocity_key(key), self.initial_velocities[key], gtsam.noiseModel.Isotropic.Sigma(
                            3, self._positive('initial_velocity_sigma_mps', .1))), 'velocity_prior')
        self._reset_marginal_prior()
        return True

    def detect_and_add_loops(self, keyframedb, tar_kidx, coverage_graph):
        target = keyframedb[tar_kidx]
        matched = []
        for reference_key in range(tar_kidx):
            reference = keyframedb[reference_key]
            if reference.trial_group == target.trial_group and reference_key == tar_kidx - 1:
                continue
            if len(reference.kp) == 0 or len(target.kp) == 0:
                continue
            if reference_key < coverage_graph.size() and not coverage_graph[reference_key].is_active:
                continue
            matches = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True).match(
                reference.des, target.des
            )
            if len(matches) < int(self.pose_graph_config["min_sift_inliers"]):
                continue
            affine, inliers = cv2.estimateAffinePartial2D(
                np.float32([reference.kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2),
                np.float32([target.kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2),
                maxIters=100, ransacReprojThreshold=3,
            )
            if affine is None or inliers is None:
                continue
            if int(np.sum(inliers)) < int(self.pose_graph_config["min_sift_inliers"]):
                continue
            initial = M2T(affine, target.L.shape[1], target.L.shape[0], target.ppmm)
            try:
                tar_T_ref = normalflow(
                    reference.N, reference.C, reference.H, reference.L,
                    target.N, target.C, target.H, target.L,
                    initial, target.ppmm,
                    scr_threshold=self.pose_graph_config["scr_threshold"],
                    ccs_threshold=self.pose_graph_config["ccs_threshold"],
                )
            except LoseTrackError:
                continue
            # Registration is used for association and initialization only.
            # The optimization factors read the original normal/height maps.
            if self._add_loop_pair(reference_key, tar_kidx, tar_T_ref):
                matched.append(reference_key)
        return matched

    def remove_prior_factors(self, removed_trial_groups):
        for group in removed_trial_groups:
            if 0 <= group < len(self.prior_factor_idxs):
                index = self.prior_factor_idxs[group]
                if index >= 0:
                    self.graph.remove(index)
                    self.prior_factor_idxs[group] = -1
                    self.prior_keys[group] = -1
                    self._reset_marginal_prior()

    def _reset_marginal_prior(self):
        self.window_start = 0
        self.marginal_prior = gtsam.NonlinearFactorGraph()
        self._marginalized_factor_idxs.clear()

    def _marginalize_before(self, start):
        if start < self.window_start:
            self._reset_marginal_prior()
        absorbed, expired = set(), set()
        factors = gtsam.NonlinearFactorGraph()
        for index in range(self.marginal_prior.size()):
            factor = self.marginal_prior.at(index)
            factors.add(factor)
            expired.update(key for key in factor.keys() if state_index(key) < start)
        for index in range(self.graph.size()):
            factor = self.graph.at(index)
            if factor is None or index in self._marginalized_factor_idxs:
                continue
            if any(state_index(key) < start for key in factor.keys()):
                factors.add(factor)
                absorbed.add(index)
                expired.update(key for key in factor.keys() if state_index(key) < start)
        if expired:
            linearized = factors.linearize(self.graph_init)
            keys = {key for index in range(factors.size()) for key in factors.at(index).keys()}
            retained = sorted(keys - expired)
            ordering = gtsam.Ordering()
            for key in sorted(expired) + retained:
                ordering.push_back(key)
            H, g = linearized.hessian(ordering)
            reduced = schur_complement(H, g, sum(variable_width(key) for key in expired))
            prior = gtsam.NonlinearFactorGraph()
            if retained:
                # Preserve the existing zero-gradient window-prior policy,
                # now retaining all 15D cross terms and their frozen anchors.
                A, rhs = information_rows(reduced.information)
                prior.add(frozen_navigation_factor(retained, A, rhs, self.graph_init))
            self.marginal_prior = prior
        self._marginalized_factor_idxs.update(absorbed)
        self.window_start = start

    def _window_graph(self, start):
        graph, values = gtsam.NonlinearFactorGraph(), gtsam.Values()
        for index in range(start, self.keyframe_count):
            for key in state_keys(index):
                values.insert(key, self.graph_init.atPose3(key) if variable_type(key) == 'pose' else
                              self.graph_init.atVector(key) if variable_type(key) == 'vector' else
                              self.graph_init.atConstantBias(key))
        if self.pose_graph_config.get('marginalization_enabled', True):
            self._marginalize_before(start)
            for index in range(self.marginal_prior.size()):
                graph.add(self.marginal_prior.at(index))
        else:
            # Condition on all 15 dimensions at each component's window boundary.
            for index in range(start, self.keyframe_count):
                if index == start or self.keyframes[index].new_trial_flag:
                    keys = state_keys(index)
                    widths = sum(variable_width(key) for key in keys)
                    graph.add(frozen_navigation_factor(keys, np.eye(widths) * 1e6,
                                                       np.zeros(widths), self.graph_init))
        for index in range(self.graph.size()):
            factor = self.graph.at(index)
            if factor is not None and all(state_index(key) >= start for key in factor.keys()):
                graph.add(factor)
        return graph, values

    def solve(self, global_optimization=False):
        if not self.poses:
            return self.raw_solutions()
        if any(state is None for state in self.states):
            raise ValueError('pose-only saved data cannot run the 15D optimizer; replay with IMU/navigation data')
        if (self.pose_graph_config.get('loop_closure_only_optimization', False)
                and not global_optimization):
            return self.raw_solutions()
        if self.pose_graph_config.get('local_window_optimization', True) and not global_optimization:
            window = max(2, int(self.pose_graph_config.get('local_window_size', 10)))
            start = max(0, self.keyframe_count - window)
            graph, values = self._window_graph(start)
        else:
            start, graph, values = 0, self.graph, self.graph_init
        result = optimize_split_pose_graph(
            graph, values, max_iterations=int(self.pose_graph_config.get('max_iterations', 10)),
            variable_types={key: variable_type(key) for key in values.keys()})
        for key in range(start, self.keyframe_count):
            state = self.states[key]
            state.pose = result.atPose3(key).matrix()
            state.velocity = result.atVector(velocity_key(key))
            bias = result.atConstantBias(bias_key(key))
            state.accel_bias, state.gyro_bias = bias.accelerometer(), bias.gyroscope()
            self.poses[key] = gelsight_pose(state.pose, self.gelsight_T_imu)
            self._put_state(key, state)
        if graph is self.graph:
            self._reset_marginal_prior()
        return self._solutions()

    def save(self, save_dir):
        data = dict(format_version=2, config=self.config, keyframes=self.keyframes,
                    states=self.states, edges=self.edges, prior_keys=self.prior_keys,
                    poses=self.poses, trial_transforms=self.trial_transforms)
        with open(os.path.join(save_dir, 'pose_graph.pkl'), 'wb') as handle:
            pickle.dump(data, handle)

    @property
    def visual_edges(self):
        return [SimpleNamespace(reference=r, target=t, kind=kind) for r, t, _, kind in self.edges]

    @classmethod
    def load(cls, load_dir):
        with open(os.path.join(load_dir, 'pose_graph.pkl'), 'rb') as handle:
            data = pickle.load(handle)
        if data.get('format_version') != 2:
            # Keep historical reconstruction viewing available. Never invent
            # velocity, biases or IMU factors to optimize a pose-only snapshot.
            if 'poses' not in data:
                raise ValueError('unsupported graph snapshot; replay the source data')
            instance = cls({'gelslam_config': {'pose_graph_config': data['pose_graph_config']},
                            'imu_config': {'gelsight_T_imu': data.get('gelsight_T_imu', np.eye(4))}})
            instance.poses = [np.asarray(pose).copy() for pose in data['poses']]
            instance.states = [None] * len(instance.poses)
            instance.edges = data.get('edges', [])
            instance.keyframe_count = len(instance.poses)
            instance._solutions()
            return instance
        instance = cls(data['config'])
        records = data['keyframes']
        if not records:
            instance.poses, instance.states = data['poses'], data['states']
            instance.edges = data['edges']
            instance.keyframe_count = len(instance.poses)
            return instance
        instance.trial_transforms = data['trial_transforms']
        frames = []
        for record in records:
            frame = SimpleNamespace(**vars(record))
            frame.is_new_trial = lambda record=record: record.new_trial_flag
            frames.append(frame)
        instance.add_keyframes(frames, 0, len(frames))
        for key, state in enumerate(data['states']):
            instance.states[key] = state.copy()
            instance.poses[key] = gelsight_pose(state.pose, instance.gelsight_T_imu)
            instance._put_state(key, state)
        instance.trial_transforms = data['trial_transforms']
        # Recreate direct loop observations at the saved (already aligned) poses.
        for reference, target, _, kind in data['edges']:
            if kind == 'loop':
                instance._add_tactile(reference, target, 'loop')
        instance.remove_prior_factors([group for group, key in enumerate(data['prior_keys']) if key < 0])
        instance._solutions()
        return instance


class PoseGraphSolutions:
    def __init__(self, solved_start_T_currs=None):
        self.solved_start_T_currs = [] if solved_start_T_currs is None else solved_start_T_currs

    def __getitem__(self, index):
        return self.solved_start_T_currs[index]

    def size(self):
        return len(self.solved_start_T_currs)

    def save(self, save_dir):
        with open(os.path.join(save_dir, "pose_graph_solutions.pkl"), "wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, load_dir):
        with open(os.path.join(load_dir, "pose_graph_solutions.pkl"), "rb") as handle:
            return pickle.load(handle)
