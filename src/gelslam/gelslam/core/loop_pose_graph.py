"""Gravity-preserving 4DoF loop correction, separate from the 15D local graph.

Like gelslam-origin, loop measurements are reference_T_target transforms.
All translations and noise sigmas here are in metres. Local VI supplies the
odometry chain; no normal/height map is evaluated by this optimizer.
"""
import gtsam
import numpy as np

from gelslam.core.split_pose import optimize_split_pose_graph
from gelslam.core.gravity_pose import (
    align_directions, gravity_between_factor, gravity_pose, gravity_prior_factor,
    up_direction, yaw_angle,
)


class LoopPoseGraph:
    degrees_of_freedom = 4

    def __init__(self, config):
        self.config = dict(config)
        sigmas = np.repeat([
            self._positive('loop_pgo_rotation_sigma_rad', .01),
            self._positive('loop_pgo_translation_sigma_m', .0001),
        ], 3)
        self.noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
        self.anchor_noise = gtsam.noiseModel.Isotropic.Sigma(
            6, self._positive('anchor_sigma', 1e-6))
        self.graph = gtsam.NonlinearFactorGraph()
        self.graph_init = gtsam.Values()
        self.graph_result = None
        self.factor_kinds = []
        self.loops = []
        self.corrections = []
        self.roots = {}
        self.anchor_poses = {}
        self.removed_groups = set()
        self.local_poses = []
        self.up = np.array([0., 0., 1.])
        self.local_up = []
        self.temporal_links = []
        self.last_initial_cost = None
        self.last_final_cost = None

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

    def synchronize(self, local_poses, records, temporal_links):
        """Extend map corrections without changing the local navigation states."""
        self.local_poses = [np.asarray(pose).copy() for pose in local_poses]
        self.local_up = [up_direction(getattr(record, 'gravity', [0., 0., -1.]))
                         for record in records]
        if self.local_up:
            self.up = self.local_up[0].copy()
        self.temporal_links = list(temporal_links)
        for key in range(len(self.corrections), len(local_poses)):
            record = records[key]
            correction = np.eye(4)
            if record.new_trial_flag:
                correction[:3, :3] = align_directions(self.local_up[key], self.up)
            else:
                correction = self.corrections[key - 1].copy()
            self.corrections.append(correction)
            if record.new_trial_flag:
                self.roots[record.trial_group] = key
                self.anchor_poses[record.trial_group] = correction @ local_poses[key]

    def map_poses(self):
        return [correction @ pose for correction, pose in
                zip(self.corrections, self.local_poses)]

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
            # Align only the map estimate of a disconnected component. Local
            # IMU states, gravity, initialization priors and biases stay intact.
            poses = self.map_poses()
            desired = poses[reference] @ measurement
            shift = np.eye(4)
            shift[:3, :3] = gtsam.Rot3.Expmap(
                self.up * yaw_angle(desired[:3, :3] @ poses[target][:3, :3].T, self.up)).matrix()
            shift[:3, 3] = desired[:3, 3] - shift[:3, :3] @ poses[target][:3, 3]
            for key in component:
                self.corrections[key] = shift @ self.corrections[key]
        self.loops.append((reference, target, measurement))
        return True

    def remove_prior_factors(self, groups):
        # This removes PGO gauge anchors only. Each disconnected local VI
        # trial still needs its own original initialization prior.
        self.removed_groups.update(groups)

    def solve(self):
        """Optimize world yaw/XYZ, retain local tilt and six measured residuals."""
        graph, values = gtsam.NonlinearFactorGraph(), gtsam.Values()
        kinds = []
        rotations = [align_directions(up, self.up) @ pose[:3, :3]
                     for up, pose in zip(self.local_up, self.local_poses)]
        for key, pose in enumerate(self.map_poses()):
            yaw = yaw_angle(pose[:3, :3] @ rotations[key].T, self.up)
            values.insert(key, np.r_[yaw, pose[:3, 3]])
        fixed = set()
        for group, key in self.roots.items():
            if group not in self.removed_groups:
                anchor = self.anchor_poses[group]
                yaw = yaw_angle(anchor[:3, :3] @ rotations[key].T, self.up)
                vector = np.r_[yaw, anchor[:3, 3]]
                values.update(key, vector)
                graph.add(gravity_prior_factor(key, gravity_pose(vector, rotations[key], self.up),
                                              self.anchor_noise, rotations[key], self.up))
                kinds.append('pose_anchor')
                fixed.add(key)
        for reference, target in self.temporal_links:
            measurement = np.linalg.inv(self.local_poses[reference]) @ self.local_poses[target]
            graph.add(gravity_between_factor(reference, target, measurement, self.noise,
                                             rotations, self.up))
            kinds.append('odometry')
        for reference, target, measurement in self.loops:
            graph.add(gravity_between_factor(reference, target, measurement, self.noise,
                                             rotations, self.up))
            kinds.append('loop')
        initial_cost = float(graph.error(values))
        result = optimize_split_pose_graph(
            graph, values, max_iterations=int(self.config.get('max_iterations', 10)),
            variable_types={key: 'vector' for key in values.keys()}, fixed_keys=fixed)
        final_cost = float(graph.error(result))
        if not np.isfinite(final_cost) or final_cost > initial_cost + 1e-10 * max(initial_cost, 1.):
            raise ValueError('loop PGO did not produce a finite non-increasing cost')
        corrections = [gravity_pose(result.atVector(key), rotations[key], self.up) @ np.linalg.inv(pose)
                       for key, pose in enumerate(self.local_poses)]
        if not all(np.isfinite(value).all() for value in corrections):
            raise ValueError('loop PGO produced a non-finite map correction')
        self.graph, self.graph_init, self.graph_result = graph, values, result
        self.factor_kinds, self.corrections = kinds, corrections
        self.last_initial_cost, self.last_final_cost = initial_cost, final_cost

    def snapshot(self):
        return dict(degrees_of_freedom=self.degrees_of_freedom,
                    loops=self.loops, corrections=self.corrections, roots=self.roots,
                    anchor_poses=self.anchor_poses, removed_groups=self.removed_groups)

    def restore(self, data):
        self.loops = [(r, t, self.transform(measurement)) for r, t, measurement in data['loops']]
        self.corrections = [self.transform(value) for value in data['corrections']]
        self.roots = dict(data['roots'])
        self.anchor_poses = dict(data['anchor_poses'])
        self.removed_groups = set(data['removed_groups'])
