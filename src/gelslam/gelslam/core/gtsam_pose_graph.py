"""15D keyframe navigation store and separate global loop PGO."""

import os
import pickle
from types import SimpleNamespace

import cv2
import gtsam
import numpy as np
from normalflow.registration import LoseTrackError, normalflow

from gelslam.utils import M2T
from gelslam.core.imu_navigation import gelsight_pose
from gelslam.core.loop_pose_graph import LoopPoseGraph


def loop_search_exclusion_count(pose_graph_config):
    """Return the number of preceding keyframes excluded from loop matching."""
    value = pose_graph_config.get('loop_search_exclude_recent_keyframes', 5)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError('loop_search_exclude_recent_keyframes must be a nonnegative integer')
    return int(value)


class PoseGraph:
    """Public backend API; each keyframe has Pose3 + velocity + ConstantBias.

    Keyframe states are the tracking navigation seeds. A separate 15D global
    PGO contains contact odometry, IMU, bias and relative-pose loops. Its map
    corrections never replace the stored keyframe seeds.
    """

    def __init__(self, config):
        self.config = config
        self.pose_graph_config = dict(config['gelslam_config']['pose_graph_config'])
        self.imu_config = dict(config.get('imu_config', {}))
        transform = self.imu_config.get('gelsight_T_imu', np.eye(4))
        self.mounting_calibrated = transform is not None
        self.gelsight_T_imu = np.asarray(np.eye(4) if transform is None else transform, dtype=float)
        self.imu_T_gelsight = np.linalg.inv(self.gelsight_T_imu)
        self.poses, self.states, self.keyframes = [], [], []
        self.edges, self.temporal_links = [], []
        self.trial_transforms = {}
        self.keyframe_count = 0
        self.loop_pgo = LoopPoseGraph(self.pose_graph_config, self.imu_config)

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
            if not packet.new_trial_flag:
                # PGO uses frontend seed odometry and raw IMU for this link.
                self.temporal_links.append((key - 1, key))
        self.keyframe_count = targeted_size

    def _solutions(self):
        return PoseGraphSolutions([pose.astype(np.float32) for pose in self.map_poses()])

    def _sync_loop_pgo(self):
        if self.keyframes:
            self.loop_pgo.synchronize(self.poses, self.keyframes, self.temporal_links,
                                      navigation_states=self.states)

    def map_poses(self):
        if not self.keyframes:
            return self.poses
        self._sync_loop_pgo()
        return self.loop_pgo.map_poses()

    def map_state(self, key):
        """Return the loop-corrected navigation estimate for map publication."""
        state = self.states[key]
        if state is None:
            return None
        if len(self.loop_pgo.corrections) != len(self.poses):
            self._sync_loop_pgo()
        self.loop_pgo.local_states[key] = state.copy()
        return self.loop_pgo.map_state(key)

    def map_T_trial_group(self, group):
        if group not in self.loop_pgo.roots:
            self._sync_loop_pgo()
        return self.loop_pgo.trial_frame_correction(group) @ self.trial_transforms[group]

    def raw_solutions(self):
        return self._solutions()

    def _add_loop_pair(self, reference, target, target_T_reference):
        # Loop registration validates this pair independently of temporal recovery.
        self._sync_loop_pgo()
        try:
            added = self.loop_pgo.add_loop(reference, target, target_T_reference)
        except ValueError:
            return False
        if not added:
            return False
        self.edges.append((reference, target, np.linalg.inv(target_T_reference).copy(), 'loop'))
        return True

    @staticmethod
    def detect_loop_candidates(keyframedb, tar_kidx, pose_graph_config, active_indices=None):
        exclude_recent = loop_search_exclusion_count(pose_graph_config)
        target = keyframedb[tar_kidx]
        matched = []
        # Count from this target's DB index, including across trial boundaries.
        # K_t excludes K_(t-1)..K_(t-N); K_(t-N-1) is the newest candidate.
        for reference_key in range(max(0, tar_kidx - exclude_recent)):
            reference = keyframedb[reference_key]
            if len(reference.kp) == 0 or len(target.kp) == 0:
                continue
            if active_indices is not None and reference_key not in active_indices:
                continue
            matches = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True).match(
                reference.des, target.des
            )
            if len(matches) < int(pose_graph_config["min_sift_inliers"]):
                continue
            affine, inliers = cv2.estimateAffinePartial2D(
                np.float32([reference.kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2),
                np.float32([target.kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2),
                maxIters=100, ransacReprojThreshold=3,
            )
            if affine is None or inliers is None:
                continue
            if int(np.sum(inliers)) < int(pose_graph_config["min_sift_inliers"]):
                continue
            initial = M2T(affine, target.L.shape[1], target.L.shape[0], target.ppmm)
            try:
                tar_T_ref = normalflow(
                    reference.N, reference.C, reference.H, reference.L,
                    target.N, target.C, target.H, target.L,
                    initial, target.ppmm,
                    scr_threshold=pose_graph_config["scr_threshold"],
                    ccs_threshold=pose_graph_config["ccs_threshold"],
                )
            except LoseTrackError:
                continue
            # Registration supplies the fixed relative-pose measurement for
            # the separate loop PGO, after SCR/CCS and SIFT validation.
            matched.append((reference_key, tar_kidx, tar_T_ref))
        return matched

    def detect_and_add_loops(self, keyframedb, tar_kidx, coverage_graph):
        active = {key for key in range(tar_kidx + 1)
                  if key >= coverage_graph.size() or coverage_graph[key].is_active}
        return [reference for reference, target, transform in self.detect_loop_candidates(
            keyframedb, tar_kidx, self.pose_graph_config, active)
            if self._add_loop_pair(reference, target, transform)]

    def remove_prior_factors(self, removed_trial_groups):
        """Remove merged PGO gauge anchors; retain velocity/bias observations."""
        self.loop_pgo.remove_prior_factors(removed_trial_groups)

    def solve(self, global_optimization=False):
        """Run loop PGO when requested, then publish map estimates."""
        if not self.poses:
            return self.raw_solutions()
        if any(state is None for state in self.states):
            raise ValueError('pose-only saved data cannot run the 15D optimizer; replay with IMU/navigation data')
        if global_optimization and self.loop_pgo.loops:
            self._sync_loop_pgo()
            self.loop_pgo.solve()
        return self._solutions()

    def save(self, save_dir):
        self._sync_loop_pgo()
        data = dict(format_version=4, config=self.config, keyframes=self.keyframes,
                    states=self.states, edges=self.edges, poses=self.poses, trial_transforms=self.trial_transforms,
                    loop_pgo=self.loop_pgo.snapshot())
        with open(os.path.join(save_dir, 'pose_graph.pkl'), 'wb') as handle:
            pickle.dump(data, handle)

    @property
    def visual_edges(self):
        return [SimpleNamespace(reference=r, target=t, kind=kind) for r, t, _, kind in self.edges]

    @classmethod
    def load(cls, load_dir):
        with open(os.path.join(load_dir, 'pose_graph.pkl'), 'rb') as handle:
            data = pickle.load(handle)
        if data.get('format_version') not in {2, 3, 4}:
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
        if data['format_version'] == 2 and any(kind == 'loop' for _, _, _, kind in data['edges']):
            raise ValueError('legacy direct-loop snapshot lacks registered loop measurements; replay the source data')
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
        instance.trial_transforms = data['trial_transforms']
        if data['format_version'] >= 3:
            instance.loop_pgo.restore(data['loop_pgo'])
            instance.edges = data['edges']
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
