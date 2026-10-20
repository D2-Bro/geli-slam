"""Single-owner Local BA service; loop search has a separate private cache."""
from dataclasses import dataclass
import numpy as np
from gelslam.core.keyframe import KeyFrameDB
from gelslam.core.slam_packets import ArrayBridge
from gelslam.core.gtsam_pose_graph import PoseGraph
from gelslam.core.parent_groups_info import ParentGroupsInfo
from gelslam.core.coverage_graph import CoverageGraph
from gelslam.core.shared_map import MapEstimate
from gelslam.core.imu_navigation import gelsight_pose


@dataclass(frozen=True)
class LoopCandidate:
    epoch: int
    revision: int
    reference_kid: int
    target_kid: int
    target_T_reference: np.ndarray


def database(config, logger):
    db = KeyFrameDB(config['device_config']['ppmm'], logger,
                    config.get('imu_config', {}).get('gelsight_T_imu'),
                    config['gelslam_config']['pose_graph_config'].get('imu_max_sample_gap_s', .05))
    db.bridge = ArrayBridge()
    return db


class LocalMapping:
    def __init__(self, config, logger, enabled=True):
        self.config, self.logger, self.enabled = config, logger, enabled
        self.graph = PoseGraph(config)
        if enabled and not self.graph.mounting_calibrated:
            raise ValueError('15D backend requires calibrated imu_config.gelsight_T_imu')
        self.db = database(config, logger)
        self.groups = ParentGroupsInfo()
        self.coverage = CoverageGraph(config)
        self.solutions = self.graph.raw_solutions()
        self.epoch = 0
        self.accepted_pairs = set()

    def process(self, snapshot, candidates, shared_map):
        if self.epoch != snapshot.epoch:
            self.__init__(self.config, self.logger, self.enabled)
            self.epoch = snapshot.epoch
        old_size = self.db.size()
        for packet in snapshot.observations[old_size:]:
            if self.enabled and not packet.navigation_valid:
                raise ValueError('15D backend requires synchronized navigation keyframe seed')
            self.db.insert(packet, log_prefix="Local BA", compute_features=False)
            keyframe = self.db[-1]
            # Incoming tracking seeds can already contain loop map corrections.
            # Express their transmitted anchor basis in the independent local
            # VI graph before initializing a new state. This is not a factor.
            anchor_index = self.db.get_kid2kidx().get(packet.anchor_kid)
            if (self.enabled and not keyframe.is_new_trial() and anchor_index is not None
                    and anchor_index < self.db.size() - 1
                    and packet.anchor_state is not None):
                current = keyframe.navigation_state.copy()
                keyframe.navigation_state = current
                basis = packet.anchor_state
                # A queued batch can refer to a keyframe decoded earlier in
                # this batch, before add_keyframes() appends it to the graph.
                anchor_frame = self.db[anchor_index]
                corrected = (self.graph.states[anchor_index]
                             if anchor_index < len(self.graph.states)
                             else anchor_frame.navigation_state)
                if corrected is not None:
                    trial_T_local = (anchor_frame.tracker_trial_T_group
                                     @ np.linalg.inv(self.graph.trial_transforms.get(
                                         anchor_frame.trial_group, np.eye(4))))
                    corrected = corrected.copy()
                    corrected.pose = trial_T_local @ corrected.pose
                    corrected.velocity = trial_T_local[:3, :3] @ corrected.velocity
                    # Both anchor states now use the tracking trial axes.
                    shift = corrected.pose @ np.linalg.inv(basis.pose)
                    group_T_trial = np.linalg.inv(keyframe.tracker_trial_T_group)
                    trial_pose = keyframe.tracker_trial_T_group @ current.pose
                    current.pose = group_T_trial @ shift @ trial_pose
                    trial_velocity = keyframe.tracker_trial_T_group[:3, :3] @ current.velocity
                    current.velocity = group_T_trial[:3, :3] @ (
                        corrected.velocity + shift[:3, :3] @ (trial_velocity - basis.velocity))
                    current.accel_bias += corrected.accel_bias - basis.accel_bias
                    current.gyro_bias += corrected.gyro_bias - basis.gyro_bias
                    keyframe.trial_T_curr = gelsight_pose(current.pose, self.graph.gelsight_T_imu)
        size = self.db.size()
        factor_start = len(self.graph.factor_kinds)
        self.graph.add_keyframes(self.db, old_size, size, build_factors=self.enabled)
        # A batch can end with an IMU recovery after adding valid tactile pairs.
        # Trigger local VI from actual new observations, not the last packet.
        local_solve = 'gelsight' in self.graph.factor_kinds[factor_start:]
        self.groups.update_wrt_new_keyframes(self.db, old_size, size)
        self.coverage.add_new_coverage_nodes(old_size, size)
        global_solve = False
        merged_original, merged_new = [], []
        indices = self.db.get_kid2kidx()
        for candidate in candidates:
            pair = (candidate.reference_kid, candidate.target_kid)
            if (not self.enabled or candidate.epoch != snapshot.epoch
                    or pair in self.accepted_pairs
                    or any(kid not in indices for kid in pair)):
                continue
            reference, target = (indices[kid] for kid in pair)
            transform = np.asarray(candidate.target_T_reference)
            if (reference >= target or transform.shape != (4, 4)
                    or not np.all(np.isfinite(transform))):
                continue
            # Registration was validated by loop search. Store its relative
            # pose in the separate PGO; never add a local tactile loop factor.
            if not self.graph._add_loop_pair(reference, target, transform):
                continue
            self.accepted_pairs.add(pair)
            original, new, _, removed = self.groups.update_wrt_loop_closure(
                self.db, target, [reference], size)
            self.graph.remove_prior_factors(removed)
            merged_original.extend(original)
            merged_new.extend(new)
            global_solve = True
        if not size:
            return False
        self.solutions = (self.graph.solve(global_optimization=global_solve,
                                          local_optimization=local_solve)
                          if self.enabled and (local_solve or global_solve)
                          else self.graph.raw_solutions())
        estimates = {}
        for index, packet in enumerate(snapshot.observations[:size]):
            keyframe = self.db[index]
            state = self.graph.map_state(index)
            if state is not None:
                map_T_trial = (self.graph.map_T_trial_group(keyframe.trial_group)
                               @ np.linalg.inv(keyframe.tracker_trial_T_group))
                trial_T_map = np.linalg.inv(map_T_trial)
                state = state.copy()
                state.pose = trial_T_map @ state.pose
                state.velocity = trial_T_map[:3, :3] @ state.velocity
                trial_pose = gelsight_pose(state.pose, self.graph.gelsight_T_imu)
            else:
                trial_pose = np.asarray(packet.trial_t_curr).reshape(4, 4)
            estimates[packet.kid] = MapEstimate(
                state, self.solutions[index], trial_pose, 0,
                self.groups.get_parent_group(index, self.db))
        if not shared_map.commit_solution(snapshot.epoch, estimates):
            return False
        # Geometry/coverage work has no shared-map lock.
        if self.enabled:
            self.coverage.update_wrt_new_keyframes(list(range(old_size, size)), self.db,
                                                 self.solutions, self.groups)
            if global_solve:
                self.coverage.update_wrt_loop_closure(sorted(set(merged_original)),
                    sorted(set(merged_new)), self.db, self.solutions)
        return True

    def save(self, directory):
        self.graph.save(directory)
        self.db.save(directory)
        self.groups.save(directory)
        self.solutions.save(directory)


class LoopSearch:
    def __init__(self, config, logger):
        self.config, self.logger = config, logger
        self.db = database(config, logger)
        self.epoch = 0

    def detect(self, snapshot):
        if self.epoch != snapshot.epoch:
            self.__init__(self.config, self.logger)
            self.epoch = snapshot.epoch
        for packet in snapshot.observations[self.db.size():]:
            # Feature extraction belongs to this worker, never the BA worker.
            self.db.insert(packet, log_prefix="Loop Search", compute_features=True)
        if self.db.size() < 2:
            return []
        pairs = PoseGraph.detect_loop_candidates(
            self.db, self.db.size() - 1,
            self.config['gelslam_config']['pose_graph_config'])
        result = []
        for reference, target, transform in pairs:
            transform = np.array(transform, copy=True)
            transform.setflags(write=False)
            result.append(LoopCandidate(snapshot.epoch, snapshot.revision,
                self.db[reference].kid, self.db[target].kid, transform))
        return result
