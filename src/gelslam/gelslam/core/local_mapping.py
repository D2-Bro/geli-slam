"""Single-owner backend map service; loop search has a separate private cache."""
from dataclasses import dataclass
import numpy as np
from gelslam.core.keyframe import KeyFrameDB
from gelslam.core.slam_packets import ArrayBridge
from gelslam.core.gtsam_pose_graph import PoseGraph, loop_search_exclusion_count
from gelslam.core.parent_groups_info import ParentGroupsInfo
from gelslam.core.coverage_graph import CoverageGraph
from gelslam.core.shared_map import MapEstimate
from gelslam.core.imu_navigation import gelsight_pose
from gelslam.core.slam_packets import apply_keyframe_update


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

    def _apply_keyframe_update(self, row):
        """Replace an earlier keyframe's stored seed by the frontend's later estimate.

        The frontend sends the change relative to the anchor basis that maps onto
        this seed, so the change is applied to the stored seed directly (pose by
        right multiplication, velocity in seed body axes). KeyFrame.published_T_current
        records the change from the published pose; the final evaluation uses it to
        tell updated keyframes and re-resolves their frames (pose_graph_node).
        """
        kid, timestamp = int(round(row[0])), float(row[1])
        index = self.db.get_kid2kidx().get(kid)
        keyframe = self.db[index] if index is not None else None
        state = getattr(keyframe, 'navigation_state', None)
        if state is None or abs(state.timestamp - timestamp) > 1e-9:
            self.logger.warning('Local BA -- keyframe update for unknown or mismatched kid %d ignored' % kid)
            return False
        if keyframe.is_new_trial():
            # A group root fixes its PGO anchor when added; keep it unchanged.
            self.logger.info('Local BA -- keyframe update for group root kid %d not applied' % kid)
            return False
        published = np.asarray(keyframe.trial_T_curr, dtype=float)
        keyframe.navigation_state = apply_keyframe_update(state, row)
        keyframe.trial_T_curr = gelsight_pose(keyframe.navigation_state.pose, self.graph.gelsight_T_imu)
        keyframe.published_T_current = (getattr(keyframe, 'published_T_current', np.eye(4))
                                        @ np.linalg.solve(published, keyframe.trial_T_curr))
        if index < len(self.graph.states) and self.graph.states[index] is not None:
            updated = apply_keyframe_update(self.graph.states[index], row)
            self.graph.states[index] = updated
            self.graph.poses[index] = gelsight_pose(updated.pose, self.graph.gelsight_T_imu)
            if index < len(self.graph.keyframes):
                record = self.graph.keyframes[index]
                record.trial_T_curr = np.asarray(keyframe.trial_T_curr).copy()
                record.navigation_state = keyframe.navigation_state.copy()
        elif index < len(self.graph.poses):
            # Optimization disabled: the graph holds raw keyframe poses only.
            self.graph.poses[index] = np.asarray(keyframe.trial_T_curr).copy()
        change = keyframe.published_T_current
        self.logger.info('Local BA -- keyframe %d seed updated by the frontend: %.4f mm from its published pose'
                         % (kid, np.linalg.norm(change[:3, 3]) * 1e3))
        return True

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
            # Express their transmitted anchor basis in the stored keyframe
            # states before initializing a new state. This is not a factor.
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
            # The frontend's later estimate of the previous keyframe, relative to
            # the same anchor basis. Apply it after the rebasing above, which
            # used the old seed with that basis. Also with optimization disabled:
            # the raw keyframe poses then follow the frontend's final estimates.
            update = np.asarray(getattr(packet, 'keyframe_update', ()), dtype=float)
            if update.size:
                self._apply_keyframe_update(update)
        size = self.db.size()
        self.graph.add_keyframes(self.db, old_size, size, build_factors=self.enabled)
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
            # pose in the separate PGO.
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
        self.solutions = (self.graph.solve(global_optimization=True)
                          if self.enabled and global_solve
                          else self.graph.raw_solutions())
        if global_solve:
            pgo = self.graph.loop_pgo
            self.logger.info(
                "Global PGO -- 15D per keyframe; keyframes=%d, state dimension=%d, "
                "cost=%.6f -> %.6f" % (size, pgo.graph_result.dim(),
                                        pgo.last_initial_cost, pgo.last_final_cost))
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
        self.exclude_recent_keyframes = loop_search_exclusion_count(
            config['gelslam_config']['pose_graph_config'])
        value = config['gelslam_config']['pose_graph_config'].get(
            'loop_search_min_interval_s', 0.)
        try:
            self.min_interval_s = float(value)
        except (TypeError, ValueError):
            raise ValueError('loop_search_min_interval_s must be finite and nonnegative') from None
        if isinstance(value, bool) or not np.isfinite(self.min_interval_s) or self.min_interval_s < 0:
            raise ValueError('loop_search_min_interval_s must be finite and nonnegative')
        self.db = database(config, logger)
        self.epoch = 0
        # DB index zero has no preceding reference. Reset with the map epoch.
        self._next_target = 1
        self._last_search_timestamp = None

    def detect(self, snapshot):
        if self.epoch != snapshot.epoch:
            self.__init__(self.config, self.logger)
            self.epoch = snapshot.epoch
        for packet in snapshot.observations[self.db.size():]:
            # Feature extraction belongs to this worker, never the BA worker.
            self.db.insert(packet, log_prefix="Loop Search", compute_features=True)
        result = []
        last_timestamp = self._last_search_timestamp
        for target_index in range(self._next_target, self.db.size()):
            keyframe = self.db[target_index]
            state = keyframe.navigation_state
            timestamp = state.timestamp if state is not None else keyframe.timestamp
            if timestamp is not None and not np.isfinite(timestamp):
                timestamp = None
            if self.min_interval_s > 0 and timestamp is not None and last_timestamp is not None:
                elapsed = timestamp - last_timestamp
                # Respect the boundary even for epoch-sized float timestamps.
                tolerance = max(1e-9, abs(np.spacing(timestamp)), abs(np.spacing(last_timestamp)))
                if 0 <= elapsed < self.min_interval_s - tolerance:
                    continue
            # Missing or backwards time starts a new interval, avoiding a stall.
            pairs = PoseGraph.detect_loop_candidates(
                self.db, target_index,
                self.config['gelslam_config']['pose_graph_config'])
            for reference, target, transform in pairs:
                transform = np.array(transform, copy=True)
                transform.setflags(write=False)
                result.append(LoopCandidate(snapshot.epoch, snapshot.revision,
                    self.db[reference].kid, self.db[target].kid, transform))
            last_timestamp = timestamp
        # Commit only after the whole result is ready for the caller.
        self._next_target = max(1, self.db.size())
        self._last_search_timestamp = last_timestamp
        return result
