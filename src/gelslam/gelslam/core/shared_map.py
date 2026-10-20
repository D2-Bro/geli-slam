"""Short-lock, versioned storage shared by tracking, Local BA and loop search.

Observations are immutable. GTSAM graphs and mutable KeyFrameDB objects never
enter this store. A solution revision describes one atomic batch of estimates.
"""
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from time import monotonic
import numpy as np
from gelslam.core.slam_packets import readonly


@dataclass(frozen=True)
class FrozenNavigationState:
    timestamp: float
    pose: np.ndarray
    velocity: np.ndarray
    accel_bias: np.ndarray
    gyro_bias: np.ndarray

    def copy(self):
        from gelslam.core.numpy_vi import NavigationState
        return NavigationState(self.timestamp, self.pose.copy(), self.velocity.copy(),
                               self.accel_bias.copy(), self.gyro_bias.copy())


def frozen_state(state):
    if state is None:
        return None
    return FrozenNavigationState(state.timestamp, *(readonly(getattr(state, name))
        for name in ('pose', 'velocity', 'accel_bias', 'gyro_bias')))


@dataclass(frozen=True)
class MapEstimate:
    # Navigation state in the original tracking trial's coordinates.
    state: object
    map_pose: np.ndarray
    trial_pose: np.ndarray
    revision: int
    parent_group: int
    committed_at: float = 0.


@dataclass(frozen=True)
class MapSnapshot:
    epoch: int
    revision: int
    observations: tuple
    estimates: object
    kid_to_index: object
    committed_at: float


class SharedMap:
    def __init__(self):
        self._lock = RLock()
        self._epoch = 0
        self._revision = 0
        self._committed_at = 0.
        self._observations = []
        self._indices = {}
        self._estimates = {}

    def work_token(self):
        with self._lock:
            return self._epoch, len(self._observations)

    def append_keyframe(self, packet):
        with self._lock:
            if packet.kid in self._indices:
                raise ValueError('duplicate keyframe ID')
            if self._observations and packet.kid <= self._observations[-1].kid:
                raise ValueError('keyframe IDs must increase')
            self._indices[packet.kid] = len(self._observations)
            self._observations.append(packet)
            return self._epoch

    def snapshot_for_optimization(self):
        with self._lock:
            return MapSnapshot(self._epoch, self._revision, tuple(self._observations),
                               MappingProxyType(dict(self._estimates)),
                               MappingProxyType(dict(self._indices)), self._committed_at)

    snapshot_for_tracking = snapshot_for_optimization

    def commit_solution(self, epoch, estimates):
        with self._lock:
            if epoch != self._epoch:
                return False
            expected_revision = self._revision
            previous = {kid: self._estimates.get(kid) for kid in estimates}
        # Array copies and numerical comparisons run outside the map lock.
        prepared = {kid: MapEstimate(frozen_state(value.state), readonly(value.map_pose),
                                    readonly(value.trial_pose), 0, value.parent_group)
                    for kid, value in estimates.items()}
        changed = {}
        for kid, value in prepared.items():
            old = previous[kid]
            differs = old is None or old.parent_group != value.parent_group
            if not differs:
                differs = (not np.allclose(old.trial_pose, value.trial_pose, rtol=0, atol=1e-12)
                           or not np.allclose(old.map_pose, value.map_pose, rtol=0, atol=1e-12)
                           or (old.state is None) != (value.state is None))
                if not differs and value.state is not None:
                    differs = any(not np.allclose(getattr(old.state, field),
                        getattr(value.state, field), rtol=0, atol=1e-12)
                        for field in ('pose', 'velocity', 'accel_bias', 'gyro_bias'))
            changed[kid] = differs
        with self._lock:
            if epoch != self._epoch or self._revision != expected_revision:
                return False
            if not set(prepared).issubset(self._indices):
                raise ValueError('solution contains unknown keyframes')
            self._revision += 1
            self._committed_at = monotonic()
            for kid, value in prepared.items():
                old = previous[kid]
                self._estimates[kid] = MapEstimate(
                    value.state, value.map_pose, value.trial_pose,
                    self._revision if changed[kid] else old.revision, value.parent_group,
                    self._committed_at if changed[kid] else old.committed_at)
            return True

    def reset(self):
        with self._lock:
            self._epoch += 1
            self._revision = 0
            self._committed_at = 0.
            self._observations.clear()
            self._indices.clear()
            self._estimates.clear()
