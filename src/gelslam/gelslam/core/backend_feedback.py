"""Small, versioned ROS correction messages; no surface maps on the return path."""
from dataclasses import dataclass
from threading import Lock
from time import monotonic
from types import MappingProxyType

import numpy as np

from gelslam.core.imu_navigation import gelsight_pose
from gelslam.core.shared_map import MapEstimate, MapSnapshot, frozen_state
from gelslam.core.slam_packets import pack_navigation_state, unpack_navigation_state, readonly


@dataclass(frozen=True)
class FeedbackKeyframe:
    kid: int
    trial_id: int


def feedback_message(snapshot, session_id):
    from gelslam_msgs.msg import BackendFeedbackMsg
    message = BackendFeedbackMsg()
    message.session_id = session_id
    message.epoch, message.revision = snapshot.epoch, snapshot.revision
    for packet in snapshot.observations:
        estimate = snapshot.estimates.get(packet.kid)
        if estimate is None or estimate.state is None:
            continue
        message.kids.append(packet.kid)
        message.trial_ids.append(packet.trial_id)
        message.state_revisions.append(estimate.revision)
        message.states.extend(pack_navigation_state(estimate.state))
    return message


class FeedbackReceiver:
    """A ROS callback replaces one immutable snapshot; tracking reads at F_j."""
    def __init__(self, session_id, gelsight_T_imu):
        self.session_id = session_id
        self.extrinsic = np.asarray(gelsight_T_imu)
        self._lock = Lock()
        self._token = (-1, -1)
        self._snapshot = MapSnapshot(0, 0, (), MappingProxyType({}), MappingProxyType({}), 0.)

    def receive(self, message):
        if message.session_id != self.session_id:
            return False
        token = (message.epoch, message.revision)
        with self._lock:
            if token <= self._token:
                return False
        count = len(message.kids)
        if (len(message.trial_ids) != count or len(message.state_revisions) != count
                or len(message.states) != 26 * count
                or len(set(message.kids)) != count):
            raise ValueError('backend feedback has inconsistent state rows')
        now = monotonic()
        packets, estimates, indices = [], {}, {}
        for i, kid in enumerate(message.kids):
            revision = message.state_revisions[i]
            if revision > message.revision:
                raise ValueError('keyframe revision exceeds solution revision')
            state = frozen_state(unpack_navigation_state(message.states[26*i:26*(i+1)]))
            pose = readonly(gelsight_pose(state.pose, self.extrinsic))
            packets.append(FeedbackKeyframe(kid, message.trial_ids[i]))
            estimates[kid] = MapEstimate(state, pose, pose, revision, 0, now)
            indices[kid] = i
        snapshot = MapSnapshot(message.epoch, message.revision, tuple(packets),
            MappingProxyType(estimates), MappingProxyType(indices), now)
        with self._lock:
            if token <= self._token:
                return False
            self._token, self._snapshot = token, snapshot
        return True

    def snapshot(self):
        with self._lock:
            return self._snapshot
