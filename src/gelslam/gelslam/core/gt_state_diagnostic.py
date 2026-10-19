"""GT IMU states for an opt-in frontend input diagnostic."""

from bisect import bisect_left

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def resolve_gt_state_test(config):
    """Resolve legacy bools or independent input/prior diagnostic controls."""
    supplied = config.get('frontend_gt_previous_state_test', False)
    defaults = dict(enabled=False, previous_pose_from_gt=True,
                    previous_velocity_from_gt=True, reference_pose_from_gt=True,
                    reset_marginal_prior=True, previous_state_time='accepted_state')
    if isinstance(supplied, bool):
        # Preserve the original all-on/raw-previous-image diagnostic exactly.
        return dict(defaults, enabled=supplied, previous_state_time='previous_image')
    if not isinstance(supplied, dict):
        raise ValueError('frontend_gt_previous_state_test must be a boolean or mapping')
    if supplied.keys() - defaults.keys():
        raise ValueError('unknown GT state test controls: ' + ', '.join(sorted(supplied.keys() - defaults.keys())))
    settings = dict(defaults, **supplied)
    for key, value in settings.items():
        if key != 'previous_state_time' and not isinstance(value, bool):
            raise ValueError('GT state test %s must be a boolean' % key)
    if settings['previous_state_time'] not in {'accepted_state', 'previous_image'}:
        raise ValueError('GT state test previous_state_time must be accepted_state or previous_image')
    return settings


def gt_state_test_needs_gt(settings):
    return settings['enabled'] and any(settings[key] for key in (
        'previous_pose_from_gt', 'previous_velocity_from_gt', 'reference_pose_from_gt'))


class GroundTruthImuStates:
    """Bounded native poses; interpolate position, rotation and central velocity."""

    def __init__(self, max_samples=10000):
        self.samples = []
        self.max_samples = max_samples

    def add(self, timestamp, rotation, position):
        rotation, position = np.asarray(rotation), np.asarray(position)
        if (not np.isfinite(timestamp) or rotation.shape != (3, 3)
                or position.shape != (3,) or not np.all(np.isfinite(rotation))
                or not np.all(np.isfinite(position))):
            return False
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation, position
        index = bisect_left([t for t, _ in self.samples], timestamp)
        if index < len(self.samples) and self.samples[index][0] == timestamp:
            self.samples[index] = (float(timestamp), pose)
        else:
            self.samples.insert(index, (float(timestamp), pose))
        if len(self.samples) > self.max_samples:
            del self.samples[:-self.max_samples]
        return True

    def at(self, timestamp):
        """Return a world IMU pose and velocity; never extrapolate GT."""
        if not np.isfinite(timestamp) or not self.samples:
            return None
        times = [t for t, _ in self.samples]
        right = bisect_left(times, timestamp)
        if right == len(times):
            return None
        centers = (right,) if times[right] == timestamp else (right - 1, right)
        velocities = []
        for center in centers:
            if center < 1 or center + 1 >= len(times):
                return None
            before, current, after = self.samples[center - 1:center + 2]
            left_dt, right_dt = current[0] - before[0], after[0] - current[0]
            left = (current[1][:3, 3] - before[1][:3, 3]) / left_dt
            right_slope = (after[1][:3, 3] - current[1][:3, 3]) / right_dt
            velocities.append((right_dt * left + left_dt * right_slope) / (left_dt + right_dt))
        if len(centers) == 1:
            return self.samples[right][1].copy(), velocities[0]
        left = right - 1
        fraction = (timestamp - times[left]) / (times[right] - times[left])
        a, b = self.samples[left][1], self.samples[right][1]
        pose = np.eye(4)
        pose[:3, 3] = (1 - fraction) * a[:3, 3] + fraction * b[:3, 3]
        pose[:3, :3] = Slerp(
            [times[left], times[right]], Rotation.from_matrix([a[:3, :3], b[:3, :3]])
        )([timestamp]).as_matrix()[0]
        velocity = (1 - fraction) * velocities[0] + fraction * velocities[1]
        return pose, velocity
