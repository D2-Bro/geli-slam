"""In-process observations. ROS images are created only at publication."""
from dataclasses import dataclass, field
from copy import deepcopy
import numpy as np


def readonly(value):
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class FramePacket:
    header: object
    height_map: np.ndarray
    contact_mask: np.ndarray
    normal_map: np.ndarray
    laplacian_map: np.ndarray
    fid: int
    ref_kid: int
    ref_t_curr: np.ndarray
    tracking_status: str = "contact_tracking"
    pose_valid: bool = True
    pose_timestamp: float = 0.0
    # Frontend estimate of ref_kid at publication, inv(published) @ estimate (FrameMsg).
    published_t_ref_estimate: np.ndarray = field(default_factory=lambda: readonly([]))


@dataclass(frozen=True)
class KeyFramePacket:
    header: object
    height_map: np.ndarray
    contact_mask: np.ndarray
    normal_map: np.ndarray
    laplacian_map: np.ndarray
    kid: int
    fid: int
    new_trial_flag: bool
    trial_t_curr: np.ndarray
    navigation_valid: bool = False
    navigation_timestamp: float = 0.
    velocity: np.ndarray = field(default_factory=lambda: readonly(np.zeros(3)))
    accel_bias: np.ndarray = field(default_factory=lambda: readonly(np.zeros(3)))
    gyro_bias: np.ndarray = field(default_factory=lambda: readonly(np.zeros(3)))
    gravity: np.ndarray = field(default_factory=lambda: readonly(np.zeros(3)))
    imu_measurements: np.ndarray = field(default_factory=lambda: readonly([]))
    tactile_valid: bool = True
    trial_id: int = -1
    anchor_kid: int = -1
    anchor_state: object = None
    solution_revision: int = 0
    session_id: str = ''
    navigation_pose: np.ndarray = field(default_factory=lambda: readonly([]))
    # Frontend re-estimate of anchor_kid after its publication (keyframe_update_row).
    keyframe_update: np.ndarray = field(default_factory=lambda: readonly([]))


KEYFRAME_UPDATE_SIZE = 27


def keyframe_update_row(kid, basis, estimate):
    """Encode the change from ``basis`` to ``estimate`` of keyframe ``kid``.

    Both states describe the same keyframe time in the frontend's current axes.
    The pose change is right-multiplied (basis body frame) and the velocity change
    is in basis body axes, so the receiver can apply it to its own stored seed of
    the keyframe without the frontend's world axes. Return [] for no change.
    """
    if basis is None or estimate is None or abs(estimate.timestamp - basis.timestamp) > 1e-9:
        return []
    basis_pose, pose = np.asarray(basis.pose, float), np.asarray(estimate.pose, float)
    delta = np.linalg.solve(basis_pose, pose)
    delta[3] = (0., 0., 0., 1.)
    velocity = basis_pose[:3, :3].T @ (np.asarray(estimate.velocity) - np.asarray(basis.velocity))
    accel_bias = np.asarray(estimate.accel_bias) - np.asarray(basis.accel_bias)
    gyro_bias = np.asarray(estimate.gyro_bias) - np.asarray(basis.gyro_bias)
    change = np.r_[delta[:3].ravel() - np.eye(4)[:3].ravel(), velocity, accel_bias, gyro_bias]
    if not np.all(np.isfinite(change)) or np.max(np.abs(change)) <= 1e-12:
        return []
    return np.r_[kid, basis.timestamp, delta.ravel(), velocity, accel_bias, gyro_bias].tolist()


def apply_keyframe_update(state, row):
    """Apply a keyframe_update_row to a stored NavigationState of the same keyframe."""
    row = np.asarray(row, dtype=float)
    if row.shape != (KEYFRAME_UPDATE_SIZE,) or not np.isfinite(row).all():
        raise ValueError('keyframe update must contain %d finite values' % KEYFRAME_UPDATE_SIZE)
    delta = row[2:18].reshape(4, 4)
    if (not np.allclose(delta[3], [0., 0., 0., 1.], rtol=0, atol=1e-12)
            or not np.allclose(delta[:3, :3].T @ delta[:3, :3], np.eye(3), rtol=0, atol=1e-8)):
        raise ValueError('keyframe update pose change must be rigid')
    updated = state.copy()
    updated.pose = np.asarray(state.pose, float) @ delta
    updated.velocity = np.asarray(state.velocity, float) + np.asarray(state.pose, float)[:3, :3] @ row[18:21]
    updated.accel_bias = np.asarray(state.accel_bias, float) + row[21:24]
    updated.gyro_bias = np.asarray(state.gyro_bias, float) + row[24:27]
    return updated


def pack_navigation_state(state):
    if state is None:
        return []
    return np.r_[state.timestamp, state.pose.ravel(), state.velocity,
                 state.accel_bias, state.gyro_bias].tolist()


def unpack_navigation_state(values):
    from gelslam.core.numpy_vi import NavigationState
    row = np.asarray(values, dtype=float)
    if row.shape != (26,) or not np.isfinite(row).all():
        raise ValueError('navigation state must contain 26 finite values')
    pose = row[1:17].reshape(4, 4).copy()
    if not np.allclose(pose[3], [0., 0., 0., 1.], rtol=0, atol=1e-10):
        raise ValueError('invalid navigation pose last row')
    return NavigationState(float(row[0]), pose, row[17:20].copy(),
                           row[20:23].copy(), row[23:26].copy())


def keyframe_from_ros(message, bridge):
    """Copy DDS-owned arrays once into an immutable backend observation."""
    from gelslam.core.shared_map import frozen_state
    return KeyFramePacket(
        header=deepcopy(message.header),
        **{name: readonly(bridge.imgmsg_to_cv2(getattr(message, name)))
           for name in ('height_map', 'contact_mask', 'normal_map', 'laplacian_map')},
        **{name: getattr(message, name) for name in
           ('kid', 'fid', 'new_trial_flag', 'navigation_valid', 'navigation_timestamp',
            'tactile_valid', 'trial_id', 'anchor_kid', 'solution_revision', 'session_id')},
        **{name: readonly(getattr(message, name)) for name in
           ('trial_t_curr', 'velocity', 'accel_bias', 'gyro_bias', 'gravity',
            'imu_measurements', 'navigation_pose', 'keyframe_update')},
        anchor_state=(frozen_state(unpack_navigation_state(message.anchor_state))
                      if len(message.anchor_state) else None))


class ArrayBridge:
    """Small compatibility adapter; never serializes internal observations."""
    internal_packets = True

    def cv2_to_imgmsg(self, image, encoding=None):
        return readonly(image)

    def imgmsg_to_cv2(self, image, *args, **kwargs):
        return image


def to_ros(packet, bridge):
    from gelslam_msgs.msg import FrameMsg, KeyFrameMsg
    keyframe = isinstance(packet, KeyFramePacket)
    message = KeyFrameMsg() if keyframe else FrameMsg()
    message.header = deepcopy(packet.header)
    for name, encoding in (("height_map", "32FC1"), ("contact_mask", "8UC1"),
                           ("normal_map", "32FC3"), ("laplacian_map", "32FC1")):
        message_image = bridge.cv2_to_imgmsg(getattr(packet, name), encoding=encoding)
        setattr(message, name, message_image)
    names = (('kid', 'fid', 'new_trial_flag', 'trial_t_curr', 'navigation_valid',
              'navigation_timestamp', 'velocity', 'accel_bias', 'gyro_bias',
              'gravity', 'imu_measurements', 'tactile_valid', 'session_id', 'trial_id',
              'anchor_kid', 'solution_revision', 'navigation_pose', 'keyframe_update') if keyframe else
             ('fid', 'ref_kid', 'ref_t_curr', 'tracking_status', 'pose_valid', 'pose_timestamp',
              'published_t_ref_estimate'))
    for name in names:
        value = getattr(packet, name)
        setattr(message, name, value.flatten().tolist() if isinstance(value, np.ndarray) else value)
    if keyframe:
        message.anchor_state = pack_navigation_state(packet.anchor_state)
    return message
