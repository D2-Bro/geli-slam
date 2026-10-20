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
            'imu_measurements', 'navigation_pose')},
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
              'anchor_kid', 'solution_revision', 'navigation_pose') if keyframe else
             ('fid', 'ref_kid', 'ref_t_curr'))
    for name in names:
        value = getattr(packet, name)
        setattr(message, name, value.flatten().tolist() if isinstance(value, np.ndarray) else value)
    if keyframe:
        message.anchor_state = pack_navigation_state(packet.anchor_state)
    return message
