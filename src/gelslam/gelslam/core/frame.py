import numpy as np
from normalflow.utils import height2pointcloud

from gelslam.utils import pointcloud2mesh
from gelslam_msgs.msg import FrameMsg, KeyFrameMsg


def _copy_header(message, header):
    """Copy a ROS header when a source image header is available."""
    if header is not None:
        message.header.stamp = header.stamp
        message.header.frame_id = header.frame_id


def create_keyframe_msg(bridge, frame, kid, fid, is_new_trial, trial_T_curr,
                        header=None, navigation_state=None, gravity=None,
                        imu_samples=(), tactile_valid=True):
    """
    Creates a KeyFrameMsg from a Frame object.

    :param bridge: CvBridge; To convert images.
    :param frame: Frame; Object containing data.
    :param kid: int; Keyframe ID.
    :param fid: int; Frame ID.
    :param is_new_trial: bool; Flag if this is a new trial.
    :param trial_T_curr: np.ndarray; Current keyframe pose in trial coordinates.
    :param header: std_msgs.msg.Header; Source image header, if available.
    :return: KeyFrameMsg; The keyframe message.
    """
    if getattr(bridge, "internal_packets", False):
        from copy import deepcopy
        from gelslam.core.slam_packets import KeyFramePacket, readonly
        state = navigation_state
        navigation = {} if state is None else dict(
            navigation_valid=True, navigation_timestamp=state.timestamp,
            navigation_pose=readonly(state.pose).reshape(-1),
            velocity=readonly(state.velocity), accel_bias=readonly(state.accel_bias),
            gyro_bias=readonly(state.gyro_bias), gravity=readonly(gravity),
            imu_measurements=readonly([[s.timestamp, *s.acceleration, *s.angular_velocity]
                                       for s in imu_samples]).reshape(-1),
        )
        return KeyFramePacket(
            deepcopy(header), readonly(frame.H), readonly(frame.C.astype(np.uint8)),
            readonly(frame.N), readonly(frame.L), kid, fid, is_new_trial,
            readonly(trial_T_curr).reshape(-1), tactile_valid=tactile_valid, **navigation)
    keyframe_msg = KeyFrameMsg()
    _copy_header(keyframe_msg, header)
    keyframe_msg.height_map = bridge.cv2_to_imgmsg(frame.H, encoding="32FC1")
    keyframe_msg.contact_mask = bridge.cv2_to_imgmsg(
        frame.C.astype(np.uint8), encoding="8UC1"
    )
    keyframe_msg.normal_map = bridge.cv2_to_imgmsg(frame.N, encoding="32FC3")
    keyframe_msg.laplacian_map = bridge.cv2_to_imgmsg(frame.L, encoding="32FC1")
    keyframe_msg.kid = kid
    keyframe_msg.fid = fid
    keyframe_msg.new_trial_flag = is_new_trial
    keyframe_msg.trial_t_curr = (
        np.asarray(trial_T_curr, dtype=np.float32).flatten().tolist()
    )
    keyframe_msg.tactile_valid = bool(tactile_valid)
    if navigation_state is not None:
        keyframe_msg.navigation_valid = True
        keyframe_msg.navigation_timestamp = float(navigation_state.timestamp)
        keyframe_msg.navigation_pose = np.asarray(navigation_state.pose).ravel().tolist()
        keyframe_msg.velocity = np.asarray(navigation_state.velocity).tolist()
        keyframe_msg.accel_bias = np.asarray(navigation_state.accel_bias).tolist()
        keyframe_msg.gyro_bias = np.asarray(navigation_state.gyro_bias).tolist()
        keyframe_msg.gravity = np.asarray(gravity).tolist()
        keyframe_msg.imu_measurements = [
            float(value) for sample in imu_samples
            for value in (sample.timestamp, *sample.acceleration, *sample.angular_velocity)
        ]
    return keyframe_msg


def create_frame_msg(bridge, frame, fid, ref_kid, ref_T_curr, header=None, *,
                     tracking_status="contact_tracking", pose_valid=True, pose_timestamp=None,
                     published_T_ref_estimate=None):
    """
    Creates a FrameMsg from a Frame object.

    :param bridge: CvBridge; To convert images.
    :param frame: Frame; Object containing data.
    :param fid: int; Frame ID.
    :param ref_kid: int; Reference Keyframe ID.
    :param ref_T_curr: np.ndarray; Transformation from current to reference.
    :param header: std_msgs.msg.Header; Source image header, if available.
    :param published_T_ref_estimate: np.ndarray (4, 4) or None; the frontend's estimate of
        the reference keyframe relative to its published pose (None: the published pose).
    :return: FrameMsg; The frame message.
    """
    if pose_timestamp is None:
        pose_timestamp = (float(header.stamp.sec) + header.stamp.nanosec * 1e-9
                          if header is not None else 0.0)
    estimate = ([] if published_T_ref_estimate is None
                else np.asarray(published_T_ref_estimate, dtype=np.float64).reshape(-1))
    if getattr(bridge, "internal_packets", False):
        from copy import deepcopy
        from gelslam.core.slam_packets import FramePacket, readonly
        return FramePacket(
            deepcopy(header), readonly(frame.H), readonly(frame.C.astype(np.uint8)),
            readonly(frame.N), readonly(frame.L), fid, ref_kid, readonly(ref_T_curr).reshape(-1),
            tracking_status, bool(pose_valid), float(pose_timestamp), readonly(estimate))
    frame_msg = FrameMsg()
    frame_msg.published_t_ref_estimate = [float(value) for value in estimate]
    frame_msg.tracking_status = tracking_status
    frame_msg.pose_valid = bool(pose_valid)
    frame_msg.pose_timestamp = float(pose_timestamp)
    _copy_header(frame_msg, header)
    frame_msg.height_map = bridge.cv2_to_imgmsg(frame.H, encoding="32FC1")
    frame_msg.contact_mask = bridge.cv2_to_imgmsg(
        frame.C.astype(np.uint8), encoding="8UC1"
    )
    frame_msg.normal_map = bridge.cv2_to_imgmsg(frame.N, encoding="32FC3")
    frame_msg.laplacian_map = bridge.cv2_to_imgmsg(frame.L, encoding="32FC1")
    frame_msg.fid = fid
    frame_msg.ref_kid = ref_kid
    frame_msg.ref_t_curr = ref_T_curr.flatten().tolist()
    return frame_msg


def pose_of_frame_msg(frame_msg):
    """
    Extracts the pose of the frame message relative to its reference keyframe.

    :param frame_msg: FrameMsg; The frame message.
    :return: np.ndarray (4, 4); The pose matrix.
    """
    return np.array(frame_msg.ref_t_curr).reshape(4, 4).astype(np.float32)


def center_of_frame_msg(bridge, frame_msg, ppmm):
    """
    Compute the center of the frame.

    :param bridge: CvBridge; To convert images.
    :param frame_msg: FrameMsg; The frame message.
    :param ppmm: float; Pixels size in milimeters.
    :return: np.ndarray (3,); The center point in meters.
    """
    C = bridge.imgmsg_to_cv2(frame_msg.contact_mask) > 0
    H = bridge.imgmsg_to_cv2(frame_msg.height_map)
    xys = np.column_stack(np.where(C))
    cy, cx = np.mean(xys, axis=0)
    cz = H[int(cy), int(cx)]
    point = np.array([cx - C.shape[1] / 2.0, cy - C.shape[0] / 2.0, cz]) * ppmm / 1000
    return point


def frame_msg2mesh(bridge, frame_msg, start_T_ref, ppmm, raise_dist=0.0002):
    """
    Convert the frame message to a mesh.
    Transform the mesh to the correct position based on the transformation.

    :param bridge: CvBridge; To convert images.
    :param frame_msg: FrameMsg; The frame message.
    :param start_T_ref: np.ndarray; Transformation.
    :param ppmm: float; Pixels size in milimeters.
    :param raise_dist: float; Distance to raise the mesh.
    :return: open3d.geometry.TriangleMesh; The mesh.
    """
    start_T_frame = start_T_ref @ pose_of_frame_msg(frame_msg)
    C = bridge.imgmsg_to_cv2(frame_msg.contact_mask) > 0
    H = bridge.imgmsg_to_cv2(frame_msg.height_map)
    pointcloud = height2pointcloud(H, C, ppmm)
    if raise_dist != 0.0:
        pointcloud[:, 2] -= raise_dist
    pointcloud = np.dot(start_T_frame[:3, :3], pointcloud.T).T + start_T_frame[:3, 3]
    mesh = pointcloud2mesh(pointcloud, C)
    mesh.paint_uniform_color([1.0, 0.0, 0.0])
    return mesh
