"""ORB-SLAM3-style body-local pose increments and separate pose errors.

Used by the frontend (retained prior, reference tracking). The backend graph
uses native GTSAM Pose3 factors and gtsam.LevenbergMarquardtOptimizer.
"""

import gtsam
import numpy as np


def split_pose_plus(pose, increment):
    """Use the pre-update R for p += R dp, then R *= Exp(dtheta)."""
    pose = np.asarray(pose, dtype=np.float64)
    increment = np.asarray(increment, dtype=np.float64)
    result = pose.copy()
    result[:3, 3] = pose[:3, 3] + pose[:3, :3] @ increment[3:6]
    result[:3, :3] = pose[:3, :3] @ gtsam.Rot3.Expmap(increment[:3]).matrix()
    return result


def split_pose_coordinates(anchor, pose):
    """Return [Log(R0.T R), R0.T (p-p0)], with no SE(3) log coupling."""
    relative_rotation = anchor[:3, :3].T @ pose[:3, :3]
    return np.concatenate((
        gtsam.Rot3.Logmap(gtsam.Rot3(relative_rotation)),
        anchor[:3, :3].T @ (pose[:3, 3] - anchor[:3, 3]),
    ))


def split_pose_coordinate_jacobian(anchor, pose):
    """Derivative of split coordinates under a split body-local increment."""
    coordinates = split_pose_coordinates(anchor, pose)
    J = np.zeros((6, 6))
    J[:3, :3] = gtsam.Rot3.LogmapDerivative(coordinates[:3])
    J[3:, 3:] = anchor[:3, :3].T @ pose[:3, :3]
    return J
