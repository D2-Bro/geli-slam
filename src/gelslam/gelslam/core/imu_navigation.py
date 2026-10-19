"""IMU navigation states and GelSight observation poses.

T_GI maps IMU coordinates into GelSight coordinates. Navigation poses are
T_WI; all tactile frame and keyframe poses remain T_WG.
"""

import numpy as np

from gelslam.core.numpy_vi import adjoint


def gelsight_pose(imu_pose, gelsight_T_imu):
    """Return T_WG = T_WI T_IG."""
    return np.asarray(imu_pose) @ np.linalg.inv(gelsight_T_imu)


def imu_pose(gel_pose, gelsight_T_imu):
    """Return T_WI = T_WG T_GI."""
    return np.asarray(gel_pose) @ gelsight_T_imu


def gelsight_pose_jacobian(gelsight_T_imu):
    """Map split IMU body increments to split GelSight body increments.

    The lower-left block retains displacement of the GelSight origin when the
    IMU rotates. This is the derivative at zero, also for the split retraction.
    """
    return adjoint(np.asarray(gelsight_T_imu))


def imu_velocity(gel_velocity, world_R_imu, omega_imu, gelsight_T_imu):
    """Convert GelSight-origin velocity to IMU-origin velocity in world axes."""
    imu_T_gel = np.linalg.inv(gelsight_T_imu)
    return (np.asarray(gel_velocity)
            - world_R_imu @ np.cross(omega_imu, imu_T_gel[:3, 3]))


def gelsight_velocity(state, omega_imu, gelsight_T_imu):
    """Convert IMU-origin velocity to GelSight-origin velocity in world axes."""
    imu_T_gel = np.linalg.inv(gelsight_T_imu)
    return (state.velocity
            + state.pose[:3, :3] @ np.cross(omega_imu, imu_T_gel[:3, 3]))
