"""Synchronized stationary IMU and tactile fixtures for backend tests."""
from types import SimpleNamespace
import numpy as np
from gelslam.core.inertial import ImuSample
from gelslam.core.numpy_vi import NavigationState

def config(**overrides):
    return {'imu_config': {
        'gelsight_T_imu': np.eye(4).tolist(), 'accelerometer_noise_density': .02,
        'gyroscope_noise_density': .002, 'accelerometer_bias_random_walk': .0002,
        'gyroscope_bias_random_walk': .00002,
    }, 'gelslam_config': {'pose_graph_config': {
        'local_window_optimization': True, 'local_window_size': 3,
        'marginalization_enabled': True, 'max_iterations': 10,
        'tactile_max_samples': 100, **overrides,
    }}}


def frame(index=0, z=0., new=False, group=0, height_m=0., tactile=True):
    yy, xx = np.mgrid[:40, :40]
    normal = np.zeros((40, 40, 3), dtype=np.float32)
    normal[:, :, 2] = 1.
    pose = np.eye(4)
    pose[2, 3] = z
    time = 1 + index * .1
    return SimpleNamespace(
        N=normal, H=np.full((40, 40), height_m / .001, dtype=np.float32),
        C=np.ones((40, 40), dtype=bool),
        L=np.maximum(400 - (xx - 20)**2 - (yy - 20)**2, 0).astype(float),
        ppmm=1., kid=index, fid=index, trial_T_curr=pose, trial_group=group,
        navigation_state=NavigationState(time, pose.copy(), np.zeros(3), np.zeros(3), np.zeros(3)),
        gravity=np.array([0., 0., -9.81]), tactile_valid=tactile,
        is_new_trial=lambda: new,
        imu_samples=tuple(ImuSample(float(t), [0., 0., 9.81], [0., 0., 0.])
                          for t in np.linspace(time - .1, time, 11)),
    )
