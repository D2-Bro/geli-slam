"""IMU graph variables with GelSight observations, frames and saved solutions."""

from backend_test_support import config as backend_config, frame as backend_frame

from types import SimpleNamespace

import gtsam
import numpy as np
import pytest
import pickle

from gelslam.core.gtsam_pose_graph import PoseGraph
from gelslam.core.gtsam_vi import GtsamVisualInertialSolver
from gelslam.core.imu_navigation import (
    gelsight_pose, gelsight_pose_jacobian, gelsight_velocity, imu_pose, imu_velocity,
)
from gelslam.core.inertial import ImuSample
from gelslam.core.numpy_vi import NavigationState, so3_exp
from gelslam.core.split_pose import (
    split_between_factor, split_pose_coordinates, split_pose_plus, split_prior_factor,
)
from gelslam.core.tracker import Tracker


def mount():
    pose = np.eye(4)
    pose[:3, :3] = so3_exp(np.array([.4, -.3, .2]))
    pose[:3, 3] = [.04, -.02, .03]
    return pose


def test_angular_acceleration_integrates_imu_origin_and_moves_gelsight_origin():
    extrinsic = mount()
    initial = np.eye(4)
    initial[:3, 3] = extrinsic[:3, :3].T @ extrinsic[:3, 3]
    previous = NavigationState(0., initial, np.zeros(3), np.zeros(3), np.zeros(3))
    assert np.allclose(gelsight_pose(previous.pose, extrinsic)[:3, 3], 0.)
    alpha, end = 4., .4
    samples = [ImuSample(float(t), [0, 0, 9.81], [0, 0, alpha * t])
               for t in np.linspace(0., end, 81)]
    current, _ = GtsamVisualInertialSolver({}).predict_current(previous, samples, extrinsic)
    expected = initial.copy()
    expected[:3, :3] = so3_exp(np.array([0., 0., .5 * alpha * end**2]))
    assert np.allclose(current.pose, expected, atol=1e-12)
    assert np.allclose(current.velocity, 0., atol=1e-12)
    gel = gelsight_pose(current.pose, extrinsic)
    assert np.allclose(gel, expected @ np.linalg.inv(extrinsic))
    assert np.linalg.norm(gel[:3, 3]) > .005
    gel_v = gelsight_velocity(current, np.array([0., 0., alpha * end]), extrinsic)
    assert np.linalg.norm(gel_v) > .05
    assert np.allclose(imu_velocity(gel_v, current.pose[:3, :3],
                                   np.array([0., 0., alpha * end]), extrinsic), 0.)


def test_gelsight_pose_chain_jacobian_includes_rotation_offset_coupling():
    extrinsic = mount()
    body = np.eye(4)
    body[:3, :3] = so3_exp(np.array([-.2, .3, .1]))
    body[:3, 3] = [.3, -.1, .2]
    gel = gelsight_pose(body, extrinsic)
    numerical = np.empty((6, 6))
    for k in range(6):
        step = np.eye(6)[k] * 1e-7
        plus = gelsight_pose(split_pose_plus(body, step), extrinsic)
        minus = gelsight_pose(split_pose_plus(body, -step), extrinsic)
        numerical[:, k] = (split_pose_coordinates(gel, plus)
                           - split_pose_coordinates(gel, minus)) / 2e-7
    assert np.allclose(numerical, gelsight_pose_jacobian(extrinsic), atol=1e-8)
    assert np.linalg.norm(numerical[3:, :3]) > .05


class PositionObservation:
    def __init__(self, target):
        self.target = target
        self.seen = []

    def __len__(self):
        return 10

    def evaluate(self, reference, target):
        self.seen.append(target.copy())
        residual = np.tile(target[:3, 3] - self.target, 10)
        J = np.zeros((30, 6))
        J[:, 3:] = np.tile(target[:3, :3], (10, 1))
        return residual, np.zeros_like(J), J


def test_frontend_observation_receives_gelsight_pose_and_retains_imu_prior():
    extrinsic = mount()
    previous = NavigationState(0., extrinsic.copy(), np.zeros(3), np.zeros(3), np.zeros(3))
    force = extrinsic[:3, :3].T @ [0., 0., 9.81]
    samples = [ImuSample(float(t), force, [0., 0., 0.]) for t in np.linspace(0., .1, 11)]
    solver = GtsamVisualInertialSolver({"gtsam_vi_max_iterations": 3,
                                      "dense_normalflow_residual_weight": 1e4})
    current, pim = solver.predict_current(previous, samples, extrinsic)
    visual = PositionObservation(np.array([.001, -.002, .003]))
    refined = solver.refine_current(visual, np.eye(4), previous, current,
                                    samples, extrinsic, preintegrated=pim)
    assert refined is not None
    assert np.allclose(visual.seen[0], np.eye(4), atol=1e-12)
    assert np.linalg.norm(gelsight_pose(refined.pose, extrinsic)[:3, 3] - visual.target) < .003
    assert solver.pending_marginal_prior.matches(refined)
    assert np.allclose(solver.pending_marginal_prior.anchor.pose, refined.pose)
    assert not np.allclose(refined.pose, gelsight_pose(refined.pose, extrinsic))


@pytest.mark.parametrize("between", [False, True])
def test_backend_gelsight_factor_jacobians_use_imu_increments(between):
    extrinsic = mount()
    a = np.eye(4)
    b = np.eye(4)
    b[:3, :3] = so3_exp(np.array([-.1, .3, .2]))
    b[:3, 3] = [.1, -.03, .02]
    values = gtsam.Values()
    values.insert(0, gtsam.Pose3(imu_pose(a, extrinsic)))
    values.insert(1, gtsam.Pose3(imu_pose(b, extrinsic)))
    noise = gtsam.noiseModel.Unit.Create(6)
    offset = np.linalg.inv(extrinsic)
    factor = (split_between_factor(0, 1, np.eye(4), noise, offset) if between
              else split_prior_factor(1, np.eye(4), noise, offset))
    A, _ = factor.linearize(values).jacobian()
    numerical = np.empty_like(A)
    for column in range(A.shape[1]):
        key = list(factor.keys())[column // 6]
        pose = values.atPose3(key).matrix()
        step = np.eye(6)[column % 6] * 1e-7
        plus, minus = gtsam.Values(values), gtsam.Values(values)
        plus.update(key, gtsam.Pose3(split_pose_plus(pose, step)))
        minus.update(key, gtsam.Pose3(split_pose_plus(pose, -step)))
        numerical[:, column] = (factor.unwhitenedError(plus)
                                - factor.unwhitenedError(minus)) / 2e-7
    assert np.allclose(A, numerical, atol=1e-8)


def test_backend_states_are_imu_and_output_and_reload_remain_gelsight(tmp_path):
    extrinsic = mount()
    config = backend_config()
    config['imu_config']['gelsight_T_imu'] = extrinsic.tolist()
    graph = PoseGraph(config)
    frames = [backend_frame(0, new=True), backend_frame(1)]
    for keyframe in frames:
        keyframe.navigation_state.pose = extrinsic.copy()
        keyframe.imu_samples = tuple(ImuSample(s.timestamp, extrinsic[:3, :3].T @ [0., 0., 9.81],
                                               [0., 0., 0.]) for s in keyframe.imu_samples)
    graph.add_keyframes(frames, 0, 2)
    assert np.allclose(graph.states[1].pose, extrinsic)
    solution = graph.solve(global_optimization=True)
    assert np.allclose(solution[1], np.eye(4), atol=1e-7)
    assert np.allclose(graph.map_state(1).pose, extrinsic, atol=1e-7)
    graph.save(tmp_path)
    loaded = PoseGraph.load(tmp_path)
    assert np.allclose(loaded.raw_solutions()[1], np.eye(4), atol=1e-7)

def test_old_active_gelsight_tracker_state_is_not_loaded_as_imu(tmp_path):
    tracker = object.__new__(Tracker)
    tracker._frontend_vi_state = NavigationState(1., np.eye(4), np.zeros(3), np.zeros(3), np.zeros(3))
    with (tmp_path / 'tracker.pkl').open('wb') as stream:
        pickle.dump(tracker, stream)
    with pytest.raises(ValueError, match='GelSight navigation states'):
        Tracker.load(tmp_path, 'unused', SimpleNamespace())

