"""Global loops jointly constrain full poses, velocities and sensor biases."""
import pickle

import gtsam
import numpy as np
import pytest

from backend_test_support import config, frame
from gelslam.core.backend_factors import bias_key, velocity_key
from gelslam.core.gtsam_pose_graph import PoseGraph


def graph_with_loop(extrinsic=None):
    settings = config(max_iterations=30)
    if extrinsic is not None:
        settings['imu_config']['gelsight_T_imu'] = extrinsic.tolist()
    graph = PoseGraph(settings)
    frames = [frame(i, new=i == 0, z=.001 * i) for i in range(4)]
    for i, item in enumerate(frames):
        item.trial_T_curr[:3, :3] = gtsam.Rot3.Expmap(np.array([.01*i, -.008*i, .003*i])).matrix()
        item.navigation_state.pose = item.trial_T_curr @ graph.gelsight_T_imu
        item.navigation_state.velocity[:] = [.01*i, -.02*i, .03*i]
        item.navigation_state.accel_bias[:] = [.01*i, -.02*i, .005*i]
        item.navigation_state.gyro_bias[:] = [.001*i, -.002*i, .0005*i]
    graph.add_keyframes(frames, 0, len(frames))
    assert graph._add_loop_pair(0, 3, np.eye(4))
    return graph


def test_global_loop_has_15d_values_and_updates_roll_pitch_velocity_and_bias():
    graph = graph_with_loop()
    before = [state.copy() for state in graph.states]
    result = graph.solve(global_optimization=True)
    loop = graph.loop_pgo
    assert loop.degrees_of_freedom == 15
    assert loop.graph_result.size() == 3 * len(before)
    assert loop.graph_init.dim() == 15 * len(before)
    assert loop.last_final_cost < loop.last_initial_cost
    assert loop.factor_kinds.count('imu') == 3
    assert loop.factor_kinds.count('bias') == 3
    assert loop.factor_kinds.count('odometry') == 3
    assert loop.factor_kinds.count('loop') == 1
    corrected = graph.map_state(3)
    rotation = gtsam.Rot3.Logmap(gtsam.Rot3(loop.corrections[3][:3, :3]))
    assert np.linalg.norm(rotation[:2]) > 1e-4
    assert np.linalg.norm(corrected.velocity - before[3].velocity) > .001
    assert np.linalg.norm(corrected.accel_bias - before[3].accel_bias) > .001
    assert np.linalg.norm(corrected.gyro_bias - before[3].gyro_bias) > .0001
    np.testing.assert_allclose(corrected.pose, loop.graph_result.atPose3(3).matrix())
    np.testing.assert_allclose(corrected.velocity, loop.graph_result.atVector(velocity_key(3)))
    np.testing.assert_allclose(np.r_[corrected.accel_bias, corrected.gyro_bias],
                               loop.graph_result.atConstantBias(bias_key(3)).vector())
    np.testing.assert_allclose(result[3], corrected.pose @ graph.imu_T_gelsight, atol=1e-8)
    for actual, original in zip(graph.states, before):
        for field in ('pose', 'velocity', 'accel_bias', 'gyro_bias'):
            np.testing.assert_array_equal(getattr(actual, field), getattr(original, field))


def test_global_imu_recovery_has_motion_and_bias_without_pose_odometry():
    graph = PoseGraph(config())
    graph.add_keyframes([frame(new=True), frame(1, tactile=False), frame(2)], 0, 3)
    graph._add_loop_pair(0, 2, np.eye(4))
    graph.solve(global_optimization=True)
    assert graph.loop_pgo.factor_kinds.count('imu') == 2
    assert graph.loop_pgo.factor_kinds.count('bias') == 2
    assert graph.loop_pgo.factor_kinds.count('odometry') == 1
    factors = [graph.loop_pgo.graph.at(i) for i, kind in
               enumerate(graph.loop_pgo.factor_kinds) if kind == 'odometry']
    assert list(factors[0].keys()) == [1, 2]


def test_recovery_odometry_links_the_imu_recovery_keyframe_when_configured():
    graph = PoseGraph(config(recovery_odometry_translation_sigma_m=.001))
    graph.add_keyframes([frame(new=True), frame(1, tactile=False), frame(2)], 0, 3)
    graph._add_loop_pair(0, 2, np.eye(4))
    graph.solve(global_optimization=True)
    kinds = graph.loop_pgo.factor_kinds
    assert kinds.count('odometry') == 1 and kinds.count('recovery_odometry') == 1
    factor = graph.loop_pgo.graph.at(kinds.index('recovery_odometry'))
    assert list(factor.keys()) == [0, 1]
    np.testing.assert_allclose(graph.loop_pgo.recovery_noise.sigmas(), [.01] * 3 + [.001] * 3)


def test_global_navigation_save_reload_and_new_keyframe_keep_all_corrections(tmp_path):
    graph = graph_with_loop()
    graph.solve(global_optimization=True)
    expected = [graph.map_state(i) for i in range(4)]
    graph.save(tmp_path)
    loaded = PoseGraph.load(tmp_path)
    for key, original in enumerate(expected):
        for field in ('pose', 'velocity', 'accel_bias', 'gyro_bias'):
            np.testing.assert_allclose(getattr(loaded.map_state(key), field), getattr(original, field), atol=1e-12)
    loaded.add_keyframes([None]*4 + [frame(4, tactile=False)], 4, 5)
    loaded.raw_solutions()
    loop = loaded.loop_pgo
    np.testing.assert_array_equal(loop.velocity_corrections[4], loop.velocity_corrections[3])
    np.testing.assert_array_equal(loop.bias_corrections[4], loop.bias_corrections[3])
    loaded.solve(global_optimization=True)
    assert loaded.loop_pgo.graph_result.dim() == 75
    assert np.isfinite(loaded.loop_pgo.last_final_cost)


def test_old_pose_correction_snapshot_migrates_without_inventing_navigation_updates(tmp_path):
    graph = graph_with_loop()
    graph.solve(global_optimization=True)
    graph.save(tmp_path)
    path = tmp_path / 'pose_graph.pkl'
    with path.open('rb') as handle:
        saved = pickle.load(handle)
    saved['format_version'] = 3
    snapshot = saved['loop_pgo']
    snapshot['degrees_of_freedom'] = 4
    for key in ('velocity_corrections', 'bias_corrections', 'anchor_states'):
        snapshot.pop(key)
    with path.open('wb') as handle:
        pickle.dump(saved, handle)
    loaded = PoseGraph.load(tmp_path)
    np.testing.assert_allclose(loaded.map_poses(), graph.map_poses(), atol=1e-12)
    assert not np.any(loaded.loop_pgo.velocity_corrections)
    assert not np.any(loaded.loop_pgo.bias_corrections)
    loaded.solve(global_optimization=True)
    assert loaded.loop_pgo.degrees_of_freedom == 15
    assert loaded.loop_pgo.graph_result.dim() == 60


def test_global_loop_uses_gelsight_relative_measurement_with_nonzero_imu_mount():
    settings = config()
    extrinsic = np.eye(4)
    extrinsic[:3, 3] = [.02, -.01, .04]
    settings['imu_config']['gelsight_T_imu'] = extrinsic.tolist()
    graph = PoseGraph(settings)
    first, second = frame(new=True), frame(1, new=True, group=1)
    first.navigation_state.pose = extrinsic.copy()
    second.navigation_state.pose = extrinsic.copy()
    graph.add_keyframes([first, second], 0, 2)
    desired = np.eye(4)
    desired[:3, :3] = gtsam.Rot3.Expmap(np.array([.1, -.15, .2])).matrix()
    desired[:3, 3] = [.003, -.002, .001]
    graph._add_loop_pair(0, 1, np.linalg.inv(desired))
    graph.remove_prior_factors([1])
    graph.solve(global_optimization=True)
    np.testing.assert_allclose(graph.map_poses()[1], desired, atol=1e-10)
    np.testing.assert_allclose(graph.map_state(1).pose, desired @ extrinsic, atol=1e-10)
    trial_frame = graph.map_T_trial_group(1)
    np.testing.assert_allclose(trial_frame[:3, :3] @ np.array([0., 0., -9.81]),
                               [0., 0., -9.81], atol=1e-12)
    # Trial-frame feedback must retain the estimated roll/pitch correction.
    feedback = np.linalg.inv(trial_frame) @ graph.map_state(1).pose
    tilt = gtsam.Rot3.Logmap(gtsam.Rot3(feedback[:3, :3]))
    assert np.linalg.norm(tilt[:2]) > .05


def test_global_solver_rejects_missing_imu_instead_of_filling_velocity_and_bias():
    graph = graph_with_loop()
    graph.keyframes[2].imu_samples = ()
    with pytest.raises(ValueError, match='global PGO IMU interval'):
        graph.solve(global_optimization=True)


def _gap_imu_factor(settings):
    graph = PoseGraph(settings)
    graph.add_keyframes([frame(new=True), frame(1, tactile=False), frame(2)], 0, 3)
    graph._add_loop_pair(0, 2, np.eye(4))
    graph.solve(global_optimization=True)
    loop = graph.loop_pgo
    factors = [loop.graph.at(i) for i, kind in enumerate(loop.factor_kinds) if kind in ('imu', 'bias')]
    by_target = {}
    for factor in factors:
        keys = [k if k < 3 else gtsam.Symbol(k).index() for k in factor.keys()]
        by_target.setdefault(max(keys), []).append(factor)
    return loop, by_target


def test_loop_noise_uses_odometry_sigmas_with_the_robust_kernel():
    graph = PoseGraph(config(loop_pgo_loop_robust='huber'))
    loop = graph.loop_pgo
    assert loop.loop_noise.isRobust() if hasattr(loop.loop_noise, 'isRobust') else 'Robust' in type(loop.loop_noise).__name__
    np.testing.assert_allclose(loop.noise.sigmas(), np.repeat([.01, .0001], 3))
    with pytest.raises(ValueError, match='loop_pgo_loop_robust'):
        PoseGraph(config(loop_pgo_loop_robust='tukey'))


def test_backend_only_integration_sigma_does_not_touch_the_frontend_config():
    settings = config(imu_integration_sigma_m=.01)
    graph = PoseGraph(settings)
    assert graph.loop_pgo.imu_config['gtsam_integration_sigma_m'] == .01
    assert 'gtsam_integration_sigma_m' not in settings['imu_config']
