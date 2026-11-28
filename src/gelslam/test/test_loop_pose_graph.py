"""Loop measurements affect map poses, never the stored keyframe seeds."""
import pickle
import numpy as np
import pytest

from gelslam.core.gtsam_pose_graph import PoseGraph
from gelslam.core.loop_pose_graph import LoopPoseGraph
from backend_test_support import config, frame, loop_records


def translation(x):
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


def test_navigation_loop_distributes_closure_error_without_changing_local_odometry():
    graph = LoopPoseGraph({'max_iterations': 20})
    local = [translation(x) for x in (0., .01, .02)]
    records = loop_records(local)
    graph.synchronize(local, records, [(0, 1), (1, 2)])
    assert graph.add_loop(0, 2, translation(-.018))
    graph.solve()
    assert graph.factor_kinds == ['pose_anchor', 'velocity_prior', 'bias_prior',
                                  'imu', 'bias', 'odometry', 'imu', 'bias', 'odometry', 'loop']
    positions = [p[0, 3] for p in graph.map_poses()]
    assert positions[0] == 0. and 0. < positions[1] < .01 and .018 < positions[2] < .02
    assert np.array_equal(graph.local_poses, local)
    assert graph.last_final_cost < graph.last_initial_cost
    # The registration is a fixed metric measurement, not reprojected maps.
    assert np.allclose(graph.loops[0][2], translation(.018))
    assert np.allclose(graph.noise.sigmas(), [.01] * 3 + [.0001] * 3)


def test_disconnected_trial_loop_uses_registration_and_preserves_local_seeds(tmp_path):
    graph = PoseGraph(config())
    first, second = frame(new=True), frame(1, new=True, group=1)
    second.navigation_state.pose = translation(.15)
    second.trial_T_curr = translation(.15)
    graph.add_keyframes([first, second], 0, 2)
    local = [state.copy() for state in graph.states]
    assert graph._add_loop_pair(0, 1, translation(-.002))
    graph.remove_prior_factors([1])
    solutions = graph.solve(global_optimization=True)
    assert np.allclose(solutions[1], translation(.002), atol=1e-10)
    assert graph.loop_pgo.factor_kinds == ['pose_anchor', 'velocity_prior', 'bias_prior',
                                         'velocity_prior', 'bias_prior', 'loop']
    for actual, expected in zip(graph.states, local):
        assert np.allclose(actual.pose, expected.pose, atol=1e-12)
        assert np.allclose(actual.velocity, expected.velocity, atol=1e-12)
    graph.save(tmp_path)
    restored = PoseGraph.load(tmp_path)
    assert np.allclose(restored.map_poses(), graph.map_poses())
    assert np.allclose(restored.loop_pgo.loops[0][2], translation(.002))
    restored.solve()
    assert np.allclose(restored.map_poses()[1], translation(.002), atol=1e-10)


def test_map_state_rotates_world_velocity_and_keeps_sensor_biases():
    graph = PoseGraph(config())
    graph.add_keyframes([frame(new=True)], 0, 1)
    graph.raw_solutions()
    local = graph.states[0]
    local.velocity[:] = [1, 2, 3]
    local.accel_bias[:] = [.1, .2, .3]
    local.gyro_bias[:] = [.01, .02, .03]
    correction = translation(.04)
    correction[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    graph.loop_pgo.corrections[0] = correction
    published = graph.map_state(0)
    assert np.allclose(published.pose, correction @ local.pose)
    assert np.array_equal(published.velocity, [-2, 1, 3])
    assert np.array_equal(published.accel_bias, local.accel_bias)
    assert np.array_equal(published.gyro_bias, local.gyro_bias)
    assert published.timestamp == local.timestamp
    assert np.array_equal(graph.map_T_trial_group(0), correction)
    assert np.array_equal(local.pose, np.eye(4))


def test_direct_solve_and_reload_keep_keyframe_seeds(tmp_path):
    graph = PoseGraph(config())
    frames = [frame(i, new=i == 0, z=.001 * i) for i in range(3)]
    frames[1].navigation_state.velocity[:] = [.1, .2, .3]
    frames[1].navigation_state.accel_bias[:] = [.01, .02, .03]
    frames[1].navigation_state.gyro_bias[:] = [.001, .002, .003]
    graph.add_keyframes(frames, 0, len(frames))
    seeds = [state.copy() for state in graph.states]
    graph.solve()
    for actual, expected in zip(graph.states, seeds):
        for field in ('pose', 'velocity', 'accel_bias', 'gyro_bias'):
            np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    measured = np.eye(4)
    measured[2, 3] = -.001
    assert graph._add_loop_pair(0, 2, measured)
    graph.solve(global_optimization=True)
    graph.save(tmp_path)
    restored = PoseGraph.load(tmp_path)
    assert restored.temporal_links == graph.temporal_links
    np.testing.assert_allclose(restored.map_poses(), graph.map_poses(), atol=1e-12)
    for actual, expected in zip(restored.states, graph.states):
        for field in ('pose', 'velocity', 'accel_bias', 'gyro_bias'):
            np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    restored.solve(global_optimization=True)
    assert restored.loop_pgo.last_final_cost <= restored.loop_pgo.last_initial_cost


@pytest.mark.parametrize('invalid', [np.full((4, 4), np.nan), np.zeros((3, 3)),
                                     np.diag([2., 1., 1., 1.]), np.diag([-1., 1., 1., 1.])])
def test_invalid_registration_cannot_modify_either_graph(invalid):
    graph = PoseGraph(config())
    graph.add_keyframes([frame(new=True), frame(1)], 0, 2)
    assert not graph._add_loop_pair(0, 1, invalid)
    assert not graph.loop_pgo.loops
    assert all(kind != 'loop' for _, _, _, kind in graph.edges)


def test_legacy_direct_loop_snapshot_requires_replay(tmp_path):
    graph = PoseGraph(config())
    graph.add_keyframes([frame(new=True), frame(1)], 0, 2)
    graph.save(tmp_path)
    path = tmp_path / 'pose_graph.pkl'
    with path.open('rb') as handle:
        data = pickle.load(handle)
    data['format_version'] = 2
    data.pop('loop_pgo')
    with path.open('wb') as handle:
        pickle.dump(data, handle)
    assert len(PoseGraph.load(tmp_path).states) == 2
    data['edges'].append((0, 1, None, 'loop'))
    with path.open('wb') as handle:
        pickle.dump(data, handle)
    with pytest.raises(ValueError, match='lacks registered loop measurements'):
        PoseGraph.load(tmp_path)


def _pose_only(**overrides):
    return LoopPoseGraph(dict({'max_iterations': 20, 'imu_factors': False,
                               'recovery_odometry_translation_sigma_m': .001}, **overrides))


def test_pose_only_graph_has_no_imu_velocity_or_bias_factor():
    graph = _pose_only()
    local = [translation(x) for x in (0., .01, .02)]
    records = loop_records(local)
    graph.synchronize(local, records, [(0, 1), (1, 2)])
    assert graph.add_loop(0, 2, translation(-.018))
    graph.solve()
    assert graph.factor_kinds == ['pose_anchor', 'odometry', 'odometry', 'loop']
    assert graph.degrees_of_freedom == 6 and graph.graph_result.dim() == 18
    positions = [p[0, 3] for p in graph.map_poses()]
    assert positions[0] == 0. and 0. < positions[1] < .01 and .018 < positions[2] < .02
    assert np.array_equal(graph.local_poses, local)
    assert graph.last_final_cost < graph.last_initial_cost
    # No velocity or bias estimate: the offsets stay zero, the seeds are kept.
    assert all(not v.any() for v in graph.velocity_corrections + graph.bias_corrections)
    state = graph.map_state(2)
    np.testing.assert_array_equal(state.accel_bias, graph.local_states[2].accel_bias)
    # A second solve from the corrected poses stays at the solution.
    before = graph.map_poses()
    graph.solve()
    np.testing.assert_allclose(graph.map_poses(), before, atol=1e-9)


def test_pose_only_graph_links_an_imu_recovery_keyframe_by_recovery_odometry():
    graph = _pose_only()
    local = [translation(x) for x in (0., .01, .02)]
    records = loop_records(local)
    records[1].tactile_valid = False
    # A missing IMU interval is not an error without IMU factors.
    records[1].imu_samples = ()
    graph.synchronize(local, records, [(0, 1), (1, 2)])
    graph.solve()
    assert graph.factor_kinds == ['pose_anchor', 'recovery_odometry', 'odometry']
    np.testing.assert_allclose(graph.map_poses(), local, atol=1e-9)


def test_pose_only_graph_settings_are_validated():
    with pytest.raises(ValueError, match='requires recovery_odometry_translation_sigma_m'):
        LoopPoseGraph({'imu_factors': False})
    with pytest.raises(ValueError, match='imu_factors must be true or false'):
        LoopPoseGraph({'imu_factors': 0})
    assert LoopPoseGraph({}).imu_factors and LoopPoseGraph({}).degrees_of_freedom == 15


def test_loop_rotation_sigma_is_separate_from_the_odometry_sigma():
    graph = LoopPoseGraph({'loop_rotation_sigma_rad': .05, 'loop_pgo_loop_robust': 'huber'})
    assert np.allclose(graph.noise.sigmas(), [.01] * 3 + [.0001] * 3)
    assert np.allclose(graph.loop_noise.noise().sigmas(), [.05] * 3 + [.0001] * 3)
    plain = LoopPoseGraph({'loop_rotation_sigma_rad': .05})
    assert np.allclose(plain.loop_noise.sigmas(), [.05] * 3 + [.0001] * 3)
    default = LoopPoseGraph({})
    assert default.loop_noise is default.noise
    with pytest.raises(ValueError, match='loop_rotation_sigma_rad'):
        LoopPoseGraph({'loop_rotation_sigma_rad': 0.})


def _rotation(rotvec):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(rotvec).as_matrix()


def _pose(rotvec, position):
    pose = np.eye(4)
    pose[:3, :3] = _rotation(rotvec)
    pose[:3, 3] = position
    return pose


def _twist(rotation, axis):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(rotation).as_quat()
    return 2. * np.arctan2(np.dot([x, y, z], axis), w)


def test_loop_tilt_source_is_validated():
    assert LoopPoseGraph({}).loop_tilt_source == 'registration'
    with pytest.raises(ValueError, match='loop_tilt_source'):
        LoopPoseGraph({'loop_tilt_source': 'gyro'})


def test_keyframe_gravity_loop_keeps_vertical_rotation_and_translation():
    graph = _pose_only(loop_tilt_source='keyframe_gravity')
    local = [_pose([.05, -.02, .3], [0., 0., 0.]), _pose([-.03, .08, -.4], [.01, .002, 0.])]
    graph.synchronize(local, loop_records(local), [(0, 1)])
    seed = np.linalg.inv(local[0]) @ local[1]
    up_reference = local[0][:3, :3].T @ [0., 0., 1.]
    up_target = local[1][:3, :3].T @ [0., 0., 1.]
    # Registration: 0.2 rad about the vertical (kept) and a 0.15 rad tilt error about GelSight x (removed).
    registered = seed.copy()
    registered[:3, :3] = _rotation([.15, 0., 0.]) @ _rotation(.2 * up_reference) @ seed[:3, :3]
    registered[:3, 3] += [.0003, -.0002, .0001]
    adjusted = graph.loop_measurement(0, 1, registered)
    np.testing.assert_allclose(adjusted[:3, :3] @ up_target, up_reference, atol=1e-12)
    np.testing.assert_array_equal(adjusted[:3, 3], registered[:3, 3])
    assert np.isclose(_twist(adjusted[:3, :3] @ seed[:3, :3].T, up_reference),
                      _twist(registered[:3, :3] @ seed[:3, :3].T, up_reference), atol=1e-12)
    # The stored loop keeps the registration.
    assert graph.add_loop(0, 1, np.linalg.inv(registered))
    np.testing.assert_allclose(graph.loops[0][2], registered, atol=1e-12)


def test_keyframe_gravity_loop_uses_each_keyframes_own_trial_gravity():
    graph = _pose_only(loop_tilt_source='keyframe_gravity')
    local = [_pose([.1, 0., 0.], [0., 0., 0.]), _pose([0., -.2, .5], [.03, 0., 0.])]
    gravities = [np.array([0., 0., -9.81]), _rotation([.3, .1, 0.]) @ [0., 0., -9.81]]
    graph.synchronize(local, loop_records(local, gravities, groups=[0, 1]), [])
    up = [pose[:3, :3].T @ (-g / np.linalg.norm(g)) for pose, g in zip(local, gravities)]
    registered = _pose([.2, -.1, .4], [.001, .002, 0.])
    adjusted = graph.loop_measurement(0, 1, registered)
    np.testing.assert_allclose(adjusted[:3, :3] @ up[1], up[0], atol=1e-12)
    np.testing.assert_array_equal(adjusted[:3, 3], registered[:3, 3])


@pytest.mark.parametrize('source', ['registration', 'keyframe_gravity'])
def test_keyframe_gravity_loop_does_not_tilt_the_pose_only_map(source):
    graph = _pose_only(loop_tilt_source=source, loop_rotation_sigma_rad=.05)
    local = [translation(x) for x in (0., .01, .02)]
    graph.synchronize(local, loop_records(local), [(0, 1), (1, 2)])
    # Correct translation, 0.2 rad tilt error about GelSight x.
    registered = _pose([.2, 0., 0.], [.02, 0., 0.])
    assert graph.add_loop(0, 2, np.linalg.inv(registered))
    graph.solve()
    tilt = max(np.arccos(np.clip(pose[:3, :3][2, 2], -1., 1.)) for pose in graph.map_poses())
    if source == 'registration':
        assert tilt > 1e-3
    else:
        assert tilt < 1e-9
        np.testing.assert_allclose(graph.map_poses(), local, atol=1e-9)
