"""15D backend observations, timing, marginalization and persistence."""
from types import SimpleNamespace

import gtsam
import numpy as np
import pytest
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from gelslam.core.backend_factors import (
    TactilePair, velocity_key, bias_key, state_keys, frozen_navigation_factor,
    variable_type, variable_width,
)
from gelslam.core.gtsam_pose_graph import PoseGraph
from gelslam.core.imu_navigation import imu_pose
from gelslam.core.inertial import ImuSample, ImuBuffer
from gelslam.core.numpy_vi import NavigationState
from gelslam.core.split_pose import split_pose_plus, optimize_split_pose_graph
from gelslam.core.tracker import Tracker
from gelslam.core.keyframe import KeyFrameDB


from backend_test_support import config, frame


def test_15d_graph_has_direct_tactile_motion_bias_and_no_relative_pose_factors():
    graph = PoseGraph(config())
    frames = [frame(0, new=True), frame(1)]
    graph.add_keyframes(frames, 0, 2)
    assert graph.graph_init.size() == 6  # two (Pose3, velocity, ConstantBias) triples
    assert graph.factor_kinds == ['pose_anchor', 'velocity_prior', 'bias_prior',
                                  'imu', 'bias', 'gelsight']
    assert all(measurement is None for _, _, measurement, _ in graph.edges)
    factors = [graph.graph.at(i) for i in range(graph.graph.size())]
    assert factors[3].dim() == 9 and factors[4].dim() == 6
    assert list(factors[3].keys()) == [0, velocity_key(0), 1, velocity_key(1), bias_key(0)]
    assert graph.graph.error(graph.graph_init) == pytest.approx(0., abs=1e-12)
    graph.solve()
    assert np.allclose(graph.states[1].velocity, 0., atol=1e-8)


def test_height_observation_corrects_an_inconsistent_tracker_seed():
    graph = PoseGraph(config())
    frames = [frame(0, new=True), frame(1, z=.002)]
    frames[1].navigation_state.velocity[:] = [.01, -.02, .01]
    frames[1].navigation_state.accel_bias[:] = [.01, -.02, .03]
    graph.add_keyframes(frames, 0, 2)
    before = graph.graph.error(graph.graph_init)
    graph.solve()
    assert graph.graph.error(graph.graph_init) < before * 1e-4
    assert abs(graph.poses[1][2, 3]) < 1e-6
    assert np.linalg.norm(graph.states[1].velocity) < 1e-4
    assert np.linalg.norm(graph.states[1].accel_bias) < 1e-4


def test_tactile_height_updates_both_poses_and_includes_imu_mounting_offset():
    extrinsic = np.eye(4)
    extrinsic[:3, 3] = [.02, -.01, .03]
    first, second = frame(), frame(1, height_m=.001)
    pair = TactilePair(first, second, np.eye(4), np.eye(4), extrinsic, {})
    factors = (pair.factor(0, 1),)
    values = gtsam.Values()
    for key in (0, 1):
        values.insert(key, gtsam.Pose3(imu_pose(np.eye(4), extrinsic)))
    factor = factors[0]
    A, _ = factor.linearize(values).jacobian()
    A = A[-1:]
    # Differentiate the intended conditional model: only displacement of the
    # GelSight origin along its frozen local Z, not the full warped height.
    from gelslam.core.imu_navigation import gelsight_pose
    epsilon = 1e-6
    target_numeric = np.zeros(6)
    pose = values.atPose3(1).matrix()
    for axis in range(6):
        step = np.eye(6)[axis] * epsilon
        plus = gelsight_pose(split_pose_plus(pose, step), extrinsic)
        minus = gelsight_pose(split_pose_plus(pose, -step), extrinsic)
        target_numeric[axis] = (plus[2, 3] - minus[2, 3]) / (2 * epsilon) / pair.height_sigma
    numeric = np.r_[-target_numeric, target_numeric][None, :]
    np.testing.assert_allclose(A, numeric, atol=1e-5, rtol=1e-5)
    assert np.linalg.norm(A[:, :6]) > 0 and np.linalg.norm(A[:, 6:]) > 0


def test_imu_only_recovery_has_no_tactile_factor():
    graph = PoseGraph(config())
    frames = [frame(0, new=True), frame(1, tactile=False), frame(2)]
    graph.add_keyframes(frames, 0, 3)
    assert graph.factor_kinds.count('imu') == 2
    assert 'gelsight' not in graph.factor_kinds
    assert len(graph.tactile_skipped) == 2


def test_backend_imu_factor_stores_midpoint_preintegration():
    graph = PoseGraph(config())
    frames = [frame(0, new=True), frame(1, tactile=False)]
    acceleration = np.array([.3, -.2, 9.81])
    omega = np.array([.4, -.3, .5])
    samples = tuple(ImuSample(float(t), acceleration, omega)
                    for t in np.linspace(1., 1.1, 11))
    frames[1].imu_samples = samples
    graph.add_keyframes(frames, 0, 2)
    factor_index = graph.factor_kinds.index('imu')
    pim = graph.graph.at(factor_index).preintegratedMeasurements()
    midpoint_velocity = np.zeros(3)
    start_velocity = np.zeros(3)
    for first, second in zip(samples, samples[1:]):
        dt = second.timestamp - first.timestamp
        elapsed = first.timestamp - samples[0].timestamp
        midpoint_velocity += gtsam.Rot3.Expmap(omega*(elapsed+dt/2)).rotate(acceleration)*dt
        start_velocity += gtsam.Rot3.Expmap(omega*elapsed).rotate(acceleration)*dt
    np.testing.assert_allclose(pim.deltaVij(), midpoint_velocity, atol=1e-13, rtol=0.)
    assert np.linalg.norm(pim.deltaVij()-start_velocity) > 1e-4


@pytest.mark.parametrize('damage', ['missing', 'start', 'end', 'gap'])
def test_incomplete_or_gapped_imu_interval_is_rejected(damage):
    graph = PoseGraph(config())
    frames = [frame(0, new=True), frame(1)]
    samples = frames[1].imu_samples
    frames[1].imu_samples = {'missing': (), 'start': samples[1:], 'end': samples[:-1],
                             'gap': (samples[0], samples[-1])}[damage]
    with pytest.raises(ValueError, match='IMU interval'):
        graph.add_keyframes(frames, 0, 2)


def test_window_prior_retains_velocity_bias_cross_terms_without_double_counting():
    graph = PoseGraph(config())
    frames = [frame(i, new=i == 0) for i in range(6)]
    graph.add_keyframes(frames, 0, 6)
    graph.solve()
    assert graph.window_start == 3
    keys = {key for i in range(graph.marginal_prior.size()) for key in graph.marginal_prior.at(i).keys()}
    assert set(state_keys(3)) <= keys
    prior = graph.marginal_prior.linearize(graph.graph_init)
    A, _ = prior.jacobian()
    assert np.linalg.norm(A.T @ A - np.diag(np.diag(A.T @ A))) > 1.
    H = A.T @ A
    graph._window_graph(3)
    A2, _ = graph.marginal_prior.linearize(graph.graph_init).jacobian()
    assert np.allclose(A2.T @ A2, H)
    frames.append(frame(6))
    graph.add_keyframes(frames, 6, 7)
    graph.solve()
    assert graph.window_start == 4
    graph.solve(global_optimization=True)
    assert graph.marginal_prior.size() == 1  # A PGO request does not reset local VI information.


def test_fixed_local_window_keeps_predecessor_factors_and_allows_first_state_to_move():
    graph = PoseGraph(config(marginalization_enabled=False))
    frames = [frame(i, new=i == 0, z=.002 if i >= 3 else 0.) for i in range(6)]
    graph.add_keyframes(frames, 0, 6)
    window, values, fixed = graph._window_graph(3)
    assert fixed == set(state_keys(2))
    assert set(values.keys()) == {key for i in range(2, 6) for key in state_keys(i)}
    # Three real boundary/interior edge sets, without an artificial prior.
    assert window.size() == 9
    keys = [list(window.at(i).keys()) for i in range(window.size())]
    assert [2, velocity_key(2), 3, velocity_key(3), bias_key(2)] in keys
    assert [bias_key(2), bias_key(3)] in keys and [2, 3] in keys
    before = graph.states[2].copy()
    graph.solve()
    for field in ('pose', 'velocity', 'accel_bias', 'gyro_bias'):
        np.testing.assert_array_equal(getattr(graph.states[2], field), getattr(before, field))
    assert abs(graph.states[3].pose[2, 3]) < 1e-6
    assert graph.marginal_prior.size() == 0


def test_fixed_local_window_excludes_outside_loop_endpoint():
    graph = PoseGraph(config(marginalization_enabled=False))
    graph.add_keyframes([frame(i, new=i == 0) for i in range(6)], 0, 6)
    assert graph._add_loop_pair(0, 5, np.eye(4))
    window, values, fixed = graph._window_graph(3)
    assert fixed == set(state_keys(2))
    assert all(values.exists(key) for key in fixed)
    keys = [list(window.at(i).keys()) for i in range(window.size())]
    assert [0, 5] not in keys
    assert not set(state_keys(0)).intersection(values.keys())
    assert [0] not in keys
    snapshots = {i: graph.states[i].copy() for i in (0, 2)}
    graph.solve()
    for index, before in snapshots.items():
        np.testing.assert_array_equal(graph.states[index].pose, before.pose)


@pytest.mark.parametrize('count', [1, 3])
def test_fixed_local_window_fixes_only_first_pose_without_predecessor(count):
    graph = PoseGraph(config(marginalization_enabled=False))
    graph.add_keyframes([frame(i, new=i == 0, z=.002 if i else 0.) for i in range(count)], 0, count)
    window, _, fixed = graph._window_graph(0)
    assert fixed == {0}
    keys = [list(window.at(i).keys()) for i in range(window.size())]
    assert [velocity_key(0)] in keys and [bias_key(0)] in keys
    assert [0] not in keys  # The pose prior has no free increment.
    before = graph.states[0].copy()
    graph.solve()
    np.testing.assert_array_equal(graph.states[0].pose, before.pose)
    if count > 1:
        assert abs(graph.states[1].pose[2, 3]) < 1e-6


def test_new_trial_without_temporal_predecessor_has_only_pose_fixed_inside_window():
    graph = PoseGraph(config(marginalization_enabled=False, local_window_size=4))
    frames = [frame(i, new=i in (0, 3), group=int(i >= 3)) for i in range(6)]
    graph.add_keyframes(frames, 0, 6)
    window, values, fixed = graph._window_graph(2)
    assert fixed == set(state_keys(1)) | {3}
    assert all(values.exists(key) for key in state_keys(2) + state_keys(4) + state_keys(5))
    assert not set(state_keys(0)).intersection(values.keys())
    keys = [list(window.at(i).keys()) for i in range(window.size())]
    assert [velocity_key(3)] in keys and [bias_key(3)] in keys


def test_inside_fixed_pose_retains_priors_and_commits_velocity_and_both_biases():
    graph = PoseGraph(config(marginalization_enabled=False))
    graph.add_keyframes([frame(0, new=True)], 0, 1)
    state = graph.states[0]
    state.pose[0, 3] = .002  # Conflict with the old pose prior must not move it.
    state.velocity[:] = [.1, -.2, .3]
    state.accel_bias[:] = [.01, -.02, .03]
    state.gyro_bias[:] = [.001, -.002, .003]
    graph.poses[0] = state.pose.copy()
    graph._put_state(0, state)
    pose, observation_pose = state.pose.copy(), graph.poses[0].copy()
    window, values, fixed = graph._window_graph(0)
    assert fixed == {0} and window.size() == 2
    assert window.error(values) > 0.
    graph.solve()
    np.testing.assert_array_equal(state.pose, pose)
    np.testing.assert_array_equal(graph.poses[0], observation_pose)
    np.testing.assert_allclose(state.velocity, 0., atol=1e-8)
    np.testing.assert_allclose(state.accel_bias, 0., atol=1e-8)
    np.testing.assert_allclose(state.gyro_bias, 0., atol=1e-8)
    np.testing.assert_array_equal(graph.graph_init.atVector(velocity_key(0)), state.velocity)
    np.testing.assert_array_equal(graph.graph_init.atConstantBias(bias_key(0)).vector(),
                                  np.r_[state.accel_bias, state.gyro_bias])


def test_inside_anchor_velocity_biases_update_with_pose_fixed_outside_predecessor():
    graph = PoseGraph(config(marginalization_enabled=False, local_window_size=4))
    graph.add_keyframes([frame(i, new=i in (0, 3), group=int(i >= 3)) for i in range(6)], 0, 6)
    state = graph.states[3]
    state.velocity[:] = [.1, -.2, .3]
    state.accel_bias[:] = [.01, -.02, .03]
    state.gyro_bias[:] = [.001, -.002, .003]
    graph._put_state(3, state)
    pose = state.pose.copy()
    outside = graph.states[1].copy()
    graph.solve()
    np.testing.assert_array_equal(state.pose, pose)
    assert np.linalg.norm(state.velocity) < 1e-5
    assert np.linalg.norm(state.accel_bias) < 1e-5
    assert np.linalg.norm(state.gyro_bias) < 1e-5
    for field in ('pose', 'velocity', 'accel_bias', 'gyro_bias'):
        np.testing.assert_array_equal(getattr(graph.states[1], field), getattr(outside, field))


@pytest.mark.parametrize('loop_reference', [0, 1])
def test_outside_predecessor_keeps_all_15d_fixed_with_separate_loop(loop_reference):
    graph = PoseGraph(config(marginalization_enabled=False))
    frames = [frame(i, new=i in (0, 2), group=int(i >= 2)) for i in range(6)]
    graph.add_keyframes(frames, 0, 6)
    assert graph._add_loop_pair(loop_reference, 5, np.eye(4))
    fixed_index = 2
    state = graph.states[fixed_index]
    state.pose[0, 3] = .002
    state.velocity[:] = [.1, -.2, .3]
    state.accel_bias[:] = [.01, -.02, .03]
    state.gyro_bias[:] = [.001, -.002, .003]
    graph.poses[fixed_index] = state.pose.copy()
    graph._put_state(fixed_index, state)
    pose, observation_pose = state.pose.copy(), graph.poses[fixed_index].copy()
    frozen = state.copy()
    excluded = graph.states[1].copy()
    window, values, fixed = graph._window_graph(3)
    assert fixed == set(state_keys(2))
    assert not set(state_keys(1)).intersection(values.keys())
    keys = [list(window.at(i).keys()) for i in range(window.size())]
    assert [velocity_key(fixed_index)] not in keys and [bias_key(fixed_index)] not in keys
    assert [fixed_index] not in keys
    assert window.error(values) > 0.
    graph.solve()
    np.testing.assert_array_equal(state.pose, pose)
    np.testing.assert_array_equal(graph.poses[fixed_index], observation_pose)
    for field in ('velocity', 'accel_bias', 'gyro_bias'):
        np.testing.assert_array_equal(getattr(state, field), getattr(frozen, field))
    np.testing.assert_array_equal(graph.graph_init.atVector(velocity_key(fixed_index)), state.velocity)
    np.testing.assert_array_equal(graph.graph_init.atConstantBias(bias_key(fixed_index)).vector(),
                                  np.r_[state.accel_bias, state.gyro_bias])
    for field in ('pose', 'velocity', 'accel_bias', 'gyro_bias'):
        np.testing.assert_array_equal(getattr(graph.states[1], field), getattr(excluded, field))


def test_loop_pgo_request_keeps_the_local_window_boundary_fixed():
    graph = PoseGraph(config(marginalization_enabled=False))
    graph.add_keyframes([frame(i, new=i == 0) for i in range(5)], 0, 5)
    graph.states[1].pose[2, 3] = .004
    graph.poses[1][2, 3] = .004
    graph._put_state(1, graph.states[1])
    graph.solve()
    assert graph.states[1].pose[2, 3] == .004
    graph.solve(global_optimization=True)
    assert graph.states[1].pose[2, 3] == pytest.approx(.004)


def test_graph_round_trip_restores_navigation_and_direct_observations(tmp_path):
    graph = PoseGraph(config())
    frames = [frame(i, new=i == 0) for i in range(5)]
    graph.add_keyframes(frames, 0, 5)
    graph.solve()
    graph.save(tmp_path)
    loaded = PoseGraph.load(tmp_path)
    assert loaded.factor_kinds == graph.factor_kinds
    assert np.allclose(loaded.poses, graph.poses)
    for a, b in zip(graph.states, loaded.states):
        assert np.allclose(a.velocity, b.velocity)
        assert np.allclose(a.accel_bias, b.accel_bias)
    loaded.solve()
    assert np.allclose(loaded.poses, graph.poses, atol=1e-6)


def test_missing_navigation_seed_does_not_silently_create_pose_odometry():
    graph = PoseGraph(config())
    missing = frame(0, new=True)
    missing.navigation_state = None
    with pytest.raises(ValueError, match='navigation seed'):
        graph.add_keyframes([missing], 0, 1)
    assert graph.graph.size() == 0


def test_backend_off_accepts_pose_only_messages():
    graph = PoseGraph(config())
    frames = [SimpleNamespace(trial_T_curr=np.eye(4))]
    graph.add_keyframes(frames, 0, 1, build_factors=False)
    assert graph.graph.size() == 0 and graph.raw_solutions().size() == 1


def test_tracker_packet_uses_keyframe_imu_times_and_separate_buffer(monkeypatch):
    import gelslam.core.tracker as module
    import gelslam.core.keyframe as keyframe_module
    tracker = Tracker.__new__(Tracker)
    tracker.bridge = SimpleNamespace(cv2_to_imgmsg=lambda *_a, **_k: Image())
    tracker.imu_config = {'buffer_max_samples': 100}
    tracker.imu_enabled = True
    tracker.imu_buffer = ImuBuffer(100)
    tracker.image_to_imu_time_offset_s = .02
    tracker.frontend_visual_inertial = SimpleNamespace(gravity=np.array([0., 0., -9.81]))
    for t in np.linspace(1., 1.3, 31):
        tracker.add_imu_sample(ImuSample(float(t), [0., 0., 9.81], [0., 0., 0.]))
    def header(t):
        h = Header()
        ns = round(t * 1e9)
        h.stamp.sec, h.stamp.nanosec = divmod(ns, 10**9)
        return h
    tracker._frontend_vi_state = frame(0).navigation_state
    tracker._frontend_vi_state.timestamp = 1.02
    first = tracker._make_keyframe_msg(frame(), 0, 0, True, np.eye(4), header(1.))
    # Frontend history is consumed independently between the keyframes.
    tracker.imu_buffer.begin_online_interval(1.2)
    tracker._frontend_vi_state = frame(1).navigation_state
    tracker._frontend_vi_state.timestamp = 1.22
    second = tracker._make_keyframe_msg(frame(), 1, 1, False, np.eye(4), header(1.2))
    assert first.navigation_valid and second.navigation_valid
    rows = np.asarray(second.imu_measurements).reshape(-1, 7)
    assert rows[0, 0] == pytest.approx(1.02)
    assert rows[-1, 0] == pytest.approx(1.22)
    assert np.allclose(np.diff(rows[:, 0]), .01)
    # Convert through the actual ROS message path without SIFT overhead.
    monkeypatch.setattr(keyframe_module, 'KeyFrame', lambda *_a, **_k: SimpleNamespace(kid=_a[4]))
    db = KeyFrameDB(1., SimpleNamespace(info=lambda *_: None, warning=lambda *_: None))
    db.bridge = SimpleNamespace(imgmsg_to_cv2=lambda _m: np.zeros((2, 2)))
    db.insert(first)
    from rclpy.serialization import serialize_message, deserialize_message
    from gelslam_msgs.msg import KeyFrameMsg
    second = deserialize_message(serialize_message(second), KeyFrameMsg)
    db.insert(second)
    assert db[1].navigation_state.timestamp == pytest.approx(1.22)
    assert len(db[1].imu_samples) == len(rows)


def test_lost_overlap_has_nonzero_cost_and_cannot_win_an_lm_step():
    pair = TactilePair(frame(), frame(1), np.eye(4), np.eye(4), np.eye(4), {})
    values = gtsam.Values()
    values.insert(0, gtsam.Pose3())
    values.insert(1, gtsam.Pose3())
    factors = (pair.factor(0, 1),)
    before = sum(f.error(values) for f in factors)
    pose = np.eye(4)
    pose[0, 3] = 1.
    values.update(1, gtsam.Pose3(pose))
    assert sum(f.error(values) for f in factors) > before + 1e6


def test_opencv_tactile_factor_corrects_known_pose_error_on_textured_maps():
    first, second = frame(), frame(1)
    yy, xx = np.mgrid[:40, :40]
    nx, ny = .2 * np.sin(xx / 5), .2 * np.cos(yy / 7)
    normal = np.dstack((nx, ny, np.sqrt(1 - nx * nx - ny * ny))).astype(np.float32)
    height = (.1 * np.sin(xx / 7) + .2 * np.cos(yy / 6)).astype(np.float32)
    for observation in (first, second):
        observation.N, observation.H = normal.copy(), height.copy()
    target = np.eye(4)
    target[:3, 3] = [.0003, -.0002, .0001]
    pair = TactilePair(first, second, np.eye(4), target, np.eye(4),
                       {'tactile_normal_sigma': .02})
    values = gtsam.Values()
    values.insert(0, gtsam.Pose3())
    values.insert(1, gtsam.Pose3(target))
    graph = gtsam.NonlinearFactorGraph()
    graph.add(pair.factor(0, 1))
    graph.add(frozen_navigation_factor([0], np.eye(6) * 1e6, np.zeros(6), values))
    before = graph.error(values)
    result = optimize_split_pose_graph(graph, values, max_iterations=30)
    assert graph.error(result) < before * 1e-4
    # The quantized sampler has a subpixel plateau; do not demand an exact
    # zero XY translation. Its known identity solution is recovered within
    # one table step (1 mm pixel pitch / 32).
    assert np.linalg.norm(result.atPose3(1).translation()) < .001 / 32


def test_raw_backend_save_and_load(tmp_path):
    graph = PoseGraph(config())
    graph.add_keyframes([frame()], 0, 1, build_factors=False)
    graph.save(tmp_path)
    loaded = PoseGraph.load(tmp_path)
    assert loaded.raw_solutions().size() == 1
    assert loaded.graph.size() == 0


def test_uncalibrated_mount_is_rejected_for_optimization_but_allows_raw_output():
    settings = config()
    settings['imu_config']['gelsight_T_imu'] = None
    graph = PoseGraph(settings)
    with pytest.raises(ValueError, match='calibrated'):
        graph.add_keyframes([frame(0, new=True)], 0, 1)
    graph.add_keyframes([frame(0, new=True)], 0, 1, build_factors=False)
    assert graph.raw_solutions().size() == 1


def test_legacy_graph_remains_viewable_without_inventing_navigation(tmp_path):
    import pickle
    with (tmp_path / 'pose_graph.pkl').open('wb') as stream:
        pickle.dump(dict(poses=[np.eye(4)], pose_graph_config={},
                         gelsight_T_imu=np.eye(4), edges=[]), stream)
    graph = PoseGraph.load(tmp_path)
    assert np.allclose(graph.raw_solutions()[0], np.eye(4))
    assert graph.states == [None] and graph.graph.size() == 0
    with pytest.raises(ValueError, match='pose-only saved data'):
        graph.solve()
