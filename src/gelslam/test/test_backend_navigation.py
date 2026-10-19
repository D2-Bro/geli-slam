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
                                  'imu', 'bias', 'normal', 'height']
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
    factors = pair.factors(0, 1)
    values = gtsam.Values()
    for key in (0, 1):
        values.insert(key, gtsam.Pose3(imu_pose(np.eye(4), extrinsic)))
    factor = factors[1]
    A, _ = factor.linearize(values).jacobian()
    epsilon = 1e-6
    numeric = np.zeros_like(A)
    for slot, key in enumerate((0, 1)):
        for axis in range(6):
            step = np.eye(6)[axis] * epsilon
            plus, minus = gtsam.Values(values), gtsam.Values(values)
            pose = values.atPose3(key).matrix()
            plus.update(key, gtsam.Pose3(split_pose_plus(pose, step)))
            minus.update(key, gtsam.Pose3(split_pose_plus(pose, -step)))
            numeric[:, slot * 6 + axis] = (factor.whitenedError(plus) - factor.whitenedError(minus)) / (2 * epsilon)
    assert np.allclose(A, numeric, atol=1e-4, rtol=1e-4)
    assert np.linalg.norm(A[:, :6]) > 0 and np.linalg.norm(A[:, 6:]) > 0


def test_imu_only_recovery_has_no_tactile_factor():
    graph = PoseGraph(config())
    frames = [frame(0, new=True), frame(1, tactile=False), frame(2)]
    graph.add_keyframes(frames, 0, 3)
    assert graph.factor_kinds.count('imu') == 2
    assert 'normal' not in graph.factor_kinds and 'height' not in graph.factor_kinds
    assert len(graph.tactile_skipped) == 2


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
    assert graph.marginal_prior.size() == 0


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


def test_both_tactile_factor_jacobians_match_their_bilinear_residuals():
    first, second = frame(), frame(1)
    yy, xx = np.mgrid[:40, :40]
    second.N = np.dstack((.005 * xx, .003 * yy, np.ones_like(xx))).astype(np.float64)
    second.H = (.02 * xx + .01 * yy).astype(np.float64)
    mount = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([.2, -.1, .3])), np.array([.02, -.01, .03])).matrix()
    target = gtsam.Pose3(gtsam.Rot3.Expmap(np.array([.01, -.02, .03])), np.array([.00031, .00027, .001])).matrix()
    pair = TactilePair(first, second, np.eye(4), target, mount,
                       {'tactile_normal_sigma': .02, 'tactile_normal_huber_delta': .4})
    values = gtsam.Values()
    values.insert(0, gtsam.Pose3(imu_pose(np.eye(4), mount)))
    values.insert(1, gtsam.Pose3(imu_pose(target, mount)))
    for factor in pair.factors(0, 1):
        A, _ = factor.linearize(values).jacobian()
        numeric = np.zeros_like(A)
        epsilon = 1e-7
        for slot, key in enumerate((0, 1)):
            for axis in range(6):
                step = np.eye(6)[axis] * epsilon
                plus, minus = gtsam.Values(values), gtsam.Values(values)
                pose = values.atPose3(key).matrix()
                plus.update(key, gtsam.Pose3(split_pose_plus(pose, step)))
                minus.update(key, gtsam.Pose3(split_pose_plus(pose, -step)))
                numeric[:, slot * 6 + axis] = (factor.whitenedError(plus) - factor.whitenedError(minus)) / (2 * epsilon)
        assert np.allclose(A, numeric, atol=1e-4, rtol=1e-4)


def test_lost_overlap_has_nonzero_cost_and_cannot_win_an_lm_step():
    pair = TactilePair(frame(), frame(1), np.eye(4), np.eye(4), np.eye(4), {})
    values = gtsam.Values()
    values.insert(0, gtsam.Pose3())
    values.insert(1, gtsam.Pose3())
    factors = pair.factors(0, 1)
    before = sum(f.error(values) for f in factors)
    pose = np.eye(4)
    pose[0, 3] = 1.
    values.update(1, gtsam.Pose3(pose))
    assert sum(f.error(values) for f in factors) > before + 1e6


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
