"""15D backend keyframe seeds, IMU timing and persistence."""
from types import SimpleNamespace

import gtsam
import numpy as np
import pytest
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from gelslam.core.gtsam_pose_graph import PoseGraph
from gelslam.core.inertial import ImuSample, ImuBuffer
from gelslam.core.tracker import Tracker
from gelslam.core.keyframe import KeyFrameDB


from backend_test_support import config, frame


def test_backend_imu_factor_uses_configured_preintegration():
    settings = config()
    settings['imu_config']['integration_method'] = 'midpoint'
    graph = PoseGraph(settings)
    frames = [frame(0, new=True), frame(1, tactile=False)]
    acceleration = np.array([.3, -.2, 9.81])
    omega = np.array([.4, -.3, .5])
    samples = tuple(ImuSample(float(t), acceleration, omega)
                    for t in np.linspace(1., 1.1, 11))
    frames[1].imu_samples = samples
    graph.add_keyframes(frames, 0, 2)
    graph._sync_loop_pgo()
    imu, _ = graph.loop_pgo._imu_factors(0, 1, graph.loop_pgo.map_state(0))
    pim = imu.preintegratedMeasurements()
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


def test_graph_round_trip_restores_navigation_seeds(tmp_path):
    graph = PoseGraph(config())
    frames = [frame(i, new=i == 0) for i in range(5)]
    graph.add_keyframes(frames, 0, 5)
    graph.solve()
    graph.save(tmp_path)
    loaded = PoseGraph.load(tmp_path)
    assert loaded.temporal_links == graph.temporal_links
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
    assert graph.states == [] and graph.temporal_links == []


def test_backend_off_accepts_pose_only_messages():
    graph = PoseGraph(config())
    frames = [SimpleNamespace(trial_T_curr=np.eye(4))]
    graph.add_keyframes(frames, 0, 1, build_factors=False)
    assert graph.states == [None] and graph.raw_solutions().size() == 1


def test_tracker_packet_uses_keyframe_imu_times_and_separate_buffer(monkeypatch):
    import gelslam.core.keyframe as keyframe_module
    tracker = Tracker.__new__(Tracker)
    tracker.bridge = SimpleNamespace(cv2_to_imgmsg=lambda *_a, **_k: Image())
    tracker.imu_config = {'buffer_max_samples': 100}
    tracker.imu_enabled = True
    tracker.imu_buffer = ImuBuffer(100)
    tracker.image_to_imu_time_offset_s = .02
    tracker.gelsight_T_imu = np.eye(4)
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


def test_raw_backend_save_and_load(tmp_path):
    graph = PoseGraph(config())
    graph.add_keyframes([frame()], 0, 1, build_factors=False)
    graph.save(tmp_path)
    loaded = PoseGraph.load(tmp_path)
    assert loaded.raw_solutions().size() == 1
    assert loaded.states == [None]


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
    assert graph.states == [None] and graph.temporal_links == []
    with pytest.raises(ValueError, match='pose-only saved data'):
        graph.solve()
