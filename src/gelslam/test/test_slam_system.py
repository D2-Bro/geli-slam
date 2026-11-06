"""Thread ownership, atomic correction and shutdown tests."""
from dataclasses import FrozenInstanceError, replace
from threading import Event, Thread
import numpy as np
import pytest
from std_msgs.msg import Header
from gelslam.core.slam_packets import ArrayBridge
from gelslam.core.frame import create_keyframe_msg
from gelslam.core.shared_map import SharedMap, MapEstimate
from gelslam.core.slam_system import SlamSystem
from gelslam.core.local_mapping import LocalMapping, LoopCandidate
from gelslam.core.numpy_vi import NavigationState
from gelslam.core.inertial import ImuBuffer, ImuSample
from gelslam.utils import Logger
from backend_test_support import frame, config


def packet(index=0, new=None, trial=0):
    observation = frame(index, new=index == 0 if new is None else new)
    header = Header()
    timestamp = observation.navigation_state.timestamp
    header.stamp.sec = int(timestamp)
    header.stamp.nanosec = int(round((timestamp - int(timestamp)) * 1e9))
    return replace(create_keyframe_msg(ArrayBridge(), observation, index, index,
        observation.is_new_trial(), observation.trial_T_curr, header,
        observation.navigation_state, observation.gravity, observation.imu_samples), trial_id=trial)


def estimate(x=0.):
    state = NavigationState(1., np.eye(4), np.zeros(3), np.zeros(3), np.zeros(3))
    state.pose[0, 3] = x
    return MapEstimate(state, state.pose.copy(), state.pose.copy(), 0, 0)


def test_observations_and_atomic_solution_are_immutable_and_versioned():
    store = SharedMap()
    first = packet()
    store.append_keyframe(first)
    snap = store.snapshot_for_tracking()
    assert snap.revision == 0
    with pytest.raises(ValueError):
        first.normal_map[0, 0] = 0
    assert store.commit_solution(snap.epoch, {0: estimate(1)})
    old = store.snapshot_for_tracking()
    store.append_keyframe(packet(1))
    assert store.commit_solution(old.epoch, {0: estimate(2)})
    assert old.estimates[0].state.pose[0, 3] == 1
    current = store.snapshot_for_tracking()
    assert len(current.observations) == 2 and current.revision == 2
    assert current.estimates[0].state.pose[0, 3] == 2
    with pytest.raises(ValueError):
        current.estimates[0].state.pose[0, 3] = 0
    with pytest.raises(FrozenInstanceError):
        current.estimates[0].state.velocity = np.zeros(3)
    store.reset()
    assert not store.commit_solution(old.epoch, {0: estimate(3)})
    assert store.snapshot_for_tracking().epoch == 1


def test_long_ba_keeps_ingestion_and_snapshots_available_and_drains_late_loop():
    entered, release, input_done = Event(), Event(), Event()
    calls, frames = [], []
    class Mapper:
        def process(self, snap, loops, store):
            calls.append((len(snap.observations), len(loops)))
            if snap.observations and not entered.is_set():
                entered.set()
                assert release.wait(3)
            return False
    class Search:
        def detect(self, snap):
            return [object()] if snap.observations else []
    system = SlamSystem(frames.append, Mapper(), Search())
    system.start()
    system.accept_keyframes([packet()])
    assert entered.wait(3)
    system.submit_input(input_done.set)
    system.submit_image('F_j')
    assert input_done.wait(1)
    assert len(system.shared_map.snapshot_for_tracking().observations) == 1
    system.accept_keyframes([packet(1)])
    release.set()
    system.stop()
    assert frames == ['F_j']
    assert calls[-1][0] == 2 and sum(count for _, count in calls) >= 1
    assert all(not thread.is_alive() and not thread.daemon for thread in system._threads)


def test_image_latest_queue_retains_keyframes_and_reports_worker_failure():
    enter, release = Event(), Event()
    seen = []
    class Mapper:
        def process(self, *args):
            return False
    def track(value):
        seen.append(value)
        if value == 0:
            enter.set()
            assert release.wait(3)
    system = SlamSystem(track, Mapper())
    system.start()
    system.submit_image(0)
    assert enter.wait(3)
    system.submit_image(1)
    system.submit_image(2)
    release.set()
    system.stop()
    assert seen == [0, 2]
    assert system.metrics['images_replaced'] == 1
    broken = SlamSystem(lambda _: (_ for _ in ()).throw(ValueError('worker sentinel')), Mapper())
    broken.start()
    broken.submit_image(0)
    with pytest.raises(RuntimeError, match='worker sentinel'):
        broken.stop()
    assert all(not t.is_alive() for t in broken._threads)


def test_local_mapping_stores_seeds_and_save_reloads(tmp_path):
    logger = Logger(print_output=False)
    cfg = config(coverage_graph_overlap_threshold=.8)
    cfg['device_config'] = {'ppmm': 1.}
    store = SharedMap()
    mapper = LocalMapping(cfg, logger)
    for index in range(12):
        store.append_keyframe(packet(index))
        assert mapper.process(store.snapshot_for_optimization(), [], store)
    assert len(mapper.graph.temporal_links) == 11
    assert all(edge[2] is None for edge in mapper.graph.edges)
    mapper.save(tmp_path)
    from gelslam.core.gtsam_pose_graph import PoseGraph
    restored = PoseGraph.load(tmp_path)
    restored.solve()
    assert len(restored.states) == 12


def test_late_loop_candidate_is_valid_after_a_new_solution_revision():
    cfg = config(coverage_graph_overlap_threshold=.8)
    cfg['device_config'] = {'ppmm': 1.}
    store, logger = SharedMap(), Logger(print_output=False)
    mapper = LocalMapping(cfg, logger)
    store.append_keyframe(packet())
    store.append_keyframe(packet(1))
    mapper.process(store.snapshot_for_optimization(), [], store)
    old = store.snapshot_for_tracking()
    store.append_keyframe(packet(2))
    mapper.process(store.snapshot_for_optimization(), [], store)
    candidate = LoopCandidate(old.epoch, old.revision, 0, 2, np.eye(4))
    mapper.process(store.snapshot_for_optimization(), [candidate], store)
    assert (0, 2) in mapper.accepted_pairs
    count = len(mapper.graph.edges)
    mapper.process(store.snapshot_for_optimization(), [candidate], store)
    assert len(mapper.graph.edges) == count


def test_loop_corrected_seeds_in_one_batch_return_to_local_anchor():
    cfg = config(coverage_graph_overlap_threshold=.8)
    cfg['device_config'] = {'ppmm': 1.}
    store = SharedMap()
    mapper = LocalMapping(cfg, Logger(print_output=False))
    store.append_keyframe(packet())
    store.append_keyframe(packet(1))
    mapper.process(store.snapshot_for_optimization(), [], store)
    measured = np.eye(4)
    measured[0, 3] = -.002
    mapper.process(store.snapshot_for_optimization(),
        [LoopCandidate(0, 0, 0, 1, measured)], store)
    corrected = store.snapshot_for_tracking().estimates[1].state
    correction_x = corrected.pose[0, 3]
    assert 0. < correction_x < .002
    assert abs(mapper.graph.states[1].pose[0, 3]) < 1e-12
    # Both packets use loop-corrected tracking coordinates. K_3 refers to
    # K_2, which has not yet been appended to the local graph.
    basis = corrected.copy()
    for index in (2, 3):
        current = packet(index)
        current_pose = corrected.pose.copy()
        store.append_keyframe(replace(current,
            navigation_pose=current_pose, trial_t_curr=current_pose.reshape(-1),
            velocity=corrected.velocity.copy(), accel_bias=corrected.accel_bias.copy(),
            gyro_bias=corrected.gyro_bias.copy(),
            anchor_kid=index - 1, anchor_state=basis.copy()))
        basis.timestamp = current.navigation_timestamp
    mapper.process(store.snapshot_for_optimization(), [], store)
    for index in (2, 3):
        assert abs(mapper.db[index].navigation_state.pose[0, 3]) < 1e-12
        assert abs(mapper.graph.states[index].pose[0, 3]) < 1e-10
        assert np.isclose(store.snapshot_for_tracking().estimates[index].state.pose[0, 3],
                          correction_x, atol=1e-9)


class PositionVisual:
    def __len__(self):
        return 10
    def evaluate(self, reference, target):
        jacobian = np.tile(np.c_[np.zeros((3, 3)), target[:3, :3]], (10, 1))
        return np.zeros(30), np.zeros_like(jacobian), jacobian


def test_seed_appended_during_ba_rebases_against_local_anchor_without_pose_factor():
    cfg = config(coverage_graph_overlap_threshold=.8)
    cfg['device_config'] = {'ppmm': 1.}
    store = SharedMap()
    mapper = LocalMapping(cfg, Logger(print_output=False))
    store.append_keyframe(packet())
    mapper.process(store.snapshot_for_optimization(), [], store)
    stale_anchor = store.snapshot_for_tracking().estimates[0].state
    changed = estimate(.001)
    mapper.graph.states[0] = changed.state.copy()
    mapper.graph.poses[0] = changed.trial_pose.copy()
    store.commit_solution(0, {0: changed})
    store.append_keyframe(replace(packet(1), anchor_kid=0, anchor_state=stale_anchor))
    mapper.process(store.snapshot_for_optimization(), [], store)
    assert mapper.db[1].navigation_state.pose[0, 3] == pytest.approx(.001)
    assert mapper.graph.temporal_links == [(0, 1)]
    assert all(edge[2] is None for edge in mapper.graph.edges)


def test_backend_off_has_raw_poses_and_no_measurement_factors():
    cfg = config(coverage_graph_overlap_threshold=.8)
    cfg['device_config'] = {'ppmm': 1.}
    store = SharedMap()
    mapper = LocalMapping(cfg, Logger(print_output=False), enabled=False)
    store.append_keyframe(packet())
    store.append_keyframe(packet(1))
    mapper.process(store.snapshot_for_optimization(), [], store)
    assert mapper.graph.states == [None, None]
    assert mapper.solutions.size() == 2
    assert store.snapshot_for_tracking().estimates[1].state is None


def test_imu_retention_buffer_can_be_saved_and_reopened():
    import pickle
    buffer = ImuBuffer(max_samples=4)
    for t in np.linspace(1., 1.03, 4):
        buffer.add(ImuSample(t, [0, 0, 9.81], [0, 0, 0]))
    restored = pickle.loads(pickle.dumps(buffer))
    assert len(restored.samples_covering(1., 1.03)) == 4
    restored.add(ImuSample(1.04, [0, 0, 9.81], [0, 0, 0]))
    assert len(restored) == 4 and not restored.samples_covering(1., 1.04)


def test_concurrent_batch_matches_sequential_reference():
    cfg = config(coverage_graph_overlap_threshold=.8)
    cfg['device_config'] = {'ppmm': 1.}
    logger = Logger(print_output=False)
    reference_store, concurrent_store = SharedMap(), SharedMap()
    for index in range(6):
        item = packet(index)
        reference_store.append_keyframe(item)
        concurrent_store.append_keyframe(item)
    reference = LocalMapping(cfg, logger)
    reference.process(reference_store.snapshot_for_optimization(), [], reference_store)
    concurrent = LocalMapping(cfg, logger)
    system = SlamSystem(lambda _: None, concurrent, shared_map=concurrent_store)
    system.start()
    system.stop()
    assert len(concurrent.graph.states) == 6
    for expected, actual in zip(reference.graph.states, concurrent.graph.states):
        for field in ('pose', 'velocity', 'accel_bias', 'gyro_bias'):
            assert np.allclose(getattr(expected, field), getattr(actual, field), atol=1e-10)


def test_late_candidate_merges_disconnected_trials_and_ignores_old_epoch():
    cfg = config(coverage_graph_overlap_threshold=.8)
    cfg['device_config'] = {'ppmm': 1.}
    store = SharedMap()
    mapper = LocalMapping(cfg, Logger(print_output=False))
    store.append_keyframe(packet())
    store.append_keyframe(packet(1, new=True, trial=1))
    mapper.process(store.snapshot_for_optimization(), [], store)
    candidate = LoopCandidate(0, 0, 0, 1, np.eye(4))
    mapper.process(store.snapshot_for_optimization(), [candidate], store)
    assert mapper.groups.get_parent_group(0, mapper.db) == mapper.groups.get_parent_group(1, mapper.db)
    store.reset()
    store.append_keyframe(packet())
    mapper.process(store.snapshot_for_optimization(), [candidate], store)
    assert mapper.db.size() == 1 and not mapper.accepted_pairs


def test_reader_never_sees_mixed_state_batch_revisions():
    store = SharedMap()
    store.append_keyframe(packet())
    store.append_keyframe(packet(1))
    finished = Event()
    errors = []
    def write():
        try:
            for value in range(100):
                store.commit_solution(0, {0: estimate(value), 1: estimate(value)})
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()
    worker = Thread(target=write)
    worker.start()
    while not finished.is_set():
        snap = store.snapshot_for_tracking()
        if snap.estimates:
            assert snap.estimates[0].revision == snap.estimates[1].revision
            assert snap.estimates[0].state.pose[0, 3] == snap.estimates[1].state.pose[0, 3]
    worker.join()
    assert not errors


def test_estimator_input_mutations_run_on_tracking_owner_thread():
    from threading import get_ident
    input_ids, image_ids, ba_ids = [], [], []
    class Mapper:
        def process(self, *args):
            ba_ids.append(get_ident())
            return False
    system = SlamSystem(lambda _: image_ids.append(get_ident()), Mapper())
    system.start()
    system.submit_input(lambda: input_ids.append(get_ident()))
    system.submit_image('F_j')
    system.stop()
    assert len(set(input_ids + image_ids)) == 1
    assert ba_ids and ba_ids[0] != input_ids[0]
