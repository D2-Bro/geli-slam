"""Retire initialization GT only after queued and active callbacks finish."""
import threading
from types import SimpleNamespace

import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.context import Context
from rclpy.node import Node

import gelslam.nodes.tracker_node as node_module
from gelslam.nodes.tracker_executor import TrackerExecutor
from gelslam.nodes.tracker_node import TrackerNode


@pytest.mark.parametrize('single_process', [False, True])
def test_spin_owner_drains_before_deleting_and_restarts_same_node(monkeypatch, single_process):
    if single_process:
        import gelslam.nodes.gelslam_node as module
    else:
        module = node_module
    events, executors = [], []
    owner = threading.get_ident()
    ready = False

    def release():
        nonlocal ready
        assert threading.get_ident() == owner
        assert executors[0].stopped
        events.append('delete_subscription')
        ready = False

    node = SimpleNamespace(
        _initialization_imu_gt_release_ready=lambda: ready,
        _release_initialization_imu_gt=release,
        system=SimpleNamespace(raise_if_failed=lambda: None),
        destroy_node=lambda: events.append('destroy_node'))

    class Executor:
        def __init__(self, **_kwargs):
            self.index, self.stopped = len(executors), False
            executors.append(self)

        def add_node(self, same_node):
            assert same_node is node
            events.append(('add', self.index))

        def spin_once(self, timeout_sec):
            nonlocal ready
            assert not self.stopped and timeout_sec == .1
            events.append(('spin', self.index))
            ready = self.index == 0

        def shutdown(self):
            assert threading.get_ident() == owner
            self.stopped = True
            events.append(('drain', self.index))

    running = iter((True, True, False))
    monkeypatch.setattr(module, 'GelSlamNode' if single_process else 'TrackerNode', lambda: node)
    monkeypatch.setattr(module, 'TrackerExecutor', Executor)
    monkeypatch.setattr(module.rclpy, 'init', lambda **_kwargs: None)
    monkeypatch.setattr(module.rclpy, 'ok', lambda: next(running))
    monkeypatch.setattr(module.rclpy, 'try_shutdown', lambda: events.append('ros_shutdown'))
    module.main()
    assert events == [('add', 0), ('spin', 0), ('drain', 0), 'delete_subscription',
                      ('add', 1), ('spin', 1), ('drain', 1), 'destroy_node', 'ros_shutdown']


@pytest.mark.parametrize('repeat', range(3))
def test_queued_gt_handler_finishes_before_real_subscription_destruction(
        tmp_path, monkeypatch, repeat):
    monkeypatch.setenv('ROS_LOG_DIR', str(tmp_path / 'ros_log'))
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    context = Context()
    rclpy.init(context=context)
    node = Node('gelslam_gt_retirement_test', context=context,
                enable_rosout=False, start_parameter_services=False)
    executor = TrackerExecutor(num_threads=2, context=context)
    executor.add_node(node)
    release_workers = threading.Event()
    started = [threading.Event(), threading.Event()]
    draining, finished, continued = threading.Event(), threading.Event(), threading.Event()
    events, errors, replacements = [], [], []
    positions = [(1., (0., 0., 0.))]

    def receive(message):
        positions.append((2., (message.pose.position.x, 0., 0.)))
        events.append('gt_callback')

    subscription = node.create_subscription(PoseStamped, 'initialization_gt', receive, 10)
    # Evaluation has its own subscription on the same topic and must survive.
    evaluation = node.create_subscription(PoseStamped, 'initialization_gt', lambda _m: None, 10)
    state, prior, imu_buffer = object(), object(), object()
    node.tracker = SimpleNamespace(
        imu_config={}, _frontend_vi_state=state, _trial_index=0,
        _pending_frontend_initialization=None, _pending_trial_contact=None,
        _ground_truth_imu_positions=positions,
        marginal_prior=prior, imu_buffer=imu_buffer)
    node.ground_truth_imu_pose_sub = subscription
    node._initialization_imu_gt_release_ready = lambda: (
        TrackerNode._initialization_imu_gt_release_ready(node))
    real_destroy = node.destroy_subscription

    def destroy(sub):
        events.append('delete_subscription')
        assert events[-2] == 'gt_callback'
        return real_destroy(sub)

    node.destroy_subscription = destroy

    def occupy_worker(index):
        started[index].set()
        assert release_workers.wait(5.), 'worker was not released'

    busy = [executor._executor.submit(occupy_worker, i) for i in range(2)]
    retire_thread = None
    try:
        assert all(event.wait(2.) for event in started)
        message = PoseStamped()
        message.pose.position.x = .25

        def take(sub):
            # Exercise the same handle acquisition as rclpy._take_subscription.
            # This handler is deliberately queued behind the occupied workers.
            with sub.handle:
                return message

        handler = executor._make_handler(
            subscription, node, take, executor._execute_subscription)
        executor._futures.append(handler)
        submitted = executor._executor.submit(handler)

        def retire():
            try:
                draining.set()
                executor.shutdown()
                assert submitted.done()
                assert TrackerNode._release_initialization_imu_gt(node)
                replacement = TrackerExecutor(num_threads=2, context=context)
                replacements.append(replacement)
                replacement.add_node(node)
                guard = node.create_guard_condition(continued.set)
                guard.trigger()
                for _ in range(20):
                    replacement.spin_once(timeout_sec=.01)
                    if continued.is_set():
                        break
            except BaseException as error:
                errors.append(error)
            finally:
                finished.set()

        retire_thread = threading.Thread(target=retire)
        retire_thread.start()
        assert draining.wait(2.)
        assert not finished.wait(.05)
        assert node.ground_truth_imu_pose_sub is subscription
        assert positions == [(1., (0., 0., 0.))]
        release_workers.set()
        assert finished.wait(3.)
        assert not errors
        assert handler.done() and handler.exception() is None
        assert events == ['gt_callback', 'delete_subscription']
        assert node.ground_truth_imu_pose_sub is None and positions == []
        assert evaluation in node.subscriptions
        assert node.tracker._frontend_vi_state is state
        assert node.tracker.marginal_prior is prior and node.tracker.imu_buffer is imu_buffer
        assert continued.is_set(), 'callbacks did not resume on the replacement executor'
    finally:
        release_workers.set()
        if retire_thread is not None:
            retire_thread.join(timeout=5.)
        for future in busy:
            future.result(timeout=2.)
        for replacement in replacements:
            replacement.shutdown()
        executor.shutdown()
        node.destroy_subscription = real_destroy
        Node.destroy_node(node)
        context.try_shutdown()
    assert retire_thread is not None and not retire_thread.is_alive()


def test_draining_preserves_callback_errors(tmp_path, monkeypatch):
    monkeypatch.setenv('ROS_LOG_DIR', str(tmp_path / 'ros_log'))
    context = Context()
    rclpy.init(context=context)
    executor = TrackerExecutor(num_threads=2, context=context)
    ran, release = threading.Event(), threading.Event()

    def fail():
        ran.set()
        assert release.wait(3.)
        raise RuntimeError('callback failure during drain')

    executor.create_task(fail)
    try:
        executor.spin_once(timeout_sec=.1)
        assert ran.wait(2.)
        release.set()
        with pytest.raises(RuntimeError, match='callback failure during drain'):
            executor.shutdown()
    finally:
        release.set()
        executor.shutdown()
        context.try_shutdown()


def test_single_process_retirement_waits_for_queued_tracking_inputs():
    from gelslam.core.slam_system import SlamSystem
    from gelslam.nodes.gelslam_node import GelSlamNode

    events, positions = [], [(1., (0., 0., 0.))]
    system = SlamSystem(lambda _image: None, SimpleNamespace(process=lambda *_args: False))
    tracker = SimpleNamespace(
        imu_config={}, _frontend_vi_state=object(), _trial_index=0,
        _pending_frontend_initialization=None, _pending_trial_contact=None,
        _ground_truth_imu_positions=positions)
    node = SimpleNamespace(
        system=system, tracker=tracker, ground_truth_imu_pose_sub=object(),
        destroy_subscription=lambda _sub: events.append(('delete', threading.get_ident())) or True,
        get_logger=lambda: SimpleNamespace(info=lambda _text: None))
    node._initialization_imu_gt_release_ready = lambda: (
        TrackerNode._initialization_imu_gt_release_ready(node))

    def pending_gt():
        positions.append((2., (1., 0., 0.)))
        events.append(('gt_input', threading.get_ident()))

    system.start()
    try:
        assert system.submit_input(pending_gt)
        assert GelSlamNode._release_initialization_imu_gt(node)
        assert [name for name, _ in events] == ['gt_input', 'delete']
        assert events[0][1] == events[1][1] != threading.get_ident()
        assert node.ground_truth_imu_pose_sub is None and positions == []
    finally:
        system.stop()
