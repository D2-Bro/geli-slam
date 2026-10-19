"""Reject failed frontend calls without a GelSight-only fallback."""
from types import SimpleNamespace

import numpy as np
import pytest

from gelslam.core.gtsam_vi import GtsamVisualInertialSolver
from gelslam.core.inertial import ImuBuffer, ImuSample
from gelslam.core.numpy_vi import NavigationState
from gelslam.core.tracker import Tracker, QualityGateError, LoseTrackError
import gelslam.core.tracker as tm
import gelslam.nodes.tracker_node as node_module


def state(t):
    return NavigationState(t, np.eye(4), np.zeros(3), np.zeros(3), np.zeros(3))


class FlatVisual:
    def __init__(self, count=10):
        self.count = count

    def __len__(self):
        return self.count

    def evaluate(self, *_args):
        return np.zeros(30), np.zeros((30, 6)), np.zeros((30, 6))


def test_low_shared_points_reason_uses_actual_gate_and_resets_on_next_solve():
    solver = GtsamVisualInertialSolver({"gtsam_vi_max_iterations": 1})
    samples = [ImuSample(t, (0, 0, 9.81), (0, 0, 0)) for t in (1.0, 1.1)]
    args = (np.eye(4), state(1.0), state(1.1), samples, np.eye(4))
    assert solver.refine_current(FlatVisual(9), *args) is None
    assert solver.last_refinement_failure == "initial_shared_points_lt_10"
    assert solver.refine_current(FlatVisual(10), *args) is not None
    assert solver.last_refinement_failure is None
    assert solver.refine_current(None, *args) is None
    assert solver.last_refinement_failure is None


def make_tracker(monkeypatch):
    tracker = object.__new__(Tracker)
    tracker.logger = SimpleNamespace(info=lambda _message: None)
    tracker.imu_config = {}
    tracker.gelsight_T_imu = np.eye(4)
    tracker.imu_buffer = ImuBuffer()
    for t in (1.0, 1.1):
        tracker.imu_buffer.add(ImuSample(t, (0, 0, 9.81), (0, 0, 0)))
    tracker._frontend_vi_state = state(1.0)
    tracker.trial_T_ref = np.eye(4)
    tracker.ref_frame = object()
    tracker.ppmm = 0.059
    tracker.track_config = {"scr_threshold": 0.3, "ccs_threshold": 0.7}
    tracker._record_normal_correspondences = lambda *_args, **_kwargs: None
    tracker.frontend_visual_inertial = SimpleNamespace(
        predict_current=lambda *_args: (state(1.1), object()),
        refine_current=lambda *_args, **_kwargs: None,
        last_refinement_failure="initial_shared_points_lt_10",
    )
    monkeypatch.setattr(tm, "DenseNormalResidual", lambda *_args, **_kwargs: FlatVisual())
    return tracker


def refine(tracker):
    header = SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=100000000))
    return tracker._refine_normalflow_with_imu(np.eye(4), header, tracker.ref_frame, object())


@pytest.mark.parametrize("reason", [
    "vi_unavailable", "imu_prediction_unavailable",
    "initial_shared_points_lt_10", "vi_refinement_failed_other",
])
def test_failure_reasons_reject_without_a_tactile_only_fallback(monkeypatch, reason):
    tracker = make_tracker(monkeypatch)
    previous = tracker._frontend_vi_state
    old_pose = previous.pose.copy()
    if reason == "vi_unavailable":
        tracker.frontend_visual_inertial = None
    elif reason == "imu_prediction_unavailable":
        tracker.imu_buffer = ImuBuffer()
    elif reason == "vi_refinement_failed_other":
        tracker.frontend_visual_inertial.last_refinement_failure = None
    with pytest.raises(LoseTrackError, match=reason):
        refine(tracker)
    assert tracker._frontend_vi_state is previous
    assert np.array_equal(previous.pose, old_pose)
    assert tracker.latest_normal_correspondences["status"] == "Frontend failed: " + reason
    counts = tracker.frontend_tracking_statistics()
    assert counts[reason] == counts["frontend_failed_calls"] == counts["refine_calls"] == 1
    assert counts["normalflow_only_calls"] == counts["normalflow_only_accepted"] == 0
    assert counts["vi_accepted"] == counts["imu_prediction_only_accepted"] == 0
    reasons = ("vi_unavailable", "imu_prediction_unavailable",
               "initial_shared_points_lt_10", "vi_refinement_failed_other")
    assert sum(counts[key] for key in reasons) == 1


def test_failure_discards_pending_posterior_and_preserves_accepted_prior(monkeypatch):
    tracker = make_tracker(monkeypatch)
    solver = tracker.frontend_visual_inertial
    accepted_prior = object()
    solver.marginal_prior = accepted_prior
    solver.pending_marginal_prior = object()
    solver.discard_pending_prior = lambda: setattr(solver, 'pending_marginal_prior', None)
    with pytest.raises(LoseTrackError):
        refine(tracker)
    assert solver.marginal_prior is accepted_prior
    assert solver.pending_marginal_prior is None


def test_tactile_only_fallback_implementation_is_removed():
    assert not hasattr(Tracker, '_run_normalflow_fallback')
    assert not hasattr(Tracker, '_track_normalflow_only')
    assert not hasattr(tm, 'normalflow')


def test_vi_quality_rejection_is_not_a_fallback(monkeypatch):
    tracker = make_tracker(monkeypatch)
    tracker.frontend_visual_inertial.refine_current = lambda *_args, **_kwargs: state(1.1)
    monkeypatch.setattr(tm, "normalflow_quality", lambda *_args: (0.1, 1.0))
    with pytest.raises(QualityGateError):
        refine(tracker)
    counts = tracker.frontend_tracking_statistics()
    assert counts["vi_solve_calls"] == counts["vi_quality_rejected"] == 1
    assert counts["normalflow_only_calls"] == counts["vi_accepted"] == 0


def test_accepted_vi_result_is_separate_from_normalflow_only(monkeypatch):
    tracker = make_tracker(monkeypatch)
    tracker.frontend_visual_inertial.refine_current = lambda *_args, **_kwargs: state(1.1)
    monkeypatch.setattr(tm, "normalflow_quality", lambda *_args: (1.0, 1.0))
    refine(tracker)
    counts = tracker.frontend_tracking_statistics()
    assert counts["vi_solve_calls"] == counts["vi_accepted"] == 1
    assert counts["normalflow_only_calls"] == counts["vi_quality_rejected"] == 0


def test_missing_images_reject_without_a_normalflow_solve(monkeypatch):
    tracker = make_tracker(monkeypatch)
    with pytest.raises(LoseTrackError, match='vi_unavailable'):
        tracker._refine_normalflow_with_imu(np.eye(4), None, None, None)
    assert tracker.frontend_tracking_statistics()["normalflow_only_calls"] == 0


def test_counts_survive_solver_replacement_and_summary_includes_zeros(monkeypatch):
    tracker = make_tracker(monkeypatch)
    with pytest.raises(LoseTrackError):
        refine(tracker)
    tracker.frontend_visual_inertial = SimpleNamespace(
        predict_current=lambda *_args: (state(1.1), object()),
        refine_current=lambda *_args, **_kwargs: None,
        last_refinement_failure="initial_shared_points_lt_10",
    )
    with pytest.raises(LoseTrackError):
        refine(tracker)
    logs = []
    tracker.logger = SimpleNamespace(info=logs.append)
    tracker.log_frontend_tracking_summary()
    assert len(logs) == 1
    assert "initial_shared_points_lt_10=2" in logs[0]
    assert "frontend_failed_calls=2" in logs[0]
    assert "normalflow_only_calls=0" in logs[0]
    assert "imu_prediction_unavailable=0" in logs[0]
    assert "retries counted separately" in logs[0]


@pytest.mark.parametrize("spin_error", [False, True])
def test_shutdown_waits_for_callbacks_before_node_destruction(monkeypatch, spin_error):
    events = []
    node = SimpleNamespace(destroy_node=lambda: events.append("destroy"))
    def spin_once(timeout_sec):
        assert timeout_sec == 0.1
        if spin_error:
            raise RuntimeError("callback failed")
    executor = SimpleNamespace(add_node=lambda _node: None, spin_once=spin_once,
                               shutdown=lambda: events.append("executor_shutdown"))
    running = iter((True, False))
    monkeypatch.setattr(node_module.rclpy, "ok", lambda: next(running))
    monkeypatch.setattr(node_module.rclpy, "init", lambda **_kwargs: None)
    monkeypatch.setattr(node_module.rclpy, "try_shutdown", lambda: events.append("ros_shutdown"))
    monkeypatch.setattr(node_module, "TrackerNode", lambda: node)
    monkeypatch.setattr(node_module, "MultiThreadedExecutor", lambda **_kwargs: executor)
    if spin_error:
        with pytest.raises(RuntimeError, match="callback failed"):
            node_module.main()
    else:
        node_module.main()
    assert events == ["executor_shutdown", "destroy", "ros_shutdown"]
