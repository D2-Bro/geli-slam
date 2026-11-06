"""Retry low initial overlap with an anchor-State seed at the target time."""
from types import SimpleNamespace

import numpy as np
import pytest

import gelslam.core.tracker as tm
from gelslam.core.imu_navigation import gelsight_pose
from test_frontend_fallback_statistics import FlatVisual
from test_tracker_inertial import header, make_failing_contact_tracker


def assert_same_state(actual, expected):
    assert actual.timestamp == expected.timestamp
    for name in ("pose", "velocity", "accel_bias", "gyro_bias"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def retry_tracker(monkeypatch, counts=(9, 10), initialize=True):
    tracker, previous = make_failing_contact_tracker(
        monkeypatch, already_keyframe=True, initialize_from_imu=initialize)
    solver = tracker.frontend_visual_inertial
    logs, visuals = [], []
    tracker.logger = SimpleNamespace(info=logs.append)
    tracker._record_normal_correspondences = lambda *a, **k: None
    point_counts = iter(counts)

    def visual(reference, target, ppmm, reference_pose, target_pose, **kwargs):
        residual = FlatVisual(next(point_counts))
        visuals.append((reference, target, reference_pose.copy(), target_pose.copy(), residual))
        return residual

    monkeypatch.setattr(tm, "DenseNormalResidual", visual)
    monkeypatch.setattr(tm, "normalflow_quality", lambda *a: (1., 1.))
    return tracker, previous, solver, visuals, logs


def run_target(tracker):
    return tracker._refine_normalflow_with_imu(
        tracker.ref_T_prev, header(1.04), tracker.ref_frame, tracker.prev_frame)


@pytest.mark.parametrize("mode", ["ordinary", "recovery_reference"])
def test_retry_rebuilds_target_seed_and_reuses_the_vi_solver(monkeypatch, mode):
    tracker, previous, solver, visuals, logs = retry_tracker(monkeypatch)
    original = previous.copy()
    samples = tracker.imu_buffer.samples_covering(previous.timestamp, 1.04)
    predicted, _ = solver.predict_current(previous, samples, tracker.gelsight_T_imu)
    if mode == "recovery_reference":
        # An IMU-only reference stays a variable of the same contact solve.
        tracker._reference_is_imu_only = True
    method = "refine_current"
    solve = getattr(solver, method)
    attempts, consistency_predictions, commits = [], [], []

    def capture(visual, reference, anchor, target, interval, extrinsic, **kwargs):
        attempts.append((visual, reference.copy(), anchor, target.copy(), interval,
                         extrinsic, kwargs["preintegrated"]))
        return solve(visual, reference, anchor, target, interval, extrinsic, **kwargs)

    distance = solver.prediction_pose_distance
    def capture_distance(prediction, candidate, prior):
        consistency_predictions.append(prediction.copy())
        return distance(prediction, candidate, prior)

    commit = solver.commit_marginal_prior
    def capture_commit(state):
        commits.append(state)
        commit(state)

    monkeypatch.setattr(solver, method, capture)
    monkeypatch.setattr(solver, "prediction_pose_distance", capture_distance)
    monkeypatch.setattr(solver, "commit_marginal_prior", capture_commit)
    run_target(tracker)

    assert len(attempts) == len(visuals) == 2
    first, retry = attempts
    assert first[0] is visuals[0][-1] and retry[0] is visuals[1][-1]
    assert_same_state(first[3], predicted)
    seed = original.copy()
    seed.timestamp = 1.04
    assert_same_state(retry[3], seed)
    assert_same_state(previous, original)
    assert first[2] is retry[2] is previous
    assert first[4] is retry[4]  # Same raw interval, including both endpoints.
    assert first[4][0].timestamp == 1. and first[4][-1].timestamp == 1.04
    assert first[5] is retry[5] is tracker.gelsight_T_imu
    assert first[6] is retry[6]  # Keep the original IMU preintegration.
    np.testing.assert_array_equal(first[1], retry[1])
    for reference, target, reference_pose, target_pose, _ in visuals:
        assert reference is tracker.ref_frame and target is tracker.prev_frame
    np.testing.assert_array_equal(visuals[0][2], visuals[1][2])
    np.testing.assert_allclose(visuals[1][3], gelsight_pose(seed.pose, tracker.gelsight_T_imu))
    assert len(consistency_predictions) == len(commits) == 1
    assert_same_state(consistency_predictions[0], predicted)
    np.testing.assert_allclose(tracker._frontend_vi_state.pose, predicted.pose, atol=1e-10)
    assert solver.marginal_prior.matches(tracker._frontend_vi_state)
    assert solver.pending_marginal_prior is None
    counts = tracker.frontend_tracking_statistics()
    assert counts["refine_calls"] == counts["vi_accepted"] == 1
    assert counts["vi_solve_calls"] == 2
    assert counts["previous_state_retry_calls"] == counts["previous_state_retry_accepted"] == 1
    assert counts["frontend_failed_calls"] == counts["initial_shared_points_lt_10"] == 0
    assert any("anchor=1.000000000 s, target=1.040000000 s, shared points=9 -> 10" in line
               for line in logs)


@pytest.mark.parametrize("failure", ["initial_shared_points_lt_10", "invalid_height_residual"])
def test_failed_retry_preserves_anchor_and_prior_and_does_not_loop(monkeypatch, failure):
    tracker, previous, solver, visuals, _ = retry_tracker(monkeypatch, counts=(9, 9))
    original = previous.copy()
    prior = solver.marginal_prior
    solve = solver.refine_current
    calls = []

    def reject(*args, **kwargs):
        calls.append(args[3].copy())
        if len(calls) == 2 and failure != "initial_shared_points_lt_10":
            solver.last_refinement_failure = failure
            return None
        return solve(*args, **kwargs)

    monkeypatch.setattr(solver, "refine_current", reject)
    expected = (failure if failure == "initial_shared_points_lt_10"
                else "vi_refinement_failed_other")
    with pytest.raises(tm.LoseTrackError, match=expected):
        run_target(tracker)
    assert len(calls) == len(visuals) == 2
    assert tracker._frontend_vi_state is previous
    assert_same_state(previous, original)
    assert solver.marginal_prior is prior and solver.pending_marginal_prior is None
    counts = tracker.frontend_tracking_statistics()
    assert counts["previous_state_retry_calls"] == counts[expected] == counts["frontend_failed_calls"] == 1
    assert counts["vi_solve_calls"] == 2
    assert counts["previous_state_retry_accepted"] == counts["vi_accepted"] == 0


@pytest.mark.parametrize("rejection", ["ccs", "imu_consistency"])
def test_retry_must_pass_quality_before_committing_state_or_prior(monkeypatch, rejection):
    tracker, previous, solver, visuals, _ = retry_tracker(monkeypatch)
    original = previous.copy()
    prior = solver.marginal_prior
    if rejection == "ccs":
        monkeypatch.setattr(tm, "normalflow_quality", lambda *a: (1., .1))
    else:
        monkeypatch.setattr(solver, "prediction_pose_distance", lambda *a: 23.)
    with pytest.raises(tm.QualityGateError):
        run_target(tracker)
    assert len(visuals) == 2
    assert tracker._frontend_vi_state is previous
    assert_same_state(previous, original)
    assert solver.marginal_prior is prior and solver.pending_marginal_prior is None
    counts = tracker.frontend_tracking_statistics()
    assert counts["previous_state_retry_calls"] == counts["vi_quality_rejected"] == 1
    assert counts["previous_state_retry_accepted"] == counts["vi_accepted"] == 0


def test_previous_state_initialization_does_not_repeat_the_same_seed(monkeypatch):
    tracker, previous, solver, visuals, _ = retry_tracker(
        monkeypatch, counts=(9,), initialize=False)
    with pytest.raises(tm.LoseTrackError, match="initial_shared_points_lt_10"):
        run_target(tracker)
    assert len(visuals) == 1
    counts = tracker.frontend_tracking_statistics()
    assert counts["vi_solve_calls"] == counts["frontend_failed_calls"] == 1
    assert counts["previous_state_retry_calls"] == 0


def test_other_solver_failure_does_not_retry(monkeypatch):
    tracker, previous, solver, visuals, _ = retry_tracker(monkeypatch, counts=(10,))
    def reject(*args, **kwargs):
        solver.last_refinement_failure = "invalid_height_residual"
        return None
    monkeypatch.setattr(solver, "refine_current", reject)
    with pytest.raises(tm.LoseTrackError, match="vi_refinement_failed_other"):
        run_target(tracker)
    assert len(visuals) == 1
    assert tracker.frontend_tracking_statistics()["previous_state_retry_calls"] == 0
