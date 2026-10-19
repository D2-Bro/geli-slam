"""Prediction-only contact tracking preserves IMU States without a solve."""
from types import SimpleNamespace

import gtsam
import numpy as np
import pytest

import gelslam.core.gtsam_vi as gv
import gelslam.core.tracker as tm
from gelslam.core.gtsam_vi import GtsamVisualInertialSolver
from gelslam.core.imu_navigation import gelsight_pose
from gelslam.core.inertial import ImuBuffer, ImuSample
from gelslam.core.marginal_prior import NavigationPrior
from gelslam.core.numpy_vi import NavigationState
from gelslam.core.tracker import Tracker, LoseTrackError


def header(t):
    return SimpleNamespace(stamp=SimpleNamespace(sec=int(t), nanosec=round((t-int(t))*1e9)))


def initial_state():
    pose = np.eye(4)
    pose[:3, :3] = gtsam.Rot3.Expmap(np.array([0., 0., .3])).matrix()
    pose[:3, 3] = [.1, -.2, .3]
    return NavigationState(1., pose, np.array([.01, -.02, .03]),
                           np.array([.003, -.002, .004]), np.array([.0002, -.0001, .00015]))


def assert_state_equal(a, b):
    assert a.timestamp == b.timestamp
    for key in ('pose', 'velocity', 'accel_bias', 'gyro_bias'):
        assert np.array_equal(getattr(a, key), getattr(b, key))


def make_tracker(monkeypatch, initialize=True):
    tracker = object.__new__(Tracker)
    tracker.imu_config = dict(frontend_imu_prediction_only=True,
                              frontend_initialize_from_imu_preintegration=initialize,
                              gtsam_vi_marginalization=True)
    tracker.frontend_visual_inertial = GtsamVisualInertialSolver(tracker.imu_config)
    tracker._frontend_vi_state = initial_state()
    tracker.imu_buffer = ImuBuffer()
    for t in np.linspace(1., 1.3, 31):
        tracker.imu_buffer.add(ImuSample(t, np.array([.4, 0., 9.81]) + initial_state().accel_bias,
                                        np.array([0., 0., .1]) + initial_state().gyro_bias))
    tracker.gelsight_T_imu = np.eye(4)
    tracker.gelsight_T_imu[:3, 3] = [.02, -.01, -.04]
    tracker.trial_T_ref = np.eye(4)
    tracker.trial_T_ref[:3, 3] = [-.1, .2, .1]
    tracker.ref_frame, tracker.prev_frame = object(), object()
    tracker.ref_header, tracker.prev_header = header(.9), header(1.)
    tracker.ppmm = .059
    tracker.track_config = dict(scr_threshold=.3, ccs_threshold=.7)
    tracker.logger = SimpleNamespace(info=lambda _: None)
    tracker._imu_coasting_ready = lambda: True

    def no_solve(*args, **kwargs):
        pytest.fail('Prediction-only tracking must not build or solve a visual factor')
    monkeypatch.setattr(tm, 'DenseNormalResidual', no_solve)
    monkeypatch.setattr(gv, 'solve_gaussian_qr', no_solve)
    monkeypatch.setattr(tm, 'normalflow_quality', no_solve)
    projected = []
    def project(ref, target, relative, ppmm):
        projected.append(relative.copy())
        return np.empty((0, 2)), np.empty((0, 2))
    monkeypatch.setattr(tm, 'project_normal_correspondences', project)
    return tracker, projected


@pytest.mark.parametrize('initialize', [False, True])
def test_prediction_is_kept_and_used_at_the_next_target_without_optimization(monkeypatch, initialize):
    tracker, projected = make_tracker(monkeypatch, initialize=initialize)
    solver = tracker.frontend_visual_inertial
    previous = tracker._frontend_vi_state
    solver.marginal_prior = NavigationPrior.initial(previous, np.ones(15))
    solver.marginal_prior.rhs[9] = 100.
    solver.pending_marginal_prior = solver.marginal_prior
    for target in (1.1, 1.2):
        samples = tracker.imu_buffer.samples_covering(previous.timestamp, target)
        # An ordinary IMU initializer is the independent expected output,
        # even when the ordinary initializer flag is false in the new mode.
        ordinary = GtsamVisualInertialSolver({})
        expected, _ = ordinary.predict_current(previous, samples, tracker.gelsight_T_imu)
        original = previous.copy()
        relative = tracker._refine_normalflow_with_imu(
            np.eye(4), header(target), tracker.ref_frame, object())
        assert_state_equal(tracker._frontend_vi_state, expected)
        assert_state_equal(previous, original)
        gel_pose = gelsight_pose(expected.pose, tracker.gelsight_T_imu)
        expected_relative = np.linalg.inv(tracker.trial_T_ref) @ gel_pose
        assert np.allclose(relative, expected_relative.astype(np.float32))
        assert projected == []
        assert tracker._frontend_vi_prev_timestamp == target
        assert np.array_equal(tracker._frontend_vi_prev_velocity, expected.velocity)
        assert np.array_equal(tracker._frontend_vi_prev_bias,
                              np.r_[expected.accel_bias, expected.gyro_bias])
        assert solver.marginal_prior is None and solver.pending_marginal_prior is None
        assert solver.last_optimization_trace == []
        assert solver.last_prior_source == 'imu_prediction_only'
        previous = tracker._frontend_vi_state
    counts = tracker.frontend_tracking_statistics()
    assert counts['imu_prediction_only_calls'] == counts['imu_prediction_only_accepted'] == 2
    assert counts['vi_solve_calls'] == counts['vi_accepted'] == counts['normalflow_only_calls'] == 0
    assert tracker.latest_normal_correspondences['status'] == 'IMU prediction pose: accepted; SCR/CCS skipped'


def test_prediction_does_not_require_quality_thresholds_or_correspondences(monkeypatch):
    tracker, projected = make_tracker(monkeypatch)
    tracker.track_config = {}  # No SCR/CCS thresholds are needed in this mode.
    def no_visual_processing(*args, **kwargs):
        pytest.fail('Prediction-only mode must not evaluate visual acceptance')
    tracker._record_normal_correspondences = no_visual_processing
    original = tracker._frontend_vi_state.copy()
    tracker._refine_normalflow_with_imu(np.eye(4), header(1.1), tracker.ref_frame, object())
    assert tracker._frontend_vi_state.timestamp == 1.1
    assert not np.array_equal(tracker._frontend_vi_state.pose, original.pose)
    assert projected == []
    counts = tracker.frontend_tracking_statistics()
    assert counts['imu_prediction_only_accepted'] == 1
    assert counts['imu_prediction_only_quality_rejected'] == counts['vi_solve_calls'] == 0


def test_nonfinite_prediction_is_still_rejected_without_replacing_previous_state(monkeypatch):
    tracker, _ = make_tracker(monkeypatch)
    previous = tracker._frontend_vi_state
    snapshot = previous.copy()
    bad_navstate = SimpleNamespace(pose=lambda: gtsam.Pose3(),
                                    velocity=lambda: np.array([np.nan, 0., 0.]))
    bad_pim = SimpleNamespace(predict=lambda *args: bad_navstate,
                              deltaTij=lambda: 0.1)
    monkeypatch.setattr(gv, 'preintegrate_imu', lambda *args: bad_pim)
    with pytest.raises(LoseTrackError, match='imu_prediction_unavailable'):
        tracker._refine_normalflow_with_imu(np.eye(4), header(1.1), tracker.ref_frame, object())
    assert tracker._frontend_vi_state is previous
    assert_state_equal(previous, snapshot)
    counts = tracker.frontend_tracking_statistics()
    assert counts['imu_prediction_only_accepted'] == counts['normalflow_only_calls'] == 0


def test_missing_imu_rejects_without_normalflow_fallback(monkeypatch):
    tracker, _ = make_tracker(monkeypatch)
    tracker.imu_buffer = ImuBuffer()
    original = tracker._frontend_vi_state.copy()
    with pytest.raises(LoseTrackError, match='imu_prediction_unavailable'):
        tracker._refine_normalflow_with_imu(np.eye(4), header(1.1), tracker.ref_frame, object())
    assert_state_equal(tracker._frontend_vi_state, original)
    assert tracker.frontend_tracking_statistics()['imu_prediction_only_calls'] == 0
    assert tracker.frontend_tracking_statistics()['normalflow_only_calls'] == 0


@pytest.mark.parametrize('invalid', ['true', 'false', None, 0, 1])
def test_option_requires_a_yaml_boolean(invalid):
    with pytest.raises(ValueError, match='frontend_imu_prediction_only must be a boolean'):
        GtsamVisualInertialSolver({'frontend_imu_prediction_only': invalid})


def test_omitted_and_false_options_keep_the_existing_mode():
    assert GtsamVisualInertialSolver({}).imu_prediction_only is False
    assert GtsamVisualInertialSolver({'frontend_imu_prediction_only': False}).imu_prediction_only is False
