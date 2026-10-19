"""Time selection is common to frontend modes and preserves navigation State."""
from types import SimpleNamespace

import numpy as np
import pytest

import gelslam.core.tracker as tm
from gelslam.core.gtsam_vi import GtsamVisualInertialSolver
from gelslam.core.marginal_prior import NavigationPrior
from gelslam.core.numpy_vi import NavigationState
from gelslam.core.tracker import Tracker, TrackingMode


def header(t):
    return SimpleNamespace(stamp=SimpleNamespace(sec=int(t), nanosec=round((t-int(t))*1e9)))


class Surface:
    contact = True
    def __init__(self, *args, **kwargs):
        self.is_contacted = self.contact
        self.N = self.C = self.H = self.L = np.zeros((1, 1))


def make_tracker(monkeypatch, mode='joint'):
    t = object.__new__(Tracker)
    t.track_config = dict(contact_threshold=1, keyframe_max_interval_s=.5)
    t.imu_config = dict(frontend_fix_previous_state=mode=='fixed',
                        frontend_imu_prediction_only=mode=='prediction')
    t.frontend_visual_inertial = GtsamVisualInertialSolver(t.imu_config)
    pose = np.eye(4)
    pose[0, 3] = .2
    t._frontend_vi_state = NavigationState(1., pose, np.array([.1, .2, .3]),
                                          np.ones(3)*.001, np.ones(3)*.0001)
    t.frontend_visual_inertial.marginal_prior = NavigationPrior.initial(
        t._frontend_vi_state, np.ones(15))
    t.trial_T_ref = pose.copy()
    t.ref_T_prev = np.eye(4, dtype=np.float32)
    t.ref_frame = t.prev_frame = Surface()
    t.ref_header = t.prev_header = header(1.)
    t.prev_in_contact_flag = t.prev_is_keyframe_flag = True
    t.curr_fid = t.prev_fid = 0
    t.curr_kid = 0
    t._last_keyframe_timestamp = 1.
    t.last_tactile_ref_T = np.eye(4)
    t.last_tactile_header = t.prev_header
    t.gelsight_T_imu = np.eye(4)
    t.bg_images = [None]*10
    t.ppmm = 1.
    t.contact_mask_config = dict(color_dist_threshold=1, height_threshold=1)
    t.recon = SimpleNamespace(get_surface_info=lambda *a, **k: (None, None, None))
    t.logger = SimpleNamespace(info=lambda _: None)
    t.bridge = object()
    t._static_imu_settings = None
    monkeypatch.setattr(tm, 'Frame', Surface)
    monkeypatch.setattr(tm, 'create_keyframe_msg', lambda bridge, frame, **kw: dict(kw, surface=frame))
    monkeypatch.setattr(tm, 'create_frame_msg', lambda bridge, frame, **kw: dict(kw, surface=frame))

    def accept(relative, h, *args, **kwargs):
        # Isolate keyframe selection from the tested frontend estimation paths.
        state = t._frontend_vi_state.copy()
        state.timestamp = t._header_timestamp(h)
        state.pose[0, 3] = .2 + .01*(state.timestamp-1.)
        t._frontend_vi_state = state
        return (np.linalg.inv(t.trial_T_ref) @ state.pose).astype(np.float32)
    t._refine_normalflow_with_imu = accept
    return t


@pytest.mark.parametrize('mode', ['joint', 'fixed', 'prediction'])
def test_common_deadline_promotes_current_contact_without_resetting_state(monkeypatch, mode):
    t = make_tracker(monkeypatch, mode)
    prior = t.frontend_visual_inertial.marginal_prior
    initial = t._frontend_vi_state.copy()
    image = np.zeros((1, 1, 3), np.uint8)
    for time, expected_kid, due in [(1.499999, 0, False), (1.5, 1, True),
                                     (1.999999, 1, False), (2., 2, True)]:
        valid, (frame, keyframes) = t.track(image, header(time))
        assert valid and len(keyframes)==int(due)
        assert t.curr_kid==frame['ref_kid']==expected_kid
        assert t.frontend_visual_inertial.marginal_prior is prior
        assert t._frontend_vi_state.timestamp==time
        for key in ('velocity', 'accel_bias', 'gyro_bias'):
            assert np.array_equal(getattr(t._frontend_vi_state,key),getattr(initial,key))
        assert np.allclose(t.trial_T_ref @ frame['ref_T_curr'], t._frontend_vi_state.pose)
        if due:
            k = keyframes[0]
            assert k['fid']==frame['fid']==t.curr_fid
            assert k['header'] is frame['header']
            assert not k['is_new_trial']
            assert np.allclose(k['trial_T_curr'], t._frontend_vi_state.pose)
            assert np.array_equal(frame['ref_T_curr'], np.eye(4))
            assert t.ref_frame is t.prev_frame is k['surface']
            assert t.prev_is_keyframe_flag
            assert t._last_keyframe_timestamp==time


@pytest.mark.parametrize('new_trial', [False, True])
def test_other_keyframes_restart_the_clock(monkeypatch, new_trial):
    t = make_tracker(monkeypatch)
    t._make_keyframe_msg(Surface(), 1, 1, new_trial, np.eye(4), header(1.4))
    assert t._last_keyframe_timestamp==1.4
    assert t._maybe_add_timed_keyframe(Surface(),header(1.8),'Test') is None
    assert t._maybe_add_timed_keyframe(Surface(),header(1.9),'Test') is not None


def test_no_contact_and_failed_targets_never_become_timed_keyframes(monkeypatch):
    t = make_tracker(monkeypatch)
    monkeypatch.setattr(Surface, 'contact', False)
    t._initialize_no_contact_coast = lambda: True
    t._coast_to_header = lambda *args: True
    t._coasting_frame_result = lambda *args: (True, (object(), []))
    valid, (_, keyframes) = t.track(np.zeros((1,1,3),np.uint8),header(2.))
    assert valid and keyframes==[] and t.curr_kid==0
    assert t._last_keyframe_timestamp==1.
    assert t._maybe_add_timed_keyframe(Surface(),header(2.),'Test') is None

    t = make_tracker(monkeypatch)
    monkeypatch.setattr(Surface, 'contact', True)
    def reject(*args, **kwargs):
        raise tm.LoseTrackError('missing IMU')
    t._refine_normalflow_with_imu = reject
    before = t._frontend_vi_state
    valid, emitted = t.track(np.zeros((1,1,3),np.uint8),header(2.))
    assert not valid and emitted is None
    assert t._frontend_vi_state is before and t.curr_kid==0
    assert t._last_keyframe_timestamp==1.


def test_recorded_camera_time_controls_deadline_not_header_or_imu_offset(monkeypatch):
    t = make_tracker(monkeypatch)
    t.timestamp_source='recorded_source'
    t.image_to_imu_time_offset_s=9.
    t._recorded_image_times={int(99.9*1e9):1.5}
    assert t._maybe_add_timed_keyframe(Surface(),header(99.9),'Test') is not None
    assert t._last_keyframe_timestamp==1.5


def test_disabled_missing_and_old_saved_clock(monkeypatch):
    t = make_tracker(monkeypatch)
    t.track_config['keyframe_max_interval_s']=0
    assert t._maybe_add_timed_keyframe(Surface(),header(2.),'Test') is None
    t.track_config['keyframe_max_interval_s']=.5
    assert t._maybe_add_timed_keyframe(Surface(),None,'Test') is None
    del t._last_keyframe_timestamp
    assert t._maybe_add_timed_keyframe(Surface(),header(1.5),'Test') is not None


@pytest.mark.parametrize('invalid',[-1, np.nan, np.inf, None, True, 'bad'])
def test_invalid_intervals_are_rejected(monkeypatch, invalid):
    t = make_tracker(monkeypatch)
    t.track_config['keyframe_max_interval_s']=invalid
    with pytest.raises(ValueError,match='keyframe_max_interval_s'):
        t._keyframe_interval_s()
