"""Window keyframe selection (kernel_score): contact kernels, selection, retained window poses."""
import collections
from types import SimpleNamespace

import numpy as np
import pytest

from gelslam.core.gtsam_vi import create_frontend_solver
from gelslam.core.inertial import ImuSample
from gelslam.core.keyframe_selection import (
    WindowFrame, contact_moments, kernel_overlap_ratio, projected_moments, resolve_keyframe_selection,
    select_window_frame,
)
from gelslam.core.numpy_vi import NavigationState
from gelslam.core.retained_prior import VariableBlock
from gelslam.core.split_pose import split_pose_plus

MM_PER_PIXEL = 0.0634
SHAPE = (240, 320)
SETTINGS = dict(method="kernel_score", window_size=8, kernel_scale=4.0, covariance_epsilon_px2=0.001,
                kernel_ellipse_vertices=32, kernel_score_alpha=0.5)


# ---------------------------------------------------------------- settings
def test_config_current_is_disabled_and_unknown_or_missing_keys_fail():
    assert resolve_keyframe_selection(dict(keyframe_selection=dict(SETTINGS, method="current"))) is None
    assert resolve_keyframe_selection(dict(keyframe_selection=SETTINGS))["method"] == "kernel_score"
    with pytest.raises(ValueError, match="unknown"):
        resolve_keyframe_selection(dict(keyframe_selection=dict(SETTINGS, alpha=0.5)))
    with pytest.raises(ValueError, match="missing"):
        resolve_keyframe_selection(dict(keyframe_selection={k: v for k, v in SETTINGS.items()
                                                            if k != "kernel_score_alpha"}))
    assert resolve_keyframe_selection({}) is None
    with pytest.raises(ValueError, match="mapping"):
        resolve_keyframe_selection(dict(keyframe_selection=["kernel_score"]))
    with pytest.raises(ValueError, match="method must be one of"):
        resolve_keyframe_selection(dict(keyframe_selection=dict(SETTINGS, method="mean_shift")))


@pytest.mark.parametrize("key,value,message", [
    ("kernel_score_alpha", 1.2, "kernel_score_alpha must be a number in"),
    ("window_size", 1, "window_size must be an integer >= 2"),
    ("kernel_scale", 0.0, "kernel_scale must be a positive number"),
    ("covariance_epsilon_px2", float("nan"), "covariance_epsilon_px2 must be a positive number"),
    ("kernel_ellipse_vertices", 4, "kernel_ellipse_vertices must be an integer >= 8"),
])
def test_settings_are_validated(key, value, message):
    with pytest.raises(ValueError, match=message):
        resolve_keyframe_selection(dict(keyframe_selection=dict(SETTINGS, **{key: value})))


# ---------------------------------------------------------------- contact kernels
def disk_frame(fid, centre_px, radius_px, depth=None):
    """Window frame with a disk contact at centre (u, v); depth: indentation map (pixels)."""
    vv, uu = np.mgrid[:SHAPE[0], :SHAPE[1]]
    mask = (uu - centre_px[0]) ** 2 + (vv - centre_px[1]) ** 2 <= radius_px ** 2
    height = np.zeros(SHAPE) if depth is None else -depth
    area, mean, cov = contact_moments(mask, height, MM_PER_PIXEL)
    return WindowFrame(fid, float(fid), None, None, None, SHAPE, area, mean, cov)


def _kernel(frame):
    return projected_moments(frame, np.eye(4), frame, np.eye(4), SETTINGS, MM_PER_PIXEL)


def test_disk_moments_put_the_kernel_edge_on_the_contact_boundary():
    frame = disk_frame(1, (160.5, 120.5), 30.0)
    settings = dict(SETTINGS, covariance_epsilon_px2=1e-9)
    mu, h_inv = projected_moments(frame, np.eye(4), frame, np.eye(4), settings, MM_PER_PIXEL)
    np.testing.assert_allclose(mu, [160.5, 120.5], atol=1e-6)
    for direction in ([1., 0.], [0., 1.], [.6, .8]):
        edge = 30.0 * np.asarray(direction)
        assert edge @ h_inv @ edge == pytest.approx(1.0, rel=0.03)


def test_indentation_weights_pull_the_mean_to_the_deeper_side():
    vv, uu = np.mgrid[:SHAPE[0], :SHAPE[1]]
    depth = np.where(uu > 160, 2.0, 1.0)          # right half pressed twice as deep
    depth[(uu - 160) ** 2 + (vv - 120) ** 2 > 30 ** 2] = 0.0
    frame = disk_frame(1, (160, 120), 30.0, depth)
    assert frame.mean3d[0] > 0.002 * MM_PER_PIXEL   # metres; to the right of the centre


def test_projected_moments_follow_the_relative_pose():
    vv, uu = np.mgrid[:SHAPE[0], :SHAPE[1]]
    frame = disk_frame(1, (160, 120), 20.0)
    # Elongate along u: two touching disks.
    mask = ((uu - 150) ** 2 + (vv - 120) ** 2 <= 400) | ((uu - 170) ** 2 + (vv - 120) ** 2 <= 400)
    frame.mask_area, frame.mean3d, frame.cov3d = contact_moments(mask, np.zeros(SHAPE), MM_PER_PIXEL)
    current = disk_frame(2, (160, 120), 20.0)
    pose = np.eye(4)
    pose[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]   # 90 deg about the normal
    pose[0, 3] = 10 * MM_PER_PIXEL / 1000.0                       # 10 px along u
    mu, h_inv = projected_moments(frame, pose, current, np.eye(4), SETTINGS, MM_PER_PIXEL)
    # Mean at pixel (160, 120) = centred (0.5, 0.5); rotated (-0.5, 0.5); +10 px along u.
    np.testing.assert_allclose(mu, [169., 120.], atol=0.05)
    # After the rotation the long axis is v: a step along v is "shorter" in Mahalanobis terms.
    assert h_inv[1, 1] < h_inv[0, 0]


def test_kernel_overlap_ratio_of_equal_far_and_half_overlapping_ellipses():
    a, b = _kernel(disk_frame(1, (160, 120), 30.0)), _kernel(disk_frame(2, (160, 120), 30.0))
    far = _kernel(disk_frame(3, (60, 120), 30.0))
    assert kernel_overlap_ratio(*b, *a, 64) == pytest.approx(1.0, abs=1e-4)
    assert kernel_overlap_ratio(*far, *a, 64) == 0.0
    # Two equal disks one radius apart: lens / disk = (2 acos(1/2) - sqrt(3)/2) / pi = 0.391.
    side = _kernel(disk_frame(4, (190, 120), 30.0))
    assert 0.35 < kernel_overlap_ratio(*side, *a, 64) < 0.43


# ---------------------------------------------------------------- kernel_score
def test_kernel_score_keeps_t_among_similar_contacts_and_skips_a_far_peak():
    # Similar areas around t, a slightly larger frame shifted away: O dominates, t (O = 1) wins.
    window = [disk_frame(1, (190, 120), 31.0), disk_frame(2, (175, 120), 30.5), disk_frame(3, (160, 120), 30.0)]
    selected, info = select_window_frame(window, [np.eye(4)] * 3, SETTINGS, MM_PER_PIXEL)
    assert selected is window[-1] and info["status"] == "kernel_score"
    scores = {fid: sc for fid, _, _, sc in info["frames"]}
    assert scores[3] > scores[2] > scores[1] and info["selected_overlap"] == 1.0


def test_kernel_score_prefers_a_large_covering_frame_when_t_is_weak():
    # Weak t inside a large earlier contact: area dominates, the covering frame wins.
    window = [disk_frame(1, (160, 120), 45.0), disk_frame(2, (163, 121), 30.0), disk_frame(3, (165, 122), 8.0)]
    selected, info = select_window_frame(window, [np.eye(4)] * 3, SETTINGS, MM_PER_PIXEL)
    overlap = {fid: o for fid, _, o, _ in info["frames"]}
    assert selected.fid == 1 and overlap[1] == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("alpha,expected", [(1.0, 1), (0.0, 3)])
def test_kernel_score_limits_are_area_only_and_overlap_only(alpha, expected):
    # alpha 1: the area peak, even without overlap; alpha 0: the most recent of the O = 1 frames (t).
    window = [disk_frame(1, (60, 120), 40.0), disk_frame(2, (160, 120), 35.0), disk_frame(3, (160, 120), 30.0)]
    settings = dict(SETTINGS, kernel_score_alpha=alpha)
    selected, _ = select_window_frame(window, [np.eye(4)] * 3, settings, MM_PER_PIXEL)
    assert selected.fid == expected


def test_kernel_score_keeps_t_without_contact_moments():
    window = [disk_frame(1, (160, 120), 40.0), disk_frame(2, (160, 120), 30.0)]
    window[-1].mean3d = None
    selected, info = select_window_frame(window, [np.eye(4)] * 2, SETTINGS, MM_PER_PIXEL)
    assert selected is window[-1] and info["status"] == "current_without_contact_moments"


# ---------------------------------------------------------------- solver
class RelativePoseVisual:
    """30 rows: relative pose (rotation vector, translation) minus a constant measurement."""

    def __init__(self, measurement):
        self.measurement = np.asarray(measurement, float)

    def __len__(self):
        return 10

    def _residual(self, reference, target):
        relative = np.linalg.inv(reference) @ target
        rotation = relative[:3, :3]
        angle = np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))
        axis = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0],
                         rotation[1, 0] - rotation[0, 1]]) / 2
        rotvec = axis if angle < 1e-9 else axis * angle / np.sin(angle)
        return np.tile(np.r_[rotvec, relative[:3, 3]] - self.measurement, 5)

    def evaluate(self, reference, target):
        r = self._residual(reference, target)
        h, Jr, Jt = 1e-7, np.zeros((30, 6)), np.zeros((30, 6))
        for k in range(6):
            e = np.zeros(6)
            e[k] = h
            Jr[:, k] = (self._residual(split_pose_plus(reference, e), target) - r) / h
            Jt[:, k] = (self._residual(reference, split_pose_plus(target, e)) - r) / h
        return r, Jr, Jt


def make_solver():
    return create_frontend_solver(dict(gtsam_vi_max_iterations=3, gtsam_vi_min_iterations=3,
                                       frontend_reference_linear_system="hessian"))


def step(solver, previous, visual):
    samples = tuple(ImuSample(float(t), [0., 0., 9.81], [0., 0., .2])
                    for t in np.linspace(previous.timestamp, previous.timestamp + .04, 5))
    current, pim = solver.predict_current(previous, samples, np.eye(4))
    result = solver.refine_current(visual, np.eye(4), previous, current, samples, np.eye(4),
                                   preintegrated=pim)
    assert result is not None, solver.last_refinement_failure
    solver.commit_marginal_prior(result)
    return result


def test_retained_window_poses_do_not_change_the_current_solution():
    plain, window = make_solver(), make_solver()
    s_plain = s_window = NavigationState(1., np.eye(4), np.zeros(3), np.zeros(3), np.zeros(3))
    visual = RelativePoseVisual([0., 0., 0.01, 1e-3, 0., 0.])
    requested = []
    for index in range(10):
        if index == 4:
            # Keyframe at the newest frame: both solvers move the reference there.
            plain.set_reference(s_plain.timestamp, s_plain)
            requested = []
            window.window_pose_timestamps = ()
            window.set_reference(s_window.timestamp, s_window)
        s_plain = step(plain, s_plain, visual)
        s_window = step(window, s_window, visual)
        np.testing.assert_allclose(s_window.pose, s_plain.pose, atol=1e-9)
        np.testing.assert_allclose(s_window.velocity, s_plain.velocity, atol=1e-8)
        np.testing.assert_allclose(s_window.gyro_bias, s_plain.gyro_bias, atol=1e-9)
        requested = (requested + [s_window.timestamp])[-3:]
        window.window_pose_timestamps = tuple(requested)
        window.set_reference(window.reference_id.timestamp, s_window)
        past = len([t for t in requested if t != s_window.timestamp])
        expected = plain.marginal_prior.layout.dimension + 6 * past
        assert window.marginal_prior.layout.dimension == expected
    # Move the reference to a retained past pose; the next solves still succeed.
    past = requested[0]
    window.window_pose_timestamps = tuple(t for t in requested if t > past)
    window.set_reference(past, s_window)
    assert window.reference_id == VariableBlock("pose", past)
    for _ in range(3):
        s_window = step(window, s_window, visual)
        assert np.isfinite(s_window.pose).all()
    with pytest.raises(ValueError, match="already marginalized"):
        plain.set_reference(past, s_plain)


# ---------------------------------------------------------------- tracker
def header(t):
    return SimpleNamespace(stamp=SimpleNamespace(sec=int(t), nanosec=round((t - int(t)) * 1e9)))


def contact(size):
    mask = np.zeros(SHAPE, bool)
    mask[120 - size:120 + size, 160 - size:160 + size] = True
    return SimpleNamespace(C=mask, H=np.zeros(SHAPE), is_contacted=True)


def window_tracker(monkeypatch, sizes, times=(1.1, 1.2, 1.3, 1.6), reference_size=10, **settings):
    """Tracker with reference K_0 at 1.0 s and accepted frames 1..n; no motion."""
    import gelslam.core.tracker as tm
    from gelslam.core.tracker import Tracker

    references, poses = [], {}

    class Prior:
        @property
        def values(self):
            return {VariableBlock("pose", t): p for t, p in poses.items()}

    solver = SimpleNamespace(marginal_prior=Prior(), gravity=np.array([0., 0., -9.81]),
                             window_pose_timestamps=(), reference_marginalization=True,
                             set_reference=lambda t, s: references.append((t, s.timestamp)))
    t = object.__new__(Tracker)
    t.track_config = dict(keyframe_max_interval_s=.5, contact_threshold=10)
    t.imu_config = {}
    t.frontend_visual_inertial = solver
    t.gelsight_T_imu = np.eye(4)
    t.ppmm = MM_PER_PIXEL
    t.logger = SimpleNamespace(info=lambda _: None)
    t.bridge = object()
    t._keyframe_selection = resolve_keyframe_selection(dict(keyframe_selection=dict(SETTINGS, **settings)))
    t._keyframe_window = collections.deque(maxlen=t._keyframe_selection["window_size"])
    t.trial_T_ref = np.eye(4)
    t.ref_header, t.ref_frame = header(1.), contact(reference_size)
    t.curr_kid, t.curr_fid = 0, 0
    t._last_keyframe_timestamp = 1.
    t._static_imu_settings = None
    monkeypatch.setattr(tm, "create_keyframe_msg",
                        lambda bridge, frame, **kw: dict(kw, surface=frame))
    stamps = {}
    for fid, (time, size) in enumerate(zip(times, sizes), start=1):
        t.curr_fid = fid
        stamps[fid] = t._imu_timestamp(header(time))
        t._frontend_vi_state = NavigationState(stamps[fid], np.eye(4), np.zeros(3),
                                               np.zeros(3), np.zeros(3))
        poses[stamps[fid]] = np.eye(4)
        t.prev_frame, t.prev_header = contact(size), header(time)
        t._push_keyframe_window(t.prev_frame, t.prev_header)
    t.ref_T_prev = np.eye(4, dtype=np.float32)
    return t, solver, references, stamps


def test_tracker_makes_the_selected_past_frame_the_keyframe(monkeypatch):
    # Concentric contacts, largest at frame 2 (1.2 s): every larger ellipse holds E_t (O = 1),
    # so frame 2 scores highest at the Timed trigger at frame 4 (1.6 s).
    t, solver, references, stamps = window_tracker(monkeypatch, (30, 50, 40, 20))
    logs = []
    t.logger = SimpleNamespace(info=logs.append)
    assert solver.window_pose_timestamps == tuple(stamps[k] for k in (1, 2, 3, 4))
    keyframe = t._maybe_add_timed_keyframe(t.prev_frame, t.prev_header, "test")
    assert keyframe["fid"] == 2 and keyframe["kid"] == 1
    assert t.latest_keyframe_selection["status"] == "kernel_score"
    assert any("Timed Window Keyframe" in line and "selected kernel overlap: 1.000" in line for line in logs)
    assert t.curr_kid == 1 and t.ref_header.stamp.nanosec == round(.2 * 1e9)
    assert t._last_keyframe_timestamp == pytest.approx(1.2)
    assert references[-1] == (pytest.approx(1.2), pytest.approx(1.6))
    # Candidates of the next selection are later than the new reference.
    assert [f.fid for f in t._keyframe_window] == [3, 4]
    assert solver.window_pose_timestamps == (stamps[3], stamps[4])
    assert t.frontend_tracking_statistics()["window_keyframes"] == 1


def test_tracker_keeps_t_when_it_scores_highest(monkeypatch):
    t, solver, references, stamps = window_tracker(monkeypatch, (20, 20, 20, 20))
    keyframe = t._maybe_add_timed_keyframe(t.prev_frame, t.prev_header, "test")
    assert keyframe["fid"] == 4 and t.ref_frame is t.prev_frame
    assert t.frontend_tracking_statistics()["window_keyframes"] == 0


def test_window_frames_without_retained_poses_are_not_candidates(monkeypatch):
    t, solver, references, stamps = window_tracker(monkeypatch, (30, 50, 40, 20))
    values = solver.marginal_prior.values
    solver.marginal_prior = SimpleNamespace(values={k: v for k, v in values.items()
                                                    if k.timestamp != stamps[2]})
    keyframe = t._maybe_add_timed_keyframe(t.prev_frame, t.prev_header, "test")
    assert keyframe["fid"] == 3


def test_contact_loss_empties_the_window(monkeypatch):
    t, solver, references, stamps = window_tracker(monkeypatch, (30, 50, 40, 20))
    t._trim_keyframe_window(None)
    assert not t._keyframe_window and solver.window_pose_timestamps == ()
