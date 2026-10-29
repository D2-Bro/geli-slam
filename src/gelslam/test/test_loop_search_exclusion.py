"""Exclude recent keyframes before loop matching and registration."""
from types import SimpleNamespace

import numpy as np
import pytest

import gelslam.core.gtsam_pose_graph as module
from gelslam.core.gtsam_pose_graph import PoseGraph
from gelslam.core.local_mapping import LoopSearch
from gelslam.utils import Logger
from test_keyframe_handoff_regressions import search_config


def database(size, separate_trials=False):
    points = [SimpleNamespace(pt=(float(i), 0.)) for i in range(4)]
    return [SimpleNamespace(
        kid=10 + 7 * i, trial_group=i if separate_trials else 0, kp=points,
        des=np.full((4, 2), i, dtype=np.float32), ppmm=1.,
        N=np.full((4, 4, 3), i, dtype=np.float32),
        C=np.ones((4, 4), dtype=bool), H=np.zeros((4, 4)), L=np.zeros((4, 4)),
    ) for i in range(size)]


@pytest.fixture
def successful_matching(monkeypatch):
    matched, registered = [], []

    def match(reference, target):
        matched.append((int(reference[0, 0]), int(target[0, 0])))
        return [SimpleNamespace(queryIdx=i, trainIdx=i) for i in range(4)]

    def register(reference, _cr, _hr, _lr, target, *_args, **_kwargs):
        registered.append((int(reference[0, 0, 0]), int(target[0, 0, 0])))
        return np.eye(4)

    monkeypatch.setattr(module.cv2, 'BFMatcher',
                        lambda *_a, **_k: SimpleNamespace(match=match))
    monkeypatch.setattr(module.cv2, 'estimateAffinePartial2D',
                        lambda *_a, **_k: (np.eye(2, 3), np.ones((4, 1))))
    monkeypatch.setattr(module, 'normalflow', register)
    return matched, registered


@pytest.mark.parametrize('separate_trials', [False, True])
@pytest.mark.parametrize('target,expected', [
    (0, []), (1, []), (5, []), (6, [0]), (7, [0, 1]),
])
def test_default_excludes_previous_five_before_matching(
        successful_matching, target, expected, separate_trials):
    # Future DB entries and nonconsecutive kid values cannot change the window.
    db = database(12, separate_trials)
    cfg = dict(min_sift_inliers=4, scr_threshold=.3, ccs_threshold=.7)
    found = PoseGraph.detect_loop_candidates(db, target, cfg)
    pairs = [(reference, target) for reference in expected]
    assert [(r, t) for r, t, _ in found] == pairs
    assert successful_matching == (pairs, pairs)


@pytest.mark.parametrize('count,expected', [
    (0, list(range(7))), (1, list(range(6))), (5, [0, 1]), (20, []),
])
def test_configured_window_and_zero_disable(successful_matching, count, expected):
    cfg = dict(min_sift_inliers=4, scr_threshold=.3, ccs_threshold=.7,
               loop_search_exclude_recent_keyframes=count)
    found = PoseGraph.detect_loop_candidates(database(8), 7, cfg)
    pairs = [(reference, 7) for reference in expected]
    assert [(r, t) for r, t, _ in found] == pairs
    assert successful_matching == (pairs, pairs)


def test_active_reference_filter_still_applies(successful_matching):
    cfg = dict(min_sift_inliers=4, scr_threshold=.3, ccs_threshold=.7,
               loop_search_exclude_recent_keyframes=5)
    found = PoseGraph.detect_loop_candidates(database(9), 8, cfg, active_indices={1, 7})
    assert [(r, t) for r, t, _ in found] == [(1, 8)]
    assert successful_matching == ([(1, 8)], [(1, 8)])


@pytest.mark.parametrize('value', [-1, 1.5, 5., True, False, None, '5', [], {},
                                  float('nan'), float('inf')])
def test_invalid_count_is_rejected_before_search(value):
    cfg = search_config()
    settings = cfg['gelslam_config']['pose_graph_config']
    settings['loop_search_exclude_recent_keyframes'] = value
    with pytest.raises(ValueError, match='loop_search_exclude_recent_keyframes'):
        LoopSearch(cfg, Logger(print_output=False))
    with pytest.raises(ValueError, match='loop_search_exclude_recent_keyframes'):
        PoseGraph.detect_loop_candidates(database(1), 0, settings)
