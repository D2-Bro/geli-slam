"""Loop target selection uses data time and is independent of queue batches."""
from dataclasses import replace

import numpy as np
import pytest
from std_msgs.msg import Header

from gelslam.core.gtsam_pose_graph import PoseGraph
from gelslam.core.local_mapping import LoopSearch
from gelslam.core.shared_map import SharedMap
from gelslam.utils import Logger
from test_keyframe_handoff_regressions import search_config
from test_ros_backend_transport import packet


def make_search(interval):
    cfg = search_config()
    cfg['gelslam_config']['pose_graph_config']['loop_search_min_interval_s'] = interval
    return LoopSearch(cfg, Logger(print_output=False))


def timed_packet(index, timestamp, navigation=True):
    p = packet(index)
    h = Header()
    if timestamp is not None:
        # A different header clock verifies that navigation time takes priority.
        header_time = 1000. + 10 * index if navigation else timestamp
        h.stamp.sec = int(header_time)
        h.stamp.nanosec = round((header_time - h.stamp.sec) * 1e9)
    return replace(p, header=h, navigation_valid=navigation,
                   navigation_timestamp=timestamp if navigation else 0.)


@pytest.mark.parametrize('navigation', [False, True])
@pytest.mark.parametrize('batches', [(6,), (1, 5), (2, 1, 3), (1, 1, 1, 1, 1, 1)])
def test_interval_selects_same_targets_in_every_batch_layout(monkeypatch, navigation, batches):
    search, store = make_search(.5), SharedMap()
    targets, found = [], []
    def detect(db, target, cfg):
        targets.append(target)
        # K_2 was skipped as a target but remains a possible reference for K_4.
        return [(2, 4, np.eye(4))] if target == 4 else []
    monkeypatch.setattr(PoseGraph, 'detect_loop_candidates', staticmethod(detect))
    times = [0., 1., 1.2, 1.3, 1.5, 2.]
    offset = 0
    for size in batches:
        for index in range(offset, offset + size):
            store.append_keyframe(timed_packet(index, times[index], navigation))
        offset += size
        snapshot = store.snapshot_for_optimization()
        found.extend(search.detect(snapshot))
        assert search.detect(snapshot) == []
    assert targets == [1, 4, 5]
    assert [(p.reference_kid, p.target_kid) for p in found] == [(2, 4)]
    assert search.db.size() == 6


@pytest.mark.parametrize('interval', [0., .3])
def test_epoch_sized_timestamp_boundary_and_zero_interval(monkeypatch, interval):
    search, store = make_search(interval), SharedMap()
    targets = []
    def detect(db, target, cfg):
        targets.append(target)
        return []
    monkeypatch.setattr(PoseGraph, 'detect_loop_candidates', staticmethod(detect))
    # The subtraction at this epoch rounds slightly below the 0.3 s boundary.
    times = [1791377410., 1791377410.1, 1791377410.2, 1791377410.4, 1791377410.4001]
    for index, time in enumerate(times):
        store.append_keyframe(timed_packet(index, time))
    search.detect(store.snapshot_for_optimization())
    assert targets == ([1, 2, 3, 4] if interval == 0 else [1, 3])


@pytest.mark.parametrize('times,expected', [
    ([0., 10., 10.1, 1., 1.1, 1.5], [1, 3, 5]),
    ([0., 1., 1.1, None, 1.2, 1.3, 1.7], [1, 3, 4, 6]),
])
def test_missing_or_backwards_clock_does_not_stall_search(monkeypatch, times, expected):
    search, store = make_search(.5), SharedMap()
    targets = []
    def detect(db, target, cfg):
        targets.append(target)
        return []
    monkeypatch.setattr(PoseGraph, 'detect_loop_candidates', staticmethod(detect))
    for index, time in enumerate(times):
        store.append_keyframe(timed_packet(index, time, navigation=False))
    search.detect(store.snapshot_for_optimization())
    assert targets == expected


def test_new_epoch_starts_search_without_previous_interval(monkeypatch):
    search, store = make_search(10.), SharedMap()
    targets = []
    def detect(db, target, cfg):
        targets.append((search.epoch, target))
        return []
    monkeypatch.setattr(PoseGraph, 'detect_loop_candidates', staticmethod(detect))
    for offset in (1., 2.):
        for i in range(3):
            store.append_keyframe(timed_packet(i, offset + i * .1))
        search.detect(store.snapshot_for_optimization())
        store.reset()
    assert targets == [(0, 1), (1, 1)]


def test_failed_batch_does_not_advance_search_time(monkeypatch):
    search, store = make_search(.5), SharedMap()
    for i, time in enumerate([0., 1., 1.2, 1.5]):
        store.append_keyframe(timed_packet(i, time))
    targets, fail = [], True
    def detect(db, target, cfg):
        targets.append(target)
        if fail and target == 3:
            raise RuntimeError('registration failed')
        return [(0, target, np.eye(4))]
    monkeypatch.setattr(PoseGraph, 'detect_loop_candidates', staticmethod(detect))
    with pytest.raises(RuntimeError, match='registration failed'):
        search.detect(store.snapshot_for_optimization())
    fail = False
    found = search.detect(store.snapshot_for_optimization())
    assert [p.target_kid for p in found] == [1, 3]
    assert targets == [1, 3, 1, 3]


@pytest.mark.parametrize('value', [-.1, float('nan'), float('inf'), -float('inf'),
                                  True, False, None, 'invalid', [], {}])
def test_invalid_interval_is_rejected(value):
    with pytest.raises(ValueError, match='loop_search_min_interval_s'):
        make_search(value)
