"""Loop search skips references that the coverage graph deactivated (original GelSLAM filter)."""
from types import SimpleNamespace

import pytest

from gelslam.core.gtsam_pose_graph import PoseGraph
from gelslam.core.local_mapping import LocalMapping, LoopSearch
from gelslam.core.shared_map import SharedMap
from gelslam.utils import Logger
from test_keyframe_handoff_regressions import search_config
from test_loop_search_interval import timed_packet


def make_search(coverage_filter):
    cfg = search_config()
    if coverage_filter is not None:
        cfg['gelslam_config']['pose_graph_config']['loop_search_coverage_filter'] = coverage_filter
    cfg['gelslam_config']['pose_graph_config']['loop_search_exclude_recent_keyframes'] = 0
    return LoopSearch(cfg, Logger(print_output=False))


def run(monkeypatch, search, inactive):
    store, calls = SharedMap(), []

    def detect(db, target, cfg, *active):
        calls.append((target, None if not active else sorted(active[0])))
        return []
    monkeypatch.setattr(PoseGraph, 'detect_loop_candidates', staticmethod(detect))
    for index in range(4):
        store.append_keyframe(timed_packet(index, float(index)))
    snapshot = store.snapshot_for_optimization()
    search.detect(snapshot, None if inactive is None else (snapshot.epoch, frozenset(inactive)))
    return calls, snapshot


@pytest.mark.parametrize('coverage_filter', [None, False])
def test_without_the_filter_every_reference_is_searched(monkeypatch, coverage_filter):
    calls, _ = run(monkeypatch, make_search(coverage_filter), inactive={1})
    assert calls == [(1, None), (2, None), (3, None)]


def test_the_filter_drops_deactivated_references_only(monkeypatch):
    search = make_search(True)
    calls, _ = run(monkeypatch, search, inactive=set())
    assert calls == [(1, [0]), (2, [0, 1]), (3, [0, 1, 2])]
    deactivated = search.db[1].kid
    calls, _ = run(monkeypatch, make_search(True), inactive={deactivated})
    assert calls == [(1, [0]), (2, [0]), (3, [0, 2])]


def test_a_stale_epoch_disables_the_filter(monkeypatch):
    search = make_search(True)
    store, calls = SharedMap(), []

    def detect(db, target, cfg, *active):
        calls.append((target, None if not active else sorted(active[0])))
        return []
    monkeypatch.setattr(PoseGraph, 'detect_loop_candidates', staticmethod(detect))
    for index in range(3):
        store.append_keyframe(timed_packet(index, float(index)))
    snapshot = store.snapshot_for_optimization()
    search.detect(snapshot, (snapshot.epoch + 1, frozenset({0, 1, 2})))
    assert calls == [(1, None), (2, None)]


def test_local_mapping_publishes_the_deactivated_keyframe_ids():
    mapping = LocalMapping(search_config(), Logger(print_output=False), enabled=False)
    assert mapping.coverage_inactive_kids == (0, frozenset())
    mapping.epoch = 3
    mapping.db = [SimpleNamespace(kid=k) for k in (10, 11, 12, 13)]
    # Keyframe 13 is not in the coverage graph yet, so it is not deactivated.
    nodes = [SimpleNamespace(is_active=a) for a in (True, False, True)]

    class Coverage:
        def size(self):
            return len(nodes)

        def __getitem__(self, index):
            return nodes[index]
    mapping.coverage = Coverage()
    mapping._publish_coverage_inactive()
    assert mapping.coverage_inactive_kids == (3, frozenset({11}))


def test_the_filter_setting_is_validated():
    with pytest.raises(ValueError, match='loop_search_coverage_filter must be true or false'):
        make_search(1)
