"""Chunk-based loop closure: fusion, chunk membership and a synthetic revisit."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from normalflow.utils import gxy2laplacian, gxy2normal

from gelslam.core.chunk_loop_closure import ChunkLoopDetector, fuse_keyframes
from gelslam.core.local_mapping import LoopSearch
from gelslam.utils import Logger

PPMM = 0.0634
ROWS, COLS = 240, 320


def surface(seed=3, shape=(700, 1000)):
    rng = np.random.default_rng(seed)
    noise = cv2.GaussianBlur(rng.normal(size=shape).astype(np.float32), (0, 0), 3.0)
    return 40.0 * noise / noise.std()


def keyframe(world, ox, oy, new=False, tactile=True):
    """A 240 x 320 contact patch cut from the world height field at pixel offset (ox, oy)."""
    H = world[oy:oy + ROWS, ox:ox + COLS].copy()
    yy, xx = np.mgrid[:ROWS, :COLS]
    C = ((xx - COLS / 2) / 130.0) ** 2 + ((yy - ROWS / 2) / 95.0) ** 2 < 1.0
    G = np.dstack(np.gradient(H)[::-1]).astype(np.float32) * 0.05
    pose = np.eye(4)
    pose[0, 3], pose[1, 3] = ox * PPMM / 1000.0, oy * PPMM / 1000.0
    return SimpleNamespace(H=(H * 0.05).astype(np.float32), C=C,
                           N=gxy2normal(G).astype(np.float32),
                           L=gxy2laplacian(G, C).astype(np.float32), trial_T_curr=pose, ppmm=PPMM,
                           tactile_valid=tactile, is_new_trial=lambda new=new: new)


class DB(list):
    def size(self):
        return len(self)


def config(**overrides):
    return {'scr_threshold': 0.3, 'ccs_threshold': 0.7, **overrides}


def test_fusing_one_keyframe_reproduces_it():
    kf = keyframe(surface(), 100, 100)
    H, C, N, L = fuse_keyframes([kf], [kf.trial_T_curr], kf.trial_T_curr, PPMM)
    inner = cv2.erode(kf.C.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    assert np.all(C[inner])
    assert np.abs(H[inner] - kf.H[inner]).max() < 1e-3
    assert np.abs(N[inner] - kf.N[inner]).max() < 1e-4


def test_shifted_keyframe_lands_at_its_world_position():
    world = surface()
    a, b = keyframe(world, 100, 100), keyframe(world, 140, 110)
    H, C, N, L = fuse_keyframes([b], [b.trial_T_curr], a.trial_T_curr, PPMM)
    both = C & a.C
    assert both.sum() > 10000
    assert np.median(np.abs(H[both] - a.H[both])) < 0.05


def test_revisit_gives_a_chunk_loop_with_the_true_relative_pose():
    world = surface()
    first = [keyframe(world, 120 + 25 * i, 150, new=i == 0) for i in range(12)]
    # Second pass after a no-contact gap (IMU recovery keyframe) over the same strip, 7 px lower.
    second = [keyframe(world, 135 + 25 * i, 157, tactile=i != 0) for i in range(12)]
    db = DB(first + second)
    detector = ChunkLoopDetector(config())
    loops = detector.update(db, len(db))
    cross = [(a, b, T) for a, b, T in loops if a < 12 <= b]
    assert cross, 'no chunk loop between the two passes'
    for a, b, T in loops:
        members = {u['center']: u['members'] for u in detector.units}
        assert not members[a] & members[b]
    for a, b, T in cross:
        truth = np.linalg.inv(db[b].trial_T_curr) @ db[a].trial_T_curr
        assert np.linalg.norm(T[:3, 3] - truth[:3, 3]) < 1e-4
        cos = np.clip((np.trace(T[:3, :3].T @ truth[:3, :3]) - 1) / 2, -1, 1)
        assert np.degrees(np.arccos(cos)) < 1.0


def test_chunks_complete_incrementally():
    world = surface()
    db = DB([keyframe(world, 120 + 25 * i, 150, new=i == 0) for i in range(12)])
    detector = ChunkLoopDetector(config())
    detector.update(db, 8)          # centre c is complete once keyframe c + 4 exists
    assert [u['center'] for u in detector.units] == [0, 3] and detector.next_center == 4
    detector.update(db, 10)
    assert [u['center'] for u in detector.units] == [0, 3] and detector.next_center == 6
    detector.update(db, 11)
    assert [u['center'] for u in detector.units] == [0, 3, 6] and detector.next_center == 7


def test_loop_search_option_must_be_boolean():
    base = {'gelslam_config': {'pose_graph_config': {'chunk_loop_closure': 'yes'}},
            'device_config': {'ppmm': PPMM}}
    with pytest.raises(ValueError):
        LoopSearch(base, Logger(print_output=False))
    base['gelslam_config']['pose_graph_config']['chunk_loop_closure'] = True
    assert LoopSearch(base, Logger(print_output=False)).chunk_loops is not None
