"""Per-solve sampling reuse, cache lifetime and navigation-state freshness."""

from types import SimpleNamespace

import numpy as np
import pytest

from gelslam.core import numpy_vi as nv


def visual_pair(dtype=np.float32):
    yy, xx = np.mgrid[:40, :40]
    normal = np.dstack((.01 * np.sin(xx / 5), .01 * np.cos(yy / 6),
                       np.ones_like(xx))).astype(dtype)
    height = (.001 * xx + .002 * yy).astype(dtype)
    curvature = (xx + yy * 40).astype(np.float32)
    def frame():
        return SimpleNamespace(N=normal.copy(), H=height.copy(),
                               C=np.ones((40, 40), dtype=bool), L=curvature.copy())
    reference = frame()
    reference.N = reference.N.astype(np.float32)  # get_J/Sobel requires CV_32F.
    return nv.DenseNormalResidual(reference, frame(), 1., np.eye(4), np.eye(4), 300)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_normal_height_and_post_step_share_samples_and_float_maps(monkeypatch, dtype):
    visual = visual_pair(dtype)
    calls = []
    original = nv.wide_remap
    def sample(field, *args, **kwargs):
        calls.append(field)
        return original(field, *args, **kwargs)
    monkeypatch.setattr(nv, 'wide_remap', sample)
    reference, target = np.eye(4), np.eye(4)
    with visual.cache_projections():
        visual.evaluate_ic(reference, target)
        visual.target_z_height_correction(reference, target)
        visual.residual_on_fixed_rows(reference, target, visual.last_active_rows)
        assert len(calls) == 3
        # An in-place pose update must not return the previous samples.
        target[0, 3] = .001
        visual.evaluate_ic(reference, target)
        visual.target_z_height_correction(reference, target)
        assert len(calls) == 6
        assert all(calls[i] is calls[i + 3] for i in range(3))
        assert all(field.dtype == np.float32 for field in calls)
        if dtype == np.float32:
            assert calls[0] is visual.tar.N
            assert calls[1] is visual.tar.H
    assert visual._projection_cache is None and visual._target_maps is None
    assert visual._ic_reference_blocks is None


@pytest.mark.parametrize('fixed_z', [False, True])
@pytest.mark.parametrize('mode', ['evaluate_ic', 'evaluate_relinearized'])
def test_cached_evaluations_match_uncached_for_reference_and_target_changes(fixed_z, mode):
    visual = visual_pair()
    if fixed_z:
        visual.set_fixed_reference_z(.0003)
    poses = []
    for i in range(6):
        reference, target = np.eye(4), np.eye(4)
        reference[:3, :3] = nv.so3_exp(np.array([.02, -.03, .01]))
        reference[1, 3] = i * .0001
        target[0, 3] = i * .0002
        target[2, 3] = i * .0001
        poses.append((reference, target))
    expected = []
    for reference, target in poses:
        evaluated = getattr(visual, mode)(reference, target)
        expected.append((evaluated, visual.target_z_height_correction(reference, target),
                         visual.last_active_rows.copy()))
    with visual.cache_projections():
        for (reference, target), (evaluated, height, rows) in zip(poses, expected):
            for _ in range(2):
                actual = getattr(visual, mode)(reference, target)
                for value, wanted in zip(actual, evaluated):
                    np.testing.assert_array_equal(value, wanted)
                assert visual.target_z_height_correction(reference, target) == height
                np.testing.assert_array_equal(visual.last_active_rows, rows)
        assert len(visual._projection_cache) <= 4


def test_failed_overlap_query_does_not_hide_rows_from_lower_threshold(monkeypatch):
    visual = visual_pair()
    visual.tar.C[:] = False
    xx, yy = visual._pixel_coordinates(visual.candidate_points[:5])
    visual.tar.C[np.rint(yy).astype(int), np.rint(xx).astype(int)] = True
    calls = []
    original = nv.wide_remap
    def sample(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(nv, 'wide_remap', sample)
    with visual.cache_projections():
        assert visual._shared_projection(np.eye(4), np.eye(4)) is None
        assert len(visual) == 5
        projected = visual._shared_projection(np.eye(4), np.eye(4), min_points=1)
        assert projected[-1].sum() == 5
        visual.active_count = 99
        assert visual._shared_projection(np.eye(4), np.eye(4),
                                         apply_normal_z_constraint=False) is None
        assert len(visual) == 99  # Height queries do not update the normal count.
        assert len(calls) == 3


@pytest.mark.parametrize('exit_by_exception', [False, True])
def test_scope_exit_discards_samples_and_observes_later_map_edits(exit_by_exception):
    visual = visual_pair()
    def run():
        with visual.cache_projections():
            assert visual.evaluate_ic(np.eye(4), np.eye(4)) is not None
            if exit_by_exception:
                raise RuntimeError('interrupted solve')
    if exit_by_exception:
        with pytest.raises(RuntimeError, match='interrupted solve'):
            run()
    else:
        run()
    assert visual._projection_cache is None and visual._target_maps is None
    assert visual._ic_reference_blocks is None
    visual.tar.C[:] = False
    assert visual.evaluate_ic(np.eye(4), np.eye(4)) is None
    with visual.cache_projections():
        assert visual.target_z_height_correction(np.eye(4), np.eye(4)) is None
    visual.tar.C[:] = True
    visual.tar.H[:] = np.nan
    with visual.cache_projections():
        assert visual.evaluate_ic(np.eye(4), np.eye(4)) is None


@pytest.mark.parametrize('replace_array', [False, True])
def test_ic_rows_remain_independent_and_refresh_after_reference_edits(replace_array):
    visual = visual_pair()
    reference, target = np.eye(4), np.eye(4)
    expected = visual.evaluate_ic(reference, target)
    with visual.cache_projections():
        result = visual.evaluate_ic(reference, target)
        result[1][:] = 0.  # A consumer must not overwrite the shared template.
        actual = visual.evaluate_ic(reference, target)
        for value, wanted in zip(actual, expected):
            np.testing.assert_array_equal(value, wanted)
    if replace_array:
        visual.candidate_normal_jacobian = -visual.candidate_normal_jacobian
    else:
        visual.candidate_normal_jacobian *= -1
    expected = visual.evaluate_ic(reference, target)
    with visual.cache_projections():
        actual = visual.evaluate_ic(reference, target)
        for value, wanted in zip(actual, expected):
            np.testing.assert_array_equal(value, wanted)
