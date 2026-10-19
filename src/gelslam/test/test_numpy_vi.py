"""Direct tactile normal residual tests."""

from types import SimpleNamespace

import numpy as np

from gelslam.core.numpy_vi import DenseNormalResidual


def test_dense_residual_reselects_shared_contact_rows_each_evaluation():
    """NormalFlow's fixed candidates must not imply fixed shared contacts."""
    height = width = 40
    yy, xx = np.mgrid[:height, :width]
    normals = np.dstack((
        np.zeros_like(xx), np.zeros_like(xx), np.ones_like(xx),
    )).astype(np.float32)
    contact = np.ones((height, width), dtype=bool)
    curvature = -((xx - 20) ** 2 + (yy - 20) ** 2).astype(np.float64)
    reference = SimpleNamespace(
        N=normals,
        H=np.zeros((height, width), dtype=np.float32),
        C=contact,
        L=curvature,
    )
    target = SimpleNamespace(
        N=normals,
        H=np.zeros((height, width), dtype=np.float32),
        C=xx < 20,
        L=curvature,
    )
    residual = DenseNormalResidual(
        reference, target, 0.059, np.eye(4), np.eye(4), max_samples=200
    )
    initial_count = len(residual)
    moved_target_pose = np.eye(4)
    moved_target_pose[0, 3] = -6 * 0.059 / 1000.0

    result = residual.evaluate(np.eye(4), moved_target_pose)

    assert result is not None
    assert len(residual) > initial_count
    assert result[0].shape == (3 * len(residual),)
    assert result[1].shape == (3 * len(residual), 6)
    assert result[2].shape == (3 * len(residual), 6)


def test_height_refinement_returns_original_normalflow_target_z_mean():
    """Height refinement must use the post-normal target-frame Z correction."""
    height = width = 40
    yy, xx = np.mgrid[:height, :width]
    normals = np.dstack((
        np.zeros_like(xx), np.zeros_like(xx), np.ones_like(xx),
    )).astype(np.float32)
    contact = np.ones((height, width), dtype=bool)
    curvature = -((xx - 20) ** 2 + (yy - 20) ** 2).astype(np.float64)
    reference = SimpleNamespace(
        N=normals, H=np.zeros((height, width), dtype=np.float32),
        C=contact, L=curvature,
    )
    target = SimpleNamespace(
        N=normals, H=np.full((height, width), 2.0, dtype=np.float32),
        C=contact, L=curvature,
    )
    residual = DenseNormalResidual(
        reference, target, 0.059, np.eye(4), np.eye(4), max_samples=200
    )

    correction = residual.target_z_height_correction(np.eye(4), np.eye(4))

    assert np.isclose(correction, 2.0 * 0.059 / 1000.0)
