"""Tangent-space normal residual: two rows per point, same first-order model."""

from types import SimpleNamespace

import numpy as np
import pytest

from gelslam.core.gtsam_vi import _normal_dim, _normal_huber_linearization
from gelslam.core.numpy_vi import DenseNormalResidual, normal_tangent_basis, so3_exp


def _bumpy_frames(size=48):
    yy, xx = np.mgrid[:size, :size].astype(np.float64)
    height = 0.4 * np.exp(-((xx - 24) ** 2 + (yy - 22) ** 2) / 60.0)
    gy, gx = np.gradient(height)
    normals = np.dstack((-gx, -gy, np.ones_like(gx)))
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)
    frame = SimpleNamespace(
        N=normals.astype(np.float32), H=height.astype(np.float32),
        C=np.ones((size, size), dtype=bool),
        L=-((xx - 24) ** 2 + (yy - 22) ** 2) + 0.1 * xx)
    return frame, SimpleNamespace(**vars(frame))


def _pair(space):
    ref, tar = _bumpy_frames()
    return DenseNormalResidual(ref, tar, 0.059, np.eye(4), np.eye(4),
                               max_samples=300, residual_space=space)


def test_tangent_basis_is_orthonormal_and_tangent():
    normals = np.array([[0, 0, 1], [1, 0, 0], [0.3, -0.4, 0.866], [0, 1, 0]], float)
    basis = normal_tangent_basis(normals)
    unit = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    assert basis.shape == (4, 2, 3)
    assert np.allclose(np.einsum("nij,nj->ni", basis, unit), 0.0)
    assert np.allclose(np.einsum("nij,nkj->nik", basis, basis), np.eye(2))


@pytest.mark.parametrize("method", ["evaluate_ic", "evaluate_relinearized"])
def test_tangent_residual_projects_the_vector_residual(method):
    target_pose = np.eye(4)
    target_pose[:3, :3] = so3_exp(np.array([0.01, -0.008, 0.02]))
    target_pose[:3, 3] = (0.03e-3, -0.02e-3, 0.0)
    vector, tangent = _pair("vector"), _pair("tangent")
    rv, Jrv, Jtv = getattr(vector, method)(np.eye(4), target_pose)
    rt, Jrt, Jtt = getattr(tangent, method)(np.eye(4), target_pose)
    count = len(rv) // 3
    assert _normal_dim(vector) == 3 and _normal_dim(tangent) == 2
    assert rt.shape == (2 * count,) and Jrt.shape == Jtt.shape == (2 * count, 6)
    basis = normal_tangent_basis(vector.candidate_normals[vector.last_active_rows])
    assert np.allclose(rt, np.einsum("nij,nj->ni", basis, rv.reshape(-1, 3)).reshape(-1))
    assert np.allclose(Jtt, np.einsum("nij,njk->nik", basis, Jtv.reshape(-1, 3, 6)).reshape(-1, 6))
    # Only the reference-normal component is dropped. It is cos(angle) - 1 for
    # unit normals, but also absorbs a short interpolated target normal.
    unit = vector.candidate_normals[vector.last_active_rows]
    unit = unit / np.linalg.norm(unit, axis=1, keepdims=True)
    along = np.sum(rv.reshape(-1, 3) * unit, axis=1)
    assert np.isclose(np.sum(rv**2), np.sum(rt**2) + np.sum(along**2))
    fixed = tangent.residual_on_fixed_rows(np.eye(4), target_pose, tangent.last_active_rows)
    assert np.allclose(fixed, rt)
    A, b, _, _ = _normal_huber_linearization(rt, Jtt, 1.0, 0.4, 2)
    assert A.shape == (2 * count, 6) and b.shape == (2 * count,)


def test_residual_space_is_validated():
    with pytest.raises(ValueError):
        _pair("sphere")
