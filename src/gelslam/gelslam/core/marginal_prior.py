"""Hessian Schur marginalization and a trial-frame navigation-state prior."""

from dataclasses import dataclass

import numpy as np

from gelslam.core.split_pose import (
    split_pose_coordinate_jacobian, split_pose_coordinates,
)


HESSIAN_SVD_CUTOFF = 1e-6
INFORMATION_EIGEN_CUTOFF = 1e-12


def _symmetric(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (matrix + matrix.T)


def information_rows(information, gradient=None):
    """Represent a PSD information matrix and normal-equation RHS as rows.

    Eigenvalues below 1e-12 (including negative roundoff) carry no information.
    Only observable gradient components can be represented by these rows.
    No damping is added. With no gradient, the anchor residual is exactly zero.
    """
    eigenvalues, vectors = np.linalg.eigh(_symmetric(information))
    active = eigenvalues >= INFORMATION_EIGEN_CUTOFF
    roots = np.sqrt(np.where(active, eigenvalues, 0.0))
    A = roots[:, None] * vectors.T
    rhs = np.zeros(len(roots), dtype=np.float64)
    if gradient is not None:
        projected = vectors.T @ np.asarray(gradient, dtype=np.float64)
        rhs[active] = projected[active] / roots[active]
    return A, rhs


@dataclass
class SchurComplement:
    information: np.ndarray
    gradient: np.ndarray  # RHS convention: H delta = g, not the cost derivative.
    previous_inverse: np.ndarray
    previous_cross: np.ndarray
    previous_gradient: np.ndarray
    _row_matrix: np.ndarray
    _row_rhs: np.ndarray

    def recover_previous(self, retained_step):
        return self.previous_inverse @ (
            self.previous_gradient - self.previous_cross @ retained_step
        )

    def rows(self):
        # Reuse the same eigenspace. Repeated decomposition can turn roundoff
        # in a discarded direction into a small positive information value.
        return self._row_matrix, self._row_rhs


def schur_complement(hessian, gradient, eliminate_dim):
    """Eliminate the leading variables of H delta = g using ORB cutoffs.

    Keep the same SVD pseudoinverse for elimination and back-substitution.
    Singular values <= 1e-6 have zero inverse, independent of matrix scale.
    """
    H = _symmetric(hessian)
    g = np.asarray(gradient, dtype=np.float64)
    if H.ndim != 2 or H.shape[0] != H.shape[1] or g.shape != (len(H),):
        raise ValueError("Hessian and gradient dimensions do not match")
    if not 0 <= eliminate_dim <= len(H):
        raise ValueError("invalid number of eliminated dimensions")
    if not np.all(np.isfinite(H)) or not np.all(np.isfinite(g)):
        raise ValueError("Hessian and gradient must be finite")
    n = eliminate_dim
    u, singular, vt = np.linalg.svd(H[:n, :n], full_matrices=False)
    inverse_singular = np.zeros_like(singular)
    active = singular > HESSIAN_SVD_CUTOFF
    inverse_singular[active] = 1.0 / singular[active]
    inverse = _symmetric((vt.T * inverse_singular) @ u.T)
    cross = H[:n, n:]
    reduced = _symmetric(H[n:, n:] - cross.T @ inverse @ cross)
    reduced_gradient = g[n:] - cross.T @ inverse @ g[:n]
    A, rhs = information_rows(reduced, reduced_gradient)
    return SchurComplement(_symmetric(A.T @ A), reduced_gradient,
                           inverse, cross.copy(), g[:n].copy(), A, rhs)


def marginalize_rows_system(previous_A, retained_A, rhs):
    """Build H=A.T A and g=A.T rhs for previous/retained increment rows."""
    previous_A = np.asarray(previous_A, dtype=np.float64)
    retained_A = np.asarray(retained_A, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    A = np.column_stack((previous_A, retained_A))
    return schur_complement(A.T @ A, A.T @ rhs, previous_A.shape[1])


def marginalize_rows(previous_A, retained_A, rhs):
    """Return GTSAM rows for the Schur information AND gradient of this step."""
    return marginalize_rows_system(previous_A, retained_A, rhs).rows()


@dataclass
class NavigationPrior:
    """Frozen prior: A * splitCoordinates(anchor, state) - rhs.

    Columns: Pose3 [rotation, translation], world velocity, accelerometer
    bias, gyroscope bias. The linearization anchor belongs to one IMU time.
    """

    anchor: object
    A: np.ndarray
    rhs: np.ndarray

    @classmethod
    def from_information(cls, anchor, information):
        """Store information at this accepted/target state with zero gradient."""
        A, rhs = information_rows(information)
        return cls(anchor.copy(), A, rhs)

    @classmethod
    def initial(cls, state, sigmas):
        return cls(state.copy(), np.diag(1.0 / np.asarray(sigmas)), np.zeros(15))

    def coordinates(self, state):
        return np.concatenate((
            split_pose_coordinates(self.anchor.pose, state.pose),
            state.velocity - self.anchor.velocity,
            state.accel_bias - self.anchor.accel_bias,
            state.gyro_bias - self.anchor.gyro_bias,
        ))

    def reframe(self, rotation, origin):
        """Transport the full prior with a rigid change of world coordinates.

        Pose increments are body-local. Only world-velocity columns rotate.
        Keep every cross term and the frozen residual right-hand side.
        """
        transport = np.eye(15)
        transport[6:9, 6:9] = rotation.T
        return NavigationPrior(
            self.anchor.reframe(rotation, origin), self.A @ transport, self.rhs.copy(),
        )

    def linearize(self, state):
        delta = self.coordinates(state)
        transport = np.eye(15)
        transport[:6, :6] = split_pose_coordinate_jacobian(self.anchor.pose, state.pose)
        return self.A @ transport, self.rhs - self.A @ delta

    def error(self, state):
        residual = self.A @ self.coordinates(state) - self.rhs
        return 0.5 * float(residual @ residual)

    def matches(self, state):
        return (abs(self.anchor.timestamp - state.timestamp) < 1e-9
                and np.allclose(self.coordinates(state), 0.0, atol=1e-8, rtol=0.0))
