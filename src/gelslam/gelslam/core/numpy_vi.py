"""Direct tactile normal residuals and shared navigation state."""

from contextlib import contextmanager
from dataclasses import dataclass

import cv2
import numpy as np

try:
    from normalflow.utils import get_J, get_backproj_laplacian, height2pointcloud, wide_remap
except ImportError:  # IMU-only diagnostics do not need the visual frontend.
    get_J = get_backproj_laplacian = height2pointcloud = wide_remap = None


def _skew(vector):
    x, y, z = vector
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def so3_exp(rotation):
    angle = np.linalg.norm(rotation)
    if angle < 1e-12:
        return np.eye(3) + _skew(rotation)
    axis = rotation / angle
    skew = _skew(axis)
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def adjoint(pose):
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    result = np.zeros((6, 6))
    result[:3, :3] = rotation
    result[3:, :3] = _skew(translation) @ rotation
    result[3:, 3:] = rotation
    return result


def normal_ic_pose_jacobians(reference_rows, target_T_reference):
    """Map cached reference IC rows into the two right-local pose charts."""
    return reference_rows, -reference_rows @ adjoint(np.linalg.inv(target_T_reference))


def gelsight_z_pose_jacobian(gel_from_state_increment, sigma):
    """Freeze rotation/XY; measure only target GelSight body-local Z motion."""
    row = np.zeros((1, 6))
    row[0, 5] = 1.0 / sigma
    return row @ gel_from_state_increment


def _frame_memo(frame, key, sources, compute):
    """Memoize reference-only quantities on a frame.

    One reference keyframe serves many target images. An entry is reused only
    while the frame still holds the same source array objects; replacing a map
    attribute invalidates it. Frames are not edited in place.
    """
    try:
        cache = frame.__dict__.setdefault("_reference_selection_cache", {})
    except AttributeError:
        return compute()
    entry = cache.get(key)
    if entry is not None and all(a is b for a, b in zip(entry[0], sources)):
        return entry[1]
    value = compute()
    cache[key] = (sources, value)
    return value


def curvature_selection(ref, count, ppmm):
    """Original NormalFlow strongest-curvature mask and its float32 points."""
    def compute():
        # Match normalflow.registration: its threshold is strict by 1e-6.
        threshold = np.partition(np.abs(ref.L).ravel(), -count)[-count] + 1e-6
        mask = np.abs(ref.L) >= threshold
        # Keep pixel indices, not a full-image mask, on long-lived keyframes.
        return np.flatnonzero(mask), height2pointcloud(ref.H, mask, ppmm)
    indices, points = _frame_memo(
        ref, ("selection", int(count), float(ppmm)), (ref.L, ref.H), compute)
    mask = np.zeros(np.shape(ref.L), dtype=bool)
    mask.flat[indices] = True
    return mask, points


@dataclass
class NavigationState:
    """Navigation pose, world velocity, and sensor-frame additive IMU biases.

    The active GTSAM frontend uses IMU pose and IMU-origin velocity.
    """

    timestamp: float
    pose: np.ndarray
    velocity: np.ndarray
    accel_bias: np.ndarray
    gyro_bias: np.ndarray

    def copy(self):
        return NavigationState(
            self.timestamp, self.pose.copy(), self.velocity.copy(),
            self.accel_bias.copy(), self.gyro_bias.copy(),
        )

    def reframe(self, rotation, origin):
        """Change world axes and origin; retain sensor-frame IMU biases."""
        result = self.copy()
        result.pose[:3, :3] = rotation @ self.pose[:3, :3]
        result.pose[:3, 3] = rotation @ (self.pose[:3, 3] - origin)
        result.velocity = rotation @ self.velocity
        return result


def normal_tangent_basis(normals):
    """Two orthonormal tangent vectors per normal, shape (N, 2, 3)."""
    normals = np.asarray(normals, dtype=np.float64)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    # Cross with the world axis least aligned with each normal.
    helper = np.zeros_like(normals)
    helper[np.arange(len(normals)), np.argmin(np.abs(normals), axis=1)] = 1.0
    first = np.cross(normals, helper)
    first /= np.linalg.norm(first, axis=1, keepdims=True)
    return np.stack((first, np.cross(normals, first)), axis=1)


class DenseNormalResidual:
    """NormalFlow normal residual with dynamic shared contacts.

    The strongest-curvature reference candidates stay fixed, as in original
    NormalFlow.  Their target contact overlap is recomputed at every GN
    linearization, so the residual contains exactly the current shared rows.

    residual_space "vector" keeps NormalFlow's three-component difference
    R^-1 n_tar - n_ref per point. "tangent" projects it onto two tangent
    vectors of the reference normal: a unit normal has two degrees of
    freedom, and the dropped n_ref component is cos(angle) - 1 (second order).
    """

    RESIDUAL_SPACES = {"vector": 3, "tangent": 2}

    def __init__(self, ref, tar, ppmm, initial_reference_pose,
                 initial_target_pose, max_samples=3000, dtype=np.float64,
                 residual_space="vector"):
        if residual_space not in self.RESIDUAL_SPACES:
            raise ValueError("dense_normalflow_residual_space must be 'vector' or 'tangent'")
        self.residual_space = residual_space
        # Residual rows per active point; Huber weights act on these blocks.
        self.residual_dim = self.RESIDUAL_SPACES[residual_space]
        # Precision of the normal residual and stored reference IC rows.
        # float32 (original NormalFlow) measured no faster here: projection,
        # contact selection, height and the IC pose mapping must stay float64
        # because the joint solve relies on an exact common-motion nullspace
        # and rank tolerance 1e-12, so the rows are converted back each pass.
        self.dtype = np.dtype(dtype)
        if get_J is None:
            raise ImportError(
                "normalflow is required for DenseNormalResidual; "
                "IMU-only preintegration remains available without it"
            )
        self.ref = ref
        self.tar = tar
        self.ppmm = float(ppmm)
        self.last_active_rows = None
        self.fixed_reference_z_m = None
        self._projection_cache = None
        self._target_maps = None
        self._ic_reference_blocks = None
        self.initial_reference_pose = np.asarray(
            initial_reference_pose, dtype=np.float64
        ).copy()
        self.initial_target_pose = np.asarray(
            initial_target_pose, dtype=np.float64
        ).copy()
        count = min(int(max_samples), ref.L.size)
        if count < 10 or self.ppmm <= 0.0:
            self.candidate_points = np.empty((0, 3))
            self.candidate_normals = np.empty((0, 3), dtype=self.dtype)
            self.candidate_normal_jacobian = np.empty((3, 5, 0), dtype=self.dtype)
            self.active_count = 0
            return
        def compute():
            mask, points32 = curvature_selection(ref, count, self.ppmm)
            points = points32.astype(np.float64)
            return (
                points,
                ref.N[mask].astype(self.dtype),
                get_J(ref.N, mask, points.astype(np.float32), self.ppmm).astype(self.dtype),
            )
        candidates = _frame_memo(ref, ("candidates", count, self.ppmm, self.dtype),
                                 (ref.L, ref.H, ref.N), compute)
        # Own copies: callers may edit or replace an instance's candidates.
        points, normals, normal_jacobian = (array.copy() for array in candidates)
        self.candidate_points = points
        self.candidate_normals = normals
        self.candidate_normal_jacobian = normal_jacobian
        if len(points) < 10:
            self.active_count = 0
            return
        relative = (
            np.linalg.inv(self.initial_target_pose) @ self.initial_reference_pose
        )
        mapped = points @ relative[:3, :3].T + relative[:3, 3]
        self.active_count = int(np.count_nonzero(self._shared_mask(mapped)))

    def __len__(self):
        return self.active_count

    @contextmanager
    def cache_projections(self):
        """Share immutable image-pair samples during one optimization call.

        Frames and candidates must not change inside this scope. Outside it,
        evaluations continue to read the current maps, including in-place edits.
        Limit cached poses so memory does not grow with the iteration count.
        """
        if self._projection_cache is not None:
            yield
            return
        self._target_maps = tuple(np.asarray(field, dtype=np.float32)
                                  for field in (self.tar.N, self.tar.H, self.tar.C))
        self._projection_cache = {}
        try:
            yield
        finally:
            self._projection_cache = None
            self._target_maps = None
            self._ic_reference_blocks = None

    def set_fixed_reference_z(self, z_m):
        """Hold keyframe-relative Z constant in every normal evaluation."""
        if not np.isfinite(z_m):
            raise ValueError("fixed normal reference Z must be finite")
        self.fixed_reference_z_m = float(z_m)
        if len(self.candidate_points) < 10:
            self.active_count = 0
            return
        target = self._normal_target_pose(
            self.initial_reference_pose, self.initial_target_pose
        )
        relative = np.linalg.inv(target) @ self.initial_reference_pose
        mapped = self.candidate_points @ relative[:3, :3].T + relative[:3, 3]
        self.active_count = int(np.count_nonzero(self._shared_mask(mapped)))

    def _normal_target_pose(self, reference_pose, target_pose):
        if self.fixed_reference_z_m is None:
            return target_pose
        reference_T_target = np.linalg.inv(reference_pose) @ target_pose
        reference_T_target[2, 3] = self.fixed_reference_z_m
        return reference_pose @ reference_T_target

    def _pixel_coordinates(self, points):
        xx = (
            points[:, 0] * 1000.0 / self.ppmm + self.tar.N.shape[1] / 2 - 0.5
        ).astype(np.float32)
        yy = (
            points[:, 1] * 1000.0 / self.ppmm + self.tar.N.shape[0] / 2 - 0.5
        ).astype(np.float32)
        return xx, yy

    def _in_image(self, xx, yy):
        return (
            (xx >= 0) & (xx < self.tar.N.shape[1])
            & (yy >= 0) & (yy < self.tar.N.shape[0])
        )

    def _shared_mask(self, points_tar, *, pixel_coordinates=None):
        """Return original NormalFlow's current target-contact rows.

        NormalFlow chooses reference pixels by curvature, then filters each GN
        iteration only by the reprojected *target* contact and image bounds.
        Reference contact belongs to its post-solve SCR/CCS quality gate, not
        to its least-squares row selection.
        """
        xx, yy = (self._pixel_coordinates(points_tar) if pixel_coordinates is None
                  else pixel_coordinates)
        contact = (np.asarray(self.tar.C, dtype=np.float32)
                   if self._target_maps is None else self._target_maps[2])
        target_contact = wide_remap(
            contact, xx, yy, mode=cv2.INTER_NEAREST
        )[:, 0] > 0.5
        return target_contact & self._in_image(xx, yy)

    def _shared_projection(self, reference_pose, target_pose, min_points=10,
                           apply_normal_z_constraint=True):
        """Project reference points and keep current shared contact samples."""
        if len(self.candidate_points) < 10:
            return None
        normal_target_pose = (self._normal_target_pose(reference_pose, target_pose)
                              if apply_normal_z_constraint else target_pose)
        cache = self._projection_cache
        if cache is not None:
            # Key by pose values, including the effective normal-Z constraint.
            # A caller can update a pose array in place between evaluations.
            key = (np.asarray(reference_pose).tobytes(),
                   np.asarray(normal_target_pose).tobytes())
            cached = cache.get(key)
            if cached is not None:
                projected, count = cached
                if apply_normal_z_constraint:
                    self.active_count = count
                return projected if count >= min_points else None
        relative = np.linalg.inv(normal_target_pose) @ reference_pose
        rotation = relative[:3, :3]
        points_tar = self.candidate_points @ rotation.T + relative[:3, 3]
        xx, yy = self._pixel_coordinates(points_tar)
        normal_map, height_map = (
            (np.asarray(self.tar.N, dtype=np.float32),
             np.asarray(self.tar.H, dtype=np.float32))
            if self._target_maps is None else self._target_maps[:2])
        normal_tar = wide_remap(normal_map, xx, yy)[:, 0, :]
        height_tar = wide_remap(height_map, xx, yy)[:, 0]
        shared = (
            self._shared_mask(points_tar, pixel_coordinates=(xx, yy))
            & np.isfinite(normal_tar).all(axis=1) & np.isfinite(height_tar)
        )
        count = int(np.count_nonzero(shared))
        projected = relative, rotation, points_tar, normal_tar, height_tar, shared
        if cache is not None:
            if len(cache) == 4:
                del cache[next(iter(cache))]
            # Cache the sample count separately from the caller's threshold.
            # A failed >=10-point query can still serve a >=1-point query.
            cache[key] = projected, count
        if apply_normal_z_constraint:
            self.active_count = count
        if count < min_points:
            return None
        return projected

    def correspondences(self, reference_pose, target_pose):
        """Return the active normal-row pixel pairs at the supplied solve pose."""
        projected = self._shared_projection(reference_pose, target_pose)
        if projected is None:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty.copy()
        _, _, points_tar, _, _, shared = projected
        points_ref = self.candidate_points[shared]
        reference_xy = points_ref[:, :2] * 1000.0 / self.ppmm
        reference_xy += np.array((self.ref.N.shape[1], self.ref.N.shape[0])) / 2 - 0.5
        xx, yy = self._pixel_coordinates(points_tar[shared])
        return reference_xy, np.column_stack((xx, yy))

    def _in_residual_space(self, rows, normal_error, *jacobians):
        """Flatten (N, 3) errors and (3N, 6) Jacobians in the residual space."""
        if self.residual_space == "vector":
            return (normal_error.reshape(-1),) + jacobians
        basis = normal_tangent_basis(self.candidate_normals[rows])
        residual = np.einsum("nij,nj->ni", basis, normal_error).reshape(-1)
        return (residual,) + tuple(
            np.einsum("nij,njk->nik", basis, J.reshape(-1, 3, J.shape[1])).reshape(-1, J.shape[1])
            for J in jacobians)

    def evaluate(self, reference_pose, target_pose):
        """Compatibility entry point for the reference-map IC linearization."""
        return self.evaluate_ic(reference_pose, target_pose)

    def evaluate_ic(self, reference_pose, target_pose):
        """Return the IC residual model in right body-local pose coordinates.

        get_J() differentiates the inverse-warped reference normal map at the
        identity increment once per image pair. For T = target_T_reference,
        an IC correction d satisfies T_new = T inv(d). A right target-pose
        increment eta induces correction coordinates dp = Ad(inv(T)) eta
        to first order, so the
        residual model is b - J_ref Ad(inv(T)) eta. The cached J_ref is fixed;
        contact rows and its mapping into target coordinates are not.

        This is an IC approximation, not the exact derivative of the forward
        residual at an arbitrary misaligned pose. The 5-column reference J
        is embedded in 6D with a zero reference-Z column. Height and IMU rows
        constrain the remaining motion in the joint navigation solve.
        """
        projected = self._shared_projection(reference_pose, target_pose)
        if projected is None:
            return None
        relative, rotation, _, normal_tar, _, shared = projected
        self.last_active_rows = shared.copy()
        dtype = self.candidate_normal_jacobian.dtype
        normal_error = (
            normal_tar[shared] @ np.linalg.inv(rotation).T.astype(self.candidate_normals.dtype)
            - self.candidate_normals[shared]
        )

        if self._projection_cache is not None:
            if self._ic_reference_blocks is None:
                source = np.transpose(self.candidate_normal_jacobian, (2, 0, 1))
                blocks = np.zeros((len(source), 3, 6), dtype=dtype)
                blocks[:, :, :5] = source
                self._ic_reference_blocks = blocks
            # Only the fixed reference layout is shared. Select current rows
            # into an independent array before applying the current pose map.
            normal_blocks = self._ic_reference_blocks[shared]
        else:
            source = np.transpose(self.candidate_normal_jacobian, (2, 0, 1))[shared]
            normal_blocks = np.zeros((len(source), 3, 6), dtype=dtype)
            normal_blocks[:, :, :5] = source
        normal_relative = normal_blocks.reshape(-1, 6).astype(np.float64, copy=False)

        reference_jacobian, target_jacobian = normal_ic_pose_jacobians(normal_relative, relative)
        if self.fixed_reference_z_m is not None:
            # The normal residual depends on reference-frame XY translation.
            # Project the target's right-tangent translation onto that plane.
            reference_R_target = (
                reference_pose[:3, :3].T @ target_pose[:3, :3]
            )
            xy = np.diag((1.0, 1.0, 0.0))
            tangent_xy = reference_R_target.T @ xy @ reference_R_target
            target_jacobian[:, 3:6] = target_jacobian[:, 3:6] @ tangent_xy
        return self._in_residual_space(shared, normal_error, reference_jacobian, target_jacobian)

    def evaluate_relinearized(self, reference_pose, target_pose):
        """Rebuild the normal Jacobian from the current target-map locations.

        Use pixel-scale central differences because cv2.remap quantizes its
        interpolation coordinates. Infinitesimal pose differences can otherwise
        return zero spatial derivatives. The residual and contact selection use
        the same sampler as evaluate_ic(). The target_relinearized mode calls
        this at each current GN state, including the first iteration.
        """
        projected = self._shared_projection(reference_pose, target_pose)
        if projected is None:
            return None
        relative, rotation, points_tar, normal_tar, _, shared = projected
        self.last_active_rows = shared.copy()
        points = points_tar[shared]
        normals = normal_tar[shared]
        xx, yy = self._pixel_coordinates(points)
        field = (np.asarray(self.tar.N, dtype=np.float32)
                 if self._target_maps is None else self._target_maps[0])
        gradient = np.zeros((len(points), 3, 3), dtype=np.float64)
        gradient[:, :, 0] = (
            wide_remap(field, xx + 1, yy)[:, 0, :]
            - wide_remap(field, xx - 1, yy)[:, 0, :]
        ) * (500.0 / self.ppmm)
        gradient[:, :, 1] = (
            wide_remap(field, xx, yy + 1)[:, 0, :]
            - wide_remap(field, xx, yy - 1)[:, 0, :]
        ) * (500.0 / self.ppmm)
        # For a target body-local increment, dq = [q]x dtheta - dp,
        # and d(R_ref_target n) = R_ref_target (-[n]x dtheta + dn).
        point_skew = np.cross(points[:, None, :], np.eye(3)).transpose(0, 2, 1)
        normal_skew = np.cross(normals[:, None, :], np.eye(3)).transpose(0, 2, 1)
        blocks = np.empty((len(points), 3, 6), dtype=np.float64)
        blocks[:, :, :3] = rotation.T @ (gradient @ point_skew - normal_skew)
        blocks[:, :, 3:] = -rotation.T @ gradient
        target_jacobian = blocks.reshape(-1, 6)
        reference_jacobian = -target_jacobian @ adjoint(relative)
        if self.fixed_reference_z_m is not None:
            reference_R_target = reference_pose[:3, :3].T @ target_pose[:3, :3]
            tangent_xy = reference_R_target.T @ np.diag((1., 1., 0.)) @ reference_R_target
            target_jacobian[:, 3:6] = target_jacobian[:, 3:6] @ tangent_xy
        normal_error = normals @ np.linalg.inv(rotation).T - self.candidate_normals[shared]
        return self._in_residual_space(shared, normal_error, reference_jacobian, target_jacobian)

    def residual_on_fixed_rows(self, reference_pose, target_pose, rows):
        """Evaluate the current iteration's rows at a candidate pose.

        Reject only when none of these rows remains valid at the candidate.
        Omit rows that leave contact so the mean cost counts valid points
        only. New rows enter at the next linearization.
        """
        projected = self._shared_projection(reference_pose, target_pose, min_points=1)
        if projected is None:
            return None
        _, rotation, _, normal_tar, _, shared = projected
        rows = np.asarray(rows, dtype=bool)
        if rows.shape != shared.shape or not np.any(shared[rows]):
            return None
        valid_rows = rows & shared
        normal_error = (
            normal_tar[valid_rows] @ np.linalg.inv(rotation).T
            - self.candidate_normals[valid_rows]
        )
        return self._in_residual_space(valid_rows, normal_error)[0]

    def target_z_height_correction(self, reference_pose, target_pose):
        """Return the original NormalFlow mean height correction in metres.

        Holding rotation and XY fixed, adding this value to
        ``target_T_reference`` Z makes the mean height error zero.
        This uses the actual pose even when the normal-only Z override is on.
        Its derivative with respect to target GelSight body-local translation
        along Z is +1 (the inverse relative transform moves points by -dp).
        """
        projected = self._shared_projection(
            reference_pose, target_pose, apply_normal_z_constraint=False
        )
        if projected is None:
            return None
        _, _, points_tar, _, height_tar, shared = projected
        height_error = height_tar[shared] * self.ppmm / 1000.0 - points_tar[shared, 2]
        return float(np.mean(height_error))


def normalflow_quality(ref, tar, tar_T_ref, ppmm, n_samples=3000):
    """Evaluate the original SCR/CCS gate without running registration again."""
    count = min(int(n_samples), ref.L.size)
    if count < 10:
        return 0.0, 0.0
    mask, points = curvature_selection(ref, count, ppmm)
    ref_laplacian = ref.L[mask]
    ref_contact = ref.C[mask]
    if len(points) < 10 or not np.any(ref_contact):
        return 0.0, 0.0
    target_laplacian, target_contact = get_backproj_laplacian(
        tar.L, tar.C, points, np.asarray(tar_T_ref, dtype=np.float32), ppmm
    )
    shared = ref_contact & target_contact
    denominator = np.sum(np.abs(ref_laplacian))
    if np.count_nonzero(shared) < 10 or denominator <= 0.0:
        return 0.0, 0.0
    scr = np.sum(np.abs(ref_laplacian[shared])) / denominator
    reference_norm = np.linalg.norm(ref_laplacian[shared])
    target_norm = np.linalg.norm(target_laplacian[shared])
    if reference_norm <= 0.0 or target_norm <= 0.0:
        return float(scr), 0.0
    ccs = target_laplacian[shared] @ ref_laplacian[shared] / reference_norm / target_norm
    return float(scr), float(ccs)
