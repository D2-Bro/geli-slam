"""Frozen, correlated pose/navigation priors with explicit variable identity.

Rows use 1/2 ||A dx - rhs||^2, hence H dx = A.T rhs = -g_cost.
Elimination projects out the eliminated row space with a rank-aware SVD
least-squares solve. This is the square-root form of the Schur complement;
it never forms an inverse or inserts damping into retained information.
"""

from dataclasses import dataclass

import numpy as np

from gelslam.core.numpy_vi import NavigationState
from gelslam.core.split_pose import (
    split_pose_coordinate_jacobian, split_pose_coordinates, split_pose_plus,
)


RANK_TOLERANCE = 1e-12
# Relative singular-value threshold for solves from H = A^T A, whose small
# singular values carry about sqrt(eps) * s_max noise. Real frontend systems
# had s_min / s_max >= 2.2e-4 (trial_001, 4000 scaled systems).
NORMAL_RANK_TOLERANCE = 1e-6
STATE_KINDS = ("pose", "velocity", "accel_bias", "gyro_bias")


@dataclass(frozen=True)
class VariableBlock:
    kind: str
    timestamp: float

    @property
    def width(self):
        return 6 if self.kind == "pose" else 3


def state_blocks(timestamp):
    return tuple(VariableBlock(kind, float(timestamp)) for kind in STATE_KINDS)


def state_values(state):
    return {block: np.array(getattr(state, block.kind), copy=True)
            for block in state_blocks(state.timestamp)}


def navigation_state(values, timestamp):
    return NavigationState(timestamp, *(values[b].copy() for b in state_blocks(timestamp)))


class VariableLayout:
    """Ordered unique blocks. An aliased reference/state pose has one column."""

    def __init__(self, blocks, fixed_blocks=()):
        self.fixed_blocks = frozenset(fixed_blocks)
        self.blocks = tuple(b for b in dict.fromkeys(blocks) if b not in self.fixed_blocks)
        self.slices = {}
        self._embed_plans = {}
        self.dimension = 0
        for block in self.blocks:
            if block.kind not in STATE_KINDS or not np.isfinite(block.timestamp):
                raise ValueError("invalid variable ID")
            self.slices[block] = slice(self.dimension, self.dimension + block.width)
            self.dimension += block.width

    def columns(self, blocks):
        columns = [np.arange(self.slices[b].start, self.slices[b].stop)
                   for b in blocks if b not in self.fixed_blocks]
        return np.concatenate(columns).astype(int) if columns else np.empty(0, int)

    def embed(self, rows, blocks):
        blocks = tuple(blocks)
        plans = getattr(self, "_embed_plans", None)
        if plans is None:  # Layout restored from a pickle written before this cache.
            plans = self._embed_plans = {}
        plan = plans.get(blocks)
        if plan is None:
            # The layout is immutable, so the column mapping is reused across
            # GN iterations; only the row values change.
            copies, offset = [], 0
            for block in blocks:
                if block not in self.fixed_blocks:
                    copies.append((self.slices[block], slice(offset, offset + block.width)))
                offset += block.width
            plan = plans[blocks] = (offset, tuple(copies))
        if rows.shape[1] != plan[0]:
            raise ValueError("factor block dimensions do not match")
        result = np.zeros((len(rows), self.dimension))
        for destination, source in plan[1]:
            # Sum columns when two endpoints name the same physical variable.
            result[:, destination] += rows[:, source]
        return result

    def plus(self, values, step):
        # Constants are copied exactly, without a manifold retraction at zero.
        result = {b: v.copy() for b, v in values.items()}
        for block in self.blocks:
            delta = step[self.slices[block]]
            result[block] = (split_pose_plus(values[block], delta) if block.kind == "pose"
                             else values[block] + delta)
        return result

    def expanded_step(self, blocks, step):
        return np.concatenate([np.zeros(b.width) if b in self.fixed_blocks
                               else step[self.slices[b]] for b in blocks])


def rank_solve(A, rhs):
    """Scale columns by their norm, solve in the observable subspace only."""
    A, rhs = np.asarray(A, float), np.asarray(rhs, float)
    if not np.isfinite(A).all() or not np.isfinite(rhs).all():
        raise ValueError("nonfinite linear system")
    scales = np.linalg.norm(A, axis=0)
    # A null column can acquire machine-roundoff entries during SVD row
    # compression. Do not normalize that noise into a unit-information column.
    inactive = scales <= np.finfo(float).eps * max(A.shape) * max(float(scales.max(initial=0.)), 1.)
    scales[inactive] = 1.
    scaled = A / scales
    scaled[:, inactive] = 0.
    solution, _, rank, singular = np.linalg.lstsq(scaled, rhs, rcond=RANK_TOLERANCE)
    solution[inactive] = 0.
    return solution / (scales if rhs.ndim == 1 else scales[:, None]), rank, singular


def eliminate_rows(A, rhs, eliminate_columns, retain_columns):
    """Schur Hrr-Hrm solve(Hmm,Hmr), and RHS, without normal-equation squaring."""
    A, rhs = np.asarray(A, float), np.asarray(rhs, float)
    if A.ndim != 2 or rhs.shape != (len(A),):
        raise ValueError("row dimensions do not match")
    if not np.isfinite(A).all() or not np.isfinite(rhs).all():
        raise ValueError("marginalization rows must be finite")
    Am, Ar = A[:, eliminate_columns], A[:, retain_columns]
    scales = np.linalg.norm(Ar, axis=0)
    zero_columns = scales == 0.
    scales[scales == 0.] = 1.
    if Am.shape[1]:
        solution, _, _ = rank_solve(Am, np.column_stack((Ar, rhs)))
        projected = np.column_stack((Ar, rhs)) - Am @ solution
        Ar, rhs = projected[:, :-1], projected[:, -1]
    # Compress to at most retained_dimension rows. Column scaling is used for
    # rank selection only. Undo it in the stored rows, preserving SI units.
    u, singular, vt = np.linalg.svd(Ar / scales, full_matrices=False)
    active = singular > RANK_TOLERANCE * max(float(singular[0]) if len(singular) else 0., 1.)
    compressed = singular[active, None] * vt[active] * scales
    compressed[:, zero_columns] = 0.
    return compressed, u[:, active].T @ rhs


def _scaled_normal_eigh(H, g, inactive):
    """Eigenpairs of the column-scaled H = A^T A, largest first, as singular values of A."""
    scales = np.sqrt(np.clip(np.diag(H), 0., None))
    inactive = inactive(scales)
    scales[inactive] = 1.
    Hs, gs = H / np.outer(scales, scales), g / scales
    Hs[inactive], Hs[:, inactive], gs[inactive] = 0., 0., 0.
    eigenvalues, vectors = np.linalg.eigh(Hs)
    singular = np.sqrt(np.clip(eigenvalues[::-1], 0., None))
    return scales, inactive, singular, vectors[:, ::-1], gs


def rank_solve_normal(H, g, rows):
    """rank_solve(A, rhs) from H = A^T A and g = A^T rhs of a system with `rows` rows.

    Same column scaling as rank_solve. Singular values from H are accurate only
    to about sqrt(eps) of the largest one, so an exact null direction appears
    near 1e-8 instead of 1e-16. Use NORMAL_RANK_TOLERANCE, not RANK_TOLERANCE.
    """
    H, g = np.asarray(H, float), np.asarray(g, float)
    if not np.isfinite(H).all() or not np.isfinite(g).all():
        raise ValueError("nonfinite linear system")
    scales, inactive, singular, vectors, gs = _scaled_normal_eigh(
        H, g, lambda s: s <= np.finfo(float).eps * max(rows, len(s))
        * max(float(s.max(initial=0.)), 1.))
    active = singular > NORMAL_RANK_TOLERANCE * (singular[0] if len(singular) else 0.)
    solution = vectors[:, active] @ ((vectors[:, active].T @ gs) / singular[active] ** 2)
    solution[inactive] = 0.
    return solution / scales, int(active.sum()), singular


def compress_normal(H, g):
    """eliminate_rows(A, rhs, [], all columns) from H = A^T A and g = A^T rhs."""
    H, g = np.asarray(H, float), np.asarray(g, float)
    if not np.isfinite(H).all() or not np.isfinite(g).all():
        raise ValueError("marginalization rows must be finite")
    scales, zero_columns, singular, vectors, gs = _scaled_normal_eigh(H, g, lambda s: s == 0.)
    active = singular > NORMAL_RANK_TOLERANCE * max(float(singular[0]) if len(singular) else 0., 1.)
    compressed = singular[active, None] * vectors[:, active].T * scales
    compressed[:, zero_columns] = 0.
    return compressed, (vectors[:, active].T @ gs) / singular[active]


@dataclass
class RetainedPrior:
    layout: VariableLayout
    anchors: dict
    A: np.ndarray
    rhs: np.ndarray
    timestamp: float
    # Constant-size ownership audit. All factors up to this accepted time
    # have been consumed. No old visual/IMU factor is retained separately.
    absorbed_intervals: int = 0
    fixed_values: dict = None

    def __post_init__(self):
        self.anchors = {b: np.array(self.anchors[b], copy=True) for b in self.layout.blocks}
        self.fixed_values = {b: np.array(v, copy=True) for b, v in (self.fixed_values or {}).items()}
        if self.layout.fixed_blocks != frozenset(self.fixed_values):
            raise ValueError("fixed variable IDs do not match the prior layout")
        self.A, self.rhs = np.array(self.A, copy=True), np.array(self.rhs, copy=True)
        if self.A.shape != (len(self.rhs), self.layout.dimension):
            raise ValueError("prior dimensions do not match")
        if not all(np.isfinite(v).all() for v in (*self.values.values(), self.A, self.rhs)):
            raise ValueError("prior must be finite")

    @property
    def anchor(self):
        return navigation_state(self.values, self.timestamp)

    @property
    def values(self):
        return {**self.fixed_values, **self.anchors}

    @property
    def information(self):
        return self.A.T @ self.A

    def matches(self, state):
        return (abs(self.timestamp - state.timestamp) < 1e-9
                and all(np.allclose(self.values[b], value, atol=1e-8, rtol=0.)
                        for b, value in state_values(state).items()))

    def coordinates(self, values):
        return np.concatenate([
            split_pose_coordinates(self.anchors[b], values[b]) if b.kind == "pose"
            else values[b] - self.anchors[b] for b in self.layout.blocks])

    def linearize_values(self, values, layout=None):
        transport = np.eye(self.layout.dimension)
        for b in self.layout.blocks:
            if b.kind == "pose":
                sl = self.layout.slices[b]
                transport[sl, sl] = split_pose_coordinate_jacobian(self.anchors[b], values[b])
        A = self.A @ transport
        return ((layout.embed(A, self.layout.blocks) if layout is not None else A),
                self.rhs - self.A @ self.coordinates(values))

    def error_values(self, values):
        residual = self.A @ self.coordinates(values) - self.rhs
        return .5 * float(residual @ residual)

    def condition_fixed(self, fixed_values, *, replace=False):
        """Condition exactly on constants; do not Schur-marginalize them.

        Preserve every free-variable cross term and the shifted residual RHS.
        replace is for a new trial's datum; old constants have already been
        absorbed in the conditional rows and need no longer be stored.
        """
        delta = np.zeros(self.layout.dimension)
        for b, value in fixed_values.items():
            if b in self.layout.slices:
                delta[self.layout.slices[b]] = (
                    split_pose_coordinates(self.anchors[b], value) if b.kind == "pose"
                    else np.asarray(value) - self.anchors[b])
            elif b not in self.fixed_values or not np.array_equal(value, self.fixed_values[b]):
                raise ValueError("cannot create or move an existing fixed variable")
        fixed = {} if replace else dict(self.fixed_values)
        fixed.update(fixed_values)
        layout = VariableLayout(self.layout.blocks, fixed)
        return RetainedPrior(layout, self.anchors, self.A[:, self.layout.columns(layout.blocks)],
                             self.rhs - self.A @ delta, self.timestamp,
                             self.absorbed_intervals, fixed)

    def retain(self, blocks, timestamp=None):
        """Remove actual old variables, never rename an old reference ID."""
        if not set(blocks).issubset(self.values):
            raise ValueError("new reference is not in the retained state")
        layout = VariableLayout(blocks, self.fixed_values)
        if not set(layout.blocks).issubset(self.layout.blocks):
            raise ValueError("new reference is not in the retained state")
        removed = [b for b in self.layout.blocks if b not in layout.blocks]
        A, rhs = eliminate_rows(self.A, self.rhs, self.layout.columns(removed),
                                self.layout.columns(layout.blocks))
        return RetainedPrior(layout, self.anchors, A, rhs,
                             self.timestamp if timestamp is None else timestamp,
                             self.absorbed_intervals, self.fixed_values)

    def current_prior(self):
        from gelslam.core.marginal_prior import NavigationPrior
        prior = self.retain(state_blocks(self.timestamp))
        if prior.layout.dimension != 15:
            raise ValueError("a fixed current pose has no finite 15D covariance representation")
        return NavigationPrior(prior.anchor, prior.A, prior.rhs)

    def reframe(self, rotation, origin):
        anchors = {b: v.copy() for b, v in self.anchors.items()}
        fixed = {b: v.copy() for b, v in self.fixed_values.items()}
        transport = np.eye(self.layout.dimension)
        for b in self.layout.blocks:
            if b.kind == "pose":
                anchors[b][:3, :3] = rotation @ anchors[b][:3, :3]
                anchors[b][:3, 3] = rotation @ (anchors[b][:3, 3] - origin)
            elif b.kind == "velocity":
                anchors[b] = rotation @ anchors[b]
                sl = self.layout.slices[b]
                transport[sl, sl] = rotation.T
        for b in fixed:
            if b.kind == "pose":
                fixed[b][:3, :3] = rotation @ fixed[b][:3, :3]
                fixed[b][:3, 3] = rotation @ (fixed[b][:3, 3] - origin)
            elif b.kind == "velocity":
                fixed[b] = rotation @ fixed[b]
        return RetainedPrior(self.layout, anchors, self.A @ transport, self.rhs,
                             self.timestamp, self.absorbed_intervals, fixed)
