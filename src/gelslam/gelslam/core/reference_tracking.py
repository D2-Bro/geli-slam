"""Tracker-local reference pose + current navigation marginalization mode."""

from contextlib import nullcontext
from time import monotonic

import numpy as np

from gelslam.core.gtsam_vi import (
    GtsamVisualInertialSolver, _bias_factor, _imu_values, _navigation_step_norms,
    _normal_huber_linearization, _split_imu_factor, navigation_factor_terms,
    preintegrate_imu,
)
from gelslam.core.gt_state_diagnostic import resolve_gt_state_test
from gelslam.core.imu_navigation import gelsight_pose, gelsight_pose_jacobian
from gelslam.core.marginal_prior import NavigationPrior
from gelslam.core.numpy_vi import adjoint
from gelslam.core.retained_prior import (
    RetainedPrior, VariableBlock, VariableLayout, eliminate_rows, navigation_state,
    rank_solve, state_blocks, state_values,
)
from gelslam.core.split_pose import split_pose_plus

PREVIOUS_STATE_MODES = ("joint", "eliminate_first")


def _expanded_step(layout, blocks, step):
    """Step for the given blocks; zero for fixed or eliminated blocks."""
    return np.concatenate([step[layout.slices[b]] if b in layout.slices else np.zeros(b.width)
                           for b in blocks])


class ReferenceMarginalizationSolver(GtsamVisualInertialSolver):
    """Consume the whole accepted problem once; keep no active old factors.

    Generic solve order is [reference pose, previous state, target state] (36D).
    The first contact pose of each trial is an exact constant (all six DOF).
    Bootstrap has 24 free dimensions, retaining the 15D target. While the
    reference remains that first pose, later solves have 30 free dimensions.
    A later shared reference/previous pose gives 30D, retaining 21D. Changing
    the reference to the target retains 15D until the next accepted target.
    """

    reference_marginalization = True

    def __init__(self, config):
        super().__init__(config)
        if (self.fix_previous_state or self.imu_prediction_only
                or not self.marginalization_enabled or resolve_gt_state_test(config)["enabled"]):
            raise ValueError("reference_marginalization requires estimated 15D states, "
                             "marginalization, and disabled GT state overrides/prediction-only")
        # joint: solve [reference, previous, current] and eliminate previous after GN;
        # the IMU factor is relinearized at the updated previous state every iteration.
        # eliminate_first: eliminate previous from prior+IMU+bias once at its accepted
        # value, then run GN on [reference, current] only (fixed IMU linearization at
        # the previous state). Normal/height rows never involve the previous state.
        self.previous_state_mode = str(config.get("frontend_reference_previous_state", "joint"))
        if self.previous_state_mode not in PREVIOUS_STATE_MODES:
            raise ValueError("frontend_reference_previous_state must be one of "
                             + ", ".join(PREVIOUS_STATE_MODES))
        self.reference_id = None
        self.first_pose_id = None
        self.last_reference_pose = None  # IMU-origin world pose, candidate only.
        self.last_reference_state = None
        self.last_problem = None

    def _accepted_prior(self, previous):
        prior = self.marginal_prior
        if prior is None:
            # Velocity/bias rows express actual initialization uncertainty.
            # Pose is conditioned exactly below, rather than softly anchored.
            prior = NavigationPrior.initial(previous, self.prior_sigmas)
        if isinstance(prior, NavigationPrior):
            if abs(prior.anchor.timestamp - previous.timestamp) > 1e-9:
                raise ValueError("initial navigation prior timestamp mismatch")
            prior = RetainedPrior(VariableLayout(state_blocks(previous.timestamp)),
                                  state_values(prior.anchor), prior.A, prior.rhs,
                                  previous.timestamp)
        if not isinstance(prior, RetainedPrior) or abs(prior.timestamp - previous.timestamp) > 1e-9:
            raise ValueError("accepted prior timestamp mismatch; refusing to reset history")
        if self.first_pose_id is None:
            self.first_pose_id = VariableBlock("pose", previous.timestamp)
            prior = prior.condition_fixed({self.first_pose_id: previous.pose}, replace=True)
        self.marginal_prior = prior
        if self.reference_id is None:
            self.reference_id = self.first_pose_id
        if self.reference_id not in prior.values:
            raise ValueError("reference pose missing from retained prior")
        return prior

    def set_reference(self, timestamp, state):
        """At an accepted image boundary eliminate an old reference if needed."""
        prior = self._accepted_prior(state)
        reference_id = VariableBlock("pose", float(timestamp))
        blocks = (reference_id,) + state_blocks(state.timestamp)
        if (reference_id == self.reference_id
                and prior.layout.blocks == VariableLayout(blocks, prior.fixed_values).blocks):
            return
        if (reference_id not in prior.layout.blocks
                and reference_id != VariableBlock("pose", state.timestamp)):
            raise ValueError("reference pose was already marginalized; cannot rename or recreate it")
        self.marginal_prior = prior.retain(blocks)
        self.reference_id = reference_id
        self.discard_pending_prior()

    def _problem(self, previous, current):
        prior = self._accepted_prior(previous)
        if current.timestamp <= prior.timestamp:
            raise ValueError("IMU interval already absorbed, or target is not later than anchor")
        layout = VariableLayout((self.reference_id,) + state_blocks(previous.timestamp)
                                + state_blocks(current.timestamp), prior.fixed_values)
        values = {b: v.copy() for b, v in prior.values.items()}
        values.update(state_values(previous))
        values.update(state_values(current))
        values.update({b: v.copy() for b, v in prior.fixed_values.items()})
        return prior, layout, values

    @staticmethod
    def _compress(layout, values, A, rhs, retained, timestamp, count, fixed_values):
        retained = VariableLayout(retained, fixed_values)
        removed = [b for b in layout.blocks if b not in retained.blocks]
        reduced_A, reduced_rhs = eliminate_rows(
            A, rhs, layout.columns(removed), layout.columns(retained.blocks))
        return RetainedPrior(retained, values, reduced_A, reduced_rhs, timestamp, count, fixed_values)

    def _inertial_rows(self, prior, layout, values, previous_time, current_time, factors):
        previous = navigation_state(values, previous_time)
        current = navigation_state(values, current_time)
        Ap, bp = prior.linearize_values(values, layout)
        rows, right = [Ap], [bp]
        # The IMU and bias factors linearize at the same endpoint states.
        linearization_point = _imu_values(previous, current)
        for factor in factors:
            Ai, Aj, rhs = navigation_factor_terms(factor, previous, current,
                                                  values=linearization_point)
            rows.append(np.sqrt(self.imu_weight) * layout.embed(
                np.column_stack((Ai, Aj)), state_blocks(previous_time) + state_blocks(current_time)))
            right.append(np.sqrt(self.imu_weight) * rhs)
        return rows, right

    def propagate_prior(self, previous, current, pim, *, fixed_anchor=False):
        if fixed_anchor:
            raise ValueError("backend conditioning would discard the correlated frontend prior")
        prior, layout, values = self._problem(previous, current)
        rows, right = self._inertial_rows(
            prior, layout, values, previous.timestamp, current.timestamp,
            (_split_imu_factor(pim), _bias_factor(pim, self.config)))
        A, rhs = np.vstack(rows), np.concatenate(right)
        propagated = self._compress(layout, values, A, rhs,
                                    (self.reference_id,) + state_blocks(current.timestamp),
                                    current.timestamp, prior.absorbed_intervals + 1,
                                    prior.fixed_values)
        # eliminate_first builds the same rows and elimination from the same
        # inputs when the solve starts at this prediction; let it reuse them.
        self._last_propagation = dict(prior=prior, pim=pim, blocks=layout.blocks,
                                      values=values, A=A, rhs=rhs, prior_out=propagated)
        return propagated

    def _reuse_propagation(self, prior, pim, layout, values):
        """Rows and elimination from propagate_prior() for identical inputs, else None."""
        cached = getattr(self, "_last_propagation", None)
        if (cached is None or cached["prior"] is not prior or cached["pim"] is not pim
                or cached["blocks"] != layout.blocks or cached["values"].keys() != values.keys()
                or not all(np.array_equal(cached["values"][b], v) for b, v in values.items())):
            return None
        return cached

    def prediction_pose_distance(self, predicted, candidate, prior):
        if isinstance(prior, RetainedPrior):
            prior = prior.current_prior()
        return super().prediction_pose_distance(predicted, candidate, prior)

    def _normal_pose_rows(self, visual, reference, target, Jr, Jt):
        """Preserve the selected target model when both endpoint poses vary.

        Older fixed-Z diagnostics only projected the variable target columns;
        their reference columns were unused. Map those projected rows to the
        variable reference too, so a common rigid pose change is still null.
        """
        if self.normal_zero_pose_z_jacobian:
            Jt = Jt.copy()
            Jt[:, 5] = 0.
        if (self.normal_zero_pose_z_jacobian
                or getattr(visual, "fixed_reference_z_m", None) is not None):
            Jr = -Jt @ adjoint(np.linalg.inv(target) @ reference)
        return Jr, Jt

    def refine_from_keyframe(self, *args, **kwargs):
        raise ValueError("reference marginalization uses tracker-local states; backend map is separate")

    def refine_recovery_reference(self, visual, reference_pose, previous, *args, **kwargs):
        self.set_reference(previous.timestamp, previous)
        return self.refine_current(visual, reference_pose, previous, *args, **kwargs)

    def refine_current(self, visual, reference_pose, previous, current, samples,
                       gelsight_T_imu, preintegrated=None, *, fixed_anchor=False):
        self.discard_pending_prior()
        self.last_optimization_trace = []
        self.last_refinement_failure = None
        self.last_problem = None
        started = monotonic()
        if fixed_anchor:
            raise ValueError("reference pose cannot be hard fixed in reference marginalization")
        if visual is None or len(visual) < 10:
            self.last_refinement_failure = "initial_shared_points_lt_10"
            return None
        if gelsight_T_imu is None:
            self.last_refinement_failure = "missing_extrinsic"
            return None
        cache = getattr(visual, "cache_projections", None)
        try:
            with cache() if cache is not None else nullcontext():
                result = self._solve(visual, previous, current, samples, gelsight_T_imu,
                                     preintegrated)
            return result
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            self.discard_pending_prior()
            self.last_refinement_failure = "reference_marginalization: " + str(error)
            return None
        finally:
            self.last_solve_time_s = monotonic() - started

    def _solve(self, visual, previous, current, samples, extrinsic, pim):
        prior, layout, values = self._problem(previous, current)
        pim = pim if pim is not None else preintegrate_imu(
            samples, previous, self.config, extrinsic, self.gravity)
        if pim is None or not np.isclose(pim.deltaTij(), current.timestamp - previous.timestamp,
                                         rtol=0., atol=1e-9):
            raise ValueError("invalid preintegrated interval")
        factors = (_split_imu_factor(pim), _bias_factor(pim, self.config))
        gel_J = gelsight_pose_jacobian(extrinsic)
        method = ("evaluate_ic" if self.normal_linearization == "inverse_compositional"
                  else "evaluate_relinearized")
        evaluate = getattr(visual, method, visual.evaluate)
        reference_id, current_id = self.reference_id, state_blocks(current.timestamp)[0]
        if self.normal_reference_z_from_imu:
            predicted = self.predict_preintegrated(previous, pim, current.timestamp, force=True)[0]
            relative = (np.linalg.inv(gelsight_pose(values[reference_id], extrinsic))
                        @ gelsight_pose(predicted.pose, extrinsic))
            self.last_fixed_normal_reference_z_m = float(relative[2, 3])
            visual.set_fixed_reference_z(self.last_fixed_normal_reference_z_m)
        self.last_prior_source = "retained_reference_current"
        retained_blocks = (reference_id,) + state_blocks(current.timestamp)
        eliminated = None
        if self.previous_state_mode == "eliminate_first":
            # Same factors as propagate_prior(): prior + IMU + bias at the accepted
            # previous state, previous blocks eliminated once. The result is a
            # fixed linear prior on [reference, current]; only its chart moves.
            propagated = self._reuse_propagation(prior, pim, layout, values)
            if propagated is not None:
                nav_A, nav_rhs = propagated["A"], propagated["rhs"]
                eliminated = propagated["prior_out"]
            else:
                nav_rows, nav_right = self._inertial_rows(
                    prior, layout, values, previous.timestamp, current.timestamp, factors)
                nav_A, nav_rhs = np.vstack(nav_rows), np.concatenate(nav_right)
                eliminated = self._compress(layout, values, nav_A, nav_rhs, retained_blocks,
                                            current.timestamp, prior.absorbed_intervals,
                                            prior.fixed_values)
            initial_values = {b: v.copy() for b, v in values.items()}
        solve_layout = layout if eliminated is None else eliminated.layout

        def navigation_rows(point):
            if eliminated is None:
                return self._inertial_rows(
                    prior, layout, point, previous.timestamp, current.timestamp, factors)
            A_nav, b_nav = eliminated.linearize_values(point, solve_layout)
            return [A_nav], [b_nav]

        def rows_at(point, iteration):
            layout = solve_layout
            reference = gelsight_pose(point[reference_id], extrinsic)
            target = gelsight_pose(point[current_id], extrinsic)
            evaluated = evaluate(reference, target)
            if evaluated is None or len(evaluated[0]) < 30:
                raise ValueError("invalid_normal_residual")
            residual, Jr, Jt = evaluated
            if not all(np.isfinite(v).all() for v in (residual, Jr, Jt)):
                raise ValueError("nonfinite_normal_residual")
            Jr, Jt = self._normal_pose_rows(visual, reference, target, Jr, Jt)
            normal_A, normal_b, normal_cost, _ = _normal_huber_linearization(
                residual, np.column_stack((Jr @ gel_J, Jt @ gel_J)),
                np.sqrt(self.visual_weight), self.normal_huber_delta_at(iteration))
            rows, right = navigation_rows(point)
            rows.append(layout.embed(normal_A, (reference_id, current_id)))
            right.append(-normal_b)
            height = self._height_z_linearization(visual, reference, target, gel_J)
            if height is None:
                raise ValueError("invalid_height_residual")
            if self.height_z_enabled:
                Ht, hb, _ = height
                Hr = -self._height_z_jacobian(np.eye(6)) @ adjoint(
                    np.linalg.inv(target) @ reference) @ gel_J
                rows.append(layout.embed(np.column_stack((Hr, Ht)), (reference_id, current_id)))
                right.append(-hb)
            A, rhs = np.vstack(rows), np.concatenate(right)
            if not np.isfinite(A).all() or not np.isfinite(rhs).all():
                raise ValueError("nonfinite_factor_rows")
            return A, rhs, normal_cost

        rank = 0
        for iteration in range(self.max_iterations):
            A, rhs, normal_cost = rows_at(values, iteration)
            step, rank, singular = rank_solve(A, rhs)
            if rank == 0 or not np.isfinite(step).all():
                raise ValueError("empty_observable_system")
            candidate = solve_layout.plus(values, step)
            if not all(np.isfinite(v).all() for v in candidate.values()):
                raise ValueError("nonfinite_gn_state")
            state_steps = [_navigation_step_norms(_expanded_step(solve_layout, state_blocks(t), step))
                           for t in (previous.timestamp, current.timestamp)]
            # Check every optimized pose, including the reference, in this mode.
            pose_steps = [step[solve_layout.slices[b]] for b in solve_layout.blocks
                          if b.kind == "pose"]
            converged = (iteration + 1 >= self.min_iterations
                         and max(np.linalg.norm(s[:3]) for s in pose_steps) < self.rotation_stop
                         and max(np.linalg.norm(s[3:]) for s in pose_steps) < self.translation_stop)
            self.last_optimization_trace.append(dict(
                iteration=iteration, update_rule="full_gauss_newton",
                variable_reference=reference_id not in prior.fixed_values,
                first_pose_fixed=True,
                solve_dimension=solve_layout.dimension, numerical_rank=int(rank),
                scaled_min_singular=float(singular[-1]), base_cost=float(rhs @ rhs / 2),
                base_normal_cost=normal_cost, raw_step_norms=state_steps,
                stop_reason="step_converged" if converged else "max_iterations"
                if iteration + 1 == self.max_iterations else None))
            values = candidate
            if converged:
                break
        A, rhs, _ = rows_at(values, iteration)
        retained = self._compress(solve_layout, values, A, rhs, retained_blocks,
                                  current.timestamp, prior.absorbed_intervals + 1, prior.fixed_values)
        if eliminated is not None and reference_id.timestamp == previous.timestamp:
            # The previous state is published as the reference state. Recover its
            # eliminated blocks by back-substitution in the same linear system.
            removed = [b for b in layout.blocks if b not in solve_layout.blocks]
            removed_step = rank_solve(
                nav_A[:, layout.columns(removed)],
                nav_rhs - nav_A[:, layout.columns(solve_layout.blocks)]
                @ eliminated.coordinates(values))[0]
            offset = 0
            for block in removed:
                delta = removed_step[offset:offset + block.width]
                values[block] = (split_pose_plus(initial_values[block], delta)
                                 if block.kind == "pose" else initial_values[block] + delta)
                offset += block.width
        self.pending_marginal_prior = retained
        self.last_reference_pose = values[reference_id].copy()
        self.last_reference_state = (navigation_state(values, previous.timestamp)
                                     if reference_id.timestamp == previous.timestamp else None)
        # Only the small normal equation is retained for diagnostics, not raw
        # NormalFlow pixels or old IMU measurements. It is never an active factor.
        H = A.T @ A
        self.last_problem = dict(
            blocks=solve_layout.blocks, solve_dimension=solve_layout.dimension,
            retained_dimension=retained.layout.dimension, hessian=H, rhs=A.T @ rhs,
            # Diagnostic only: rank of the last solved GN system. Avoid a
            # second full solve of the final rows just to report it.
            numerical_rank=int(rank),
            first_pose_id=self.first_pose_id,
            reference_fixed=reference_id in prior.fixed_values,
            symmetry_error=float(np.max(np.abs(H - H.T))),
            factor_ids=(("prior", previous.timestamp),
                        ("normal", reference_id.timestamp, current.timestamp),
                        ("imu", previous.timestamp, current.timestamp),
                        ("bias", previous.timestamp, current.timestamp)),
            active_measurement_factors_after_commit=0,
            absorbed_intervals=retained.absorbed_intervals)
        return navigation_state(values, current.timestamp)
