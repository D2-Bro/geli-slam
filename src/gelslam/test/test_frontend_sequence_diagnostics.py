"""Diagnostics for invalid measurements after a full frontend GN step."""
import numpy as np

from gelslam.core.gtsam_vi import GtsamVisualInertialSolver
from gelslam.core.inertial import ImuSample
from gelslam.core.numpy_vi import NavigationState


def test_losing_all_contact_rows_fails_without_mutating_input():
    """A failed refinement must not publish a result or change its input."""
    rng = np.random.default_rng(20261013)
    jacobian = rng.normal(size=(30, 6))

    class RejectedRows:
        last_active_rows = np.ones(10, dtype=bool)
        active_count = 10

        def __len__(self):
            return 10

        def evaluate(self, _reference, _target):
            return np.ones(30), jacobian, jacobian

        def residual_on_fixed_rows(self, _reference, _target, _rows):
            return None

    solver = GtsamVisualInertialSolver({"gravity_mps2": 9.80665})
    previous = NavigationState(1., np.eye(4), np.zeros(3), np.zeros(3), np.zeros(3))
    current = previous.copy()
    current.timestamp = 1.1
    samples = [ImuSample(t, [0, 0, 9.80665], [0, 0, 0]) for t in (1., 1.05, 1.1)]
    result = solver.refine_current(RejectedRows(), np.eye(4), previous, current,
                                   samples, np.eye(4))
    assert result is None
    assert np.array_equal(current.pose, np.eye(4))
    assert np.array_equal(current.velocity, np.zeros(3))
    trace = solver.last_optimization_trace
    assert len(trace) == 1
    assert trace[0]["accepted_scale"] == 1.
    assert len(trace[0]["attempts"]) == 1
    assert trace[0]["stop_reason"] == 'invalid_updated_residual'
    assert solver.last_refinement_failure == 'invalid_updated_residual'
