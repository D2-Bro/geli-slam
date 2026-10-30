"""Origin rotation/XY stopping and the five-step normal-only Huber phase."""

import gtsam
import numpy as np
import pytest

from gelslam.core import gtsam_vi as gv
from gelslam.core.inertial import ImuSample
from gelslam.core.marginal_prior import NavigationPrior
from gelslam.core.numpy_vi import NavigationState
from gelslam.core.imu_navigation import gelsight_pose, gelsight_pose_jacobian
from gelslam.core.split_pose import split_pose_plus


def state(t):
    return NavigationState(t, np.eye(4), np.zeros(3), np.zeros(3), np.zeros(3))


def samples():
    return [ImuSample(t, (0., 0., 9.81), (0., 0., 0.))
            for t in np.linspace(0., .1, 11)]


class FlatVisual:
    def __len__(self):
        return 10

    def evaluate(self, reference, target):
        return np.zeros(30), np.zeros((30, 6)), np.zeros((30, 6))


def test_large_full_pose_step_does_not_stop_as_converged():
    class Visual(FlatVisual):
        def evaluate(self, reference, target):
            residual = np.tile(np.r_[np.zeros(3), target[:3, 3] - [1., 0., 0.]], 5)
            jacobian = np.tile(np.diag([1., 1., 1., .1, 1., 1.]), (5, 1))
            return residual, jacobian, jacobian

    solver = gv.GtsamVisualInertialSolver({
        'gtsam_vi_max_iterations': 3,
        'gtsam_vi_min_iterations': 1,
        'gtsam_vi_marginalization': False,
        'gtsam_vi_translation_stop_m': 2.,
        'gtsam_vi_normal_huber_delta': 0.,
        'dense_normalflow_residual_weight': 1e8,
        'imu_residual_weight': 1e-12,
    })
    result = solver.refine_current(Visual(), np.eye(4), state(0.), state(.1),
                                   samples(), np.eye(4))
    assert result is not None
    traces = solver.last_optimization_trace
    assert len(traces) == 3
    assert all(t['accepted_scale'] == 1. for t in traces)
    assert all(t['raw_step_norms']['target']['translation_m'] > 2. for t in traces)
    assert not any(t['step_converged'] for t in traces)
    assert traces[-1]['stop_reason'] == 'max_iterations'


def force_step(monkeypatch, column=None, value=.1):
    increment = np.zeros(15)
    if column is not None:
        increment[column] = value
    values = gtsam.VectorValues()
    for key, part in ((gv.POSE_J, increment[:6]), (gv.VELOCITY_J, increment[6:9]),
                      (gv.BIAS_J, increment[9:])):
        values.insert(key, part)
    monkeypatch.setattr(gv, 'solve_gaussian_qr', lambda graph: values)


@pytest.mark.parametrize('fixed_previous', [False, True])
def test_stationary_solve_can_converge_only_after_seven_steps(fixed_previous):
    solver = gv.GtsamVisualInertialSolver({
        'frontend_fix_previous_state': fixed_previous})
    result = solver.refine_current(FlatVisual(), np.eye(4), state(0.), state(.1),
                                   samples(), np.eye(4))
    assert result is not None
    traces = solver.last_optimization_trace
    assert len(traces) == 7
    assert not any(t['step_converged'] for t in traces[:6])
    assert traces[-1]['iteration'] == 6
    assert traces[-1]['stop_reason'] == 'step_converged'


@pytest.mark.parametrize('column', [5, 6, 9, 12])
def test_z_velocity_and_bias_updates_do_not_delay_planar_convergence(monkeypatch, column):
    force_step(monkeypatch, column)
    solver = gv.GtsamVisualInertialSolver({'frontend_fix_previous_state': True})
    result = solver.refine_current(FlatVisual(), np.eye(4), state(0.), state(.1),
                                   samples(), np.eye(4))
    assert result is not None
    traces = solver.last_optimization_trace
    assert len(traces) == 7
    assert traces[-1]['step_converged']
    assert all(t['convergence_step_norms']['translation_xy_m'] == 0. for t in traces)
    # The excluded components are still updated by the optimizer.
    values = np.r_[result.pose[:3, 3], result.velocity, result.accel_bias, result.gyro_bias]
    assert values[column - 3] == pytest.approx(.7)


@pytest.mark.parametrize('column', [0, 3, 4])
def test_large_rotation_or_reference_xy_update_prevents_convergence(monkeypatch, column):
    force_step(monkeypatch, column, .001)
    solver = gv.GtsamVisualInertialSolver({
        'frontend_fix_previous_state': True, 'gtsam_vi_max_iterations': 9})
    result = solver.refine_current(FlatVisual(), np.eye(4), state(0.), state(.1),
                                   samples(), np.eye(4))
    assert result is not None
    assert len(solver.last_optimization_trace) == 9
    assert not any(t['step_converged'] for t in solver.last_optimization_trace)
    assert solver.last_optimization_trace[-1]['stop_reason'] == 'max_iterations'


def test_large_previous_update_does_not_delay_target_planar_convergence(monkeypatch):
    previous = state(0.)
    solver = gv.GtsamVisualInertialSolver({})
    solver.marginal_prior = NavigationPrior.initial(previous, np.ones(15))
    force_step(monkeypatch)
    previous_increment = np.zeros(15)
    previous_increment[6] = .1
    from gelslam.core.marginal_prior import SchurComplement
    monkeypatch.setattr(SchurComplement, 'recover_previous',
                        lambda *args, **kwargs: previous_increment.copy())
    result = solver.refine_current(FlatVisual(), np.eye(4), previous, state(.1),
                                   samples(), np.eye(4))
    assert result is not None
    traces = solver.last_optimization_trace
    assert len(traces) == 7
    assert traces[-1]['step_converged']
    assert traces[-1]['raw_step_norms']['previous']['velocity_mps'] == pytest.approx(.1)


def test_planar_stop_uses_reference_ic_coordinates_with_rotated_offset_mount():
    reference, target, mount = np.eye(4), np.eye(4), np.eye(4)
    for pose, rot, xyz in ((reference, [.2, -.1, .3], [.1, -.1, .2]),
                            (target, [-.1, .25, .12], [.12, -.08, .23]),
                            (mount, [.4, -.2, .1], [.004, -.003, .04])):
        pose[:3, :3] = gtsam.Rot3.Expmap(rot).matrix()
        pose[:3, 3] = xyz
    pose_step = np.array([.03, -.02, .01, .002, -.001, .003])
    norms = gv._normalflow_step_norms(reference, target, gelsight_pose_jacobian(mount), pose_step)
    # Independently recover the inverse-compositional transform from a small
    # physical IMU update, including the sensor lever arm and both rotations.
    epsilon = 1e-5
    updated = gelsight_pose(split_pose_plus(target @ mount, epsilon * pose_step), mount)
    T = np.linalg.inv(target) @ reference
    T_new = np.linalg.inv(updated) @ reference
    correction = np.linalg.inv(T_new) @ T
    numerical_rotation = np.linalg.norm(gtsam.Rot3.Logmap(gtsam.Rot3(correction[:3, :3]))) / epsilon
    numerical_xy = np.linalg.norm(correction[:2, 3]) / epsilon
    assert norms['rotation_rad'] == pytest.approx(numerical_rotation, rel=1e-6)
    assert norms['translation_xy_m'] == pytest.approx(numerical_xy, rel=1e-5)
    assert not np.isclose(norms['translation_xy_m'], np.linalg.norm(pose_step[3:5]))


def test_hard_iteration_cap_can_end_before_minimum_without_claiming_convergence():
    solver = gv.GtsamVisualInertialSolver({'gtsam_vi_max_iterations': 3})
    result = solver.refine_current(FlatVisual(), np.eye(4), state(0.), state(.1),
                                   samples(), np.eye(4))
    assert result is not None
    assert len(solver.last_optimization_trace) == 3
    assert solver.last_optimization_trace[-1]['stop_reason'] == 'max_iterations'


@pytest.mark.parametrize('mode', ['inverse_compositional', 'target_relinearized'])
def test_normal_huber_switches_off_at_sixth_step_and_stays_off_in_prior(monkeypatch, mode):
    class OutlierVisual(FlatVisual):
        def evaluate(self, reference, target):
            return np.tile([1., 0., 0.], 10), np.zeros((30, 6)), np.zeros((30, 6))

    deltas = []
    linearize = gv._normal_huber_linearization

    def capture(residual, J, scale, delta, dim=3):
        deltas.append(delta)
        return linearize(residual, J, scale, delta, dim)

    monkeypatch.setattr(gv, '_normal_huber_linearization', capture)
    solver = gv.GtsamVisualInertialSolver({'gtsam_vi_normal_linearization': mode})
    for _ in range(2):  # Each new tracking call restarts its five-step phase.
        deltas.clear()
        result = solver.refine_current(OutlierVisual(), np.eye(4), state(0.), state(.1),
                                       samples(), np.eye(4))
        assert result is not None
        assert deltas == [.4] * 5 + [0.] * 3  # Seven steps plus retained prior.
        traces = solver.last_optimization_trace
        assert [t['normal_huber_active'] for t in traces] == [True] * 5 + [False] * 2
        assert [t['downweighted_normal_points'] for t in traces] == [10] * 5 + [0] * 2
        for i, trace in enumerate(traces):
            expected = .32 if i < 5 else .5
            assert trace['base_normal_cost'] == pytest.approx(expected)
            assert trace['accepted_normal_cost'] == pytest.approx(expected)


@pytest.mark.parametrize('settings', [
    {'gtsam_vi_normal_huber_delta': 0.}, {'gtsam_vi_normal_huber_iterations': 0}])
def test_normal_huber_can_be_disabled(settings):
    solver = gv.GtsamVisualInertialSolver(settings)
    assert all(solver.normal_huber_delta_at(i) == 0. for i in range(10))


@pytest.mark.parametrize('key,value', [
    ('gtsam_vi_min_iterations', 0), ('gtsam_vi_min_iterations', True),
    ('gtsam_vi_min_iterations', 7.5), ('gtsam_vi_normal_huber_iterations', -1),
    ('gtsam_vi_normal_huber_iterations', False), ('gtsam_vi_normal_huber_iterations', '5')])
def test_iteration_schedule_rejects_invalid_counts(key, value):
    with pytest.raises(ValueError, match=key):
        gv.GtsamVisualInertialSolver({key: value})
