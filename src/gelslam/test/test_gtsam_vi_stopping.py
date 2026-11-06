"""Origin rotation/XY stopping and the five-step normal-only Huber phase."""

import gtsam
import numpy as np
import pytest

from gelslam.core import gtsam_vi as gv
from gelslam.core.inertial import ImuSample
from gelslam.core.numpy_vi import NavigationState


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


def force_step(monkeypatch, column=None, value=.1):
    increment = np.zeros(15)
    if column is not None:
        increment[column] = value
    values = gtsam.VectorValues()
    for key, part in ((gv.POSE_J, increment[:6]), (gv.VELOCITY_J, increment[6:9]),
                      (gv.BIAS_J, increment[9:])):
        values.insert(key, part)
    monkeypatch.setattr(gv, 'solve_gaussian_qr', lambda graph: values)


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
