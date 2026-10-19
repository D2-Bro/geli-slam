"""Verify replacement time, bias policy, and the resulting IMU prediction."""
import io
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from gelslam.core.gtsam_vi import GtsamVisualInertialSolver
from gelslam.core.inertial import ImuBuffer, ImuSample
from gelslam.core.numpy_vi import NavigationState
from gelslam.core.simulation_ideal_imu import SimulationIdealImu
from gelslam.core.tracker import Tracker


def truth_file(tmp_path, zipped=False, **changes):
    data = dict(t_s=np.array([0., .5, 1.]),
                ideal_specific_force_m_s2=np.tile([0., 0., 9.81], (3, 1)),
                ideal_gyro_rad_s=np.zeros((3, 3)),
                bias_accel_m_s2=np.tile([.1, -.2, .3], (3, 1)),
                bias_gyro_rad_s=np.tile([.01, -.02, .03], (3, 1)))
    data.update(changes)
    stream = io.BytesIO()
    np.savez(stream, **data)
    path = tmp_path / ('truth.zip' if zipped else 'truth.npz')
    if zipped:
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('sequence/assets/trajectory_truth.npz', stream.getvalue())
    else:
        path.write_bytes(stream.getvalue())
    return str(path)


def tracker(config):
    obj = object.__new__(Tracker)
    obj.imu_config = config
    obj.imu_enabled = True
    obj.imu_buffer = ImuBuffer()
    obj.messages = []
    obj.logger = SimpleNamespace(warning=obj.messages.append)
    obj._configure_simulation_ideal_imu()
    return obj


@pytest.mark.parametrize('zipped', [False, True])
@pytest.mark.parametrize('include_bias', [False, True])
def test_input_replacement_predicts_stationary_motion(tmp_path, zipped, include_bias):
    path = truth_file(tmp_path, zipped)
    obj = tracker(dict(enabled=True, simulation_ideal_imu=dict(
        enabled=True, truth_path=path, time_offset_s=10., include_bias_gt=include_bias)))
    for t in np.linspace(10., 11., 101):
        raw = ImuSample(t, [20., 30., 40.], [1., 2., 3.])
        assert obj.add_imu_sample(raw)
        assert np.array_equal(raw.acceleration, [20., 30., 40.])
    ba = np.array([.1, -.2, .3]) if include_bias else np.zeros(3)
    bg = np.array([.01, -.02, .03]) if include_bias else np.zeros(3)
    previous = NavigationState(10., np.eye(4), np.zeros(3), ba, bg)
    solver = GtsamVisualInertialSolver(dict(
        gravity_mps2=9.81, frontend_initialize_from_imu_preintegration=True))
    predicted, pim = solver.predict_current(previous, obj.imu_buffer.samples_covering(10., 11.), np.eye(4))
    assert predicted.timestamp == 11.
    assert pim.deltaTij() == pytest.approx(1.)
    assert np.allclose(predicted.velocity, 0., atol=1e-12)
    assert np.allclose(predicted.pose, np.eye(4), atol=1e-12)
    assert not obj.add_imu_sample(ImuSample(12., [20., 30., 40.], [1., 2., 3.]))
    assert obj.imu_buffer.latest_at_or_before(12.).timestamp == 11.
    assert 'outside simulation truth range' in obj.messages[-1]


def test_timestamp_interpolation_preserves_bias_variation(tmp_path):
    path = truth_file(tmp_path, bias_accel_m_s2=np.array([[0., 0., 0.], [2., 0., 0.], [4., 0., 0.]]))
    source = SimulationIdealImu(path, 1., True)
    sample = source.sample_at(1.25)
    assert sample.timestamp == 1.25
    assert np.allclose(sample.acceleration, [1., 0., 9.81])
    assert source.sample_at(.5) is None
    assert source.sample_at(2.5) is None


def test_disabled_flag_never_reads_truth_or_changes_input():
    obj = tracker(dict(simulation_ideal_imu=dict(enabled=False, truth_path='/missing/truth.npz')))
    raw = ImuSample(1., [1., 2., 3.], [.1, .2, .3])
    assert obj.add_imu_sample(raw)
    assert obj.imu_buffer.samples_between(1., 1.)[0] is raw


@pytest.mark.parametrize('settings', [dict(enabled='true'), dict(enabled=True),
                                      dict(enabled=True, truth_path='x', include_bias_gt='false')])
def test_invalid_settings_fail_before_input(settings):
    with pytest.raises(ValueError):
        SimulationIdealImu.from_config(dict(enabled=True, simulation_ideal_imu=settings))


@pytest.mark.parametrize('change', [dict(t_s=[0., 0., 1.]),
                                    dict(ideal_gyro_rad_s=np.zeros((2, 3))),
                                    dict(ideal_specific_force_m_s2=np.full((3, 3), np.nan))])
def test_invalid_truth_fails_at_load(tmp_path, change):
    with pytest.raises(ValueError):
        SimulationIdealImu(truth_file(tmp_path, **change), 1., True)
