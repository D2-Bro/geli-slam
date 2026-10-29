"""Stationary gyro mean covariance at its last measurement timestamp."""
import numpy as np
import pytest

from gelslam.core.inertial import ImuBuffer, ImuSample
from gelslam.core.static_imu_initialization import StaticImuInitializer, resolve_static_imu_config


def setup(enabled=True, mode='fixed_time'):
    settings=dict(enabled=True,mode=mode,gyro_bias_uncertainty_from_samples=enabled)
    if mode=='fixed_time':settings.update(start_time_s=1.,end_time_s=2.)
    cfg=dict(enabled=True,frontend_optimization=True,gtsam_vi_marginalization=True,
             gravity_mps2=9.81,gyroscope_noise_density=.00016,gyroscope_bias_random_walk=.000022,
             static_initialization=settings,
             initial_accel_bias_mps2=[0.,0.,0.],initial_gyro_bias_rps=[.0002,-.0001,.00015])
    effective,resolved,_=resolve_static_imu_config(cfg,np.eye(4))
    return StaticImuInitializer(resolved,effective),ImuBuffer(),cfg


def samples(times):
    return [ImuSample(float(t),[0.,0.,9.81],[.0002,-.0001,.00015]) for t in times]


def observe(detector,buffer,values):
    for s in values:buffer.add(s)
    detector.observe(values)


def test_uniform_stationary_mean_has_white_and_endpoint_random_walk_uncertainty():
    detector,buffer,cfg=setup();n=201;dt=.005
    observe(detector,buffer,samples(np.linspace(1.,2.,n)))
    expected=cfg['gyroscope_noise_density']**2/(n*dt)
    expected+=cfg['gyroscope_bias_random_walk']**2*dt*(n-1)*n*(2*n-1)/(6*n*n)
    assert detector.stationary_gyro_sigma_rps**2==pytest.approx(expected,rel=1e-13)


def test_irregular_stationary_samples_match_full_brownian_covariance_projection():
    detector,_,cfg=setup();dt=np.tile([.005,.007,.004,.006],25)
    times=np.r_[0.,np.cumsum(dt)];values=samples(times)
    detector._set_stationary_anchor(values,np.array([0.,0.,9.81]),np.array([.0002,-.0001,.00015]))
    n=len(times);coeff=-np.ones(n)/n;coeff[-1]+=1
    covariance=cfg['gyroscope_bias_random_walk']**2*np.minimum.outer(times,times)
    expected=coeff@covariance@coeff+cfg['gyroscope_noise_density']**2*np.sum(1./np.r_[dt[0],dt])/(n*n)
    assert detector.stationary_gyro_sigma_rps**2==pytest.approx(expected,rel=1e-13)


def test_stationary_uncertainty_agrees_with_independent_measurement_monte_carlo():
    detector,_,cfg=setup();n=101;dt=.01;sigma_w=.0002;sigma_b=.0003
    detector.config.update(gyroscope_noise_density=sigma_w,gyroscope_bias_random_walk=sigma_b)
    detector._set_stationary_anchor(samples(np.arange(n)*dt),np.array([0.,0.,9.81]),np.zeros(3))
    rng=np.random.default_rng(73);count=12000
    bias=np.column_stack((np.zeros(count),np.cumsum(rng.normal(size=(count,n-1))*sigma_b*np.sqrt(dt),axis=1)))
    measurements=bias+rng.normal(size=(count,n))*sigma_w/np.sqrt(dt)
    error=bias[:,-1]-measurements.mean(axis=1)
    assert np.var(error,ddof=1)==pytest.approx(detector.stationary_gyro_sigma_rps**2,rel=.05)


def test_opt_in_changes_gyro_bias_information_and_preserves_initialized_mean():
    seeds=[]
    for enabled in [False,True]:
        detector,buffer,_=setup(enabled);observe(detector,buffer,samples(np.linspace(1.,2.,201)))
        sigmas=np.repeat([.01,.0001,.1,.1,.01],3)
        seed=detector.predict_trial(buffer,2.,np.eye(4),sigmas);assert seed is not None
        seeds.append(seed)
        actual=(seed.prior.A.T@seed.prior.A)[12:15,12:15]
        sigma=detector.stationary_gyro_sigma_rps if enabled else .01
        np.testing.assert_allclose(actual,np.eye(3)/sigma**2,rtol=1e-12,atol=1e-10)
    for name in ['pose','velocity','accel_bias','gyro_bias']:
        np.testing.assert_array_equal(getattr(seeds[0].state,name),getattr(seeds[1].state,name))


def test_detector_freezes_uncertainty_at_the_selected_past_stationary_window():
    detector,buffer,_=setup(True,mode='stationary_detection')
    values=samples(np.linspace(1.,2.3,261))
    values=[s if s.timestamp<2.1 else ImuSample(s.timestamp,s.acceleration,[.5,0.,0.]) for s in values]
    observe(detector,buffer,values)
    assert detector.motion_detected and detector.stationary[0]<2.1
    before=detector.stationary_gyro_sigma_rps
    detector.observe([ImuSample(4.,[0,0,9.81],[1,0,0])])
    assert detector.stationary_gyro_sigma_rps==before


@pytest.mark.parametrize('value',[1,'true',None])
def test_sample_uncertainty_switch_requires_a_boolean(value):
    with pytest.raises(ValueError,match='gyro_bias_uncertainty_from_samples'):
        setup(value)
