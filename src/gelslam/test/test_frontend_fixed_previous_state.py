"""The explicit frontend mode fixes the full previous navigation state."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from gelslam.core.gtsam_vi import GtsamVisualInertialSolver
from gelslam.core.inertial import ImuSample
from gelslam.core.marginal_prior import NavigationPrior
from gelslam.core.numpy_vi import NavigationState

spec=importlib.util.spec_from_file_location('fixed_previous_observer',
    Path(__file__).resolve().parents[3]/'scripts/diagnose_five_observations.py')
diagnostic=importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class PositionVisual:
    def __len__(self):
        return 10
    def evaluate(self,reference,target):
        residual=np.tile(target[:3,3]-np.array([.0005,-.0002,.0001]),10)
        J=np.tile(np.c_[np.zeros((3,3)),target[:3,:3]],(10,1))
        return residual,np.zeros_like(J),J


def inputs():
    previous=NavigationState(0.,np.eye(4),np.array([.01,-.02,.005]),
                              np.array([.003,-.002,.004]),np.array([.0002,-.0001,.00015]))
    samples=[ImuSample(t,previous.accel_bias+[0.,0.,9.81],previous.gyro_bias)
              for t in np.linspace(0.,.1,11)]
    return previous,samples


def snapshot(state):
    return [state.timestamp]+[getattr(state,k).copy() for k in ('pose','velocity','accel_bias','gyro_bias')]


def assert_state_unchanged(state,expected):
    assert state.timestamp==expected[0]
    for key,value in zip(('pose','velocity','accel_bias','gyro_bias'),expected[1:]):
        assert np.array_equal(getattr(state,key),value)


@pytest.mark.parametrize('marginalization',[False,True])
def test_flag_fixes_previous_state_in_every_iteration_and_discards_old_prior(marginalization):
    previous,samples=inputs();original=snapshot(previous)
    solver=GtsamVisualInertialSolver(dict(frontend_fix_previous_state=True,
        gtsam_vi_marginalization=marginalization,gtsam_vi_max_iterations=5,
        dense_normalflow_residual_weight=1e8))
    # Even a strong old marginal gradient must not move the fixed anchor.
    solver.marginal_prior=NavigationPrior.initial(previous,np.ones(15)*.001)
    solver.marginal_prior.rhs[9]=100.
    current,pim=solver.predict_current(previous,samples,np.eye(4))
    current.accel_bias+=.02;current.gyro_bias-=.001
    target_before=snapshot(current)
    observer=diagnostic.IterationObserver()
    result=observer.run(solver.refine_current,PositionVisual(),np.eye(4),previous,current,
                        samples,np.eye(4),preintegrated=pim)
    assert result is not None and observer.iterations
    for record in observer.iterations:
        assert record['previous_step'] is None
        assert record['prior'] is None
        assert_state_unchanged(record['previous'],original)
        assert_state_unchanged(record['after_previous'],original)
    assert_state_unchanged(previous,original)
    assert_state_unchanged(current,target_before)
    assert np.linalg.norm(result.pose[:3,3]-current.pose[:3,3])>1e-6
    assert np.linalg.norm(result.velocity-current.velocity)>1e-6
    assert np.allclose(result.accel_bias,previous.accel_bias,atol=1e-10)
    assert np.allclose(result.gyro_bias,previous.gyro_bias,atol=1e-10)
    assert solver.last_prior_source=='fixed_previous'
    assert solver.marginal_prior is None and solver.pending_marginal_prior is None
    solver.commit_marginal_prior(result)
    assert solver.marginal_prior is None


def test_explicit_flag_matches_the_existing_fixed_anchor_solve():
    previous,samples=inputs()
    outputs=[];traces=[]
    for config in ({'frontend_fix_previous_state':True,'gtsam_vi_marginalization':True},
                   {'frontend_fix_previous_state':False,'gtsam_vi_marginalization':False}):
        solver=GtsamVisualInertialSolver(dict(config,gtsam_vi_max_iterations=4))
        current,pim=solver.predict_current(previous,samples,np.eye(4))
        outputs.append(solver.refine_current(PositionVisual(),np.eye(4),previous,current,samples,
                                             np.eye(4),preintegrated=pim))
        traces.append(solver.last_optimization_trace)
    assert outputs[0] is not None and outputs[1] is not None
    assert_state_unchanged(outputs[0],snapshot(outputs[1]))
    assert traces[0]==traces[1]


def test_flag_off_retains_joint_optimization_and_pending_prior():
    previous,samples=inputs()
    solver=GtsamVisualInertialSolver(dict(frontend_fix_previous_state=False,gtsam_vi_marginalization=True))
    current,pim=solver.predict_current(previous,samples,np.eye(4))
    observer=diagnostic.IterationObserver()
    result=observer.run(solver.refine_current,PositionVisual(),np.eye(4),previous,current,samples,
                        np.eye(4),preintegrated=pim)
    assert result is not None and observer.iterations
    assert any(np.linalg.norm(r['previous_step'])>1e-10 for r in observer.iterations)
    assert solver.pending_marginal_prior is not None
    solver.commit_marginal_prior(result)
    assert solver.marginal_prior.matches(result)


@pytest.mark.parametrize('invalid',['true','false',None,0,1])
def test_flag_requires_yaml_boolean(invalid):
    with pytest.raises(ValueError,match='frontend_fix_previous_state must be a boolean'):
        GtsamVisualInertialSolver({'frontend_fix_previous_state':invalid})


def test_missing_flag_preserves_the_default_joint_mode():
    assert GtsamVisualInertialSolver({}).fix_previous_state is False
