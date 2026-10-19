"""Independent controls used to distinguish prior algebra from normal rows."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('prior_normal_experiment',
    Path(__file__).resolve().parents[3]/'scripts/experiment_prior_normal_coupling.py')
experiment=importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


@pytest.mark.parametrize('column_scaling',[False,True])
def test_independent_householder_control_preserves_profiled_cost_and_gradient(column_scaling):
    rng=np.random.default_rng(181)
    Ap,Ac=rng.normal(size=(90,15)),rng.normal(size=(90,15))
    if column_scaling:
        Ap*=np.geomspace(1e-3,1e3,15)
    rhs=rng.normal(size=90)
    A,b=experiment.gv.marginalize_rows(Ap,Ac,rhs)
    independent_A,independent_b=experiment.qr_eliminate(Ap,Ac,rhs)
    joint=np.column_stack((Ap,Ac)); H=joint.T@joint; g=joint.T@rhs
    u,s,vt=np.linalg.svd(H[:15,:15])
    inverse=vt.T@np.diag([1./v if v>1e-6 else 0. for v in s])@u.T
    inverse=(inverse+inverse.T)/2
    expected_H=H[15:,15:]-H[15:,:15]@inverse@H[:15,15:]
    expected_H=(expected_H+expected_H.T)/2
    expected_g=g[15:]-H[15:,:15]@inverse@g[:15]
    assert np.allclose(A.T@A,expected_H,rtol=1e-8,atol=1e-8)
    assert np.allclose(A.T@b,expected_g,rtol=1e-8,atol=1e-8)
    # The scaled fixture has cond(Hpp) about 1e12. Forming its Hessian loses
    # precision compared with Householder QR; it is not a square-root method.
    bound=1e-5 if column_scaling else 1e-10
    assert np.linalg.norm(A.T@A-independent_A.T@independent_A)/np.linalg.norm(independent_A.T@independent_A)<bound
    assert np.linalg.norm(A.T@b-independent_A.T@independent_b)/np.linalg.norm(independent_A.T@independent_b)<bound
    for step in rng.normal(size=(5,15)):
        dp=np.linalg.lstsq(Ap,rhs-Ac@step,rcond=None)[0]
        cost=np.linalg.norm(Ap@dp+Ac@step-rhs)**2
        assert np.isclose(np.linalg.norm(independent_A@step-independent_b)**2,cost,rtol=1e-9)


def test_continuous_bilinear_matches_known_subpixel_plane():
    y,x=np.mgrid[:12,:16]
    field=np.stack((.1*x+.2*y,.05*x-.1*y,1.+.01*x),axis=2)
    px=np.array([2.125,7.625,11.375]);py=np.array([3.75,5.125,8.625])
    expected=np.column_stack((.1*px+.2*py,.05*px-.1*py,1.+.01*px))
    assert np.allclose(experiment.bilinear(field,px,py),expected,atol=1e-14)


def visual():
    y,x=np.mgrid[:16,:16]
    field=np.stack((.1*x+.2*y,.05*x-.1*y,1.+.01*x),axis=2)
    frame=SimpleNamespace(N=field,H=np.zeros((16,16)),C=np.ones((16,16)))
    value=experiment.SmoothFD.__new__(experiment.SmoothFD)
    value.ref=value.tar=frame;value.ppmm=1.
    px=np.linspace(3.125,10.625,12);py=np.linspace(4.125,9.625,12)
    value.candidate_points=np.column_stack(((px-7.5)/1000,(py-7.5)/1000,np.zeros(12)))
    value.candidate_normals=experiment.bilinear(field,px,py)
    value.candidate_normal_jacobian=np.zeros((3,5,12))
    value.fixed_reference_z_m=None
    value.last_active_rows=None;value.active_count=12
    return value


def test_consistent_normal_derivative_has_correct_translation_sign_and_units():
    value=visual()
    residual,_,J=value.evaluate(np.eye(4),np.eye(4))
    assert np.linalg.norm(residual)<1e-12
    assert np.allclose(J[:,3].reshape(-1,3),[-100.,-50.,-10.],atol=1e-7)
    assert np.allclose(J[:,4].reshape(-1,3),[-200.,100.,0.],atol=1e-7)
    assert np.allclose(J[:,5],0.,atol=1e-8)
    assert value.fd_fallback==0 and value.fd_one_sided==0


def test_consistent_normal_retraction_predicts_residual_for_rotation_and_translation():
    value=visual()
    pose=experiment.gv.split_pose_plus(np.eye(4),np.array([.01,-.02,.03,.00002,-.00001,0.]))
    residual,_,J=value.evaluate(np.eye(4),pose)
    step=np.array([2e-7,-1e-7,3e-7,2e-9,-1e-9,1e-9])
    after=value.residual_on_fixed_rows(np.eye(4),experiment.gv.split_pose_plus(pose,step),value.last_active_rows)
    assert after is not None
    assert np.linalg.norm(after-residual-J@step)<1e-10


def test_return_observer_captures_uncompressed_prior_rhs_before_local_overwrite():
    previous=experiment.exp.NavigationState(0.,np.eye(4),np.zeros(3),np.zeros(3),np.zeros(3))
    samples=[experiment.exp.ImuSample(t,[0.,0.,9.81],[0.,0.,0.]) for t in (0.,.05,.1)]
    solver=experiment.gv.GtsamVisualInertialSolver({})
    current,pim=solver.predict_current(previous,samples,np.eye(4))
    observer=experiment.ReturnObserver()
    result=observer.run(solver.refine_current,experiment.exp.NoNormal(),np.eye(4),
                        previous,current,samples,np.eye(4),preintegrated=pim)
    assert result is not None
    final=observer.final
    Ap,Ac,rhs=final['rows']
    assert len(rhs)==len(Ap)==len(Ac)
    assert len(rhs)>len(final['pending'].rhs)
    reduced=experiment.gv.marginalize_rows_system(Ap,Ac,rhs)
    check=experiment.profile_comparison(Ap,Ac,rhs,*reduced.rows())
    assert check['profile_cost_relative_max']<1e-10
    assert np.allclose(final['pending'].A.T@final['pending'].A,reduced.information)
    assert np.array_equal(final['pending'].rhs,np.zeros(15))
