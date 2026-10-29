#!/usr/bin/env python3
"""Audit prior elimination and isolate normal sampling/linearization effects.

All estimator changes are local diagnostic substitutions. Production files and
the supplied configuration remain unchanged.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_imu_objective_and_drift as exp
import numpy as np
from scipy.linalg import qr
import yaml

diag, gv = exp.diag, exp.gv
Dense = exp.DenseNormalResidual
BLOCKS = [('rotation',0,3),('translation',3,6),('velocity',6,9),
          ('accel_bias',9,12),('gyro_bias',12,15)]


def qr_eliminate(Ap, Ac, rhs):
    """Independent Householder QR elimination; requires full-rank old columns."""
    width = Ap.shape[1]
    norms = np.linalg.norm(Ap, axis=0)
    if np.any(norms == 0.):
        raise ValueError('Independent QR control requires full-rank previous columns')
    _, R = qr(np.column_stack((Ap/norms, Ac, rhs)), mode='economic')
    if np.min(np.abs(np.diag(R[:width,:width]))) < 1e-12:
        raise ValueError('Independent QR control found deficient previous columns')
    return R[width:, width:-1].copy(), R[width:, -1].copy()


def bilinear(field, x, y):
    """Continuous float64 bilinear interpolation, zero outside the image."""
    field = np.asarray(field, dtype=float)
    h, w = field.shape[:2]
    scalar = field.ndim == 2
    if scalar:
        field = field[...,None]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x-x0, y-y0
    out = np.zeros((len(x), field.shape[2]))
    for dx, dy, weight in ((0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),
                            (0,1,(1-fx)*fy),(1,1,fx*fy)):
        xx, yy = x0+dx, y0+dy
        valid = (xx>=0)&(xx<w)&(yy>=0)&(yy<h)
        out[valid] += field[yy[valid],xx[valid]]*weight[valid,None]
    return out[:,0] if scalar else out


class SmoothIC(Dense):
    """Keep the original IC Jacobian; replace only normal-map interpolation."""
    def _shared_projection(self, reference_pose, target_pose, *args, **kwargs):
        if len(self.candidate_points) < 10:
            return None
        target = self._normal_target_pose(reference_pose, target_pose)
        relative = np.linalg.inv(target) @ reference_pose
        rotation = relative[:3,:3]
        points = self.candidate_points @ rotation.T + relative[:3,3]
        # Contact selection is unchanged. Only the normal sampling is continuous.
        x = points[:,0]*1000/self.ppmm+self.tar.N.shape[1]/2-.5
        y = points[:,1]*1000/self.ppmm+self.tar.N.shape[0]/2-.5
        normals = bilinear(self.tar.N,x,y)
        old = super()._shared_projection(reference_pose,target_pose, *args, **kwargs)
        if old is None:
            return None
        _,_,_,_,height,shared = old
        return relative,rotation,points,normals,height,shared


class SmoothFD(SmoothIC):
    """Continuous normal residual and its numerical target-pose Jacobian."""
    total_evaluations=0
    total_one_sided=0
    total_fallback=0
    def evaluate(self, reference_pose, target_pose):
        evaluated = super().evaluate(reference_pose,target_pose)
        if evaluated is None:
            return None
        residual, reference_J, fallback_J = evaluated
        rows = self.last_active_rows.copy()
        J = np.zeros_like(fallback_J)
        self.fd_one_sided = getattr(self,'fd_one_sided',0)
        self.fd_fallback = getattr(self,'fd_fallback',0)
        SmoothFD.total_evaluations+=1
        for column, epsilon in enumerate((1e-5,1e-5,1e-5,1e-7,1e-7,1e-7)):
            step = np.eye(6)[column]*epsilon
            plus = self.residual_on_fixed_rows(reference_pose,gv.split_pose_plus(target_pose,step),rows)
            minus = self.residual_on_fixed_rows(reference_pose,gv.split_pose_plus(target_pose,-step),rows)
            if plus is not None and minus is not None:
                J[:,column] = (plus-minus)/(2*epsilon)
            elif plus is not None:
                J[:,column] = (plus-residual)/epsilon; self.fd_one_sided+=1; SmoothFD.total_one_sided+=1
            elif minus is not None:
                J[:,column] = (residual-minus)/epsilon; self.fd_one_sided+=1; SmoothFD.total_one_sided+=1
            else:
                J[:,column] = fallback_J[:,column]; self.fd_fallback+=1; SmoothFD.total_fallback+=1
        self.last_active_rows = rows
        self.active_count = int(np.count_nonzero(rows))
        return residual,reference_J,J


def replace_visual(original, cls):
    result = copy.copy(original)
    result.__class__ = cls
    return result


class ReturnObserver(diag.IterationObserver):
    instances = []
    def __init__(self, function=None):
        super().__init__(function)
        self.final = None
        function=gv.GtsamVisualInertialSolver.refine_current if function is None else function
        lines,first=diag.inspect.getsourcelines(function)
        self.posterior_line=next(first+i for i,s in enumerate(lines)
                                 if 'information = marginalize_rows_system(Ap, Ac, rhs)' in s)
        self.posterior_rows=None
        self.instances.append(self)
    def observe(self, frame, event, arg):
        if event=='line' and frame.f_lineno==self.posterior_line:
            self.posterior_rows=tuple(np.array(frame.f_locals[k],copy=True) for k in ('Ap','Ac','rhs'))
        if event == 'return':
            v = frame.f_locals
            pending = copy.deepcopy(v['self'].pending_marginal_prior)
            self.final = dict(pending=pending, previous=copy.deepcopy(v.get('previous')),
                              current=copy.deepcopy(v.get('current')),
                              rows=None if pending is None else self.posterior_rows,
                              fd_one_sided=getattr(v.get('visual'),'fd_one_sided',0),
                              fd_fallback=getattr(v.get('visual'),'fd_fallback',0))
        return super().observe(frame,event,arg)


def profile_comparison(Ap,Ac,rhs,A,b):
    Aq,bq = qr_eliminate(Ap,Ac,rhs)
    H,g,c = A.T@A,A.T@b,float(b@b)
    Hq,gq,cq = Aq.T@Aq,Aq.T@bq,float(bq@bq)
    rng = np.random.default_rng(231)
    scale = np.repeat([.001,.0001,.001,.001,.0001],3)
    relative_cost=[]
    for step in [np.zeros(15)]+[rng.normal(size=15)*scale for _ in range(12)]:
        dp = np.linalg.lstsq(Ap,rhs-Ac@step,rcond=None)[0]
        full = .5*np.linalg.norm(Ap@dp+Ac@step-rhs)**2
        reduced = .5*np.linalg.norm(A@step-b)**2
        relative_cost.append(abs(full-reduced)/max(1.,full))
    return dict(H_relative=float(np.linalg.norm(H-Hq)/max(1.,np.linalg.norm(Hq))),
                g_relative=float(np.linalg.norm(g-gq)/max(1.,np.linalg.norm(gq))),
                constant_abs=abs(c-cq),profile_cost_relative_max=max(relative_cost))


def prior_derivative_check(prior):
    rng = np.random.default_rng(82)
    state = gv._navigation_plus(prior.anchor,rng.normal(size=15)*np.repeat([.001,.0001,.001,.001,.0001],3))
    A,rhs = prior.linearize(state)
    numeric = np.zeros_like(A); cost_grad=np.zeros(15)
    for col,epsilon in enumerate(np.repeat([1e-6,1e-8,1e-6,1e-6,1e-7],3)):
        step = np.eye(15)[col]*epsilon
        plus,minus=gv._navigation_plus(state,step),gv._navigation_plus(state,-step)
        numeric[:,col]=prior.A@(prior.coordinates(plus)-prior.coordinates(minus))/(2*epsilon)
        cost_grad[col]=(prior.error(plus)-prior.error(minus))/(2*epsilon)
    return dict(residual_J_relative=float(np.linalg.norm(A-numeric)/np.linalg.norm(numeric)),
                cost_gradient_relative=float(np.linalg.norm(-A.T@rhs-cost_grad)/max(1.,np.linalg.norm(cost_grad))))


def linear_checks(call,records,final):
    rows=[]
    for record in records:
        solver = record['solver']
        factors=(record['imu_factor'],record['bias_factor'])
        Ap,Ac,rhs=solver._joint_rows(factors,record['prior'],record['previous'],record['current'])
        Ap=np.vstack((Ap,np.zeros((len(record['normal_b']),15))))
        Ac=np.vstack((Ac,np.pad(record['normal_A'],((0,0),(0,9)))))
        rhs=np.r_[rhs,-record['normal_b']]
        A,b=gv.marginalize_rows(Ap,Ac,rhs)
        row=profile_comparison(Ap,Ac,rhs,A,b)
        full=np.column_stack((Ap,Ac)); scales=np.linalg.norm(full,axis=0)
        solved=np.linalg.lstsq(full/scales,rhs,rcond=None)[0]/scales
        actual=np.r_[record['previous_step'],record['pose_step'],record['velocity_step'],record['bias_step']]
        row.update(iteration=record['iteration'], step_relative=float(np.linalg.norm(solved-actual)/max(1e-15,np.linalg.norm(solved))),
                   step_block_abs={side+'_'+name:float(np.max(np.abs((solved-actual)[offset+start:offset+stop])))
                                   for side,offset in (('previous',0),('target',15)) for name,start,stop in BLOCKS})
        rows.append(row)
    pending=final['pending']
    posterior=None
    if pending is not None:
        # Compare step reduction independently from stored-prior semantics.
        # Stored RHS is deliberately zero; it does not preserve profile cost.
        reduced=gv.marginalize_rows_system(*final['rows'])
        posterior=profile_comparison(*final['rows'],*reduced.rows())
        posterior['stored_information_relative']=float(np.linalg.norm(
            pending.A.T@pending.A-reduced.information)/max(1.,np.linalg.norm(reduced.information)))
        posterior['discarded_gradient_norm']=float(np.linalg.norm(reduced.gradient))
        posterior['stored_gradient_norm']=float(np.linalg.norm(pending.A.T@pending.rhs))
        posterior['derivative']=prior_derivative_check(pending)
        posterior['rhs_norm']=float(np.linalg.norm(pending.rhs))
    return dict(target_time=call['initial']['timestamp'],iterations=rows,posterior=posterior)


def normal_checks(record,extrinsic):
    reference=record['reference_pose']; current=record['current']
    target=exp.gelsight_pose(current.pose,extrinsic)
    visual=copy.copy(record['visual'])
    smooth=replace_visual(visual,SmoothIC)
    numeric=replace_visual(visual,SmoothFD)
    rs,_,Js=smooth.evaluate(reference,target)
    rn,_,Jn=numeric.evaluate(reference,target)
    J=record['normal_jacobian']
    mapping=gv.gelsight_pose_jacobian(extrinsic)
    Jn=Jn@mapping;Js=Js@mapping
    # The observer's normal_jacobian is already in IMU increment columns.
    delta = record['solver'].normal_huber_delta_at(record['iteration'])
    As,bs,_,_=gv._normal_huber_linearization(rs,Js,record['normal_scale'],delta)
    An,bn,_,_=gv._normal_huber_linearization(rn,Jn,record['normal_scale'],delta)
    oldg=record['normal_A'].T@record['normal_b']; sg=As.T@bs;ng=An.T@bn
    return dict(residual_sampling_relative=float(np.linalg.norm(rs-record['residual'])/max(1e-15,np.linalg.norm(rs))),
                smooth_IC_vs_FD_J_relative=float(np.linalg.norm(Js-Jn)/max(1e-15,np.linalg.norm(Jn))),
                IC_gradient_cosine=diag.cosine(sg,ng),
                IC_gradient_block_cosine={name:diag.cosine(sg[start:stop],ng[start:stop])
                                          for name,start,stop in BLOCKS[:2]},
                original_gradient=oldg.tolist(),smooth_IC_gradient=sg.tolist(),smooth_FD_gradient=ng.tolist(),
                numeric_one_sided=numeric.fd_one_sided,numeric_fallback=numeric.fd_fallback)


def posterior_variant(record,final,visual_cls=None,remove_normal=False):
    solver=record['solver']; previous,current=final['previous'],final['current']
    Ap,Ac,rhs=solver._joint_rows((record['imu_factor'],record['bias_factor']),record['prior'],previous,current)
    if not remove_normal:
        visual=copy.copy(record['visual']) if visual_cls is None else replace_visual(record['visual'],visual_cls)
        residual,_,J=visual.evaluate(record['reference_pose'],exp.gelsight_pose(current.pose,EXTRINSIC))
        J=J@gv.gelsight_pose_jacobian(EXTRINSIC)
        delta = solver.normal_huber_delta_at(len(solver.last_optimization_trace) - 1)
        A,b,_,_=gv._normal_huber_linearization(residual,J,record['normal_scale'],delta)
        Ap=np.vstack((Ap,np.zeros((len(b),15))));Ac=np.vstack((Ac,np.pad(A,((0,0),(0,9)))))
        rhs=np.r_[rhs,-b]
    A,b=qr_eliminate(Ap,Ac,rhs)
    return exp.NavigationPrior(current.copy(),A,b)


def downstream(call,record,prior,cls,truth,buffer,config):
    cfg=dict(config,frontend_gravity_mps2=call['gravity'])
    cfg['frontend_gt_previous_state_test']=dict(enabled=True,reset_marginal_prior=False,previous_state_time='accepted_state')
    solver=gv.GtsamVisualInertialSolver(cfg);solver.marginal_prior=copy.deepcopy(prior)
    previous,initial=exp.unpack(call['previous']),exp.unpack(call['initial'])
    visual=cls(record['visual'].ref,record['visual'].tar,record['visual'].ppmm,
               np.array(call['reference_pose']),exp.gelsight_pose(initial.pose,EXTRINSIC),max_samples=cfg['dense_normalflow_samples'])
    result=solver.refine_current(visual,np.array(call['reference_pose']),previous,initial,
                                 buffer.samples_covering(previous.timestamp,initial.timestamp),EXTRINSIC)
    return dict(result=diag.state(result),error=truth.error(result,EXTRINSIC),trace=solver.last_optimization_trace,
                fd_one_sided=getattr(visual,'fd_one_sided',0),fd_fallback=getattr(visual,'fd_fallback',0))


def compact(replay,truth):
    events=[]
    for event in replay['events']:
        if event['contact_label'] is None: continue
        events.append(event)
    return dict(events=events,statistics=replay['tracker'].frontend_tracking_statistics(),
                final_error=events[-1]['after_error'],final_state_time=events[-1]['after']['timestamp'],
                calls=[{k:v for k,v in c.items() if k in ('previous','initial','result','prior_source','trace','quality_scr_ccs','failure')}
                       for c in replay['calls']])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--bag',type=Path,required=True)
    parser.add_argument('--bias-archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'result.json').exists():raise RuntimeError('Use a fresh output directory')
    baseline=json.loads((args.baseline/'result.json').read_text()); config=baseline['config']
    (args.output/'config_supplied.yaml').write_text(yaml.safe_dump(config))
    for path,digest in baseline['code_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    inputs=diag.read_bag(args.bag);truth=diag.Truth(inputs['/ground_truth/imu_pose'],args.bias_archive)
    truth.transform=np.array(baseline['navigation_T_world'])
    global EXTRINSIC
    EXTRINSIC=np.array(baseline['extrinsic'])
    buffer=exp.ImuBuffer()
    for _,_,m in inputs['/imu/data_raw']:
        buffer.add(exp.ImuSample(diag.stamp(m),[m.linear_acceleration.x,m.linear_acceleration.y,m.linear_acceleration.z],
                                 [m.angular_velocity.x,m.angular_velocity.y,m.angular_velocity.z]))
    original_observer=diag.IterationObserver;diag.IterationObserver=ReturnObserver
    folder=args.output/'baseline';folder.mkdir(exist_ok=True)
    try:
        replay=diag.replay(inputs,config,folder,truth,True,1.1,5,True)
    finally:
        diag.IterationObserver=original_observer
    expected=[{k:v for k,v in e.items() if k not in ('image_file','target_gt')} for e in baseline['events']]
    assert replay['events']==expected
    for c,old in zip(replay['calls'],baseline['calls']):assert c['trace']==old['trace']
    print('Baseline exactly reproduced',flush=True)
    result=dict(baseline_matches=True,config=config,linear_checks=[],normal_checks=[],downstream=[],full_replays={})
    result['full_replays']['original']=compact(replay,truth)
    for call,records,observer in zip(replay['calls'],replay['records'],ReturnObserver.instances):
        if not records:continue
        result['linear_checks'].append(linear_checks(call,records,observer.final))
        if call['initial']['timestamp']<=5.15:
            for r in (records[0],records[-1]):
                result['normal_checks'].append(dict(target_time=call['initial']['timestamp'],iteration=r['iteration'],
                                                    **normal_checks(r,EXTRINSIC)))
        print('Checked algebra',call['initial']['timestamp'],flush=True)
    for index in (1,10):
        old_record=replay['records'][index-1][-1];old_final=ReturnObserver.instances[index-1].final
        pending=old_final['pending'];actual=replay['records'][index][0]['prior']
        assert np.array_equal(pending.A,actual.A) and np.array_equal(pending.rhs,actual.rhs)
        variants={'actual':copy.deepcopy(actual),
                  'independent_qr':posterior_variant(old_record,old_final),
                  'without_old_normal':posterior_variant(old_record,old_final,remove_normal=True),
                  'smooth_IC':posterior_variant(old_record,old_final,SmoothIC),
                  'smooth_FD':posterior_variant(old_record,old_final,SmoothFD)}
        for name,prior in variants.items():
            for current_name,cls in (('original',Dense),('smooth_IC',SmoothIC),('smooth_FD',SmoothFD)):
                out=downstream(replay['calls'][index],replay['records'][index][0],prior,cls,truth,buffer,config['imu_config'])
                result['downstream'].append(dict(target_time=replay['calls'][index]['initial']['timestamp'],
                                                  prior_variant=name,current_normal=current_name,**out))
                print('Downstream',index,name,current_name,flush=True)
    # Full feedback tests. Patch only the diagnostic process, then restore.
    saved_dense=diag.tm.DenseNormalResidual
    for name,cls in (('smooth_IC',SmoothIC),('smooth_FD',SmoothFD)):
        folder=args.output/name;folder.mkdir(exist_ok=True)
        diag.tm.DenseNormalResidual=cls
        SmoothFD.total_evaluations=SmoothFD.total_one_sided=SmoothFD.total_fallback=0
        try:
            rr=diag.replay(inputs,config,folder,truth,False,1.1,5,True)
        finally:
            diag.tm.DenseNormalResidual=saved_dense
        result['full_replays'][name]=compact(rr,truth)
        if name=='smooth_FD':
            result['full_replays'][name]['numeric_J_usage']=dict(evaluations=SmoothFD.total_evaluations,
                one_sided_columns=SmoothFD.total_one_sided,fallback_columns=SmoothFD.total_fallback)
        print('Full replay complete',name,result['full_replays'][name]['final_error'],flush=True)
        del rr
    for path,digest in baseline['code_hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    result['production_code_hashes']=baseline['code_hashes']
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Complete',args.output,flush=True)


if __name__=='__main__':main()
