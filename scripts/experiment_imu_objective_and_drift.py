#!/usr/bin/env python3
"""Separate IMU-model checks from the production frontend's error history.

No production configuration or estimator implementation is changed. The
fixed-anchor solve calls production factor/retraction functions, but uses a
standalone 9D least-squares loop. Staged solves call refine_current itself.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_five_observations as diag
import gtsam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plotly.offline import get_plotlyjs
from scipy.spatial.transform import Rotation
import yaml

from gelslam.core import gtsam_vi as gv
from gelslam.core.inertial import ImuBuffer, ImuSample
from gelslam.core.imu_navigation import gelsight_pose
from gelslam.core.marginal_prior import NavigationPrior
from gelslam.core.numpy_vi import DenseNormalResidual, NavigationState


PERTURBATION = np.r_[np.deg2rad([2., -1., 1.5]), [.005, -.004, .003],
                     [.03, -.02, .015], np.zeros(6)]


def unpack(s):
    return NavigationState(s['timestamp'], np.array(s['pose']), np.array(s['velocity']),
                           np.array(s['accel_bias']), np.array(s['gyro_bias']))


def difference(a, b, extrinsic=np.eye(4)):
    ap, bp = gelsight_pose(a.pose, extrinsic), gelsight_pose(b.pose, extrinsic)
    return dict(position_mm=float(np.linalg.norm(ap[:3, 3] - bp[:3, 3])*1000),
                rotation_deg=float(np.rad2deg(Rotation.from_matrix(bp[:3, :3].T @ ap[:3, :3]).magnitude())),
                velocity_mps=float(np.linalg.norm(a.velocity-b.velocity)),
                accel_bias_mps2=float(np.linalg.norm(a.accel_bias-b.accel_bias)),
                gyro_bias_rps=float(np.linalg.norm(a.gyro_bias-b.gyro_bias)))


def physical_residual(previous, current, pim, gravity):
    """Native target-axis R/p/v errors; anchor bias equals pim.biasHat()."""
    dt = current.timestamp-previous.timestamp
    R = previous.pose[:3, :3]
    Rt = current.pose[:3, :3].T
    predicted_position = (previous.pose[:3, 3] + previous.velocity*dt
                          + .5*np.asarray(gravity)*dt*dt + R @ pim.deltaPij())
    predicted_velocity = previous.velocity + np.asarray(gravity)*dt + R @ pim.deltaVij()
    return np.r_[Rotation.from_matrix(Rt @ R @ pim.deltaRij().matrix()).as_rotvec(),
                 Rt @ (predicted_position-current.pose[:3, 3]),
                 Rt @ (predicted_velocity-current.velocity)]


def fixed_anchor_solve(previous, initial, pim, max_iterations=30):
    """Fix all anchor variables and target bias; solve only target pose/velocity."""
    factor = gv._split_imu_factor(pim)
    current = initial.copy()
    trace = []
    for iteration in range(max_iterations):
        values = gv._imu_values(previous, current)
        cost = float(factor.error(values))
        _, Ac, rhs = gv.navigation_factor_terms(factor, previous, current)
        step = np.linalg.lstsq(Ac[:, :9], rhs, rcond=None)[0]
        if np.linalg.norm(step) < 1e-11:
            break
        accepted = False
        for scale in 2.**-np.arange(16):
            candidate = gv._navigation_plus(current, np.r_[step*scale, np.zeros(6)])
            after_cost = float(factor.error(gv._imu_values(previous, candidate)))
            if after_cost < cost:
                trace.append(dict(iteration=iteration, before=cost, after=after_cost,
                                  alpha=float(scale)))
                current, accepted = candidate, True
                break
        if not accepted:
            break
    return current, trace


def jacobian_check(factor, previous, current, epsilons=(1e-5, 1e-6, 1e-7)):
    """All 30 navigation columns, including the unconnected target bias block."""
    Ap, Ac, _ = gv.navigation_factor_terms(factor, previous, current)
    analytic = np.column_stack((Ap, Ac))
    results = []
    for epsilon in epsilons:
        numerical = np.zeros_like(analytic)
        for column in range(30):
            step = np.eye(15)[column % 15]*epsilon
            if column < 15:
                plus = gv._imu_values(gv._navigation_plus(previous, step), current)
                minus = gv._imu_values(gv._navigation_plus(previous, -step), current)
            else:
                plus = gv._imu_values(previous, gv._navigation_plus(current, step))
                minus = gv._imu_values(previous, gv._navigation_plus(current, -step))
            numerical[:, column] = (factor.whitenedError(plus)-factor.whitenedError(minus))/(2*epsilon)
        delta = analytic-numerical
        results.append(dict(epsilon=epsilon, max_abs=float(np.max(np.abs(delta))),
                            relative_frobenius=float(np.linalg.norm(delta)/max(np.linalg.norm(numerical), 1e-30)),
                            block_max_abs={name: float(np.max(np.abs(delta[:, start:start+width])))
                                           for name, start, width in (
                                               ('previous_pose', 0, 6), ('previous_velocity', 6, 3),
                                               ('previous_bias', 9, 6), ('target_pose', 15, 6),
                                               ('target_velocity', 21, 3), ('target_bias', 24, 6))}))
    return results


def synthetic_case(kind, dt=.001, duration=.2):
    """Analytic world trajectory and ideal body measurements; no PIM-generated GT."""
    gravity = np.array([0., 0., -9.80665])
    R0 = Rotation.from_rotvec([.4, -.3, .2]).as_matrix()
    p0, v0 = np.array([.4, -.2, .3]), np.array([.12, -.07, .03])
    acceleration, omega = np.zeros(3), np.zeros(3)
    if kind == 'stationary':
        v0 *= 0.
    elif kind == 'constant_acceleration':
        acceleration = np.array([.4, -.2, .1])
    elif kind == 'gravity_axis_rotation':
        omega = R0.T @ np.array([0., 0., .5])
    elif kind == 'general_rotation_acceleration':
        acceleration, omega = np.array([.4, -.2, .1]), np.array([.3, -.2, .4])
    elif kind != 'constant_velocity':
        raise ValueError(kind)
    ba, bg = np.array([.003, -.002, .004]), np.array([.0002, -.0001, .00015])
    def at(t):
        pose = np.eye(4)
        pose[:3, :3] = R0 @ Rotation.from_rotvec(omega*t).as_matrix()
        pose[:3, 3] = p0+v0*t+.5*acceleration*t*t
        return NavigationState(t, pose, v0+acceleration*t, ba.copy(), bg.copy())
    samples = [ImuSample(float(t), at(t).pose[:3, :3].T @ (acceleration-gravity)+ba, omega+bg)
               for t in np.linspace(0., duration, round(duration/dt)+1)]
    return at(0.), at(duration), samples, gravity


class NoNormal:
    """No tactile cost; supplies the production solver's minimum row interface."""
    def __init__(self):
        self.last_active_rows = np.ones(10, dtype=bool)
    def __len__(self):
        return 10
    def set_fixed_reference_z(self, z):
        pass
    def evaluate(self, reference, current):
        return np.zeros(30), np.zeros((30, 6)), np.zeros((30, 6))
    def residual_on_fixed_rows(self, reference, current, rows):
        return np.zeros(30)


def synthetic_experiment(config):
    rows = []
    for kind in ('stationary', 'constant_velocity', 'constant_acceleration',
                 'gravity_axis_rotation', 'general_rotation_acceleration'):
        for dt in ((.001, .0001) if kind == 'general_rotation_acceleration' else (.001,)):
            previous, target, samples, gravity = synthetic_case(kind, dt)
            pim = gv.preintegrate_imu(samples, previous, config, np.eye(4), gravity)
            factor = gv._split_imu_factor(pim)
            initial = gv._navigation_plus(target, PERTURBATION)
            fitted, trace = fixed_anchor_solve(previous, initial, pim)
            residual = factor.unwhitenedError(gv._imu_values(previous, target))
            rows.append(dict(kind=kind, dt=dt, raw_gt_residual=residual.tolist(),
                             direct_equation_max_abs=float(np.max(np.abs(
                                 factor.unwhitenedError(gv._imu_values(previous, initial))
                                 - physical_residual(previous, initial, pim, gravity)))),
                             raw_gt_norms=dict(rotation_rad=float(np.linalg.norm(residual[:3])),
                                               position_m=float(np.linalg.norm(residual[3:6])),
                                               velocity_mps=float(np.linalg.norm(residual[6:9]))),
                             gt_cost=float(factor.error(gv._imu_values(previous, target))),
                             final_error=difference(fitted, target), trace=trace,
                             jacobian_at_gt=jacobian_check(factor, previous, target),
                             jacobian_perturbed=jacobian_check(factor, previous, initial)))
    return rows


def staged_solves(record, call, truth, buffer, config, extrinsic, initial_scale=0.):
    """Independent per-interval solves; no stage result is fed to the next frame."""
    previous = truth.at(call['previous']['timestamp'])
    target = truth.at(call['initial']['timestamp'])
    initial = gv._navigation_plus(target, PERTURBATION*initial_scale)
    samples = buffer.samples_covering(previous.timestamp, target.timestamp)
    gt_reference = gelsight_pose(truth.at(call['reference_time']).pose, extrinsic)
    stages = []
    modes = ('imu_gt_prior', 'imu_recorded_prior', 'normal_gt_prior_gt_reference',
             'normal_recorded_prior_gt_reference', 'normal_recorded_prior_recorded_reference',
             'normal_recorded_prior_recorded_reference_free_z')
    for mode in modes:
        cfg = dict(config, frontend_gravity_mps2=call['gravity'])
        # Keep the supplied frozen prior when previous GT differs from its mean.
        cfg['frontend_gt_previous_state_test'] = dict(enabled=True, reset_marginal_prior=False,
                                                     previous_state_time='accepted_state')
        if mode.endswith('free_z'):
            cfg['gtsam_vi_normal_reference_z_from_imu'] = False
        solver = gv.GtsamVisualInertialSolver(cfg)
        solver.marginal_prior = (copy.deepcopy(record['prior']) if 'recorded_prior' in mode else
                                NavigationPrior.initial(previous, solver.prior_sigmas))
        assert solver.marginal_prior is not None
        reference = np.array(call['reference_pose']) if 'recorded_reference' in mode else gt_reference
        visual = (DenseNormalResidual(record['visual'].ref, record['visual'].tar,
                                      record['visual'].ppmm, reference, gelsight_pose(initial.pose, extrinsic),
                                      max_samples=config.get('dense_normalflow_samples', 3000))
                  if mode.startswith('normal') else NoNormal())
        result = solver.refine_current(visual, reference, previous, initial, samples, extrinsic)
        quality = None
        if result is not None and mode.startswith('normal'):
            quality = [float(v) for v in diag.tm.normalflow_quality(
                visual.ref, visual.tar, np.linalg.inv(gelsight_pose(result.pose, extrinsic)) @ reference,
                record['visual'].ppmm, 3000)]
        stages.append(dict(mode=mode, target_start='gt' if initial_scale == 0 else 'near_gt_perturbed',
                           initial_scale=initial_scale, target_time=target.timestamp, previous_time=previous.timestamp,
                           reference_time=call['reference_time'], initial_error=truth.error(initial, extrinsic),
                           result_error=None if result is None else truth.error(result, extrinsic),
                           result=diag.state(result), iterations=len(solver.last_optimization_trace),
                           stop_reason=(solver.last_optimization_trace[-1]['stop_reason']
                                        if solver.last_optimization_trace else solver.last_refinement_failure),
                           quality_scr_ccs=quality, prior_source=solver.last_prior_source,
                           trace=copy.deepcopy(solver.last_optimization_trace)))
    return stages


def classify(event, calls):
    if event['contact_label'] == 'F0':
        return 'anchor'
    if not event['published']:
        return 'target_not_published'
    call = calls[event['calls'][-1]] if event['calls'] else None
    if call is None:
        return 'other'
    if call['result'] is None:
        return 'normalflow_only' if call['failure'] == 'initial_shared_points_lt_10' else 'other'
    quality = call.get('quality_scr_ccs')
    if quality is not None and quality[0] >= .3 and quality[1] >= .7:
        return 'vi_accepted'
    after = unpack(event['after'])
    initial = unpack(call['initial'])
    if all(np.allclose(getattr(after, k), getattr(initial, k), atol=1e-12, rtol=0.)
           for k in ('pose', 'velocity', 'accel_bias', 'gyro_bias')):
        return 'imu_recovery'
    return 'other'


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def history_experiment(baseline):
    events, iterations = [], []
    for event in baseline['events']:
        if event['contact_label'] is None:
            continue
        call = baseline['calls'][event['calls'][-1]] if event['calls'] else None
        row = dict(frame=event['contact_label'], target_time=event['timestamp'],
                   decision=classify(event, baseline['calls']), published=event['published'],
                   retained_time=event['after']['timestamp'],
                   previous_time=None if call is None else call['previous']['timestamp'],
                   reference_time=None if call is None else call['reference_time'],
                   reference_after_time=event['reference_time'],
                   prior_source=None if call is None else call['prior_source'])
        for prefix, values in [('previous', None if call is None else call['previous_error']),
                               ('predicted', None if call is None else call['initial_error']),
                               ('optimized', None if call is None else call['result_error']),
                               ('retained', event['after_error'])]:
            for key in ('position_mm', 'rotation_deg', 'velocity_mps', 'accel_bias_mps2', 'gyro_bias_rps'):
                row[prefix+'_'+key] = None if values is None else values[key]
        events.append(row)
        if call is None:
            continue
        for step in call['iterations']:
            tr = step['trace']
            detail = dict(frame=event['contact_label'], target_time=event['timestamp'],
                          iteration=step['iteration'], alpha=tr['accepted_scale'],
                          frame_decision=row['decision'], stop_reason=tr.get('stop_reason'))
            for side in ('previous', 'current'):
                for metric in ('position_mm', 'rotation_deg', 'velocity_mps', 'accel_bias_mps2', 'gyro_bias_rps'):
                    for boundary in ('before', 'after'):
                        detail[side+'_'+boundary+'_'+metric] = step[boundary][side+'_error'][metric]
            for term in ('normal', 'imu', 'bias', 'prior'):
                detail['before_'+term+'_cost'] = tr['base_'+term+'_cost']
                detail['after_'+term+'_cost'] = tr.get('accepted_'+term+'_cost', tr['base_'+term+'_cost'])
            iterations.append(detail)
    return events, iterations


def render(output, result):
    events = result['history']['events']
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), layout='constrained')
    pairs = [('position_mm', 'GelSight position error [mm]'), ('rotation_deg', 'Rotation error [deg]'),
             ('velocity_mps', 'IMU-origin velocity error [m/s]'), ('accel_bias_mps2', 'Accel bias error [m/s²]'),
             ('gyro_bias_rps', 'Gyro bias error [rad/s]')]
    for ax, (key, label) in zip(axes.flat, pairs):
        x = [e['target_time'] for e in events]
        ax.plot(x, [e['predicted_'+key] for e in events], label='production IMU prediction', color='#e18021')
        ax.plot(x, [e['optimized_'+key] for e in events], label='production VI candidate', color='#4264ad')
        ax.plot(x, [e['retained_'+key] if e['published'] else np.nan for e in events],
                'o', ms=3, label='retained target', color='black')
        ax.set(xlabel='Target time [s]', ylabel=label); ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    rows = result['fixed_anchor']
    axes[2, 1].plot([r['target_time'] for r in rows], [r['final_error']['position_mm'] for r in rows],
                    label='GT anchor / IMU only', color='#2a8053')
    axes[2, 1].plot([e['target_time'] for e in events], [e['predicted_position_mm'] for e in events],
                    label='production prediction', color='#e18021')
    axes[2, 1].set(xlabel='Target time [s]', ylabel='Position error [mm]',
                    title='Separate tests: GT-fixed anchor vs production input state')
    axes[2, 1].legend(fontsize=8); axes[2, 1].grid(alpha=.25)
    fig.suptitle('IMU model checks and production error history | contact observations only')
    fig.savefig(output/'overview.png', dpi=150); fig.savefig(output/'overview.pdf'); plt.close(fig)
    data = json.dumps(result, allow_nan=False)
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>IMU 목적함수와 오차 전달 실험</title>
<style>body{font:15px system-ui;background:#f4f6f9;color:#18293d;margin:24px auto;max-width:1500px}section{background:white;padding:18px;margin:15px 0;border:1px solid #ddd}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}.chart{height:380px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}select{padding:8px}</style>
<script>__PLOTLY__</script><h1>IMU 목적함수 검증 · 초반 오차와 전달 경로</h1>
<section><p>실험 1: 이전 GT pose·velocity·bias와 gravity를 고정한 IMU 단독 복원. 이후 실제 frontend solver에 GT prior / 기록된 prior / NormalFlow를 단계별로 추가합니다.</p>
<p>실험 2: 기존 production frontend 전체 실행의 채택된 갱신과 IMU recovery를 추적합니다. 아래 단계별 비교의 결과는 다음 프레임으로 전달하지 않습니다.</p>
<p>접촉 전 입력은 초기화에 사용하고 표에서 제외합니다. 위치는 GelSight 원점, velocity는 IMU 원점의 world 값입니다. GT pose 보간·velocity 중앙차분, 평활화 없음. GT bias는 검증된 simulation sidecar에서 가져왔습니다.</p>
<p id="summary"></p><a href="overview.png">전체 정적 그래프</a> · <a href="report.txt">검증 보고서</a> · <a href="history_events.csv">프레임 CSV</a> · <a href="history_iterations.csv">iteration CSV</a> · <a href="result.json">전체 수치</a></section>
<section><h2>실험 1 · 고정 GT anchor 복원</h2><div id="fixed" class="chart"></div><h3>독립적인 무잡음 운동과 Jacobian 검사</h3><div id="synthetic"></div></section>
<section><h2>실험 1 · Prior / NormalFlow 단계별 비교</h2><p>이전 pose·velocity·bias는 모든 단계에서 GT 시작값입니다. IMU 단독만 GT로 고정합니다. 아래 production joint solver 단계에서는 이전 State도 최적화됩니다. target은 정확한 GT 또는 작은 보정(0.04/−0.02/0.03 deg, 0.1/−0.08/0.06 mm, velocity 0.0006/−0.0004/0.0003 m/s)으로 시작합니다. 위 IMU 단독 복원은 더 큰 ±2/−1/1.5 deg, ±5/−4/3 mm 보정을 사용합니다.</p>
<p>각 비교는 실제 실행의 previous→target 구간과 reference를 사용합니다. 기록된 prior는 실제 frozen 평균·gradient를 유지합니다. GT prior는 이전 GT State 중심의 초기 prior입니다.</p>
<select id="frame"></select><div id="stages"></div></section>
<section><h2>실험 2 · 오차가 채택되고 전달되는 과정</h2><p>최적화 candidate와 품질검사 후 보관 State는 다를 수 있습니다. 보관 State의 오차는 그 State 자신의 시각에서 평가했습니다.</p><div class="grid"><div id="position" class="chart"></div><div id="rotation" class="chart"></div><div id="velocity" class="chart"></div><div id="bias" class="chart"></div></div><h3>초반 Prior 분리 시험</h3><p>actual_A_anchor_zero_rhs는 실제 초기값·영상·고정 reference·Prior 행렬 A·anchor를 유지하고 rhs만 0으로 바꾼 진단입니다. 실제 prior 정보를 버리므로 이것을 그대로 수정안으로 적용한 것은 아닙니다. 자세한 수치는 <a href="early_prior_ablation.json">원본 JSON</a>에서 확인할 수 있습니다.</p><div id="early"></div><div id="history"></div></section>
<script>const D=__DATA__;const n=x=>x===null||x===undefined?'—':Number(x).toPrecision(5);const E=id=>document.getElementById(id);
const plot=(id,data,title,y)=>Plotly.react(id,data,{title,margin:{l:70,r:20,t:50,b:70},xaxis:{title:'target time [s]'},yaxis:{title:y},legend:{orientation:'h',y:-.25}},{responsive:true});
E('summary').textContent=`접촉 관찰 ${D.history.events.length}개 · 연속 ${D.fixed_anchor.length/2}구간 × 초기값 2개 · 단계별 비교 ${D.stages.length}개`;
Plotly.react('fixed',[{x:D.fixed_anchor.map(r=>r.target_time),y:D.fixed_anchor.map(r=>r.final_error.position_mm),name:'최종 위치 [mm]'}, {x:D.fixed_anchor.map(r=>r.target_time),y:D.fixed_anchor.map(r=>r.final_error.rotation_deg),name:'최종 회전 [deg]',yaxis:'y2'}],{title:'GT anchor IMU 단독 복원',margin:{l:70,r:70,t:50,b:70},xaxis:{title:'target time [s]'},yaxis:{title:'위치 오차 [mm]'},yaxis2:{title:'회전 오차 [deg]',overlaying:'y',side:'right'},legend:{orientation:'h',y:-.25}},{responsive:true});
E('synthetic').innerHTML='<table><tr><th>운동</th><th>IMU dt [s]</th><th>GT raw R [rad]</th><th>GT raw p [m]</th><th>GT raw v [m/s]</th><th>Jacobian 최대 상대 오차 (GT/perturbed)</th></tr>'+D.synthetic.map(r=>'<tr><td>'+r.kind+'</td><td>'+r.dt+'</td><td>'+n(r.raw_gt_norms.rotation_rad)+'</td><td>'+n(r.raw_gt_norms.position_m)+'</td><td>'+n(r.raw_gt_norms.velocity_mps)+'</td><td>'+n(Math.max(...r.jacobian_at_gt.concat(r.jacobian_perturbed).map(j=>j.relative_frobenius)))+'</td></tr>').join('')+'</table>';
const times=[...new Set(D.stages.map(r=>r.target_time))];E('frame').innerHTML=times.map(t=>`<option value="${t}">${t.toFixed(3)} s</option>`).join('');
function show(){const t=+E('frame').value;E('stages').innerHTML='<table><tr><th>구성 / target 시작값</th><th>previous / reference [s]</th><th>위치 [mm]</th><th>회전 [deg]</th><th>velocity [m/s]</th><th>accel bias [m/s²]</th><th>gyro bias [rad/s]</th><th>반복 / 종료</th><th>SCR / CCS (참고)</th></tr>'+D.stages.filter(r=>r.target_time===t).map(r=>'<tr><td>'+r.mode+' / '+r.target_start+'</td><td>'+r.previous_time+' / '+r.reference_time+'</td>'+['position_mm','rotation_deg','velocity_mps','accel_bias_mps2','gyro_bias_rps'].map(k=>'<td>'+n(r.result_error?.[k])+'</td>').join('')+'<td>'+r.iterations+' / '+r.stop_reason+'</td><td>'+(r.quality_scr_ccs?.map(x=>n(x)).join(' / ')??'—')+'</td></tr>').join('')+'</table>';}
E('frame').value=String(times[0]);E('frame').onchange=show;show();
for(const [id,key,label] of [['position','position_mm','위치 [mm]'],['rotation','rotation_deg','회전 [deg]'],['velocity','velocity_mps','velocity [m/s]'],['bias','accel_bias_mps2','accel bias [m/s²]']]){plot(id,['predicted','optimized','retained'].map(p=>({x:D.history.events.map(e=>e.target_time),y:D.history.events.map(e=>p==='retained'&&!e.published?null:e[p+'_'+key]),name:p,mode:'lines+markers'})),label+' GT 오차',label);}
E('history').innerHTML='<table><tr><th>관찰</th><th>target [s]</th><th>결정</th><th>보관 시각 [s]</th><th>prior</th><th>reference 추정 시도 → callback 후 [s]</th><th>예측 → 최적화 → 보관 회전 [deg]</th><th>보관 accel bias 오차</th></tr>'+D.history.events.map(e=>'<tr><td>'+e.frame+'</td><td>'+e.target_time+'</td><td>'+e.decision+'</td><td>'+e.retained_time+'</td><td>'+e.prior_source+'</td><td>'+e.reference_time+' → '+e.reference_after_time+'</td><td>'+[e.predicted_rotation_deg,e.optimized_rotation_deg,e.retained_rotation_deg].map(n).join(' → ')+'</td><td>'+n(e.retained_accel_bias_mps2)+'</td></tr>').join('')+'</table>';
E('early').innerHTML='<table><tr><th>target [s]</th><th>Prior 설정</th><th>위치 [mm]</th><th>회전 [deg]</th><th>velocity [m/s]</th><th>accel bias [m/s²]</th></tr>'+(D.early_prior_ablation??[]).map(r=>'<tr><td>'+r.target_time+'</td><td>'+r.mode+'</td>'+['position_mm','rotation_deg','velocity_mps','accel_bias_mps2'].map(k=>'<td>'+n(r.result_error[k])+'</td>').join('')+'</tr>').join('')+'</table>';
</script></html>'''
    page = page.replace('__PLOTLY__', get_plotlyjs()).replace('__DATA__', data)
    (output/'dashboard.html').write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--bag', type=Path, required=True)
    parser.add_argument('--bias-archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'result.json').exists():
        raise RuntimeError('Use a fresh output folder')
    baseline = json.loads((args.baseline/'result.json').read_text())
    config = baseline['config']
    (args.output/'config_supplied.yaml').write_text(yaml.safe_dump(config))
    for source, digest in baseline['code_hashes'].items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == digest, 'Production source changed: '+source
    inputs = diag.read_bag(args.bag)
    truth = diag.Truth(inputs['/ground_truth/imu_pose'], args.bias_archive)
    truth.transform = np.array(baseline['navigation_T_world'])
    buffer = ImuBuffer(max_samples=10000)
    for _, _, m in inputs['/imu/data_raw']:
        buffer.add(ImuSample(diag.stamp(m), [m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z],
                             [m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z]))
    extrinsic = np.array(baseline['extrinsic'])
    cfg = dict(config['imu_config'], frontend_gravity_mps2=baseline['calls'][0]['gravity'])
    result = dict(scope='production frontend factors; standalone fixed-anchor solve; offline replay; no backend',
                  baseline=str(args.baseline), bag=str(args.bag), config=config,
                  perturbation=PERTURBATION.tolist(), synthetic=synthetic_experiment(cfg), fixed_anchor=[], stages=[])
    contact = [e for e in baseline['events'] if e['contact_label'] is not None]
    for i, (a, b) in enumerate(zip(contact, contact[1:])):
        previous, target = truth.at(a['timestamp']), truth.at(b['timestamp'])
        samples = buffer.samples_covering(previous.timestamp, target.timestamp)
        pim = gv.preintegrate_imu(samples, previous, cfg, extrinsic, np.array(cfg['frontend_gravity_mps2']))
        factor = gv._split_imu_factor(pim)
        for sign in (-1., 1.):
            initial = gv._navigation_plus(target, PERTURBATION*sign)
            fitted, trace = fixed_anchor_solve(previous, initial, pim)
            predicted = pim.predict(gtsam.NavState(gtsam.Pose3(previous.pose), previous.velocity), gv._bias(previous))
            endpoint = target.copy(); endpoint.pose = predicted.pose().matrix(); endpoint.velocity = predicted.velocity()
            row = dict(previous_time=previous.timestamp, target_time=target.timestamp, sign=sign,
                       initial_error=difference(initial, target, extrinsic), final_error=difference(fitted, target, extrinsic),
                       final_to_pim_endpoint=difference(fitted, endpoint, extrinsic), trace=trace,
                       gt_cost=float(factor.error(gv._imu_values(previous, target))),
                       final_cost=float(factor.error(gv._imu_values(previous, fitted))))
            if sign == 1.:
                row['jacobian_at_gt'] = jacobian_check(factor, previous, target)
                row['jacobian_perturbed'] = jacobian_check(factor, previous, initial)
            result['fixed_anchor'].append(row)
        print('Fixed GT interval', i+1, '/', len(contact)-1, flush=True)
    replay = diag.replay(inputs, config, args.output, truth, True, baseline['selected_interval'][0], 5, True)
    # Compare original scalar traces without diagnostic iteration expansion.
    assert replay['events'] == [{k:v for k,v in e.items() if k not in ('image_file','target_gt')}
                                for e in baseline['events']], 'Replay changed event state or decision'
    for c, original in zip(replay['calls'], baseline['calls']):
        assert c['trace'] == original['trace'], 'Replay changed solver trace'
    assert len(replay['calls']) == len(baseline['calls'])
    result['replay_matches_baseline'] = True
    for i, (call, records) in enumerate(zip(replay['calls'], replay['records'])):
        if not records:
            continue
        for initial_scale in (0., .02):
            result['stages'].extend(staged_solves(records[0], call, truth, buffer, cfg, extrinsic, initial_scale))
        print('Staged call', i+1, '/', len(replay['calls']), flush=True)
    events, iterations = history_experiment(baseline)
    result['history'] = dict(events=events, iterations=iterations)
    result['code_hashes'] = baseline['code_hashes']
    for source, digest in result['code_hashes'].items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == digest
    (args.output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    write_csv(args.output/'history_events.csv', events)
    write_csv(args.output/'history_iterations.csv', iterations)
    render(args.output, result)
    print('Complete:', args.output, flush=True)


if __name__ == '__main__':
    main()
