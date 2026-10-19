#!/usr/bin/env python3
"""Static scientific figures and a local interactive view of recorded VI steps."""
import argparse
import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plotly.offline import get_plotlyjs


def gel_position(state, extrinsic):
    return (np.asarray(state['pose'])@np.linalg.inv(extrinsic))[:3,3]*1000


def contact_events(events):
    """Hide startup inputs; retain every input after the first contact anchor."""
    if not any('contact_label' in e for e in events):
        return events
    first=next((i for i,e in enumerate(events) if e['contact_label'] is not None),len(events))
    return events[first:]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--details-limit',type=int,default=6)
    args=parser.parse_args();directory=args.directory
    result=json.loads((directory/'result.json').read_text())
    all_events=result['events']
    result['events']=contact_events(all_events)
    result['hidden_startup_images']=len(all_events)-len(result['events'])+result.get('prefix_images',0)
    extrinsic=np.asarray(result['extrinsic'])
    frames=result['events'];calls=result['calls']
    call_frames={i:e for e in frames for i in e['calls']}
    for i,call in enumerate(calls):
        call['frame_label']=call_frames.get(i,{}).get('contact_label') or call_frames.get(i,{}).get('frame','call_%d'%i)
        call.setdefault('reference_image','F0.png')
    references={call['reference_image'] for call in calls}
    result['reference_images']={name:'data:image/png;base64,'+base64.b64encode((directory/name).read_bytes()).decode()
                                for name in references}
    assert frames,'No contact observations to display'
    fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
    metrics=[('position_mm','GelSight position error [mm]'),('rotation_deg','Rotation error [deg]'),
             ('velocity_mps','IMU-origin velocity error [m/s]'),('accel_bias_mps2','Accel bias error [m/s²]'),
             ('gyro_bias_rps','Gyro bias error [rad/s]')]
    for ax,(key,label) in zip(axes.flat,metrics):
        ax.plot([e['timestamp'] for e in frames],
                [e['after_error'][key] if e['after_error'] is not None and
                 abs(e['after']['timestamp']-e['timestamp'])<1e-8 else np.nan for e in frames],
                'o-',ms=3,label='retained target State')
        ax.plot([c['initial']['timestamp'] for c in calls],
                [c['initial_error'][key] for c in calls],'s--',ms=3,label='IMU prediction')
        ax.set_xlabel('Target header time [s]')
        ax.set_ylabel(label);ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8)
    ax=axes[2,1]
    representative=sorted(set([0]+sorted(range(len(calls)),key=lambda i:(
        calls[i]['result_error'] or calls[i]['initial_error'])['rotation_deg'],reverse=True)[:4]))
    for i in representative:
        c=calls[i]
        ax.plot(range(len(c['iterations'])+1),
                [c['initial_error']['rotation_deg']]+[x['after']['current_error']['rotation_deg'] for x in c['iterations']],
                'o-',label='%s %.2f s'%(c['frame_label'],c['initial']['timestamp']))
    ax.set_title('Selected frontend update paths')
    ax.set_xlabel('Iteration boundary (0 = initial prediction)');ax.set_ylabel('Rotation error [deg]');ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('Contact observations: %d | production frontend | no trajectory alignment\n'
                 'All bag images processed; startup hidden; rejected inputs retained' % len(frames))
    fig.savefig(directory/'overview.png',dpi=160);fig.savefig(directory/'overview.pdf');plt.close(fig)
    detail_indices=sorted(set(list(range(min(2,len(calls))))+sorted(
        range(len(calls)),key=lambda i:(calls[i]['result_error'] or calls[i]['initial_error'])['rotation_deg'],
        reverse=True)[:max(0,args.details_limit-2)]))
    for index in detail_indices:
        call=calls[index]
        iterations=call['iterations'];xs=np.arange(len(iterations)+1)
        if not iterations:
            continue
        fig,axes=plt.subplots(3,2,figsize=(12,11),layout='constrained')
        ax=axes[0,0]
        ax.plot(xs,[call['initial_error']['position_mm']]+[s['after']['current_error']['position_mm'] for s in iterations],
                'o-',color='#1c65a6',label='position [mm]')
        right=ax.twinx();right.plot(xs,[call['initial_error']['rotation_deg']]+[s['after']['current_error']['rotation_deg'] for s in iterations],
                                   's-',color='#b34228',label='rotation [deg]')
        ax.set_ylabel('Position [mm]',color='#1c65a6');right.set_ylabel('Rotation [deg]',color='#b34228');ax.set_xlabel('Iteration boundary');ax.grid(alpha=.25)
        ax=axes[0,1]
        for term in ('normal','imu','prior','bias'):
            values=[iterations[0]['trace']['base_'+term+'_cost']]+[s['trace']['accepted_'+term+'_cost']
                     if s['trace']['accepted_scale'] is not None else s['trace']['base_'+term+'_cost'] for s in iterations]
            ax.plot(xs,values,'o-',label=term)
        ax.set_yscale('symlog',linthresh=1e-5);ax.set_ylabel('Weighted cost');ax.set_xlabel('Iteration boundary');ax.legend(fontsize=8);ax.grid(alpha=.25)
        initial=gel_position(call['initial'],extrinsic)
        gt=gel_position(iterations[0]['gt_current'],extrinsic)
        positions=np.array([initial]+[gel_position(s['after']['current'],extrinsic) for s in iterations])-gt
        ax=axes[1,0];ax.plot(positions[:,0],positions[:,1],'o-',label='optimizer path')
        ax.scatter([0],[0],marker='*',s=130,c='black',label='target GT')
        for k,p in enumerate(positions):ax.annotate(str(k),p[:2],xytext=(4,4),textcoords='offset points')
        ax.set_xlabel('X error [mm], fixed navigation axes');ax.set_ylabel('Y error [mm]');ax.legend(fontsize=8);ax.grid(alpha=.25)
        ax.locator_params(axis='x',nbins=5)
        ax=axes[1,1]
        ax.plot(xs,[iterations[0]['before']['previous_error']['accel_bias_mps2']]+
                [s['after']['previous_error']['accel_bias_mps2'] for s in iterations],'s--',label='previous accel bias')
        ax.plot(xs,[call['initial_error']['accel_bias_mps2']]+
                [s['after']['current_error']['accel_bias_mps2'] for s in iterations],'o-',label='target accel bias')
        ax.set_ylabel('Bias GT error [m/s²]');ax.set_xlabel('Iteration boundary');ax.legend(fontsize=8);ax.grid(alpha=.25)
        record=iterations[0];arrays=np.load(directory/record['arrays'])
        points=arrays['reference_pixels'];norm=np.linalg.norm(arrays['raw_residual'].reshape(-1,3),axis=1)
        ax=axes[2,0];image=plt.imread(directory/call['reference_image']);ax.imshow(image)
        scatter=ax.scatter(points[:,0],points[:,1],c=norm,s=8,cmap='magma',vmin=0,vmax=max(np.percentile(norm,98),1e-9))
        fig.colorbar(scatter,ax=ax,label='Raw normal residual norm');ax.set_title('Shared points on actual reference %.3f s'%call['reference_time']);ax.set_xlim(0,320);ax.set_ylim(240,0)
        ax=axes[2,1]
        for term in ('normal','imu','prior','total'):
            ax.plot([p['alpha'] for p in record['gt_pose_cost_slice']],
                    [np.nan if p['costs'][term] is None else p['costs'][term] for p in record['gt_pose_cost_slice']],label=term)
        ax.axvline(0,color='gray',ls=':');ax.axvline(1,color='black',ls='--')
        ax.set_yscale('symlog',linthresh=1e-4);ax.set_xlabel('Pose-only path: 0=initial, 1=GT');ax.set_ylabel('Weighted cost');ax.legend(fontsize=8);ax.grid(alpha=.25)
        fig.suptitle('%s target %.2f s | previous %.2f s | reference %.2f s\n'
                     'GT pose slice holds previous state, target velocity/bias, shared points, and IMU Z fixed'
                     %(call['frame_label'],call['initial']['timestamp'],call['previous']['timestamp'],call['reference_time']))
        fig.savefig(directory/('call_%03d_details.png'%index),dpi=150)
        fig.savefig(directory/('call_%03d_details.pdf'%index));plt.close(fig)
    for call in calls:
        for step in call['iterations']:
            arrays=np.load(directory/step['arrays'])
            step['heatmap']=dict(x=arrays['reference_pixels'][:,0].round(3).tolist(),
                                 y=arrays['reference_pixels'][:,1].round(3).tolist(),
                                 residual=np.linalg.norm(arrays['raw_residual'].reshape(-1,3),axis=1).round(6).tolist())
    html=HTML.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',json.dumps(result,allow_nan=False))
    (directory/'dashboard.html').write_text(html)
    print(directory/'dashboard.html')


HTML='''<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>접촉 이후 전체 관찰 — VI 진단</title>
<style>body{font:15px system-ui;margin:0;background:#f4f6f9;color:#152335}main{max-width:1500px;margin:auto;padding:22px}h1{font-size:23px}h2{font-size:17px;margin:0 0 8px}.controls,.panel{background:white;padding:16px;border:1px solid #dce2e9;border-radius:8px;margin:12px 0}button,select{font:inherit;padding:8px 12px;margin-right:8px}button.selected{background:#174f88;color:white}input{width:350px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.chart{height:340px;background:white;border:1px solid #dce2e9;border-radius:8px}.wide{grid-column:1/-1}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #dde3e8;padding:7px;text-align:right}th:first-child,td:first-child{text-align:left}.small{font-size:13px;color:#465468}.warn{color:#934619}#thumbnail{width:240px;float:right;margin-left:18px}@media(max-width:900px){.grid{grid-template-columns:1fr}.wide{grid-column:1}input{width:180px}}</style>
<script>__PLOTLY__</script></head><body><main>
<h1>GelSight 접촉 이후 관찰 · 실제 production frontend 기록</h1>
<p>고정 좌표계 · 궤적 정합 없음 · 관측 기능 켠 실행과 끈 실행의 상태·비용·결정이 정확히 일치</p>
<div class="controls"><p id="coverage"></p><button id="prevFrame">이전 관찰</button><select id="frameSelect" aria-label="관찰 선택"></select><button id="nextFrame">다음 관찰</button>
<label>추정 시도 <select id="callSelect" aria-label="추정·재시도 선택"></select></label>
<p><label>단계 <input id="step" type="range" min="0" value="0"><b id="stepLabel"></b></label>
<label>평면 <select id="plane"><option value="01">XY</option><option value="02">XZ</option><option value="12">YZ</option></select></label></p>
<div id="meta"></div></div>
<div class="panel"><img id="thumbnail" alt="선택한 GelSight 입력"><h2>실제 설정과 상태</h2><div id="settings"></div><div id="state"></div><div style="clear:both"></div></div>
<div class="grid"><div id="path" class="chart"></div><div id="errors" class="chart"></div>
<div id="costs" class="chart"></div><div id="direction" class="chart"></div>
<div id="bias" class="chart"></div><div id="heat" class="chart"></div>
<div id="profile" class="chart"></div><div id="gradient" class="chart"></div></div>
<div class="panel"><h2>현재 iteration의 후보와 비용</h2><div id="candidates"></div></div>
<div class="panel"><h2>해석 범위</h2><p>GT pose 비용 단면은 현재 pose만 바꿉니다. 이전 State, 현재 velocity·bias, 대응점과 고정 Z는 그대로입니다. 이 단면의 높은 IMU 비용만으로 IMU 오류라고 판정할 수 없습니다.</p>
<p>NormalFlow는 OpenCV 선형 remap을 사용합니다. 현재 GN 방식은 각 반복의 target normal map에서 Jacobian을 다시 계산합니다. 수치 미분은 step 크기에 영향을 받으므로 여러 크기를 함께 표시합니다. 이를 곧바로 Jacobian 버그로 해석하지 않습니다.</p>
<p>F0의 pose 오차 0은 초기 anchor로 좌표계를 정의했기 때문입니다. 실시간 ROS 메시지 순서·드롭, backend는 이 실험에 포함되지 않습니다. IMU·GT는 이미지 이후 5 ms까지 준비한 뒤 callback을 실행했습니다.</p></div>
</main><script>
const D=__DATA__;let frameIndex=Math.min(1,D.events.length-1),callSlot=0;const colors={normal:'#1765a2',imu:'#d46b23',bias:'#8d4ea4',prior:'#238263'};
const E=id=>document.getElementById(id);const number=(x,n=6)=>x===null||x===undefined?'—':Number(x).toPrecision(n);
const stopText=reason=>({step_converged:'이전·현재 State의 전체 갱신량 임계값 충족',line_search_failed:'line search 후보 수락 실패',max_iterations:'최대 반복 횟수 도달',invalid_updated_residual:'전체 GN 갱신 후 유효 residual 없음 또는 비유한 비용',nonfinite_gn_state:'전체 GN 갱신 후 비유한 State'}[reason]??'종료 원인 기록 없음');
const plot=(id,data,title,x,y,extra={})=>Plotly.react(id,data,{title:{text:title,font:{size:15}},margin:{l:64,r:30,t:50,b:58},xaxis:{title:x},yaxis:{title:y},legend:{orientation:'h',y:-.25},...extra},{responsive:true,displaylogo:false});
function position(s){const T=s.pose,A=D.extrinsic;const offset=[0,1,2].map(i=>-[0,1,2].reduce((v,j)=>v+A[j][i]*A[j][3],0));return [0,1,2].map(i=>1000*(T[i][3]+[0,1,2].reduce((v,j)=>v+T[i][j]*offset[j],0)));}
function row(label,s,err){return '<tr><td>'+label+'</td><td>'+number(s.timestamp)+'</td><td>'+position(s).map(v=>number(v,5)).join(', ')+'</td><td>'+s.velocity.map(v=>number(v,5)).join(', ')+'</td><td>'+s.accel_bias.map(v=>number(v,5)).join(', ')+'</td><td>'+s.gyro_bias.map(v=>number(v,5)).join(', ')+'</td><td>'+number(err?.position_mm,5)+'</td><td>'+number(err?.rotation_deg,5)+'</td></tr>';}
function draw(){const e=D.events[frameIndex];callSlot=Math.min(callSlot,Math.max(0,e.calls.length-1));const c=e.calls.length?D.calls[e.calls[callSlot]]:null;const max=c?c.iterations.length:0;E('step').max=max;let n=Math.min(+E('step').value,max);E('step').value=n;
E('frameSelect').value=String(frameIndex);E('callSelect').innerHTML=e.calls.map((id,i)=>`<option value="${i}">${i+1} / ${e.calls.length} · call ${id}</option>`).join('');E('callSelect').value=String(callSlot);E('callSelect').disabled=e.calls.length<2;
E('stepLabel').textContent=n===0?'IMU 초기 예측':`iteration ${n-1} 종료 · ${c.iterations[n-1].trace.accepted_scale===null?'갱신 미적용':'갱신 적용'}`;
E('thumbnail').src=e.image_file??`${e.frame}.png`;const cfg=D.config.imu_config;E('settings').textContent='GT 진단: '+JSON.stringify(cfg.frontend_gt_previous_state_test)+' | IMU 예측만 사용: '+(cfg.frontend_imu_prediction_only??false)+' | 이전 State 고정: '+(cfg.frontend_fix_previous_state??false)+' | joint marginalization 적용: '+((cfg.gtsam_vi_marginalization??true)&&!(cfg.frontend_fix_previous_state??false)&&!(cfg.frontend_imu_prediction_only??false))+' | NormalFlow Z from IMU: '+cfg.gtsam_vi_normal_reference_z_from_imu;
E('meta').textContent=`${e.contact_label??e.frame} (입력 ${e.image_index??e.frame}) target ${e.timestamp.toFixed(3)} s · reference ${e.reference_time?.toFixed(3)??'없음'} s · ${e.mode} · track 반환/현재 frame 발행 ${e.valid}/${e.published}`+(c?` · previous ${c.previous.timestamp.toFixed(3)} s · IMU ${c.imu_start.toFixed(3)}→${c.imu_end.toFixed(3)} (${c.imu_samples} samples) · prior ${c.prior_source} · SCR/CCS ${c.quality_scr_ccs?.map(v=>v.toFixed(4)).join('/')??(c.prior_source==='imu_prediction_only'?'생략':'기록 없음')} · 반복 ${max}회 · ${max?stopText(c.iterations[max-1].trace.stop_reason):(c.prior_source==='imu_prediction_only'?'IMU 예측만 사용':c.failure)}`:' · VI 시도 없음');
if(!c||!c.iterations.length){E('stepLabel').textContent='최적화 iteration 없음';E('state').innerHTML='<p>'+(c?(c.prior_source==='imu_prediction_only'?'IMU 예측만 사용: 최적화와 marginal prior 생성을 생략했습니다. '+(c.quality_scr_ccs?'SCR/CCS 결과와 최종 보관 State를 확인하세요.':'SCR/CCS 검사도 생략했습니다. 최종 보관 State를 확인하세요.'):'VI 반복 전 종료: '+c.failure+' · 이후 추적 실패/IMU recovery 결과는 보관 State에서 확인할 수 있습니다.'):'초기 anchor 또는 VI 시도가 없는 관찰입니다.')+'</p>'+(e.after?'<table><tr><th>State</th><th>시각 [s]</th><th>GelSight 위치 [mm]</th><th>velocity [m/s]</th><th>accel bias</th><th>gyro bias</th><th>위치 오차 [mm]</th><th>회전 오차 [deg]</th></tr>'+row('보관 State',e.after,e.after_error)+(e.target_gt?row('현재 시각 GT',e.target_gt,null):'')+'</table>':'');for(const id of ['path','errors','costs','direction','bias','heat','profile','gradient'])Plotly.purge(id);E('candidates').textContent='VI 최적화 iteration 없음';return;}
const it=c.iterations[Math.max(0,n-1)],cur=n===0?c.initial:it.after.current,prev=n===0?c.previous:it.after.previous;
const ce=n===0?c.initial_error:it.after.current_error,pe=n===0?c.previous_error:it.after.previous_error;
E('state').innerHTML='<table><tr><th>State</th><th>시각 [s]</th><th>GelSight 위치 [mm]</th><th>IMU velocity [m/s]</th><th>accel bias [m/s²]</th><th>gyro bias [rad/s]</th><th>위치 오차 [mm, State 시각]</th><th>회전 오차 [deg, State 시각]</th></tr>'+(e.before?row('callback 진입 시 저장 State',e.before,null):'')+row('선택한 단계의 이전 State',prev,pe)+row('선택한 단계의 현재 State',cur,ce)+(e.after?row('품질검사·복구 후 보관 State',e.after,e.after_error):'')+row('이전 GT',it.gt_previous,null)+row('현재 GT',it.gt_current,null)+'</table>';
const states=[c.initial,...c.iterations.map(x=>x.after.current)],errs=[c.initial_error,...c.iterations.map(x=>x.after.current_error)],x=states.map((_,i)=>i),gt=position(it.gt_current),plane=E('plane').value.split('').map(Number),names=['X','Y','Z'];
const pts=states.map(s=>position(s).map((v,i)=>v-gt[i]));const selected=position(cur).map((v,i)=>v-gt[i]);
plot('path',[{x:pts.map(p=>p[plane[0]]),y:pts.map(p=>p[plane[1]]),text:x.map(i=>'stage '+i),mode:'lines+markers+text',textposition:'top center',name:'optimizer'},
{x:[0],y:[0],mode:'markers',marker:{symbol:'star',size:14,color:'black'},name:'GT'},
{x:[selected[plane[0]]],y:[selected[plane[1]]],mode:'markers',marker:{symbol:'circle-open',size:20,color:'#d46b23'},name:'선택한 단계'}],
'현재 target 위치 경로',names[plane[0]]+' 오차 [mm]',names[plane[1]]+' 오차 [mm]');
plot('errors',[{x,y:errs.map(s=>s.position_mm),mode:'lines+markers',name:'위치 [mm]'},
{x,y:errs.map(s=>s.rotation_deg),mode:'lines+markers',name:'회전 [deg]',yaxis:'y2'}],'GT 오차 · 0은 IMU 예측','iteration 종료 단계','위치 오차 [mm]',{yaxis2:{title:'회전 오차 [deg]',overlaying:'y',side:'right'}});
const tr=it.trace;const terms=['normal','imu','bias','prior'];plot('costs',terms.map(term=>({x:['iteration 시작','갱신 후'],y:[tr['base_'+term+'_cost'],tr.accepted_scale===null?tr['base_'+term+'_cost']:tr['accepted_'+term+'_cost']],name:term,type:'bar',marker:{color:colors[term]}})),
'가중 비용 · 현재 iteration','','Cost',{barmode:'group'});
const alpha=tr.accepted_scale??0,scales=[180/Math.PI,180/Math.PI,180/Math.PI,1000,1000,1000];
plot('direction',[{x:['Rx','Ry','Rz','Tx','Ty','Tz'],y:it.gt_pose_correction.map((v,i)=>v*scales[i]),name:'GT로 가는 보정',type:'bar'},
{x:['Rx','Ry','Rz','Tx','Ty','Tz'],y:it.raw_step.slice(0,6).map((v,i)=>v*alpha*scales[i]),name:'적용한 갱신',type:'bar'}],
`제안한 IMU body step · cos(R)=${number(it.direction_cosine.rotation,3)}, cos(T)=${number(it.direction_cosine.translation,3)}`,'R: deg / T: mm','별도 성분; 통합 norm 아님',{barmode:'group'});
plot('bias',[{x,y:errs.map(s=>s.accel_bias_mps2),mode:'lines+markers',name:'현재 accel bias'},
{x,y:[c.iterations[0].before.previous_error.accel_bias_mps2,...c.iterations.map(s=>s.after.previous_error.accel_bias_mps2)],mode:'lines+markers',name:'이전 accel bias'}],
'최적화가 이전·현재 bias를 바꾼 정도','iteration 종료 단계','GT bias 오차 [m/s²]');
plot('heat',[{x:it.heatmap.x,y:it.heatmap.y,mode:'markers',type:'scattergl',marker:{color:it.heatmap.residual,colorscale:'Magma',size:5,showscale:true},name:'normal error'}],
`고정 대응점 residual · ${tr.active_rows} points`,'reference pixel X','reference pixel Y',
{xaxis:{range:[0,320]},yaxis:{range:[240,0]},images:[{source:D.reference_images[c.reference_image],xref:'x',yref:'y',x:0,y:0,sizex:320,sizey:240,xanchor:'left',yanchor:'top',sizing:'stretch',layer:'below'}]});
plot('profile',terms.concat(['total']).map(term=>({x:it.gt_pose_cost_slice.map(p=>p.alpha),y:it.gt_pose_cost_slice.map(p=>p.costs[term]),mode:'lines',name:term})),
'GT로 향하는 pose-only 비용 단면','0=현재 iteration 시작 / 1=GT pose','Cost',{yaxis:{type:'log',title:'Weighted cost (log)'},shapes:[{type:'line',xref:'x',yref:'paper',x0:1,x1:1,y0:0,y1:1,line:{dash:'dot'}}]});
plot('gradient',terms.map(term=>({x:it.derivative_sweep.map(p=>p.epsilon),y:it.derivative_sweep.map(p=>p.derivative[term]),mode:'lines+markers',name:term+' 수치 미분'})).concat(terms.map(term=>({x:[.0001,.1],y:[it.directional_derivative[term],it.directional_derivative[term]],mode:'lines',name:term+' Jacobian',line:{dash:'dot'}}))),
'실제 joint step 방향의 비용 미분','원래 step에 곱한 ε','dCost / dα',{xaxis:{type:'log',title:'ε (step scale)'}});
E('candidates').innerHTML='<p>'+(tr.update_rule==='full_gauss_newton'?'GN 전체 step · 비용 감소 검사 없음 · ':'')+'적용 α='+number(tr.accepted_scale)+' · 이전 State 위치 오차 '+number(it.before.previous_error.position_mm)+' → '+number(it.after.previous_error.position_mm)+' mm · NormalFlow 고정 Z='+number(it.normal_fixed_reference_z_m)+' m</p><table><tr><th>α</th><th>Normal</th><th>IMU</th><th>Bias</th><th>Prior</th><th>합계</th><th>판정</th></tr>'+it.trace.attempts.map(a=>{const c=it.candidates.find(c=>c.scale===a.scale);return '<tr><td>'+a.scale+'</td>'+terms.map(t=>'<td>'+number(c?.costs[t])+'</td>').join('')+'<td>'+number(a.cost)+'</td><td>'+(a.scale===tr.accepted_scale?'적용':a.reason??'비용 미감소')+'</td></tr>';}).join('')+'</table>';
if(tr.raw_step_norms){const cfg=D.config.imu_config,limits={rotation_rad:cfg.gtsam_vi_rotation_stop_rad??.02,translation_m:cfg.gtsam_vi_translation_stop_m??.002,velocity_mps:cfg.gtsam_vi_velocity_stop_mps??.0001,accel_bias_mps2:cfg.gtsam_vi_accel_bias_stop_mps2??.00001,gyro_bias_rps:cfg.gtsam_vi_gyro_bias_stop_rps??.000001};E('candidates').innerHTML+='<p>GN 갱신량 검사: '+(tr.step_converged?'모든 항목 임계값 미만':'임계값 이상 항목 있음')+' · '+(tr.stop_reason?stopText(tr.stop_reason):'다음 iteration 진행')+'</p><table><tr><th>GN 갱신량 norm / 단위</th><th>이전 State</th><th>현재 target State</th><th>종료 임계값</th></tr>'+Object.keys(limits).map(k=>'<tr><td>'+k+'</td><td>'+number(tr.raw_step_norms.previous?.[k])+'</td><td>'+number(tr.raw_step_norms.target[k])+'</td><td>'+number(limits[k])+'</td></tr>').join('')+'</table>';}
}
E('coverage').textContent=`전체 처리 ${D.processed_images??D.events.length+D.hidden_startup_images}장 · 접촉 이후 표시 ${D.events.length}장 · 접촉 전 ${D.hidden_startup_images}장은 초기화에 사용하고 표에서 제외`;
E('frameSelect').innerHTML=D.events.map((e,i)=>`<option value="${i}">${e.contact_label??e.frame} · ${e.timestamp.toFixed(3)} s · ${e.published?'현재 frame 발행':'현재 frame 미발행'}</option>`).join('');
function chooseFrame(i){frameIndex=Math.max(0,Math.min(i,D.events.length-1));callSlot=0;E('step').value=0;draw();}
E('frameSelect').onchange=()=>chooseFrame(+E('frameSelect').value);E('prevFrame').onclick=()=>chooseFrame(frameIndex-1);E('nextFrame').onclick=()=>chooseFrame(frameIndex+1);
E('callSelect').onchange=()=>{callSlot=+E('callSelect').value;E('step').value=0;draw();};E('step').oninput=draw;E('plane').onchange=draw;draw();
</script></body></html>'''


if __name__=='__main__':main()
