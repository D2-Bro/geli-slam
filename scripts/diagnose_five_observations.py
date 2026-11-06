#!/usr/bin/env python3
"""Observe selected inputs or a full bag without changing solver calculations.

The Python line observer reads solver locals. It does not replace the solver.
An unobserved replay must return identical states and tracking decisions.
"""
import argparse
import copy
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace
import zipfile

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import torch
import yaml
from bisect import bisect_left
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.serialization import deserialize_message
from scipy.spatial.transform import Rotation, Slerp
from sensor_msgs.msg import Image, Imu

import gelslam.core.gtsam_vi as gv
import gelslam.core.reference_tracking as rt
import gelslam.core.tracker as tm
from gelslam.core.imu_navigation import gelsight_pose
from gelslam.core.inertial import ImuSample
from gelslam.core.numpy_vi import NavigationState
from gelslam.core.split_pose import split_pose_coordinates, split_pose_plus


class GroundTruthImuStates:
    """Bounded native poses; interpolate position, rotation and central velocity."""

    def __init__(self, max_samples=10000):
        self.samples = []
        self.max_samples = max_samples

    def add(self, timestamp, rotation, position):
        rotation, position = np.asarray(rotation), np.asarray(position)
        if (not np.isfinite(timestamp) or rotation.shape != (3, 3)
                or position.shape != (3,) or not np.all(np.isfinite(rotation))
                or not np.all(np.isfinite(position))):
            return False
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation, position
        index = bisect_left([t for t, _ in self.samples], timestamp)
        if index < len(self.samples) and self.samples[index][0] == timestamp:
            self.samples[index] = (float(timestamp), pose)
        else:
            self.samples.insert(index, (float(timestamp), pose))
        if len(self.samples) > self.max_samples:
            del self.samples[:-self.max_samples]
        return True

    def at(self, timestamp):
        """Return a world IMU pose and velocity; never extrapolate GT."""
        if not np.isfinite(timestamp) or not self.samples:
            return None
        times = [t for t, _ in self.samples]
        right = bisect_left(times, timestamp)
        if right == len(times):
            return None
        centers = (right,) if times[right] == timestamp else (right - 1, right)
        velocities = []
        for center in centers:
            if center < 1 or center + 1 >= len(times):
                return None
            before, current, after = self.samples[center - 1:center + 2]
            left_dt, right_dt = current[0] - before[0], after[0] - current[0]
            left = (current[1][:3, 3] - before[1][:3, 3]) / left_dt
            right_slope = (after[1][:3, 3] - current[1][:3, 3]) / right_dt
            velocities.append((right_dt * left + left_dt * right_slope) / (left_dt + right_dt))
        if len(centers) == 1:
            return self.samples[right][1].copy(), velocities[0]
        left = right - 1
        fraction = (timestamp - times[left]) / (times[right] - times[left])
        a, b = self.samples[left][1], self.samples[right][1]
        pose = np.eye(4)
        pose[:3, 3] = (1 - fraction) * a[:3, 3] + fraction * b[:3, 3]
        pose[:3, :3] = Slerp(
            [times[left], times[right]], Rotation.from_matrix([a[:3, :3], b[:3, :3]])
        )([timestamp]).as_matrix()[0]
        velocity = (1 - fraction) * velocities[0] + fraction * velocities[1]
        return pose, velocity


def navigation_plus(state, step):
    result = NavigationState(
        state.timestamp, split_pose_plus(state.pose, step[:6]),
        state.velocity.copy(), state.accel_bias.copy(), state.gyro_bias.copy(),
    )
    result.velocity += step[6:9]
    result.accel_bias += step[9:12]
    result.gyro_bias += step[12:15]
    return result


def normal_huber_cost(residual, scale, delta, dim=3):
    norms = np.linalg.norm(np.asarray(residual).reshape(-1, dim) * scale, axis=1)
    return gv._normal_huber_cost_from_norms(norms, delta)


def stamp(message):
    return message.header.stamp.sec + message.header.stamp.nanosec * 1e-9


def state(value):
    if value is None:
        return None
    return dict(timestamp=value.timestamp, pose=value.pose.tolist(), velocity=value.velocity.tolist(),
                accel_bias=value.accel_bias.tolist(), gyro_bias=value.gyro_bias.tolist())


def cosine(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return None if norm < 1e-15 else float(np.clip(np.dot(a, b) / norm, -1., 1.))


class IterationObserver:
    """Read the exact production iteration, candidate, and accepted states."""
    def __init__(self, target=None):
        if hasattr(target, 'reference_marginalization') or (isinstance(target, type) and issubclass(target, rt.ReferenceMarginalizationSolver)):
            self.mode = 'reference_marginalization'
            fn = rt.ReferenceMarginalizationSolver._solve
            lines, first = inspect.getsourcelines(fn)
            self.code = fn.__code__
            locate = lambda token: [first+i for i,s in enumerate(lines) if token in s]
            self.start = locate('self.last_optimization_trace.append(dict(')[0]
            self.iterations = []
            return
        self.mode = 'pairwise'
        function = gv.GtsamVisualInertialSolver.refine_current if target is None else target
        lines, first = inspect.getsourcelines(function)
        if not any('self.last_optimization_trace.append(trace)' in line for line in lines):
            # The public entry point owns the per-solve projection cache.
            function = gv.GtsamVisualInertialSolver._refine_current
            lines, first = inspect.getsourcelines(function)
        self.code = function.__code__
        locate = lambda token: [first+i for i,s in enumerate(lines) if token in s]
        self.start = locate('self.last_optimization_trace.append(trace)')[0]
        self.candidate = locate('trace["attempts"].append({')[-1]
        finishes = locate('if not valid_costs:') or locate('if trace["accepted_scale"] is None:')
        self.finish = finishes[0]
        self.iterations = []

    def dispatch(self, frame, event, arg):
        return self.observe if frame.f_code is self.code else None

    def observe(self, frame, event, arg):
        if event != 'line':
            return self.observe
        values = frame.f_locals
        if self.mode == 'reference_marginalization':
            if frame.f_lineno == self.start:
                solver = values['self']
                val_dict = values['values']
                cand_dict = values['candidate']
                layout = values.get('solve_layout', values.get('layout'))
                prior = values['prior']
                factors = values['factors']
                reference_id = values['reference_id']
                current_id = values['current_id']
                step = values['step']
                prev_time = values['previous'].timestamp
                curr_time = values['current'].timestamp
                extrinsic = values['extrinsic']
                visual = values['visual']
                iteration = values['iteration']
                converged = values['converged']
                normal_cost = values['normal_cost']
                rank = values['rank']
                singular = values['singular']
                pose_steps = values['pose_steps']

                prev_blocks = rt.state_blocks(prev_time)
                curr_blocks = rt.state_blocks(curr_time)

                prev_state_before = (rt.navigation_state(val_dict, prev_time)
                                     if prev_blocks[0] in layout.slices else values['previous'])
                curr_state_before = rt.navigation_state(val_dict, curr_time)

                prev_state_after = (rt.navigation_state(cand_dict, prev_time)
                                    if prev_blocks[0] in layout.slices else values['previous'])
                curr_state_after = rt.navigation_state(cand_dict, curr_time)

                curr_delta = rt._expanded_step(layout, curr_blocks, step)
                pose_step = curr_delta[:6]
                velocity_step = curr_delta[6:9]
                bias_step = curr_delta[9:15]

                dp = (rt._expanded_step(layout, prev_blocks, step)
                      if prev_blocks[0] in layout.slices else np.zeros(15))

                ref_world = val_dict[reference_id]
                ref_pose_gelsight = gelsight_pose(ref_world, extrinsic)
                cur_pose_gelsight = gelsight_pose(val_dict[current_id], extrinsic)

                evaluate = values['evaluate']
                evaluated = evaluate(ref_pose_gelsight, cur_pose_gelsight)
                residual, Jr, Jt = evaluated
                Jr, Jt = solver._normal_pose_rows(visual, ref_pose_gelsight, cur_pose_gelsight, Jr, Jt)
                gel_J = values['gel_J']
                normal_scale = np.sqrt(solver.visual_weight)
                dim = gv._normal_dim(visual)
                normal_A, normal_b, _, _ = gv._normal_huber_linearization(
                    residual, np.column_stack((Jr @ gel_J, Jt @ gel_J)),
                    normal_scale, solver.normal_huber_delta_at(iteration), dim)

                imu_terms = gv.navigation_factor_terms(factors[0], prev_state_before, curr_state_before)
                bias_terms = gv.navigation_factor_terms(factors[1], prev_state_before, curr_state_before)
                imu_cost = float(solver.imu_weight * (imu_terms[2] @ imu_terms[2]) / 2)
                bias_cost = float(solver.imu_weight * (bias_terms[2] @ bias_terms[2]) / 2)

                prior_vals = dict(prior.fixed_values)
                prior_vals.update(rt.state_values(prev_state_before))
                if reference_id in prior.anchors and reference_id not in prior_vals:
                    prior_vals[reference_id] = ref_world
                prior_cost = float(prior.error_values(prior_vals))

                if 'squared_norm' in values:
                    base_cost = float(values['squared_norm'] / 2)
                elif 'rhs' in values:
                    base_cost = float(values['rhs'] @ values['rhs'] / 2)
                else:
                    base_cost = normal_cost + imu_cost + bias_cost + prior_cost
                active_rows = visual.last_active_rows

                raw_norms_dict = dict(
                    previous=gv._navigation_step_norms(dp),
                    target=gv._navigation_step_norms(curr_delta)
                )

                trace = dict(
                    iteration=iteration,
                    update_rule="full_gauss_newton",
                    variable_reference=reference_id not in prior.fixed_values,
                    first_pose_fixed=True,
                    solve_dimension=layout.dimension,
                    numerical_rank=int(rank),
                    scaled_min_singular=float(singular[-1]),
                    base_cost=base_cost,
                    base_normal_cost=normal_cost,
                    base_imu_cost=imu_cost,
                    base_bias_cost=bias_cost,
                    base_prior_cost=prior_cost,
                    accepted_scale=1.0,
                    accepted_cost=base_cost,
                    accepted_normal_cost=normal_cost,
                    accepted_imu_cost=imu_cost,
                    accepted_bias_cost=bias_cost,
                    accepted_prior_cost=prior_cost,
                    active_rows=int(np.count_nonzero(active_rows)),
                    raw_step_norms=raw_norms_dict,
                    convergence_step_norms=dict(
                        rotation_rad=float(max(np.linalg.norm(s[:3]) for s in pose_steps)),
                        translation_xy_m=float(max(np.linalg.norm(s[3:5]) for s in pose_steps))
                    ),
                    attempts=[dict(
                        scale=1.0, cost=base_cost, reason="full_gn_step",
                        imu_cost=imu_cost, bias_cost=bias_cost, prior_cost=prior_cost, normal_cost=normal_cost
                    )],
                    step_converged=bool(converged),
                    stop_reason="step_converged" if converged else "max_iterations" if iteration + 1 == solver.max_iterations else None
                )

                candidates = [dict(
                    scale=1.0,
                    current=curr_state_after,
                    previous=prev_state_after,
                    costs=dict(normal=normal_cost, imu=imu_cost, bias=bias_cost, prior=prior_cost, total=base_cost)
                )]

                ref_step = step[layout.slices[reference_id]] if reference_id in layout.slices else np.zeros(6)

                record = dict(
                    previous=prev_state_before,
                    current=curr_state_before,
                    after_previous=prev_state_after,
                    after_current=curr_state_after,
                    reference_pose=ref_pose_gelsight,
                    reference_block=reference_id,
                    reference_world_pose=ref_world.copy(),
                    ref_step=ref_step,
                    prior=prior,
                    normal_A=normal_A,
                    normal_b=normal_b,
                    residual=residual,
                    pose_step=pose_step,
                    velocity_step=velocity_step,
                    bias_step=bias_step,
                    previous_step=dp,
                    active_rows=active_rows,
                    visual=copy.copy(visual),
                    imu_factor=factors[0],
                    bias_factor=factors[1],
                    normal_scale=normal_scale,
                    solver=solver,
                    candidates=candidates,
                    iteration=iteration,
                    direction_attempt=0,
                    trace=trace
                )
                self.iterations.append(record)
            return self.observe
        if frame.f_lineno == self.start:
            keys = ('previous', 'current', 'reference_pose', 'prior', 'normal_A', 'normal_b',
                    'normal_jacobian', 'residual', 'pose_step', 'velocity_step', 'bias_step',
                    'previous_step', 'active_rows')
            record = {}
            for key in keys:
                value = values[key]
                record[key] = (value.copy() if isinstance(value, (np.ndarray, NavigationState))
                               else copy.deepcopy(value))
            record.update(visual=copy.copy(values['visual']),
                          imu_factor=values['nonlinear_imu_factor'],
                          bias_factor=values['nonlinear_bias_factor'],
                          normal_scale=values['normal_scale'], solver=values['self'],
                          candidates=[], iteration=values['iteration'],
                          direction_attempt=values.get('direction_attempt', 0))
            self.iterations.append(record)
        elif (frame.f_lineno == self.candidate
              and len(self.iterations[-1]['candidates']) <= len(values['trace']['attempts'])):
            # CPython can emit this line twice while evaluating a multiline
            # append call. Capture each candidate only before its first visit.
            self.iterations[-1]['candidates'].append(dict(
                scale=values['scale'], current=values['candidate'].copy(),
                previous=values['candidate_previous'].copy(),
                costs=dict(normal=values['candidate_normal_cost'], imu=values['candidate_imu_cost'],
                           bias=values['candidate_bias_cost'], prior=values['candidate_prior_cost'],
                           total=values['candidate_cost'])))
        elif frame.f_lineno == self.finish:
            self.iterations[-1].update(after_previous=values['previous'].copy(),
                                       after_current=values['current'].copy(),
                                       trace=copy.deepcopy(values['trace']))
        return self.observe

    def run(self, function, *args, **kwargs):
        prior_trace = sys.gettrace()
        try:
            sys.settrace(self.dispatch)
            return function(*args, **kwargs)
        finally:
            sys.settrace(prior_trace)


def read_bag(path):
    connection = sqlite3.connect('file:'+str(path)+'?mode=ro', uri=True)
    kinds = {'/image': Image, '/imu/data_raw': Imu,
             '/ground_truth/gel_pose': PoseStamped, '/ground_truth/imu_pose': PoseStamped}
    result = {}
    for name, identifier in connection.execute('SELECT name,id FROM topics'):
        if name in kinds:
            result[name] = [(ns, bytes(blob), deserialize_message(bytes(blob), kinds[name]))
                            for ns, blob in connection.execute(
                                'SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp,id',
                                (identifier,))]
    connection.close()
    return result


def pose(message):
    q, p = message.pose.orientation, message.pose.position
    result = np.eye(4)
    result[:3,:3] = Rotation.from_quat([q.x,q.y,q.z,q.w]).as_matrix()
    result[:3,3] = [p.x,p.y,p.z]
    return result


class Truth:
    def __init__(self, messages, archive):
        self.states = GroundTruthImuStates(max_samples=max(10000, len(messages) + 100))
        for _, _, message in messages:
            p = pose(message)
            self.states.add(stamp(message), p[:3,:3], p[:3,3])
        self.transform = None
        with zipfile.ZipFile(archive) as z:
            entry = next(n for n in z.namelist() if n.endswith('/assets/trajectory_truth.npz'))
            data = np.load(io.BytesIO(z.read(entry)))
            self.bias_times = data['t_s'] + 1.
            self.ba, self.bg = data['bias_accel_m_s2'], data['bias_gyro_rad_s']
            omega = np.einsum('nij,nj->ni', data['T_WI'][:,:3,:3], data['ideal_gyro_rad_s'])
            self.v = data['velocity_WS_m_s'] + np.cross(omega, data['T_WI'][:,:3,3]-data['T_WS'][:,:3,3])
            self.p = data['T_WI']
            # Check the native sidecar belongs to these exact bag GT samples.
            for _, _, message in messages[::50]:
                k = int(np.argmin(abs(self.bias_times-stamp(message))))
                assert abs(self.bias_times[k]-stamp(message)) < 1e-8
                assert np.allclose(data['T_WI'][k], pose(message), atol=1e-10)

    def at(self, time):
        value = self.states.at(time)
        if value is None:
            k = int(np.argmin(abs(self.bias_times - time)))
            if abs(self.bias_times[k] - time) < 1e-4:
                value = (self.p[k], self.v[k])
            else:
                raise ValueError('No central-difference GT support at '+str(time))
        if not self.bias_times[0] <= time <= self.bias_times[-1]:
            raise ValueError('No bias GT support')
        p, v = value
        interpolate = lambda a: np.array([np.interp(time,self.bias_times,a[:,k]) for k in range(3)])
        return NavigationState(time, self.transform @ p, self.transform[:3,:3] @ v,
                               interpolate(self.ba), interpolate(self.bg))

    def error(self, estimate, extrinsic):
        if estimate is None or self.transform is None:
            return None
        target = self.at(estimate.timestamp)
        a, b = gelsight_pose(estimate.pose, extrinsic), gelsight_pose(target.pose, extrinsic)
        return dict(position_mm=float(np.linalg.norm(a[:3,3]-b[:3,3])*1000),
                    rotation_deg=float(np.rad2deg(Rotation.from_matrix(b[:3,:3].T@a[:3,:3]).magnitude())),
                    velocity_mps=float(np.linalg.norm(estimate.velocity-target.velocity)),
                    accel_bias_mps2=float(np.linalg.norm(estimate.accel_bias-target.accel_bias)),
                    gyro_bias_rps=float(np.linalg.norm(estimate.gyro_bias-target.gyro_bias)))


def costs(record, previous, current, reference=None, oracle_z=False):
    solver = record['solver']
    visual = copy.copy(record['visual'])
    reference = record['reference_pose'] if reference is None else reference
    if oracle_z:
        visual.fixed_reference_z_m = float((np.linalg.inv(reference) @
                                           gelsight_pose(current.pose, record['extrinsic']))[2,3])
    residual = visual.residual_on_fixed_rows(
        reference, gelsight_pose(current.pose, record['extrinsic']), record['active_rows'])
    dim = gv._normal_dim(visual)
    normal = None if residual is None else normal_huber_cost(
        residual, record['normal_scale'], solver.normal_huber_delta_at(record['iteration']), dim)
    values = gv._imu_values(previous,current)
    imu = solver.imu_weight * float(record['imu_factor'].error(values))
    bias = solver.imu_weight * float(record['bias_factor'].error(values))
    if record['prior'] is None:
        prior = 0.
    elif hasattr(record['prior'], 'error_values'):
        vals = dict(record['prior'].fixed_values)
        vals.update(rt.state_values(previous))
        ref_block = record.get('reference_block')
        if ref_block is not None and ref_block in record['prior'].anchors and ref_block not in vals:
            vals[ref_block] = record['reference_world_pose']
        prior = float(record['prior'].error_values(vals))
    else:
        prior = float(record['prior'].error(previous))
    return dict(normal=normal,imu=imu,bias=bias,prior=prior,
                total=None if normal is None else normal+imu+bias+prior)


def describe_iteration(record, truth, extrinsic, reference_time, path):
    previous, current, solver = record['previous'], record['current'], record['solver']
    record['extrinsic'] = extrinsic
    gt_previous, gt_current = truth.at(previous.timestamp), truth.at(current.timestamp)
    delta = np.r_[record['pose_step'], record['velocity_step'], record['bias_step']]
    dp = np.zeros(15) if record['previous_step'] is None else record['previous_step']
    correction = split_pose_coordinates(current.pose,gt_current.pose)
    if record['normal_A'].shape[1] == 12:
        step12 = np.r_[record.get('ref_step', np.zeros(6)), delta[:6]]
        projections = dict(normal=float(record['normal_b']@(record['normal_A']@step12)))
    else:
        projections = dict(normal=float(record['normal_b']@(record['normal_A']@delta[:6])))
    arrays = dict(normal_A=record['normal_A'],normal_b=record['normal_b'],
                  raw_residual=record['residual'],active_rows=record['active_rows'],
                  current_step=delta,previous_step=dp)
    for name in ('imu','bias'):
        a,b,rhs = gv.navigation_factor_terms(record[name+'_factor'],previous,current)
        projections[name] = float(-solver.imu_weight*rhs@(a@dp+b@delta))
        arrays.update({name+'_previous_A':a,name+'_current_A':b,name+'_rhs':rhs})
    if record['prior'] is None:
        projections['prior'] = 0.
    elif hasattr(record['prior'], 'linearize_values'):
        vals = dict(record['prior'].fixed_values)
        vals.update(rt.state_values(previous))
        ref_block = record.get('reference_block')
        if ref_block is not None and ref_block in record['prior'].anchors and ref_block not in vals:
            vals[ref_block] = record['reference_world_pose']
        a_prior, rhs_prior = record['prior'].linearize_values(vals)
        prev_blocks = rt.state_blocks(previous.timestamp)
        prev_cols = [record['prior'].layout.slices[b] for b in prev_blocks if b in record['prior'].layout.slices]
        if prev_cols:
            prev_cols_flat = np.concatenate([np.arange(s.start, s.stop) for s in prev_cols])
            projections['prior'] = float(-rhs_prior @ (a_prior[:, prev_cols_flat] @ dp[:len(prev_cols_flat)]))
        else:
            projections['prior'] = 0.
        arrays.update(prior_A=a_prior, prior_rhs=rhs_prior, prior_frozen_A=record['prior'].A,
                      prior_frozen_rhs=record['prior'].rhs)
    else:
        a,rhs = record['prior'].linearize(previous)
        projections['prior'] = float(-rhs@(a@dp))
        arrays.update(prior_A=a,prior_rhs=rhs,prior_frozen_A=record['prior'].A,
                      prior_frozen_rhs=record['prior'].rhs)
    derivative_sweep = []
    for epsilon in (1e-4,1e-3,1e-2,.1):
        plus = costs(record, navigation_plus(previous,dp*epsilon), navigation_plus(current,delta*epsilon))
        minus = costs(record, navigation_plus(previous,-dp*epsilon), navigation_plus(current,-delta*epsilon))
        derivative_sweep.append(dict(epsilon=epsilon,derivative={
            k:None if plus[k] is None or minus[k] is None else (plus[k]-minus[k])/(2*epsilon)
            for k in ('normal','imu','bias','prior','total')}))
    numerical = derivative_sweep[0]['derivative']
    profile = []
    for alpha in np.linspace(-.5,1.5,41):
        # Pose-only slice. All other variables and the shared rows are fixed.
        candidate = current.copy()
        candidate.pose = split_pose_plus(current.pose, correction*alpha)
        profile.append(dict(alpha=float(alpha),costs=costs(record,previous,candidate)))
    reference_gt = gelsight_pose(truth.at(reference_time).pose,extrinsic)
    points=record['visual'].candidate_points[record['active_rows']]
    arrays['reference_pixels']=(points[:,:2]*1000/record['visual'].ppmm
                                + np.array(record['visual'].ref.N.shape[1::-1])/2-.5)
    np.savez_compressed(path,**arrays)
    base = dict(previous=state(previous),current=state(current),
                previous_error=truth.error(previous,extrinsic),current_error=truth.error(current,extrinsic))
    after = dict(previous=state(record['after_previous']),current=state(record['after_current']),
                 previous_error=truth.error(record['after_previous'],extrinsic),
                 current_error=truth.error(record['after_current'],extrinsic))
    gt_pose = current.copy();gt_pose.pose = gt_current.pose.copy()
    return dict(iteration=record['iteration'],direction_attempt=record.get('direction_attempt',0),
                before=base,after=after,
                gt_previous=state(gt_previous),gt_current=state(gt_current),
                trace=record['trace'],raw_step=delta.tolist(),previous_raw_step=dp.tolist(),
                gt_pose_correction=correction.tolist(),
                direction_cosine=dict(translation=cosine(delta[3:6],correction[3:6]),
                                      rotation=cosine(delta[:3],correction[:3])),
                directional_derivative=projections,numerical_derivative=numerical,
                derivative_sweep=derivative_sweep,
                gt_pose_cost=costs(record,previous,gt_pose),
                full_gt_cost=costs(record,gt_previous,gt_current,reference_gt),
                full_gt_oracle_z_cost=costs(record,gt_previous,gt_current,reference_gt,oracle_z=True),
                gt_pose_cost_slice=profile,arrays=str(path.name),
                normal_fixed_reference_z_m=record['visual'].fixed_reference_z_m,
                normal_effective_target_pose=record['visual']._normal_target_pose(
                    record['reference_pose'],gelsight_pose(current.pose,extrinsic)).tolist(),
                candidates=[dict(scale=c['scale'],previous=state(c['previous']),current=state(c['current']),
                                 costs=c['costs'],error=truth.error(c['current'],extrinsic))
                            for c in record['candidates']])


def replay(inputs, config, output, truth, observed, analysis_start, observations, all_images=False):
    np.random.seed(17);torch.manual_seed(17);torch.set_num_threads(1)
    log=(output/('tracker.log' if observed else 'control_tracker.log')).open('w')
    logger=SimpleNamespace(info=lambda m:log.write(str(m)+'\n'),warning=lambda m:log.write(str(m)+'\n'))
    tracker=tm.Tracker(config['device_config']['calibration_model_path'],copy.deepcopy(config),True,logger)
    solver_cls=type(tracker.frontend_visual_inertial)
    is_ref_marg=getattr(tracker.frontend_visual_inertial, 'reference_marginalization', False)
    original=solver_cls.refine_current
    original_quality=tm.normalflow_quality
    calls=[];raw_records=[];event_calls=[]
    def refine(solver,visual,reference,previous,current,samples,extrinsic,preintegrated=None,**kwargs):
        observer=IterationObserver(solver if is_ref_marg else original)
        result=(observer.run(original,solver,visual,reference,previous,current,samples,extrinsic,preintegrated=preintegrated,**kwargs)
                if observed else original(solver,visual,reference,previous,current,samples,extrinsic,preintegrated=preintegrated,**kwargs))
        reference_frame=(visual.ref if visual is not None
                         else tracker.latest_normal_correspondences['reference'])
        reference_header=(tracker.ref_header if reference_frame is tracker.ref_frame else tracker.prev_header)
        item=dict(previous=state(previous),initial=state(current),result=state(result),
                  reference_pose=reference.tolist(),reference_time=tracker._imu_timestamp(reference_header),
                  imu_start=samples[0].timestamp,imu_end=samples[-1].timestamp,imu_samples=len(samples),
                  gravity=solver.gravity.tolist(),prior_source=solver.last_prior_source,
                  failure=solver.last_refinement_failure,trace=copy.deepcopy(solver.last_optimization_trace),
                  fixed_reference_z_m=solver.last_fixed_normal_reference_z_m)
        calls.append(item);event_calls.append(len(calls)-1)
        raw_records.append(observer.iterations)
        return result
    solver_cls.refine_current=refine
    def quality(*args,**kwargs):
        result=original_quality(*args,**kwargs)
        if calls:
            calls[-1]['quality_scr_ccs']=[float(x) for x in result]
        return result
    tm.normalflow_quality=quality
    bridge=CvBridge();indices={name:0 for name in inputs if name!='/image'};events=[];processed=0
    contact_index=None
    try:
        for _,_,image in inputs['/image']:
            t=stamp(image)
            for name,index in list(indices.items()):
                series=inputs[name]
                while index<len(series) and stamp(series[index][2])<=t+.005+1e-9:
                    m=series[index][2]
                    if name=='/imu/data_raw':
                        tracker.add_imu_sample(ImuSample(stamp(m),[m.linear_acceleration.x,m.linear_acceleration.y,m.linear_acceleration.z],
                                                        [m.angular_velocity.x,m.angular_velocity.y,m.angular_velocity.z]))
                    else:
                        p=pose(m)
                        if name=='/ground_truth/gel_pose':
                            tracker.add_ground_truth_gel_pose(stamp(m),p[:3,:3],p[:3,3])
                        elif hasattr(tracker, 'add_ground_truth_imu_pose'):
                            tracker.add_ground_truth_imu_pose(stamp(m),p[:3,:3],p[:3,3])
                        elif hasattr(tracker, 'add_ground_truth_imu_position'):
                            tracker.add_ground_truth_imu_position(stamp(m),p[:3,3])
                    index+=1
                indices[name]=index
            before=state(tracker._frontend_vi_state);event_calls.clear()
            valid,emitted=tracker.track(bridge.imgmsg_to_cv2(image,'bgr8'),image.header)
            processed+=1
            if truth.transform is None and tracker._frontend_vi_state is not None:
                a=tracker._frontend_vi_state
                truth.transform=a.pose@np.linalg.inv(truth.states.at(a.timestamp)[0])
            if contact_index is None and valid and tracker._frontend_vi_state is not None:
                contact_index=processed-1
            if t>=analysis_start-1e-9 and (all_images or events or (valid and tracker._frontend_vi_state is not None)):
                events.append(dict(frame=('I'+str(processed-1) if all_images else 'F'+str(len(events))),
                                   image_index=processed-1,
                                   contact_label=None if contact_index is None else 'F'+str(processed-1-contact_index),
                                   timestamp=t,before=before,
                                   after=state(tracker._frontend_vi_state),valid=bool(valid),
                                   published=bool(emitted is not None and emitted[0] is not None),
                                   calls=list(event_calls),reference_time=tracker._imu_timestamp(tracker.ref_header),
                                   mode=tracker.tracking_mode.value,
                                   after_error=truth.error(tracker._frontend_vi_state,tracker.gelsight_T_imu),
                                   diagnostic=copy.deepcopy(getattr(tracker, 'latest_gt_state_test', None))))
                if not all_images and len(events)==observations:
                    break
    finally:
        solver_cls.refine_current=original
        tm.normalflow_quality=original_quality
        log.close()
    if not all_images:
        assert len(events)==observations,'Not enough observations after the contact anchor'
    else:
        assert processed==len(inputs['/image']),'The full input bag was not processed'
    return dict(events=events,calls=calls,records=raw_records,tracker=tracker,processed=processed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bag',type=Path,required=True);p.add_argument('--prefix-bag',type=Path)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--bias-archive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--observations',type=int,default=5)
    p.add_argument('--all-images',action='store_true',help='Record every image, including startup and failed inputs')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'result.json').exists():raise RuntimeError('Choose a fresh output')
    config=yaml.safe_load(args.config.read_text());(args.output/'config_supplied.yaml').write_text(yaml.safe_dump(config))
    short=read_bag(args.bag)
    full=read_bag(args.prefix_bag) if args.prefix_bag else short
    assert len(short['/image'])>=args.observations
    audits={}
    for name,rows in short.items():
        originals={t:b for t,b,_ in full[name]}
        if args.prefix_bag:
            assert all(originals.get(t)==b for t,b,_ in rows),name
        audits[name]=dict(count=len(rows),all_payloads_match_original=True if args.prefix_bag else None)
    start=stamp(short['/image'][0][2])
    inputs={name:sorted([r for r in full[name] if r[0]<short[name][0][0]]+short[name],key=lambda r:r[0])
            for name in short if name!='/image'}
    inputs['/image']=[r for r in full['/image'] if stamp(r[2])<start-1e-9]+short['/image']
    truth=Truth(inputs['/ground_truth/imu_pose'],args.bias_archive)
    hashes={m.__file__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (tm,gv,rt)}
    observed=replay(inputs,config,args.output,truth,True,start,args.observations,args.all_images)
    control=replay(inputs,config,args.output,truth,False,start,args.observations,args.all_images)
    assert observed['events']==control['events'],'Observer changed tracking states or decisions'
    assert observed['calls']==control['calls'],'Observer changed solver states or costs'
    assert hashes=={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in hashes}
    calls=observed['calls'];extrinsic=observed['tracker'].gelsight_T_imu
    for index,(call,records) in enumerate(zip(calls,observed['records'])):
        call['iterations']=[describe_iteration(r,truth,extrinsic,call['reference_time'],
                                              args.output/('call_%02d_iteration_%02d_direction_%02d.npz'%(
                                                  index,r['iteration'],r.get('direction_attempt',0)))) for r in records]
        for stage in ('previous','initial','result'):
            s=call[stage]
            value=None if s is None else NavigationState(s['timestamp'],np.array(s['pose']),np.array(s['velocity']),
                                                          np.array(s['accel_bias']),np.array(s['gyro_bias']))
            call[stage+'_error']=truth.error(value,extrinsic)
    import cv2
    selected_times={round(e['timestamp'],9) for e in observed['events']}
    chosen=[r for r in inputs['/image'] if round(stamp(r[2]),9) in selected_times]
    for event,(_,_,message) in zip(observed['events'],chosen):
        event['image_file']=event['frame']+'.png'
        cv2.imwrite(str(args.output/event['image_file']),bridge_image(message))
        event['target_gt']=state(truth.at(event['timestamp']))
    reference_images={round(e['timestamp'],9):e['image_file'] for e in observed['events']}
    for call in calls:
        call['reference_image']=reference_images[round(call['reference_time'],9)]
    result=dict(events=observed['events'],calls=calls,navigation_T_world=truth.transform.tolist(),
                extrinsic=extrinsic.tolist(),config=config,scope='production frontend; offline callbacks; no backend',
                selected_interval=[observed['events'][0]['timestamp'],observed['events'][-1]['timestamp']],
                prefix_images=observed['processed']-len(chosen),
                processed_images=observed['processed'],all_images=args.all_images,
                short_bag_audit=audits,bias_truth_archive=str(args.bias_archive),code_hashes=hashes,
                observer_matches_control_exactly=True,callback_lookahead_s=.005,
                statistics=observed['tracker'].frontend_tracking_statistics())
    def json_default(o):
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False,default=json_default)+'\n')
    print(json.dumps(dict(frames=len(result['events']),calls=len(calls),statistics=result['statistics'],
                         observer_matches_control_exactly=True),indent=2))


def bridge_image(message):
    return CvBridge().imgmsg_to_cv2(message,'bgr8')


if __name__=='__main__':
    main()
