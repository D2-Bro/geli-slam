#!/usr/bin/env python3
"""Run isolated tactile/IMU diagnostics. Do not change estimator settings.

Use the project's gel-slam Python environment. ROS is needed only for T12.
FAIL means that a stated model/state contract failed. WARN marks an IC
approximation or missing uncertainty; it does not assert an exact-J bug.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/gelslam"))

import cv2
import gtsam
import numpy as np
import normalflow.utils as nf
import yaml
from scipy.spatial.transform import Rotation

from gelslam.core import gtsam_vi as gv
from gelslam.core import numpy_vi as nv
from gelslam.core.inertial import ImuBuffer, ImuSample

SEED = 20261013
GRAVITY = np.array([0., 0., -9.80665])
SCALES = np.array([.1] * 3 + [.01] * 3 + [.1] * 3 + [.01] * 3 + [.001] * 3)
BLOCKS = {"rotation": slice(0, 3), "translation": slice(3, 6),
          "velocity": slice(6, 9), "accel_bias": slice(9, 12),
          "gyro_bias": slice(12, 15)}


def pose(rot=(0, 0, 0), pos=(0, 0, 0)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(rot).as_matrix()
    result[:3, 3] = pos
    return result


def state(t, T=None, v=None, ba=None, bg=None):
    return nv.NavigationState(t, np.eye(4) if T is None else T.copy(),
                              np.zeros(3) if v is None else np.array(v, float),
                              np.zeros(3) if ba is None else np.array(ba, float),
                              np.zeros(3) if bg is None else np.array(bg, float))


def plus(x, d):
    """Use exactly the active frontend's GTSAM pose retraction."""
    y = x.copy()
    y.pose = gtsam.Pose3(x.pose).retract(d[:6]).matrix()
    y.velocity += d[6:9]
    y.accel_bias += d[9:12]
    y.gyro_bias += d[12:15]
    return y


def numeric(fun, x, eps):
    cols = []
    for j, h in enumerate(eps):
        d = np.zeros(len(eps))
        d[j] = h
        cols.append((fun(plus(x, d)) - fun(plus(x, -d))) / (2 * h))
    return np.column_stack(cols)


def spectrum(J, scales=None):
    scaled = J if scales is None else J * scales
    _, s, vt = np.linalg.svd(scaled, full_matrices=True)
    threshold = max(1e-10, (s[0] if len(s) else 0) * 1e-8)
    rank = int(np.count_nonzero(s > threshold))
    return {"rank": rank, "singular_values": s.tolist(), "threshold": threshold,
            "observable_condition": float(s[0] / s[rank - 1]) if rank else None,
            "full_condition": float(s[0] / s[-1]) if rank == J.shape[1] else None,
            "null_vectors_scaled": vt[rank:].tolist()}


def error_row(factor, endpoint, block, epsilon, A, B, verdict="PASS"):
    diff = A - B
    return dict(factor=factor, endpoint=endpoint, block=block, epsilon=epsilon,
                max_absolute_error=float(np.max(np.abs(diff))),
                relative_error=float(np.linalg.norm(diff) / max(np.linalg.norm(B), 1e-12)),
                analytic_norm=float(np.linalg.norm(A)), numeric_norm=float(np.linalg.norm(B)),
                status=verdict)


def normal_field(xy, kind="asymmetric"):
    x, y = np.asarray(xy).T
    if kind == "plane":
        normals = np.column_stack((x * 0, y * 0, x * 0 + 1))
    elif kind == "sphere":
        normals = np.column_stack((x * 8, y * 8, np.sqrt(1 - (x * 8)**2 - (y * 8)**2)))
    else:
        normals = np.column_stack((.1 * np.sin(x * 190) + .035 * np.cos(y * 160),
                                   .12 * np.cos(y * 130) + .04 * np.sin((x + y) * 120),
                                   np.ones_like(x)))
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


def visual_fixture(kind="asymmetric"):
    h, w, scale = 80, 80, 1.0
    yy, xx = np.mgrid[:h, :w]
    xy = np.column_stack(((xx.ravel() - w / 2 + .5) * .001,
                          (yy.ravel() - h / 2 + .5) * .001))
    N = normal_field(xy, kind).reshape(h, w, 3).astype(np.float32)
    C = np.zeros((h, w), bool)
    C[20:60, 20:60] = True
    L = np.zeros((h, w), np.float32)
    L[C] = np.linspace(.1, .6, C.sum())
    H = np.zeros((h, w), np.float32)
    frame = SimpleNamespace(N=N, C=C, H=H, L=L)
    visual = nv.DenseNormalResidual(frame, frame, scale, np.eye(4), np.eye(4), 1400)
    return visual


def smooth_forward(visual, reference, target, rows):
    """Analytic float64 normal field. Production uses quantized cv2.remap."""
    T = np.linalg.inv(target) @ reference
    P = visual.candidate_points[rows] @ T[:3, :3].T + T[:3, 3]
    return (normal_field(P[:, :2]) @ T[:3, :3]
            - visual.candidate_normals[rows]).reshape(-1)


def normal_gradient(xy):
    x,y=np.asarray(xy).T
    f=np.column_stack((.1*np.sin(x*190)+.035*np.cos(y*160),
                       .12*np.cos(y*130)+.04*np.sin((x+y)*120),np.ones_like(x)))
    df=np.zeros((len(x),3,2))
    df[:,0,0]=19*np.cos(x*190);df[:,0,1]=-5.6*np.sin(y*160)
    df[:,1,0]=4.8*np.cos((x+y)*120)
    df[:,1,1]=-15.6*np.sin(y*130)+4.8*np.cos((x+y)*120)
    length=np.linalg.norm(f,axis=1)
    n=f/length[:,None]
    D=(np.eye(3)[None]-n[:,:,None]*n[:,None,:])/length[:,None,None]
    return n,D@df


def forward_exact(visual,reference,target,rows):
    T=np.linalg.inv(target)@reference
    points=visual.candidate_points[rows]@T[:3,:3].T+T[:3,3]
    normals,grad=normal_gradient(points[:,:2])
    blocks=[]
    for q,n,D in zip(points,normals,grad):
        spatial=np.column_stack((D,np.zeros(3)))
        blocks.append(T[:3,:3].T@np.column_stack((spatial@nv._skew(q)-nv._skew(n),-spatial)))
    return np.concatenate(blocks)


def ic_local(visual, delta, rows):
    T = gtsam.Pose3(np.eye(4)).retract(delta[:6]).matrix()
    P = visual.candidate_points[rows] @ T[:3, :3].T + T[:3, 3]
    return (normal_field(P[:, :2]) @ T[:3, :3]
            - visual.candidate_normals[rows]).reshape(-1)


def analytic_motion(duration=.2, dt=.005, omega=(0, 0, 0), a=(0, 0, 0),
                    v=(.03, -.01, .02), ext=None, ba=(.003, -.002, .004),
                    bg=(.0002, -.0001, .00015), alpha=(0,0,0)):
    """Independent Gel-origin trajectory. Include IMU lever-arm acceleration."""
    ext = np.eye(4) if ext is None else ext
    omega, a, v, ba, bg = map(lambda z: np.asarray(z, float), (omega, a, v, ba, bg))
    alpha=np.asarray(alpha,float)
    if np.linalg.norm(np.cross(omega,alpha))>1e-12:
        raise ValueError("analytic angular acceleration must share its rotation axis")
    R0 = Rotation.from_rotvec([.12, -.08, .2]).as_matrix()
    times = np.linspace(0, duration, int(round(duration / dt)) + 1)
    lever = ext[:3, 3]
    samples = []
    for t in times:
        R = R0 @ Rotation.from_rotvec(omega*t+.5*alpha*t*t).as_matrix()
        wt=omega+alpha*t
        ai = a + R @ (np.cross(alpha,lever)+np.cross(wt,np.cross(wt,lever)))
        RI = R @ ext[:3, :3]
        samples.append(ImuSample(float(t), RI.T @ (ai - GRAVITY) + ba,
                                 ext[:3, :3].T @ wt + bg))
    def truth(t, imu_velocity=False):
        R = R0 @ Rotation.from_rotvec(omega*t+.5*alpha*t*t).as_matrix()
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, v * t + .5 * a * t * t
        velocity = v + a * t
        if imu_velocity:
            velocity = velocity + R @ np.cross(omega+alpha*t,lever)
        return state(t, T, velocity, ba, bg)
    return samples, truth(0), truth(duration), truth(0, True), truth(duration, True)


def run_audit(output, config_path):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(config_path).read_text())["imu_config"]
    rng = np.random.default_rng(SEED)
    results, jacrows, arrays = {}, [], {}
    def record(key, status, **data):
        results[key] = dict(status=status, **data)
    # T01: independent factors; production GaussianFactorGraph assembly.
    J1, J2, r1, r2 = rng.normal(size=(24, 15)), rng.normal(size=(15, 15)), rng.normal(size=24), rng.normal(size=15)
    J, r = np.vstack((J1, J2)), np.r_[r1, r2]
    H, g = J1.T @ J1 + J2.T @ J2, J1.T @ r1 + J2.T @ r2
    graph = gtsam.GaussianFactorGraph()
    for A, b in ((J1, -r1), (J2, -r2)):
        graph.add(gtsam.JacobianFactor(1, A[:, :6], 2, A[:, 6:9], 3, A[:, 9:], b,
                                      gtsam.noiseModel.Isotropic.Sigma(len(b), 1)))
    Ag, bg = graph.jacobian()
    sol = graph.optimize()
    delta = np.r_[sol.at(1), sol.at(2), sol.at(3)]
    assembly = max(np.max(np.abs(H - Ag.T @ Ag)), np.max(np.abs(g + Ag.T @ bg)),
                   np.max(np.abs(delta + np.linalg.solve(H, g))))
    record("T01", "PASS" if assembly < 1e-10 else "FAIL", max_error=float(assembly),
           stack_accumulated_H_error=float(np.max(np.abs(H - J.T @ J))),
           stack_accumulated_g_error=float(np.max(np.abs(g - J.T @ r))))
    # T02: the visual missing direction can receive an IMU update.
    nfJ = np.eye(15)[:6]; nfJ[5] = 0
    imuJ = np.eye(15)
    rhs = np.zeros(15); rhs[5] = .004
    delta = np.linalg.solve(nfJ.T @ nfJ + imuJ.T @ imuJ, imuJ.T @ rhs)
    record("T02", "PASS" if abs(delta[5] - .004) < 1e-12 else "FAIL", missing_translation_update_m=float(delta[5]))
    # T03/T04 NormalFlow: test forward, IC local, and increment mapping separately.
    visual = visual_fixture()
    _, Kref, _ = visual.evaluate(np.eye(4), np.eye(4))
    rows = visual.last_active_rows.copy()
    cases = {"aligned": (pose(), pose(), pose()),
             "global_tilt": (pose((.3, -.4, .1)), pose((.3, -.4, .1)), pose()),
             "relative_tilt": (pose(), pose((.35, -.2, .1)), pose()),
             "sensor_z_world_x": (pose((0, np.pi / 2, 0)), pose((0, np.pi / 2, 0)), pose()),
             "extrinsic_rotation": (pose(), pose((.2, .1, -.15)), pose((-.3, .2, .4))),
             "extrinsic_lever_arm": (pose(), pose((.2, .1, -.15)), pose((-.3, .2, .4), (.012, -.018, .03)))}
    normal_cases = []
    for name, (reference, target, ext) in cases.items():
        relative = np.linalg.inv(target) @ reference
        # Same production IC chain on fixed candidates; no image-boundary switch.
        Ktarget = -Kref @ nv.adjoint(np.linalg.inv(relative))
        x = state(.2, target)
        Jnum = numeric(lambda z: smooth_forward(visual, reference, z.pose, rows), x,
                       np.r_[np.full(3, 1e-5), np.full(3, 1e-7), np.full(9, 1e-6)])[:, :6]
        null_ic = np.r_[np.zeros(3), relative[:3, :3] @ [0., 0., 1.]]
        null_target = np.array([0., 0., 0., 0., 0., 1.])
        # Map Gel tangent to IMU tangent; lever arm rotates and mixes pose blocks.
        C = nv.adjoint(ext)
        Kbody, Jbody = Ktarget @ C, Jnum @ C
        body_null = np.linalg.solve(C, null_target)
        world_null = target[:3, :3] @ [0., 0., 1.]
        normal_cases.append(dict(case=name, ic_column_5_norm=float(np.linalg.norm(Ktarget[:, 5])),
            forward_column_5_norm=float(np.linalg.norm(Jnum[:, 5])),
            ic_null=null_ic.tolist(), sensor_null=null_target.tolist(), world_null=world_null.tolist(),
            ic_null_residual=float(np.linalg.norm(Ktarget @ null_ic)),
            forward_on_ic_null=float(np.linalg.norm(Jnum @ null_ic)),
            body_null=body_null.tolist(), body_null_residual=float(np.linalg.norm(Jbody @ body_null)),
            body_ic_column_5_norm=float(np.linalg.norm(Kbody[:, 5]))))
        arrays["NF_IC_" + name], arrays["NF_forward_" + name] = Ktarget, Jnum
        Jexact=forward_exact(visual,reference,target,rows)
        for block in ("rotation","translation"):
            sl=BLOCKS[block]
            row=error_row("NF_exact_smooth",name,block,1e-5 if block=="rotation" else 1e-7,Jexact[:,sl],Jnum[:,sl])
            row["status"]="PASS" if row["relative_error"]<1e-6 else "FAIL"
            jacrows.append(row)
    record("T03", "WARN", cases=normal_cases,
           meaning="IC reference-plane null differs from forward target-plane null at relative tilt; not an exact IC-Jacobian failure")
    ref,tar,_=cases["relative_tilt"]
    T=np.linalg.inv(tar)@ref
    d=rng.normal(size=6)*SCALES[:6]
    moved=np.linalg.inv(gtsam.Pose3(tar).retract(d*1e-5).matrix())@ref
    chain_numeric=gtsam.Pose3.Logmap(gtsam.Pose3(np.linalg.inv(T)@moved))/1e-5
    results["T03"]["IC_increment_chain_error"]=float(np.linalg.norm(chain_numeric+nv.adjoint(np.linalg.inv(T))@d))
    Jexact=forward_exact(visual,ref,tar,rows)
    f=lambda D:smooth_forward(visual,ref,gtsam.Pose3(tar).retract(D).matrix(),rows)
    results["T03"]["forward_directional_error"]=float(np.linalg.norm((f(d*1e-5)-f(-d*1e-5))/2e-5-Jexact@d))
    results["T03"]["forward_taylor_errors"]=[dict(step=h,error=float(np.linalg.norm(f(d*h)-f(np.zeros(6))-Jexact@(d*h)))) for h in (1e-2,1e-3,1e-4)]
    for mult in (10., 1., .1):
        eps = np.r_[np.full(3, 1e-5 * mult), np.full(3, 1e-7 * mult), np.full(9, 1e-6)]
        Jic = numeric(lambda z: ic_local(visual, gtsam.Pose3.Logmap(gtsam.Pose3(z.pose)), rows), state(0), eps)[:, :6]
        for block in ("rotation", "translation"):
            sl = BLOCKS[block]
            jacrows.append(error_row("NF_IC_smooth_vs_Sobel", "reference_local", block, float(eps[sl][0]),
                                     Kref[:, sl], Jic[:, sl], "WARN"))
    # Production sampler epsilon sweep; fixed mask and continuous warp kept.
    for eps_t in (1e-8, 1e-6, 4e-5, 1e-4):
        eps = np.r_[np.full(3, 5e-4), np.full(3, eps_t), np.full(9, 1e-6)]
        Jp = numeric(lambda z: visual.residual_on_fixed_rows(np.eye(4), z.pose, rows), state(0), eps)[:, :6]
        for block in ("rotation", "translation"):
            sl = BLOCKS[block]
            jacrows.append(error_row("NF_production_forward_vs_IC", "current", block, float(eps[sl][0]),
                                     -Kref[:, sl], Jp[:, sl], "WARN"))
    # Native IMU both endpoints; covariance is fixed inside this GTSAM factor.
    ext = pose((.3, -.2, .1), (.015, -.012, .03))
    samples, prev, curr, _, _ = analytic_motion(omega=(.3, -.2, .4), a=(.2, -.1, .05), ext=ext)
    curr = plus(curr, rng.normal(size=15) * SCALES * .04)
    pim = gv.preintegrate_combined(samples, prev, config, ext, GRAVITY)
    factor = gv._combined_imu_factor(pim)
    A, b = factor.linearize(gv._imu_values(prev, curr)).jacobian()
    Rwhite = factor.noiseModel().R()
    native_max = 0.
    for endpoint, x, analytic in (("previous", prev, np.column_stack((A[:, :9], A[:, 18:24]))),
                                   ("current", curr, np.column_stack((A[:, 9:18], A[:, 24:30])))):
        def fun(y):
            values = gv._imu_values(y, curr) if endpoint == "previous" else gv._imu_values(prev, y)
            return Rwhite @ factor.unwhitenedError(values)
        for mult in (10., 1., .1):
            eps = np.array([1e-5] * 3 + [1e-6] * 3 + [1e-5] * 3 + [1e-5] * 3 + [1e-6] * 3) * mult
            Jn = numeric(fun, x, eps)
            for block, sl in BLOCKS.items():
                row = error_row("GTSAM_CombinedIMU", endpoint, block, float(eps[sl][0]), analytic[:, sl], Jn[:, sl])
                native_max = max(native_max, row["max_absolute_error"])
                row["status"] = "PASS" if row["relative_error"] < 1e-5 else "FAIL"
                jacrows.append(row)
        d = rng.normal(size=15) * SCALES
        directional = (fun(plus(x, d * 1e-5)) - fun(plus(x, -d * 1e-5))) / 2e-5
        arrays["IMU_" + endpoint] = analytic
        results["directional_" + endpoint] = float(np.linalg.norm(analytic @ d - directional))
    # Native pose and bias priors; these are test-only, not frontend factors.
    for name, f, key in (("pose_prior", gtsam.PriorFactorPose3(1, gtsam.Pose3(prev.pose), gtsam.noiseModel.Isotropic.Sigma(6, .1)), 1),
                          ("bias_prior", gtsam.PriorFactorConstantBias(5, gv._bias(prev), gtsam.noiseModel.Isotropic.Sigma(6, .01)), 5)):
        value = gv._imu_values(prev, curr)
        Aprior, _ = f.linearize(value).jacobian()
        def prior_fun(y):
            return f.noiseModel().R() @ f.unwhitenedError(gv._imu_values(y, curr))
        Jn = numeric(prior_fun, prev, np.full(15, 1e-6))
        sl = slice(0, 6) if key == 1 else slice(9, 15)
        jacrows.append(error_row(name, "previous_test_only", "pose" if key == 1 else "bias", 1e-6, Aprior, Jn[:, sl]))
    reductions = []
    fcur = lambda y: Rwhite @ factor.unwhitenedError(gv._imu_values(prev, y))
    Acur = arrays["IMU_current"]
    d = rng.normal(size=15) * SCALES
    for h in (1e-2, 1e-3, 1e-4):
        reductions.append(dict(step=h, linearization_error=float(np.linalg.norm(fcur(plus(curr, h*d)) - fcur(curr) - Acur @ (h*d)))))
    record("T04", "PASS" if native_max < .02 else "FAIL", native_max_absolute_error=native_max,
           linearization_error=reductions, NF_exact_comparison="WARN: distinct forward and IC objectives; Sobel and cv2 precision recorded in CSV")
    results["T04"]["native_whitening_identity_error"]=float(np.max(np.abs(Rwhite@factor.noiseModel().covariance()@Rwhite.T-np.eye(15))))
    results["T04"]["raw_preintegration_vs_factor_covariance_difference"]=float(np.max(np.abs(pim.preintMeasCov()-factor.noiseModel().covariance())))
    results["T04"]["covariance_off_diagonal_norm"]=float(np.linalg.norm(pim.preintMeasCov()-np.diag(np.diag(pim.preintMeasCov()))))
    results["T04"]["numeric_checks_status"]="PASS" if all(r["status"]!="FAIL" for r in jacrows) else "FAIL"
    # T05: points, inverse, extrinsic lever arm, and actual retraction.
    point = np.array([.01, -.02, .03, 1.])
    WB0, WB1, BS = pose((.2, -.1, .3), (.1, .2, -.1)), pose((-.1, .3, .2), (.11, .17, -.08)), np.linalg.inv(ext)
    WS0, WS1 = WB0 @ BS, WB1 @ BS
    rel = ext @ np.linalg.inv(WB1) @ WB0 @ np.linalg.inv(ext)
    transform_error = np.linalg.norm(rel @ point - np.linalg.inv(WS1) @ WS0 @ point)
    inversion_control = np.linalg.norm(np.linalg.inv(rel) @ point - np.linalg.inv(WS1) @ WS0 @ point)
    increment = np.array([.02, -.01, .03, .001, -.002, .003])
    retract_error = np.linalg.norm(gtsam.Pose3(WB0).retract(increment).matrix() - (WB0 @ gtsam.Pose3.Expmap(increment).matrix()))
    hand_pose=pose((0,0,np.pi/2),(1,2,3))
    hand_error=np.linalg.norm(hand_pose@np.array([1,0,0,1])-np.array([1,3,3,1]))
    record("T05", "PASS" if max(transform_error, retract_error) < 1e-12 else "FAIL",
           transform_error=float(transform_error), inverse_negative_control_error=float(inversion_control),
           retract_error=float(retract_error), hand_point_error=float(hand_error), extrinsic_translation_m=ext[:3, 3].tolist())
    # T06: independent motion, sample-rate convergence, bias correction.
    truth_cases = []
    for name, omega, a, v, E in (("stationary", (0,0,0), (0,0,0), (0,0,0), pose()),
                               ("constant_velocity", (0,0,0), (0,0,0), (.03,-.02,.01), pose()),
                               ("constant_acceleration", (0,0,0), (.2,-.1,.3), (.03,0,0), pose()),
                               ("single_axis", (0,0,.5), (0,0,0), (.03,0,0), pose()),
                               ("multi_axis", (.4,-.3,.5), (.2,-.1,0), (.03,0,0), pose()),
                               ("lever_arm", (.4,-.3,.5), (.2,-.1,0), (.03,0,0), ext)):
        for dt in (.01, .005, .0025):
            sm, pi, pj, pii, pji = analytic_motion(dt=dt, omega=omega, a=a, v=v, ext=E)
            pp = gv.preintegrate_combined(sm, pi, config, E, GRAVITY)
            fac = gv._combined_imu_factor(pp)
            raw = fac.unwhitenedError(gv._imu_values(pi, pj))
            raw_wrong_v = fac.unwhitenedError(gv._imu_values(pii, pji))
            truth_cases.append(dict(case=name, dt=dt, native_raw_norm=float(np.linalg.norm(raw)),
                                    native_with_IMU_origin_velocity=float(np.linalg.norm(raw_wrong_v))))
    sm, pi, pj, _, _ = analytic_motion(omega=(.4,-.3,.5))
    base = gv.preintegrate_combined(sm, pi, config, pose(), GRAVITY)
    corrections = []
    for h in (1e-2, 1e-3, 1e-4):
        biased = pi.copy(); biased.accel_bias += h * np.array([1,-.5,.3]); biased.gyro_bias += h * np.array([.3,-.2,.1])
        fresh = gv.preintegrate_combined(sm, biased, config, pose(), GRAVITY)
        old_prediction = base.predict(gtsam.NavState(gtsam.Pose3(pi.pose), pi.velocity), gv._bias(biased))
        new_prediction = fresh.predict(gtsam.NavState(gtsam.Pose3(pi.pose), pi.velocity), gv._bias(biased))
        corrections.append(dict(bias_delta=h, pose_error=float(np.linalg.norm(old_prediction.pose().localCoordinates(new_prediction.pose()))),
                                velocity_error=float(np.linalg.norm(old_prediction.velocity()-new_prediction.velocity()))))
    record("T06", "PASS", trajectories=truth_cases, bias_correction=corrections,
           velocity_contract="native: Gel origin; no-contact: IMU origin")
    angular_acceleration=[]
    for dt in (.01,.005,.0025,.00125):
        for label,E in (("zero_arm",pose()),("nonzero_arm",ext)):
            ss,pi,pj,pii,pji=analytic_motion(dt=dt,omega=(.4,-.3,.5),alpha=(.8,-.6,1.),ext=E)
            pp=gv.preintegrate_combined(ss,pi,config,E,GRAVITY)
            raw=gv._combined_imu_factor(pp).unwhitenedError(gv._imu_values(pi,pj))
            angular_acceleration.append(dict(case=label,dt=dt,native_rotation_error=float(np.linalg.norm(raw[:3])),
                native_position_error=float(np.linalg.norm(raw[3:6])),native_velocity_error=float(np.linalg.norm(raw[6:9]))))
    record("T06_angular_acceleration","WARN",cases=angular_acceleration,
           meaning="bodyPSensor compensates omega x (omega x lever) but omits alpha x lever")
    # T07: active connectivity and conditional rank; preserve both endpoints.
    active_pim = gv.preintegrate_imu(samples, prev, config, ext, GRAVITY)
    active_factors = (gv._split_imu_factor(active_pim), gv._bias_factor(active_pim, config))
    active_rows = [gv.navigation_factor_terms(f, prev, curr) for f in active_factors]
    active_A = np.vstack([np.column_stack(row[:2]) for row in active_rows])
    blocks, rhs = gv.inertial_current_terms(active_pim, prev, curr, config)
    Icur = np.column_stack((blocks[gv.POSE_J], blocks[gv.VELOCITY_J], blocks[gv.BIAS_J]))
    Nfull = np.zeros((Kref.shape[0], 15)); Nfull[:, :6] = -Kref
    stacked = np.vstack((Nfull, Icur))
    production_graph = gtsam.GaussianFactorGraph()
    nA, nr, _, _ = gv._normal_huber_linearization(
        np.zeros(len(Nfull)), Nfull[:,:6],
        np.sqrt(config.get("dense_normalflow_residual_weight",1.))/config.get("dense_normalflow_normal_sigma",1.),
        config.get("gtsam_vi_normal_huber_delta",.4))
    production_graph.add(gtsam.JacobianFactor(gv.POSE_J,nA,-nr,gtsam.noiseModel.Isotropic.Sigma(len(nr),1)))
    iw=np.sqrt(config.get("imu_residual_weight",1.))
    production_graph.add(gtsam.JacobianFactor(gv.POSE_J,blocks[gv.POSE_J][:9]*iw,
        gv.VELOCITY_J,blocks[gv.VELOCITY_J][:9]*iw,
        rhs[:9]*iw,gtsam.noiseModel.Unit.Create(9)))
    production_graph.add(gtsam.JacobianFactor(gv.BIAS_J,blocks[gv.BIAS_J][9:]*iw,
        rhs[9:]*iw,gtsam.noiseModel.Unit.Create(6)))
    pgA, pgb=production_graph.jacobian()
    expected_J=np.vstack((np.column_stack((nA,np.zeros((len(nA),9)))),Icur*iw))
    expected_r=np.r_[nr,-rhs*iw]
    production_assembly_error=max(np.max(np.abs(pgA.T@pgA-expected_J.T@expected_J)),
                                  np.max(np.abs(-pgA.T@pgb-expected_J.T@expected_r)))
    results["T01"]["native_and_normal_production_assembly_error"]=float(production_assembly_error)
    Hjoint = stacked.T @ stacked
    Hpp, Hpn, Hnn = Hjoint[:6,:6], Hjoint[:6,6:], Hjoint[6:,6:]
    marginal = Hpp - Hpn @ np.linalg.solve(Hnn, Hpn.T)
    eig = np.linalg.eigvalsh((marginal + marginal.T) / 2)
    arrays.update(NF_full=Nfull, IMU_current=Icur, stack=stacked, H_before_damping=Hjoint,
                  solver_matrix=Hjoint + np.eye(15)*float(config.get("gtsam_vi_damping",0)), pose_marginal=marginal)
    missing_bias = Icur.copy(); missing_bias[:,9:] = 0
    record("T07", "PASS", current=spectrum(Icur, SCALES), joint=spectrum(stacked, SCALES),
           full_endpoint=spectrum(active_A, np.r_[SCALES, SCALES]),
           motion_residual_dim=9, bias_residual_dim=6,
           missing_bias_negative_control=spectrum(np.vstack((Nfull, missing_bias)), SCALES),
           pose_marginal_eigenvalues= eig.tolist(), note="conditioned on exact previous state; no propagated prior")
    # T08: algebra only. Active frontend/backend do not marginalize.
    M = rng.normal(size=(25,9)); rr = rng.normal(size=25)
    HH, gg = M.T@M, M.T@rr
    Schur = HH[3:,3:] - HH[3:,:3] @ np.linalg.solve(HH[:3,:3], HH[:3,3:])
    gm = gg[3:] - HH[3:,:3] @ np.linalg.solve(HH[:3,:3], gg[:3])
    diff = np.max(np.abs(-np.linalg.solve(Schur,gm) + np.linalg.solve(HH,gg)[3:]))
    record("T08", "PASS" if diff<1e-12 else "FAIL", illustrative_Schur_error=float(diff), production_marginalization="SKIP: absent")
    ranks = []
    for kind in ("plane", "sphere", "asymmetric"):
        vis = visual_fixture(kind); _, Kr, _ = vis.evaluate(np.eye(4),np.eye(4))
        fixed=vis.last_active_rows.copy()
        def exact_shape(z):
            T=z.pose;P=vis.candidate_points[fixed]@T[:3,:3].T+T[:3,3]
            return (normal_field(P[:,:2],kind)@T[:3,:3]-vis.candidate_normals[fixed]).reshape(-1)
        Js=numeric(exact_shape,state(0),np.r_[np.full(3,1e-5),np.full(3,1e-7),np.full(9,1e-6)])[:,:6]
        ranks.append(dict(shape=kind, production=spectrum(Kr,SCALES[:6]), smooth=spectrum(Js,SCALES[:6])))
        arrays["NF_shape_"+kind] = Kr
    record("T09", "PASS" if [r["smooth"]["rank"] for r in ranks]==[2,4,5] else "FAIL", shapes=ranks, note="sphere uses flat point support; normal symmetry test, not a full sphere surface")
    # T10: same inputs and oracle rank-5 measurement for real native solver.
    class OracleVisual:
        def __len__(self): return 10
        def evaluate(self, reference, target):
            fun=lambda y: np.tile(gtsam.Pose3(np.eye(4)).localCoordinates(gtsam.Pose3(y.pose))[:5],6)
            rr=fun(state(.2,target))
            JJ=numeric(fun,state(.2,target),np.full(15,1e-6))[:,:6]
            return rr, -JJ, JJ
    sm, p0, truth, _, _ = analytic_motion(v=(0,0,0), ba=(0,0,0), bg=(0,0,0))
    # Set a stationary level anchor for this oracle experiment.
    p0.pose = np.eye(4); truth.pose = np.eye(4)
    sm = [ImuSample(float(t), -GRAVITY, np.zeros(3)) for t in np.linspace(0,.2,41)]
    initial = plus(truth, np.array([.003,-.002,.001,.001,-.001,.002,.01,-.02,.03,.001,-.002,.003,.0001,-.0002,.0003]))
    solver = gv.GtsamVisualInertialSolver(dict(config, frontend_gravity_mps2=GRAVITY.tolist(),
           gtsam_vi_rotation_stop_rad=1e-9, gtsam_vi_translation_stop_m=1e-9, gtsam_vi_max_iterations=20))
    result = solver.refine_current(OracleVisual(), np.eye(4), p0, initial, sm, np.eye(4))
    delta_truth = np.linalg.norm(gtsam.Pose3(truth.pose).localCoordinates(gtsam.Pose3(result.pose)))
    record("T10", "PASS" if delta_truth<1e-7 else "FAIL", pose_error=float(delta_truth),
           velocity_error=float(np.linalg.norm(result.velocity)), bias_error=float(np.linalg.norm(np.r_[result.accel_bias,result.gyro_bias])),
           traces=solver.last_optimization_trace, previous_state="fixed exact; current biases estimated with combined factor")
    # A-D/E/G/H ablations share a stationary truth and one initial state.
    # This is a model consistency test, not a dataset accuracy comparison.
    class NoVisual:
        def __len__(self):return 10
        def evaluate(self,_reference,_target):return np.zeros(30),np.zeros((30,6)),np.zeros((30,6))
    class SmoothExactVisual:
        def __init__(self,vis):self.vis=vis;self.rows=vis.last_active_rows.copy();self.active_count=int(self.rows.sum())
        def __len__(self):return self.active_count
        def evaluate(self,ref,tar):
            J=forward_exact(self.vis,ref,tar,self.rows)
            return smooth_forward(self.vis,ref,tar,self.rows),-J,J
    actual=visual_fixture();actual.evaluate(np.eye(4),np.eye(4))
    ablations=[]
    for label,V in (("B_IMU_only",NoVisual()),("D_E_H_joint_bias_free_fixed_previous_oracle",OracleVisual()),
                    ("G_production_IC",actual),("G_exact_smooth",SmoothExactVisual(actual))):
        final=solver.refine_current(V,np.eye(4),p0,initial,sm,np.eye(4))
        ablations.append(dict(case=label,returned=final is not None,
            pose_error=None if final is None else float(np.linalg.norm(gtsam.Pose3(np.eye(4)).localCoordinates(gtsam.Pose3(final.pose)))),
            velocity_error=None if final is None else float(np.linalg.norm(final.velocity)),
            bias_error=None if final is None else float(np.linalg.norm(np.r_[final.accel_bias,final.gyro_bias])),
            accepted_steps=sum(t["accepted_scale"] is not None for t in solver.last_optimization_trace)))
    # A: NF-only linear oracle at identity; hold its missing component at the seed.
    z0=gtsam.Pose3(np.eye(4)).localCoordinates(gtsam.Pose3(initial.pose))
    oracleJ=np.eye(6)[:5]
    dA=np.linalg.lstsq(oracleJ,-z0[:5],rcond=None)[0]
    ablations.append(dict(case="A_NF_only_oracle_linear",observed_linear_error=float(np.linalg.norm((z0+dA)[:5])),
        missing_z_at_seed=float(z0[5]),note="missing component held; no demand to recover it"))
    # C: the native production factors with current bias fixed at known truth.
    known_initial=initial.copy();known_initial.accel_bias=p0.accel_bias.copy();known_initial.gyro_bias=p0.gyro_bias.copy()
    pp=gv.preintegrate_combined(sm,p0,config,np.eye(4),GRAVITY)
    known=known_initial.copy()
    for _ in range(8):
        nr,_,nj=OracleVisual().evaluate(np.eye(4),known.pose)
        ib,ir=gv.combined_imu_current_terms(pp,p0,known)
        JJ=np.vstack((np.column_stack((nj,np.zeros((len(nj),3)))),np.column_stack((ib[gv.POSE_J],ib[gv.VELOCITY_J]))))
        rr=np.r_[nr,-ir]
        step=np.linalg.lstsq(JJ,-rr,rcond=None)[0]
        known=plus(known,np.r_[step,np.zeros(6)])
    ablations.append(dict(case="C_joint_known_bias",pose_error=float(np.linalg.norm(gtsam.Pose3(np.eye(4)).localCoordinates(gtsam.Pose3(known.pose)))),
        velocity_error=float(np.linalg.norm(known.velocity)),bias_error=0.))
    record("ablations","PASS",cases=ablations,window_F="SKIP: no active uncertainty-bearing frontend window",
        limits="A is linear oracle only. Production/smooth samplers differ. Same truth and initial state; no performance ordering imposed.")
    # T11: errors cannot be repaired by absent visual-Z information alone.
    sensitivity=[]
    for name in ("velocity", "accel_bias", "gyro_bias", "gravity_tilt"):
        p = p0.copy(); gravity = GRAVITY.copy()
        if name == "velocity": p.velocity[2]=.05
        if name == "accel_bias": p.accel_bias[2]=.1
        if name == "gyro_bias": p.gyro_bias[0]=.01
        if name == "gravity_tilt": gravity=Rotation.from_rotvec([0,.01,0]).apply(GRAVITY)
        pred = gv.preintegrate_combined(sm,p,config,np.eye(4),gravity).predict(gtsam.NavState(gtsam.Pose3(p.pose),p.velocity),gv._bias(p))
        sensitivity.append(dict(injected=name, position_error_m=pred.position().tolist(),
                                velocity_error_mps=pred.velocity().tolist(),
                                rotation_error_rad=Rotation.from_matrix(pred.pose().rotation().matrix()).as_rotvec().tolist()))
    record("T11", "WARN", cases=sensitivity, note="fixed previous state does not carry its uncertainty")
    # T12: real contact -> no-contact bridge. Fused bias should persist if it is the same estimator.
    try:
        from gelslam.core.tracker import Tracker
        tr = object.__new__(Tracker)
        tr.imu_config = dict(gravity_mps2=9.80665, gravity_initialization="fixed_world_z",
                             initial_accel_bias_mps2=[0,0,0], initial_gyro_bias_rps=[0,0,0])
        tr.gelsight_T_imu=np.eye(4); tr.imu_buffer=ImuBuffer()
        for t in (1.,1.1,1.2): tr.imu_buffer.add(ImuSample(t,[.1,0,9.80665],[.002,0,0]))
        hdr=lambda t: SimpleNamespace(stamp=SimpleNamespace(sec=int(t),nanosec=int(round((t-int(t))*1e9))))
        tr.trial_T_ref=np.eye(4); tr.frontend_visual_inertial=gv.GtsamVisualInertialSolver({})
        tr._frontend_vi_state=state(1.,v=[.2,0,0],ba=[.1,0,0],bg=[.002,0,0])
        tr._record_tactile_pose(np.eye(4),hdr(1.))
        anchor_ok=tr._initialize_no_contact_coast()
        coast_bias=tr.inertial_state.accel_bias.copy()
        coast_velocity=tr.inertial_state.velocity.copy()
        fused_bias=tr._frontend_vi_state.accel_bias.copy()
        coast=tr.inertial_propagator.propagate(tr.inertial_state,tr.imu_buffer.samples_covering(1.,1.2))
        record("T12", "FAIL" if np.linalg.norm(coast_bias-fused_bias)>.01 else "PASS", anchor_ok=anchor_ok,
               fused_accel_bias=fused_bias.tolist(), coast_accel_bias=coast_bias.tolist(),
               fused_velocity=tr._frontend_vi_state.velocity.tolist(), coast_velocity=coast_velocity.tolist(),
               coast_position_at_1p2=coast.position.tolist(),expected_position_at_1p2=[.04,0,0],
               velocity_policy_status="WARN" if np.linalg.norm(coast_velocity-tr._frontend_vi_state.velocity)>.01 else "PASS",
               reason=("frontend bias preserved at matching anchor; velocity still uses the tactile history policy"
                       if np.linalg.norm(coast_bias-fused_bias)<=.01
                       else "no-contact initializer uses config bias and tactile difference velocity"))
    except ImportError as exc:
        record("T12","SKIP",reason=str(exc))
    # Weak contact can have no selected curvature rows despite a large mask.
    flat=visual_fixture("plane").ref
    flat.L=np.zeros_like(flat.L)
    try:
        empty=nv.DenseNormalResidual(flat,flat,1.,np.eye(4),np.eye(4),1400)
        record("T12_empty_support","PASS",selected_rows=len(empty))
    except ValueError as exc:
        record("T12_empty_support","FAIL",exception=str(exc),
               contact_pixels=int(flat.C.sum()),selected_curvature_rows=0)
    # Negative controls: wrong world-Z removal and residual/J sign reversal.
    tilted=normal_cases[2]
    Jw=arrays["NF_forward_relative_tilt"] @ nv.adjoint(np.linalg.inv(cases["relative_tilt"][1]))
    Jwrong=Jw.copy(); Jwrong[:,5]=0
    actual_world_null=tilted["world_null"]
    Nw=np.r_[np.zeros(3),actual_world_null]
    wrong_null=float(np.linalg.norm(Jwrong@Nw))
    signJ=np.eye(2); signr=np.array([.3,-.2]); wrongstep=signr.copy()
    record("negative_controls", "PASS" if wrong_null>1 and inversion_control>1e-3 else "FAIL",
           wrong_world_z_null_error=wrong_null, wrong_inverse_point_error=float(inversion_control),
           missing_bias_rank=results["T07"]["missing_bias_negative_control"]["rank"],
           sign_error_cost_before=float(.5*signr@signr), sign_error_cost_after=float(.5*(signr+signJ@wrongstep)@(signr+signJ@wrongstep)))
    # Record evidence version. No claim that old logs used today's settings.
    tracked = [ROOT/"src/gelslam/gelslam/core"/name for name in ("gtsam_vi.py","numpy_vi.py","tracker.py","inertial.py","gtsam_pose_graph.py")]
    tracked += [Path(config_path),Path(nf.__file__)]
    provenance=dict(seed=SEED, python=sys.executable, numpy=np.__version__, cv2=cv2.__version__,
                    gtsam=importlib.metadata.version("gtsam"),normalflow_path=nf.__file__,
                    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                    branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip(),
                    config_path=str(Path(config_path).resolve()), config=config,
                    hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked},
                    variable_scaling=SCALES.tolist(), rank_threshold="max(1e-10, s_max*1e-8)")
    (output/"results.json").write_text(json.dumps(dict(provenance=provenance,tests=results),indent=2,allow_nan=False)+"\n")
    with (output/"jacobians.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(jacrows[0]));writer.writeheader();writer.writerows(jacrows)
    np.savez_compressed(output/"matrices.npz",**arrays)
    print(json.dumps({key:value["status"] for key,value in results.items() if isinstance(value,dict) and "status" in value}))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"diagnostics/tactile_imu_audit_20261013")
    parser.add_argument("--config",type=Path,default=ROOT/"src/gelslam/config/real_config.yaml")
    args=parser.parse_args()
    run_audit(args.output,args.config)


if __name__=="__main__":
    main()
