"""Diagnostics and regression checks for the applied tactile/IMU fixes."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"scripts"))
from audit_tactile_imu import state, visual_fixture
from gelslam.core.numpy_vi import DenseNormalResidual
from gelslam.core.tracker import Tracker
from gelslam.core.inertial import ImuBuffer, ImuSample, InertialState
from gelslam.core.gtsam_vi import GtsamVisualInertialSolver


# Removed 2026-11-05: the audit fixture tests (T01-T12, negative controls, IC/forward
# null directions) ran scripts/audit_tactile_imu.py, which needs the pairwise
# GtsamVisualInertialSolver.refine_current. The working tree removed the pairwise
# frontend (frontend_state_mode); reference marginalization is the only frontend.
# The two legacy no-contact inertial_state bias tests were removed with that path.


def build_bridge(cls=Tracker,frontend_time=1.,initialize=True):
    tr=object.__new__(cls)
    tr.imu_config=dict(gravity_mps2=9.80665,gravity_initialization="fixed_world_z",
        initial_accel_bias_mps2=[0,0,0],initial_gyro_bias_rps=[0,0,0])
    tr.gelsight_T_imu=np.eye(4);tr.imu_buffer=ImuBuffer()
    tr.imu_buffer.add(ImuSample(1.,[.1,0,9.80665],[0,0,0]))
    tr.imu_buffer.add(ImuSample(1.2,[.1,0,9.80665],[0,0,0]))
    tr.trial_T_ref=np.eye(4);tr.frontend_visual_inertial=GtsamVisualInertialSolver({})
    tr._frontend_vi_state=state(1.,v=[.2,0,0],ba=[.1,0,0],bg=[.002,0,0])
    header=SimpleNamespace(stamp=SimpleNamespace(sec=1,nanosec=0))
    tr._record_tactile_pose(np.eye(4),header)
    if frontend_time is None:
        tr._frontend_vi_state=None
    else:
        tr._frontend_vi_state.timestamp=frontend_time
    if initialize:
        tr._initialize_no_contact_coast()
    return tr


def test_empty_curvature_support_is_safe():
    flat=visual_fixture("plane").ref;flat.L=np.zeros_like(flat.L)
    visual=DenseNormalResidual(flat,flat,1.,np.eye(4),np.eye(4),1400)
    assert len(visual)==0
    assert visual.evaluate(np.eye(4),np.eye(4)) is None


def test_no_contact_keeps_existing_coast_bias_before_frontend_bias():
    tr=build_bridge(initialize=False)
    tr.inertial_state=InertialState(1.,np.eye(3),np.zeros(3),np.zeros(3),np.array([.3,.2,.1]),np.array([.01,.02,.03]))
    assert tr._initialize_no_contact_coast()
    assert np.allclose(tr.inertial_state.accel_bias,[.3,.2,.1])
    assert np.allclose(tr.inertial_state.gyro_bias,[.01,.02,.03])


@pytest.mark.parametrize("selected_count",[1,9])
def test_small_curvature_support_is_inactive(selected_count):
    flat=visual_fixture("plane").ref;flat.L=np.zeros_like(flat.L)
    flat.L[25,25:25+selected_count]=np.arange(1,selected_count+1)
    vis=DenseNormalResidual(flat,flat,1.,np.eye(4),np.eye(4),1400)
    assert len(vis.candidate_points)==selected_count
    assert len(vis)==0
    assert vis.evaluate(np.eye(4),np.eye(4)) is None
