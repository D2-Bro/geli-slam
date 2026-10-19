"""The diagnostic observer must not change a production VI solve."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from types import SimpleNamespace

from gelslam.core.gtsam_vi import GtsamVisualInertialSolver
from gelslam.core.inertial import ImuSample
from gelslam.core.numpy_vi import NavigationState


spec = importlib.util.spec_from_file_location(
    'five_observation_diagnostic',
    Path(__file__).resolve().parents[3]/'scripts/diagnose_five_observations.py')
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)
render_spec = importlib.util.spec_from_file_location(
    'render_five_observation_diagnostic',
    Path(__file__).resolve().parents[3]/'scripts/render_five_observations.py')
renderer = importlib.util.module_from_spec(render_spec)
render_spec.loader.exec_module(renderer)


class LinearVisual:
    def __len__(self):
        return 10

    def evaluate(self, reference, target):
        self.last_active_rows = np.ones(10,dtype=bool)
        residual = np.tile(target[:3,3]-np.array([.0001,0.,0.]),10)
        jacobian = np.tile(np.c_[np.zeros((3,3)),target[:3,:3]],(10,1))
        return residual,np.zeros((30,6)),jacobian

    def residual_on_fixed_rows(self, reference, target, rows):
        return np.tile(target[:3,3]-np.array([.0001,0.,0.]),10)


def test_observer_records_real_steps_without_changing_solve():
    previous = NavigationState(1.,np.eye(4),np.zeros(3),np.zeros(3),np.zeros(3))
    samples = [ImuSample(t,[0.,0.,9.81],[0.,0.,0.]) for t in (1.,1.05)]
    outputs=[];traces=[];observer=diagnostic.IterationObserver()
    for observed in (False,True):
        solver=GtsamVisualInertialSolver({'gtsam_vi_max_iterations':3})
        current,pim=solver.predict_current(previous,samples,np.eye(4))
        function=lambda:solver.refine_current(LinearVisual(),np.eye(4),previous,current,samples,
                                             np.eye(4),preintegrated=pim)
        outputs.append(observer.run(function) if observed else function())
        traces.append(solver.last_optimization_trace)
    assert traces[0]==traces[1]
    assert outputs[0] is not None
    for attr in ('pose','velocity','accel_bias','gyro_bias'):
        assert np.array_equal(getattr(outputs[0],attr),getattr(outputs[1],attr))
    assert observer.iterations
    record=observer.iterations[0]
    assert record['trace']['accepted_scale'] is not None
    assert record['candidates']
    accepted=next(c for c in record['candidates'] if c['scale']==record['trace']['accepted_scale'])
    assert np.array_equal(record['after_current'].pose,accepted['current'].pose)
    assert np.array_equal(record['after_previous'].pose,accepted['previous'].pose)


def test_observer_records_full_gn_with_target_jacobian_from_first_step():
    class WrongDirection(LinearVisual):
        def evaluate(self, reference, target):
            residual, reference_j, target_j = super().evaluate(reference, target)
            return residual, reference_j, -target_j

        def evaluate_relinearized(self, reference, target):
            return super().evaluate(reference, target)

    previous = NavigationState(1.,np.eye(4),np.zeros(3),np.zeros(3),np.zeros(3))
    samples = [ImuSample(t,[0.,0.,9.81],[0.,0.,0.]) for t in (1.,1.05)]
    outputs=[];traces=[];observer=diagnostic.IterationObserver()
    for observed in (False,True):
        solver=GtsamVisualInertialSolver({'gtsam_vi_max_iterations':1,
            'gtsam_vi_normal_linearization':'target_relinearized',
            'dense_normalflow_residual_weight':1e8,'imu_residual_weight':1e-8})
        current,pim=solver.predict_current(previous,samples,np.eye(4))
        function=lambda:solver.refine_current(WrongDirection(),np.eye(4),previous,current,
                                              samples,np.eye(4),preintegrated=pim)
        outputs.append(observer.run(function) if observed else function())
        traces.append(solver.last_optimization_trace)
    assert traces[0]==traces[1]
    assert np.array_equal(outputs[0].pose,outputs[1].pose)
    first,=observer.iterations
    assert first['iteration']==0
    assert first['trace']['normal_jacobian_source']=='target_normal_map'
    assert first['trace']['accepted_scale']==1.
    assert len(first['candidates'])==1
    assert np.array_equal(first['after_current'].pose,outputs[0].pose)


@pytest.mark.parametrize('all_images',[False,True])
def test_whole_bag_processes_startup_but_table_keeps_failed_contact_input(monkeypatch,tmp_path,all_images):
    class Tracker:
        def __init__(self,*args):
            self._frontend_vi_state=None
            self.gelsight_T_imu=np.eye(4)
            self.ref_header=None
            self.tracking_mode=SimpleNamespace(value='startup')
            self.latest_gt_state_test=None
        def _imu_timestamp(self,header):
            return None if header is None else float(header.stamp.sec)
        def track(self,image,header):
            t=self._imu_timestamp(header)
            if t<3:
                return False,None
            if self.ref_header is None:
                self.ref_header=header
            self.tracking_mode.value='contact'
            if t==4:  # A rejected target must remain visible in the table.
                return False,None
            self._frontend_vi_state=NavigationState(t,np.eye(4),np.zeros(3),np.zeros(3),np.zeros(3))
            return True,(object(),[])
    monkeypatch.setattr(diagnostic.tm,'Tracker',Tracker)
    monkeypatch.setattr(diagnostic,'CvBridge',lambda:SimpleNamespace(imgmsg_to_cv2=lambda *a:np.zeros((2,2,3))))
    images=[(t,b'',SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=t,nanosec=0))))
            for t in range(1,6)]
    truth=SimpleNamespace(transform=None,states=SimpleNamespace(at=lambda t:(np.eye(4),np.zeros(3))),
                          error=lambda *args:None)
    result=diagnostic.replay({'/image':images},{'device_config':{'calibration_model_path':'unused'}},
                             tmp_path,truth,False,1.,2,all_images)
    assert result['processed']==(5 if all_images else 4)
    assert len(result['events'])==(5 if all_images else 2)
    shown=renderer.contact_events(result['events'])
    assert [e['timestamp'] for e in shown]==([3.,4.,5.] if all_images else [3.,4.])
    assert shown[1]['valid'] is False
    assert shown[1]['after']['timestamp']==3.
