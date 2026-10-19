"""GT input intervention while retaining the production predictor/solve path."""

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from gelslam.core.gt_state_diagnostic import GroundTruthImuStates, gt_state_test_needs_gt, resolve_gt_state_test
from gelslam.core.gtsam_vi import GtsamVisualInertialSolver
from gelslam.core.imu_navigation import gelsight_pose
from gelslam.core.inertial import ImuBuffer, ImuSample
from gelslam.core.marginal_prior import NavigationPrior
from gelslam.core.numpy_vi import NavigationState
from gelslam.core.tracker import Tracker
from gelslam.nodes.tracker_node import TrackerNode
from geometry_msgs.msg import PoseStamped
import gelslam.core.tracker as tracker_module


def header(t):
    return SimpleNamespace(stamp=SimpleNamespace(sec=int(t), nanosec=round((t-int(t))*1e9)))


def test_gt_interpolation_matches_native_nonuniform_central_differences():
    states = GroundTruthImuStates()
    times = np.array([.9, .97, 1.02, 1.13, 1.2])
    positions = np.array([[.2*t*t + .1*t, .03*t, 0.] for t in times])
    for t, p in zip(times, positions):
        assert states.add(t, Rotation.from_euler('z', .4*t).as_matrix(), p)
    pose, velocity = states.at(1.)
    native = np.gradient(positions, times, axis=0, edge_order=2)
    expected = [np.interp(1., times, native[:, k]) for k in range(3)]
    assert np.allclose(velocity, expected, atol=1e-12)
    assert np.allclose(pose[:3, :3], Rotation.from_euler('z', .4).as_matrix())
    assert states.at(.8) is None and states.at(1.3) is None
    assert states.at(.9) is None  # No past sample for a central derivative.


def fixture_tracker(enabled):
    tracker = object.__new__(Tracker)
    tracker.imu_config = {'frontend_gt_previous_state_test': enabled}
    tracker.logger = SimpleNamespace(info=lambda _: None, warning=lambda _: None)
    tracker.gelsight_T_imu = np.eye(4)
    tracker.gelsight_T_imu[:3, 3] = [.02, -.01, .03]
    tracker.trial_T_ref = np.eye(4)
    tracker.trial_T_ref[:3, 3] = [.4, -.2, .1]  # Deliberately wrong visual reference.
    tracker.ref_frame, tracker.prev_frame = object(), object()
    tracker.ref_header, tracker.prev_header = header(.95), header(1.)
    tracker.ppmm = .059
    tracker.track_config = {'scr_threshold': .3, 'ccs_threshold': .7}
    tracker._record_normal_correspondences = lambda *args, **kwargs: None
    tracker.frontend_visual_inertial = GtsamVisualInertialSolver(tracker.imu_config)
    tracker._frontend_vi_state = NavigationState(1., np.eye(4), np.array([2., 3., 4.]),
                                                np.array([.001, -.002, .003]), np.zeros(3))
    tracker._frontend_vi_state.pose[:3, 3] = [3., 2., 1.]
    solver = tracker.frontend_visual_inertial
    solver.marginal_prior = NavigationPrior.initial(tracker._frontend_vi_state, solver.prior_sigmas)
    tracker._gt_test_states = GroundTruthImuStates()
    for t in np.round(np.arange(.9, 1.201, .01), 2):
        tracker._gt_test_states.add(t, np.eye(3), [.1*t, .2*t, .3*t])
    tracker._gt_test_initial_state = NavigationState(.95, tracker.gelsight_T_imu.copy(),
                                                    np.zeros(3), np.zeros(3), np.zeros(3))
    tracker._gt_test_navigation_T_world = None
    tracker._gt_test_previous_image_timestamp = 1.05  # The 1.05 image was rejected.
    tracker.imu_buffer = ImuBuffer()
    for t in np.round(np.arange(.99, 1.111, .01), 2):
        tracker.imu_buffer.add(ImuSample(t, [0., 0., 9.81], [0., 0., 0.]))
    return tracker


@pytest.mark.parametrize('enabled', [False, True])
def test_existing_predictor_receives_gt_previous_and_visual_reference_only_when_enabled(monkeypatch, enabled):
    tracker = fixture_tracker(enabled)
    solver = tracker.frontend_visual_inertial
    original = tracker._frontend_vi_state.copy()
    prior = solver.marginal_prior
    seen = {}

    class Visual:
        def __init__(self, _ref, _target, _ppmm, reference, initial, **kwargs):
            seen['reference'] = reference.copy()

    def refine(visual, reference, previous, current, samples, extrinsic, preintegrated=None):
        seen.update(previous=previous.copy(), current=current.copy(), samples=samples,
                    prior=solver.marginal_prior, pim=preintegrated)
        return current

    monkeypatch.setattr(tracker_module, 'DenseNormalResidual', Visual)
    monkeypatch.setattr(tracker_module, 'normalflow_quality', lambda *a: (1., 1.))
    monkeypatch.setattr(solver, 'refine_current', refine)
    requested = []
    original_at = tracker._gt_test_states.at
    def at(t):
        requested.append(t)
        assert t < 1.1  # Target GT must never enter initialization or optimization.
        return original_at(t)
    tracker._gt_test_states.at = at
    relative = tracker._refine_normalflow_with_imu(
        np.eye(4), header(1.1), tracker.ref_frame, object(),
    )
    previous = seen['previous']
    assert seen['pim'] is not None
    assert np.array_equal(previous.accel_bias, original.accel_bias)
    if enabled:
        assert previous.timestamp == 1.05
        assert np.allclose(previous.velocity, [.1, .2, .3])
        assert np.allclose(previous.pose[:3, 3], tracker.gelsight_T_imu[:3, 3] + [.01, .02, .03])
        assert np.allclose(seen['reference'], np.eye(4))
        assert seen['prior'] is None
        assert seen['samples'][0].timestamp == 1.05
    else:
        assert previous.timestamp == original.timestamp
        assert np.array_equal(previous.pose, original.pose)
        assert np.array_equal(previous.velocity, original.velocity)
        assert np.array_equal(seen['reference'], tracker.trial_T_ref)
        assert seen['prior'] is prior and not requested
    # Returned relative pose must still use the caller's stored reference basis.
    assert np.allclose(tracker.trial_T_ref @ relative,
                       gelsight_pose(seen['current'].pose, tracker.gelsight_T_imu), atol=1e-6)


def test_missing_diagnostic_gt_stops_instead_of_silently_using_estimated_state():
    tracker = fixture_tracker(True)
    tracker._gt_test_states = GroundTruthImuStates()
    with pytest.raises(RuntimeError, match='bracketing'):
        tracker._gt_test_inputs(tracker._frontend_vi_state, 1.05, .95)


def tracker_after_gt_anchor_eviction():
    tracker = fixture_tracker(dict(enabled=True, reference_pose_from_gt=False,
                                   previous_state_time='accepted_state'))
    tracker._gt_test_inputs(tracker._frontend_vi_state, 1., .95, tracker.trial_T_ref)
    tracker._gt_test_states.max_samples = 8
    for t in np.round(np.arange(1.21, 1.401, .01), 2):
        tracker._gt_test_states.add(t, np.eye(3), [.1*t, .2*t, .3*t])
    assert tracker._gt_test_states.at(.95) is None
    assert len(tracker._gt_test_states.samples) == 8
    return tracker


def test_cached_trial_transform_survives_gt_anchor_eviction():
    tracker = tracker_after_gt_anchor_eviction()
    transform = tracker._gt_test_navigation_T_world.copy()
    previous = tracker._frontend_vi_state.copy()
    previous.timestamp = 1.36
    corrected, reference = tracker._gt_test_inputs(previous, 1.36, .95, tracker.trial_T_ref)
    gt_pose, gt_velocity = tracker._gt_test_states.at(1.36)
    assert np.allclose(corrected.pose, transform @ gt_pose)
    assert np.allclose(corrected.velocity, transform[:3, :3] @ gt_velocity)
    assert np.array_equal(tracker._gt_test_navigation_T_world, transform)
    assert np.array_equal(reference, tracker.trial_T_ref)


def test_cached_transform_still_requires_current_gt_coverage():
    tracker = tracker_after_gt_anchor_eviction()
    with pytest.raises(RuntimeError, match='bracketing.*previous'):
        tracker._gt_test_inputs(tracker._frontend_vi_state, 1., .95, tracker.trial_T_ref)


def test_new_trial_resets_cached_transform_after_old_anchor_eviction():
    tracker = tracker_after_gt_anchor_eviction()
    old_transform = tracker._gt_test_navigation_T_world.copy()
    tracker._frontend_vi_state.timestamp = 1.36
    tracker._set_gt_test_anchor()
    assert tracker._gt_test_navigation_T_world is None
    corrected, _ = tracker._gt_test_inputs(tracker._frontend_vi_state, 1.37, .95, tracker.trial_T_ref)
    new_transform = (tracker._gt_test_initial_state.pose
                     @ np.linalg.inv(tracker._gt_test_states.at(1.36)[0]))
    assert np.allclose(tracker._gt_test_navigation_T_world, new_transform)
    assert not np.allclose(new_transform, old_transform)
    assert np.allclose(corrected.pose, new_transform @ tracker._gt_test_states.at(1.37)[0])


def test_background_or_rejected_images_still_set_the_previous_input_timestamp():
    tracker = fixture_tracker(True)
    tracker.bg_images = []
    tracker._static_imu_settings = None
    tracker._gt_test_last_image_timestamp = None
    for t in (1., 1.05, 1.1):
        assert tracker.track(np.zeros((2, 2, 3)), header(t)) == (False, None)
    assert tracker._gt_test_previous_image_timestamp == 1.05
    assert tracker._frontend_vi_state.timestamp == 1.  # Not the raw previous image.


def test_diagnostic_callback_supplies_full_imu_rotation_and_position():
    received = []
    node = SimpleNamespace(
        tracker=SimpleNamespace(
            imu_config={'frontend_gt_previous_state_test': True},
            add_ground_truth_imu_pose=lambda *args: received.append(args) or True,
        ),
        get_logger=lambda: SimpleNamespace(warning=lambda _: None),
    )
    message = PoseStamped()
    message.header.stamp.sec = 2
    quaternion = Rotation.from_euler('xyz', [.3, -.2, .1]).as_quat()
    message.pose.orientation.x, message.pose.orientation.y = quaternion[:2]
    message.pose.orientation.z, message.pose.orientation.w = quaternion[2:]
    message.pose.position.x = .4
    TrackerNode.ground_truth_imu_pose_callback(node, message)
    assert received[0][0] == 2.
    assert np.allclose(received[0][1], Rotation.from_quat(quaternion).as_matrix())
    assert received[0][2] == (.4, 0., 0.)


@pytest.mark.parametrize('pose_gt', [False, True])
@pytest.mark.parametrize('velocity_gt', [False, True])
@pytest.mark.parametrize('reference_gt', [False, True])
@pytest.mark.parametrize('reset_prior', [False, True])
def test_all_interventions_are_independent(monkeypatch, pose_gt, velocity_gt, reference_gt, reset_prior):
    controls = dict(enabled=True, previous_pose_from_gt=pose_gt,
                    previous_velocity_from_gt=velocity_gt, reference_pose_from_gt=reference_gt,
                    reset_marginal_prior=reset_prior, previous_state_time='accepted_state')
    tracker = fixture_tracker(controls)
    solver = tracker.frontend_visual_inertial
    original = tracker._frontend_vi_state.copy()
    prior = solver.marginal_prior
    seen = {}
    class Visual:
        def __init__(self, *args, **kwargs):
            pass
    def refine(visual, reference, previous, current, samples, extrinsic, preintegrated=None):
        seen.update(previous=previous, reference=reference, prior=solver.marginal_prior)
        return current
    monkeypatch.setattr(tracker_module, 'DenseNormalResidual', Visual)
    monkeypatch.setattr(tracker_module, 'normalflow_quality', lambda *a: (1., 1.))
    monkeypatch.setattr(solver, 'refine_current', refine)
    if not (pose_gt or velocity_gt or reference_gt):
        tracker._gt_test_states = None  # Prior-only and no-op require no GT.
    tracker._refine_normalflow_with_imu(np.eye(4), header(1.1), tracker.ref_frame, object())
    previous = seen['previous']
    assert previous.timestamp == original.timestamp
    expected_pose = tracker.gelsight_T_imu.copy()
    expected_pose[:3, 3] += [.005, .01, .015]
    assert np.allclose(previous.pose, expected_pose if pose_gt else original.pose)
    assert np.allclose(previous.velocity, [.1, .2, .3] if velocity_gt else original.velocity)
    assert np.allclose(seen['reference'], np.eye(4) if reference_gt else tracker.trial_T_ref)
    assert seen['prior'] is (None if reset_prior else prior)
    assert np.array_equal(previous.accel_bias, original.accel_bias)


@pytest.mark.parametrize('config_name', ['config.yaml', 'real_config.yaml', 'bag_simulation_config.yaml'])
@pytest.mark.parametrize('pose_gt', [False, True])
def test_shipped_config_keeps_one_field_gt_at_accepted_time_after_rejected_image(
        monkeypatch, config_name, pose_gt):
    config_path = Path(__file__).resolve().parents[1] / 'config' / config_name
    controls = yaml.safe_load(config_path.read_text())['imu_config']['frontend_gt_previous_state_test']
    # Change only the user-facing switches. Keep the shipped time policy.
    controls.update(enabled=True, previous_pose_from_gt=pose_gt,
                    previous_velocity_from_gt=not pose_gt)
    tracker = fixture_tracker(controls)
    original = tracker._frontend_vi_state.copy()
    seen = {}

    class Visual:
        def __init__(self, *args, **kwargs):
            pass

    def refine(visual, reference, previous, current, samples, extrinsic, preintegrated=None):
        seen.update(previous=previous.copy(), samples=samples)
        return current

    monkeypatch.setattr(tracker_module, 'DenseNormalResidual', Visual)
    monkeypatch.setattr(tracker_module, 'normalflow_quality', lambda *args: (1., 1.))
    monkeypatch.setattr(tracker.frontend_visual_inertial, 'refine_current', refine)
    tracker._refine_normalflow_with_imu(np.eye(4), header(1.1), tracker.ref_frame, object())
    previous = seen['previous']
    # Accepted state t=1.0; rejected input t=1.05; target t=1.1.
    assert previous.timestamp == original.timestamp == 1.
    assert seen['samples'][0].timestamp == 1.
    assert seen['samples'][-1].timestamp == 1.1
    expected_pose = tracker.gelsight_T_imu.copy()
    expected_pose[:3, 3] += [.005, .01, .015]
    assert np.allclose(previous.pose, expected_pose if pose_gt else original.pose)
    assert np.allclose(previous.velocity, original.velocity if pose_gt else [.1, .2, .3])
    assert np.array_equal(previous.accel_bias, original.accel_bias)
    assert np.array_equal(previous.gyro_bias, original.gyro_bias)


class EmptyNormalRows:
    def __len__(self):
        return 10
    def evaluate(self, *args):
        return np.zeros(30), np.zeros((30, 6)), np.zeros((30, 6))


def test_solver_really_retains_frozen_prior_when_gt_start_differs():
    controls = dict(enabled=True, reset_marginal_prior=False, previous_state_time='accepted_state')
    tracker = fixture_tracker(controls)
    solver = tracker.frontend_visual_inertial
    solver.max_iterations = 1
    prior = solver.marginal_prior
    anchor_pose, A, rhs = prior.anchor.pose.copy(), prior.A.copy(), prior.rhs.copy()
    previous, reference = tracker._gt_test_inputs(tracker._frontend_vi_state, 1., .95)
    assert not prior.matches(previous)
    samples = tracker.imu_buffer.samples_covering(1., 1.1)
    current, pim = solver.predict_current(previous, samples, tracker.gelsight_T_imu)
    solver.refine_current(EmptyNormalRows(), reference, previous, current, samples,
                          tracker.gelsight_T_imu, preintegrated=pim)
    assert solver.last_prior_source == 'retained'
    assert np.array_equal(prior.anchor.pose, anchor_pose)
    assert np.array_equal(prior.A, A) and np.array_equal(prior.rhs, rhs)


def test_retaining_a_prior_at_another_timestamp_is_rejected():
    tracker = fixture_tracker(dict(enabled=True, reset_marginal_prior=False,
                                   previous_state_time='previous_image'))
    solver = tracker.frontend_visual_inertial
    previous, reference = tracker._gt_test_inputs(tracker._frontend_vi_state, 1.05, .95)
    samples = tracker.imu_buffer.samples_covering(1.05, 1.1)
    current, pim = solver.predict_current(previous, samples, tracker.gelsight_T_imu)
    with pytest.raises(RuntimeError, match='different IMU timestamp'):
        solver.refine_current(EmptyNormalRows(), reference, previous, current, samples,
                              tracker.gelsight_T_imu, preintegrated=pim)


def test_accepted_state_time_reports_required_prior_restart_after_a_state_jump():
    tracker = fixture_tracker(dict(enabled=True, reset_marginal_prior=False,
                                   previous_state_time='accepted_state'))
    solver = tracker.frontend_visual_inertial
    solver.max_iterations = 1
    previous, reference = tracker._gt_test_inputs(tracker._frontend_vi_state, 1.05, .95)
    samples = tracker.imu_buffer.samples_covering(1.05, 1.1)
    current, pim = solver.predict_current(previous, samples, tracker.gelsight_T_imu)
    solver.refine_current(EmptyNormalRows(), reference, previous, current, samples,
                          tracker.gelsight_T_imu, preintegrated=pim)
    assert solver.last_prior_source == 'initial_timestamp_change'


def test_legacy_boolean_and_disabled_mapping_keep_their_meanings():
    assert not resolve_gt_state_test({})['enabled']
    old = resolve_gt_state_test({'frontend_gt_previous_state_test': True})
    assert old['enabled'] and old['reset_marginal_prior']
    assert old['previous_state_time'] == 'previous_image'
    assert not gt_state_test_needs_gt(resolve_gt_state_test({
        'frontend_gt_previous_state_test': dict(enabled=False)}))
    assert not gt_state_test_needs_gt(resolve_gt_state_test({
        'frontend_gt_previous_state_test': dict(enabled=True, previous_pose_from_gt=False,
            previous_velocity_from_gt=False, reference_pose_from_gt=False, reset_marginal_prior=True)}))
