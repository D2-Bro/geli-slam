"""Causal coast-entry velocity, coordinate frames, and prior ownership."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gelslam.core.contact_velocity import CoastVelocitySeed, ContactVelocityHistory
from gelslam.core.numpy_vi import NavigationState


def state(t, p, v=(0., 0., 0.)):
    pose = np.eye(4)
    pose[:3, 3] = p
    return NavigationState(float(t), pose, np.array(v, float), np.zeros(3), np.zeros(3))


def history(**settings):
    return ContactVelocityHistory.from_config(dict(coast_velocity_fit=dict(enabled=True, **settings)))


def trajectory(h, endpoint=1., trial=0, rotation=np.eye(3), translation=np.zeros(3)):
    velocity, acceleration = np.array([.005, -.003, .002]), np.array([.02, .01, -.005])
    for offset in [-.4, -.35, -.26, -.2, -.1, -.05, 0.]:
        p = translation + rotation @ (velocity*offset + .5*acceleration*offset*offset)
        current = state(endpoint+offset, p)
        h.observe(current, trial)
    return current, rotation@velocity


def test_irregular_contact_positions_recover_endpoint_velocity_and_preserve_anchor():
    h = history()
    anchor, expected = trajectory(h)
    before = anchor.copy()
    seed = h.seed(anchor)
    assert seed is not None and seed.samples == 7
    np.testing.assert_allclose(seed.velocity_delta, expected, atol=1e-15)
    initial = seed.apply(anchor)
    np.testing.assert_array_equal(initial.pose, anchor.pose)
    np.testing.assert_allclose(initial.velocity, expected, atol=1e-15)
    np.testing.assert_array_equal(anchor.pose, before.pose)
    np.testing.assert_array_equal(anchor.velocity, before.velocity)
    target = state(1.7, [.01, -.02, .03], [.002, .001, -.003])
    changed = seed.apply(target)
    np.testing.assert_allclose(changed.pose[:3,3], target.pose[:3,3]+.7*expected, atol=1e-15)
    np.testing.assert_allclose(changed.velocity, target.velocity+expected, atol=1e-15)
    np.testing.assert_array_equal(changed.pose[:3,:3], target.pose[:3,:3])
    np.testing.assert_array_equal(changed.accel_bias, target.accel_bias)
    np.testing.assert_array_equal(changed.gyro_bias, target.gyro_bias)


def test_fit_respects_fixed_trial_axes_and_imu_origin():
    rotation = Rotation.from_rotvec([.4, -.3, .2]).as_matrix()
    anchor, expected = trajectory(h := history(), rotation=rotation, translation=np.array([1., -2., 3.]))
    np.testing.assert_allclose(h.seed(anchor).velocity_delta, expected, atol=1e-13)


def test_future_contact_and_imu_recovery_anchor_cannot_seed_an_older_gap():
    anchor, _ = trajectory(h := history())
    assert h.seed(state(.95, [0., 0., 0.])) is None
    assert h.last_status == 'anchor_is_not_last_contact'
    assert h.seed(state(1.05, [0., 0., 0.])) is None
    assert h.seed(state(1., [.001, 0., 0.])) is None


def test_contact_gap_trial_change_and_clear_discard_old_positions():
    anchor, _ = trajectory(h := history())
    h.observe(state(1.2, [0., 0., 0.]), 0)
    assert len(h.samples) == 1 and h.seed(anchor) is None
    trajectory(h, endpoint=2.)
    h.observe(state(2.05, [0., 0., 0.]), 1)
    assert len(h.samples) == 1
    h.clear()
    assert not h.samples


def test_repeated_timestamp_replaces_position_and_backward_clock_clears_history():
    anchor, _ = trajectory(h := history())
    count = len(h.samples)
    h.observe(anchor.copy(), 0)
    assert len(h.samples) == count
    h.observe(state(.8, [0., 0., 0.]), 0)
    assert len(h.samples) == 1


def test_curve_residual_and_velocity_change_reject_bad_fits():
    h = history(max_fit_rmse_m=1e-6)
    for k, t in enumerate(np.linspace(.6, 1., 9)):
        anchor=state(t,[.001*(k%2), 0., 0.]);h.observe(anchor,0)
    assert h.seed(anchor) is None
    assert h.last_status == 'contact_curve_residual_too_large'
    anchor,_=trajectory(h := history(max_velocity_change_mps=.001))
    assert h.seed(anchor) is None
    assert h.last_status == 'velocity_change_too_large'


def test_short_span_and_too_few_points_keep_contact_solver_velocity():
    h=history()
    for t in np.linspace(.95,1.,9):anchor=state(t,[.001*t,0.,0.]);h.observe(anchor,0)
    assert h.seed(anchor) is None
    h.clear()
    for t in [.6,.75,.9,1.]:anchor=state(t,[.001*t,0.,0.]);h.observe(anchor,0)
    assert h.seed(anchor) is None


@pytest.mark.parametrize('settings', [dict(enabled=1), dict(window_s=0),
    dict(window_s=float('nan')),dict(min_samples=4),dict(min_samples=True),
    dict(min_span_s=.5),dict(max_fit_rmse_m=-1),dict(unknown=1)])
def test_invalid_enabled_configuration_is_rejected(settings):
    with pytest.raises(ValueError):
        ContactVelocityHistory.from_config(dict(coast_velocity_fit=dict({'enabled':True},**settings)))


def test_disabled_configuration_does_not_change_existing_tracker_behavior():
    assert ContactVelocityHistory.from_config({}) is None
    assert ContactVelocityHistory.from_config(dict(coast_velocity_fit=dict(enabled=False))) is None


def test_inertial_fit_handles_changing_acceleration_and_sensor_axes_without_future_samples():
    from gelslam.core.inertial import ImuSample
    velocity=np.array([.01,-.005,.002]);jerk=np.array([1.,-.8,.3])
    rotation=Rotation.from_rotvec([.4,-.2,.3]).as_matrix()
    gravity=np.array([0.,0.,-9.80665]);ba=np.array([.02,-.01,.015]);bg=np.array([.001,-.002,.0005])
    ic=dict(coast_velocity_fit=dict(enabled=True,model='inertial_quadratic',max_velocity_change_mps=1.),max_sample_gap_s=.02)
    h=ContactVelocityHistory.from_config(ic)
    last=None
    for t in np.linspace(.6,1.,9):
        offset=t-1.
        p=velocity*offset+jerk*offset**3/6
        anchor=state(t,p);anchor.pose[:3,:3]=rotation;anchor.accel_bias=ba.copy();anchor.gyro_bias=bg.copy()
        samples=[]
        if last is not None:
            for stamp in np.linspace(last,t,11):
                samples.append(ImuSample(float(stamp),rotation.T@(jerk*(stamp-1.)-gravity)+ba,bg))
        h.observe(anchor,0,samples);last=t
    seed=h.seed(anchor,gravity)
    assert seed is not None
    # 5 ms midpoint intervals approximate a linearly changing acceleration;
    # the analytic jerk trajectory bounds this discretization at 3 um/s.
    np.testing.assert_allclose(seed.velocity_delta,velocity,atol=3e-6)
    assert h.imu_samples[-1].timestamp==anchor.timestamp
    h.clear()
    assert not h.imu_samples


def test_inertial_fit_rejects_missing_or_gapped_imu_history():
    from gelslam.core.inertial import ImuSample
    h=history(model='inertial_quadratic');anchor,_=trajectory(h)
    assert h.seed(anchor,np.array([0.,0.,-9.81])) is None
    assert h.last_status=='missing_contact_imu_interval'
    h.imu_samples.extend([ImuSample(.6,[0,0,9.81],[0,0,0]),ImuSample(1.,[0,0,9.81],[0,0,0])])
    assert h.seed(anchor,np.array([0.,0.,-9.81])) is None


def test_tracker_never_uses_prediction_only_contact_states_as_pose_observations():
    from test_tracker_inertial import make_coasting_tracker, header
    from gelslam.core.gtsam_vi import GtsamVisualInertialSolver
    tracker=make_coasting_tracker()
    tracker.frontend_visual_inertial=GtsamVisualInertialSolver(dict(frontend_imu_prediction_only=True))
    anchor,_=trajectory(h := history())
    tracker._frontend_vi_state=anchor
    tracker._contact_velocity_history=h
    tracker._trial_index=0
    tracker._record_tactile_pose(anchor.pose,header(1.))
    assert not h.samples
    assert tracker._coast_velocity_seed is None


def test_tracker_history_records_navigation_origin_in_trial_axes():
    from test_tracker_inertial import make_coasting_tracker, header
    from gelslam.core.gtsam_vi import GtsamVisualInertialSolver
    tracker=make_coasting_tracker()
    tracker.frontend_visual_inertial=GtsamVisualInertialSolver({})
    tracker._contact_velocity_history=h=history()
    tracker._trial_index=0
    navigation=state(1.,[.01,.02,.04])
    tracker._frontend_vi_state=navigation
    gel_pose=np.eye(4);gel_pose[:3,3]=[.01,.02,0.]
    tracker._record_tactile_pose(gel_pose,header(1.))
    np.testing.assert_array_equal(h.samples[-1][1],navigation.pose[:3,3])


def test_no_contact_seed_is_frozen_and_never_changes_accepted_state_or_prior():
    from test_tracker_inertial import make_coasting_tracker, header
    from gelslam.core.gtsam_vi import GtsamVisualInertialSolver
    from gelslam.core.marginal_prior import NavigationPrior
    from gelslam.core.tracker import TrackingMode
    tracker=make_coasting_tracker()
    tracker.frontend_visual_inertial=GtsamVisualInertialSolver({})
    anchor,expected=trajectory(h := history())
    tracker._contact_velocity_history=h
    tracker._frontend_vi_state=anchor
    solver=tracker.frontend_visual_inertial
    solver.marginal_prior=prior=NavigationPrior.initial(anchor,np.ones(15))
    before=anchor.copy()
    assert tracker._initialize_no_contact_coast()
    seed=tracker._coast_velocity_seed
    assert seed is not None and not h.samples
    assert tracker._coast_to_header(header(2.),TrackingMode.NO_CONTACT_COASTING)
    predicted,_=solver.predict_preintegrated(anchor,tracker._frontend_imu_preintegration.pim,2.,force=True)
    np.testing.assert_allclose(tracker._frontend_coast_state.pose[:3,3],predicted.pose[:3,3]+expected,atol=1e-14)
    assert tracker._initialize_no_contact_coast()
    assert tracker._coast_velocity_seed is seed
    np.testing.assert_allclose(tracker._frontend_coast_state.velocity,predicted.velocity+expected,atol=1e-14)
    assert tracker._frontend_vi_state is anchor and solver.marginal_prior is prior
    for name in ['pose','velocity','accel_bias','gyro_bias']:
        np.testing.assert_array_equal(getattr(anchor,name),getattr(before,name))
    tracker._clear_contact_velocity_history()
    assert tracker._coast_velocity_seed is None


@pytest.mark.parametrize('state_mode', ['pairwise', 'reference_marginalization'])
def test_coast_seed_translates_prior_distribution_and_preserves_cross_information(state_mode):
    from gelslam.core.gtsam_vi import create_frontend_solver
    from gelslam.core.inertial import ImuSample
    from gelslam.core.marginal_prior import NavigationPrior
    from gelslam.core.retained_prior import state_blocks, state_values
    from gelslam.core.split_pose import split_pose_plus

    solver = create_frontend_solver(dict(frontend_state_mode=state_mode))
    previous = state(1., [.1, -.2, .3], [.01, -.02, .03])
    previous.pose[:3, :3] = Rotation.from_rotvec([.2, -.1, .3]).as_matrix()
    solver.marginal_prior = NavigationPrior.initial(previous, np.ones(15))
    if state_mode == 'reference_marginalization':
        solver.set_reference(previous.timestamp, previous)
        bootstrap = tuple(ImuSample(float(t), [0, 0, 9.81], [0, 0, .1])
                          for t in np.linspace(1., 1.1, 11))
        target, pim = solver.predict_current(previous, bootstrap, np.eye(4), force=True)
        solver.marginal_prior = solver.propagate_prior(previous, target, pim)
        previous = target
        solver.set_reference(previous.timestamp, previous)
    samples = tuple(ImuSample(float(t), [0, 0, 9.81], [0, 0, .1])
                    for t in np.linspace(previous.timestamp, 1.2, 21))
    current, pim = solver.predict_current(previous, samples, np.eye(4), force=True)
    prior = solver.propagate_prior(previous, current, pim)
    prior.rhs += np.linspace(-.01, .02, len(prior.rhs))
    seed = CoastVelocitySeed(previous.timestamp, np.array([.0044, -.0026, -.00014]), 0., 8, .35)
    shifted = seed.apply_prior(prior)
    assert shifted.matches(seed.apply(current))
    assert prior.matches(current)
    np.testing.assert_array_equal(shifted.A, prior.A)
    np.testing.assert_array_equal(shifted.rhs, prior.rhs)
    assert not np.shares_memory(shifted.A, prior.A)
    assert not np.shares_memory(shifted.rhs, prior.rhs)
    perturbation = np.linspace(-.002, .003, 15)
    probe = current.copy()
    probe.pose = split_pose_plus(probe.pose, perturbation[:6])
    probe.velocity += perturbation[6:9]
    probe.accel_bias += perturbation[9:12]
    probe.gyro_bias += perturbation[12:15]
    if state_mode == 'pairwise':
        assert shifted.error(seed.apply(probe)) == pytest.approx(prior.error(probe), abs=1e-12)
    else:
        assert prior.layout.dimension == 21
        assert np.linalg.norm(prior.information[:6, 6:]) > 0.
        original_values = dict(prior.values)
        original_values.update(state_values(probe))
        shifted_values = dict(original_values)
        shifted_values.update(state_values(seed.apply(probe)))
        assert shifted.error_values(shifted_values) == pytest.approx(
            prior.error_values(original_values), abs=1e-12)
        assert shifted.absorbed_intervals == prior.absorbed_intervals
        for block in prior.fixed_values:
            np.testing.assert_array_equal(shifted.fixed_values[block], prior.fixed_values[block])
        # The free retained reference is a distinct event from the target.
        np.testing.assert_array_equal(shifted.values[state_blocks(previous.timestamp)[0]], previous.pose)


@pytest.mark.parametrize('state_mode', ['pairwise', 'reference_marginalization'])
@pytest.mark.parametrize('initialize_from_imu', [False, True])
def test_imu_recovery_keeps_coast_state_and_prior_through_next_optimization(
        monkeypatch, state_mode, initialize_from_imu):
    import gelslam.core.tracker as tm
    from gelslam.core.gtsam_vi import create_frontend_solver
    from gelslam.core.marginal_prior import NavigationPrior
    from gelslam.core.tracker import TrackingMode
    from test_contact_recovery_policy import FlatVisual, tracker_with_imu
    from test_tracker_inertial import header

    tracker, anchor = tracker_with_imu(monkeypatch)
    solver = create_frontend_solver(dict(
        frontend_state_mode=state_mode,
        frontend_initialize_from_imu_preintegration=initialize_from_imu))
    tracker.frontend_visual_inertial = solver
    solver.marginal_prior = NavigationPrior.initial(anchor, solver.prior_sigmas)
    tracker._sync_reference_prior()
    original_prior = solver.marginal_prior
    seed = CoastVelocitySeed(anchor.timestamp, np.array([.00443557, -.00258236, -.00014073]),
                             .000042782, 8, .35)
    tracker._coast_velocity_seed = seed
    tracker._coast_velocity_anchor_timestamp = anchor.timestamp
    assert tracker._initialize_no_contact_coast()
    assert tracker._coast_to_header(header(1.24), TrackingMode.NO_CONTACT_COASTING)
    coast = tracker._frontend_coast_state.copy()
    assert tracker._frontend_vi_state is anchor
    assert solver.marginal_prior is original_prior

    ok, (message, packets) = tracker._recover_current_frame_with_imu(
        tracker.ref_frame, header(1.24), 'test')
    assert ok and len(packets) == 1 and not packets[0]['tactile_valid']
    recovered = tracker._frontend_vi_state
    for name in ('pose', 'velocity', 'accel_bias', 'gyro_bias'):
        np.testing.assert_allclose(getattr(recovered, name), getattr(coast, name), atol=1e-12)
    np.testing.assert_allclose(packets[0]['navigation_state'].pose, recovered.pose, atol=1e-12)
    assert solver.marginal_prior.matches(recovered)
    assert tracker._coast_velocity_seed is None
    assert tracker._coast_velocity_anchor_timestamp is None
    assert tracker._frontend_imu_preintegration is None
    np.testing.assert_array_equal(message['ref_T_curr'], np.eye(4))

    interval = tracker.imu_buffer.samples_covering(recovered.timestamp, 1.28)
    expected, _ = solver.predict_current(recovered, interval, np.eye(4), force=True)
    monkeypatch.setattr(tm, 'DenseNormalResidual', lambda *a, **k: FlatVisual())
    monkeypatch.setattr(tm, 'normalflow_quality', lambda *a: (.9, .9))
    ok, (_, packets) = tracker.track(np.zeros((1, 1, 3), np.uint8), header(1.28))
    assert ok and not packets
    np.testing.assert_allclose(tracker._frontend_vi_state.pose, expected.pose, atol=1e-10)
    np.testing.assert_allclose(tracker._frontend_vi_state.velocity, expected.velocity, atol=1e-10)
    assert solver.marginal_prior.matches(tracker._frontend_vi_state)

    # A later recovery starts at the accepted corrected state; do not add the old seed again.
    accepted = tracker._frontend_vi_state
    interval = tracker.imu_buffer.samples_covering(accepted.timestamp, 1.32)
    expected, _ = solver.predict_current(accepted, interval, np.eye(4), force=True)
    assert tracker._recover_current_frame_with_imu(tracker.prev_frame, header(1.32), 'test')[0]
    np.testing.assert_allclose(tracker._frontend_vi_state.pose, expected.pose, atol=1e-10)
    np.testing.assert_allclose(tracker._frontend_vi_state.velocity, expected.velocity, atol=1e-10)
    assert solver.marginal_prior.matches(tracker._frontend_vi_state)
