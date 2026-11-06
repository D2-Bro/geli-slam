"""Tracker-level tests for tactile tracking and no-contact IMU coasting."""

from types import SimpleNamespace

import numpy as np
import pytest

from gelslam.core.inertial import ImuBuffer, ImuSample
from gelslam.core.numpy_vi import NavigationState
from gelslam.core.tracker import Tracker, TrackingMode
import gelslam.core.tracker as tracker_module
from gelslam.core.gtsam_vi import create_frontend_solver
from gelslam.core.imu_navigation import imu_pose
from gelslam.core.marginal_prior import NavigationPrior


def rm_stub(predict, refine):
    """Fake reference-marginalization solver around test predict/refine callables.

    The reference keeps the visual reference pose that the tracker passes in.
    """
    solver = SimpleNamespace(
        predict_current=predict, gravity=np.array([0., 0., -9.81]),
        set_reference=lambda *_args: None, reference_id=0, first_pose_id=0,
        last_reference_pose=None)

    def refine_current(visual, reference, previous, current, samples, extrinsic,
                       preintegrated=None):
        solver.last_reference_pose = imu_pose(reference, extrinsic)
        return refine(visual, reference, previous, current, samples, extrinsic,
                      preintegrated=preintegrated)

    solver.refine_current = refine_current
    return solver


def header(timestamp):
    sec = int(timestamp)
    return SimpleNamespace(
        stamp=SimpleNamespace(sec=sec, nanosec=int(round((timestamp - sec) * 1e9)))
    )


def make_coasting_tracker():
    """Build only the inertial tracker state; no hardware reconstructor is needed."""
    tracker = object.__new__(Tracker)
    tracker.logger = SimpleNamespace(info=lambda _message: None)
    tracker.gelsight_T_imu = np.eye(4)
    tracker.imu_buffer = ImuBuffer()
    tracker.imu_buffer.add(
        ImuSample(1.0, (0.0, 0.0, 9.81), (0.0, 0.0, np.pi / 2.0))
    )
    tracker.imu_buffer.add(
        ImuSample(2.0, (0.0, 0.0, 9.81), (0.0, 0.0, 0.0))
    )
    tracker.imu_config = {"gravity_mps2": 9.81}
    tracker.inertial_state = None
    tracker.inertial_propagator = None
    tracker.ref_T_prev = np.eye(4, dtype=np.float32)
    tracker.trial_T_ref = np.eye(4, dtype=np.float64)
    tracker.ppmm = 1.0
    tracker.track_config = {"scr_threshold": 0.3, "ccs_threshold": 0.7}
    tracker.frontend_visual_inertial = None
    return tracker


def test_successful_tactile_anchor_can_coast_to_a_later_image_timestamp():
    tracker = make_coasting_tracker()

    assert tracker._anchor_inertial_state(np.eye(4), header(1.0))
    assert tracker._coast_to_header(header(2.0), TrackingMode.IMU_COASTING)

    assert tracker.tracking_mode is TrackingMode.IMU_COASTING
    assert np.allclose(
        tracker.ref_T_prev[:3, :3],
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        atol=1e-6,
    )


def test_no_contact_coast_retains_imu_from_last_accepted_vi_state():
    tracker = make_coasting_tracker()
    tracker.imu_config["max_coast_duration_s"] = 3.0
    tracker._frontend_vi_state = NavigationState(
        1.0, np.eye(4), np.zeros(3), np.zeros(3), np.zeros(3)
    )
    tracker.imu_buffer.add(
        ImuSample(3.0, (0.0, 0.0, 9.81), (0.0, 0.0, 0.0))
    )

    assert tracker._anchor_inertial_state(np.eye(4), header(1.0))
    assert tracker._coast_to_header(header(2.0), TrackingMode.NO_CONTACT_COASTING)
    assert tracker._coast_to_header(header(3.0), TrackingMode.NO_CONTACT_COASTING)

    samples = tracker.imu_buffer.samples_covering(1.0, 3.0)
    assert [sample.timestamp for sample in samples] == [1.0, 2.0, 3.0]
    assert tracker._frontend_vi_state.timestamp == 1.0


def test_image_to_imu_offset_moves_the_integration_boundary():
    tracker = make_coasting_tracker()
    tracker.image_to_imu_time_offset_s = 0.025

    assert tracker._imu_timestamp(header(1.0)) == 1.025


def test_delayed_gt_does_not_publish_fallback_frontend_gravity():
    tracker = make_coasting_tracker()
    tracker.imu_config = {
        "gravity_mps2": 9.81,
        "frontend_optimization": True,
        "frontend_gravity_initialization": "ground_truth_initial_pose",
    }
    tracker.frontend_visual_inertial = SimpleNamespace(gravity=np.array((0.0, 0.0, -9.81)))
    tracker._frontend_vi_state = None
    tracker._frontend_trial_start_timestamp = 1.0
    tracker.initial_ground_truth_world_R_gel = None
    tracker._ground_truth_gel_rotations = []
    tracker.logger = SimpleNamespace(info=lambda _message: None)

    tracker._record_tactile_pose(np.eye(4), header(1.0))
    assert tracker._frontend_vi_state is None
    assert tracker.frontend_gravity_for_backend() is None

    world_R_gel = np.array(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))
    tracker._ground_truth_gel_rotations.append((1.0, world_R_gel))
    tracker._record_tactile_pose(np.eye(4), header(1.1))

    assert tracker._frontend_vi_state is not None
    assert np.allclose(tracker.frontend_gravity_for_backend(), (0.0, 0.0, -9.81))
    assert np.allclose(tracker.frontend_visual_inertial.gravity, (0.0, 0.0, -9.81))
    assert np.allclose(tracker._frontend_vi_state.pose[:3, :3], world_R_gel)
    assert np.allclose(tracker._frontend_vi_state.pose[:3, 3], 0.0)


def test_first_trial_velocity_uses_gt_imu_positions_in_trial_axes():
    tracker = make_coasting_tracker()
    tracker.curr_kid = 0
    tracker.imu_config = {
        "gravity_mps2": 9.81,
        "frontend_optimization": True,
        "frontend_gravity_initialization": "ground_truth_initial_pose",
        "frontend_first_trial_velocity_from_gt": True,
    }
    tracker.frontend_visual_inertial = object()
    tracker.initial_ground_truth_world_R_gel = None
    tracker._ground_truth_gel_rotations = []
    tracker._ground_truth_gel_positions = []
    tracker.logger = SimpleNamespace(info=lambda _message: None)
    world_R_gel = np.array(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))
    world_velocity = np.array((0.03, 0.0, -0.02))
    for timestamp in np.linspace(0.95, 1.0, 6):
        position = np.array((0.1, 0.2, 0.3)) + world_velocity * (timestamp - 1.0)
        assert tracker.add_ground_truth_gel_pose(timestamp, world_R_gel, position)
        assert tracker.add_ground_truth_imu_position(timestamp, position)
    tracker.add_ground_truth_gel_pose(
        1.01, world_R_gel, np.array((0.1, 0.2, 0.3)) + world_velocity * .01
    )
    tracker.add_ground_truth_imu_position(
        1.01, np.array((0.1, 0.2, 0.3)) + world_velocity * .01
    )

    tracker._reset_frontend_visual_inertial(np.eye(4), header(1.0))

    assert tracker._frontend_vi_state is not None
    assert np.allclose(
        tracker._frontend_vi_state.velocity,
        world_velocity,
        atol=1e-10,
    )


def _make_gt_velocity_tracker():
    tracker = make_coasting_tracker()
    tracker.imu_config.update({
        "frontend_optimization": True,
        "frontend_gravity_initialization": "ground_truth_initial_pose",
        "frontend_first_trial_velocity_from_gt": True,
    })
    tracker.frontend_visual_inertial = object()
    tracker.initial_ground_truth_world_R_gel = None
    tracker._ground_truth_gel_rotations = []
    tracker._ground_truth_gel_positions = []
    tracker._trial_index = 0
    tracker._frontend_trial_start_timestamp = 1.0
    tracker.logger = SimpleNamespace(info=lambda _message: None)
    return tracker


def test_gt_imu_velocity_does_not_use_gelsight_motion_or_measured_gyro():
    tracker = _make_gt_velocity_tracker()
    tracker.gelsight_T_imu[:3, 3] = [.04, -.02, 0.]
    tracker.imu_buffer = ImuBuffer()
    tracker.imu_buffer.add(ImuSample(1., [0., 0., 9.81], [0., 0., 5.]))
    imu_velocity_world = np.array([.1, -.2, .03])
    tracker._ground_truth_imu_positions = [
        (t, imu_velocity_world * t) for t in (.99, 1., 1.01)
    ]
    for t in (.99, 1., 1.01):
        tracker.add_ground_truth_gel_pose(t, np.eye(3), [0., 0., 0.])

    tracker._reset_frontend_visual_inertial(np.eye(4), header(1.))

    assert tracker._frontend_vi_state is not None
    assert np.allclose(tracker._frontend_vi_state.velocity, imu_velocity_world, atol=1e-12)


def test_online_gt_attitudes_keep_recent_rotations_without_positions():
    tracker = make_coasting_tracker()
    for index in range(1200):
        assert tracker.add_recent_ground_truth_gel_rotation(1.0 + index * .005, np.eye(3))
    samples = tracker._ground_truth_gel_rotations
    assert len(samples) <= 201
    assert samples[0][0] >= samples[-1][0] - 1.0
    assert getattr(tracker, '_ground_truth_gel_positions', []) == []
    count = len(samples)
    changed = np.diag([1., -1., -1.])
    assert tracker.add_recent_ground_truth_gel_rotation(samples[-2][0], changed)
    assert len(tracker._ground_truth_gel_rotations) == count
    np.testing.assert_array_equal(tracker._ground_truth_gel_rotations[-2][1], changed)


def test_late_online_gt_attitude_completes_original_pending_anchor():
    tracker = _make_gt_velocity_tracker()
    for timestamp in (.99, 1., 1.01):
        tracker.add_ground_truth_imu_position(timestamp, [.2 * timestamp**2, 0., 0.])
    tracker._reset_frontend_visual_inertial(np.eye(4), header(1.))
    assert tracker._frontend_vi_state is None
    for timestamp in (2., 3., 4.):
        tracker.add_recent_ground_truth_gel_rotation(timestamp, np.eye(3))
    assert tracker._frontend_vi_state is None
    tracker.add_recent_ground_truth_gel_rotation(1., np.eye(3))
    assert tracker._frontend_vi_state.timestamp == 1.
    np.testing.assert_allclose(tracker._frontend_vi_state.velocity, [.4, 0., 0.])


def test_recent_gt_attitudes_allow_next_trial_after_imu_gt_is_released():
    tracker = _make_gt_velocity_tracker()
    for timestamp in (.99, 1., 1.01):
        tracker.add_recent_ground_truth_gel_rotation(timestamp, np.eye(3))
        tracker.add_ground_truth_imu_position(timestamp, [.1 * timestamp, 0., 0.])
    tracker._reset_frontend_visual_inertial(np.eye(4), header(1.))
    first_pose = tracker._frontend_vi_state.pose.copy()
    tracker._ground_truth_imu_positions.clear()
    next_rotation = np.diag([1., -1., -1.])
    for timestamp in np.linspace(2., 5., 601):
        tracker.add_recent_ground_truth_gel_rotation(timestamp, next_rotation)
    np.testing.assert_array_equal(tracker._frontend_vi_state.pose, first_pose)
    assert tracker._ground_truth_gel_rotations[0][0] >= 4.
    tracker._trial_index = 1
    tracker._frontend_trial_start_timestamp = 5.
    tracker._trial_start_world_R_gel = None
    tracker.trial_T_ref = np.eye(4)
    tracker._reset_frontend_visual_inertial(np.eye(4), header(5.))
    assert tracker._frontend_vi_state.timestamp == 5.
    np.testing.assert_array_equal(tracker._trial_start_world_R_gel, next_rotation)
    assert not np.allclose(tracker._frontend_vi_state.pose[:3, :3], first_pose[:3, :3])
    np.testing.assert_allclose(tracker.frontend_visual_inertial.gravity, [0., 0., -9.81])
    np.testing.assert_array_equal(tracker._frontend_vi_state.velocity, np.zeros(3))


def test_gt_imu_velocity_initializes_without_a_measured_gyro():
    tracker = _make_gt_velocity_tracker()
    tracker.gelsight_T_imu[:3, 3] = [.04, -.02, 0.]
    tracker.imu_buffer = ImuBuffer()
    imu_velocity_world = np.array([.1, .2, .3])
    for time in (.99, 1., 1.01):
        tracker.add_ground_truth_gel_pose(time, np.eye(3), [0., 0., 0.])
        tracker.add_ground_truth_imu_position(time, imu_velocity_world * time)
    tracker._reset_frontend_visual_inertial(np.eye(4), header(1.))
    assert np.allclose(tracker._frontend_vi_state.pose, tracker.gelsight_T_imu)
    assert np.allclose(tracker._frontend_vi_state.velocity, imu_velocity_world)


def test_gt_velocity_central_difference_matches_accelerating_motion():
    tracker = _make_gt_velocity_tracker()
    rotation = np.array(((1., 0., 0.), (0., 0., -1.), (0., 1., 0.)))
    quadratic = np.array((.03, -.02, .01))
    linear = np.array((.01, .02, -.03))
    for t in (.97, .99, 1., 1.02):
        tracker.add_ground_truth_gel_pose(t, rotation, quadratic * t**2 + linear * t)
        tracker.add_ground_truth_imu_position(t, quadratic * t**2 + linear * t)

    tracker._reset_frontend_visual_inertial(np.eye(4), header(1.0))

    assert np.allclose(tracker._frontend_vi_state.velocity,
                       2 * quadratic + linear, atol=1e-12)


def test_gt_velocity_interpolates_native_nonuniform_central_differences():
    tracker = _make_gt_velocity_tracker()
    tracker.initial_ground_truth_world_R_gel = np.eye(3)
    for t in (.98, 1., 1.01, 1.03):
        tracker.add_ground_truth_imu_position(t, (.2 * t**2 + .1 * t, 0., 0.))

    velocity = tracker._first_trial_velocity_from_gt(1.005)

    assert velocity is not None
    assert velocity[0] == pytest.approx(.4 * 1.005 + .1, abs=1e-12)


def test_gt_velocity_waits_for_future_sample_without_moving_initial_anchor():
    tracker = _make_gt_velocity_tracker()
    for t in (.98, .99, 1.):
        tracker.add_ground_truth_gel_pose(t, np.eye(3), (.2 * t**2, 0., 0.))
        tracker.add_ground_truth_imu_position(t, (.2 * t**2, 0., 0.))

    tracker._reset_frontend_visual_inertial(np.eye(4), header(1.0))
    assert tracker._frontend_vi_state is None
    later_pose = np.eye(4)
    later_pose[0, 3] = .003
    tracker._reset_frontend_visual_inertial(later_pose, header(1.02))
    assert tracker._frontend_vi_state is None

    tracker.add_ground_truth_gel_pose(1.01, np.eye(3), (.2 * 1.01**2, 0., 0.))
    assert tracker._frontend_vi_state is None  # GelSight GT cannot supply IMU velocity.
    tracker.add_ground_truth_imu_position(1.01, (.2 * 1.01**2, 0., 0.))

    state = tracker._frontend_vi_state
    assert state is not None
    assert state.timestamp == 1.0
    assert np.array_equal(state.pose, np.eye(4))
    assert np.allclose(state.velocity, (.4, 0., 0.), atol=1e-12)


def test_gt_imu_velocity_uses_the_navigation_timestamp_after_camera_offset():
    tracker = _make_gt_velocity_tracker()
    tracker.image_to_imu_time_offset_s = .005
    for t in (.98, 1., 1.01, 1.03):
        tracker.add_ground_truth_gel_pose(t, np.eye(3), [0., 0., 0.])
        tracker.add_ground_truth_imu_position(t, [.2 * t**2, 0., 0.])

    tracker._reset_frontend_visual_inertial(np.eye(4), header(1.))

    assert tracker._frontend_vi_state.timestamp == pytest.approx(1.005)
    assert tracker._frontend_vi_state.velocity[0] == pytest.approx(.4 * 1.005)


def test_vi_state_uses_public_trial_axes_with_gt_gravity(monkeypatch):
    tracker = make_coasting_tracker()
    tracker.curr_kid = 0
    tracker.imu_config = {
        "gravity_mps2": 9.81,
        "frontend_optimization": True,
        "frontend_gravity_initialization": "ground_truth_initial_pose",
    }
    tracker.frontend_visual_inertial = object()
    tracker.initial_ground_truth_world_R_gel = None
    tracker._ground_truth_gel_rotations = []
    tracker.logger = SimpleNamespace(info=lambda _message: None)
    world_R_gel = np.array(
        ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))
    )
    tracker.add_ground_truth_gel_rotation(1.0, world_R_gel)
    tracker._reset_frontend_visual_inertial(np.eye(4), header(1.0))
    assert np.allclose(tracker._frontend_vi_state.pose[:3, :3], world_R_gel)
    assert np.allclose(tracker._frontend_vi_state.pose[:3, 3], 0.0)
    assert np.allclose(tracker.frontend_visual_inertial.gravity, (0.0, 0.0, -9.81))
    assert np.allclose(tracker.frontend_gravity_for_backend(), (0.0, 0.0, -9.81))

    reference_pose = tracker._frontend_vi_state.pose.copy()
    reference_pose[0, 3] = 0.03
    tracker.trial_T_ref = reference_pose
    tracker._record_tactile_pose(np.eye(4), header(1.5))
    assert np.allclose(tracker._frontend_vi_state.pose, reference_pose)

    relative = np.eye(4)
    relative[1, 3] = 0.004
    observed = []

    predicted_pose = reference_pose.copy()
    predicted_pose[0, 3] += 0.006
    predicted_velocity = np.array([0.02, 0.0, 0.0])
    integration_marker = object()

    def predict(previous, samples, _extrinsic):
        return NavigationState(
            samples[-1].timestamp, predicted_pose.copy(), predicted_velocity.copy(),
            previous.accel_bias.copy(), previous.gyro_bias.copy(),
        ), integration_marker

    def refine(_visual, reference, previous, current, _samples, _extrinsic,
               preintegrated=None):
        assert preintegrated is integration_marker
        observed.append((reference.copy(), previous.pose.copy(), current.pose.copy()))
        assert np.allclose(current.velocity, predicted_velocity)
        result = current.copy()
        result.pose = reference @ relative
        return result

    tracker.frontend_visual_inertial = rm_stub(predict, refine)
    monkeypatch.setattr(tracker_module, "DenseNormalResidual", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        tracker_module, "normalflow_quality", lambda *_args: (1.0, 1.0)
    )
    tracker.ref_frame = object()  # The solved pair does not use the keyframe surface.
    output = tracker._refine_normalflow_with_imu(
        relative, header(2.0), object(), object()
    )
    assert np.allclose(observed[0][0], reference_pose)
    assert np.allclose(observed[0][1], reference_pose)
    assert np.allclose(observed[0][2], predicted_pose)
    assert np.allclose(output, relative)


def test_new_trial_waits_for_its_own_gt_attitude():
    tracker = make_coasting_tracker()
    tracker.imu_config = {
        "gravity_mps2": 9.81,
        "frontend_optimization": True,
        "frontend_gravity_initialization": "ground_truth_initial_pose",
        "gt_trial_attitude_max_age_s": 0.05,
    }
    tracker.frontend_visual_inertial = object()
    tracker.initial_ground_truth_world_R_gel = None
    tracker._ground_truth_gel_rotations = []
    tracker.logger = SimpleNamespace(info=lambda _message: None)
    first_rotation = np.array(
        ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))
    )
    second_rotation = np.array(
        ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))
    )
    tracker.curr_kid = 0
    tracker._frontend_trial_start_timestamp = 1.0
    tracker.add_ground_truth_gel_rotation(1.0, first_rotation)
    tracker._reset_frontend_visual_inertial(np.eye(4), header(1.0))
    assert tracker._frontend_vi_state is not None
    first_gravity = tracker.frontend_visual_inertial.gravity.copy()
    first_pose = tracker._frontend_vi_state.pose.copy()

    tracker.curr_kid = 1
    tracker._trial_index = 1
    tracker._frontend_trial_start_timestamp = 2.0
    tracker._trial_start_world_R_gel = None
    tracker.trial_T_ref = np.eye(4)
    tracker._reset_frontend_visual_inertial(np.eye(4), header(2.0))
    assert tracker._frontend_vi_state is None
    assert tracker.frontend_gravity_for_backend() is None

    tracker.add_ground_truth_gel_rotation(1.98, second_rotation)
    tracker._reset_frontend_visual_inertial(np.eye(4), header(2.1))
    assert tracker._frontend_vi_state is not None
    assert np.allclose(tracker.frontend_visual_inertial.gravity, first_gravity)
    assert not np.allclose(tracker._frontend_vi_state.pose, first_pose)
    assert np.allclose(
        tracker.frontend_visual_inertial.gravity,
        np.array((0.0, 0.0, -9.81)),
    )
    assert np.allclose(
        tracker.frontend_gravity_for_backend(),
        np.array((0.0, 0.0, -9.81)),
    )
    assert np.allclose(
        tracker._frontend_vi_state.pose[:3, :3], second_rotation
    )
    assert np.allclose(tracker._frontend_vi_state.pose[:3, 3], 0.0)
    tracker.add_ground_truth_gel_rotation(2.0, np.eye(3))
    assert np.allclose(
        tracker._trial_initial_gt_rotation(2.0), second_rotation
    )


def test_new_keyframe_in_first_trial_does_not_select_new_gt_gravity():
    tracker = make_coasting_tracker()
    tracker.imu_config = {
        "gravity_mps2": 9.81,
        "frontend_optimization": True,
        "frontend_gravity_initialization": "ground_truth_initial_pose",
    }
    tracker.frontend_visual_inertial = object()
    tracker.initial_ground_truth_world_R_gel = None
    tracker._ground_truth_gel_rotations = []
    tracker.logger = SimpleNamespace(info=lambda _message: None)
    first_rotation = np.array(
        ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))
    )
    tracker.add_ground_truth_gel_rotation(1.0, first_rotation)
    tracker.add_ground_truth_gel_rotation(2.0, np.eye(3))
    tracker._trial_index = 0
    tracker._frontend_trial_start_timestamp = 1.0
    tracker.curr_kid = 3
    tracker._reset_frontend_visual_inertial(np.eye(4), header(2.0))

    assert tracker._frontend_vi_state is not None
    assert np.allclose(tracker.frontend_gravity_for_backend(), (0.0, 0.0, -9.81))
    assert np.allclose(tracker.initial_ground_truth_world_R_gel, first_rotation)


def test_gt_velocity_mode_waits_for_recent_positions_instead_of_using_zero():
    tracker = make_coasting_tracker()
    tracker.curr_kid = 0
    tracker.imu_config = {
        "gravity_mps2": 9.81,
        "frontend_optimization": True,
        "frontend_gravity_initialization": "ground_truth_initial_pose",
        "frontend_first_trial_velocity_from_gt": True,
    }
    tracker.frontend_visual_inertial = object()
    tracker.initial_ground_truth_world_R_gel = None
    tracker._ground_truth_gel_rotations = []
    tracker._ground_truth_gel_positions = []
    tracker.logger = SimpleNamespace(info=lambda _message: None)
    tracker.add_ground_truth_gel_pose(1.0, np.eye(3), (0.0, 0.0, 0.0))

    tracker._reset_frontend_visual_inertial(np.eye(4), header(1.0))

    assert tracker._frontend_vi_state is None


def test_contact_record_preserves_the_jointly_estimated_velocity_and_biases():
    tracker = make_coasting_tracker()
    tracker.frontend_visual_inertial = object()
    tracker._frontend_vi_state = NavigationState(
        1.0, np.eye(4), np.array((0.1, -0.2, 0.3)),
        np.array((0.01, -0.02, 0.03)), np.array((0.001, -0.002, 0.003)),
    )
    tracker.last_tactile_ref_T = np.eye(4)
    tracker.last_tactile_header = header(1.0)

    tracker._record_tactile_pose(np.eye(4), header(2.0))

    state = tracker._frontend_vi_state
    assert state.timestamp == 2.0
    assert np.allclose(state.velocity, (0.1, -0.2, 0.3))
    assert np.allclose(state.accel_bias, (0.01, -0.02, 0.03))
    assert np.allclose(state.gyro_bias, (0.001, -0.002, 0.003))


def test_reference_change_can_preserve_the_frontend_vi_state():
    """A promoted keyframe must not discard the state at that frame."""
    tracker = make_coasting_tracker()
    tracker.frontend_visual_inertial = object()
    state = NavigationState(
        1.0, np.eye(4), np.array((0.1, -0.2, 0.3)),
        np.array((0.01, -0.02, 0.03)), np.array((0.001, -0.002, 0.003)),
    )
    tracker._frontend_vi_state = state
    tracker.last_tactile_ref_T = np.eye(4)
    tracker.last_tactile_header = header(0.5)

    tracker._record_tactile_pose(
        np.eye(4), header(1.0), reference_changed=True,
        preserve_frontend_state=True,
    )

    assert tracker._frontend_vi_state is state
    assert tracker.previous_tactile_ref_T is None
    assert np.allclose(tracker.last_tactile_ref_T, np.eye(4))


def test_simulation_biases_initialize_state_and_anchor_sample_gravity():
    """Configured biases initialize the state; gravity comes from the anchor sample."""
    tracker = make_coasting_tracker()
    tracker.imu_config.update(
        {
            "gravity_mps2": 9.80665,
            "initial_accel_bias_mps2": [0.003, -0.002, 0.004],
            "initial_gyro_bias_rps": [0.0002, -0.0001, 0.00015],
        }
    )
    tracker.imu_buffer = ImuBuffer()
    tracker.imu_buffer.add(
        ImuSample(0.5, (0.003, -0.002, 9.81065), (0.0, 0.0, 0.0))
    )
    tracker.imu_buffer.add(
        ImuSample(1.0, (0.003, -0.002, 9.81065), (0.0, 0.0, 0.0))
    )

    assert tracker._anchor_inertial_state(np.eye(4), header(1.0))
    assert np.allclose(tracker.inertial_state.accel_bias, [0.003, -0.002, 0.004])
    assert np.allclose(tracker.inertial_state.gyro_bias, [0.0002, -0.0001, 0.00015])
    assert np.allclose(tracker.inertial_propagator.gravity, [0.0, 0.0, -9.80665])


def test_contact_poses_defer_imu_initialization_until_contact_is_lost():
    tracker = make_coasting_tracker()
    first_tactile_pose = np.eye(4)
    last_tactile_pose = np.eye(4)
    last_tactile_pose[0, 3] = 0.2

    tracker._record_tactile_pose(
        first_tactile_pose, header(1.0), reference_changed=True
    )
    tracker._record_tactile_pose(last_tactile_pose, header(2.0))

    assert tracker.inertial_state is None
    assert tracker.inertial_propagator is None
    assert tracker._initialize_no_contact_coast()
    assert np.allclose(tracker.inertial_state.position, [0.2, 0.0, 0.0])
    assert np.allclose(tracker.inertial_state.velocity, [0.2, 0.0, 0.0])


def test_imu_ready_with_configured_extrinsic_and_buffer():
    tracker = make_coasting_tracker()
    assert tracker._imu_coasting_ready()

    extrinsic = tracker.gelsight_T_imu
    tracker.gelsight_T_imu = None
    assert not tracker._imu_coasting_ready()
    tracker.gelsight_T_imu = extrinsic

    imu_buffer = tracker.imu_buffer
    tracker.imu_buffer = None
    assert not tracker._imu_coasting_ready()
    tracker.imu_buffer = imu_buffer
    assert tracker._imu_coasting_ready()


def test_contact_failure_is_not_published_or_imu_propagated(
    monkeypatch,
):
    tracker = make_coasting_tracker()
    assert tracker._anchor_inertial_state(np.eye(4), header(1.0))

    class FakeFrame:
        def __init__(self, *_, **__):
            self.is_contacted = True
            self.N = self.C = self.H = self.L = np.zeros((1, 1))

    tracker.bg_images = [object()] * 10
    tracker.recon = SimpleNamespace(
        get_surface_info=lambda *_args, **_kwargs: (None, None, None)
    )
    tracker.contact_mask_config = {"color_dist_threshold": 1, "height_threshold": 1}
    tracker.track_config = {
        "contact_threshold": 1,
        "scr_threshold": 1,
        "ccs_threshold": 1,
    }
    tracker.ppmm = 1.0
    tracker.prev_in_contact_flag = True
    tracker.prev_is_keyframe_flag = True
    tracker.prev_frame = FakeFrame()
    tracker.ref_frame = tracker.prev_frame
    tracker.prev_header = header(1.0)
    tracker.ref_header = header(1.0)
    tracker.curr_fid = 0
    tracker.curr_kid = 4
    tracker.bridge = object()
    tracker.logger = SimpleNamespace(
        info=lambda _message: None, warning=lambda _message: None
    )

    monkeypatch.setattr(tracker_module, "Frame", FakeFrame)
    monkeypatch.setattr(
        tracker,
        "_refine_normalflow_with_imu",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            tracker_module.LoseTrackError()
        ),
    )
    monkeypatch.setattr(
        tracker_module,
        "create_frame_msg",
        lambda _bridge, _frame, **kwargs: kwargs,
    )

    ret, result = tracker.track(
        np.zeros((1, 1, 3), dtype=np.uint8), header(2.0)
    )

    assert not ret
    assert result is None
    assert tracker.curr_kid == 4
    assert tracker.curr_fid == 0
    assert tracker.inertial_state.timestamp == 1.0
    assert tracker.tracking_mode is TrackingMode.TACTILE_HOLD


def test_contact_failure_does_not_propagate_even_with_contact_history(monkeypatch):
    tracker = make_coasting_tracker()
    assert tracker._anchor_inertial_state(np.eye(4), header(1.0))

    class FakeFrame:
        def __init__(self, *_, **__):
            self.is_contacted = True
            self.N = self.C = self.H = self.L = np.zeros((1, 1))

    tracker.bg_images = [object()] * 10
    tracker.recon = SimpleNamespace(
        get_surface_info=lambda *_args, **_kwargs: (None, None, None)
    )
    tracker.contact_mask_config = {"color_dist_threshold": 1, "height_threshold": 1}
    tracker.track_config = {
        "contact_threshold": 1,
        "scr_threshold": 1,
        "ccs_threshold": 1,
    }
    tracker.ppmm = 1.0
    tracker.prev_in_contact_flag = True
    tracker.prev_is_keyframe_flag = True
    tracker.prev_frame = FakeFrame()
    tracker.ref_frame = tracker.prev_frame
    tracker.last_tactile_ref_T = np.eye(4)
    tracker.curr_fid = 0
    tracker.curr_kid = 4
    tracker.bridge = object()
    tracker.logger = SimpleNamespace(
        info=lambda _message: None, warning=lambda _message: None
    )

    monkeypatch.setattr(tracker_module, "Frame", FakeFrame)
    monkeypatch.setattr(
        tracker,
        "_refine_normalflow_with_imu",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            tracker_module.LoseTrackError()
        ),
    )
    monkeypatch.setattr(
        tracker_module,
        "create_frame_msg",
        lambda _bridge, _frame, **kwargs: kwargs,
    )

    ret, result = tracker.track(
        np.zeros((1, 1, 3), dtype=np.uint8), header(2.0)
    )

    assert not ret
    assert result is None
    assert tracker.curr_kid == 4
    assert tracker.curr_fid == 0
    assert tracker.inertial_state.timestamp == 1.0
    assert tracker.tracking_mode is TrackingMode.TACTILE_HOLD


def test_reacquire_does_not_propagate_to_contact_time_before_normal_flow(monkeypatch):
    """Recontact registration must use the last trusted pose, not IMU coasting."""
    tracker = make_coasting_tracker()
    assert tracker._anchor_inertial_state(np.eye(4), header(1.0))

    class FakeFrame:
        def __init__(self, *_, **__):
            self.is_contacted = True
            self.N = self.C = self.H = self.L = np.zeros((1, 1))

    tracker.bg_images = [object()] * 10
    tracker.recon = SimpleNamespace(
        get_surface_info=lambda *_args, **_kwargs: (None, None, None)
    )
    tracker.contact_mask_config = {"color_dist_threshold": 1, "height_threshold": 1}
    tracker.track_config = {
        "contact_threshold": 1,
        "scr_threshold": 1,
        "ccs_threshold": 1,
    }
    tracker.ppmm = 1.0
    tracker.prev_in_contact_flag = False
    tracker.prev_is_keyframe_flag = False
    tracker.prev_frame = FakeFrame()
    tracker.ref_frame = tracker.prev_frame
    tracker.last_tactile_ref_T = np.eye(4)
    tracker.prev_header = header(1.0)
    tracker.ref_header = header(1.0)
    tracker.curr_fid = 0
    tracker.curr_kid = 4
    tracker.bridge = object()
    tracker.logger = SimpleNamespace(info=lambda _message: None)

    captured_initial_poses = []
    captured_state_timestamps = []

    def successful_refinement(ref_T_curr, *_args, **_kwargs):
        captured_initial_poses.append(np.linalg.inv(ref_T_curr))
        captured_state_timestamps.append(tracker.inertial_state.timestamp)
        return np.eye(4)

    tracker.track_config["keyframe_max_interval_s"] = 0.0  # Isolate the original recovery path.
    monkeypatch.setattr(tracker_module, "Frame", FakeFrame)
    monkeypatch.setattr(tracker, "_refine_normalflow_with_imu", successful_refinement)
    monkeypatch.setattr(
        tracker_module,
        "create_frame_msg",
        lambda _bridge, _frame, **kwargs: kwargs,
    )

    ret, (frame_msg, keyframes) = tracker.track(
        np.zeros((1, 1, 3), dtype=np.uint8), header(2.0)
    )

    assert ret
    assert keyframes == []
    assert frame_msg["ref_kid"] == 4
    assert len(captured_initial_poses) == 1
    assert np.allclose(captured_initial_poses[0], np.eye(4), atol=1e-6)
    assert captured_state_timestamps == [1.0]


def test_reacquire_failure_does_not_start_a_new_trial(
    monkeypatch,
):
    """A rejected recontact frame must not reset the tactile trial."""
    tracker = make_coasting_tracker()
    assert tracker._anchor_inertial_state(np.eye(4), header(1.0))

    class FakeFrame:
        def __init__(self, *_, **__):
            self.is_contacted = True
            self.N = self.C = self.H = self.L = np.zeros((1, 1))

    tracker.bg_images = [object()] * 10
    tracker.recon = SimpleNamespace(
        get_surface_info=lambda *_args, **_kwargs: (None, None, None)
    )
    tracker.contact_mask_config = {"color_dist_threshold": 1, "height_threshold": 1}
    tracker.track_config = {
        "contact_threshold": 1,
        "scr_threshold": 1,
        "ccs_threshold": 1,
    }
    tracker.ppmm = 1.0
    tracker.gelsight_T_imu = None
    tracker.prev_in_contact_flag = False
    tracker.prev_is_keyframe_flag = False
    tracker.ref_frame = FakeFrame()
    tracker.prev_frame = FakeFrame()
    tracker.last_tactile_ref_T = np.eye(4)
    tracker.last_tactile_header = header(1.0)
    tracker.prev_header = header(1.0)
    tracker.ref_header = header(1.0)
    tracker.curr_fid = 0
    tracker.curr_kid = 4
    tracker.bridge = object()
    tracker.logger = SimpleNamespace(
        info=lambda _message: None, warning=lambda _message: None
    )

    monkeypatch.setattr(tracker_module, "Frame", FakeFrame)
    monkeypatch.setattr(
        tracker,
        "_refine_normalflow_with_imu",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            tracker_module.LoseTrackError()
        ),
    )
    monkeypatch.setattr(
        tracker_module,
        "create_frame_msg",
        lambda _bridge, _frame, **kwargs: kwargs,
    )
    monkeypatch.setattr(
        tracker_module,
        "create_keyframe_msg",
        lambda _bridge, _frame, **kwargs: kwargs,
    )

    ret, result = tracker.track(
        np.zeros((1, 1, 3), dtype=np.uint8), header(2.0)
    )

    assert not ret
    assert result is None
    assert tracker.curr_kid == 4
    assert tracker.curr_fid == 0
    assert tracker.inertial_state.timestamp == 1.0
    assert tracker.tracking_mode is TrackingMode.TACTILE_HOLD


def test_contact_history_initializes_imu_coasting_only_for_no_contact():
    tracker = make_coasting_tracker()
    tracker.logger = SimpleNamespace(warning=lambda _message: None)
    tracker._record_tactile_pose(np.eye(4), header(1.0), reference_changed=True)

    assert tracker.inertial_state is None
    assert tracker._initialize_no_contact_coast()
    assert tracker._coast_to_header(header(2.0), TrackingMode.NO_CONTACT_COASTING)
    assert tracker.tracking_mode is TrackingMode.NO_CONTACT_COASTING


def test_unavailable_visual_inertial_refinement_rejects_without_tactile_normalflow(monkeypatch):
    tracker = make_coasting_tracker()
    tracker.logger = SimpleNamespace(info=lambda _message: None)
    class FakeFrame:
        N = C = H = L = np.zeros((1, 1))
    with pytest.raises(tracker_module.LoseTrackError, match='vi_unavailable'):
        tracker._refine_normalflow_with_imu(
            np.eye(4, dtype=np.float32), header(2.0), FakeFrame(), FakeFrame())
    assert tracker.frontend_tracking_statistics()['normalflow_only_calls'] == 0


def test_visual_inertial_quality_uses_current_to_reference_warp(monkeypatch):
    tracker = make_coasting_tracker()
    tracker._frontend_vi_state = NavigationState(
        1.0, np.eye(4), np.zeros(3), np.zeros(3), np.zeros(3)
    )
    reference_pose = np.eye(4)
    reference_pose[0, 3] = 0.3
    tracker.trial_T_ref = reference_pose
    tracker._frontend_vi_state.pose = reference_pose.copy()
    refined_pose = np.eye(4)
    refined_pose[0, 3] = 0.4
    refined = NavigationState(
        2.0, refined_pose, np.zeros(3), np.zeros(3), np.zeros(3)
    )
    solver_inputs = []

    def predict(previous, samples, _extrinsic):
        current = previous.copy()
        current.timestamp = samples[-1].timestamp
        return current, object()

    def refine(_visual, reference, previous, current, _samples, _extrinsic,
               preintegrated=None):
        solver_inputs.append((reference.copy(), previous.pose.copy(), current.pose.copy()))
        return refined

    tracker.frontend_visual_inertial = rm_stub(predict, refine)
    monkeypatch.setattr(tracker_module, "DenseNormalResidual", lambda *_args, **_kwargs: object())
    observed_warps = []

    def quality(_reference, _current, curr_T_ref, _ppmm, _samples):
        observed_warps.append(curr_T_ref)
        return 1.0, 1.0

    monkeypatch.setattr(tracker_module, "normalflow_quality", quality)
    tracker.ref_frame = object()  # The solved pair does not use the keyframe surface.
    pose = tracker._refine_normalflow_with_imu(
        np.eye(4), header(2.0), object(), object()
    )

    assert len(observed_warps) == 1
    assert len(solver_inputs) == 1
    assert all(np.allclose(pose, reference_pose) for pose in solver_inputs[0])
    assert np.isclose(observed_warps[0][0, 3], -0.1)
    assert np.isclose(pose[0, 3], 0.1)


def test_failed_contact_frame_promotes_previous_without_retry(monkeypatch):
    tracker = make_coasting_tracker()
    tracker.gelsight_T_imu = None

    class FakeFrame:
        def __init__(self, *_, **__):
            self.is_contacted = True
            self.N = self.C = self.H = self.L = np.zeros((1, 1))

    tracker.bg_images = [object()] * 10
    tracker.recon = SimpleNamespace(
        get_surface_info=lambda *_args, **_kwargs: (None, None, None)
    )
    tracker.contact_mask_config = {"color_dist_threshold": 1, "height_threshold": 1}
    tracker.track_config = {
        "contact_threshold": 1,
        "scr_threshold": 1,
        "ccs_threshold": 1,
    }
    tracker.ppmm = 1.0
    tracker.prev_in_contact_flag = True
    tracker.prev_is_keyframe_flag = False
    tracker.prev_frame = FakeFrame()
    tracker.prev_header = header(1.0)
    tracker.ref_frame = FakeFrame()
    tracker.ref_header = header(1.0)
    tracker.last_tactile_ref_T = None
    tracker.last_tactile_header = None
    tracker.curr_fid = 0
    tracker.curr_kid = 4
    tracker.bridge = object()
    tracker.logger = SimpleNamespace(
        info=lambda _message: None, warning=lambda _message: None
    )

    monkeypatch.setattr(tracker_module, "Frame", FakeFrame)
    monkeypatch.setattr(
        tracker,
        "_refine_normalflow_with_imu",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            tracker_module.LoseTrackError()
        ),
    )
    monkeypatch.setattr(
        tracker_module,
        "create_frame_msg",
        lambda _bridge, _frame, **kwargs: kwargs,
    )
    monkeypatch.setattr(
        tracker_module,
        "create_keyframe_msg",
        lambda _bridge, _frame, **kwargs: kwargs,
    )

    ret, result = tracker.track(
        np.zeros((1, 1, 3), dtype=np.uint8), header(2.0)
    )

    assert ret
    frame_msg, keyframes = result
    assert frame_msg is None
    assert len(keyframes) == 1
    assert keyframes[0]["fid"] == 0
    assert tracker.curr_kid == 5
    assert tracker.curr_fid == 0
    assert tracker.prev_is_keyframe_flag
    assert tracker.tracking_mode is TrackingMode.TACTILE_HOLD


def test_scr_failure_promotes_previous_accepted_frame(monkeypatch):
    tracker = make_coasting_tracker()
    tracker.gelsight_T_imu = None

    class FakeFrame:
        def __init__(self, *_, **__):
            self.is_contacted = True
            self.N = self.C = self.H = self.L = np.zeros((1, 1))

    tracker.bg_images = [object()] * 10
    tracker.recon = SimpleNamespace(
        get_surface_info=lambda *_args, **_kwargs: (None, None, None)
    )
    tracker.contact_mask_config = {"color_dist_threshold": 1, "height_threshold": 1}
    # Short fixture interval; production/default 0.5 s is tested separately.
    tracker.track_config.update(contact_threshold=1, contact_failure_keyframe_interval_s=0.04)
    tracker.prev_in_contact_flag = True
    tracker.prev_is_keyframe_flag = False
    previous_frame = FakeFrame()
    tracker.prev_frame = previous_frame
    tracker.ref_frame = FakeFrame()
    tracker.prev_header = header(1.5)
    tracker.ref_header = header(1.0)
    tracker.ref_T_prev[0, 3] = 0.05
    tracker.curr_fid = 4
    tracker.prev_fid = 4
    tracker.curr_kid = 2
    tracker.bridge = object()
    tracker.logger = SimpleNamespace(info=lambda _message: None)
    monkeypatch.setattr(tracker_module, "Frame", FakeFrame)
    def reject_quality(*args, **kwargs):
        raise tracker_module.QualityGateError(np.eye(4))
    monkeypatch.setattr(tracker, '_refine_normalflow_with_imu', reject_quality)
    seen_frames = []
    monkeypatch.setattr(
        tracker_module, "create_keyframe_msg",
        lambda _bridge, source_frame, **kwargs: (
            seen_frames.append(source_frame) or kwargs
        ),
    )
    observed_frame_sources = []
    monkeypatch.setattr(
        tracker_module, "create_frame_msg",
        lambda _bridge, source_frame, **kwargs: (
            observed_frame_sources.append(source_frame) or kwargs
        ),
    )

    ret, (frame_msg, keyframes) = tracker.track(
        np.zeros((1, 1, 3), dtype=np.uint8), header(2.0)
    )

    assert ret and frame_msg is None
    assert len(keyframes) == 1
    assert seen_frames == [previous_frame]
    assert observed_frame_sources == []
    assert tracker.ref_frame is previous_frame
    assert keyframes[0]["fid"] == 4
    assert keyframes[0]["kid"] == 3
    assert keyframes[0]["header"].stamp.sec == 1
    assert np.isclose(keyframes[0]["trial_T_curr"][0, 3], 0.05)
    assert tracker.curr_fid == 4
    assert np.allclose(tracker.ref_T_prev, np.eye(4))
    assert tracker.tracking_mode is TrackingMode.TACTILE_HOLD


def test_quality_failure_drops_current_when_previous_is_already_keyframe(monkeypatch):
    tracker = make_coasting_tracker()
    tracker.prev_is_keyframe_flag = True
    tracker.prev_frame = tracker.ref_frame = object()
    tracker.prev_header = tracker.ref_header = header(1.0)
    tracker.curr_fid = 5
    tracker.curr_kid = 2
    tracker.logger = SimpleNamespace(info=lambda _message: None)
    state = NavigationState(
        1.0, np.eye(4), np.zeros(3), np.zeros(3), np.zeros(3)
    )
    tracker._frontend_vi_state = state
    monkeypatch.setattr(
        tracker, "_make_keyframe_msg",
        lambda *_args: (_ for _ in ()).throw(AssertionError("duplicate keyframe")),
    )

    assert tracker._promote_previous_frame("Track", "SCR/CCS failure") == (
        False, None
    )
    assert tracker.curr_kid == 2
    assert tracker.curr_fid == 4
    assert tracker._frontend_vi_state is state


def test_quality_failure_after_coast_uses_last_tactile_pose_and_fid(monkeypatch):
    tracker = make_coasting_tracker()
    tracker.prev_is_keyframe_flag = False
    tracker.prev_in_contact_flag = False
    tracker.prev_frame = previous_frame = object()
    tracker.prev_header = header(1.0)
    tracker.prev_fid = 4
    tracker.curr_fid = 7
    tracker.curr_kid = 2
    tracker.logger = SimpleNamespace(info=lambda _message: None)
    tracker.ref_T_prev[0, 3] = 0.9  # Coasted prediction, not tactile pose.
    tracker.last_tactile_ref_T = np.eye(4)
    tracker.last_tactile_ref_T[0, 3] = 0.1
    state = NavigationState(
        1.0, tracker.last_tactile_ref_T.copy(),
        np.zeros(3), np.zeros(3), np.zeros(3),
    )
    tracker._frontend_vi_state = state
    monkeypatch.setattr(
        tracker, "_make_keyframe_msg",
        lambda frame, kid, fid, is_new_trial, pose, source_header: (
            frame, kid, fid, is_new_trial, pose.copy(), source_header
        ),
    )

    ret, (frame_msg, keyframes) = tracker._promote_previous_frame(
        "Track", "SCR/CCS failure"
    )

    assert ret and frame_msg is None
    assert keyframes[0][0] is previous_frame
    assert keyframes[0][2] == 4
    assert np.isclose(keyframes[0][4][0, 3], 0.1)
    assert tracker.curr_fid == 6
    assert tracker._frontend_vi_state is state
    assert tracker._frontend_vi_state.timestamp == 1.0


def test_ccs_failure_promotes_previous_and_keeps_imu_interval(monkeypatch):
    tracker = make_coasting_tracker()

    class FakeFrame:
        def __init__(self, *_, **__):
            self.is_contacted = True
            self.N = self.C = self.H = self.L = np.zeros((1, 1))

    tracker.bg_images = [object()] * 10
    tracker.recon = SimpleNamespace(
        get_surface_info=lambda *_args, **_kwargs: (None, None, None)
    )
    tracker.contact_mask_config = {"color_dist_threshold": 1, "height_threshold": 1}
    # Short fixture interval; production/default 0.5 s is tested separately.
    tracker.track_config.update(contact_threshold=1, contact_failure_keyframe_interval_s=0.04)
    tracker.prev_in_contact_flag = True
    tracker.prev_is_keyframe_flag = False
    previous_frame = FakeFrame()
    tracker.prev_frame = previous_frame
    tracker.ref_frame = FakeFrame()
    tracker.prev_header = tracker.ref_header = header(1.0)
    tracker.ref_T_prev[0, 3] = 0.03
    tracker.curr_fid = 4
    tracker.prev_fid = 4
    tracker.curr_kid = 2
    tracker.bridge = object()
    tracker.logger = SimpleNamespace(info=lambda _message: None)
    accepted_pose = np.eye(4)
    accepted_pose[0, 3] = 0.03
    accepted = NavigationState(
        1.0, accepted_pose, np.array([0.01, 0, 0]), np.zeros(3), np.zeros(3)
    )
    tracker._frontend_vi_state = accepted
    refined_pose = np.eye(4)
    refined_pose[0, 3] = 0.14
    refined = NavigationState(
        2.0, refined_pose, np.array([0.02, 0, 0]), np.zeros(3), np.zeros(3)
    )
    intervals = []

    def predict(previous, samples, _extrinsic):
        current = previous.copy()
        current.timestamp = samples[-1].timestamp
        return current, object()

    def refine(_visual, _reference, previous, _current, samples, _extrinsic,
               preintegrated=None):
        intervals.append((previous.timestamp, [sample.timestamp for sample in samples]))
        if len(intervals) == 1:
            return refined
        pose = np.eye(4)
        pose[0, 3] = 0.05
        return NavigationState(
            3.0, pose, previous.velocity.copy(),
            previous.accel_bias.copy(), previous.gyro_bias.copy(),
        )

    tracker.frontend_visual_inertial = rm_stub(predict, refine)
    tracker.track_config["keyframe_max_interval_s"] = 0.0  # Isolate the original recovery path.
    monkeypatch.setattr(tracker_module, "Frame", FakeFrame)
    monkeypatch.setattr(tracker_module, "DenseNormalResidual", lambda *_args, **_kwargs: object())
    qualities = iter(((0.9, 0.2), (0.9, 0.9)))
    monkeypatch.setattr(
        tracker_module, "normalflow_quality", lambda *_args: next(qualities)
    )
    monkeypatch.setattr(
        tracker_module, "create_keyframe_msg",
        lambda _bridge, _frame, **kwargs: kwargs,
    )
    monkeypatch.setattr(
        tracker_module, "create_frame_msg",
        lambda _bridge, _frame, **kwargs: kwargs,
    )

    ret, (frame_msg, keyframes) = tracker.track(
        np.zeros((1, 1, 3), dtype=np.uint8), header(2.0)
    )

    assert ret and frame_msg is None
    assert len(keyframes) == 1
    assert keyframes[0]["fid"] == 4
    assert np.isclose(keyframes[0]["trial_T_curr"][0, 3], 0.03)
    assert tracker._frontend_vi_state is accepted
    assert tracker._frontend_vi_state.timestamp == 1.0
    assert np.allclose(tracker._frontend_vi_state.velocity, [0.01, 0, 0])
    assert tracker.ref_frame is previous_frame
    assert tracker.curr_fid == 4

    tracker.imu_buffer.add(
        ImuSample(3.0, (0.0, 0.0, 9.81), (0.0, 0.0, 0.0))
    )
    ret, (frame_msg, keyframes) = tracker.track(
        np.zeros((1, 1, 3), dtype=np.uint8), header(3.0)
    )

    assert ret and keyframes == []
    assert frame_msg["fid"] == 5
    assert frame_msg["ref_kid"] == 3
    assert np.isclose(frame_msg["ref_T_curr"][0, 3], 0.02)
    assert intervals == [(1.0, [1.0, 2.0]), (1.0, [1.0, 2.0, 3.0])]


def test_quality_failure_promotes_previous_then_tracks_next_frame(monkeypatch):
    tracker = make_coasting_tracker()
    tracker.gelsight_T_imu = None

    class FakeFrame:
        def __init__(self, *_, **__):
            self.is_contacted = True
            self.N = self.C = self.H = self.L = np.zeros((1, 1))

    tracker.bg_images = [object()] * 10
    tracker.recon = SimpleNamespace(
        get_surface_info=lambda *_args, **_kwargs: (None, None, None)
    )
    tracker.contact_mask_config = {"color_dist_threshold": 1, "height_threshold": 1}
    tracker.track_config = {
        "contact_threshold": 1,
        "scr_threshold": 1,
        "ccs_threshold": 1,
    }
    tracker.prev_in_contact_flag = True
    tracker.prev_is_keyframe_flag = False
    previous_frame = FakeFrame()
    tracker.prev_frame = previous_frame
    tracker.prev_header = header(1.0)
    tracker.ref_frame = FakeFrame()
    tracker.ref_header = header(0.5)
    tracker.ref_T_prev = np.eye(4, dtype=np.float32)
    tracker.ref_T_prev[0, 3] = 0.1
    tracker._frontend_vi_state = NavigationState(
        1.0, tracker.ref_T_prev.copy(), np.array([0.01, 0.0, 0.0]),
        np.zeros(3), np.zeros(3),
    )
    tracker.frontend_visual_inertial = SimpleNamespace(
        gravity=np.array([0.0, 0.0, -9.81]), set_reference=lambda *_args: None)
    tracker.gelsight_T_imu = np.eye(4)
    tracker.curr_fid = 7
    tracker.curr_kid = 4
    tracker.bridge = object()
    tracker.logger = SimpleNamespace(
        info=lambda _message: None, warning=lambda _message: None
    )

    tracker.track_config["keyframe_max_interval_s"] = 0.0  # Isolate the original recovery path.
    monkeypatch.setattr(tracker_module, "Frame", FakeFrame)
    retracked_pose = np.eye(4, dtype=np.float32)
    retracked_pose[1, 3] = 0.02
    refine_calls = []

    def refine(*args, **kwargs):
        refine_calls.append((args, kwargs))
        if len(refine_calls) == 1:
            raise tracker_module.QualityGateError(np.eye(4))
        return retracked_pose

    monkeypatch.setattr(tracker, "_refine_normalflow_with_imu", refine)
    monkeypatch.setattr(
        tracker_module,
        "create_frame_msg",
        lambda _bridge, _frame, **kwargs: kwargs,
    )
    created_keyframe_frames = []
    monkeypatch.setattr(
        tracker_module,
        "create_keyframe_msg",
        lambda _bridge, keyframe, **kwargs: (
            created_keyframe_frames.append(keyframe) or kwargs
        ),
    )

    ret, (frame_msg, keyframe_msgs) = tracker.track(
        np.zeros((1, 1, 3), dtype=np.uint8), header(2.0)
    )

    assert ret
    assert len(keyframe_msgs) == 1
    assert keyframe_msgs[0]["kid"] == 5
    assert keyframe_msgs[0]["fid"] == 7
    assert created_keyframe_frames == [previous_frame]
    assert np.allclose(keyframe_msgs[0]["trial_T_curr"], [
        [1.0, 0.0, 0.0, 0.1],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    assert frame_msg is None
    assert tracker.ref_frame is previous_frame
    assert np.allclose(tracker.trial_T_ref, keyframe_msgs[0]["trial_T_curr"])
    assert tracker.prev_is_keyframe_flag is True
    assert tracker.curr_fid == 7
    assert tracker.tracking_mode is TrackingMode.TACTILE_HOLD
    assert len(refine_calls) == 1
    assert np.allclose(tracker._frontend_vi_state.pose, keyframe_msgs[0]["trial_T_curr"])
    assert np.allclose(tracker._frontend_vi_state.velocity, [0.01, 0.0, 0.0])

    ret, (frame_msg, keyframe_msgs) = tracker.track(
        np.zeros((1, 1, 3), dtype=np.uint8), header(3.0)
    )

    assert ret
    assert keyframe_msgs == []
    assert frame_msg["fid"] == 8
    assert frame_msg["ref_kid"] == 5
    assert np.allclose(frame_msg["ref_T_curr"], retracked_pose)
    assert tracker.prev_is_keyframe_flag is False
    assert tracker.tracking_mode is TrackingMode.CONTACT_TRACKING
    assert len(refine_calls) == 2
    assert refine_calls[1][0][2] is previous_frame
    assert refine_calls[1][0][3] is not refine_calls[0][0][3]


def test_promoted_keyframe_preserves_trial_frame_vi_state():
    tracker = make_coasting_tracker()
    old_ref_T_new_ref = np.eye(4)
    old_ref_T_new_ref[:3, :3] = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    old_ref_T_new_ref[0, 3] = 0.1
    tracker._frontend_vi_state = NavigationState(
        1.0, old_ref_T_new_ref.copy(), np.array([1.0, 0.0, 0.0]),
        np.array([0.01, 0.02, 0.03]), np.array([0.001, 0.002, 0.003]),
    )
    tracker.frontend_visual_inertial = SimpleNamespace(
        gravity=np.array([0.0, -9.81, 0.0])
    )

    tracker.trial_T_ref = old_ref_T_new_ref.copy()
    tracker._record_tactile_pose(
        np.eye(4), header(1.0), reference_changed=True,
        preserve_frontend_state=True,
    )

    assert np.allclose(tracker._frontend_vi_state.pose, old_ref_T_new_ref)
    assert np.allclose(tracker._frontend_vi_state.velocity, [1.0, 0.0, 0.0])
    assert np.allclose(tracker.frontend_visual_inertial.gravity, [0.0, -9.81, 0.0])
    assert np.allclose(tracker._frontend_vi_state.accel_bias, [0.01, 0.02, 0.03])
    assert np.allclose(tracker._frontend_vi_state.gyro_bias, [0.001, 0.002, 0.003])


def test_failure_after_keyframe_promotion_keeps_committed_navigation_prior(monkeypatch):
    tracker = make_coasting_tracker()
    accepted = NavigationState(1.0, np.eye(4), np.zeros(3), np.zeros(3), np.zeros(3))
    tracker._frontend_vi_state = accepted
    solver = create_frontend_solver({"gtsam_vi_max_iterations": 1})
    tracker.frontend_visual_inertial = solver
    solver.marginal_prior = NavigationPrior.initial(accepted, solver.prior_sigmas)
    tracker.prev_frame = tracker.ref_frame = object()
    tracker.prev_header = tracker.ref_header = header(1.0)
    tracker._sync_reference_prior()
    accepted_prior = solver.marginal_prior
    tracker.prev_is_keyframe_flag = False
    tracker.prev_in_contact_flag = True
    tracker.curr_fid = 2
    tracker.prev_fid = 1
    tracker.curr_kid = 0
    tracker.bridge = object()
    tracker.logger = SimpleNamespace(info=lambda _: None)

    class FlatVisual:
        def __len__(self):
            return 10

        def evaluate(self, _reference, _pose):
            return np.zeros(30), np.zeros((30, 6)), np.zeros((30, 6))

    monkeypatch.setattr(tracker_module, "DenseNormalResidual", lambda *a, **k: FlatVisual())
    qualities = iter(((0.2, 0.2), (0.9, 0.2), (0.9, 0.9)))
    monkeypatch.setattr(tracker_module, "normalflow_quality", lambda *a: next(qualities))
    monkeypatch.setattr(tracker_module, "create_keyframe_msg", lambda *a, **k: k)

    with pytest.raises(tracker_module.QualityGateError):
        tracker._refine_normalflow_with_imu(np.eye(4), header(2.0), object(), object())
    assert solver.marginal_prior is accepted_prior
    assert solver.pending_marginal_prior is None
    promoted, _ = tracker._promote_previous_frame("test", "SCR/CCS failure")
    assert promoted and tracker.prev_is_keyframe_flag
    assert tracker._frontend_vi_state is accepted
    assert solver.marginal_prior is accepted_prior

    tracker.imu_buffer.add(ImuSample(3.0, (0, 0, 9.81), (0, 0, 0)))
    with pytest.raises(tracker_module.QualityGateError):
        tracker._refine_normalflow_with_imu(np.eye(4), header(3.0), object(), object())
    promoted, _ = tracker._promote_previous_frame("test", "SCR/CCS failure")
    assert not promoted
    assert tracker._frontend_vi_state is accepted
    assert solver.marginal_prior is accepted_prior
    assert solver.pending_marginal_prior is None

    tracker.imu_buffer.add(ImuSample(4.0, (0, 0, 9.81), (0, 0, 0)))
    tracker._refine_normalflow_with_imu(np.eye(4), header(4.0), object(), object())
    assert solver.marginal_prior.anchor.timestamp == 4.0
    assert tracker._frontend_vi_state.timestamp == 4.0


def make_failing_contact_tracker(monkeypatch, already_keyframe=False,
                                 initialize_from_imu=True):
    """Keep real IMU prediction and replace only surface reconstruction."""
    tracker = make_coasting_tracker()

    class ContactFrame:
        def __init__(self, *args, **kwargs):
            self.is_contacted = True
            self.N = self.C = self.H = self.L = np.zeros((1, 1))

    monkeypatch.setattr(tracker_module, "Frame", ContactFrame)
    monkeypatch.setattr(tracker_module, "create_frame_msg", lambda *a, **k: k)
    monkeypatch.setattr(tracker_module, "create_keyframe_msg", lambda *a, **k: k)
    tracker.bg_images = [object()] * 10
    tracker.recon = SimpleNamespace(get_surface_info=lambda *a, **k: (None, None, None))
    tracker.contact_mask_config = {"color_dist_threshold": 1, "height_threshold": 1}
    # Short fixture interval; production/default 0.5 s is tested separately.
    tracker.track_config.update(contact_threshold=1, contact_failure_keyframe_interval_s=0.04)
    tracker.prev_in_contact_flag = True
    tracker.prev_is_keyframe_flag = already_keyframe
    tracker.prev_frame = ContactFrame()
    tracker.ref_frame = tracker.prev_frame if already_keyframe else ContactFrame()
    tracker.prev_header = tracker.ref_header = header(1.0)
    tracker.curr_fid = tracker.prev_fid = 4
    tracker.curr_kid = 2
    tracker.bridge = object()
    tracker.logger = SimpleNamespace(info=lambda _: None)
    # Use a nonidentity trial pose, a velocity, and nonzero sensor biases.
    pose = np.eye(4)
    pose[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    pose[:3, 3] = [0.1, -0.2, 0.3]
    tracker.trial_T_ref = pose.copy()
    accepted = NavigationState(
        1.0, pose, np.array([0.1, 0.02, 0]),
        np.array([0.02, 0.03, 0.04]), np.array([0.01, 0.02, 0.03]),
    )
    tracker._frontend_vi_state = accepted
    solver = create_frontend_solver({
        "frontend_initialize_from_imu_preintegration": initialize_from_imu,
    })
    solver.marginal_prior = NavigationPrior.initial(accepted, solver.prior_sigmas)
    # Retain the reference keyframe pose (the accepted state at the reference time).
    solver.set_reference(accepted.timestamp, accepted)
    tracker.frontend_visual_inertial = solver
    tracker.imu_buffer = ImuBuffer()
    for time in np.linspace(1.0, 1.6, 31):
        tracker.imu_buffer.add(ImuSample(
            time, np.array([0, 0, 9.81]) + accepted.accel_bias,
            np.array([0, 0, 0.4]) + accepted.gyro_bias,
        ))
    return tracker, accepted


@pytest.mark.parametrize("failure", ["quality", "registration"])
@pytest.mark.parametrize("already_keyframe", [False, True])
@pytest.mark.parametrize("initialize_from_imu", [False, True])
def test_quality_failure_waits_but_registration_failure_recovers_immediately(
        monkeypatch, failure, already_keyframe, initialize_from_imu):
    tracker, accepted = make_failing_contact_tracker(
        monkeypatch, already_keyframe, initialize_from_imu,
    )
    solver = tracker.frontend_visual_inertial
    prior = solver.marginal_prior
    calls = []

    def reject(*args, **kwargs):
        calls.append(args[3])
        if failure == "quality":
            bad_pose = np.eye(4)
            bad_pose[0, 3] = 100.0
            raise tracker_module.QualityGateError(bad_pose)
        raise tracker_module.LoseTrackError()

    monkeypatch.setattr(tracker, "_refine_normalflow_with_imu", reject)
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    first_ret, first_result = tracker.track(image, header(1.04))
    if failure == "quality":
        assert first_ret is (not already_keyframe)
        if first_ret:
            assert first_result[0] is None
            assert first_result[1][0]["fid"] == 4
        assert tracker._frontend_vi_state is accepted
        assert solver.marginal_prior is prior
        ret, result = tracker.track(image, header(1.08))
        recovery_time = 1.08
        expected_kid = 3 if already_keyframe else 4
    else:
        ret, result = first_ret, first_result
        recovery_time = 1.04
        expected_kid = 3
        assert tracker._contact_failure_since is None
    assert ret
    frame_msg, keyframes = result
    assert len(keyframes) == 1
    keyframe = keyframes[0]
    assert keyframe["fid"] == frame_msg["fid"] == 5
    assert keyframe["kid"] == frame_msg["ref_kid"] == expected_kid
    assert keyframe["is_new_trial"] is False
    assert keyframe["header"] is frame_msg["header"]
    assert tracker._imu_timestamp(keyframe["header"]) == pytest.approx(recovery_time)
    # Constant gyro rotates by 0.4 * duration rad. Bias-corrected acceleration
    # cancels gravity. Translation therefore follows the accepted velocity.
    duration = recovery_time - accepted.timestamp
    angle = 0.4 * duration
    delta_rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0], [0, 0, 1],
    ])
    expected = accepted.pose.copy()
    expected[:3, :3] = accepted.pose[:3, :3] @ delta_rotation
    expected[:3, 3] += accepted.velocity * duration
    assert np.allclose(keyframe["trial_T_curr"], expected, atol=1e-8)
    assert np.allclose(frame_msg["ref_T_curr"], np.eye(4))
    assert np.allclose(tracker.trial_T_ref, expected)
    assert tracker.prev_frame is tracker.ref_frame is calls[-1]
    assert tracker.prev_fid == tracker.curr_fid == 5
    assert tracker.prev_header is tracker.ref_header is frame_msg["header"]
    assert tracker.prev_is_keyframe_flag and tracker.prev_in_contact_flag
    assert tracker.tracking_mode is TrackingMode.IMU_ONLY
    current = tracker._frontend_vi_state
    assert current.timestamp == pytest.approx(recovery_time)
    assert np.allclose(current.pose, expected)
    assert np.allclose(current.velocity, accepted.velocity)
    assert np.array_equal(current.accel_bias, accepted.accel_bias)
    assert np.array_equal(current.gyro_bias, accepted.gyro_bias)
    assert solver.initialize_from_imu_preintegration is initialize_from_imu
    assert solver.marginal_prior is not prior
    assert solver.marginal_prior.matches(current)
    assert solver.pending_marginal_prior is None


def test_repeated_quality_failures_refresh_at_each_elapsed_interval(monkeypatch):
    tracker, _ = make_failing_contact_tracker(monkeypatch)
    monkeypatch.setattr(
        tracker, "_refine_normalflow_with_imu",
        lambda *a, **k: (_ for _ in ()).throw(tracker_module.QualityGateError(np.eye(4))),
    )
    dropped = 0
    recoveries = 0
    for time in np.linspace(1.04, 1.4, 10):
        ret, result = tracker.track(np.zeros((1, 1, 3), dtype=np.uint8), header(time))
        if ret and result[0] is not None:
            dropped = 0
            recoveries += 1
            assert result[1][0]["header"] is result[0]["header"]
        else:
            dropped += 1
        assert dropped < 3, "three successive inputs were rejected"
    assert recoveries == 9


def test_imu_recovery_rejects_stale_tail_samples(monkeypatch):
    tracker, accepted = make_failing_contact_tracker(monkeypatch, already_keyframe=True)
    prior = tracker.frontend_visual_inertial.marginal_prior
    tracker.imu_buffer = ImuBuffer()
    for time in (1.0, 1.02, 1.04):
        tracker.imu_buffer.add(ImuSample(time, (0, 0, 9.81), (0, 0, 0)))
    monkeypatch.setattr(
        tracker, "_refine_normalflow_with_imu",
        lambda *a, **k: (_ for _ in ()).throw(tracker_module.LoseTrackError()),
    )
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    assert tracker.track(image, header(1.08)) == (False, None)
    assert tracker._frontend_vi_state is accepted
    assert tracker.frontend_visual_inertial.marginal_prior is prior
    assert tracker.curr_fid == 4
    assert tracker.ref_header.stamp.sec == 1


def test_imu_recovery_before_next_imu_callback_holds_one_sample_period(monkeypatch):
    tracker, accepted = make_failing_contact_tracker(monkeypatch)
    tracker.imu_buffer = ImuBuffer()

    def add_sample(time):
        tracker.imu_buffer.add(ImuSample(
            time, np.array([0, 0, 9.81]) + accepted.accel_bias,
            np.array([0, 0, 0.4]) + accepted.gyro_bias,
        ))

    add_sample(1.0)
    add_sample(1.02)
    monkeypatch.setattr(
        tracker, "_refine_normalflow_with_imu",
        lambda *a, **k: (_ for _ in ()).throw(tracker_module.LoseTrackError()),
    )
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    tracker.track(image, header(1.03))
    add_sample(1.04)
    add_sample(1.06)
    buffered = tracker.imu_buffer.samples_between(1.0, 1.07)
    ret, (frame_msg, keyframes) = tracker.track(image, header(1.07))
    assert ret and frame_msg is not None and len(keyframes) == 1
    current = tracker._frontend_vi_state
    assert current.timestamp == pytest.approx(1.07)
    assert np.allclose(current.pose[:3, 3], accepted.pose[:3, 3] + accepted.velocity * 0.07)
    angle = np.arctan2(current.pose[1, 0], current.pose[0, 0])
    assert angle == pytest.approx(np.pi / 2 + 0.4 * 0.07)
    assert tracker.imu_buffer.samples_between(1.0, 1.07) == buffered
    assert tracker.imu_buffer.sample_at(1.07) is None
    add_sample(1.08)
    assert tracker.imu_buffer.sample_at(current.timestamp) is not None


def test_tactile_success_resets_failure_sequence(monkeypatch):
    tracker, _ = make_failing_contact_tracker(monkeypatch)
    solver = tracker.frontend_visual_inertial
    outcomes = iter((False, True, False, False))

    def refine(_ref_T_curr, target_header, *args, **kwargs):
        if not next(outcomes):
            raise tracker_module.QualityGateError(np.eye(4))
        # Accept the IMU prediction at the target time with its prior.
        previous = tracker._frontend_vi_state
        samples = tracker.imu_buffer.samples_covering(
            previous.timestamp, tracker._imu_timestamp(target_header))
        current, pim = solver.predict_current(
            previous, samples, tracker.gelsight_T_imu, force=True)
        solver.marginal_prior = solver.propagate_prior(previous, current, pim)
        tracker._frontend_vi_state = current
        return np.eye(4, dtype=np.float32)

    monkeypatch.setattr(tracker, "_refine_normalflow_with_imu", refine)
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    tracker.track(image, header(1.04))
    ret, result = tracker.track(image, header(1.08))
    assert ret and result[0] is not None and result[1] == []
    ret, result = tracker.track(image, header(1.12))
    assert ret and result[0] is None
    assert result[1][0]["header"].stamp.nanosec == header(1.08).stamp.nanosec


def test_quality_recovery_then_visual_solve_uses_predicted_navigation_anchor(monkeypatch):
    tracker, accepted = make_failing_contact_tracker(monkeypatch)
    solver = tracker.frontend_visual_inertial
    original_refine = solver.refine_current
    anchors = []

    class FlatVisual:
        def __len__(self):
            return 10

        def evaluate(self, _reference, _pose):
            return np.zeros(30), np.zeros((30, 6)), np.zeros((30, 6))

    def refine(visual, reference, previous, current, samples, extrinsic,
               preintegrated=None):
        anchors.append(previous.timestamp)
        if len(anchors) <= 2:
            solver.last_reference_pose = imu_pose(reference, extrinsic)
            rejected = current.copy()
            rejected.pose[0, 3] = 100.0
            return rejected
        return original_refine(
            visual, reference, previous, current, samples, extrinsic,
            preintegrated=preintegrated,
        )

    monkeypatch.setattr(solver, "refine_current", refine)
    monkeypatch.setattr(tracker_module, "DenseNormalResidual", lambda *a, **k: FlatVisual())
    qualities = iter(((0.1, 0.9), (0.9, 0.1), (0.9, 0.9)))
    monkeypatch.setattr(tracker_module, "normalflow_quality", lambda *a: next(qualities))
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    tracker.track(image, header(1.04))
    assert tracker._frontend_vi_state is accepted
    ret, (frame_msg, keyframes) = tracker.track(image, header(1.08))
    assert ret and frame_msg is not None and len(keyframes) == 1
    recovery_kid = tracker.curr_kid
    recovery_pose = tracker.trial_T_ref.copy()
    assert recovery_pose[0, 3] < 1.0
    assert tracker.latest_normal_correspondences["status"] == (
        "IMU prediction: normal-flow rejected"
    )
    assert tracker.latest_normal_correspondences["current_xy"].shape == (0, 2)
    ret, (frame_msg, keyframes) = tracker.track(image, header(1.12))
    assert ret and not keyframes
    assert frame_msg["ref_kid"] == tracker.curr_kid == recovery_kid
    assert anchors == pytest.approx([1.0, 1.0, 1.08])
    assert frame_msg["fid"] == 6
    assert solver.marginal_prior.matches(tracker._frontend_vi_state)
    assert solver.marginal_prior.anchor.timestamp == pytest.approx(1.12)
    assert np.allclose(
        tracker.trial_T_ref @ frame_msg["ref_T_curr"], tracker._frontend_vi_state.pose,
    )


def test_reacquire_registration_failure_recovers_with_imu_immediately(monkeypatch):
    tracker, _ = make_failing_contact_tracker(monkeypatch)
    tracker.prev_in_contact_flag = False
    tracker.inertial_state = SimpleNamespace(timestamp=1.0)
    tracker.last_tactile_ref_T = np.eye(4)
    calls = []

    def reject(*args, **kwargs):
        calls.append(args[3])
        raise tracker_module.LoseTrackError()

    monkeypatch.setattr(tracker, "_refine_normalflow_with_imu", reject)
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    ret, (frame_msg, keyframes) = tracker.track(image, header(1.04))
    assert ret and frame_msg["tracking_status"] == "imu_only"
    assert len(keyframes) == 1 and not keyframes[0]["tactile_valid"]
    assert len(calls) == 1  # Keyframe only; no retry on an older tactile frame.
    assert tracker._consecutive_tracking_failures == 0
    assert tracker._contact_failure_since is None
    assert tracker.curr_kid == 3
    ret, (frame_msg, keyframes) = tracker.track(image, header(1.08))
    assert ret and frame_msg is not None and len(keyframes) == 1
    assert len(calls) == 2
    assert keyframes[0]["is_new_trial"] is False
    assert tracker.ref_frame is calls[-1]
    assert tracker._consecutive_tracking_failures == 0


def test_no_contact_resets_contact_failure_sequence(monkeypatch):
    tracker, _ = make_failing_contact_tracker(monkeypatch)
    monkeypatch.setattr(
        tracker, "_refine_normalflow_with_imu",
        lambda *a, **k: (_ for _ in ()).throw(tracker_module.QualityGateError(np.eye(4))),
    )
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    tracker.track(image, header(1.04))
    assert tracker._consecutive_tracking_failures == 1
    monkeypatch.setattr(
        tracker_module, "Frame", lambda *a, **k: SimpleNamespace(is_contacted=False),
    )
    monkeypatch.setattr(tracker, "_initialize_no_contact_coast", lambda: False)
    assert tracker.track(image, header(1.08)) == (False, None)
    assert tracker._consecutive_tracking_failures == 0


@pytest.mark.parametrize("initialize_from_imu", [False, True])
def test_frontend_no_contact_preserves_trial_state_after_reference_reset(initialize_from_imu):
    tracker = make_coasting_tracker()
    tracker.logger = SimpleNamespace(warning=lambda _: None)
    tracker.imu_config["max_coast_duration_s"] = 0.4
    tracker.gelsight_T_imu = np.diag([1., -1., -1., 1.])
    tracker.gelsight_T_imu[2, 3] = -0.04
    tracker.trial_T_ref[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    tracker.trial_T_ref[:3, 3] = [0.3, -0.2, 0.1]
    local_pose = np.eye(4)
    local_pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    local_pose[:3, 3] = [0.02, -0.01, 0.03]
    accepted = NavigationState(
        1.0, tracker.trial_T_ref @ local_pose @ tracker.gelsight_T_imu,
        np.array([0.03, -0.04, 0.02]),
        np.array([0.003, -0.002, 0.004]), np.array([0.0002, -0.0001, 0.00015]),
    )
    gravity = np.array([1.3, -2.1, -9.4])
    acceleration_trial = np.array([0.2, -0.1, 0.05])
    tracker.frontend_visual_inertial = create_frontend_solver({
        "frontend_initialize_from_imu_preintegration": initialize_from_imu,
    }, gravity=gravity)
    solver = tracker.frontend_visual_inertial
    solver.marginal_prior = NavigationPrior.initial(accepted, solver.prior_sigmas)
    prior = solver.marginal_prior
    tracker._frontend_vi_state = accepted
    tracker.imu_buffer = ImuBuffer()
    acceleration_imu = accepted.pose[:3, :3].T @ (acceleration_trial - gravity) + accepted.accel_bias
    for time in (1., 1.125, 1.25, 1.375, 1.5):
        tracker.imu_buffer.add(ImuSample(time, acceleration_imu, accepted.gyro_bias))
    tracker.ref_T_prev = local_pose.astype(np.float32)
    tracker._record_tactile_pose(
        local_pose, header(1.), reference_changed=True, preserve_frontend_state=True,
    )
    assert tracker.previous_tactile_ref_T is None
    assert tracker._initialize_no_contact_coast()
    # An image before a new IMU measurement must hold the anchor pose.
    assert tracker._coast_to_header(header(1.0625), TrackingMode.NO_CONTACT_COASTING)
    assert np.allclose(tracker.ref_T_prev, local_pose, atol=1e-7)
    for time in (1.125, 1.25, 1.375):
        assert tracker._initialize_no_contact_coast()  # Must not reset the cursor.
        assert tracker._coast_to_header(header(time), TrackingMode.NO_CONTACT_COASTING)
        dt = time - accepted.timestamp
        expected = accepted.pose.copy()
        expected[:3, 3] += accepted.velocity * dt + 0.5 * acceleration_trial * dt**2
        assert np.allclose(tracker.trial_T_ref @ tracker.ref_T_prev,
                           expected @ np.linalg.inv(tracker.gelsight_T_imu), atol=1e-7)
        predicted = tracker._frontend_coast_state
        assert predicted.timestamp == time
        assert np.allclose(predicted.pose, expected, atol=1e-10)
        assert np.allclose(predicted.velocity, accepted.velocity + acceleration_trial * dt)
        assert np.array_equal(predicted.accel_bias, accepted.accel_bias)
        assert np.array_equal(predicted.gyro_bias, accepted.gyro_bias)
    assert tracker._frontend_vi_state is accepted
    assert solver.marginal_prior is prior
    assert solver.initialize_from_imu_preintegration is initialize_from_imu
    assert np.array_equal(solver.gravity, gravity)
    assert len(tracker.imu_buffer.samples_covering(1., 1.375)) == 4
    assert not tracker._coast_to_header(header(1.5), TrackingMode.NO_CONTACT_COASTING)
    assert tracker._frontend_coast_state.timestamp == 1.375


def test_frontend_no_contact_waits_for_frontend_initialization():
    tracker = make_coasting_tracker()
    tracker.frontend_visual_inertial = create_frontend_solver({})
    tracker._frontend_vi_state = None
    tracker.last_tactile_ref_T = np.eye(4)
    tracker.last_tactile_header = header(1.)
    assert not tracker._initialize_no_contact_coast()
    assert tracker.inertial_state is None


@pytest.mark.parametrize("initialize_from_imu", [False, True])
def test_contact_gap_reuses_preintegration_through_rejection_and_acceptance(
        monkeypatch, initialize_from_imu):
    import gelslam.core.gtsam_vi as gv

    tracker, accepted = make_failing_contact_tracker(
        monkeypatch, initialize_from_imu=initialize_from_imu,
    )
    solver = tracker.frontend_visual_inertial
    solver.config["integration_method"] = "midpoint"
    tracker.imu_config["integration_method"] = "midpoint"
    prior = solver.marginal_prior
    tracker.imu_buffer = ImuBuffer()
    for time in np.linspace(1., 1.6, 31):
        tracker.imu_buffer.add(ImuSample(
            time, np.array([.2 * time, .1 * time**2, 9.81]) + accepted.accel_bias,
            np.array([.1 * time, -.15 * time, .4]) + accepted.gyro_bias,
        ))
    real_preintegrate = gv.preintegrate_imu
    integrated = []
    integrate = gv._integrate_measurements

    def record(pim, samples):
        integrated.extend((a.timestamp, b.timestamp) for a, b in zip(samples, samples[1:]))
        return integrate(pim, samples)

    def no_full_reintegration(*args, **kwargs):
        pytest.fail("contact gap must reuse the accumulated IMU prefix")

    monkeypatch.setattr(gv, "_integrate_measurements", record)
    monkeypatch.setattr(gv, "preintegrate_imu", no_full_reintegration)
    for time in (1.125, 1.245):
        assert tracker._initialize_no_contact_coast()
        assert tracker._coast_to_header(header(time), TrackingMode.NO_CONTACT_COASTING)
    cache = tracker._frontend_imu_preintegration
    assert tracker._frontend_vi_state is accepted
    assert solver.marginal_prior is prior
    assert cache.last_sample.timestamp == pytest.approx(1.24)
    prefix_intervals = list(integrated)

    class FlatVisual:
        def __len__(self):
            return 10

        def evaluate(self, _reference, _pose):
            return np.zeros(30), np.zeros((30, 6)), np.zeros((30, 6))

    monkeypatch.setattr(tracker_module, "DenseNormalResidual", lambda *a, **k: FlatVisual())
    qualities = iter(((0., 0.), (0., 0.), (1., 1.)))
    monkeypatch.setattr(tracker_module, "normalflow_quality", lambda *a: next(qualities))
    refine = solver.refine_current
    snapshots = []

    def check_factor(visual, reference, previous, current, samples, extrinsic,
                     preintegrated=None):
        assert previous is accepted
        snapshots.append(preintegrated)
        # Exclude the independent batch oracle from production interval counts.
        with monkeypatch.context() as oracle:
            oracle.setattr(gv, "_integrate_measurements", integrate)
            expected = real_preintegrate(samples, previous, solver.config, extrinsic, solver.gravity)
        np.testing.assert_allclose(preintegrated.deltaRij().matrix(), expected.deltaRij().matrix(),
                                   rtol=0., atol=1e-14)
        np.testing.assert_allclose(preintegrated.deltaPij(), expected.deltaPij(), rtol=0., atol=1e-14)
        np.testing.assert_allclose(preintegrated.deltaVij(), expected.deltaVij(), rtol=0., atol=1e-14)
        np.testing.assert_allclose(preintegrated.preintMeasCov(), expected.preintMeasCov(),
                                   rtol=0., atol=1e-14)
        return refine(visual, reference, previous, current, samples, extrinsic,
                      preintegrated=preintegrated)

    monkeypatch.setattr(solver, "refine_current", check_factor)
    for _ in range(2):
        with pytest.raises(tracker_module.QualityGateError):
            tracker._refine_normalflow_with_imu(
                np.eye(4), header(1.275), tracker.ref_frame, tracker.prev_frame,
            )
        assert tracker._frontend_vi_state is accepted
        assert solver.marginal_prior is prior
        assert tracker._frontend_imu_preintegration is cache
        assert cache.last_sample.timestamp == pytest.approx(1.26)
    # Continue with raw samples after the failed, interpolated contact boundary.
    assert tracker._initialize_no_contact_coast()
    assert tracker._coast_to_header(header(1.325), TrackingMode.NO_CONTACT_COASTING)
    tracker._refine_normalflow_with_imu(
        np.eye(4), header(1.375), tracker.ref_frame, tracker.prev_frame,
    )
    assert len(snapshots) == 3
    assert snapshots[0].deltaTij() == pytest.approx(.275)
    assert len(prefix_intervals) == len(set(prefix_intervals))
    assert all(integrated.count(pair) == 1 for pair in prefix_intervals)
    assert tracker._frontend_vi_state.timestamp == pytest.approx(1.375)
    assert solver.marginal_prior.matches(tracker._frontend_vi_state)
    assert tracker._frontend_imu_preintegration is None
    tracker._record_tactile_pose(tracker.ref_T_prev, header(1.375))
    assert tracker._initialize_no_contact_coast()
    assert tracker._frontend_imu_preintegration.anchor.timestamp == pytest.approx(1.375)


@pytest.mark.parametrize("change", ["bias", "gravity", "anchor", "noise"])
def test_contact_gap_discards_incompatible_preintegration(monkeypatch, change):
    tracker, accepted = make_failing_contact_tracker(monkeypatch)
    solver = tracker.frontend_visual_inertial
    assert tracker._initialize_no_contact_coast()
    assert tracker._coast_to_header(header(1.12), TrackingMode.NO_CONTACT_COASTING)
    previous = accepted.copy()
    if change == "bias":
        previous.gyro_bias[0] += .01
    elif change == "gravity":
        solver.gravity[0] += .1
    elif change == "anchor":
        previous.timestamp = 1.02
    else:
        solver.config["accelerometer_noise_density"] = .03
    samples = tracker.imu_buffer.samples_covering(previous.timestamp, 1.275)
    expected, full = solver.predict_current(previous, samples, tracker.gelsight_T_imu)
    actual, pim = tracker._predict_frontend_imu(previous, samples)
    assert tracker._frontend_imu_preintegration is None
    np.testing.assert_allclose(actual.pose, expected.pose, rtol=0., atol=1e-14)
    np.testing.assert_allclose(actual.velocity, expected.velocity, rtol=0., atol=1e-14)
    np.testing.assert_allclose(pim.preintMeasCov(), full.preintMeasCov(), rtol=0., atol=1e-14)


@pytest.mark.parametrize("initialize_from_imu", [False, True])
def test_frontend_recovery_coast_reacquire_keeps_accepted_optimization_anchor(
        monkeypatch, initialize_from_imu):
    tracker, accepted = make_failing_contact_tracker(
        monkeypatch, initialize_from_imu=initialize_from_imu,
    )
    tracker.imu_buffer = ImuBuffer()
    for time in np.arange(1., 1.5625, 0.0625):
        tracker.imu_buffer.add(ImuSample(
            time, np.array([0, 0, 9.81]) + accepted.accel_bias,
            np.array([0, 0, 0.4]) + accepted.gyro_bias,
        ))
    solver = tracker.frontend_visual_inertial
    original_refine = solver.refine_current
    anchors = []

    class FlatVisual:
        def __len__(self):
            return 10

        def evaluate(self, _reference, _pose):
            return np.zeros(30), np.zeros((30, 6)), np.zeros((30, 6))

    def refine(visual, reference, previous, current, samples, extrinsic,
               preintegrated=None):
        anchors.append((previous.timestamp, [s.timestamp for s in samples]))
        return original_refine(
            visual, reference, previous, current, samples, extrinsic,
            preintegrated=preintegrated,
        )

    monkeypatch.setattr(solver, "refine_current", refine)
    monkeypatch.setattr(tracker_module, "DenseNormalResidual", lambda *a, **k: FlatVisual())
    qualities = iter(((0., 0.), (0., 0.), (1., 1.), (1., 1.)))
    monkeypatch.setattr(tracker_module, "normalflow_quality", lambda *a: next(qualities))
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    tracker.track(image, header(1.0625))
    ret, (_, keyframes) = tracker.track(image, header(1.125))
    assert ret and len(keyframes) == 1
    recovered = tracker._frontend_vi_state
    assert getattr(tracker, "previous_tactile_ref_T", None) is None
    monkeypatch.setattr(
        tracker_module, "Frame", lambda *a, **k: SimpleNamespace(is_contacted=False),
    )
    ret, (frame_msg, keyframes) = tracker.track(image, header(1.25))
    assert ret and keyframes == []
    # The recovered velocity is nonzero, even though tactile history was cleared.
    assert np.allclose(
        (tracker.trial_T_ref @ frame_msg["ref_T_curr"])[:3, 3],
        recovered.pose[:3, 3] + recovered.velocity * 0.125, atol=1e-7,
    )
    assert tracker._frontend_vi_state is recovered
    assert tracker.inertial_state is None
    coast_pose = tracker._frontend_coast_state.pose.copy()
    assert tracker._frontend_coast_state.timestamp == 1.25
    monkeypatch.setattr(
        tracker_module, "Frame", lambda *a, **k: tracker.prev_frame,
    )
    ret, (frame_msg, keyframes) = tracker.track(image, header(1.375))
    assert ret and not keyframes
    assert frame_msg["ref_kid"] == tracker.curr_kid == 4  # Keep the IMU recovery keyframe.
    assert anchors[-1] == (1.125, [1.125, 1.1875, 1.25, 1.3125, 1.375])
    assert tracker._frontend_vi_state.timestamp == 1.375
    assert solver.marginal_prior.matches(tracker._frontend_vi_state)
    assert tracker._frontend_coast_state is None
    assert tracker.tracking_mode is TrackingMode.CONTACT_TRACKING
    # A new gap starts at the newly accepted contact state, not the old prediction.
    assert tracker._initialize_no_contact_coast()
    assert tracker._frontend_coast_state.timestamp == 1.375
    assert not np.allclose(tracker._frontend_coast_state.pose, coast_pose)
