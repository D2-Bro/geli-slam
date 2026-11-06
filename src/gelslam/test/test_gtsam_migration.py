"""Active frontend and backend GTSAM paths."""

from backend_test_support import config as backend_config, frame as backend_frame

from types import SimpleNamespace

import numpy as np
import gtsam
import pytest
from normalflow.utils import get_J, height2pointcloud, wide_remap

from gelslam.core.gtsam_pose_graph import PoseGraph
from gelslam.core.gtsam_vi import (
    GtsamVisualInertialSolver, _combined_imu_factor, _imu_values, combined_imu_current_terms,
    preintegrate_combined,
)
from gelslam.core.inertial import (
    ImuSample, rotation_aligning_gravity_to_negative_z,
)
from gelslam.core.numpy_vi import DenseNormalResidual, NavigationState, adjoint


def _state(time, pose=None):
    return NavigationState(
        time, np.eye(4) if pose is None else pose,
        np.zeros(3), np.zeros(3), np.zeros(3),
    )


def _samples(rate=0.0):
    return [
        ImuSample(time, (0.0, 0.0, 9.81), (0.0, 0.0, rate))
        for time in np.linspace(0.0, 0.1, 11)
    ]


class _FlatVisual:
    def __len__(self):
        return 10

    def evaluate(self, _reference, _target):
        return np.zeros(30), np.zeros((30, 6)), np.zeros((30, 6))

    def target_z_height_correction(self, _reference, _target):
        raise AssertionError("joint visual-IMU solve must not correct Z after optimization")


class _MislinearizedVisual:
    def __init__(self, jacobian_x):
        self.jacobian_x = jacobian_x

    def __len__(self):
        return 10

    def evaluate(self, _reference, target):
        residual = np.full(30, target[0, 3] - 1.0)
        jacobian = np.zeros((30, 6))
        jacobian[:, 3] = self.jacobian_x
        return residual, jacobian, jacobian


def test_frontend_predicts_pose_and_velocity_from_the_same_imu_interval():
    previous = _state(0.0)
    previous.velocity = np.array([0.3, -0.1, 0.0])
    previous.accel_bias = np.array([0.2, 0.0, 0.0])
    previous.gyro_bias = np.array([0.0, 0.0, 0.1])
    samples = [
        ImuSample(float(t), (0.2, 0.0, 9.81), (0.0, 0.0, 0.1))
        for t in np.linspace(0.0, 0.1, 11)
    ]
    solver = GtsamVisualInertialSolver({"gtsam_vi_max_iterations": 2})

    prediction = solver.predict_current(previous, samples, np.eye(4))

    assert prediction is not None
    current, preintegrated = prediction
    assert np.isclose(current.timestamp, 0.1)
    assert np.allclose(current.pose[:3, 3], [0.03, -0.01, 0.0], atol=1e-8)
    assert np.allclose(current.pose[:3, :3], np.eye(3), atol=1e-8)
    assert np.allclose(current.velocity, previous.velocity, atol=1e-8)
    assert np.allclose(current.accel_bias, previous.accel_bias)
    assert np.allclose(current.gyro_bias, previous.gyro_bias)
    assert np.isclose(preintegrated.deltaTij(), 0.1)


def test_frontend_prediction_retains_stationary_imu_pose_with_rotated_mount():
    gelsight_T_imu = np.eye(4)
    gelsight_T_imu[:3, :3] = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ])
    gelsight_T_imu[:3, 3] = [0.02, -0.01, 0.03]
    stationary_specific_force = gelsight_T_imu[:3, :3].T @ [0.0, 0.0, 9.81]
    samples = [
        ImuSample(float(t), stationary_specific_force, (0.0, 0.0, 0.0))
        for t in np.linspace(0.0, 0.1, 11)
    ]

    current, _ = GtsamVisualInertialSolver({}).predict_current(
        _state(0.0, gelsight_T_imu.copy()), samples, gelsight_T_imu
    )

    assert np.allclose(current.pose, gelsight_T_imu, atol=1e-8)
    assert np.allclose(current.velocity, np.zeros(3), atol=1e-8)


def test_native_imu_factor_stationary_and_turning():
    previous = _state(0.0)
    current = _state(0.1)
    current.pose[0, 3] = 0.001
    solver = GtsamVisualInertialSolver({"gtsam_vi_max_iterations": 2})
    stationary = preintegrate_combined(
        _samples(), previous, solver.config, np.eye(4), solver.gravity
    )
    moving = preintegrate_combined(
        _samples(0.5), previous, solver.config, np.eye(4), solver.gravity
    )
    assert np.isclose(stationary.deltaTij(), 0.1)
    assert not np.allclose(stationary.deltaRij().matrix(), moving.deltaRij().matrix())
    blocks, rhs = combined_imu_current_terms(stationary, previous, current)
    assert [block.shape for block in blocks.values()] == [(15, 6), (15, 3), (15, 6)]
    assert rhs.shape == (15,)
    assert np.all(np.isfinite(rhs))
    assert np.isclose(
        _combined_imu_factor(stationary).error(_imu_values(previous, current)),
        0.5 * rhs @ rhs,
    )


def test_native_imu_factor_uses_nonzero_initial_velocity():
    previous = _state(0.0)
    previous.velocity = np.array((0.0, 0.0, 0.02))
    current = _state(0.1)
    current.pose[2, 3] = 0.002
    current.velocity = previous.velocity.copy()
    gelsight_T_imu = np.eye(4)
    gelsight_T_imu[:3, :3] = np.array(
        ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))
    )
    gelsight_T_imu[2, 3] = -0.04
    previous.pose = previous.pose @ gelsight_T_imu
    current.pose = current.pose @ gelsight_T_imu
    gravity = np.array((0.0, 0.0, -9.81))
    sensor_acceleration = -gelsight_T_imu[:3, :3].T @ gravity
    samples = [
        ImuSample(time, sensor_acceleration, (0.0, 0.0, 0.0))
        for time in np.linspace(0.0, 0.1, 11)
    ]
    pim = preintegrate_combined(
        samples, previous, {}, gelsight_T_imu, gravity
    )
    _, correct_rhs = combined_imu_current_terms(pim, previous, current)
    wrong_previous = previous.copy()
    wrong_previous.velocity[:] = 0.0
    _, wrong_rhs = combined_imu_current_terms(pim, wrong_previous, current)

    assert np.linalg.norm(correct_rhs) < 1e-7
    assert np.linalg.norm(wrong_rhs) > 1.0


def test_combined_imu_residual_is_invariant_to_gravity_frame_alignment():
    gravity_trial = np.array((1.444253, -4.934384, 8.350831))
    alignment = np.eye(4)
    alignment[:3, :3] = rotation_aligning_gravity_to_negative_z(gravity_trial)
    previous = _state(0.0)
    previous.velocity = np.array((0.05, -0.04, 0.03))
    previous.accel_bias = np.array((0.003, -0.002, 0.004))
    previous.gyro_bias = np.array((0.0002, -0.0001, 0.00015))
    current = _state(0.1)
    current.pose[:3, 3] = (0.01, -0.03, 0.02)
    current.velocity = np.array((0.06, 0.01, -0.02))
    current.accel_bias = previous.accel_bias.copy()
    current.gyro_bias = previous.gyro_bias.copy()
    gelsight_T_imu = np.eye(4)
    gelsight_T_imu[:3, :3] = np.diag((1.0, -1.0, -1.0))
    gelsight_T_imu[2, 3] = -0.04
    sensor_acceleration = -gelsight_T_imu[:3, :3].T @ gravity_trial
    samples = [
        ImuSample(time, sensor_acceleration, (0.0, 0.0, 0.0))
        for time in np.linspace(0.0, 0.1, 11)
    ]
    unaligned_pim = preintegrate_combined(
        samples, previous, {}, gelsight_T_imu, gravity_trial
    )
    _, unaligned_rhs = combined_imu_current_terms(
        unaligned_pim, previous, current
    )

    aligned_previous = previous.copy()
    aligned_current = current.copy()
    for state in (aligned_previous, aligned_current):
        state.pose = alignment @ state.pose
        state.velocity = alignment[:3, :3] @ state.velocity
    aligned_gravity = alignment[:3, :3] @ gravity_trial
    aligned_pim = preintegrate_combined(
        samples, aligned_previous, {}, gelsight_T_imu, aligned_gravity
    )
    _, aligned_rhs = combined_imu_current_terms(
        aligned_pim, aligned_previous, aligned_current
    )

    assert np.allclose(aligned_gravity, (0.0, 0.0, -np.linalg.norm(gravity_trial)))
    assert np.allclose(aligned_rhs, unaligned_rhs, atol=1e-9)
    assert np.allclose(
        aligned_pim.preintMeasCov(), unaligned_pim.preintMeasCov()
    )


def test_fixed_imu_reference_z_removes_normal_z_dependence_with_tilted_poses():
    height, width = 80, 80
    yy, xx = np.mgrid[:height, :width]
    normals = np.stack((
        0.12 * np.sin(xx / 8), 0.11 * np.cos(yy / 9),
        np.ones((height, width)),
    ), axis=2).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)
    frame = SimpleNamespace(
        N=normals,
        C=np.ones((height, width), dtype=bool),
        H=np.zeros((height, width), dtype=np.float32),
        L=np.arange(height * width, dtype=np.float32).reshape(height, width),
    )
    reference = np.eye(4)
    reference[:3, :3] = gtsam.Rot3.Rodrigues(0.15, -0.08, 0.1).matrix()
    target = reference @ gtsam.Pose3(
        gtsam.Rot3.Rodrigues(-0.05, 0.04, 0.02),
        gtsam.Point3(0.002, -0.001, 0.004),
    ).matrix()
    visual = DenseNormalResidual(frame, frame, 1.0, reference, target, 1500)
    visual.set_fixed_reference_z(0.002)
    base_residual, _, jacobian = visual.evaluate(reference, target)
    base_rows = visual.last_active_rows.copy()
    target_shifted = target.copy()
    target_shifted[:3, 3] += reference[:3, 2] * 0.003
    shifted_residual, _, _ = visual.evaluate(reference, target_shifted)
    assert np.array_equal(base_rows, visual.last_active_rows)
    assert np.allclose(shifted_residual, base_residual, atol=1e-9)
    tangent_reference_z = target[:3, :3].T @ reference[:3, 2]
    assert np.linalg.norm(jacobian[:, 3:6] @ tangent_reference_z) < 1e-10


def test_normalflow_rows_and_pose_jacobian_follow_original_method():
    height, width = 40, 40
    yy, xx = np.mgrid[:height, :width]
    normals = np.stack((
        0.1 * np.sin(xx / 7), 0.1 * np.cos(yy / 7),
        np.ones((height, width)),
    ), axis=2).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)
    laplacian = (xx * width + yy).astype(np.float32)
    contact = np.ones((height, width), dtype=bool)
    target_contact = np.zeros_like(contact)
    target_contact[:, 5:35] = True
    reference = SimpleNamespace(
        N=normals, C=contact, H=np.zeros((height, width), np.float32), L=laplacian,
    )
    target = SimpleNamespace(
        N=normals, C=target_contact, H=reference.H, L=laplacian,
    )
    reference_pose = np.eye(4)
    target_pose = np.eye(4)
    target_pose[0, 3] = 0.005
    visual = DenseNormalResidual(
        reference, target, 1.0, reference_pose, target_pose, max_samples=200
    )
    residual, _, target_jacobian = visual.evaluate(reference_pose, target_pose)
    fixed_rows = visual.last_active_rows.copy()
    assert np.allclose(
        visual.residual_on_fixed_rows(reference_pose, target_pose, fixed_rows),
        residual,
    )
    lost_contact = target_pose.copy()
    lost_contact[0, 3] = 0.003
    partial_residual = visual.residual_on_fixed_rows(
        reference_pose, lost_contact, fixed_rows
    )
    assert partial_residual is not None
    overlap = visual._shared_projection(reference_pose, lost_contact, min_points=1)[-1]
    assert 0 < np.count_nonzero(overlap & fixed_rows) < np.count_nonzero(fixed_rows)
    assert partial_residual.shape == (3 * np.count_nonzero(overlap & fixed_rows),)
    all_lost = target_pose.copy()
    all_lost[0, 3] = 1.
    assert visual.residual_on_fixed_rows(reference_pose, all_lost, fixed_rows) is None
    count = len(residual)
    shifted = target_pose.copy()
    shifted[0, 3] = 0.003
    assert len(visual.evaluate(reference_pose, shifted)[0]) != count

    threshold = np.partition(np.abs(laplacian).ravel(), -200)[-200] + 1e-6
    mask = np.abs(laplacian) >= threshold
    points = height2pointcloud(reference.H, mask, 1.0)
    relative = np.linalg.inv(target_pose) @ reference_pose
    projected = points @ relative[:3, :3].T + relative[:3, 3]
    pixel_x = (projected[:, 0] * 1000.0 + width / 2 - 0.5).astype(np.float32)
    pixel_y = (projected[:, 1] * 1000.0 + height / 2 - 0.5).astype(np.float32)
    active = (
        (wide_remap(target_contact.astype(np.float32), pixel_x, pixel_y,
                    mode=0)[:, 0] > 0.5)
        & (pixel_x >= 0) & (pixel_x < width)
        & (pixel_y >= 0) & (pixel_y < height)
    )
    remapped = wide_remap(normals, pixel_x, pixel_y)[:, 0, :]
    original_residual = (remapped - normals[mask])[active].reshape(-1)
    original_jacobian = np.transpose(
        get_J(normals, mask, points, 1.0), (2, 0, 1)
    )[active].reshape(-1, 5)
    original_six = np.zeros((len(original_residual), 6))
    original_six[:, :3] = original_jacobian[:, :3]
    original_six[:, 3:5] = original_jacobian[:, 3:5]
    assert np.allclose(residual, original_residual, atol=1e-6)
    assert np.allclose(
        target_jacobian, -original_six @ adjoint(np.linalg.inv(relative)),
        atol=1e-6,
    )
    # GTSAM uses a right Pose3 tangent. The smooth rotation direction agrees
    # with a finite difference; OpenCV's remap quantizes subpixel translation.
    delta = np.array((1e-6, 0, 0, 0, 0, 0))
    moved = gtsam.Pose3(target_pose).retract(delta).matrix()
    moved_residual = visual.evaluate(reference_pose, moved)[0]
    assert np.allclose(
        moved_residual - residual, target_jacobian @ delta, atol=1e-7
    )
    alignment = np.eye(4)
    alignment[:3, :3] = rotation_aligning_gravity_to_negative_z(
        np.array((1.0, -3.0, 9.0))
    )
    aligned_reference = alignment @ reference_pose
    aligned_target = alignment @ target_pose
    aligned_visual = DenseNormalResidual(
        reference, target, 1.0, aligned_reference, aligned_target,
        max_samples=200,
    )
    aligned_residual, _, aligned_jacobian = aligned_visual.evaluate(
        aligned_reference, aligned_target
    )
    assert np.allclose(aligned_residual, residual, atol=1e-7)
    assert np.allclose(aligned_jacobian, target_jacobian, atol=1e-7)


def test_candidate_residual_accepts_one_remaining_contact_point():
    height = width = 40
    yy, xx = np.mgrid[:height, :width]
    normals = np.stack((.1*np.sin(xx/7), .1*np.cos(yy/7), np.ones_like(xx)), axis=2).astype(np.float32)
    reference = SimpleNamespace(N=normals, C=np.ones((height,width), bool),
                                H=np.zeros((height,width), np.float32), L=(xx*width+yy).astype(np.float32))
    target = SimpleNamespace(N=normals.copy(), C=reference.C.copy(), H=reference.H, L=reference.L)
    visual = DenseNormalResidual(reference,target,1.,np.eye(4),np.eye(4),max_samples=200)
    residual,_,_ = visual.evaluate(np.eye(4),np.eye(4))
    rows = visual.last_active_rows.copy()
    selected = int(np.flatnonzero(rows)[0])
    px,py = visual._pixel_coordinates(visual.candidate_points)
    x,y = int(round(float(px[selected]))),int(round(float(py[selected])))
    target.C[:] = False
    target.C[y,x] = True
    target.N[y,x,0] += .2

    # Candidate scoring needs just one row; the linearization minimum stays 10.
    candidate = visual.residual_on_fixed_rows(np.eye(4),np.eye(4),rows)
    assert candidate is not None
    assert candidate.shape == (3,)
    assert np.allclose(candidate, [.2,0.,0.],atol=1e-6)
    assert visual.evaluate(np.eye(4),np.eye(4)) is None

    rows_without_selected = rows.copy()
    rows_without_selected[selected] = False
    assert visual.residual_on_fixed_rows(np.eye(4),np.eye(4),rows_without_selected) is None
    target.C[:] = False
    assert visual.residual_on_fixed_rows(np.eye(4),np.eye(4),rows) is None


def _keyframe(x, new, group):
    pose = np.eye(4)
    pose[0, 3] = x
    return SimpleNamespace(
        trial_T_curr=pose,
        trial_group=group,
        is_new_trial=lambda: new,
    )


def test_pose_graph_keeps_aligned_anchor_during_solve_and_reload(tmp_path):
    rotation = rotation_aligning_gravity_to_negative_z([1., -3., 9.])
    keyframes = [backend_frame(0, new=True), backend_frame(1)]
    for keyframe in keyframes:
        keyframe.trial_T_curr[:3, :3] = rotation
        keyframe.navigation_state.pose[:3, :3] = rotation
        keyframe.gravity = -rotation @ np.array([0., 0., 9.81])
    graph = PoseGraph(backend_config())
    graph.add_keyframes(keyframes, 0, 2)
    graph.solve(global_optimization=True)
    assert np.allclose(graph.poses[0], keyframes[0].trial_T_curr, atol=1e-10)
    graph.save(tmp_path)
    loaded = PoseGraph.load(tmp_path)
    loaded.solve(global_optimization=True)
    assert np.allclose(loaded.poses, graph.poses, atol=1e-8)
    assert np.allclose(loaded.loop_pgo.anchor_poses[0], keyframes[0].trial_T_curr)

def test_navigation_graph_separate_loop_pgo_merge_and_reload(tmp_path):
    graph = PoseGraph(backend_config())
    keyframes = [backend_frame(0, new=True), backend_frame(1),
                 backend_frame(2, new=True, group=1), backend_frame(3, group=1)]
    graph.add_keyframes(keyframes, 0, 4)
    assert graph.temporal_links == [(0, 1), (2, 3)]
    assert all(edge[2] is None for edge in graph.edges)
    assert graph._add_loop_pair(1, 3, np.eye(4))
    graph.remove_prior_factors([1])
    graph.solve(global_optimization=True)
    assert graph.loop_pgo.removed_groups == {1}
    graph.save(tmp_path)
    loaded = PoseGraph.load(tmp_path)
    assert np.allclose(loaded.poses, graph.poses)
    assert loaded.loop_pgo.removed_groups == {1}
    assert len(loaded.edges) == 1
    keyframes.append(backend_frame(4, group=1))
    loaded.add_keyframes(keyframes, 4, 5)
    assert loaded.states[4].timestamp == pytest.approx(1.4)

def test_loop_registration_adds_relative_measurement_only_to_pgo(monkeypatch):
    import gelslam.core.gtsam_pose_graph as module
    graph = PoseGraph(backend_config())
    anchors = [backend_frame(0, new=True), backend_frame(1, new=True, group=1)]
    graph.add_keyframes(anchors, 0, 2)
    points = [SimpleNamespace(pt=(float(i), 0.0)) for i in range(8)]
    for keyframe in anchors:
        keyframe.kp = points
        keyframe.des = np.zeros((8, 2), dtype=np.float32)
    matches = [SimpleNamespace(queryIdx=i, trainIdx=i) for i in range(8)]
    monkeypatch.setattr(module.cv2, 'BFMatcher',
                        lambda *_a, **_k: SimpleNamespace(match=lambda *_: matches))
    monkeypatch.setattr(module.cv2, 'estimateAffinePartial2D',
                        lambda *_a, **_k: (np.eye(2, 3), np.ones((8, 1))))
    target_from_reference = np.eye(4)
    target_from_reference[0, 3] = -.002
    monkeypatch.setattr(module, 'normalflow', lambda *_a, **_k: target_from_reference)
    graph.pose_graph_config.update(min_sift_inliers=8, scr_threshold=.3, ccs_threshold=.7,
                                   loop_search_exclude_recent_keyframes=0)
    matched = graph.detect_and_add_loops(anchors, 1, SimpleNamespace(size=lambda: 0))
    assert matched == [0]
    assert graph.map_poses()[1][0, 3] == pytest.approx(.002)
    assert graph.poses[1][0, 3] == 0.  # The local trial frame did not move.
    assert graph.edges[-1][2][0, 3] == pytest.approx(.002)
