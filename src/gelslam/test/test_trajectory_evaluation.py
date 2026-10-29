"""Tests for final-trajectory SE(3) evaluation utilities."""

import numpy as np
from scipy.spatial.transform import Rotation

from gelslam.core.trajectory_evaluation import (
    apply_se3,
    estimate_first_pose_alignment,
    match_trajectory_timestamps,
    save_aligned_trajectory,
    trajectory_output_dir,
)


def pose_at(position):
    pose = np.eye(4)
    pose[:3, 3] = position
    return pose


def test_timestamp_matching_interpolates_position_and_rotation_at_frame_time():
    estimated = [(1.025, 1, 0, pose_at((0, 0, 0)))]
    first = pose_at((0, 0, 0))
    last = pose_at((2, 4, 6))
    last[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    ground_truth = [(1.00, -1, -1, first), (1.05, -1, -1, last)]

    matches = match_trajectory_timestamps(estimated, ground_truth, 0.03)

    assert len(matches) == 1
    interpolated = matches[0][1]
    assert interpolated[:3] == (estimated[0][0], -1, -1)
    assert np.allclose(interpolated[3][:3, 3], (1, 2, 3))
    assert np.allclose(
        interpolated[3][:3, :3], Rotation.from_euler("z", 45, degrees=True).as_matrix()
    )


def test_first_pose_alignment_uses_anchor_attitude_and_does_not_refit_later_drift(tmp_path):
    anchor = pose_at((4, -2, 3))
    anchor[:3, :3] = Rotation.from_euler("xyz", (20, -10, 45), degrees=True).as_matrix()
    gt_anchor = pose_at((-3, 1, 8))
    gt_anchor[:3, :3] = Rotation.from_euler("xyz", (-5, 30, -70), degrees=True).as_matrix()
    fixed = estimate_first_pose_alignment(anchor, gt_anchor)
    assert np.allclose(fixed @ anchor, gt_anchor)
    estimates = []
    truth = []
    for index in range(3):
        estimate = anchor.copy()
        estimate[:3, 3] += (index, index * index, 0)
        target = fixed @ estimate
        if index:
            target[:3, 3] += (3, -1, 2)
        estimates.append((float(index), index, 0, estimate))
        truth.append((float(index), -1, -1, target))
    result = save_aligned_trajectory(str(tmp_path), estimates, truth, 0.03)
    assert result["status"] == "ok"
    assert result["alignment_method"] == "first_matched_pose"
    assert result["alignment_anchor_timestamp_s"] == 0.
    assert np.allclose(result["alignment_estimate_to_gt"], fixed)
    assert result["translation_rmse_m"] > 1.
    assert result["rotation_rmse_deg"] < 1e-6


def test_timestamp_matching_skips_extrapolation_and_missing_gt_intervals():
    estimated = [(time, index, 0, pose_at((0, 0, 0)))
                 for index, time in enumerate((0.99, 1.0, 1.025, 1.051, 1.999, 2.001))]
    ground_truth = [(time, -1, -1, pose_at((0, 0, 0)))
                    for time in (1.0, 1.05, 2.0)]

    matches = match_trajectory_timestamps(estimated, ground_truth, 0.03)

    assert [estimate[0] for estimate, _ in matches] == [1.0, 1.025]


def test_trajectory_output_dir_uses_explicit_sequence_name_or_data_dir_name():
    assert trajectory_output_dir("/results", "sequence_01", "/data/ignored") == (
        "/results/trajectory_evaluation/sequence_01"
    )
    assert trajectory_output_dir("/results", "", "/data/sequence_02") == (
        "/results/trajectory_evaluation/sequence_02"
    )
    assert trajectory_output_dir("/results", "", "") == (
        "/results/trajectory_evaluation/default"
    )


def test_final_trajectory_export_writes_aligned_csv_metrics_and_plot(tmp_path):
    estimated = [
        (float(index) + 0.25, index, 0, pose_at(position))
        for index, position in enumerate(((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    ]
    transform = np.eye(4)
    transform[:3, 3] = (3.0, -2.0, 1.0)
    ground_truth = []
    for timestamp, _, _, pose in estimated:
        for offset in (-0.25, 0.25):
            sample = pose.copy()
            sample[2, 3] += 2 * offset
            ground_truth.append((timestamp + offset, -1, -1, apply_se3(transform, sample)))

    metrics = save_aligned_trajectory(str(tmp_path), estimated, ground_truth, 0.3)

    assert metrics["status"] == "ok"
    assert metrics["translation_rmse_m"] < 1e-10
    assert (tmp_path / "trajectory_final_raw.csv").is_file()
    assert (tmp_path / "trajectory_final_aligned.csv").is_file()
    assert (tmp_path / "trajectory_ground_truth_matched.csv").is_file()
    assert (tmp_path / "trajectory_metrics.json").is_file()
    assert (tmp_path / "trajectory_se3_aligned.png").is_file()
    assert (tmp_path / "trajectory_translation_error.png").is_file()
    assert (tmp_path / "trajectory_rotation_error.png").is_file()
    matched_gt = np.genfromtxt(tmp_path / "trajectory_ground_truth_matched.csv", delimiter=",", skip_header=1)
    assert np.allclose(matched_gt[:, 0], [sample[0] for sample in estimated])
