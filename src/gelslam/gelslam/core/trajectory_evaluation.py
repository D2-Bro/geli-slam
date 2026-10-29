"""Final-pose trajectory export, rigid alignment, and ground-truth plots."""

import csv
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from gelslam.core.trajectory_error_plot import OUTPUT_FILES, write_trajectory_error_colormaps


def trajectory_output_dir(save_dir, sequence_name, data_dir):
    """Return the per-sequence directory for trajectory evaluation outputs.

    An explicitly supplied sequence name takes priority.  Otherwise the final
    component of ``data_dir`` identifies the sequence.  A sequence name must
    be one directory component so it cannot write outside the evaluation
    directory.
    """
    sequence_name = str(sequence_name or "").strip()
    if not sequence_name:
        data_dir = str(data_dir or "").strip()
        if data_dir:
            sequence_name = os.path.basename(
                os.path.normpath(os.path.expanduser(data_dir))
            )
    if not sequence_name:
        sequence_name = "default"
    if sequence_name in (os.curdir, os.pardir) or os.path.basename(sequence_name) != sequence_name:
        raise ValueError("sequence_name must be a single directory name")
    return os.path.join(save_dir, "trajectory_evaluation", sequence_name)


def match_trajectory_timestamps(estimated, ground_truth, max_delta_s):
    """Interpolate GT poses at estimate times within the sampling tolerance.

    Translation is linear and orientation uses quaternion SLERP.  A frame
    outside the GT time range is never extrapolated.  The nearest GT sample
    must be within ``max_delta_s`` and the bracketing GT gap must be at most
    twice that tolerance, so missing GT intervals are not bridged.
    """
    if max_delta_s < 0.0:
        raise ValueError("max_delta_s must be non-negative")
    ground_truth = sorted(ground_truth, key=lambda sample: sample[0])
    if not ground_truth:
        return []
    gt_times = np.asarray([sample[0] for sample in ground_truth], dtype=np.float64)
    matches = []
    for estimate in estimated:
        timestamp = estimate[0]
        index = int(np.searchsorted(gt_times, timestamp))
        if index < len(gt_times) and gt_times[index] == timestamp:
            pose = ground_truth[index][3]
        elif index == 0 or index == len(gt_times):
            continue
        else:
            before_time = gt_times[index - 1]
            after_time = gt_times[index]
            before_gap = timestamp - before_time
            after_gap = after_time - timestamp
            interval = after_time - before_time
            if (min(before_gap, after_gap) > max_delta_s
                    or interval > 2.0 * max_delta_s):
                continue
            fraction = before_gap / interval
            before_pose = ground_truth[index - 1][3]
            after_pose = ground_truth[index][3]
            pose = np.eye(4)
            pose[:3, 3] = (1.0 - fraction) * before_pose[:3, 3] + fraction * after_pose[:3, 3]
            rotations = Rotation.from_matrix([before_pose[:3, :3], after_pose[:3, :3]])
            pose[:3, :3] = Slerp([0.0, 1.0], rotations)([fraction]).as_matrix()[0]
        matches.append((estimate, (timestamp, -1, -1, pose)))
    return matches


def estimate_first_pose_alignment(estimated_pose, ground_truth_pose):
    """Map the first matched final pose to GT, then keep that transform fixed.

    R_GT_map = R_GT_sensor(t0) R_map_sensor(t0)^T.
    Translation puts the same sensor origin at the GT position at t0.
    No later samples, trajectory fit, scale fit or repeated alignment are used.
    """
    estimated_pose = np.asarray(estimated_pose, dtype=np.float64)
    ground_truth_pose = np.asarray(ground_truth_pose, dtype=np.float64)
    if (estimated_pose.shape != (4, 4) or ground_truth_pose.shape != (4, 4)
            or not np.isfinite(estimated_pose).all() or not np.isfinite(ground_truth_pose).all()):
        raise ValueError("First-pose alignment requires two finite 4x4 poses")
    estimated_rotation = Rotation.from_matrix(estimated_pose[:3, :3]).as_matrix()
    gt_rotation = Rotation.from_matrix(ground_truth_pose[:3, :3]).as_matrix()
    alignment = np.eye(4)
    alignment[:3, :3] = gt_rotation @ estimated_rotation.T
    alignment[:3, 3] = ground_truth_pose[:3, 3] - alignment[:3, :3] @ estimated_pose[:3, 3]
    return alignment


def apply_se3(alignment, pose):
    """Apply a world-frame rigid alignment to one pose."""
    return np.asarray(alignment, dtype=np.float64) @ np.asarray(pose, dtype=np.float64)


def _quaternion_from_rotation(rotation):
    """Convert a proper rotation matrix to x/y/z/w quaternion ordering."""
    rotation = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        return np.array(((rotation[2, 1] - rotation[1, 2]) / scale,
                         (rotation[0, 2] - rotation[2, 0]) / scale,
                         (rotation[1, 0] - rotation[0, 1]) / scale,
                         0.25 * scale))
    diagonal = np.diag(rotation)
    index = int(np.argmax(diagonal))
    if index == 0:
        scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        return np.array((0.25 * scale, (rotation[0, 1] + rotation[1, 0]) / scale,
                         (rotation[0, 2] + rotation[2, 0]) / scale,
                         (rotation[2, 1] - rotation[1, 2]) / scale))
    if index == 1:
        scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        return np.array(((rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale,
                         (rotation[1, 2] + rotation[2, 1]) / scale,
                         (rotation[0, 2] - rotation[2, 0]) / scale))
    scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
    return np.array(((rotation[0, 2] + rotation[2, 0]) / scale,
                     (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale,
                     (rotation[1, 0] - rotation[0, 1]) / scale))


def _write_trajectory_csv(path, samples):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("timestamp_s", "fid", "ref_kid", "tx_m", "ty_m", "tz_m", "qx", "qy", "qz", "qw"))
        for timestamp, fid, ref_kid, pose in samples:
            quaternion = _quaternion_from_rotation(pose[:3, :3])
            writer.writerow((timestamp, fid, ref_kid, *pose[:3, 3], *quaternion))


def _write_plot(path, estimated, ground_truth):
    cache_dir = os.path.join(os.path.dirname(path), ".matplotlib")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", cache_dir)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    estimate_times = np.asarray([sample[0] for sample in estimated])
    estimate_positions = np.asarray([sample[3][:3, 3] for sample in estimated])
    gt_times = np.asarray([sample[0] for sample in ground_truth])
    gt_positions = np.asarray([sample[3][:3, 3] for sample in ground_truth])
    figure = plt.figure(figsize=(13, 5))
    spatial = figure.add_subplot(1, 2, 1, projection="3d")
    spatial.plot(*estimate_positions.T, label="GelSLAM final", color="tab:blue")
    spatial.plot(*gt_positions.T, label="ground truth", color="tab:orange")
    spatial.set_xlabel("x [m]")
    spatial.set_ylabel("y [m]")
    spatial.set_zlabel("z [m]")
    spatial.legend()
    spatial.set_title("Final backend trajectory: fixed first-pose alignment")
    series = figure.add_subplot(1, 2, 2)
    for index, axis in enumerate("xyz"):
        series.plot(estimate_times, estimate_positions[:, index], label="estimate " + axis)
        series.plot(gt_times, gt_positions[:, index], "--", label="GT " + axis)
    series.set_xlabel("timestamp [s]")
    series.set_ylabel("position [m]")
    series.legend(ncol=2, fontsize="small")
    series.grid()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_aligned_trajectory(output_dir, estimated, ground_truth, max_delta_s):
    """Save raw/final trajectory, interpolated GT, SE(3) metrics, and a PNG plot.

    ``estimated`` and ``ground_truth`` are tuples of
    ``(timestamp_s, fid, ref_kid, map_T_pose)``.  GT fid/ref_kid are retained
    as -1.  The returned metrics make it possible for the ROS node to report a
    concise completion message without parsing files again.
    """
    os.makedirs(output_dir, exist_ok=True)
    for filename in OUTPUT_FILES:
        path = os.path.join(output_dir, filename)
        if os.path.isfile(path):
            os.remove(path)
    estimated = sorted(estimated, key=lambda sample: sample[0])
    _write_trajectory_csv(os.path.join(output_dir, "trajectory_final_raw.csv"), estimated)
    matches = match_trajectory_timestamps(estimated, ground_truth, max_delta_s)
    if len(matches) < 2:
        result = {"matched_poses": len(matches), "status": "insufficient_matches"}
        with open(os.path.join(output_dir, "trajectory_metrics.json"), "w") as handle:
            json.dump(result, handle, indent=2)
        return result
    alignment = estimate_first_pose_alignment(matches[0][0][3], matches[0][1][3])
    aligned_estimated = [
        (estimate[0], estimate[1], estimate[2], apply_se3(alignment, estimate[3]))
        for estimate, _ in matches
    ]
    matched_ground_truth = [gt for _, gt in matches]
    translation_errors = np.asarray([
        np.linalg.norm(estimate[3][:3, 3] - gt[3][:3, 3])
        for estimate, gt in zip(aligned_estimated, matched_ground_truth)
    ])
    rotation_errors_deg = np.degrees(Rotation.from_matrix([
        gt[3][:3, :3].T @ estimate[3][:3, :3]
        for estimate, gt in zip(aligned_estimated, matched_ground_truth)
    ]).magnitude())
    _write_trajectory_csv(os.path.join(output_dir, "trajectory_final_aligned.csv"), aligned_estimated)
    _write_trajectory_csv(os.path.join(output_dir, "trajectory_ground_truth_matched.csv"), matched_ground_truth)
    _write_plot(os.path.join(output_dir, "trajectory_se3_aligned.png"), aligned_estimated, matched_ground_truth)
    write_trajectory_error_colormaps(
        output_dir, aligned_estimated, matched_ground_truth,
        translation_errors, rotation_errors_deg, label="GelSLAM",
    )
    result = {
        "status": "ok",
        "matched_poses": len(matches),
        "alignment_method": "first_matched_pose",
        "alignment": "one fixed SE(3) transform from first matched final backend pose; no scale or trajectory fit",
        "alignment_anchor_timestamp_s": matches[0][0][0],
        "alignment_anchor_fid": matches[0][0][1],
        "alignment_anchor_ref_kid": matches[0][0][2],
        "alignment_estimate_to_gt": alignment.tolist(),
        "translation_rmse_m": float(np.sqrt(np.mean(translation_errors**2))),
        "translation_mean_m": float(np.mean(translation_errors)),
        "rotation_rmse_deg": float(np.sqrt(np.mean(rotation_errors_deg**2))),
        "rotation_mean_deg": float(np.mean(rotation_errors_deg)),
    }
    with open(os.path.join(output_dir, "trajectory_metrics.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    return result
