#!/usr/bin/env python3
"""Inspect recorded IMU values strictly before the first GelSight image.

GT IMU orientation supplies the expected stationary specific force. No sensor
calibration or timing offset is fitted. The optional short propagation check
starts at the first GT-covered IMU sample with zero velocity and configured
biases. Its target is the last IMU sample before the first image.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/gelslam"))
from gelslam.core.gtsam_vi import GtsamVisualInertialSolver
from gelslam.core.inertial import ImuSample, gyro_to_rad_per_sec
from gelslam.core.numpy_vi import NavigationState


def stamp_ns(header):
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def expected_stationary_specific_force(world_R_imu, gravity_world):
    """Accelerometers measure -R_WI.T*g_W when the sensor origin is stationary."""
    return np.einsum("...ji,j->...i", world_R_imu, -np.asarray(gravity_world))


def statistics(values):
    values = np.asarray(values, dtype=float)
    return dict(
        count=len(values), mean=values.mean(axis=0).tolist(),
        std=values.std(axis=0, ddof=1).tolist(),
        rms=np.sqrt(np.mean(values**2, axis=0)).tolist(),
        minimum=values.min(axis=0).tolist(), maximum=values.max(axis=0).tolist(),
        median=np.median(values, axis=0).tolist(), p95=np.quantile(values, .95, axis=0).tolist(),
    )


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    from geometry_msgs.msg import PoseStamped
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image, Imu

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "src/gelslam/config/real_config.yaml")
    args = parser.parse_args()
    database = args.bag
    if database.is_dir():
        matches = list(database.glob("*.db3"))
        if len(matches) != 1:
            parser.error("exactly one database is required")
        database = matches[0]
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists():
        parser.error("output already contains results; choose a fresh output path")
    config = yaml.safe_load(args.config.read_text())
    ic = config["imu_config"]
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    topics = dict(connection.execute("SELECT name,id FROM topics"))
    image_row = connection.execute(
        "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp,id LIMIT 1",
        (topics[config["device_config"]["image_topic"]],),
    ).fetchone()
    image = deserialize_message(bytes(image_row[1]), Image)
    first_image_ns = stamp_ns(image.header)
    if first_image_ns != image_row[0]:
        raise ValueError("first image header and bag time differ")
    imu_ns, accelerations, gyros, samples = [], [], [], []
    imu_frames = set()
    for bag_ns, blob in connection.execute(
            "SELECT timestamp,data FROM messages WHERE topic_id=? AND timestamp<? ORDER BY timestamp,id",
            (topics[ic["topic"]], first_image_ns)):
        message = deserialize_message(bytes(blob), Imu)
        ns = stamp_ns(message.header)
        if ns != bag_ns:
            raise ValueError("IMU header and bag timestamps differ")
        acceleration = np.array([getattr(message.linear_acceleration, axis) for axis in "xyz"])
        gyro = gyro_to_rad_per_sec(
            [getattr(message.angular_velocity, axis) for axis in "xyz"],
            ic.get("gyro_input_unit", "rad_s"), ic.get("gyro_max_rate_rad_s", 5),
        )
        imu_ns.append(ns)
        accelerations.append(acceleration)
        gyros.append(gyro)
        samples.append(ImuSample((ns - first_image_ns) * 1e-9, acceleration, gyro))
        imu_frames.add(message.header.frame_id)
    if len(samples) < 2 or any(a >= b for a, b in zip(imu_ns, imu_ns[1:])):
        raise ValueError("at least two strictly ordered pre-image IMU samples are required")
    # Read GT through a little after the cutoff only to bracket the last
    # pre-image IMU time. No IMU sample at/after the first image enters analysis.
    gt_ns, gt_positions, gt_rotations = [], [], []
    gt_frames = set()
    for bag_ns, blob in connection.execute(
            "SELECT timestamp,data FROM messages WHERE topic_id=? AND timestamp<=? ORDER BY timestamp,id",
            (topics["/ground_truth/imu_pose"], first_image_ns + 100_000_000)):
        message = deserialize_message(bytes(blob), PoseStamped)
        ns = stamp_ns(message.header)
        if ns != bag_ns:
            raise ValueError("GT header and bag timestamps differ")
        q = message.pose.orientation
        gt_ns.append(ns)
        gt_positions.append([getattr(message.pose.position, axis) for axis in "xyz"])
        gt_rotations.append(Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix())
        gt_frames.add(message.header.frame_id)
    connection.close()
    times = np.array([(ns - first_image_ns) * 1e-9 for ns in imu_ns])
    gt_times = np.array([(ns - first_image_ns) * 1e-9 for ns in gt_ns])
    gt_positions = np.asarray(gt_positions)
    gt_rotation_series = Rotation.from_matrix(np.asarray(gt_rotations))
    if len(gt_times) < 2 or np.any(np.diff(gt_times) <= 0):
        raise ValueError("GT orientation must be bracketed and strictly ordered")
    valid = (times >= gt_times[0]) & (times <= gt_times[-1])
    gravity_world = np.array([0., 0., -float(ic["gravity_mps2"])])
    accelerations, gyros = np.asarray(accelerations), np.asarray(gyros)
    interpolated_rotations = Slerp(gt_times, gt_rotation_series)(times[valid]).as_matrix()
    expected = np.full_like(accelerations, np.nan)
    expected[valid] = expected_stationary_specific_force(interpolated_rotations, gravity_world)
    residuals = accelerations - expected
    world_residual = np.einsum("nij,nj->ni", interpolated_rotations, residuals[valid])
    accel_norm = np.linalg.norm(accelerations, axis=1)
    gyro_norm = np.linalg.norm(gyros, axis=1)
    angles = np.rad2deg(np.arccos(np.clip(
        np.sum(accelerations[valid] * expected[valid], axis=1)
        / (accel_norm[valid] * np.linalg.norm(expected[valid], axis=1)), -1, 1,
    )))
    gt_before = gt_times < 0
    positions_before = gt_positions[gt_before]
    rotations_before = gt_rotation_series[gt_before]
    stationary = dict(
        samples=int(gt_before.sum()),
        maximum_position_change_from_first_mm=float(np.max(np.linalg.norm(
            positions_before - positions_before[0], axis=1)) * 1000),
        position_peak_to_peak_xyz_mm=(np.ptp(positions_before, axis=0) * 1000).tolist(),
        maximum_rotation_change_from_first_deg=float(np.rad2deg(
            (rotations_before[0].inv() * rotations_before).magnitude()).max()),
    )
    indices = np.flatnonzero(valid)
    first, last = int(indices[0]), int(indices[-1])
    reference = np.eye(4)
    reference[:3, :3] = interpolated_rotations[0]
    reference[:3, 3] = [np.interp(times[first], gt_times, gt_positions[:, axis]) for axis in range(3)]
    target_gt = np.eye(4)
    target_gt[:3, :3] = interpolated_rotations[-1]
    target_gt[:3, 3] = [np.interp(times[last], gt_times, gt_positions[:, axis]) for axis in range(3)]
    initial = NavigationState(
        times[first], reference, np.zeros(3),
        np.asarray(ic.get("initial_accel_bias_mps2", [0, 0, 0]), dtype=float),
        np.asarray(ic.get("initial_gyro_bias_rps", [0, 0, 0]), dtype=float),
    )
    propagation_config = dict(ic, frontend_gravity_mps2=gravity_world.tolist(),
                              frontend_initialize_from_imu_preintegration=True)
    prediction = GtsamVisualInertialSolver(propagation_config).predict_current(
        initial, samples[first:last+1], np.eye(4),
    )
    if prediction is None:
        raise ValueError("short stationary propagation did not produce a prediction")
    predicted, _ = prediction
    propagation = dict(
        anchor_imu_sample=first, target_imu_sample=last,
        anchor_elapsed_s=float(times[first]), target_elapsed_s=float(times[last]),
        duration_s=float(times[last] - times[first]),
        initialization="GT IMU pose, zero velocity, configured biases; static-setting control",
        predicted_displacement_world_mm=((predicted.pose[:3, 3]-reference[:3, 3])*1000).tolist(),
        gt_displacement_world_mm=((target_gt[:3, 3]-reference[:3, 3])*1000).tolist(),
        position_error_mm=float(np.linalg.norm(predicted.pose[:3, 3]-target_gt[:3, 3])*1000),
        predicted_velocity_world_mps=predicted.velocity.tolist(),
        rotation_error_deg=float(np.rad2deg(Rotation.from_matrix(
            target_gt[:3, :3].T @ predicted.pose[:3, :3]).magnitude())),
    )
    summary = dict(
        bag=str(database), cutoff="strictly before first corrected GelSight image header/bag time",
        first_image_stamp_ns=first_image_ns, imu_samples=len(samples),
        window_start_relative_to_image_s=float(times[0]), window_end_relative_to_image_s=float(times[-1]),
        first_imu_to_first_image_s=float(-times[0]), imu_sample_span_s=float(times[-1]-times[0]),
        gt_matched_imu_samples=int(valid.sum()), imu_without_bracketing_gt=int((~valid).sum()),
        imu_coordinate_frames=sorted(imu_frames), gt_world_frames=sorted(gt_frames),
        gravity_world_mps2=gravity_world.tolist(),
        acceleration_xyz_mps2=statistics(accelerations), acceleration_norm_mps2=statistics(accel_norm),
        acceleration_norm_minus_gravity_mps2=statistics(accel_norm-np.linalg.norm(gravity_world)),
        angular_velocity_xyz_rad_s=statistics(gyros), angular_velocity_xyz_deg_s=statistics(np.rad2deg(gyros)),
        angular_velocity_norm_rad_s=statistics(gyro_norm),
        all_recorded_gyro_values_exactly_zero=bool(np.all(gyros == 0.)),
        matched_acceleration_xyz_mps2=statistics(accelerations[valid]),
        expected_stationary_specific_force_imu_mps2=statistics(expected[valid]),
        gravity_residual_xyz_imu_mps2=statistics(residuals[valid]),
        gravity_residual_norm_mps2=statistics(np.linalg.norm(residuals[valid], axis=1)),
        gravity_residual_xyz_world_mps2=statistics(world_residual),
        measured_expected_gravity_angle_deg=statistics(angles),
        gt_stationarity=stationary, short_propagation_control=propagation,
        notes=[
            "Recorded accelerometer values are specific force: stationary expectation is -R_WI^T*g_W.",
            "Corrected header/bag times are used; no additional timing shifts or spatial alignment.",
            "GT rotations are SLERP interpolated without extrapolation. Leading unbracketed IMU values remain in raw statistics.",
            "Residual mean is a stationary offset estimate, not a proven accelerometer bias; extrinsic/GT orientation errors also contribute.",
            "All-zero recorded gyro values do not demonstrate zero physical sensor noise.",
            "No calibration, bias setting, tracker state, or runtime configuration is changed.",
        ],
        input_sha256={str(p): digest(p) for p in (database, args.config, Path(__file__))},
    )
    policy = database.parent.parent / "TIMING_POLICY.json"
    if policy.exists():
        summary["timing_policy"] = json.loads(policy.read_text())
    (args.output / "summary.json").write_text(json.dumps(clean(summary), indent=2, allow_nan=False) + "\n")
    (args.output / "config_snapshot.yaml").write_text(yaml.safe_dump(config))
    rows = []
    for i, ns in enumerate(imu_ns):
        row = dict(sample=i, stamp_ns=ns, time_relative_to_first_image_s=times[i], gt_bracketed=bool(valid[i]))
        for j, axis in enumerate("xyz"):
            row[f"accel_{axis}_mps2"] = accelerations[i, j]
            row[f"gyro_{axis}_rad_s"] = gyros[i, j]
            row[f"gyro_{axis}_deg_s"] = np.rad2deg(gyros[i, j])
            row[f"expected_static_accel_{axis}_mps2"] = expected[i, j] if valid[i] else None
            row[f"gravity_residual_{axis}_mps2"] = residuals[i, j] if valid[i] else None
        row["accel_norm_mps2"] = accel_norm[i]
        row["gyro_norm_rad_s"] = gyro_norm[i]
        rows.append(row)
    with (args.output / "samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(args.output, times, accelerations, gyros, expected, residuals, valid,
         gt_times[gt_before], positions_before, rotations_before, gravity_world)
    print(json.dumps(clean({k: summary[k] for k in (
        "imu_samples", "first_imu_to_first_image_s", "gt_matched_imu_samples",
        "all_recorded_gyro_values_exactly_zero", "acceleration_xyz_mps2", "acceleration_norm_mps2",
        "gravity_residual_xyz_imu_mps2", "measured_expected_gravity_angle_deg",
        "gt_stationarity", "short_propagation_control",
    )}), indent=2, allow_nan=False), flush=True)


def plot(output, times, accel, gyro, expected, residuals, valid,
         gt_times, gt_positions, gt_rotations, gravity):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#176bb5", "#dd711d", "#2e956a"]
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True, layout="constrained")
    for j, axis in enumerate("XYZ"):
        axes[0].plot(times, accel[:, j], color=colors[j], lw=1.3, label=f"measured {axis}")
        axes[0].plot(times, expected[:, j], color=colors[j], lw=1, ls="--", label=f"GT gravity {axis}")
        axes[1].plot(times, residuals[:, j], color=colors[j], lw=1.2, label=axis)
        axes[3].plot(times, np.rad2deg(gyro[:, j]), color=colors[j], lw=1.2, label=axis)
    axes[0].set_ylabel("IMU specific force [m/s²]")
    axes[0].legend(ncol=3, fontsize=9)
    axes[1].axhline(0, color="#333333", ls="--", lw=.8)
    axes[1].set_ylabel("Measured - GT gravity [m/s²]")
    axes[1].legend(ncol=3)
    axes[2].plot(times, np.linalg.norm(accel, axis=1), color="#754aa0", label="measured |a|")
    axes[2].axhline(np.linalg.norm(gravity), color="#333333", ls="--", label="configured gravity")
    axes[2].set_ylabel("Specific-force magnitude [m/s²]")
    axes[2].legend()
    axes[3].axhline(0, color="#333333", ls="--", lw=.8)
    axes[3].set_ylabel("Recorded angular velocity [deg/s]")
    axes[3].legend(ncol=3)
    axes[3].set_ylim(-.01, .01) if np.all(gyro == 0.) else None
    axes[3].set_xlabel("Time relative to first GelSight image [s]")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.axvline(0, color="#b62d35", ls=":", lw=1)
        ax.set_xlim(times[0], .01)
        if not valid[0]:
            ax.axvspan(times[0], times[np.flatnonzero(valid)[0]], color="#888888", alpha=.12)
    fig.suptitle("IMU before first GelSight image | raw recorded values\n"
                 "Dashed: stationary GT gravity; grey start: no bracketing GT; red line: first image")
    fig.savefig(output / "imu_before_first_image.png", dpi=160)
    fig.savefig(output / "imu_before_first_image.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, layout="constrained")
    for j, axis in enumerate("XYZ"):
        axes[0].plot(gt_times, (gt_positions[:, j]-gt_positions[0, j])*1000, color=colors[j], label=axis)
    angles = np.rad2deg((gt_rotations[0].inv()*gt_rotations).as_rotvec())
    for j, axis in enumerate("XYZ"):
        axes[1].plot(gt_times, angles[:, j], color=colors[j], label=axis)
    axes[0].set_ylabel("GT position change [mm]")
    axes[1].set_ylabel("GT rotation-vector change [deg]")
    axes[1].set_xlabel("Time relative to first GelSight image [s]")
    for ax in axes:
        ax.legend(ncol=3)
        ax.grid(alpha=.2)
    fig.suptitle("GT IMU-origin stationarity check | differences from first pre-image GT pose")
    fig.savefig(output / "gt_stationarity.png", dpi=160)
    fig.savefig(output / "gt_stationarity.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
