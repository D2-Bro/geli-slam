#!/usr/bin/env python3
"""Compare independent consecutive-image motions with GT reference states.

No tracker, keyframe selection, pose graph, spatial alignment, timing fit, or
cross-interval state update is used. All reported transforms map target
GelSight coordinates into reference GelSight coordinates.
"""

import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import sqlite3
import sys
import warnings

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import gtsam
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/gelslam"))
from gelslam.core.gtsam_vi import preintegrate_combined
from gelslam.core.inertial import ImuBuffer, ImuSample, gyro_to_rad_per_sec
from gelslam.core.numpy_vi import NavigationState


class GroundTruthSeries:
    """Linear position and velocity interpolation, with SLERP orientation."""

    def __init__(self, times, poses):
        self.times = np.asarray(times, dtype=np.float64)
        self.poses = np.asarray(poses, dtype=np.float64)
        if len(self.times) < 3 or np.any(np.diff(self.times) <= 0):
            raise ValueError("GT requires at least three strictly ordered samples")
        self.positions = self.poses[:, :3, 3]
        # Nonuniform-time central differences; second-order one-sided ends.
        # This is numerical differentiation, without a smoothing filter.
        self.velocities = np.gradient(
            self.positions, self.times, axis=0, edge_order=2,
        )
        self.rotations = Slerp(self.times, Rotation.from_matrix(self.poses[:, :3, :3]))

    def at(self, time):
        if not self.times[0] <= time <= self.times[-1]:
            raise ValueError("GT does not bracket the requested timestamp")
        pose = np.eye(4)
        pose[:3, :3] = self.rotations([time]).as_matrix()[0]
        pose[:3, 3] = [np.interp(time, self.times, self.positions[:, k]) for k in range(3)]
        velocity = np.array([
            np.interp(time, self.times, self.velocities[:, k]) for k in range(3)
        ])
        return pose, velocity


def integrate_gelsight_motion(reference_imu_pose, reference_imu_velocity,
                             start, end, samples, gel_T_imu, gravity, config):
    """Integrate at the measured IMU origin, then convert both poses to GelSight."""
    if len(samples) < 2:
        raise ValueError("IMU does not bracket both frame timestamps")
    if abs(samples[0].timestamp - start) > 1e-9 or abs(samples[-1].timestamp - end) > 1e-9:
        raise ValueError("IMU interval boundaries differ from frame timestamps")
    previous = NavigationState(
        start, reference_imu_pose.copy(), reference_imu_velocity.copy(),
        np.asarray(config.get("initial_accel_bias_mps2", (0, 0, 0)), dtype=float),
        np.asarray(config.get("initial_gyro_bias_rps", (0, 0, 0)), dtype=float),
    )
    pim = preintegrate_combined(samples, previous, config, np.eye(4), gravity)
    if pim is None:
        raise ValueError("IMU preintegration failed")
    bias = gtsam.imuBias.ConstantBias(previous.accel_bias, previous.gyro_bias)
    predicted = pim.predict(
        gtsam.NavState(gtsam.Pose3(previous.pose), previous.velocity), bias,
    )
    imu_T_gel = np.linalg.inv(gel_T_imu)
    world_T_reference_gel = reference_imu_pose @ imu_T_gel
    world_T_target_gel = np.asarray(predicted.pose().matrix()) @ imu_T_gel
    relative = np.linalg.inv(world_T_reference_gel) @ world_T_target_gel
    if not np.all(np.isfinite(relative)):
        raise ValueError("IMU prediction is not finite")
    return relative, np.asarray(predicted.velocity()), pim


def compare_transforms(estimate, reference, direction_min_mm=0.1):
    """Compare translation vectors in one reference chart and SO(3) rotations."""
    a, b = estimate[:3, 3], reference[:3, 3]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    error = Rotation.from_matrix(reference[:3, :3].T @ estimate[:3, :3]).magnitude()
    direction = None
    if min(na, nb) >= direction_min_mm * 1e-3:
        direction = float(np.rad2deg(np.arccos(np.clip(a @ b / (na * nb), -1, 1))))
    return dict(
        translation_error_mm=float(np.linalg.norm(a - b) * 1000),
        rotation_error_deg=float(np.rad2deg(error)),
        translation_direction_deg=direction,
        translation_magnitude_difference_mm=float((na - nb) * 1000),
        translation_magnitude_ratio=float(na / nb) if nb >= direction_min_mm * 1e-3 else None,
        translation_error_vector_mm=((a - b) * 1000).tolist(),
    )


def stamp_ns(header):
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def pose_matrix(message):
    q, p = message.pose.orientation, message.pose.position
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat((q.x, q.y, q.z, q.w)).as_matrix()
    matrix[:3, 3] = (p.x, p.y, p.z)
    return matrix


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(x) for x in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def summarize(rows):
    def group(selected):
        result = {"interval_count": len(selected)}
        for name in ("normalflow_vs_gt", "imu_vs_gt", "normalflow_vs_imu"):
            result[name] = {}
            for metric in ("translation_error_mm", "rotation_error_deg",
                           "translation_direction_deg", "translation_magnitude_ratio"):
                values = np.array([
                    r[name][metric] for r in selected
                    if name in r and r[name].get(metric) is not None
                ], dtype=float)
                if len(values):
                    result[name][metric] = dict(
                        count=len(values), mean=float(values.mean()),
                        median=float(np.median(values)), rms=float(np.sqrt(np.mean(values**2))),
                        p95=float(np.quantile(values, .95)), maximum=float(values.max()),
                    )
                    if metric == "translation_direction_deg":
                        result[name][metric]["fraction_within_30_deg"] = float(np.mean(values <= 30))
                        result[name][metric]["fraction_opposite_over_90_deg"] = float(np.mean(values > 90))
        return result
    statuses = {s: sum(r["normalflow_status"] == s for r in rows)
                for s in sorted({r["normalflow_status"] for r in rows})}
    return dict(
        all_intervals=group(rows), normalflow_status_counts=statuses,
        accepted_normalflow=group([r for r in rows if r["normalflow_status"] == "accepted"]),
        all_finite_normalflow=group([r for r in rows if "normalflow_T" in r]),
    )


def write_csv(path, rows):
    flat = []
    for row in rows:
        item = {k: row[k] for k in (
            "reference_frame", "target_frame", "reference_stamp_ns", "target_stamp_ns",
            "reference_elapsed_s", "dt_s", "reference_contact", "target_contact",
            "normalflow_status", "imu_status", "scr", "ccs", "imu_samples", "imu_max_dt_s",
        ) if k in row}
        for method in ("gt", "imu", "normalflow"):
            transform = row.get(method + "_T")
            if transform is not None:
                T = np.asarray(transform)
                rv = np.rad2deg(Rotation.from_matrix(T[:3, :3]).as_rotvec())
                for k, axis in enumerate("xyz"):
                    item[f"{method}_t{axis}_mm"] = T[k, 3] * 1000
                    item[f"{method}_r{axis}_deg"] = rv[k]
                item[f"{method}_translation_norm_mm"] = np.linalg.norm(T[:3, 3]) * 1000
                for a in range(4):
                    for b in range(4):
                        item[f"{method}_T{a}{b}"] = T[a, b]
        for comparison in ("normalflow_vs_gt", "imu_vs_gt", "normalflow_vs_imu"):
            for metric, value in row.get(comparison, {}).items():
                if isinstance(value, list):
                    for axis, x in zip("xyz", value):
                        item[f"{comparison}_{metric}_{axis}"] = x
                else:
                    item[f"{comparison}_{metric}"] = value
        for state in ("reference_imu_velocity_world_mps", "reference_gel_velocity_world_mps",
                      "reference_gravity_gel_mps2"):
            for axis, x in zip("xyz", row.get(state, [])):
                item[f"{state}_{axis}"] = x
        for name in ("gt_target_pose_world", "normalflow_target_pose_world", "imu_target_pose_world"):
            if name in row:
                for axis, value in zip("xyz", np.asarray(row[name])[:3, 3]):
                    item[f"{name}_t{axis}_m"] = value
        flat.append(item)
    names = list(dict.fromkeys(k for row in flat for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(flat)


def plot_results(output, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .22})
    x = np.array([r["reference_elapsed_s"] for r in rows])
    colors = {"gt": "#222222", "imu": "#dd711d", "normalflow": "#176bb5"}
    rejected = np.array([r["normalflow_status"] == "quality_rejected" for r in rows])

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True, layout="constrained")
    for method in colors:
        translation = np.full((len(rows), 3), np.nan)
        rotation = np.full((len(rows), 3), np.nan)
        for i, row in enumerate(rows):
            if method + "_T" in row:
                T = np.asarray(row[method + "_T"])
                translation[i] = T[:3, 3] * 1000
                rotation[i] = np.rad2deg(Rotation.from_matrix(T[:3, :3]).as_rotvec())
        for k, axis in enumerate("XYZ"):
            axes[k, 0].plot(x, translation[:, k], color=colors[method], lw=1, label=method)
            axes[k, 1].plot(x, rotation[:, k], color=colors[method], lw=1, label=method)
            if method == "normalflow":
                axes[k, 0].scatter(x[rejected], translation[rejected, k], s=8, color="#b62d35", zorder=3)
                axes[k, 1].scatter(x[rejected], rotation[rejected, k], s=8, color="#b62d35", zorder=3)
            axes[k, 0].set_ylabel(f"Reference-frame {axis} translation [mm]")
            axes[k, 1].set_ylabel(f"Reference-frame {axis} rotation vector [deg]")
    axes[0, 0].legend(ncol=3)
    axes[0, 1].set_title("Red dots: finite NormalFlow result rejected by SCR/CCS")
    axes[-1, 0].set_xlabel("Reference time from first image [s]")
    axes[-1, 1].set_xlabel("Reference time from first image [s]")
    fig.suptitle("Independent consecutive-frame relative motion | no SE(3) alignment")
    fig.savefig(output / "relative_motion_components.png", dpi=170)
    fig.savefig(output / "relative_motion_components.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True, layout="constrained")
    comparisons = ("normalflow_vs_gt", "imu_vs_gt", "normalflow_vs_imu")
    for comparison, color in zip(comparisons, (colors["normalflow"], colors["imu"], "#754aa0")):
        for ax, metric in zip(axes[:2], ("translation_error_mm", "rotation_error_deg")):
            y = [r.get(comparison, {}).get(metric, np.nan) for r in rows]
            ax.plot(x, y, label=comparison.replace("_", " "), lw=1, color=color)
            ax.set_yscale("symlog", linthresh=.01)
    axes[0].set_ylabel("Translation-vector error [mm]")
    axes[1].set_ylabel("SO(3) rotation error [deg]")
    axes[0].legend(ncol=3)
    direction = [r.get("normalflow_vs_imu", {}).get("translation_direction_deg") for r in rows]
    axes[2].plot(x, [np.nan if v is None else v for v in direction], color="#754aa0", lw=1)
    axes[2].axhline(90, color="#b62d35", ls="--", lw=.8)
    axes[2].set_ylim(0, 180)
    axes[2].set_ylabel("NF / IMU translation direction [deg]")
    axes[2].set_title("Direction shown only when both displacement norms are at least 0.1 mm")
    labels = ["accepted", "quality_rejected", "solver_failed", "no_contact"]
    for code, label in enumerate(labels):
        mask = np.array([r["normalflow_status"] == label for r in rows])
        axes[3].scatter(x[mask], np.full(mask.sum(), code), s=9, label=label)
    axes[3].set_yticks(range(len(labels)), labels)
    axes[3].set_xlabel("Reference time from first image [s]")
    axes[3].set_ylabel("NormalFlow status")
    fig.suptitle("Per-interval errors | every reference state is reset from GT")
    fig.savefig(output / "interval_errors.png", dpi=170)
    fig.savefig(output / "interval_errors.pdf")
    plt.close(fig)


def main():
    from cv_bridge import CvBridge
    from geometry_msgs.msg import PoseStamped
    from gs_sdk.gs_reconstruct import Reconstructor
    from normalflow.registration import LoseTrackError, normalflow
    from normalflow.utils import Frame
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image, Imu

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "src/gelslam/config/real_config.yaml")
    parser.add_argument("--calibration", type=Path, help="JSON containing the fixed T_GI matrix")
    parser.add_argument("--model", type=Path, help="surface reconstruction checkpoint")
    parser.add_argument("--timing-policy", type=Path, help="JSON describing the input timestamp policy")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    database = args.bag
    if database.is_dir():
        matches = list(database.glob("*.db3"))
        if len(matches) != 1:
            parser.error("exactly one bag database is required")
        database = matches[0]
    trial = database.parent.parent
    calibration = args.calibration or trial / "original_evidence/CALIBRATION_PILOT.json"
    model = args.model or trial / "original_evidence/nnmodel.pth"
    timing_policy = args.timing_policy or trial / "TIMING_POLICY.json"
    config = yaml.safe_load(args.config.read_text())
    imu_config = config["imu_config"]
    gel_T_imu = np.asarray(json.loads(calibration.read_text())["T_GI"], dtype=float)
    imu_T_gel = np.linalg.inv(gel_T_imu)
    gravity = np.array([0., 0., -float(imu_config["gravity_mps2"])])
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "intervals.json").exists():
        parser.error("output already contains experiment results; choose a fresh directory")
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    topics = dict(connection.execute("SELECT name,id FROM topics"))

    def read(name, message_type):
        result = []
        for bag_ns, blob in connection.execute(
                "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp,id",
                (topics[name],)):
            message = deserialize_message(bytes(blob), message_type)
            ns = stamp_ns(message.header)
            if ns != bag_ns:
                raise ValueError(f"bag/header timestamps differ for {name}")
            result.append((ns, message))
        if any(a[0] >= b[0] for a, b in zip(result, result[1:])):
            raise ValueError(f"non-increasing {name} timestamps")
        return result

    images = read(config["device_config"]["image_topic"], Image)
    imus = read(imu_config["topic"], Imu)
    gt_imu = read("/ground_truth/imu_pose", PoseStamped)
    gt_gel = read("/ground_truth/gel_pose", PoseStamped)
    connection.close()
    origin_ns = images[0][0]
    gt = GroundTruthSeries(
        [(ns - origin_ns) * 1e-9 for ns, _ in gt_imu],
        [pose_matrix(m) for _, m in gt_imu],
    )
    gt_gel_series = GroundTruthSeries(
        [(ns - origin_ns) * 1e-9 for ns, _ in gt_gel],
        [pose_matrix(m) for _, m in gt_gel],
    )
    if [ns for ns, _ in gt_imu] != [ns for ns, _ in gt_gel]:
        raise ValueError("IMU and GelSight GT source timestamps differ")
    gt_closure = max(np.max(np.abs(pose_matrix(i) @ imu_T_gel - pose_matrix(g)))
                     for (_, i), (_, g) in zip(gt_imu, gt_gel))
    if gt_closure > 1e-8:
        raise ValueError("fixed sensor calibration disagrees with GT pose definitions")
    buffer = ImuBuffer(max_samples=len(imus) + 2)
    for ns, m in imus:
        angular = gyro_to_rad_per_sec(
            [m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z],
            imu_config.get("gyro_input_unit", "rad_s"), imu_config.get("gyro_max_rate_rad_s", 5),
        )
        buffer.add(ImuSample((ns - origin_ns) * 1e-9,
                            [m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z],
                            angular))
    bridge = CvBridge()
    background = np.mean([
        bridge.imgmsg_to_cv2(m, desired_encoding="bgr8") for _, m in images[:10]
    ], axis=0)
    recon = Reconstructor(str(model))
    recon.load_bg(background)
    ppmm = float(config["device_config"]["ppmm"])
    track = config["gelslam_config"]["track_config"]
    mask = config["contact_mask_config"]
    rows = []
    previous_frame = None
    for target_index, (target_ns, image_message) in enumerate(images):
        image = bridge.imgmsg_to_cv2(image_message, desired_encoding="bgr8")
        G, H, C = recon.get_surface_info(image, ppmm, **mask)
        frame = Frame(G, H, C, contact_threshold=track["contact_threshold"])
        if previous_frame is None:
            previous_frame = frame
            continue
        reference_index = target_index - 1
        reference_ns = images[reference_index][0]
        t0, t1 = (reference_ns - origin_ns) * 1e-9, (target_ns - origin_ns) * 1e-9
        row = dict(reference_frame=reference_index, target_frame=target_index,
                   reference_stamp_ns=reference_ns, target_stamp_ns=target_ns,
                   reference_elapsed_s=t0, dt_s=t1-t0,
                   reference_contact=bool(previous_frame.is_contacted), target_contact=bool(frame.is_contacted),
                   normalflow_status="no_contact", imu_status="unavailable")
        try:
            reference_imu_pose, velocity = gt.at(t0)
            target_imu_pose, _ = gt.at(t1)
            reference_gel_pose = reference_imu_pose @ imu_T_gel
            target_gel_pose = target_imu_pose @ imu_T_gel
            row["gt_T"] = np.linalg.inv(reference_gel_pose) @ target_gel_pose
            row["reference_imu_pose_world"] = reference_imu_pose
            row["reference_gel_pose_world"] = reference_gel_pose
            row["gt_target_pose_world"] = target_gel_pose
            row["reference_imu_velocity_world_mps"] = velocity
            row["reference_gel_velocity_world_mps"] = gt_gel_series.at(t0)[1]
            row["reference_gravity_gel_mps2"] = reference_gel_pose[:3, :3].T @ gravity
            samples = buffer.samples_covering(t0, t1)
            row["imu_samples"] = len(samples)
            row["imu_max_dt_s"] = float(np.max(np.diff([s.timestamp for s in samples]))) if len(samples) > 1 else None
            imu_T, target_velocity, _ = integrate_gelsight_motion(
                reference_imu_pose, velocity, t0, t1, samples, gel_T_imu, gravity, imu_config,
            )
            row["imu_T"] = imu_T
            row["imu_target_pose_world"] = reference_gel_pose @ imu_T
            row["imu_target_velocity_world_mps"] = target_velocity
            row["imu_status"] = "predicted"
            row["imu_vs_gt"] = compare_transforms(imu_T, row["gt_T"])
            row["imu_vs_gt"]["target_position_error_world_mm"] = (
                (row["imu_target_pose_world"][:3, 3] - target_gel_pose[:3, 3]) * 1000
            )
        except (ValueError, RuntimeError) as error:
            row["imu_error"] = str(error)
        if previous_frame.is_contacted and frame.is_contacted:
            try:
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    target_T_reference, quality = normalflow(
                        previous_frame.N, previous_frame.C, previous_frame.H, previous_frame.L,
                        frame.N, frame.C, frame.H, frame.L,
                        np.eye(4, dtype=np.float32), ppmm, n_samples=3000,
                        scr_threshold=-np.inf, ccs_threshold=-np.inf, return_quality=True,
                    )
                row["normalflow_warnings"] = [str(w.message) for w in captured]
                normal_T = np.linalg.inv(target_T_reference)
                if not np.all(np.isfinite(normal_T)):
                    raise LoseTrackError("NormalFlow returned a nonfinite transform")
                row["normalflow_T"] = normal_T
                row["scr"], row["ccs"] = float(quality["scr"]), float(quality["ccs"])
                passed = (np.isfinite(row["scr"]) and np.isfinite(row["ccs"])
                          and row["scr"] >= track["scr_threshold"] and row["ccs"] >= track["ccs_threshold"])
                row["normalflow_status"] = "accepted" if passed else "quality_rejected"
                if "gt_T" in row:
                    row["normalflow_target_pose_world"] = reference_gel_pose @ normal_T
                    row["normalflow_vs_gt"] = compare_transforms(normal_T, row["gt_T"])
                    row["normalflow_vs_gt"]["target_position_error_world_mm"] = (
                        (row["normalflow_target_pose_world"][:3, 3] - target_gel_pose[:3, 3]) * 1000
                    )
                if "imu_T" in row:
                    row["normalflow_vs_imu"] = compare_transforms(normal_T, row["imu_T"])
            except (LoseTrackError, ValueError, np.linalg.LinAlgError) as error:
                row["normalflow_status"] = "solver_failed"
                row["normalflow_error"] = str(error)
        rows.append(clean(row))
        previous_frame = frame  # Always the immediately preceding raw image.
        if target_index % 50 == 0:
            print(f"processed {target_index}/{len(images)-1} consecutive intervals", flush=True)
    metadata = dict(
        bag=str(database), image_count=len(images), interval_count=len(rows), origin_stamp_ns=origin_ns,
        gt_source_samples=len(gt_imu), imu_source_samples=len(imus), gt_calibration_closure_max=gt_closure,
        calibration_file=str(calibration), model_file=str(model), config_file=str(args.config),
        timing_policy_file=str(timing_policy),
        timing_policy=json.loads(timing_policy.read_text()),
        assumptions=dict(
            transforms="target GelSight -> reference GelSight; no spatial alignment",
            primary_error="For each estimator: estimated world target = GT world reference * estimated relative transform; compare with GT world target. Relative-chart norm errors are identical by rotational invariance.",
            pose_interpolation="GT IMU position: linear; rotation: SLERP; Gel pose: fixed rigid conversion",
            velocity="GT IMU position central differences at native nonuniform GT times, linearly interpolated; no smoothing",
            gravity_world_mps2=gravity.tolist(),
            initial_accel_bias_mps2=imu_config.get("initial_accel_bias_mps2", [0, 0, 0]),
            initial_gyro_bias_rps=imu_config.get("initial_gyro_bias_rps", [0, 0, 0]),
            imu="native GTSAM IMU-origin preintegration; interval-start held measurements; no visual refinement",
            timestamps="input bag/header nanoseconds must match; no extra shifts, fitting, resampling, extrapolation, or sample holding beyond endpoints; input policy recorded separately",
            normalflow="identity initialization; 3000 curvature candidates; final SCR/CCS recorded separately",
            normalflow_scr_min=track["scr_threshold"], normalflow_ccs_min=track["ccs_threshold"],
            background="average of first ten raw images; no startup repair",
            direction_min_displacement_mm=.1,
            gt_initialization="offline diagnostic oracle; each interval independent; not a deployed estimator accuracy result",
        ),
        source_sha256={str(p): digest(p) for p in (
            database, calibration, model, timing_policy, args.config, Path(__file__),
            Path(inspect.getfile(normalflow)), Path(inspect.getfile(preintegrate_combined)),
        )},
    )
    result = dict(metadata=metadata, summary=summarize(rows), intervals=rows)
    (args.output / "intervals.json").write_text(json.dumps(clean(result), indent=2, allow_nan=False) + "\n")
    (args.output / "summary.json").write_text(json.dumps(clean(result["summary"]), indent=2, allow_nan=False) + "\n")
    (args.output / "config_snapshot.yaml").write_text(yaml.safe_dump(config))
    write_csv(args.output / "intervals.csv", rows)
    plot_results(args.output, rows)
    print(json.dumps(clean(result["summary"]), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
