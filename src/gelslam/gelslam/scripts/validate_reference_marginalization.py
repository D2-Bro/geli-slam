#!/usr/bin/env python3
"""Replay stored ROS images/IMU through the actual Python frontend, without DDS.

The effective configuration and source hashes are saved with timing/dimension
evidence. This is not a live ROS executor or backend optimization benchmark.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
from time import monotonic

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, Imu

from gelslam.core import reference_tracking, retained_prior, tracker as tracker_module
from gelslam.core.inertial import ImuSample, gyro_to_rad_per_sec
from gelslam.core.imu_navigation import gelsight_pose


def _timestamp(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def _bag_records(database, topics, start, end):
    types = dict(image=Image, imu=Imu, gt=PoseStamped)
    records = []
    with sqlite3.connect('file:' + str(database) + '?mode=ro', uri=True) as connection:
        ids = {name: key for key, name in connection.execute('SELECT id, name FROM topics')}
        for kind, topic in topics.items():
            if kind == 'gt' and topic not in ids:
                continue
            if topic not in ids:
                raise ValueError('bag has no %s topic: %s' % (kind, topic))
            for stamp, payload in connection.execute(
                    'SELECT timestamp, data FROM messages WHERE topic_id=? '
                    'AND timestamp BETWEEN ? AND ? ORDER BY timestamp',
                    (ids[topic], int(start * 1e9), int(end * 1e9))):
                records.append((stamp, kind, deserialize_message(bytes(payload), types[kind])))
    return sorted(records, key=lambda item: (item[0], dict(imu=0, gt=1, image=2)[item[1]]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--start", type=float, default=1.)
    parser.add_argument("--end", type=float, default=6.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    config["imu_config"]["frontend_state_mode"] = "reference_marginalization"
    imu = config["imu_config"]
    if imu.get("timestamp_source", "header") != "header":
        parser.error("this replay requires header timestamps; use ROS launch for recorded_source")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config_supplied.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    log = []
    logger = type("Logger", (), {
        "info": lambda self, message: log.append(str(message)),
        "warning": lambda self, message: log.append(str(message)),
    })()
    tracker = tracker_module.Tracker(config["device_config"]["calibration_model_path"],
                                      config, True, logger)
    (args.output / "config_effective_imu.yaml").write_text(yaml.safe_dump(tracker.imu_config, sort_keys=False))
    database = args.bag / "bag_0.db3" if args.bag.is_dir() else args.bag
    records = _bag_records(str(database), dict(
        image=config["device_config"]["image_topic"], imu=imu["topic"],
        gt=imu.get("ground_truth_gel_pose_topic", "/ground_truth/gel_pose")), args.start, args.end)
    # Offline readiness: all bracketing samples are available before processing
    # each image. The estimator still consumes accepted-anchor intervals only.
    for _, kind, message in records:
        time = _timestamp(message.header)
        if kind == "imu":
            gyro = gyro_to_rad_per_sec(
                [message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z],
                imu.get("gyro_input_unit", "rad_s"), imu.get("gyro_max_rate_rad_s", 5.))
            if not tracker.add_imu_sample(ImuSample(time,
                    [message.linear_acceleration.x, message.linear_acceleration.y,
                     message.linear_acceleration.z], gyro)):
                raise RuntimeError("replay IMU was rejected at %.9f" % time)
        elif kind == "gt":
            from scipy.spatial.transform import Rotation
            q, p = message.pose.orientation, message.pose.position
            tracker.add_ground_truth_gel_pose(time,
                Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(), [p.x, p.y, p.z])
    images, bridge = [], CvBridge()
    solves, trial_datums = [], {}
    first_reference_poses = {}
    original_solve = reference_tracking.ReferenceMarginalizationSolver.refine_current

    def capture(solver, *solve_args, **kwargs):
        result = original_solve(solver, *solve_args, **kwargs)
        problem = solver.last_problem
        solves.append(dict(target=solve_args[3].timestamp, seconds=solver.last_solve_time_s,
            success=result is not None, failure=solver.last_refinement_failure,
            solve_dimension=None if problem is None else problem["solve_dimension"],
            retained_dimension=None if problem is None else problem["retained_dimension"],
            rank=None if problem is None else problem["numerical_rank"],
            symmetry_error=None if problem is None else problem["symmetry_error"],
            absorbed_intervals=None if problem is None else problem["absorbed_intervals"]))
        return result

    reference_tracking.ReferenceMarginalizationSolver.refine_current = capture
    try:
        for _, kind, message in records:
            if kind != "image":
                continue
            started = monotonic()
            valid, data = tracker.track(bridge.imgmsg_to_cv2(message, desired_encoding="bgr8"),
                                        message.header, "reference validation")
            s = tracker.frontend_visual_inertial
            prior = s.marginal_prior
            current = tracker._frontend_vi_state
            datum = prior.fixed_values[s.first_pose_id] if isinstance(prior, retained_prior.RetainedPrior) else None
            if datum is not None:
                trial = tracker._trial_index
                original = trial_datums.setdefault(trial, datum.copy())
                if not np.array_equal(datum, original):
                    raise RuntimeError("first contact pose changed within trial %d" % trial)
                if s.reference_id == s.first_pose_id:
                    original_ref = first_reference_poses.setdefault(trial, tracker.trial_T_ref.copy())
                    if not np.array_equal(tracker.trial_T_ref, original_ref):
                        raise RuntimeError("first GelSight reference pose changed")
            images.append(dict(timestamp=_timestamp(message.header), valid=bool(valid),
                seconds=monotonic() - started, keyframes=0 if data is None else len(data[1]),
                mode=tracker.tracking_mode.value,
                reference_timestamp=None if tracker.ref_header is None else _timestamp(tracker.ref_header),
                state_timestamp=None if current is None else current.timestamp,
                retained_dimension=prior.layout.dimension if isinstance(prior, retained_prior.RetainedPrior) else None,
                first_pose_timestamp=None if datum is None else s.first_pose_id.timestamp,
                first_pose=None if datum is None else gelsight_pose(datum, tracker.gelsight_T_imu).tolist()))
    finally:
        reference_tracking.ReferenceMarginalizationSolver.refine_current = original_solve
    successful = [s for s in solves if s["success"]]
    durations = np.array([s["seconds"] for s in successful]) * 1000.
    frame_times = np.array([i["seconds"] for i in images]) * 1000.
    sources = [Path(m.__file__).resolve() for m in (tracker_module, reference_tracking, retained_prior)]
    summary = dict(images=len(images), accepted_images=sum(i["valid"] for i in images),
        keyframes=sum(i["keyframes"] for i in images), solves=len(solves),
        successful_solves=len(successful), failed_solves=len(solves) - len(successful),
        first_poses_exactly_fixed=bool(trial_datums),
        first_pose_trials=len(trial_datums),
        solve_dimensions=sorted({s["solve_dimension"] for s in successful}),
        retained_dimensions=sorted({i["retained_dimension"] for i in images if i["retained_dimension"] is not None}),
        solve_ms_median=None if not len(durations) else float(np.median(durations)),
        solve_ms_p95=None if not len(durations) else float(np.percentile(durations, 95)),
        image_ms_median=None if not len(frame_times) else float(np.median(frame_times)),
        image_ms_p95=None if not len(frame_times) else float(np.percentile(frame_times, 95)),
        finite_states=all(np.isfinite(v).all() for v in (current.pose, current.velocity,
            current.accel_bias, current.gyro_bias)) if current is not None else False,
        scope="offline ROS message replay; CPU; prebuffered IMU/GT; no live DDS/backend")
    result = dict(summary=summary, config=str(args.config.resolve()), bag=str(database),
        interval=[args.start, args.end], python=sys.executable,
        sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        images=images, solves=solves, logs=log)
    (args.output / "evidence.json").write_text(json.dumps(result, indent=2, allow_nan=False,
        default=lambda value: value.item() if isinstance(value, np.generic) else value.tolist()) + "\n")
    (args.output / "tracker.log").write_text("\n".join(log) + "\n")
    print(json.dumps(summary, indent=2))
    if not successful or 36 not in summary["solve_dimensions"] or summary["keyframes"] < 2:
        raise SystemExit("insufficient successful tracking/reference-change evidence")


if __name__ == "__main__":
    main()
