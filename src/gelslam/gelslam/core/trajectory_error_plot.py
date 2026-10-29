"""Error-coloured final trajectories and rendering from saved evaluation CSVs."""

import csv
import json
import os

import numpy as np


OUTPUT_FILES = (
    "trajectory_translation_error.png",
    "trajectory_rotation_error.png",
)


def write_trajectory_error_colormaps(output_dir, estimated, ground_truth,
                                    translation_errors_m, rotation_errors_deg, label="GelSLAM",
                                    alignment_label="first-pose aligned"):
    """Save pose error maps; State.velocity evaluation owns the velocity outputs."""
    for filename, values, title, unit in (
            (OUTPUT_FILES[0], np.asarray(translation_errors_m) * 1000, "Translation error", "mm"),
            (OUTPUT_FILES[1], np.asarray(rotation_errors_deg), "Rotation error", "deg")):
        write_error_colormap(output_dir, estimated, ground_truth, values, filename,
                             label + ": " + alignment_label + " final backend GelSight-origin trajectory",
                             title, unit)


def write_error_colormap(output_dir, estimated, ground_truth, values, filename,
                         label, title, unit, groups=None):
    """Colour sample errors and connect every adjacent stored pose.

    ``groups`` is retained for callers; time gaps and trial changes do not
    interrupt the displayed estimate or matched GT lines.
    """
    values = np.asarray(values, dtype=float)
    os.makedirs(output_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(output_dir, ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    positions = np.asarray([sample[3][:3, 3] for sample in estimated])
    truth = np.asarray([sample[3][:3, 3] for sample in ground_truth])
    times = np.asarray([sample[0] for sample in estimated])
    if len(positions) < 2 or positions.shape != truth.shape:
        raise ValueError("At least two matched aligned poses are required")
    if not np.allclose(times, [sample[0] for sample in ground_truth], atol=1e-9, rtol=0):
        raise ValueError("Estimate and GT image timestamps must match")
    segments = np.stack((positions[:-1], positions[1:]), axis=1)
    bounds = np.vstack((positions, truth))
    centers = (bounds.min(axis=0) + bounds.max(axis=0)) / 2
    radius = max(float(np.ptp(bounds, axis=0).max()) * 0.55, 1e-6)

    if values.shape != times.shape or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Each matched pose needs one finite non-negative error")
    norm = Normalize(vmin=0, vmax=max(float(values.max()), 1e-9))
    segment_values = (values[:-1] + values[1:]) / 2
    figure = plt.figure(figsize=(15, 7))
    grid = figure.add_gridspec(1, 3, width_ratios=(1, 1, 0.035),
                              left=0.03, right=0.95, bottom=0.13,
                              top=0.84, wspace=0.3)
    spatial = figure.add_subplot(grid[0], projection="3d")
    planar = figure.add_subplot(grid[1])
    color_axis = figure.add_subplot(grid[2])
    for axis, dimensions, collection_type in (
            (spatial, 3, Line3DCollection), (planar, 2, LineCollection)):
        collection = collection_type(segments[:, :, :dimensions],
                                     cmap="turbo", norm=norm, linewidths=1.5)
        collection.set_array(segment_values)
        axis.add_collection(collection)
        axis.plot(*truth[:, :dimensions].T, color="0.35",
                  linestyle="--", linewidth=1, alpha=0.7)
        scatter_options = {"depthshade": False} if dimensions == 3 else {"zorder": 3}
        axis.scatter(*positions[:, :dimensions].T, c=values, cmap="turbo",
                     norm=norm, s=7, **scatter_options)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.grid(alpha=0.25)
    spatial.set_zlabel("z [m]")
    spatial.set_xlim(centers[0] - radius, centers[0] + radius)
    spatial.set_ylim(centers[1] - radius, centers[1] + radius)
    spatial.set_zlim(centers[2] - radius, centers[2] + radius)
    spatial.set_box_aspect((1, 1, 1))
    for axis in (spatial.xaxis, spatial.yaxis, spatial.zaxis):
        axis.set_major_locator(MaxNLocator(5))
    spatial.view_init(elev=25, azim=-55)
    spatial.set_title("3D trajectory")
    planar.set_aspect("equal", adjustable="box")
    planar.autoscale_view()
    planar.set_title("XY position projection")
    planar.legend(handles=[
        Line2D([], [], color=plt.get_cmap("turbo")(0.6), label="Estimate (error colour)"),
        Line2D([], [], color="0.35", linestyle="--", label="Matched GT"),
    ], loc="best", fontsize=9)
    colorbar = figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="turbo"),
                              cax=color_axis)
    colorbar.set_label(f"{title} [{unit}]")
    rmse = np.sqrt(np.mean(values**2))
    figure.suptitle(f"{label}\n"
                   f"{title} | RMSE {rmse:.4f} {unit} | {len(values)} matched poses")
    figure.savefig(os.path.join(output_dir, filename), dpi=180)
    plt.close(figure)


def write_velocity_component_colormaps(output_dir, estimated, ground_truth, errors_mps, groups=None):
    """One figure: signed X/Y/Z State velocity errors, 3D above XY.

    Components are in the aligned GT world axes. All six panels share a
    symmetric colour scale; positive means estimate exceeds GT on that axis.
    All stored poses are connected, including time gaps and trial changes.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(output_dir, ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    positions = np.asarray([sample[3][:3, 3] for sample in estimated])
    truth = np.asarray([sample[3][:3, 3] for sample in ground_truth])
    times = np.asarray([sample[0] for sample in estimated])
    errors = np.asarray(errors_mps, dtype=float) * 1000
    if (len(times) < 2 or positions.shape != truth.shape or errors.shape != (len(times), 3)
            or not np.isfinite(errors).all() or not np.allclose(
                times, [sample[0] for sample in ground_truth], atol=1e-9, rtol=0)):
        raise ValueError("Component maps require matched poses and finite Nx3 velocity errors")
    segments = np.stack((positions[:-1], positions[1:]), axis=1)
    bounds = np.vstack((positions, truth))
    center = (bounds.max(axis=0) + bounds.min(axis=0)) / 2
    radius = max(float(np.ptp(bounds, axis=0).max()) * 0.55, 1e-6)
    maximum = max(float(np.max(np.abs(errors))), 1e-9)
    norm = TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum)
    figure = plt.figure(figsize=(20, 11))
    grid = figure.add_gridspec(2, 4, width_ratios=(1, 1, 1, 0.04),
                              left=0.03, right=0.94, bottom=0.08, top=0.88,
                              wspace=0.32, hspace=0.25)
    for index, component in enumerate("XYZ"):
        values = errors[:, index]
        segment_values = (values[:-1] + values[1:]) / 2
        rmse = np.sqrt(np.mean(values**2))
        for row, dimensions, collection_type in ((0, 3, Line3DCollection), (1, 2, LineCollection)):
            axis = figure.add_subplot(grid[row, index], projection="3d" if row == 0 else None)
            collection = collection_type(segments[:, :, :dimensions], cmap="coolwarm", norm=norm,
                                         linewidths=1.5)
            collection.set_array(segment_values)
            axis.add_collection(collection)
            axis.plot(*truth[:, :dimensions].T, color="0.4", linestyle="--", linewidth=1)
            scatter_options = {"depthshade": False} if row == 0 else {"zorder": 3}
            axis.scatter(*positions[:, :dimensions].T, c=values, cmap="coolwarm", norm=norm,
                         s=7, **scatter_options)
            axis.set_xlabel("x [m]")
            axis.set_ylabel("y [m]")
            axis.grid(alpha=0.25)
            if row == 0:
                axis.set_zlabel("z [m]")
                axis.set_xlim(center[0] - radius, center[0] + radius)
                axis.set_ylim(center[1] - radius, center[1] + radius)
                axis.set_zlim(center[2] - radius, center[2] + radius)
                axis.set_box_aspect((1, 1, 1))
                axis.view_init(elev=25, azim=-55)
                for coordinate in (axis.xaxis, axis.yaxis, axis.zaxis):
                    coordinate.set_major_locator(MaxNLocator(4))
                axis.set_title(f"{component} velocity error | RMSE {rmse:.4f} mm/s")
            else:
                axis.set_aspect("equal", adjustable="box")
                axis.autoscale_view()
                axis.set_title(f"{component} error: IMU-origin XY trajectory")
    colorbar = figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="coolwarm"),
                              cax=figure.add_subplot(grid[:, 3]))
    colorbar.set_label("Signed component error: estimate - GT [mm/s]")
    figure.suptitle("GelSLAM: frontend State.velocity component errors on the IMU-origin trajectory\n"
                   "Components in GT world axes | "
                   "Blue: negative | White: zero | Red: positive | Shared colour scale", y=0.97)
    figure.legend(handles=[Line2D([], [], color="tab:red", label="State trajectory (component error colour)"),
                           Line2D([], [], color="0.4", linestyle="--", label="Matched GT IMU trajectory")],
                  loc="lower center", ncol=2, bbox_to_anchor=(0.48, 0.01))
    figure.savefig(os.path.join(output_dir, "trajectory_velocity_error_components.png"), dpi=180)
    plt.close(figure)


def render_saved_trajectory(output_dir, label="GelSLAM"):
    """Add colour plots to saved results without changing CSVs or metrics."""
    from scipy.spatial.transform import Rotation

    def read_poses(filename):
        samples = []
        with open(os.path.join(output_dir, filename), newline="") as handle:
            for row in csv.DictReader(handle):
                pose = np.eye(4)
                pose[:3, 3] = [float(row[key]) for key in ("tx_m", "ty_m", "tz_m")]
                pose[:3, :3] = Rotation.from_quat(
                    [float(row[key]) for key in ("qx", "qy", "qz", "qw")]).as_matrix()
                samples.append((float(row["timestamp_s"]), int(row["fid"]),
                                int(row["ref_kid"]), pose))
        return samples

    estimated = read_poses("trajectory_final_aligned.csv")
    truth = read_poses("trajectory_ground_truth_matched.csv")
    if len(estimated) != len(truth):
        raise ValueError("Saved estimate and GT must have equal sample counts")
    errors_path = os.path.join(output_dir, "trajectory_errors.csv")
    if os.path.isfile(errors_path):
        with open(errors_path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(estimated) or not np.allclose(
                [float(row["timestamp_s"]) for row in rows],
                [sample[0] for sample in estimated], atol=1e-9, rtol=0):
            raise ValueError("Saved error rows must match aligned trajectory timestamps")
        translations = np.asarray([float(row["translation_error_m"]) for row in rows])
        rotations = np.asarray([float(row["rotation_error_deg"]) for row in rows])
    else:
        translations = np.asarray([np.linalg.norm(a[3][:3, 3] - b[3][:3, 3])
                                   for a, b in zip(estimated, truth)])
        rotations = np.asarray([Rotation.from_matrix(a[3][:3, :3].T @ b[3][:3, :3]).magnitude()
                                for a, b in zip(estimated, truth)]) * 180 / np.pi
    metadata_path = os.path.join(output_dir, "trajectory_metrics.json")
    metadata = {}
    if os.path.isfile(metadata_path):
        with open(metadata_path) as handle:
            metadata = json.load(handle)
    alignment_label = ("first-pose aligned" if metadata.get("alignment_method") == "first_matched_pose"
                       else "SE(3)-aligned")
    write_trajectory_error_colormaps(output_dir, estimated, truth, translations, rotations, label,
                                    alignment_label=alignment_label)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="Directory containing matched aligned trajectory CSVs")
    parser.add_argument("--label", default="GelSLAM")
    args = parser.parse_args()
    render_saved_trajectory(os.path.abspath(os.path.expanduser(args.output_dir)), args.label)


if __name__ == "__main__":
    main()
