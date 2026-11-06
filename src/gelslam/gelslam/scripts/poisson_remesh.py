"""Offline re-meshing of a GelSLAM fast-fusion mesh (paper Sec. IV-C, second phase).

GelSLAM writes ``reconstructed_mesh.ply``: per coverage keyframe, the neighbour-averaged
point cloud is triangulated on its pixel grid and the patches are concatenated (fast
fusion; not watertight).  This script applies Poisson Surface Reconstruction to that
fused surface to get one closed (watertight) mesh.

Oriented normals come from the input triangles.  ``pointcloud2mesh`` winds every
patch triangle so its normal points along -z of the keyframe GelSight frame.  Contact
lowers the gs_sdk height map (contact mask: H below the 85th percentile cutoff), so
the object lies on +z of the measured surface and -z points out of the object, which
is the orientation Poisson needs.  ``--flip-normals`` reverses it.

The paper does not give its Poisson settings; the defaults below are not the paper's.
"""

import argparse
import json
import os
import time

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def fused_points_with_normals(mesh, voxel_size_m=0.0, flip_normals=False):
    """Return the mesh vertices as a point cloud with area-weighted vertex normals."""
    if not mesh.has_triangles():
        raise ValueError("input mesh has no triangles")
    mesh = o3d.geometry.TriangleMesh(mesh)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    cloud = o3d.geometry.PointCloud(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    cloud.normals = o3d.utility.Vector3dVector(-normals if flip_normals else normals)
    if mesh.has_vertex_colors():
        cloud.colors = mesh.vertex_colors
    if voxel_size_m > 0.0:
        cloud = cloud.voxel_down_sample(voxel_size_m)
        cloud.normalize_normals()
    return cloud


def poisson_remesh(cloud, depth=9, scale=1.1, linear_fit=False, density_quantile=0.0,
                   n_threads=-1):
    """Run Poisson reconstruction; optionally drop the lowest-density vertices.

    density_quantile 0 keeps the closed surface.  A positive value removes vertices
    whose Poisson density is below that quantile (extrapolated, unscanned regions);
    the result then has boundaries.
    """
    if not 0.0 <= density_quantile < 1.0:
        raise ValueError("density_quantile must be in [0, 1)")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=depth, scale=scale, linear_fit=linear_fit, n_threads=n_threads)
    densities = np.asarray(densities)
    if density_quantile > 0.0:
        mesh.remove_vertices_by_mask(densities < np.quantile(densities, density_quantile))
    mesh.compute_vertex_normals()
    return mesh, densities


def is_closed_manifold(mesh):
    """True when every edge has exactly two triangles and every vertex is manifold.

    Open3D's is_watertight() also requires no self-intersection; Poisson output can
    contain near-degenerate slivers that fail that test on a closed surface.
    """
    return bool(mesh.is_edge_manifold(allow_boundary_edges=False) and mesh.is_vertex_manifold())


def mesh_summary(mesh, self_intersection=False):
    summary = {
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "closed_manifold": is_closed_manifold(mesh),
        "boundary_or_nonmanifold_edges": len(mesh.get_non_manifold_edges(allow_boundary_edges=False)),
        "extent_mm": (1e3 * (mesh.get_max_bound() - mesh.get_min_bound())).round(3).tolist(),
    }
    if self_intersection:
        summary["self_intersecting"] = bool(mesh.is_self_intersecting())
    return summary


def distance_to_input(mesh, cloud, far_m=0.0005):
    """Distance of each output vertex to the nearest fused input point.

    Vertices far from every input point lie in extrapolated (unscanned) regions; on an
    open, one-sided scan Poisson closes the surface toward the reconstruction cube.
    """
    distances, _ = cKDTree(np.asarray(cloud.points)).query(np.asarray(mesh.vertices))
    return {
        "median_mm": float(1e3 * np.median(distances)),
        "p99_mm": float(1e3 * np.quantile(distances, 0.99)),
        "max_mm": float(1e3 * distances.max()),
        "fraction_beyond_%.1f_mm" % (1e3 * far_m): float(np.mean(distances > far_m)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", help="fast-fusion mesh, e.g. gelslam_online/reconstructed_mesh.ply")
    parser.add_argument("-o", "--output", help="output mesh (default: <input>_poisson.ply)")
    parser.add_argument("--depth", type=int, default=9,
                        help="octree depth; cell = scale * largest extent / 2^depth (default 9)")
    parser.add_argument("--scale", type=float, default=1.1,
                        help="reconstruction cube / sample bounding cube ratio (default 1.1)")
    parser.add_argument("--linear-fit", action="store_true",
                        help="place iso-vertices by linear interpolation")
    parser.add_argument("--voxel-size-m", type=float, default=0.0,
                        help="voxel down-sampling of the fused points before Poisson; 0 keeps all")
    parser.add_argument("--density-quantile", type=float, default=0.0,
                        help="remove vertices below this Poisson-density quantile; 0 keeps the "
                             "watertight surface")
    parser.add_argument("--flip-normals", action="store_true", help="reverse the input normals")
    parser.add_argument("--threads", type=int, default=-1, help="Poisson threads (-1: all)")
    parser.add_argument("--check-self-intersection", action="store_true",
                        help="also test the output for self-intersecting triangles (slow)")
    args = parser.parse_args()

    output = args.output or os.path.splitext(args.input)[0] + "_poisson.ply"
    start = time.time()
    fused = o3d.io.read_triangle_mesh(args.input)
    cloud = fused_points_with_normals(fused, args.voxel_size_m, args.flip_normals)
    mesh, densities = poisson_remesh(cloud, args.depth, args.scale, args.linear_fit,
                                     args.density_quantile, args.threads)
    if not o3d.io.write_triangle_mesh(output, mesh, write_ascii=False, compressed=False):
        raise RuntimeError("could not write %s" % output)
    extent = float(np.max(fused.get_max_bound() - fused.get_min_bound()))
    report = {
        "input": os.path.abspath(args.input),
        "output": os.path.abspath(output),
        "input_mesh": mesh_summary(fused),
        "poisson_points": len(cloud.points),
        "settings": {"depth": args.depth, "scale": args.scale, "linear_fit": args.linear_fit,
                     "voxel_size_m": args.voxel_size_m,
                     "density_quantile": args.density_quantile,
                     "flip_normals": args.flip_normals},
        "approx_cell_mm": 1e3 * args.scale * extent / 2 ** args.depth,
        "density_range": [float(densities.min()), float(densities.max())],
        "output_mesh": mesh_summary(mesh, args.check_self_intersection),
        "output_vertex_distance_to_input": distance_to_input(mesh, cloud),
        "seconds": round(time.time() - start, 2),
    }
    with open(os.path.splitext(output)[0] + ".json", "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
