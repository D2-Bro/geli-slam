import numpy as np
import open3d as o3d

from gelslam.scripts.poisson_remesh import (
    fused_points_with_normals, is_closed_manifold, poisson_remesh)
from gelslam.utils import pointcloud2mesh


def _sphere_indentation(radius_px=40.0, size=101, depth_px=10.0):
    """gs_sdk-style height map: a sphere pressed into the gel lowers H."""
    yy, xx = np.mgrid[:size, :size] - (size - 1) / 2.0
    r2 = xx ** 2 + yy ** 2
    contact = r2 < radius_px ** 2 - (radius_px - depth_px) ** 2
    height = np.zeros((size, size), dtype=np.float32)
    # Sphere centre on +z: the object occupies z > H.
    centre_z = radius_px - depth_px
    height[contact] = centre_z - np.sqrt(radius_px ** 2 - r2[contact])
    return height, contact, centre_z


def test_patch_normals_point_out_of_the_object():
    height, contact, centre_z = _sphere_indentation()
    yy, xx = np.nonzero(contact)
    points = np.stack([xx - 50.0, yy - 50.0, height[contact]], axis=1)
    mesh = pointcloud2mesh(points, contact)
    cloud = fused_points_with_normals(mesh)
    outward = np.asarray(cloud.points) - [0.0, 0.0, centre_z]
    cosine = np.sum(outward * np.asarray(cloud.normals), axis=1) / np.linalg.norm(outward, axis=1)
    assert np.mean(cosine > 0.9) > 0.95
    flipped = fused_points_with_normals(mesh, flip_normals=True)
    np.testing.assert_allclose(np.asarray(flipped.normals), -np.asarray(cloud.normals))


def test_poisson_closes_sphere_patches():
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01, resolution=60)
    # Two overlapping open patches, as from two coverage keyframes.
    vertices = np.asarray(sphere.vertices)
    patches = o3d.geometry.TriangleMesh()
    for keep in (vertices[:, 2] > -0.004, vertices[:, 2] < 0.004):
        patch = o3d.geometry.TriangleMesh(sphere)
        patch.remove_vertices_by_mask(~keep)
        patches += patch
    assert not is_closed_manifold(patches)
    mesh, _ = poisson_remesh(fused_points_with_normals(patches), depth=6)
    assert is_closed_manifold(mesh)
    # Genus-0 closed surface: Euler characteristic V - E + F = 2 gives F = 2V - 4.
    assert len(mesh.triangles) == 2 * len(mesh.vertices) - 4
    radius = np.linalg.norm(np.asarray(mesh.vertices), axis=1)
    assert abs(np.median(radius) - 0.01) < 3e-4
    trimmed, _ = poisson_remesh(fused_points_with_normals(patches), depth=6, density_quantile=0.2)
    assert len(trimmed.vertices) < len(mesh.vertices)
    assert not is_closed_manifold(trimmed)
