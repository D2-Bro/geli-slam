"""Contact kernels for keyframe selection (kernel_score) and the kernel_overlap keyframe rule.

Config block ``gelslam_config.track_config.keyframe_selection``::

    keyframe_selection:
      method: kernel_score          # current | kernel_score
      window_size: 8                # accepted contact frames after the reference, newest last
      kernel_scale: 4.0             # c: H = c Sigma (d^2 = 1 on the edge of a uniform disk)
      covariance_epsilon_px2: 0.001  # eps I_2 added to each projected 2D covariance
      kernel_ellipse_vertices: 32   # polygon vertices of each kernel ellipse (>= 8)
      kernel_score_alpha: 0.5       # weight of A_m / A_peak against O(m)

Contact kernel of a frame m: its eroded contact pixels (normalflow Frame.C) lifted to
3D with the height map (normalflow axes, metres); p_bar_m and Sigma_m are their
indentation-weighted mean and covariance, weight = percentile_85(H) - H(u) clipped at
0 (the depth below the level that gs_sdk's height contact test uses). In image t,
with (R, t) = T_t^-1 T_m and s metres per pixel: mu_{m->t} = pixel of R p_bar_m + t,
Sigma_{m->t} = [R Sigma_m R^T]_xy / s^2 + eps I_2, H_m = c Sigma_{m->t}, and the
kernel ellipse E_m = {x : (x - mu)^T H_m^-1 (x - mu) <= 1}, drawn as an inscribed
polygon. O(m) = area(E_m ∩ E_t) / area(E_t) (convex polygon intersection, OpenCV).

kernel_score: at a keyframe trigger for the newest accepted frame t, the window holds
the accepted contact frames after the reference keyframe K_k (t last). Select
argmax_m alpha A_m / A_peak + (1 - alpha) O(m), A_m = |M_m| (eroded contact pixels),
A_peak = max_m A_m, O(t) = 1; equal scores: the more recent frame. A past frame m
becomes K_{k+1} with the pose that the reference-marginalization prior retains for
it (ReferenceMarginalizationSolver.window_pose_timestamps).

keyframe_rule kernel_overlap (tracker): t becomes a keyframe when the reference
keyframe's ellipse moved into t covers less than keyframe_overlap_threshold of E_t,
O_K(t) = area(E_K->t ∩ E_t) / area(E_t). It uses the kernel definition above for any
method.
"""

from collections import namedtuple
from dataclasses import dataclass

import cv2
import numpy as np

CONFIG_KEY = "keyframe_selection"
METHODS = ("current", "kernel_score")
_REQUIRED = ("method", "window_size", "kernel_scale", "covariance_epsilon_px2",
             "kernel_ellipse_vertices", "kernel_score_alpha")
# Kernel definition shared by kernel_score and keyframe_rule kernel_overlap.
KERNEL_KEYS = ("kernel_scale", "covariance_epsilon_px2", "kernel_ellipse_vertices")


def _block(track_config):
    block = (track_config or {}).get(CONFIG_KEY)
    if not isinstance(block, dict):
        raise ValueError("track_config.%s must be a mapping with %s" % (CONFIG_KEY, ", ".join(_REQUIRED)))
    unknown = set(block) - set(_REQUIRED)
    missing = [key for key in _REQUIRED if key not in block]
    if unknown or missing:
        raise ValueError("track_config.%s: unknown %s, missing %s"
                         % (CONFIG_KEY, sorted(unknown), missing))
    return block


def _validate_kernel(block):
    """Kernel definition: H = kernel_scale * (projected 2D covariance + covariance_epsilon_px2 I),
    its ellipse E = {d^2 <= 1} drawn as an inscribed polygon of kernel_ellipse_vertices."""
    kernel = {}
    for key in ("kernel_scale", "covariance_epsilon_px2"):
        value = block[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value) or value <= 0.0):
            raise ValueError("track_config.%s.%s must be a positive number" % (CONFIG_KEY, key))
        kernel[key] = float(value)
    vertices = block["kernel_ellipse_vertices"]
    if isinstance(vertices, bool) or not isinstance(vertices, int) or vertices < 8:
        raise ValueError("track_config.%s.kernel_ellipse_vertices must be an integer >= 8" % CONFIG_KEY)
    kernel["kernel_ellipse_vertices"] = vertices
    return kernel


def resolve_kernel_settings(track_config):
    """Validated KERNEL_KEYS of the keyframe_selection block, for any method."""
    return _validate_kernel(_block(track_config))


def resolve_keyframe_selection(track_config):
    """Validated settings, or None for the current (newest-frame) selection (also without the block)."""
    if (track_config or {}).get(CONFIG_KEY) is None:
        return None
    settings = dict(_block(track_config))
    if settings["method"] not in METHODS:
        raise ValueError("track_config.%s.method must be one of %s" % (CONFIG_KEY, ", ".join(METHODS)))
    settings.update(_validate_kernel(settings))
    size = settings["window_size"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 2:
        raise ValueError("track_config.%s.window_size must be an integer >= 2" % CONFIG_KEY)
    alpha = settings["kernel_score_alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0.0 <= alpha <= 1.0:
        raise ValueError("track_config.%s.kernel_score_alpha must be a number in [0, 1]" % CONFIG_KEY)
    settings["kernel_score_alpha"] = float(alpha)
    return None if settings["method"] == "current" else settings


def _twice_signed_area(poly):
    """Twice the signed shoelace area; positive for (u right, v down) positive orientation."""
    x, y = poly[:, 0], poly[:, 1]
    return float(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]) + x[-1] * y[0] - y[-1] * x[0])


def polygon_area(poly):
    """Shoelace area in pixel^2."""
    return 0.5 * abs(_twice_signed_area(poly)) if len(poly) >= 3 else 0.0


def convex_hull(points):
    """Convex hull (N, 2) of 2D points, float64 vertices, positive orientation (OpenCV order)."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3:
        return np.empty((0, 2))
    index = cv2.convexHull(points.astype(np.float32).reshape(-1, 1, 2), returnPoints=False).ravel()
    hull = points[index] if len(index) >= 3 else np.empty((0, 2))
    if len(hull) < 3:
        return np.empty((0, 2))
    return hull[::-1].copy() if _twice_signed_area(hull) < 0 else hull


def convex_intersection(subject, clip):
    """(area, polygon) of the intersection of the convex hulls of two polygons (OpenCV, float32)."""
    subject, clip = convex_hull(subject), convex_hull(clip)
    if len(subject) < 3 or len(clip) < 3:
        return 0.0, np.empty((0, 2))
    area, polygon = cv2.intersectConvexConvex(
        subject.astype(np.float32).reshape(-1, 1, 2), clip.astype(np.float32).reshape(-1, 1, 2))
    if polygon is None or area <= 0.0:
        return 0.0, np.empty((0, 2))
    return float(area), polygon.reshape(-1, 2).astype(np.float64)


def contact_moments(contact_mask, height_map, mm_per_pixel):
    """(|M|, mean (3,), covariance (3, 3)) of the contact pixels lifted to 3D (metres).

    Axes as normalflow.utils.height2pointcloud. Weight = percentile_85(H) - H(u),
    clipped at 0 (uniform if every weight is 0).
    """
    mask = np.asarray(contact_mask, dtype=bool)
    v, u = np.nonzero(mask)
    if len(u) == 0:
        return 0.0, None, None
    height = np.asarray(height_map, dtype=np.float64)
    rows, cols = height.shape[:2]
    weight = np.clip(np.percentile(height, 85) - height[v, u], 0.0, None)
    if weight.sum() <= 0.0:
        weight = np.ones_like(weight)
    points = np.column_stack((u - cols / 2 + 0.5, v - rows / 2 + 0.5, height[v, u])) * (
        mm_per_pixel / 1000.0)
    weight = weight / weight.sum()
    mean = weight @ points
    centred = points - mean
    return float(len(u)), mean, (centred * weight[:, None]).T @ centred


KernelMoments = namedtuple("KernelMoments", "mask_area mean3d cov3d image_shape")


def kernel_moments(frame, mm_per_pixel):
    """KernelMoments of a frame's eroded contact mask C and height map H (see contact_moments)."""
    return KernelMoments(*contact_moments(frame.C, frame.H, mm_per_pixel), tuple(np.shape(frame.C)[:2]))


@dataclass
class WindowFrame:
    """One accepted contact frame of the selection window."""
    fid: int
    timestamp: float          # IMU-clock time of its accepted navigation state
    header: object
    frame: object             # normalflow Frame (reference surface if selected)
    state: object             # accepted NavigationState at timestamp
    image_shape: tuple
    mask_area: float          # |M_m|, eroded contact mask pixels
    mean3d: np.ndarray        # (3,) indentation-weighted contact mean, metres
    cov3d: np.ndarray         # (3, 3) indentation-weighted contact covariance, m^2


def make_window_frame(fid, timestamp, header, frame, state, mm_per_pixel):
    moments = kernel_moments(frame, mm_per_pixel)
    return WindowFrame(fid, float(timestamp), header, frame, state.copy(), moments.image_shape,
                       moments.mask_area, moments.mean3d, moments.cov3d)


def projected_moments(entry, pose, current, current_pose, kernel, mm_per_pixel):
    """(mu_{m->t} pixels (2,), H_m^-1 (2, 2)) of frame m (entry, GelSight pose) in image t."""
    current_T_entry = np.linalg.solve(current_pose, pose)
    rotation, translation = current_T_entry[:3, :3], current_T_entry[:3, 3]
    rows, cols = current.image_shape
    mean = (rotation @ entry.mean3d + translation)[:2] * (1000.0 / mm_per_pixel)
    mu = mean + np.array([cols / 2 - 0.5, rows / 2 - 0.5])
    metres_per_pixel = mm_per_pixel / 1000.0
    covariance = ((rotation @ entry.cov3d @ rotation.T)[:2, :2] / metres_per_pixel ** 2
                  + kernel["covariance_epsilon_px2"] * np.eye(2))
    return mu, np.linalg.inv(kernel["kernel_scale"] * covariance)


def kernel_ellipse(mu, h_inv, vertices):
    """(vertices, 2) polygon inscribed in E = {x : (x - mu)^T h_inv (x - mu) <= 1}."""
    values, vectors = np.linalg.eigh(np.linalg.inv(h_inv))
    angles = 2.0 * np.pi * np.arange(vertices) / vertices
    unit = np.vstack((np.cos(angles), np.sin(angles)))
    return np.asarray(mu, dtype=float) + (vectors @ (np.sqrt(values)[:, None] * unit)).T


def kernel_overlap_ratio(mu_m, h_inv_m, mu_t, h_inv_t, vertices):
    """O = area(E_m ∩ E_t) / area(E_t), both ellipses as inscribed polygons."""
    ellipse_t = kernel_ellipse(mu_t, h_inv_t, vertices)
    area_t = polygon_area(ellipse_t)
    if area_t <= 0.0:
        return 0.0
    shared = convex_intersection(kernel_ellipse(mu_m, h_inv_m, vertices), ellipse_t)[0]
    return min(1.0, shared / area_t)


def reference_kernel_overlap(reference, reference_pose, current, current_pose, kernel, mm_per_pixel):
    """O_K(t) = area(E_{K->t} ∩ E_t) / area(E_t): share of frame t's kernel ellipse that the
    reference keyframe K's ellipse covers once moved into image t (T_t^-1 T_K), at most 1.

    reference, current: KernelMoments; poses: GelSight trial poses of K and t. 0 if either
    frame has no contact pixels.
    """
    if reference.mean3d is None or current.mean3d is None:
        return 0.0
    mu_k, h_k = projected_moments(reference, reference_pose, current, current_pose, kernel, mm_per_pixel)
    mu_t, h_t = projected_moments(current, current_pose, current, current_pose, kernel, mm_per_pixel)
    return kernel_overlap_ratio(mu_k, h_k, mu_t, h_t, kernel["kernel_ellipse_vertices"])


def select_window_frame(window, poses, settings, mm_per_pixel):
    """kernel_score: return (selected WindowFrame, info). window[-1] is the newest frame t.

    poses[i] is the current GelSight trial pose estimate of window[i]. Frames
    without contact moments are not candidates; without them at t, t is kept.
    """
    current, current_pose = window[-1], poses[-1]
    valid = [(f, p) for f, p in zip(window, poses) if f.mean3d is not None]
    if not any(f is current for f, _ in valid):
        return current, dict(status="current_without_contact_moments")
    peak = max((f for f, _ in valid), key=lambda f: f.mask_area)
    mu_t, h_t = projected_moments(current, current_pose, current, current_pose, settings, mm_per_pixel)
    alpha, vertices = settings["kernel_score_alpha"], settings["kernel_ellipse_vertices"]
    rows = []
    for index, (f, pose) in enumerate(valid):
        overlap = 1.0 if f is current else kernel_overlap_ratio(
            *projected_moments(f, pose, current, current_pose, settings, mm_per_pixel), mu_t, h_t, vertices)
        rows.append((index, f, overlap, alpha * f.mask_area / peak.mask_area + (1.0 - alpha) * overlap))
    # Highest score; equal scores: the more recent frame (later in the window).
    _, selected, overlap, score = max(rows, key=lambda row: (row[3], row[0]))
    return selected, dict(status="kernel_score", peak_fid=peak.fid, selected_overlap=float(overlap),
                          selected_score=float(score),
                          frames=[(f.fid, float(f.mask_area), float(o), float(sc)) for _, f, o, sc in rows])
