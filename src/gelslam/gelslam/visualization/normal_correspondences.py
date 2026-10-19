"""Render frontend normal-flow projections in a side-by-side RGB image."""

import cv2
import numpy as np
from normalflow.utils import height2pointcloud, wide_remap


DEFAULT_CORRESPONDENCE_MAX_LINES = 30


def normal_map_to_rgb8(normals):
    """Encode sensor-frame XYZ normals as RGB; invalid pixels are black."""
    normals = np.asarray(normals)
    valid = np.isfinite(normals).all(axis=-1)
    rgb = np.zeros(normals.shape, dtype=np.uint8)
    rgb[valid] = np.rint(
        (np.clip(normals[valid], -1.0, 1.0) + 1.0) * 127.5
    ).astype(np.uint8)
    return rgb


def project_normal_correspondences(ref, current, ref_T_current, ppmm,
                                   n_samples=3000):
    """Use original NormalFlow curvature selection and target contact filtering.

    Return reference pixel centers and their subpixel current-image positions
    at the supplied pose. These are geometric projections, not feature matches.
    """
    count = min(int(n_samples), ref.L.size)
    empty = np.empty((0, 2), dtype=np.float32)
    if count == 0:
        return empty, empty.copy()
    threshold = np.partition(np.abs(ref.L).ravel(), -count)[-count] + 1e-6
    mask = np.abs(ref.L) >= threshold
    y, x = np.nonzero(mask)
    if len(x) == 0:
        return empty, empty.copy()
    points = height2pointcloud(ref.H, mask, ppmm)
    current_T_ref = np.linalg.inv(ref_T_current).astype(np.float32)
    mapped = points @ current_T_ref[:3, :3].T + current_T_ref[:3, 3]
    xx = (mapped[:, 0] * 1000.0 / ppmm + ref.N.shape[1] / 2 - 0.5)
    yy = (mapped[:, 1] * 1000.0 / ppmm + ref.N.shape[0] / 2 - 0.5)
    contact = wide_remap(current.C.astype(np.float32), xx, yy,
                         mode=cv2.INTER_NEAREST)[:, 0] > 0.5
    valid = (contact & np.isfinite(mapped).all(axis=1)
             & (xx >= 0) & (xx < current.N.shape[1])
             & (yy >= 0) & (yy < current.N.shape[0]))
    return np.column_stack((x, y))[valid], np.column_stack((xx, yy))[valid]


def render_normal_correspondences(reference, current, reference_xy, current_xy,
                                  reference_label, current_label, status,
                                  max_lines=DEFAULT_CORRESPONDENCE_MAX_LINES):
    """Place reference left, current right, and connect pairs with green lines."""
    title_height, gap = 48, 16
    current_rgb = normal_map_to_rgb8(current.N)
    reference_rgb = (np.zeros_like(current_rgb) if reference is None
                     else normal_map_to_rgb8(reference.N))
    rh, rw = reference_rgb.shape[:2]
    ch, cw = current_rgb.shape[:2]
    canvas = np.zeros((max(rh, ch) + title_height, rw + gap + cw, 3), np.uint8)
    canvas[title_height:title_height + rh, :rw] = reference_rgb
    offset = rw + gap
    canvas[title_height:title_height + ch, offset:offset + cw] = current_rgb
    reference_xy = np.asarray(reference_xy).reshape(-1, 2)
    current_xy = np.asarray(current_xy).reshape(-1, 2)
    if len(reference_xy) != len(current_xy):
        raise ValueError("Reference and current correspondence counts differ")
    indices = np.arange(len(reference_xy))
    if max_lines > 0 and len(indices) > max_lines:
        indices = np.linspace(0, len(indices) - 1, max_lines, dtype=int)
    a, b = reference_xy[indices], current_xy[indices]
    valid = (np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
             & (a[:, 0] >= 0) & (a[:, 0] < rw) & (a[:, 1] >= 0) & (a[:, 1] < rh)
             & (b[:, 0] >= 0) & (b[:, 0] < cw) & (b[:, 1] >= 0) & (b[:, 1] < ch))
    p = np.rint(a[valid]).astype(np.int32)
    q = np.rint(b[valid]).astype(np.int32)
    p = np.minimum(p, (rw - 1, rh - 1)) + (0, title_height)
    q = np.minimum(q, (cw - 1, ch - 1)) + (offset, title_height)
    segments = np.stack((p, q), axis=1).astype(np.int32)
    if len(segments):
        cv2.polylines(canvas, segments, False, (0, 255, 0), 1, cv2.LINE_8)
    for p, q in segments:
        cv2.circle(canvas, p, 2, (0, 255, 0), -1)
        cv2.circle(canvas, q, 2, (0, 255, 0), -1)
    drawn = len(segments)
    for label, origin in ((reference_label, (6, 17)),
                          (current_label, (offset + 6, 17)),
                          (f"{status} | pairs {drawn}/{len(reference_xy)}",
                           (6, 37))):
        cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas
