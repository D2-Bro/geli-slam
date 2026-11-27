"""Chunk-based loop closure: register fused keyframe chunks instead of single keyframes.

A single keyframe of a texture-poor surface gives few repeatable SIFT keypoints, so the keyframe
loop search (SIFT contrast 0.02) finds no candidate on such objects.  Fusing a short run of
keyframes averages the normal noise and widens the patch: low-contrast SIFT then matches revisits
that the keyframe search misses, and NormalFlow verifies them with the loop SCR/CCS thresholds.

Chunks (one per ``chunk_loop_stride`` keyframes):
  * centre c: a keyframe with at least ``chunk_loop_center_min_contact_px`` contact pixels;
  * members: keyframes c - w .. c + w (w = ``chunk_loop_half_width``) of the same contact segment
    (a segment ends at a new trial or at a keyframe whose link is not tactile) with at least
    ``chunk_loop_member_min_contact_px`` contact pixels;
  * the members are fused into the centre keyframe frame with their published (frontend) poses:
    height and normal are weighted means (weight rises with the distance to the contact edge),
    the Laplacian is recomputed from the fused normals.
A chunk is complete when keyframe c + w is in the database.  Each complete chunk (target) is
matched against every older chunk that shares no keyframe with it: SIFT
(``chunk_loop_sift_contrast``) cross-check matches, affine RANSAC with at least
``chunk_loop_min_inliers`` inliers, then NormalFlow from the affine initial guess.  Per reference
contact segment the candidate with the most inliers is kept.  The result is a loop between the
two centre keyframes with the same meaning as a keyframe loop (target_T_reference).

Offline replay of the 12 real-objects trials (main ea134264 dump, 2026-11-27): keyframe RMSE
3.23 -> 2.77 mm, 10 of 12 trials better, no trial worse by more than 0.5 mm (trial_006
7.53 -> 4.97 mm, trial_005 7.66 -> 6.62 mm) with SCR 0.3 / CCS 0.7 and no further gate.
"""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from normalflow.registration import LoseTrackError, normalflow
from normalflow.utils import gxy2laplacian

from gelslam.utils import M2T, get_sift_features


def fuse_keyframes(keyframes, poses, center_pose, ppmm, edge_px=8.0, min_weight=0.25):
    """Fuse keyframe surfaces into the frame of ``center_pose``.

    :param keyframes: sequence of objects with H (px), C (bool), N (H x W x 3).
    :param poses: GelSight poses of the keyframes (4 x 4), same frame as center_pose.
    :return: (H, C, N, L) float32 / bool arrays of the keyframe image size.
    """
    rows, cols = keyframes[0].C.shape
    s = ppmm / 1000.0
    acc_w = np.zeros(rows * cols)
    acc_h = np.zeros(rows * cols)
    acc_n = np.zeros((rows * cols, 3))
    center_inv = np.linalg.inv(np.asarray(center_pose, dtype=np.float64))
    for keyframe, pose in zip(keyframes, poses):
        C = np.asarray(keyframe.C, dtype=bool)
        ys, xs = np.nonzero(C)
        if not len(ys):
            continue
        edge = cv2.distanceTransform(C.astype(np.uint8), cv2.DIST_L2, 3)[ys, xs]
        weight = np.clip(edge / edge_px, 0.05, 1.0)
        T = center_inv @ np.asarray(pose, dtype=np.float64)
        points = np.column_stack(((xs - cols / 2 + .5) * s, (ys - rows / 2 + .5) * s,
                                  np.asarray(keyframe.H)[ys, xs] * s)) @ T[:3, :3].T + T[:3, 3]
        normals = np.asarray(keyframe.N)[ys, xs] @ T[:3, :3].T
        u = points[:, 0] / s + cols / 2 - .5
        v = points[:, 1] / s + rows / 2 - .5
        u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
        fu, fv = u - u0, v - v0
        for du, dv, w in ((0, 0, (1 - fu) * (1 - fv)), (1, 0, fu * (1 - fv)),
                          (0, 1, (1 - fu) * fv), (1, 1, fu * fv)):
            uu, vv = u0 + du, v0 + dv
            ok = (uu >= 0) & (uu < cols) & (vv >= 0) & (vv < rows)
            index = vv[ok] * cols + uu[ok]
            ww = (w * weight)[ok]
            np.add.at(acc_w, index, ww)
            np.add.at(acc_h, index, ww * points[ok, 2] / s)
            np.add.at(acc_n, index, ww[:, None] * normals[ok])
    C = (acc_w > min_weight).reshape(rows, cols)
    H = np.where(C.reshape(-1), acc_h / np.maximum(acc_w, 1e-12), 0.0).reshape(rows, cols)
    N = acc_n / np.maximum(np.linalg.norm(acc_n, axis=1, keepdims=True), 1e-12)
    N[~C.reshape(-1)] = (0.0, 0.0, 1.0)
    N = N.reshape(rows, cols, 3)
    G = -N[..., :2] / np.maximum(N[..., 2:3], 1e-3)
    if C.any() and not C.all():
        # Nearest valid gradient outside the mask: no Laplacian spike at the contact edge.
        iy, ix = distance_transform_edt(~C, return_distances=False, return_indices=True)
        G = G[iy, ix]
    L = gxy2laplacian(G.astype(np.float32), C)
    return H.astype(np.float32), C, N.astype(np.float32), L.astype(np.float32)


class ChunkLoopDetector:
    """Incremental chunk loop search over a KeyFrameDB (owned by LoopSearch)."""

    def __init__(self, pose_graph_config):
        get = pose_graph_config.get
        self.stride = int(get('chunk_loop_stride', 3))
        self.half_width = int(get('chunk_loop_half_width', 4))
        self.contrast = float(get('chunk_loop_sift_contrast', 0.005))
        self.min_inliers = int(get('chunk_loop_min_inliers', 12))
        self.center_min_contact = int(get('chunk_loop_center_min_contact_px', 2000))
        self.member_min_contact = int(get('chunk_loop_member_min_contact_px', 500))
        self.scr_threshold = float(get('chunk_loop_scr_threshold', get('scr_threshold', 0.3)))
        self.ccs_threshold = float(get('chunk_loop_ccs_threshold', get('ccs_threshold', 0.7)))
        if self.stride < 1 or self.half_width < 0 or self.min_inliers < 3:
            raise ValueError('chunk loop stride >= 1, half width >= 0 and min inliers >= 3 '
                             'are required')
        self.sift = cv2.SIFT_create(contrastThreshold=self.contrast, nOctaveLayers=3)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        self.segments = []
        self.units = []
        self.next_center = 0

    def _segment(self, db, index):
        """Contact-segment id: new at a new trial or at a keyframe without a tactile link."""
        while len(self.segments) <= index:
            key = len(self.segments)
            if key == 0:
                self.segments.append(0)
                continue
            keyframe = db[key]
            new = keyframe.is_new_trial() or not getattr(keyframe, 'tactile_valid', True)
            self.segments.append(self.segments[-1] + (1 if new else 0))
        return self.segments[index]

    def _unit(self, db, center, size):
        if int(np.count_nonzero(db[center].C)) < self.center_min_contact:
            return None
        segment = self._segment(db, center)
        first, last = max(0, center - self.half_width), min(size, center + self.half_width + 1)
        members = [j for j in range(first, last)
                   if self._segment(db, j) == segment
                   and int(np.count_nonzero(db[j].C)) > self.member_min_contact]
        if center not in members:
            return None
        image = fuse_keyframes([db[j] for j in members], [db[j].trial_T_curr for j in members],
                               db[center].trial_T_curr, db[center].ppmm)
        keypoints, descriptors = get_sift_features(image[3], image[1], self.sift)
        points = np.array([k.pt for k in keypoints], np.float32).reshape(-1, 2)
        return dict(center=center, members=frozenset(members), segment=segment, image=image,
                    points=points, descriptors=descriptors, ppmm=db[center].ppmm)

    def _register(self, reference, target):
        """target_T_reference and the RANSAC inlier count, or None."""
        if reference['descriptors'] is None or len(reference['points']) < self.min_inliers:
            return None
        matches = self.matcher.match(reference['descriptors'], target['descriptors'])
        if len(matches) < self.min_inliers:
            return None
        affine, inliers = cv2.estimateAffinePartial2D(
            reference['points'][[m.queryIdx for m in matches]].reshape(-1, 1, 2),
            target['points'][[m.trainIdx for m in matches]].reshape(-1, 1, 2),
            maxIters=200, ransacReprojThreshold=3)
        if affine is None or inliers is None or int(inliers.sum()) < self.min_inliers:
            return None
        Ha, Ca, Na, La = reference['image']
        Hb, Cb, Nb, Lb = target['image']
        rows, cols = Cb.shape
        initial = M2T(affine, cols, rows, target['ppmm'])
        try:
            transform = normalflow(Na, Ca, Ha, La, Nb, Cb, Hb, Lb, initial, target['ppmm'],
                                   scr_threshold=self.scr_threshold,
                                   ccs_threshold=self.ccs_threshold)
        except (LoseTrackError, np.linalg.LinAlgError, ValueError):
            return None
        return np.asarray(transform, dtype=np.float64), int(inliers.sum())

    def update(self, db, size):
        """(reference_index, target_index, target_T_reference) of newly complete chunks."""
        result = []
        while self.next_center + self.half_width < size:
            center = self.next_center
            self.next_center += 1
            if center % self.stride:
                continue
            target = self._unit(db, center, size)
            if target is None:
                continue
            if target['descriptors'] is not None and len(target['points']) >= self.min_inliers:
                best = {}
                for reference in self.units:
                    if reference['members'] & target['members']:
                        continue
                    found = self._register(reference, target)
                    if found is None:
                        continue
                    key = reference['segment']
                    if key not in best or found[1] > best[key][2]:
                        best[key] = (reference['center'], found[0], found[1])
                result += [(reference, center, transform)
                           for reference, transform, _ in best.values()]
            self.units.append(target)
        return result
