import numpy as np

from gelslam.core.keyframe import compute_adjusted_pointcloud
from gelslam.core.pose_graph import PoseGraphSolutions
from gelslam.utils import pointcloud2mesh


def compute_frontend_tracking_meshes(keyframedb, *, create_meshes=True):
    """Build the current trial mesh from tracker odometry only.

    This path deliberately does not use loop closures, coverage selection, or
    pose-graph solutions. Each keyframe point cloud uses the trial pose that
    the tracker published. A new trial
    has no tracked transform to an earlier trial, so only the newest trial is
    displayed in its own local map frame.
    Set create_meshes=False to keep only XYZ point arrays for RViz.
    """
    if keyframedb.size() == 0:
        return VisibleCoverageMeshes()

    trial_start_kidx = 0
    for kidx in range(keyframedb.size() - 1, -1, -1):
        if keyframedb[kidx].is_new_trial():
            trial_start_kidx = kidx
            break
    kidxs = list(range(trial_start_kidx, keyframedb.size()))

    n_points = sum(keyframedb[kidx].pointcloud.shape[0] for kidx in kidxs)
    if n_points > 1000000:
        viz_level = 3
    elif n_points > 350000:
        viz_level = 2
    elif n_points > 150000:
        viz_level = 1
    else:
        viz_level = 0

    meshes = []
    pointclouds = []
    for kidx in kidxs:
        keyframe = keyframedb[kidx]
        start_T_curr = keyframe.trial_T_curr
        pointcloud = keyframe.get_pointcloud(viz_level)
        pointcloud = (
            start_T_curr[:3, :3] @ pointcloud.T
        ).T + start_T_curr[:3, 3]
        if create_meshes:
            meshes.append(pointcloud2mesh(pointcloud, keyframe.get_C(viz_level)))
        else:
            pointclouds.append(pointcloud)

    return VisibleCoverageMeshes(
        meshes,
        trial_start_kidx,
        np.asarray(keyframedb[trial_start_kidx].trial_T_curr, dtype=np.float32).copy(),
        viz_level,
        pointclouds=pointclouds,
    )


def compute_visible_coverage_meshes(visible_coverage_graph_msg, keyframedb, kid2kidx,
                                  *, create_meshes=True):
    """
    Compute the visible coverage meshes.
    Determines visualization level based on point count.

    :param visible_coverage_graph_msg: VisibleCoverageGraphMsg; The visible coverage graph message.
    :param keyframedb: KeyFrameDB; The keyframe database.
    :param kid2kidx: dict; The mapping from kid to kidx.
    :param create_meshes: bool; False skips triangulation and retains XYZ arrays.
    :return: VisibleCoverageMeshes; The visible coverage meshes.
    """
    kidxs = []
    for visible_coverage_node_msg in visible_coverage_graph_msg.visible_coverage_nodes:
        try:
            kidxs.append(kid2kidx[visible_coverage_node_msg.kid])
        except KeyError:
            continue
    # Compute visualization level
    n_points = 0
    for kidx in kidxs:
        n_points += keyframedb[kidx].pointcloud.shape[0]
    if n_points > 1000000:
        viz_level = 3
    elif n_points > 350000:
        viz_level = 2
    elif n_points > 150000:
        viz_level = 1
    else:
        viz_level = 0
    # construct pose graph solutions until the last loop closed keyframe, only visible coverage keyframes have values
    if len(kidxs) == 0:
        start_T_currs = np.zeros((0, 4, 4), dtype=np.float32)
    else:
        # Visible coverage nodes are not guaranteed to arrive in kidx order
        # (loop-closure updates can reorder them).  Size by the largest index,
        # not the final list element, so every mapped keyframe has a slot.
        start_T_currs = np.zeros((max(kidxs) + 1, 4, 4), dtype=np.float32)
    for visible_coverage_node_msg in visible_coverage_graph_msg.visible_coverage_nodes:
        try:
            kidx = kid2kidx[visible_coverage_node_msg.kid]
            start_T_curr = np.array(visible_coverage_node_msg.ref_t_curr).reshape(4, 4)
            start_T_currs[kidx] = start_T_curr
        except KeyError:
            continue
    pose_graph_solutions = PoseGraphSolutions(start_T_currs)
    # Create the meshes
    meshes = []
    pointclouds = []
    for visible_coverage_node_msg in visible_coverage_graph_msg.visible_coverage_nodes:
        try:
            kidx = kid2kidx[visible_coverage_node_msg.kid]
            neighbor_kidxs = []
            for neighbor_kid in visible_coverage_node_msg.neighbor_kids:
                try:
                    neighbor_kidxs.append(kid2kidx[neighbor_kid])
                except KeyError:
                    continue
            adjusted_pointcloud = compute_adjusted_pointcloud(
                kidx,
                neighbor_kidxs,
                keyframedb,
                pose_graph_solutions,
                viz_level,
            )
            if create_meshes:
                adjusted_mesh = pointcloud2mesh(
                    adjusted_pointcloud, keyframedb[kidx].get_C(viz_level)
                )
                meshes.append(adjusted_mesh)
            else:
                pointclouds.append(adjusted_pointcloud)
        except KeyError:
            pass
    # Create the visible coverage meshes
    if len(kidxs) == 0:
        visible_coverage_meshes = VisibleCoverageMeshes(meshes)
    else:
        visible_coverage_meshes = VisibleCoverageMeshes(
            meshes, kidxs[-1], pose_graph_solutions[kidxs[-1]], viz_level,
            pointclouds=pointclouds,
        )
    return visible_coverage_meshes


class VisibleCoverageMeshes:
    """
    It contains all coverage meshes in the same parent group as the last loop closing keyframe.
    It also saves the kidx and the pose of the loop closing keyframe.
    """

    def __init__(
        self,
        visible_coverage_meshes=[],
        lc_kidx=None,
        start_T_lc=None,
        viz_level=0,
        pointclouds=None,
    ):
        self.visible_coverage_meshes = visible_coverage_meshes
        self.visible_coverage_pointclouds = [] if pointclouds is None else pointclouds
        # The last loop closing keyframe's kidx
        self.lc_kidx = lc_kidx
        self.start_T_lc = start_T_lc
        self.viz_level = viz_level

    def get_keyframe_pose(self, tar_kidx, keyframedb):
        """
        Get the pose of the keyframe.

        :param tar_kidx: int; The keyframe index.
        :param keyframedb: KeyFrameDB; The keyframe database.
        :return: (bool, np.ndarray);
            - bool: If the keyframe is not in the same parent group as the visible coverage meshes.
            - np.ndarray: The pose of the keyframe reference to the visible coverage meshes.
        """
        if self.lc_kidx is None:
            keyframe = keyframedb[tar_kidx]
            return keyframe.is_new_trial(), np.asarray(
                keyframe.trial_T_curr, dtype=np.float32
            ).copy()
        for kidx in range(self.lc_kidx + 1, tar_kidx + 1):
            if keyframedb[kidx].is_new_trial():
                return True, np.asarray(keyframedb[tar_kidx].trial_T_curr, dtype=np.float32).copy()
        trial_T_lc = keyframedb[self.lc_kidx].trial_T_curr
        trial_T_target = keyframedb[tar_kidx].trial_T_curr
        return False, self.start_T_lc @ np.linalg.inv(trial_T_lc) @ trial_T_target
