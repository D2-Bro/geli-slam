"""Geometry tests for the RViz point-cloud/TF bridge."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Vector3Stamped
from scipy.spatial.transform import Rotation

from gelslam.nodes.rviz_visualizer_node import (
    GRAVITY_ALIGNED_FRAME,
    RVIZ_FRAME,
    RvizVisualizerNode,
    rotation_matrix_to_quaternion,
    vertices_to_pointcloud,
)
import gelslam.visualization.visible_coverage_meshes as coverage_module
from gelslam.nodes.tracker_node import TrackerNode, normal_map_to_rgb8
from gelslam.nodes.pose_graph_node import PoseGraphNode
from gelslam.visualization.visible_coverage_meshes import (
    VisibleCoverageMeshes,
    compute_frontend_tracking_meshes,
    compute_visible_coverage_meshes,
)


class _TrackingKeyframe:
    """Minimal keyframe fixture for frontend-only mesh placement."""

    def __init__(self, trial_t_curr, new_trial=False):
        self.trial_T_curr = np.asarray(trial_t_curr, dtype=np.float32)
        self.new_trial_flag = new_trial
        self.pointcloud = np.array(
            [[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.0, 0.001, 0.0]],
            dtype=np.float32,
        )
        self.contact = np.array([[True, True], [True, False]])

    def is_new_trial(self):
        return self.new_trial_flag

    def get_pointcloud(self, level):
        assert level == 0
        return self.pointcloud

    def get_C(self, level):
        assert level == 0
        return self.contact


class _TrackingKeyframeDB:
    def __init__(self, keyframes):
        self.keyframes = keyframes

    def size(self):
        return len(self.keyframes)

    def __getitem__(self, index):
        return self.keyframes[index]


def test_rotation_matrix_to_quaternion_for_identity_and_half_turn_x():
    """TF quaternions follow ROS's xyzw ordering."""
    assert np.allclose(rotation_matrix_to_quaternion(np.eye(3)), [0.0, 0.0, 0.0, 1.0])
    half_turn_x = np.diag([1.0, -1.0, -1.0])
    quaternion = rotation_matrix_to_quaternion(half_turn_x)
    assert np.allclose(np.abs(quaternion), [1.0, 0.0, 0.0, 0.0])


def test_vertices_to_pointcloud_serializes_xyz32():
    """Surface rendering uses one XYZ32 point for every mesh vertex."""
    vertices = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 0.5]])
    cloud = vertices_to_pointcloud(vertices, Time(sec=7, nanosec=8))
    assert cloud.header.frame_id == "gelslam_map"
    assert cloud.header.stamp.sec == 7
    assert cloud.width == 2
    assert cloud.point_step == 12
    assert np.allclose(np.frombuffer(cloud.data, dtype=np.float32).reshape(-1, 3), vertices)


def test_current_frame_waits_for_its_keyframe_across_topics():
    """The red surface updates when FrameMsg arrives before KeyFrameMsg."""
    keyframes = []
    keyframedb = SimpleNamespace(
        insert=keyframes.append,
        size=lambda: len(keyframes),
        find_kidx_from_kid=lambda kid, _size: (
            0 if keyframes and keyframes[0].kid == kid else None
        ),
    )
    clouds = []
    transforms = []
    node = SimpleNamespace(
        keyframedb=keyframedb,
        visible_coverage_meshes=SimpleNamespace(
            get_keyframe_pose=lambda *_args: (False, np.eye(4))
        ),
        use_backend_poses=True,
        pending_frame_msg=None,
        latest_map_T_gelsight=np.eye(4),
        latest_contact_local_cloud=None,
        bridge=object(),
        ppmm=0.0634,
        current_cloud_pub=SimpleNamespace(publish=clouds.append),
        tf_broadcaster=SimpleNamespace(sendTransform=transforms.append),
        matrix_to_transform=RvizVisualizerNode.matrix_to_transform,
    )
    node.frame_callback = lambda message: RvizVisualizerNode.frame_callback(node, message)
    node._publish_current_contact_cloud = lambda message: (
        RvizVisualizerNode._publish_current_contact_cloud(node, message)
    )
    frame = SimpleNamespace(
        ref_kid=3,
        ref_t_curr=np.eye(4).flatten().tolist(),
        header=SimpleNamespace(stamp=Time(sec=7, nanosec=8)),
    )

    node.frame_callback(frame)
    assert node.pending_frame_msg is frame
    assert clouds == []

    RvizVisualizerNode.keyframe_callback(node, SimpleNamespace(kid=3))

    assert node.pending_frame_msg is None
    assert clouds == []
    assert len(transforms) == 1
    assert transforms[0].header.stamp == frame.header.stamp


def test_current_contact_cloud_updates_and_clears_without_frame_pose():
    """RViz uses each current mask even when the tracker holds the last pose."""
    published = []
    pose = np.eye(4)
    pose[0, 3] = 0.1
    node = SimpleNamespace(
        latest_map_T_gelsight=pose,
        latest_contact_local_cloud=None,
        current_cloud_pub=SimpleNamespace(publish=published.append),
        get_logger=lambda: SimpleNamespace(warning=lambda _: None),
    )
    node._publish_current_contact_cloud = lambda message: (
        RvizVisualizerNode._publish_current_contact_cloud(node, message)
    )
    local = vertices_to_pointcloud([[0.001, 0.0, 0.0]], Time(sec=2))
    RvizVisualizerNode.current_contact_callback(node, local)
    assert len(published) == 1
    point = np.frombuffer(published[-1].data, dtype=np.float32).reshape(-1, 3)
    assert np.allclose(point, [[0.101, 0.0, 0.0]])

    empty = vertices_to_pointcloud([], Time(sec=3))
    RvizVisualizerNode.current_contact_callback(node, empty)
    assert len(published) == 2
    assert published[-1].width == 0
    assert published[-1].header.stamp.sec == 3


def test_normal_map_rgb_encoding_preserves_axes_and_marks_invalid_pixels():
    normals = np.array(
        [[[0.0, 0.0, 1.0], [1.0, -1.0, 0.0], [np.nan, 0.0, 1.0]]],
        dtype=np.float32,
    )
    rgb = normal_map_to_rgb8(normals)
    assert rgb.dtype == np.uint8
    assert rgb.tolist() == [[[128, 128, 255], [255, 0, 128], [0, 0, 0]]]


def test_tracker_publishes_normal_map_when_contact_tracking_is_rejected():
    normal_images = []
    contact_clouds = []
    header = SimpleNamespace(stamp=Time(sec=3), frame_id="gel_frame")
    observation = SimpleNamespace(N=np.array([[[0.0, 0.0, 1.0]]]))
    node = SimpleNamespace(
        recorded_source_clock=None,
        bridge=SimpleNamespace(
            imgmsg_to_cv2=lambda *_args, **_kwargs: np.zeros((1, 1, 3)),
            cv2_to_imgmsg=lambda rgb, encoding: SimpleNamespace(
                data=rgb, encoding=encoding, header=None
            ),
        ),
        tracker=SimpleNamespace(
            track=lambda *_args, **_kwargs: (False, None),
            latest_observation_frame=observation,
        ),
        _publish_frontend_gravity=lambda _header: None,
        _publish_normal_correspondences=lambda _frame, _header: None,
        _contact_surface_cloud=lambda _frame, _stamp: object(),
        normal_map_pub=SimpleNamespace(publish=normal_images.append),
        contact_surface_pub=SimpleNamespace(publish=contact_clouds.append),
    )

    TrackerNode.image_callback(node, SimpleNamespace(header=header))

    assert len(normal_images) == len(contact_clouds) == 1
    assert normal_images[0].encoding == "rgb8"
    assert normal_images[0].header is header
    assert normal_images[0].data.tolist() == [[[128, 128, 255]]]


def test_tracker_contact_cloud_uses_current_mask_and_empty_mask():
    node = SimpleNamespace(tracker=SimpleNamespace(ppmm=0.1))
    frame = SimpleNamespace(
        H=np.array([[0.0, 2.0], [0.0, 0.0]], dtype=np.float32),
        C=np.array([[False, True], [False, False]]),
    )
    cloud = TrackerNode._contact_surface_cloud(node, frame, Time(sec=2))
    points = np.frombuffer(cloud.data, dtype=np.float32).reshape(-1, 3)
    assert cloud.width == 1
    assert np.allclose(points, [[0.00005, -0.00005, 0.0002]])

    frame.C[:] = False
    empty = TrackerNode._contact_surface_cloud(node, frame, Time(sec=3))
    assert empty.width == 0
    assert len(empty.data) == 0


@pytest.mark.parametrize("gravity", [(4., -2., -8.), (0., 0., -9.80665)])
def test_rviz_static_tf_aligns_first_trial_gravity_and_keeps_imu_extrinsic(gravity):
    """Late RViz subscribers receive the map alignment and IMU TF together."""
    published = []
    node = SimpleNamespace(
        gravity_alignment=None,
        gelsight_T_imu=np.eye(4),
        static_tf_broadcaster=SimpleNamespace(sendTransform=published.append),
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=7, nanosec=8))
        ),
        get_logger=lambda: SimpleNamespace(info=lambda _: None, warning=lambda _: None),
        matrix_to_transform=RvizVisualizerNode.matrix_to_transform,
    )
    node.publish_static_imu_transform = lambda: (
        RvizVisualizerNode.publish_static_imu_transform(node)
    )
    node.set_gravity_alignment = lambda gravity: (
        RvizVisualizerNode.set_gravity_alignment(node, gravity)
    )
    message = Vector3Stamped()
    message.header.frame_id = "gelslam_first_trial"
    message.vector.x, message.vector.y, message.vector.z = gravity

    RvizVisualizerNode.frontend_gravity_callback(node, message)

    assert len(published) == 1
    assert {(tf.header.frame_id, tf.child_frame_id) for tf in published[0]} == {
        (GRAVITY_ALIGNED_FRAME, RVIZ_FRAME),
        ("gelsight_estimate", "imu_estimate"),
    }
    alignment = next(tf for tf in published[0] if tf.child_frame_id == RVIZ_FRAME)
    quaternion = alignment.transform.rotation
    rotation = Rotation.from_quat(
        [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    )
    assert np.allclose(
        rotation.apply(gravity),
        [0.0, 0.0, -np.linalg.norm(gravity)],
        atol=1e-12,
    )
    RvizVisualizerNode.frontend_gravity_callback(node, message)
    assert len(published) == 1


def test_frontend_mesh_and_current_tf_keep_the_first_aligned_rotation():
    first = np.eye(4)
    first[:3, :3] = Rotation.from_rotvec([2.7, .4, -.1]).as_matrix()
    second = first.copy()
    second[:3, 3] = [.01, -.02, .03]
    db = _TrackingKeyframeDB([
        _TrackingKeyframe(first, new_trial=True), _TrackingKeyframe(second),
    ])
    meshes = compute_frontend_tracking_meshes(db)
    for index, pose in enumerate((first, second)):
        _, displayed_pose = meshes.get_keyframe_pose(index, db)
        assert np.allclose(displayed_pose, pose, atol=1e-7)
        expected = db[index].pointcloud @ pose[:3, :3].T + pose[:3, 3]
        assert np.allclose(np.asarray(meshes.visible_coverage_meshes[index].vertices), expected)


def test_frontend_tracking_meshes_use_keyframe_tracking_poses_without_pgo():
    """Tracker-only reconstruction uses the published trial poses."""
    second_pose = np.eye(4, dtype=np.float32)
    second_pose[0, 3] = 0.01
    third_pose = np.eye(4, dtype=np.float32)
    third_pose[0, 3] = 0.025
    keyframedb = _TrackingKeyframeDB(
        [
            _TrackingKeyframe(np.eye(4), new_trial=True),
            _TrackingKeyframe(second_pose),
            _TrackingKeyframe(third_pose),
        ]
    )

    meshes = compute_frontend_tracking_meshes(keyframedb)

    assert len(meshes.visible_coverage_meshes) == 3
    second_vertices = np.asarray(meshes.visible_coverage_meshes[1].vertices)
    assert np.allclose(
        second_vertices,
        keyframedb[1].pointcloud + np.array([0.01, 0.0, 0.0]),
    )
    new_trial, reference_pose = meshes.get_keyframe_pose(1, keyframedb)
    assert not new_trial
    assert np.allclose(reference_pose, second_pose)
    third_vertices = np.asarray(meshes.visible_coverage_meshes[2].vertices)
    assert np.allclose(third_vertices, keyframedb[2].pointcloud + [0.025, 0.0, 0.0])
    new_trial, reference_pose = meshes.get_keyframe_pose(2, keyframedb)
    assert not new_trial
    assert np.allclose(reference_pose, third_pose)

    corrected_lc_pose = np.eye(4)
    corrected_lc_pose[0, 3] = 1.0
    corrected = VisibleCoverageMeshes([object()], 1, corrected_lc_pose)
    new_trial, corrected_target = corrected.get_keyframe_pose(2, keyframedb)
    assert not new_trial
    assert np.isclose(corrected_target[0, 3], 1.015)


def test_reference_pose_uses_tracker_trial_pose_before_coverage_exists():
    """An identity keyframe-relative pose must retain its trial motion."""
    first = _TrackingKeyframe(np.eye(4), new_trial=True)
    moved_pose = np.eye(4)
    moved_pose[0, 3] = 0.012
    second = _TrackingKeyframe(moved_pose)
    keyframedb = _TrackingKeyframeDB([first, second])

    coverage = VisibleCoverageMeshes()
    is_new_trial, reference_pose = coverage.get_keyframe_pose(1, keyframedb)

    assert not is_new_trial
    assert np.allclose(reference_pose, moved_pose)
    assert coverage.get_keyframe_pose(0, keyframedb)[0]


def test_rviz_current_pose_moves_with_keyframe_before_coverage():
    moved_pose = np.eye(4)
    moved_pose[0, 3] = 0.012
    keyframe = _TrackingKeyframe(moved_pose)

    class KeyframeDB(_TrackingKeyframeDB):
        def find_kidx_from_kid(self, kid, _size):
            return 0 if kid == 4 else None

    transforms = []
    node = SimpleNamespace(
        keyframedb=KeyframeDB([keyframe]),
        visible_coverage_meshes=VisibleCoverageMeshes(),
        pending_frame_msg=None,
        latest_map_T_gelsight=np.eye(4),
        latest_contact_local_cloud=None,
        bridge=object(),
        ppmm=0.0634,
        tf_broadcaster=SimpleNamespace(sendTransform=transforms.append),
        matrix_to_transform=RvizVisualizerNode.matrix_to_transform,
    )
    frame = SimpleNamespace(
        ref_kid=4,
        ref_t_curr=np.eye(4).flatten().tolist(),
        header=SimpleNamespace(stamp=Time(sec=7)),
    )

    RvizVisualizerNode.frame_callback(node, frame)

    assert np.isclose(node.latest_map_T_gelsight[0, 3], 0.012)
    assert np.isclose(transforms[0].transform.translation.x, 0.012)
    assert len(transforms) == 1


def test_pointcloud_only_surfaces_preserve_geometry_and_backend_pose(monkeypatch):
    """Both RViz map paths avoid triangulation and retain pose corrections."""
    second_pose = np.eye(4)
    second_pose[0, 3] = .01
    db = _TrackingKeyframeDB([
        _TrackingKeyframe(np.eye(4), new_trial=True), _TrackingKeyframe(second_pose),
    ])
    expected = compute_frontend_tracking_meshes(db)

    def forbid_triangles(*args):
        pytest.fail("RViz point clouds must not construct triangle meshes")

    monkeypatch.setattr(coverage_module, "pointcloud2mesh", forbid_triangles)
    raw = compute_frontend_tracking_meshes(db, create_meshes=False)
    assert raw.visible_coverage_meshes == []
    for cloud, mesh in zip(raw.visible_coverage_pointclouds, expected.visible_coverage_meshes):
        np.testing.assert_allclose(cloud, np.asarray(mesh.vertices))
    np.testing.assert_allclose(raw.get_keyframe_pose(1, db)[1], second_pose)

    corrected = np.eye(4)
    corrected[0, 3] = 1.0
    monkeypatch.setattr(coverage_module, "compute_adjusted_pointcloud",
                        lambda *args: db[0].pointcloud + [1.0, 0.0, 0.0])
    message = SimpleNamespace(visible_coverage_nodes=[SimpleNamespace(
        kid=10, ref_t_curr=corrected.flatten().tolist(), neighbor_kids=[]
    )])
    surface = compute_visible_coverage_meshes(message, db, {10: 0}, create_meshes=False)
    assert surface.visible_coverage_meshes == []
    np.testing.assert_allclose(surface.visible_coverage_pointclouds[0],
                               db[0].pointcloud + [1.0, 0.0, 0.0])
    assert surface.get_keyframe_pose(1, db)[1][0, 3] == pytest.approx(1.01)


def test_global_surface_publishes_xyz_and_clears_previous_geometry():
    published = []
    surface = VisibleCoverageMeshes(pointclouds=[np.array([[1., 2., 3.]])])
    node = SimpleNamespace(
        visible_coverage_meshes=surface,
        global_cloud_pub=SimpleNamespace(publish=published.append),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=2))),
    )
    RvizVisualizerNode._publish_global_surface(node)
    np.testing.assert_allclose(np.frombuffer(published[-1].data, dtype=np.float32), [1, 2, 3])
    surface.visible_coverage_pointclouds = []
    RvizVisualizerNode._publish_global_surface(node)
    assert published[-1].width == 0


def test_backend_keyframe_trajectory_uses_only_active_solved_states():
    """The RViz path must connect backend keyframes, not every frame."""
    published = []
    poses = [np.eye(4) for _ in range(3)]
    poses[0][:3, 3] = (0.0, 0.0, 0.0)
    poses[1][:3, 3] = (0.1, 0.2, 0.3)
    poses[2][:3, 3] = (9.0, 9.0, 9.0)

    class _KeyframeDB:
        def __init__(self):
            self.keyframes = [
                SimpleNamespace(trial_group=0),
                SimpleNamespace(trial_group=0),
                SimpleNamespace(trial_group=1),
            ]

        def size(self):
            return len(self.keyframes)

        def __getitem__(self, index):
            return self.keyframes[index]

    class _Solutions:
        def size(self):
            return len(poses)

        def __getitem__(self, index):
            return poses[index]

    class _ParentGroups:
        parent_group_sizes = [2, 1]

        @staticmethod
        def get_largest_parent_group():
            return 0

        @staticmethod
        def get_parent_group(kidx, keyframedb):
            return keyframedb[kidx].trial_group

    node = SimpleNamespace(
        add_keyframe_lock=threading.Lock(),
        keyframedb=_KeyframeDB(),
        pose_graph_solutions=_Solutions(),
        parent_groups_info=_ParentGroups(),
        gelsight_trajectory_pub=SimpleNamespace(publish=published.append),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(
            to_msg=lambda: Time(sec=7, nanosec=8)
        )),
    )

    PoseGraphNode.publish_gelsight_keyframe_trajectory(node)

    assert len(published) == 1
    path = published[0]
    assert path.header.frame_id == "gelslam_map"
    assert len(path.poses) == 2
    assert np.allclose(
        [pose.pose.position.x for pose in path.poses], (0.0, 0.1)
    )
