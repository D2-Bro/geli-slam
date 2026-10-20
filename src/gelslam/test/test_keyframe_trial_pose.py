"""Keyframe message poses and recovery after a dropped keyframe."""

from types import SimpleNamespace

import numpy as np
from sensor_msgs.msg import Image

import gelslam.core.keyframe as keyframe_module
from gelslam.core.frame import create_keyframe_msg
from gelslam.core.keyframe import KeyFrameDB
from gelslam_msgs.msg import KeyFrameMsg


def _message(kid, x, new_trial=False):
    message = KeyFrameMsg()
    message.kid = kid
    message.fid = kid
    message.new_trial_flag = new_trial
    message.header.stamp.sec = kid + 1
    message.height_map = Image()
    message.contact_mask = Image()
    message.normal_map = Image()
    message.laplacian_map = Image()
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = x
    message.trial_t_curr = pose.ravel().tolist()
    return message


def test_tracker_serializes_keyframe_trial_pose():
    bridge = SimpleNamespace(cv2_to_imgmsg=lambda *_args, **_kwargs: Image())
    frame = SimpleNamespace(
        H=np.zeros((2, 2), dtype=np.float32),
        C=np.ones((2, 2), dtype=bool),
        N=np.zeros((2, 2, 3), dtype=np.float32),
        L=np.zeros((2, 2), dtype=np.float32),
    )
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = 0.25

    message = create_keyframe_msg(bridge, frame, 2, 4, False, pose)

    assert np.allclose(np.asarray(message.trial_t_curr).reshape(4, 4), pose)
    assert not hasattr(message, "ref_t_curr")
    assert not hasattr(message, "imu_preintegrated")


def test_keyframe_db_keeps_trial_poses_and_reanchors_after_a_drop(monkeypatch):
    monkeypatch.setattr(
        keyframe_module,
        "KeyFrame",
        lambda _h, _c, _n, _l, kid, fid, group, new, pose, _ppmm, _stamp, **_options:
        SimpleNamespace(
            kid=kid, fid=fid, trial_group=group, new_trial_flag=new,
            trial_T_curr=pose,
        ),
    )
    logger = SimpleNamespace(info=lambda *_args: None, warning=lambda *_args: None)
    db = KeyFrameDB(1.0, logger)
    db.bridge = SimpleNamespace(imgmsg_to_cv2=lambda _image: np.zeros((2, 2)))

    for message in (
        _message(0, 0.0, new_trial=True),
        _message(1, 0.1),
        _message(3, 0.3),
        _message(4, 0.4),
    ):
        db.insert(message)

    assert [keyframe.trial_group for keyframe in db.keyframes] == [0, 0, 1, 1]
    assert np.allclose(
        [keyframe.trial_T_curr[0, 3] for keyframe in db.keyframes],
        [0.0, 0.1, 0.0, 0.1],
    )


def test_keyframe_group_recovery_keeps_gravity_axes(monkeypatch):
    from scipy.spatial.transform import Rotation

    monkeypatch.setattr(keyframe_module, "KeyFrame",
        lambda _h, _c, _n, _l, kid, fid, group, new, pose, _ppmm, _stamp, **_options:
        SimpleNamespace(kid=kid, trial_group=group, trial_T_curr=pose))
    db = KeyFrameDB(1., SimpleNamespace(info=lambda *_: None, warning=lambda *_: None))
    db.bridge = SimpleNamespace(imgmsg_to_cv2=lambda _: np.zeros((2, 2)))
    Q = Rotation.from_rotvec([.5, -.2, .1]).as_matrix()
    for kid, x in ((0, 0.), (1, .1), (3, .3), (4, .4)):
        msg = _message(kid, x, new_trial=kid == 0)
        pose = np.asarray(msg.trial_t_curr).reshape(4, 4)
        pose[:3, :3] = Q
        msg.trial_t_curr = pose.ravel().tolist()
        db.insert(msg)
    assert [k.trial_group for k in db.keyframes] == [0, 0, 1, 1]
    for keyframe in db.keyframes:
        assert np.allclose(keyframe.trial_T_curr[:3, :3], Q)
    assert np.allclose([k.trial_T_curr[0, 3] for k in db.keyframes], [0., .1, 0., .1])
