import json
import os

import numpy as np
import pytest
import yaml

from gelslam.core.imu_mount_calibration import (
    CALIBRATION_KEY, SOURCE_KEY, describe_imu_mount, load_config, require_resolved,
    resolve_imu_mount_calibration)
from gelslam.core.loop_pose_graph import LoopPoseGraph

LIVE_CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "live_thick_far_config.yaml")


def _transform(rotation_deg, translation):
    angle = np.radians(rotation_deg)
    matrix = np.eye(4)
    matrix[:3, :3] = [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0],
                      [0.0, 0.0, 1.0]]
    matrix[:3, 3] = translation
    return matrix


NEAR = _transform(90.0, [0.003, 0.0005, -0.033])
TOP = _transform(-90.0, [-0.003, -0.001, -0.158])


def _write_candidates(tmp_path, convention="T_A_B maps coordinates in B into A; metres"):
    path = tmp_path / "CALIBRATION_CANDIDATES.json"
    path.write_text(json.dumps({
        "convention": convention,
        "imu_slots": {
            "near": {"status": "near_status", "T_G_Inear": NEAR.tolist()},
            "top": {"status": "top_status", "T_G_Itop": TOP.tolist()},
        },
    }))
    return path


def _config(path, slot="near"):
    return {"imu_config": {CALIBRATION_KEY: {
        "path": str(path), "slot": slot, "slots": {"near": "near", "far": "top"}}}}


def test_default_slot_and_override_select_the_file_entry(tmp_path):
    config = _config(_write_candidates(tmp_path))
    near = resolve_imu_mount_calibration(config)["imu_config"]
    far = resolve_imu_mount_calibration(config, "far")["imu_config"]
    np.testing.assert_array_equal(near["gelsight_T_imu"], NEAR)
    np.testing.assert_array_equal(far["gelsight_T_imu"], TOP)
    assert CALIBRATION_KEY not in far
    assert far[SOURCE_KEY]["slot"] == "far"
    assert far[SOURCE_KEY]["entry"] == "imu_slots.top.T_G_Itop"
    assert far[SOURCE_KEY]["status"] == "top_status"
    # The input configuration is not changed.
    assert CALIBRATION_KEY in config["imu_config"]
    require_resolved(far)
    assert "-158.00] mm" in describe_imu_mount(far)


def test_load_config_resolves_relative_path_and_parameter_slot(tmp_path):
    _write_candidates(tmp_path)
    config_path = tmp_path / "config.yaml"
    config = _config("CALIBRATION_CANDIDATES.json")
    config_path.write_text(yaml.safe_dump(config))
    resolved = load_config(str(config_path), "far")
    np.testing.assert_array_equal(resolved["imu_config"]["gelsight_T_imu"], TOP)
    np.testing.assert_array_equal(load_config(str(config_path), "")["imu_config"]["gelsight_T_imu"], NEAR)


@pytest.mark.parametrize("change, message", [
    (lambda c: c["imu_config"].update(gelsight_T_imu=np.eye(4).tolist()), "both"),
    (lambda c: c["imu_config"][CALIBRATION_KEY].update(slot="middle"), "not one of"),
    (lambda c: c["imu_config"][CALIBRATION_KEY].update(extra=1), "unknown"),
    (lambda c: c["imu_config"][CALIBRATION_KEY]["slots"].update(far="side"), None),
])
def test_invalid_configuration_is_rejected(tmp_path, change, message):
    config = _config(_write_candidates(tmp_path), slot="far" if message is None else "near")
    change(config)
    with pytest.raises(ValueError, match=message or "imu_slots.side.T_G_Iside is missing"):
        resolve_imu_mount_calibration(config)


def test_convention_and_rotation_are_checked(tmp_path):
    with pytest.raises(ValueError, match="convention"):
        resolve_imu_mount_calibration(_config(_write_candidates(tmp_path, "T_A_B maps A into B")))
    path = _write_candidates(tmp_path)
    data = json.loads(path.read_text())
    data["imu_slots"]["near"]["T_G_Inear"][0][0] = 2.0
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="orthonormal"):
        resolve_imu_mount_calibration(_config(path))


def test_slot_without_calibration_block_is_rejected():
    config = {"imu_config": {"gelsight_T_imu": np.eye(4).tolist()}}
    assert resolve_imu_mount_calibration(config) is config
    with pytest.raises(ValueError, match="imu_slot"):
        resolve_imu_mount_calibration(config, "near")


def test_unresolved_block_is_not_read_as_identity(tmp_path):
    config = _config(_write_candidates(tmp_path))
    with pytest.raises(ValueError, match="not resolved"):
        LoopPoseGraph({}, config["imu_config"])


def test_live_config_uses_calibration_file_for_both_slots():
    with open(LIVE_CONFIG, encoding="utf-8") as handle:
        block = yaml.safe_load(handle)["imu_config"][CALIBRATION_KEY]
    if not os.path.isfile(block["path"]):
        pytest.skip("calibration file not available: %s" % block["path"])
    with open(block["path"], encoding="utf-8") as handle:
        slots = json.load(handle)["imu_slots"]
    for slot, key in (("near", "near"), ("far", "top")):
        resolved = load_config(LIVE_CONFIG, slot)["imu_config"]
        np.testing.assert_array_equal(resolved["gelsight_T_imu"], slots[key]["T_G_I" + key])
