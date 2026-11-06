"""Select T_GI (imu_config.gelsight_T_imu) from an IMU-slot calibration file.

A configuration can give T_GI directly (``imu_config.gelsight_T_imu``) or name a
calibration file and an IMU mounting slot::

    imu_config:
      gelsight_T_imu_calibration:
        path: ".../CALIBRATION_CANDIDATES.json"
        slot: "near"            # default slot; a launch/node parameter can override it
        slots:                  # slot name -> imu_slots key in the file
          near: "near"
          far: "top"

The file stores ``imu_slots.<key>.T_G_I<key>`` with the convention "T_A_B maps
coordinates in B into A", so T_G_I<key> maps IMU coordinates into GelSight
coordinates, the same as ``gelsight_T_imu``.  ``load_config`` replaces the block
with ``gelsight_T_imu`` and an informational ``gelsight_T_imu_source`` record.
"""

import copy
import json
import os

import numpy as np
import yaml

CALIBRATION_KEY = "gelsight_T_imu_calibration"
SOURCE_KEY = "gelsight_T_imu_source"
CONVENTION = "T_A_B maps coordinates in B into A"


def _homogeneous(matrix, name):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("%s must be a finite 4x4 matrix" % name)
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0)):
        raise ValueError("%s must be homogeneous" % name)
    rotation = matrix[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)):
        raise ValueError("%s rotation must be orthonormal with determinant +1" % name)
    return matrix


def resolve_imu_mount_calibration(config, imu_slot=None, config_dir=None):
    """Return a copy of ``config`` with T_GI read from the slot calibration file.

    ``imu_slot`` overrides the block's default ``slot``; None or "" keeps it.  A
    config without the block is returned unchanged; then ``imu_slot`` must be empty.
    """
    imu_config = (config or {}).get("imu_config") or {}
    block = imu_config.get(CALIBRATION_KEY)
    if block is None:
        if imu_slot:
            raise ValueError("imu_slot %r needs imu_config.%s in the configuration"
                             % (imu_slot, CALIBRATION_KEY))
        return config
    if imu_config.get("gelsight_T_imu") is not None:
        raise ValueError("imu_config gives both gelsight_T_imu and %s; keep one"
                         % CALIBRATION_KEY)
    if not isinstance(block, dict):
        raise ValueError("imu_config.%s must be a mapping" % CALIBRATION_KEY)
    unknown = block.keys() - {"path", "slot", "slots"}
    if unknown:
        raise ValueError("unknown imu_config.%s settings: %s"
                         % (CALIBRATION_KEY, ", ".join(sorted(unknown))))
    slots = block.get("slots")
    if not isinstance(slots, dict) or not slots:
        raise ValueError("imu_config.%s.slots must map slot names to file keys"
                         % CALIBRATION_KEY)
    slot = imu_slot or block.get("slot")
    if slot not in slots:
        raise ValueError("IMU slot %r is not one of: %s" % (slot, ", ".join(sorted(slots))))
    key = slots[slot]

    path = os.path.expanduser(str(block.get("path") or ""))
    if not path:
        raise ValueError("imu_config.%s.path is empty" % CALIBRATION_KEY)
    if not os.path.isabs(path) and config_dir is not None:
        path = os.path.join(config_dir, path)
    with open(path, "r", encoding="utf-8") as handle:
        candidates = json.load(handle)
    if not str(candidates.get("convention", "")).startswith(CONVENTION):
        raise ValueError("%s: convention must start with %r" % (path, CONVENTION))
    entry = (candidates.get("imu_slots") or {}).get(key)
    name = "T_G_I" + key
    if entry is None or name not in entry:
        raise ValueError("%s: imu_slots.%s.%s is missing" % (path, key, name))
    matrix = _homogeneous(entry[name], "%s imu_slots.%s.%s" % (path, key, name))

    resolved = copy.deepcopy(config)
    resolved_imu = resolved["imu_config"]
    del resolved_imu[CALIBRATION_KEY]
    resolved_imu["gelsight_T_imu"] = matrix.tolist()
    resolved_imu[SOURCE_KEY] = {
        "path": os.path.abspath(path),
        "slot": slot,
        "entry": "imu_slots.%s.%s" % (key, name),
        "status": entry.get("status"),
    }
    return resolved


def require_resolved(imu_config):
    """Reject a config whose slot calibration was not resolved by ``load_config``."""
    if (imu_config or {}).get(CALIBRATION_KEY) is not None:
        raise ValueError("imu_config.%s is not resolved; load the configuration with "
                         "gelslam.core.imu_mount_calibration.load_config" % CALIBRATION_KEY)


def load_config(config_path, imu_slot=None):
    """Read a GelSLAM YAML configuration and resolve its IMU slot calibration."""
    config_path = os.path.abspath(os.path.expanduser(config_path))
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return resolve_imu_mount_calibration(config, imu_slot, os.path.dirname(config_path))


def describe_imu_mount(imu_config):
    """Return a one-line log text for the active T_GI, or None without one."""
    transform = (imu_config or {}).get("gelsight_T_imu")
    if transform is None:
        return None
    translation_mm = 1e3 * np.asarray(transform, dtype=np.float64)[:3, 3]
    source = (imu_config or {}).get(SOURCE_KEY)
    origin = ("config value" if source is None else "slot %s, %s, %s (status %s)"
              % (source["slot"], source["entry"], source["path"], source["status"]))
    return "T_GI from %s; IMU origin in GelSight frame [%.2f, %.2f, %.2f] mm" % (
        origin, *translation_mm)
