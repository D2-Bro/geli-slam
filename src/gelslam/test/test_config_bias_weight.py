"""Shipped configs keep the bias random-walk weight apart from the IMU factor weight."""

from pathlib import Path

import pytest
import yaml

from gelslam.core.gtsam_vi import create_frontend_solver

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def _weight_blocks(node):
    """Every mapping that sets imu_residual_weight."""
    if isinstance(node, dict):
        if "imu_residual_weight" in node:
            yield node
        for value in node.values():
            yield from _weight_blocks(value)
    elif isinstance(node, list):
        for value in node:
            yield from _weight_blocks(value)


CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))
BLOCKS = [(path.name, block) for path in CONFIGS
          for block in _weight_blocks(yaml.safe_load(path.read_text()))]


def test_some_shipped_config_sets_the_imu_weight():
    assert BLOCKS


@pytest.mark.parametrize("name, block", BLOCKS, ids=[name for name, _ in BLOCKS])
def test_shipped_config_sets_an_explicit_bias_weight(name, block):
    weight = block.get("bias_residual_weight")
    assert weight is not None, name + " lets the bias weight follow imu_residual_weight"
    assert float(weight) > 0


def test_explicit_bias_weight_does_not_follow_the_imu_weight():
    for imu_weight in (1e-4, 1e-2, 1e-1):
        solver = create_frontend_solver(dict(
            gtsam_vi_max_iterations=3, gtsam_vi_min_iterations=1, imu_residual_weight=imu_weight,
            bias_residual_weight=1e-4))
        assert solver.imu_weight == imu_weight
        assert solver.bias_weight == 1e-4
