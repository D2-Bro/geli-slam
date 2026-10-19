"""NormalFlow residual exports for the active GTSAM frontend."""

from gelslam.core.numpy_vi import DenseNormalResidual, normalflow_quality

DenseNormalFlowFactor = DenseNormalResidual

__all__ = ("DenseNormalResidual", "DenseNormalFlowFactor", "normalflow_quality")
