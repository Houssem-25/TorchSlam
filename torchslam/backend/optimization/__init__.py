"""
Optimization module for PyTorch SLAM backend.

This module contains optimization solvers and algorithms for various SLAM problems,
including bundle adjustment, pose graph optimization, and differentiable optimization.
"""

from .base import OptimizationProblem, OptimizationResult, Optimizer

# from .factor_graph import FactorGraph, Variable, Factor # File was deleted, comment out import
# from .pose_graph import PoseGraphOptimization # File was deleted, comment out import
from .bundle_adjustment import BundleAdjustment

# from .differentiable_solver import DifferentiableSolver # File was deleted, comment out import

__all__ = [
    "Optimizer",
    "OptimizationProblem",
    "OptimizationResult",
    # 'FactorGraph', # Comment out from __all__ as well
    # 'Variable',
    # 'Factor',
    # 'PoseGraphOptimization', # Comment out from __all__ as well
    "BundleAdjustment",
    # 'DifferentiableSolver', # Comment out from __all__ as well
]
