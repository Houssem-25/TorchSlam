"""
Frontend module for PyTorch SLAM library.

This module contains classes and functions for the frontend components of the SLAM system,
including feature extraction, tracking, odometry estimation, and loop detection.
"""

from .keyframe import CovisibilityGraph, Keyframe, KeyframeManager, KeyframeStatus
from .loop_detection import LoopCandidate, LoopDetector

__all__ = [
    "Keyframe",
    "KeyframeManager",
    "CovisibilityGraph",
    "KeyframeStatus",
    "LoopDetector",
    "LoopCandidate",
]
