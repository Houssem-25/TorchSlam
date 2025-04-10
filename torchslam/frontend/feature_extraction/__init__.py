"""
Feature extraction module for PyTorch SLAM library.

This module contains classes and functions for detecting and describing features
in images, which are essential for visual SLAM systems.
"""

from .base import BaseFeatureExtractor, FeatureType, KeyPoint
from .feature_matcher import FeatureMatcher, MatchingMethod
from .orb import ORBFeatureExtractor
from .sift import SIFTFeatureExtractor

__all__ = [
    "FeatureType",
    "KeyPoint",
    "BaseFeatureExtractor",
    "SIFTFeatureExtractor",
    "ORBFeatureExtractor",
    "FeatureMatcher",
    "MatchingMethod",
]
