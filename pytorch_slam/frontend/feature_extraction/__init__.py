"""
Feature extraction module for PyTorch SLAM library.

This module contains classes and functions for detecting and describing features
in images, which are essential for visual SLAM systems.
"""

from .feature_extraction import (
    BaseFeatureExtractor,
    FeatureMatcher,
    FeatureType,
    KeyPoint,
    ORBFeatureExtractor,
    SIFTFeatureExtractor,
)

__all__ = [
    "FeatureType",
    "KeyPoint",
    "BaseFeatureExtractor",
    "SIFTFeatureExtractor",
    "ORBFeatureExtractor",
    "FeatureMatcher",
]
