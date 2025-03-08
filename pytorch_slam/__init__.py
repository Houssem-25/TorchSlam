"""
PyTorch SLAM Library

A PyTorch-based Simultaneous Localization and Mapping (SLAM) library that provides
a complete framework for visual and LiDAR SLAM systems. The library is designed to be
modular, GPU-accelerated, and easily extensible.

Major Components:
- Frontend: Feature extraction, tracking, visual odometry, and loop detection
- Backend: Map optimization, bundle adjustment, and pose graph optimization
- Mapping: Point cloud and mesh-based mapping
- Visualization: Trajectory and map visualization tools
- Dataset: Loaders for common SLAM datasets

This library is implemented entirely in PyTorch, enabling GPU acceleration and
differentiable operations throughout the SLAM pipeline.
"""
# Import frontend components
from pytorch_slam.frontend import (
    CovisibilityGraph,
    Keyframe,
    KeyframeManager,
    KeyframeStatus,
    LoopCandidate,
    LoopDetector,
)

# Import frontend/feature_extraction components
from pytorch_slam.frontend.feature_extraction import (
    BaseFeatureExtractor,
    FeatureMatcher,
    FeatureType,
    KeyPoint,
    ORBFeatureExtractor,
    SIFTFeatureExtractor,
)

# Import frontend/odometry components
from pytorch_slam.frontend.odometry import (
    BaseOdometry,
    FramePose,
    ICPOdometry,
    ICPVariant,
    OdometryStatus,
    PnPOdometry,
)

# Import frontend/tracking components
from pytorch_slam.frontend.tracking import (
    BaseTracker,
    DeepTracker,
    KLTTracker,
    SiameseNetwork,
    Track,
    TrackStatus,
)

# Version information
from pytorch_slam.version import __version__

# Define the public API
__all__ = [
    # Version
    "__version__",
    # Frontend
    "Keyframe",
    "KeyframeManager",
    "CovisibilityGraph",
    "KeyframeStatus",
    "LoopDetector",
    "LoopCandidate",
    # Odometry
    "BaseOdometry",
    "FramePose",
    "OdometryStatus",
    "PnPOdometry",
    "ICPOdometry",
    "ICPVariant",
    # Feature Extraction
    "FeatureType",
    "KeyPoint",
    "BaseFeatureExtractor",
    "SIFTFeatureExtractor",
    "ORBFeatureExtractor",
    "FeatureMatcher",
    # Tracking
    "BaseTracker",
    "Track",
    "TrackStatus",
    "KLTTracker",
    "DeepTracker",
    "SiameseNetwork",
]
