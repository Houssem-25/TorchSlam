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
from torchslam.frontend import (
    CovisibilityGraph,
    Keyframe,
    KeyframeManager,
    KeyframeStatus,
    LoopCandidate,
    LoopDetector,
)

# Import frontend/feature_extraction components
from torchslam.frontend.feature_extraction import (
    BaseFeatureExtractor,
    FeatureMatcher,
    FeatureType,
    KeyPoint,
    ORBFeatureExtractor,
    SIFTFeatureExtractor,
)

# Import frontend/odometry components
from torchslam.frontend.odometry import (
    BaseOdometry,
    FramePose,
    ICPOdometry,
    ICPVariant,
    OdometryStatus,
    PnPOdometry,
)

# Import frontend/tracking components
from torchslam.frontend.tracking import BaseTracker, KLTTracker, Track, TrackStatus

# Version information
from torchslam.version import __version__

# Define the public API
__all__ = [
    # Top-level modules
    "frontend",
    "backend",
    "mapping",
    "state_estimation",
    "slam",
    "utils",
    "config",
    "datasets",
    # Specific classes/functions (example)
    "BaseFeatureExtractor",
    "KeyPoint",
    "ORBFeatureExtractor",
    "FramePose",
    "OdometryStatus",
    "Track",
    "TrackStatus",
    "__version__",
]

# Aliases/Re-exports (example - keep if needed, remove otherwise)
# Import key components to the top level
from torchslam.frontend.feature_extraction import BaseFeatureExtractor
from torchslam.frontend.feature_extraction import (
    BaseFeatureExtractor as FeatureExtractorBase,
)
from torchslam.frontend.feature_extraction import KeyPoint
from torchslam.frontend.feature_extraction import KeyPoint as KP
from torchslam.frontend.feature_extraction import ORBFeatureExtractor
from torchslam.frontend.feature_extraction import ORBFeatureExtractor as ORB
from torchslam.frontend.odometry import FramePose
from torchslam.frontend.odometry import FramePose as Pose
from torchslam.frontend.odometry import OdometryStatus
from torchslam.frontend.odometry import OdometryStatus as OdomStatus
from torchslam.frontend.tracking import Track
from torchslam.frontend.tracking import Track as FeatureTrack
from torchslam.frontend.tracking import TrackStatus
from torchslam.frontend.tracking import TrackStatus as FeatTrackStatus
