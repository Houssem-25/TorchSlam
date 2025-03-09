import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


class TrackStatus(Enum):
    """Status of a feature track."""

    TRACKED = 0  # Successfully tracked in current frame
    LOST = 1  # Failed to track in current frame
    NEW = 2  # New feature, first detection
    INACTIVE = 3  # Track exists but not active in current frame


class Track:
    """Represents a tracked feature across multiple frames."""

    def __init__(
        self, feature_id: int, keypoint, descriptor: Optional[torch.Tensor] = None
    ):
        """
        Initialize a track.

        Args:
            feature_id: Unique identifier for this track
            keypoint: Initial keypoint of the track
            descriptor: Feature descriptor (optional)
        """
        self.feature_id = feature_id
        self.keypoints = [keypoint]  # List of keypoints across frames
        self.descriptor = descriptor  # Latest descriptor
        self.status = TrackStatus.NEW
        self.last_seen_frame = 0
        self.age = 1  # Number of frames this track has been observed
        self.positions = [(keypoint.x, keypoint.y)]  # Positions across frames
        self.is_keypoint = True  # Whether this track corresponds to a keypoint
        self.depth = None  # Depth or distance information if available

    def update(
        self, keypoint, descriptor: Optional[torch.Tensor] = None, frame_idx: int = None
    ):
        """
        Update track with a new observation.

        Args:
            keypoint: New keypoint observation
            descriptor: Updated feature descriptor (optional)
            frame_idx: Current frame index
        """
        self.keypoints.append(keypoint)
        if descriptor is not None:
            self.descriptor = descriptor
        self.status = TrackStatus.TRACKED
        if frame_idx is not None:
            self.last_seen_frame = frame_idx
        self.age += 1
        self.positions.append((keypoint.x, keypoint.y))

    def mark_lost(self):
        """Mark track as lost."""
        self.status = TrackStatus.LOST

    def mark_inactive(self):
        """Mark track as inactive."""
        self.status = TrackStatus.INACTIVE

    def get_latest_position(self) -> Tuple[float, float]:
        """Get the latest position of the track."""
        return self.positions[-1]

    def get_motion_vector(self) -> Optional[Tuple[float, float]]:
        """
        Get motion vector between the last two positions.

        Returns:
            Tuple of (dx, dy) or None if track has only one position
        """
        if len(self.positions) < 2:
            return None

        last_pos = self.positions[-1]
        prev_pos = self.positions[-2]
        return (last_pos[0] - prev_pos[0], last_pos[1] - prev_pos[1])

    def has_moved(self, threshold: float = 1.0) -> bool:
        """
        Check if the track has moved.

        Args:
            threshold: Minimum distance to consider as movement

        Returns:
            True if track has moved, False otherwise
        """
        if len(self.positions) < 2:
            return False

        motion = self.get_motion_vector()
        if motion is None:
            return False

        dx, dy = motion
        return (dx**2 + dy**2) > threshold**2

    def set_depth(self, depth: float):
        """
        Set the depth/distance for this track.

        Args:
            depth: Depth or distance value
        """
        self.depth = depth

    def get_keypoint(self, idx: int = -1):
        """
        Get keypoint at a specific index.

        Args:
            idx: Index of keypoint to retrieve (-1 for latest)

        Returns:
            Keypoint at the specified index
        """
        return self.keypoints[idx]


class BaseTracker(ABC):
    """Base class for feature tracking."""

    def __init__(self, config: Dict = None):
        """
        Initialize tracker.

        Args:
            config: Configuration dictionary
        """
        self.config = config if config is not None else {}
        self.tracks = {}  # Dictionary mapping feature_id to Track
        self.next_feature_id = 0
        self.current_frame_idx = 0

        # Extract common configuration parameters
        self.max_tracks = self.config.get("max_tracks", 1000)
        self.min_distance = self.config.get("min_distance", 10)
        self.max_age = self.config.get("max_age", 100)
        self.max_frames_lost = self.config.get("max_frames_lost", 3)

        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def track_features(
        self,
        current_frame: torch.Tensor,
        prev_frame: Optional[torch.Tensor] = None,
        current_keypoints: Optional[List] = None,
    ) -> Dict[int, Track]:
        """
        Track features between frames.

        Args:
            current_frame: Current frame tensor
            prev_frame: Previous frame tensor (optional)
            current_keypoints: List of keypoints detected in current frame (optional)

        Returns:
            Dictionary of track_id to Track object
        """
        pass

    def initialize_tracks(self, keypoints, descriptors: Optional[torch.Tensor] = None):
        """
        Initialize tracks from keypoints.

        Args:
            keypoints: List of keypoints
            descriptors: Tensor of descriptors (optional)
        """
        for i, kp in enumerate(keypoints):
            track_id = self.next_feature_id
            desc = descriptors[i].unsqueeze(0) if descriptors is not None else None
            self.tracks[track_id] = Track(track_id, kp, desc)
            self.next_feature_id += 1

    def update_track(
        self, track_id: int, keypoint, descriptor: Optional[torch.Tensor] = None
    ):
        """
        Update an existing track.

        Args:
            track_id: ID of track to update
            keypoint: New keypoint observation
            descriptor: Updated descriptor (optional)
        """
        if track_id in self.tracks:
            self.tracks[track_id].update(keypoint, descriptor, self.current_frame_idx)

    def remove_track(self, track_id: int):
        """
        Remove a track.

        Args:
            track_id: ID of track to remove
        """
        if track_id in self.tracks:
            del self.tracks[track_id]

    def prune_tracks(self):
        """Prune old or lost tracks."""
        track_ids_to_remove = []

        for track_id, track in self.tracks.items():
            # Remove tracks that haven't been seen for too long
            if (self.current_frame_idx - track.last_seen_frame) > self.max_frames_lost:
                track_ids_to_remove.append(track_id)

            # Remove tracks that are too old
            elif track.age > self.max_age:
                track_ids_to_remove.append(track_id)

        for track_id in track_ids_to_remove:
            self.remove_track(track_id)

    def get_active_tracks(self) -> Dict[int, Track]:
        """
        Get currently active tracks.

        Returns:
            Dictionary of active tracks (track_id -> Track)
        """
        return {
            tid: track
            for tid, track in self.tracks.items()
            if track.status == TrackStatus.TRACKED or track.status == TrackStatus.NEW
        }

    def get_active_keypoints(self) -> List:
        """
        Get keypoints of active tracks.

        Returns:
            List of keypoints
        """
        active_tracks = self.get_active_tracks()
        return [track.get_keypoint() for track in active_tracks.values()]

    def get_track_motion(self) -> Dict[int, Tuple[float, float]]:
        """
        Get motion vectors for all active tracks.

        Returns:
            Dictionary mapping track_id to motion vector (dx, dy)
        """
        motions = {}
        for track_id, track in self.get_active_tracks().items():
            motion = track.get_motion_vector()
            if motion is not None:
                motions[track_id] = motion
        return motions

    def reset(self):
        """Reset tracker state."""
        self.tracks = {}
        self.current_frame_idx = 0
        self.next_feature_id = 0
