import copy
import logging
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch

from .odometry.base import FramePose
from .tracking.base import Track, TrackStatus


class KeyframeStatus(Enum):
    """Status of a keyframe within the mapping system."""

    ACTIVE = 0  # Currently used for tracking and mapping
    INACTIVE = 1  # Not actively used for tracking but still in map
    MARGINALIZED = 2  # Marginalized in the backend optimization
    BAD = 3  # Identified as a bad keyframe (e.g., during loop correction)


class CovisibilityGraph:
    """Graph representing covisibility between keyframes."""

    def __init__(self):
        """Initialize covisibility graph."""
        self.keyframes = set()  # Set of keyframe IDs
        self.connections = (
            {}
        )  # Dictionary mapping keyframe ID to set of connected keyframe IDs
        self.weights = (
            {}
        )  # Dictionary mapping (kf1_id, kf2_id) to weight (number of shared observations)

    def add_keyframe(self, keyframe_id: int):
        """
        Add a keyframe to the covisibility graph.

        Args:
            keyframe_id: ID of the keyframe to add
        """
        if keyframe_id not in self.keyframes:
            self.keyframes.add(keyframe_id)
            self.connections[keyframe_id] = set()

    def add_connection(self, kf1_id: int, kf2_id: int, weight: int = 1):
        """
        Add or update a connection between two keyframes.

        Args:
            kf1_id: ID of the first keyframe
            kf2_id: ID of the second keyframe
            weight: Connection weight (number of shared observations)
        """
        # Ensure keyframes exist in the graph
        self.add_keyframe(kf1_id)
        self.add_keyframe(kf2_id)

        # Add connection both ways
        self.connections[kf1_id].add(kf2_id)
        self.connections[kf2_id].add(kf1_id)

        # Store weight (both directions have the same weight)
        self.weights[(kf1_id, kf2_id)] = weight
        self.weights[(kf2_id, kf1_id)] = weight

    def update_connection(self, kf1_id: int, kf2_id: int, weight: int):
        """
        Update the weight of a connection.

        Args:
            kf1_id: ID of the first keyframe
            kf2_id: ID of the second keyframe
            weight: New connection weight
        """
        self.add_connection(kf1_id, kf2_id, weight)

    def remove_connection(self, kf1_id: int, kf2_id: int):
        """
        Remove a connection between two keyframes.

        Args:
            kf1_id: ID of the first keyframe
            kf2_id: ID of the second keyframe
        """
        if kf1_id in self.connections and kf2_id in self.connections[kf1_id]:
            self.connections[kf1_id].remove(kf2_id)

        if kf2_id in self.connections and kf1_id in self.connections[kf2_id]:
            self.connections[kf2_id].remove(kf1_id)

        # Remove weights
        if (kf1_id, kf2_id) in self.weights:
            del self.weights[(kf1_id, kf2_id)]

        if (kf2_id, kf1_id) in self.weights:
            del self.weights[(kf2_id, kf1_id)]

    def remove_keyframe(self, keyframe_id: int):
        """
        Remove a keyframe from the covisibility graph.

        Args:
            keyframe_id: ID of the keyframe to remove
        """
        if keyframe_id in self.keyframes:
            # Remove all connections to this keyframe
            connections_to_remove = list(self.connections[keyframe_id])
            for connected_kf in connections_to_remove:
                self.remove_connection(keyframe_id, connected_kf)

            # Remove keyframe from the graph
            self.keyframes.remove(keyframe_id)
            del self.connections[keyframe_id]

    def get_connected_keyframes(
        self, keyframe_id: int, min_weight: int = 0
    ) -> List[int]:
        """
        Get keyframes connected to the given keyframe.

        Args:
            keyframe_id: ID of the keyframe
            min_weight: Minimum connection weight to consider

        Returns:
            List of connected keyframe IDs
        """
        if keyframe_id not in self.connections:
            return []

        if min_weight <= 0:
            return list(self.connections[keyframe_id])

        # Filter by weight
        connected_kfs = []
        for connected_kf in self.connections[keyframe_id]:
            weight = self.weights.get((keyframe_id, connected_kf), 0)
            if weight >= min_weight:
                connected_kfs.append(connected_kf)

        return connected_kfs

    def get_connection_weight(self, kf1_id: int, kf2_id: int) -> int:
        """
        Get the weight of a connection between two keyframes.

        Args:
            kf1_id: ID of the first keyframe
            kf2_id: ID of the second keyframe

        Returns:
            Connection weight or 0 if not connected
        """
        return self.weights.get((kf1_id, kf2_id), 0)

    def get_best_covisible_keyframes(self, keyframe_id: int, n: int = 10) -> List[int]:
        """
        Get the n best covisible keyframes for a given keyframe.

        Args:
            keyframe_id: ID of the keyframe
            n: Number of keyframes to return

        Returns:
            List of keyframe IDs sorted by covisibility weight
        """
        if keyframe_id not in self.connections:
            return []

        # Get all connected keyframes with weights
        connected_kfs = [
            (kf_id, self.weights.get((keyframe_id, kf_id), 0))
            for kf_id in self.connections[keyframe_id]
        ]

        # Sort by weight (descending)
        connected_kfs.sort(key=lambda x: x[1], reverse=True)

        # Return top n
        return [kf_id for kf_id, _ in connected_kfs[:n]]

    def get_mst_spanning_tree(self) -> Dict[int, int]:
        """
        Get a minimum spanning tree of the covisibility graph.
        Uses a simple implementation of Prim's algorithm.

        Returns:
            Dictionary mapping child keyframe ID to parent keyframe ID
        """
        if not self.keyframes:
            return {}

        # Initialize
        included = set()
        excluded = set(self.keyframes)
        parent = {}

        # Start with the first keyframe
        start_kf = next(iter(self.keyframes))
        included.add(start_kf)
        excluded.remove(start_kf)

        # Grow the minimum spanning tree
        while excluded:
            best_weight = -1
            best_in_kf = None
            best_out_kf = None

            # Find the best edge connecting the included tree to an excluded vertex
            for in_kf in included:
                for out_kf in self.connections[in_kf]:
                    if out_kf in excluded:
                        weight = self.weights.get((in_kf, out_kf), 0)
                        if weight > best_weight:
                            best_weight = weight
                            best_in_kf = in_kf
                            best_out_kf = out_kf

            if best_in_kf is None:
                # Graph is disconnected, add remaining vertices with no parent
                break

            # Add the best edge to the tree
            parent[best_out_kf] = best_in_kf
            included.add(best_out_kf)
            excluded.remove(best_out_kf)

        return parent


class Keyframe:
    """Keyframe class for SLAM system."""

    def __init__(
        self,
        keyframe_id: int,
        pose: FramePose,
        frame: torch.Tensor,
        keypoints: List,
        descriptors: Optional[torch.Tensor] = None,
        tracks: Optional[Dict[int, Track]] = None,
        depth: Optional[torch.Tensor] = None,
        timestamp: Optional[float] = None,
    ):
        """
        Initialize a keyframe.

        Args:
            keyframe_id: Unique identifier for this keyframe
            pose: Pose of the keyframe
            frame: Image frame tensor
            keypoints: List of keypoints
            descriptors: Feature descriptors
            tracks: Dictionary of feature tracks
            depth: Depth or disparity information
            timestamp: Timestamp of the frame
        """
        self.id = keyframe_id
        self.pose = pose
        self.frame = frame  # Store the image
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.tracks = tracks if tracks is not None else {}
        self.depth = depth
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.status = KeyframeStatus.ACTIVE

        # Landmark associations
        self.landmark_observations = {}  # Map from track_id to landmark_id
        self.points_3d = {}  # Map from track_id to 3D point

        # Additional metadata
        self.creation_time = time.time()
        self.is_loop_keyframe = False
        self.loop_connections = set()  # Set of keyframe IDs connected via loop closures

        # For place recognition
        self.bow_vector = None  # Bag of Words vector for loop detection
        self.global_descriptor = None  # Global image descriptor

    def set_pose(self, pose: FramePose):
        """
        Update the pose of the keyframe.

        Args:
            pose: New pose
        """
        self.pose = pose

    def get_num_matched_landmarks(self) -> int:
        """
        Get the number of matches to 3D landmarks.

        Returns:
            Number of matched landmarks
        """
        return len(self.landmark_observations)

    def get_track_ids(self) -> List[int]:
        """
        Get IDs of tracks in this keyframe.

        Returns:
            List of track IDs
        """
        return list(self.tracks.keys())

    def get_matched_track_ids(self) -> List[int]:
        """
        Get IDs of tracks that have 3D point associations.

        Returns:
            List of track IDs with 3D points
        """
        return list(self.points_3d.keys())

    def add_landmark_observation(
        self, track_id: int, landmark_id: int, point_3d: torch.Tensor
    ):
        """
        Add an observation of a 3D landmark.

        Args:
            track_id: Track ID
            landmark_id: Landmark ID
            point_3d: 3D point coordinates
        """
        self.landmark_observations[track_id] = landmark_id
        self.points_3d[track_id] = point_3d

    def remove_landmark_observation(self, track_id: int):
        """
        Remove a landmark observation.

        Args:
            track_id: Track ID to remove
        """
        if track_id in self.landmark_observations:
            del self.landmark_observations[track_id]

        if track_id in self.points_3d:
            del self.points_3d[track_id]

    def get_landmark_id(self, track_id: int) -> Optional[int]:
        """
        Get the landmark ID associated with a track.

        Args:
            track_id: Track ID

        Returns:
            Landmark ID or None if not associated
        """
        return self.landmark_observations.get(track_id)

    def get_3d_point(self, track_id: int) -> Optional[torch.Tensor]:
        """
        Get the 3D point associated with a track.

        Args:
            track_id: Track ID

        Returns:
            3D point or None if not associated
        """
        return self.points_3d.get(track_id)

    def set_as_loop_keyframe(self, is_loop: bool = True):
        """
        Mark this keyframe as part of a loop closure.

        Args:
            is_loop: Whether this is a loop keyframe
        """
        self.is_loop_keyframe = is_loop

    def add_loop_connection(self, keyframe_id: int):
        """
        Add a loop connection to another keyframe.

        Args:
            keyframe_id: ID of the connected keyframe
        """
        self.loop_connections.add(keyframe_id)

    def get_loop_connections(self) -> Set[int]:
        """
        Get IDs of keyframes connected via loop closures.

        Returns:
            Set of keyframe IDs
        """
        return self.loop_connections

    def set_bow_vector(self, bow_vector: torch.Tensor):
        """
        Set the Bag of Words vector for this keyframe.

        Args:
            bow_vector: BoW vector
        """
        self.bow_vector = bow_vector

    def set_global_descriptor(self, descriptor: torch.Tensor):
        """
        Set the global image descriptor for this keyframe.

        Args:
            descriptor: Global descriptor
        """
        self.global_descriptor = descriptor

    def compute_visible_landmarks(
        self, camera_intrinsics: torch.Tensor, image_size: Tuple[int, int]
    ) -> List[int]:
        """
        Compute which landmarks are visible from this keyframe.

        Args:
            camera_intrinsics: Camera intrinsic matrix (3x3)
            image_size: Image size as (width, height)

        Returns:
            List of visible landmark IDs
        """
        width, height = image_size
        visible_landmarks = []

        for track_id, point_3d in self.points_3d.items():
            if track_id not in self.landmark_observations:
                continue

            # Transform point to camera frame
            point_cam = (
                torch.matmul(self.pose.rotation, point_3d) + self.pose.translation
            )

            # Check if point is in front of camera
            if point_cam[2] <= 0:
                continue

            # Project to image
            point_img = torch.matmul(camera_intrinsics, point_cam)
            x = point_img[0] / point_img[2]
            y = point_img[1] / point_img[2]

            # Check if point is within image bounds
            if 0 <= x < width and 0 <= y < height:
                visible_landmarks.append(self.landmark_observations[track_id])

        return visible_landmarks

    def get_projected_points(
        self, camera_intrinsics: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """
        Project 3D points to this keyframe's image plane.

        Args:
            camera_intrinsics: Camera intrinsic matrix (3x3)

        Returns:
            Dictionary mapping track_id to projected 2D point
        """
        projected_points = {}

        for track_id, point_3d in self.points_3d.items():
            # Transform point to camera frame
            point_cam = (
                torch.matmul(self.pose.rotation, point_3d) + self.pose.translation
            )

            # Check if point is in front of camera
            if point_cam[2] <= 0:
                continue

            # Project to image
            point_img = torch.matmul(camera_intrinsics, point_cam)
            x = point_img[0] / point_img[2]
            y = point_img[1] / point_img[2]

            projected_points[track_id] = torch.tensor([x, y], device=point_3d.device)

        return projected_points

    def compute_reprojection_errors(
        self, camera_intrinsics: torch.Tensor
    ) -> Dict[int, float]:
        """
        Compute reprojection errors for 3D points.

        Args:
            camera_intrinsics: Camera intrinsic matrix (3x3)

        Returns:
            Dictionary mapping track_id to reprojection error
        """
        errors = {}

        # Get projected points
        projected_points = self.get_projected_points(camera_intrinsics)

        # Compute errors
        for track_id, projected_point in projected_points.items():
            if track_id in self.tracks:
                track = self.tracks[track_id]
                observed_point = torch.tensor(
                    [track.get_latest_position()], device=projected_point.device
                )

                # Compute Euclidean distance
                error = torch.norm(projected_point - observed_point).item()
                errors[track_id] = error

        return errors


class KeyframeManager:
    """Manager for keyframes in the SLAM system."""

    def __init__(self, config: Dict = None):
        """
        Initialize keyframe manager.

        Args:
            config: Configuration dictionary with the following keys:
                - max_keyframes: Maximum number of keyframes to maintain
                - keyframe_interval: Minimum number of frames between keyframes
                - min_translation: Minimum translation for creating a new keyframe
                - min_rotation: Minimum rotation for creating a new keyframe
                - min_tracked_ratio: Minimum ratio of tracked features for a new keyframe
                - redundant_culling_threshold: Threshold for culling redundant keyframes
                - keyframe_validity_threshold: Threshold for valid keyframe creation
        """
        self.config = config if config is not None else {}

        # Extract configuration
        self.max_keyframes = self.config.get("max_keyframes", 100)
        self.keyframe_interval = self.config.get("keyframe_interval", 5)
        self.min_translation = self.config.get("min_translation", 0.1)
        self.min_rotation = self.config.get("min_rotation", 0.1)
        self.min_tracked_ratio = self.config.get("min_tracked_ratio", 0.5)
        self.redundant_culling_threshold = self.config.get(
            "redundant_culling_threshold", 0.9
        )
        self.keyframe_validity_threshold = self.config.get(
            "keyframe_validity_threshold", 20
        )

        # Keyframe storage
        self.keyframes = {}  # Map from keyframe ID to Keyframe object
        self.next_keyframe_id = 0

        # Frame counter
        self.frame_counter = 0
        self.last_keyframe_idx = -1

        # Covisibility graph
        self.covisibility_graph = CovisibilityGraph()

        # Reference keyframe for tracking
        self.reference_keyframe_id = None

        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_keyframe(
        self,
        pose: FramePose,
        frame: torch.Tensor,
        keypoints: List,
        descriptors: Optional[torch.Tensor] = None,
        tracks: Optional[Dict[int, Track]] = None,
        depth: Optional[torch.Tensor] = None,
        timestamp: Optional[float] = None,
    ) -> Optional[Keyframe]:
        """
        Create a new keyframe.

        Args:
            pose: Pose of the keyframe
            frame: Image frame tensor
            keypoints: List of keypoints
            descriptors: Feature descriptors
            tracks: Dictionary of feature tracks
            depth: Depth or disparity information
            timestamp: Timestamp of the frame

        Returns:
            New keyframe object or None if creation failed
        """
        # Check if we should create a keyframe
        if not self._should_create_keyframe(pose, tracks):
            return None

        # Create new keyframe
        keyframe_id = self.next_keyframe_id
        keyframe = Keyframe(
            keyframe_id, pose, frame, keypoints, descriptors, tracks, depth, timestamp
        )

        # Add to storage
        self.keyframes[keyframe_id] = keyframe
        self.covisibility_graph.add_keyframe(keyframe_id)

        # Update covisibility graph with connections to other keyframes
        self._update_covisibility_graph(keyframe)

        # Set as reference if we don't have one
        if self.reference_keyframe_id is None:
            self.reference_keyframe_id = keyframe_id

        # Update state
        self.next_keyframe_id += 1
        self.last_keyframe_idx = self.frame_counter

        # Perform culling if we exceed the maximum number of keyframes
        if len(self.keyframes) > self.max_keyframes:
            self._cull_keyframes()

        return keyframe

    def _should_create_keyframe(
        self, pose: FramePose, tracks: Optional[Dict[int, Track]] = None
    ) -> bool:
        """
        Determine if a new keyframe should be created.

        Args:
            pose: Current pose
            tracks: Current feature tracks

        Returns:
            True if a new keyframe should be created, False otherwise
        """
        self.frame_counter += 1

        # Always create a keyframe if we don't have any
        if len(self.keyframes) == 0:
            return True

        # Check if enough frames have passed since the last keyframe
        if self.frame_counter - self.last_keyframe_idx < self.keyframe_interval:
            return False

        # Get reference keyframe
        if self.reference_keyframe_id not in self.keyframes:
            self.reference_keyframe_id = next(iter(self.keyframes))

        reference_keyframe = self.keyframes[self.reference_keyframe_id]

        # Check if we've moved enough from the reference keyframe
        relative_pose = reference_keyframe.pose.inverse().compose(pose)

        # Check translation
        translation_magnitude = torch.norm(relative_pose.translation).item()
        if translation_magnitude < self.min_translation:
            rotation_magnitude = torch.norm(
                relative_pose.rotation
                - torch.eye(3, device=relative_pose.rotation.device)
            ).item()

            if rotation_magnitude < self.min_rotation:
                return False

        # Check feature tracking ratio if tracks are provided
        if tracks is not None:
            # Count tracked features
            tracked_count = sum(
                1 for track in tracks.values() if track.status == TrackStatus.TRACKED
            )

            # Check ratio against reference keyframe's tracks
            if self.reference_keyframe_id in self.keyframes:
                ref_tracks = self.keyframes[self.reference_keyframe_id].tracks
                if ref_tracks:
                    # Find tracks that are common between reference and current
                    common_tracks = set(ref_tracks.keys()).intersection(tracks.keys())

                    # Calculate tracking ratio
                    if len(ref_tracks) > 0:
                        tracking_ratio = len(common_tracks) / len(ref_tracks)

                        # If we're still tracking most features, don't create a new keyframe
                        if tracking_ratio > self.min_tracked_ratio:
                            return False

        # Check if we have enough keypoints
        if keypoints is not None and len(keypoints) < self.keyframe_validity_threshold:
            return False

        return True

    def _update_covisibility_graph(self, keyframe: Keyframe):
        """
        Update the covisibility graph with a new keyframe.

        Args:
            keyframe: New keyframe
        """
        # Find common observations with other keyframes
        for other_id, other_keyframe in self.keyframes.items():
            if other_id == keyframe.id:
                continue

            # Find common tracks
            common_tracks = set(keyframe.tracks.keys()).intersection(
                other_keyframe.tracks.keys()
            )

            if common_tracks:
                # Update covisibility graph
                self.covisibility_graph.add_connection(
                    keyframe.id, other_id, len(common_tracks)
                )

    def _cull_keyframes(self):
        """Remove redundant keyframes."""
        # Find redundant keyframes
        redundant_ids = []

        for kf_id, keyframe in self.keyframes.items():
            # Skip the reference keyframe
            if kf_id == self.reference_keyframe_id:
                continue

            # Skip recently created keyframes
            if self.frame_counter - kf_id < 2 * self.keyframe_interval:
                continue

            # Skip loop keyframes
            if keyframe.is_loop_keyframe:
                continue

            # Check if this keyframe is redundant
            connected_kfs = self.covisibility_graph.get_connected_keyframes(
                kf_id, min_weight=10
            )

            if not connected_kfs:
                continue

            # Check how many observations in this keyframe are also observed in connected keyframes
            observed_in_connected = set()
            for connected_kf_id in connected_kfs:
                if connected_kf_id in self.keyframes:
                    connected_kf = self.keyframes[connected_kf_id]
                    observed_in_connected.update(
                        connected_kf.landmark_observations.keys()
                    )

            # Get observations in this keyframe
            observations = set(keyframe.landmark_observations.keys())

            # Compute redundancy ratio
            if observations:
                redundancy_ratio = len(
                    observations.intersection(observed_in_connected)
                ) / len(observations)

                if redundancy_ratio > self.redundant_culling_threshold:
                    redundant_ids.append(kf_id)

        # Remove redundant keyframes
        for kf_id in redundant_ids:
            self.remove_keyframe(kf_id)

    def remove_keyframe(self, keyframe_id: int):
        """
        Remove a keyframe.

        Args:
            keyframe_id: ID of the keyframe to remove
        """
        if keyframe_id not in self.keyframes:
            return

        # Check if it's the reference keyframe
        if keyframe_id == self.reference_keyframe_id:
            # Find a new reference keyframe
            connected_kfs = self.covisibility_graph.get_connected_keyframes(keyframe_id)
            if connected_kfs:
                self.reference_keyframe_id = connected_kfs[0]
            else:
                # Just pick any other keyframe
                remaining_kfs = [
                    kf_id for kf_id in self.keyframes.keys() if kf_id != keyframe_id
                ]
                if remaining_kfs:
                    self.reference_keyframe_id = remaining_kfs[0]
                else:
                    self.reference_keyframe_id = None

        # Remove from storage
        del self.keyframes[keyframe_id]

        # Remove from covisibility graph
        self.covisibility_graph.remove_keyframe(keyframe_id)

    def get_keyframe(self, keyframe_id: int) -> Optional[Keyframe]:
        """
        Get a keyframe by ID.

        Args:
            keyframe_id: Keyframe ID

        Returns:
            Keyframe object or None if not found
        """
        return self.keyframes.get(keyframe_id)

    def get_all_keyframes(self) -> Dict[int, Keyframe]:
        """
        Get all keyframes.

        Returns:
            Dictionary of keyframe ID to Keyframe object
        """
        return self.keyframes

    def get_reference_keyframe(self) -> Optional[Keyframe]:
        """
        Get the reference keyframe.

        Returns:
            Reference keyframe or None if not set
        """
        if self.reference_keyframe_id is not None:
            return self.keyframes.get(self.reference_keyframe_id)
        return None

    def set_reference_keyframe(self, keyframe_id: int) -> bool:
        """
        Set the reference keyframe.

        Args:
            keyframe_id: ID of the keyframe to set as reference

        Returns:
            True if successful, False otherwise
        """
        if keyframe_id in self.keyframes:
            self.reference_keyframe_id = keyframe_id
            return True
        return False

    def get_best_covisible_keyframes(
        self, keyframe_id: int, n: int = 10
    ) -> List[Keyframe]:
        """
        Get the n best covisible keyframes for a given keyframe.

        Args:
            keyframe_id: ID of the keyframe
            n: Number of keyframes to return

        Returns:
            List of Keyframe objects
        """
        kf_ids = self.covisibility_graph.get_best_covisible_keyframes(keyframe_id, n)
        return [self.keyframes[kf_id] for kf_id in kf_ids if kf_id in self.keyframes]

    def get_keyframes_in_view(
        self,
        pose: FramePose,
        camera_intrinsics: torch.Tensor,
        image_size: Tuple[int, int],
        min_overlap: int = 10,
    ) -> List[Keyframe]:
        """
        Get keyframes that share view with the given pose.

        Args:
            pose: Camera pose
            camera_intrinsics: Camera intrinsic matrix
            image_size: Image size as (width, height)
            min_overlap: Minimum number of common landmarks

        Returns:
            List of keyframes in view
        """
        in_view_keyframes = []

        for keyframe in self.keyframes.values():
            # Compute relative pose
            relative_pose = keyframe.pose.inverse().compose(pose)

            # Check if poses are close
            translation_magnitude = torch.norm(relative_pose.translation).item()
            rotation_magnitude = torch.norm(
                relative_pose.rotation
                - torch.eye(3, device=relative_pose.rotation.device)
            ).item()

            # Skip if too far away
            if translation_magnitude > 10.0 or rotation_magnitude > 1.0:
                continue

            # Check for landmark overlap
            visible_landmarks = keyframe.compute_visible_landmarks(
                camera_intrinsics, image_size
            )

            if len(visible_landmarks) >= min_overlap:
                in_view_keyframes.append(keyframe)

        return in_view_keyframes
