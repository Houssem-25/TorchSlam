"""
PyTorch-based Visual SLAM implementation.

This module implements a monocular Visual SLAM system using RGB images.
It integrates frontend components (feature extraction, tracking) with
backend optimization (bundle adjustment).
"""
import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ..backend.optimization.bundle_adjustment import BundleAdjustment
from ..backend.se3 import SE3
from ..frontend.feature_extraction import (
    BaseFeatureExtractor,
    FeatureType,
    KeyPoint,
    ORBFeatureExtractor,
)
from ..frontend.feature_extraction.feature_matcher import FeatureMatcher, MatchingMethod
from ..frontend.keyframe import Keyframe, KeyframeManager, KeyframeStatus
from ..frontend.loop_detection import LoopDetector
from ..frontend.odometry import FramePose, OdometryStatus, PnPOdometry
from ..frontend.tracking import KLTTracker, Track, TrackStatus


class VisualSLAMStatus(Enum):
    """Status of the Visual SLAM system."""

    INITIALIZING = 0  # System is initializing
    TRACKING_GOOD = 1  # Tracking is good
    TRACKING_BAD = 2  # Tracking is poor but system is still functioning
    LOST = 3  # Tracking is lost, needs recovery
    RELOCALIZATION = 4  # Trying to relocalize


class VisualSLAM:
    """
    Visual SLAM using RGB images.

    This class implements a monocular visual SLAM system that tracks camera
    motion from RGB images and builds a 3D map of the environment.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize the Visual SLAM system.

        Args:
            config: Configuration dictionary with SLAM parameters
        """
        self.config = config if config is not None else {}
        self.device = torch.device(
            self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        # Current status
        self.status = VisualSLAMStatus.INITIALIZING
        self.frame_idx = 0
        self.last_keyframe_idx = -1

        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing Visual SLAM on device: {self.device}")

        # Initialize components
        self._init_frontend()
        self._init_backend()

    def _init_frontend(self):
        """Initialize frontend components."""
        # Camera parameters
        self.camera_intrinsics = self.config.get("camera_intrinsics", None)
        if self.camera_intrinsics is None:
            raise ValueError("Camera intrinsics must be provided in the config")

        if isinstance(self.camera_intrinsics, np.ndarray):
            self.camera_intrinsics = torch.tensor(
                self.camera_intrinsics, dtype=torch.float32, device=self.device
            )

        # Feature extraction
        feature_config = self.config.get("feature_extraction", {})
        self.max_features = feature_config.get("max_features", 1000)
        self.feature_extractor = ORBFeatureExtractor(max_features=self.max_features)

        # Feature matching
        matcher_config = self.config.get("feature_matching", {})
        self.feature_matcher = FeatureMatcher(
            method=matcher_config.get("method", MatchingMethod.RATIO_TEST),
            ratio_threshold=matcher_config.get("ratio_threshold", 0.7),
            cross_check=matcher_config.get("cross_check", True),
            max_distance=matcher_config.get("max_distance", float("inf")),
            num_neighbors=matcher_config.get("num_neighbors", 2),
        )

        # Tracking
        tracking_config = self.config.get("tracking", {})
        self.tracker = KLTTracker(config=tracking_config)

        # Odometry
        odometry_config = self.config.get("odometry", {})
        self.odometry = PnPOdometry(odometry_config)

        # Keyframe management
        keyframe_config = self.config.get("keyframe", {})
        self.keyframe_manager = KeyframeManager(keyframe_config)

        # Loop detection
        loop_config = self.config.get("loop_detection", {})
        if loop_config.get("enabled", True):
            self.loop_detector = LoopDetector(loop_config)
            self.loop_detection_enabled = True
        else:
            self.loop_detection_enabled = False

        # Initialize current and previous frame data
        self.current_frame = None
        self.current_keypoints = None
        self.current_descriptors = None
        self.current_pose = FramePose.identity(device=self.device)
        self.previous_frame = None
        self.previous_keypoints = None
        self.previous_descriptors = None
        self.previous_pose = FramePose.identity(device=self.device)

        # Tracks
        self.tracks = {}  # Track ID to Track object
        self.next_track_id = 0

    def _init_backend(self):
        """Initialize backend components."""
        # Map of 3D points
        self.map_points = {}  # Landmark ID to 3D position
        self.landmark_observations = (
            {}
        )  # Landmark ID to list of (keyframe_id, track_id)
        self.next_landmark_id = 0

        # Bundle adjustment
        self.bundle_adjustment = BundleAdjustment(device=self.device)

        # Local mapping
        self.local_ba_frequency = self.config.get("local_ba_frequency", 5)
        self.local_ba_window = self.config.get("local_ba_window", 10)
        self.last_local_ba_frame = 0

        # Global bundle adjustment
        self.global_ba_enabled = self.config.get("global_ba_enabled", True)
        self.global_ba_frequency = self.config.get("global_ba_frequency", 20)

    def process_frame(self, frame: torch.Tensor, timestamp: float = None) -> Dict:
        """
        Process a new RGB frame.

        Args:
            frame: RGB image tensor (C, H, W)
            timestamp: Optional timestamp for the frame

        Returns:
            Dictionary with processed data including camera pose and status
        """
        start_time = time.time()
        self.frame_idx += 1

        # Ensure frame is on the correct device
        frame = frame.to(self.device)

        # Store current frame as previous
        self.previous_frame = self.current_frame
        self.previous_keypoints = self.current_keypoints
        self.previous_descriptors = self.current_descriptors
        self.previous_pose = self.current_pose

        # Extract features
        self.current_frame = frame
        (
            self.current_keypoints,
            self.current_descriptors,
        ) = self.feature_extractor.detect_and_compute(frame)

        # If this is the first frame, initialize and return
        if self.previous_frame is None:
            self.logger.info("Initializing with first frame")
            self._initialize_first_frame()
            processing_time = time.time() - start_time

            return {
                "pose": self.current_pose,
                "status": self.status,
                "processing_time": processing_time,
                "num_keypoints": len(self.current_keypoints),
                "frame_idx": self.frame_idx,
            }

        # Track features between frames
        tracked_keypoints, status, new_tracks = self._track_features()

        # Update pose using tracked features
        num_tracked = sum(status)
        if num_tracked < self.config.get("min_tracked_points", 10):
            self.logger.warning(f"Low number of tracked points: {num_tracked}")
            if self.status != VisualSLAMStatus.LOST:
                self.status = VisualSLAMStatus.TRACKING_BAD

            if num_tracked < 5:  # Not enough points for PnP
                self.status = VisualSLAMStatus.LOST
                self.logger.warning("Tracking lost, not enough points")
                processing_time = time.time() - start_time

                return {
                    "pose": self.current_pose,  # Use previous pose
                    "status": self.status,
                    "processing_time": processing_time,
                    "num_keypoints": len(self.current_keypoints),
                    "num_tracked": num_tracked,
                    "frame_idx": self.frame_idx,
                }

        # Update pose using tracked 3D-2D correspondences
        try:
            self._update_pose(tracked_keypoints, status)
        except Exception as e:
            self.logger.error(f"Error updating pose: {e}")
            self.status = VisualSLAMStatus.TRACKING_BAD

        # Check if we should create a new keyframe
        if self._should_create_keyframe():
            self.logger.info(f"Creating new keyframe at frame {self.frame_idx}")
            self._create_keyframe()

            # Check for loop closures if enabled
            if self.loop_detection_enabled and self.frame_idx > 10:
                self._detect_loop_closures()

            # Run local bundle adjustment if needed
            if (self.frame_idx - self.last_local_ba_frame) >= self.local_ba_frequency:
                self.logger.info("Running local bundle adjustment")
                self._run_local_bundle_adjustment()
                self.last_local_ba_frame = self.frame_idx

        # Prepare result
        processing_time = time.time() - start_time
        result = {
            "pose": self.current_pose,
            "status": self.status,
            "processing_time": processing_time,
            "num_keypoints": len(self.current_keypoints),
            "num_tracked": num_tracked,
            "frame_idx": self.frame_idx,
        }

        return result

    def _initialize_first_frame(self):
        """Initialize the system with the first frame."""
        # Set identity pose for the first frame
        self.current_pose = FramePose.identity(device=self.device)

        # Create initial tracks for all keypoints
        for idx, kp in enumerate(self.current_keypoints):
            track = Track(
                feature_id=self.next_track_id,
                keypoint=kp,
                descriptor=self.current_descriptors[idx].unsqueeze(0)
                if self.current_descriptors is not None
                else None,
            )
            self.tracks[self.next_track_id] = track
            self.next_track_id += 1

        # Create first keyframe
        self._create_keyframe()

        # Update status
        self.status = VisualSLAMStatus.INITIALIZING

    def _track_features(self) -> Tuple[List[KeyPoint], List[bool], List[int]]:
        """
        Track features between previous and current frames.

        Returns:
            Tuple of (tracked keypoints, status, new track IDs)
        """
        # Track features using optical flow
        prev_points = np.array(
            [kp.pt() for kp in self.previous_keypoints], dtype=np.float32
        )
        tracked_points, status, _ = self.tracker.track(
            self.previous_frame, self.current_frame, prev_points
        )

        # Convert tracked points to KeyPoint objects
        tracked_keypoints = []
        for i, (x, y) in enumerate(tracked_points):
            if status[i]:
                # Use the original keypoint properties but update position
                kp = KeyPoint(
                    x=float(x),
                    y=float(y),
                    response=self.previous_keypoints[i].response,
                    size=self.previous_keypoints[i].size,
                    angle=self.previous_keypoints[i].angle,
                    octave=self.previous_keypoints[i].octave,
                )
                tracked_keypoints.append(kp)
            else:
                tracked_keypoints.append(None)

        # Update tracks
        valid_track_ids = []
        for i, (is_tracked, kp) in enumerate(zip(status, tracked_keypoints)):
            if i >= len(self.previous_keypoints):
                continue

            # Find the track ID for this keypoint
            track_id = None
            for tid, track in self.tracks.items():
                if track.last_keypoint == self.previous_keypoints[i]:
                    track_id = tid
                    break

            if track_id is None:
                continue

            if is_tracked:
                # Update track
                self.tracks[track_id].update(
                    keypoint=kp,
                    descriptor=None,  # No descriptor for tracked points
                )
                valid_track_ids.append(track_id)
            else:
                # Mark track as lost
                self.tracks[track_id].status = TrackStatus.LOST

        # Create new tracks for unmatched keypoints in current frame
        new_track_ids = []
        if self.current_descriptors is not None:
            # Find keypoints that are not matched to any track
            current_kp_used = [False] * len(self.current_keypoints)

            # Mark keypoints that are already tracked
            for tid in valid_track_ids:
                for i, kp in enumerate(self.current_keypoints):
                    if kp.pt() == self.tracks[tid].last_keypoint.pt():
                        current_kp_used[i] = True

            # Create new tracks for unused keypoints
            for i, used in enumerate(current_kp_used):
                if not used:
                    track = Track(
                        feature_id=self.next_track_id,
                        keypoint=self.current_keypoints[i],
                        descriptor=self.current_descriptors[i].unsqueeze(0)
                        if self.current_descriptors is not None
                        else None,
                    )
                    self.tracks[self.next_track_id] = track
                    new_track_ids.append(self.next_track_id)
                    self.next_track_id += 1

        return tracked_keypoints, status, new_track_ids

    def _update_pose(self, tracked_keypoints: List[KeyPoint], status: List[bool]):
        """
        Update camera pose using tracked features.

        Args:
            tracked_keypoints: List of tracked keypoints
            status: List of tracking status booleans
        """
        # Collect 3D-2D correspondences for PnP
        points_3d = []
        points_2d = []

        for i, (is_tracked, kp) in enumerate(zip(status, tracked_keypoints)):
            if not is_tracked or kp is None:
                continue

            # Find track ID
            track_id = None
            for tid, track in self.tracks.items():
                if track.last_keypoint == kp:
                    track_id = tid
                    break

            if track_id is None:
                continue

            # Check if track has a 3D point
            landmark_id = None
            for keyframe in self.keyframe_manager.get_all_keyframes().values():
                lid = keyframe.get_landmark_id(track_id)
                if lid is not None:
                    landmark_id = lid
                    break

            if landmark_id is not None and landmark_id in self.map_points:
                # Add correspondence
                points_3d.append(self.map_points[landmark_id])
                points_2d.append(torch.tensor([kp.x, kp.y], device=self.device))

        # If we have enough correspondences, estimate pose with PnP
        if len(points_3d) >= 4:
            try:
                # Convert to tensors
                points_3d_tensor = torch.stack(points_3d)
                points_2d_tensor = torch.stack(points_2d)

                # Estimate pose
                data = {
                    "points_3d": points_3d_tensor,
                    "points_2d": points_2d_tensor,
                    "camera_intrinsics": self.camera_intrinsics,
                    "previous_pose": self.previous_pose,
                }

                self.current_pose = self.odometry.process_frame(data)

                # Check status
                if self.odometry.get_status() == OdometryStatus.OK:
                    self.status = VisualSLAMStatus.TRACKING_GOOD
                else:
                    self.status = VisualSLAMStatus.TRACKING_BAD
            except Exception as e:
                self.logger.error(f"PnP failed: {e}")
                # Use relative motion from previous frames as a fallback
                if self.status != VisualSLAMStatus.LOST:
                    relative_motion = self.odometry.get_relative_motion()
                    self.current_pose = self.previous_pose.compose(relative_motion)
                    self.status = VisualSLAMStatus.TRACKING_BAD
        else:
            self.logger.warning(
                f"Not enough 3D-2D correspondences for PnP: {len(points_3d)}"
            )
            if self.status != VisualSLAMStatus.LOST:
                # Use previous pose as a fallback
                self.current_pose = self.previous_pose
                self.status = VisualSLAMStatus.TRACKING_BAD

    def _should_create_keyframe(self) -> bool:
        """
        Determine if a new keyframe should be created.

        Returns:
            True if a new keyframe should be created, False otherwise
        """
        # If we're initializing, create keyframes more frequently
        if self.status == VisualSLAMStatus.INITIALIZING:
            return self.frame_idx > 0 and self.frame_idx % 5 == 0

        # If tracking is bad, don't create keyframes
        if self.status != VisualSLAMStatus.TRACKING_GOOD:
            return False

        # If this is one of the first frames, create keyframes more frequently
        if self.frame_idx < 10:
            return self.frame_idx % 5 == 0

        # Create a keyframe if there are not enough visible map points
        visible_landmarks = 0
        for track_id, track in self.tracks.items():
            if track.status == TrackStatus.TRACKED:
                for keyframe in self.keyframe_manager.get_all_keyframes().values():
                    if keyframe.get_landmark_id(track_id) is not None:
                        visible_landmarks += 1
                        break

        if visible_landmarks < self.config.get("min_visible_landmarks", 30):
            return True

        # Create a keyframe if enough frames have passed since the last one
        min_frames_between_keyframes = self.config.get(
            "min_frames_between_keyframes", 20
        )
        if self.frame_idx - self.last_keyframe_idx > min_frames_between_keyframes:
            return True

        return False

    def _create_keyframe(self):
        """Create a new keyframe from the current frame."""
        # Create tracks dictionary for the keyframe
        keyframe_tracks = {}
        for track_id, track in self.tracks.items():
            if track.status == TrackStatus.TRACKED:
                keyframe_tracks[track_id] = track

        # Create the keyframe
        keyframe = self.keyframe_manager.create_keyframe(
            pose=self.current_pose,
            frame=self.current_frame,
            keypoints=self.current_keypoints,
            descriptors=self.current_descriptors,
            tracks=keyframe_tracks,
            timestamp=time.time(),
        )

        if keyframe is None:
            self.logger.warning("Failed to create keyframe")
            return

        # Update the last keyframe index
        self.last_keyframe_idx = self.frame_idx

        # Initialize new landmarks for tracks without 3D points
        self._initialize_new_landmarks(keyframe)

        # Add keyframe to bundle adjustment
        self.bundle_adjustment.add_camera(
            camera_id=f"cam_{keyframe.keyframe_id}",
            pose=keyframe.pose.as_matrix(),
            intrinsics=self.camera_intrinsics,
        )

        # Fix the first camera to set the global reference frame
        if keyframe.keyframe_id == 0:
            self.bundle_adjustment.set_fixed_camera(f"cam_{keyframe.keyframe_id}")

    def _initialize_new_landmarks(self, keyframe: Keyframe):
        """
        Initialize new landmarks for tracks in the keyframe.

        Args:
            keyframe: The keyframe to initialize landmarks for
        """
        # Get all tracks in the keyframe
        for track_id in keyframe.get_track_ids():
            # Skip tracks that already have landmarks
            if keyframe.get_landmark_id(track_id) is not None:
                continue

            # Check if this track has a landmark in other keyframes
            landmark_id = None
            for kf in self.keyframe_manager.get_all_keyframes().values():
                if kf.keyframe_id == keyframe.keyframe_id:
                    continue

                lid = kf.get_landmark_id(track_id)
                if lid is not None:
                    landmark_id = lid
                    break

            if landmark_id is not None:
                # Add observation of existing landmark
                keyframe.add_landmark_observation(
                    track_id=track_id,
                    landmark_id=landmark_id,
                    point_3d=self.map_points[landmark_id],
                )

                # Update landmark observations
                if landmark_id in self.landmark_observations:
                    self.landmark_observations[landmark_id].append(
                        (keyframe.keyframe_id, track_id)
                    )
                else:
                    self.landmark_observations[landmark_id] = [
                        (keyframe.keyframe_id, track_id)
                    ]
            else:
                # If we have at least 2 keyframes, try to triangulate new landmarks
                if len(self.keyframe_manager.get_all_keyframes()) >= 2:
                    self._triangulate_landmark(keyframe, track_id)

    def _triangulate_landmark(self, keyframe: Keyframe, track_id: int):
        """
        Triangulate a new landmark from a track observed in multiple keyframes.

        Args:
            keyframe: The current keyframe
            track_id: The track ID to triangulate
        """
        # Find other keyframes that observe this track
        other_observations = []
        for kf in self.keyframe_manager.get_all_keyframes().values():
            if kf.keyframe_id == keyframe.keyframe_id:
                continue

            if track_id in kf.get_track_ids():
                other_observations.append(kf)

        if not other_observations:
            return

        # Get the best keyframe for triangulation (largest baseline)
        best_keyframe = None
        best_angle = 0.0
        current_pos = keyframe.pose.translation
        current_rot = keyframe.pose.rotation

        for kf in other_observations:
            kf_pos = kf.pose.translation
            baseline = torch.norm(current_pos - kf_pos)

            # Skip if baseline is too small
            if baseline < 0.1:
                continue

            # Compute viewing angle
            ray1 = current_rot.transpose(0, 1) @ torch.tensor(
                [0, 0, 1.0], device=self.device
            )
            ray2 = kf.pose.rotation.transpose(0, 1) @ torch.tensor(
                [0, 0, 1.0], device=self.device
            )
            angle = torch.acos(torch.clamp(torch.dot(ray1, ray2), -1.0, 1.0))

            if angle > best_angle:
                best_angle = angle
                best_keyframe = kf

        if best_keyframe is None:
            return

        # Triangulate the landmark
        try:
            # Get the keypoints
            current_kp = None
            other_kp = None

            for track in keyframe.tracks.values():
                if track.track_id == track_id:
                    current_kp = track.last_keypoint
                    break

            for track in best_keyframe.tracks.values():
                if track.track_id == track_id:
                    other_kp = track.last_keypoint
                    break

            if current_kp is None or other_kp is None:
                return

            # Set up the triangulation matrices
            P1 = torch.cat(
                (keyframe.pose.rotation, keyframe.pose.translation.unsqueeze(1)), dim=1
            )
            P1 = self.camera_intrinsics @ P1

            P2 = torch.cat(
                (
                    best_keyframe.pose.rotation,
                    best_keyframe.pose.translation.unsqueeze(1),
                ),
                dim=1,
            )
            P2 = self.camera_intrinsics @ P2

            # Construct the linear system
            A = torch.zeros((4, 4), device=self.device)
            A[0] = current_kp.x * P1[2] - P1[0]
            A[1] = current_kp.y * P1[2] - P1[1]
            A[2] = other_kp.x * P2[2] - P2[0]
            A[3] = other_kp.y * P2[2] - P2[1]

            # Solve using SVD
            _, _, V = torch.svd(A)
            point_homo = V[:, -1]
            point_3d = point_homo[:3] / point_homo[3]

            # Check if point is in front of both cameras
            point_cam1 = keyframe.pose.rotation @ point_3d + keyframe.pose.translation
            point_cam2 = (
                best_keyframe.pose.rotation @ point_3d + best_keyframe.pose.translation
            )

            if point_cam1[2] <= 0 or point_cam2[2] <= 0:
                return

            # Add the landmark
            landmark_id = self.next_landmark_id
            self.next_landmark_id += 1

            # Store the 3D point
            self.map_points[landmark_id] = point_3d

            # Add landmark observations
            keyframe.add_landmark_observation(
                track_id=track_id,
                landmark_id=landmark_id,
                point_3d=point_3d,
            )

            best_keyframe.add_landmark_observation(
                track_id=track_id,
                landmark_id=landmark_id,
                point_3d=point_3d,
            )

            # Update landmark observations
            self.landmark_observations[landmark_id] = [
                (keyframe.keyframe_id, track_id),
                (best_keyframe.keyframe_id, track_id),
            ]

            # Add landmark to bundle adjustment
            self.bundle_adjustment.add_landmark(
                landmark_id=f"lm_{landmark_id}",
                position=point_3d,
            )

            # Add observations to bundle adjustment
            self.bundle_adjustment.add_observation(
                camera_id=f"cam_{keyframe.keyframe_id}",
                landmark_id=f"lm_{landmark_id}",
                observation=torch.tensor(
                    [current_kp.x, current_kp.y], device=self.device
                ),
            )

            self.bundle_adjustment.add_observation(
                camera_id=f"cam_{best_keyframe.keyframe_id}",
                landmark_id=f"lm_{landmark_id}",
                observation=torch.tensor([other_kp.x, other_kp.y], device=self.device),
            )
        except Exception as e:
            self.logger.error(f"Triangulation failed: {e}")

    def _run_local_bundle_adjustment(self):
        """Run local bundle adjustment on recent keyframes."""
        # Get the most recent keyframes
        all_keyframes = self.keyframe_manager.get_all_keyframes()
        keyframe_ids = sorted(list(all_keyframes.keys()))

        if len(keyframe_ids) <= 1:
            return

        # Select keyframes for local BA
        recent_keyframe_ids = keyframe_ids[
            -min(self.local_ba_window, len(keyframe_ids)) :
        ]
        recent_keyframes = [all_keyframes[kid] for kid in recent_keyframe_ids]

        # Run optimization
        try:
            result = self.bundle_adjustment.optimize(
                max_iterations=self.config.get("local_ba_iterations", 10),
                verbose=False,
            )

            # Update keyframe poses
            for kf in recent_keyframes:
                pose_matrix = self.bundle_adjustment.get_camera_pose(
                    f"cam_{kf.keyframe_id}"
                )
                kf.set_pose(FramePose.from_matrix(pose_matrix))

            # Update landmark positions
            for landmark_id in self.map_points.keys():
                if f"lm_{landmark_id}" in self.bundle_adjustment.get_all_landmarks():
                    new_position = self.bundle_adjustment.get_landmark_position(
                        f"lm_{landmark_id}"
                    )
                    self.map_points[landmark_id] = new_position
        except Exception as e:
            self.logger.error(f"Local bundle adjustment failed: {e}")

    def _detect_loop_closures(self):
        """Detect and process loop closures."""
        if not hasattr(self, "loop_detector") or self.loop_detector is None:
            return

        # Get the current keyframe
        current_keyframe = self.keyframe_manager.get_keyframe(self.last_keyframe_idx)
        if current_keyframe is None:
            return

        # Detect loop candidates
        try:
            loop_candidates = self.loop_detector.detect_loop_candidates(
                current_keyframe,
                self.keyframe_manager.get_all_keyframes(),
                min_matches=self.config.get("min_loop_matches", 30),
            )

            if not loop_candidates:
                return

            # Process the best loop candidate
            best_candidate = loop_candidates[0]
            self.logger.info(
                f"Loop detected between keyframes {current_keyframe.keyframe_id} and {best_candidate.keyframe_id}"
            )

            # Add loop connection
            current_keyframe.set_as_loop_keyframe(True)
            current_keyframe.add_loop_connection(best_candidate.keyframe_id)

            # Run pose graph optimization to correct the loop
            self._optimize_pose_graph()
        except Exception as e:
            self.logger.error(f"Loop closure detection failed: {e}")

    def _optimize_pose_graph(self):
        """Optimize the pose graph after a loop closure."""
        # TODO: Implement pose graph optimization for loop closure
        # This would typically involve:
        # 1. Building a pose graph
        # 2. Adding constraints from keyframe covisibility
        # 3. Adding loop closure constraints
        # 4. Optimizing the graph
        # 5. Updating keyframe poses
        self.logger.info("Pose graph optimization not yet implemented")

    def get_camera_pose(self) -> FramePose:
        """
        Get the current camera pose.

        Returns:
            Current camera pose
        """
        return self.current_pose

    def get_status(self) -> VisualSLAMStatus:
        """
        Get the current system status.

        Returns:
            Current SLAM status
        """
        return self.status

    def get_map_points(self) -> Dict[int, torch.Tensor]:
        """
        Get the current map points.

        Returns:
            Dictionary mapping landmark ID to 3D position
        """
        return self.map_points

    def get_keyframes(self) -> Dict[int, Keyframe]:
        """
        Get all keyframes.

        Returns:
            Dictionary mapping keyframe ID to Keyframe object
        """
        return self.keyframe_manager.get_all_keyframes()

    def reset(self):
        """Reset the SLAM system."""
        self.frame_idx = 0
        self.last_keyframe_idx = -1
        self.status = VisualSLAMStatus.INITIALIZING

        # Reset frontend
        self.current_frame = None
        self.current_keypoints = None
        self.current_descriptors = None
        self.current_pose = FramePose.identity(device=self.device)
        self.previous_frame = None
        self.previous_keypoints = None
        self.previous_descriptors = None
        self.previous_pose = FramePose.identity(device=self.device)

        # Reset tracks
        self.tracks = {}
        self.next_track_id = 0

        # Reset backend
        self.map_points = {}
        self.landmark_observations = {}
        self.next_landmark_id = 0

        # Reset bundle adjustment
        self.bundle_adjustment = BundleAdjustment(device=self.device)
        self.last_local_ba_frame = 0
