import logging
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch

from ..frontend.keyframe import Keyframe, KeyframeStatus
from .optimization.pose_graph import PoseGraphOptimization
from .se3 import SE3


class PoseGraphCorrection:
    """
    Pose graph correction for loop closure.

    This class implements pose graph optimization to correct the trajectory
    after loop closures are detected. It uses the pose graph optimizer to
    distribute the error across the trajectory.
    """

    def __init__(self, device: torch.device = None):
        """
        Initialize pose graph correction.

        Args:
            device: PyTorch device
        """
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.optimizer = PoseGraphOptimization(device=self.device)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Uncertainty parameters
        self.odom_information = (
            torch.eye(6, device=self.device) * 100.0
        )  # Default odometry information
        self.loop_information = (
            torch.eye(6, device=self.device) * 10.0
        )  # Default loop closure information

    def set_odometry_uncertainty(self, uncertainty: torch.Tensor):
        """
        Set the uncertainty for odometry constraints.

        Args:
            uncertainty: 6x6 covariance matrix for odometry
        """
        if uncertainty.shape != (6, 6):
            raise ValueError(f"Expected 6x6 covariance matrix, got {uncertainty.shape}")

        # Information matrix is the inverse of covariance
        self.odom_information = torch.inverse(uncertainty.to(self.device))

    def set_loop_uncertainty(self, uncertainty: torch.Tensor):
        """
        Set the uncertainty for loop closure constraints.

        Args:
            uncertainty: 6x6 covariance matrix for loop closures
        """
        if uncertainty.shape != (6, 6):
            raise ValueError(f"Expected 6x6 covariance matrix, got {uncertainty.shape}")

        # Information matrix is the inverse of covariance
        self.loop_information = torch.inverse(uncertainty.to(self.device))

    def correct_trajectory(
        self,
        keyframes: Dict[int, Keyframe],
        loop_closures: List[Tuple[int, int, torch.Tensor]],
        fix_first_pose: bool = True,
    ) -> Dict[int, torch.Tensor]:
        """
        Correct the trajectory using pose graph optimization.

        Args:
            keyframes: Dictionary of keyframes by ID
            loop_closures: List of (query_id, match_id, relative_pose) tuples
            fix_first_pose: Whether to fix the first pose

        Returns:
            Dictionary mapping keyframe ID to corrected pose
        """
        self.logger.info(
            f"Correcting trajectory with {len(loop_closures)} loop closures"
        )

        # Build pose graph
        self._build_pose_graph(keyframes, loop_closures, fix_first_pose)

        # Optimize
        result = self.optimizer.optimize(max_iterations=100, verbose=True)

        # Check if optimization was successful
        if not result.success:
            self.logger.warning("Pose graph optimization failed")
            return {kf_id: kf.pose.to_matrix() for kf_id, kf in keyframes.items()}

        # Get corrected poses
        corrected_poses = {}
        for kf_id in keyframes.keys():
            corrected_poses[kf_id] = self.optimizer.get_pose(str(kf_id))

        self.logger.info(
            f"Pose graph optimization completed with final cost {result.final_cost:.6f}"
        )
        self.logger.info(
            f"Initial cost: {result.initial_cost:.6f}, iterations: {result.num_iterations}"
        )

        return corrected_poses

    def _build_pose_graph(
        self,
        keyframes: Dict[int, Keyframe],
        loop_closures: List[Tuple[int, int, torch.Tensor]],
        fix_first_pose: bool,
    ):
        """
        Build pose graph from keyframes and loop closures.

        Args:
            keyframes: Dictionary of keyframes by ID
            loop_closures: List of (query_id, match_id, relative_pose) tuples
            fix_first_pose: Whether to fix the first pose
        """
        # Reset the optimizer
        self.optimizer = PoseGraphOptimization(device=self.device)

        # Add keyframe poses to the graph
        keyframe_ids = sorted(keyframes.keys())
        for kf_id in keyframe_ids:
            keyframe = keyframes[kf_id]
            pose_matrix = keyframe.pose.to_matrix().to(self.device)
            self.optimizer.add_pose(str(kf_id), pose_matrix)

        # Fix the first pose if requested
        if fix_first_pose and keyframe_ids:
            self.optimizer.set_fixed_pose(str(keyframe_ids[0]))

        # Add sequential pose constraints (odometry)
        for i in range(len(keyframe_ids) - 1):
            kf_id_i = keyframe_ids[i]
            kf_id_j = keyframe_ids[i + 1]

            pose_i = keyframes[kf_id_i].pose.to_matrix().to(self.device)
            pose_j = keyframes[kf_id_j].pose.to_matrix().to(self.device)

            # Compute relative pose: T_i^-1 * T_j
            pose_i_se3 = SE3.from_matrix(pose_i)
            pose_j_se3 = SE3.from_matrix(pose_j)
            relative_pose = pose_i_se3.inverse().compose(pose_j_se3).to_matrix()

            # Add constraint
            self.optimizer.add_relative_pose_constraint(
                str(kf_id_i), str(kf_id_j), relative_pose, self.odom_information
            )

        # Add loop closure constraints
        for query_id, match_id, relative_pose in loop_closures:
            if (
                str(query_id) not in self.optimizer.graph.variables
                or str(match_id) not in self.optimizer.graph.variables
            ):
                self.logger.warning(
                    f"Loop closure involves unknown keyframe: {query_id} -> {match_id}"
                )
                continue

            # Make sure the relative pose is on the right device
            relative_pose = relative_pose.to(self.device)

            # Add loop closure constraint
            self.optimizer.add_loop_closure_constraint(
                str(match_id), str(query_id), relative_pose, self.loop_information
            )


class LandmarkCorrection:
    """
    Landmark position correction after pose graph optimization.

    This class applies corrections to landmark positions after the
    camera poses have been corrected by pose graph optimization.
    """

    def __init__(self, device: torch.device = None):
        """
        Initialize landmark correction.

        Args:
            device: PyTorch device
        """
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def correct_landmarks(
        self,
        keyframes: Dict[int, Keyframe],
        corrected_poses: Dict[int, torch.Tensor],
        landmarks: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """
        Correct landmark positions based on corrected poses.

        Args:
            keyframes: Dictionary of keyframes by ID
            corrected_poses: Dictionary of corrected poses by keyframe ID
            landmarks: Dictionary of landmark positions by landmark ID

        Returns:
            Dictionary of corrected landmark positions by landmark ID
        """
        self.logger.info(f"Correcting {len(landmarks)} landmarks")

        # Build keyframe observations
        landmark_observations = self._build_landmark_observations(keyframes)

        # Create corrected SE3 poses
        corrected_se3_poses = {
            kf_id: SE3.from_matrix(pose.to(self.device))
            for kf_id, pose in corrected_poses.items()
        }

        # Original SE3 poses
        original_se3_poses = {
            kf_id: SE3.from_matrix(kf.pose.to_matrix().to(self.device))
            for kf_id, kf in keyframes.items()
        }

        # Compute transformations between original and corrected poses
        pose_transforms = {}
        for kf_id in corrected_se3_poses.keys():
            if kf_id in original_se3_poses:
                # T_corrected * T_original^-1
                pose_transforms[kf_id] = corrected_se3_poses[kf_id].compose(
                    original_se3_poses[kf_id].inverse()
                )

        # Correct landmark positions
        corrected_landmarks = {}

        for landmark_id, landmark_pos in landmarks.items():
            # Skip if no observations
            if landmark_id not in landmark_observations:
                corrected_landmarks[landmark_id] = landmark_pos.clone()
                continue

            # Collect all corrected landmark positions from different observations
            corrected_positions = []
            weights = []

            landmark_pos = landmark_pos.to(self.device)

            for kf_id in landmark_observations[landmark_id]:
                if kf_id not in pose_transforms:
                    continue

                # Transform landmark using the pose correction
                corrected_pos = pose_transforms[kf_id].transform_point(landmark_pos)
                corrected_positions.append(corrected_pos)

                # Weight by inverse of distance (closer keyframes have higher weight)
                keyframe_pos = corrected_se3_poses[kf_id].translation_vector()
                distance = torch.norm(corrected_pos - keyframe_pos)
                weight = 1.0 / (distance + 1e-6)
                weights.append(weight)

            if corrected_positions:
                # Compute weighted average
                weights = torch.tensor(weights, device=self.device)
                weights = weights / weights.sum()

                corrected_pos = torch.zeros_like(landmark_pos)
                for i, pos in enumerate(corrected_positions):
                    corrected_pos += weights[i] * pos

                corrected_landmarks[landmark_id] = corrected_pos
            else:
                # Keep original position if no observations
                corrected_landmarks[landmark_id] = landmark_pos.clone()

        self.logger.info(f"Landmark correction completed")

        return corrected_landmarks

    def _build_landmark_observations(
        self, keyframes: Dict[int, Keyframe]
    ) -> Dict[int, List[int]]:
        """
        Build a dictionary mapping landmark IDs to lists of observing keyframe IDs.

        Args:
            keyframes: Dictionary of keyframes by ID

        Returns:
            Dictionary mapping landmark ID to list of keyframe IDs
        """
        landmark_observations = defaultdict(list)

        for kf_id, keyframe in keyframes.items():
            for track_id, landmark_id in keyframe.landmark_observations.items():
                landmark_observations[landmark_id].append(kf_id)

        return landmark_observations


class LoopCorrectionManager:
    """
    Manager for the loop closure correction process.

    This class coordinates the pose graph optimization and landmark correction
    after loop closures are detected.
    """

    def __init__(self, device: torch.device = None):
        """
        Initialize loop correction manager.

        Args:
            device: PyTorch device
        """
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.pose_corrector = PoseGraphCorrection(device=self.device)
        self.landmark_corrector = LandmarkCorrection(device=self.device)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Correction parameters
        self.min_loop_closures = (
            1  # Minimum number of loop closures to trigger correction
        )
        self.correction_interval = 5  # Minimum interval between corrections

        # State variables
        self.last_correction_keyframe_id = -1
        self.pending_loop_closures = []  # List of (query_id, match_id, relative_pose)

    def add_loop_closure(
        self,
        query_keyframe_id: int,
        match_keyframe_id: int,
        relative_pose: torch.Tensor,
    ):
        """
        Add a loop closure to be processed.

        Args:
            query_keyframe_id: ID of the query keyframe
            match_keyframe_id: ID of the matched keyframe
            relative_pose: Relative pose from match to query
        """
        self.pending_loop_closures.append(
            (query_keyframe_id, match_keyframe_id, relative_pose)
        )
        self.logger.info(
            f"Added loop closure: {match_keyframe_id} -> {query_keyframe_id}"
        )

    def should_correct(self, current_keyframe_id: int) -> bool:
        """
        Check if correction should be performed.

        Args:
            current_keyframe_id: ID of the current keyframe

        Returns:
            True if correction should be performed, False otherwise
        """
        # Check if we have enough loop closures
        if len(self.pending_loop_closures) < self.min_loop_closures:
            return False

        # Check if enough keyframes have passed since last correction
        if (
            current_keyframe_id - self.last_correction_keyframe_id
            < self.correction_interval
        ):
            return False

        return True

    def correct(
        self, keyframes: Dict[int, Keyframe], landmarks: Dict[int, torch.Tensor]
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Perform trajectory and landmark correction.

        Args:
            keyframes: Dictionary of keyframes by ID
            landmarks: Dictionary of landmark positions by landmark ID

        Returns:
            Tuple of (corrected_poses, corrected_landmarks)
        """
        if not self.pending_loop_closures:
            return {
                kf_id: kf.pose.to_matrix() for kf_id, kf in keyframes.items()
            }, landmarks

        self.logger.info(
            f"Starting loop correction with {len(self.pending_loop_closures)} loop closures"
        )

        # Correct camera poses
        corrected_poses = self.pose_corrector.correct_trajectory(
            keyframes, self.pending_loop_closures, fix_first_pose=True
        )

        # Correct landmark positions
        corrected_landmarks = self.landmark_corrector.correct_landmarks(
            keyframes, corrected_poses, landmarks
        )

        # Update state
        self.last_correction_keyframe_id = max(keyframes.keys())
        self.pending_loop_closures = []

        self.logger.info(f"Loop correction completed")

        return corrected_poses, corrected_landmarks

    def update_keyframes_and_landmarks(
        self,
        keyframes: Dict[int, Keyframe],
        landmarks: Dict[int, torch.Tensor],
        corrected_poses: Dict[int, torch.Tensor],
        corrected_landmarks: Dict[int, torch.Tensor],
    ):
        """
        Update keyframes and landmarks with corrected poses and positions.

        Args:
            keyframes: Dictionary of keyframes by ID
            landmarks: Dictionary of landmark positions by landmark ID
            corrected_poses: Dictionary of corrected poses by keyframe ID
            corrected_landmarks: Dictionary of corrected landmark positions by landmark ID
        """
        # Update keyframe poses
        for kf_id, pose in corrected_poses.items():
            if kf_id in keyframes:
                keyframes[kf_id].set_pose(FramePose.from_matrix(pose))

        # Update landmark positions
        for landmark_id, position in corrected_landmarks.items():
            landmarks[landmark_id] = position

        self.logger.info(
            f"Updated {len(corrected_poses)} keyframes and {len(corrected_landmarks)} landmarks"
        )


# Define the FramePose class for compatibility
class FramePose:
    """
    Represents a pose of a frame.

    This is a simplified version for use in the loop correction module.
    The full implementation would be imported from the frontend.
    """

    def __init__(
        self, rotation: torch.Tensor, translation: torch.Tensor, timestamp: float = None
    ):
        """
        Initialize frame pose.

        Args:
            rotation: Rotation matrix (3x3)
            translation: Translation vector (3)
            timestamp: Optional timestamp
        """
        self.rotation = rotation
        self.translation = translation
        self.timestamp = timestamp

    def to_matrix(self) -> torch.Tensor:
        """
        Return pose as a 4x4 transformation matrix.

        Returns:
            4x4 transformation matrix (rotation and translation)
        """
        device = self.rotation.device
        T = torch.eye(4, device=device)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.translation
        return T

    @classmethod
    def from_matrix(cls, matrix: torch.Tensor, timestamp: float = None) -> "FramePose":
        """
        Create a pose from a 4x4 transformation matrix.

        Args:
            matrix: 4x4 transformation matrix
            timestamp: Optional timestamp

        Returns:
            FramePose object
        """
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]
        return cls(rotation, translation, timestamp)
