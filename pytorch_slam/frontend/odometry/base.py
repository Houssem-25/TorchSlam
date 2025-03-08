import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


class OdometryStatus(Enum):
    """Status of odometry estimation."""

    OK = 0
    LOST = 1
    UNCERTAIN = 2
    INITIALIZING = 3


class FramePose:
    """Represents a pose of a frame."""

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

    @property
    def device(self) -> torch.device:
        """Get device of tensors."""
        return self.rotation.device

    def as_matrix(self) -> torch.Tensor:
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

    def inverse(self) -> "FramePose":
        """
        Compute the inverse of this pose.

        Returns:
            Inverse pose
        """
        inv_rotation = self.rotation.transpose(0, 1)
        inv_translation = -torch.matmul(inv_rotation, self.translation)
        return FramePose(inv_rotation, inv_translation, self.timestamp)

    def compose(self, other: "FramePose") -> "FramePose":
        """
        Compose this pose with another pose: self * other

        Args:
            other: Other pose to compose with

        Returns:
            Composed pose
        """
        new_rotation = torch.matmul(self.rotation, other.rotation)
        new_translation = (
            torch.matmul(self.rotation, other.translation) + self.translation
        )
        new_timestamp = (
            other.timestamp if other.timestamp is not None else self.timestamp
        )
        return FramePose(new_rotation, new_translation, new_timestamp)

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

    @classmethod
    def identity(cls, device: torch.device = None) -> "FramePose":
        """
        Create an identity pose.

        Args:
            device: PyTorch device

        Returns:
            Identity pose
        """
        rotation = torch.eye(3, device=device)
        translation = torch.zeros(3, device=device)
        return cls(rotation, translation)

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert to NumPy arrays.

        Returns:
            Tuple of (rotation, translation) as NumPy arrays
        """
        return self.rotation.cpu().numpy(), self.translation.cpu().numpy()

    @classmethod
    def from_numpy(
        cls,
        rotation: np.ndarray,
        translation: np.ndarray,
        timestamp: float = None,
        device: torch.device = None,
    ) -> "FramePose":
        """
        Create a pose from NumPy arrays.

        Args:
            rotation: Rotation matrix as NumPy array
            translation: Translation vector as NumPy array
            timestamp: Optional timestamp
            device: PyTorch device

        Returns:
            FramePose object
        """
        rot_tensor = torch.tensor(rotation, dtype=torch.float32, device=device)
        trans_tensor = torch.tensor(translation, dtype=torch.float32, device=device)
        return cls(rot_tensor, trans_tensor, timestamp)


class BaseOdometry(ABC):
    """Base class for odometry estimation."""

    def __init__(self, config: Dict = None):
        """
        Initialize odometry estimator.

        Args:
            config: Configuration dictionary
        """
        self.config = config if config is not None else {}
        self.current_pose = FramePose.identity()
        self.previous_pose = FramePose.identity()
        self.status = OdometryStatus.INITIALIZING
        self.frame_idx = 0
        self.relative_motion = FramePose.identity()
        self.is_initialized = False

        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def process_frame(self, data: Dict[str, Any]) -> FramePose:
        """
        Process a new frame to estimate odometry.

        Args:
            data: Dictionary containing sensor data (depends on the odometry method)

        Returns:
            Estimated pose for the current frame
        """
        pass

    def reset(self):
        """Reset odometry estimator."""
        self.current_pose = FramePose.identity()
        self.previous_pose = FramePose.identity()
        self.status = OdometryStatus.INITIALIZING
        self.frame_idx = 0
        self.relative_motion = FramePose.identity()
        self.is_initialized = False

    def update_pose(self, relative_motion: FramePose):
        """
        Update the pose with relative motion.

        Args:
            relative_motion: Relative motion between frames
        """
        # Update poses
        self.previous_pose = self.current_pose
        self.current_pose = self.previous_pose.compose(relative_motion)
        self.relative_motion = relative_motion
        self.frame_idx += 1

        # If we're starting with good estimations, change status to OK
        if self.status == OdometryStatus.INITIALIZING and self.frame_idx > 5:
            self.status = OdometryStatus.OK
            self.is_initialized = True

    def get_current_pose(self) -> FramePose:
        """
        Get the current estimated pose.

        Returns:
            Current pose
        """
        return self.current_pose

    def get_relative_motion(self) -> FramePose:
        """
        Get the relative motion between the last two frames.

        Returns:
            Relative motion
        """
        return self.relative_motion

    def get_status(self) -> OdometryStatus:
        """
        Get the current status of the odometry estimation.

        Returns:
            Odometry status
        """
        return self.status
