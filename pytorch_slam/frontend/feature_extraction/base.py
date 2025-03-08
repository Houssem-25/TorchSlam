import math
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


class FeatureType(Enum):
    """Enum for different feature types."""

    SIFT = 1
    ORB = 2


class KeyPoint:
    """Class representing a keypoint."""

    def __init__(
        self,
        x: float,
        y: float,
        response: float = 0.0,
        size: float = 1.0,
        angle: float = -1.0,
        octave: int = 0,
    ):
        self.x = x
        self.y = y
        self.response = response  # Strength of the keypoint
        self.size = size  # Diameter of the meaningful keypoint neighborhood
        self.angle = angle  # Orientation in degrees (-1 if not applicable)
        self.octave = (
            octave  # Octave (pyramid layer) from which the keypoint was extracted
        )

    def pt(self) -> Tuple[float, float]:
        """Get point coordinates."""
        return (self.x, self.y)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "x": self.x,
            "y": self.y,
            "response": self.response,
            "size": self.size,
            "angle": self.angle,
            "octave": self.octave,
        }

    @staticmethod
    def from_dict(data: Dict) -> "KeyPoint":
        """Create from dictionary."""
        return KeyPoint(
            x=data["x"],
            y=data["y"],
            response=data["response"],
            size=data["size"],
            angle=data["angle"],
            octave=data["octave"],
        )


class BaseFeatureExtractor:
    """Base class for feature extraction.

    This class defines the interface for feature extraction and provides common
    utility functions used by different feature extractors."""

    def __init__(self, max_features: int = 1000):
        self.max_features = max_features

    def extract(self, image: torch.Tensor) -> List[KeyPoint]:
        """
        Extract features from image.

        Args:
            image: PyTorch tensor with shape (C, H, W) or (H, W)

        Returns:
            List of KeyPoint objects
        """
        raise NotImplementedError("Subclasses must implement extract method")

    def compute_descriptors(
        self, image: torch.Tensor, keypoints: List[KeyPoint]
    ) -> torch.Tensor:
        """
        Compute descriptors for keypoints.

        Args:
            image: PyTorch tensor with shape (C, H, W) or (H, W)
            keypoints: List of KeyPoint objects

        Returns:
            Tensor of descriptors
        """
        raise NotImplementedError(
            "Subclasses must implement compute_descriptors method"
        )

    def detect_and_compute(
        self, image: torch.Tensor
    ) -> Tuple[List[KeyPoint], torch.Tensor]:
        """
        Extract features and compute descriptors.

        Args:
            image: PyTorch tensor with shape (C, H, W) or (H, W)

        Returns:
            Tuple of (keypoints, descriptors)
        """
        keypoints = self.extract(image)
        descriptors = self.compute_descriptors(image, keypoints)
        return keypoints, descriptors

    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """Preprocess image for feature extraction."""
        # Ensure image is on the correct device
        device = image.device

        # Convert to grayscale if color
        if image.dim() == 3 and image.shape[0] > 1:
            # Simple rgb to grayscale - just average the channels
            # In a more sophisticated implementation, we'd use the correct weighting
            gray = image.mean(dim=0, keepdim=True)
        elif image.dim() == 3 and image.shape[0] == 1:
            gray = image
        elif image.dim() == 2:
            gray = image.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported image format with shape {image.shape}")

        # Ensure float dtype in range [0, 1]
        if gray.dtype != torch.float32:
            gray = gray.float()

        if gray.max() > 1.0 + 1e-6:
            gray = gray / 255.0

        return gray

    def _compute_image_gradients(
        self, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute image gradients using Sobel operators.

        Args:
            image: Single-channel image tensor (1, H, W)

        Returns:
            Tuple of (dx, dy) gradient tensors
        """
        device = image.device

        # Define Sobel kernels
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device
        ).view(1, 1, 3, 3)

        # Apply padding to handle borders
        padded = F.pad(image, (1, 1, 1, 1), mode="reflect")

        # Apply convolution
        dx = F.conv2d(padded, sobel_x)
        dy = F.conv2d(padded, sobel_y)

        return dx, dy

    def _gaussian_kernel_2d(
        self, kernel_size: int, sigma: float, device: torch.device
    ) -> torch.Tensor:
        """Create a 2D Gaussian kernel."""
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")

        # Create 1D kernels
        coords = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2
        x = coords.repeat(kernel_size, 1)
        y = x.t()

        # Create 2D Gaussian
        kernel = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * sigma * sigma))
        kernel = kernel / kernel.sum()  # Normalize

        return kernel

    def _non_maximum_suppression(
        self, response_map: torch.Tensor, max_points: int, window_size: int = 3
    ) -> List[KeyPoint]:
        """
        Perform non-maximum suppression to find local maxima in the response map.

        Args:
            response_map: Tensor containing corner/feature response values
            max_points: Maximum number of points to return
            window_size: Size of window for non-maximum suppression

        Returns:
            List of KeyPoint objects
        """
        device = response_map.device
        height, width = response_map.shape

        # Pad response map to handle borders
        pad = window_size // 2
        padded = F.pad(
            response_map.unsqueeze(0).unsqueeze(0),
            (pad, pad, pad, pad),
            mode="constant",
            value=0,
        )

        # Find local maxima
        max_pooled = F.max_pool2d(padded, kernel_size=window_size, stride=1, padding=0)

        # Find pixels that are equal to the local max (i.e., local maxima)
        is_max = response_map == max_pooled.squeeze(0).squeeze(0)

        # Threshold to eliminate weak responses
        min_response = torch.max(response_map) * 0.01
        is_max = torch.logical_and(is_max, response_map > min_response)

        # Get coordinates and response values of detected keypoints
        y_coords, x_coords = torch.nonzero(is_max, as_tuple=True)

        if len(y_coords) == 0:
            return []

        # Get response values
        responses = response_map[y_coords, x_coords]

        # Create a tensor of [x, y, response]
        keypoints_data = torch.stack(
            [x_coords.float(), y_coords.float(), responses], dim=1
        )

        # Sort by response (descending)
        keypoints_data = keypoints_data[
            torch.argsort(keypoints_data[:, 2], descending=True)
        ]

        # Limit to max_points
        if len(keypoints_data) > max_points:
            keypoints_data = keypoints_data[:max_points]

        # Convert to KeyPoint objects
        keypoints = []
        for x, y, response in keypoints_data:
            keypoints.append(
                KeyPoint(
                    x=x.item(),
                    y=y.item(),
                    response=response.item(),
                    size=1.0,  # Default size
                )
            )

        return keypoints
