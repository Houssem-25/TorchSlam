import math
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseFeatureExtractor, KeyPoint


class SIFTFeatureExtractor(BaseFeatureExtractor):
    """SIFT (Scale-Invariant Feature Transform) feature detector and descriptor.

    This is a PyTorch implementation of SIFT that doesn't use OpenCV or neural networks.
    The implementation follows the original Lowe's paper algorithm."""

    def __init__(
        self,
        max_features: int = 1000,
        n_octaves: int = 4,
        n_scales: int = 5,
        sigma: float = 1.6,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10.0,
        descriptor_width: int = 4,
    ):
        """
        Initialize SIFT detector.

        Args:
            max_features: Maximum number of features to detect
            n_octaves: Number of octaves in the scale space
            n_scales: Number of scales per octave
            sigma: Base scale for Gaussian blur
            contrast_threshold: Threshold for contrast filtering
            edge_threshold: Threshold for edge filtering
            descriptor_width: Width of the descriptor grid (descriptor dimension will be width^2 * 8)
        """
        super().__init__(max_features)
        self.n_octaves = n_octaves
        self.n_scales = n_scales
        self.sigma = sigma
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.descriptor_width = descriptor_width

    def extract(self, image: torch.Tensor) -> List[KeyPoint]:
        """
        Extract SIFT keypoints from image.

        Args:
            image: Image tensor

        Returns:
            List of KeyPoint objects
        """
        # Preprocess image
        img = self._preprocess_image(image)
        device = img.device

        # Create scale space
        gaussian_pyramid = self._create_gaussian_pyramid(img)
        dog_pyramid = self._create_difference_of_gaussian_pyramid(gaussian_pyramid)

        # Find keypoint candidates in DoG pyramid
        keypoints = []

        for octave in range(len(dog_pyramid)):
            for scale in range(
                1, len(dog_pyramid[octave]) - 1
            ):  # Skip first and last scale in each octave
                # Local extrema detection
                extrema = self._detect_local_extrema(
                    dog_pyramid[octave][scale - 1 : scale + 2]
                )

                # Refine keypoint locations and filter by contrast and edge response
                refined_keypoints = self._refine_keypoints(
                    dog_pyramid[octave], scale, extrema, octave
                )

                # Compute orientations
                for kp in refined_keypoints:
                    # Get corresponding Gaussian blurred image
                    gauss_image = gaussian_pyramid[octave][scale]

                    # Compute gradient magnitude and orientation
                    orientations = self._compute_orientations(gauss_image, kp)

                    # Create a keypoint for each significant orientation
                    for angle in orientations:
                        new_kp = KeyPoint(
                            x=kp.x
                            * (2**octave),  # Scale back to original image coordinates
                            y=kp.y * (2**octave),
                            response=kp.response,
                            size=kp.size * (2**octave),
                            angle=angle,
                            octave=octave,
                        )
                        keypoints.append(new_kp)

        # Sort by response and limit to max_features
        keypoints.sort(key=lambda x: x.response, reverse=True)
        if len(keypoints) > self.max_features:
            keypoints = keypoints[: self.max_features]

        return keypoints

    def compute_descriptors(
        self, image: torch.Tensor, keypoints: List[KeyPoint]
    ) -> torch.Tensor:
        """
        Compute SIFT descriptors for keypoints.

        Args:
            image: Image tensor
            keypoints: List of KeyPoint objects

        Returns:
            Tensor of descriptors (n_keypoints, descriptor_width^2 * 8)
        """
        # Preprocess image
        img = self._preprocess_image(image)
        device = img.device
        height, width = img.shape[1:]

        # Compute gradients
        dx, dy = self._compute_image_gradients(img)

        # Calculate gradient magnitude and orientation
        gradient_magnitude = torch.sqrt(dx.pow(2) + dy.pow(2))
        gradient_orientation = (
            torch.atan2(dy, dx) * 180.0 / math.pi
        )  # Convert to degrees
        gradient_orientation = torch.where(
            gradient_orientation < 0, gradient_orientation + 360.0, gradient_orientation
        )

        # Create empty descriptor tensor
        descriptor_size = self.descriptor_width * self.descriptor_width * 8
        descriptors = torch.zeros(
            (len(keypoints), descriptor_size), dtype=torch.float32, device=device
        )

        # Compute descriptor for each keypoint
        for i, kp in enumerate(keypoints):
            # Determine scale factor
            scale_factor = 2**kp.octave

            # Calculate window size based on keypoint scale
            window_radius = int(kp.size * 1.5 * scale_factor)

            # Skip keypoints too close to the edge
            if (
                kp.x < window_radius
                or kp.x >= width - window_radius
                or kp.y < window_radius
                or kp.y >= height - window_radius
            ):
                continue

            # Get angle in radians
            angle_rad = kp.angle * math.pi / 180.0
            cos_angle = math.cos(angle_rad)
            sin_angle = math.sin(angle_rad)

            # Compute descriptor
            desc = self._compute_sift_descriptor(
                gradient_magnitude.squeeze(0),
                gradient_orientation.squeeze(0),
                kp.x,
                kp.y,
                kp.size,
                angle_rad,
            )

            # Normalize descriptor
            norm = torch.norm(desc)
            if norm > 1e-5:
                desc = desc / norm

            # Clamp values to 0.2 to reduce influence of large gradient magnitudes
            desc = torch.clamp(desc, 0, 0.2)

            # Normalize again
            norm = torch.norm(desc)
            if norm > 1e-5:
                desc = desc / norm

            descriptors[i] = desc

        return descriptor
