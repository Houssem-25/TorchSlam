import math
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseFeatureExtractor, KeyPoint


class ORBFeatureExtractor(BaseFeatureExtractor):
    """ORB (Oriented FAST and Rotated BRIEF) feature detector and descriptor.

    This is a PyTorch implementation of ORB that doesn't use OpenCV or neural networks.
    It implements the FAST corner detector with orientation computation and
    a rotation-aware BRIEF descriptor."""

    def __init__(
        self,
        max_features: int = 1000,
        n_scale_levels: int = 8,
        scale_factor: float = 1.2,
        fast_threshold: float = 0.1,
        n_points: int = 256,
        patch_size: int = 31,
    ):
        """
        Initialize ORB detector.

        Args:
            max_features: Maximum number of features to detect
            n_scale_levels: Number of scale levels in the image pyramid
            scale_factor: Scale factor between levels in the image pyramid
            fast_threshold: Threshold for FAST corner detection
            n_points: Number of binary tests in the descriptor
            patch_size: Size of patch to extract around each keypoint
        """
        super().__init__(max_features)
        self.n_scale_levels = n_scale_levels
        self.scale_factor = scale_factor
        self.fast_threshold = fast_threshold
        self.n_points = n_points
        self.patch_size = patch_size
        self.half_patch_size = patch_size // 2

        # Generate sampling patterns for the descriptor
        self.sampling_patterns = self._generate_sampling_patterns()

    def extract(self, image: torch.Tensor) -> List[KeyPoint]:
        """
        Extract ORB keypoints from image.

        Args:
            image: Image tensor

        Returns:
            List of KeyPoint objects
        """
        # Preprocess image
        img = self._preprocess_image(image)
        device = img.device
        # print(f"[DEBUG] Initial img shape: {img.shape}") # Debug print removed

        all_keypoints = []

        # Create image pyramid
        current_img = img  # Start with the preprocessed image
        for level in range(self.n_scale_levels):
            # Compute scale
            current_scale = 1.0 / (self.scale_factor**level)

            # Resize image to current scale (skip resize for first level)
            if level == 0:
                scaled_img = current_img
            else:
                # Calculate target size based on the *previous* level's image
                prev_h, prev_w = current_img.shape[2:]  # Use indices 2 and 3 for H, W
                new_h = int(prev_h / self.scale_factor)  # Scale down from previous
                new_w = int(prev_w / self.scale_factor)
                # Ensure size is at least 1x1
                new_h = max(1, new_h)
                new_w = max(1, new_w)

                # Interpolate the *previous* level's image
                # print(f"[DEBUG] Level {level}: Interpolating current_img shape: {current_img.shape} to size ({new_h}, {new_w})") # Debug print removed
                scaled_img = F.interpolate(
                    current_img,
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                )
                current_img = scaled_img  # Update current_img for the next level

            # Detect FAST keypoints
            # Ensure scaled_img is (1, H, W) for _extract_fast
            if scaled_img.dim() == 2:  # Should not happen if preprocess is correct
                scaled_img = scaled_img.unsqueeze(0)
            elif scaled_img.dim() == 3 and scaled_img.shape[0] != 1:
                # Should also not happen, maybe take the first channel?
                scaled_img = scaled_img[0].unsqueeze(0)

            # Ensure scaled_img is (1, C, H, W) for _extract_fast
            # This check might be redundant now, but kept for safety
            if (
                scaled_img.dim() != 4
                or scaled_img.shape[0] != 1
                or scaled_img.shape[1] != 1
            ):
                # Attempt to reshape if possible, otherwise raise error
                try:
                    scaled_img = scaled_img.view(
                        1, 1, scaled_img.shape[-2], scaled_img.shape[-1]
                    )
                except Exception as e:
                    raise ValueError(
                        f"Unexpected shape for scaled_img: {scaled_img.shape}. Error: {e}"
                    )

            keypoints = self._extract_fast(scaled_img)

            if len(keypoints) == 0:
                continue

            # Compute orientation for each keypoint
            # --- Optimized Orientation Calculation ---
            kp_coords = torch.tensor(
                [(kp.x * current_scale, kp.y * current_scale) for kp in keypoints],
                device=device,
            )
            kp_coords_int = torch.round(kp_coords).long()

            # Create grid for patch indices
            patch_radius = 4
            patch_size = 2 * patch_radius + 1
            iy, ix = torch.meshgrid(
                torch.arange(-patch_radius, patch_radius + 1, device=device),
                torch.arange(-patch_radius, patch_radius + 1, device=device),
                indexing="ij",
            )

            # Calculate coordinates for all patches
            all_patch_y = kp_coords_int[:, 1, None, None] + iy[None, :, :]
            all_patch_x = kp_coords_int[:, 0, None, None] + ix[None, :, :]

            # Clamp coordinates to be within image bounds
            # Use shape[2] for H and shape[3] for W due to (N, C, H, W) format
            h, w = scaled_img.shape[2:]
            all_patch_y = torch.clamp(all_patch_y, 0, h - 1)
            all_patch_x = torch.clamp(all_patch_x, 0, w - 1)

            # Extract all patches using advanced indexing from the first batch and channel
            all_patches = scaled_img[
                0, 0, all_patch_y, all_patch_x
            ]  # Shape: (num_kp, patch_size, patch_size)

            # Compute moments using tensor operations
            # ix and iy already represent the coordinates relative to the center (-4 to +4)
            # For moment calculation, we need coordinates from 0 to 8
            moment_iy = iy + patch_radius  # Shape: (patch_size, patch_size)
            moment_ix = ix + patch_radius  # Shape: (patch_size, patch_size)

            m01 = torch.sum(all_patches * moment_iy[None, :, :], dim=(1, 2))
            m10 = torch.sum(all_patches * moment_ix[None, :, :], dim=(1, 2))

            # Avoid division by zero or near-zero sums
            denominator = m01 + m10
            valid_moment = denominator.abs() > 1e-6

            # Compute centroid coordinates (relative to patch center, i.e., -4 to 4)
            # Centroid relative to top-left of patch (0 to 8)
            centroid_x_rel_patch = torch.zeros_like(m10)
            centroid_y_rel_patch = torch.zeros_like(m01)
            centroid_x_rel_patch[valid_moment] = (
                m10[valid_moment] / denominator[valid_moment]
            )
            centroid_y_rel_patch[valid_moment] = (
                m01[valid_moment] / denominator[valid_moment]
            )

            # Centroid relative to patch center (-4 to 4)
            centroid_x = centroid_x_rel_patch - patch_radius
            centroid_y = centroid_y_rel_patch - patch_radius

            # Calculate orientation angle in degrees
            angles = torch.atan2(centroid_y, centroid_x) * 180.0 / math.pi
            angles = torch.where(angles < 0, angles + 360.0, angles)
            angles[~valid_moment] = 0.0  # Set angle to 0 for invalid moments

            # Update keypoints with calculated angles and scale level
            for i, kp in enumerate(keypoints):
                # Convert back to original image coordinates (already done before loop)
                # kp.x /= current_scale
                # kp.y /= current_scale
                # kp.size /= current_scale

                # Check bounds (can potentially be vectorized too, but might be less clear)
                x_int, y_int = kp_coords_int[i].tolist()
                if (
                    x_int < patch_radius
                    or y_int < patch_radius
                    or x_int >= w - patch_radius
                    or y_int >= h - patch_radius
                ):
                    kp.angle = 0.0
                else:
                    kp.angle = angles[i].item()

                kp.octave = level
                all_keypoints.append(kp)
            # --- End Optimized Orientation Calculation ---

            # Original loop:
            # for kp in keypoints:
            #     # Convert back to original image coordinates
            #     kp.x /= current_scale
            #     kp.y /= current_scale
            #     kp.size /= current_scale
            #
            #     # Compute orientation using intensity centroid method
            #     x, y = int(kp.x * current_scale), int(kp.y * current_scale)
            #
            #     # Ensure we're within bounds
            #     if (
            #         x < 4
            #         or y < 4
            #         or x >= scaled_img.shape[2] - 4
            #         or y >= scaled_img.shape[1] - 4
            #     ):
            #         kp.angle = 0.0
            #         continue
            #
            #     # Extract patch around keypoint (9x9)
            #     patch = scaled_img[0, y - 4 : y + 5, x - 4 : x + 5]
            #
            #     # Compute moments
            #     m01 = 0.0
            #     m10 = 0.0
            #
            #     for i in range(9):
            #         for j in range(9):
            #             pixel_value = patch[i, j].item()
            #             m01 += i * pixel_value
            #             m10 += j * pixel_value
            #
            #     # Compute centroid
            #     if m01 == 0 or m10 == 0:
            #         kp.angle = 0.0
            #     else:
            #         centroid_x = m10 / (m01 + m10)
            #         centroid_y = m01 / (m01 + m10)
            #
            #         # Calculate orientation (in degrees)
            #         angle = (
            #             math.atan2(centroid_y - 4.0, centroid_x - 4.0) * 180.0 / math.pi
            #         )
            #         if angle < 0:
            #             angle += 360.0
            #
            #         kp.angle = angle
            #
            #     # Store scale level in octave
            #     kp.octave = level
            #
            #     all_keypoints.append(kp)

        # Sort keypoints by response and limit to max_features
        all_keypoints.sort(key=lambda x: x.response, reverse=True)
        if len(all_keypoints) > self.max_features:
            all_keypoints = all_keypoints[: self.max_features]

        return all_keypoints

    def compute_descriptors(
        self, image: torch.Tensor, keypoints: List[KeyPoint]
    ) -> torch.Tensor:
        """
        Compute ORB descriptors for keypoints (Vectorized Implementation).

        Args:
            image: PyTorch tensor (preprocessed, likely 1, 1, H, W)
            keypoints: List of KeyPoint objects

        Returns:
            Tensor of descriptors (num_keypoints, n_points // 8) dtype=torch.uint8
        """
        if not keypoints:
            return torch.zeros(
                (0, self.n_points // 8), dtype=torch.uint8, device=image.device
            )

        image = self._preprocess_image(image)
        device = image.device
        height, width = image.shape[2:]  # Now image is guaranteed to be 4D

        # Pad image for sampling (using reflect padding)
        # Image is already preprocessed (grayscale, float, normalized)
        padded = F.pad(
            image,
            (
                self.half_patch_size,
                self.half_patch_size,
                self.half_patch_size,
                self.half_patch_size,
            ),
            mode="reflect",
        )
        padded_h, padded_w = padded.shape[2:]

        # Move sampling patterns to device
        sampling_patterns = self.sampling_patterns.to(device)  # Shape: (n_points, 4)

        # Extract keypoint data into tensors
        kp_data = torch.tensor(
            [(kp.x, kp.y, kp.angle) for kp in keypoints],
            dtype=torch.float32,
            device=device,
        )  # Shape: (num_keypoints, 3)
        kp_x = kp_data[:, 0]
        kp_y = kp_data[:, 1]
        kp_angles = kp_data[:, 2]

        # --- Identify valid keypoints (not too close to the border) ---
        valid_mask = (
            (kp_x >= self.half_patch_size)
            & (kp_x < width - self.half_patch_size)
            & (kp_y >= self.half_patch_size)
            & (kp_y < height - self.half_patch_size)
        )
        valid_indices = torch.where(valid_mask)[0]
        num_valid_kp = len(valid_indices)

        # Initialize the result tensor with zeros
        all_descriptors = torch.zeros(
            (len(keypoints), self.n_points // 8), dtype=torch.uint8, device=device
        )

        if num_valid_kp == 0:
            return all_descriptors  # All keypoints were near the edge

        # --- Process only valid keypoints ---
        valid_kp_x = kp_x[valid_indices]  # Shape: (N,)
        valid_kp_y = kp_y[valid_indices]  # Shape: (N,)
        valid_kp_angles = kp_angles[valid_indices]  # Shape: (N,)

        # Calculate angles in radians (handle negative angles)
        angles_rad = torch.where(
            valid_kp_angles >= 0, valid_kp_angles * (math.pi / 180.0), 0.0
        )  # Shape: (N,)

        # Compute sine and cosine for rotation
        cos_theta = torch.cos(angles_rad)  # Shape: (N,)
        sin_theta = torch.sin(angles_rad)  # Shape: (N,)

        # --- Batch Rotate Sampling Patterns ---
        # sampling_patterns shape: (n_points, 4) -> (x1, y1, x2, y2)
        # Expand tensors for broadcasting:
        # cos/sin: (N,) -> (N, 1)
        # patterns: (n_points, 4) -> (1, n_points, 4)
        cos_t = cos_theta[:, None]
        sin_t = sin_theta[:, None]
        sp = sampling_patterns[None, :, :]  # Shape (1, n_points, 4)

        # Rotate points (x', y') = (x*cos - y*sin, x*sin + y*cos)
        # Shape: (N, n_points)
        r_x1 = cos_t * sp[:, :, 0] - sin_t * sp[:, :, 1]
        r_y1 = sin_t * sp[:, :, 0] + cos_t * sp[:, :, 1]
        r_x2 = cos_t * sp[:, :, 2] - sin_t * sp[:, :, 3]
        r_y2 = sin_t * sp[:, :, 2] + cos_t * sp[:, :, 3]

        # --- Batch Calculate Absolute Pixel Coordinates ---
        # Expand kp coordinates: (N,) -> (N, 1)
        # Add rotated offsets (N, n_points) and padding offset
        # Shape: (N, n_points)
        px1 = valid_kp_x[:, None] + r_x1 + self.half_patch_size
        py1 = valid_kp_y[:, None] + r_y1 + self.half_patch_size
        px2 = valid_kp_x[:, None] + r_x2 + self.half_patch_size
        py2 = valid_kp_y[:, None] + r_y2 + self.half_patch_size

        # --- Batch Clamp Coordinates ---
        # Shape: (N, n_points)
        px1 = torch.clamp(px1, 0, padded_w - 1).long()
        py1 = torch.clamp(py1, 0, padded_h - 1).long()
        px2 = torch.clamp(px2, 0, padded_w - 1).long()
        py2 = torch.clamp(py2, 0, padded_h - 1).long()

        # --- Batch Sample Pixel Values ---
        # Use advanced indexing on the padded image (1, 1, H_pad, W_pad)
        # Shape: (N, n_points)
        vals1 = padded[0, 0, py1, px1]
        vals2 = padded[0, 0, py2, px2]

        # --- Batch Compare ---
        # Shape: (N, n_points), dtype=torch.bool
        descriptor_bits = vals1 < vals2

        # --- Batch Pack Bits into Bytes ---
        # Reshape bits: (N, n_points) -> (N, n_points // 8, 8)
        desc_bits_reshaped = descriptor_bits.view(num_valid_kp, self.n_points // 8, 8)

        # Create powers of 2 for packing: [1, 2, 4, ..., 128]
        powers = 2 ** torch.arange(8, dtype=torch.uint8, device=device)  # Shape: (8,)

        # Pack bits using matrix multiplication or summation
        # Convert bits to uint8 and multiply by powers, then sum
        # Shape: (N, n_points // 8)
        valid_descriptors = torch.sum(
            desc_bits_reshaped.byte() * powers, dim=2, dtype=torch.uint8
        )

        # --- Place valid descriptors into the final tensor ---
        all_descriptors[valid_indices] = valid_descriptors

        return all_descriptors

    def _generate_sampling_patterns(self) -> torch.Tensor:
        """
        Generate sampling patterns for ORB descriptor.
        These are the pixel pairs to compare in the BRIEF descriptor.

        Returns:
            Tensor of shape (n_points, 4) containing (x1, y1, x2, y2) coordinates
        """
        # Use numpy for random number generation
        np.random.seed(1234)  # Fixed seed for reproducibility

        # Generate sampling patterns using Gaussian distribution
        # For a proper implementation, the sampling points should be learned from data
        # or use the original ORB sampling pattern, but for simplicity we use random sampling

        # We generate coordinates within a circle of radius half_patch_size
        points = []
        for _ in range(self.n_points * 2):  # Generate twice as many points as needed
            # Generate point in a circle
            theta = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, self.half_patch_size)
            x = int(r * np.cos(theta))
            y = int(r * np.sin(theta))

            # Ensure the point is within the patch
            if abs(x) <= self.half_patch_size and abs(y) <= self.half_patch_size:
                points.append((x, y))

        # Ensure we have enough points
        if len(points) < self.n_points * 2:
            # Fill with points on a grid if we don't have enough
            for x in range(-self.half_patch_size, self.half_patch_size + 1, 2):
                for y in range(-self.half_patch_size, self.half_patch_size + 1, 2):
                    points.append((x, y))

        # Take only the points we need and pair them
        points = points[: self.n_points * 2]

        # Create sampling pairs
        pairs = []
        for i in range(0, self.n_points * 2, 2):
            if i + 1 < len(points):
                pairs.append(
                    (points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
                )

        # Convert to tensor
        sampling_patterns = torch.tensor(pairs, dtype=torch.float32)

        return sampling_patterns

    def _extract_fast(
        self, image: torch.Tensor, n_consistent: int = 9
    ) -> List[KeyPoint]:
        """
        Vectorized FAST (Features from Accelerated Segment Test) corner detector.

        Args:
            image: Image tensor (1, 1, H, W)
            n_consistent: Minimum number of consistent pixels to detect as corner (9-16)

        Returns:
            List of KeyPoint objects
        """
        device = image.device
        # Input image is now (1, 1, H, W)
        height, width = image.shape[2:]

        # Define 16-point Bresenham circle offsets
        circle_offsets = torch.tensor(
            [
                [0, 3],
                [1, 3],
                [2, 2],
                [3, 1],
                [3, 0],
                [3, -1],
                [2, -2],
                [1, -3],
                [0, -3],
                [-1, -3],
                [-2, -2],
                [-3, -1],
                [-3, 0],
                [-3, 1],
                [-2, 2],
                [-1, 3],
            ],
            dtype=torch.long,
            device=device,
        )

        # Pad image to handle borders
        pad = 3
        padded = F.pad(image, (pad, pad, pad, pad), mode="reflect")

        # Get center pixel values for the valid image area
        center_pixels = image  # Shape: (1, 1, H, W)

        # Calculate coordinates for the 16 neighbors for all center pixels
        # Create base coordinates for the center pixels
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )

        # Add offsets (broadcast) and adjust for padding
        # Shape: (16, H, W)
        neighbor_y = y_coords[None, :, :] + circle_offsets[:, 1, None, None] + pad
        neighbor_x = x_coords[None, :, :] + circle_offsets[:, 0, None, None] + pad

        # Gather neighbor pixel values using advanced indexing
        # Shape: (16, H, W)
        neighbor_pixels = padded[0, 0, neighbor_y, neighbor_x]

        # Calculate thresholds for all pixels
        lower_bounds = center_pixels - self.fast_threshold  # Shape (1, 1, H, W)
        upper_bounds = center_pixels + self.fast_threshold  # Shape (1, 1, H, W)

        # Check brightness/darkness conditions for all neighbors
        # Compare (16, H, W) with (1, 1, H, W) using broadcasting
        is_brighter = neighbor_pixels > upper_bounds  # Remove squeeze
        is_darker = neighbor_pixels < lower_bounds  # Remove squeeze

        # --- Vectorized check for n_consistent consecutive pixels ---
        # Double the condition arrays for wrap-around check
        # Shape is likely (1, 16, H, W) -> cat -> (1, 32, H, W)
        is_brighter_double = torch.cat([is_brighter, is_brighter], dim=1)
        is_darker_double = torch.cat([is_darker, is_darker], dim=1)

        # Use convolution with a kernel of ones to count consecutive pixels
        kernel = torch.ones((1, 1, n_consistent), device=device, dtype=torch.float32)

        # Treat is_brighter/is_darker as (Batch, Channel, Sequence)
        # Reshape (1, 32, H, W) -> (H*W, 1, 32)
        h_w = height * width
        # Permute to (H, W, 1, 32) then reshape
        brighter_flat = is_brighter_double.permute(2, 3, 0, 1).reshape(h_w, 1, 32)
        darker_flat = is_darker_double.permute(2, 3, 0, 1).reshape(h_w, 1, 32)

        brighter_seq = F.conv1d(brighter_flat.float(), kernel, padding=0)
        darker_seq = F.conv1d(darker_flat.float(), kernel, padding=0)

        # Check if any convolution result equals n_consistent
        # Shape: (H*W,)
        has_consecutive_brighter = torch.any(
            brighter_seq >= n_consistent, dim=2
        ).squeeze()
        has_consecutive_darker = torch.any(darker_seq >= n_consistent, dim=2).squeeze()

        # Combine conditions: a pixel is a corner if it has either streak
        # Shape: (H, W)
        is_corner = (has_consecutive_brighter | has_consecutive_darker).view(
            height, width
        )
        # --- End Vectorized Check ---

        # Compute corner score (sum of absolute differences) only for potential corners
        corner_scores = torch.zeros((height, width), dtype=torch.float32, device=device)
        # Compare (16, H, W) with (H, W) after squeezing center_pixels
        abs_diff = torch.abs(
            neighbor_pixels - center_pixels.squeeze()
        )  # Squeeze center_pixels
        summed_scores = torch.sum(abs_diff, dim=0)  # Calculate sum first (H, W)
        # Assign scores only where is_corner is True (two-step)
        scores_to_assign = summed_scores[is_corner]  # Extract relevant scores
        corner_scores[is_corner] = scores_to_assign  # Assign extracted scores

        # Non-maximum suppression
        keypoints = self._non_maximum_suppression(corner_scores, self.max_features)

        return keypoints
