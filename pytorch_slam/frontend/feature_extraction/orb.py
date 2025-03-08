import math

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseFeatureExtractor


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

        all_keypoints = []

        # Create image pyramid
        for level in range(self.n_scale_levels):
            # Compute scale
            current_scale = 1.0 / (self.scale_factor**level)

            # Resize image to current scale (skip resize for first level)
            if level == 0:
                scaled_img = img
            else:
                new_h = int(img.shape[1] * current_scale)
                new_w = int(img.shape[2] * current_scale)
                scaled_img = F.interpolate(
                    img, size=(new_h, new_w), mode="bilinear", align_corners=False
                )

            # Detect FAST keypoints
            keypoints = self._extract_fast(scaled_img)

            if len(keypoints) == 0:
                continue

            # Compute orientation for each keypoint
            for kp in keypoints:
                # Convert back to original image coordinates
                kp.x /= current_scale
                kp.y /= current_scale
                kp.size /= current_scale

                # Compute orientation using intensity centroid method
                x, y = int(kp.x * current_scale), int(kp.y * current_scale)

                # Ensure we're within bounds
                if (
                    x < 4
                    or y < 4
                    or x >= scaled_img.shape[2] - 4
                    or y >= scaled_img.shape[1] - 4
                ):
                    kp.angle = 0.0
                    continue

                # Extract patch around keypoint (9x9)
                patch = scaled_img[0, y - 4 : y + 5, x - 4 : x + 5]

                # Compute moments
                m01 = 0.0
                m10 = 0.0

                for i in range(9):
                    for j in range(9):
                        pixel_value = patch[i, j].item()
                        m01 += i * pixel_value
                        m10 += j * pixel_value

                # Compute centroid
                if m01 == 0 or m10 == 0:
                    kp.angle = 0.0
                else:
                    centroid_x = m10 / (m01 + m10)
                    centroid_y = m01 / (m01 + m10)

                    # Calculate orientation (in degrees)
                    angle = (
                        math.atan2(centroid_y - 4.0, centroid_x - 4.0) * 180.0 / math.pi
                    )
                    if angle < 0:
                        angle += 360.0

                    kp.angle = angle

                # Store scale level in octave
                kp.octave = level

                all_keypoints.append(kp)

        # Sort keypoints by response and limit to max_features
        all_keypoints.sort(key=lambda x: x.response, reverse=True)
        if len(all_keypoints) > self.max_features:
            all_keypoints = all_keypoints[: self.max_features]

        return all_keypoints

    def compute_descriptors(
        self, image: torch.Tensor, keypoints: List[KeyPoint]
    ) -> torch.Tensor:
        """
        Compute ORB descriptors for keypoints.

        Args:
            image: Image tensor
            keypoints: List of KeyPoint objects

        Returns:
            Tensor of binary descriptors (n_keypoints, n_points//8)
        """
        # Ensure image is grayscale and in the right format
        if image.dim() == 3 and image.shape[0] > 1:
            # Convert to grayscale - simple average
            image = image.mean(dim=0, keepdim=True)
        elif image.dim() == 2:
            image = image.unsqueeze(0)

        device = image.device
        height, width = image.shape[1:]

        # Pad image to handle keypoints near edges
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

        # Move sampling patterns to device
        sampling_patterns = self.sampling_patterns.to(device)

        descriptors = []

        for kp in keypoints:
            # Skip keypoints too close to the edge
            if (
                kp.x < self.half_patch_size
                or kp.x >= width - self.half_patch_size
                or kp.y < self.half_patch_size
                or kp.y >= height - self.half_patch_size
            ):
                # Add a zero descriptor
                descriptors.append(
                    torch.zeros((self.n_points // 8,), dtype=torch.uint8, device=device)
                )
                continue

            # Get angle in radians
            angle_rad = kp.angle * math.pi / 180.0 if kp.angle >= 0 else 0.0

            # Compute sine and cosine for rotation
            cos_theta = math.cos(angle_rad)
            sin_theta = math.sin(angle_rad)

            # Rotate sampling patterns according to keypoint orientation
            rotated_patterns = torch.zeros_like(sampling_patterns)

            # Rotate first point
            rotated_patterns[:, 0] = (
                cos_theta * sampling_patterns[:, 0]
                - sin_theta * sampling_patterns[:, 1]
            )
            rotated_patterns[:, 1] = (
                sin_theta * sampling_patterns[:, 0]
                + cos_theta * sampling_patterns[:, 1]
            )

            # Rotate second point
            rotated_patterns[:, 2] = (
                cos_theta * sampling_patterns[:, 2]
                - sin_theta * sampling_patterns[:, 3]
            )
            rotated_patterns[:, 3] = (
                sin_theta * sampling_patterns[:, 2]
                + cos_theta * sampling_patterns[:, 3]
            )

            # Round to nearest integer
            rotated_patterns = torch.round(rotated_patterns).long()

            # Compute descriptor
            descriptor_bits = torch.zeros(
                self.n_points, dtype=torch.bool, device=device
            )

            for i in range(self.n_points):
                # Get coordinates of the sampling points
                x1, y1, x2, y2 = rotated_patterns[i]

                # Get pixel values at the sampling points
                # Add half_patch_size to adjust for padding
                px1 = kp.x + x1 + self.half_patch_size
                py1 = kp.y + y1 + self.half_patch_size
                px2 = kp.x + x2 + self.half_patch_size
                py2 = kp.y + y2 + self.half_patch_size

                # Ensure the coordinates are valid
                px1 = torch.clamp(torch.tensor(px1), 0, padded.shape[2] - 1).long()
                py1 = torch.clamp(torch.tensor(py1), 0, padded.shape[1] - 1).long()
                px2 = torch.clamp(torch.tensor(px2), 0, padded.shape[2] - 1).long()
                py2 = torch.clamp(torch.tensor(py2), 0, padded.shape[1] - 1).long()

                # Compare pixel values for the binary test
                descriptor_bits[i] = padded[0, py1, px1] < padded[0, py2, px2]

            # Pack bits into bytes
            descriptor_bytes = torch.zeros(
                self.n_points // 8, dtype=torch.uint8, device=device
            )

            for i in range(self.n_points // 8):
                for j in range(8):
                    if descriptor_bits[i * 8 + j]:
                        descriptor_bytes[i] |= 1 << j

            descriptors.append(descriptor_bytes)

        # Stack all descriptors
        if descriptors:
            return torch.stack(descriptors)
        else:
            return torch.zeros(
                (0, self.n_points // 8), dtype=torch.uint8, device=device
            )

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
        FAST (Features from Accelerated Segment Test) corner detector.

        Args:
            image: Image tensor
            n_consistent: Minimum number of consistent pixels to detect as corner (9-16)

        Returns:
            List of KeyPoint objects
        """
        # Preprocess image
        img = image

        device = img.device
        height, width = img.shape[1:]

        # Define 16-point Bresenham circle offsets
        # This array represents the 16 pixels around the pixel at (0,0) in a circle
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
            dtype=torch.int,
            device=device,
        )

        # Pad image to handle borders
        padded = F.pad(img, (3, 3, 3, 3), mode="reflect")

        # Initialize corner score map
        corner_scores = torch.zeros((height, width), dtype=torch.float32, device=device)

        # For each pixel in the image (excluding borders that would go out of bounds)
        for y in range(height):
            for x in range(width):
                # Get pixel value and threshold bounds
                center_val = img[0, y, x]
                lower_bound = center_val - self.fast_threshold
                upper_bound = center_val + self.fast_threshold

                # Get values of the 16 pixels in the Bresenham circle
                circle_values = torch.zeros(16, dtype=torch.float32, device=device)
                for i, (dx, dy) in enumerate(circle_offsets):
                    circle_values[i] = padded[0, y + dy + 3, x + dx + 3]

                # Check if we have consecutive pixels that are all brighter or all darker
                is_brighter = circle_values > upper_bound
                is_darker = circle_values < lower_bound

                # Using a rolling window of size n_consistent to check for consecutive pixels
                # This is a simplified version that works for 9-point FAST
                max_consistent_brighter = 0
                max_consistent_darker = 0

                for start in range(16):
                    consistent_brighter = 0
                    consistent_darker = 0
                    for offset in range(16):
                        idx = (start + offset) % 16
                        if is_brighter[idx]:
                            consistent_brighter += 1
                            consistent_darker = 0
                        elif is_darker[idx]:
                            consistent_darker += 1
                            consistent_brighter = 0
                        else:  # Not consistent with either condition
                            consistent_brighter = 0
                            consistent_darker = 0

                        max_consistent_brighter = max(
                            max_consistent_brighter, consistent_brighter
                        )
                        max_consistent_darker = max(
                            max_consistent_darker, consistent_darker
                        )

                # Compute corner score (very basic - just sum the absolute differences)
                if (
                    max_consistent_brighter >= n_consistent
                    or max_consistent_darker >= n_consistent
                ):
                    score = torch.sum(torch.abs(circle_values - center_val))
                    corner_scores[y, x] = score

        # Non-maximum suppression
        keypoints = self._non_maximum_suppression(corner_scores, self.max_features)

        return keypoints
