import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseTracker, Track, TrackStatus


class KLTTracker(BaseTracker):
    """
    Kanade-Lucas-Tomasi (KLT) feature tracker.

    Tracks features across frames using the Lucas-Kanade optical flow algorithm.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize KLT tracker.

        Args:
            config: Configuration dictionary with the following keys:
                - window_size: Size of the window for KLT tracking
                - num_pyramids: Number of pyramid levels
                - num_iterations: Maximum number of iterations per pyramid level
                - min_eigenvalue: Minimum eigenvalue for good features
                - min_displacement: Minimum displacement to consider as tracked
                - max_displacement: Maximum displacement to allow
                - convergence_eps: Convergence threshold
        """
        super().__init__(config)

        # Extract KLT-specific parameters
        self.window_size = self.config.get("window_size", 21)
        self.num_pyramids = self.config.get("num_pyramids", 3)
        self.num_iterations = self.config.get("num_iterations", 30)
        self.min_eigenvalue = self.config.get("min_eigenvalue", 0.01)
        self.min_displacement = self.config.get("min_displacement", 0.1)
        self.max_displacement = self.config.get("max_displacement", 20.0)
        self.convergence_eps = self.config.get("convergence_eps", 0.01)

        # Initialize previous frame and pyramid
        self.prev_frame = None
        self.prev_pyramid = None

    def track_features(
        self,
        current_frame: torch.Tensor,
        prev_frame: Optional[torch.Tensor] = None,
        current_keypoints: Optional[List] = None,
    ) -> Dict[int, Track]:
        """
        Track features between frames using KLT.

        Args:
            current_frame: Current frame tensor (C, H, W)
            prev_frame: Previous frame tensor (C, H, W), if None uses stored previous frame
            current_keypoints: List of keypoints detected in current frame (optional)

        Returns:
            Dictionary of track_id to Track object
        """
        device = current_frame.device

        # Convert to grayscale if needed
        current_gray = self._ensure_grayscale(current_frame)

        # If no previous frame, initialize with current frame
        if self.prev_frame is None:
            self.prev_frame = current_gray
            self.current_frame_idx += 1

            # If keypoints provided, initialize tracks
            if current_keypoints is not None:
                self.initialize_tracks(current_keypoints)

            # Build pyramid for the first frame
            self.prev_pyramid = self._build_pyramid(current_gray)

            return self.tracks

        # Use provided previous frame if available
        if prev_frame is not None:
            self.prev_frame = self._ensure_grayscale(prev_frame)
            self.prev_pyramid = self._build_pyramid(self.prev_frame)

        # Build pyramid for current frame
        current_pyramid = self._build_pyramid(current_gray)

        # Get previous keypoints from tracks
        prev_keypoints = []
        track_ids = []

        for track_id, track in self.tracks.items():
            if track.status == TrackStatus.TRACKED or track.status == TrackStatus.NEW:
                kp = track.get_keypoint()
                prev_keypoints.append((kp.x, kp.y))
                track_ids.append(track_id)

        # If no keypoints to track, initialize from current keypoints if provided
        if len(prev_keypoints) == 0:
            if current_keypoints is not None:
                self.initialize_tracks(current_keypoints)

            self.prev_frame = current_gray
            self.prev_pyramid = current_pyramid
            self.current_frame_idx += 1
            return self.tracks

        # Convert keypoints to tensor
        prev_pts = torch.tensor(prev_keypoints, dtype=torch.float32, device=device)

        # Track points using pyramidal Lucas-Kanade
        new_pts, status = self._track_points_lk(
            self.prev_pyramid, current_pyramid, prev_pts
        )

        # Update tracks based on tracking results
        for i, track_id in enumerate(track_ids):
            if status[i]:
                # Successfully tracked
                x, y = new_pts[i]
                kp = self.tracks[track_id].get_keypoint().clone()
                kp.x = x.item()
                kp.y = y.item()
                self.update_track(track_id, kp)
            else:
                # Failed to track
                self.tracks[track_id].mark_lost()

        # Add new keypoints if provided
        if current_keypoints is not None:
            # Filter out keypoints that are too close to existing tracks
            filtered_keypoints = self._filter_keypoints_by_distance(current_keypoints)

            # Add remaining keypoints as new tracks if we have space
            max_new_tracks = max(0, self.max_tracks - len(self.get_active_tracks()))
            if len(filtered_keypoints) > max_new_tracks:
                filtered_keypoints = filtered_keypoints[:max_new_tracks]

            self.initialize_tracks(filtered_keypoints)

        # Prune old tracks
        self.prune_tracks()

        # Store current frame for next iteration
        self.prev_frame = current_gray
        self.prev_pyramid = current_pyramid
        self.current_frame_idx += 1

        return self.tracks

    def _filter_keypoints_by_distance(self, keypoints: List) -> List:
        """
        Filter out keypoints that are too close to existing tracks.

        Args:
            keypoints: List of keypoints

        Returns:
            Filtered list of keypoints
        """
        if not keypoints:
            return []

        # Get positions of active tracks
        active_tracks = self.get_active_tracks()
        active_positions = [
            track.get_latest_position() for track in active_tracks.values()
        ]

        if not active_positions:
            return keypoints

        filtered_keypoints = []

        for kp in keypoints:
            kp_pos = (kp.x, kp.y)
            too_close = False

            for track_pos in active_positions:
                dx = kp_pos[0] - track_pos[0]
                dy = kp_pos[1] - track_pos[1]
                dist_squared = dx * dx + dy * dy

                if dist_squared < self.min_distance * self.min_distance:
                    too_close = True
                    break

            if not too_close:
                filtered_keypoints.append(kp)

        return filtered_keypoints

    def _ensure_grayscale(self, image: torch.Tensor) -> torch.Tensor:
        """
        Ensure image is grayscale.

        Args:
            image: Input image tensor (C, H, W)

        Returns:
            Grayscale image tensor (1, H, W)
        """
        if image.dim() == 2:
            # Already grayscale, add channel dimension
            return image.unsqueeze(0)
        elif image.dim() == 3 and image.shape[0] == 1:
            # Already grayscale with channel dimension
            return image
        elif image.dim() == 3 and image.shape[0] == 3:
            # Convert RGB to grayscale
            # Use standard RGB to grayscale conversion weights
            weights = torch.tensor([0.299, 0.587, 0.114], device=image.device).view(
                3, 1, 1
            )
            gray = torch.sum(image * weights, dim=0, keepdim=True)
            return gray
        else:
            raise ValueError(f"Unsupported image format with shape {image.shape}")

    def _build_pyramid(self, image: torch.Tensor) -> List[torch.Tensor]:
        """
        Build image pyramid for multi-scale tracking.

        Args:
            image: Input image tensor (1, H, W)

        Returns:
            List of image tensors at different scales
        """
        pyramid = [image]

        for i in range(1, self.num_pyramids):
            prev_level = pyramid[i - 1]
            # Downsample by factor of 2
            next_level = F.avg_pool2d(prev_level, kernel_size=2, stride=2)
            pyramid.append(next_level)

        return pyramid

    def _track_points_lk(
        self,
        prev_pyramid: List[torch.Tensor],
        curr_pyramid: List[torch.Tensor],
        points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Track points using pyramidal Lucas-Kanade.

        Args:
            prev_pyramid: Pyramid of previous frame
            curr_pyramid: Pyramid of current frame
            points: Tensor of points to track (N, 2)

        Returns:
            Tuple of (new_points, status)
        """
        device = points.device
        num_points = points.shape[0]

        # Initialize guesses and status
        guesses = points.clone()
        status = torch.ones(num_points, dtype=torch.bool, device=device)

        # Process from coarsest to finest level
        for level in range(self.num_pyramids - 1, -1, -1):
            # Scale points for this pyramid level
            level_scale = 1.0 / (2**level)
            scaled_points = points * level_scale
            scaled_guesses = guesses * level_scale

            # Get images at this level
            prev_img = prev_pyramid[level]
            curr_img = curr_pyramid[level]

            # Track at this level
            scaled_guesses, status = self._track_points_lk_single_level(
                prev_img, curr_img, scaled_points, scaled_guesses, status
            )

            # Scale back and update guesses
            guesses = scaled_guesses / level_scale

        return guesses, status

    def _track_points_lk_single_level(
        self,
        prev_img: torch.Tensor,
        curr_img: torch.Tensor,
        points: torch.Tensor,
        initial_guess: torch.Tensor,
        status: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Track points using Lucas-Kanade at a single pyramid level.

        Args:
            prev_img: Previous image at this level
            curr_img: Current image at this level
            points: Points from previous image
            initial_guess: Initial guess for new locations
            status: Current tracking status

        Returns:
            Tuple of (new_points, updated_status)
        """
        device = points.device
        height, width = prev_img.shape[1:3]
        num_points = points.shape[0]
        half_window = self.window_size // 2

        # Initialize new points with initial guess
        new_points = initial_guess.clone()

        for i in range(num_points):
            if not status[i]:
                continue

            # Get point coordinates
            x, y = points[i]

            # Check if point is within bounds (with window margin)
            if (
                x < half_window
                or x >= width - half_window
                or y < half_window
                or y >= height - half_window
            ):
                status[i] = False
                continue

            # Extract patch around point in previous image
            prev_patch = self._extract_patch(prev_img, x, y, half_window)

            # Compute spatial gradients of previous patch
            gx, gy = self._compute_gradient(prev_patch)

            # Compute G matrix (2x2 matrix from gradients)
            G = torch.zeros(2, 2, device=device)
            G[0, 0] = torch.sum(gx * gx)
            G[0, 1] = G[1, 0] = torch.sum(gx * gy)
            G[1, 1] = torch.sum(gy * gy)

            # Check if G is invertible (minimum eigenvalue)
            det = G[0, 0] * G[1, 1] - G[0, 1] * G[1, 0]
            trace = G[0, 0] + G[1, 1]

            # Simple check for minimum eigenvalue
            if det < self.min_eigenvalue * trace * trace:
                status[i] = False
                continue

            # Invert G matrix
            try:
                G_inv = torch.inverse(G)
            except Exception:
                status[i] = False
                continue

            # Start iterations from initial guess
            nx, ny = initial_guess[i]

            # LK iterations
            for _ in range(self.num_iterations):
                # Ensure point is within image bounds
                if (
                    nx < half_window
                    or nx >= width - half_window
                    or ny < half_window
                    or ny >= height - half_window
                ):
                    status[i] = False
                    break

                # Extract patch around current guess in current image
                curr_patch = self._extract_patch(curr_img, nx, ny, half_window)

                # Compute residual (difference between patches)
                residual = prev_patch - curr_patch

                # Compute b vector
                bx = torch.sum(gx * residual)
                by = torch.sum(gy * residual)
                b = torch.tensor([bx, by], device=device)

                # Compute delta
                delta = torch.matmul(G_inv, b)

                # Update guess
                nx += delta[0]
                ny += delta[1]

                # Check for convergence
                if torch.norm(delta) < self.convergence_eps:
                    break

            # Store new point location
            new_points[i, 0] = nx
            new_points[i, 1] = ny

            # Check if displacement is valid
            dx = nx - x
            dy = ny - y
            displacement = math.sqrt(dx * dx + dy * dy)

            if (
                displacement < self.min_displacement
                or displacement > self.max_displacement
            ):
                status[i] = False

        return new_points, status

    def _extract_patch(
        self, image: torch.Tensor, x: torch.Tensor, y: torch.Tensor, half_window: int
    ) -> torch.Tensor:
        """
        Extract patch around a point.

        Args:
            image: Image tensor (1, H, W)
            x, y: Center coordinates
            half_window: Half window size

        Returns:
            Patch tensor (window_size, window_size)
        """
        device = image.device
        patch_size = 2 * half_window + 1

        # Convert to integer coordinates
        x_int = int(x.item())
        y_int = int(y.item())

        # Extract patch (using simple indexing for exact integer coordinates)
        patch = image[
            0,
            y_int - half_window : y_int + half_window + 1,
            x_int - half_window : x_int + half_window + 1,
        ]

        # Handle subpixel coordinates with bilinear interpolation
        if not (x == x_int and y == y_int):
            # Create sampling grid for bilinear interpolation
            # Scale coordinates to [-1, 1] range expected by grid_sample
            h, w = image.shape[1:3]

            # Create a grid of coordinates for the patch
            y_coords, x_coords = torch.meshgrid(
                torch.arange(-half_window, half_window + 1, device=device),
                torch.arange(-half_window, half_window + 1, device=device),
            )

            # Add the subpixel offset
            x_coords = x_coords + (x - x_int)
            y_coords = y_coords + (y - y_int)

            # Convert to normalized [-1, 1] coordinates
            x_coords = 2 * (x_int + x_coords) / (w - 1) - 1
            y_coords = 2 * (y_int + y_coords) / (h - 1) - 1

            # Combine to grid
            grid = torch.stack([x_coords, y_coords], dim=2).unsqueeze(0)

            # Sample from image
            patch = (
                F.grid_sample(
                    image.unsqueeze(0), grid, mode="bilinear", align_corners=True
                )
                .squeeze(0)
                .squeeze(0)
            )

        return patch

    def _compute_gradient(
        self, patch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute x and y gradients of a patch.

        Args:
            patch: Patch tensor (window_size, window_size)

        Returns:
            Tuple of (gx, gy) gradients
        """
        device = patch.device

        # Define Sobel filters
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device
        )

        # Apply convolution for gradients
        gx = F.conv2d(
            patch.unsqueeze(0).unsqueeze(0),
            sobel_x.unsqueeze(0).unsqueeze(0),
            padding=1,
        ).squeeze()

        gy = F.conv2d(
            patch.unsqueeze(0).unsqueeze(0),
            sobel_y.unsqueeze(0).unsqueeze(0),
            padding=1,
        ).squeeze()

        return gx, gy
