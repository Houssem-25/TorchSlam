import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ..feature_extraction import FeatureMatcher, MatchingMethod
from .base import BaseOdometry, FramePose, OdometryStatus


class PnPOdometry(BaseOdometry):
    """
    Perspective-n-Point (PnP) based visual odometry.

    Estimates camera motion using 3D-2D point correspondences.
    Can work with monocular, stereo, or RGB-D data.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize PnP odometry estimator.

        Args:
            config: Configuration dictionary with the following keys:
                - camera_matrix: Intrinsic camera matrix (3x3)
                - dist_coeffs: Distortion coefficients (optional)
                - min_inliers: Minimum number of inliers for reliable estimation
                - ransac_threshold: RANSAC threshold for outlier rejection
                - confidence: RANSAC confidence level
                - method: PnP method ('epnp', 'dls', 'p3p', etc.)
                - use_extrinsic_guess: Whether to use previous pose as initial guess
        """
        super().__init__(config)

        # Extract configuration
        self.camera_matrix = self.config.get("camera_matrix")
        if self.camera_matrix is not None:
            self.camera_matrix = torch.tensor(self.camera_matrix, dtype=torch.float32)

        self.dist_coeffs = self.config.get("dist_coeffs")
        if self.dist_coeffs is not None:
            self.dist_coeffs = torch.tensor(self.dist_coeffs, dtype=torch.float32)

        self.min_inliers = self.config.get("min_inliers", 10)
        self.ransac_threshold = self.config.get("ransac_threshold", 2.0)
        self.confidence = self.config.get("confidence", 0.99)
        self.method = self.config.get("method", "epnp")
        self.use_extrinsic_guess = self.config.get("use_extrinsic_guess", False)

        # Initialize feature matcher (pass relevant config options)
        # Use defaults suitable for PnP context if not specified in config
        self.matcher = FeatureMatcher(
            method=self.config.get("matcher_method", MatchingMethod.RATIO_TEST),
            ratio_threshold=self.config.get("matcher_ratio_threshold", 0.7),
            cross_check=self.config.get("matcher_cross_check", True),
            max_distance=self.config.get("matcher_max_distance", float("inf")),
            num_neighbors=self.config.get("matcher_num_neighbors", 2),
        )

        # Store previous frame data
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_points_3d = None

    def process_frame(self, data: Dict[str, Any]) -> FramePose:
        """
        Process a new frame to estimate odometry using PnP.

        Args:
            data: Dictionary containing:
                - keypoints: List of keypoints
                - descriptors: Tensor of descriptors
                - points_3d: Tensor of 3D points corresponding to keypoints (N, 3)
                - timestamp: Optional timestamp

        Returns:
            Estimated pose for the current frame
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Extract data
        keypoints = data.get("keypoints")
        descriptors = data.get("descriptors")
        points_3d = data.get("points_3d")
        timestamp = data.get("timestamp")

        if keypoints is None or descriptors is None or points_3d is None:
            self.logger.error("Missing required data for PnP odometry")
            self.status = OdometryStatus.LOST
            return self.current_pose

        # Ensure camera matrix is on the right device
        if self.camera_matrix is not None:
            self.camera_matrix = self.camera_matrix.to(device)

        if self.dist_coeffs is not None:
            self.dist_coeffs = self.dist_coeffs.to(device)

        # Handle the first frame
        if self.prev_keypoints is None or not self.is_initialized:
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_points_3d = points_3d
            self.status = OdometryStatus.INITIALIZING
            return self.current_pose

        # Match features between frames
        matches = self.matcher.match_descriptors(self.prev_descriptors, descriptors)

        if len(matches) < self.min_inliers:
            self.logger.warning(
                f"Not enough matches for PnP: {len(matches)} < {self.min_inliers}"
            )
            self.status = OdometryStatus.UNCERTAIN
            return self.current_pose

        # Extract matched points
        prev_indices = [m[0] for m in matches]
        curr_indices = [m[1] for m in matches]

        # Get 3D points from previous frame
        object_points = self.prev_points_3d[prev_indices].to(device)

        # Get 2D points from current frame
        image_points = torch.tensor(
            [keypoints[i].pt() for i in curr_indices],
            dtype=torch.float32,
            device=device,
        )

        # Solve PnP problem
        relative_pose = self._solve_pnp(object_points, image_points)

        if relative_pose is None:
            self.logger.warning("PnP solver failed")
            self.status = OdometryStatus.UNCERTAIN
            return self.current_pose

        # Update pose
        self.update_pose(relative_pose)

        # Store current frame data for next iteration
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.prev_points_3d = points_3d

        return self.current_pose

    def _solve_pnp(
        self, object_points: torch.Tensor, image_points: torch.Tensor
    ) -> Optional[FramePose]:
        """
        Solve the PnP problem to get camera pose.

        Args:
            object_points: 3D points in the world coordinate (N, 3)
            image_points: 2D points in the image plane (N, 2)

        Returns:
            Estimated relative pose or None if estimation fails
        """
        device = object_points.device

        # Check if we have enough points
        if len(object_points) < 4:
            self.logger.warning(f"Not enough points for PnP: {len(object_points)} < 4")
            return None

        # If camera matrix is not provided, use a default one
        if self.camera_matrix is None:
            fx = fy = max(image_points[:, 0].max(), image_points[:, 1].max())
            cx = image_points[:, 0].mean()
            cy = image_points[:, 1].mean()
            self.camera_matrix = torch.tensor(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                dtype=torch.float32,
                device=device,
            )

        # Since PyTorch doesn't have a direct PnP solver, we need to implement it ourselves
        # or use an external library. Here we'll implement a simple EPnP algorithm:

        # This is a placeholder for a real EPnP implementation
        rotation, translation, inliers = self._epnp_ransac(object_points, image_points)

        if rotation is None or translation is None:
            return None

        # Check if we have enough inliers
        if inliers is not None and len(inliers) < self.min_inliers:
            self.logger.warning(
                f"Not enough inliers for PnP: {len(inliers)} < {self.min_inliers}"
            )
            self.status = OdometryStatus.UNCERTAIN
            return None

        # Create pose from rotation and translation
        pose = FramePose(rotation, translation)

        return pose

    def _epnp_ransac(
        self, object_points: torch.Tensor, image_points: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        EPnP with RANSAC for robust pose estimation.

        Args:
            object_points: 3D points in world coordinates (N, 3)
            image_points: 2D points in image plane (N, 2)

        Returns:
            Tuple of (rotation, translation, inliers)
        """
        device = object_points.device
        n_points = object_points.shape[0]
        best_inliers = []
        best_rotation = None
        best_translation = None

        # Convert to homogeneous coordinates
        object_points_h = torch.cat(
            [object_points, torch.ones(n_points, 1, device=device)], dim=1
        )

        # Normalize image points
        K_inv = torch.inverse(self.camera_matrix)
        image_points_h = torch.cat(
            [image_points, torch.ones(n_points, 1, device=device)], dim=1
        )
        normalized_points = torch.matmul(image_points_h, K_inv.t())[:, :2]

        # RANSAC loop
        max_iterations = 100
        min_points = 6  # Minimum points for EPnP

        for _ in range(max_iterations):
            # Randomly select a minimal set of points
            if n_points <= min_points:
                indices = torch.arange(n_points, device=device)
            else:
                indices = torch.randperm(n_points, device=device)[:min_points]

            selected_obj_points = object_points[indices]
            selected_img_points = normalized_points[indices]

            # Solve PnP for this subset
            try:
                rotation, translation = self._epnp(
                    selected_obj_points, selected_img_points
                )
            except Exception as e:
                self.logger.debug(f"EPnP solver failed: {e}")
                continue

            # Evaluate model
            projection = torch.matmul(
                object_points_h,
                torch.cat([rotation, translation.unsqueeze(1)], dim=1).t(),
            )

            # Normalize homogeneous coordinates
            projection = projection / projection[:, 2:3]

            # Project back to image coordinates
            reprojected = torch.matmul(projection, self.camera_matrix.t())[:, :2]

            # Compute reprojection errors
            errors = torch.sqrt(torch.sum((reprojected - image_points) ** 2, dim=1))

            # Find inliers
            inliers = torch.where(errors < self.ransac_threshold)[0]

            # Update best model if we found more inliers
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_rotation = rotation
                best_translation = translation

                # Early termination if we found a good model
                if len(inliers) > n_points * 0.8:
                    break

        # Refine model using all inliers
        if len(best_inliers) >= min_points:
            try:
                refined_rotation, refined_translation = self._epnp(
                    object_points[best_inliers], normalized_points[best_inliers]
                )
                best_rotation = refined_rotation
                best_translation = refined_translation
            except Exception as e:
                self.logger.warning(f"EPnP refinement failed: {e}")

        return best_rotation, best_translation, best_inliers

    def _epnp(
        self, object_points: torch.Tensor, normalized_points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Efficient PnP algorithm implementation.

        Args:
            object_points: 3D points in world coordinates (N, 3)
            normalized_points: Normalized 2D points (N, 2)

        Returns:
            Tuple of (rotation, translation)
        """
        device = object_points.device
        n_points = object_points.shape[0]

        # Compute mean of the 3D points
        mean_obj = object_points.mean(dim=0)
        centered_obj = object_points - mean_obj

        # Compute the covariance matrix
        cov = torch.matmul(centered_obj.t(), centered_obj) / (n_points - 1)

        # Perform SVD on the covariance matrix
        U, S, V = torch.svd(cov)

        # Control points are defined along the 3 principal directions
        # c0 is the centroid, c1, c2, c3 are along principal directions
        scale = torch.sqrt(S).max() * 2.0
        control_points = torch.zeros((4, 3), device=device)
        control_points[0] = mean_obj
        control_points[1] = mean_obj + scale * U[:, 0]
        control_points[2] = mean_obj + scale * U[:, 1]
        control_points[3] = mean_obj + scale * U[:, 2]

        # Compute barycentric coordinates
        alphas = self._compute_barycentric(object_points, control_points)

        # Build the system matrix M
        M = torch.zeros((2 * n_points, 12), device=device)

        for i in range(n_points):
            # Get 2D point
            x, y = normalized_points[i]

            # Fill M matrix
            for j in range(4):
                alpha_j = alphas[i, j]

                M[2 * i, 3 * j : 3 * j + 3] = alpha_j * torch.tensor(
                    [1, 0, -x], device=device
                )
                M[2 * i + 1, 3 * j : 3 * j + 3] = alpha_j * torch.tensor(
                    [0, 1, -y], device=device
                )

        # Solve for control points in camera coordinates using SVD
        _, _, Vt = torch.svd(M)
        kernel = Vt[-1].reshape(4, 3)

        # Compute rotation and translation using procrustes analysis
        rotation, translation = self._procrustes(control_points, kernel)

        return rotation, translation

    def _compute_barycentric(
        self, points: torch.Tensor, control_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute barycentric coordinates of points with respect to control points.

        Args:
            points: 3D points (N, 3)
            control_points: Control points (4, 3)

        Returns:
            Barycentric coordinates (N, 4)
        """
        device = points.device
        n_points = points.shape[0]

        # Compute matrix C for the linear system
        C = torch.zeros((3, 3), device=device)
        C[0] = control_points[1] - control_points[0]
        C[1] = control_points[2] - control_points[0]
        C[2] = control_points[3] - control_points[0]

        # Compute inverse of C
        C_inv = torch.inverse(C)

        # Compute barycentric coordinates
        alphas = torch.zeros((n_points, 4), device=device)

        for i in range(n_points):
            # Compute coefficients for control points
            W = points[i] - control_points[0]
            coeffs = torch.matmul(C_inv, W)

            # Set barycentric coordinates
            alphas[i, 0] = 1.0 - coeffs.sum()
            alphas[i, 1:] = coeffs

        return alphas

    def _procrustes(
        self, control_points: torch.Tensor, estimated_control_points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotation and translation using Procrustes analysis.

        Args:
            control_points: Original control points (4, 3)
            estimated_control_points: Estimated control points in camera frame (4, 3)

        Returns:
            Tuple of (rotation, translation)
        """
        device = control_points.device

        # Normalize estimated control points to have the same scale as original
        scale = torch.norm(control_points[1:] - control_points[0]) / torch.norm(
            estimated_control_points[1:] - estimated_control_points[0]
        )
        estimated_control_points = estimated_control_points * scale

        # Compute centroids
        centroid1 = control_points.mean(dim=0)
        centroid2 = estimated_control_points.mean(dim=0)

        # Center points
        centered1 = control_points - centroid1
        centered2 = estimated_control_points - centroid2

        # Compute covariance matrix
        H = torch.matmul(centered1.t(), centered2)

        # SVD decomposition
        U, _, V = torch.svd(H)

        # Compute rotation matrix
        rotation = torch.matmul(V, U.t())

        # Ensure proper rotation matrix (det = 1)
        det = torch.det(rotation)
        if det < 0:
            V[:, 2] = -V[:, 2]
            rotation = torch.matmul(V, U.t())

        # Compute translation
        translation = centroid2 - torch.matmul(rotation, centroid1)

        return rotation, translation
