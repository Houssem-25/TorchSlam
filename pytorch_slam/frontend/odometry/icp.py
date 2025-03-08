import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseOdometry, FramePose, OdometryStatus


class ICPVariant(Enum):
    """Variants of ICP algorithm."""

    POINT_TO_POINT = 0
    POINT_TO_PLANE = 1


class ICPOdometry(BaseOdometry):
    """
    Iterative Closest Point (ICP) based odometry.

    Estimates motion between consecutive point clouds.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize ICP odometry estimator.

        Args:
            config: Configuration dictionary with the following keys:
                - max_iterations: Maximum number of iterations
                - distance_threshold: Threshold for point correspondences
                - convergence_threshold: Convergence criteria for transformation change
                - point_selection_ratio: Ratio of points to randomly select
                - variant: ICP variant ('point_to_point' or 'point_to_plane')
                - use_normals: Whether to use point normals (required for point-to-plane)
                - min_correspondences: Minimum number of point correspondences
                - neighborhood_size: Size of neighborhood for normal estimation
                - voxel_size: Size of voxels for downsampling
        """
        super().__init__(config)

        # Extract configuration
        self.max_iterations = self.config.get("max_iterations", 50)
        self.distance_threshold = self.config.get("distance_threshold", 0.5)
        self.convergence_threshold = self.config.get("convergence_threshold", 1e-5)
        self.point_selection_ratio = self.config.get("point_selection_ratio", 1.0)

        # ICP variant
        variant_str = self.config.get("variant", "point_to_point")
        self.variant = (
            ICPVariant.POINT_TO_POINT
            if variant_str == "point_to_point"
            else ICPVariant.POINT_TO_PLANE
        )

        self.use_normals = self.config.get("use_normals", False)
        self.min_correspondences = self.config.get("min_correspondences", 100)
        self.neighborhood_size = self.config.get("neighborhood_size", 30)
        self.voxel_size = self.config.get("voxel_size", None)

        # Previous point cloud
        self.prev_points = None
        self.prev_normals = None

    def process_frame(self, data: Dict[str, Any]) -> FramePose:
        """
        Process a new frame to estimate odometry using ICP.

        Args:
            data: Dictionary containing:
                - points: Point cloud as tensor (N, 3)
                - normals: Point normals as tensor (N, 3) (optional, used for point-to-plane ICP)
                - timestamp: Optional timestamp

        Returns:
            Estimated pose for the current frame
        """
        # Extract data
        points = data.get("points")
        normals = data.get("normals")
        timestamp = data.get("timestamp")

        if points is None:
            self.logger.error("Missing point cloud data for ICP odometry")
            self.status = OdometryStatus.LOST
            return self.current_pose

        # Check if we need normals but they're not provided
        if (
            self.variant == ICPVariant.POINT_TO_PLANE
            and normals is None
            and self.use_normals
        ):
            if points.shape[0] > self.neighborhood_size:
                # Estimate normals
                normals = self._estimate_normals(points)
            else:
                self.logger.warning(
                    "Not enough points to estimate normals, falling back to point-to-point ICP"
                )
                self.variant = ICPVariant.POINT_TO_POINT

        # Downsample point cloud if needed
        if self.voxel_size is not None:
            points, normals = self._voxel_downsample(points, normals, self.voxel_size)

        # Handle the first frame
        if self.prev_points is None or not self.is_initialized:
            self.prev_points = points
            self.prev_normals = normals
            self.status = OdometryStatus.INITIALIZING
            return self.current_pose

        # Run ICP to estimate transformation
        success, rotation, translation = self._icp_align(
            self.prev_points, points, self.prev_normals, normals
        )

        if not success:
            self.logger.warning("ICP alignment failed")
            self.status = OdometryStatus.UNCERTAIN
            return self.current_pose

        # Create pose from rotation and translation
        relative_pose = FramePose(rotation, translation, timestamp)

        # Update pose
        self.update_pose(relative_pose)

        # Store current point cloud for next iteration
        self.prev_points = points
        self.prev_normals = normals

        return self.current_pose

    def _voxel_downsample(
        self, points: torch.Tensor, normals: Optional[torch.Tensor], voxel_size: float
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Downsample point cloud using voxel grid.

        Args:
            points: Input point cloud (N, 3)
            normals: Point normals (N, 3) or None
            voxel_size: Size of voxels

        Returns:
            Tuple of (downsampled_points, downsampled_normals)
        """
        device = points.device

        # Compute voxel indices for each point
        voxel_indices = torch.floor(points / voxel_size).int()

        # Create a unique ID for each voxel
        voxel_ids = (
            voxel_indices[:, 0] * 1000000
            + voxel_indices[:, 1] * 1000
            + voxel_indices[:, 2]
        )

        # Get unique voxels
        unique_voxel_ids, inverse_indices = torch.unique(voxel_ids, return_inverse=True)

        # Create downsampled point cloud
        downsampled_points = torch.zeros((len(unique_voxel_ids), 3), device=device)
        point_counts = torch.zeros(len(unique_voxel_ids), device=device)

        # Accumulate points in each voxel
        for i in range(len(points)):
            voxel_idx = torch.nonzero(unique_voxel_ids == voxel_ids[i]).item()
            downsampled_points[voxel_idx] += points[i]
            point_counts[voxel_idx] += 1

        # Average points in each voxel
        downsampled_points = downsampled_points / point_counts.unsqueeze(1)

        # Downsample normals if provided
        downsampled_normals = None
        if normals is not None:
            downsampled_normals = torch.zeros((len(unique_voxel_ids), 3), device=device)
            normal_counts = torch.zeros(len(unique_voxel_ids), device=device)

            # Accumulate normals in each voxel
            for i in range(len(normals)):
                voxel_idx = torch.nonzero(unique_voxel_ids == voxel_ids[i]).item()
                downsampled_normals[voxel_idx] += normals[i]
                normal_counts[voxel_idx] += 1

            # Average and normalize normals
            downsampled_normals = downsampled_normals / normal_counts.unsqueeze(1)
            downsampled_normals = F.normalize(downsampled_normals, p=2, dim=1)

        return downsampled_points, downsampled_normals

    def _estimate_normals(self, points: torch.Tensor) -> torch.Tensor:
        """
        Estimate point normals using PCA on local neighborhoods.

        Args:
            points: Input point cloud (N, 3)

        Returns:
            Point normals (N, 3)
        """
        device = points.device
        num_points = points.shape[0]

        # Compute pairwise distances
        expanded_points = points.unsqueeze(1)
        expanded_points_t = points.unsqueeze(0)

        # Compute squared distances
        squared_distances = torch.sum((expanded_points - expanded_points_t) ** 2, dim=2)

        # Find k-nearest neighbors
        _, indices = torch.topk(
            squared_distances, self.neighborhood_size, dim=1, largest=False
        )

        # Initialize normals
        normals = torch.zeros((num_points, 3), device=device)

        # Compute normal for each point
        for i in range(num_points):
            # Get neighbors
            neighbors = points[indices[i]]

            # Compute centered neighbors
            centered = neighbors - neighbors.mean(dim=0, keepdim=True)

            # Compute covariance matrix
            cov = torch.matmul(centered.t(), centered) / (self.neighborhood_size - 1)

            # Perform SVD
            try:
                U, S, V = torch.svd(cov)
                # Normal is the eigenvector corresponding to the smallest eigenvalue
                normal = V[:, 2]
                normals[i] = normal
            except Exception as e:
                # Fall back to a default normal if SVD fails
                self.logger.debug(f"SVD failed for point {i}: {e}")
                normals[i] = torch.tensor([0.0, 0.0, 1.0], device=device)

        return normals

    def _find_correspondences(
        self, source_points: torch.Tensor, target_points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find nearest neighbor correspondences between point clouds.

        Args:
            source_points: Source point cloud (N, 3)
            target_points: Target point cloud (M, 3)

        Returns:
            Tuple of (source_indices, target_indices) for matching points
        """
        device = source_points.device

        # Compute pairwise distances
        expanded_source = source_points.unsqueeze(1)
        expanded_target = target_points.unsqueeze(0)

        # Compute squared distances
        squared_distances = torch.sum((expanded_source - expanded_target) ** 2, dim=2)

        # Find nearest neighbors
        distances, target_indices = torch.min(squared_distances, dim=1)

        # Filter by distance threshold
        valid_mask = distances < self.distance_threshold**2

        if torch.sum(valid_mask) < self.min_correspondences:
            self.logger.warning(
                f"Not enough correspondences: {torch.sum(valid_mask).item()} < {self.min_correspondences}"
            )
            return torch.tensor([], device=device, dtype=torch.long), torch.tensor(
                [], device=device, dtype=torch.long
            )

        source_indices = torch.arange(len(source_points), device=device)[valid_mask]
        target_indices = target_indices[valid_mask]

        return source_indices, target_indices

    def _icp_align(
        self,
        source_points: torch.Tensor,
        target_points: torch.Tensor,
        source_normals: Optional[torch.Tensor] = None,
        target_normals: Optional[torch.Tensor] = None,
    ) -> Tuple[bool, torch.Tensor, torch.Tensor]:
        """
        Align source point cloud to target using ICP.

        Args:
            source_points: Source point cloud (N, 3)
            target_points: Target point cloud (M, 3)
            source_normals: Source point normals (N, 3)
            target_normals: Target point normals (M, 3)

        Returns:
            Tuple of (success, rotation, translation)
        """
        device = source_points.device

        # Initialize transformation
        current_rotation = torch.eye(3, device=device)
        current_translation = torch.zeros(3, device=device)

        # Current transformed source points
        current_source = source_points.clone()

        for iteration in range(self.max_iterations):
            # Find correspondences
            source_indices, target_indices = self._find_correspondences(
                current_source, target_points
            )

            if len(source_indices) < self.min_correspondences:
                return False, current_rotation, current_translation

            # Get corresponding points
            src_corr = current_source[source_indices]
            tgt_corr = target_points[target_indices]

            # Compute transformation based on ICP variant
            if self.variant == ICPVariant.POINT_TO_PLANE and target_normals is not None:
                tgt_normals_corr = target_normals[target_indices]
                (
                    success,
                    delta_rotation,
                    delta_translation,
                ) = self._compute_point_to_plane_transform(
                    src_corr, tgt_corr, tgt_normals_corr
                )
            else:
                (
                    success,
                    delta_rotation,
                    delta_translation,
                ) = self._compute_point_to_point_transform(src_corr, tgt_corr)

            if not success:
                return False, current_rotation, current_translation

            # Update transformation
            current_rotation = torch.matmul(delta_rotation, current_rotation)
            current_translation = (
                torch.matmul(delta_rotation, current_translation) + delta_translation
            )

            # Apply transformation to source points
            current_source = (
                torch.matmul(source_points, current_rotation.t()) + current_translation
            )

            # Check for convergence
            if (
                torch.norm(delta_translation) < self.convergence_threshold
                and torch.norm(delta_rotation - torch.eye(3, device=device))
                < self.convergence_threshold
            ):
                break

        return True, current_rotation, current_translation

    def _compute_point_to_point_transform(
        self, source_points: torch.Tensor, target_points: torch.Tensor
    ) -> Tuple[bool, torch.Tensor, torch.Tensor]:
        """
        Compute transformation using point-to-point ICP.

        Args:
            source_points: Source points (N, 3)
            target_points: Target points (N, 3)

        Returns:
            Tuple of (success, rotation, translation)
        """
        device = source_points.device

        # Compute centroids
        source_centroid = torch.mean(source_points, dim=0)
        target_centroid = torch.mean(target_points, dim=0)

        # Center points
        centered_source = source_points - source_centroid
        centered_target = target_points - target_centroid

        # Compute covariance matrix
        covariance = torch.matmul(centered_source.t(), centered_target)

        # SVD decomposition
        try:
            U, _, V = torch.svd(covariance)
        except Exception as e:
            self.logger.warning(f"SVD failed: {e}")
            return False, torch.eye(3, device=device), torch.zeros(3, device=device)

        # Compute rotation matrix
        rotation = torch.matmul(V, U.t())

        # Ensure proper rotation matrix (det = 1)
        det = torch.det(rotation)
        if det < 0:
            V[:, 2] = -V[:, 2]
            rotation = torch.matmul(V, U.t())

        # Compute translation
        translation = target_centroid - torch.matmul(rotation, source_centroid)

        return True, rotation, translation

    def _compute_point_to_plane_transform(
        self,
        source_points: torch.Tensor,
        target_points: torch.Tensor,
        target_normals: torch.Tensor,
    ) -> Tuple[bool, torch.Tensor, torch.Tensor]:
        """
        Compute transformation using point-to-plane ICP.

        Args:
            source_points: Source points (N, 3)
            target_points: Target points (N, 3)
            target_normals: Target point normals (N, 3)

        Returns:
            Tuple of (success, rotation, translation)
        """
        device = source_points.device
        num_points = source_points.shape[0]

        # For point-to-plane ICP, we solve a linear system:
        # A * x = b, where x = [alpha, beta, gamma, tx, ty, tz]
        # alpha, beta, gamma are rotation angles, tx, ty, tz are translation components

        # Initialize A and b
        A = torch.zeros((num_points, 6), device=device)
        b = torch.zeros(num_points, device=device)

        # Fill A and b
        for i in range(num_points):
            p = source_points[i]
            q = target_points[i]
            n = target_normals[i]

            # Cross product: p × n
            cross = torch.tensor(
                [
                    p[1] * n[2] - p[2] * n[1],
                    p[2] * n[0] - p[0] * n[2],
                    p[0] * n[1] - p[1] * n[0],
                ],
                device=device,
            )

            # Fill row of A
            A[i, :3] = cross
            A[i, 3:] = n

            # Fill element of b
            b[i] = torch.dot(n, q - p)

        # Solve the system using SVD
        try:
            # Compute pseudo-inverse using SVD
            U, S, V = torch.svd(A)

            # Handle singular values
            eps = 1e-10
            S_inv = torch.zeros_like(S)
            S_inv[S > eps] = 1.0 / S[S > eps]

            # Compute solution
            x = torch.matmul(V, torch.matmul(torch.diag(S_inv), torch.matmul(U.t(), b)))
        except Exception as e:
            self.logger.warning(f"SVD failed: {e}")
            return False, torch.eye(3, device=device), torch.zeros(3, device=device)

        # Extract rotation and translation
        alpha, beta, gamma = x[:3]
        translation = x[3:]

        # Compute rotation matrix using small angle approximation
        # R ≈ I + [0 -gamma beta; gamma 0 -alpha; -beta alpha 0]
        rotation = torch.eye(3, device=device)
        rotation[0, 1] = -gamma
        rotation[0, 2] = beta
        rotation[1, 0] = gamma
        rotation[1, 2] = -alpha
        rotation[2, 0] = -beta
        rotation[2, 1] = alpha

        # Ensure orthogonality (important for small angle approximation)
        U, _, V = torch.svd(rotation)
        rotation = torch.matmul(U, V.t())

        return True, rotation, translation
