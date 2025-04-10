from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


class MatchingMethod(Enum):
    """Enum for different matching methods."""

    NEAREST_NEIGHBOR = 1
    RATIO_TEST = 2
    MUTUAL_NEAREST = 3


class Match:
    """Represents a match between two keypoints."""

    def __init__(self, query_idx: int, train_idx: int, distance: float):
        """
        Initialize a match.

        Args:
            query_idx: Index of the keypoint in the query image
            train_idx: Index of the keypoint in the train image
            distance: Distance/similarity between the descriptors
        """
        self.query_idx = query_idx
        self.train_idx = train_idx
        self.distance = distance

    def __repr__(self) -> str:
        return f"Match(query_idx={self.query_idx}, train_idx={self.train_idx}, distance={self.distance:.4f})"


class FeatureMatcher:
    """Class for matching features between images using FLANN and RANSAC."""

    def __init__(
        self,
        method: MatchingMethod = MatchingMethod.RATIO_TEST,
        ratio_threshold: float = 0.7,
        cross_check: bool = True,
        max_distance: float = float("inf"),
        num_neighbors: int = 2,
    ):
        """
        Initialize feature matcher.

        Args:
            method: Method for matching features
            ratio_threshold: Threshold for ratio test (if applicable)
            cross_check: Whether to perform cross-checking for mutual matches
            max_distance: Maximum allowable distance between matched descriptors
            num_neighbors: Number of neighbors to find in FLANN
        """
        self.method = method
        self.ratio_threshold = ratio_threshold
        self.cross_check = cross_check
        self.max_distance = max_distance
        self.num_neighbors = num_neighbors

    def match(
        self,
        query_descriptors: torch.Tensor,
        train_descriptors: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        train_mask: Optional[torch.Tensor] = None,
    ) -> List[Match]:
        """
        Match descriptors between two sets using FLANN.

        Args:
            query_descriptors: Descriptors from the query image (N, D)
            train_descriptors: Descriptors from the train image (M, D)
            query_mask: Binary mask for query descriptors (N, )
            train_mask: Binary mask for train descriptors (M, )

        Returns:
            List of Match objects
        """
        # Apply masks if provided
        if query_mask is not None:
            query_indices = torch.nonzero(query_mask).squeeze(1)
            query_descriptors = query_descriptors[query_indices]
        else:
            query_indices = torch.arange(
                query_descriptors.shape[0], device=query_descriptors.device
            )

        if train_mask is not None:
            train_indices = torch.nonzero(train_mask).squeeze(1)
            train_descriptors = train_descriptors[train_indices]
        else:
            train_indices = torch.arange(
                train_descriptors.shape[0], device=train_descriptors.device
            )

        # Use FLANN matching
        matches = self._flann_match(query_descriptors, train_descriptors)

        # Map indices back to original positions
        for match in matches:
            match.query_idx = query_indices[match.query_idx].item()
            match.train_idx = train_indices[match.train_idx].item()

        return matches

    def _flann_match(
        self, query_descriptors: torch.Tensor, train_descriptors: torch.Tensor
    ) -> List[Match]:
        """
        Perform FLANN matching (approximate nearest neighbor).

        Args:
            query_descriptors: Descriptors from the query image (N, D)
            train_descriptors: Descriptors from the train image (M, D)

        Returns:
            List of Match objects
        """
        # Handle empty descriptor sets
        if query_descriptors.shape[0] == 0 or train_descriptors.shape[0] == 0:
            return []

        # Determine descriptor type and prepare for matching
        if query_descriptors.dtype == torch.uint8:
            # Convert binary descriptors to float for kNN
            query_float = query_descriptors.float()
            train_float = train_descriptors.float()

            # For binary descriptors, use Hamming distance
            distances, indices = self._knn_match_binary(
                query_descriptors, train_descriptors, self.num_neighbors
            )
        else:
            # For floating point descriptors, normalize for cosine similarity
            query_norm = F.normalize(query_descriptors, p=2, dim=1)
            train_norm = F.normalize(train_descriptors, p=2, dim=1)

            # Find k nearest neighbors
            distances, indices = self._knn_match(
                query_norm, train_norm, self.num_neighbors
            )

        # Apply the selected matching method
        matches = []

        if self.method == MatchingMethod.RATIO_TEST and self.num_neighbors >= 2:
            # Apply ratio test
            for i in range(distances.shape[0]):
                if distances[i, 0] < self.ratio_threshold * distances[i, 1]:
                    matches.append(
                        Match(i, indices[i, 0].item(), distances[i, 0].item())
                    )

        elif self.method == MatchingMethod.NEAREST_NEIGHBOR:
            # Use only the nearest neighbor
            for i in range(distances.shape[0]):
                matches.append(Match(i, indices[i, 0].item(), distances[i, 0].item()))

        elif self.method == MatchingMethod.MUTUAL_NEAREST:
            # Reverse the matching direction
            if query_descriptors.dtype == torch.uint8:
                rev_distances, rev_indices = self._knn_match_binary(
                    train_descriptors, query_descriptors, 1
                )
            else:
                train_norm = F.normalize(train_descriptors, p=2, dim=1)
                query_norm = F.normalize(query_descriptors, p=2, dim=1)
                rev_distances, rev_indices = self._knn_match(train_norm, query_norm, 1)

            # Find mutual nearest neighbors
            for i in range(distances.shape[0]):
                t_idx = indices[i, 0].item()
                if rev_indices[t_idx, 0].item() == i:
                    matches.append(Match(i, t_idx, distances[i, 0].item()))

        # Apply cross-check if needed and not already done by mutual nearest
        elif self.cross_check and self.method != MatchingMethod.MUTUAL_NEAREST:
            # Reverse the matching direction
            if query_descriptors.dtype == torch.uint8:
                rev_distances, rev_indices = self._knn_match_binary(
                    train_descriptors, query_descriptors, 1
                )
            else:
                train_norm = F.normalize(train_descriptors, p=2, dim=1)
                query_norm = F.normalize(query_descriptors, p=2, dim=1)
                rev_distances, rev_indices = self._knn_match(train_norm, query_norm, 1)

            # Keep only mutual matches
            for match in matches:
                if rev_indices[match.train_idx, 0].item() == match.query_idx:
                    filtered_matches.append(match)
            matches = filtered_matches

        # Filter by max distance
        matches = [m for m in matches if m.distance <= self.max_distance]

        return matches

    def _knn_match(
        self, query: torch.Tensor, train: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform k-nearest neighbor matching for float descriptors using cosine similarity.

        Args:
            query: Query descriptors (N, D)
            train: Train descriptors (M, D)
            k: Number of nearest neighbors to find

        Returns:
            Tuple of (distances, indices)
        """
        # Compute similarity matrix
        similarity = torch.mm(query, train.t())

        # Convert to distance (1 - similarity)
        distances = 1.0 - similarity

        # Find k nearest neighbors
        k = min(k, train.shape[0])
        top_distances, top_indices = torch.topk(distances, k, dim=1, largest=False)

        return top_distances, top_indices

    def _knn_match_binary(
        self, query: torch.Tensor, train: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform k-nearest neighbor matching for binary descriptors using Hamming distance.

        Args:
            query: Query descriptors (N, D)
            train: Train descriptors (M, D)
            k: Number of nearest neighbors to find

        Returns:
            Tuple of (distances, indices)
        """
        device = query.device
        n, d = query.shape
        m = train.shape[0]

        # Use a more efficient implementation for binary descriptors
        # Compute Hamming distances between all pairs of descriptors
        # We'll compute this in batches to avoid memory issues with large descriptor sets

        # Initialize distance matrix
        distances = torch.zeros((n, m), device=device)

        # Batch size for processing (adjust as needed)
        batch_size = 128

        for i in range(0, n, batch_size):
            batch_end = min(i + batch_size, n)
            query_batch = query[i:batch_end]

            # Compute Hamming distance for this batch
            batch_distances = torch.zeros((batch_end - i, m), device=device)

            # For each descriptor in the batch
            for j in range(m):
                # XOR the batch descriptors with the train descriptor
                xor_result = query_batch ^ train[j].unsqueeze(0)

                # Count bits using lookup table or bit operations
                # This can be done efficiently for uint8 data
                bit_count = torch.zeros(batch_end - i, device=device)
                for b in range(d):
                    bit_count += torch.tensor(
                        [bin(x.item()).count("1") for x in xor_result[:, b]],
                        device=device,
                    )

                batch_distances[:, j] = bit_count

            # Update the full distance matrix
            distances[i:batch_end] = batch_distances

        # Find k nearest neighbors
        k = min(k, m)
        top_distances, top_indices = torch.topk(distances, k, dim=1, largest=False)

        return top_distances, top_indices


def filter_matches_by_ransac(
    matches: List[Match],
    query_keypoints: List,
    train_keypoints: List,
    ransac_threshold: float = 3.0,
    confidence: float = 0.99,
    max_iterations: int = 1000,
    min_inliers: int = 8,
) -> Tuple[List[Match], torch.Tensor]:
    """
    Filter matches using RANSAC to find a consistent transformation.

    Args:
        matches: List of matches
        query_keypoints: List of keypoints from query image
        train_keypoints: List of keypoints from train image
        ransac_threshold: Threshold for RANSAC inlier detection
        confidence: Confidence level for RANSAC
        max_iterations: Maximum number of RANSAC iterations
        min_inliers: Minimum number of inliers required

    Returns:
        Tuple of (filtered_matches, homography/fundamental_matrix)
    """
    # Need at least 8 points to estimate a fundamental matrix or 4 for homography
    if len(matches) < min_inliers:
        return [], None

    # Extract matched points
    query_pts = torch.tensor(
        [query_keypoints[m.query_idx].pt() for m in matches], dtype=torch.float32
    )
    train_pts = torch.tensor(
        [train_keypoints[m.train_idx].pt() for m in matches], dtype=torch.float32
    )

    # Choose between fundamental matrix or homography based on the number of matches
    use_fundamental = True if len(matches) >= 8 else False

    # Compute transformation matrix and inliers using RANSAC
    if use_fundamental:
        transformation, inlier_mask = compute_fundamental_matrix_ransac(
            query_pts, train_pts, ransac_threshold, confidence, max_iterations
        )
    else:
        transformation, inlier_mask = compute_homography_ransac(
            query_pts, train_pts, ransac_threshold, confidence, max_iterations
        )

    # Check if we have enough inliers
    inlier_count = torch.sum(inlier_mask).item()
    if inlier_count < min_inliers:
        return [], transformation

    # Filter matches
    filtered_matches = [match for i, match in enumerate(matches) if inlier_mask[i]]

    return filtered_matches, transformation


def compute_fundamental_matrix_ransac(
    pts1: torch.Tensor,
    pts2: torch.Tensor,
    ransac_threshold: float = 3.0,
    confidence: float = 0.99,
    max_iterations: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute fundamental matrix from point correspondences using RANSAC.

    Args:
        pts1: First set of points (N, 2)
        pts2: Second set of points (N, 2)
        ransac_threshold: Threshold for RANSAC inlier detection
        confidence: Confidence level for RANSAC
        max_iterations: Maximum number of RANSAC iterations

    Returns:
        Tuple of (fundamental_matrix, inlier_mask)
    """
    device = pts1.device
    num_points = pts1.shape[0]

    # Need at least 8 points for fundamental matrix estimation
    if num_points < 8:
        return torch.eye(3, device=device), torch.zeros(
            num_points, dtype=torch.bool, device=device
        )

    # RANSAC parameters
    best_num_inliers = 0
    best_inlier_mask = torch.zeros(num_points, dtype=torch.bool, device=device)
    best_F = torch.eye(3, device=device)

    # Calculate number of iterations based on confidence
    # p = probability of selecting only inliers
    # w = inlier ratio (initially unknown, assume 0.5)
    # k = number of iterations
    # 1 - p = (1 - w^8)^k
    # k = log(1 - p) / log(1 - w^8)
    w = 0.5  # Initial inlier ratio estimate
    k = int(np.log(1 - confidence) / np.log(1 - w**8))
    k = min(max(k, 8), max_iterations)

    # RANSAC iterations
    for _ in range(k):
        # Randomly select 8 points
        indices = torch.randperm(num_points, device=device)[:8]
        sample_pts1 = pts1[indices]
        sample_pts2 = pts2[indices]

        try:
            # Compute fundamental matrix from sample
            F = compute_fundamental_matrix_8point(sample_pts1, sample_pts2)

            # Compute epipolar distances for all points
            distances = compute_epipolar_distances(F, pts1, pts2)

            # Find inliers
            inlier_mask = distances < ransac_threshold
            num_inliers = inlier_mask.sum().item()

            # Update best solution if better
            if num_inliers > best_num_inliers:
                best_num_inliers = num_inliers
                best_inlier_mask = inlier_mask
                best_F = F

                # Update inlier ratio estimate
                w = num_inliers / num_points
                k = int(np.log(1 - confidence) / np.log(1 - w**8))
                k = min(max(k, 8), max_iterations)

        except RuntimeError as e:
            # Handle numerical errors in fundamental matrix computation
            continue

    # If no good solution found, return identity matrix
    if best_num_inliers < 8:
        return torch.eye(3, device=device), torch.zeros(
            num_points, dtype=torch.bool, device=device
        )

    # Refine fundamental matrix using all inliers
    if best_num_inliers > 8:
        inlier_pts1 = pts1[best_inlier_mask]
        inlier_pts2 = pts2[best_inlier_mask]
        try:
            best_F = compute_fundamental_matrix_8point(inlier_pts1, inlier_pts2)
        except RuntimeError:
            # If refinement fails, keep the previous best solution
            pass

    return best_F, best_inlier_mask


def compute_homography_ransac(
    pts1: torch.Tensor,
    pts2: torch.Tensor,
    ransac_threshold: float = 3.0,
    confidence: float = 0.99,
    max_iterations: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute homography matrix from point correspondences using RANSAC.

    Args:
        pts1: First set of points (N, 2)
        pts2: Second set of points (N, 2)
        ransac_threshold: Threshold for RANSAC inlier detection
        confidence: Confidence level for RANSAC
        max_iterations: Maximum number of RANSAC iterations

    Returns:
        Tuple of (homography_matrix, inlier_mask)
    """
    device = pts1.device
    num_points = pts1.shape[0]

    # Need at least 4 points for homography estimation
    if num_points < 4:
        return torch.eye(3, device=device), torch.zeros(
            num_points, dtype=torch.bool, device=device
        )

    # RANSAC parameters
    best_num_inliers = 0
    best_inlier_mask = torch.zeros(num_points, dtype=torch.bool, device=device)
    best_H = torch.eye(3, device=device)

    # Calculate number of iterations based on confidence
    w = 0.5  # Initial inlier ratio estimate
    k = int(np.log(1 - confidence) / np.log(1 - w**4))
    k = min(max(k, 4), max_iterations)

    # RANSAC loop
    for _ in range(k):
        # Randomly select 4 points
        indices = torch.randperm(num_points)[:4]
        selected_pts1 = pts1[indices]
        selected_pts2 = pts2[indices]

        # Compute homography using Direct Linear Transform (DLT)
        H = compute_homography_dlt(selected_pts1, selected_pts2)

        # Check if H is valid
        if H is None:
            continue

        # Compute reprojection error
        distances = compute_homography_distances(H, pts1, pts2)

        # Determine inliers
        inlier_mask = distances < ransac_threshold
        num_inliers = torch.sum(inlier_mask).item()

        # Update best model if we found more inliers
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_inlier_mask = inlier_mask
            best_H = H

            # Update inlier ratio estimate and number of iterations
            w = num_inliers / num_points
            k_new = int(np.log(1 - confidence) / np.log(1 - w**4))
            k = min(k_new, max_iterations)

    # Refine homography using all inliers
    if best_num_inliers >= 4:
        refined_H = compute_homography_dlt(
            pts1[best_inlier_mask], pts2[best_inlier_mask]
        )
        if refined_H is not None:
            return refined_H, best_inlier_mask

    return best_H, best_inlier_mask


def compute_fundamental_matrix_8point(
    pts1: torch.Tensor, pts2: torch.Tensor
) -> torch.Tensor:
    """
    Compute fundamental matrix using normalized 8-point algorithm.

    Args:
        pts1: First set of points (N, 2)
        pts2: Second set of points (N, 2)

    Returns:
        Fundamental matrix (3, 3)
    """
    # Normalize points
    pts1_normalized, T1 = normalize_points(pts1)
    pts2_normalized, T2 = normalize_points(pts2)

    # Build the constraint matrix
    n = pts1.shape[0]
    A = torch.zeros((n, 9), device=pts1.device)

    for i in range(n):
        x1, y1 = pts1_normalized[i]
        x2, y2 = pts2_normalized[i]
        A[i] = torch.tensor(
            [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1], device=pts1.device
        )

    # Solve Af = 0 using SVD
    _, _, Vt = torch.svd(A)
    f = Vt[-1]  # Last row of Vt is the solution
    F = f.reshape(3, 3)

    # Enforce rank-2 constraint
    U, S, Vt = torch.svd(F)
    S[2] = 0  # Set smallest singular value to zero
    F = torch.matmul(U, torch.matmul(torch.diag(S), Vt))

    # Denormalize
    F = torch.matmul(T2.t(), torch.matmul(F, T1))

    # Normalize to ensure F[2,2] = 1
    F = F / (F[2, 2] + 1e-10)

    return F


def compute_homography_dlt(
    pts1: torch.Tensor, pts2: torch.Tensor
) -> Optional[torch.Tensor]:
    """
    Compute homography matrix using Direct Linear Transform (DLT).

    Args:
        pts1: First set of points (N, 2)
        pts2: Second set of points (N, 2)

    Returns:
        Homography matrix (3, 3) or None if computation fails
    """
    # Normalize points
    pts1_normalized, T1 = normalize_points(pts1)
    pts2_normalized, T2 = normalize_points(pts2)

    # Build the constraint matrix
    n = pts1.shape[0]
    A = torch.zeros((2 * n, 9), device=pts1.device)

    for i in range(n):
        x1, y1 = pts1_normalized[i]
        x2, y2 = pts2_normalized[i]

        A[2 * i] = torch.tensor(
            [0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1, y2], device=pts1.device
        )
        A[2 * i + 1] = torch.tensor(
            [x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2], device=pts1.device
        )

    # Solve Ah = 0 using SVD
    try:
        _, _, Vt = torch.svd(A)
        h = Vt[-1]  # Last row of Vt is the solution
        H = h.reshape(3, 3)

        # Denormalize
        H = torch.matmul(T2.inverse(), torch.matmul(H, T1))

        # Normalize to ensure H[2,2] = 1
        H = H / (H[2, 2] + 1e-10)

        return H
    except RuntimeError:
        # SVD computation failed
        return None


def normalize_points(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize points for better numerical stability.

    Args:
        points: Points to normalize (N, 2)

    Returns:
        Tuple of (normalized_points, transformation_matrix)
    """
    # Compute centroid
    centroid = torch.mean(points, dim=0)

    # Center points
    centered = points - centroid

    # Compute average distance from origin
    avg_dist = torch.mean(torch.sqrt(torch.sum(centered**2, dim=1)))

    # Scale factor to make average distance sqrt(2)
    scale = np.sqrt(2) / (avg_dist + 1e-8)

    # Create transformation matrix
    T = torch.eye(3, device=points.device)
    T[0, 0] = T[1, 1] = scale
    T[0, 2] = -scale * centroid[0]
    T[1, 2] = -scale * centroid[1]

    # Apply transformation
    normalized = centered * scale

    return normalized, T


def compute_epipolar_distances(
    F: torch.Tensor, pts1: torch.Tensor, pts2: torch.Tensor
) -> torch.Tensor:
    """
    Compute Sampson distances from points to their corresponding epipolar lines.

    Args:
        F: Fundamental matrix (3, 3)
        pts1: First set of points (N, 2)
        pts2: Second set of points (N, 2)

    Returns:
        Sampson distances
    """
    # Convert to homogeneous coordinates
    pts1_homog = torch.cat(
        [pts1, torch.ones(pts1.shape[0], 1, device=pts1.device)], dim=1
    )
    pts2_homog = torch.cat(
        [pts2, torch.ones(pts2.shape[0], 1, device=pts2.device)], dim=1
    )

    # Compute epipolar lines
    lines2 = torch.matmul(F, pts1_homog.t()).t()  # l' = F * x
    lines1 = torch.matmul(F.t(), pts2_homog.t()).t()  # l = F^T * x'

    # Compute numerators: x'^T * F * x
    numerators = torch.sum(pts2_homog * torch.matmul(F, pts1_homog.t()).t(), dim=1)
    numerators = torch.abs(numerators)

    # Compute denominators
    denominators = (
        lines1[:, 0] ** 2 + lines1[:, 1] ** 2 + lines2[:, 0] ** 2 + lines2[:, 1] ** 2
    )

    # Compute Sampson distances
    distances = numerators**2 / (denominators + 1e-10)

    return distances


def compute_homography_distances(
    H: torch.Tensor, pts1: torch.Tensor, pts2: torch.Tensor
) -> torch.Tensor:
    """
    Compute reprojection distances for homography.

    Args:
        H: Homography matrix (3, 3)
        pts1: First set of points (N, 2)
        pts2: Second set of points (N, 2)

    Returns:
        Reprojection distances
    """
    # Convert to homogeneous coordinates
    pts1_homog = torch.cat(
        [pts1, torch.ones(pts1.shape[0], 1, device=pts1.device)], dim=1
    )

    # Apply homography: pts1 -> pts2
    pts2_proj_homog = torch.matmul(H, pts1_homog.t()).t()

    # Convert back to inhomogeneous coordinates
    pts2_proj = pts2_proj_homog[:, :2] / pts2_proj_homog[:, 2:3]

    # Compute reprojection error
    forward_error = torch.sqrt(torch.sum((pts2 - pts2_proj) ** 2, dim=1))

    # Also compute reverse error (pts2 -> pts1)
    pts2_homog = torch.cat(
        [pts2, torch.ones(pts2.shape[0], 1, device=pts2.device)], dim=1
    )
    pts1_proj_homog = torch.matmul(torch.inverse(H), pts2_homog.t()).t()
    pts1_proj = pts1_proj_homog[:, :2] / pts1_proj_homog[:, 2:3]
    backward_error = torch.sqrt(torch.sum((pts1 - pts1_proj) ** 2, dim=1))

    # Return symmetric reprojection error
    return (forward_error + backward_error) / 2
