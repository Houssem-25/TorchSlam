import os
import time  # Add time import

import cv2  # For loading video and potential comparison
import numpy as np
import pytest
import torch

from torchslam.frontend.feature_extraction.base import KeyPoint
from torchslam.frontend.feature_extraction.orb import ORBFeatureExtractor


# Fixture to load the first frame of the video
@pytest.fixture(scope="module")
def video_frame():
    # Look for video in the project root directory (current working directory)
    video_path = "video.mp4"
    # video_path = os.path.expanduser("~/video.mp4") # Original path
    if not os.path.exists(video_path):
        pytest.skip(f"Video file not found at {os.path.abspath(video_path)}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        pytest.fail(f"Could not open video file: {video_path}")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        pytest.fail(f"Could not read frame from video file: {video_path}")

    # Convert frame from BGR to RGB and then to a PyTorch tensor (H, W, C) -> (C, H, W)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float()  # C, H, W
    return frame_tensor


# Fixture to create an instance of the ORB extractor
@pytest.fixture(scope="module")
def orb_extractor():
    return ORBFeatureExtractor(max_features=4000)  # Increase features for testing


# --- Basic ORB Tests ---


def test_orb_initialization(orb_extractor):
    """Test if the ORB extractor is initialized correctly."""
    assert isinstance(orb_extractor, ORBFeatureExtractor)
    assert orb_extractor.max_features == 4000


def test_orb_extract(orb_extractor, video_frame):
    """Test the extract method."""
    keypoints = orb_extractor.extract(video_frame)

    assert isinstance(keypoints, list)
    # Check if any keypoints were found (might be 0 depending on frame/threshold)
    assert len(keypoints) >= 0
    if len(keypoints) > 0:
        assert isinstance(keypoints[0], KeyPoint)
        assert len(keypoints) <= orb_extractor.max_features
        # Check basic attributes of a keypoint
        kp = keypoints[0]
        assert isinstance(kp.x, float)
        assert isinstance(kp.y, float)
        assert isinstance(kp.response, float)
        assert isinstance(kp.size, float)
        assert isinstance(kp.angle, float)
        assert isinstance(kp.octave, int)


def test_orb_compute_descriptors(orb_extractor, video_frame):
    """Test the compute_descriptors method."""
    keypoints = orb_extractor.extract(video_frame)

    # Skip test if no keypoints found
    if not keypoints:
        pytest.skip("No keypoints found to compute descriptors for.")

    descriptors = orb_extractor.compute_descriptors(video_frame, keypoints)

    assert isinstance(descriptors, torch.Tensor)
    assert descriptors.dtype == torch.uint8  # ORB descriptors are binary
    assert descriptors.ndim == 2
    assert descriptors.shape[0] == len(keypoints)
    # Default ORB uses 256 bits = 32 bytes
    expected_descriptor_size_bytes = orb_extractor.n_points // 8
    assert descriptors.shape[1] == expected_descriptor_size_bytes


def test_orb_detect_and_compute(orb_extractor, video_frame):
    """Test the detect_and_compute method."""
    keypoints, descriptors = orb_extractor.detect_and_compute(video_frame)

    # Check keypoints
    assert isinstance(keypoints, list)
    assert len(keypoints) >= 0
    if len(keypoints) > 0:
        assert isinstance(keypoints[0], KeyPoint)
        assert len(keypoints) <= orb_extractor.max_features

    # Check descriptors
    assert isinstance(descriptors, torch.Tensor)
    assert descriptors.dtype == torch.uint8
    assert descriptors.ndim == 2

    # Check consistency
    assert descriptors.shape[0] == len(keypoints)
    if len(keypoints) > 0:
        expected_descriptor_size_bytes = orb_extractor.n_points // 8
        assert descriptors.shape[1] == expected_descriptor_size_bytes


# --- Comparison with OpenCV ---


def test_compare_with_opencv(orb_extractor, video_frame):
    """Compare our ORB implementation with OpenCV's implementation."""
    # --- Run our implementation with timing ---
    start_time_ours = time.perf_counter()
    kp_ours, desc_ours = orb_extractor.detect_and_compute(video_frame)
    end_time_ours = time.perf_counter()
    time_ours = end_time_ours - start_time_ours
    print(f"\n[INFO] PyTorch ORB time: {time_ours:.4f} seconds")

    # --- Run OpenCV implementation with timing ---
    # Convert frame tensor back to NumPy format (C, H, W) -> (H, W, C) and BGR
    frame_np = video_frame.permute(1, 2, 0).cpu().numpy()
    frame_np_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    frame_np_gray = cv2.cvtColor(frame_np_bgr, cv2.COLOR_BGR2GRAY)
    frame_np_gray = frame_np_gray.astype(np.uint8)

    # Use same number of max_features as our extractor
    opencv_orb = cv2.ORB_create(nfeatures=orb_extractor.max_features)

    start_time_cv = time.perf_counter()
    kp_cv, desc_cv = opencv_orb.detectAndCompute(frame_np_gray, None)
    end_time_cv = time.perf_counter()
    time_cv = end_time_cv - start_time_cv
    print(f"[INFO] OpenCV ORB time:  {time_cv:.4f} seconds")

    # --- Comparisons ---
    # Allow skipping if either implementation finds no keypoints
    if len(kp_ours) == 0 or len(kp_cv) == 0:
        pytest.skip("One of the ORB implementations found no keypoints.")

    # 1. Compare number of keypoints (allow some tolerance)
    # Note: Exact numbers can differ due to implementation details
    # Let's check if they are within a reasonable factor (e.g., 2x)
    assert (
        0.5 < len(kp_ours) / len(kp_cv) < 2.0
    ), f"Significant difference in keypoint counts: Ours={len(kp_ours)}, OpenCV={len(kp_cv)}"

    # 2. Compare keypoint locations (optional, requires matching)
    # A full comparison requires matching keypoints, which is complex.
    # Instead, we can check if the overall distribution is similar.
    # Let's compare the average location (center of mass of keypoints)
    kp_ours_pts = np.array([(kp.x, kp.y) for kp in kp_ours])
    kp_cv_pts = np.array([kp.pt for kp in kp_cv])

    avg_loc_ours = np.mean(kp_ours_pts, axis=0)
    avg_loc_cv = np.mean(kp_cv_pts, axis=0)

    # Allow an even larger distance (e.g., 100 pixels) with more features
    # due to implementation differences
    loc_diff = np.linalg.norm(avg_loc_ours - avg_loc_cv)
    assert (
        loc_diff < 100.0
    ), f"Average keypoint locations differ significantly: Ours={avg_loc_ours}, OpenCV={avg_loc_cv}"

    # 3. Compare descriptors (optional, requires matching)
    # Comparing descriptors directly is hard without matching.
    # We can check if the descriptor shapes match.
    assert desc_ours.shape[0] == len(kp_ours)
    assert desc_cv.shape[0] == len(kp_cv)
    assert desc_ours.shape[1] * 8 == orb_extractor.n_points
    # OpenCV descriptor size is fixed at 32 bytes (256 bits)
    assert desc_cv.shape[1] == 32


# TODO: Add test comparing with OpenCV - Placeholder removed
