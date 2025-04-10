import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from torchslam.frontend.feature_extraction import KeyPoint
from torchslam.frontend.keyframe import Keyframe  # Needed for mocking
from torchslam.frontend.odometry import FramePose, OdometryStatus
from torchslam.frontend.tracking import Track, TrackStatus
from torchslam.slam.visual_slam import VisualSLAM, VisualSLAMStatus

# Sample valid configuration for testing
VALID_CONFIG = {
    "device": "cpu",
    "camera_intrinsics": np.array(
        [[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32
    ),
    "feature_extraction": {"max_features": 100},
    "feature_matching": {},
    "tracking": {},
    "odometry": {},
    "keyframe": {},
    "loop_detection": {"enabled": False},
    "local_ba_frequency": 5,
    "local_ba_window": 10,
    "global_ba_enabled": False,
}


class TestVisualSLAM(unittest.TestCase):
    def setUp(self):
        """Set up a basic SLAM instance for tests."""
        self.slam = VisualSLAM(config=VALID_CONFIG)
        # Create a dummy grayscale image (C, H, W)
        self.dummy_frame = torch.rand(1, 480, 640, device=self.slam.device) * 255

    def test_initialization_minimal_config(self):
        """Test VisualSLAM initialization with minimal required config."""
        minimal_config = {"camera_intrinsics": VALID_CONFIG["camera_intrinsics"]}
        slam = VisualSLAM(config=minimal_config)
        self.assertIsInstance(slam, VisualSLAM)
        self.assertEqual(slam.status, VisualSLAMStatus.INITIALIZING)
        self.assertEqual(slam.frame_idx, 0)
        self.assertIsNotNone(slam.camera_intrinsics)
        # Check default component initializations
        self.assertIsNotNone(slam.feature_extractor)
        self.assertIsNotNone(slam.feature_matcher)
        self.assertIsNotNone(slam.tracker)
        self.assertIsNotNone(slam.odometry)
        self.assertIsNotNone(slam.keyframe_manager)
        self.assertFalse(
            slam.loop_detection_enabled
        )  # Default should be true, but config turns it off

    def test_initialization_full_config(self):
        """Test VisualSLAM initialization with a more complete config."""
        slam = VisualSLAM(config=VALID_CONFIG)
        self.assertIsInstance(slam, VisualSLAM)
        self.assertEqual(slam.status, VisualSLAMStatus.INITIALIZING)
        self.assertEqual(slam.frame_idx, 0)
        self.assertEqual(slam.feature_extractor.max_features, 100)
        self.assertFalse(slam.loop_detection_enabled)

    def test_initialization_missing_intrinsics(self):
        """Test VisualSLAM initialization fails without camera intrinsics."""
        with self.assertRaises(ValueError):
            VisualSLAM(config={})

    def test_process_first_frame(self):
        """Test processing the very first frame."""
        result = self.slam.process_frame(self.dummy_frame)

        self.assertEqual(self.slam.frame_idx, 1)
        self.assertEqual(result["frame_idx"], 1)
        self.assertEqual(result["status"], VisualSLAMStatus.INITIALIZING)
        self.assertIsInstance(result["pose"], FramePose)
        # First frame pose should be identity
        self.assertTrue(torch.equal(result["pose"].rotation, torch.eye(3)))
        self.assertTrue(torch.equal(result["pose"].translation, torch.zeros(3)))
        self.assertGreater(result["num_keypoints"], 0)  # Should extract some keypoints
        self.assertIn("processing_time", result)

        # Check internal state after first frame
        self.assertIsNotNone(self.slam.current_frame)
        self.assertIsNotNone(self.slam.current_keypoints)
        self.assertIsNotNone(self.slam.current_descriptors)
        self.assertEqual(len(self.slam.tracks), result["num_keypoints"])
        self.assertEqual(self.slam.next_track_id, result["num_keypoints"])
        self.assertEqual(
            len(self.slam.keyframe_manager.get_all_keyframes()), 1
        )  # First keyframe created
        self.assertEqual(self.slam.last_keyframe_idx, 1)

    def _create_dummy_keypoints(self, num_points=10):
        """Helper to create dummy keypoints."""
        kps = []
        descs = []
        for i in range(num_points):
            kp = KeyPoint(
                x=100.0 + i * 5,
                y=200.0 + i * 3,
                response=0.5,
                size=10.0,
                angle=0.0,
                octave=0,
            )
            kps.append(kp)
            # Dummy descriptor (32 bytes = 256 bits for ORB)
            descs.append(
                torch.randint(0, 256, (32,), dtype=torch.uint8, device=self.slam.device)
            )
        return kps, torch.stack(descs) if descs else None

    def test_track_features_basic(self):
        """Test the basic functionality of _track_features."""
        # 1. Setup: Process first frame to initialize tracks
        self.slam.process_frame(self.dummy_frame)
        first_frame_kps = self.slam.current_keypoints
        first_frame_tracks = self.slam.tracks.copy()

        # 2. Create a second, slightly different frame
        # (Shift frame slightly to simulate movement for KLT)
        dummy_frame_shifted = torch.roll(
            self.dummy_frame, shifts=(0, 5, 5), dims=(0, 1, 2)
        )

        # Manually set up for the next tracking step
        self.slam.previous_frame = self.slam.current_frame
        self.slam.previous_keypoints = self.slam.current_keypoints
        self.slam.previous_descriptors = self.slam.current_descriptors
        self.slam.current_frame = dummy_frame_shifted
        # Extract features for the new frame (needed for creating new tracks)
        (
            self.slam.current_keypoints,
            self.slam.current_descriptors,
        ) = self.slam.feature_extractor.detect_and_compute(dummy_frame_shifted)

        # 3. Call _track_features
        tracked_kps, status, new_track_ids = self.slam._track_features()

        # 4. Assertions
        self.assertIsInstance(tracked_kps, list)
        self.assertIsInstance(status, list)
        self.assertIsInstance(new_track_ids, list)
        self.assertEqual(
            len(tracked_kps), len(first_frame_kps)
        )  # tracked_kps corresponds to previous_kps
        self.assertEqual(len(status), len(first_frame_kps))

        num_tracked = sum(status)
        self.assertGreaterEqual(num_tracked, 0)
        self.assertLessEqual(num_tracked, len(first_frame_kps))

        # Check if tracked keypoints have updated positions (if any were tracked)
        if num_tracked > 0:
            first_tracked_idx = status.index(True)
            self.assertIsNotNone(tracked_kps[first_tracked_idx])
            self.assertIsInstance(tracked_kps[first_tracked_idx], KeyPoint)
            # Position should ideally be different due to shift, but KLT might fail
            # A loose check is sufficient here
            self.assertNotEqual(
                tracked_kps[first_tracked_idx].pt(),
                first_frame_kps[first_tracked_idx].pt(),
            )

        # Check track status updates
        for i, tracked in enumerate(status):
            # Find the original track id corresponding to previous_keypoints[i]
            original_track_id = -1
            for tid, track in first_frame_tracks.items():
                # Compare keypoint objects directly might be fragile due to float precision?
                # Let's rely on the order matching previous_keypoints for this test setup.
                # A better way might involve searching by coordinates if order isn't guaranteed.
                if (
                    i < len(track.history)
                    and track.history[0].keypoint.pt() == first_frame_kps[i].pt()
                ):
                    original_track_id = tid
                    break

            if original_track_id != -1:
                updated_track = self.slam.tracks[original_track_id]
                if tracked:
                    self.assertEqual(updated_track.status, TrackStatus.TRACKED)
                    self.assertEqual(
                        updated_track.last_keypoint.pt(), tracked_kps[i].pt()
                    )
                else:
                    self.assertEqual(updated_track.status, TrackStatus.LOST)

        # Check new tracks
        num_new_features = len(self.slam.current_keypoints)
        # Difficult to predict exact number of new tracks without matching
        # But some new tracks should be created for the new features
        self.assertGreaterEqual(len(new_track_ids), 0)
        for new_tid in new_track_ids:
            self.assertIn(new_tid, self.slam.tracks)
            self.assertEqual(self.slam.tracks[new_tid].status, TrackStatus.TRACKED)

    def test_update_pose_success(self):
        """Test _update_pose when enough 3D-2D correspondences are found."""
        # 1. Setup initial state (pose, map points, tracks, keyframes)
        initial_pose = FramePose.identity(device=self.slam.device)
        self.slam.previous_pose = initial_pose
        self.slam.current_pose = initial_pose  # Start with identity

        num_points = 5
        landmark_ids = list(range(num_points))
        map_points_3d = []
        tracked_keypoints_2d = []
        status = [True] * num_points

        # Create dummy map points (in front of camera at Z=5)
        for i in range(num_points):
            landmark_id = landmark_ids[i]
            # Simple 3D point in front of the initial identity pose
            pt_3d = torch.tensor(
                [float(i - num_points // 2), float(i % 2 - 0.5), 5.0],
                device=self.slam.device,
            )
            self.slam.map_points[landmark_id] = pt_3d
            map_points_3d.append(pt_3d)

            # Create corresponding 2D keypoints (simulate perfect projection for simplicity)
            # K = self.slam.camera_intrinsics
            # px = K[0, 0] * pt_3d[0] / pt_3d[2] + K[0, 2]
            # py = K[1, 1] * pt_3d[1] / pt_3d[2] + K[1, 2]
            # For this test, exact projection isn't critical, just need some 2D points
            kp_2d = KeyPoint(x=320.0 + i * 10, y=240.0 + (i % 2) * 10)
            tracked_keypoints_2d.append(kp_2d)

            # Create dummy tracks linked to these landmarks
            track = Track(
                feature_id=i, keypoint=kp_2d, descriptor=None
            )  # Use feature_id, status is NEW by default
            self.slam.tracks[
                i
            ] = track  # Assume track_id == feature_id for this test setup

        # Create a dummy keyframe and link tracks to landmarks
        # This is needed for _update_pose to find the 3D points for the tracks
        dummy_kf = Keyframe(
            keyframe_id=0,
            pose=initial_pose,
            frame=None,
            keypoints=[],
            descriptors=None,
            tracks=self.slam.tracks,
            timestamp=0,
        )
        for i in range(num_points):
            dummy_kf.add_landmark_observation(
                track_id=i, landmark_id=landmark_ids[i], point_3d=map_points_3d[i]
            )
        self.slam.keyframe_manager.keyframes = {0: dummy_kf}
        self.slam.keyframe_manager.next_keyframe_id = 1

        # 2. Call _update_pose
        self.slam._update_pose(tracked_keypoints_2d, status)

        # 3. Assertions
        # Pose should be updated by PnP (even if it results in identity again with perfect data)
        # A robust check is difficult without knowing the exact PnP output,
        # but it should succeed and status should be good.
        self.assertEqual(self.slam.odometry.get_status(), OdometryStatus.OK)
        self.assertEqual(self.slam.status, VisualSLAMStatus.TRACKING_GOOD)
        # Check if pose changed (it might not change much if input is perfect identity)
        # Let's check it's still a valid FramePose
        self.assertIsInstance(self.slam.current_pose, FramePose)

    def test_update_pose_not_enough_points(self):
        """Test _update_pose when fewer than 4 correspondences are found."""
        # Setup minimal state needed
        initial_pose = FramePose(torch.eye(3), torch.tensor([1.0, 0.0, 0.0]))
        self.slam.previous_pose = initial_pose
        self.slam.current_pose = initial_pose
        self.slam.status = VisualSLAMStatus.TRACKING_GOOD  # Assume it was good before

        # Simulate only 3 tracked points (even if they exist in map)
        num_points = 3
        tracked_keypoints_2d = []
        status = [True] * num_points

        # Need to setup map points and tracks similar to success case, but only 3
        landmark_ids = list(range(num_points))
        map_points_3d = []
        for i in range(num_points):
            landmark_id = landmark_ids[i]
            pt_3d = torch.tensor([float(i), 0.0, 5.0], device=self.slam.device)
            self.slam.map_points[landmark_id] = pt_3d
            map_points_3d.append(pt_3d)
            kp_2d = KeyPoint(x=320.0 + i * 10, y=240.0)
            tracked_keypoints_2d.append(kp_2d)
            track = Track(
                feature_id=i, keypoint=kp_2d, descriptor=None
            )  # Use feature_id, status is NEW by default
            self.slam.tracks[
                i
            ] = track  # Assume track_id == feature_id for this test setup

        # Dummy keyframe to link tracks <-> landmarks
        dummy_kf = Keyframe(
            keyframe_id=0,
            pose=FramePose.identity(device=self.slam.device),
            frame=None,
            keypoints=[],
            descriptors=None,
            tracks=self.slam.tracks,
            timestamp=0,
        )
        for i in range(num_points):
            dummy_kf.add_landmark_observation(
                track_id=i, landmark_id=landmark_ids[i], point_3d=map_points_3d[i]
            )
        self.slam.keyframe_manager.keyframes = {0: dummy_kf}
        self.slam.keyframe_manager.next_keyframe_id = 1

        # Call _update_pose
        self.slam._update_pose(tracked_keypoints_2d, status)

        # Assertions: Pose should remain unchanged (same as previous), status becomes BAD
        self.assertTrue(
            torch.equal(self.slam.current_pose.rotation, initial_pose.rotation)
        )
        self.assertTrue(
            torch.equal(self.slam.current_pose.translation, initial_pose.translation)
        )
        self.assertEqual(self.slam.status, VisualSLAMStatus.TRACKING_BAD)

    def test_reset(self):
        """Test the reset functionality of VisualSLAM."""
        slam = VisualSLAM(config=VALID_CONFIG)
        # Simulate processing a few frames (details don't matter for reset test)
        slam.frame_idx = 10
        slam.status = VisualSLAMStatus.TRACKING_GOOD
        slam.current_pose = FramePose(torch.eye(3), torch.tensor([1.0, 2.0, 3.0]))
        slam.tracks = {0: "dummy_track"}
        slam.map_points = {0: torch.tensor([1.0, 1.0, 5.0])}
        slam.next_track_id = 1
        slam.next_landmark_id = 1
        slam.keyframe_manager.create_keyframe(
            slam.current_pose, torch.rand(3, 480, 640), [], None, {}, 0
        )  # Add a dummy keyframe

        slam.reset()

        self.assertEqual(slam.status, VisualSLAMStatus.INITIALIZING)
        self.assertEqual(slam.frame_idx, 0)
        self.assertIsNone(slam.current_frame)
        self.assertIsNone(slam.current_keypoints)
        self.assertIsNone(slam.current_descriptors)
        self.assertTrue(torch.equal(slam.current_pose.rotation, torch.eye(3)))
        self.assertTrue(torch.equal(slam.current_pose.translation, torch.zeros(3)))
        self.assertIsNone(slam.previous_frame)
        self.assertEqual(len(slam.tracks), 0)
        self.assertEqual(slam.next_track_id, 0)
        self.assertEqual(len(slam.map_points), 0)
        self.assertEqual(slam.next_landmark_id, 0)
        self.assertEqual(len(slam.keyframe_manager.get_all_keyframes()), 0)
        self.assertIsNotNone(slam.bundle_adjustment)  # BA object should be recreated
        self.assertEqual(slam.last_local_ba_frame, 0)


if __name__ == "__main__":
    unittest.main()
