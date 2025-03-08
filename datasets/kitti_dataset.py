import glob
import logging
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from dataset.base_sensor import *
from PIL import Image
from torch.utils.data import Dataset

from .base_dataset import BaseDataset, CalibrationData, SensorData, SensorType


class KITTIOdometryDataset(BaseDataset):
    """
    KITTI Odometry dataset loader.

    The KITTI Odometry dataset contains 22 sequences (00-21), with ground truth available
    for sequences 00-10. Each sequence includes stereo images, LiDAR scans, and calibration data.

    Directory structure:
    dataset_path/
        ├── sequences/
        │   ├── 00/
        │   │   ├── image_0/              # Left grayscale images
        │   │   ├── image_1/              # Right grayscale images
        │   │   ├── image_2/              # Left color images
        │   │   ├── image_3/              # Right color images
        │   │   ├── velodyne/             # LiDAR point clouds
        │   │   └── calib.txt             # Calibration data
        │   ├── 01/
        │   │   └── ...
        │   └── ...
        └── poses/
            ├── 00.txt                    # Ground truth poses for sequence 00
            ├── 01.txt                    # Ground truth poses for sequence 01
            └── ...
    """

    def __init__(
        self,
        dataset_path: Union[str, Path],
        sequence_id: Union[str, int] = "00",
        sensors: Optional[List[SensorType]] = None,
        temporal_window: int = 1,
        transforms: Optional[Dict] = None,
        cache_size: int = 100,
        load_calibration: bool = True,
        use_color: bool = True,
        use_lidar: bool = False,
        return_stereo: bool = True,
        max_points: Optional[int] = None,
    ):
        """
        Initialize KITTI Odometry dataset.

        Args:
            dataset_path: Path to the KITTI dataset root
            sequence_id: Sequence ID ('00' to '21')
            sensors: List of sensor types to load
            temporal_window: Number of frames to include in each sample
            transforms: Dictionary of transforms to apply to each sensor data
            cache_size: Maximum number of samples to keep in memory
            load_calibration: Whether to load calibration data
            use_color: Whether to use color images (image_2, image_3) or grayscale (image_0, image_1)
            use_lidar: Whether to use LiDAR
            return_stereo: Whether to return both left and right images
            max_points: Maximum number of LiDAR points to return (None for all)
        """
        self.use_color = use_color
        self.return_stereo = return_stereo
        self.max_points = max_points

        # Convert dataset_path to Path if needed
        self.dataset_path = (
            Path(dataset_path) if not isinstance(dataset_path, Path) else dataset_path
        )

        # Ensure sequence_id is a 2-digit string
        if isinstance(sequence_id, int):
            sequence_id = f"{sequence_id:02d}"
        self.sequence_id = sequence_id

        # Sensor mapping
        self.sensor_mapping = {
            "left_camera": "image_2" if use_color else "image_0",
            "right_camera": "image_3" if use_color else "image_1",
            "lidar": "velodyne",
        }

        # Default sensor types if not specified
        if sensors is None:
            if return_stereo:
                sensors = [SensorType.STEREO_CAMERA, SensorType.LIDAR]
            else:
                sensors = [SensorType.RGB_CAMERA if use_color else SensorType.LIDAR]
        self.sensors = sensors

        # Initialize calibration container
        self.calibration_data = {}

        super().__init__(
            dataset_path=self.dataset_path,
            sequence_id=self.sequence_id,
            sensors=self.sensors,
            temporal_window=temporal_window,
            transforms=transforms,
            cache_size=cache_size,
            load_calibration=load_calibration,
        )

        # Load ground truth poses if available
        self.poses = self._load_poses()

    def _load_metadata(self):
        """Load KITTI dataset metadata."""
        sequence_path = self.dataset_path / "sequences" / self.sequence_id
        if not sequence_path.exists():
            raise ValueError(
                f"Sequence {self.sequence_id} not found at {sequence_path}"
            )

        # Count image files for left camera
        left_cam_path = sequence_path / self.sensor_mapping["left_camera"]
        if not left_cam_path.exists():
            raise ValueError(f"Left camera images not found at {left_cam_path}")

        left_cam_files = sorted(left_cam_path.glob("*.png"))
        num_frames = len(left_cam_files)
        if num_frames == 0:
            raise ValueError(f"No image files found in {left_cam_path}")

        # Check LiDAR files if LiDAR sensor is requested
        lidar_files = []
        lidar_path = sequence_path / self.sensor_mapping["lidar"]
        if lidar_path.exists():
            lidar_files = sorted(lidar_path.glob("*.bin"))
            if len(lidar_files) != num_frames:
                logging.warning(
                    f"Number of LiDAR scans ({len(lidar_files)}) doesn't match number of images ({num_frames})"
                )
                num_frames = min(num_frames, len(lidar_files))
        else:
            if SensorType.LIDAR in self.sensors:
                logging.warning(f"LiDAR data requested but not found at {lidar_path}")
                self.sensors.remove(SensorType.LIDAR)

        # Set data indices
        self.data_indices = {"left_camera": num_frames}
        if self.return_stereo:
            self.data_indices["right_camera"] = num_frames
        if SensorType.LIDAR in self.sensors:
            self.data_indices["lidar"] = num_frames

        # Store file paths for efficient loading
        self.file_paths = {"left_camera": [str(f) for f in left_cam_files[:num_frames]]}
        if self.return_stereo:
            right_cam_path = sequence_path / self.sensor_mapping["right_camera"]
            right_cam_files = sorted(right_cam_path.glob("*.png"))
            self.file_paths["right_camera"] = [
                str(f) for f in right_cam_files[:num_frames]
            ]
        if SensorType.LIDAR in self.sensors:
            self.file_paths["lidar"] = [str(f) for f in lidar_files[:num_frames]]

    def _load_calibration(self):
        """Load KITTI calibration data."""
        calib_file = self.dataset_path / "sequences" / self.sequence_id / "calib.txt"
        if not calib_file.exists():
            logging.warning(f"Calibration file not found: {calib_file}")
            return

        calib_data = {}
        with open(calib_file, "r") as f:
            for line in f:
                key, value = line.split(":", 1)
                calib_data[key.strip()] = np.array(
                    [float(x) for x in value.strip().split()]
                )

        # Select calibration keys based on color usage
        if self.use_color:
            left_key = "P2"
            right_key = "P3"
        else:
            left_key = "P0"
            right_key = "P1"

        # Left camera calibration
        if left_key in calib_data:
            P_left = calib_data[left_key].reshape(3, 4)
            calib = CalibrationData(
                "left_camera",
                SensorType.RGB_CAMERA if self.use_color else SensorType.STEREO_CAMERA,
            )
            calib.set_intrinsics(P_left[:, :3])
            calib.set_extrinsics(np.eye(4))  # Identity transform (reference frame)
            self.calibration_data["left_camera"] = calib

        # Right camera calibration
        if right_key in calib_data and self.return_stereo:
            P_right = calib_data[right_key].reshape(3, 4)
            calib = CalibrationData(
                "right_camera",
                SensorType.RGB_CAMERA if self.use_color else SensorType.STEREO_CAMERA,
            )
            calib.set_intrinsics(P_right[:, :3])
            baseline = P_right[0, 3] / P_right[0, 0]
            extrinsics = np.eye(4)
            extrinsics[
                0, 3
            ] = (
                -baseline
            )  # Negative because right camera is to the left in vehicle frame
            calib.set_extrinsics(extrinsics)
            self.calibration_data["right_camera"] = calib

        # LiDAR to camera transform
        if "Tr" in calib_data and SensorType.LIDAR in self.sensors:
            Tr = np.eye(4)
            Tr[:3, :4] = calib_data["Tr"].reshape(3, 4)
            calib = CalibrationData("lidar", SensorType.LIDAR)
            calib.set_extrinsics(Tr)
            self.calibration_data["lidar"] = calib

    def _load_poses(self):
        """Load ground truth poses if available."""
        pose_file = self.dataset_path / "poses" / f"{self.sequence_id}.txt"
        if not pose_file.exists():
            logging.warning(f"Ground truth poses not found: {pose_file}")
            return None

        poses = []
        with open(pose_file, "r") as f:
            for line in f:
                # Each line contains 12 values for the 3x4 pose matrix
                values = [float(x) for x in line.strip().split()]
                pose = np.eye(4)
                pose[:3, :4] = np.array(values).reshape(3, 4)
                poses.append(pose)
        return np.array(poses)

    def _load_sensor_data(self, sensor_id: str, idx: int) -> Any:
        """Load data for a specific sensor at a specific index."""
        if idx >= len(self.file_paths[sensor_id]):
            raise IndexError(f"Index {idx} out of range for sensor {sensor_id}")

        file_path = self.file_paths[sensor_id][idx]

        if sensor_id in ["left_camera", "right_camera"]:
            # Load image
            image = cv2.imread(
                file_path, cv2.IMREAD_COLOR if self.use_color else cv2.IMREAD_GRAYSCALE
            )
            if image is None:
                raise IOError(f"Failed to load image: {file_path}")

            if self.use_color:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(image.astype(np.float32) / 255.0)

            if not self.use_color:
                image = image.unsqueeze(0)
            else:
                image = image.permute(2, 0, 1)

            return image

        elif sensor_id == "lidar":
            # Load LiDAR point cloud
            points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
            if self.max_points is not None and points.shape[0] > self.max_points:
                indices = np.random.choice(
                    points.shape[0], self.max_points, replace=False
                )
                points = points[indices]
            return torch.from_numpy(points)

        else:
            raise ValueError(f"Unknown sensor type: {sensor_id}")

    def get_timestamp(self, sensor_id: str, idx: int) -> Optional[float]:
        """
        Get timestamp for a specific sensor at a specific index.
        KITTI doesn't provide explicit timestamps, so we return the frame index instead.
        """
        return float(idx)

    def get_pose_at_index(self, idx: int) -> Optional[np.ndarray]:
        """Get the ground truth pose at a specific index."""
        if self.poses is None or idx >= len(self.poses):
            return None
        return self.poses[idx]

    def get_sequence_length(self) -> int:
        """Get the number of frames in the sequence."""
        return len(self)

    def get_camera_intrinsics(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get camera intrinsics for left and right cameras."""
        left_calib = self.calibration_data.get("left_camera")
        right_calib = (
            self.calibration_data.get("right_camera") if self.return_stereo else None
        )
        left_intrinsics = left_calib.intrinsics if left_calib is not None else None
        right_intrinsics = right_calib.intrinsics if right_calib is not None else None
        return left_intrinsics, right_intrinsics

    def get_lidar_to_camera_transform(self) -> Optional[np.ndarray]:
        """Get transform from LiDAR to camera frame."""
        if "lidar" in self.calibration_data:
            return self.calibration_data["lidar"].extrinsics
        return None

    def get_camera_baseline(self) -> Optional[float]:
        """Get stereo camera baseline."""
        if self.return_stereo and "right_camera" in self.calibration_data:
            extrinsics = self.calibration_data["right_camera"].extrinsics
            return abs(extrinsics[0, 3])
        return None

    def get_trajectory(self) -> Optional[np.ndarray]:
        """Get full trajectory of the sequence."""
        return self.poses


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create dataset
    dataset = KITTIOdometryDataset(
        dataset_path="/path/to/kitti/dataset",
        sequence_id="00",
        use_color=True,
        return_stereo=True,
    )

    # Print dataset info
    print(f"Dataset length: {len(dataset)}")
    print(f"Camera baseline: {dataset.get_camera_baseline()} meters")

    # Get a sample
    sample = dataset[0]
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}, {value.dtype}")
        else:
            print(f"{key}: {type(value)}")

    # Get ground truth pose
    pose = dataset.get_pose_at_index(0)
    if pose is not None:
        print(f"Ground truth pose:\n{pose}")
