import json
import logging
import os
import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from dataset.base_sensor import *
from torch.utils.data import Dataset


class CalibrationData:
    """Class to store sensor calibration data"""

    def __init__(self, sensor_id: str, sensor_type: SensorType):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.intrinsics = None
        self.extrinsics = None  # Transform from sensor to base frame
        self.distortion = None
        self.additional_params = {}

    def set_intrinsics(self, intrinsics: np.ndarray):
        """Set camera intrinsic parameters."""
        self.intrinsics = intrinsics

    def set_extrinsics(self, extrinsics: np.ndarray):
        """Set extrinsic parameters (transform from sensor to base frame)."""
        self.extrinsics = extrinsics

    def set_distortion(self, distortion: np.ndarray):
        """Set distortion parameters."""
        self.distortion = distortion

    def add_param(self, name: str, value: Any):
        """Add additional calibration parameter."""
        self.additional_params[name] = value

    def to_dict(self) -> Dict:
        """Convert calibration data to dictionary for serialization."""
        return {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type.name,
            "intrinsics": self.intrinsics.tolist()
            if isinstance(self.intrinsics, np.ndarray)
            else self.intrinsics,
            "extrinsics": self.extrinsics.tolist()
            if isinstance(self.extrinsics, np.ndarray)
            else self.extrinsics,
            "distortion": self.distortion.tolist()
            if isinstance(self.distortion, np.ndarray)
            else self.distortion,
            "additional_params": self.additional_params,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CalibrationData":
        """Create a CalibrationData object from a dictionary."""
        sensor_type = SensorType[data["sensor_type"]]
        calib = cls(data["sensor_id"], sensor_type)
        if data["intrinsics"] is not None:
            calib.set_intrinsics(np.array(data["intrinsics"]))
        if data["extrinsics"] is not None:
            calib.set_extrinsics(np.array(data["extrinsics"]))
        if data["distortion"] is not None:
            calib.set_distortion(np.array(data["distortion"]))
        calib.additional_params = data["additional_params"]
        return calib


class BaseDataset(Dataset, ABC):
    """
    Base dataset class for SLAM datasets.
    Handles sensor calibration and provides common functionality.
    """

    def __init__(
        self,
        dataset_path: str,
        sequence_id: str = None,
        sensors: List[SensorType] = None,
        temporal_window: int = 1,
        transforms: Dict = None,
        cache_size: int = 100,
        load_calibration: bool = True,
    ):
        """
        Initialize the base dataset.

        Args:
            dataset_path: Path to the dataset root
            sequence_id: Specific sequence identifier
            sensors: List of sensor types to load
            temporal_window: Number of frames to include in each sample
            transforms: Dictionary of transforms to apply to each sensor data
            cache_size: Maximum number of samples to keep in memory
            load_calibration: Whether to load calibration data
        """
        self.dataset_path = Path(dataset_path)
        self.sequence_id = sequence_id
        self.sensors = sensors if sensors is not None else []
        self.temporal_window = temporal_window
        self.transforms = transforms if transforms is not None else {}
        self.cache_size = cache_size

        # Initialize data structures
        self.calibration_data = {}  # sensor_id -> CalibrationData
        self.data_indices = {}  # sensor_id -> number of data points
        self.data_cache = {}  # (index, sensor_id) -> data

        # Load dataset metadata
        self._load_metadata()

        # Load calibration if requested
        if load_calibration:
            self._load_calibration()

        # Log basic dataset information
        self._log_dataset_info()

    @abstractmethod
    def _load_metadata(self):
        """
        Load dataset-specific metadata.
        This should populate basic information about the sequence and available sensors.
        Must be implemented by subclasses to set self.data_indices with number of samples per sensor.
        """
        pass

    def _load_calibration(self):
        """
        Load calibration data from files.
        This provides a default implementation that looks for calibration files
        in common formats (YAML, JSON).
        """
        raise NotImplementedError

    def _log_dataset_info(self):
        """Log basic information about the dataset."""
        logging.info(f"Dataset path: {self.dataset_path}")
        logging.info(f"Sequence: {self.sequence_id}")
        logging.info(f"Number of sensors: {len(self.data_indices)}")
        for sensor_id, count in self.data_indices.items():
            logging.info(f"  {sensor_id}: {count} data points")
        logging.info(f"Total samples: {len(self)}")

    def get_calibration(self, sensor_id: str) -> Optional[CalibrationData]:
        """Get calibration data for a specific sensor."""
        return self.calibration_data.get(sensor_id)

    def get_all_calibrations(self) -> Dict[str, CalibrationData]:
        """Get all calibration data."""
        return self.calibration_data

    @abstractmethod
    def _load_sensor_data(self, sensor_id: str, idx: int) -> Any:
        """
        Load data for a specific sensor at a specific index.
        To be implemented by subclasses for each specific dataset format.
        """
        raise NotImplementedError

    def get_timestamp(self, sensor_id: str, idx: int) -> Optional[float]:
        """
        Get timestamp for a specific sensor at a specific index.
        To be implemented by subclasses if timestamps are available.
        """
        raise NotImplementedError

    def apply_transforms(self, data: Any, sensor_id: str) -> Any:
        """Apply transforms to sensor data."""
        if sensor_id in self.transforms and self.transforms[sensor_id] is not None:
            for transform in self.transforms[sensor_id]:
                data = transform(data)
        return data

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        By default, uses the length of the first sensor in data_indices.
        Subclasses may override this for dataset-specific behavior.
        """
        if not self.data_indices:
            return 0
        # Default behavior: use the first sensor's count
        return next(iter(self.data_indices.values()))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a data sample by index.

        Returns:
            Dictionary mapping sensor_id to sensor data
        """
        result = {}

        # Load data for the primary sensor
        primary_sensor = next(iter(self.data_indices.keys()))

        # Try to get timestamp if available
        timestamp = self.get_timestamp(primary_sensor, idx)
        if timestamp is not None:
            result["timestamp"] = timestamp

        # Load data for each sensor
        for sensor_id in self.data_indices.keys():
            # Skip sensors that don't have this index
            if idx >= self.data_indices[sensor_id]:
                continue

            cache_key = (idx, sensor_id)

            # Try to get from cache first
            if cache_key in self.data_cache:
                data = self.data_cache[cache_key]
            else:
                # Load and apply transforms
                data = self._load_sensor_data(sensor_id, idx)
                data = self.apply_transforms(data, sensor_id)

                # Add to cache
                if len(self.data_cache) >= self.cache_size:
                    # Remove oldest item from cache
                    self.data_cache.pop(next(iter(self.data_cache)))
                self.data_cache[cache_key] = data

            result[sensor_id] = data

        # Handle temporal window
        if self.temporal_window > 1:
            for offset in range(1, self.temporal_window):
                if idx + offset < len(self):
                    next_item = self[idx + offset]
                    for sensor_id in next_item:
                        if sensor_id != "timestamp":
                            result[f"{sensor_id}_{offset}"] = next_item[sensor_id]

        return result

    def get_sequence_info(self) -> Dict:
        """Get information about the sequence."""
        return {
            "sequence_id": self.sequence_id,
            "num_frames": len(self),
            "sensors": list(self.data_indices.keys()),
        }

    def get_pose_at_index(self, idx: int) -> Optional[np.ndarray]:
        """
        Get the ground truth pose at a specific index.
        To be implemented by subclasses if ground truth is available.

        Returns:
            4x4 transformation matrix or None if not available
        """
        raise NotImplementedError

    def export_calibration(self, output_path: str):
        """Export calibration data to a file."""
        calib_dict = {
            sensor_id: calib.to_dict()
            for sensor_id, calib in self.calibration_data.items()
        }

        with open(output_path, "w") as f:
            json.dump(calib_dict, f, indent=2)

        logging.info(f"Exported calibration data to {output_path}")
