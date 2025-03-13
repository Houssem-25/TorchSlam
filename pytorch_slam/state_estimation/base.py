import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


class State:
    """
    Base class for state representation.

    This class encapsulates the state vector and covariance matrix for
    state estimation algorithms.
    """

    def __init__(self, mean: torch.Tensor, covariance: Optional[torch.Tensor] = None):
        """
        Initialize state.

        Args:
            mean: State mean vector
            covariance: State covariance matrix (optional)
        """
        self.mean = mean
        self.covariance = covariance

        # Check dimensions
        if covariance is not None and covariance.shape[0] != mean.shape[0]:
            raise ValueError(
                f"Covariance shape {covariance.shape} does not match mean shape {mean.shape}"
            )

    @property
    def dim(self) -> int:
        """Get state dimension."""
        return self.mean.shape[0]

    @property
    def device(self) -> torch.device:
        """Get device of state tensors."""
        return self.mean.device

    def to(self, device: torch.device) -> "State":
        """
        Move state to device.

        Args:
            device: Target PyTorch device

        Returns:
            State on the target device
        """
        mean = self.mean.to(device)
        covariance = self.covariance.to(device) if self.covariance is not None else None
        return State(mean, covariance)

    def clone(self) -> "State":
        """
        Create a copy of the state.

        Returns:
            Cloned state
        """
        mean = self.mean.clone()
        covariance = self.covariance.clone() if self.covariance is not None else None
        return State(mean, covariance)

    def __str__(self) -> str:
        """String representation of state."""
        return f"State(mean={self.mean}, cov_diag={torch.diag(self.covariance) if self.covariance is not None else None})"


class StateEstimator(ABC):
    """
    Base class for state estimation algorithms.

    This is an abstract base class that defines the interface for
    various state estimation algorithms like EKF, UKF, or particle filters.
    """

    def __init__(self, state_dim: int, device: Optional[torch.device] = None):
        """
        Initialize state estimator.

        Args:
            state_dim: Dimension of the state vector
            device: PyTorch device
        """
        self.state_dim = state_dim
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.logger = logging.getLogger(self.__class__.__name__)

        # Current state estimate
        self.state = None

        # Time of last update
        self.last_update_time = None

    @abstractmethod
    def predict(self, control: torch.Tensor, dt: float) -> State:
        """
        Predict step of the state estimation.

        Args:
            control: Control input
            dt: Time step

        Returns:
            Predicted state
        """
        pass

    @abstractmethod
    def update(
        self,
        measurement: torch.Tensor,
        measurement_func: callable,
        measurement_noise: torch.Tensor,
    ) -> State:
        """
        Update step of the state estimation.

        Args:
            measurement: Measurement vector
            measurement_func: Function that maps state to measurement
            measurement_noise: Measurement noise covariance

        Returns:
            Updated state
        """
        pass

    @abstractmethod
    def reset(self, initial_state: Optional[State] = None):
        """
        Reset the state estimator.

        Args:
            initial_state: Optional initial state
        """
        pass

    def get_state(self) -> State:
        """
        Get current state estimate.

        Returns:
            Current state
        """
        if self.state is None:
            raise RuntimeError("State estimator has not been initialized")
        return self.state

    def get_current_pose(self) -> torch.Tensor:
        """
        Get current pose estimate.

        This method extracts the pose (position and orientation) from the state vector.
        The specific implementation depends on the state representation.

        Returns:
            Current pose as a transformation matrix
        """
        # Default implementation assumes first 6 elements are [x, y, z, roll, pitch, yaw]
        if self.state is None:
            raise RuntimeError("State estimator has not been initialized")

        # Extract position and orientation from state
        x, y, z = self.state.mean[:3]
        roll, pitch, yaw = self.state.mean[3:6]

        # Create transformation matrix
        cos_r, sin_r = torch.cos(roll), torch.sin(roll)
        cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)

        # Rotation matrix (ZYX order)
        R = torch.tensor(
            [
                [
                    cos_y * cos_p,
                    cos_y * sin_p * sin_r - sin_y * cos_r,
                    cos_y * sin_p * cos_r + sin_y * sin_r,
                ],
                [
                    sin_y * cos_p,
                    sin_y * sin_p * sin_r + cos_y * cos_r,
                    sin_y * sin_p * cos_r - cos_y * sin_r,
                ],
                [-sin_p, cos_p * sin_r, cos_p * cos_r],
            ],
            device=self.device,
        )

        # Create homogeneous transformation matrix
        T = torch.eye(4, device=self.device)
        T[:3, :3] = R
        T[:3, 3] = torch.tensor([x, y, z], device=self.device)

        return T

    def set_state(self, state: State):
        """
        Set current state estimate.

        Args:
            state: New state
        """
        if state.dim != self.state_dim:
            raise ValueError(
                f"State dimension {state.dim} does not match estimator dimension {self.state_dim}"
            )
        self.state = state

    def get_uncertainty(self) -> torch.Tensor:
        """
        Get current state uncertainty.

        Returns:
            Diagonal of the state covariance matrix
        """
        if self.state is None or self.state.covariance is None:
            raise RuntimeError("State estimator has no covariance information")
        return torch.diag(self.state.covariance)

    def get_mahalanobis_distance(self, state: State) -> float:
        """
        Compute Mahalanobis distance between current state and given state.

        Args:
            state: State to compute distance to

        Returns:
            Mahalanobis distance
        """
        if self.state is None or self.state.covariance is None:
            raise RuntimeError("State estimator has no covariance information")

        # Compute difference
        diff = state.mean - self.state.mean

        # Compute inverse of covariance
        try:
            cov_inv = torch.inverse(self.state.covariance)
        except RuntimeError:
            # Add small regularization if inversion fails
            reg_cov = (
                self.state.covariance
                + torch.eye(self.state_dim, device=self.device) * 1e-8
            )
            cov_inv = torch.inverse(reg_cov)

        # Compute Mahalanobis distance
        dist = torch.sqrt(
            torch.matmul(torch.matmul(diff.unsqueeze(0), cov_inv), diff.unsqueeze(1))
        )

        return dist.item()


class MotionModel(ABC):
    """
    Base class for motion models.

    This class defines the interface for motion models that predict
    how the state evolves over time given control inputs.
    """

    def __init__(
        self, state_dim: int, control_dim: int, device: Optional[torch.device] = None
    ):
        """
        Initialize motion model.

        Args:
            state_dim: Dimension of the state vector
            control_dim: Dimension of the control vector
            device: PyTorch device
        """
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    @abstractmethod
    def predict(self, state: State, control: torch.Tensor, dt: float) -> State:
        """
        Predict next state given current state and control.

        Args:
            state: Current state
            control: Control input
            dt: Time step

        Returns:
            Predicted next state
        """
        pass

    @abstractmethod
    def jacobian(self, state: State, control: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Compute Jacobian of state transition function.

        This is used by EKF and other methods that linearize the motion model.

        Args:
            state: Current state
            control: Control input
            dt: Time step

        Returns:
            Jacobian matrix of shape (state_dim, state_dim)
        """
        pass

    @abstractmethod
    def noise_covariance(
        self, state: State, control: torch.Tensor, dt: float
    ) -> torch.Tensor:
        """
        Compute process noise covariance.

        Args:
            state: Current state
            control: Control input
            dt: Time step

        Returns:
            Process noise covariance matrix of shape (state_dim, state_dim)
        """
        pass


class MeasurementModel(ABC):
    """
    Base class for measurement models.

    This class defines the interface for measurement models that map
    the state to expected measurements.
    """

    def __init__(
        self,
        state_dim: int,
        measurement_dim: int,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize measurement model.

        Args:
            state_dim: Dimension of the state vector
            measurement_dim: Dimension of the measurement vector
            device: PyTorch device
        """
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    @abstractmethod
    def predict_measurement(self, state: State) -> torch.Tensor:
        """
        Predict measurement given state.

        Args:
            state: State

        Returns:
            Predicted measurement
        """
        pass

    @abstractmethod
    def jacobian(self, state: State) -> torch.Tensor:
        """
        Compute Jacobian of measurement function.

        This is used by EKF and other methods that linearize the measurement model.

        Args:
            state: State

        Returns:
            Jacobian matrix of shape (measurement_dim, state_dim)
        """
        pass

    @abstractmethod
    def noise_covariance(self, state: State) -> torch.Tensor:
        """
        Compute measurement noise covariance.

        Args:
            state: State

        Returns:
            Measurement noise covariance matrix of shape (measurement_dim, measurement_dim)
        """
        pass
