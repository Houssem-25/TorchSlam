import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .base import MeasurementModel, MotionModel, State, StateEstimator


class EKF(StateEstimator):
    """
    Extended Kalman Filter for state estimation.

    The EKF is a recursive state estimator for nonlinear systems. It linearizes
    the motion and measurement models around the current state estimate.
    """

    def __init__(
        self,
        motion_model: MotionModel,
        initial_state: Optional[State] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize Extended Kalman Filter.

        Args:
            motion_model: Motion model
            initial_state: Initial state (optional)
            device: PyTorch device
        """
        super().__init__(motion_model.state_dim, device)

        self.motion_model = motion_model

        # Initialize state if provided
        if initial_state is not None:
            self.state = initial_state
            self.last_update_time = time.time()

    def predict(self, control: torch.Tensor, dt: float) -> State:
        """
        Predict step of the EKF.

        Args:
            control: Control input
            dt: Time step

        Returns:
            Predicted state
        """
        if self.state is None:
            raise RuntimeError("EKF has not been initialized with a state")

        # Get current state
        current_state = self.state

        # Predict next state using motion model
        predicted_state = self.motion_model.predict(current_state, control, dt)

        # Compute Jacobian of motion model
        F = self.motion_model.jacobian(current_state, control, dt)

        # Compute process noise covariance
        Q = self.motion_model.noise_covariance(current_state, control, dt)

        # Update covariance: P = F * P * F^T + Q
        if current_state.covariance is not None:
            predicted_covariance = (
                torch.matmul(F, torch.matmul(current_state.covariance, F.t())) + Q
            )
            predicted_state.covariance = predicted_covariance

        # Update state
        self.state = predicted_state

        # Update time of last state update
        self.last_update_time = time.time()

        return self.state

    def update(
        self,
        measurement: torch.Tensor,
        measurement_func: Callable[[torch.Tensor], torch.Tensor],
        measurement_noise: torch.Tensor,
        jacobian_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> State:
        """
        Update step of the EKF.

        Args:
            measurement: Measurement vector
            measurement_func: Function that maps state to measurement
            measurement_noise: Measurement noise covariance
            jacobian_func: Function to compute measurement Jacobian (optional)

        Returns:
            Updated state
        """
        if self.state is None:
            raise RuntimeError("EKF has not been initialized with a state")

        # If covariance is not available, we can't perform an update
        if self.state.covariance is None:
            self.logger.warning("No covariance information available, skipping update")
            return self.state

        # Get current state
        current_state = self.state

        # Predict measurement
        predicted_measurement = measurement_func(current_state.mean)

        # Compute measurement residual: y = z - h(x)
        residual = measurement - predicted_measurement

        # Compute measurement Jacobian
        if jacobian_func is not None:
            H = jacobian_func(current_state.mean)
        else:
            # Numerically estimate the Jacobian if not provided
            H = self._numerical_jacobian(measurement_func, current_state.mean)

        # Compute innovation covariance: S = H * P * H^T + R
        S = (
            torch.matmul(H, torch.matmul(current_state.covariance, H.t()))
            + measurement_noise
        )

        # Compute Kalman gain: K = P * H^T * S^-1
        try:
            S_inv = torch.inverse(S)
        except RuntimeError:
            # Add small regularization if inversion fails
            reg_S = S + torch.eye(S.shape[0], device=self.device) * 1e-8
            S_inv = torch.inverse(reg_S)

        K = torch.matmul(current_state.covariance, torch.matmul(H.t(), S_inv))

        # Update state mean: x = x + K * y
        updated_mean = current_state.mean + torch.matmul(K, residual)

        # Update state covariance: P = (I - K * H) * P
        I = torch.eye(self.state_dim, device=self.device)
        updated_covariance = torch.matmul(
            I - torch.matmul(K, H), current_state.covariance
        )

        # Ensure covariance is symmetric
        updated_covariance = 0.5 * (updated_covariance + updated_covariance.t())

        # Create updated state
        updated_state = State(updated_mean, updated_covariance)

        # Update state
        self.state = updated_state

        # Update time of last state update
        self.last_update_time = time.time()

        return self.state

    def reset(self, initial_state: Optional[State] = None):
        """
        Reset the EKF.

        Args:
            initial_state: Optional initial state
        """
        if initial_state is not None:
            if initial_state.dim != self.state_dim:
                raise ValueError(
                    f"Initial state dimension {initial_state.dim} does not match EKF dimension {self.state_dim}"
                )
            self.state = initial_state
            self.last_update_time = time.time()
        else:
            self.state = None
            self.last_update_time = None

    def _numerical_jacobian(
        self,
        func: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Compute numerical Jacobian of a function.

        Args:
            func: Function to differentiate
            x: Point at which to compute Jacobian
            eps: Small perturbation for numerical differentiation

        Returns:
            Jacobian matrix
        """
        n = x.shape[0]  # Input dimension
        y = func(x)
        m = y.shape[0]  # Output dimension

        # Initialize Jacobian
        J = torch.zeros((m, n), device=self.device)

        # Perturb each dimension and compute partial derivatives
        for i in range(n):
            x_plus = x.clone()
            x_plus[i] += eps

            y_plus = func(x_plus)

            # Compute partial derivative
            J[:, i] = (y_plus - y) / eps

        return J
