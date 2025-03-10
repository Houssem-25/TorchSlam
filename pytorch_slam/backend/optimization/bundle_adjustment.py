import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ..se3 import SE3
from .base import LevenbergMarquardtOptimizer, OptimizationResult, Optimizer
from .factor_graph import Factor, FactorGraph, Variable


class CameraVariable(Variable):
    """
    Variable representing a camera pose.

    This variable includes the camera's extrinsic parameters (pose).
    Camera intrinsics are typically held constant during optimization.
    """

    def __init__(self, var_id: str, initial_value: Optional[torch.Tensor] = None):
        """
        Initialize camera variable.

        Args:
            var_id: Unique identifier for the variable
            initial_value: Initial SE(3) matrix (4x4) or None for identity
        """
        super().__init__(var_id, dim=6, initial_value=None)

        # Store actual pose as SE3 object for easy manipulation
        if initial_value is None:
            self.pose = SE3.identity()
        else:
            self.pose = SE3.from_matrix(initial_value)

        # Update the internal value to store the 6D parameterization
        self.value = self.pose.log()

    def set_value(self, value: torch.Tensor):
        """
        Set the value of the camera variable.

        Args:
            value: New value as 6D vector or 4x4 SE(3) matrix
        """
        if value.shape == (6,):
            # 6D vector (se3 parameters)
            self.value = value
            self.pose = SE3.exp(value)
        elif value.shape == (4, 4):
            # SE(3) matrix
            self.pose = SE3.from_matrix(value)
            self.value = self.pose.log()
        else:
            raise ValueError(f"Invalid shape for camera variable: {value.shape}")

    def get_value(self) -> torch.Tensor:
        """
        Get the current value of the variable as 6D vector.

        Returns:
            Current 6D parameterization
        """
        return self.value

    def get_matrix(self) -> torch.Tensor:
        """
        Get the current pose as 4x4 SE(3) matrix.

        Returns:
            Current pose matrix
        """
        return self.pose.to_matrix()


class LandmarkVariable(Variable):
    """
    Variable representing a 3D landmark point.

    This variable represents a 3D point in the world coordinate system.
    """

    def __init__(self, var_id: str, initial_value: Optional[torch.Tensor] = None):
        """
        Initialize landmark variable.

        Args:
            var_id: Unique identifier for the variable
            initial_value: Initial 3D position or None for origin
        """
        super().__init__(var_id, dim=3, initial_value=None)

        # Store position
        if initial_value is None:
            self.value = torch.zeros(3)
        else:
            self.value = initial_value

    def set_value(self, value: torch.Tensor):
        """
        Set the value of the landmark variable.

        Args:
            value: New 3D position
        """
        if value.shape != (3,):
            raise ValueError(f"Invalid shape for landmark variable: {value.shape}")
        self.value = value

    def get_value(self) -> torch.Tensor:
        """
        Get the current value of the variable.

        Returns:
            Current 3D position
        """
        return self.value


class ReprojectionFactor(Factor):
    """
    Factor for the reprojection error in bundle adjustment.

    This factor represents the error between the projected 3D point and
    the observed 2D point in a camera image.
    """

    def __init__(
        self,
        camera_id: str,
        landmark_id: str,
        observation: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        robust_kernel: bool = False,
        robust_delta: float = 1.0,
        weight: float = 1.0,
    ):
        """
        Initialize reprojection factor.

        Args:
            camera_id: ID of the camera variable
            landmark_id: ID of the landmark variable
            observation: 2D observation point in image coordinates
            camera_intrinsics: Camera intrinsic matrix (3x3)
            robust_kernel: Whether to use a robust error kernel
            robust_delta: Delta parameter for robust kernel
            weight: Weight for this factor
        """
        super().__init__([camera_id, landmark_id], weight)

        self.camera_id = camera_id
        self.landmark_id = landmark_id
        self.observation = observation
        self.camera_intrinsics = camera_intrinsics
        self.robust_kernel = robust_kernel
        self.robust_delta = robust_delta

        # Extract camera intrinsics for faster computation
        self.fx = camera_intrinsics[0, 0]
        self.fy = camera_intrinsics[1, 1]
        self.cx = camera_intrinsics[0, 2]
        self.cy = camera_intrinsics[1, 2]

    def compute_error(self, variables: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute reprojection error.

        Args:
            variables: Dictionary mapping variable ID to value
                       camera_id -> 6D camera pose parameters
                       landmark_id -> 3D landmark position

        Returns:
            Error vector (2,)
        """
        # Get camera pose and landmark position
        camera_pose = variables[self.camera_id]
        landmark_position = variables[self.landmark_id]

        # Convert camera pose to SE3
        camera_pose_se3 = SE3.exp(camera_pose)

        # Transform landmark to camera frame
        point_camera = camera_pose_se3.transform_point(landmark_position)

        # Check if point is in front of camera
        if point_camera[2] <= 0:
            # Point is behind camera, use a large error
            return (
                torch.tensor([1000.0, 1000.0], device=self.observation.device)
                * self.weight
            )

        # Project point to image plane
        projected_x = self.fx * (point_camera[0] / point_camera[2]) + self.cx
        projected_y = self.fy * (point_camera[1] / point_camera[2]) + self.cy

        # Compute reprojection error
        error = torch.tensor(
            [projected_x - self.observation[0], projected_y - self.observation[1]],
            device=self.observation.device,
        )

        # Apply robust kernel if enabled
        if self.robust_kernel:
            error_norm = torch.norm(error)
            if error_norm > self.robust_delta:
                # Apply Huber loss
                scale = self.robust_delta / error_norm
                error = error * scale

        # Apply weight
        weighted_error = error * self.weight

        return weighted_error

    def compute_jacobians(
        self, variables: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Jacobians of the error with respect to variables.

        Args:
            variables: Dictionary mapping variable ID to value

        Returns:
            Dictionary mapping variable ID to Jacobian matrix
        """
        # Get camera pose and landmark position
        camera_pose = variables[self.camera_id]
        landmark_position = variables[self.landmark_id]

        # Convert camera pose to SE3
        camera_pose_se3 = SE3.exp(camera_pose)

        # Transform landmark to camera frame
        point_camera = camera_pose_se3.transform_point(landmark_position)

        # Check if point is in front of camera
        if point_camera[2] <= 0:
            # Return zero Jacobians for points behind camera
            return {
                self.camera_id: torch.zeros((2, 6), device=camera_pose.device),
                self.landmark_id: torch.zeros((2, 3), device=landmark_position.device),
            }

        # Extract camera rotation
        R = camera_pose_se3.rotation_matrix()

        # Pre-compute values
        x, y, z = point_camera[0], point_camera[1], point_camera[2]
        inv_z = 1.0 / z
        inv_z_squared = inv_z * inv_z

        # Compute Jacobian with respect to camera pose (2x6 matrix)
        # The Jacobian has the structure [J_rotation | J_translation]
        # This follows the derivation in multiple view geometry texts
        J_camera = torch.zeros((2, 6), device=camera_pose.device)

        # Jacobian of projection with respect to translation
        J_camera[0, 3] = self.fx * inv_z
        J_camera[0, 4] = 0
        J_camera[0, 5] = -self.fx * x * inv_z_squared

        J_camera[1, 3] = 0
        J_camera[1, 4] = self.fy * inv_z
        J_camera[1, 5] = -self.fy * y * inv_z_squared

        # Jacobian of projection with respect to rotation
        J_camera[0, 0] = -self.fx * y * x * inv_z_squared
        J_camera[0, 1] = self.fx * (1 + x * x * inv_z_squared)
        J_camera[0, 2] = -self.fx * y * inv_z

        J_camera[1, 0] = -self.fy * (1 + y * y * inv_z_squared)
        J_camera[1, 1] = self.fy * x * y * inv_z_squared
        J_camera[1, 2] = self.fy * x * inv_z

        # Compute Jacobian with respect to landmark position (2x3 matrix)
        J_landmark = torch.zeros((2, 3), device=landmark_position.device)

        # Jacobian of projection with respect to point coordinates
        J_landmark[0, 0] = self.fx * inv_z
        J_landmark[0, 1] = 0
        J_landmark[0, 2] = -self.fx * x * inv_z_squared

        J_landmark[1, 0] = 0
        J_landmark[1, 1] = self.fy * inv_z
        J_landmark[1, 2] = -self.fy * y * inv_z_squared

        # Transform landmark Jacobian to world frame
        J_landmark = torch.matmul(J_landmark, R)

        # Apply robust kernel if enabled
        if self.robust_kernel:
            error = self.compute_error(variables) / self.weight
            error_norm = torch.norm(error)

            if error_norm > self.robust_delta:
                # Compute scale factor for Jacobians
                scale = self.robust_delta / error_norm
                J_camera = J_camera * scale
                J_landmark = J_landmark * scale

        # Apply weight
        J_camera = J_camera * self.weight
        J_landmark = J_landmark * self.weight

        return {self.camera_id: J_camera, self.landmark_id: J_landmark}


class BundleAdjustment:
    """
    Bundle adjustment for structure from motion.

    This class provides methods to build and optimize a bundle adjustment problem,
    refining camera poses and 3D landmark positions.
    """

    def __init__(self, device: torch.device = None):
        """
        Initialize bundle adjustment.

        Args:
            device: PyTorch device
        """
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.graph = FactorGraph()
        self.optimizer = LevenbergMarquardtOptimizer(device=self.device)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Camera intrinsics
        self.camera_intrinsics = {}  # camera_id -> intrinsics matrix

    def add_camera(
        self, camera_id: str, pose: torch.Tensor, intrinsics: torch.Tensor
    ) -> "BundleAdjustment":
        """
        Add a camera to the bundle adjustment problem.

        Args:
            camera_id: Unique identifier for the camera
            pose: Initial camera pose as SE(3) matrix
            intrinsics: Camera intrinsic matrix (3x3)

        Returns:
            Self for method chaining
        """
        # Move tensors to device
        pose = pose.to(self.device)
        intrinsics = intrinsics.to(self.device)

        # Create camera variable
        camera_var = CameraVariable(camera_id, pose)
        self.graph.add_variable(camera_var)

        # Store intrinsics
        self.camera_intrinsics[camera_id] = intrinsics

        return self

    def add_landmark(
        self, landmark_id: str, position: torch.Tensor
    ) -> "BundleAdjustment":
        """
        Add a landmark to the bundle adjustment problem.

        Args:
            landmark_id: Unique identifier for the landmark
            position: Initial 3D position

        Returns:
            Self for method chaining
        """
        # Move tensor to device
        position = position.to(self.device)

        # Create landmark variable
        landmark_var = LandmarkVariable(landmark_id, position)
        self.graph.add_variable(landmark_var)

        return self

    def add_observation(
        self,
        camera_id: str,
        landmark_id: str,
        observation: torch.Tensor,
        weight: float = 1.0,
        robust: bool = False,
    ) -> "BundleAdjustment":
        """
        Add an observation of a landmark by a camera.

        Args:
            camera_id: ID of the observing camera
            landmark_id: ID of the observed landmark
            observation: 2D image coordinates of the observation
            weight: Weight for this observation
            robust: Whether to use robust error kernel

        Returns:
            Self for method chaining
        """
        # Check if camera and landmark exist
        if camera_id not in self.camera_intrinsics:
            raise ValueError(f"Camera {camera_id} not found")

        # Move tensor to device
        observation = observation.to(self.device)

        # Get camera intrinsics
        intrinsics = self.camera_intrinsics[camera_id]

        # Create reprojection factor
        factor = ReprojectionFactor(
            camera_id=camera_id,
            landmark_id=landmark_id,
            observation=observation,
            camera_intrinsics=intrinsics,
            robust_kernel=robust,
            weight=weight,
        )

        # Add factor to graph
        self.graph.add_factor(factor)

        return self

    def set_fixed_camera(
        self, camera_id: str, fixed: bool = True
    ) -> "BundleAdjustment":
        """
        Set a camera as fixed during optimization.

        Args:
            camera_id: ID of the camera to fix
            fixed: Whether to fix the camera

        Returns:
            Self for method chaining
        """
        # In a full implementation, we would set a flag on the variable
        # and exclude it from the optimization variables list
        # For simplicity, we'll keep this as a placeholder for now
        self.logger.info(
            f"Setting camera {camera_id} as {'fixed' if fixed else 'free'}"
        )

        return self

    def set_fixed_landmark(
        self, landmark_id: str, fixed: bool = True
    ) -> "BundleAdjustment":
        """
        Set a landmark as fixed during optimization.

        Args:
            landmark_id: ID of the landmark to fix
            fixed: Whether to fix the landmark

        Returns:
            Self for method chaining
        """
        # Placeholder similar to set_fixed_camera
        self.logger.info(
            f"Setting landmark {landmark_id} as {'fixed' if fixed else 'free'}"
        )

        return self

    def optimize(
        self, max_iterations: int = 100, verbose: bool = False
    ) -> OptimizationResult:
        """
        Optimize the bundle adjustment problem.

        Args:
            max_iterations: Maximum number of iterations
            verbose: Whether to print optimization progress

        Returns:
            Optimization result
        """
        # Configure optimizer
        self.optimizer.max_iterations = max_iterations
        self.optimizer.verbose = verbose

        # Run optimization
        result = self.optimizer.solve(self.graph)

        # Update graph variables with optimized values
        self.graph.set_variable_values(result.variables)

        return result

    def get_camera_pose(self, camera_id: str) -> torch.Tensor:
        """
        Get the current pose of a camera.

        Args:
            camera_id: ID of the camera

        Returns:
            SE(3) matrix of the camera pose
        """
        camera_var = self.graph.get_variable(camera_id)
        if isinstance(camera_var, CameraVariable):
            return camera_var.get_matrix()
        else:
            # Convert from 6D parameterization to matrix
            return SE3.exp(camera_var.get_value()).to_matrix()

    def get_landmark_position(self, landmark_id: str) -> torch.Tensor:
        """
        Get the current position of a landmark.

        Args:
            landmark_id: ID of the landmark

        Returns:
            3D position of the landmark
        """
        return self.graph.get_variable(landmark_id).get_value()

    def get_all_cameras(self) -> Dict[str, torch.Tensor]:
        """
        Get all camera poses.

        Returns:
            Dictionary mapping camera ID to SE(3) matrix
        """
        cameras = {}
        for var_id, variable in self.graph.variables.items():
            if isinstance(variable, CameraVariable):
                cameras[var_id] = self.get_camera_pose(var_id)
        return cameras

    def get_all_landmarks(self) -> Dict[str, torch.Tensor]:
        """
        Get all landmark positions.

        Returns:
            Dictionary mapping landmark ID to 3D position
        """
        landmarks = {}
        for var_id, variable in self.graph.variables.items():
            if isinstance(variable, LandmarkVariable):
                landmarks[var_id] = variable.get_value()
        return landmarks

    def get_reprojection_errors(self) -> Dict[Tuple[str, str], float]:
        """
        Get reprojection errors for all observations.

        Returns:
            Dictionary mapping (camera_id, landmark_id) to reprojection error
        """
        errors = {}

        # Get current variable values
        variables = self.graph.get_variable_values()

        # Compute errors for each factor
        for factor in self.graph.factors:
            if isinstance(factor, ReprojectionFactor):
                camera_id = factor.camera_id
                landmark_id = factor.landmark_id

                # Compute error for this factor
                error = factor.compute_error(variables)
                error_norm = torch.norm(error).item()

                errors[(camera_id, landmark_id)] = error_norm

        return errors

    def get_total_reprojection_error(self) -> float:
        """
        Get total reprojection error.

        Returns:
            Sum of squared reprojection errors
        """
        errors = self.get_reprojection_errors()
        return sum(error**2 for error in errors.values())
