from typing import Optional, Tuple, Union

import numpy as np
import torch


class SE3:
    """
    SE(3) Lie group implementation for rigid body transformations.

    This class provides methods for working with 3D rigid transformations
    using the SE(3) Lie group and its corresponding Lie algebra se(3).
    All operations are implemented in PyTorch for GPU compatibility and
    differentiability.
    """

    def __init__(self, rotation: torch.Tensor, translation: torch.Tensor):
        """
        Initialize SE(3) transformation.

        Args:
            rotation: 3x3 rotation matrix
            translation: 3D translation vector
        """
        self.R = rotation
        self.t = translation

        # Store device for operations
        self.device = rotation.device

    @classmethod
    def from_matrix(cls, matrix: torch.Tensor) -> "SE3":
        """
        Create SE(3) object from 4x4 transformation matrix.

        Args:
            matrix: 4x4 transformation matrix

        Returns:
            SE3 object
        """
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected 4x4 matrix, got {matrix.shape}")

        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]

        return cls(rotation, translation)

    @classmethod
    def from_rotation_translation(
        cls, rotation: torch.Tensor, translation: torch.Tensor
    ) -> "SE3":
        """
        Create SE(3) object from rotation matrix and translation vector.

        Args:
            rotation: 3x3 rotation matrix
            translation: 3D translation vector

        Returns:
            SE3 object
        """
        if rotation.shape != (3, 3):
            raise ValueError(f"Expected 3x3 rotation matrix, got {rotation.shape}")

        if translation.shape != (3,):
            raise ValueError(f"Expected 3D translation vector, got {translation.shape}")

        return cls(rotation, translation)

    @classmethod
    def from_elements(
        cls,
        rx: float,
        ry: float,
        rz: float,
        tx: float,
        ty: float,
        tz: float,
        device: Optional[torch.device] = None,
    ) -> "SE3":
        """
        Create SE(3) object from individual rotation and translation elements.

        Args:
            rx, ry, rz: Rotation angles around x, y, z axes (in radians)
            tx, ty, tz: Translation along x, y, z axes
            device: PyTorch device

        Returns:
            SE3 object
        """
        device = device or torch.device("cpu")

        # Create rotation matrix using Rodrigues' formula for each axis
        Rx = torch.tensor(
            [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]],
            dtype=torch.float32,
            device=device,
        )

        Ry = torch.tensor(
            [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]],
            dtype=torch.float32,
            device=device,
        )

        Rz = torch.tensor(
            [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]],
            dtype=torch.float32,
            device=device,
        )

        # Combine rotations
        R = torch.matmul(Rz, torch.matmul(Ry, Rx))

        # Create translation vector
        t = torch.tensor([tx, ty, tz], dtype=torch.float32, device=device)

        return cls(R, t)

    @classmethod
    def identity(cls, device: Optional[torch.device] = None) -> "SE3":
        """
        Create identity transformation.

        Args:
            device: PyTorch device

        Returns:
            Identity SE3 object
        """
        device = device or torch.device("cpu")
        R = torch.eye(3, device=device)
        t = torch.zeros(3, device=device)
        return cls(R, t)

    @classmethod
    def exp(cls, xi: torch.Tensor) -> "SE3":
        """
        Exponential map from se(3) to SE(3).

        Args:
            xi: 6D twist coordinates (v, omega) in se(3)
                First 3 elements are translation components
                Last 3 elements are rotation components

        Returns:
            SE3 object
        """
        if xi.shape != (6,):
            raise ValueError(f"Expected 6D vector, got {xi.shape}")

        device = xi.device
        v = xi[:3]  # Translation part
        omega = xi[3:]  # Rotation part

        # Get rotation using the Rodrigues' formula
        theta = torch.norm(omega)

        if theta < 1e-8:
            # For small rotations, use approximation
            R = torch.eye(3, device=device)
            V = torch.eye(3, device=device)
        else:
            # Normalize rotation axis
            omega_normalized = omega / theta

            # Skew-symmetric matrix
            K = cls._skew_symmetric(omega_normalized)

            # Rodrigues' formula
            R = (
                torch.eye(3, device=device)
                + torch.sin(theta) * K
                + (1 - torch.cos(theta)) * torch.matmul(K, K)
            )

            # V matrix for translation part
            V = (
                torch.eye(3, device=device)
                + (1 - torch.cos(theta)) / theta * K
                + (theta - torch.sin(theta)) / theta * torch.matmul(K, K)
            )

        # Translation part
        t = torch.matmul(V, v)

        return cls(R, t)

    def log(self) -> torch.Tensor:
        """
        Logarithmic map from SE(3) to se(3).

        Returns:
            6D twist coordinates (v, omega) in se(3)
        """
        # Extract rotation angle from trace
        trace = torch.trace(self.R)
        cos_theta = (trace - 1) / 2

        if cos_theta > 0.999999:
            # For small rotations, use approximation
            omega = torch.zeros(3, device=self.device)
            V_inv = torch.eye(3, device=self.device)
        else:
            # Extract rotation angle
            theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))

            # Extract rotation axis
            sin_theta = torch.sin(theta)
            if abs(sin_theta) < 1e-8:
                # For 180-degree rotations, find rotation axis differently
                # We need to find an eigenvector corresponding to eigenvalue 1
                d = torch.diag(self.R)
                if d[0] > 0.999:
                    omega = torch.tensor([1.0, 0.0, 0.0], device=self.device)
                elif d[1] > 0.999:
                    omega = torch.tensor([0.0, 1.0, 0.0], device=self.device)
                else:
                    omega = torch.tensor([0.0, 0.0, 1.0], device=self.device)
                omega = omega * theta
            else:
                # Extract axis from skew-symmetric component
                K = (self.R - self.R.t()) / (2 * sin_theta)
                omega = (
                    torch.tensor([K[2, 1], K[0, 2], K[1, 0]], device=self.device)
                    * theta
                )

            # Compute V inverse
            K = self._skew_symmetric(omega / theta)
            V_inv = (
                torch.eye(3, device=self.device)
                - 0.5 * K
                + (1 - theta * torch.cos(theta) / (2 * torch.sin(theta)))
                * torch.matmul(K, K)
            )

        # Compute translational component
        v = torch.matmul(V_inv, self.t)

        # Concatenate to form se(3) element
        xi = torch.cat([v, omega])

        return xi

    def to_matrix(self) -> torch.Tensor:
        """
        Convert to 4x4 transformation matrix.

        Returns:
            4x4 transformation matrix
        """
        matrix = torch.eye(4, device=self.device)
        matrix[:3, :3] = self.R
        matrix[:3, 3] = self.t
        return matrix

    def adjoint(self) -> torch.Tensor:
        """
        Compute the adjoint matrix of the SE(3) element.

        The adjoint is used for transforming twists and wrenches.

        Returns:
            6x6 adjoint matrix
        """
        adj = torch.zeros((6, 6), device=self.device)

        # Top-left block: rotation matrix
        adj[:3, :3] = self.R

        # Top-right block: skew(t) * R
        adj[:3, 3:] = torch.matmul(self._skew_symmetric(self.t), self.R)

        # Bottom-right block: rotation matrix
        adj[3:, 3:] = self.R

        return adj

    def transform_point(self, point: torch.Tensor) -> torch.Tensor:
        """
        Transform a 3D point.

        Args:
            point: 3D point

        Returns:
            Transformed point
        """
        return torch.matmul(self.R, point) + self.t

    def transform_points(self, points: torch.Tensor) -> torch.Tensor:
        """
        Transform multiple 3D points.

        Args:
            points: Tensor of shape (N, 3) containing 3D points

        Returns:
            Transformed points of shape (N, 3)
        """
        return torch.matmul(points, self.R.t()) + self.t

    def inverse(self) -> "SE3":
        """
        Compute the inverse transformation.

        Returns:
            Inverse SE3 object
        """
        R_inv = self.R.t()
        t_inv = -torch.matmul(R_inv, self.t)
        return SE3(R_inv, t_inv)

    def compose(self, other: "SE3") -> "SE3":
        """
        Compose with another SE(3) transformation: self * other

        Args:
            other: Another SE3 object

        Returns:
            Composed transformation
        """
        R = torch.matmul(self.R, other.R)
        t = torch.matmul(self.R, other.t) + self.t
        return SE3(R, t)

    def rotation_matrix(self) -> torch.Tensor:
        """
        Get the rotation matrix component.

        Returns:
            3x3 rotation matrix
        """
        return self.R

    def translation_vector(self) -> torch.Tensor:
        """
        Get the translation vector component.

        Returns:
            3D translation vector
        """
        return self.t

    def interpolate(self, other: "SE3", alpha: float) -> "SE3":
        """
        Interpolate between this and another SE(3) transformation.

        Args:
            other: Another SE3 object
            alpha: Interpolation parameter [0, 1]

        Returns:
            Interpolated SE3 object
        """
        # Convert both transformations to se(3)
        xi1 = self.log()
        xi2 = other.log()

        # Linear interpolation in the Lie algebra
        xi_interp = xi1 + alpha * (xi2 - xi1)

        # Convert back to SE(3)
        return SE3.exp(xi_interp)

    @staticmethod
    def _skew_symmetric(v: torch.Tensor) -> torch.Tensor:
        """
        Create a skew-symmetric matrix from a 3D vector.

        Args:
            v: 3D vector

        Returns:
            3x3 skew-symmetric matrix
        """
        device = v.device
        zero = torch.tensor(0.0, device=device)

        return torch.tensor(
            [[zero, -v[2], v[1]], [v[2], zero, -v[0]], [-v[1], v[0], zero]],
            device=device,
        )

    def __repr__(self) -> str:
        return f"SE3(R=\n{self.R},\nt={self.t})"


# Additional utility functions for conversions
def euler_to_rotation_matrix(euler: torch.Tensor) -> torch.Tensor:
    """
    Convert Euler angles to rotation matrix using the ZYX convention.

    Args:
        euler: Tensor of Euler angles [roll, pitch, yaw] in radians

    Returns:
        3x3 rotation matrix
    """
    device = euler.device

    # Extract Euler angles
    roll, pitch, yaw = euler

    # Compute trigonometric functions
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)

    # Compute rotation matrix
    R = torch.zeros((3, 3), device=device)

    # ZYX convention
    R[0, 0] = cy * cp
    R[0, 1] = cy * sp * sr - sy * cr
    R[0, 2] = cy * sp * cr + sy * sr
    R[1, 0] = sy * cp
    R[1, 1] = sy * sp * sr + cy * cr
    R[1, 2] = sy * sp * cr - cy * sr
    R[2, 0] = -sp
    R[2, 1] = cp * sr
    R[2, 2] = cp * cr

    return R


def rotation_matrix_to_euler(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to Euler angles using the ZYX convention.

    Args:
        R: 3x3 rotation matrix

    Returns:
        Tensor of Euler angles [roll, pitch, yaw] in radians
    """
    device = R.device

    # Handle singularities
    if abs(R[2, 0]) > 0.99999:
        # Gimbal lock case
        pitch = -torch.sign(R[2, 0]) * torch.pi / 2
        yaw = torch.atan2(-R[1, 2], R[1, 1])
        roll = torch.tensor(0.0, device=device)
    else:
        pitch = -torch.asin(R[2, 0])
        roll = torch.atan2(R[2, 1], R[2, 2])
        yaw = torch.atan2(R[1, 0], R[0, 0])

    return torch.tensor([roll, pitch, yaw], device=device)


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to rotation matrix.

    Args:
        q: Quaternion [w, x, y, z] where w is the scalar part

    Returns:
        3x3 rotation matrix
    """
    device = q.device

    # Normalize quaternion
    q = q / torch.norm(q)

    # Extract quaternion components
    w, x, y, z = q

    # Compute rotation matrix
    R = torch.zeros((3, 3), device=device)

    R[0, 0] = 1 - 2 * (y * y + z * z)
    R[0, 1] = 2 * (x * y - w * z)
    R[0, 2] = 2 * (x * z + w * y)

    R[1, 0] = 2 * (x * y + w * z)
    R[1, 1] = 1 - 2 * (x * x + z * z)
    R[1, 2] = 2 * (y * z - w * x)

    R[2, 0] = 2 * (x * z - w * y)
    R[2, 1] = 2 * (y * z + w * x)
    R[2, 2] = 1 - 2 * (x * x + y * y)

    return R


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to quaternion.

    Args:
        R: 3x3 rotation matrix

    Returns:
        Quaternion [w, x, y, z] where w is the scalar part
    """
    device = R.device

    # Calculate trace
    trace = torch.trace(R)

    if trace > 0:
        # If trace is positive
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        # If trace is negative or zero
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    # Return quaternion
    return torch.tensor([w, x, y, z], device=device)
