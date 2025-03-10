import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


@dataclass
class OptimizationResult:
    """Results from an optimization run."""

    success: bool  # Whether optimization was successful
    initial_cost: float  # Initial cost before optimization
    final_cost: float  # Final cost after optimization
    variables: Dict[str, torch.Tensor]  # Optimized variables
    num_iterations: int  # Number of iterations performed
    time_seconds: float  # Time taken in seconds
    convergence_info: Dict[str, Any]  # Additional convergence information


class OptimizationProblem(ABC):
    """
    Base class for optimization problems in SLAM.

    This abstract class defines the interface for optimization problems
    that can be solved by various optimizers.
    """

    @abstractmethod
    def compute_residuals(self, variables: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute residuals for the current variable values.

        Args:
            variables: Dictionary of variables

        Returns:
            Tensor of residuals
        """
        pass

    @abstractmethod
    def compute_cost(self, variables: Dict[str, torch.Tensor]) -> float:
        """
        Compute total cost for the current variable values.

        Args:
            variables: Dictionary of variables

        Returns:
            Total cost value
        """
        pass

    @abstractmethod
    def compute_jacobians(
        self, variables: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Jacobians for each variable.

        Args:
            variables: Dictionary of variables

        Returns:
            Dictionary mapping variable IDs to Jacobian matrices
        """
        pass

    @abstractmethod
    def get_variable_dimensions(self) -> Dict[str, int]:
        """
        Get dimensions of each variable.

        Returns:
            Dictionary mapping variable IDs to their dimensions
        """
        pass

    @abstractmethod
    def get_residual_dimensions(self) -> int:
        """
        Get total dimension of residuals.

        Returns:
            Total dimension of all residuals
        """
        pass

    @abstractmethod
    def get_initial_values(self) -> Dict[str, torch.Tensor]:
        """
        Get initial values for all variables.

        Returns:
            Dictionary mapping variable IDs to their initial values
        """
        pass


class Optimizer(ABC):
    """
    Base class for optimization algorithms.

    This abstract class defines the interface for optimizers that
    can solve SLAM optimization problems.
    """

    def __init__(
        self,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
        verbose: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize optimizer.

        Args:
            max_iterations: Maximum number of iterations
            convergence_threshold: Threshold for convergence check
            verbose: Whether to print optimization progress
            device: PyTorch device
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.verbose = verbose

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    @abstractmethod
    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
        """
        Solve an optimization problem.

        Args:
            problem: Optimization problem to solve

        Returns:
            Optimization result
        """
        pass

    def measure_convergence(
        self, prev_cost: float, curr_cost: float, update_norm: float
    ) -> bool:
        """
        Check if optimization has converged.

        Args:
            prev_cost: Cost from previous iteration
            curr_cost: Cost from current iteration
            update_norm: Norm of the update step

        Returns:
            True if converged, False otherwise
        """
        # Check if cost decrease is small
        cost_decrease = prev_cost - curr_cost
        relative_decrease = cost_decrease / (prev_cost + 1e-10)

        # Check if update norm is small
        if (
            update_norm < self.convergence_threshold
            and relative_decrease < self.convergence_threshold
        ):
            return True

        return False


class GaussNewtonOptimizer(Optimizer):
    """
    Gauss-Newton optimizer for nonlinear least squares problems.

    This optimizer is commonly used for SLAM problems due to its
    fast convergence near the minimum.
    """

    def __init__(
        self,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
        damping_factor: float = 1e-4,
        verbose: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize Gauss-Newton optimizer.

        Args:
            max_iterations: Maximum number of iterations
            convergence_threshold: Threshold for convergence check
            damping_factor: Damping factor for numerical stability
            verbose: Whether to print optimization progress
            device: PyTorch device
        """
        super().__init__(max_iterations, convergence_threshold, verbose, device)
        self.damping_factor = damping_factor

    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
        """
        Solve an optimization problem using Gauss-Newton algorithm.

        Args:
            problem: Optimization problem to solve

        Returns:
            Optimization result
        """
        start_time = time.time()

        # Get initial values
        variables = problem.get_initial_values()

        # Move variables to device
        for var_id, var_value in variables.items():
            variables[var_id] = var_value.to(self.device)

        # Compute initial cost
        initial_cost = problem.compute_cost(variables)
        current_cost = initial_cost

        # Track convergence
        convergence_info = {
            "costs": [initial_cost],
            "update_norms": [],
            "gradient_norms": [],
        }

        # Main optimization loop
        iteration = 0
        for iteration in range(self.max_iterations):
            # Compute residuals
            residuals = problem.compute_residuals(variables)

            # Compute Jacobians
            jacobians = problem.compute_jacobians(variables)

            # Build normal equations (J^T * J) * delta = -J^T * r
            JTJ = {}  # Sparse block structure
            JTr = {}  # Right-hand side

            # Initialize JTr
            for var_id, var_dim in problem.get_variable_dimensions().items():
                JTr[var_id] = torch.zeros(var_dim, device=self.device)

            # Build JTJ and JTr
            for var_id, jacobian in jacobians.items():
                # JTr = J^T * r
                JTr[var_id] = torch.matmul(jacobian.t(), residuals)

                # JTJ diagonal blocks with damping
                var_dim = problem.get_variable_dimensions()[var_id]
                JTJ[(var_id, var_id)] = torch.matmul(jacobian.t(), jacobian)
                JTJ[(var_id, var_id)] += self.damping_factor * torch.eye(
                    var_dim, device=self.device
                )

                # Off-diagonal blocks
                for other_id, other_jacobian in jacobians.items():
                    if var_id != other_id:
                        JTJ[(var_id, other_id)] = torch.matmul(
                            jacobian.t(), other_jacobian
                        )

            # Solve normal equations
            delta = self._solve_normal_equations(
                JTJ, JTr, problem.get_variable_dimensions()
            )

            # Update variables
            update_norm = 0.0
            for var_id, var_delta in delta.items():
                variables[var_id] = variables[var_id] + var_delta
                update_norm += torch.norm(var_delta).item() ** 2

            update_norm = update_norm**0.5

            # Compute new cost
            new_cost = problem.compute_cost(variables)

            # Track convergence
            convergence_info["costs"].append(new_cost)
            convergence_info["update_norms"].append(update_norm)

            # Check convergence
            if self.measure_convergence(current_cost, new_cost, update_norm):
                break

            current_cost = new_cost

            # Print progress
            if self.verbose and (
                iteration % 5 == 0 or iteration == self.max_iterations - 1
            ):
                print(
                    f"Iteration {iteration}: cost = {current_cost:.6f}, update_norm = {update_norm:.6f}"
                )

        end_time = time.time()

        # Create result
        result = OptimizationResult(
            success=True,
            initial_cost=initial_cost,
            final_cost=current_cost,
            variables=variables,
            num_iterations=iteration + 1,
            time_seconds=end_time - start_time,
            convergence_info=convergence_info,
        )

        return result

    def _solve_normal_equations(
        self,
        JTJ: Dict[Tuple[str, str], torch.Tensor],
        JTr: Dict[str, torch.Tensor],
        var_dims: Dict[str, int],
    ) -> Dict[str, torch.Tensor]:
        """
        Solve the normal equations for Gauss-Newton optimization.

        This method solves the system (J^T * J) * delta = -J^T * r.
        It builds a sparse block matrix structure for efficiency.

        Args:
            JTJ: Dictionary mapping (var_id1, var_id2) to block matrices
            JTr: Dictionary mapping var_id to right-hand side vectors
            var_dims: Dictionary mapping var_id to variable dimensions

        Returns:
            Dictionary mapping var_id to delta updates
        """
        # Get sorted variable IDs for consistent ordering
        var_ids = sorted(var_dims.keys())

        # Negate right-hand side
        for var_id in var_ids:
            JTr[var_id] = -JTr[var_id]

        # Build full matrix for normal equations
        # In a real implementation, this would use sparse block solvers
        # But for simplicity, we use a dense solver here

        # Calculate total dimension
        total_dim = sum(var_dims.values())

        # Build full matrices
        A = torch.zeros((total_dim, total_dim), device=self.device)
        b = torch.zeros(total_dim, device=self.device)

        # Fill matrices
        row_offset = 0
        for i, var_i in enumerate(var_ids):
            col_offset = 0
            i_dim = var_dims[var_i]

            # Fill right-hand side
            b[row_offset : row_offset + i_dim] = JTr[var_i]

            for j, var_j in enumerate(var_ids):
                j_dim = var_dims[var_j]

                # Fill JTJ block
                if (var_i, var_j) in JTJ:
                    A[
                        row_offset : row_offset + i_dim, col_offset : col_offset + j_dim
                    ] = JTJ[(var_i, var_j)]

                col_offset += j_dim

            row_offset += i_dim

        # Solve system
        try:
            x = torch.linalg.solve(A, b)
        except RuntimeError:
            # If matrix is singular, use pseudo-inverse
            x = torch.linalg.lstsq(A, b).solution

        # Extract individual deltas
        delta = {}
        offset = 0
        for var_id in var_ids:
            dim = var_dims[var_id]
            delta[var_id] = x[offset : offset + dim]
            offset += dim

        return delta


class LevenbergMarquardtOptimizer(GaussNewtonOptimizer):
    """
    Levenberg-Marquardt optimizer for nonlinear least squares problems.

    This optimizer extends Gauss-Newton with adaptive damping to improve
    robustness and convergence in difficult problems.
    """

    def __init__(
        self,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
        initial_lambda: float = 1e-4,
        lambda_factor: float = 10.0,
        min_lambda: float = 1e-10,
        max_lambda: float = 1e10,
        verbose: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize Levenberg-Marquardt optimizer.

        Args:
            max_iterations: Maximum number of iterations
            convergence_threshold: Threshold for convergence check
            initial_lambda: Initial damping parameter
            lambda_factor: Factor for changing lambda
            min_lambda: Minimum value for lambda
            max_lambda: Maximum value for lambda
            verbose: Whether to print optimization progress
            device: PyTorch device
        """
        super().__init__(
            max_iterations, convergence_threshold, initial_lambda, verbose, device
        )
        self.lambda_value = initial_lambda
        self.lambda_factor = lambda_factor
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda

    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
        """
        Solve an optimization problem using Levenberg-Marquardt algorithm.

        Args:
            problem: Optimization problem to solve

        Returns:
            Optimization result
        """
        start_time = time.time()

        # Get initial values
        variables = problem.get_initial_values()

        # Move variables to device
        for var_id, var_value in variables.items():
            variables[var_id] = var_value.to(self.device)

        # Compute initial cost
        initial_cost = problem.compute_cost(variables)
        current_cost = initial_cost

        # Track convergence
        convergence_info = {
            "costs": [initial_cost],
            "update_norms": [],
            "lambda_values": [self.lambda_value],
        }

        # Main optimization loop
        iteration = 0
        for iteration in range(self.max_iterations):
            # Compute residuals and Jacobians
            residuals = problem.compute_residuals(variables)
            jacobians = problem.compute_jacobians(variables)

            # Build normal equations with current lambda
            JTJ = {}
            JTr = {}

            # Initialize JTr
            for var_id, var_dim in problem.get_variable_dimensions().items():
                JTr[var_id] = torch.zeros(var_dim, device=self.device)

            # Build JTJ and JTr
            for var_id, jacobian in jacobians.items():
                JTr[var_id] = torch.matmul(jacobian.t(), residuals)

                var_dim = problem.get_variable_dimensions()[var_id]
                JTJ[(var_id, var_id)] = torch.matmul(jacobian.t(), jacobian)

                # Apply Levenberg-Marquardt damping to diagonal blocks
                diag = torch.diag(JTJ[(var_id, var_id)])
                damping = self.lambda_value * diag
                JTJ[(var_id, var_id)] += torch.diag(damping)

                for other_id, other_jacobian in jacobians.items():
                    if var_id != other_id:
                        JTJ[(var_id, other_id)] = torch.matmul(
                            jacobian.t(), other_jacobian
                        )

            # Solve normal equations
            delta = self._solve_normal_equations(
                JTJ, JTr, problem.get_variable_dimensions()
            )

            # Try the update
            new_variables = {}
            for var_id, var_value in variables.items():
                new_variables[var_id] = var_value + delta[var_id]

            # Compute new cost
            new_cost = problem.compute_cost(new_variables)

            # Compute update norm
            update_norm = 0.0
            for var_id, var_delta in delta.items():
                update_norm += torch.norm(var_delta).item() ** 2
            update_norm = update_norm**0.5

            # Check if update improved the cost
            if new_cost < current_cost:
                # Accept update
                variables = new_variables
                current_cost = new_cost

                # Decrease lambda (become more like Gauss-Newton)
                self.lambda_value = max(
                    self.min_lambda, self.lambda_value / self.lambda_factor
                )

                # Check convergence
                if self.measure_convergence(current_cost, new_cost, update_norm):
                    break
            else:
                # Reject update and increase lambda (become more like gradient descent)
                self.lambda_value = min(
                    self.max_lambda, self.lambda_value * self.lambda_factor
                )

            # Track convergence
            convergence_info["costs"].append(current_cost)
            convergence_info["update_norms"].append(update_norm)
            convergence_info["lambda_values"].append(self.lambda_value)

            # Print progress
            if self.verbose and (
                iteration % 5 == 0 or iteration == self.max_iterations - 1
            ):
                print(
                    f"Iteration {iteration}: cost = {current_cost:.6f}, lambda = {self.lambda_value:.6e}"
                )

        end_time = time.time()

        # Create result
        result = OptimizationResult(
            success=True,
            initial_cost=initial_cost,
            final_cost=current_cost,
            variables=variables,
            num_iterations=iteration + 1,
            time_seconds=end_time - start_time,
            convergence_info=convergence_info,
        )

        return result
