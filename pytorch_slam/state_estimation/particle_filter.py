import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from .base import MeasurementModel, MotionModel, State, StateEstimator


class Particle:
    """
    Particle for particle filter.

    Each particle represents a possible state of the system.
    """

    def __init__(self, state: torch.Tensor, weight: float = 1.0):
        """
        Initialize particle.

        Args:
            state: State vector
            weight: Particle weight
        """
        self.state = state
        self.weight = weight

    def __repr__(self) -> str:
        """String representation of particle."""
        return f"Particle(state={self.state}, weight={self.weight})"


class ParticleFilter(StateEstimator):
    """
    Particle filter for state estimation.

    Particle filters approximate the posterior distribution using a set of weighted particles.
    They can handle arbitrary distributions and are well-suited for highly nonlinear problems.
    """

    def __init__(
        self,
        motion_model: MotionModel,
        num_particles: int = 100,
        initial_state: Optional[State] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize particle filter.

        Args:
            motion_model: Motion model
            num_particles: Number of particles
            initial_state: Initial state (optional)
            device: PyTorch device
        """
        super().__init__(motion_model.state_dim, device)

        self.motion_model = motion_model
        self.num_particles = num_particles

        # Initialize particles
        self.particles = []

        # If initial state is provided, initialize particles
        if initial_state is not None:
            self.initialize_particles(initial_state)
            self.last_update_time = time.time()

    def initialize_particles(self, initial_state: State):
        """
        Initialize particles around the initial state.

        Args:
            initial_state: Initial state
        """
        # Clear existing particles
        self.particles = []

        # Check if covariance is available for sampling
        if initial_state.covariance is not None:
            # Sample from multivariate normal distribution
            dist = MultivariateNormal(initial_state.mean, initial_state.covariance)
            particle_states = dist.sample((self.num_particles,))

            # Create particles
            for i in range(self.num_particles):
                self.particles.append(
                    Particle(particle_states[i], 1.0 / self.num_particles)
                )
        else:
            # If no covariance, use identical particles
            for i in range(self.num_particles):
                self.particles.append(
                    Particle(initial_state.mean.clone(), 1.0 / self.num_particles)
                )

        # Compute state estimate from particles
        self._update_state_from_particles()

    def predict(self, control: torch.Tensor, dt: float) -> State:
        """
        Predict step of the particle filter.

        Args:
            control: Control input
            dt: Time step

        Returns:
            Predicted state
        """
        if not self.particles:
            raise RuntimeError(
                "Particle filter has not been initialized with particles"
            )

        # Process noise covariance
        Q = self.motion_model.noise_covariance(self.state, control, dt)

        # Process each particle
        for i in range(len(self.particles)):
            # Get current particle state
            particle_state = self.particles[i].state

            # Create a temporary State object for the motion model
            temp_state = State(particle_state)

            # Predict next state using motion model
            predicted_state = self.motion_model.predict(temp_state, control, dt)

            # Add process noise
            if Q is not None:
                noise_dist = MultivariateNormal(
                    torch.zeros(self.state_dim, device=self.device), Q
                )
                noise = noise_dist.sample()
                predicted_mean = predicted_state.mean + noise
            else:
                predicted_mean = predicted_state.mean

            # Update particle
            self.particles[i].state = predicted_mean

        # Update state estimate from particles
        self._update_state_from_particles()

        # Update time of last state update
        self.last_update_time = time.time()

        return self.state

    def update(
        self,
        measurement: torch.Tensor,
        measurement_func: Callable[[torch.Tensor], torch.Tensor],
        measurement_noise: torch.Tensor,
    ) -> State:
        """
        Update step of the particle filter.

        Args:
            measurement: Measurement vector
            measurement_func: Function that maps state to measurement
            measurement_noise: Measurement noise covariance

        Returns:
            Updated state
        """
        if not self.particles:
            raise RuntimeError(
                "Particle filter has not been initialized with particles"
            )

        # Create measurement noise distribution
        noise_dist = MultivariateNormal(
            torch.zeros_like(measurement), measurement_noise
        )

        # Update weights of each particle
        max_log_weight = -float("inf")
        log_weights = []

        for i, particle in enumerate(self.particles):
            # Predict measurement for this particle
            predicted_measurement = measurement_func(particle.state)

            # Compute measurement likelihood
            log_likelihood = noise_dist.log_prob(measurement - predicted_measurement)

            # Update particle log-weight
            log_weight = np.log(particle.weight) + log_likelihood.item()
            log_weights.append(log_weight)

            # Track maximum log-weight for numerical stability
            max_log_weight = max(max_log_weight, log_weight)

        # Normalize weights
        total_weight = 0.0
        for i, log_weight in enumerate(log_weights):
            # Subtract max_log_weight for numerical stability
            weight = np.exp(log_weight - max_log_weight)
            self.particles[i].weight = weight
            total_weight += weight

        # Normalize weights
        if total_weight > 0:
            for particle in self.particles:
                particle.weight /= total_weight
        else:
            # If all weights are zero, reset to uniform weights
            for particle in self.particles:
                particle.weight = 1.0 / len(self.particles)

        # Compute effective sample size
        ess = 1.0 / sum(p.weight**2 for p in self.particles)

        # If effective sample size is too small, resample
        if ess < self.num_particles / 2:
            self._resample()

        # Update state estimate from particles
        self._update_state_from_particles()

        # Update time of last state update
        self.last_update_time = time.time()

        return self.state

    def reset(self, initial_state: Optional[State] = None):
        """
        Reset the particle filter.

        Args:
            initial_state: Optional initial state
        """
        # Clear particles
        self.particles = []

        # Initialize with provided state if available
        if initial_state is not None:
            self.initialize_particles(initial_state)
            self.last_update_time = time.time()
        else:
            self.state = None
            self.last_update_time = None

    def _resample(self):
        """
        Resample particles based on their weights.

        This uses the systematic resampling algorithm for lower variance.
        """
        # Extract weights
        weights = torch.tensor([p.weight for p in self.particles], device=self.device)

        # Prepare new particles
        new_particles = []

        # Compute cumulative sum of weights
        cumsum = torch.cumsum(weights, dim=0)

        # Scale cumulative sum to [0, 1]
        cumsum = cumsum / cumsum[-1]

        # Draw starting point from uniform distribution
        u = torch.rand(1, device=self.device) / self.num_particles

        # Systematic resampling
        i = 0
        for j in range(self.num_particles):
            while u > cumsum[i]:
                i += 1

            # Add new particle (clone state to avoid reference issues)
            new_particles.append(
                Particle(self.particles[i].state.clone(), 1.0 / self.num_particles)
            )

            # Update u
            u += 1.0 / self.num_particles

        # Replace old particles
        self.particles = new_particles

    def _update_state_from_particles(self):
        """
        Update the state estimate from particles.

        Computes the weighted mean and covariance of the particles.
        """
        # Check if we have particles
        if not self.particles:
            return

        # Extract states and weights
        states = torch.stack([p.state for p in self.particles])
        weights = torch.tensor([p.weight for p in self.particles], device=self.device)

        # Compute weighted mean
        mean = torch.sum(weights.unsqueeze(1) * states, dim=0)

        # Compute weighted covariance
        centered = states - mean
        cov = torch.zeros((self.state_dim, self.state_dim), device=self.device)

        for i in range(self.num_particles):
            outer_product = torch.outer(centered[i], centered[i])
            cov += weights[i] * outer_product

        # Create state
        self.state = State(mean, cov)

    def get_particles(self) -> List[Particle]:
        """
        Get the current particles.

        Returns:
            List of particles
        """
        return self.particles


class RaoBlackwellizedParticleFilter(ParticleFilter):
    """
    Rao-Blackwellized Particle Filter (RBPF) for SLAM.

    RBPF is a variant of particle filter that factorizes the state into
    a part estimated by particles (typically robot pose) and a part estimated
    analytically (typically map features).
    """

    def __init__(
        self,
        motion_model: MotionModel,
        num_particles: int = 100,
        initial_state: Optional[State] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize Rao-Blackwellized particle filter.

        Args:
            motion_model: Motion model
            num_particles: Number of particles
            initial_state: Initial state (optional)
            device: PyTorch device
        """
        super().__init__(motion_model, num_particles, initial_state, device)

        # Map features for each particle
        self.particle_maps = [None] * num_particles

    def predict(self, control: torch.Tensor, dt: float) -> State:
        """
        Predict step of the RBPF.

        Args:
            control: Control input
            dt: Time step

        Returns:
            Predicted state
        """
        # Update particle poses using parent class method
        state = super().predict(control, dt)

        # No need to update map features in prediction step

        return state

    def update(
        self,
        measurement: torch.Tensor,
        measurement_func: Callable[[torch.Tensor], torch.Tensor],
        measurement_noise: torch.Tensor,
        feature_id: Optional[int] = None,
    ) -> State:
        """
        Update step of the RBPF.

        Args:
            measurement: Measurement vector
            measurement_func: Function that maps state to measurement
            measurement_noise: Measurement noise covariance
            feature_id: ID of the observed map feature (optional)

        Returns:
            Updated state
        """
        if not self.particles:
            raise RuntimeError("RBPF has not been initialized with particles")

        # Create measurement noise distribution
        noise_dist = MultivariateNormal(
            torch.zeros_like(measurement), measurement_noise
        )

        # Update weights and map features for each particle
        max_log_weight = -float("inf")
        log_weights = []

        for i, particle in enumerate(self.particles):
            # If feature_id is provided, update map feature
            if feature_id is not None and self.particle_maps[i] is not None:
                # Get current feature estimate
                feature_mean, feature_cov = self.particle_maps[i].get(
                    feature_id, (None, None)
                )

                if feature_mean is not None:
                    # Update feature using EKF update
                    # TODO: Implement feature update
                    pass
                else:
                    # Initialize new feature
                    # TODO: Implement feature initialization
                    pass

            # Predict measurement for this particle
            predicted_measurement = measurement_func(particle.state)

            # Compute measurement likelihood
            log_likelihood = noise_dist.log_prob(measurement - predicted_measurement)

            # Update particle log-weight
            log_weight = np.log(particle.weight) + log_likelihood.item()
            log_weights.append(log_weight)

            # Track maximum log-weight for numerical stability
            max_log_weight = max(max_log_weight, log_weight)

        # Normalize weights
        total_weight = 0.0
        for i, log_weight in enumerate(log_weights):
            # Subtract max_log_weight for numerical stability
            weight = np.exp(log_weight - max_log_weight)
            self.particles[i].weight = weight
            total_weight += weight

        # Normalize weights
        if total_weight > 0:
            for particle in self.particles:
                particle.weight /= total_weight
        else:
            # If all weights are zero, reset to uniform weights
            for particle in self.particles:
                particle.weight = 1.0 / len(self.particles)

        # Compute effective sample size
        ess = 1.0 / sum(p.weight**2 for p in self.particles)

        # If effective sample size is too small, resample
        if ess < self.num_particles / 2:
            self._resample_with_maps()

        # Update state estimate from particles
        self._update_state_from_particles()

        # Update time of last state update
        self.last_update_time = time.time()

        return self.state

    def _resample_with_maps(self):
        """
        Resample particles and their associated maps.
        """
        # Extract weights
        weights = torch.tensor([p.weight for p in self.particles], device=self.device)

        # Prepare new particles and maps
        new_particles = []
        new_maps = []

        # Compute cumulative sum of weights
        cumsum = torch.cumsum(weights, dim=0)

        # Scale cumulative sum to [0, 1]
        cumsum = cumsum / cumsum[-1]

        # Draw starting point from uniform distribution
        u = torch.rand(1, device=self.device) / self.num_particles

        # Systematic resampling
        i = 0
        for j in range(self.num_particles):
            while u > cumsum[i]:
                i += 1

            # Add new particle (clone state to avoid reference issues)
            new_particles.append(
                Particle(self.particles[i].state.clone(), 1.0 / self.num_particles)
            )

            # Clone map if exists
            if self.particle_maps[i] is not None:
                new_maps.append(self.particle_maps[i].copy())
            else:
                new_maps.append(None)

            # Update u
            u += 1.0 / self.num_particles

        # Replace old particles and maps
        self.particles = new_particles
        self.particle_maps = new_maps

    def reset(self, initial_state: Optional[State] = None):
        """
        Reset the RBPF.

        Args:
            initial_state: Optional initial state
        """
        # Reset particles
        super().reset(initial_state)

        # Reset maps
        self.particle_maps = [None] * self.num_particles
