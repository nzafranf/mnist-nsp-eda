"""
Instrumented Generation - Captures trajectories during inference
Hooks into the ODE solver to log intermediate states.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Callable
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trajectory_logger import TrajectoryLogger
from models.fm import ImageFlowMatcher


class InstrumentedODESolver:
    """
    Wrapper around ODE solver that logs trajectory data.
    Uses the flow_matching library's ODESolver with instrumentation.
    """

    def __init__(self, velocity_model, logger: Optional[TrajectoryLogger] = None):
        """
        Initialize instrumented ODE solver.

        Args:
            velocity_model: The velocity field neural network
            logger: TrajectoryLogger instance (creates new if None)
        """
        self.velocity_model = velocity_model
        self.logger = logger or TrajectoryLogger(velocity_model)
        self.velocity_calls = 0

    def velocity_wrapper(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Wrapper around velocity model that logs calls.

        Args:
            t: Time(s)
            x: State(s)

        Returns:
            Velocity field value
        """
        # Reshape x to 4D [batch, channels, height, width]
        # ODE solver may pass: [784], [batch, 784], [height, width], or [batch, height, width]
        if x.dim() == 1:
            if x.shape[0] == 784:
                # Flattened single sample: [784] -> [1, 1, 28, 28]
                x = x.view(1, 1, 28, 28)
            else:
                # Other 1D: reshape to 2D
                x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            if x.shape[1] == 784:
                # Batch of flattened samples: [batch, 784] -> [batch, 1, 28, 28]
                x = x.view(x.shape[0], 1, 28, 28)
            elif x.shape[0] == 28 and x.shape[1] == 28:
                # Single sample as 2D: [28, 28] -> [1, 1, 28, 28]
                x = x.unsqueeze(0).unsqueeze(0)
            else:
                # Assume [batch, features], reshape to image
                x = x.view(x.shape[0], 1, 28, 28)
        elif x.dim() == 3:
            if x.shape[1] == 28 and x.shape[2] == 28:
                # Batch of 2D samples: [batch, 28, 28] -> [batch, 1, 28, 28]
                x = x.unsqueeze(1)
            else:
                # Other 3D: try to infer shape
                x = x.unsqueeze(1) if x.shape[0] > 1 else x.unsqueeze(0)
        elif x.dim() != 4:
            # Fallback: pad dimensions until 4D
            while x.dim() < 4:
                x = x.unsqueeze(0)

        # Ensure t is the right shape for the model
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.shape[0] == 1 and x.shape[0] > 1:
            t = t.expand(x.shape[0])

        # Final validation
        assert x.dim() == 4, f"Expected 4D tensor, got {x.dim()}D with shape {x.shape}"

        v = self.velocity_model(x, t)

        # Reshape output back to match expected format from ODE solver
        # If input was 1D or 2D, flatten output
        if v.dim() == 4:
            v = v.view(v.shape[0], -1)  # [batch, 1, 28, 28] -> [batch, 784]
            if v.shape[0] == 1:
                v = v.squeeze(0)  # [1, 784] -> [784]

        self.velocity_calls += 1
        return v

    def sample(
        self,
        time_grid: torch.Tensor,
        x_init: torch.Tensor,
        method: str = 'dopri5',
        step_size: Optional[float] = None,
        atol: float = 1e-4,
        rtol: float = 1e-4,
        target_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Integrate ODE with logging using flow_matching's ODESolver.

        Args:
            time_grid: Times at which to log states
            x_init: Initial state
            method: ODE solver method
            step_size: Step size (if applicable)
            atol: Absolute tolerance
            rtol: Relative tolerance
            target_state: Target state for alignment (optional)

        Returns:
            Integrated trajectory
        """
        from flow_matching.solver import ODESolver

        device = x_init.device
        self.logger.reset()

        # Create ODESolver from flow_matching library
        solver = ODESolver(velocity_model=self.velocity_model)

        # Sample trajectory (return_intermediates=True to get states at all time_grid points)
        trajectory = solver.sample(
            x_init=x_init,
            time_grid=time_grid,
            method=method,
            step_size=step_size,
            atol=atol,
            rtol=rtol,
            return_intermediates=True,
        )

        # Handle return_intermediates=True which returns list of tensors
        if isinstance(trajectory, (list, tuple)):
            # Stack list of tensors into [num_steps, batch, ...]
            trajectory = torch.stack(trajectory, dim=0)
        elif trajectory.shape[0] != len(time_grid):
            # Handle batch-first format [batch, num_steps, ...]
            if trajectory.shape[1] == len(time_grid):
                trajectory = trajectory.transpose(0, 1)

        # Log each timestep
        for t_idx, t_val in enumerate(time_grid):
            state = trajectory[t_idx]

            # Get velocity at this state
            with torch.no_grad():
                t_expanded = t_val.view(1) if t_val.dim() == 0 else t_val
                if t_expanded.shape[0] == 1:
                    t_expanded = t_expanded.expand(state.shape[0])
                velocity = self.velocity_wrapper(t_expanded, state)

            self.logger.log_step(
                state,
                t_val.item() if t_val.dim() > 0 else t_val,
                velocity,
                target_s1=target_state
            )

        return trajectory


def generate_with_logging(
    model: ImageFlowMatcher,
    num_trajectories: int = 10,
    num_steps: int = 50,
    batch_size: int = 1,
    save_dir: str = 'results/phase1',
    target_images: Optional[torch.Tensor] = None,
) -> Tuple[list, list]:
    """
    Generate trajectories with full logging.

    Args:
        model: Trained ImageFlowMatcher model
        num_trajectories: Number of independent trajectories to capture
        num_steps: Number of ODE integration steps
        batch_size: Batch size for generation
        save_dir: Directory to save trajectories
        target_images: Optional target images to use as alignment targets

    Returns:
        Tuple of (loggers_list, trajectories_list)
    """
    model.eval()
    device = model.device

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    loggers = []
    trajectories = []

    print(f"Generating {num_trajectories} instrumented trajectories...")

    for traj_idx in range(num_trajectories):
        logger = TrajectoryLogger(model.model)

        # Create instrumented solver
        solver = InstrumentedODESolver(model.model, logger)

        # Initialize random noise
        x_init = torch.randn(batch_size, 1, 28, 28, device=device)

        # Optional target image for alignment
        target = None
        if target_images is not None and traj_idx < len(target_images):
            target = target_images[traj_idx:traj_idx+1]

        # Solve ODE
        time_grid = torch.linspace(0, 1, num_steps, device=device)

        with torch.no_grad():
            trajectory = solver.sample(
                time_grid=time_grid,
                x_init=x_init,
                method='dopri5',
                atol=1e-4,
                rtol=1e-4,
                target_state=target,
            )

        # Use solver's logger which has the actual logged data
        logger = solver.logger

        # Denormalize if needed
        if model.normalize_data:
            trajectory = (trajectory + 1) / 2
            trajectory = torch.clamp(trajectory, 0.0, 1.0)

        # Save logger
        save_path = Path(save_dir) / f"trajectory_{traj_idx:03d}.npz"
        logger.save(str(save_path))
        loggers.append(logger)
        trajectories.append(trajectory)

        print(f"  [{traj_idx+1}/{num_trajectories}] Saved to {save_path}")

    print(f"\nGeneration complete! Saved {num_trajectories} trajectories to {save_dir}")

    return loggers, trajectories


def load_trajectories(
    save_dir: str = 'results/phase1',
    num_trajectories: Optional[int] = None
) -> list:
    """
    Load previously saved trajectory logs.

    Args:
        save_dir: Directory containing saved trajectories
        num_trajectories: Number of trajectories to load (None = load all)

    Returns:
        List of trajectory dictionaries
    """
    save_path = Path(save_dir)
    trajectory_files = sorted(save_path.glob('trajectory_*.npz'))

    if num_trajectories:
        trajectory_files = trajectory_files[:num_trajectories]

    trajectories = []
    for traj_file in trajectory_files:
        logger = TrajectoryLogger()
        logger.load(str(traj_file))
        traj_dict = logger.get_trajectory_dict()
        trajectories.append(traj_dict)

    print(f"Loaded {len(trajectories)} trajectories from {save_dir}")
    return trajectories


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate instrumented trajectories')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--num-trajectories', type=int, default=20,
                       help='Number of trajectories to generate')
    parser.add_argument('--num-steps', type=int, default=100,
                       help='Number of ODE integration steps')
    parser.add_argument('--save-dir', type=str, default='results/phase1',
                       help='Directory to save trajectories')

    args = parser.parse_args()

    # Load model
    if args.checkpoint:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ImageFlowMatcher()
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
    else:
        print("Error: --checkpoint required")
        exit(1)

    # Generate with logging
    loggers, trajectories = generate_with_logging(
        model,
        num_trajectories=args.num_trajectories,
        num_steps=args.num_steps,
        save_dir=args.save_dir
    )
