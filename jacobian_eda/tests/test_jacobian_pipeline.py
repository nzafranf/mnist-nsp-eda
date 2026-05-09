"""
Test pipeline for Jacobian EDA with dummy data.

Validates the pipeline before running on real model.
"""

import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.jacobian_utils import (
    power_iteration_spectral_norm,
    compute_geodesic_deviation_grid,
    compute_straightenability_mask
)


class DummyVelocityField:
    """
    Simple test velocity field: v(x,t) = -2*x*t (quadratic decay)

    This has ∇_x v = -2*t*I, so spectral norm = 2*t
    Integral from t_a to t_b: 2*(t_b^2 - t_a^2) / 2 = t_b^2 - t_a^2
    """

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Dummy velocity field: v(x, t) = -2 * x * t
        """
        if isinstance(t, (int, float)):
            t_val = torch.tensor(t, dtype=x.dtype, device=x.device)
        else:
            t_val = t

        return -2.0 * x * t_val


def generate_dummy_trajectories(num_trajectories: int = 5, num_steps: int = 20,
                                 dim: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate dummy trajectories using Euler integration.

    Integrate x' = -2*x*t with x(0) ~ N(0,1)

    Returns:
        trajectories: [num_trajectories, num_steps, dim]
        times: [num_steps]
    """
    device = torch.device('cpu')
    velocity_fn = DummyVelocityField()

    times = np.linspace(0, 0.999, num_steps)
    dt = times[1] - times[0]

    trajectories = np.zeros((num_trajectories, num_steps, dim))
    spectral_norms = np.zeros((num_trajectories, num_steps))

    for traj_idx in range(num_trajectories):
        # Initial state
        x = torch.randn(1, dim, device=device)
        trajectories[traj_idx, 0] = x.cpu().numpy()

        # Spectral norm at t=0: ||∇_x v|| = 2*0 = 0
        spectral_norms[traj_idx, 0] = 0.0

        # Euler integration
        for step_idx, t in enumerate(times[:-1]):
            # Compute velocity
            vel = velocity_fn(x, torch.tensor(t, dtype=x.dtype, device=device))

            # Euler step
            x = x + vel * dt

            # Store
            trajectories[traj_idx, step_idx + 1] = x.detach().cpu().numpy()

            # Spectral norm at this time: ||∇_x v|| = 2*t
            spectral_norms[traj_idx, step_idx + 1] = 2.0 * (t + dt)

    return trajectories, spectral_norms, times


def test_pipeline():
    """Run test pipeline with dummy data."""
    print("=" * 80)
    print("JACOBIAN EDA TEST PIPELINE")
    print("=" * 80)
    print()

    device = torch.device('cpu')

    # 1. Generate dummy trajectories
    print("[1/4] Generating dummy trajectories...")
    trajectories, spectral_norms, times = generate_dummy_trajectories(
        num_trajectories=5,
        num_steps=20,
        dim=4
    )
    print(f"  Trajectories shape: {trajectories.shape}")
    print(f"  Spectral norms shape: {spectral_norms.shape}")
    print()

    # 2. Compute geodesic deviation grid
    print("[2/4] Computing geodesic deviation integrals...")
    integrals, t_grid, stats = compute_geodesic_deviation_grid(
        trajectories, spectral_norms, grid_size=30
    )
    print(f"  Integrals shape: {integrals.shape}")
    print(f"  Mean integral range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"  Mean heatmap: min={stats['mean'].min():.4f}, max={stats['mean'].max():.4f}")
    print(f"  Std heatmap: min={stats['std'].min():.4f}, max={stats['std'].max():.4f}")
    print()

    # 3. Compute straightenability mask
    print("[3/4] Computing straightenability mask...")
    epsilon = 0.3
    mask = compute_straightenability_mask(stats['mean'], stats['std'], epsilon=epsilon)
    straightenable_count = mask.sum()
    total_count = mask.size
    pct = 100.0 * straightenable_count / total_count
    print(f"  Straightenable regions (eps={epsilon}): {straightenable_count}/{total_count} ({pct:.1f}%)")
    print()

    # 4. Visualize
    print("[4/4] Creating visualizations...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Mean integral
    sns.heatmap(stats['mean'], ax=axes[0], cmap='viridis', cbar_kws={'label': 'Mean Integral'})
    axes[0].set_title('Mean Geodesic Deviation')
    axes[0].set_xlabel('t_b')
    axes[0].set_ylabel('t_a')

    # Std dev integral
    sns.heatmap(stats['std'], ax=axes[1], cmap='plasma', cbar_kws={'label': 'Std Dev'})
    axes[1].set_title('Std Dev Geodesic Deviation')
    axes[1].set_xlabel('t_b')
    axes[1].set_ylabel('t_a')

    # Straightenability mask
    sns.heatmap(mask.astype(float), ax=axes[2], cmap='RdYlGn', cbar_kws={'label': 'Straightenable'},
                vmin=0, vmax=1)
    axes[2].set_title(f'Straightenability Mask (ε={epsilon})')
    axes[2].set_xlabel('t_b')
    axes[2].set_ylabel('t_a')

    plt.tight_layout()
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / 'test_jacobian_heatmaps.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved heatmaps to {save_path}")
    plt.close()

    print()
    print("=" * 80)
    print("TEST COMPLETE - Pipeline is error-free!")
    print("=" * 80)
    print()
    print("Next: Run jacobian_eda/scripts/compute_jacobian_eda.py on real model")


if __name__ == '__main__':
    test_pipeline()
