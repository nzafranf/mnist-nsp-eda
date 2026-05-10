#!/usr/bin/env python
"""
Jacobian EDA on Flow Matching MNIST Model

Computes geodesic deviation integral to identify linearizable (straightenable) time intervals.

References:
- Theorem 4.1 (Linearizability)
- Section IV (Jacobi Fields) from "Adaptive and Neural Schedule Predictor" paper
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add paths
project_root = Path(__file__).resolve().parent.parent.parent  # flow-matching-mnist
jacobian_eda_root = Path(__file__).resolve().parent.parent     # jacobian_eda

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(jacobian_eda_root))

from models.fm import ImageFlowMatcher
from flow_matching.solver import ODESolver

# Import jacobian utilities (from parent directory)
import importlib.util
utils_path = jacobian_eda_root / 'utils' / 'jacobian_utils.py'
spec = importlib.util.spec_from_file_location("jacobian_utils", utils_path)
jacobian_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jacobian_utils)

finite_difference_spectral_norm = jacobian_utils.finite_difference_spectral_norm
compute_geodesic_deviation_grid = jacobian_utils.compute_geodesic_deviation_grid
compute_straightenability_mask = jacobian_utils.compute_straightenability_mask


class JacobianEDAConfig:
    """Configuration for Jacobian EDA."""

    def __init__(self):
        self.ckpt_path = "outputs/fm/2026-05-01/22-58-12/training_logs/version_0/checkpoints/fm-balanced-epoch=014-train_loss=0.1829.ckpt"
        self.num_trajectories = 50
        self.num_steps_integration = 100
        self.grid_size = 50
        self.power_iterations = 5
        self.epsilon_straightenable = 0.2
        self.device = torch.device('cpu')
        self.results_dir = Path("jacobian_eda/results")


def load_model(config: JacobianEDAConfig) -> ImageFlowMatcher:
    """Load the trained Flow Matching model."""
    print(f"Loading model from {config.ckpt_path}...")
    model = ImageFlowMatcher.load_from_checkpoint(str(config.ckpt_path))
    model.eval()
    model = model.to(config.device)
    return model


def generate_trajectories(model: ImageFlowMatcher, config: JacobianEDAConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate trajectories with spectral norm computation at each step.

    Returns:
        trajectories: [N, num_steps, 784]
        spectral_norms: [N, num_steps]
    """
    print(f"\nGenerating {config.num_trajectories} trajectories with {config.num_steps_integration} steps...")

    trajectories = np.zeros((config.num_trajectories, config.num_steps_integration, 28 * 28))
    spectral_norms = np.zeros((config.num_trajectories, config.num_steps_integration))

    time_grid = torch.linspace(0, 0.999, config.num_steps_integration, device=config.device)

    # Phase 1: Generate trajectories (no grad needed)
    trajectory_tensors = []

    with torch.no_grad():
        for traj_idx in tqdm(range(config.num_trajectories), desc="Generating Trajectories"):
            # Initialize random noise
            x_init = torch.randn(1, 1, 28, 28, device=config.device)

            # Get ODE solver
            solver = ODESolver(velocity_model=model.model)

            # Solve ODE and get intermediate states
            traj_output = solver.sample(
                x_init=x_init,
                time_grid=time_grid,
                method='dopri5',
                step_size=None,
                atol=1e-4,
                rtol=1e-4,
                return_intermediates=True
            )

            # Handle output format
            if isinstance(traj_output, (list, tuple)):
                # Stack along time dimension: [num_steps, 1, 1, 28, 28]
                traj_tensor = torch.stack(traj_output, dim=0)
                # Squeeze out extra batch dimensions: [num_steps, 1, 28, 28]
                while traj_tensor.ndim > 4:
                    traj_tensor = traj_tensor.squeeze(1)
            else:
                traj_tensor = traj_output
                if traj_tensor.shape[0] != len(time_grid):
                    traj_tensor = traj_tensor.transpose(0, 1)

            # Denormalize
            if model.normalize_data:
                traj_tensor = (traj_tensor + 1) / 2
                traj_tensor = torch.clamp(traj_tensor, 0, 1)

            # Store trajectory (flatten to 784)
            traj_flat = traj_tensor.reshape(config.num_steps_integration, -1).cpu().numpy()
            trajectories[traj_idx] = traj_flat

            # Keep tensor version for spectral norm computation
            trajectory_tensors.append(traj_tensor)

    # Phase 2: Compute spectral norms WITH gradient computation enabled
    print("\nComputing spectral norms (gradients enabled)...")
    for traj_idx in tqdm(range(config.num_trajectories), desc="Spectral Norms"):
        traj_tensor = trajectory_tensors[traj_idx].to(config.device)  # Original shape from stack

        # Force correct shape by checking and reshaping
        while traj_tensor.ndim > 4:
            traj_tensor = traj_tensor.squeeze(1)

        for step_idx in range(config.num_steps_integration):
            # Use index to get single frame and add batch dimension
            x_single = traj_tensor[step_idx]  # [1, 28, 28] or [channels, height, width]
            # Ensure it's [1, channels, height, width]
            if x_single.ndim == 3:
                x_t = x_single.unsqueeze(0)  # Add batch dim
            else:
                x_t = x_single.unsqueeze(0)

            t_scalar = time_grid[step_idx]  # Keep as tensor

            try:
                spec_norm = finite_difference_spectral_norm(
                    velocity_fn=model.model,
                    x=x_t,
                    t=t_scalar,
                    num_samples=5,
                    eps=1e-4,
                    device=config.device
                )
                spectral_norms[traj_idx, step_idx] = spec_norm
            except Exception as e:
                print(f"  Warning: Failed to compute spectral norm at traj {traj_idx}, step {step_idx}: {e}")
                spectral_norms[traj_idx, step_idx] = 0.0

    return trajectories, spectral_norms


def save_artifacts(config: JacobianEDAConfig, trajectories: np.ndarray, spectral_norms: np.ndarray,
                  integrals: np.ndarray, t_grid: np.ndarray, stats: dict, mask: np.ndarray):
    """Save all artifacts to disk."""
    config.results_dir.mkdir(parents=True, exist_ok=True)

    # Save numerical data
    print("\nSaving artifacts...")

    np.save(config.results_dir / "trajectories.npy", trajectories)
    np.save(config.results_dir / "spectral_norms.npy", spectral_norms)
    np.save(config.results_dir / "integrals.npy", integrals)
    np.save(config.results_dir / "t_grid.npy", t_grid)
    np.save(config.results_dir / "straightenability_mask.npy", mask)

    print(f"  Saved trajectory data and integrals")

    # Save statistics as JSON
    stats_json = {
        'mean_integral': stats['mean'].tolist(),
        'std_integral': stats['std'].tolist(),
        'min_integral': float(stats['min']),
        'max_integral': float(stats['max']),
        'num_trajectories': config.num_trajectories,
        'grid_size': config.grid_size,
        'num_steps': config.num_steps_integration,
        'epsilon': config.epsilon_straightenable,
        'straightenable_count': int(mask.sum()),
        'total_regions': int(mask.size),
        'straightenable_percentage': 100.0 * mask.sum() / mask.size,
        'timestamp': datetime.now().isoformat()
    }

    with open(config.results_dir / "statistics.json", 'w') as f:
        json.dump(stats_json, f, indent=2)

    print(f"  Saved statistics")


def create_enhanced_lower_triangle_viz(config: JacobianEDAConfig, stats: dict, mask: np.ndarray):
    """Create enhanced visualization with lower triangle only and KDE bounds."""
    from scipy.ndimage import gaussian_filter

    print("\nCreating enhanced lower triangle visualization with KDE bounds...")

    mean_integral = stats['mean']

    # Create lower triangle mask
    grid_size = config.grid_size
    lower_triangle = np.tril(np.ones((grid_size, grid_size)))

    # Mask upper triangle with NaNs for better visualization
    mean_integral_lt = np.where(lower_triangle, mean_integral, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Jacobian EDA: Lower Triangle Analysis with KDE Bounds',
                fontsize=14, fontweight='bold')

    # Panel 1: Heatmap with KDE contours
    im0 = axes[0].imshow(mean_integral_lt, cmap='viridis', aspect='auto', origin='lower',
                        vmin=mean_integral.min(), vmax=mean_integral.max())
    axes[0].set_title('Geodesic Deviation (Lower Triangle) with Threshold Boundaries',
                     fontsize=13, fontweight='bold')
    axes[0].set_xlabel('t_b (end time)', fontsize=12)
    axes[0].set_ylabel('t_a (start time)', fontsize=12)

    # Add KDE contour bounds for various epsilon thresholds
    epsilon_thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    colors = plt.cm.Set1(np.linspace(0, 1, len(epsilon_thresholds)))

    # Create a smoothed version for contours using KDE-like gaussian filter
    mean_smooth = gaussian_filter(mean_integral, sigma=1.5)

    # Add contour lines for each threshold
    x_grid = np.arange(grid_size)
    y_grid = np.arange(grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)

    for eps_thresh, color in zip(epsilon_thresholds, colors):
        contours = axes[0].contour(X, Y, mean_smooth, levels=[eps_thresh],
                                  colors=[color], linewidths=2.5, alpha=0.8)
        # Label the contour
        axes[0].clabel(contours, inline=True, fontsize=9, fmt=f'ε={eps_thresh:.2f}')

    cbar0 = plt.colorbar(im0, ax=axes[0])
    cbar0.set_label('Geodesic Deviation Integral', fontsize=11)

    # Panel 2: Straightenability mask with lower triangle
    mask_lt = np.where(lower_triangle, mask.astype(float), np.nan)

    im1 = axes[1].imshow(mask_lt, cmap='RdYlGn', aspect='auto', origin='lower',
                        vmin=0, vmax=1)
    axes[1].set_title(f'Straightenability Mask (ε={config.epsilon_straightenable})',
                     fontsize=13, fontweight='bold')
    axes[1].set_xlabel('t_b (end time)', fontsize=12)
    axes[1].set_ylabel('t_a (start time)', fontsize=12)

    cbar1 = plt.colorbar(im1, ax=axes[1])
    cbar1.set_ticks([0, 1])
    cbar1.set_ticklabels(['No', 'Yes'])
    cbar1.set_label('Straightenable', fontsize=11)

    plt.tight_layout()
    save_path = config.results_dir / "jacobian_eda_lower_triangle_kde.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved enhanced visualization to {save_path}")
    plt.close()


def create_visualizations(config: JacobianEDAConfig, stats: dict, mask: np.ndarray, t_grid: np.ndarray):
    """Create and save visualization heatmaps."""
    print("\nCreating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Mean integral
    im0 = axes[0, 0].imshow(stats['mean'], cmap='viridis', aspect='auto', origin='lower')
    axes[0, 0].set_title('Mean Geodesic Deviation I(t_a, t_b)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('t_b (end time)', fontsize=11)
    axes[0, 0].set_ylabel('t_a (start time)', fontsize=11)
    plt.colorbar(im0, ax=axes[0, 0], label='Integral value')

    # Std dev
    im1 = axes[0, 1].imshow(stats['std'], cmap='plasma', aspect='auto', origin='lower')
    axes[0, 1].set_title('Std Dev Geodesic Deviation', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('t_b (end time)', fontsize=11)
    axes[0, 1].set_ylabel('t_a (start time)', fontsize=11)
    plt.colorbar(im1, ax=axes[0, 1], label='Std Dev')

    # Straightenability mask
    im2 = axes[1, 0].imshow(mask.astype(float), cmap='RdYlGn', aspect='auto', origin='lower',
                            vmin=0, vmax=1)
    axes[1, 0].set_title(f'Straightenability Mask (Mean+1*Std < {config.epsilon_straightenable})',
                        fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('t_b (end time)', fontsize=11)
    axes[1, 0].set_ylabel('t_a (start time)', fontsize=11)
    cbar = plt.colorbar(im2, ax=axes[1, 0], label='Straightenable')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['No', 'Yes'])

    # Summary statistics text
    straightenable_pct = 100.0 * mask.sum() / mask.size
    axes[1, 1].axis('off')
    summary_text = f"""
JACOBIAN EDA SUMMARY

Number of Trajectories: {config.num_trajectories}
Integration Steps: {config.num_steps_integration}
Grid Resolution: {config.grid_size}x{config.grid_size}
Power Iteration Steps: {config.power_iterations}

Geodesic Deviation Statistics:
  Min Integral: {stats['min']:.6f}
  Max Integral: {stats['max']:.6f}
  Mean Integral Range: [{stats['mean'].min():.6f}, {stats['mean'].max():.6f}]

Straightenability (eps={config.epsilon_straightenable}):
  Straightenable Regions: {int(mask.sum())} / {int(mask.size)}
  Percentage: {straightenable_pct:.1f}%

Interpretation:
  Purple/Dark regions: Low linearization (needs small steps)
  Green/Bright regions: High linearization (can use large steps)
"""
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_path = config.results_dir / "jacobian_eda_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to {save_path}")
    plt.close()


def main():
    """Main EDA pipeline."""
    print("=" * 80)
    print("JACOBIAN EDA - LINEARIZABILITY ANALYSIS")
    print("=" * 80)

    config = JacobianEDAConfig()

    # Load model
    model = load_model(config)

    # Generate trajectories and compute spectral norms
    trajectories, spectral_norms = generate_trajectories(model, config)

    # Compute geodesic deviation integrals
    print("\nComputing geodesic deviation integrals...")
    integrals, t_grid, stats = compute_geodesic_deviation_grid(
        trajectories, spectral_norms, grid_size=config.grid_size
    )
    print(f"  Integrals shape: {integrals.shape}")
    print(f"  Mean integral range: [{stats['mean'].min():.6f}, {stats['mean'].max():.6f}]")

    # Compute straightenability mask
    print("\nComputing straightenability mask...")
    mask = compute_straightenability_mask(
        stats['mean'], stats['std'], epsilon=config.epsilon_straightenable
    )
    straightenable_pct = 100.0 * mask.sum() / mask.size
    print(f"  Straightenable regions: {mask.sum()}/{mask.size} ({straightenable_pct:.1f}%)")

    # Save artifacts
    save_artifacts(config, trajectories, spectral_norms, integrals, t_grid, stats, mask)

    # Create visualizations
    create_visualizations(config, stats, mask, t_grid)

    # Create enhanced lower triangle visualization with KDE bounds
    create_enhanced_lower_triangle_viz(config, stats, mask)

    print("\n" + "=" * 80)
    print("JACOBIAN EDA COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {config.results_dir}")
    print(f"  - trajectories.npy: Full trajectory data")
    print(f"  - spectral_norms.npy: Spectral norms at each step")
    print(f"  - integrals.npy: Geodesic deviation integrals")
    print(f"  - straightenability_mask.npy: Binary mask for straightenable regions")
    print(f"  - statistics.json: Summary statistics")
    print(f"  - jacobian_eda_analysis.png: Visualization heatmaps")
    print(f"  - jacobian_eda_lower_triangle_kde.png: Lower triangle with KDE bounds")


if __name__ == '__main__':
    main()
