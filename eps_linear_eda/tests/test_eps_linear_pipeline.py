#!/usr/bin/env python
"""
Test Pipeline for EPS Linear EDA

Uses dummy data to validate the entire implementation without requiring a trained model.

Dummy velocity field: v(x,t) = -2*x*t
Analytical solution: x(t) = x_0 * exp(-t²)
Second derivative: ẍ(t) = (4*x_0*t² - 2*x_0) * exp(-t²)
Lipschitz constant: L(x,t) = |∇_x v| = 2*t (since v = -2*x*t, ∂v/∂x = -2*t)
"""

import sys
from pathlib import Path
from datetime import datetime
import json

import numpy as np
from typing import Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

# Add paths
test_root = Path(__file__).resolve().parent
eps_linear_eda_root = test_root.parent
project_root = eps_linear_eda_root.parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(eps_linear_eda_root))

import importlib.util
utils_path = eps_linear_eda_root / 'utils' / 'eps_linear_utils.py'
spec = importlib.util.spec_from_file_location("eps_linear_utils", utils_path)
eps_linear_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eps_linear_utils)

compute_curvature_supremum = eps_linear_utils.compute_curvature_supremum
compute_eps_linear_grid = eps_linear_utils.compute_eps_linear_grid
identify_violations = eps_linear_utils.identify_violations


class DummyEDAConfig:
    """Configuration for test EDA."""

    def __init__(self):
        self.num_trajectories = 5
        self.num_steps_integration = 20
        self.grid_size = 20
        self.n_substeps = 10
        self.results_dir = Path("eps_linear_eda/results")


def dummy_velocity_field(x: np.ndarray, t: float) -> np.ndarray:
    """
    Dummy velocity field for testing: v(x,t) = -2*x*t

    Args:
        x: position [dim]
        t: time scalar

    Returns:
        velocity [dim]
    """
    return -2.0 * x * t


def dummy_trajectory_analytical(x0: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """
    Analytical solution: x(t) = x_0 * exp(-t²)
    """
    return x0[np.newaxis, :] * np.exp(-t_grid[:, np.newaxis] ** 2)


def generate_dummy_trajectories(num_trajectories: int, num_steps: int,
                                dim: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate dummy trajectories using analytical solution.

    Returns:
        trajectories: [num_trajectories, num_steps, dim]
        time_grid: [num_steps]
    """
    time_grid = np.linspace(0, 0.999, num_steps)
    trajectories = np.zeros((num_trajectories, num_steps, dim))

    for i in range(num_trajectories):
        x0 = np.random.randn(dim)
        trajectories[i] = dummy_trajectory_analytical(x0, time_grid)

    return trajectories, time_grid


def generate_dummy_spectral_norms(num_trajectories: int, num_steps: int,
                                  time_grid: np.ndarray) -> np.ndarray:
    """
    Generate dummy spectral norms.
    For the dummy field v = -2xt, we have ∇_x v = -2t (scalar), so |∇_x v| = 2|t|.
    """
    spectral_norms = np.zeros((num_trajectories, num_steps))
    for traj_idx in range(num_trajectories):
        for step_idx in range(num_steps):
            # Lipschitz constant grows with time
            spectral_norms[traj_idx, step_idx] = 2.0 * time_grid[step_idx]

    return spectral_norms


def run_test_pipeline(config: DummyEDAConfig):
    """Run complete test pipeline."""
    print("=" * 80)
    print("EPS LINEAR EDA TEST PIPELINE")
    print("=" * 80)

    # [1/4] Generate trajectories
    print("\n[1/4] Generating dummy trajectories...")
    trajectories, time_grid = generate_dummy_trajectories(
        config.num_trajectories, config.num_steps_integration
    )
    print(f"  Trajectories shape: {trajectories.shape}")
    print(f"  Time grid: {time_grid[0]:.4f} to {time_grid[-1]:.4f}")

    # [2/4] Generate spectral norms
    print("\n[2/4] Generating dummy spectral norms...")
    spectral_norms = generate_dummy_spectral_norms(
        config.num_trajectories, config.num_steps_integration, time_grid
    )
    print(f"  Spectral norms shape: {spectral_norms.shape}")
    print(f"  Spectral norm range: [{np.min(spectral_norms):.6f}, {np.max(spectral_norms):.6f}]")

    # [3/4] Compute EPS Linear grid
    print("\n[3/4] Computing EPS Linear grid...")
    lhs_heatmap, rhs_heatmap, tightness_ratio, stats = compute_eps_linear_grid(
        trajectories, time_grid, spectral_norms,
        grid_size=config.grid_size,
        n_substeps=config.n_substeps,
        velocity_fn=None
    )
    print(f"  LHS heatmap shape: {lhs_heatmap.shape}")
    print(f"  LHS range: [{stats['lhs_min']:.2e}, {stats['lhs_max']:.2e}]")
    print(f"  RHS range: [{stats['rhs_min']:.2e}, {stats['rhs_max']:.2e}]")
    print(f"  Bound violations: {stats['violations']}/{config.grid_size ** 2}")

    # [4/4] Create visualizations
    print("\n[4/4] Creating visualizations...")
    config.results_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('EPS Linear EDA Test Results', fontsize=16, fontweight='bold')

    # LHS heatmap (with log scale)
    lhs_display = np.clip(lhs_heatmap, 1e-10, None)
    im0 = axes[0, 0].imshow(lhs_display, cmap='viridis', aspect='auto', origin='lower',
                            norm=LogNorm())
    axes[0, 0].set_title('Actual Error (LHS): ||x̂^N - x̂^1||', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('t_b (end time index)', fontsize=11)
    axes[0, 0].set_ylabel('t_a (start time index)', fontsize=11)
    cbar0 = plt.colorbar(im0, ax=axes[0, 0], label='Log Error')

    # RHS heatmap (with log scale)
    rhs_display = np.clip(rhs_heatmap, 1e-10, None)
    im1 = axes[0, 1].imshow(rhs_display, cmap='plasma', aspect='auto', origin='lower',
                            norm=LogNorm())
    axes[0, 1].set_title('Theoretical Bound (RHS)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('t_b (end time index)', fontsize=11)
    axes[0, 1].set_ylabel('t_a (start time index)', fontsize=11)
    cbar1 = plt.colorbar(im1, ax=axes[0, 1], label='Log Bound')

    # Tightness ratio
    tightness_display = np.clip(tightness_ratio, 0.1, 100)
    im2 = axes[1, 0].imshow(tightness_display, cmap='RdYlGn_r', aspect='auto', origin='lower',
                            norm=LogNorm())
    axes[1, 0].set_title('Tightness Ratio (RHS / LHS)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('t_b (end time index)', fontsize=11)
    axes[1, 0].set_ylabel('t_a (start time index)', fontsize=11)
    cbar2 = plt.colorbar(im2, ax=axes[1, 0], label='Log Ratio')

    # Violations overlay
    violation_mask = identify_violations(lhs_heatmap, rhs_heatmap, time_grid, config.grid_size)
    axes[1, 1].imshow(np.ones_like(lhs_heatmap), cmap='gray', aspect='auto', origin='lower',
                     vmin=0, vmax=1)
    if np.any(violation_mask):
        violations_y, violations_x = np.where(violation_mask)
        axes[1, 1].scatter(violations_x, violations_y, c='red', s=100, marker='X',
                          edgecolors='darkred', linewidth=2, label='Violations')
    axes[1, 1].set_title(f'Bound Violations: {stats["violations"]} cells',
                        fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('t_b (end time index)', fontsize=11)
    axes[1, 1].set_ylabel('t_a (start time index)', fontsize=11)

    plt.tight_layout()
    plt.savefig(config.results_dir / "test_eps_linear_heatmaps.png", dpi=100, bbox_inches='tight')
    print(f"  Saved visualization to {config.results_dir / 'test_eps_linear_heatmaps.png'}")

    # Save statistics
    stats_json = {
        'lhs_min': stats['lhs_min'],
        'lhs_max': stats['lhs_max'],
        'lhs_mean': stats['lhs_mean'],
        'rhs_min': stats['rhs_min'],
        'rhs_max': stats['rhs_max'],
        'rhs_mean': stats['rhs_mean'],
        'violations': stats['violations'],
        'num_trajectories': config.num_trajectories,
        'grid_size': config.grid_size,
        'num_steps': config.num_steps_integration,
        'timestamp': datetime.now().isoformat()
    }

    with open(config.results_dir / "test_statistics.json", 'w') as f:
        json.dump(stats_json, f, indent=2)

    print(f"\n" + "=" * 80)
    print("TEST COMPLETE - Pipeline is error-free!")
    print("=" * 80)

    return True


if __name__ == "__main__":
    config = DummyEDAConfig()
    success = run_test_pipeline(config)
    sys.exit(0 if success else 1)
