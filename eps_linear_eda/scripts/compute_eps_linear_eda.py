#!/usr/bin/env python
"""
EPS Linear EDA on Flow Matching MNIST Model

Validates Section III (Step Compression) and Theorem 3.1 by comparing:
- Actual Error (LHS): ||x̂^N(t_b) - x̂^1(t_b)||
- Theoretical Bound (RHS): (ε H² / 2) * (e^(LH) - 1) / (LH) * (1 + 1/N)

Uses pre-computed trajectories from Jacobian EDA to ensure consistency.

References:
- Theorem 3.1 (Step Compression with Error Bound)
- Section III (Step Compression) from "Adaptive and Neural Schedule Predictor" paper
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
from matplotlib.colors import LogNorm
from tqdm import tqdm

# Add paths
project_root = Path(__file__).resolve().parent.parent.parent  # flow-matching-mnist
eps_linear_eda_root = Path(__file__).resolve().parent.parent  # eps_linear_eda

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(eps_linear_eda_root))

# Import utilities
import importlib.util
utils_path = eps_linear_eda_root / 'utils' / 'eps_linear_utils.py'
spec = importlib.util.spec_from_file_location("eps_linear_utils", utils_path)
eps_linear_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eps_linear_utils)

compute_eps_linear_grid = eps_linear_utils.compute_eps_linear_grid
identify_violations = eps_linear_utils.identify_violations


class EPSLinearEDAConfig:
    """Configuration for EPS Linear EDA."""

    def __init__(self):
        # Data source: use saved trajectories from Jacobian EDA
        self.jacobian_eda_results = Path("jacobian_eda/results")

        # Analysis parameters
        self.grid_size = 50
        self.n_substeps = 10  # N for N-step Euler comparison
        self.log_scale_heatmaps = True

        # Output
        self.results_dir = Path("eps_linear_eda/results")


def load_jacobian_eda_artifacts(config: EPSLinearEDAConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pre-computed trajectories and spectral norms from Jacobian EDA.

    Returns:
        trajectories: [N, num_steps, 784]
        time_grid: [num_steps] - reconstructed to match trajectory shape
        spectral_norms: [N, num_steps]
    """
    print(f"Loading Jacobian EDA artifacts from {config.jacobian_eda_results}...")

    if not config.jacobian_eda_results.exists():
        raise FileNotFoundError(f"Jacobian EDA results not found at {config.jacobian_eda_results}")

    # Load trajectories
    trajectories_path = config.jacobian_eda_results / "trajectories.npy"
    if not trajectories_path.exists():
        raise FileNotFoundError(f"trajectories.npy not found at {trajectories_path}")

    trajectories = np.load(trajectories_path)
    print(f"  Loaded trajectories: {trajectories.shape}")

    # Load time grid (may be downsampled, so reconstruct it)
    t_grid_path = config.jacobian_eda_results / "t_grid.npy"
    if t_grid_path.exists():
        time_grid_loaded = np.load(t_grid_path)
        print(f"  Loaded time grid: {time_grid_loaded.shape}")
    else:
        time_grid_loaded = None

    # Reconstruct time grid to match trajectory shape
    num_steps = trajectories.shape[1]
    time_grid = np.linspace(0, 0.999, num_steps)
    print(f"  Reconstructed time grid: {time_grid.shape} (to match trajectory {num_steps} steps)")

    # Load spectral norms (Lipschitz estimates)
    spectral_norms_path = config.jacobian_eda_results / "spectral_norms.npy"
    if not spectral_norms_path.exists():
        raise FileNotFoundError(f"spectral_norms.npy not found at {spectral_norms_path}")

    spectral_norms = np.load(spectral_norms_path)
    print(f"  Loaded spectral norms: {spectral_norms.shape}")

    # Verify shapes match
    if spectral_norms.shape[1] != num_steps:
        raise ValueError(f"Spectral norms shape {spectral_norms.shape} doesn't match trajectory steps {num_steps}")

    return trajectories, time_grid, spectral_norms


def run_eps_linear_analysis(config: EPSLinearEDAConfig):
    """Run complete EPS Linear EDA analysis."""
    print("=" * 80)
    print("EPS LINEAR EDA - THEOREM 3.1 VALIDATION")
    print("=" * 80)

    # [1/3] Load Jacobian EDA artifacts
    print("\n[1/3] Loading Jacobian EDA artifacts...")
    try:
        trajectories, time_grid, spectral_norms = load_jacobian_eda_artifacts(config)
    except Exception as e:
        print(f"ERROR: Failed to load artifacts: {e}")
        return False

    # [2/3] Compute EPS Linear grid
    print(f"\n[2/3] Computing EPS Linear grid ({config.grid_size}x{config.grid_size})...")
    print(f"  Using {config.n_substeps}-step Euler for N-step comparison")

    try:
        lhs_heatmap, rhs_heatmap, tightness_ratio, stats = compute_eps_linear_grid(
            trajectories, time_grid, spectral_norms,
            grid_size=config.grid_size,
            n_substeps=config.n_substeps,
            velocity_fn=None
        )
        print(f"  [OK] LHS (Actual Error) computed")
        print(f"    Range: [{stats['lhs_min']:.2e}, {stats['lhs_max']:.2e}]")
        print(f"    Mean:  {stats['lhs_mean']:.2e}")

        print(f"  [OK] RHS (Theoretical Bound) computed")
        print(f"    Range: [{stats['rhs_min']:.2e}, {stats['rhs_max']:.2e}]")
        print(f"    Mean:  {stats['rhs_mean']:.2e}")

        print(f"  [OK] Tightness ratio computed")
        print(f"    Violations (LHS > RHS): {stats['violations']}/{config.grid_size ** 2}")

    except Exception as e:
        print(f"ERROR: Failed to compute grid: {e}")
        import traceback
        traceback.print_exc()
        return False

    # [3/3] Create supporting 1D analysis graphs
    print(f"\n[3/3] Creating supporting analysis graphs...")
    try:
        config.results_dir.mkdir(parents=True, exist_ok=True)

        # Supporting graph 1: Curvature and Lipschitz vs time
        eps_by_time = stats['eps_grid'].mean(axis=1)  # Average over all ending times
        L_by_time = stats['L_grid'].mean(axis=1)  # Average Lipschitz over all ending times
        time_points = np.linspace(0, 0.999, config.grid_size)

        fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
        fig1.suptitle('Supporting Analysis: Curvature, Lipschitz, and RHS Bound Trends',
                     fontsize=14, fontweight='bold')

        # Graph 1: Curvature vs time
        ax1.plot(time_points, eps_by_time, linewidth=2.5, color='darkblue', marker='o', markersize=5)
        ax1.fill_between(time_points, eps_by_time, alpha=0.3, color='lightblue')
        ax1.set_xlabel('Time t (start of interval)', fontsize=12)
        ax1.set_ylabel('Average Curvature ε(t)', fontsize=12)
        ax1.set_title('Curvature: How ε Grows Over Time', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])

        # Graph 2: Lipschitz constant vs time
        ax2.plot(time_points, L_by_time, linewidth=2.5, color='darkgreen', marker='s', markersize=5)
        ax2.fill_between(time_points, L_by_time, alpha=0.3, color='lightgreen')
        ax2.set_xlabel('Time t (start of interval)', fontsize=12)
        ax2.set_ylabel('Average Lipschitz Constant L(t)', fontsize=12)
        ax2.set_title('Lipschitz: How L Varies Over Time', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 1])

        # Graph 3: RHS bound vs window size H
        # Compute average RHS bound for different window sizes
        window_sizes = []
        avg_bounds = []

        for i in range(config.grid_size):
            for j in range(i + 1, config.grid_size):
                H = time_points[j] - time_points[i]
                rhs_val = stats['rhs_heatmap'][i, j] if 'rhs_heatmap' in stats else rhs_heatmap[i, j]

                if rhs_val > 0:
                    window_sizes.append(H)
                    avg_bounds.append(rhs_val)

        if window_sizes:
            # Sort by window size for cleaner plot
            sorted_idx = np.argsort(window_sizes)
            window_sizes_sorted = np.array(window_sizes)[sorted_idx]
            avg_bounds_sorted = np.array(avg_bounds)[sorted_idx]

            # Bin by window size and compute averages
            n_bins = 20
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_means = []

            for k in range(len(bin_edges) - 1):
                mask = (window_sizes_sorted >= bin_edges[k]) & (window_sizes_sorted < bin_edges[k + 1])
                if np.any(mask):
                    bin_means.append(np.mean(np.array(avg_bounds_sorted)[mask]))
                else:
                    bin_means.append(np.nan)

            valid_bins = ~np.isnan(bin_means)
            ax3.semilogy(bin_centers[valid_bins], np.array(bin_means)[valid_bins], linewidth=2.5,
                        color='darkred', marker='^', markersize=6, label='RHS Bound')
            ax3.set_xlabel('Window Size H = t_b - t_a', fontsize=12)
            ax3.set_ylabel('Average RHS Bound (log scale)', fontsize=12)
            ax3.set_title('RHS Bound: How It Grows with Window Size', fontsize=13, fontweight='bold')
            ax3.grid(True, alpha=0.3, which='both')
            ax3.set_xlim([0, 1])

        plt.tight_layout()
        support_path = config.results_dir / "eps_linear_supporting_analysis.png"
        plt.savefig(support_path, dpi=150, bbox_inches='tight')
        print(f"  [OK] Saved supporting analysis to {support_path}")
        plt.close(fig1)

    except Exception as e:
        print(f"ERROR: Failed to create supporting graphs: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Add rhs_heatmap to stats for reference
    stats['rhs_heatmap'] = rhs_heatmap

    # [4/3] Create main triple heatmap visualization
    print(f"\n[4/3] Creating main triple heatmap visualization...")
    try:
        config.results_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle('EPS Linear EDA: Theorem 3.1 Validation\nLHS vs RHS Bound',
                    fontsize=16, fontweight='bold')

        # Create time labels for axes
        time_labels = np.linspace(0, 1, config.grid_size)

        # [Panel 1] LHS Heatmap (log scale)
        if config.log_scale_heatmaps:
            lhs_display = np.clip(lhs_heatmap, 1e-10, None)
            im0 = axes[0, 0].imshow(lhs_display, cmap='viridis', aspect='auto', origin='lower',
                                   norm=LogNorm())
            axes[0, 0].set_title(
                'LHS: Actual Error ||x̂^N - x̂^1||' + '\n(Log Scale)',
                fontsize=13, fontweight='bold'
            )
        else:
            im0 = axes[0, 0].imshow(lhs_heatmap, cmap='viridis', aspect='auto', origin='lower')
            axes[0, 0].set_title(
                'LHS: Actual Error ||x̂^N - x̂^1||',
                fontsize=13, fontweight='bold'
            )

        axes[0, 0].set_xlabel('t_b (end time)', fontsize=12)
        axes[0, 0].set_ylabel('t_a (start time)', fontsize=12)
        cbar0 = plt.colorbar(im0, ax=axes[0, 0])
        cbar0.set_label('Error magnitude', fontsize=11)

        # [Panel 2] RHS Heatmap (log scale)
        if config.log_scale_heatmaps:
            rhs_display = np.clip(rhs_heatmap, 1e-10, None)
            im1 = axes[0, 1].imshow(rhs_display, cmap='plasma', aspect='auto', origin='lower',
                                   norm=LogNorm())
            axes[0, 1].set_title(
                'RHS: Theoretical Bound' + '\n(Log Scale)',
                fontsize=13, fontweight='bold'
            )
        else:
            im1 = axes[0, 1].imshow(rhs_heatmap, cmap='plasma', aspect='auto', origin='lower')
            axes[0, 1].set_title(
                'RHS: Theoretical Bound',
                fontsize=13, fontweight='bold'
            )

        axes[0, 1].set_xlabel('t_b (end time)', fontsize=12)
        axes[0, 1].set_ylabel('t_a (start time)', fontsize=12)
        cbar1 = plt.colorbar(im1, ax=axes[0, 1])
        cbar1.set_label('Bound magnitude', fontsize=11)

        # [Panel 3] Tightness Ratio (RHS / LHS)
        tightness_display = np.clip(tightness_ratio, 0.1, 1000)
        im2 = axes[1, 0].imshow(tightness_display, cmap='RdYlGn_r', aspect='auto', origin='lower',
                               norm=LogNorm(vmin=0.1, vmax=1000))
        axes[1, 0].set_title(
            'Tightness: RHS / LHS' + '\n(Green=Tight, Red=Conservative)',
            fontsize=13, fontweight='bold'
        )
        axes[1, 0].set_xlabel('t_b (end time)', fontsize=12)
        axes[1, 0].set_ylabel('t_a (start time)', fontsize=12)
        cbar2 = plt.colorbar(im2, ax=axes[1, 0])
        cbar2.set_label('Ratio (log scale)', fontsize=11)

        # [Panel 4] Violations and Statistics
        violation_mask = identify_violations(lhs_heatmap, rhs_heatmap, time_grid, config.grid_size)

        axes[1, 1].imshow(np.ones_like(lhs_heatmap), cmap='Greys', aspect='auto', origin='lower',
                         vmin=0, vmax=1, alpha=0.3)

        if np.any(violation_mask):
            violations_y, violations_x = np.where(violation_mask)
            axes[1, 1].scatter(violations_x, violations_y, c='red', s=150, marker='X',
                             edgecolors='darkred', linewidth=2.5, label='Violations',
                             zorder=5)

        status_text = f'Violations: {stats["violations"]} cells (out of {config.grid_size ** 2})'
        status_text += '\n[PASS]' if stats['violations'] == 0 else '\n[FAIL]'
        axes[1, 1].set_title(
            status_text,
            fontsize=13, fontweight='bold',
            color='green' if stats['violations'] == 0 else 'red'
        )
        axes[1, 1].set_xlabel('t_b (end time)', fontsize=12)
        axes[1, 1].set_ylabel('t_a (start time)', fontsize=12)

        if np.any(violation_mask):
            axes[1, 1].legend(fontsize=11, loc='upper right')

        plt.tight_layout()
        viz_path = config.results_dir / "eps_linear_eda_analysis.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"  [OK] Saved visualization to {viz_path}")

        plt.close(fig)

    except Exception as e:
        print(f"ERROR: Failed to create visualizations: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Save artifacts
    print(f"\n[4/3] Saving artifacts...")
    try:
        config.results_dir.mkdir(parents=True, exist_ok=True)

        # Save heatmaps
        np.save(config.results_dir / "lhs_heatmap.npy", lhs_heatmap)
        np.save(config.results_dir / "rhs_heatmap.npy", rhs_heatmap)
        np.save(config.results_dir / "tightness_ratio.npy", tightness_ratio)
        np.save(config.results_dir / "violation_mask.npy", violation_mask)
        np.save(config.results_dir / "eps_grid.npy", stats['eps_grid'])
        np.save(config.results_dir / "L_grid.npy", stats['L_grid'])

        print(f"  [OK] Saved heatmaps and grids")

        # Save statistics
        stats_json = {
            'lhs_min': stats['lhs_min'],
            'lhs_max': stats['lhs_max'],
            'lhs_mean': stats['lhs_mean'],
            'rhs_min': stats['rhs_min'],
            'rhs_max': stats['rhs_max'],
            'rhs_mean': stats['rhs_mean'],
            'violations': stats['violations'],
            'violations_percentage': 100.0 * stats['violations'] / (config.grid_size ** 2),
            'num_trajectories': trajectories.shape[0],
            'num_steps': trajectories.shape[1],
            'grid_size': config.grid_size,
            'n_substeps': config.n_substeps,
            'timestamp': datetime.now().isoformat(),
            'theorem_status': 'PASS' if stats['violations'] == 0 else 'FAIL'
        }

        with open(config.results_dir / "statistics.json", 'w') as f:
            json.dump(stats_json, f, indent=2)

        print(f"  [OK] Saved statistics")

    except Exception as e:
        print(f"ERROR: Failed to save artifacts: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Print summary
    print(f"\n" + "=" * 80)
    print(f"ANALYSIS COMPLETE")
    print(f"=" * 80)
    print(f"\nTheorem 3.1 Validation Summary:")
    print(f"  Actual errors (LHS):       [{stats['lhs_min']:.2e}, {stats['lhs_max']:.2e}]")
    print(f"  Theoretical bounds (RHS):  [{stats['rhs_min']:.2e}, {stats['rhs_max']:.2e}]")
    print(f"  Bound violations:          {stats['violations']} / {config.grid_size ** 2}")
    print(f"  Status:                    {'PASS' if stats['violations'] == 0 else 'FAIL'}")
    print(f"\nResults saved to: {config.results_dir}")
    print(f"=" * 80)

    return True


if __name__ == "__main__":
    config = EPSLinearEDAConfig()
    success = run_eps_linear_analysis(config)
    sys.exit(0 if success else 1)
