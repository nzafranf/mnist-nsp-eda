#!/usr/bin/env python
"""
Per-Trajectory EDA Analysis

Analyzes Jacobian and EPS Linear characteristics for individual trajectories.
Uses pre-computed artifacts to avoid recomputation.

Generates 10 visualizations showing diverse straightenability profiles.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json

# Load artifacts
jacobian_results = Path("jacobian_eda/results")
eps_linear_results = Path("eps_linear_eda/results")

print("Loading pre-computed artifacts...")
trajectories = np.load(jacobian_results / "trajectories.npy")  # (50, 100, 784)
spectral_norms = np.load(jacobian_results / "spectral_norms.npy")  # (50, 100)
integrals = np.load(jacobian_results / "integrals.npy")  # (50, 50, 50)
lhs_heatmap = np.load(eps_linear_results / "lhs_heatmap.npy")  # (50, 50)
rhs_heatmap = np.load(eps_linear_results / "rhs_heatmap.npy")  # (50, 50)

N_traj, num_steps, dim = trajectories.shape
time_grid = np.linspace(0, 0.999, num_steps)
grid_size = 50

print(f"Loaded: {N_traj} trajectories, {num_steps} steps each")

# Select 10 diverse trajectories
# Strategy: pick trajectories with different characteristics
selected_indices = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]

results_dir = Path("per_trajectory_eda_results")
results_dir.mkdir(exist_ok=True)

print(f"\nGenerating per-trajectory visualizations for {len(selected_indices)} trajectories...")

for idx, traj_idx in enumerate(selected_indices):
    print(f"\n[{idx+1}/10] Analyzing trajectory {traj_idx}...")

    # Get trajectory-specific metrics
    traj = trajectories[traj_idx]
    spec_norm = spectral_norms[traj_idx]
    traj_integrals = integrals[traj_idx]  # (50, 50)

    # Compute trajectory-specific curvature
    curvatures = []
    for i in range(1, len(traj) - 1):
        x_prev = traj[i - 1]
        x_curr = traj[i]
        x_next = traj[i + 1]
        dt = (time_grid[i + 1] - time_grid[i - 1]) / 2
        if dt > 0:
            x_ddot = (x_next - 2 * x_curr + x_prev) / (dt ** 2)
            curvatures.append(np.linalg.norm(x_ddot))
    curvatures = np.array(curvatures)

    # Create per-trajectory visualization
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    fig.suptitle(f'Per-Trajectory EDA Analysis: Trajectory #{traj_idx}\n(Diverse Straightenability Profile)',
                fontsize=16, fontweight='bold')

    # [1,1] Trajectory visualization (2D projection via PCA-like)
    ax1 = fig.add_subplot(gs[0, 0])
    # Simple 2D projection: first two principal components
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    traj_2d = pca.fit_transform(traj)
    ax1.plot(traj_2d[:, 0], traj_2d[:, 1], 'b-', linewidth=2, alpha=0.7)
    ax1.scatter(traj_2d[0, 0], traj_2d[0, 1], c='green', s=200, marker='o', label='Start (noise)', zorder=5)
    ax1.scatter(traj_2d[-1, 0], traj_2d[-1, 1], c='red', s=200, marker='s', label='End (data)', zorder=5)
    ax1.set_xlabel('PC1', fontsize=10)
    ax1.set_ylabel('PC2', fontsize=10)
    ax1.set_title(f'Trajectory Path (2D Projection)', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # [1,2] Curvature over time
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time_grid[1:-1], curvatures, 'b-', linewidth=2.5, marker='o', markersize=3)
    ax2.fill_between(time_grid[1:-1], curvatures, alpha=0.3, color='lightblue')
    ax2.set_xlabel('Time t', fontsize=10)
    ax2.set_ylabel('Curvature ||ẍ(t)||', fontsize=10)
    ax2.set_title(f'Curvature Profile\n(Mean: {np.mean(curvatures):.3e}, Max: {np.max(curvatures):.3e})',
                 fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # [1,3] Lipschitz constant over time
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(time_grid, spec_norm, 'g-', linewidth=2.5, marker='s', markersize=3)
    ax3.fill_between(time_grid, spec_norm, alpha=0.3, color='lightgreen')
    ax3.set_xlabel('Time t', fontsize=10)
    ax3.set_ylabel('Lipschitz ||∇_x v||', fontsize=10)
    ax3.set_title(f'Lipschitz Profile\n(Mean: {np.mean(spec_norm):.3f}, Max: {np.max(spec_norm):.3f})',
                 fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # [2,1] Jacobian: Straightenability mask
    ax4 = fig.add_subplot(gs[1, 0])
    # Use integrals to compute straightenability for this trajectory
    traj_integral_mean = traj_integrals  # Already aggregated
    # Normalize for visualization
    integral_norm = (traj_integral_mean - traj_integral_mean.min()) / (traj_integral_mean.max() - traj_integral_mean.min() + 1e-8)
    im4 = ax4.imshow(integral_norm, cmap='viridis', aspect='auto', origin='lower')
    ax4.set_xlabel('t_b', fontsize=10)
    ax4.set_ylabel('t_a', fontsize=10)
    ax4.set_title('Jacobian: Geodesic Deviation\n(Lower = More Straightenable)', fontsize=11, fontweight='bold')
    plt.colorbar(im4, ax=ax4, label='Integral')

    # [2,2] EPS Linear: LHS heatmap
    ax5 = fig.add_subplot(gs[1, 1])
    # Extract LHS for this trajectory region
    lhs_display = np.clip(lhs_heatmap, 1e-10, None)
    im5 = ax5.imshow(lhs_display, cmap='viridis', aspect='auto', origin='lower', norm=plt.matplotlib.colors.LogNorm())
    ax5.set_xlabel('t_b', fontsize=10)
    ax5.set_ylabel('t_a', fontsize=10)
    ax5.set_title('EPS Linear: Actual Error (LHS)\n(Log Scale)', fontsize=11, fontweight='bold')
    plt.colorbar(im5, ax=ax5, label='Error')

    # [2,3] EPS Linear: RHS heatmap
    ax6 = fig.add_subplot(gs[1, 2])
    rhs_display = np.clip(rhs_heatmap, 1e-10, None)
    im6 = ax6.imshow(rhs_display, cmap='plasma', aspect='auto', origin='lower', norm=plt.matplotlib.colors.LogNorm())
    ax6.set_xlabel('t_b', fontsize=10)
    ax6.set_ylabel('t_a', fontsize=10)
    ax6.set_title('EPS Linear: Theoretical Bound (RHS)\n(Log Scale)', fontsize=11, fontweight='bold')
    plt.colorbar(im6, ax=ax6, label='Bound')

    # [3,1] Distribution of curvature
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.hist(curvatures, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax7.axvline(np.mean(curvatures), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(curvatures):.3e}')
    ax7.axvline(np.median(curvatures), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(curvatures):.3e}')
    ax7.set_xlabel('Curvature', fontsize=10)
    ax7.set_ylabel('Frequency', fontsize=10)
    ax7.set_title('Curvature Distribution', fontsize=11, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3, axis='y')

    # [3,2] Distribution of Lipschitz
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.hist(spec_norm, bins=20, color='seagreen', alpha=0.7, edgecolor='black')
    ax8.axvline(np.mean(spec_norm), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(spec_norm):.3f}')
    ax8.axvline(np.median(spec_norm), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(spec_norm):.3f}')
    ax8.set_xlabel('Lipschitz Constant', fontsize=10)
    ax8.set_ylabel('Frequency', fontsize=10)
    ax8.set_title('Lipschitz Distribution', fontsize=11, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3, axis='y')

    # [3,3] Summary statistics
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    stats_text = f"""
TRAJECTORY SUMMARY

Trajectory ID: {traj_idx}

Curvature Stats:
  Mean: {np.mean(curvatures):.3e}
  Std:  {np.std(curvatures):.3e}
  Min:  {np.min(curvatures):.3e}
  Max:  {np.max(curvatures):.3e}
  Range: [{np.min(curvatures):.3e}, {np.max(curvatures):.3e}]

Lipschitz Stats:
  Mean: {np.mean(spec_norm):.3f}
  Std:  {np.std(spec_norm):.3f}
  Min:  {np.min(spec_norm):.3f}
  Max:  {np.max(spec_norm):.3f}
  Range: [{np.min(spec_norm):.3f}, {np.max(spec_norm):.3f}]

Jacobian Characteristics:
  Integral Range: [{traj_integral_mean.min():.3f}, {traj_integral_mean.max():.3f}]
  Straightenable: {(traj_integral_mean < 0.2).sum()} / {traj_integral_mean.size} cells

Error Bounds:
  LHS Range: [{lhs_heatmap.min():.3e}, {lhs_heatmap.max():.3e}]
  RHS Range: [{rhs_heatmap.min():.3e}, {rhs_heatmap.max():.3e}]
  Mean Tightness: {(rhs_heatmap / np.maximum(lhs_heatmap, 1e-10)).mean():.1f}x
"""

    ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Save figure
    save_path = results_dir / f"trajectory_{traj_idx:02d}_eda.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close(fig)

print(f"\n[SUCCESS] Generated 10 per-trajectory visualizations in {results_dir}/")
print(f"\nVisualization files:")
for i, traj_idx in enumerate(selected_indices, 1):
    print(f"  {i}. trajectory_{traj_idx:02d}_eda.png")

# Create a summary JSON
summary = {
    "analysis_type": "Per-Trajectory EDA",
    "num_trajectories_analyzed": len(selected_indices),
    "selected_trajectory_indices": selected_indices,
    "timestamp": str(np.datetime64('today')),
    "source_artifacts": {
        "trajectories": "jacobian_eda/results/trajectories.npy",
        "spectral_norms": "jacobian_eda/results/spectral_norms.npy",
        "integrals": "jacobian_eda/results/integrals.npy",
        "lhs_heatmap": "eps_linear_eda/results/lhs_heatmap.npy",
        "rhs_heatmap": "eps_linear_eda/results/rhs_heatmap.npy"
    },
    "purpose": "Show diverse straightenability characteristics for NSP training"
}

with open(results_dir / "per_trajectory_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved summary to: per_trajectory_eda_results/per_trajectory_summary.json")
