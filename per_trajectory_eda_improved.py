#!/usr/bin/env python
"""
Per-Trajectory EDA Analysis (Improved Version)

Enhanced visualizations with:
1. PCA explainability metrics and unified axis ranges
2. KDE contours for epsilon thresholds on Jacobian heatmap
3. X markers for LHS > RHS violations (theorem failures)
4. Log scale for curvature and Lipschitz profiles
5. RHS/LHS ratio plot showing bound tightness

Uses pre-computed artifacts to avoid recomputation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA

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
selected_indices = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]

results_dir = Path("per_trajectory_eda_results_improved")
results_dir.mkdir(exist_ok=True)

# STEP 1: Compute global PCA bounds for unified axis ranges
print("\nComputing global PCA bounds from all 10 trajectories...")
global_pca = PCA(n_components=2)
all_2d_projections = []

for traj_idx in selected_indices:
    traj = trajectories[traj_idx]
    traj_2d = global_pca.fit_transform(traj)
    all_2d_projections.append(traj_2d)

# Fit on all data to get consistent basis
all_traj_data = np.vstack([trajectories[i] for i in selected_indices])
global_pca = PCA(n_components=2)
global_pca.fit(all_traj_data)
explained_var = global_pca.explained_variance_ratio_

# Compute global bounds
all_points = np.vstack([global_pca.transform(trajectories[i]) for i in selected_indices])
x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()

# Add padding
pad = 0.1 * max(x_max - x_min, y_max - y_min)
x_lim = [x_min - pad, x_max + pad]
y_lim = [y_min - pad, y_max + pad]

print(f"Global PCA bounds: X=[{x_lim[0]:.2f}, {x_lim[1]:.2f}], Y=[{y_lim[0]:.2f}, {y_lim[1]:.2f}]")
print(f"Explained variance: PC1={explained_var[0]:.1%}, PC2={explained_var[1]:.1%}")

# Epsilon thresholds for KDE contours
epsilon_thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]

print(f"\nGenerating improved per-trajectory visualizations for {len(selected_indices)} trajectories...")

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
    curvatures_clipped = np.clip(curvatures, 1e-10, None)

    # Clip spectral norms for log scale
    spec_norm_clipped = np.clip(spec_norm, 1e-10, None)

    # Compute RHS/LHS ratio (bound tightness)
    lhs_safe = np.clip(lhs_heatmap, 1e-10, None)
    rhs_ratio = np.clip(rhs_heatmap / lhs_safe, 1e-10, 1e6)

    # Identify violations (LHS > RHS)
    violations = lhs_heatmap > rhs_heatmap

    # Create improved per-trajectory visualization with 4 rows
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.35)

    fig.suptitle(
        f'Per-Trajectory EDA Analysis (Improved): Trajectory #{traj_idx}\n(PC1: {explained_var[0]:.1%}, PC2: {explained_var[1]:.1%})',
        fontsize=16, fontweight='bold'
    )

    # ===== ROW 1: Trajectory & Basic Properties =====

    # [1,1] Trajectory visualization (2D projection with unified axes)
    ax1 = fig.add_subplot(gs[0, 0])
    traj_2d = global_pca.transform(traj)
    ax1.plot(traj_2d[:, 0], traj_2d[:, 1], 'b-', linewidth=2.5, alpha=0.8)
    ax1.scatter(traj_2d[0, 0], traj_2d[0, 1], c='green', s=250, marker='o', label='Start (noise)', zorder=5, edgecolors='darkgreen', linewidths=1.5)
    ax1.scatter(traj_2d[-1, 0], traj_2d[-1, 1], c='red', s=250, marker='s', label='End (data)', zorder=5, edgecolors='darkred', linewidths=1.5)
    ax1.set_xlim(x_lim)
    ax1.set_ylim(y_lim)
    ax1.set_xlabel(f'PC1 ({explained_var[0]:.1%})', fontsize=11, fontweight='bold')
    ax1.set_ylabel(f'PC2 ({explained_var[1]:.1%})', fontsize=11, fontweight='bold')
    ax1.set_title(f'Trajectory Path (2D Projection)\nUnified Axes for All 10 Trajectories', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)

    # [1,2] Curvature over time (LOG SCALE)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(time_grid[1:-1], curvatures_clipped, 'b-', linewidth=2.5, marker='o', markersize=4, label='Curvature')
    ax2.fill_between(time_grid[1:-1], curvatures_clipped, alpha=0.2, color='lightblue')
    ax2.set_xlabel('Time t', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Curvature ||ẍ(t)|| (log)', fontsize=11, fontweight='bold')
    ax2.set_title(f'Curvature Profile (LOG SCALE)\nMean: {np.mean(curvatures):.3e}, Max: {np.max(curvatures):.3e}',
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')

    # [1,3] Lipschitz constant over time (LOG SCALE)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.semilogy(time_grid, spec_norm_clipped, 'g-', linewidth=2.5, marker='s', markersize=4, label='Lipschitz')
    ax3.fill_between(time_grid, spec_norm_clipped, alpha=0.2, color='lightgreen')
    ax3.set_xlabel('Time t', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Lipschitz ||∇_x v|| (log)', fontsize=11, fontweight='bold')
    ax3.set_title(f'Lipschitz Profile (LOG SCALE)\nMean: {np.mean(spec_norm):.3f}, Max: {np.max(spec_norm):.3f}',
                 fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, which='both')

    # ===== ROW 2: Jacobian & Error Analysis =====

    # [2,1] Jacobian: Geodesic Deviation with KDE contours
    ax4 = fig.add_subplot(gs[1, 0])
    traj_integral_mean = traj_integrals
    # Normalize for visualization
    integral_norm = (traj_integral_mean - traj_integral_mean.min()) / (traj_integral_mean.max() - traj_integral_mean.min() + 1e-8)
    im4 = ax4.imshow(integral_norm, cmap='viridis', aspect='auto', origin='lower', alpha=0.85)

    # Add KDE contours for epsilon thresholds
    kde_smooth = gaussian_filter(traj_integral_mean, sigma=1.5)
    contours = ax4.contour(kde_smooth, levels=epsilon_thresholds, colors='white', linewidths=1.5, alpha=0.9)
    ax4.clabel(contours, inline=True, fontsize=8, fmt='%.2f')

    ax4.set_xlabel('t_b', fontsize=11, fontweight='bold')
    ax4.set_ylabel('t_a', fontsize=11, fontweight='bold')
    ax4.set_title('Jacobian: Geodesic Deviation + KDE Contours\n(Lower = More Straightenable)', fontsize=12, fontweight='bold')
    cbar4 = plt.colorbar(im4, ax=ax4, label='Integral')

    # [2,2] EPS Linear: LHS heatmap with violation markers
    ax5 = fig.add_subplot(gs[1, 1])
    lhs_display = np.clip(lhs_heatmap, 1e-10, None)
    im5 = ax5.imshow(lhs_display, cmap='viridis', aspect='auto', origin='lower', norm=plt.matplotlib.colors.LogNorm())

    # Mark violations with red X symbols
    violation_indices = np.where(violations)
    ax5.scatter(violation_indices[1], violation_indices[0], marker='x', c='red', s=50, linewidths=2, alpha=0.8, label=f'Violations: {len(violation_indices[0])}')

    ax5.set_xlabel('t_b', fontsize=11, fontweight='bold')
    ax5.set_ylabel('t_a', fontsize=11, fontweight='bold')
    ax5.set_title(f'EPS Linear: Actual Error (LHS)\nRed X = LHS > RHS Violations (Log Scale)', fontsize=12, fontweight='bold')
    cbar5 = plt.colorbar(im5, ax=ax5, label='Error')
    if len(violation_indices[0]) > 0:
        ax5.legend(fontsize=9, loc='upper left')

    # [2,3] EPS Linear: RHS heatmap
    ax6 = fig.add_subplot(gs[1, 2])
    rhs_display = np.clip(rhs_heatmap, 1e-10, None)
    im6 = ax6.imshow(rhs_display, cmap='plasma', aspect='auto', origin='lower', norm=plt.matplotlib.colors.LogNorm())
    ax6.set_xlabel('t_b', fontsize=11, fontweight='bold')
    ax6.set_ylabel('t_a', fontsize=11, fontweight='bold')
    ax6.set_title('EPS Linear: Theoretical Bound (RHS)\n(Log Scale)', fontsize=12, fontweight='bold')
    cbar6 = plt.colorbar(im6, ax=ax6, label='Bound')

    # ===== ROW 3: RHS/LHS Ratio & Distributions =====

    # [3,1] RHS/LHS ratio (tightness of bounds)
    ax7 = fig.add_subplot(gs[2, 0])
    ratio_display = np.clip(rhs_ratio, 1e-1, 1e3)
    im7 = ax7.imshow(ratio_display, cmap='RdYlGn_r', aspect='auto', origin='lower', norm=plt.matplotlib.colors.LogNorm(vmin=1, vmax=1e3))
    ax7.set_xlabel('t_b', fontsize=11, fontweight='bold')
    ax7.set_ylabel('t_a', fontsize=11, fontweight='bold')
    ax7.set_title('Bound Tightness: RHS/LHS Ratio\n(Lower = Tighter Bounds, Green = Tight)', fontsize=12, fontweight='bold')
    cbar7 = plt.colorbar(im7, ax=ax7, label='Ratio')

    # [3,2] Distribution of curvature
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.hist(curvatures, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax8.axvline(np.mean(curvatures), color='red', linestyle='--', linewidth=2.5, label=f'Mean: {np.mean(curvatures):.3e}')
    ax8.axvline(np.median(curvatures), color='green', linestyle='--', linewidth=2.5, label=f'Median: {np.median(curvatures):.3e}')
    ax8.set_xlabel('Curvature', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax8.set_title('Curvature Distribution', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3, axis='y')

    # [3,3] Distribution of Lipschitz
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.hist(spec_norm, bins=20, color='seagreen', alpha=0.7, edgecolor='black')
    ax9.axvline(np.mean(spec_norm), color='red', linestyle='--', linewidth=2.5, label=f'Mean: {np.mean(spec_norm):.3f}')
    ax9.axvline(np.median(spec_norm), color='orange', linestyle='--', linewidth=2.5, label=f'Median: {np.median(spec_norm):.3f}')
    ax9.set_xlabel('Lipschitz Constant', fontsize=11, fontweight='bold')
    ax9.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax9.set_title('Lipschitz Distribution', fontsize=12, fontweight='bold')
    ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.3, axis='y')

    # ===== ROW 4: Summary Statistics =====

    # [4,1-3] Summary statistics (spanning all 3 columns)
    ax10 = fig.add_subplot(gs[3, :])
    ax10.axis('off')

    stats_text = f"""
TRAJECTORY SUMMARY

Trajectory ID: {traj_idx}

[PCA EXPLAINABILITY]
  PC1 Variance: {explained_var[0]:.1%} | PC2 Variance: {explained_var[1]:.1%}
  Axis Ranges (Unified): X=[{x_lim[0]:.2f}, {x_lim[1]:.2f}], Y=[{y_lim[0]:.2f}, {y_lim[1]:.2f}]

[CURVATURE STATS]
  Mean: {np.mean(curvatures):.3e} | Median: {np.median(curvatures):.3e} | Std: {np.std(curvatures):.3e}
  Min: {np.min(curvatures):.3e} | Max: {np.max(curvatures):.3e}

[LIPSCHITZ STATS]
  Mean: {np.mean(spec_norm):.3f} | Median: {np.median(spec_norm):.3f} | Std: {np.std(spec_norm):.3f}
  Min: {np.min(spec_norm):.3f} | Max: {np.max(spec_norm):.3f}

[JACOBIAN CHARACTERISTICS]
  Integral Range: [{traj_integral_mean.min():.3f}, {traj_integral_mean.max():.3f}]
  Straightenable (eps<0.2): {(traj_integral_mean < 0.2).sum()} / {traj_integral_mean.size} cells ({100*((traj_integral_mean < 0.2).sum()/traj_integral_mean.size):.1f}%)
  KDE Contour Levels: {epsilon_thresholds}

[ERROR BOUNDS & VIOLATIONS]
  LHS Range: [{lhs_heatmap.min():.3e}, {lhs_heatmap.max():.3e}]
  RHS Range: [{rhs_heatmap.min():.3e}, {rhs_heatmap.max():.3e}]
  Violations (LHS > RHS): {len(violation_indices[0])} / {violations.size} cells ({100*(len(violation_indices[0])/violations.size):.2f}%)
  Mean RHS/LHS Ratio (Tightness): {rhs_ratio[rhs_ratio > 0].mean():.2f}x | Median: {np.median(rhs_ratio[rhs_ratio > 0]):.2f}x
"""

    ax10.text(0.05, 0.95, stats_text, transform=ax10.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85, pad=1))

    # Save figure
    save_path = results_dir / f"trajectory_{traj_idx:02d}_eda_improved.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close(fig)

print(f"\n[SUCCESS] Generated 10 improved per-trajectory visualizations in {results_dir}/")
print(f"\nVisualization files:")
for i, traj_idx in enumerate(selected_indices, 1):
    print(f"  {i}. trajectory_{traj_idx:02d}_eda_improved.png")

# Create an enhanced summary JSON
summary = {
    "analysis_type": "Per-Trajectory EDA (Improved)",
    "num_trajectories_analyzed": len(selected_indices),
    "selected_trajectory_indices": selected_indices,
    "timestamp": str(np.datetime64('today')),
    "enhancements": [
        "PCA explainability metrics (variance explained %)",
        "Unified axis ranges across all 10 trajectories for direct comparison",
        "KDE contours for epsilon thresholds on Jacobian heatmap",
        "Violation markers (red X) for LHS > RHS failures",
        "Log scale for curvature and Lipschitz profiles",
        "RHS/LHS ratio plot showing bound tightness"
    ],
    "pca_explainability": {
        "pc1_variance_explained": f"{explained_var[0]:.1%}",
        "pc2_variance_explained": f"{explained_var[1]:.1%}",
        "unified_x_limits": [float(x_lim[0]), float(x_lim[1])],
        "unified_y_limits": [float(y_lim[0]), float(y_lim[1])]
    },
    "kde_contour_levels": epsilon_thresholds,
    "source_artifacts": {
        "trajectories": "jacobian_eda/results/trajectories.npy",
        "spectral_norms": "jacobian_eda/results/spectral_norms.npy",
        "integrals": "jacobian_eda/results/integrals.npy",
        "lhs_heatmap": "eps_linear_eda/results/lhs_heatmap.npy",
        "rhs_heatmap": "eps_linear_eda/results/rhs_heatmap.npy"
    },
    "purpose": "Show diverse straightenability characteristics for NSP training with enhanced interpretability"
}

with open(results_dir / "per_trajectory_summary_improved.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved enhanced summary to: per_trajectory_eda_results_improved/per_trajectory_summary_improved.json")
print(f"\n[OK] All improvements applied:")
print(f"  1. PCA explainability: {explained_var[0]:.1%} + {explained_var[1]:.1%}")
print(f"  2. Unified axes: X=[{x_lim[0]:.2f}, {x_lim[1]:.2f}], Y=[{y_lim[0]:.2f}, {y_lim[1]:.2f}]")
print(f"  3. KDE contours: {epsilon_thresholds}")
print(f"  4. Log scales applied to curvature and Lipschitz profiles")
print(f"  5. RHS/LHS ratio plot added")
print(f"  6. Violation markers on LHS heatmap")
