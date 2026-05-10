#!/usr/bin/env python
"""
Trajectory & Curvature Correlation Analysis

Validates that adaptive step size H(t) is inversely proportional to curvature κ₂(t).
Demonstrates coupling between theory (Section III) and algorithm (Section V).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from adaptive_solver import AdaptiveFlowSolver, SolverConfig, velocity_from_trajectory_finite_diff

def compute_curvature_from_trajectory(
    trajectory: np.ndarray,
    t_grid: np.ndarray
) -> np.ndarray:
    """
    Compute curvature ||ẍ(t)|| from trajectory via finite differences.

    Returns:
        curvatures: [N_steps-2] array of curvature values
    """
    dt = np.diff(t_grid)
    velocity = np.diff(trajectory, axis=0) / dt[:, None]
    acceleration = np.diff(velocity, axis=0) / dt[:-1, None]
    curvatures = np.linalg.norm(acceleration, axis=1)

    return curvatures

def analyze_trajectory_curvature_correlation(
    trajectories: np.ndarray,
    t_grid: np.ndarray,
    traj_idx: int,
    tolerance: float = 1e-3,
):
    """
    Run adaptive solver and correlate H(t) with κ₂(t).

    Returns:
        Dictionary with H(t), κ₂(t), correlation analysis
    """
    x0 = trajectories[traj_idx, 0].copy()
    t_start, t_end = t_grid[0], t_grid[-1]

    # Get velocity function
    velocity_fn = velocity_from_trajectory_finite_diff(trajectories, t_grid, traj_idx)

    # Run adaptive solver
    config = SolverConfig(tolerance=tolerance, alpha=0.9, f_min=0.1, f_max=5.0)
    solver = AdaptiveFlowSolver(config)
    adaptive_result = solver.solve(x0, (t_start, t_end), velocity_fn, H0=0.01)

    # Compute curvature of reference trajectory
    traj_full = trajectories[traj_idx]
    curvature_full = compute_curvature_from_trajectory(traj_full, t_grid)
    t_curvature = t_grid[1:-1]  # Valid time points for curvature

    # For each adaptive step, get its time and step size
    adaptive_times = adaptive_result['times'][:-1]  # Times at which steps are taken
    adaptive_H = adaptive_result['step_sizes']

    # Interpolate curvature to adaptive time points
    curvature_at_adaptive = np.interp(adaptive_times, t_curvature, curvature_full, left=curvature_full[0], right=curvature_full[-1])

    # Compute correlation
    if len(adaptive_H) > 1 and len(curvature_at_adaptive) > 1:
        correlation = np.corrcoef(adaptive_H, curvature_at_adaptive)[0, 1]
        # Should be negative (H inversely proportional to κ₂)
    else:
        correlation = np.nan

    return {
        'trajectory_idx': traj_idx,
        'tolerance': tolerance,
        'adaptive_times': adaptive_times,
        'adaptive_step_sizes': adaptive_H,
        'curvature_full': curvature_full,
        't_curvature': t_curvature,
        'curvature_at_adaptive_times': curvature_at_adaptive,
        'correlation': correlation,
        'n_steps': len(adaptive_H),
        'mean_step_size': np.mean(adaptive_H),
        'mean_curvature': np.mean(curvature_full),
    }

def plot_trajectory_analysis(analysis: dict, save_path: Path):
    """
    Plot dual-axis visualization of H(t) and κ₂(t).

    Left Y-axis: Adaptive step size H(t)
    Right Y-axis: Theoretical curvature κ₂(t)
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Step size on left axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Time t', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Adaptive Step Size H(t)', color=color1, fontsize=12, fontweight='bold')
    line1 = ax1.plot(analysis['adaptive_times'], analysis['adaptive_step_sizes'],
                     color=color1, linewidth=2.5, marker='o', markersize=4, label='H(t)', alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.2)

    # Curvature on right axis
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Theoretical Curvature (norm of acceleration)', color=color2, fontsize=12, fontweight='bold')
    line2 = ax2.plot(analysis['t_curvature'], analysis['curvature_full'],
                     color=color2, linewidth=2.5, label='Curvature(t)', alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Title with correlation
    corr_str = f"r = {analysis['correlation']:.3f}" if not np.isnan(analysis['correlation']) else "r = N/A"
    fig.suptitle(
        f"Trajectory #{analysis['trajectory_idx']}: Adaptive H(t) vs Curvature\n"
        f"Inverse Proportionality Check (Correlation: {corr_str})",
        fontsize=13, fontweight='bold', y=1.00
    )

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved trajectory analysis: {save_path}")
    plt.close()

def generate_correlation_report(
    trajectories: np.ndarray,
    t_grid: np.ndarray,
    selected_indices: list,
    save_dir: Path
):
    """
    Analyze 3 diverse trajectories for H(t) vs κ₂(t) correlation.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("CURVATURE & ADAPTIVE STEP SIZE CORRELATION")
    print(f"{'='*70}")

    analyses = []
    for traj_idx in selected_indices:
        print(f"\nAnalyzing Trajectory #{traj_idx}...", end='', flush=True)

        analysis = analyze_trajectory_curvature_correlation(
            trajectories, t_grid, traj_idx, tolerance=1e-3
        )
        analyses.append(analysis)

        print(f" [OK]")
        print(f"  Correlation (H vs curvature): {analysis['correlation']:.4f}")
        print(f"  Mean H: {analysis['mean_step_size']:.6f}")
        print(f"  Mean curvature: {analysis['mean_curvature']:.6e}")
        print(f"  Steps taken: {analysis['n_steps']}")

        # Plot
        plot_trajectory_analysis(
            analysis,
            save_dir / f"trajectory_{traj_idx:02d}_correlation.png"
        )

    # Save summary
    summary = {
        'selected_trajectories': [int(i) for i in selected_indices],
        'analyses': [
            {
                'trajectory_idx': int(a['trajectory_idx']),
                'correlation': float(a['correlation']) if not np.isnan(a['correlation']) else None,
                'mean_step_size': float(a['mean_step_size']),
                'mean_curvature': float(a['mean_curvature']),
                'n_steps': int(a['n_steps']),
            }
            for a in analyses
        ]
    }

    with open(save_dir / "correlation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    print("\nExpected behavior: H(t) should be INVERSELY proportional to curvature")
    print("- High curvature (large) -> Small step size (H small)")
    print("- Low curvature (small) -> Large step size (H large)")
    print("\nExpected correlation: r ~ -0.3 to -0.8 (negative)")
    print("\nResults:")
    for a in analyses:
        interp = "GOOD" if a['correlation'] < -0.3 else "MODERATE" if a['correlation'] < 0 else "WEAK"
        print(f"  Trajectory #{a['trajectory_idx']:2d}: r = {a['correlation']:7.4f} ({interp})")

    return analyses

if __name__ == '__main__':
    # Load artifacts
    jacobian_results = Path("jacobian_eda/results")
    trajectories = np.load(jacobian_results / "trajectories.npy")

    # Reconstruct proper time grid (100 steps from 0 to 0.999)
    n_steps = trajectories.shape[1]
    t_grid = np.linspace(0, 0.999, n_steps)

    print(f"Loaded trajectories: {trajectories.shape}")

    # Select 3 diverse trajectories
    n_total = trajectories.shape[0]
    selected = [0, n_total // 2, n_total - 1]  # Simple, middle, complex

    generate_correlation_report(trajectories, t_grid, selected, Path("richardson_eda/results"))
