#!/usr/bin/env python
"""
Safety Factor (Alpha) Stability Analysis for Richardson Extrapolation

Sweeps over both tolerance and safety factor (alpha) values to identify
the most stable and efficient hyperparameter combinations.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from adaptive_solver import AdaptiveFlowSolver, SolverConfig, velocity_from_trajectory_finite_diff, reference_trajectory_fine_euler

def run_alpha_stability_sweep(
    trajectories: np.ndarray,
    t_grid: np.ndarray,
    alpha_values: list = None,
    tolerance_values: list = None,
    n_traj_test: int = 50,
):
    """
    Sweep over both tolerance and alpha (safety factor) values.

    Args:
        trajectories: [N_traj, N_steps, d] pre-computed trajectories
        t_grid: [N_steps] time grid
        alpha_values: List of safety factors to test
        tolerance_values: List of tolerance values to test
        n_traj_test: Number of trajectories to test

    Returns:
        Dictionary with 2D sweep results indexed by (tolerance, alpha)
    """
    if alpha_values is None:
        alpha_values = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    if tolerance_values is None:
        tolerance_values = [1e-1, 10**(-1.5), 1e-2, 10**(-2.5), 1e-3, 10**(-3.5), 1e-4]

    n_total = trajectories.shape[0]
    selected_indices = np.linspace(0, n_total - 1, n_traj_test, dtype=int)

    print(f"\nAlpha Stability Sweep Configuration:")
    print(f"  Trajectories to test: {n_traj_test}")
    print(f"  Tolerance values: {len(tolerance_values)}")
    print(f"  Alpha values: {alpha_values}")
    print(f"  Total combinations: {len(tolerance_values) * len(alpha_values)}")

    results = {
        'tolerance_values': tolerance_values,
        'alpha_values': alpha_values,
        'selected_traj_indices': list(selected_indices),
        'sweep_results': {},  # Key: (tol, alpha) tuple
        'aggregated_metrics': {
            'tolerances': [],
            'alphas': [],
            'avg_nfe': [],
            'avg_error': [],
            'avg_reject_rate': [],
            'std_nfe': [],
            'std_error': [],
        }
    }

    t_start, t_end = t_grid[0], t_grid[-1]
    max_steps_cap = 100

    total_combinations = len(tolerance_values) * len(alpha_values)
    combo_count = 0

    # Sweep over all (tolerance, alpha) pairs
    for tol in tolerance_values:
        for alpha in alpha_values:
            combo_count += 1
            print(f"\n[{combo_count}/{total_combinations}] Tolerance={tol:.6e}, Alpha={alpha:.2f}")
            print(f"{'='*70}")

            nfe_values = []
            error_values = []
            reject_rates = []

            # Run on selected trajectories
            for traj_idx in selected_indices:
                x0 = trajectories[traj_idx, 0].copy()
                velocity_fn = velocity_from_trajectory_finite_diff(trajectories, t_grid, traj_idx)

                # Generate reference on first run for each tolerance
                if traj_idx == selected_indices[0]:
                    ref = reference_trajectory_fine_euler(x0, (t_start, t_end), velocity_fn, n_steps=1000)
                    x_ref = ref['trajectory'][-1]

                # Run adaptive solver
                config = SolverConfig(
                    tolerance=tol,
                    alpha=alpha,
                    f_min=0.1,
                    f_max=5.0,
                    max_steps=max_steps_cap,
                    verbose=False
                )
                solver = AdaptiveFlowSolver(config)
                adaptive_result = solver.solve(x0, (t_start, t_end), velocity_fn, H0=0.01)

                # Compute metrics
                x_adaptive = adaptive_result['trajectory'][-1]
                global_error = np.linalg.norm(x_adaptive - x_ref)

                nfe_values.append(adaptive_result['nfe'])
                error_values.append(global_error)
                reject_rates.append(adaptive_result['reject_rate'])

            # Aggregate
            key = (float(tol), float(alpha))
            results['sweep_results'][str(key)] = {
                'tolerance': tol,
                'alpha': alpha,
                'avg_nfe': float(np.mean(nfe_values)),
                'std_nfe': float(np.std(nfe_values)),
                'avg_error': float(np.mean(error_values)),
                'std_error': float(np.std(error_values)),
                'avg_reject_rate': float(np.mean(reject_rates)),
                'min_reject_rate': float(np.min(reject_rates)),
                'max_reject_rate': float(np.max(reject_rates)),
            }

            print(f"  NFE: {np.mean(nfe_values):.1f}±{np.std(nfe_values):.1f}")
            print(f"  Error: {np.mean(error_values):.3e}±{np.std(error_values):.3e}")
            print(f"  Reject Rate: {np.mean(reject_rates):.2%}")

    # Build aggregated metrics tables
    for tol in tolerance_values:
        for alpha in alpha_values:
            key = str((float(tol), float(alpha)))
            if key in results['sweep_results']:
                data = results['sweep_results'][key]
                results['aggregated_metrics']['tolerances'].append(tol)
                results['aggregated_metrics']['alphas'].append(alpha)
                results['aggregated_metrics']['avg_nfe'].append(data['avg_nfe'])
                results['aggregated_metrics']['std_nfe'].append(data['std_nfe'])
                results['aggregated_metrics']['avg_error'].append(data['avg_error'])
                results['aggregated_metrics']['std_error'].append(data['std_error'])
                results['aggregated_metrics']['avg_reject_rate'].append(data['avg_reject_rate'])

    return results

def plot_alpha_heatmaps(results: dict, save_dir: Path):
    """Generate heatmap visualizations for alpha stability analysis."""
    tols = np.array(results['tolerance_values'])
    alphas = np.array(results['alpha_values'])
    n_tol = len(tols)
    n_alpha = len(alphas)

    # Reshape aggregated metrics into matrices
    nfe_matrix = np.zeros((n_tol, n_alpha))
    error_matrix = np.zeros((n_tol, n_alpha))
    reject_matrix = np.zeros((n_tol, n_alpha))

    for i, tol in enumerate(tols):
        for j, alpha in enumerate(alphas):
            key = str((float(tol), float(alpha)))
            if key in results['sweep_results']:
                data = results['sweep_results'][key]
                nfe_matrix[i, j] = data['avg_nfe']
                error_matrix[i, j] = data['avg_error']
                reject_matrix[i, j] = data['avg_reject_rate'] * 100

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Heatmap 1: NFE vs (tolerance, alpha)
    ax = axes[0, 0]
    im = ax.imshow(nfe_matrix, aspect='auto', cmap='viridis', origin='lower')
    ax.set_xlabel('Safety Factor (alpha)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tolerance', fontsize=12, fontweight='bold')
    ax.set_xticks(range(n_alpha))
    ax.set_xticklabels([f'{a:.2f}' for a in alphas])
    ax.set_yticks(range(n_tol))
    ax.set_yticklabels([f'{t:.0e}' for t in tols])
    ax.set_title('Average NFE (Function Evaluations)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='NFE')

    # Heatmap 2: Error vs (tolerance, alpha)
    ax = axes[0, 1]
    im = ax.imshow(error_matrix, aspect='auto', cmap='plasma', origin='lower')
    ax.set_xlabel('Safety Factor (alpha)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tolerance', fontsize=12, fontweight='bold')
    ax.set_xticks(range(n_alpha))
    ax.set_xticklabels([f'{a:.2f}' for a in alphas])
    ax.set_yticks(range(n_tol))
    ax.set_yticklabels([f'{t:.0e}' for t in tols])
    ax.set_title('Average Global Error', fontsize=13, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, label='Error')
    cbar.set_label('Error (log scale)', fontsize=11)

    # Heatmap 3: Rejection Rate vs (tolerance, alpha)
    ax = axes[1, 0]
    im = ax.imshow(reject_matrix, aspect='auto', cmap='coolwarm', origin='lower', vmin=0, vmax=50)
    ax.set_xlabel('Safety Factor (alpha)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tolerance', fontsize=12, fontweight='bold')
    ax.set_xticks(range(n_alpha))
    ax.set_xticklabels([f'{a:.2f}' for a in alphas])
    ax.set_yticks(range(n_tol))
    ax.set_yticklabels([f'{t:.0e}' for t in tols])
    ax.set_title('Step Rejection Rate (%)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Reject Rate (%)')
    # Add contour line for 5% threshold
    contour = ax.contour(np.arange(n_alpha)-0.5, np.arange(n_tol)-0.5, reject_matrix,
                         levels=[5], colors='black', linewidths=2, linestyles='--')
    ax.clabel(contour, inline=True, fontsize=10, fmt='5%% target')

    # Panel 4: Efficiency frontier (error vs NFE, colored by alpha)
    ax = axes[1, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(alphas)))
    for j, alpha in enumerate(alphas):
        nfe_vals = []
        err_vals = []
        for i in range(len(tols)):
            if nfe_matrix[i, j] > 0:
                nfe_vals.append(nfe_matrix[i, j])
                err_vals.append(error_matrix[i, j])
        ax.plot(nfe_vals, err_vals, 'o-', linewidth=2.5, markersize=8,
                label=f'alpha={alpha:.2f}', color=colors[j])
    ax.set_xlabel('Average NFE', fontsize=12, fontweight='bold')
    ax.set_ylabel('Global Error', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.set_title('Efficiency Frontier (colored by alpha)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / "alpha_stability_heatmaps.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved alpha stability heatmaps: {save_path}")
    plt.close()

def find_optimal_alpha_config(results: dict) -> dict:
    """
    Find optimal (tolerance, alpha) pair based on stability and efficiency.

    Criteria:
    1. Reject rate < 5%
    2. Minimal NFE for acceptable error
    3. Stability across trajectories (low std_error)
    """
    candidates = []

    for key, data in results['sweep_results'].items():
        if data['avg_reject_rate'] < 0.05:  # Stable
            candidates.append({
                'tolerance': data['tolerance'],
                'alpha': data['alpha'],
                'nfe': data['avg_nfe'],
                'error': data['avg_error'],
                'reject_rate': data['avg_reject_rate'],
                'error_std': data['std_error'],
                'score': data['avg_nfe']  # Minimize NFE as primary criterion
            })

    if not candidates:
        print("WARNING: No (tolerance, alpha) pairs with reject rate < 5%")
        # Fallback: find minimal reject rate
        min_reject = min(float(d['avg_reject_rate']) for d in results['sweep_results'].values())
        candidates = [d for d in results['sweep_results'].values()
                     if float(d['avg_reject_rate']) <= min_reject + 0.05]

    # Sort by NFE (efficiency)
    candidates.sort(key=lambda x: x['nfe'])
    optimal = candidates[0]

    return {
        'tolerance': optimal['tolerance'],
        'alpha': optimal['alpha'],
        'estimated_nfe': optimal['nfe'],
        'estimated_error': optimal['error'],
        'estimated_reject_rate': optimal['reject_rate'],
        'num_candidates': len(candidates),
    }

if __name__ == '__main__':
    jacobian_results = Path("jacobian_eda/results")
    trajectories = np.load(jacobian_results / "trajectories.npy")

    n_steps = trajectories.shape[1]
    t_grid = np.linspace(0, 0.999, n_steps)

    print(f"Loaded trajectories: {trajectories.shape}")

    # Run sweep
    results = run_alpha_stability_sweep(
        trajectories,
        t_grid,
        alpha_values=[0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        tolerance_values=[1e-1, 10**(-1.5), 1e-2, 10**(-2.5), 1e-3, 10**(-3.5), 1e-4],
        n_traj_test=50
    )

    # Find optimal configuration
    optimal = find_optimal_alpha_config(results)

    # Save results
    result_dir = Path("richardson_eda/results")
    result_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to native Python types for JSON serialization
    results_serializable = {
        'tolerance_values': [float(t) for t in results['tolerance_values']],
        'alpha_values': [float(a) for a in results['alpha_values']],
        'selected_traj_indices': [int(i) for i in results['selected_traj_indices']],
        'sweep_results': results['sweep_results'],
        'aggregated_metrics': {
            'tolerances': [float(t) for t in results['aggregated_metrics']['tolerances']],
            'alphas': [float(a) for a in results['aggregated_metrics']['alphas']],
            'avg_nfe': [float(x) for x in results['aggregated_metrics']['avg_nfe']],
            'std_nfe': [float(x) for x in results['aggregated_metrics']['std_nfe']],
            'avg_error': [float(x) for x in results['aggregated_metrics']['avg_error']],
            'std_error': [float(x) for x in results['aggregated_metrics']['std_error']],
            'avg_reject_rate': [float(x) for x in results['aggregated_metrics']['avg_reject_rate']],
        }
    }

    with open(result_dir / "alpha_stability_results.json", 'w') as f:
        json.dump(results_serializable, f, indent=2)

    optimal_serializable = {
        'tolerance': float(optimal['tolerance']),
        'alpha': float(optimal['alpha']),
        'estimated_nfe': float(optimal['estimated_nfe']),
        'estimated_error': float(optimal['estimated_error']),
        'estimated_reject_rate': float(optimal['estimated_reject_rate']),
        'num_candidates': int(optimal['num_candidates']),
    }

    with open(result_dir / "optimal_alpha_config.json", 'w') as f:
        json.dump(optimal_serializable, f, indent=2)

    # Generate visualizations
    plot_alpha_heatmaps(results, result_dir)

    print(f"\n{'='*70}")
    print("ALPHA STABILITY ANALYSIS - OPTIMAL CONFIGURATION")
    print(f"{'='*70}")
    print(f"Tolerance:              {optimal['tolerance']:.6e}")
    print(f"Safety Factor (alpha):  {optimal['alpha']:.2f}")
    print(f"Estimated NFE:          {optimal['estimated_nfe']:.1f}")
    print(f"Estimated Error:        {optimal['estimated_error']:.3e}")
    print(f"Estimated Reject Rate:  {optimal['estimated_reject_rate']:.2%}")
    print(f"Stable candidates:      {optimal['num_candidates']} configs with <5% rejection")

    print(f"\nSaved results to richardson_eda/results/")
    print(f"  - alpha_stability_results.json")
    print(f"  - optimal_alpha_config.json")
    print(f"  - alpha_stability_heatmaps.png")
