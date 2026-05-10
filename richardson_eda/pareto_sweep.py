#!/usr/bin/env python
"""
Pareto Optimization Sweep for Richardson Extrapolation

Sweeps over tolerance values and compares:
- NFE (Number of Function Evaluations)
- Global Error (vs reference trajectory)
- Reject Rate

Generates Pareto frontier and optimal hyperparameter recommendations.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from adaptive_solver import AdaptiveFlowSolver, SolverConfig, velocity_from_trajectory_finite_diff, reference_trajectory_fine_euler

def run_pareto_sweep(
    trajectories: np.ndarray,
    t_grid: np.ndarray,
    n_traj_test: int = 10,
    tolerance_values: list = None
):
    """
    Run Pareto sweep over tolerance values.

    Args:
        trajectories: [N_traj, N_steps, d] pre-computed trajectories
        t_grid: [N_steps] time grid
        n_traj_test: Number of trajectories to test (evenly spaced)
        tolerance_values: List of tolerance values to sweep over

    Returns:
        Dictionary with sweep results
    """
    if tolerance_values is None:
        tolerance_values = [1e-1, 10**(-1.5), 1e-2, 10**(-2.5), 1e-3, 10**(-3.5), 1e-4]

    # Select evenly spaced trajectories for testing
    n_total = trajectories.shape[0]
    selected_indices = np.linspace(0, n_total - 1, n_traj_test, dtype=int)

    print(f"\nPareto Sweep Configuration:")
    print(f"  Trajectories to test: {n_traj_test} (indices: {list(selected_indices)})")
    print(f"  Tolerance values: {tolerance_values}")
    print(f"  Time span: [{t_grid[0]:.3f}, {t_grid[-1]:.3f}]")

    results = {
        'tolerance_values': tolerance_values,
        'selected_traj_indices': list(selected_indices),
        'trajectory_results': [],
        'aggregated_metrics': {
            'tolerances': [],
            'avg_nfe': [],
            'avg_error': [],
            'avg_reject_rate': [],
            'std_nfe': [],
            'std_error': [],
        }
    }

    t_start, t_end = t_grid[0], t_grid[-1]
    max_steps_cap = 100  # Cap adaptive steps at 100 max

    for traj_idx in selected_indices:
        print(f"\n{'='*70}")
        print(f"Trajectory {traj_idx}")
        print(f"{'='*70}")

        x0 = trajectories[traj_idx, 0].copy()
        velocity_fn = velocity_from_trajectory_finite_diff(trajectories, t_grid, traj_idx)

        # Ground truth reference (fine Euler)
        print(f"  Computing reference trajectory (fine Euler)...", end='', flush=True)
        ref = reference_trajectory_fine_euler(x0, (t_start, t_end), velocity_fn, n_steps=1000)
        x_ref = ref['trajectory'][-1]
        print(f" [OK]")

        traj_result = {'index': traj_idx, 'tolerance_results': []}

        for tolerance in tolerance_values:
            print(f"  Tolerance {tolerance:.6e}...", end='', flush=True)

            # Run adaptive solver (capped at 100 steps max)
            config = SolverConfig(
                tolerance=tolerance,
                alpha=0.9,
                f_min=0.1,
                f_max=5.0,
                max_steps=max_steps_cap,  # Cap at 100 steps
                verbose=False
            )
            solver = AdaptiveFlowSolver(config)
            adaptive_result = solver.solve(x0, (t_start, t_end), velocity_fn, H0=0.01)

            # Compute global error
            x_adaptive = adaptive_result['trajectory'][-1]
            global_error = np.linalg.norm(x_adaptive - x_ref)

            nfe = adaptive_result['nfe']
            reject_rate = adaptive_result['reject_rate']

            print(f" NFE={nfe}, Error={global_error:.3e}, RejectRate={reject_rate:.2%}")

            traj_result['tolerance_results'].append({
                'tolerance': tolerance,
                'nfe': nfe,
                'global_error': global_error,
                'reject_rate': reject_rate,
                'n_accepted': adaptive_result['n_accepted'],
                'n_rejected': adaptive_result['n_rejected'],
            })

        results['trajectory_results'].append(traj_result)

    # Aggregate metrics across trajectories
    print(f"\n{'='*70}")
    print("Aggregating Results")
    print(f"{'='*70}")

    for tol in tolerance_values:
        nfe_values = []
        error_values = []
        reject_rates = []

        for traj_res in results['trajectory_results']:
            for tol_res in traj_res['tolerance_results']:
                if tol_res['tolerance'] == tol:
                    nfe_values.append(tol_res['nfe'])
                    error_values.append(tol_res['global_error'])
                    reject_rates.append(tol_res['reject_rate'])

        results['aggregated_metrics']['tolerances'].append(tol)
        results['aggregated_metrics']['avg_nfe'].append(np.mean(nfe_values))
        results['aggregated_metrics']['std_nfe'].append(np.std(nfe_values))
        results['aggregated_metrics']['avg_error'].append(np.mean(error_values))
        results['aggregated_metrics']['std_error'].append(np.std(error_values))
        results['aggregated_metrics']['avg_reject_rate'].append(np.mean(reject_rates))

        print(f"  Tolerance {tol:.6e}: NFE={np.mean(nfe_values):.1f}±{np.std(nfe_values):.1f}, "
              f"Error={np.mean(error_values):.3e}±{np.std(error_values):.3e}, "
              f"RejectRate={np.mean(reject_rates):.2%}")

    return results

def plot_pareto_frontier(results: dict, save_path: Path):
    """Generate Pareto frontier visualization."""
    metrics = results['aggregated_metrics']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Pareto frontier: Error vs NFE
    ax = axes[0, 0]
    ax.errorbar(metrics['avg_nfe'], metrics['avg_error'],
                xerr=metrics['std_nfe'], yerr=metrics['std_error'],
                fmt='o-', linewidth=2.5, markersize=8, capsize=5, capthick=2, color='steelblue')
    ax.set_xlabel('Average NFE (Function Evaluations)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Global Error (MSE vs Reference)', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.set_title('Pareto Frontier: Error vs Computational Cost', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')

    # Annotate tolerance values
    for tol, nfe, err in zip(metrics['tolerances'], metrics['avg_nfe'], metrics['avg_error']):
        ax.annotate(f'{tol:.0e}', (nfe, err), fontsize=9, ha='center', va='bottom')

    # Tolerance vs Error
    ax = axes[0, 1]
    ax.loglog(metrics['tolerances'], metrics['avg_error'], 'go-', linewidth=2.5, markersize=8, label='Actual Error')
    ax.loglog(metrics['tolerances'], metrics['tolerances'], 'r--', linewidth=2, label='y=tolerance (ideal)')
    ax.set_xlabel('Tolerance Setting', fontsize=12, fontweight='bold')
    ax.set_ylabel('Achieved Global Error', fontsize=12, fontweight='bold')
    ax.set_title('Error vs Tolerance Setting', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    # Tolerance vs NFE
    ax = axes[1, 0]
    ax.loglog(metrics['tolerances'], metrics['avg_nfe'], 'bs-', linewidth=2.5, markersize=8)
    ax.set_xlabel('Tolerance Setting', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average NFE', fontsize=12, fontweight='bold')
    ax.set_title('Computational Cost vs Tolerance', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')

    # Reject rate vs Tolerance
    ax = axes[1, 1]
    ax.semilogx(metrics['tolerances'], [100*r for r in metrics['avg_reject_rate']], 'mo-', linewidth=2.5, markersize=8)
    ax.axhline(5, color='r', linestyle='--', linewidth=2, label='Target: <5%')
    ax.set_xlabel('Tolerance Setting', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reject Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Step Rejection Rate vs Tolerance', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(20, 100*max(metrics['avg_reject_rate']))])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved Pareto frontier visualization: {save_path}")
    plt.close()

def find_optimal_hyperparameters(results: dict) -> dict:
    """
    Find optimal hyperparameter set based on Pareto analysis.

    Criteria:
    1. Reject rate < 5%
    2. Minimal NFE for acceptable error
    """
    metrics = results['aggregated_metrics']

    # Filter by reject rate < 5%
    valid_idx = [i for i, r in enumerate(metrics['avg_reject_rate']) if r < 0.05]

    if not valid_idx:
        print("WARNING: No tolerances with reject rate < 5%. Using most conservative.")
        valid_idx = [len(metrics['tolerances']) - 1]

    # Among valid, find minimum NFE
    valid_nfe = [metrics['avg_nfe'][i] for i in valid_idx]
    best_idx = valid_idx[np.argmin(valid_nfe)]

    optimal = {
        'tolerance': metrics['tolerances'][best_idx],
        'alpha': 0.9,
        'f_min': 0.1,
        'f_max': 5.0,
        'estimated_nfe': metrics['avg_nfe'][best_idx],
        'estimated_error': metrics['avg_error'][best_idx],
        'estimated_reject_rate': metrics['avg_reject_rate'][best_idx],
    }

    return optimal

if __name__ == '__main__':
    # Load artifacts
    jacobian_results = Path("jacobian_eda/results")
    trajectories = np.load(jacobian_results / "trajectories.npy")

    # Reconstruct proper time grid (100 steps from 0 to 0.999)
    n_steps = trajectories.shape[1]
    t_grid = np.linspace(0, 0.999, n_steps)

    print(f"Loaded trajectories: {trajectories.shape}")
    print(f"Time grid: {t_grid.shape}, range [{t_grid[0]:.3f}, {t_grid[-1]:.3f}]")

    # Run sweep
    results = run_pareto_sweep(
        trajectories,
        t_grid,
        n_traj_test=8,
        tolerance_values=[1e-1, 10**(-1.5), 1e-2, 10**(-2.5), 1e-3, 10**(-3.5), 1e-4]
    )

    # Find optimal
    optimal = find_optimal_hyperparameters(results)

    print(f"\n{'='*70}")
    print("OPTIMAL HYPERPARAMETERS")
    print(f"{'='*70}")
    print(f"Tolerance (t):         {optimal['tolerance']:.6e}")
    print(f"Safety Factor (α):     {optimal['alpha']}")
    print(f"Min Step Multiplier:   {optimal['f_min']}")
    print(f"Max Step Multiplier:   {optimal['f_max']}")
    print(f"\nEstimated Performance:")
    print(f"  NFE:                 {optimal['estimated_nfe']:.1f}")
    print(f"  Global Error:        {optimal['estimated_error']:.3e}")
    print(f"  Reject Rate:         {optimal['estimated_reject_rate']:.2%}")

    # Save results
    result_dir = Path("richardson_eda/results")
    result_dir.mkdir(parents=True, exist_ok=True)

    with open(result_dir / "pareto_results.json", 'w') as f:
        # Convert numpy types for JSON serialization
        results_serializable = {
            'tolerance_values': [float(t) for t in results['tolerance_values']],
            'selected_traj_indices': [int(i) for i in results['selected_traj_indices']],
            'aggregated_metrics': {
                'tolerances': [float(t) for t in results['aggregated_metrics']['tolerances']],
                'avg_nfe': [float(x) for x in results['aggregated_metrics']['avg_nfe']],
                'avg_error': [float(x) for x in results['aggregated_metrics']['avg_error']],
                'avg_reject_rate': [float(x) for x in results['aggregated_metrics']['avg_reject_rate']],
                'std_nfe': [float(x) for x in results['aggregated_metrics']['std_nfe']],
                'std_error': [float(x) for x in results['aggregated_metrics']['std_error']],
            }
        }
        json.dump(results_serializable, f, indent=2)

    with open(result_dir / "optimal_hyperparameters.json", 'w') as f:
        json.dump(optimal, f, indent=2)

    # Plot
    plot_pareto_frontier(results, result_dir / "pareto_frontier.png")

    print(f"\nSaved results to richardson_eda/results/")
    print(f"  - pareto_results.json")
    print(f"  - optimal_hyperparameters.json")
    print(f"  - pareto_frontier.png")
