"""
Flow Field Visualization - Trajectories in PCA space
Shows the learned flow from noise z(0) to data z(1)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
from pathlib import Path
from typing import Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.fm import ImageFlowMatcher


def capture_full_trajectories(
    model: ImageFlowMatcher,
    num_trajectories: int = 100,
    num_steps: int = 50,
    device: str = 'cpu'
) -> Tuple[list, list, torch.Tensor]:
    """
    Capture full trajectories at all intermediate timesteps.

    Args:
        model: Trained ImageFlowMatcher
        num_trajectories: Number of different random seeds
        num_steps: Number of timesteps per trajectory
        device: Device to run on

    Returns:
        Tuple of (all_trajectories, z_init_list, time_grid)
        - all_trajectories: List of [num_steps, 784] trajectory arrays
        - z_init_list: List of initial noise [784]
        - time_grid: Timesteps
    """
    from flow_matching.solver import ODESolver

    model.eval()
    model.to(device)

    all_trajectories = []
    z_init_list = []
    time_grid = torch.linspace(0, 1, num_steps, device=device)

    print(f"Capturing {num_trajectories} full trajectories with {num_steps} steps each...")

    for traj_idx in range(num_trajectories):
        if (traj_idx + 1) % 20 == 0:
            print(f"  {traj_idx + 1}/{num_trajectories}")

        # Random initialization
        torch.manual_seed(traj_idx)
        x_init = torch.randn(1, 1, 28, 28, device=device)
        z_init = x_init.reshape(-1).cpu()  # Flatten to 784
        z_init_list.append(z_init)

        # Create solver
        solver = ODESolver(velocity_model=model.model)

        # Generate trajectory
        with torch.no_grad():
            trajectory = solver.sample(
                x_init=x_init,
                time_grid=time_grid,
                method='dopri5',
                step_size=None,
                atol=1e-4,
                rtol=1e-4,
                return_intermediates=True,
            )

        # Handle list or tensor
        if isinstance(trajectory, (list, tuple)):
            trajectory = torch.stack(trajectory, dim=0)
        elif trajectory.shape[0] != len(time_grid):
            if trajectory.shape[1] == len(time_grid):
                trajectory = trajectory.transpose(0, 1)

        # Denormalize if needed
        if model.normalize_data:
            trajectory = (trajectory + 1) / 2
            trajectory = torch.clamp(trajectory, 0.0, 1.0)

        # Flatten: [num_steps, 784]
        trajectory_flat = trajectory.reshape(num_steps, -1).cpu().numpy()
        all_trajectories.append(trajectory_flat)

    return all_trajectories, z_init_list, time_grid


def fit_pca_on_trajectories(
    all_trajectories: list,
    z_init_list: list,
    n_components: int = 2
) -> PCA:
    """Fit PCA on all states (both noise and generated)."""
    all_states = []

    # Collect all states from all trajectories
    for traj in all_trajectories:
        all_states.extend(traj)

    # Add initial noise
    z_init_array = np.array([z.numpy() for z in z_init_list])
    all_states.extend(z_init_array)

    all_states = np.array(all_states)
    print(f"Fitting PCA on {len(all_states)} total states...")

    pca = PCA(n_components=n_components)
    pca.fit(all_states)

    return pca


def plot_flow_field_trajectories(
    all_trajectories: list,
    z_init_list: list,
    time_grid: torch.Tensor,
    pca: PCA,
    save_path: str = 'flow_field.png'
) -> None:
    """
    Plot trajectories in PCA space showing flow from noise to data.

    Args:
        all_trajectories: List of [num_steps, 784] arrays
        z_init_list: List of initial noise vectors
        time_grid: Timesteps
        pca: Fitted PCA transformer
        save_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(14, 12))

    # Project initial noise (z(0)) - blue points
    z_init_array = np.array([z.numpy() for z in z_init_list])
    z_init_proj = pca.transform(z_init_array)

    # Project final states (z(1)) - dark points
    z_final_list = [traj[-1] for traj in all_trajectories]
    z_final_array = np.array(z_final_list)
    z_final_proj = pca.transform(z_final_array)

    # Draw trajectories as lines
    print("Drawing flow field trajectories...")
    num_traj = len(all_trajectories)

    colors = plt.cm.Greens(np.linspace(0.3, 0.8, num_traj))

    for traj_idx, traj in enumerate(all_trajectories):
        # Project all timesteps
        traj_proj = pca.transform(traj)

        # Draw line from start to end
        ax.plot(traj_proj[:, 0], traj_proj[:, 1],
               color=colors[traj_idx], alpha=0.4, linewidth=1.5)

    # Plot initial states (noise) - blue
    ax.scatter(z_init_proj[:, 0], z_init_proj[:, 1],
              c='blue', s=50, alpha=0.6, edgecolors='darkblue',
              linewidth=1, label='z(0) - Noise', zorder=5)

    # Plot final states (data) - dark
    ax.scatter(z_final_proj[:, 0], z_final_proj[:, 1],
              c='black', s=50, alpha=0.6, edgecolors='darkgray',
              linewidth=1, label='z(1) - Data', zorder=5)

    # Add flow arrows along some trajectories
    print("Adding flow field arrows...")
    for traj_idx in range(0, num_traj, max(1, num_traj // 15)):  # Show ~15 arrows
        traj = all_trajectories[traj_idx]
        traj_proj = pca.transform(traj)

        # Sample points along trajectory for arrows
        num_arrows = 3
        arrow_indices = np.linspace(0, len(traj_proj) - 1, num_arrows + 2, dtype=int)[1:-1]

        for arr_idx in arrow_indices:
            if arr_idx > 0 and arr_idx < len(traj_proj) - 1:
                start = traj_proj[arr_idx - 1]
                end = traj_proj[arr_idx]
                dx = end[0] - start[0]
                dy = end[1] - start[1]

                ax.arrow(start[0], start[1], dx, dy,
                        head_width=0.3, head_length=0.2,
                        fc='olive', ec='olive', alpha=0.5, linewidth=0.8)

    # Labels and formatting
    var1 = pca.explained_variance_ratio_[0]
    var2 = pca.explained_variance_ratio_[1]

    ax.set_xlabel(f'PC1 ({var1:.1%})', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'PC2 ({var2:.1%})', fontsize=13, fontweight='bold')
    ax.set_title('Flow Matching: Learned Trajectories from Noise to Data\n' +
                f'(Green: flow field, Blue: z(0) initialization, Black: z(1) target)',
                fontsize=14, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=12, markerscale=1.5)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Flow field visualization saved to {save_path}")
    plt.close()


def plot_flow_field_density(
    all_trajectories: list,
    z_init_list: list,
    pca: PCA,
    save_path: str = 'flow_field_density.png'
) -> None:
    """
    Alternative visualization with trajectory density heatmap.

    Args:
        all_trajectories: List of [num_steps, 784] arrays
        z_init_list: List of initial noise vectors
        pca: Fitted PCA transformer
        save_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(14, 12))

    # Collect all projected states for density plot
    all_proj_states = []
    for traj in all_trajectories:
        traj_proj = pca.transform(traj)
        all_proj_states.extend(traj_proj)

    all_proj_states = np.array(all_proj_states)

    # 2D histogram for density
    h = ax.hist2d(all_proj_states[:, 0], all_proj_states[:, 1],
                  bins=30, cmap='YlOrRd', alpha=0.7)
    plt.colorbar(h[3], ax=ax, label='Trajectory Density')

    # Draw individual trajectories lightly
    for traj_idx, traj in enumerate(all_trajectories):
        traj_proj = pca.transform(traj)
        ax.plot(traj_proj[:, 0], traj_proj[:, 1],
               color='gray', alpha=0.15, linewidth=0.8)

    # Plot initial and final
    z_init_array = np.array([z.numpy() for z in z_init_list])
    z_init_proj = pca.transform(z_init_array)
    z_final_array = np.array([traj[-1] for traj in all_trajectories])
    z_final_proj = pca.transform(z_final_array)

    ax.scatter(z_init_proj[:, 0], z_init_proj[:, 1],
              c='blue', s=80, alpha=0.7, edgecolors='darkblue',
              linewidth=1.5, label='z(0) - Noise', zorder=10)
    ax.scatter(z_final_proj[:, 0], z_final_proj[:, 1],
              c='black', s=80, alpha=0.7, edgecolors='darkgray',
              linewidth=1.5, label='z(1) - Data', zorder=10)

    var1 = pca.explained_variance_ratio_[0]
    var2 = pca.explained_variance_ratio_[1]

    ax.set_xlabel(f'PC1 ({var1:.1%})', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'PC2 ({var2:.1%})', fontsize=13, fontweight='bold')
    ax.set_title('Flow Matching: Trajectory Density in PCA Space',
                fontsize=14, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=12, markerscale=1.5)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Density visualization saved to {save_path}")
    plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize flow field in PCA space')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--num-trajectories', type=int, default=100,
                       help='Number of trajectories')
    parser.add_argument('--num-steps', type=int, default=50, help='Steps per trajectory')
    parser.add_argument('--output-dir', type=str, default='results/flow_field',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    print("[LOADING] Loading model...")
    device = torch.device(args.device)
    model = ImageFlowMatcher()
    state_dict = torch.load(args.checkpoint, map_location=device)
    if 'state_dict' in state_dict:
        model.load_state_dict(state_dict['state_dict'])
    else:
        model.load_state_dict(state_dict)
    model.to(device)
    print("[OK] Model loaded\n")

    # Capture trajectories
    all_trajectories, z_init_list, time_grid = capture_full_trajectories(
        model,
        num_trajectories=args.num_trajectories,
        num_steps=args.num_steps,
        device=str(device)
    )

    # Fit PCA
    print()
    pca = fit_pca_on_trajectories(all_trajectories, z_init_list, n_components=2)
    print(f"PCA variance explained: PC1={pca.explained_variance_ratio_[0]:.1%}, " +
          f"PC2={pca.explained_variance_ratio_[1]:.1%}\n")

    # Create visualizations
    print("="*80)
    print("FLOW FIELD VISUALIZATION")
    print("="*80 + "\n")

    plot_flow_field_trajectories(
        all_trajectories, z_init_list, time_grid, pca,
        save_path=f'{args.output_dir}/flow_field_trajectories.png'
    )

    plot_flow_field_density(
        all_trajectories, z_init_list, pca,
        save_path=f'{args.output_dir}/flow_field_density.png'
    )

    print(f"\n[SUCCESS] Visualizations saved to {args.output_dir}")
