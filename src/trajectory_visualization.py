"""
Trajectory Visualization - Visualize noise-to-image evolution
Shows both latent space paths and actual image progression
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from sklearn.decomposition import PCA
from typing import List, Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.fm import ImageFlowMatcher


def capture_trajectory_images(
    model: ImageFlowMatcher,
    num_seeds: int = 6,
    num_steps: int = 50,
    device: str = 'cpu'
) -> Tuple[List[np.ndarray], List[np.ndarray], List[torch.Tensor]]:
    """
    Generate trajectories capturing images at each timestep.

    Args:
        model: Trained ImageFlowMatcher
        num_seeds: Number of different noise initializations
        num_steps: Number of timesteps to capture
        device: Device to run on

    Returns:
        Tuple of (all_images, all_states, time_grid)
        - all_images: List of [num_seeds, num_steps, 1, 28, 28] image arrays
        - all_states: List of final states for each trajectory
        - time_grid: Timesteps used
    """
    from flow_matching.solver import ODESolver

    model.eval()
    model.to(device)

    all_images = []
    all_states = []

    time_grid = torch.linspace(0, 1, num_steps, device=device)

    print(f"Generating {num_seeds} trajectories with {num_steps} timesteps each...")

    for seed_idx in range(num_seeds):
        print(f"  Trajectory {seed_idx + 1}/{num_seeds}")

        # Random initialization
        torch.manual_seed(seed_idx * 42)  # Deterministic for reproducibility
        x_init = torch.randn(1, 1, 28, 28, device=device)

        # Create solver
        solver = ODESolver(velocity_model=model.model)

        # Generate with intermediate returns
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

        # Handle list or tensor output
        if isinstance(trajectory, (list, tuple)):
            trajectory = torch.stack(trajectory, dim=0)
        elif trajectory.shape[0] != len(time_grid):
            if trajectory.shape[1] == len(time_grid):
                trajectory = trajectory.transpose(0, 1)

        # trajectory shape: [num_steps, 1, 1, 28, 28]

        # Denormalize if needed
        if model.normalize_data:
            trajectory = (trajectory + 1) / 2
            trajectory = torch.clamp(trajectory, 0.0, 1.0)

        # Convert to numpy
        trajectory_np = trajectory.detach().cpu().numpy()
        all_images.append(trajectory_np)
        all_states.append(trajectory[-1].detach().cpu())  # Final state

    return all_images, all_states, time_grid


def plot_latent_trajectories(
    all_states: List[torch.Tensor],
    save_path: Optional[str] = None,
    title: str = "Inference Trajectories in PCA Space"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Plot trajectories in 2D PCA space of final states.

    Args:
        all_states: List of final state tensors
        save_path: Where to save the figure
        title: Plot title

    Returns:
        Tuple of (projected_states, pca_object)
    """
    # Flatten states for PCA
    states_flat = torch.cat([s.reshape(1, -1) for s in all_states], dim=0).numpy()

    # PCA projection
    pca = PCA(n_components=2)
    proj = pca.fit_transform(states_flat)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_states)))

    for i, color in enumerate(colors):
        ax.scatter(proj[i, 0], proj[i, 1], s=200, color=color,
                  edgecolor='black', linewidth=2, zorder=3, label=f'Seed {i}')
        ax.annotate(f'{i}', (proj[i, 0], proj[i, 1]),
                   fontsize=12, fontweight='bold', ha='center', va='center')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Latent space plot saved to {save_path}")

    plt.close()
    return proj, pca


def plot_image_evolution(
    all_images: List[np.ndarray],
    time_grid: torch.Tensor,
    save_path: Optional[str] = None,
    title: str = "Image Evolution: Noise → Digit"
) -> None:
    """
    Plot image evolution from noise to digit for each trajectory.

    Args:
        all_images: List of trajectory image arrays [num_steps, 1, 1, 28, 28]
        time_grid: Timesteps
        save_path: Where to save the figure
        title: Plot title
    """
    num_seeds = len(all_images)
    num_steps = len(time_grid)

    # Sample timesteps to display (show every Nth step to avoid too many columns)
    display_indices = np.linspace(0, num_steps - 1, min(12, num_steps), dtype=int)
    num_display = len(display_indices)

    fig = plt.figure(figsize=(16, num_seeds * 2.5))
    gs = GridSpec(num_seeds, num_display + 1, figure=fig,
                  hspace=0.3, wspace=0.1, width_ratios=[0.5] + [1] * num_display)

    for seed_idx, images in enumerate(all_images):
        # Left: timestep label
        ax = fig.add_subplot(gs[seed_idx, 0])
        ax.text(0.5, 0.5, f'Seed {seed_idx}', ha='center', va='center',
               fontsize=14, fontweight='bold', transform=ax.transAxes)
        ax.axis('off')

        # Right: image evolution
        for col_idx, step_idx in enumerate(display_indices):
            ax = fig.add_subplot(gs[seed_idx, col_idx + 1])

            # Get image at this timestep
            img = images[step_idx, 0, 0, :, :]  # [28, 28]

            ax.imshow(img, cmap='gray', vmin=0, vmax=1)

            # Title: timestep
            t_val = time_grid[step_idx].item()
            ax.set_title(f't={t_val:.2f}', fontsize=10)
            ax.axis('off')

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Image evolution plot saved to {save_path}")

    plt.close()


def plot_combined_visualization(
    all_images: List[np.ndarray],
    all_states: List[torch.Tensor],
    time_grid: torch.Tensor,
    save_path_latent: Optional[str] = None,
    save_path_images: Optional[str] = None,
) -> None:
    """
    Create both latent space and image evolution visualizations.

    Args:
        all_images: List of trajectory images
        all_states: List of final states for latent visualization
        time_grid: Timesteps
        save_path_latent: Path to save latent space plot
        save_path_images: Path to save image evolution plot
    """
    print("\n" + "="*80)
    print("TRAJECTORY VISUALIZATION")
    print("="*80)

    # Plot 1: Latent trajectories
    print("\n[1/2] Creating latent space trajectory plot...")
    plot_latent_trajectories(
        all_states,
        save_path=save_path_latent,
        title="Inference Trajectories in PCA Space (Final States)"
    )

    # Plot 2: Image evolution
    print("[2/2] Creating image evolution plot...")
    plot_image_evolution(
        all_images,
        time_grid,
        save_path=save_path_images,
        title="Image Evolution: Pure Noise → Coherent Digit"
    )

    print("\n[OK] Visualization complete!")


if __name__ == '__main__':
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    parser = argparse.ArgumentParser(description='Visualize inference trajectories')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--num-seeds', type=int, default=6, help='Number of trajectories')
    parser.add_argument('--num-steps', type=int, default=50, help='Timesteps per trajectory')
    parser.add_argument('--output-dir', type=str, default='results/trajectory_viz',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    print("[LOADING] Loading model from checkpoint...")
    device = torch.device(args.device)
    model = ImageFlowMatcher()
    state_dict = torch.load(args.checkpoint, map_location=device)
    if 'state_dict' in state_dict:
        model.load_state_dict(state_dict['state_dict'])
    else:
        model.load_state_dict(state_dict)
    model.to(device)
    print("[OK] Model loaded")

    # Generate trajectories
    all_images, all_states, time_grid = capture_trajectory_images(
        model,
        num_seeds=args.num_seeds,
        num_steps=args.num_steps,
        device=str(device)
    )

    # Create visualizations
    save_path_latent = Path(args.output_dir) / 'latent_trajectories.png'
    save_path_images = Path(args.output_dir) / 'image_evolution.png'

    plot_combined_visualization(
        all_images,
        all_states,
        time_grid,
        save_path_latent=str(save_path_latent),
        save_path_images=str(save_path_images),
    )

    print(f"\n[RESULTS] Visualizations saved to {args.output_dir}")
