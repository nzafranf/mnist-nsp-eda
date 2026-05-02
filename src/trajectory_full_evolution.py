"""
Full Trajectory Evolution - Show all timesteps from noise to digit
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.fm import ImageFlowMatcher


def capture_trajectory_images(
    model: ImageFlowMatcher,
    num_seeds: int = 6,
    num_steps: int = 50,
    device: str = 'cpu'
) -> tuple:
    """Generate trajectories capturing ALL images at each timestep."""
    from flow_matching.solver import ODESolver

    model.eval()
    model.to(device)

    all_images = []
    time_grid = torch.linspace(0, 1, num_steps, device=device)

    print(f"Generating {num_seeds} trajectories with {num_steps} timesteps each...")

    for seed_idx in range(num_seeds):
        print(f"  Trajectory {seed_idx + 1}/{num_seeds}")

        torch.manual_seed(seed_idx * 42)
        x_init = torch.randn(1, 1, 28, 28, device=device)

        solver = ODESolver(velocity_model=model.model)

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

        if isinstance(trajectory, (list, tuple)):
            trajectory = torch.stack(trajectory, dim=0)
        elif trajectory.shape[0] != len(time_grid):
            if trajectory.shape[1] == len(time_grid):
                trajectory = trajectory.transpose(0, 1)

        if model.normalize_data:
            trajectory = (trajectory + 1) / 2
            trajectory = torch.clamp(trajectory, 0.0, 1.0)

        trajectory_np = trajectory.detach().cpu().numpy()
        all_images.append(trajectory_np)

    return all_images, time_grid


def plot_all_timesteps_per_trajectory(
    all_images: List[np.ndarray],
    time_grid: torch.Tensor,
    output_dir: str = 'results/trajectory_full',
) -> None:
    """
    Create separate detailed plots for each trajectory showing all timesteps.

    Args:
        all_images: List of [num_steps, 1, 1, 28, 28] arrays
        time_grid: Timesteps
        output_dir: Output directory
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    num_seeds = len(all_images)
    num_steps = len(time_grid)

    print("\nCreating detailed trajectory plots (all timesteps)...")

    for seed_idx, images in enumerate(all_images):
        print(f"  Trajectory {seed_idx + 1}/{num_seeds}")

        # Create figure with all timesteps
        # Layout: num_steps columns, organized in rows of 10 for readability
        cols_per_row = 10
        num_rows = (num_steps + cols_per_row - 1) // cols_per_row

        fig = plt.figure(figsize=(20, num_rows * 2.5))
        gs = GridSpec(num_rows, cols_per_row, figure=fig, hspace=0.4, wspace=0.2)

        for step_idx in range(num_steps):
            row = step_idx // cols_per_row
            col = step_idx % cols_per_row

            ax = fig.add_subplot(gs[row, col])

            img = images[step_idx, 0, 0, :, :]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)

            t_val = time_grid[step_idx].item()
            ax.set_title(f't={t_val:.3f}', fontsize=9, fontweight='bold')
            ax.axis('off')

        fig.suptitle(f'Seed {seed_idx}: Complete Trajectory Evolution (t=0 → t=1)',
                    fontsize=14, fontweight='bold', y=0.995)

        save_path = Path(output_dir) / f'trajectory_seed{seed_idx:02d}_all_steps.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"    Saved to {save_path}")
        plt.close()


def plot_all_trajectories_grid(
    all_images: List[np.ndarray],
    time_grid: torch.Tensor,
    output_dir: str = 'results/trajectory_full',
    sample_every_n: int = 1,
) -> None:
    """
    Create a single large grid showing all trajectories and timesteps.

    Args:
        all_images: List of [num_steps, 1, 1, 28, 28] arrays
        time_grid: Timesteps
        output_dir: Output directory
        sample_every_n: Sample every Nth timestep (to avoid too many columns)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    num_seeds = len(all_images)
    num_steps = len(time_grid)

    # Sample timesteps
    if sample_every_n > 1:
        indices = range(0, num_steps, sample_every_n)
    else:
        indices = range(num_steps)

    num_cols = len(list(indices))

    print(f"\nCreating combined grid ({num_seeds} trajectories × {num_cols} timesteps)...")

    # Create large figure
    fig = plt.figure(figsize=(num_cols * 0.6, num_seeds * 1.2))
    gs = GridSpec(num_seeds, num_cols, figure=fig, hspace=0.15, wspace=0.05)

    for seed_idx, images in enumerate(all_images):
        for col_idx, step_idx in enumerate(indices):
            ax = fig.add_subplot(gs[seed_idx, col_idx])

            img = images[step_idx, 0, 0, :, :]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')

            # Add timestep label only on first row
            if seed_idx == 0:
                t_val = time_grid[step_idx].item()
                ax.set_title(f'{t_val:.2f}', fontsize=7)

            # Add seed label only on first column
            if col_idx == 0:
                ax.set_ylabel(f'Seed {seed_idx}', fontsize=10, fontweight='bold')

    fig.suptitle('All Trajectories: Evolution from Noise to Digit\n(Each row = one trajectory, each column = one timestep)',
                fontsize=12, fontweight='bold', y=0.99)

    save_path = Path(output_dir) / 'all_trajectories_grid.png'
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"  Saved to {save_path}")
    plt.close()


def plot_single_trajectory_detailed(
    images: np.ndarray,
    time_grid: torch.Tensor,
    seed_idx: int = 0,
    output_dir: str = 'results/trajectory_full',
) -> None:
    """
    Create a detailed strip showing one complete trajectory evolution.

    Args:
        images: [num_steps, 1, 1, 28, 28] array
        time_grid: Timesteps
        seed_idx: Seed number for title
        output_dir: Output directory
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    num_steps = len(time_grid)

    # Create horizontal strip
    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 0.5, 1.5))

    if num_steps == 1:
        axes = [axes]

    for step_idx, ax in enumerate(axes):
        img = images[step_idx, 0, 0, :, :]
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])

        # Add thin border
        for spine in ax.spines.values():
            spine.set_edgecolor('lightgray')
            spine.set_linewidth(0.5)

        # Label only every 5th step to reduce clutter
        if step_idx % 5 == 0:
            t_val = time_grid[step_idx].item()
            ax.set_xlabel(f'{t_val:.2f}', fontsize=8)

    fig.suptitle(f'Seed {seed_idx}: Continuous Evolution (t=0 → t=1)',
                fontsize=11, fontweight='bold')
    plt.tight_layout()

    save_path = Path(output_dir) / f'trajectory_seed{seed_idx:02d}_strip.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved strip for Seed {seed_idx}")
    plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Show complete trajectory evolution')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--num-seeds', type=int, default=6, help='Number of trajectories')
    parser.add_argument('--num-steps', type=int, default=50, help='Timesteps per trajectory')
    parser.add_argument('--output-dir', type=str, default='results/trajectory_full',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')

    args = parser.parse_args()

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
    print("[OK] Model loaded\n")

    # Generate trajectories
    all_images, time_grid = capture_trajectory_images(
        model,
        num_seeds=args.num_seeds,
        num_steps=args.num_steps,
        device=str(device)
    )

    # Create visualizations
    print("\n" + "="*80)
    print("TRAJECTORY EVOLUTION VISUALIZATIONS")
    print("="*80)

    # 1. Detailed per-trajectory plots
    plot_all_timesteps_per_trajectory(all_images, time_grid, args.output_dir)

    # 2. Combined grid view
    plot_all_trajectories_grid(all_images, time_grid, args.output_dir, sample_every_n=1)

    # 3. Strip visualizations
    print("\nCreating strip visualizations (continuous evolution)...")
    for seed_idx, images in enumerate(all_images):
        plot_single_trajectory_detailed(images, time_grid, seed_idx, args.output_dir)

    print(f"\n[SUCCESS] All visualizations saved to {args.output_dir}")
    print(f"\nGenerated:")
    print(f"  - trajectory_seed##_all_steps.png (detailed per-trajectory)")
    print(f"  - all_trajectories_grid.png (combined view)")
    print(f"  - trajectory_seed##_strip.png (continuous evolution)")
