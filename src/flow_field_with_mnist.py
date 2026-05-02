"""
Flow Field with Ground Truth MNIST - Show trajectories aligned with true digit clusters
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from pathlib import Path
from typing import Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.fm import ImageFlowMatcher


def load_mnist_dataset(num_samples_per_class: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """Load MNIST and return flattened images and labels."""
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor()])
    mnist_test = datasets.MNIST('data', train=False, download=True, transform=transform)

    images_list = []
    labels_list = []

    # Sample evenly from each digit
    samples_per_digit = {i: 0 for i in range(10)}
    max_per_digit = num_samples_per_class

    for img, label in mnist_test:
        if samples_per_digit[label] < max_per_digit:
            images_list.append(img.reshape(-1).numpy())
            labels_list.append(label)
            samples_per_digit[label] += 1

        if all(v >= max_per_digit for v in samples_per_digit.values()):
            break

    images = np.array(images_list)
    labels = np.array(labels_list)

    print(f"Loaded {len(images)} MNIST images")
    return images, labels


def capture_full_trajectories(
    model: ImageFlowMatcher,
    num_trajectories: int = 100,
    num_steps: int = 50,
    device: str = 'cpu'
) -> Tuple[list, list, torch.Tensor]:
    """Capture full trajectories at all intermediate timesteps."""
    from flow_matching.solver import ODESolver

    model.eval()
    model.to(device)

    all_trajectories = []
    z_init_list = []
    time_grid = torch.linspace(0, 1, num_steps, device=device)

    print(f"Capturing {num_trajectories} trajectories...")

    for traj_idx in range(num_trajectories):
        if (traj_idx + 1) % 20 == 0:
            print(f"  {traj_idx + 1}/{num_trajectories}")

        torch.manual_seed(traj_idx)
        x_init = torch.randn(1, 1, 28, 28, device=device)
        z_init = x_init.reshape(-1).cpu()
        z_init_list.append(z_init)

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

        trajectory_flat = trajectory.reshape(num_steps, -1).cpu().numpy()
        all_trajectories.append(trajectory_flat)

    return all_trajectories, z_init_list, time_grid


def fit_pca_on_all_data(
    all_trajectories: list,
    z_init_list: list,
    mnist_images: np.ndarray,
    n_components: int = 2
) -> PCA:
    """Fit PCA on trajectories + MNIST to use same coordinate system."""
    all_states = []

    # Add trajectory states
    for traj in all_trajectories:
        all_states.extend(traj)

    # Add initial noise
    z_init_array = np.array([z.numpy() for z in z_init_list])
    all_states.extend(z_init_array)

    # Add MNIST
    all_states.extend(mnist_images)

    all_states = np.array(all_states)
    print(f"Fitting PCA on {len(all_states)} total states (trajectories + MNIST)...")

    pca = PCA(n_components=n_components)
    pca.fit(all_states)

    return pca


def plot_flow_field_with_mnist(
    all_trajectories: list,
    z_init_list: list,
    mnist_images: np.ndarray,
    mnist_labels: np.ndarray,
    time_grid: torch.Tensor,
    pca: PCA,
    save_path: str = 'flow_field_with_mnist.png'
) -> None:
    """
    Plot trajectories overlaid with ground truth MNIST digit clusters.

    Args:
        all_trajectories: List of [num_steps, 784] arrays
        z_init_list: List of initial noise vectors
        mnist_images: [N, 784] MNIST test images
        mnist_labels: [N] digit labels 0-9
        time_grid: Timesteps
        pca: Fitted PCA transformer
        save_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(16, 13))

    # Color map for digits
    digit_colors = {
        0: '#e74c3c',  # red
        1: '#3498db',  # blue
        2: '#2ecc71',  # green
        3: '#f39c12',  # orange
        4: '#9b59b6',  # purple
        5: '#1abc9c',  # teal
        6: '#e67e22',  # dark orange
        7: '#34495e',  # dark gray
        8: '#95a5a6',  # light gray
        9: '#c0392b',  # dark red
    }

    # Project MNIST images by digit
    print("Projecting MNIST images to PCA space...")
    mnist_proj = pca.transform(mnist_images)

    for digit in range(10):
        digit_mask = mnist_labels == digit
        digit_points = mnist_proj[digit_mask]

        ax.scatter(digit_points[:, 0], digit_points[:, 1],
                  c=digit_colors[digit], s=30, alpha=0.4,
                  edgecolors='none', label=f'Digit {digit}')

    # Project and draw trajectories
    print("Drawing trajectories...")
    z_init_array = np.array([z.numpy() for z in z_init_list])
    z_init_proj = pca.transform(z_init_array)

    z_final_list = [traj[-1] for traj in all_trajectories]
    z_final_array = np.array(z_final_list)
    z_final_proj = pca.transform(z_final_array)

    # Draw trajectory lines
    num_traj = len(all_trajectories)
    colors_traj = plt.cm.Greens(np.linspace(0.3, 0.8, num_traj))

    for traj_idx, traj in enumerate(all_trajectories):
        traj_proj = pca.transform(traj)
        ax.plot(traj_proj[:, 0], traj_proj[:, 1],
               color=colors_traj[traj_idx], alpha=0.3, linewidth=1.2, zorder=2)

    # Plot trajectory endpoints
    print("Marking trajectory endpoints...")
    ax.scatter(z_init_proj[:, 0], z_init_proj[:, 1],
              c='blue', s=80, marker='o', alpha=0.7,
              edgecolors='darkblue', linewidth=2,
              label='z(0) - Noise Start', zorder=10)

    ax.scatter(z_final_proj[:, 0], z_final_proj[:, 1],
              c='black', s=80, marker='X', alpha=0.7,
              edgecolors='darkgray', linewidth=2,
              label='z(1) - Generated End', zorder=10)

    # Add flow arrows
    print("Adding flow arrows...")
    for traj_idx in range(0, num_traj, max(1, num_traj // 12)):
        traj = all_trajectories[traj_idx]
        traj_proj = pca.transform(traj)

        num_arrows = 3
        arrow_indices = np.linspace(0, len(traj_proj) - 1, num_arrows + 2, dtype=int)[1:-1]

        for arr_idx in arrow_indices:
            if arr_idx > 0 and arr_idx < len(traj_proj) - 1:
                start = traj_proj[arr_idx - 1]
                end = traj_proj[arr_idx]
                dx = end[0] - start[0]
                dy = end[1] - start[1]

                ax.arrow(start[0], start[1], dx * 0.7, dy * 0.7,
                        head_width=0.4, head_length=0.3,
                        fc='olive', ec='olive', alpha=0.4, linewidth=1)

    # Formatting
    var1 = pca.explained_variance_ratio_[0]
    var2 = pca.explained_variance_ratio_[1]

    ax.set_xlabel(f'PC1 ({var1:.1%})', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'PC2 ({var2:.1%})', fontsize=13, fontweight='bold')

    title = ('Flow Matching: Learned Trajectories Aligned with MNIST Ground Truth\n' +
             'Pastel: MNIST digits by class | Green: Generated trajectories | ' +
             'Blue circles: noise start | Black X: generated end')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.2, linestyle='--')

    # Create custom legend
    digit_patches = [Patch(facecolor=digit_colors[d], alpha=0.6, label=f'Digit {d}')
                     for d in range(10)]
    trajectory_patches = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                   markersize=10, label='z(0) Noise Start', markeredgecolor='darkblue', markeredgewidth=2),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='black',
                   markersize=10, label='z(1) Generated End', markeredgecolor='darkgray', markeredgewidth=2),
        plt.Line2D([0], [0], color='olive', linewidth=2, alpha=0.6, label='Flow Trajectory'),
    ]

    ax.legend(handles=digit_patches + trajectory_patches,
             loc='upper left', fontsize=11, ncol=2, framealpha=0.95)

    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Visualization saved to {save_path}")
    plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Flow field with MNIST ground truth')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--num-trajectories', type=int, default=100,
                       help='Number of generated trajectories')
    parser.add_argument('--num-steps', type=int, default=50, help='Steps per trajectory')
    parser.add_argument('--mnist-samples', type=int, default=500,
                       help='Samples per digit from MNIST')
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

    # Load MNIST
    print("[MNIST] Loading ground truth dataset...")
    mnist_images, mnist_labels = load_mnist_dataset(num_samples_per_class=args.mnist_samples)
    print()

    # Capture trajectories
    all_trajectories, z_init_list, time_grid = capture_full_trajectories(
        model,
        num_trajectories=args.num_trajectories,
        num_steps=args.num_steps,
        device=str(device)
    )

    # Fit PCA on all data
    print()
    pca = fit_pca_on_all_data(all_trajectories, z_init_list, mnist_images, n_components=2)
    print(f"PCA variance: PC1={pca.explained_variance_ratio_[0]:.1%}, " +
          f"PC2={pca.explained_variance_ratio_[1]:.1%}\n")

    # Create visualization
    print("="*80)
    print("FLOW FIELD WITH MNIST GROUND TRUTH")
    print("="*80 + "\n")

    plot_flow_field_with_mnist(
        all_trajectories, z_init_list, mnist_images, mnist_labels,
        time_grid, pca,
        save_path=f'{args.output_dir}/flow_field_with_mnist.png'
    )

    print(f"\n[SUCCESS] Visualization saved to {args.output_dir}/flow_field_with_mnist.png")
