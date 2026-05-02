"""
Improved Flow Field with LDA - Better visibility of digit clusters and trajectories
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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


def fit_lda_on_mnist(mnist_images: np.ndarray, mnist_labels: np.ndarray) -> LinearDiscriminantAnalysis:
    """Fit LDA on MNIST."""
    print(f"Fitting LDA on {len(mnist_images)} MNIST images...")
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(mnist_images, mnist_labels)
    return lda


def plot_flow_field_lda_improved(
    all_trajectories: list,
    z_init_list: list,
    mnist_images: np.ndarray,
    mnist_labels: np.ndarray,
    time_grid: torch.Tensor,
    lda: LinearDiscriminantAnalysis,
    save_path: str = 'flow_field_lda_improved.png'
) -> None:
    """
    Improved visualization with digit boundaries and better layering.
    """
    fig, ax = plt.subplots(figsize=(18, 14))

    digit_colors = {
        0: '#e74c3c', 1: '#3498db', 2: '#2ecc71', 3: '#f39c12', 4: '#9b59b6',
        5: '#1abc9c', 6: '#e67e22', 7: '#34495e', 8: '#95a5a6', 9: '#c0392b',
    }

    print("Projecting MNIST images...")
    mnist_proj = lda.transform(mnist_images)

    # Draw digit clusters with convex hulls as boundaries
    print("Drawing digit cluster boundaries...")
    for digit in range(10):
        digit_mask = mnist_labels == digit
        digit_points = mnist_proj[digit_mask]

        # Plot digit points
        ax.scatter(digit_points[:, 0], digit_points[:, 1],
                  c=digit_colors[digit], s=100, alpha=0.7,
                  edgecolors='darkgray', linewidth=0.3, label=f'Digit {digit}', zorder=5)

        # Draw convex hull boundary if enough points
        if len(digit_points) > 3:
            try:
                hull = ConvexHull(digit_points)
                hull_points = digit_points[hull.vertices]
                hull_polygon = Polygon(hull_points, fill=False,
                                      edgecolor=digit_colors[digit],
                                      linewidth=2, linestyle='--', alpha=0.4, zorder=4)
                ax.add_patch(hull_polygon)
            except:
                pass

    # Project trajectories
    print("Drawing trajectories...")
    z_init_array = np.array([z.numpy() for z in z_init_list])
    z_init_proj = lda.transform(z_init_array)

    z_final_list = [traj[-1] for traj in all_trajectories]
    z_final_array = np.array(z_final_list)
    z_final_proj = lda.transform(z_final_array)

    # Draw trajectory lines
    num_traj = len(all_trajectories)
    colors_traj = plt.cm.Greens(np.linspace(0.3, 0.85, num_traj))

    for traj_idx, traj in enumerate(all_trajectories):
        traj_proj = lda.transform(traj)
        ax.plot(traj_proj[:, 0], traj_proj[:, 1],
               color=colors_traj[traj_idx], alpha=0.4, linewidth=1.8, zorder=6)

    # Plot noise initialization (smaller, less visible)
    print("Marking noise points...")
    ax.scatter(z_init_proj[:, 0], z_init_proj[:, 1],
              c='lightblue', s=60, marker='o', alpha=0.3,
              edgecolors='blue', linewidth=1,
              label='z(0) - Noise Start', zorder=7)

    # Plot generated endpoints (prominent)
    print("Marking generated points...")
    ax.scatter(z_final_proj[:, 0], z_final_proj[:, 1],
              c='lime', s=150, marker='*', alpha=0.95,
              edgecolors='darkgreen', linewidth=2,
              label='z(1) - Generated End', zorder=15)

    # Add flow arrows
    print("Adding flow direction...")
    for traj_idx in range(0, num_traj, max(1, num_traj // 12)):
        traj = all_trajectories[traj_idx]
        traj_proj = lda.transform(traj)

        num_arrows = 4
        arrow_indices = np.linspace(0, len(traj_proj) - 1, num_arrows + 2, dtype=int)[1:-1]

        for arr_idx in arrow_indices:
            if arr_idx > 0 and arr_idx < len(traj_proj) - 1:
                start = traj_proj[arr_idx - 1]
                end = traj_proj[arr_idx]
                dx = end[0] - start[0]
                dy = end[1] - start[1]

                ax.arrow(start[0], start[1], dx * 0.5, dy * 0.5,
                        head_width=0.6, head_length=0.5,
                        fc='darkgreen', ec='darkgreen', alpha=0.5,
                        linewidth=1.5, zorder=8)

    # Formatting
    var1 = lda.explained_variance_ratio_[0]
    var2 = lda.explained_variance_ratio_[1]

    ax.set_xlabel(f'LDA1 ({var1:.1%})', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'LDA2 ({var2:.1%})', fontsize=14, fontweight='bold')

    title = ('Flow Matching: Trajectories Converging to MNIST Digit Manifold\n' +
             'Colored regions: MNIST digit clusters with boundaries | ' +
             'Green lines & ★: learned trajectories | Light blue○: noise initialization')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.15, linestyle=':', linewidth=0.8)

    # Legend with better organization
    digit_patches = [Patch(facecolor=digit_colors[d], alpha=0.7,
                          edgecolor='darkgray', label=f'Digit {d}')
                     for d in range(10)]
    trajectory_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
                   markersize=10, label='z(0) Noise', markeredgecolor='blue', markeredgewidth=1.5),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='lime',
                   markersize=15, label='z(1) Generated', markeredgecolor='darkgreen', markeredgewidth=2),
        plt.Line2D([0], [0], color='darkgreen', linewidth=2.5, alpha=0.6, label='Flow Trajectory'),
        plt.Line2D([0], [0], color='darkgray', linewidth=2, linestyle='--', alpha=0.4, label='Digit Boundary'),
    ]

    ax.legend(handles=digit_patches + trajectory_elements,
             loc='upper left', fontsize=10, ncol=2, framealpha=0.97,
             edgecolor='gray', fancybox=True)

    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Improved visualization saved to {save_path}")
    plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Improved flow field with LDA')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num-trajectories', type=int, default=100)
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--mnist-samples', type=int, default=500)
    parser.add_argument('--output-dir', type=str, default='results/flow_field')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

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

    print("[MNIST] Loading dataset...")
    mnist_images, mnist_labels = load_mnist_dataset(num_samples_per_class=args.mnist_samples)
    print()

    print("[LDA] Fitting discriminant analysis...")
    lda = fit_lda_on_mnist(mnist_images, mnist_labels)
    print(f"LDA variance: {np.sum(lda.explained_variance_ratio_):.1%}\n")

    print()
    all_trajectories, z_init_list, time_grid = capture_full_trajectories(
        model, args.num_trajectories, args.num_steps, str(device)
    )

    print("\n" + "="*80)
    print("IMPROVED FLOW FIELD VISUALIZATION")
    print("="*80 + "\n")

    plot_flow_field_lda_improved(
        all_trajectories, z_init_list, mnist_images, mnist_labels,
        time_grid, lda,
        save_path=f'{args.output_dir}/flow_field_lda_improved.png'
    )

    print(f"\n[SUCCESS] Saved to {args.output_dir}/flow_field_lda_improved.png")
