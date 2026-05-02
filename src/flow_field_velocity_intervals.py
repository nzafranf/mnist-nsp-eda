"""
Flow Field with Velocity Vectors at Different Time Intervals
Shows arrows from t_i to t_{i+1} to visualize velocity magnitude and direction
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


def plot_flow_field_velocity_interval(
    all_trajectories: list,
    mnist_images: np.ndarray,
    mnist_labels: np.ndarray,
    time_grid: torch.Tensor,
    lda: LinearDiscriminantAnalysis,
    start_time: float = 0.0,
    save_path: str = 'flow_field_velocity.png'
) -> None:
    """
    Plot velocity vectors (arrows from t_i to t_{i+1}) for a specific time interval.

    Args:
        all_trajectories: List of [num_steps, 784] arrays
        mnist_images: [N, 784] MNIST test images
        mnist_labels: [N] digit labels 0-9
        time_grid: Timesteps
        lda: Fitted LDA transformer
        start_time: Start time for this interval (0.0 to 0.9)
        save_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(14, 12))

    digit_colors = {
        0: '#e74c3c', 1: '#3498db', 2: '#2ecc71', 3: '#f39c12', 4: '#9b59b6',
        5: '#1abc9c', 6: '#e67e22', 7: '#34495e', 8: '#95a5a6', 9: '#c0392b',
    }

    # Project MNIST images
    print(f"  Projecting MNIST images...")
    mnist_proj = lda.transform(mnist_images)

    # Draw MNIST digit clusters with convex hulls
    for digit in range(10):
        digit_mask = mnist_labels == digit
        digit_points = mnist_proj[digit_mask]

        ax.scatter(digit_points[:, 0], digit_points[:, 1],
                  c=digit_colors[digit], s=100, alpha=0.7,
                  edgecolors='darkgray', linewidth=0.3, label=f'Digit {digit}', zorder=5)

        # Draw convex hull boundary
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

    # Find indices corresponding to start_time
    time_array = time_grid.cpu().numpy()
    start_idx = np.argmin(np.abs(time_array - start_time))

    # Draw velocity vectors (arrows from t_i to t_{i+1})
    print(f"  Drawing velocity vectors from t={start_time:.1f}...")
    num_traj = len(all_trajectories)
    colors_traj = plt.cm.Greens(np.linspace(0.3, 0.85, num_traj))

    for traj_idx, traj in enumerate(all_trajectories):
        # Only use points from start_idx onward
        traj_subset = traj[start_idx:]
        traj_proj = lda.transform(traj_subset)

        # Draw arrows between consecutive points
        for i in range(len(traj_proj) - 1):
            start = traj_proj[i]
            end = traj_proj[i + 1]
            dx = end[0] - start[0]
            dy = end[1] - start[1]

            # Magnitude of velocity
            magnitude = np.sqrt(dx**2 + dy**2)

            if magnitude > 1e-6:  # Only draw if there's movement
                ax.arrow(start[0], start[1], dx, dy,
                        head_width=0.3, head_length=0.2,
                        fc=colors_traj[traj_idx], ec=colors_traj[traj_idx],
                        alpha=0.6, linewidth=1.2, zorder=6)

    # Plot endpoints within this interval
    print(f"  Marking trajectory states...")
    for traj_idx, traj in enumerate(all_trajectories):
        traj_subset = traj[start_idx:]
        traj_proj = lda.transform(traj_subset)

        # Mark start of this interval
        if start_idx > 0:
            ax.scatter(traj_proj[0, 0], traj_proj[0, 1],
                      c=colors_traj[traj_idx], s=80, marker='o', alpha=0.8,
                      edgecolors='black', linewidth=1.5, zorder=8)

        # Mark end of interval
        ax.scatter(traj_proj[-1, 0], traj_proj[-1, 1],
                  c=colors_traj[traj_idx], s=120, marker='*', alpha=0.95,
                  edgecolors='darkgreen', linewidth=2, zorder=15)

    # Formatting
    var1 = lda.explained_variance_ratio_[0]
    var2 = lda.explained_variance_ratio_[1]

    ax.set_xlabel(f'LDA1 ({var1:.1%})', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'LDA2 ({var2:.1%})', fontsize=14, fontweight='bold')

    title = (f'Flow Matching: Velocity Vectors [t={start_time:.1f}, t=1.0]\n' +
             f'Arrows show movement from t_i to t_{{i+1}} | Arrow length = velocity magnitude')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.15, linestyle=':', linewidth=0.8)

    # Legend
    digit_patches = [Patch(facecolor=digit_colors[d], alpha=0.7,
                          edgecolor='darkgray', label=f'Digit {d}')
                     for d in range(10)]
    trajectory_elements = [
        plt.Line2D([0], [0], color='darkgreen', linewidth=2.5, alpha=0.6, label='Velocity Vector'),
        plt.Line2D([0], [0], color='darkgray', linewidth=2, linestyle='--', alpha=0.4, label='Digit Boundary'),
    ]

    ax.legend(handles=digit_patches + trajectory_elements,
             loc='upper left', fontsize=9, ncol=2, framealpha=0.97,
             edgecolor='gray', fancybox=True)

    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"    Saved to {save_path}")
    plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Flow field velocity vectors at different time intervals')
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
    print("VELOCITY VECTOR VISUALIZATION AT DIFFERENT TIME INTERVALS")
    print("="*80 + "\n")

    # Create 10 visualizations for different time intervals
    time_intervals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for start_time in time_intervals:
        print(f"Creating visualization for interval [t={start_time:.1f}, t=1.0]...")
        plot_flow_field_velocity_interval(
            all_trajectories, mnist_images, mnist_labels,
            time_grid, lda,
            start_time=start_time,
            save_path=f'{args.output_dir}/velocity_interval_t{int(start_time*10):02d}_to_10.png'
        )

    print(f"\n[SUCCESS] All 10 visualizations saved to {args.output_dir}")
    print(f"\nGenerated files:")
    for i in range(10):
        print(f"  - velocity_interval_t{i:02d}_to_10.png  (interval [{i/10:.1f}, 1.0])")
