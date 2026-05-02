#!/usr/bin/env python
"""
Simplified LDA visualization: Static 2D and 3D plots at key time points
"""
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pathlib import Path
from models.fm import ImageFlowMatcher
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torchvision import datasets, transforms
from flow_matching.solver import ODESolver
from tqdm import tqdm

def load_data():
    """Load model and MNIST data."""
    ckpt_path = Path("outputs/fm/2026-05-01/22-58-12/training_logs/version_0/checkpoints/fm-balanced-epoch=014-train_loss=0.1829.ckpt")
    print("Loading model...")
    model = ImageFlowMatcher.load_from_checkpoint(str(ckpt_path))
    model.eval()
    device = torch.device('cpu')
    model = model.to(device)

    print("Loading MNIST data...")
    transform = transforms.ToTensor()
    test_data = datasets.MNIST(root='./data', train=False, transform=transform, download=False)
    indices = np.random.choice(len(test_data), 400, replace=False)

    real_images = torch.stack([test_data[i][0] for i in indices])
    real_labels = np.array([test_data[i][1] for i in indices])

    return model, device, real_images, real_labels

def generate_at_t(model, device, t, num_samples=150):
    """Generate samples at specific time t."""
    samples = []
    with torch.no_grad():
        for i in range(0, num_samples, 32):
            batch_size = min(32, num_samples - i)
            x_t = torch.randn(batch_size, 1, 28, 28, device=device)

            num_steps = max(2, int(50 * (1 - t)))
            if t >= 0.99:
                t_use = 0.98
            else:
                t_use = t
            time_grid = torch.linspace(t_use, 1.0, num_steps, device=device)
            time_grid = torch.unique(time_grid, sorted=True)

            solver = ODESolver(velocity_model=model.model)
            generated = solver.sample(
                time_grid=time_grid, x_init=x_t, method='dopri5',
                step_size=None, atol=1e-4, rtol=1e-4
            )

            if model.normalize_data:
                generated = (generated + 1) / 2
                generated = torch.clamp(generated, 0, 1)

            samples.append(generated)

    return torch.cat(samples, dim=0)

def fit_lda(real_images, real_labels, gen_images):
    """Fit LDA models."""
    real_flat = real_images.reshape(len(real_images), -1).numpy()
    gen_flat = gen_images.reshape(len(gen_images), -1).numpy()

    lda_digit = LinearDiscriminantAnalysis(n_components=2)
    lda_digit.fit(real_flat, real_labels)

    lda_realfake = LinearDiscriminantAnalysis(n_components=1)
    combined = np.vstack([real_flat, gen_flat])
    labels_rf = np.hstack([np.zeros(len(real_flat)), np.ones(len(gen_flat))])
    lda_realfake.fit(combined, labels_rf)

    return lda_digit, lda_realfake

def transform_all(images, labels, lda_digit, lda_realfake):
    """Transform to LDA space."""
    flat = images.reshape(len(images), -1).numpy()
    lda_2d = lda_digit.transform(flat)
    lda_1d = lda_realfake.transform(flat)
    return np.hstack([lda_2d, lda_1d])

def create_visualizations(real_images, real_labels, model, device):
    """Create 2D and 3D static visualizations."""
    print("\nGenerating samples at key time points...")
    gen_t0 = generate_at_t(model, device, 0.0, 150)
    gen_t05 = generate_at_t(model, device, 0.5, 150)
    gen_t1 = generate_at_t(model, device, 0.99, 150)

    print("Fitting LDA models...")
    lda_digit, lda_realfake = fit_lda(real_images, real_labels, gen_t1)

    print("Transforming to LDA space...")
    real_3d = transform_all(real_images, real_labels, lda_digit, lda_realfake)
    gen_t0_3d = transform_all(gen_t0, None, lda_digit, lda_realfake)
    gen_t05_3d = transform_all(gen_t05, None, lda_digit, lda_realfake)
    gen_t1_3d = transform_all(gen_t1, None, lda_digit, lda_realfake)

    # Create 2D visualization
    print("Creating 2D visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('2D LDA: Digit Discrimination Across Time', fontsize=16, fontweight='bold')

    # Ground truth
    ax = axes[0, 0]
    for digit in range(10):
        mask = real_labels == digit
        ax.scatter(real_3d[mask, 0], real_3d[mask, 1], alpha=0.6, s=40, label=f'{digit}', c=[digit], cmap='tab10')
    ax.set_xlabel('LDA Component 1', fontweight='bold')
    ax.set_ylabel('LDA Component 2', fontweight='bold')
    ax.set_title('Ground Truth (Real MNIST)', fontweight='bold', fontsize=12)
    ax.legend(loc='best', ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    # t=0
    ax = axes[0, 1]
    for digit in range(10):
        mask = real_labels == digit
        ax.scatter(real_3d[mask, 0], real_3d[mask, 1], alpha=0.2, s=20, c=[digit], cmap='tab10')
    ax.scatter(gen_t0_3d[:, 0], gen_t0_3d[:, 1], c='red', alpha=0.8, s=100, marker='x', linewidth=2)
    ax.set_xlabel('LDA Component 1', fontweight='bold')
    ax.set_ylabel('LDA Component 2', fontweight='bold')
    ax.set_title('Generated at t=0 (Pure Noise)', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)

    # t=0.5
    ax = axes[1, 0]
    for digit in range(10):
        mask = real_labels == digit
        ax.scatter(real_3d[mask, 0], real_3d[mask, 1], alpha=0.2, s=20, c=[digit], cmap='tab10')
    ax.scatter(gen_t05_3d[:, 0], gen_t05_3d[:, 1], c='orange', alpha=0.8, s=100, marker='x', linewidth=2)
    ax.set_xlabel('LDA Component 1', fontweight='bold')
    ax.set_ylabel('LDA Component 2', fontweight='bold')
    ax.set_title('Generated at t=0.5 (Intermediate)', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)

    # t=1.0
    ax = axes[1, 1]
    for digit in range(10):
        mask = real_labels == digit
        ax.scatter(real_3d[mask, 0], real_3d[mask, 1], alpha=0.3, s=30, label=f'{digit}', c=[digit], cmap='tab10')
    ax.scatter(gen_t1_3d[:, 0], gen_t1_3d[:, 1], c='green', alpha=0.8, s=100, marker='x', linewidth=2)
    ax.set_xlabel('LDA Component 1', fontweight='bold')
    ax.set_ylabel('LDA Component 2', fontweight='bold')
    ax.set_title('Generated at t=0.99 (Data Distribution)', fontweight='bold', fontsize=12)
    ax.legend(loc='best', ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path("results").mkdir(exist_ok=True)
    plt.savefig("results/lda_2d_timeline.png", dpi=150, bbox_inches='tight')
    print("Saved: results/lda_2d_timeline.png")
    plt.close()

    # Create 3D visualization
    print("Creating 3D visualization...")
    fig = plt.figure(figsize=(20, 15))

    titles = [
        ('Ground Truth\n(Real MNIST)', real_3d, None, 'blue'),
        ('Generated at t=0\n(Pure Noise)', gen_t0_3d, real_3d, 'red'),
        ('Generated at t=0.5\n(Intermediate)', gen_t05_3d, real_3d, 'orange'),
        ('Generated at t=0.99\n(Data Distribution)', gen_t1_3d, real_3d, 'green'),
    ]

    for idx, (title, gen_data, ref_data, color) in enumerate(titles, 1):
        ax = fig.add_subplot(2, 2, idx, projection='3d')

        if ref_data is not None:
            # Plot reference in background
            for digit in range(10):
                mask = real_labels == digit
                ax.scatter(ref_data[mask, 0], ref_data[mask, 1], ref_data[mask, 2],
                          alpha=0.1, s=10, c=[digit], cmap='tab10')
            # Plot generated
            ax.scatter(gen_data[:, 0], gen_data[:, 1], gen_data[:, 2],
                      c=color, alpha=0.8, s=80, marker='x', linewidth=2)
        else:
            # Plot ground truth
            for digit in range(10):
                mask = real_labels == digit
                ax.scatter(gen_data[mask, 0], gen_data[mask, 1], gen_data[mask, 2],
                          alpha=0.6, s=40, c=[digit], cmap='tab10', label=f'{digit}')
            ax.legend(loc='best', fontsize=7, ncol=2)

        ax.set_xlabel('LDA 1\n(Digit)', fontweight='bold', fontsize=10)
        ax.set_ylabel('LDA 2\n(Digit)', fontweight='bold', fontsize=10)
        ax.set_zlabel('LDA 3\n(Real/Fake)', fontweight='bold', fontsize=10)
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.view_init(elev=20, azim=45)

    fig.suptitle('3D LDA: Digit (X,Y) + Real/Fake (Z) Discrimination', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/lda_3d_timeline.png", dpi=150, bbox_inches='tight')
    print("Saved: results/lda_3d_timeline.png")
    plt.close()

def main():
    print("="*80)
    print("SIMPLIFIED LDA TRAJECTORY VISUALIZATION")
    print("="*80)
    print()

    model, device, real_images, real_labels = load_data()
    print()

    create_visualizations(real_images, real_labels, model, device)

    print()
    print("="*80)
    print("COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - results/lda_2d_timeline.png (4-panel 2D comparison)")
    print("  - results/lda_3d_timeline.png (4-panel 3D comparison)")

if __name__ == '__main__':
    main()
