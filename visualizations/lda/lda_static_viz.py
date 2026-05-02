#!/usr/bin/env python
"""
Static 2D and 3D LDA visualizations (no animations, fast generation)
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

def load_model_and_data():
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
    indices = np.random.choice(len(test_data), 500, replace=False)

    images = [test_data[i][0] for i in indices]
    labels = np.array([test_data[i][1] for i in indices])

    real_images = torch.stack(images)
    return model, device, real_images, labels

def generate_samples(model, device, t=1.0, num_samples=150):
    """Generate samples at time t."""
    print(f"Generating {num_samples} samples at t={t}...")

    samples = []
    with torch.no_grad():
        for i in range(0, num_samples, 32):
            batch_size = min(32, num_samples - i)
            x_t = torch.randn(batch_size, 1, 28, 28, device=device)

            num_steps = max(5, int(50 * (1 - t)))
            time_grid = torch.linspace(t, 1, num_steps, device=device)

            from flow_matching.solver import ODESolver
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

def create_2d_viz(real_images, real_labels, gen_images_t0, gen_images_t05, gen_images_t1):
    """Create 2D LDA visualization."""
    print("\nCreating 2D visualization...")
    Path("results").mkdir(exist_ok=True)

    real_flat = real_images.reshape(len(real_images), -1).numpy()

    # Fit LDA on digit classification
    lda_digit = LinearDiscriminantAnalysis(n_components=2)
    lda_digit.fit(real_flat, real_labels)

    # Project all data
    real_2d = lda_digit.transform(real_flat)
    gen_t0_2d = lda_digit.transform(gen_images_t0.reshape(len(gen_images_t0), -1).numpy())
    gen_t05_2d = lda_digit.transform(gen_images_t05.reshape(len(gen_images_t05), -1).numpy())
    gen_t1_2d = lda_digit.transform(gen_images_t1.reshape(len(gen_images_t1), -1).numpy())

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('2D LDA: Digit Discrimination Across Time Points', fontsize=16, fontweight='bold')

    # Plot 1: Ground truth
    ax = axes[0, 0]
    for digit in range(10):
        mask = real_labels == digit
        ax.scatter(real_2d[mask, 0], real_2d[mask, 1], alpha=0.6, s=50, label=f'Digit {digit}', c=[digit], cmap='tab10')
    ax.set_xlabel('LDA Component 1', fontweight='bold')
    ax.set_ylabel('LDA Component 2', fontweight='bold')
    ax.set_title('Ground Truth (Real MNIST)', fontweight='bold')
    ax.legend(loc='best', ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Generated at t=0
    ax = axes[0, 1]
    for digit in range(10):
        ax.scatter([], [], alpha=0.6, s=50, label=f'Digit {digit}', c=[digit], cmap='tab10')
    ax.scatter(gen_t0_2d[:, 0], gen_t0_2d[:, 1], c='red', alpha=0.8, s=100, marker='x', linewidth=2)
    ax.scatter([], [], c='red', alpha=0.8, s=100, marker='x', linewidth=2, label='Generated (t=0)')
    ax.set_xlabel('LDA Component 1', fontweight='bold')
    ax.set_ylabel('LDA Component 2', fontweight='bold')
    ax.set_title('Generated Samples at t=0 (Pure Noise)', fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 3: Generated at t=0.5
    ax = axes[1, 0]
    for digit in range(10):
        mask = real_labels == digit
        ax.scatter(real_2d[mask, 0], real_2d[mask, 1], alpha=0.2, s=20, c=[digit], cmap='tab10')
    ax.scatter(gen_t05_2d[:, 0], gen_t05_2d[:, 1], c='orange', alpha=0.8, s=100, marker='x', linewidth=2)
    ax.scatter([], [], c='orange', alpha=0.8, s=100, marker='x', linewidth=2, label='Generated (t=0.5)')
    for digit in range(10):
        ax.scatter([], [], alpha=0.2, s=20, label=f'Real {digit}', c=[digit], cmap='tab10')
    ax.set_xlabel('LDA Component 1', fontweight='bold')
    ax.set_ylabel('LDA Component 2', fontweight='bold')
    ax.set_title('Generated Samples at t=0.5 (Intermediate)', fontweight='bold')
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Plot 4: Generated at t=1.0
    ax = axes[1, 1]
    for digit in range(10):
        mask = real_labels == digit
        ax.scatter(real_2d[mask, 0], real_2d[mask, 1], alpha=0.3, s=30, label=f'Digit {digit}', c=[digit], cmap='tab10')
    ax.scatter(gen_t1_2d[:, 0], gen_t1_2d[:, 1], c='green', alpha=0.8, s=100, marker='x', linewidth=2)
    ax.scatter([], [], c='green', alpha=0.8, s=100, marker='x', linewidth=2, label='Generated (t=1)')
    ax.set_xlabel('LDA Component 1', fontweight='bold')
    ax.set_ylabel('LDA Component 2', fontweight='bold')
    ax.set_title('Generated Samples at t=1.0 (Data Distribution)', fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/lda_2d_static.png", dpi=150, bbox_inches='tight')
    print("Saved: results/lda_2d_static.png")
    plt.close()

    return lda_digit

def create_3d_viz(real_images, real_labels, gen_images_t0, gen_images_t05, gen_images_t1, lda_digit):
    """Create 3D LDA visualization with real/fake discrimination."""
    print("\nCreating 3D visualization...")

    real_flat = real_images.reshape(len(real_images), -1).numpy()
    gen_t0_flat = gen_images_t0.reshape(len(gen_images_t0), -1).numpy()
    gen_t1_flat = gen_images_t1.reshape(len(gen_images_t1), -1).numpy()

    # Fit real/fake LDA
    lda_realfake = LinearDiscriminantAnalysis(n_components=1)
    combined = np.vstack([real_flat, gen_t1_flat])
    labels_rf = np.hstack([np.zeros(len(real_flat)), np.ones(len(gen_t1_flat))])
    lda_realfake.fit(combined, labels_rf)

    # Project all
    real_2d = lda_digit.transform(real_flat)
    real_1d = lda_realfake.transform(real_flat)
    real_3d = np.hstack([real_2d, real_1d])

    gen_t0_2d = lda_digit.transform(gen_t0_flat)
    gen_t0_1d = lda_realfake.transform(gen_t0_flat)
    gen_t0_3d = np.hstack([gen_t0_2d, gen_t0_1d])

    gen_t05_flat = gen_images_t05.reshape(len(gen_images_t05), -1).numpy()
    gen_t05_2d = lda_digit.transform(gen_t05_flat)
    gen_t05_1d = lda_realfake.transform(gen_t05_flat)
    gen_t05_3d = np.hstack([gen_t05_2d, gen_t05_1d])

    gen_t1_2d = lda_digit.transform(gen_t1_flat)
    gen_t1_1d = lda_realfake.transform(gen_t1_flat)
    gen_t1_3d = np.hstack([gen_t1_2d, gen_t1_1d])

    fig = plt.figure(figsize=(18, 12))

    for plot_idx, (title, gen_3d, gen_t, color) in enumerate([
        ('at t=0 (Pure Noise)', gen_t0_3d, 0.0, 'red'),
        ('at t=0.5 (Intermediate)', gen_t05_3d, 0.5, 'orange'),
        ('at t=1.0 (Data)', gen_t1_3d, 1.0, 'green'),
    ]):
        ax = fig.add_subplot(2, 2, plot_idx + 1, projection='3d')

        # Plot real data
        for digit in range(10):
            mask = real_labels == digit
            ax.scatter(real_3d[mask, 0], real_3d[mask, 1], real_3d[mask, 2],
                      alpha=0.2, s=20, c=[digit], cmap='tab10', label=f'Real {digit}')

        # Plot generated
        ax.scatter(gen_3d[:, 0], gen_3d[:, 1], gen_3d[:, 2],
                  c=color, alpha=0.9, s=80, marker='x', linewidth=2)

        ax.set_xlabel('LDA 1: Digit', fontweight='bold', fontsize=10)
        ax.set_ylabel('LDA 2: Digit', fontweight='bold', fontsize=10)
        ax.set_zlabel('LDA 3: Real/Fake', fontweight='bold', fontsize=10)
        ax.set_title(f'Generated Samples {title}\n(Color = digit class, Red/Orange/Green = time)', fontweight='bold', fontsize=11)
        ax.view_init(elev=20, azim=45)

    # Panel 4: Summary with all three
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    for digit in range(10):
        mask = real_labels == digit
        ax.scatter(real_3d[mask, 0], real_3d[mask, 1], real_3d[mask, 2],
                  alpha=0.15, s=15, c=[digit], cmap='tab10')

    ax.scatter(gen_t0_3d[:, 0], gen_t0_3d[:, 1], gen_t0_3d[:, 2],
              c='red', alpha=0.6, s=50, marker='x', linewidth=2, label='t=0')
    ax.scatter(gen_t05_3d[:, 0], gen_t05_3d[:, 1], gen_t05_3d[:, 2],
              c='orange', alpha=0.6, s=50, marker='x', linewidth=2, label='t=0.5')
    ax.scatter(gen_t1_3d[:, 0], gen_t1_3d[:, 1], gen_t1_3d[:, 2],
              c='green', alpha=0.6, s=50, marker='x', linewidth=2, label='t=1.0')

    ax.set_xlabel('LDA 1: Digit', fontweight='bold', fontsize=10)
    ax.set_ylabel('LDA 2: Digit', fontweight='bold', fontsize=10)
    ax.set_zlabel('LDA 3: Real/Fake', fontweight='bold', fontsize=10)
    ax.set_title('All Time Points Combined\n(Red→Orange→Green shows trajectory)', fontweight='bold', fontsize=11)
    ax.legend(loc='best', fontsize=10)
    ax.view_init(elev=20, azim=45)

    fig.suptitle('3D LDA: Digit Discrimination (X,Y) + Real/Fake (Z)', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig("results/lda_3d_static.png", dpi=150, bbox_inches='tight')
    print("Saved: results/lda_3d_static.png")
    plt.close()

def main():
    print("="*80)
    print("STATIC LDA VISUALIZATION (2D and 3D)")
    print("="*80)
    print()

    model, device, real_images, real_labels = load_model_and_data()
    print()

    print("[1/4] Generating samples at t=0...")
    gen_t0 = generate_samples(model, device, t=0.0, num_samples=150)

    print("[2/4] Generating samples at t=0.5...")
    gen_t05 = generate_samples(model, device, t=0.5, num_samples=150)

    print("[3/4] Generating samples at t=1.0...")
    gen_t1 = generate_samples(model, device, t=1.0, num_samples=150)

    print()
    print("[4/4] Creating visualizations...")

    lda_digit = create_2d_viz(real_images, real_labels, gen_t0, gen_t05, gen_t1)
    create_3d_viz(real_images, real_labels, gen_t0, gen_t05, gen_t1, lda_digit)

    print()
    print("="*80)
    print("COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - results/lda_2d_static.png")
    print("  - results/lda_3d_static.png")

if __name__ == '__main__':
    main()
