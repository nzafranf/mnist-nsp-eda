#!/usr/bin/env python
"""
Generate 51 individual LDA visualization frames (t=0/50 to t=50/50)
Each frame shows trajectory evolution at that specific time point
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
    print("Loading model and MNIST data...")
    model = ImageFlowMatcher.load_from_checkpoint(str(ckpt_path))
    model.eval()
    device = torch.device('cpu')
    model = model.to(device)

    transform = transforms.ToTensor()
    test_data = datasets.MNIST(root='./data', train=False, transform=transform, download=False)
    indices = np.random.choice(len(test_data), 300, replace=False)

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
    """Fit LDA models on ground truth and final generated."""
    real_flat = real_images.reshape(len(real_images), -1).numpy()
    gen_flat = gen_images.reshape(len(gen_images), -1).numpy()

    lda_digit = LinearDiscriminantAnalysis(n_components=2)
    lda_digit.fit(real_flat, real_labels)

    lda_realfake = LinearDiscriminantAnalysis(n_components=1)
    combined = np.vstack([real_flat, gen_flat])
    labels_rf = np.hstack([np.zeros(len(real_flat)), np.ones(len(gen_flat))])
    lda_realfake.fit(combined, labels_rf)

    return lda_digit, lda_realfake

def transform_to_3d(images, lda_digit, lda_realfake):
    """Transform images to 3D LDA space."""
    flat = images.reshape(len(images), -1).numpy()
    lda_2d = lda_digit.transform(flat)
    lda_1d = lda_realfake.transform(flat)
    return np.hstack([lda_2d, lda_1d])

def create_frame_sequence(real_images, real_labels, model, device):
    """Create 51 individual frame images (t=0 to t=50 in steps of 1)."""

    print("\nGenerating reference data for LDA fitting...")
    gen_final = generate_at_t(model, device, 0.99, 150)

    print("Fitting LDA models...")
    lda_digit, lda_realfake = fit_lda(real_images, real_labels, gen_final)

    print("Transforming ground truth to LDA space...")
    real_3d = transform_to_3d(real_images, lda_digit, lda_realfake)

    # Create output directory
    frame_dir = Path("results/lda_frames")
    frame_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating 51 frames (t=0/50 to t=50/50)...")

    # Generate 51 frames
    for frame_idx in tqdm(range(51), desc="Frames"):
        # t normalized: 0 to 50 means t from 0.0 to 1.0
        t = frame_idx / 50.0

        # Generate samples at this time point
        gen_data = generate_at_t(model, device, t, 150)
        gen_3d = transform_to_3d(gen_data, lda_digit, lda_realfake)

        # Create both 2D and 3D visualizations for this frame

        # ===== 2D Plot =====
        fig, ax = plt.subplots(figsize=(14, 11))

        # Plot ground truth in background
        cmap = plt.cm.get_cmap('tab10')
        for digit in range(10):
            mask = real_labels == digit
            ax.scatter(real_3d[mask, 0], real_3d[mask, 1],
                      alpha=0.3, s=30, label=f'Real {digit}',
                      color=cmap(digit / 10.0))

        # Plot generated at this time point
        ax.scatter(gen_3d[:, 0], gen_3d[:, 1],
                  c='red', alpha=0.85, s=100, marker='x', linewidth=2.5,
                  label=f'Generated (t={t:.2f})')

        ax.set_xlabel('LDA Component 1 (Digit Discrimination)', fontweight='bold', fontsize=12)
        ax.set_ylabel('LDA Component 2 (Digit Discrimination)', fontweight='bold', fontsize=12)
        ax.set_title(f'2D Trajectory at Frame {frame_idx:02d}/50 (t={frame_idx}/50 = {t:.2f})',
                    fontweight='bold', fontsize=13)
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(frame_dir / f"frame_2d_{frame_idx:02d}.png", dpi=120, bbox_inches='tight')
        plt.close()

        # ===== 3D Plot =====
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot ground truth in background
        cmap = plt.cm.get_cmap('tab10')
        for digit in range(10):
            mask = real_labels == digit
            ax.scatter(real_3d[mask, 0], real_3d[mask, 1], real_3d[mask, 2],
                      alpha=0.15, s=20, color=cmap(digit / 10.0))

        # Plot generated at this time point
        ax.scatter(gen_3d[:, 0], gen_3d[:, 1], gen_3d[:, 2],
                  c='red', alpha=0.9, s=100, marker='x', linewidth=2.5,
                  label=f't={frame_idx}/50 ({t:.2f})')

        ax.set_xlabel('LDA 1: Digit', fontweight='bold', fontsize=11)
        ax.set_ylabel('LDA 2: Digit', fontweight='bold', fontsize=11)
        ax.set_zlabel('LDA 3: Real/Fake', fontweight='bold', fontsize=11)
        ax.set_title(f'3D Trajectory at Frame {frame_idx:02d}/50 (t={t:.2f})',
                    fontweight='bold', fontsize=12)
        ax.view_init(elev=20, azim=45)
        ax.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(frame_dir / f"frame_3d_{frame_idx:02d}.png", dpi=120, bbox_inches='tight')
        plt.close()

    print(f"\nSaved 102 frames (51 × 2D + 51 × 3D) to {frame_dir}/")
    print("Files: frame_2d_00.png to frame_2d_50.png")
    print("       frame_3d_00.png to frame_3d_50.png")

    return frame_dir

def main():
    print("="*80)
    print("LDA TRAJECTORY - 51 FRAME SEQUENCE GENERATION")
    print("="*80)
    print()

    model, device, real_images, real_labels = load_data()
    print()

    frame_dir = create_frame_sequence(real_images, real_labels, model, device)

    print()
    print("="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nFrames saved to: {frame_dir}")
    print("\nFrame naming:")
    print("  - frame_2d_00.png to frame_2d_50.png (2D LDA space)")
    print("  - frame_3d_00.png to frame_3d_50.png (3D LDA space)")
    print("\nUse these frames to:")
    print("  1. Create animated GIF/video")
    print("  2. Show progression sequence")
    print("  3. Analyze individual timesteps in detail")

if __name__ == '__main__':
    main()
