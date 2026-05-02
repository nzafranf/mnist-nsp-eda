#!/usr/bin/env python
"""
Fast LDA-based visualization (fewer frames for quicker generation)
"""
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from models.fm import ImageFlowMatcher
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torchvision import datasets, transforms
import plotly.graph_objects as go
from tqdm import tqdm

def load_best_checkpoint():
    """Load the best trained model checkpoint."""
    ckpt_path = Path("outputs/fm/2026-05-01/22-58-12/training_logs/version_0/checkpoints/fm-balanced-epoch=014-train_loss=0.1829.ckpt")
    print(f"Loading checkpoint: {ckpt_path}")
    model = ImageFlowMatcher.load_from_checkpoint(str(ckpt_path))
    model.eval()
    device = torch.device('cpu')
    model = model.to(device)
    return model, device

def load_mnist_data(num_samples=500):
    """Load MNIST dataset."""
    print(f"Loading MNIST dataset ({num_samples} samples)...")
    transform = transforms.ToTensor()
    test_data = datasets.MNIST(root='./data', train=False, transform=transform, download=False)

    indices = np.random.choice(len(test_data), num_samples, replace=False)
    images = []
    labels = []

    for idx in indices:
        img, label = test_data[idx]
        images.append(img)
        labels.append(label)

    images = torch.stack(images)
    labels = np.array(labels)
    return images, labels

def flatten_images(images):
    """Flatten images for LDA."""
    return images.reshape(images.shape[0], -1)

def generate_samples_at_t(model, device, t, num_samples=200):
    """Generate samples by solving ODE from time t to 1."""
    print(f"  Generating {num_samples} samples at t={t:.1f}...")

    samples = []
    with torch.no_grad():
        for i in range(0, num_samples, 32):
            batch_size = min(32, num_samples - i)

            # Start from noise
            x_0 = torch.randn(batch_size, 1, 28, 28, device=device)

            # Time grid from t to 1
            num_steps = max(5, int(50 * (1 - t)))
            time_grid = torch.linspace(t, 1, num_steps, device=device)

            # Solve ODE
            from flow_matching.solver import ODESolver
            solver = ODESolver(velocity_model=model.model)

            generated = solver.sample(
                time_grid=time_grid,
                x_init=x_0,
                method='dopri5',
                step_size=None,
                atol=1e-4,
                rtol=1e-4
            )

            if model.normalize_data:
                generated = (generated + 1) / 2
                generated = torch.clamp(generated, 0, 1)

            samples.append(generated)

    return torch.cat(samples, dim=0)

def create_3d_interactive_plot(real_images, real_labels, time_steps=[0.0, 0.5, 1.0]):
    """Create interactive 3D plot with multiple time points."""
    print("\nCreating interactive 3D visualization...")
    Path("results").mkdir(exist_ok=True)

    model, device = load_best_checkpoint()

    # Fit LDA models
    real_flat = flatten_images(real_images)
    lda_digit = LinearDiscriminantAnalysis(n_components=2)
    lda_digit.fit(real_flat, real_labels)

    lda_realfake = LinearDiscriminantAnalysis(n_components=1)
    # Fit on dummy data to initialize
    dummy_labels = np.hstack([np.zeros(250), np.ones(250)])
    lda_realfake.fit(real_flat[:500], dummy_labels)

    # Project real data
    real_2d = lda_digit.transform(real_flat)
    real_1d = lda_realfake.transform(real_flat)
    real_3d = np.hstack([real_2d, real_1d])

    fig = go.Figure()

    # Add real data (background)
    for digit in range(10):
        mask = real_labels == digit
        fig.add_trace(go.Scatter3d(
            x=real_3d[mask, 0],
            y=real_3d[mask, 1],
            z=real_3d[mask, 2],
            mode='markers',
            name=f'Real Digit {digit}',
            marker=dict(size=4, opacity=0.4, color=digit, colorscale='Tab10'),
            hovertemplate=f'<b>Ground Truth: {digit}</b><br>LDA1: %{{x:.2f}}<br>LDA2: %{{y:.2f}}<br>Real/Fake: %{{z:.2f}}<extra></extra>'
        ))

    # Add generated samples at different times
    colors_time = ['red', 'orange', 'yellow']
    for t_idx, t in enumerate(time_steps):
        print(f"Generating trajectory at t={t}...")
        gen_samples = generate_samples_at_t(model, device, t, num_samples=150)
        gen_flat = flatten_images(gen_samples)
        gen_2d = lda_digit.transform(gen_flat)

        # Fit and transform for real/fake
        combined = np.vstack([real_flat[:500], gen_flat])
        labels_rf = np.hstack([np.zeros(500), np.ones(len(gen_flat))])
        lda_realfake.fit(combined, labels_rf)
        gen_1d = lda_realfake.transform(gen_flat)
        gen_3d = np.hstack([gen_2d, gen_1d])

        fig.add_trace(go.Scatter3d(
            x=gen_3d[:, 0],
            y=gen_3d[:, 1],
            z=gen_3d[:, 2],
            mode='markers',
            name=f'Generated (t={t})',
            marker=dict(size=7, color=colors_time[t_idx], symbol='x', opacity=0.8),
            hovertemplate=f'<b>Generated (t={t})</b><br>LDA1: %{{x:.2f}}<br>LDA2: %{{y:.2f}}<br>Real/Fake: %{{z:.2f}}<extra></extra>'
        ))

    fig.update_layout(
        title='3D LDA Interactive Visualization<br><sub>Digit Discrimination (X,Y) + Real/Fake (Z) at Different Time Points</sub>',
        scene=dict(
            xaxis_title='LDA 1: Digit Discrimination',
            yaxis_title='LDA 2: Digit Discrimination',
            zaxis_title='LDA 3: Real vs Generated',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        width=1400,
        height=1000,
        hovermode='closest',
        showlegend=True
    )

    output_file = "results/lda_interactive_3d_fast.html"
    fig.write_html(output_file)
    print(f"Saved: {output_file}")
    print("\n✓ Open in browser to interact with the visualization")
    print("  - Rotate: Click + drag")
    print("  - Zoom: Scroll wheel")
    print("  - Pan: Right-click + drag")
    print("  - Hover: See sample details")

def main():
    print("="*80)
    print("FAST LDA TRAJECTORY VISUALIZATION")
    print("="*80)
    print()

    print("Loading MNIST data...")
    real_images, real_labels = load_mnist_data(num_samples=500)
    print()

    create_3d_interactive_plot(real_images, real_labels, time_steps=[0.0, 0.5, 1.0])
    print()

    print("="*80)
    print("COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
