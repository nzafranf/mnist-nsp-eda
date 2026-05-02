#!/usr/bin/env python
"""
LDA-based Flow Field Trajectory Visualization with Animations
- 2D animation: Digit discrimination (LDA components 1-2)
- 3D animation: Digit + Real/Fake discrimination
- 3D interactive: Plotly interactive exploration
"""
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path
from models.fm import ImageFlowMatcher
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import torch.nn.functional as F
from torchvision import datasets, transforms
import plotly.graph_objects as go
import plotly.express as px
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

def load_mnist_data(num_samples=1000):
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

def generate_trajectories(model, device, num_samples=500, time_steps=None):
    """Generate trajectories from noise to data across time."""
    if time_steps is None:
        time_steps = np.linspace(0, 1, 11)  # [0, 0.1, 0.2, ..., 1.0]

    print(f"Generating {num_samples} trajectories at {len(time_steps)} time points...")

    all_trajectories = {}  # {t: tensor of shape (num_samples, 1, 28, 28)}

    with torch.no_grad():
        for t_idx, t in enumerate(tqdm(time_steps, desc="Time steps")):
            # For each time point, generate samples
            # We'll use ODE solver with specific endpoint time
            samples = []

            for _ in range(0, num_samples, 32):
                batch_size = min(32, num_samples - len(samples))

                # Generate from t to 1 (t is start, 1 is data)
                x_t = torch.randn(batch_size, 1, 28, 28, device=device)

                # Create time grid from t to 1 (ensure at least 2 points and t < 1)
                num_steps = max(2, int(50 * (1 - t)))
                if t >= 1.0:
                    t = 0.99  # Cap at 0.99 to avoid issues
                time_grid = torch.linspace(t, 1.0, num_steps, device=device)
                # Ensure strictly increasing
                time_grid = torch.unique(time_grid, sorted=True)
                if len(time_grid) < 2:
                    time_grid = torch.tensor([t, 1.0], device=device)

                from flow_matching.solver import ODESolver
                solver = ODESolver(velocity_model=model.model)

                generated = solver.sample(
                    time_grid=time_grid,
                    x_init=x_t,
                    method='dopri5',
                    step_size=None,
                    atol=1e-4,
                    rtol=1e-4
                )

                if model.normalize_data:
                    generated = (generated + 1) / 2
                    generated = torch.clamp(generated, 0, 1)

                samples.append(generated)

            all_trajectories[float(t)] = torch.cat(samples, dim=0)[:num_samples]

    return all_trajectories, time_steps

def fit_lda_models(real_images, generated_images):
    """Fit two LDA models:
    1. Digit discrimination (on real + generated mixed)
    2. Real vs Generated discrimination
    """
    print("Fitting LDA models...")

    # Flatten images
    real_flat = flatten_images(real_images)
    gen_flat = flatten_images(generated_images)

    # LDA 1: Digit classification (real only, 10 classes)
    print("  LDA 1: Digit discrimination (0-9)...")
    lda_digit = LinearDiscriminantAnalysis(n_components=2)

    # Use only real images with their labels for digit LDA
    lda_digit.fit(real_flat, np.arange(len(real_images)) % 10)

    # LDA 2: Real vs Generated discrimination
    print("  LDA 2: Real vs Generated discrimination...")
    lda_realfake = LinearDiscriminantAnalysis(n_components=1)

    # Combine real and generated with binary labels (0=real, 1=generated)
    combined = np.vstack([real_flat, gen_flat])
    labels_realfake = np.hstack([np.zeros(len(real_flat)), np.ones(len(gen_flat))])

    lda_realfake.fit(combined, labels_realfake)

    return lda_digit, lda_realfake

def project_to_lda_space(images, lda_digit, lda_realfake, digit_labels=None):
    """Project images to LDA space."""
    flat = flatten_images(images)

    # Project to digit LDA space (2D)
    lda_2d = lda_digit.transform(flat)

    # Project to real/fake LDA space (1D)
    lda_1d = lda_realfake.transform(flat)

    # Combine to 3D
    lda_3d = np.hstack([lda_2d, lda_1d])

    return lda_2d, lda_1d, lda_3d

def create_2d_animation(trajectories, lda_models, real_labels, time_steps, output_file="results/lda_trajectory_2d.gif"):
    """Create 2D animation of trajectories evolving over time."""
    print("\nCreating 2D animation...")
    Path("results").mkdir(exist_ok=True)

    lda_digit, lda_realfake = lda_models

    fig, ax = plt.subplots(figsize=(12, 10))

    frames = []

    for t_idx, t in enumerate(tqdm(time_steps, desc="Creating 2D frames")):
        ax.clear()

        # Project current trajectory to 2D
        traj_data = trajectories[float(t)]
        traj_2d, _, _ = project_to_lda_space(traj_data, lda_digit, lda_realfake)

        # Plot real data points (background)
        real_2d, _, _ = project_to_lda_space(flatten_images(real_images).reshape(-1, 1, 28, 28),
                                             lda_digit, lda_realfake)

        for digit in range(10):
            mask = real_labels == digit
            ax.scatter(real_2d[mask, 0], real_2d[mask, 1],
                      alpha=0.3, s=30, label=f'Real {digit}', c=[digit], cmap='tab10')

        # Plot generated trajectory
        scatter = ax.scatter(traj_2d[:, 0], traj_2d[:, 1],
                            c='red', s=100, alpha=0.8, marker='x',
                            linewidth=3, label='Generated (t)')

        # Add arrows showing direction
        if t_idx > 0:
            prev_t = time_steps[t_idx - 1]
            prev_traj = trajectories[float(prev_t)]
            prev_2d, _, _ = project_to_lda_space(prev_traj, lda_digit, lda_realfake)

            # Show movement vectors
            dx = traj_2d[:, 0] - prev_2d[:, 0]
            dy = traj_2d[:, 1] - prev_2d[:, 1]
            ax.quiver(prev_2d[:, 0], prev_2d[:, 1], dx, dy,
                     alpha=0.3, scale=30, width=0.003)

        ax.set_xlabel('LDA Component 1 (Digit Discrimination)', fontsize=11, fontweight='bold')
        ax.set_ylabel('LDA Component 2 (Digit Discrimination)', fontsize=11, fontweight='bold')
        ax.set_title(f'Flow Matching Trajectories - 2D LDA Space\nTime window: [t={t:.1f}, 1.0]',
                    fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)

    plt.close(fig)

    # Save as gif
    imageio = __import__('imageio')
    imageio.mimsave(output_file, frames, fps=2)
    print(f"Saved: {output_file}")

def create_3d_animation(trajectories, lda_models, real_labels, time_steps, output_file="results/lda_trajectory_3d.gif"):
    """Create 3D animation with real/fake discrimination."""
    print("\nCreating 3D animation...")
    Path("results").mkdir(exist_ok=True)

    lda_digit, lda_realfake = lda_models

    frames = []

    for t_idx, t in enumerate(tqdm(time_steps, desc="Creating 3D frames")):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Project current trajectory to 3D
        traj_data = trajectories[float(t)]
        traj_2d, traj_1d, traj_3d = project_to_lda_space(traj_data, lda_digit, lda_realfake)

        # Plot real data points (background)
        real_2d, real_1d, real_3d = project_to_lda_space(real_images, lda_digit, lda_realfake)

        # Color by digit class
        colors = plt.cm.tab10(real_labels / 9)
        for digit in range(10):
            mask = real_labels == digit
            ax.scatter(real_3d[mask, 0], real_3d[mask, 1], real_3d[mask, 2],
                      alpha=0.2, s=30, c=[plt.cm.tab10(digit/9)], label=f'Real {digit}')

        # Plot generated samples
        ax.scatter(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2],
                  c='red', s=100, alpha=0.9, marker='x', linewidth=3,
                  label=f'Generated (t={t:.1f})')

        ax.set_xlabel('LDA 1: Digit Disc.', fontsize=10, fontweight='bold')
        ax.set_ylabel('LDA 2: Digit Disc.', fontsize=10, fontweight='bold')
        ax.set_zlabel('LDA 3: Real/Fake', fontsize=10, fontweight='bold')
        ax.set_title(f'3D LDA: Digit + Real/Fake Discrimination\nTime: [t={t:.1f}, 1.0]',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=7, ncol=2)
        ax.view_init(elev=20, azim=45)

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)

        plt.close(fig)

    # Save as gif
    imageio = __import__('imageio')
    imageio.mimsave(output_file, frames, fps=2)
    print(f"Saved: {output_file}")

def create_interactive_3d_plot(real_3d, real_labels, traj_3d, output_file="results/lda_interactive_3d.html"):
    """Create interactive 3D Plotly visualization."""
    print("\nCreating interactive 3D plot...")
    Path("results").mkdir(exist_ok=True)

    # Create figure with both real and generated
    fig = go.Figure()

    # Add real data points (one trace per digit for legend)
    for digit in range(10):
        mask = real_labels == digit
        fig.add_trace(go.Scatter3d(
            x=real_3d[mask, 0],
            y=real_3d[mask, 1],
            z=real_3d[mask, 2],
            mode='markers',
            name=f'Real Digit {digit}',
            marker=dict(size=4, opacity=0.5),
            hovertemplate=f'<b>Digit {digit}</b><br>LDA1: %{{x:.2f}}<br>LDA2: %{{y:.2f}}<br>Real/Fake: %{{z:.2f}}<extra></extra>'
        ))

    # Add generated samples
    fig.add_trace(go.Scatter3d(
        x=traj_3d[:, 0],
        y=traj_3d[:, 1],
        z=traj_3d[:, 2],
        mode='markers',
        name='Generated Samples',
        marker=dict(size=8, color='red', symbol='x', opacity=0.9),
        hovertemplate='<b>Generated</b><br>LDA1: %{x:.2f}<br>LDA2: %{y:.2f}<br>Real/Fake: %{z:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title='3D LDA Interactive Visualization<br><sub>Digit Discrimination (X,Y) + Real/Fake (Z)</sub>',
        scene=dict(
            xaxis_title='LDA Component 1<br>(Digit Discrimination)',
            yaxis_title='LDA Component 2<br>(Digit Discrimination)',
            zaxis_title='LDA Component 3<br>(Real vs Generated)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        width=1200,
        height=900,
        hovermode='closest',
        showlegend=True
    )

    fig.write_html(output_file)
    print(f"Saved: {output_file}")
    print(f"Open in browser to interact (rotate, zoom, etc.)")

def main():
    print("="*80)
    print("LDA-BASED TRAJECTORY VISUALIZATION WITH ANIMATIONS")
    print("="*80)
    print()

    # Load model
    print("[1/6] Loading model...")
    model, device = load_best_checkpoint()
    print()

    # Load MNIST data
    print("[2/6] Loading MNIST data...")
    real_images, real_labels = load_mnist_data(num_samples=500)
    print()

    # Generate trajectories
    print("[3/6] Generating trajectories...")
    time_steps = np.linspace(0, 1, 11)  # [0, 0.1, ..., 1.0]
    trajectories, ts = generate_trajectories(model, device, num_samples=200, time_steps=time_steps)
    print()

    # Fit LDA models
    print("[4/6] Fitting LDA models...")
    # Use final trajectory (closest to t=1) as representative generated samples
    final_t = max(trajectories.keys())  # Get the last/highest t value
    gen_images_final = trajectories[final_t]
    lda_digit, lda_realfake = fit_lda_models(real_images, gen_images_final)
    print()

    # Project real data for reference
    print("[5/6] Creating animations...")
    create_2d_animation(trajectories, (lda_digit, lda_realfake), real_labels, time_steps)
    create_3d_animation(trajectories, (lda_digit, lda_realfake), real_labels, time_steps)
    print()

    # Create interactive plot
    print("[6/6] Creating interactive visualization...")
    gen_final_2d, gen_final_1d, gen_final_3d = project_to_lda_space(
        gen_images_final, lda_digit, lda_realfake
    )
    real_2d, real_1d, real_3d = project_to_lda_space(real_images, lda_digit, lda_realfake)

    create_interactive_3d_plot(real_3d, real_labels, gen_final_3d)
    print()

    print("="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - results/lda_trajectory_2d.gif (2D animation)")
    print("  - results/lda_trajectory_3d.gif (3D animation)")
    print("  - results/lda_interactive_3d.html (Interactive 3D - open in browser)")
    print("\nVisualization details:")
    print("  - 11 time frames: t ∈ [0.0, 1.0] (step 0.1)")
    print("  - 2D LDA: Digit classification (2 components)")
    print("  - 3D LDA: Digit (2 comp) + Real/Fake (1 comp)")
    print("  - 500 real MNIST samples (10 colors for digits 0-9)")
    print("  - 200 generated samples per time step")

if __name__ == '__main__':
    main()
