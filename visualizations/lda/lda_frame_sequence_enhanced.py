#!/usr/bin/env python
"""
Enhanced 51-Frame LDA Visualization with Velocity Arrows, Explainability, and Caching

Features:
  - Full trajectory caching (run model once, reuse for all 51 frames)
  - Velocity arrows showing sample motion between timesteps
  - Arrow color = per-sample variance NOT captured by LDA (missed information)
  - LDA explainability % on axis labels
  - Live progress log to results/lda_frames/progress.txt
"""

import sys
from pathlib import Path as PathLib
sys.path.insert(0, str(PathLib(__file__).resolve().parent.parent.parent))

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm

from models.fm import ImageFlowMatcher
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torchvision import datasets, transforms
from flow_matching.solver import ODESolver


# ============================================================================
# Logging
# ============================================================================

def log(msg, log_path):
    """Write timestamped message to console and log file."""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


# ============================================================================
# Data Loading & Caching
# ============================================================================

def load_model_and_real_data(ckpt_path, log_path):
    """Load model and real MNIST data."""
    log("Loading model and real MNIST data...", log_path)

    model = ImageFlowMatcher.load_from_checkpoint(str(ckpt_path))
    model.eval()
    device = torch.device('cpu')
    model = model.to(device)

    transform = transforms.ToTensor()
    test_data = datasets.MNIST(root='./data', train=False, transform=transform, download=False)

    # Use first 300 samples for LDA fitting
    indices = np.arange(300)
    real_images = torch.stack([test_data[i][0] for i in indices])
    real_labels = np.array([test_data[i][1] for i in indices])

    return model, device, real_images, real_labels


def load_or_generate_trajectories(model, device, cache_dir, num_traj=150, num_steps=51,
                                   ckpt_hash="", log_path=None, force_regenerate=False):
    """Load cached trajectories or generate new ones with return_intermediates=True."""
    cache_path = Path(cache_dir) / "trajectories.npz"
    meta_path = Path(cache_dir) / "cache_meta.json"

    # Check cache validity
    cache_valid = False
    if cache_path.exists() and meta_path.exists() and not force_regenerate:
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get('num_traj') == num_traj and meta.get('num_steps') == num_steps:
            cache_valid = True

    if cache_valid:
        if log_path:
            log(f"Loading cached trajectories from {cache_path}", log_path)
        data = np.load(cache_path)
        return data['trajectories'], data['time_grid']

    # Generate trajectories
    if log_path:
        log(f"Generating {num_traj} full trajectories with {num_steps} timesteps...", log_path)

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    trajectories = []
    time_grid = torch.linspace(0, 1, num_steps, device=device)

    with torch.no_grad():
        for traj_idx in tqdm(range(num_traj), desc="Trajectories", disable=(log_path is None)):
            # Fresh Gaussian noise for each trajectory
            x_init = torch.randn(1, 1, 28, 28, device=device)

            # Capture full trajectory with intermediate states
            solver = ODESolver(velocity_model=model.model)
            trajectory = solver.sample(
                x_init=x_init,
                time_grid=time_grid,
                method='dopri5',
                step_size=None,
                atol=1e-4,
                rtol=1e-4,
                return_intermediates=True
            )

            # Handle return format (can be list or tensor)
            if isinstance(trajectory, (list, tuple)):
                trajectory = torch.stack(trajectory, dim=0)
            elif trajectory.shape[0] != len(time_grid):
                if trajectory.shape[1] == len(time_grid):
                    trajectory = trajectory.transpose(0, 1)

            # Denormalize if needed
            if model.normalize_data:
                trajectory = (trajectory + 1) / 2
                trajectory = torch.clamp(trajectory, 0.0, 1.0)

            # Flatten to (num_steps, 784)
            traj_flat = trajectory.reshape(num_steps, -1).cpu().numpy()
            trajectories.append(traj_flat)

    trajectories = np.array(trajectories)  # (num_traj, num_steps, 784)

    # Cache
    np.savez_compressed(cache_path, trajectories=trajectories, time_grid=time_grid.cpu().numpy())
    meta = {
        'num_traj': num_traj,
        'num_steps': num_steps,
        'ckpt_hash': ckpt_hash,
        'timestamp': datetime.now().isoformat()
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    if log_path:
        log(f"Cached trajectories to {cache_path}", log_path)

    return trajectories, time_grid.cpu().numpy()


def load_or_fit_lda(real_images, real_labels, gen_final, cache_dir, log_path=None, force_refit=False):
    """Load cached LDA or fit new models."""
    cache_path = Path(cache_dir) / "lda_models.npz"

    if cache_path.exists() and not force_refit:
        if log_path:
            log(f"Loading cached LDA models from {cache_path}", log_path)
        data = np.load(cache_path, allow_pickle=True)
        lda_digit = data['lda_digit'].item()
        lda_realfake = data['lda_realfake'].item()
        return lda_digit, lda_realfake

    if log_path:
        log("Fitting LDA models...", log_path)

    # Handle both tensor and numpy array inputs
    if isinstance(real_images, np.ndarray):
        real_flat = real_images.reshape(len(real_images), -1)
    else:
        real_flat = real_images.reshape(len(real_images), -1).numpy()

    if isinstance(gen_final, np.ndarray):
        gen_flat = gen_final if gen_final.ndim == 2 else gen_final.reshape(len(gen_final), -1)
    else:
        gen_flat = gen_final.reshape(len(gen_final), -1).numpy()

    # LDA for digit discrimination (2 components)
    lda_digit = LinearDiscriminantAnalysis(n_components=2)
    lda_digit.fit(real_flat, real_labels)

    # LDA for real vs generated (1 component)
    lda_realfake = LinearDiscriminantAnalysis(n_components=1)
    combined = np.vstack([real_flat, gen_flat])
    labels_rf = np.hstack([np.zeros(len(real_flat)), np.ones(len(gen_flat))])
    lda_realfake.fit(combined, labels_rf)

    # Cache
    np.savez_compressed(cache_path, lda_digit=lda_digit, lda_realfake=lda_realfake)

    if log_path:
        var_digit = lda_digit.explained_variance_ratio_
        var_rf = lda_realfake.explained_variance_ratio_
        log(f"LDA fitted. Digit: {var_digit[0]:.1%} + {var_digit[1]:.1%} = {sum(var_digit):.1%}. RealFake: {var_rf[0]:.1%}",
            log_path)

    return lda_digit, lda_realfake


def transform_to_3d(images, lda_digit, lda_realfake):
    """Transform images (already in flat form) to 3D LDA space."""
    flat = images if images.ndim == 2 else images.reshape(len(images), -1)
    lda_2d = lda_digit.transform(flat)
    lda_1d = lda_realfake.transform(flat)
    return np.hstack([lda_2d, lda_1d])


# ============================================================================
# Residual Variance (Missed LDA Information)
# ============================================================================

def compute_lda_confidence(samples_flat, lda_digit):
    """
    Per-sample LDA classification confidence.
    Returns values in [0, 1]:
      1.0 = High confidence (looks like a real digit)
      0.1 = Low confidence (looks like noise)

    Uses the maximum class probability from LDA's probabilistic model.
    At random chance with 10 classes: 0.1, at perfect: 1.0
    """
    # Get class probabilities
    proba = lda_digit.predict_proba(samples_flat)  # (N, 10) - probability for each digit
    max_proba = proba.max(axis=1)  # (N,) - max probability across all classes

    # Normalize to [0, 1]: 0.1 (random) -> 0, 1.0 (certain) -> 1
    confidence = np.clip((max_proba - 0.1) / 0.9, 0, 1)
    return confidence


# ============================================================================
# Frame Rendering
# ============================================================================

def render_frame_2d(frame_idx, trajectories, time_grid, real_3d, real_labels,
                    lda_digit, lda_realfake, frame_dir, log_path=None):
    """Render 2D frame with full trajectory history and confidence coloring."""
    t = frame_idx / 50.0
    step_idx = frame_idx

    # Get states at current timestep
    states_at_t = trajectories[:, step_idx, :]  # (N_traj, 784)
    states_2d = lda_digit.transform(states_at_t)
    states_1d = lda_realfake.transform(states_at_t)
    states_3d = np.hstack([states_2d, states_1d])

    # Compute LDA classification confidence
    confidence = compute_lda_confidence(states_at_t, lda_digit)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 11))

    # Plot real MNIST clusters
    cmap_tab = plt.colormaps['tab10']
    for digit in range(10):
        mask = real_labels == digit
        ax.scatter(real_3d[mask, 0], real_3d[mask, 1],
                  alpha=0.2, s=20, color=cmap_tab(digit / 10.0), label=f'{digit}')

    # Draw full trajectory history for each sample
    cmap_rb = plt.colormaps['RdBu']  # Red-Blue: red=low confidence, blue=high confidence
    for i in range(len(trajectories)):
        # Get history of this sample up to current frame
        history = trajectories[i, :step_idx+1, :]  # (step_idx+1, 784)
        history_2d = lda_digit.transform(history)   # (step_idx+1, 2)

        # Draw line connecting trajectory points
        ax.plot(history_2d[:, 0], history_2d[:, 1],
               color=cmap_rb(confidence[i]),
               alpha=0.6, linewidth=1.5, zorder=5)

        # Mark start and end points
        ax.scatter(history_2d[0, 0], history_2d[0, 1],
                  color='gray', s=30, alpha=0.5, marker='o', zorder=6)  # Start
        ax.scatter(history_2d[-1, 0], history_2d[-1, 1],
                  color=cmap_rb(confidence[i]), s=80, alpha=0.9,
                  marker='*', edgecolors='black', linewidth=0.5, zorder=7)  # End

    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_rb, norm=plt.Normalize(vmin=0, vmax=1)),
                       ax=ax)
    cbar.set_label('LDA Confidence\n(Red=noise, Blue=digit)', fontsize=10)

    # Labels with explainability %
    var1, var2 = lda_digit.explained_variance_ratio_[:2]
    total_var = lda_digit.explained_variance_ratio_.sum()

    ax.set_xlabel(f'LDA1 ({var1:.1%} discriminant var)', fontweight='bold', fontsize=11)
    ax.set_ylabel(f'LDA2 ({var2:.1%} discriminant var)', fontweight='bold', fontsize=11)

    ax.set_title(
        f'Frame {frame_idx:02d}/50 (t={t:.2f}) | Digit LDA captures {total_var:.1%} | Mean confidence: {confidence.mean():.1%}',
        fontweight='bold', fontsize=12
    )

    ax.legend(loc='upper left', fontsize=8, ncol=5)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    save_path = frame_dir / f"frame_2d_{frame_idx:02d}.png"
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def render_frame_3d(frame_idx, trajectories, time_grid, real_3d, real_labels,
                    lda_digit, lda_realfake, frame_dir, log_path=None):
    """Render 3D frame with full trajectory history and confidence coloring."""
    t = frame_idx / 50.0
    step_idx = frame_idx

    # Get states at current timestep
    states_at_t = trajectories[:, step_idx, :]  # (N_traj, 784)
    states_2d = lda_digit.transform(states_at_t)
    states_1d = lda_realfake.transform(states_at_t)
    states_3d = np.hstack([states_2d, states_1d])

    # Compute LDA classification confidence
    confidence = compute_lda_confidence(states_at_t, lda_digit)

    # Create 3D figure
    fig = plt.figure(figsize=(13, 11))
    ax = fig.add_subplot(111, projection='3d')

    # Plot real MNIST clusters
    cmap_tab = plt.colormaps['tab10']
    for digit in range(10):
        mask = real_labels == digit
        ax.scatter(real_3d[mask, 0], real_3d[mask, 1], real_3d[mask, 2],
                  alpha=0.1, s=15, color=cmap_tab(digit / 10.0))

    # Draw full trajectory history for each sample
    cmap_rb = plt.colormaps['RdBu']  # Red-Blue
    for i in range(len(trajectories)):
        # Get history up to current frame
        history = trajectories[i, :step_idx+1, :]  # (step_idx+1, 784)
        history_2d = lda_digit.transform(history)
        history_1d = lda_realfake.transform(history)
        history_3d = np.hstack([history_2d, history_1d])

        # Draw line connecting trajectory points
        ax.plot(history_3d[:, 0], history_3d[:, 1], history_3d[:, 2],
               color=cmap_rb(confidence[i]),
               alpha=0.6, linewidth=1.5, zorder=5)

        # Mark start and end points
        ax.scatter(history_3d[0, 0], history_3d[0, 1], history_3d[0, 2],
                  color='gray', s=30, alpha=0.5, marker='o', zorder=6)  # Start
        ax.scatter(history_3d[-1, 0], history_3d[-1, 1], history_3d[-1, 2],
                  color=cmap_rb(confidence[i]), s=80, alpha=0.9,
                  marker='*', edgecolors='black', linewidth=0.5, zorder=7)  # End

    # Create colorbar for confidence
    sm = plt.cm.ScalarMappable(cmap=cmap_rb, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('LDA Confidence\n(Red=noise, Blue=digit)', fontsize=9)

    # Labels with explainability
    var1, var2 = lda_digit.explained_variance_ratio_[:2]
    var_rf = lda_realfake.explained_variance_ratio_[0]
    total_var = lda_digit.explained_variance_ratio_.sum()

    ax.set_xlabel(f'LDA1 ({var1:.1%})', fontweight='bold', fontsize=11)
    ax.set_ylabel(f'LDA2 ({var2:.1%})', fontweight='bold', fontsize=11)
    ax.set_zlabel(f'Real/Fake ({var_rf:.1%})', fontweight='bold', fontsize=11)

    ax.set_title(
        f'3D Frame {frame_idx:02d}/50 (t={t:.2f}) | LDA Captures {total_var:.1%} | Mean Confidence: {confidence.mean():.1%}',
        fontweight='bold', fontsize=12
    )

    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    save_path = frame_dir / f"frame_3d_{frame_idx:02d}.png"
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("ENHANCED LDA TRAJECTORY - 51 FRAME SEQUENCE WITH ARROWS & CACHING")
    print("=" * 80)
    print()

    # Setup paths
    ckpt_path = Path("outputs/fm/2026-05-01/22-58-12/training_logs/version_0/checkpoints/fm-balanced-epoch=014-train_loss=0.1829.ckpt")
    frame_dir = Path("results/lda_frames")
    cache_dir = Path("results/lda_cache")
    log_path = frame_dir / "progress.txt"

    frame_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Clear and start log
    log_path.write_text("")
    log("=" * 80, log_path)
    log("ENHANCED LDA TRAJECTORY VISUALIZATION WITH ARROWS", log_path)
    log("=" * 80, log_path)
    log("", log_path)

    # Load model and real data
    model, device, real_images, real_labels = load_model_and_real_data(ckpt_path, log_path)
    log("", log_path)

    # Load or generate trajectories
    num_traj = 150
    num_steps = 51
    ckpt_hash = str(ckpt_path)[-20:]
    trajectories, time_grid = load_or_generate_trajectories(
        model, device, cache_dir, num_traj=num_traj, num_steps=num_steps,
        ckpt_hash=ckpt_hash, log_path=log_path
    )
    log("", log_path)

    # Transform real data to LDA space
    log("Transforming real MNIST data to LDA space...", log_path)
    real_flat = real_images.reshape(len(real_images), -1).numpy()

    # Get final generated samples for LDA fitting (already numpy, shape: num_traj x 784)
    gen_final = trajectories[:, -1, :]

    # Fit LDA
    lda_digit, lda_realfake = load_or_fit_lda(
        real_images, real_labels, gen_final, cache_dir, log_path=log_path
    )
    log("", log_path)

    # Transform real data to 3D
    real_3d = transform_to_3d(real_flat, lda_digit, lda_realfake)

    # Render 51 frames
    log(f"Rendering {51 * 2} frames (51x 2D + 51x 3D)...", log_path)
    log("", log_path)

    for frame_idx in tqdm(range(51), desc="Frames"):
        render_frame_2d(frame_idx, trajectories, time_grid, real_3d, real_labels,
                       lda_digit, lda_realfake, frame_dir, log_path)
        render_frame_3d(frame_idx, trajectories, time_grid, real_3d, real_labels,
                       lda_digit, lda_realfake, frame_dir, log_path)

        if (frame_idx + 1) % 10 == 0:
            log(f"  Completed {frame_idx + 1}/51 frames", log_path)

    log("", log_path)
    log("=" * 80, log_path)
    log("COMPLETE", log_path)
    log("=" * 80, log_path)
    log(f"Frames saved to: {frame_dir}", log_path)
    log(f"Cache saved to: {cache_dir}", log_path)
    log(f"Log file: {log_path}", log_path)
    log("", log_path)
    log("Frame features:", log_path)
    log("  - Full trajectory history: each line shows complete path from t=0 to current t", log_path)
    log("  - Trajectory color: red=noise-like (low confidence), blue=digit-like (high confidence)", log_path)
    log("  - Confidence metric: max class probability from LDA's probabilistic model", log_path)
    log("  - Start point: small gray circle, End point: blue/red star", log_path)
    log("  - Axis labels: show LDA explained variance percentages", log_path)


if __name__ == '__main__':
    main()
