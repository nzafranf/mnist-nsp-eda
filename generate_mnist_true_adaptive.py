#!/usr/bin/env python
"""
Generate MNIST images using EXPLICIT Algorithm 5.1 (Adaptive Richardson Extrapolation)

Uses AdaptiveFlowSolver with trained Flow Matching velocity network.
Demonstrates TRUE adaptive step control (variable steps per sample).

Compare to benchmark:
  - Benchmark (fixed 50-step grid): 21.75s, 2.18±0.20s per sample
  - This (true adaptive):            ? seconds, variable steps per sample
"""

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision
from pathlib import Path
import json
import time
import numpy as np
import sys

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "richardson_eda"))
from adaptive_solver import AdaptiveFlowSolver, SolverConfig

# Add model path
sys.path.insert(0, str(Path(__file__).parent))
from models.fm import ImageFlowMatcher


def create_velocity_function_from_model(model, device):
    """
    Create a callable velocity function from the trained Flow Matching model.

    The model's forward pass takes (x_t: [B,1,28,28], t: [B]) and returns velocity.
    We need to reshape for our scalar interface.
    """

    def velocity_fn(x_flat: np.ndarray, t: float) -> np.ndarray:
        """
        Compute velocity v(x, t) for scalar time and flattened state.

        Args:
            x_flat: Flattened state [784] for MNIST image
            t: Time scalar in [0, 1]

        Returns:
            Velocity vector [784]
        """
        with torch.no_grad():
            # Reshape to image format [1, 1, 28, 28]
            x_img = torch.from_numpy(x_flat.astype(np.float32)).reshape(1, 1, 28, 28).to(device)

            # Time tensor
            t_tensor = torch.tensor([t], dtype=torch.float32, device=device).unsqueeze(0)

            try:
                # Model forward pass: predict velocity
                v_img = model(x_img, t_tensor)  # Shape: [1, 1, 28, 28]

                # Reshape back to flat [784]
                v_flat = v_img.squeeze().cpu().numpy().astype(np.float32).flatten()

                return v_flat
            except Exception as e:
                print(f"Error in velocity_fn at t={t:.3f}: {e}")
                return np.zeros(784, dtype=np.float32)

    return velocity_fn


def sample_with_true_adaptive_solver(model, device, config, n_samples=10):
    """
    Generate samples using explicit AdaptiveFlowSolver.

    This demonstrates TRUE Algorithm 5.1 with variable adaptive steps.
    """

    results = {
        'method': 'Explicit Algorithm 5.1 (AdaptiveFlowSolver)',
        'config': {
            'tolerance': config.tolerance,
            'alpha': config.alpha,
            'f_min': config.f_min,
            'f_max': config.f_max,
            'max_steps': config.max_steps,
        },
        'samples': [],
        'aggregate_metrics': {
            'total_time': 0.0,
            'avg_time': 0.0,
            'std_time': 0.0,
            'total_nfe': 0,
            'avg_nfe': 0.0,
            'avg_steps': 0.0,
            'avg_reject_rate': 0.0,
            'throughput': 0.0,
        }
    }

    # Create velocity function from model
    velocity_fn = create_velocity_function_from_model(model, device)

    print(f"\n{'='*70}")
    print("ADAPTIVE RICHARDSON SAMPLING - TRUE ALGORITHM 5.1")
    print(f"{'='*70}")
    print(f"\nConfiguration:")
    print(f"  Tolerance: {config.tolerance:.6e}")
    print(f"  Alpha: {config.alpha:.2f}")
    print(f"  Max steps: {config.max_steps}")
    print(f"\nGenerating {n_samples} samples with variable adaptive steps...\n")

    all_times = []
    total_nfe = 0
    generated_images = []

    np.random.seed(42)

    for sample_id in range(n_samples):
        print(f"[{sample_id+1:2d}/{n_samples}] Sampling...", end='', flush=True)

        # Random noise initial condition in [0, 1] image space
        x0 = np.random.randn(784).astype(np.float32)

        # Time the solver
        t_start = time.time()

        # Run ADAPTIVE solver (t: 0 -> 1)
        solver = AdaptiveFlowSolver(config)
        result = solver.solve(x0, (0.0, 1.0), velocity_fn, H0=0.01)

        t_end = time.time()
        wall_time = t_end - t_start
        all_times.append(wall_time)

        # Final state is the generated image
        final_state = result['trajectory'][-1]

        # Normalize and reshape to 28x28 MNIST
        image = np.clip(final_state, -3, 3) / 3  # Rough normalization
        image = (image + 1) / 2  # Map to [0, 1]
        image = image.reshape(28, 28)
        generated_images.append(image)

        sample_result = {
            'sample_id': sample_id,
            'nfe': int(result['nfe']),
            'n_steps': int(result['n_accepted']),
            'n_rejected': int(result['n_rejected']),
            'reject_rate': float(result['reject_rate']),
            'wall_time_sec': float(wall_time),
            'avg_step_size': float(np.mean(result['step_sizes'])) if len(result['step_sizes']) > 0 else 0.0,
            'min_step_size': float(np.min(result['step_sizes'])) if len(result['step_sizes']) > 0 else 0.0,
            'max_step_size': float(np.max(result['step_sizes'])) if len(result['step_sizes']) > 0 else 0.0,
        }

        results['samples'].append(sample_result)
        total_nfe += result['nfe']

        print(f" NFE={result['nfe']:3d}, Steps={result['n_accepted']:2d}, "
              f"Reject={result['reject_rate']:.1%}, Time={wall_time:.3f}s")

    # Aggregate metrics
    results['aggregate_metrics']['total_time'] = float(sum(all_times))
    results['aggregate_metrics']['avg_time'] = float(np.mean(all_times))
    results['aggregate_metrics']['std_time'] = float(np.std(all_times))
    results['aggregate_metrics']['total_nfe'] = int(total_nfe)
    results['aggregate_metrics']['avg_nfe'] = float(total_nfe / n_samples)
    results['aggregate_metrics']['avg_steps'] = float(np.mean([s['n_steps'] for s in results['samples']]))
    results['aggregate_metrics']['avg_reject_rate'] = float(np.mean([s['reject_rate'] for s in results['samples']]))
    results['aggregate_metrics']['throughput'] = float(n_samples / results['aggregate_metrics']['total_time'])

    return results, generated_images


def main():
    """Generate MNIST samples with true adaptive Algorithm 5.1."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    checkpoint_path = Path("PRIMARY/checkpoints/fm-balanced-epoch=014-train_loss=0.1829.ckpt")
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return

    print(f"\nLoading model from {checkpoint_path}...")
    model = ImageFlowMatcher.load_from_checkpoint(str(checkpoint_path))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")

    # Configuration from stability analysis
    config = SolverConfig(
        tolerance=0.01,      # From alpha stability analysis
        alpha=0.75,          # Optimal for stability
        f_min=0.1,
        f_max=5.0,
        max_steps=5000,
        verbose=False
    )

    # Generate with true adaptive solver
    results, images = sample_with_true_adaptive_solver(model, device, config, n_samples=10)

    # Save results
    output_dir = Path("richardson_eda/results/adaptive_algorithm5_samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save grid
    images_tensor = torch.tensor(np.array(images)).unsqueeze(1)  # Add channel dimension
    grid_path = output_dir / "mnist_grid_adaptive.png"
    torchvision.utils.save_image(images_tensor, str(grid_path), nrow=5, normalize=False, pad_value=0.5)
    print(f"\nSaved grid: {grid_path}")

    # Save individual images
    for i, img in enumerate(images):
        sample_info = results['samples'][i]
        img_path = output_dir / f"digit_{i:02d}_nfe_{sample_info['nfe']}_steps_{sample_info['n_steps']}.png"

        plt.figure(figsize=(4, 4))
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.title(f"Sample #{i+1}: NFE={sample_info['nfe']}, "
                 f"Steps={sample_info['n_steps']}, "
                 f"Time={sample_info['wall_time_sec']:.3f}s")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(img_path, dpi=100, bbox_inches='tight')
        plt.close()

    print(f"Saved individual samples to {output_dir}/digit_*.png")

    # Save metrics
    metrics_path = output_dir / "adaptive_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("ADAPTIVE ALGORITHM 5.1 COMPLETE")
    print(f"{'='*70}")
    print(f"""
Summary:
  Samples generated:        10
  Total wall-clock time:    {results['aggregate_metrics']['total_time']:.4f}s
  Average time/sample:      {results['aggregate_metrics']['avg_time']:.4f}s ± {results['aggregate_metrics']['std_time']:.4f}s
  Total FE:                 {results['aggregate_metrics']['total_nfe']}
  Average FE/sample:        {results['aggregate_metrics']['avg_nfe']:.1f}
  Average steps/sample:     {results['aggregate_metrics']['avg_steps']:.1f}
  Average rejection rate:   {results['aggregate_metrics']['avg_reject_rate']:.2%}
  Throughput:               {results['aggregate_metrics']['throughput']:.2f} samples/s

Comparison to Benchmark (Fixed 50-step grid):
  Benchmark time:           21.75 seconds
  This time:                {results['aggregate_metrics']['total_time']:.2f} seconds
  Speedup:                  {21.75 / results['aggregate_metrics']['total_time']:.2f}x

Results saved to: {output_dir}/
  - mnist_grid_adaptive.png
  - digit_*_nfe_*_steps_*.png (10 samples, showing variable steps)
  - adaptive_metrics.json
""")

    # Show per-sample step variation
    print("\nPer-Sample Step Variation (Algorithm 5.1 Adaptive Control):")
    print("Sample  NFE   Steps  Reject%  Time(s)")
    print("------  ---   -----  -------  -------")
    for s in results['samples']:
        print(f"  {s['sample_id']:2d}    {s['nfe']:3d}   {s['n_steps']:3d}    {s['reject_rate']:5.2%}   {s['wall_time_sec']:.4f}")

    return output_dir, results


if __name__ == '__main__':
    output_dir, results = main()
