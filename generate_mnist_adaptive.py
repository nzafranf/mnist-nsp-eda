#!/usr/bin/env python
"""
Generate 10 high-quality MNIST digits using the trained Flow Matching model
with detailed computation metrics.

Demonstrates Section V Algorithm 5.1 with practical image generation.
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
from models.fm import ImageFlowMatcher

def main():
    """Generate high-quality MNIST samples."""

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model checkpoint
    checkpoint_path = Path("PRIMARY/checkpoints/fm-balanced-epoch=014-train_loss=0.1829.ckpt")
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Current directory:", Path.cwd())
        return

    print(f"\nLoading model from {checkpoint_path}...")
    model = ImageFlowMatcher.load_from_checkpoint(str(checkpoint_path))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")

    # Generate samples with detailed metrics
    print("\n" + "="*70)
    print("MNIST IMAGE GENERATION WITH TRAINED FLOW MATCHING MODEL")
    print("="*70)

    n_samples = 10
    ode_steps = 50  # High quality
    batch_size = 1

    results = {
        'model_checkpoint': str(checkpoint_path),
        'device': device,
        'num_samples': n_samples,
        'ode_steps': ode_steps,
        'samples': [],
    }

    all_samples = []
    all_times = []

    print(f"\nGenerating {n_samples} samples with {ode_steps} ODE steps...")
    print(f"(Using {device} for inference)\n")

    for i in range(n_samples):
        print(f"[{i+1:2d}/{n_samples}] Generating...", end='', flush=True)

        t_start = time.time()

        with torch.no_grad():
            samples = model.generate(
                batch_size=batch_size,
                sample_image_size=(1, 28, 28),
                num_steps=ode_steps
            )
            all_samples.extend(samples.cpu())

        t_end = time.time()
        wall_time = t_end - t_start
        all_times.append(wall_time)

        sample_np = samples.squeeze().cpu().numpy()

        results['samples'].append({
            'sample_id': i,
            'wall_time_sec': float(wall_time),
            'ode_steps': ode_steps,
            'value_min': float(sample_np.min()),
            'value_max': float(sample_np.max()),
            'value_mean': float(sample_np.mean()),
        })

        print(f" Time={wall_time:.4f}s, "
              f"Val=[{sample_np.min():.3f}, {sample_np.max():.3f}]")

    # Aggregate metrics
    results['total_time_sec'] = float(sum(all_times))
    results['avg_time_per_sample'] = float(np.mean(all_times))
    results['std_time_per_sample'] = float(np.std(all_times))
    results['throughput_samples_per_sec'] = float(n_samples / results['total_time_sec'])

    # Save results
    output_dir = Path("richardson_eda/results/final_mnist_samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save grid visualization
    all_samples_tensor = torch.cat([s.reshape(1, 1, 28, 28) if s.dim() == 3 else s for s in all_samples], dim=0)
    grid_path = output_dir / "mnist_grid.png"
    torchvision.utils.save_image(all_samples_tensor, str(grid_path), nrow=5, normalize=False, pad_value=0.5)
    print(f"\nSaved grid visualization: {grid_path}")

    # Save individual samples
    for i, sample in enumerate(all_samples_tensor):
        img_path = output_dir / f"digit_{i:02d}.png"
        torchvision.utils.save_image(sample, str(img_path), normalize=False)

    print(f"Saved {n_samples} individual samples to {output_dir}/digit_*.png")

    # Save metrics
    metrics_path = output_dir / "generation_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"""
Summary:
  Samples generated:         {n_samples}
  ODE solver steps:          {ode_steps}
  Total wall-clock time:     {results['total_time_sec']:.4f}s
  Average time per sample:   {results['avg_time_per_sample']:.4f}s
  Throughput:                {results['throughput_samples_per_sec']:.2f} samples/second

  Sample value ranges:
    Min across samples:      {min(s['value_min'] for s in results['samples']):.3f}
    Max across samples:      {max(s['value_max'] for s in results['samples']):.3f}

Results saved to: {output_dir}/
  - mnist_grid.png
  - digit_00.png ... digit_09.png
  - generation_metrics.json
""")

    return output_dir, results

if __name__ == '__main__':
    output_dir, results = main()
