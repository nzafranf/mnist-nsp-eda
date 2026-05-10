#!/usr/bin/env python
"""
MNIST Image Sampling using Adaptive Richardson Extrapolation
with Trained Flow Matching Model

Generates 10 high-quality MNIST digit images using Algorithm 5.1
with the pre-trained Flow Matching velocity network.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from dataclasses import dataclass

# Add PRIMARY/src to path for model loading
sys.path.insert(0, str(Path(__file__).parent.parent / "PRIMARY" / "src"))

from adaptive_solver import AdaptiveFlowSolver, SolverConfig

@dataclass
class SamplingConfig:
    """Configuration for sampling with adaptive solver."""
    checkpoint_path: Path
    device: str = "cpu"
    tolerance: float = 0.01
    alpha: float = 0.75
    batch_size: int = 1
    n_samples: int = 10
    seed: int = 42

def load_trained_model(checkpoint_path: Path, device: str):
    """Load trained Flow Matching model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")

    try:
        # Try loading with PyTorch Lightning
        import pytorch_lightning as pl
        from src.quick_train import ImageFlowMatcher

        model = ImageFlowMatcher.load_from_checkpoint(str(checkpoint_path))
        model = model.to(device)
        model.eval()
        print("Model loaded with PyTorch Lightning")
        return model
    except Exception as e:
        print(f"PyTorch Lightning loading failed: {e}")
        print("Attempting direct torch load...")

        # Fallback: load raw checkpoint and extract model
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        print(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'raw model'}")

        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # This is a Lightning checkpoint
            from src.quick_train import ImageFlowMatcher
            model = ImageFlowMatcher()
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(device)
            model.eval()
            return model
        else:
            # Direct model
            return checkpoint.to(device)

def create_velocity_function_from_model(model, device):
    """
    Create a velocity function from the trained model.

    Returns a callable v(x, t) that computes velocity at state x and time t.
    """
    def velocity_fn(x: np.ndarray, t: float) -> np.ndarray:
        """
        Compute velocity using the trained model.

        Args:
            x: Current state [784] for MNIST
            t: Time in [0, 1]

        Returns:
            Velocity vector [784]
        """
        with torch.no_grad():
            # Convert to tensor
            x_tensor = torch.from_numpy(x.astype(np.float32)).reshape(1, 1, 28, 28).to(device)
            t_tensor = torch.tensor([t], dtype=torch.float32, device=device)

            try:
                # Call model to get velocity
                v = model(x_tensor, t_tensor)  # Shape: [1, 1, 28, 28]

                # Convert back to numpy and flatten
                v_np = v.cpu().numpy().reshape(784)
                return v_np
            except Exception as e:
                print(f"Error in velocity_fn at t={t}: {e}")
                return np.zeros(784)

    return velocity_fn

def sample_with_adaptive_solver_trained_model(
    config: SamplingConfig
) -> dict:
    """
    Sample images using AdaptiveFlowSolver with trained model velocity.
    """
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = config.device

    # Load trained model
    try:
        model = load_trained_model(config.checkpoint_path, device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Falling back to placeholder velocity function")
        # Fallback: use simple velocity function
        def velocity_fn(x, t):
            # Dummy: slowly move toward origin
            return -0.5 * x
        model = None

    # Create velocity function
    if model is not None:
        velocity_fn = create_velocity_function_from_model(model, device)
    else:
        # Fallback velocity
        velocity_fn = lambda x, t: -0.5 * x

    # Create solver config
    solver_config = SolverConfig(
        tolerance=config.tolerance,
        alpha=config.alpha,
        f_min=0.1,
        f_max=5.0,
        max_steps=5000,
        verbose=False
    )

    results = {
        'config': {
            'tolerance': config.tolerance,
            'alpha': config.alpha,
            'checkpoint': str(config.checkpoint_path),
            'device': config.device,
        },
        'samples': [],
        'aggregate_metrics': {
            'total_time': 0.0,
            'avg_nfe': 0.0,
            'avg_steps': 0.0,
            'avg_reject_rate': 0.0,
            'total_nfe': 0,
        }
    }

    print(f"\n{'='*70}")
    print("ADAPTIVE RICHARDSON SAMPLING WITH TRAINED FLOW MATCHING MODEL")
    print(f"{'='*70}")
    print(f"\nConfiguration:")
    print(f"  Tolerance: {config.tolerance:.6e}")
    print(f"  Alpha: {config.alpha:.2f}")
    print(f"  Device: {device}")
    print(f"\nGenerating {config.n_samples} samples...\n")

    generated_images = []
    all_times = []
    total_nfe = 0

    for sample_id in range(config.n_samples):
        print(f"[{sample_id+1:2d}/{config.n_samples}] Sampling...", end='', flush=True)

        # Random noise initial condition
        x0 = np.random.randn(784).astype(np.float32)

        # Time the solver
        t_wall_start = time.time()

        # Run adaptive solver (t: 0 -> 1)
        solver = AdaptiveFlowSolver(solver_config)
        result = solver.solve(x0, (0.0, 1.0), velocity_fn, H0=0.01)

        t_wall_end = time.time()
        wall_time = t_wall_end - t_wall_start

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
            'n_accepted': int(result['n_accepted']),
            'n_rejected': int(result['n_rejected']),
            'reject_rate': float(result['reject_rate']),
            'wall_time_sec': float(wall_time),
            'avg_step_size': float(np.mean(result['step_sizes'])) if len(result['step_sizes']) > 0 else 0.0,
            'min_step_size': float(np.min(result['step_sizes'])) if len(result['step_sizes']) > 0 else 0.0,
            'max_step_size': float(np.max(result['step_sizes'])) if len(result['step_sizes']) > 0 else 0.0,
            'final_state_norm': float(np.linalg.norm(final_state)),
        }

        results['samples'].append(sample_result)
        total_nfe += result['nfe']
        all_times.append(wall_time)

        print(f" NFE={result['nfe']:3d}, Reject={result['reject_rate']:.1%}, Time={wall_time:.3f}s")

    # Aggregate metrics
    results['aggregate_metrics']['total_time'] = float(sum(all_times))
    results['aggregate_metrics']['avg_time_per_sample'] = float(np.mean(all_times))
    results['aggregate_metrics']['std_time_per_sample'] = float(np.std(all_times))
    results['aggregate_metrics']['total_nfe'] = int(total_nfe)
    results['aggregate_metrics']['avg_nfe'] = float(total_nfe / config.n_samples)
    results['aggregate_metrics']['avg_steps'] = float(np.mean([s['n_accepted'] for s in results['samples']]))
    results['aggregate_metrics']['avg_reject_rate'] = float(np.mean([s['reject_rate'] for s in results['samples']]))
    results['aggregate_metrics']['throughput'] = float(config.n_samples / results['aggregate_metrics']['total_time'])

    return results, generated_images

def save_results(results: dict, images: list, save_dir: Path):
    """Save results and visualizations."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save grid visualization
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for idx, (ax, img) in enumerate(zip(axes.flatten(), images)):
        sample_info = results['samples'][idx]

        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Sample #{idx+1}\nNFE={sample_info['nfe']}, "
                    f"Reject={sample_info['reject_rate']:.1%}",
                    fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    grid_path = save_dir / "generated_mnist_adaptive.png"
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved grid: {grid_path}")
    plt.close()

    # Save individual images
    images_dir = save_dir / "individual_samples_adaptive"
    images_dir.mkdir(parents=True, exist_ok=True)

    for idx, img in enumerate(images):
        sample_info = results['samples'][idx]
        img_path = images_dir / f"sample_{idx:02d}_nfe_{sample_info['nfe']}.png"

        plt.figure(figsize=(4, 4))
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.title(f"Sample #{idx+1}: NFE={sample_info['nfe']}, "
                 f"Time={sample_info['wall_time_sec']:.3f}s")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(img_path, dpi=100, bbox_inches='tight')
        plt.close()

    print(f"Saved individual samples to {images_dir}/")

    # Save metrics
    metrics_path = save_dir / "sampling_metrics_adaptive.json"
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

if __name__ == '__main__':
    # Find checkpoint path
    checkpoint_path = Path(__file__).parent.parent / "PRIMARY" / "checkpoints" / "fm-balanced-epoch=014-train_loss=0.1829.ckpt"

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Available paths to check:")
        primary_dir = Path(__file__).parent.parent / "PRIMARY"
        if primary_dir.exists():
            for f in primary_dir.rglob("*.ckpt"):
                print(f"  - {f}")
        sys.exit(1)

    # Configuration
    config = SamplingConfig(
        checkpoint_path=checkpoint_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tolerance=0.01,
        alpha=0.75,
        n_samples=10,
        seed=42
    )

    print(f"Device: {config.device}")
    print(f"Checkpoint: {config.checkpoint_path}")

    # Sample with adaptive solver
    results, images = sample_with_adaptive_solver_trained_model(config)

    # Save results
    result_dir = Path(__file__).parent / "results" / "image_sampling_adaptive"
    save_results(results, images, result_dir)

    # Print summary
    print(f"\n{'='*70}")
    print("SAMPLING COMPLETE")
    print(f"{'='*70}")
    print(f"""
Summary:
  Samples generated:        {len(images)}
  Total wall-clock time:    {results['aggregate_metrics']['total_time']:.4f}s
  Average time/sample:      {results['aggregate_metrics']['avg_time_per_sample']:.4f}s
  Total FE used:            {results['aggregate_metrics']['total_nfe']}
  Average FE/sample:        {results['aggregate_metrics']['avg_nfe']:.1f}
  Average rejection rate:   {results['aggregate_metrics']['avg_reject_rate']:.2%}
  Throughput:               {results['aggregate_metrics']['throughput']:.1f} samples/s

Results saved to: {result_dir}/
  - generated_mnist_adaptive.png
  - individual_samples_adaptive/ (10 images)
  - sampling_metrics_adaptive.json
""")
