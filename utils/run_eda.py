#!/usr/bin/env python
"""
Master EDA Runner - Orchestrates full pipeline:
1. Quick training (if no checkpoint provided)
2. Trajectory generation and capture
3. Comprehensive EDA (Phases 1-9)
4. Report generation
"""

import torch
import argparse
import sys
from pathlib import Path
from src.eda_master import EDAMaster
from src.quick_train import quick_train
from models.fm import ImageFlowMatcher


def find_latest_checkpoint(checkpoint_dir: str = 'checkpoints'):
    """Find the most recent checkpoint in the directory."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob('fm-best-*.ckpt'), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(ckpts[0]) if ckpts else None


def validate_checkpoint(checkpoint_path: str, device):
    """Test-load checkpoint to ensure it's compatible."""
    try:
        model = ImageFlowMatcher(lr=1e-4, c_unet=32)
        state_dict = torch.load(checkpoint_path, map_location=device)
        if 'state_dict' in state_dict:
            model.load_state_dict(state_dict['state_dict'])
        else:
            model.load_state_dict(state_dict)

        # Quick validation: try a forward pass
        x = torch.randn(1, 1, 28, 28, device=device)
        t = torch.tensor([0.5], device=device)
        model.to(device)
        model.eval()
        with torch.no_grad():
            _ = model(x, t)
        return True
    except Exception as e:
        print(f"   Validation failed: {e}")
        return False


def run_pipeline(
    checkpoint: str = None,
    num_trajectories: int = 20,
    num_epochs: int = 3,
    results_dir: str = 'results',
    skip_training: bool = False,
    force_retrain: bool = False,
):
    """
    Run the complete EDA pipeline.

    Args:
        checkpoint: Path to checkpoint (if None, will search for existing or train)
        num_trajectories: Number of trajectories to analyze
        num_epochs: Number of epochs for training (if needed)
        results_dir: Directory for results
        skip_training: Skip training even if no checkpoint
        force_retrain: Force retraining even if checkpoint exists
    """
    print("\n" + "="*100)
    print("FLOW MATCHING EDA PIPELINE")
    print("="*100)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = None

    # Step 1: Find or train checkpoint
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            print(f"\n[ERROR] Checkpoint not found: {checkpoint}")
            exit(1)
        print(f"\n[OK] Using provided checkpoint: {checkpoint}")
    elif not force_retrain:
        # Try to find existing checkpoint
        print("\n[CHECKPOINT] Searching for existing checkpoints...")
        existing_ckpt = find_latest_checkpoint('checkpoints')
        if existing_ckpt:
            print(f"   Found: {Path(existing_ckpt).name}")
            print(f"   Validating checkpoint compatibility...")
            if validate_checkpoint(existing_ckpt, device):
                checkpoint_path = existing_ckpt
                print(f"[OK] Checkpoint validated, reusing trained model")
            else:
                print(f"[WARNING] Checkpoint validation failed, will retrain")

        if not checkpoint_path and not skip_training:
            print("\n[TRAINING] No valid checkpoint found, training fresh model...")
            checkpoint_path = quick_train(
                num_epochs=num_epochs,
                batch_size=32,
                checkpoint_dir='checkpoints',
            )
            if not checkpoint_path:
                print("\n[ERROR] Training failed to produce checkpoint")
                exit(1)
    elif force_retrain:
        print("\n[TRAINING] Force retrain requested, training fresh model...")
        checkpoint_path = quick_train(
            num_epochs=num_epochs,
            batch_size=32,
            checkpoint_dir='checkpoints',
        )
        if not checkpoint_path:
            print("\n[ERROR] Training failed to produce checkpoint")
            exit(1)

    if not checkpoint_path:
        print("\n[ERROR] No checkpoint available and training skipped")
        exit(1)

    # Step 2: Load model
    print(f"\n[LOADING] Loading model from checkpoint...")
    print(f"   Device: {device}")

    model = ImageFlowMatcher(lr=1e-4, c_unet=32)

    # Handle both .ckpt (Lightning) and .pt (state_dict) files
    checkpoint_path = str(checkpoint_path)
    try:
        if checkpoint_path.endswith('.ckpt'):
            # Lightning checkpoint
            state_dict = torch.load(checkpoint_path, map_location=device)
            if 'state_dict' in state_dict:
                model.load_state_dict(state_dict['state_dict'])
            else:
                model.load_state_dict(state_dict)
        else:
            # Regular state dict
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
    except Exception as e:
        print(f"\n[WARNING] Error loading checkpoint: {e}")
        print("[RETRAINING] Checkpoint incompatible, retraining model...")
        checkpoint_path = quick_train(
            num_epochs=num_epochs,
            batch_size=32,
            checkpoint_dir='checkpoints',
        )
        if not checkpoint_path:
            print("\n[ERROR] Retraining failed")
            exit(1)
        state_dict = torch.load(str(checkpoint_path), map_location=device)
        if 'state_dict' in state_dict:
            model.load_state_dict(state_dict['state_dict'])
        else:
            model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    print(f"[OK] Model loaded successfully")

    # Step 3: Run EDA
    print(f"\n[EDA] Starting EDA analysis with {num_trajectories} trajectories...")
    print("   This will run all 9 phases of analysis")

    eda = EDAMaster(results_dir=results_dir)
    eda.run_full_eda(model, num_trajectories=num_trajectories)

    # Step 4: Summary
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    print(f"\nResults Directory: {Path(results_dir).absolute()}")
    print("\nGenerated Artifacts:")
    print(f"  - Phase 1: Trajectory data (NPZ files)")
    print(f"  - Phase 2: PCA visualizations")
    print(f"  - Phase 3: Curvature & alignment plots")
    print(f"  - Phase 4: Leap viability analysis")
    print(f"  - Phase 5: Information gain profiles")
    print(f"  - Phase 6: Schedule optimization results")
    print(f"  - Phase 7: Clustering analysis")
    print(f"  - Phase 8: Ablation study")
    print(f"  - Phase 9: Comprehensive summary report")
    print("\n[SUCCESS] Pipeline complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run complete Flow Matching EDA pipeline'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--num-trajectories',
        type=int,
        default=20,
        help='Number of trajectories to analyze'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=3,
        help='Number of epochs for training (if no checkpoint provided)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory for results'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training even if no checkpoint found'
    )
    parser.add_argument(
        '--force-retrain',
        action='store_true',
        help='Force retraining even if checkpoint exists'
    )

    args = parser.parse_args()

    try:
        run_pipeline(
            checkpoint=args.checkpoint,
            num_trajectories=args.num_trajectories,
            num_epochs=args.num_epochs,
            results_dir=args.results_dir,
            skip_training=args.skip_training,
            force_retrain=args.force_retrain,
        )
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Pipeline interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
