"""
Quick Training Script - Train a basic FM model on MNIST
This generates a checkpoint for EDA analysis.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.fm import ImageFlowMatcher


def quick_train(
    num_epochs: int = 5,
    batch_size: int = 32,
    data_dir: str = 'data',
    checkpoint_dir: str = 'checkpoints',
    device: str = 'auto'
):
    """
    Quick training of FM model.

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        data_dir: Directory to download MNIST
        checkpoint_dir: Directory to save checkpoints
        device: Device to train on ('cuda', 'cpu', or 'auto')
    """
    print("=" * 80)
    print("Quick Training: Flow Matching on MNIST")
    print("=" * 80)

    Path(checkpoint_dir).mkdir(exist_ok=True)
    Path(data_dir).mkdir(exist_ok=True)

    # Data setup
    print("\n[1/3] Loading MNIST dataset...")
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    val_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")

    # Model setup
    print("\n[2/3] Initializing model...")
    model = ImageFlowMatcher(lr=1e-4, c_unet=32)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    print("\n[3/3] Training...")

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='fm-best-{epoch:02d}-{train_loss:.2f}',
        monitor='train_loss',
        mode='min',
        save_top_k=2,
    )

    tb_logger = TensorBoardLogger(save_dir='.', name='tb_logs')

    trainer = pl.Trainer(
        accelerator=device,
        devices=1,
        max_epochs=num_epochs,
        callbacks=[checkpoint_callback],
        logger=tb_logger,
        limit_val_batches=0.2,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_loader, val_loader)

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)

    # Find best checkpoint
    checkpoints = list(Path(checkpoint_dir).glob('*.ckpt'))
    if checkpoints:
        best_ckpt = checkpoints[-1]  # Last saved is usually best
        print(f"\nBest checkpoint: {best_ckpt}")
        return str(best_ckpt)
    else:
        print("Warning: No checkpoint found!")
        return None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Quick train FM model')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cuda/cpu)')

    args = parser.parse_args()

    ckpt = quick_train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device
    )

    if ckpt:
        print(f"\n[OK] Ready for EDA! Use: python src/eda_master.py --checkpoint {ckpt}")
