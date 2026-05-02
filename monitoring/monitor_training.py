#!/usr/bin/env python
"""
Live Training Progress Monitor
Parses PyTorch Lightning logs and displays real-time progress
"""
import re
import time
import subprocess
from pathlib import Path
from collections import defaultdict
import sys

def parse_training_logs(log_file):
    """Extract epoch and loss from training logs."""
    epochs = defaultdict(dict)

    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return epochs

    for line in lines:
        # Match TensorBoard logging patterns
        # Pattern: Epoch X, global step Y: train_loss = Z.ZZZZ
        match = re.search(r'Epoch\s+(\d+).*train_loss=([0-9.]+)', line)
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            epochs[epoch]['train_loss'] = loss

        # Alternative pattern from Lightning output
        match = re.search(r'train_loss:\s+([0-9.]+)', line)
        if match and 'train_loss' not in line.split('train_loss')[0][-20:]:
            pass

    return epochs

def create_progress_bar(current, total, width=50):
    """Create a simple progress bar."""
    if total == 0:
        percent = 0
    else:
        percent = current / total
    filled = int(width * percent)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {percent*100:6.1f}%"

def display_progress(epochs_data, max_epochs=150):
    """Display training progress in a nice format."""
    if not epochs_data:
        print("⏳ Waiting for training to start...")
        return

    sorted_epochs = sorted(epochs_data.keys())
    latest_epoch = max(sorted_epochs)
    latest_loss = epochs_data[latest_epoch].get('train_loss', None)

    # Calculate progress
    progress = (latest_epoch + 1) / max_epochs

    print("\n" + "="*70)
    print("FLOW MATCHING TRAINING PROGRESS")
    print("="*70)
    print(f"\nEpoch: {latest_epoch + 1}/{max_epochs}")
    print(f"Progress: {create_progress_bar(latest_epoch + 1, max_epochs)}")

    if latest_loss is not None:
        print(f"\nCurrent Loss: {latest_loss:.6f}")

        # Show loss trend
        if len(sorted_epochs) > 1:
            prev_epoch = sorted_epochs[-2]
            prev_loss = epochs_data[prev_epoch].get('train_loss', latest_loss)
            improvement = prev_loss - latest_loss
            indicator = "↓" if improvement > 0 else "↑"
            print(f"Loss Change: {indicator} {abs(improvement):+.6f}")

    # Show best loss so far
    best_loss = min(ep.get('train_loss', float('inf')) for ep in epochs_data.values())
    print(f"Best Loss:   {best_loss:.6f}")

    # ETA calculation
    if len(sorted_epochs) > 5:
        recent_losses = [epochs_data[ep].get('train_loss', float('inf')) for ep in sorted_epochs[-5:]]
        if all(x < float('inf') for x in recent_losses):
            avg_improvement = (recent_losses[0] - recent_losses[-1]) / 5
            if avg_improvement > 0:
                epochs_to_target = (latest_loss - 0.01) / avg_improvement
                eta_epochs = max(0, int(epochs_to_target - (latest_epoch + 1)))
                print(f"ETA to <0.01: {eta_epochs} epochs")

    print("\n" + "="*70)

def monitor_live():
    """Monitor training live with periodic updates."""
    log_file = Path("training_improved_full.log")
    last_epoch = -1

    print("Starting live monitoring... (updates every 10 seconds)")
    print(f"Log file: {log_file}")

    try:
        while True:
            if log_file.exists():
                epochs = parse_training_logs(log_file)

                if epochs:
                    latest = max(epochs.keys())
                    if latest > last_epoch or latest == 0:
                        last_epoch = latest
                        display_progress(epochs, max_epochs=150)

                # Check if training has reached high epochs
                if latest >= 149:
                    print("\n✓ Training completed!")
                    break

            time.sleep(10)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")

if __name__ == '__main__':
    monitor_live()
