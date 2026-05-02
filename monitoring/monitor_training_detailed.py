#!/usr/bin/env python
"""
Detailed Training Progress Monitor with Time Tracking
Shows elapsed time, per-epoch time, ETA, and loss trends
"""
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import sys

def parse_training_logs(log_file):
    """Extract epoch times and losses from training logs."""
    epochs_data = defaultdict(dict)
    start_time = None

    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return epochs_data, None

    # Find training start time from first log line
    for line in lines:
        if "Starting training" in line:
            match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if match:
                start_time = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
                break

    # Parse epoch completions and losses
    for i, line in enumerate(lines):
        # Look for epoch completion pattern: "Epoch X: 100%"
        if "Epoch" in line and "100%" in line:
            match = re.search(r'Epoch (\d+):', line)
            if match:
                epoch_num = int(match.group(1))
                epochs_data[epoch_num]['completed'] = True

        # Look for train_loss from epoch end
        match = re.search(r'train_loss_epoch=([0-9.]+)', line)
        if match and "Epoch" in line:
            epoch_match = re.search(r'Epoch (\d+)', line)
            if epoch_match:
                epoch_num = int(epoch_match.group(1))
                epochs_data[epoch_num]['loss'] = float(match.group(1))

    return epochs_data, start_time


def format_duration(seconds):
    """Format seconds as HH:MM:SS"""
    if seconds is None or seconds < 0:
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}h {minutes:02d}m {secs:02d}s"


def display_progress(epochs_data, start_time, max_epochs=24):
    """Display detailed training progress with time estimates."""
    if not epochs_data:
        print("\n[WAITING] Training initializing...")
        return None

    # Get latest epoch
    completed_epochs = [e for e in epochs_data.keys() if epochs_data[e].get('completed')]

    if not completed_epochs:
        # Still in first epoch
        current_epoch = 0
        latest_epoch = 0
    else:
        latest_epoch = max(completed_epochs)
        current_epoch = latest_epoch + 1

    current_time = datetime.now()
    elapsed_total = (current_time - start_time).total_seconds() if start_time else None

    print("\n" + "="*80)
    print("FLOW MATCHING TRAINING - DETAILED PROGRESS REPORT")
    print("="*80)

    # Progress summary
    print(f"\n[PROGRESS]")
    print(f"   Current Epoch: {current_epoch}/{max_epochs}")
    progress_pct = (current_epoch / max_epochs) * 100
    bar_filled = int(progress_pct / 5)
    bar = "#" * bar_filled + "-" * (20 - bar_filled)
    print(f"   [{bar}] {progress_pct:5.1f}%")

    # Time tracking
    print(f"\n[TIME]")
    if elapsed_total:
        print(f"   Total Elapsed: {format_duration(elapsed_total)}")

    # Per-epoch analysis
    if len(completed_epochs) > 0:
        epoch_times = []
        for i in range(len(completed_epochs) - 1):
            epoch_times.append((completed_epochs[i+1] - completed_epochs[i], i+1))

        if len(completed_epochs) >= 2:
            avg_epoch_time = elapsed_total / len(completed_epochs) if elapsed_total else 0
            print(f"   Avg Time/Epoch: {format_duration(avg_epoch_time)}")

        # Recent epoch time
        if len(completed_epochs) >= 1:
            recent_time = elapsed_total / len(completed_epochs) if elapsed_total else 0
            print(f"   Last Epoch: {format_duration(recent_time)}")

    # ETA calculation
    if len(completed_epochs) > 0 and elapsed_total:
        avg_epoch_time = elapsed_total / len(completed_epochs)
        remaining_epochs = max_epochs - current_epoch
        eta_seconds = remaining_epochs * avg_epoch_time
        print(f"   Remaining Time: {format_duration(eta_seconds)}")

        finish_time = current_time + timedelta(seconds=eta_seconds)
        print(f"   Estimated Finish: {finish_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Loss metrics
    print(f"\n[LOSS METRICS]")
    if latest_epoch >= 0 and 'loss' in epochs_data[latest_epoch]:
        latest_loss = epochs_data[latest_epoch]['loss']
        print(f"   Latest Loss (Epoch {latest_epoch}): {latest_loss:.6f}")

        # Loss trend
        if len(completed_epochs) > 1:
            prev_epoch = completed_epochs[-2]
            prev_loss = epochs_data[prev_epoch].get('loss', latest_loss)
            improvement = prev_loss - latest_loss
            indicator = "v" if improvement > 0 else "^"
            print(f"   Loss Change: {indicator} {abs(improvement):+.6f}")

        # Best loss so far
        best_loss = min(ep.get('loss', float('inf')) for ep in epochs_data.values() if 'loss' in ep)
        best_epoch = [e for e, d in epochs_data.items() if d.get('loss') == best_loss][0]
        print(f"   Best Loss: {best_loss:.6f} (Epoch {best_epoch})")

    # Checkpoint info
    print(f"\n[CHECKPOINTS]")
    print(f"   Saving top 12 by loss")
    print(f"   Early stopping: if no improvement for 5 epochs")

    print("\n" + "="*80 + "\n")

    return elapsed_total


def monitor_live():
    """Monitor training live with periodic updates."""
    log_file = Path("training_improved.log")
    last_epoch = -1
    start_time = None

    print("🚀 Starting detailed live monitoring...")
    print(f"📝 Log file: {log_file}")
    print(f"⏱️  Updates every 15 seconds\n")

    try:
        first_read = True
        while True:
            if log_file.exists():
                epochs, file_start_time = parse_training_logs(log_file)

                if first_read and file_start_time:
                    start_time = file_start_time
                    first_read = False

                if epochs and start_time:
                    completed = [e for e in epochs.keys() if epochs[e].get('completed')]

                    if completed:
                        latest = max(completed)
                        if latest > last_epoch:
                            last_epoch = latest
                            elapsed = display_progress(epochs, start_time, max_epochs=24)

                            # Check if training complete
                            if latest >= 23:
                                print("✓ Training completed!")
                                break

                time.sleep(15)  # Update every 15 seconds

            else:
                print("⏳ Waiting for training to start and create log file...")
                time.sleep(5)

    except KeyboardInterrupt:
        print("\n\n⏸️  Monitoring stopped by user.")


if __name__ == '__main__':
    monitor_live()
