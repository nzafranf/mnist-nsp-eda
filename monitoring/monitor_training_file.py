#!/usr/bin/env python
"""
Training Progress File Logger
Writes detailed progress to training_progress.txt - check anytime
"""
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

def parse_training_logs(log_file):
    """Extract epoch times and losses from training logs."""
    epochs_data = defaultdict(dict)
    start_time = None

    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return epochs_data, None

    # Find training start time
    for line in lines:
        if "Starting training" in line:
            match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if match:
                start_time = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
                break

    # Parse epoch completions and losses
    for line in lines:
        # Look for epoch completion
        if "Epoch" in line and "100%" in line:
            match = re.search(r'Epoch (\d+):', line)
            if match:
                epoch_num = int(match.group(1))
                epochs_data[epoch_num]['completed'] = True

        # Look for train_loss
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


def write_progress(epochs_data, start_time, max_epochs=24):
    """Write detailed training progress to file."""
    output_file = Path("training_progress.txt")

    completed_epochs = [e for e in epochs_data.keys() if epochs_data[e].get('completed')]

    if not completed_epochs:
        current_epoch = 0
        latest_epoch = 0
    else:
        latest_epoch = max(completed_epochs)
        current_epoch = latest_epoch + 1

    current_time = datetime.now()
    elapsed_total = (current_time - start_time).total_seconds() if start_time else None

    # Build report
    lines = []
    lines.append("=" * 80)
    lines.append("FLOW MATCHING TRAINING PROGRESS")
    lines.append("=" * 80)
    lines.append(f"Last Updated: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Progress
    lines.append("[PROGRESS]")
    lines.append(f"Current Epoch: {current_epoch}/{max_epochs}")
    progress_pct = (current_epoch / max_epochs) * 100
    bar_filled = int(progress_pct / 5)
    bar = "#" * bar_filled + "-" * (20 - bar_filled)
    lines.append(f"[{bar}] {progress_pct:5.1f}%")
    lines.append("")

    # Time tracking
    lines.append("[TIME]")
    if elapsed_total:
        lines.append(f"Total Elapsed:  {format_duration(elapsed_total)}")

    if len(completed_epochs) > 0 and elapsed_total:
        avg_epoch_time = elapsed_total / len(completed_epochs)
        lines.append(f"Avg Time/Epoch: {format_duration(avg_epoch_time)}")

        remaining_epochs = max_epochs - current_epoch
        eta_seconds = remaining_epochs * avg_epoch_time
        lines.append(f"Remaining Time: {format_duration(eta_seconds)}")

        finish_time = current_time + timedelta(seconds=eta_seconds)
        lines.append(f"Estimated Finish: {finish_time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Loss metrics
    lines.append("[LOSS METRICS]")
    if latest_epoch >= 0 and 'loss' in epochs_data[latest_epoch]:
        latest_loss = epochs_data[latest_epoch]['loss']
        lines.append(f"Latest Loss (Epoch {latest_epoch}): {latest_loss:.6f}")

        if len(completed_epochs) > 1:
            prev_epoch = completed_epochs[-2]
            prev_loss = epochs_data[prev_epoch].get('loss', latest_loss)
            improvement = prev_loss - latest_loss
            indicator = "DOWN" if improvement > 0 else "UP"
            lines.append(f"Loss Change: {indicator} {abs(improvement):+.6f}")

        best_loss = min(ep.get('loss', float('inf')) for ep in epochs_data.values() if 'loss' in ep)
        best_epoch = [e for e, d in epochs_data.items() if d.get('loss') == best_loss][0]
        lines.append(f"Best Loss: {best_loss:.6f} (Epoch {best_epoch})")
    lines.append("")

    # Epoch history
    if completed_epochs:
        lines.append("[EPOCH HISTORY]")
        for epoch in sorted(completed_epochs):
            loss = epochs_data[epoch].get('loss', 'N/A')
            if isinstance(loss, float):
                lines.append(f"Epoch {epoch:2d}: Loss = {loss:.6f}")
            else:
                lines.append(f"Epoch {epoch:2d}: Loss = {loss}")
        lines.append("")

    # Checkpoint info
    lines.append("[CHECKPOINTS]")
    lines.append("Saving: Top 12 checkpoints by loss")
    lines.append("Early Stopping: 5 epochs patience")
    lines.append("")
    lines.append("=" * 80)

    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return output_file


def monitor_live():
    """Monitor training and write progress to file."""
    log_file = Path("training_improved.log")
    progress_file = Path("training_progress.txt")
    last_epoch = -1
    start_time = None

    print(f"Started file-based progress logger")
    print(f"Check anytime: {progress_file.absolute()}")
    print("")

    try:
        first_read = True
        update_count = 0

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
                        if latest > last_epoch or update_count % 4 == 0:
                            last_epoch = latest
                            outfile = write_progress(epochs, start_time, max_epochs=24)
                            update_count += 1

                            # Also print brief update to console
                            with open(outfile, 'r', encoding='utf-8') as f:
                                content = f.read()
                                # Print just the progress section
                                for line in content.split('\n')[4:15]:
                                    if line.strip():
                                        print(line)

                            # Check completion
                            if latest >= 23:
                                print("\nTraining completed!")
                                break

                time.sleep(30)  # Update every 30 seconds

            else:
                print("Waiting for training log to be created...")
                time.sleep(5)

    except KeyboardInterrupt:
        print("\nLogger stopped.")


if __name__ == '__main__':
    monitor_live()
