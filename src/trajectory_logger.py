"""
Trajectory Logger for Flow Matching Inference
Captures intermediate states along ODE integration trajectories.
"""

import numpy as np
import torch
from typing import Optional, List, Dict, Any
import json
from pathlib import Path


class TrajectoryLogger:
    """Logs intermediate states, velocities, and metadata during ODE integration."""

    def __init__(self, model=None):
        """
        Initialize the trajectory logger.

        Args:
            model: The velocity field model (for reference)
        """
        self.model = model
        self.states = []
        self.times = []
        self.velocities = []
        self.targets = []
        self.metadata = {}

    def log_step(self, s_tau, tau, v_theta_value, target_s1=None, **extra):
        """
        Log a single ODE integration step.

        Args:
            s_tau: Current state at time tau
            tau: Current time value
            v_theta_value: Velocity field value at this state
            target_s1: Target state (for alignment computation)
            **extra: Additional metadata to log
        """
        # Handle tensors
        if isinstance(s_tau, torch.Tensor):
            s_tau = s_tau.detach().cpu().numpy()
        if isinstance(v_theta_value, torch.Tensor):
            v_theta_value = v_theta_value.detach().cpu().numpy()
        if isinstance(target_s1, torch.Tensor):
            target_s1 = target_s1.detach().cpu().numpy()

        # Handle scalar time
        if isinstance(tau, torch.Tensor):
            tau = tau.item() if tau.numel() == 1 else tau.detach().cpu().numpy()

        self.states.append(s_tau)
        self.times.append(tau)
        self.velocities.append(v_theta_value)
        if target_s1 is not None:
            self.targets.append(target_s1)

        # Log extra metadata
        for key, value in extra.items():
            if key not in self.metadata:
                self.metadata[key] = []
            self.metadata[key].append(value)

    def reset(self):
        """Clear all logged data."""
        self.states = []
        self.times = []
        self.velocities = []
        self.targets = []
        self.metadata = {}

    def save(self, filename):
        """
        Save logged trajectories to disk in NPZ format.

        Args:
            filename: Path to save file
        """
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        data = {
            'states': np.array(self.states) if self.states else np.array([]),
            'times': np.array(self.times) if self.times else np.array([]),
            'velocities': np.array(self.velocities) if self.velocities else np.array([]),
            'targets': np.array(self.targets) if self.targets else np.array([]),
        }

        # Save with pickle support for metadata
        np.savez_compressed(filename, **data, allow_pickle=True)

        # Save metadata separately as JSON if present
        if self.metadata:
            metadata_file = filename.replace('.npz', '_metadata.json')
            try:
                metadata_json = {}
                for key, val_list in self.metadata.items():
                    # Convert numpy arrays to lists for JSON serialization
                    metadata_json[key] = [
                        v.tolist() if isinstance(v, np.ndarray) else v
                        for v in val_list
                    ]
                with open(metadata_file, 'w') as f:
                    json.dump(metadata_json, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not save metadata: {e}")

    def load(self, filename):
        """
        Load trajectories from disk.

        Args:
            filename: Path to load file
        """
        data = np.load(filename, allow_pickle=True)

        self.states = [data['states'][i] for i in range(len(data['states']))]
        self.times = data['times'].tolist()
        self.velocities = [data['velocities'][i] for i in range(len(data['velocities']))]

        if len(data['targets']) > 0:
            self.targets = [data['targets'][i] for i in range(len(data['targets']))]

        # Load metadata if exists
        metadata_file = filename.replace('.npz', '_metadata.json')
        if Path(metadata_file).exists():
            try:
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load metadata: {e}")

    def get_trajectory_dict(self) -> Dict[str, Any]:
        """
        Get trajectory as a dictionary.

        Returns:
            Dictionary with 'states', 'times', 'velocities', 'targets' arrays
        """
        return {
            'states': np.array(self.states) if self.states else np.array([]),
            'times': np.array(self.times) if self.times else np.array([]),
            'velocities': np.array(self.velocities) if self.velocities else np.array([]),
            'target': np.array(self.targets[-1]) if self.targets else None,
        }

    def __len__(self):
        """Return number of logged steps."""
        return len(self.states)

    def __repr__(self):
        return (
            f"TrajectoryLogger(steps={len(self.states)}, "
            f"state_shape={self.states[0].shape if self.states else 'empty'}, "
            f"has_targets={len(self.targets) > 0})"
        )
