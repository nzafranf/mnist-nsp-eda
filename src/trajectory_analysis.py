"""
Trajectory Analysis Utilities for Flow Matching EDA
Computes curvature, alignment, leap viability, and other metrics.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


class TrajectoryAnalyzer:
    """Core analysis routines for FM trajectories."""

    @staticmethod
    def compute_curvature_proxy(velocities: np.ndarray) -> np.ndarray:
        """
        Estimate curvature via velocity magnitude differences.

        Args:
            velocities: Velocity vectors (flattened or multi-dimensional)

        Returns:
            (T,) array of curvature proxy values
        """
        # Flatten if multi-dimensional
        velocities = velocities.reshape(velocities.shape[0], -1) if velocities.ndim > 2 else velocities

        if len(velocities) < 2:
            return np.array([0.0] * len(velocities))

        diffs = np.linalg.norm(np.diff(velocities, axis=0), axis=1)
        # Pad to match original length
        curvature = np.concatenate([[diffs[0] if len(diffs) > 0 else 0.0], diffs])
        return curvature

    @staticmethod
    def compute_straightness(states: np.ndarray) -> float:
        """
        Compute straightness metric: chord_distance / arc_length.
        1.0 = perfectly straight, close to 0 = very curved.

        Args:
            states: Trajectory states (flattened or multi-dimensional)

        Returns:
            Straightness score in [0, 1]
        """
        # Flatten if multi-dimensional
        states = states.reshape(states.shape[0], -1) if states.ndim > 2 else states

        if len(states) < 2:
            return 1.0

        chord = np.linalg.norm(states[-1] - states[0], ord=2)
        arc = np.sum([np.linalg.norm(states[i+1] - states[i], ord=2)
                      for i in range(len(states)-1)])

        return chord / (arc + 1e-8)

    @staticmethod
    def compute_alignment(
        states: np.ndarray,
        velocities: np.ndarray,
        target: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute alignment score: how well velocity points toward target.
        Uses cosine similarity, shifted to [0, 1].

        Args:
            states: Trajectory states (flattened or multi-dimensional)
            velocities: Trajectory velocities (flattened or multi-dimensional)
            target: Target state (defaults to states[-1])

        Returns:
            (T,) alignment scores in [0, 1]
        """
        # Flatten multi-dimensional inputs
        states = states.reshape(states.shape[0], -1) if states.ndim > 2 else states
        velocities = velocities.reshape(velocities.shape[0], -1) if velocities.ndim > 2 else velocities

        if target is None:
            target = states[-1]
        else:
            target = target.reshape(-1) if target.ndim > 1 else target

        alignment = []
        for i in range(len(states)):
            direction_to_target = target - states[i]

            # Avoid division by zero
            v_norm = np.linalg.norm(velocities[i]) + 1e-8
            d_norm = np.linalg.norm(direction_to_target) + 1e-8

            cos_angle = np.dot(velocities[i], direction_to_target) / (v_norm * d_norm)
            # Clip to [-1, 1] and shift to [0, 1]
            alignment.append((np.clip(cos_angle, -1, 1) + 1) / 2)

        return np.array(alignment)

    @staticmethod
    def compute_velocity_magnitude(velocities: np.ndarray) -> np.ndarray:
        """Compute L2 norm of velocity vectors."""
        # Flatten if multi-dimensional
        velocities = velocities.reshape(velocities.shape[0], -1) if velocities.ndim > 2 else velocities
        return np.linalg.norm(velocities, axis=1)

    @staticmethod
    def analyze_curvature_alignment_correlation(trajectories_list: List[Dict]) -> Tuple[float, float]:
        """
        Compute correlation between curvature and alignment across trajectories.

        Args:
            trajectories_list: List of dicts with 'states', 'velocities', 'target'

        Returns:
            Tuple of (correlation coefficient, p-value)
        """
        all_curvatures = []
        all_alignments = []

        for traj in trajectories_list:
            states = traj['states']
            velocities = traj['velocities']
            target = traj.get('target')

            curvature = TrajectoryAnalyzer.compute_curvature_proxy(velocities)
            alignment = TrajectoryAnalyzer.compute_alignment(states, velocities, target)

            all_curvatures.extend(curvature)
            all_alignments.extend(alignment)

        corr, pval = pearsonr(all_curvatures, all_alignments)

        return corr, pval

    @staticmethod
    def compute_leap_error(
        state_tau_i: np.ndarray,
        velocity_tau_i: np.ndarray,
        tau_i: float,
        tau_j: float,
        true_state_tau_j: np.ndarray,
        delta_tau: Optional[float] = None
    ) -> Tuple[float, np.ndarray]:
        """
        Compute error for a single Euler step (leap).

        Args:
            state_tau_i: State at tau_i
            velocity_tau_i: Velocity at tau_i
            tau_i: Current time
            tau_j: Target time
            true_state_tau_j: Ground truth state at tau_j
            delta_tau: Time step size (defaults to tau_j - tau_i)

        Returns:
            Tuple of (error, leap_state)
        """
        if delta_tau is None:
            delta_tau = tau_j - tau_i

        leap_state = state_tau_i + delta_tau * velocity_tau_i
        error = np.linalg.norm(leap_state - true_state_tau_j, ord=2)

        return error, leap_state

    @staticmethod
    def evaluate_all_possible_leaps(
        states: np.ndarray,
        velocities: np.ndarray,
        alignment: np.ndarray,
        target: Optional[np.ndarray] = None,
        epsilon: float = 0.5
    ) -> List[Dict]:
        """
        For each timestep, find maximum valid leap size.

        Args:
            states: Trajectory states (flattened or multi-dimensional)
            velocities: Trajectory velocities (flattened or multi-dimensional)
            alignment: (T,) alignment scores
            target: Target state (defaults to states[-1])
            epsilon: Error threshold for valid leap

        Returns:
            List of leap dicts with start_idx, end_idx, leap_size, etc.
        """
        # Flatten if multi-dimensional
        states = states.reshape(states.shape[0], -1) if states.ndim > 2 else states
        velocities = velocities.reshape(velocities.shape[0], -1) if velocities.ndim > 2 else velocities

        if target is None:
            target = states[-1]
        else:
            target = target.reshape(-1) if target.ndim > 1 else target

        T = len(states)
        leap_data = []

        for i in range(T - 1):
            best_j = i

            for j in range(i + 1, T):
                delta_tau = 1.0 / (T - 1)
                error, _ = TrajectoryAnalyzer.compute_leap_error(
                    states[i], velocities[i],
                    i * delta_tau, j * delta_tau,
                    states[j],
                    delta_tau * (j - i)
                )

                avg_alignment = np.mean(alignment[i:j+1])

                # Leap is valid if low error AND high alignment
                is_valid = (error < epsilon) and (avg_alignment > 0.5)

                if is_valid:
                    best_j = j
                else:
                    break

            leap_data.append({
                'start_idx': i,
                'end_idx': best_j,
                'leap_size': best_j - i,
                'avg_alignment': np.mean(alignment[i:best_j+1]),
                'avg_curvature': np.mean(TrajectoryAnalyzer.compute_curvature_proxy(
                    velocities[i:best_j+1]
                )),
            })

        return leap_data

    @staticmethod
    def detect_phase_transition(
        velocities: np.ndarray,
        percentile: int = 75
    ) -> Tuple[int, np.ndarray]:
        """
        Identify transition from low to high information gain phase.

        Args:
            velocities: (T, d) velocity array
            percentile: Threshold percentile for high-gain region

        Returns:
            Tuple of (transition_timestep, velocity_magnitudes)
        """
        v_mag = TrajectoryAnalyzer.compute_velocity_magnitude(velocities)
        threshold = np.percentile(v_mag, percentile)
        high_gain_region = np.where(v_mag > threshold)[0]

        if len(high_gain_region) > 0:
            phase_transition = int(high_gain_region[0])
        else:
            phase_transition = len(velocities) // 2

        return phase_transition, v_mag

    @staticmethod
    def extract_trajectory_features(
        states: np.ndarray,
        velocities: np.ndarray,
        alignment: np.ndarray,
        leap_data: List[Dict],
        phase_transition: int
    ) -> Dict[str, float]:
        """
        Extract hand-crafted features for each trajectory.

        Args:
            states: Trajectory states
            velocities: Trajectory velocities
            alignment: Alignment scores
            leap_data: Leap analysis results
            phase_transition: Phase transition timestep

        Returns:
            Dictionary of trajectory features
        """
        curvature = TrajectoryAnalyzer.compute_curvature_proxy(velocities)

        return {
            'mean_curvature': float(np.mean(curvature)),
            'max_curvature': float(np.max(curvature)),
            'curvature_std': float(np.std(curvature)),
            'mean_alignment': float(np.mean(alignment)),
            'alignment_std': float(np.std(alignment)),
            'straightness': float(TrajectoryAnalyzer.compute_straightness(states)),
            'mean_leap_size': float(np.mean([d['leap_size'] for d in leap_data])),
            'phase_transition_idx': int(phase_transition),
            'complexity': float(phase_transition / len(states)) if len(states) > 0 else 0.0,
        }

    @staticmethod
    def greedy_schedule_discovery(
        states: np.ndarray,
        velocities: np.ndarray,
        alignment: np.ndarray,
        target: Optional[np.ndarray] = None,
        epsilon: float = 0.5,
        max_leap: int = 10
    ) -> List[int]:
        """
        Find schedule greedily by taking longest valid leap at each step.

        Args:
            states: Trajectory states (flattened or multi-dimensional)
            velocities: Trajectory velocities (flattened or multi-dimensional)
            alignment: Alignment scores
            target: Target state
            epsilon: Error threshold
            max_leap: Maximum leap size to try

        Returns:
            List of timestep indices in the schedule
        """
        # Flatten if multi-dimensional
        states = states.reshape(states.shape[0], -1) if states.ndim > 2 else states
        velocities = velocities.reshape(velocities.shape[0], -1) if velocities.ndim > 2 else velocities

        if target is None:
            target = states[-1]
        else:
            target = target.reshape(-1) if target.ndim > 1 else target

        schedule = [0]
        current_idx = 0
        T = len(states)

        while current_idx < T - 1:
            best_leap = 1

            for leap_size in range(1, min(max_leap, T - current_idx)):
                next_idx = current_idx + leap_size

                error, _ = TrajectoryAnalyzer.compute_leap_error(
                    states[current_idx], velocities[current_idx],
                    current_idx / T, next_idx / T,
                    states[next_idx],
                    leap_size / T
                )

                avg_alignment = np.mean(alignment[current_idx:next_idx+1])

                if error < epsilon and avg_alignment > 0.5:
                    best_leap = leap_size
                else:
                    break

            current_idx += best_leap
            if current_idx not in schedule:
                schedule.append(current_idx)

        if (T - 1) not in schedule:
            schedule.append(T - 1)

        schedule = sorted(list(set(schedule)))
        return schedule


class VisualizationUtils:
    """Visualization routines for trajectory analysis."""

    @staticmethod
    def visualize_trajectory_pca(states: np.ndarray, title: str = "Trajectory in PCA space",
                                 save_path: Optional[str] = None) -> Tuple[PCA, np.ndarray]:
        """
        Project trajectory onto 2D PCA space and visualize.

        Args:
            states: Trajectory states (flattened or multi-dimensional)
            title: Plot title
            save_path: Path to save figure (if None, just plot)

        Returns:
            Tuple of (pca_object, projection_array)
        """
        # Flatten states if multi-dimensional (e.g., [T, 1, 1, 28, 28] -> [T, 784])
        states = states.reshape(states.shape[0], -1) if states.ndim > 2 else states

        pca = PCA(n_components=2)
        proj = pca.fit_transform(states)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(proj[:, 0], proj[:, 1], c=np.arange(len(states)),
                            cmap='viridis', s=50, alpha=0.6)
        plt.colorbar(scatter, label='Timestep')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.title(title)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return pca, proj

    @staticmethod
    def plot_leap_profile(leap_data: List[Dict], save_path: Optional[str] = None) -> None:
        """
        Plot leap size vs timestep across trajectory.

        Args:
            leap_data: List of leap dicts
            save_path: Path to save figure
        """
        starts = [d['start_idx'] for d in leap_data]
        sizes = [d['leap_size'] for d in leap_data]
        alignments = [d['avg_alignment'] for d in leap_data]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.plot(starts, sizes, 'o-', linewidth=2, markersize=6, color='steelblue')
        ax1.set_xlabel('Timestep $\\tau_i$', fontsize=12)
        ax1.set_ylabel('Maximum Leap Size', fontsize=12)
        ax1.set_title('Leap Viability Profile', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        scatter = ax2.scatter(starts, sizes, c=alignments, cmap='RdYlGn', s=100, alpha=0.7)
        ax2.set_xlabel('Timestep $\\tau_i$', fontsize=12)
        ax2.set_ylabel('Maximum Leap Size', fontsize=12)
        ax2.set_title('Leap Size Colored by Average Alignment', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Avg Alignment', fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_information_gain(states: np.ndarray, velocities: np.ndarray,
                             leap_data: List[Dict], save_path: Optional[str] = None) -> None:
        """
        Plot velocity magnitude (information gain) and leap boundaries.

        Args:
            states: Trajectory states
            velocities: Trajectory velocities
            leap_data: Leap analysis data
            save_path: Path to save figure
        """
        T = len(states)
        v_mag = TrajectoryAnalyzer.compute_velocity_magnitude(velocities)

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(np.arange(T), v_mag, 'b-', linewidth=2.5, label='||v_θ(s_τ)||', marker='o', markersize=4)

        # Mark leap regions
        leap_regions_drawn = False
        for leap in leap_data:
            start = leap['start_idx']
            end = leap['end_idx']
            if end > start:
                ax.axvspan(start, end, alpha=0.1, color='green')
                if not leap_regions_drawn:
                    ax.axvspan(start, end, alpha=0.1, color='green', label='Valid leap regions')
                    leap_regions_drawn = True

        ax.set_xlabel('Timestep Index', fontsize=12)
        ax.set_ylabel('Velocity Magnitude', fontsize=12)
        ax.set_title('Information Gain Profile with Leap Boundaries', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_curvature_alignment(alignment: np.ndarray, curvature: np.ndarray,
                                save_path: Optional[str] = None) -> None:
        """
        Plot alignment and curvature across trajectory.

        Args:
            alignment: Alignment scores
            curvature: Curvature values
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        timesteps = np.arange(len(alignment))

        ax1.fill_between(timesteps, alignment, alpha=0.6, color='orange', label='Alignment')
        ax1.plot(timesteps, alignment, color='darkorange', linewidth=2)
        ax1.set_ylabel('Alignment Score', fontsize=12)
        ax1.set_title('Alignment and Curvature Along Trajectory', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)

        ax2.fill_between(timesteps, curvature, alpha=0.6, color='steelblue', label='Curvature')
        ax2.plot(timesteps, curvature, color='darkblue', linewidth=2)
        ax2.set_xlabel('Timestep', fontsize=12)
        ax2.set_ylabel('Curvature Proxy', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
