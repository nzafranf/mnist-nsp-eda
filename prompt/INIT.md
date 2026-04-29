# Coding Agent Prompt: First-Principles EDA on Flow Matching Inference Trajectories

## Objective

Perform **exploratory data analysis (EDA) on Flow Matching (FM) inference trajectories** using the MNIST flow-matching codebase. Focus on **trajectory geometry, curvature, information gain, and straightness** rather than SwaraTTS or speech synthesis specifics. The goal is to ground the Neural Schedule Predictor (NSP) concept with empirical observations that don't require formal proofs yet—just strong heuristic observations.

## Setup

**Repository:** `https://github.com/Michedev/flow-matching-mnist.git`

**Task:** Clone, install, and instrument the FM model to **capture and analyze inference trajectories** before and after schedule optimization.

---

## Phase 1: Repository Setup and Instrumentation

### 1.1 Clone and Install
```bash
git clone https://github.com/Michedev/flow-matching-mnist.git
cd flow-matching-mnist
# Install dependencies (likely: pytorch, torchvision, numpy, matplotlib, scikit-learn)
pip install -r requirements.txt  # if exists, else: pip install torch torchvision matplotlib scikit-learn scipy
```

### 1.2 Understand the Codebase
- Locate the main FM model class (likely `FlowMatching`, `FM`, or similar).
- Identify the inference loop (sampling, integration, ODE solver).
- Find the velocity field network (`v_theta` or `velocity_model`).
- Identify the ODE solver (e.g., RK4, Euler, DPM-Solver).

### 1.3 Instrument the Model
**Goal:** Capture intermediate states $(s_\tau, \tau)$ along a trajectory.

Create a wrapper or callback that:
1. **Hooks into the ODE solver** to log states at each step.
2. **Stores intermediate trajectories** to disk (HDF5 or pickle) for later analysis.
3. **Logs metadata:** velocity field values, gradients, alignment angles (defined below).

**Pseudocode:**
```python
class TrajectoryLogger:
    def __init__(self, model):
        self.states = []
        self.times = []
        self.velocities = []
        self.targets = []  # For supervised case, if available
        
    def log_step(self, s_tau, tau, v_theta_value, target_s1=None):
        """Called at each ODE integration step."""
        self.states.append(s_tau.detach().cpu().numpy())
        self.times.append(tau)
        self.velocities.append(v_theta_value.detach().cpu().numpy())
        if target_s1 is not None:
            self.targets.append(target_s1.detach().cpu().numpy())
    
    def save(self, filename):
        """Save logged trajectories to disk."""
        import numpy as np
        data = {
            'states': np.array(self.states),
            'times': np.array(self.times),
            'velocities': np.array(self.velocities),
            'targets': np.array(self.targets) if self.targets else None,
        }
        np.save(filename, data, allow_pickle=True)
```

Integrate this logger into the FM sampling function. Capture **10–50 trajectories** across different random seeds and classes.

---

## Phase 2: Basic Trajectory Visualization

### 2.1 PCA Projection (Geometric Intuition)
**For each trajectory $\{\mathbf{s}_\tau\}_{\tau \in [0,1]}$:**

1. Stack states into a matrix `X ∈ ℝ^(T × d)` (T timesteps, d=784 for MNIST flattened).
2. Apply PCA, retain top 2–3 components (explaining $>95\%$ variance).
3. Plot each trajectory as a 2D/3D curve, colored by timestep index or another metric.

**Questions to answer:**
- Do trajectories look "straight" or "curved"?
- Do they all have similar shape, or are there distinct patterns?
- Are there regions where the trajectory "bends sharply" (high curvature)?

**Code sketch:**
```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def visualize_trajectory_pca(states, title="Trajectory in PCA space"):
    """
    states: (T, 784) array of MNIST images along trajectory
    """
    pca = PCA(n_components=2)
    proj = pca.fit_transform(states)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(proj[:, 0], proj[:, 1], c=np.arange(len(states)), cmap='viridis', s=50)
    plt.colorbar(label='Timestep')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{title.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.close()
```

**Observation to seek:** "Do 10–20 of the first timesteps (navigation phase) show little movement in PCA space, while later timesteps show rapid change?"

---

### 2.2 Epoch-by-Epoch Snapshots
Generate PCA plots for trajectories at different training stages (epoch 1, 10, 50, final). Observe how trajectory "shape" evolves.

---

## Phase 3: Curvature and Alignment Analysis

### 3.1 Local Curvature Estimation
**Heuristic: Without formal differential geometry, estimate curvature via finite differences.**

For each timestep $\tau_i$:

$$\text{Curvature estimate: } \kappa_i \approx \frac{\|\dot{s}_{i+1} - \dot{s}_i\|}{\Delta\tau} = \frac{\|\mathbf{v}(\mathbf{s}_{i+1}) - \mathbf{v}(\mathbf{s}_i)\|}{\Delta\tau}$$

**Simpler heuristic:** Just compute the **velocity magnitude difference**:

$$\text{Velocity change: } \Delta v_i = \|\mathbf{v}_{\tau_i} - \mathbf{v}_{\tau_{i-1}}\|_2$$

High $\Delta v$ indicates the velocity field is changing rapidly (sharp turn ahead).

**Code:**
```python
def compute_curvature_proxy(velocities):
    """
    velocities: (T, d) array
    Returns: (T,) array of curvature proxies
    """
    diffs = np.linalg.norm(np.diff(velocities, axis=0), axis=1)
    # Pad to match original length
    curvature = np.concatenate([[diffs[0]], diffs])
    return curvature

def compute_straightness(states):
    """
    Straightness = chord_distance / arc_length
    Values close to 1 = straight, close to 0 = very curved
    """
    chord = np.linalg.norm(states[-1] - states[0], ord=2)
    arc = np.sum([np.linalg.norm(states[i+1] - states[i], ord=2) for i in range(len(states)-1)])
    return chord / (arc + 1e-8)
```

### 3.2 Alignment Score (the key metric)
**Definition (heuristic, no $s_1$ dependency):**

At each step $\tau_i$, the **alignment score** measures how well the velocity vector $\mathbf{v}(\mathbf{s}_{\tau_i})$ points toward the final state $\mathbf{s}_1$:

$$\text{Alignment}(\tau_i) = \cos\left(\angle\left(\mathbf{v}(\mathbf{s}_{\tau_i}), \; \mathbf{s}_1 - \mathbf{s}_{\tau_i}\right)\right)$$

**Practical simplification (training regime):** 
- During EDA, you have access to ground-truth targets (e.g., "generate digit 5").
- Use the true target $\mathbf{s}_1$ to compute alignment.
- **Key observation:** High alignment = velocity points toward target, low error on a leap.

**Code:**
```python
def compute_alignment(states, velocities, target=None):
    """
    states: (T, d) trajectory
    velocities: (T, d) velocity field values
    target: (d,) true target state (or use states[-1])
    
    Returns: (T,) alignment scores in [0, 1] (cosine similarity shifted)
    """
    if target is None:
        target = states[-1]
    
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
```

### 3.3 Compute Alignment vs. Curvature Correlation
**Question:** Does high curvature correlate with low alignment (velocity misaligned from target)?

```python
from scipy.stats import pearsonr

def analyze_curvature_alignment_correlation(trajectories_list):
    """
    trajectories_list: list of dicts, each with 'states', 'velocities', 'target'
    """
    all_curvatures = []
    all_alignments = []
    
    for traj in trajectories_list:
        states = traj['states']
        velocities = traj['velocities']
        target = traj['target']
        
        curvature = compute_curvature_proxy(velocities)
        alignment = compute_alignment(states, velocities, target)
        
        all_curvatures.extend(curvature)
        all_alignments.extend(alignment)
    
    corr, pval = pearsonr(all_curvatures, all_alignments)
    print(f"Correlation (curvature vs. alignment): {corr:.3f}, p={pval:.2e}")
    print(f"Interpretation: {'Strong' if abs(corr) > 0.6 else 'Weak'} "
          f"{'negative' if corr < 0 else 'positive'} correlation")
    
    return corr, pval
```

**Expected observation:** High curvature ↔ low alignment (negative correlation $< -0.6$).

---

## Phase 4: Leap Viability Analysis

### 4.1 Define and Test Leaps
**Leap:** Skip from timestep $\tau_i$ to $\tau_j$ (where $j > i+1$), using a single RK4 step with large $\Delta\tau = \tau_j - \tau_i$.

**Metrics for each leap:**

1. **Leap error:** $\text{Error}(\tau_i, \tau_j) = \|\mathbf{s}_{\tau_j}^{\text{leap}} - \mathbf{s}_{\tau_j}^{\text{true}}\|_2$
   - True state taken from fine-grained trajectory.
   - Leap state computed with single Euler/RK4 step.

2. **Leap alignment:** Average alignment over $[\tau_i, \tau_j]$.

3. **Leap validity:** Is error $< \epsilon$ (threshold)?

**Code:**
```python
def compute_leap_error(state_tau_i, velocity_tau_i, tau_i, tau_j, true_state_tau_j, delta_tau=None):
    """
    Simulate a single RK4 step (the "leap").
    """
    if delta_tau is None:
        delta_tau = tau_j - tau_i
    
    # Simple Euler step
    leap_state = state_tau_i + delta_tau * velocity_tau_i
    
    # Error relative to ground truth
    error = np.linalg.norm(leap_state - true_state_tau_j, ord=2)
    return error, leap_state

def evaluate_all_possible_leaps(states, velocities, alignment, target, epsilon=0.5):
    """
    For each starting position, find the longest valid leap.
    """
    T = len(states)
    leap_data = []
    
    for i in range(T - 1):
        # Try all possible endpoints j > i
        best_j = i  # At minimum, we can stay at current step
        
        for j in range(i + 1, T):
            delta_tau = 1.0 / (T - 1)  # Normalized step
            error, _ = compute_leap_error(
                states[i], velocities[i], i * delta_tau, j * delta_tau,
                states[j], delta_tau * (j - i)
            )
            
            # Check alignment over interval
            avg_alignment = np.mean(alignment[i:j+1])
            
            # Leap is valid if error < epsilon AND alignment is high
            is_valid = (error < epsilon) and (avg_alignment > 0.5)
            
            if is_valid:
                best_j = j
            else:
                # Once invalid, further jumps are likely invalid
                break
        
        leap_data.append({
            'start_idx': i,
            'end_idx': best_j,
            'leap_size': best_j - i,
            'avg_alignment': np.mean(alignment[i:best_j+1]),
            'avg_curvature': np.mean(compute_curvature_proxy(velocities[i:best_j+1])),
        })
    
    return leap_data
```

### 4.2 Visualize Leap Size vs. Timestep
**Plot:** X-axis = timestep $\tau_i$, Y-axis = maximum leap size $(\tau_j - \tau_i)$.

**Expected observation:**
- Early timesteps (navigation phase): small leaps allowed (high curvature).
- Middle timesteps: larger leaps.
- Late timesteps (refinement): variable, may shrink again.

**Code:**
```python
def plot_leap_profile(leap_data):
    """leap_data: list of dicts from evaluate_all_possible_leaps"""
    starts = [d['start_idx'] for d in leap_data]
    sizes = [d['leap_size'] for d in leap_data]
    alignments = [d['avg_alignment'] for d in leap_data]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Leap size
    ax1.plot(starts, sizes, 'o-', linewidth=2, markersize=6)
    ax1.set_xlabel('Timestep $\\tau_i$')
    ax1.set_ylabel('Maximum Leap Size $\\tau_j - \\tau_i$')
    ax1.set_title('Leap Viability Profile Across Trajectory')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Alignment (color code)
    scatter = ax2.scatter(starts, sizes, c=alignments, cmap='RdYlGn', s=80)
    ax2.set_xlabel('Timestep $\\tau_i$')
    ax2.set_ylabel('Maximum Leap Size $\\tau_j - \\tau_i$')
    ax2.set_title('Leap Size colored by Average Alignment')
    plt.colorbar(scatter, ax=ax2, label='Avg Alignment')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('leap_profile.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Mean leap size: {np.mean(sizes):.2f}")
    print(f"Leap size range: [{np.min(sizes)}, {np.max(sizes)}]")
```

---

## Phase 5: Information Gain Proxy

### 5.1 Velocity Magnitude as Information Proxy
**Heuristic:** High velocity $\|\mathbf{v}_\tau\|$ = high information transfer in that step.

```python
def compute_information_gain_proxy(velocities):
    """
    Proxy for information gain: ||v_theta(s_tau)||
    """
    v_magnitudes = np.linalg.norm(velocities, axis=1)
    return v_magnitudes

def plot_information_gain(states, velocities, leap_data):
    """
    Plot trajectory information gain and overlay leap boundaries.
    """
    T = len(states)
    v_mag = compute_information_gain_proxy(velocities)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot velocity magnitude
    ax.plot(np.arange(T), v_mag, 'b-', linewidth=2, label='||v_θ||')
    
    # Mark leap boundaries
    for leap in leap_data:
        start = leap['start_idx']
        end = leap['end_idx']
        if end > start:
            ax.axvspan(start, end, alpha=0.1, color='green', label='Valid leap region' if start == leap_data[0]['start_idx'] else '')
    
    ax.set_xlabel('Timestep Index')
    ax.set_ylabel('Velocity Magnitude ||v_θ(s_τ)||')
    ax.set_title('Information Gain Proxy and Leap Boundaries')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig('information_gain.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Statistics
    print(f"Max information gain: {np.max(v_mag):.3f} at timestep {np.argmax(v_mag)}")
    print(f"Min information gain: {np.min(v_mag):.3f} at timestep {np.argmin(v_mag)}")
    print(f"Information gain concentrated in middle {np.where(v_mag > 0.5 * np.max(v_mag))[0]}")
```

### 5.2 Navigation vs. Refinement Phase Detection
**Hypothesis:** Early phase has low $\|\mathbf{v}\|$ (broad uncertainty), late phase has variable $\|\mathbf{v}\|$ (fine details).

```python
def detect_phase_transition(velocities, percentile=75):
    """
    Identify where trajectory transitions from low to high information gain.
    """
    v_mag = compute_information_gain_proxy(velocities)
    threshold = np.percentile(v_mag, percentile)
    high_gain_region = np.where(v_mag > threshold)[0]
    
    if len(high_gain_region) > 0:
        phase_transition = high_gain_region[0]
    else:
        phase_transition = len(velocities) // 2
    
    return phase_transition, v_mag
```

---

## Phase 6: Binary Search Oracle (Simple Version)

### 6.1 Greedy Schedule Discovery
**Algorithm:** Greedily find the longest admissible leap at each step.

```python
def greedy_schedule_discovery(states, velocities, alignment, target, epsilon=0.5, max_leap=10):
    """
    Find a greedy schedule by taking the longest valid leap at each step.
    Returns: schedule as list of timestep indices.
    """
    schedule = [0]  # Start at time 0
    current_idx = 0
    T = len(states)
    
    while current_idx < T - 1:
        # Try increasingly larger leaps
        best_leap = 1  # At least move to next step
        
        for leap_size in range(1, min(max_leap, T - current_idx)):
            next_idx = current_idx + leap_size
            
            # Check validity
            error, _ = compute_leap_error(
                states[current_idx],
                velocities[current_idx],
                current_idx / T,
                next_idx / T,
                states[next_idx],
                leap_size / T
            )
            
            avg_alignment = np.mean(alignment[current_idx:next_idx+1])
            
            if error < epsilon and avg_alignment > 0.5:
                best_leap = leap_size
            else:
                break
        
        current_idx += best_leap
        schedule.append(current_idx)
    
    schedule.append(T - 1)  # Ensure we reach the end
    schedule = list(set(schedule))  # Remove duplicates
    schedule.sort()
    
    return schedule

def compare_schedules(schedule_uniform, schedule_greedy):
    """
    Compare uniform vs. greedy scheduling.
    """
    print(f"Uniform schedule: {len(schedule_uniform)} steps")
    print(f"Greedy schedule: {len(schedule_greedy)} steps")
    print(f"Speedup: {len(schedule_uniform) / len(schedule_greedy):.2f}x")
```

---

## Phase 7: Clustering Analysis (NSP Learnability)

### 7.1 Trajectory Feature Extraction
**For each trajectory, extract simple features:**

```python
def extract_trajectory_features(states, velocities, alignment, leap_data, phase_transition):
    """
    Extract hand-crafted features that might predict good schedules.
    """
    features = {
        'mean_curvature': np.mean(compute_curvature_proxy(velocities)),
        'max_curvature': np.max(compute_curvature_proxy(velocities)),
        'curvature_std': np.std(compute_curvature_proxy(velocities)),
        'mean_alignment': np.mean(alignment),
        'alignment_std': np.std(alignment),
        'straightness': compute_straightness(states),
        'mean_leap_size': np.mean([d['leap_size'] for d in leap_data]),
        'phase_transition_idx': phase_transition,
        'complexity': phase_transition / len(states),  # How early does high-gain begin?
    }
    return features
```

### 7.2 Cluster Trajectories
**Question:** Can we group trajectories by their geometry, and would these groups benefit from different schedules?

```python
from sklearn.cluster import KMeans

def cluster_trajectories(feature_list, n_clusters=3):
    """
    feature_list: list of feature dicts from extract_trajectory_features
    """
    import pandas as pd
    
    # Convert to matrix
    df = pd.DataFrame(feature_list)
    X = df.values
    
    # Standardize
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    print(f"Trajectory clusters found:")
    for i in range(n_clusters):
        count = np.sum(clusters == i)
        print(f"  Cluster {i}: {count} trajectories")
        print(f"    Example features: {df[clusters == i].mean()}")
    
    return clusters, kmeans
```

### 7.3 Do different clusters need different schedules?

```python
def analyze_schedule_by_cluster(trajectories, clusters, target_classes):
    """
    For each cluster, see if schedules are different.
    """
    cluster_schedules = {i: [] for i in set(clusters)}
    
    for traj, cluster_id in zip(trajectories, clusters):
        states = traj['states']
        velocities = traj['velocities']
        alignment = compute_alignment(states, velocities, traj['target'])
        schedule = greedy_schedule_discovery(states, velocities, alignment, traj['target'])
        cluster_schedules[cluster_id].append(len(schedule))
    
    print("Schedule complexity by cluster:")
    for cluster_id, schedule_lengths in cluster_schedules.items():
        mean_len = np.mean(schedule_lengths)
        print(f"  Cluster {cluster_id}: avg {mean_len:.1f} steps")
```

---

## Phase 8: Ablation Study - Oracle vs. Naive Schedules

### 8.1 Uniform, Greedy, and Random Schedules
Compare three schedules on the same trajectories:

1. **Uniform:** $\tau_i = i / K$ for fixed $K \in \{4, 8, 16\}$
2. **Greedy oracle:** Binary-search greedy from Phase 6
3. **Random:** Random subset of timesteps

```python
def evaluate_all_schedules(states, velocities, target, epsilon=0.5):
    """
    Evaluate uniform, greedy, and random schedules.
    """
    T = len(states)
    results = {}
    
    # Uniform schedules
    for K in [4, 8, 16]:
        uniform_schedule = np.linspace(0, T-1, K, dtype=int)
        # Evaluate error (simplified)
        results[f'uniform_K={K}'] = {
            'steps': K,
            # Error would be computed via actual ODE integration
        }
    
    # Greedy
    alignment = compute_alignment(states, velocities, target)
    leap_data = evaluate_all_possible_leaps(states, velocities, alignment, target, epsilon)
    greedy_schedule = greedy_schedule_discovery(states, velocities, alignment, target, epsilon)
    results['greedy'] = {
        'steps': len(greedy_schedule),
    }
    
    # Random
    random_schedule = np.sort(np.random.choice(T, K=8, replace=False))
    results['random'] = {
        'steps': len(random_schedule),
    }
    
    return results
```

---

## Phase 9: Summary and Key Observations

Generate a summary report capturing:

1. **Trajectory straightness:** How much do trajectories deviate from a straight line?
2. **Curvature-alignment correlation:** Is there a strong relationship?
3. **Information gain profile:** Where is most gain concentrated?
4. **Phase transitions:** When does the trajectory switch from navigation to refinement?
5. **Leap feasibility:** How much can steps be increased without error growth?
6. **Greedy efficiency:** How much speedup does greedy scheduling achieve?
7. **Trajectory diversity:** How different are trajectories from one digit class to another?
8. **Clustering insights:** Can trajectory geometry predict useful schedule properties?

---

## Expected Outputs

Create a results folder with:
- `trajectory_pca_*.png` — PCA visualizations of 10 trajectories
- `leap_profile_*.png` — Leap size vs. timestep plots
- `information_gain_*.png` — Velocity magnitude and phase transitions
- `cluster_analysis.json` — Cluster statistics
- `summary_report.txt` — High-level findings

**Example summary:**
```
=== EDA Summary ===
Total trajectories analyzed: 50
Average straightness: 0.72 (±0.12)
Curvature-alignment correlation: -0.68 (p < 0.001)
Phase transition occurs at timestep: 40% into trajectory
Average leap size (greedy): 3.2 steps (±1.5)
Greedy speedup vs. uniform (K=16): 2.1x
Trajectory clusters: 3 groups identified
  - Group A (simple digits): 16 trajectories, mean steps=5.2
  - Group B (complex): 20 trajectories, mean steps=7.8
  - Group C (mixed): 14 trajectories, mean steps=6.3
```

---

## Notes for Coding Agent

1. **Do NOT try to use formal proofs**—just compute and visualize.
2. **Be pragmatic:** If an observation is "almost true" for 80% of trajectories, report it.
3. **Focus on FM trajectory dynamics, not MNIST classification**—we care about geometry, not accuracy.
4. **Instrument carefully:** The ODE solver itself; log everything at each step.
5. **Use simple heuristics:** Curvature from velocity differences, alignment from dot products.
6. **Visualize everything:** Plots are worth 1000 equations.
7. **Save intermediate results:** Don't recompute trajectories.
8. **If stuck:** Print diagnostics (shapes, ranges, NaNs) and adjust thresholds.

---

## Success Criteria

✓ Can visualize 10+ trajectories in PCA space  
✓ Correlation between curvature and alignment is computed and reported  
✓ Leap feasibility analysis shows realistic leap sizes  
✓ Greedy schedule achieves measurable speedup  
✓ Clustering reveals at least 2 distinct trajectory types  
✓ Summary report with 5+ key observations that ground NSP design  

This EDA is **not meant to prove NSP optimal**—it's meant to show that the geometry is real and worth exploiting.