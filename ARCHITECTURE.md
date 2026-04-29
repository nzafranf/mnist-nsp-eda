# 🏗️ Architecture & Data Flow Guide

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     FLOW MATCHING EDA SYSTEM                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────┐
│   Input: Model      │
│  Checkpoint or      │
│  Train from scratch │
└──────────┬──────────┘
           │
           ▼
    ┌──────────────────────────┐
    │  Phase 1: Generation     │
    │  - Load/train model      │
    │  - Create ODE logger     │
    │  - Capture trajectories  │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │ Trajectory Data (NPZ)    │
    │ - States: T×D array      │
    │ - Velocities: T×D array  │
    │ - Timesteps: T array     │
    │ - Targets: D array       │
    └──────────┬───────────────┘
               │
         ┌─────┴─────┬──────────┬──────────┬──────────┐
         │           │          │          │          │
         ▼           ▼          ▼          ▼          ▼
    ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
    │Phase 2 │ │Phase 3 │ │Phase 4 │ │Phase 5 │ │Phase 6 │
    │  PCA   │ │Curve & │ │  Leap  │ │  Info  │ │Greedy  │
    │ Visual │ │ Align  │ │Viability│ │  Gain  │ │Schedule│
    └────────┘ └────────┘ └────────┘ └────────┘ └────────┘
         │           │          │          │          │
         └─────┬─────┴──────────┴──────────┴──────────┘
               │
               ▼
    ┌──────────────────────────┐
    │  Phase 7: Clustering     │
    │  - Extract features      │
    │  - K-means clustering    │
    │  - Cluster analysis      │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │  Phase 8: Ablation       │
    │  - Compare schedules     │
    │  - Uniform vs Greedy     │
    │  - Statistics            │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │  Phase 9: Report         │
    │  - Synthesize findings   │
    │  - Generate summary      │
    │  - Actionable insights   │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │  Output: Results folder  │
    │  - 30+ PNG plots         │
    │  - 9 JSON summaries      │
    │  - Final report (TXT)    │
    └──────────────────────────┘
```

---

## Module Dependencies

```
run_eda.py (entry point)
    │
    ├─► models.fm.ImageFlowMatcher
    │   └─► flow_matching.solver.ODESolver
    │
    ├─► src.quick_train.quick_train()
    │
    └─► src.eda_master.EDAMaster
        │
        ├─► src.instrumented_generation
        │   ├─► TrajectoryLogger
        │   ├─► InstrumentedODESolver
        │   │   └─► ImageFlowMatcher
        │   └─► load_trajectories()
        │
        └─► src.trajectory_analysis
            ├─► TrajectoryAnalyzer (static methods)
            │   ├─► compute_curvature_proxy()
            │   ├─► compute_alignment()
            │   ├─► evaluate_all_possible_leaps()
            │   ├─► greedy_schedule_discovery()
            │   └─► extract_trajectory_features()
            │
            └─► VisualizationUtils (static methods)
                ├─► visualize_trajectory_pca()
                ├─► plot_leap_profile()
                ├─► plot_information_gain()
                └─► plot_curvature_alignment()
```

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────┐
│                     PHASE 1: GENERATION                   │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
                  ┌─────────────────┐
                  │ Model Checkpoint │
                  │ (*.ckpt or *.pt) │
                  └────────┬────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
        ┌──────────────┐          ┌──────────────┐
        │ Load Model   │          │ Create Solver│
        │ on Device    │          │ + Logger     │
        └────────┬─────┘          └──────┬───────┘
                 │                       │
                 └───────────┬───────────┘
                             ▼
                  ┌──────────────────────┐
                  │ ODE Integration Loop │
                  │ for t=0 to t=1       │
                  └──────────┬───────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
      ┌──────────┐    ┌──────────┐    ┌──────────┐
      │State s_t │    │Velocity  │    │Metadata  │
      │(image)   │    │v(s_t)    │    │timestamp │
      └─────┬────┘    └────┬─────┘    └────┬─────┘
            │              │              │
            └──────────────┬──────────────┘
                           ▼
                  ┌──────────────────────┐
                  │  TrajectoryLogger    │
                  │  Accumulates steps   │
                  └──────────┬───────────┘
                             │
              ┌──────────────┴───────────────┐
              ▼                              ▼
         (repeat for                    Save to NPZ
          each step)                        │
                                            ▼
                                  ┌────────────────┐
                                  │ trajectory_NNN │
                                  │     .npz       │
                                  └────────────────┘
```

---

## Analysis Pipeline (Phases 2-9)

```
┌────────────────────────────────────────────────────────────┐
│                   TRAJECTORY DATA (NPZ)                    │
│  states[T,D], times[T], velocities[T,D], targets[D]        │
└────────────────────────┬─────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
    ┌────────┐      ┌────────┐      ┌────────┐
    │States  │      │Times   │      │Velocity│
    │T×D     │      │T       │      │T×D     │
    └────┬───┘      └────┬───┘      └────┬───┘
         │               │              │
         └───────────────┬──────────────┘
                         │
    ┌────────────────────▼──────────────────────┐
    │       PHASE 2: PCA Visualization          │
    │  • Fit PCA to states                      │
    │  • Project to 2D                          │
    │  • Color by timestep                      │
    │  • Save PNG plot                          │
    └────────────────────┬──────────────────────┘
                         │
    ┌────────────────────▼──────────────────────┐
    │    PHASE 3: Curvature & Alignment         │
    │  • Δv = ||v[i+1] - v[i]||  (curvature)   │
    │  • a = cos(angle(v, target - s))          │
    │  • straightness = chord / arc             │
    │  • correlation(Δv, a)                     │
    │  • Save plots + JSON                      │
    └────────────────────┬──────────────────────┘
                         │
    ┌────────────────────▼──────────────────────┐
    │      PHASE 4: Leap Viability              │
    │  • For each timestep i:                   │
    │    - Try all jumps j > i+1                │
    │    - Euler step: s' = s[i] + Δt·v[i]     │
    │    - Error = ||s' - s[j]||                │
    │    - Valid if error < ε AND align > 0.5  │
    │    - Record max leap                      │
    │  • Save leap profile plots                │
    └────────────────────┬──────────────────────┘
                         │
    ┌────────────────────▼──────────────────────┐
    │   PHASE 5: Information Gain                │
    │  • ||v[i]|| = velocity magnitude          │
    │  • Find phase transition:                 │
    │    - Where 75th percentile exceeded       │
    │  • Identify navigation vs refinement      │
    │  • Save information plots                 │
    └────────────────────┬──────────────────────┘
                         │
    ┌────────────────────▼──────────────────────┐
    │    PHASE 6: Greedy Scheduling             │
    │  • schedule = [0]                         │
    │  • current = 0                            │
    │  • while current < T-1:                   │
    │    - Find max leap from current           │
    │    - Add to schedule                      │
    │    - Move current forward                 │
    │  • Compare vs uniform (K=4,8,16)          │
    │  • Save schedule stats                    │
    └────────────────────┬──────────────────────┘
                         │
    ┌────────────────────▼──────────────────────┐
    │  PHASE 7: Clustering                      │
    │  • Extract features for each traj:        │
    │    - mean_curvature                       │
    │    - max_curvature                        │
    │    - mean_alignment                       │
    │    - straightness                         │
    │    - mean_leap_size                       │
    │    - complexity                           │
    │  • Standardize features                   │
    │  • K-means clustering (k=2-3)             │
    │  • Analyze cluster properties             │
    │  • Save cluster results                   │
    └────────────────────┬──────────────────────┘
                         │
    ┌────────────────────▼──────────────────────┐
    │    PHASE 8: Ablation Study                │
    │  • Evaluate multiple schedules:           │
    │    - Uniform K=4,8,16                     │
    │    - Greedy (from Phase 6)                │
    │    - Random                               │
    │  • Compare step counts                    │
    │  • Save comparison stats                  │
    └────────────────────┬──────────────────────┘
                         │
    ┌────────────────────▼──────────────────────┐
    │   PHASE 9: Summary Report                 │
    │  • Aggregate all metrics                  │
    │  • Synthesize findings                    │
    │  • Generate text report                   │
    │  • List key observations                  │
    │  • NSP design implications                │
    │  • Save summary_report.txt                │
    └──────────────────────────────────────────┘
```

---

## Class Hierarchy

```
TrajectoryAnalyzer
├─ compute_curvature_proxy(velocities)
├─ compute_straightness(states)
├─ compute_alignment(states, velocities, target)
├─ compute_velocity_magnitude(velocities)
├─ analyze_curvature_alignment_correlation(trajectories_list)
├─ compute_leap_error(state_i, velocity_i, tau_i, tau_j, true_state_j)
├─ evaluate_all_possible_leaps(states, velocities, alignment, target)
├─ detect_phase_transition(velocities, percentile)
├─ extract_trajectory_features(states, velocities, alignment, leap_data)
└─ greedy_schedule_discovery(states, velocities, alignment, target)

VisualizationUtils
├─ visualize_trajectory_pca(states, title, save_path)
├─ plot_leap_profile(leap_data, save_path)
├─ plot_information_gain(states, velocities, leap_data, save_path)
└─ plot_curvature_alignment(alignment, curvature, save_path)

TrajectoryLogger
├─ log_step(s_tau, tau, v_theta_value, target_s1, **extra)
├─ reset()
├─ save(filename)
├─ load(filename)
├─ get_trajectory_dict()
└─ __len__()

InstrumentedODESolver
├─ velocity_wrapper(t, x)
└─ sample(time_grid, x_init, method, step_size, ...)

EDAMaster
├─ phase1_setup_and_generation(model, num_trajectories, num_steps)
├─ phase2_pca_visualization()
├─ phase3_curvature_alignment()
├─ phase4_leap_viability()
├─ phase5_information_gain()
├─ phase6_greedy_schedule()
├─ phase7_clustering()
├─ phase8_ablation_study()
├─ phase9_summary_report()
└─ run_full_eda(model, num_trajectories)
```

---

## Execution Flow

```
┌─────────────────┐
│  run_eda.py     │
│  (entry point)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Parse command-line arguments       │
│  - checkpoint path                  │
│  - num_trajectories                 │
│  - num_epochs                       │
│  - results_dir                      │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Check for checkpoint                │
│                                      │
│  if not exists:                      │
│    train with quick_train.py        │
│    (3-5 epochs)                     │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Load model                          │
│  - Handle .ckpt (Lightning)         │
│  - Handle .pt (state_dict)          │
│  - Move to device                    │
│  - Set to eval mode                  │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Create EDAMaster instance           │
│  eda = EDAMaster(results_dir)        │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Call eda.run_full_eda()             │
│  Sequential execution of phases:     │
│  1. Generate trajectories            │
│  2. PCA visualization                │
│  3. Curvature & alignment            │
│  4. Leap viability                   │
│  5. Information gain                 │
│  6. Greedy scheduling                │
│  7. Clustering                       │
│  8. Ablation study                   │
│  9. Summary report                   │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Output generation                   │
│  - PNG plots saved                   │
│  - JSON summaries written            │
│  - Report text generated             │
│  - Status printed to console         │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  results/ folder ready              │
│  - phase1-9 subdirectories          │
│  - All visualizations               │
│  - All data and reports             │
└─────────────────────────────────────┘
```

---

## File I/O Operations

```
INPUT FILES:
├─ Checkpoint (*.ckpt or *.pt)
│  └─ Loaded by EDAMaster.phase1_setup_and_generation()
│
└─ MNIST dataset (auto-downloaded)
   └─ Downloaded by quick_train.py

OUTPUT FILES:
├─ results/
│  │
│  ├─ phase1/
│  │  ├─ trajectory_000.npz
│  │  ├─ trajectory_001.npz
│  │  ├─ ...
│  │  └─ summary.json
│  │
│  ├─ phase2/
│  │  ├─ trajectory_pca_00.png
│  │  ├─ trajectory_pca_01.png
│  │  ├─ ...
│  │  └─ summary.json
│  │
│  ├─ phase3/
│  │  ├─ curvature_alignment_00.png
│  │  ├─ curvature_alignment_01.png
│  │  ├─ ...
│  │  └─ summary.json
│  │
│  ├─ phase4/
│  │  ├─ leap_profile_00.png
│  │  ├─ leap_profile_01.png
│  │  ├─ ...
│  │  └─ summary.json
│  │
│  ├─ phase5/
│  │  ├─ information_gain_00.png
│  │  ├─ information_gain_01.png
│  │  ├─ ...
│  │  └─ summary.json
│  │
│  ├─ phase6/
│  │  └─ summary.json
│  │
│  ├─ phase7/
│  │  └─ summary.json
│  │
│  ├─ phase8/
│  │  └─ summary.json
│  │
│  └─ phase9/
│     ├─ summary_report.txt
│     └─ summary.json
│
├─ checkpoints/
│  └─ fm-best-*.ckpt
│
├─ data/
│  ├─ MNIST/
│  │  ├─ processed/
│  │  └─ raw/
│  │
│  └─ fid_ref_paths.pkl (for FID computation)
│
└─ tb_logs/ (TensorBoard logs from training)
```

---

## Metric Computation Pipeline

```
Raw Trajectory Data
├─ States: (T, 28, 28)
├─ Velocities: (T, 28, 28)
├─ Times: (T,)
└─ Target: (28, 28)

┌─────────────────────────────────────┐
│  Geometric Metrics                   │
├─────────────────────────────────────┤
│ • Curvature Proxy                   │
│   κ_i = ||v[i+1] - v[i]||           │
│                                      │
│ • Alignment                          │
│   A_i = (cos(∠(v, target-s)) + 1)/2│
│                                      │
│ • Straightness                       │
│   S = ||s_final - s_0|| / arc_length│
└──────────┬───────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  Phase-Based Metrics                 │
├─────────────────────────────────────┤
│ • Velocity Magnitude: ||v_i||        │
│                                      │
│ • Phase Transition Idx               │
│   = argmax(||v|| > 75th percentile)  │
│                                      │
│ • Navigation Phase: low ||v||        │
│ • Refinement Phase: high ||v||       │
└──────────┬───────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  Leap Viability Metrics              │
├─────────────────────────────────────┤
│ • Leap Error for each (i,j):         │
│   E(i,j) = ||s[i] + Δt·v[i] - s[j]|| │
│                                      │
│ • Max Valid Leap at each i:          │
│   max j where E(i,j) < ε             │
│                                      │
│ • Leap Profile: leap_size vs time    │
└──────────┬───────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  Schedule Metrics                    │
├─────────────────────────────────────┤
│ • Greedy Schedule                   │
│   [0, best_j_1, best_j_2, ..., T-1] │
│                                      │
│ • Schedule Length                    │
│   Number of timesteps needed         │
│                                      │
│ • Speedup Ratio                      │
│   uniform_length / greedy_length     │
└──────────┬───────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  Feature Vector (for Clustering)     │
├─────────────────────────────────────┤
│ f = [                                │
│   mean_curvature,                   │
│   max_curvature,                    │
│   curvature_std,                    │
│   mean_alignment,                   │
│   alignment_std,                    │
│   straightness,                     │
│   mean_leap_size,                   │
│   phase_transition_idx,             │
│   complexity                        │
│ ]                                    │
└──────────┬───────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  Clustering Results                  │
├─────────────────────────────────────┤
│ • K-means (k=2-3)                   │
│ • Cluster assignments                │
│ • Cluster centroids                  │
│ • Intra-cluster statistics           │
└──────────────────────────────────────┘
```

---

## Summary of Key Components

| Component | Type | Purpose | Output |
|-----------|------|---------|--------|
| **TrajectoryLogger** | Class | Log ODE steps | NPZ files |
| **InstrumentedODESolver** | Class | Wrap ODE solver | Logged trajectories |
| **TrajectoryAnalyzer** | Class | Compute metrics | Numerical results |
| **VisualizationUtils** | Class | Create plots | PNG images |
| **EDAMaster** | Class | Orchestrate phases | All outputs |
| **run_eda.py** | Script | Main entry | Complete results |
| **quick_train.py** | Script | Train model | Checkpoint |

---

## Complexity Analysis

```
Time Complexity:
├─ Phase 1 (Generation): O(N × T × M²)
│  (N trajectories, T steps, M² model forward pass)
│
├─ Phase 2-5: O(N × T)
│  (Linear scan of trajectories)
│
├─ Phase 6: O(N × T²)
│  (Nested loop for leap testing)
│
├─ Phase 7: O(N × F + N × k)
│  (Feature extraction + K-means)
│
├─ Phase 8: O(N × T)
│  (Schedule comparison)
│
└─ Phase 9: O(N)
   (Aggregation only)

Total: O(N × T² × M²) dominated by generation

Space Complexity:
├─ Trajectory storage: O(N × T × D)
│  (20 traj × 100 steps × 28² pixels ≈ 1.5 GB)
│
├─ Feature vectors: O(N × F)
│  (20 traj × 9 features ≈ 2 KB)
│
└─ Plots: O(N × image_size)
   (PNG compression)
```

---

## Performance Characteristics

```
Training Phase:
├─ Data loading: ~5 sec (MNIST download if needed)
├─ 3 epochs: ~5-10 min (GPU) / 30+ min (CPU)
└─ Checkpoint save: ~2 sec

Generation Phase:
├─ ODE integration (20 traj, 100 steps): ~2-5 min
└─ NPZ storage: ~1 sec

Analysis Phases:
├─ PCA computation: ~30 sec
├─ Metric calculation: ~1-2 min
├─ Plotting: ~2-3 min
└─ Clustering: ~30 sec

Report Generation:
├─ Aggregation: ~10 sec
└─ Text generation: ~5 sec

Total Execution: ~15-20 min (GPU) / 1 hour (CPU)
```

---

## Error Handling & Robustness

```
Potential Issues & Mitigations:

1. Empty Trajectories
   └─ Check len() > 0 before operations
   └─ Skip or use default values

2. NaN/Inf Values
   └─ Add epsilon (1e-8) to denominators
   └─ Clip values to valid ranges

3. Memory Constraints
   └─ Process trajectories one at a time
   └─ Save intermediate results

4. GPU Memory
   └─ Reduce batch_size if OOM
   └─ Fallback to CPU

5. Checkpoint Loading
   └─ Handle .ckpt (Lightning) and .pt formats
   └─ Use map_location for device compatibility

6. File I/O
   └─ Create directories if needed
   └─ Check write permissions
   └─ Handle existing files gracefully
```

---

## Extensibility Points

```
To Add New Analysis:

1. Add to TrajectoryAnalyzer
   └─ Implement static method
   └─ Use existing trajectory dict format
   └─ Return JSON-serializable results

2. Add to VisualizationUtils
   └─ Implement static visualization method
   └─ Save PNG to appropriate phase dir
   └─ Follow existing color schemes

3. Add to EDAMaster
   └─ Create phase_N method
   └─ Update run_full_eda() order
   └─ Save summary.json per phase

4. New Feature Type
   └─ Extract in Phase 7
   └─ Use in Phase 7 clustering
   └─ Document in CODE_INDEX.md
```

---

## Configuration Hierarchy

```
run_eda.py
├─ CLI Arguments
│  ├─ --checkpoint (model path)
│  ├─ --num-trajectories (20)
│  ├─ --num-epochs (3)
│  ├─ --results-dir ('results')
│  └─ --skip-training (False)
│
├─ EDAMaster
│  ├─ results_dir (from CLI)
│  └─ trajectories (from Phase 1)
│
└─ Hardcoded Parameters (in analysis methods)
   ├─ Leap epsilon: 0.5
   ├─ Alignment threshold: 0.5
   ├─ Phase transition percentile: 75
   ├─ K-means clusters: auto (2-3)
   └─ Max leap size: 10
```

---

**This architecture diagram provides a complete view of how all components fit together.**

---

*For detailed implementation, see CODE_INDEX.md*
*For usage guide, see IMPLEMENTATION_SUMMARY.md*
*For detailed analysis, see EDA_README.md*
