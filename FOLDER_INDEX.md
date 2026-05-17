# Flow Matching MNIST - Folder Index & Contents

**Purpose**: Understand what each folder contains and when to use it.  
**Last Updated**: 2026-05-17

---

## Core Codebase (Implementation)

### `/models` - Flow Matching Model Implementations
**Purpose**: Core neural network models for Flow Matching  
**Status**: Production-ready  
**Key Files**:
- `fm.py` - **Main model** (ImageFlowMatcher class)
- `class_cond_fm.py` - Class-conditional variant
- `utils.py` - Model utilities
- `velocity_architectures/unet.py` - UNet velocity network

**When to use**:
- Loading trained models: `from models.fm import ImageFlowMatcher`
- Understanding model architecture: Read `fm.py` (main entry point)
- Extending model: Modify velocity_architectures

**Training checkpoint**:
- Location: `PRIMARY/checkpoints/fm-balanced-epoch=014-train_loss=0.1829.ckpt`
- Size: ~350MB (8.6M parameters, loss=0.1829)

---

### `/richardson_eda` - Algorithm 5.1 Implementation & Analysis
**Purpose**: Core Richardson adaptive solver + comprehensive analysis suite  
**Status**: Complete and tested  
**Structure**:
```
richardson_eda/
├── adaptive_solver.py              # ⭐ Core Algorithm 5.1 implementation
├── pareto_sweep.py                 # Tolerance optimization (7 values)
├── alpha_stability_sweep.py         # Safety factor α optimization
├── curvature_correlation.py         # Validates H ∝ 1/κ₂ relationship
├── analyze_timestep_decisions.py    # Iteration-level EDA
└── results/                         # All analysis outputs
    ├── RICHARDSON_COMPLETE_ANALYSIS.md    # 2450+ solver runs
    ├── FINAL_IMAGE_GENERATION_REPORT.md   # Image generation metrics
    └── timestep_decision_eda/             # Iteration-level analysis
```

**Key Functions**:
- `AdaptiveFlowSolver` - Main solver class
  - `richardson_step()` - Computes error via coarse/fine steps
  - `compute_new_step_size()` - Applies H_new = H × α × (t/ê)^(1/2)
  - `solve()` - Main adaptive loop
- `SolverConfig` - Configuration dataclass

**Configuration (optimal from analysis)**:
```python
config = SolverConfig(
    tolerance=0.01,      # Error threshold
    alpha=0.75,          # Safety factor
    f_min=0.1,           # Min H shrinking
    f_max=5.0,           # Max H growth
    max_steps=5000       # Hard limit
)
```

**When to use**:
- Implementing adaptive ODE solving: Import `AdaptiveFlowSolver`
- Understanding algorithm: Read `adaptive_solver.py` + `prompt/PROOF.md` Section V
- Tuning parameters: See analysis in `results/RICHARDSON_COMPLETE_ANALYSIS.md`
- Analyzing decision-making: See `results/timestep_decision_eda/`

**Analysis outputs**:
- Timestep evolution plots
- Error landscape analysis
- Decision statistics (JSON)
- Iteration-by-iteration logs

---

### `/src` - Source Code & Configuration
**Purpose**: Model utilities and configuration classes  
**Status**: Stable  
**Contents**:
- `src/models/` - Model utilities
- `src/config/` - Configuration management

**When to use**: Custom model loading, configuration inheritance

---

### `/config` - Configuration Files
**Purpose**: Model training & inference configurations  
**Status**: Reference (not actively used in Algorithm 5.1)  
**Contents**:
- `config/model/` - Model-specific configs

**When to use**: Understanding original training setup

---

### `/utils` - Utility Functions
**Purpose**: Helper functions (data loading, preprocessing, etc.)  
**Status**: Stable  

**When to use**: Data pipeline setup, utility operations

---

### `/scripts` - Helper Scripts
**Purpose**: Standalone utility scripts  
**Status**: Stable  

**When to use**: Data preparation, batch operations

---

### `/monitoring` - Training Monitoring
**Purpose**: TensorBoard logging and training metrics  
**Status**: Legacy (for reference)  
**Files**:
- `monitor_training.py`
- `monitor_training_detailed.py`

**When to use**: Understanding training setup, not needed for Algorithm 5.1

---

### `/prompt` - Mathematical Documentation & References
**Purpose**: Mathematical proofs, algorithms, and specification documents  
**Status**: Reference  
**Key Files**:
- `PROOF.md` - ⭐ **Main reference** (Section V contains Algorithm 5.1 specification)
- `RICHARDSON_HYP_PARM_EDA.md` - Hyperparameter optimization guide
- Other reference documents

**When to use**:
- Understanding theory: Read `PROOF.md` Section III & V
- Hyperparameter tuning: Read `RICHARDSON_HYP_PARM_EDA.md`

---

### `/docs` - Project Documentation
**Purpose**: Setup, architecture, and deployment guides  
**Status**: Comprehensive  
**Key Files**:
- `03_START_HERE.md` - Quick start
- `01_ARCHITECTURE.md` - System design
- `05_EDA_README.md` - Analysis guide
- `07_DEPLOYMENT_GUIDE.md` - Setup instructions

**When to use**: First-time setup, understanding system architecture

---

## Data & Checkpoints

### `/data/MNIST` - Dataset
**Purpose**: MNIST training/validation data  
**Status**: Ready to use  
**Contents**: Images and labels (standard PyTorch MNIST format)

**When to use**: Training new models, data pipeline setup

---

### `/PRIMARY` - Main Trained Model & Results
**Purpose**: ⭐ Production model checkpoint and artifacts  
**Status**: Ready for inference  
**Structure**:
```
PRIMARY/
├── checkpoints/
│   └── fm-balanced-epoch=014-train_loss=0.1829.ckpt  # ⭐ USE THIS
├── src/                    # Model source code
├── results/                # Training results
└── config/                 # Training configuration
```

**Key File**: 
- `fm-balanced-epoch=014-train_loss=0.1829.ckpt` (8.6M parameters)

**When to use**:
- Image generation: Load this checkpoint
- Model inference: Use this model
- Velocity field computation: Use this trained network

**Loading example**:
```python
from models.fm import ImageFlowMatcher
from pathlib import Path

path = Path("PRIMARY/checkpoints/fm-balanced-epoch=014-train_loss=0.1829.ckpt")
model = ImageFlowMatcher.load_from_checkpoint(str(path))
model.eval()
```

---

### `/SECONDARY` - Alternative Checkpoint (Experimental)
**Purpose**: Backup/alternative trained model  
**Status**: Available but not primary  
**Contents**: Checkpoint, training logs, results

**When to use**: Only if PRIMARY checkpoint is unavailable; experimental comparisons

---

### `/tb_logs` - TensorBoard Training Logs
**Purpose**: Training metrics and visualization  
**Status**: Archive (10 training runs)  
**Contents**:
- `version_0/` through `version_9/` (10 runs)
- Event files for TensorBoard

**When to use**: Analyzing training dynamics (not needed for Algorithm 5.1)

---

## Results & Artifacts

### `/richardson_eda/results` - Algorithm Analysis Results
**Purpose**: ⭐ Main analysis output directory  
**Status**: Complete  
**Structure**:
```
richardson_eda/results/
├── RICHARDSON_COMPLETE_ANALYSIS.md              # 2450+ runs analysis
├── FINAL_IMAGE_GENERATION_REPORT.md             # Image generation
├── timestep_decision_eda/                       # ⭐ Iteration-level EDA
│   ├── TIMESTEP_DECISION_ANALYSIS.md
│   ├── timestep_evolution_traj*.png (3 files)
│   ├── error_landscape_traj*.png (3 files)
│   └── decision_statistics.json
├── pareto_sweep_results/                        # Tolerance optimization
├── alpha_stability_results/                     # Safety factor optimization
└── curvature_correlation_results/               # Curvature analysis
```

**Key Contents**:
1. **Timestep Decision EDA** (NEW):
   - 4-panel iteration dynamics visualizations
   - Error landscape analysis
   - Decision statistics (JSON)
   - Comprehensive markdown report

2. **Richardson Analysis**:
   - 2450+ solver runs across tolerance & alpha values
   - Pareto frontier analysis
   - Stability metrics
   - Rejection rate analysis

3. **Image Generation**:
   - Benchmark results (21.75s for 10 samples)
   - Generated MNIST digits
   - Computation metrics

**When to use**:
- Understanding algorithm performance: Read TIMESTEP_DECISION_ANALYSIS.md
- Hyperparameter justification: Read RICHARDSON_COMPLETE_ANALYSIS.md
- Image quality verification: Check FINAL_IMAGE_GENERATION_REPORT.md

---

### `/results` - Main Results Directory
**Purpose**: Generated experiment outputs  
**Status**: Organized into phases  
**Structure**:
```
results/
├── final_mnist_samples/             # ⭐ Main output (benchmark)
│   ├── mnist_grid.png
│   ├── digit_00.png ... digit_09.png
│   └── generation_metrics.json
├── trajectory_viz/                  # Trajectory visualizations
├── phase1/ through phase9/          # EDA phases (experimental)
├── lda_frames/                      # LDA visualization frames
└── lda_cache/                       # LDA computation cache
```

**When to use**:
- Viewing generated images: Check `final_mnist_samples/`
- Understanding trajectory behavior: Check `trajectory_viz/`

---

### `/outputs` - Alternative Output Directory
**Purpose**: Secondary output location  
**Status**: Organized by model type  
**Structure**:
```
outputs/
├── fm/                   # Flow Matching outputs
└── fid/                  # FID score results
```

**When to use**: Comparing different model variants (FID evaluation)

---

### `/results_simple` & `/per_trajectory_eda_results` - Experimental Outputs
**Purpose**: Previous EDA attempts  
**Status**: Archive (kept for reference)  
**Note**: Use `richardson_eda/results/` for current analysis instead

**When to use**: Historical comparison only

---

## Generation Scripts (Top-Level)

### `generate_mnist_adaptive.py` - Benchmark Generation
**Purpose**: Generate MNIST images using fixed 50-step grid  
**Status**: Working, production-ready  
**Run**: `python generate_mnist_adaptive.py`  
**Output**:
- Location: `richardson_eda/results/final_mnist_samples/`
- 10 MNIST digits
- Time: ~21.75 seconds
- Metrics: generation_metrics.json

**Key Metrics**:
- Throughput: 0.46 samples/second
- Consistency: CV = 9.05%
- All samples valid (pixel range [0,1])

---

### `generate_mnist_true_adaptive.py` - Algorithm 5.1 Generation
**Purpose**: Generate MNIST images using variable adaptive steps  
**Status**: Working, validates Algorithm 5.1  
**Run**: `python generate_mnist_true_adaptive.py`  
**Output**:
- Location: `richardson_eda/results/adaptive_algorithm5_samples/`
- 10 MNIST digits with variable steps
- Time: ~18.99 seconds
- Adaptive steps: 41-50 per sample

**Key Metrics**:
- Speedup: 1.14x vs benchmark
- Variable steps per sample
- Detailed per-sample metrics

---

## Supporting Scripts

### `convert_md_to_pdf.py` - Markdown to PDF Conversion
**Purpose**: Convert markdown reports to PDF  
**Usage**: `python convert_md_to_pdf.py <markdown_file>`

---

### `md2html.py` & `md2pdf.py` - Format Converters
**Purpose**: Convert markdown to HTML or PDF  
**Usage**: Standalone converters for documentation

---

## Organizational Notes

### What to Keep (Codebase)
- ✓ `/models` - Model implementations
- ✓ `/richardson_eda` - Core algorithm & analysis
- ✓ `/src`, `/utils`, `/scripts` - Support code
- ✓ `/prompt`, `/docs` - Documentation
- ✓ `/config` - Configuration
- ✓ `/PRIMARY` - Main checkpoint
- ✓ `generate_mnist_*.py` - Generation scripts

### What's Archive (Reference)
- `/SECONDARY` - Alternative checkpoint
- `/tb_logs` - Historical training logs
- `/results_simple`, `/per_trajectory_eda_results*` - Old analysis
- `/jacobian_eda`, `/eps_linear_eda` - Experimental analyses

### What to Generate (Outputs)
- `richardson_eda/results/` - New analysis results
- `results/final_mnist_samples/` - Generated images
- Image grids and individual samples

---

## File Quick Reference

| File/Folder | Type | Purpose | Status |
|-------------|------|---------|--------|
| [README_NEW.md](./README_NEW.md) | Doc | Main navigation & overview | ⭐ START HERE |
| [FOLDER_INDEX.md](./FOLDER_INDEX.md) | Doc | This file (folder contents) | Reference |
| [EXPERIMENTS.md](./EXPERIMENTS.md) | Doc | How to run experiments | Guide |
| [richardson_eda/adaptive_solver.py](./richardson_eda/adaptive_solver.py) | Code | Core Algorithm 5.1 | ⭐ CORE |
| [prompt/PROOF.md](./prompt/PROOF.md) | Doc | Mathematical foundation | ⭐ THEORY |
| [richardson_eda/analyze_timestep_decisions.py](./richardson_eda/analyze_timestep_decisions.py) | Code | Iteration EDA | Analysis |
| [generate_mnist_adaptive.py](./generate_mnist_adaptive.py) | Code | Benchmark generation | Demo |
| [generate_mnist_true_adaptive.py](./generate_mnist_true_adaptive.py) | Code | Algorithm 5.1 generation | Demo |
| [models/fm.py](./models/fm.py) | Code | Flow Matching model | Production |
| [PRIMARY/checkpoints/...ckpt](./PRIMARY/checkpoints/) | Data | Trained model | ⭐ USE |
| [docs/03_START_HERE.md](./docs/03_START_HERE.md) | Doc | Quick start | Guide |

---

## Summary

**Codebase is organized into 3 main areas**:

1. **Implementation** (`/richardson_eda`, `/models`, `/src`)
   - Algorithm 5.1 adaptive solver
   - Flow Matching model
   - Support code

2. **Analysis** (`/richardson_eda/results`)
   - Timestep decision EDA (NEW)
   - Richardson algorithm analysis (2450+ runs)
   - Image generation metrics

3. **Documentation** (`/docs`, `/prompt`, this file)
   - Setup & architecture
   - Mathematical foundations
   - Navigation & reference

**Start with**: `README_NEW.md` → `docs/03_START_HERE.md` → `richardson_eda/adaptive_solver.py`

