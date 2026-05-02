# Inference Guide

How to generate new MNIST samples using the trained Flow Matching model.

---

## Quick Start (5 minutes)

### 1. Download and Setup

```bash
# Clone repository
git clone <your-repo-url>
cd flow-matching-mnist

# Download best model checkpoint
bash scripts/download_primary.sh

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Samples

```bash
# Generate and save a grid of 16 samples
python visualizations/training/generate_samples.py

# View results
open results/generated_samples.png      # macOS
xdg-open results/generated_samples.png  # Linux
start results/generated_samples.png     # Windows
```

### 3. Generate with Different Quality Levels

```bash
# Generate with different ODE step counts
# More steps = higher quality but slower
python visualizations/training/generate_samples.py

# Results show quality progression:
# - 5 steps:   Fast (real-time capable)
# - 10 steps:  Good quality
# - 20 steps:  High quality
# - 50 steps:  Maximum quality (recommended)
```

---

## Using the Model Directly

### Custom Sample Generation

Create a Python script `generate_my_samples.py`:

```python
import torch
from src.models.fm import ImageFlowMatcher
from src.utils.paths import get_checkpoint_path

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ImageFlowMatcher.load_from_checkpoint(
    "outputs/fm/2026-05-01/22-58-12/training_logs/version_0/checkpoints/fm-balanced-epoch=014-train_loss=0.1829.ckpt"
)
model = model.to(device)
model.eval()

# Generate 10 samples with 50 ODE steps
solver = model.get_solver(device, num_steps=50)
samples = solver.sample(batch_size=10)  # Shape: [10, 1, 28, 28]

# Save samples
import torchvision
torchvision.utils.save_image(samples.cpu(), "my_samples.png", nrow=5, normalize=True)
```

Run it:
```bash
python generate_my_samples.py
```

---

## Model Details

| Property | Value |
|----------|-------|
| Architecture | UNet-based Flow Matching |
| Input | Gaussian noise [batch, 1, 28, 28] |
| Output | MNIST digit samples [batch, 1, 28, 28] |
| Channels | 64 base channels |
| Parameters | 8.6M |
| Training Loss | 0.1829 (best) |
| Training Time | ~5 hours on CPU |
| Inference Time | ~2-5 seconds per batch of 16 (50 steps) |

---

## Performance

### Memory Requirements

| Setting | GPU Memory | CPU Memory |
|---------|-----------|-----------|
| Single sample | 100 MB | 200 MB |
| Batch of 16 | 500 MB | 1 GB |
| With visualizations | 2 GB | 3 GB |

### Speed (approximate)

| ODE Steps | Speed | Quality |
|-----------|-------|---------|
| 5 | Very fast (~0.1s) | OK (noise visible) |
| 10 | Fast (~0.3s) | Good |
| 20 | Medium (~0.8s) | Very good |
| 50 | Slow (~2s) | Excellent |

---

## Visualizations Available

After downloading, you can generate:

### 1. Sample Quality Grid
```bash
python visualizations/training/generate_samples.py
```
Output: `results/generated_samples.png` (4×4 grid of 16 samples)

### 2. Quality Comparison (5 vs 50 steps)
```bash
python visualizations/training/generate_samples.py
```
Output: `results/generation_process.png` (shows quality improvement)

### 3. Training Analysis
```bash
python visualizations/training/analyze_training.py
```
Output: `results/loss_curves.png` + `results/training_report.png`

### 4. Interactive 3D Visualization
```bash
# Generate interactive plot with 300 real + 300 generated samples
python visualizations/lda/quick_3d_interactive.py
```
Output: `results/lda_3d_interactive_quick.html` (open in browser)

### 5. Trajectory Frame Sequences
```bash
# Generate 51 frames showing noise → digit evolution
python visualizations/lda/lda_frame_sequence.py
```
Output: `results/lda_frames/frame_2d_*.png` (102 frames total)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Run `bash scripts/download_primary.sh` first |
| Out of memory | Reduce batch size or ODE steps |
| Slow inference | Reduce ODE steps (use 20 instead of 50) |
| CUDA not available | Model will automatically use CPU |
| Import errors | Run `pip install -r requirements.txt` |

---

## Next Steps

- Modify the model for conditional generation (class-specific digits)
- Fine-tune on other datasets
- Experiment with different ODE solvers
- Analyze learned trajectories in detail
- Create animation from frame sequences

See the main [README.md](00_README.md) for full documentation.
