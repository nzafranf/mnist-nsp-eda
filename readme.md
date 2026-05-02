# Flow Matching on MNIST

A complete implementation of Flow Matching for generative modeling, with trajectory visualization and LDA-based analysis.

**Status:** ✓ Training Complete | Best Loss: 0.1829 | Duration: ~5.2 hours (CPU)

## Quick Start (2 minutes)

```bash
# 1. Clone repository
git clone <repo-url>
cd flow-matching-mnist

# 2. Download best model
bash scripts/download_primary.sh

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate samples
python visualizations/training/generate_samples.py
```

Results will be saved in `results/` directory.

---

## Features

- **Trained Model:** Best checkpoint with loss 0.1829
- **Sample Quality:** High-fidelity MNIST digit generation
- **Trajectory Analysis:** LDA-based 3D visualization and frame sequences
- **Comprehensive Docs:** Full architecture, training, and inference guides

---

## Documentation

| Document | Purpose |
|----------|---------|
| [INFERENCE_GUIDE.md](docs/10_INFERENCE_GUIDE.md) | How to generate samples and customize inference |
| [DEPLOYMENT_STRUCTURE.md](docs/09_DEPLOYMENT_STRUCTURE.md) | Google Drive folder organization |
| [VISUALIZATIONS_GUIDE.md](docs/VISUALIZATIONS_GUIDE.md) | All available visualizations and how to run them |
| [00_README.md](docs/00_README.md) | Full project overview and results |
| [01_ARCHITECTURE.md](docs/01_ARCHITECTURE.md) | Model architecture details |

---

## Available Visualizations

After downloading, generate:

```bash
# Sample quality grid (1 min)
python visualizations/training/generate_samples.py

# Training analysis charts (1 min)
python visualizations/training/analyze_training.py

# Interactive 3D visualization (5 min)
python visualizations/lda/quick_3d_interactive.py

# Trajectory frame sequences (30-40 min)
python visualizations/lda/lda_frame_sequence.py
```

See [VISUALIZATIONS_GUIDE.md](docs/VISUALIZATIONS_GUIDE.md) for detailed information.

---

## Project Structure

```
flow-matching-mnist/
├── src/                    # Source code
│   ├── train.py          # Training script
│   ├── models/           # Model definitions
│   └── config/           # Configuration files
├── visualizations/       # Visualization scripts
│   ├── training/         # Sample quality visualizations
│   └── lda/              # Trajectory analysis
├── docs/                 # Documentation
├── scripts/              # Download and utility scripts
├── results/              # Generated outputs
└── requirements.txt      # Dependencies
```

---

## Model Details

| Property | Value |
|----------|-------|
| Framework | PyTorch Lightning |
| Architecture | UNet-based Flow Matcher |
| Parameters | 8.6M |
| Best Loss | 0.1829 |
| Training Time | 5.2 hours (CPU) |
| Inference Speed | ~2-5s per batch of 16 (50 ODE steps) |

---

## Downloads

### PRIMARY (Essential - 300 MB)
Best model checkpoint + code + key visualizations. Everything needed for inference.

```bash
bash scripts/download_primary.sh
```

### SECONDARY (Optional - 3 GB)
All training artifacts, alternative checkpoints, frame sequences, and logs. Useful for analysis and research.

```bash
bash scripts/download_secondary.sh
```

---

## Citation

If you use this work, please cite:

```bibtex
@article{albergo2023flow,
  title={Flow Matching for Generative Modeling},
  author={Albergo, Michael S and others},
  year={2023}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) file

---

## Next Steps

- See [INFERENCE_GUIDE.md](docs/10_INFERENCE_GUIDE.md) for detailed inference instructions
- Check [VISUALIZATIONS_GUIDE.md](docs/VISUALIZATIONS_GUIDE.md) for all visualization options
- Read [00_README.md](docs/00_README.md) for full project overview
