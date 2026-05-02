#!/bin/bash

# Create directory structure
echo "Creating organized directory structure..."

mkdir -p src/models/velocity_architectures
mkdir -p src/config/model
mkdir -p visualizations/lda
mkdir -p visualizations/training
mkdir -p monitoring
mkdir -p scripts
mkdir -p docs
mkdir -p results/examples
mkdir -p utils

# Move core training files to src/
echo "Organizing training code..."
mv train.py src/ 2>/dev/null
mv utils.py utils/ 2>/dev/null
mv paths.py utils/ 2>/dev/null

# Move model files (already in correct place, just verify)
echo "Verifying model files..."
# Models should already be in models/ directory

# Move all visualization scripts
echo "Organizing visualization scripts..."
mv generate_samples.py visualizations/training/ 2>/dev/null
mv generate.py visualizations/training/ 2>/dev/null
mv analyze_training.py visualizations/training/ 2>/dev/null

# LDA-specific visualizations
mv lda_frame_sequence.py visualizations/lda/ 2>/dev/null
mv lda_trajectory_visualization.py visualizations/lda/ 2>/dev/null
mv lda_simplified.py visualizations/lda/ 2>/dev/null
mv lda_static_viz.py visualizations/lda/ 2>/dev/null
mv lda_trajectory_fast.py visualizations/lda/ 2>/dev/null
mv quick_3d_interactive.py visualizations/lda/ 2>/dev/null

# Move monitoring scripts
echo "Organizing monitoring scripts..."
mv monitor_training.py monitoring/ 2>/dev/null
mv monitor_training_detailed.py monitoring/ 2>/dev/null
mv monitor_training_file.py monitoring/ 2>/dev/null

# Move test/utility scripts
echo "Organizing test scripts..."
mv test_unet.py utils/ 2>/dev/null
mv run_eda.py utils/ 2>/dev/null
mv run_eda_simple.py utils/ 2>/dev/null

# Move all documentation to docs/
echo "Organizing documentation..."
mv README.md docs/00_README.md 2>/dev/null || mv readme.md docs/00_README.md 2>/dev/null
mv ARCHITECTURE.md docs/01_ARCHITECTURE.md 2>/dev/null
mv CODE_INDEX.md docs/02_CODE_INDEX.md 2>/dev/null
mv START_HERE.md docs/03_START_HERE.md 2>/dev/null
mv DELIVERY_SUMMARY.md docs/04_DELIVERY_SUMMARY.md 2>/dev/null
mv EDA_README.md docs/05_EDA_README.md 2>/dev/null
mv LDA_VISUALIZATIONS_README.md docs/06_LDA_VISUALIZATIONS.md 2>/dev/null
mv DEPLOYMENT_GUIDE.md docs/07_DEPLOYMENT_GUIDE.md 2>/dev/null
mv GITHUB_PUSH_CHECKLIST.md docs/08_GITHUB_CHECKLIST.md 2>/dev/null

# Move scripts
echo "Organizing scripts..."
mv downloads.sh scripts/ 2>/dev/null

# Keep requirements.txt in root
echo "Keeping requirements.txt in root..."

# Create __init__.py files for packages
touch src/__init__.py
touch visualizations/__init__.py
touch monitoring/__init__.py
touch utils/__init__.py

echo "✓ Directory structure organized!"
echo ""
echo "New structure:"
echo "├── src/                    # Training code"
echo "├── visualizations/         # Visualization scripts"
echo "│   ├── training/          # Sample generation, training analysis"
echo "│   └── lda/               # LDA-based visualizations"
echo "├── monitoring/            # Training monitoring"
echo "├── utils/                 # Utilities and tests"
echo "├── scripts/               # Setup and download scripts"
echo "├── docs/                  # Documentation"
echo "├── results/               # Output visualizations"
echo "├── config/                # Training configs"
echo "├── models/                # Model definitions"
echo "├── data/                  # MNIST data"
echo "├── outputs/               # Training logs"
echo "└── requirements.txt"

