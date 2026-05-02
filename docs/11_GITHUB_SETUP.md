# GitHub Setup & Push Checklist

Complete guide for preparing and pushing the project to GitHub.

---

## Pre-Push Checklist

### 1. Verify Project Structure

- [ ] Check `.gitignore` excludes large files (*.ckpt, results/lda_frames/, etc.)
- [ ] Verify `docs/` folder has all markdown files
- [ ] Confirm `scripts/` folder has `download_primary.sh` and `download_secondary.sh`
- [ ] Check `requirements.txt` is up-to-date
- [ ] Root-level README.md exists and is clear

### 2. Clean Up Local Files

```bash
# Remove any local-only directories
rm -rf __pycache__ *.pyc
find . -type d -name "__pycache__" -exec rm -rf {} +

# Remove large artifact directories (these go to Google Drive)
rm -rf results/lda_frames/  # Keep only key visualizations
rm -rf tb_logs/             # TensorBoard logs not needed
rm -rf checkpoints/         # Checkpoints go to Google Drive
```

### 3. Update Download Scripts

Before pushing, ensure placeholder IDs are in place:

**scripts/download_primary.sh:**
```bash
PRIMARY_FOLDER_ID="YOUR_PRIMARY_FOLDER_ID"
```

**scripts/download_secondary.sh:**
```bash
SECONDARY_FOLDER_ID="YOUR_SECONDARY_FOLDER_ID"
```

These will be replaced after uploading to Google Drive.

### 4. Verify Key Files

Essential files to include:

- [x] `README.md` - Main entry point
- [x] `requirements.txt` - All dependencies
- [x] `src/train.py` - Training script
- [x] `src/models/fm.py` - Model definition
- [x] `visualizations/training/generate_samples.py` - Inference
- [x] `visualizations/lda/quick_3d_interactive.py` - Interactive viz
- [x] `scripts/download_primary.sh` - Model download
- [x] `docs/` - All documentation

### 5. Verify .gitignore

```
# Check that these are excluded:
*.ckpt                      # Checkpoints
*.log                       # Training logs
outputs/                    # Training outputs
results/lda_frames/        # Large frame sequences
tb_logs/                   # TensorBoard logs
data/                      # MNIST data (can redownload)
__pycache__/               # Python cache
*.pyc
.DS_Store
```

---

## Step 1: Initialize/Prepare Git

### If Not Already a Git Repo

```bash
cd flow-matching-mnist
git init
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### If Already a Git Repo

```bash
# Check status
git status

# Add all files (respecting .gitignore)
git add -A

# Review what will be committed
git status
```

---

## Step 2: Create Initial Commit

```bash
git commit -m "Initial commit: Flow Matching MNIST implementation

- Trained model with best loss: 0.1829
- Complete visualization pipeline (training analysis, 3D LDA, trajectory frames)
- Comprehensive documentation and guides
- Ready for deployment with Google Drive artifact links
- Training completed in ~5.2 hours on CPU"
```

---

## Step 3: Create GitHub Repository

1. Go to https://github.com/new
2. Fill in:
   - **Repository name:** `flow-matching-mnist`
   - **Description:** Flow Matching for generative MNIST modeling with trajectory visualization
   - **Public:** ✓ (make it public)
   - **Initialize repository:** Leave unchecked (we have local files)

3. Click "Create repository"

4. You'll see push instructions like:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/flow-matching-mnist.git
   git branch -M main
   git push -u origin main
   ```

---

## Step 4: Add Remote and Push

```bash
# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/flow-matching-mnist.git

# Set main branch
git branch -M main

# Push to GitHub
git push -u origin main
```

Check that files appear on GitHub.

---

## Step 5: Add Topics and Description

On GitHub repository page:

1. Click "Edit" (gear icon) near repo name
2. Add topics: `flow-matching`, `diffusion`, `mnist`, `generative-model`
3. Update description if needed
4. Add useful URLs in "Website" field

---

## Step 6: Upload to Google Drive

### Create PRIMARY Folder

1. Go to Google Drive
2. Create new folder: `flow-matching-mnist-PRIMARY`
3. Upload to this folder:
   ```
   - Best checkpoint (fm-balanced-epoch=014-train_loss=0.1829.ckpt)
   - src/ directory
   - config/ directory
   - requirements.txt
   - Key result images from results/
   ```

4. Right-click folder → Share → Get link
5. Copy folder ID from URL: `https://drive.google.com/drive/folders/FOLDER_ID_HERE`
6. Save the ID

### Create SECONDARY Folder

1. Create new folder: `flow-matching-mnist-SECONDARY`
2. Upload to this folder:
   ```
   - All other checkpoints (11 files in checkpoints/)
   - Training logs
   - LDA frame sequences (results/lda_frames/)
   - Full outputs/ directory
   ```

3. Get folder ID same way

---

## Step 7: Update Download Scripts

After getting folder IDs:

**scripts/download_primary.sh:**
```bash
PRIMARY_FOLDER_ID="ACTUAL_ID_FROM_STEP_6"
```

**scripts/download_secondary.sh:**
```bash
SECONDARY_FOLDER_ID="ACTUAL_ID_FROM_STEP_6"
```

Commit these changes:
```bash
git add scripts/download_*.sh
git commit -m "Update Google Drive folder IDs for downloads"
git push
```

---

## Step 8: Test Complete Workflow

In a clean directory:

```bash
# Clone fresh
git clone https://github.com/YOUR_USERNAME/flow-matching-mnist.git
cd flow-matching-mnist

# Download primary
bash scripts/download_primary.sh

# Verify checkpoint exists
ls outputs/fm/2026-05-01/22-58-12/training_logs/version_0/checkpoints/

# Install dependencies
pip install -r requirements.txt

# Generate samples
python visualizations/training/generate_samples.py

# Check results
ls results/generated_samples.png
```

If successful: ✓ Deployment is working!

---

## Step 9: Create GitHub Release (Optional)

```bash
# Create tag
git tag -a v1.0 -m "Initial release: Flow Matching MNIST

- Trained model with checkpoint
- Complete documentation
- Deployment ready"

# Push tag
git push origin v1.0
```

Then on GitHub:
1. Go to Releases
2. "Draft a new release"
3. Select tag `v1.0`
4. Add description and attach checkpoint file
5. Publish release

---

## Final Verification

- [ ] GitHub repository public
- [ ] All source files present
- [ ] README.md displays correctly
- [ ] Download scripts have folder IDs
- [ ] Fresh clone can download and run
- [ ] Visualizations generate successfully
- [ ] Documentation links all work

---

## Troubleshooting

### Files not appearing on GitHub

```bash
# Check what will be pushed
git ls-files

# If large files are missing, check .gitignore
cat .gitignore
```

### Download script not working

```bash
# Test gdown directly
pip install gdown
gdown --folder "FOLDER_ID" -O test_folder/
```

### Permission denied on clone

```bash
# Use HTTPS instead of SSH
git remote set-url origin https://github.com/YOUR_USERNAME/flow-matching-mnist.git
```

---

## After Deployment

- [ ] Update README.md with GitHub link
- [ ] Test download links from docs
- [ ] Monitor GitHub for issues/questions
- [ ] Consider adding GitHub Actions for:
  - Automated testing
  - Documentation building
  - Release automation

---

**You're ready to deploy!** Once the download folder IDs are added, the project is fully self-contained and reproducible.
