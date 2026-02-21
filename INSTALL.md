# Installation Guide

## Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA 11.0+ (optional, for GPU acceleration)

## Installation Methods

### Method 1: Install from Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/Hesitation Margins are Necessary:
Ambiguity-Aware Backpropagation for Robust
Learning Under Label Noise/AAB-SGN.git
cd AAB-SGN

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### Method 2: Pip Install (when released)

```bash
pip install aab-sgn
```

### Method 3: Conda Environment

```bash
# Create conda environment
conda create -n aabsgn python=3.10
conda activate aabsgn

# Install PyTorch (example for CUDA 11.8)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install AAB-SGN
pip install -e .
```

## Verification

Test your installation:

```python
import torch
from aab_sgn import VagueSetMembership, AABLoss

# Create vague set
vague_set = VagueSetMembership()

# Create dummy data
logits = torch.randn(10, 5)
targets = torch.randint(0, 5, (10,))

# Compute AAB loss
aab_loss = AABLoss(vague_set)
loss = aab_loss(logits, targets)

print(f"✓ Installation successful! Loss: {loss.item():.4f}")
```

## Troubleshooting

### CUDA Issues

If you encounter CUDA errors:

```bash
# Check CUDA version
nvcc --version

# Install matching PyTorch version
# Visit: https://pytorch.org/get-started/locally/
```

### Import Errors

If you get import errors:

```bash
# Make sure package is installed
pip list | grep aab-sgn

# Reinstall if needed
pip uninstall aab-sgn
pip install -e .
```

### Scikit-learn Compatibility

If GMM fitting fails:

```bash
# Upgrade scikit-learn
pip install --upgrade scikit-learn>=1.0.0
```

## GPU Setup

For optimal performance with GPU:

```python
import torch

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

## Next Steps

- See `README.md` for usage examples
- Run `python experiments/train_cifar10.py --help` for training options
- Check `configs/default.yaml` for configuration templates
