Hesitation Margins are Necessary:
Ambiguity-Aware Backpropagation for Robust
Learning Under Label Noise

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

Official PyTorch implementation of **AAB-SGN** from the paper:

> **AAB-SGN: Robust Learning Under Label Noise via Hesitation-Aware Gradient Modulation**  
> Ramsha Mehreen, Renikunta Ramesh  
> *Journal of Experimental & Theoretical Artificial Intelligence*, 2026  
> [[Paper]](link-to-paper) [[Supplementary]](link-to-supplement)

## Overview

AAB-SGN is a novel framework for robust deep learning under label noise that embeds **vague set theory** directly into gradient computation. Unlike existing methods that operate at the loss or data level, AAB-SGN modulates gradient flow using **hesitation margins** (π_A ≥ 0.1) to enable uncertainty-aware optimization.

### Key Features

✅ **First Necessity Theorem**: Proves that positive hesitation margins (π_min > 0) are mathematically required for robust convergence under label noise  
✅ **Adaptive Mode Selection**: O(n^{-1/2}) convergence guarantee for automatic hyperparameter tuning  
✅ **Efficient**: 18% computational overhead (vs 119% for DivideMix)  
✅ **Interpretable**: Explicit failure modes and deployment diagnostics  
✅ **State-of-the-Art**: 94.23% accuracy on CIFAR-10 (40% noise)

## Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU training)

### Install from source

```bash
git clone https://github.com/m24ds001/AAB-SGN-PROJECT
cd AAB-SGN
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Basic Usage

```python
import torch
from torchvision.models import resnet18
from aab_sgn import create_aab_sgn_model, AABSGNTrainer, create_noisy_cifar10

# Create model
model = resnet18(num_classes=10)
model, aab_loss = create_aab_sgn_model(
    model,
    sigma_S=0.1,    # Minor residual width
    sigma_M=0.2,    # Moderate residual width  
    sigma_L=1.0,    # Severe residual width
    pi_min=0.1      # Minimum hesitation margin
)

# Create noisy dataset
train_dataset = create_noisy_cifar10(
    root='./data',
    noise_type='symmetric',
    noise_rate=0.4,
    download=True
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True
)

# Train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

trainer = AABSGNTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device
)

history = trainer.train(num_epochs=200)
```

### Running Experiments

Train on CIFAR-10 with 40% symmetric noise:

```bash
python experiments/train_cifar10.py \
    --noise-type symmetric \
    --noise-rate 0.4 \
    --epochs 200 \
    --batch-size 128 \
    --lr 0.1
```

## Supported Datasets

- **CIFAR-10/100**: Standard image classification benchmarks
- **Fashion-MNIST**: Fashion product images
- **Clothing1M**: Real-world noisy labels (~1M images, 38% noise)
- **Food101-N**: Food images with noisy annotations
- **CIFAR-10N**: Human-annotated noisy CIFAR-10

## Noise Types

AAB-SGN supports multiple noise patterns:

1. **Symmetric Noise**: Uniform random label flipping
2. **Asymmetric Noise**: Class-conditional corruption (e.g., truck→automobile)
3. **Pair-Flip Noise**: Specific class pair swapping
4. **Custom**: Provide your own transition matrix

## Key Results

| Dataset | Noise Type | Noise Rate | AAB-SGN | SGN (2024) | DivideMix |
|---------|-----------|------------|---------|-----------|-----------|
| CIFAR-10 | Symmetric | 40% | **94.23%** | 88.12% | 87.83% |
| CIFAR-10 | Asymmetric | 40% | **88.39%** | 84.56% | 84.59% |
| Clothing1M | Real-world | 38% | **75.23%** | 73.12% | 72.56% |
| CIFAR-100 | Symmetric | 40% | **71.45%** | 65.32% | 67.89% |

*Note: All results averaged over 5 random seeds with statistical significance testing (p < 0.05).*

## Mode Selection

AAB-SGN automatically selects between two operating modes based on residual separability:

- **Standalone AAB** (KL ≥ 1.5): Direct vague set modulation on gradients
- **Two-stage AAB-SGN** (KL < 1.5): SGN preprocessing → AAB modulation

To check which mode is appropriate for your dataset:

```python
from aab_sgn import AdaptiveKLThreshold

selector = AdaptiveKLThreshold()
threshold, use_two_stage, info = selector.update(residuals)

print(f"KL divergence: {info['kl_divergence']:.3f}")
print(f"Threshold: {threshold:.3f}")
print(f"Mode: {info['mode']}")
```

## Hyperparameters

Default values (from paper):

```python
{
    'sigma_S': 0.1,      # Minor residual width
    'sigma_M': 0.2,      # Moderate residual width
    'sigma_L': 1.0,      # Severe residual width
    'pi_min': 0.1,       # Minimum hesitation margin
    'alpha': 0.3,        # Learning rate scaling factor
    'kl_threshold': 1.5, # Mode selection threshold (or None for adaptive)
}
```

See `configs/default.yaml` for full configuration options.

## When to Use AAB-SGN

### ✅ Recommended Use Cases

- Data augmentation pipelines with 10-40% label corruption
- Semi-supervised learning with noisy pseudo-labels
- Adversarial training (57% gradient variance reduction)
- Large-scale datasets where ensemble methods are cost-prohibitive

### ⚠️ Not Recommended

- Medical imaging with expert disagreement (fundamental semantic overlap)
- Systematic crowdsourced bias (use confusion matrix estimation instead)
- Extreme noise rates (τ > 0.70) — ensemble methods preferred
- Asymmetric noise with high τ_max (use two-stage AAB-SGN)

See paper Section 9.2 for detailed deployment guidelines.

## Citation

If you use AAB-SGN in your research, please cite:

```bibtex
@article{mehreen2026aabsgn,
  title={Hesitation Margins are Necessary:
Ambiguity-Aware Backpropagation for Robust
Learning Under Label Noise},
  author={Mehreen, Ramsha and Ramesh, Renikunta},
  journal={Journal of Experimental \& Theoretical Artificial Intelligence},
  year={2026},
  publisher={Taylor \& Francis}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank the anonymous reviewers for their constructive feedback and the open-source community for providing baseline implementations.

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Email: ramshamehreen2208@gmail.com, rr.mh@kitsw.ac.in

## Reproducibility

All experiments use fixed random seeds (0, 42, 123, 456, 789) and detailed hyperparameter specifications are provided in the paper's supplementary material. Pretrained models for Clothing1M and Food101-N are available upon request.

---

**Note**: This implementation is provided for research purposes. For production deployments, please refer to the deployment guidelines in Section 9.2 of the paper.
