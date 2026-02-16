# AAB-SGN Code Structure

```
AAB-SGN/
├── README.md                    # Main documentation
├── LICENSE                      # MIT License
├── CITATION.cff                 # Citation information
├── INSTALL.md                   # Installation guide
├── setup.py                     # Package setup configuration
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore patterns
│
├── src/aab_sgn/                # Main package
│   ├── __init__.py             # Package initialization
│   ├── aab_core.py             # Core AAB-SGN implementation
│   ├── adaptive_threshold.py   # Adaptive KL threshold (Algorithm 1)
│   ├── trainer.py              # Training loop
│   └── noisy_data.py           # Noise injection utilities
│
├── experiments/                 # Experiment scripts
│   └── train_cifar10.py        # CIFAR-10 training script
│
├── configs/                     # Configuration files
│   └── default.yaml            # Default hyperparameters
│
├── models/                      # (Placeholder for pretrained models)
├── data/                        # (Created when downloading datasets)
├── checkpoints/                 # (Created during training)
└── tests/                       # (Placeholder for unit tests)
```

## Key Files

### Core Implementation

- **aab_core.py**: Contains `VagueSetMembership`, `AABLoss`, and `TwoStageAABSGN` classes
- **adaptive_threshold.py**: Implements Algorithm 1 from the paper with O(n^-1/2) convergence
- **trainer.py**: Complete training loop with mode selection and logging
- **noisy_data.py**: Dataset wrapper for label noise injection

### Experiments

- **train_cifar10.py**: Standalone script for reproducing paper results

### Configuration

- **default.yaml**: All hyperparameters from the paper (Table S33)

## Usage Patterns

### Quick Start
```python
from aab_sgn import create_aab_sgn_model, AABSGNTrainer
```

### Custom Training Loop
```python
from aab_sgn import VagueSetMembership, AABLoss, AdaptiveKLThreshold
```

### Noise Injection Only
```python
from aab_sgn import NoisyLabelDataset, create_noisy_cifar10
```
