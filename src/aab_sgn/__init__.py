"""
AAB-SGN: Ambiguity-Aware Backpropagation with Symbolic Gradient Modulation

A PyTorch implementation of vague set-based gradient modulation for robust learning under label noise.

Reference:
    Mehreen, R., & Ramesh, R. (2026). AAB-SGN: Robust Learning Under Label Noise 
    via Hesitation-Aware Gradient Modulation. Journal of Experimental & Theoretical 
    Artificial Intelligence.

Example usage:
    >>> from aab_sgn import create_aab_sgn_model, AABSGNTrainer
    >>> model, aab_loss = create_aab_sgn_model(resnet18())
    >>> trainer = AABSGNTrainer(model, train_loader, val_loader, optimizer, device)
    >>> history = trainer.train(num_epochs=200)
"""

__version__ = "1.0.0"
__author__ = "Ramsha Mehreen, Renikunta Ramesh"
__email__ = "ramshamehreen2208@gmail.com, rr.mh@kitsw.ac.in"

from .aab_core import (
    VagueSetMembership,
    AABLoss,
    TwoStageAABSGN,
    create_aab_sgn_model
)

from .adaptive_threshold import (
    AdaptiveKLThreshold,
    ModeSelector
)

from .trainer import AABSGNTrainer

from .noisy_data import (
    NoisyLabelDataset,
    create_noisy_cifar10
)

__all__ = [
    'VagueSetMembership',
    'AABLoss',
    'TwoStageAABSGN',
    'create_aab_sgn_model',
    'AdaptiveKLThreshold',
    'ModeSelector',
    'AABSGNTrainer',
    'NoisyLabelDataset',
    'create_noisy_cifar10',
]
