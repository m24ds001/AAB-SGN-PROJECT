"""
datasets/noisy_dataset.py
=========================
Dataset wrappers with configurable synthetic label noise.

Supported noise protocols  [Paper Section 6.1.0.2]:
    · Symmetric     : each label flipped to uniform random other class with prob τ
    · Asymmetric    : class-conditional (truck→automobile, bird→airplane, etc.)
    · Pair-flip     : class i → class (i+1) % C with prob τ

Paper seeds:  {0, 42, 123, 456, 789}  (Table 6 / Table S3)
Paper rates:  τ ∈ {0.2, 0.4, 0.6}  for CIFAR-10 experiments
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from typing import Optional, Tuple


# ── Noise functions ───────────────────────────────────────────────────────────

def symmetric_noise(
    labels: np.ndarray, num_classes: int, rate: float, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Flip each label to a uniformly random other class with probability τ."""
    rng   = np.random.RandomState(seed)
    noisy = labels.copy()
    n     = len(labels)
    idx   = rng.choice(n, size=int(n * rate), replace=False)
    for i in idx:
        choices = list(range(num_classes))
        choices.remove(int(labels[i]))
        noisy[i] = rng.choice(choices)
    mask = np.zeros(n, dtype=bool); mask[idx] = True
    return noisy, mask


def asymmetric_noise(
    labels: np.ndarray, num_classes: int, rate: float,
    transition: Optional[dict] = None, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Class-conditional noise.  Default CIFAR-10 map [Section 6.9, Table 12]:
        truck (9) → automobile (1)
        bird  (2) → airplane   (0)
        deer  (4) → horse      (7)
        cat   (3) → dog        (5)
    """
    if transition is None:
        transition = {9: 1, 2: 0, 4: 7, 3: 5}
    rng   = np.random.RandomState(seed)
    noisy = labels.copy()
    n     = len(labels)
    mask  = np.zeros(n, dtype=bool)
    for src, tgt in transition.items():
        src_idx = np.where(labels == src)[0]
        flip    = rng.choice(src_idx, size=int(len(src_idx) * rate), replace=False)
        noisy[flip] = tgt; mask[flip] = True
    return noisy, mask


def pair_flip_noise(
    labels: np.ndarray, num_classes: int, rate: float, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Flip class i → (i+1) % C with probability τ."""
    rng   = np.random.RandomState(seed)
    noisy = labels.copy()
    n     = len(labels)
    idx   = rng.choice(n, size=int(n * rate), replace=False)
    for i in idx:
        noisy[i] = (int(labels[i]) + 1) % num_classes
    mask = np.zeros(n, dtype=bool); mask[idx] = True
    return noisy, mask


# ── Dataset wrappers ──────────────────────────────────────────────────────────

class NoisyCIFAR10(datasets.CIFAR10):
    """
    CIFAR-10 with configurable synthetic label noise.

    Parameters
    ----------
    noise_rate   : τ ∈ {0.2, 0.4, 0.6} — paper experiments.
    noise_type   : 'symmetric' | 'asymmetric' | 'pair_flip'.
    seed         : RNG seed — paper uses {0, 42, 123, 456, 789}.
    return_index : If True, __getitem__ returns (img, label, index).
                   Required for two-stage mode (SGNWeights need indices).

    Expected results (Table 6, main paper):
        τ=0.4, symmetric, 5 seeds → AAB-SGN 94.23 ± 0.45%
    """

    def __init__(self, root, train=True, transform=None,
                 download=True, noise_rate=0.40, noise_type="symmetric",
                 seed=42, return_index=False):
        super().__init__(root, train=train, transform=transform, download=download)
        self.return_index = return_index
        self.clean_labels = np.array(self.targets).copy()
        if noise_rate > 0 and train:
            self._inject(noise_rate, noise_type, seed)

    def _inject(self, rate, kind, seed):
        lbl = np.array(self.targets)
        fn  = dict(symmetric=symmetric_noise,
                   asymmetric=asymmetric_noise,
                   pair_flip=pair_flip_noise)[kind]
        noisy, mask = fn(lbl, 10, rate, seed)
        self.targets    = noisy.tolist()
        self.noise_mask = mask
        print(f"[NoisyCIFAR10] type={kind} τ_req={rate:.2f} "
              f"τ_actual={mask.mean():.4f} seed={seed}")

    def __getitem__(self, idx):
        img, lbl = super().__getitem__(idx)
        return (img, lbl, idx) if self.return_index else (img, lbl)


class NoisyCIFAR100(datasets.CIFAR100):
    """CIFAR-100 with symmetric noise (Table S4, Supplementary)."""

    def __init__(self, root, train=True, transform=None,
                 download=True, noise_rate=0.40, seed=42, return_index=False):
        super().__init__(root, train=train, transform=transform, download=download)
        self.return_index = return_index
        self.clean_labels = np.array(self.targets).copy()
        if noise_rate > 0 and train:
            noisy, mask = symmetric_noise(np.array(self.targets), 100, noise_rate, seed)
            self.targets    = noisy.tolist()
            self.noise_mask = mask

    def __getitem__(self, idx):
        img, lbl = super().__getitem__(idx)
        return (img, lbl, idx) if self.return_index else (img, lbl)
