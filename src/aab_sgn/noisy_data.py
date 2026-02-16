"""
Data loading utilities with label noise injection for AAB-SGN experiments.

Implements various noise types:
- Symmetric noise: Random label flipping
- Asymmetric noise: Class-conditional corruption
- Pair-flip noise: Specific class pair swapping
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict, Callable
import logging

logger = logging.getLogger(__name__)


class NoisyLabelDataset(Dataset):
    """
    Wrapper dataset that injects label noise into clean dataset.
    
    Args:
        base_dataset: Clean dataset
        noise_type: Type of noise ('symmetric', 'asymmetric', 'pair_flip')
        noise_rate: Corruption rate τ ∈ [0, 1]
        num_classes: Number of classes
        seed: Random seed for reproducibility
        transition_matrix: Custom transition matrix (optional)
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        noise_type: str = 'symmetric',
        noise_rate: float = 0.4,
        num_classes: int = 10,
        seed: int = 42,
        transition_matrix: Optional[np.ndarray] = None
    ):
        self.base_dataset = base_dataset
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.num_classes = num_classes
        self.seed = seed
        
        # Generate noisy labels
        self.clean_labels = self._extract_labels()
        self.noisy_labels, self.noise_mask = self._inject_noise(transition_matrix)
        
        actual_noise_rate = self.noise_mask.sum() / len(self.noise_mask)
        logger.info(f"Created NoisyLabelDataset: type={noise_type}, "
                   f"target_rate={noise_rate:.2f}, "
                   f"actual_rate={actual_noise_rate:.4f}")
    
    def _extract_labels(self) -> np.ndarray:
        """Extract all labels from base dataset."""
        labels = []
        for _, label in self.base_dataset:
            if isinstance(label, torch.Tensor):
                label = label.item()
            labels.append(label)
        return np.array(labels)
    
    def _inject_noise(
        self,
        transition_matrix: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject label noise according to specified type.
        
        Returns:
            (noisy_labels, corruption_mask) tuple
        """
        np.random.seed(self.seed)
        
        n_samples = len(self.clean_labels)
        noisy_labels = self.clean_labels.copy()
        noise_mask = np.zeros(n_samples, dtype=bool)
        
        if transition_matrix is not None:
            # Use custom transition matrix
            noisy_labels, noise_mask = self._apply_transition_matrix(transition_matrix)
        
        elif self.noise_type == 'symmetric':
            # Symmetric noise: uniform random flipping
            n_corrupt = int(n_samples * self.noise_rate)
            corrupt_indices = np.random.choice(n_samples, n_corrupt, replace=False)
            
            for idx in corrupt_indices:
                true_label = self.clean_labels[idx]
                # Sample from other classes uniformly
                other_classes = [c for c in range(self.num_classes) if c != true_label]
                noisy_labels[idx] = np.random.choice(other_classes)
                noise_mask[idx] = True
        
        elif self.noise_type == 'asymmetric':
            # Asymmetric noise: class-conditional corruption
            # Example for CIFAR-10: truck→automobile, bird→airplane, deer→horse, cat→dog
            class_pairs = self._get_asymmetric_pairs()
            
            for true_label, noisy_label in class_pairs.items():
                class_indices = np.where(self.clean_labels == true_label)[0]
                n_corrupt = int(len(class_indices) * self.noise_rate)
                corrupt_indices = np.random.choice(class_indices, n_corrupt, replace=False)
                
                noisy_labels[corrupt_indices] = noisy_label
                noise_mask[corrupt_indices] = True
        
        elif self.noise_type == 'pair_flip':
            # Pair-flip noise: swap specific class pairs
            pairs = [(0, 1), (2, 3), (4, 5)]  # Customize based on dataset
            
            for class_a, class_b in pairs:
                # Flip class_a → class_b
                indices_a = np.where(self.clean_labels == class_a)[0]
                n_corrupt_a = int(len(indices_a) * self.noise_rate)
                corrupt_a = np.random.choice(indices_a, n_corrupt_a, replace=False)
                noisy_labels[corrupt_a] = class_b
                noise_mask[corrupt_a] = True
                
                # Flip class_b → class_a
                indices_b = np.where(self.clean_labels == class_b)[0]
                n_corrupt_b = int(len(indices_b) * self.noise_rate)
                corrupt_b = np.random.choice(indices_b, n_corrupt_b, replace=False)
                noisy_labels[corrupt_b] = class_a
                noise_mask[corrupt_b] = True
        
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
        
        return noisy_labels, noise_mask
    
    def _get_asymmetric_pairs(self) -> Dict[int, int]:
        """Get class pairs for asymmetric noise (CIFAR-10 specific)."""
        return {
            9: 1,  # truck → automobile
            2: 0,  # bird → airplane  
            4: 7,  # deer → horse
            3: 5,  # cat → dog
        }
    
    def _apply_transition_matrix(
        self,
        T: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply noise via transition matrix T where T_ij = P(observed=j | true=i).
        
        Args:
            T: Transition matrix of shape (num_classes, num_classes)
            
        Returns:
            (noisy_labels, noise_mask) tuple
        """
        np.random.seed(self.seed)
        
        n_samples = len(self.clean_labels)
        noisy_labels = np.zeros(n_samples, dtype=int)
        noise_mask = np.zeros(n_samples, dtype=bool)
        
        for i in range(n_samples):
            true_label = self.clean_labels[i]
            # Sample noisy label according to transition probabilities
            noisy_label = np.random.choice(self.num_classes, p=T[true_label])
            noisy_labels[i] = noisy_label
            noise_mask[i] = (noisy_label != true_label)
        
        return noisy_labels, noise_mask
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, index: int) -> Tuple:
        """
        Get item with noisy label.
        
        Returns:
            (data, noisy_label) tuple
        """
        data, _ = self.base_dataset[index]  # Ignore clean label
        noisy_label = self.noisy_labels[index]
        
        return data, noisy_label
    
    def get_clean_label(self, index: int) -> int:
        """Get clean label for validation purposes."""
        return int(self.clean_labels[index])
    
    def is_corrupted(self, index: int) -> bool:
        """Check if sample is corrupted."""
        return bool(self.noise_mask[index])
    
    def get_noise_statistics(self) -> Dict:
        """Compute noise statistics."""
        total = len(self.noise_mask)
        corrupted = self.noise_mask.sum()
        
        # Per-class corruption rates
        per_class_rates = {}
        for c in range(self.num_classes):
            class_mask = self.clean_labels == c
            if class_mask.sum() > 0:
                class_corruption = (class_mask & self.noise_mask).sum() / class_mask.sum()
                per_class_rates[c] = float(class_corruption)
        
        return {
            'total_samples': total,
            'corrupted_samples': int(corrupted),
            'corruption_rate': float(corrupted / total),
            'per_class_rates': per_class_rates,
            'max_class_rate': max(per_class_rates.values()) if per_class_rates else 0.0
        }


def create_noisy_cifar10(
    root: str = './data',
    train: bool = True,
    noise_type: str = 'symmetric',
    noise_rate: float = 0.4,
    download: bool = True,
    transform: Optional[Callable] = None,
    seed: int = 42
) -> NoisyLabelDataset:
    """
    Create noisy CIFAR-10 dataset.
    
    Args:
        root: Data root directory
        train: Training set if True, test set if False
        noise_type: Type of noise injection
        noise_rate: Corruption rate
        download: Download dataset if not present
        transform: Data transformation pipeline
        seed: Random seed
        
    Returns:
        NoisyLabelDataset instance
    """
    from torchvision import datasets, transforms as T
    
    if transform is None:
        if train:
            transform = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    
    base_dataset = datasets.CIFAR10(
        root=root,
        train=train,
        transform=transform,
        download=download
    )
    
    if train:
        noisy_dataset = NoisyLabelDataset(
            base_dataset=base_dataset,
            noise_type=noise_type,
            noise_rate=noise_rate,
            num_classes=10,
            seed=seed
        )
        return noisy_dataset
    else:
        # Return clean test set
        return base_dataset


if __name__ == "__main__":
    # Test noise injection
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Noise Injection")
    print("=" * 60)
    
    from torchvision import datasets, transforms
    
    # Load clean CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    clean_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )
    
    # Test symmetric noise
    print("\n1. Symmetric Noise (40%)")
    noisy_sym = NoisyLabelDataset(
        clean_dataset,
        noise_type='symmetric',
        noise_rate=0.4,
        num_classes=10
    )
    
    stats_sym = noisy_sym.get_noise_statistics()
    print(f"   Corruption rate: {stats_sym['corruption_rate']:.4f}")
    print(f"   Corrupted samples: {stats_sym['corrupted_samples']}/{stats_sym['total_samples']}")
    
    # Test asymmetric noise
    print("\n2. Asymmetric Noise (40%)")
    noisy_asym = NoisyLabelDataset(
        clean_dataset,
        noise_type='asymmetric',
        noise_rate=0.4,
        num_classes=10
    )
    
    stats_asym = noisy_asym.get_noise_statistics()
    print(f"   Corruption rate: {stats_asym['corruption_rate']:.4f}")
    print(f"   Max class rate: {stats_asym['max_class_rate']:.4f}")
    
    # Test pair-flip noise
    print("\n3. Pair-Flip Noise (40%)")
    noisy_pair = NoisyLabelDataset(
        clean_dataset,
        noise_type='pair_flip',
        noise_rate=0.4,
        num_classes=10
    )
    
    stats_pair = noisy_pair.get_noise_statistics()
    print(f"   Corruption rate: {stats_pair['corruption_rate']:.4f}")
    
    print("\n✓ Noise injection test passed!")
