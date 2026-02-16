"""
Training script for CIFAR-10 with AAB-SGN.

Example usage:
    python experiments/train_cifar10.py --noise-type symmetric --noise-rate 0.4 --epochs 200
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
import logging

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.aab_sgn import (
    create_aab_sgn_model,
    AABSGNTrainer,
    create_noisy_cifar10
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train AAB-SGN on CIFAR-10')
    
    # Data parameters
    parser.add_argument('--data-root', type=str, default='./data',
                        help='Root directory for data')
    parser.add_argument('--noise-type', type=str, default='symmetric',
                        choices=['symmetric', 'asymmetric', 'pair_flip'],
                        help='Type of label noise')
    parser.add_argument('--noise-rate', type=float, default=0.4,
                        help='Label corruption rate (0.0-1.0)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='Dataset to use')
    
    # Model parameters
    parser.add_argument('--arch', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='Model architecture')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # AAB-SGN parameters
    parser.add_argument('--sigma-S', type=float, default=0.1,
                        help='Minor residual width')
    parser.add_argument('--sigma-M', type=float, default=0.2,
                        help='Moderate residual width')
    parser.add_argument('--sigma-L', type=float, default=1.0,
                        help='Severe residual width')
    parser.add_argument('--pi-min', type=float, default=0.1,
                        help='Minimum hesitation margin')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Learning rate scaling factor')
    parser.add_argument('--kl-threshold', type=float, default=None,
                        help='KL threshold for mode selection (None for adaptive)')
    parser.add_argument('--sgn-warmup', type=int, default=10,
                        help='SGN warmup epochs for two-stage mode')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--save-path', type=str, default='./checkpoints',
                        help='Path to save checkpoints')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='Logging interval (batches)')
    
    return parser.parse_args()


def create_data_loaders(args):
    """Create train and validation data loaders."""
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Datasets
    num_classes = 10 if args.dataset == 'cifar10' else 100
    
    # Create noisy training set
    train_dataset = create_noisy_cifar10(
        root=args.data_root,
        train=True,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        download=True,
        transform=train_transform,
        seed=args.seed
    )
    
    # Log noise statistics
    noise_stats = train_dataset.get_noise_statistics()
    logger.info(f"Noise statistics: {noise_stats}")
    
    # Clean test set
    if args.dataset == 'cifar10':
        test_dataset = datasets.CIFAR10(
            root=args.data_root,
            train=False,
            transform=test_transform,
            download=True
        )
    else:
        test_dataset = datasets.CIFAR100(
            root=args.data_root,
            train=False,
            transform=test_transform,
            download=True
        )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, num_classes


def create_model(args, num_classes):
    """Create model architecture."""
    
    if args.arch == 'resnet18':
        from torchvision.models import resnet18
        base_model = resnet18(num_classes=num_classes)
    elif args.arch == 'resnet34':
        from torchvision.models import resnet34
        base_model = resnet34(num_classes=num_classes)
    elif args.arch == 'resnet50':
        from torchvision.models import resnet50
        base_model = resnet50(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")
    
    # Wrap with AAB-SGN
    model, aab_loss = create_aab_sgn_model(
        base_model,
        sigma_S=args.sigma_S,
        sigma_M=args.sigma_M,
        sigma_L=args.sigma_L,
        pi_min=args.pi_min,
        kl_threshold=args.kl_threshold or 1.5,
        alpha=args.alpha
    )
    
    return model


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    train_loader, test_loader, num_classes = create_data_loaders(args)
    logger.info(f"Dataset: {args.dataset}, Classes: {num_classes}")
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    model = create_model(args, num_classes)
    logger.info(f"Model: {args.arch}")
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler (cosine annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )
    
    # Create checkpoint directory
    os.makedirs(args.save_path, exist_ok=True)
    checkpoint_path = os.path.join(
        args.save_path,
        f'{args.dataset}_{args.noise_type}_{args.noise_rate}_best.pth'
    )
    
    # Create trainer
    trainer = AABSGNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        device=device,
        vague_set_params={
            'sigma_S': args.sigma_S,
            'sigma_M': args.sigma_M,
            'sigma_L': args.sigma_L,
            'pi_min': args.pi_min
        },
        kl_threshold=args.kl_threshold,
        sgn_warmup_epochs=args.sgn_warmup,
        log_interval=args.log_interval,
        scheduler=scheduler
    )
    
    # Train
    logger.info("=" * 80)
    logger.info("Starting training")
    logger.info("=" * 80)
    
    history = trainer.train(
        num_epochs=args.epochs,
        save_checkpoint_path=checkpoint_path
    )
    
    # Print final results
    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info(f"Best validation accuracy: {history['best_val_acc']:.2f}%")
    logger.info(f"Model saved to: {checkpoint_path}")
    logger.info("=" * 80)
    
    # Save training history
    import json
    history_path = checkpoint_path.replace('.pth', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(history['training_stats'], f, indent=2)
    logger.info(f"Training history saved to: {history_path}")


if __name__ == '__main__':
    main()
