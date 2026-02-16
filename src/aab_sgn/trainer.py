"""
Training loop for AAB-SGN with support for both standalone and two-stage modes.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Callable
import logging
from tqdm import tqdm
import numpy as np

from .aab_core import TwoStageAABSGN, VagueSetMembership
from .adaptive_threshold import ModeSelector

logger = logging.getLogger(__name__)


class AABSGNTrainer:
    """
    Trainer for AAB-SGN with automatic mode selection and comprehensive logging.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Torch optimizer
        device: Device for training
        vague_set_params: Parameters for VagueSetMembership
        kl_threshold: Fixed KL threshold (None for adaptive)
        sgn_warmup_epochs: Epochs for SGN stage if two-stage mode
        log_interval: Logging frequency (batches)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        vague_set_params: Optional[Dict] = None,
        kl_threshold: Optional[float] = None,
        sgn_warmup_epochs: int = 10,
        log_interval: int = 50,
        scheduler: Optional[object] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.log_interval = log_interval
        self.scheduler = scheduler
        
        # Initialize vague set
        vague_params = vague_set_params or {}
        self.vague_set = VagueSetMembership(**vague_params)
        
        # Initialize AAB-SGN loss
        self.aab_sgn = TwoStageAABSGN(
            vague_set=self.vague_set,
            kl_threshold=kl_threshold or 1.5,
            sgn_warmup_epochs=sgn_warmup_epochs
        )
        
        # Initialize mode selector
        self.mode_selector = ModeSelector(
            fixed_threshold=kl_threshold,
            adaptive_config={'confidence_level': 0.05, 'tau_min': 0.5}
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.training_stats = []
        
        logger.info(f"AABSGNTrainer initialized on {device}")
    
    def train_epoch(self, epoch: int) -> Dict:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training statistics
        """
        self.model.train()
        self.current_epoch = epoch
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Statistics accumulators
        residuals_list = []
        confidence_list = []
        hesitation_list = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # Compute AAB-SGN loss with statistics
            loss, stats = self.aab_sgn(
                output, target,
                epoch=epoch,
                return_stats=True
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate stats
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Collect residuals for mode selection
            with torch.no_grad():
                residuals = self.aab_sgn.aab_loss.compute_residuals(output, target)
                residuals_list.append(residuals)
                confidence_list.append(stats.get('mean_confidence', 0.0))
                hesitation_list.append(stats.get('mean_hesitation', 0.0))
            
            # Update progress bar
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100. * correct / total:.2f}%",
                    'conf': f"{stats.get('mean_confidence', 0.0):.3f}"
                })
        
        # Epoch statistics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        # Update mode selector with epoch residuals
        all_residuals = torch.cat(residuals_list)
        _, mode_info = self.mode_selector.select_mode(residuals=all_residuals)
        
        epoch_stats = {
            'epoch': epoch,
            'train_loss': avg_loss,
            'train_acc': accuracy,
            'mean_confidence': np.mean(confidence_list),
            'mean_hesitation': np.mean(hesitation_list),
            'kl_divergence': mode_info.get('kl_divergence', 0.0),
            'kl_threshold': mode_info.get('threshold', self.aab_sgn.kl_threshold),
            'mode': mode_info.get('mode', 'unknown')
        }
        
        return epoch_stats
    
    @torch.no_grad()
    def validate(self) -> Dict:
        """
        Validate on validation set.
        
        Returns:
            Dictionary of validation statistics
        """
        self.model.eval()
        
        val_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in self.val_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            
            # Use standard cross-entropy for validation
            loss = nn.functional.cross_entropy(output, target)
            val_loss += loss.item()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        val_loss /= len(self.val_loader)
        accuracy = 100. * correct / total
        
        return {
            'val_loss': val_loss,
            'val_acc': accuracy
        }
    
    def train(
        self,
        num_epochs: int,
        early_stopping_patience: Optional[int] = None,
        save_checkpoint_path: Optional[str] = None
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Epochs to wait before early stopping (None = disabled)
            save_checkpoint_path: Path to save best model checkpoint
            
        Returns:
            Dictionary of training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            # Train epoch
            train_stats = self.train_epoch(epoch)
            
            # Validate
            val_stats = self.validate()
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_stats['val_loss'])
                else:
                    self.scheduler.step()
            
            # Combine stats
            epoch_stats = {**train_stats, **val_stats}
            self.training_stats.append(epoch_stats)
            
            # Log
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_stats['train_loss']:.4f}, "
                f"Train Acc: {train_stats['train_acc']:.2f}%, "
                f"Val Loss: {val_stats['val_loss']:.4f}, "
                f"Val Acc: {val_stats['val_acc']:.2f}%, "
                f"KL: {train_stats['kl_divergence']:.3f}, "
                f"Mode: {train_stats['mode']}"
            )
            
            # Save best model
            if val_stats['val_acc'] > best_val_acc:
                best_val_acc = val_stats['val_acc']
                patience_counter = 0
                
                if save_checkpoint_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_acc': best_val_acc,
                        'training_stats': self.training_stats
                    }, save_checkpoint_path)
                    logger.info(f"Saved best model (val_acc: {best_val_acc:.2f}%)")
            else:
                patience_counter += 1
            
            # Early stopping
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        logger.info(f"Training complete. Best val acc: {best_val_acc:.2f}%")
        
        return {
            'training_stats': self.training_stats,
            'best_val_acc': best_val_acc,
            'final_epoch': epoch
        }


if __name__ == "__main__":
    # Simple test with dummy data
    logging.basicConfig(level=logging.INFO)
    
    from torchvision.models import resnet18
    from torch.utils.data import TensorDataset
    
    # Create dummy dataset
    train_data = torch.randn(100, 3, 32, 32)
    train_labels = torch.randint(0, 10, (100,))
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    val_data = torch.randn(50, 3, 32, 32)
    val_labels = torch.randint(0, 10, (50,))
    val_dataset = TensorDataset(val_data, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Create model
    model = resnet18(num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    device = torch.device('cpu')
    
    # Create trainer
    trainer = AABSGNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device
    )
    
    # Train for 2 epochs (test)
    history = trainer.train(num_epochs=2)
    
    print("\n✓ Training loop test passed!")
    print(f"Final validation accuracy: {history['best_val_acc']:.2f}%")
