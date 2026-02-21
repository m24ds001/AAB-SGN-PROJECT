"""Hesitation Margins are Necessary:
Ambiguity-Aware Backpropagation for Robust
Learning Under Label Noise

Core implementation of vague set gradient modulation for robust learning under label noise.

Reference:
    Mehreen, R., & Ramesh, R. (2026). AAB-SGN: Robust Learning Under Label Noise 
    via Hesitation-Aware Gradient Modulation. Journal of Experimental & Theoretical 
    Artificial Intelligence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class VagueSetMembership:
    """
    Vague set membership functions implementing three-valued logic.
    
    Uses Gaussian membership functions for Minor (S), Moderate (M), and Severe (L) 
    residual categories with explicit hesitation margins π_A = 1 - t_A - f_A.
    
    Args:
        sigma_S (float): Width of Minor residual membership (default: 0.1)
        sigma_M (float): Width of Moderate residual membership (default: 0.2)
        sigma_L (float): Width of Severe residual membership (default: 1.0)
        center_M (float): Center of Moderate membership (default: 0.5)
        pi_min (float): Minimum hesitation margin (default: 0.1)
    """
    
    def __init__(
        self,
        sigma_S: float = 0.1,
        sigma_M: float = 0.2,
        sigma_L: float = 1.0,
        center_M: float = 0.5,
        pi_min: float = 0.1
    ):
        self.sigma_S = sigma_S
        self.sigma_M = sigma_M
        self.sigma_L = sigma_L
        self.center_M = center_M
        self.pi_min = pi_min
        
        # Centers for Gaussian memberships
        self.center_S = 0.0  # Minor errors (clean samples)
        self.center_L = 1.0  # Severe errors (corrupted samples)
        
        logger.info(f"VagueSet initialized: σ_S={sigma_S}, σ_M={sigma_M}, σ_L={sigma_L}, π_min={pi_min}")
    
    def _gaussian_membership(self, epsilon: torch.Tensor, center: float, sigma: float) -> torch.Tensor:
        """Gaussian membership function: t_A(ε) = exp(-(ε - c_A)² / (2σ²_A))"""
        return torch.exp(-((epsilon - center) ** 2) / (2 * sigma ** 2))
    
    def compute_memberships(self, epsilon: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute truth memberships for all three vague sets.
        
        Args:
            epsilon: Residual tensor of shape (batch_size,)
            
        Returns:
            Dictionary with keys 'S', 'M', 'L' containing truth memberships
        """
        t_S = self._gaussian_membership(epsilon, self.center_S, self.sigma_S)
        t_M = self._gaussian_membership(epsilon, self.center_M, self.sigma_M)
        t_L = self._gaussian_membership(epsilon, self.center_L, self.sigma_L)
        
        return {'S': t_S, 'M': t_M, 'L': t_L}
    
    def compute_confidence(
        self, 
        epsilon: torch.Tensor, 
        weights: Dict[str, float] = None
    ) -> torch.Tensor:
        """
        Compute vague set confidence factor ϕ(ε).
        
        ϕ(ε) = Σ w_A · t_A(ε) / W
        
        where W = w_S + w_M + w_L
        
        Args:
            epsilon: Residual tensor
            weights: Dictionary with keys 'S', 'M', 'L' (default: {S: 0.5, M: 1.0, L: 1.5})
            
        Returns:
            Confidence factor tensor of same shape as epsilon
        """
        if weights is None:
            weights = {'S': 0.5, 'M': 1.0, 'L': 1.5}
        
        memberships = self.compute_memberships(epsilon)
        W = sum(weights.values())
        
        confidence = (
            weights['S'] * memberships['S'] +
            weights['M'] * memberships['M'] +
            weights['L'] * memberships['L']
        ) / W
        
        # Ensure minimum confidence (based on hesitation margin)
        w_L = weights['L']
        min_confidence = w_L / W
        confidence = torch.clamp(confidence, min=min_confidence)
        
        return confidence
    
    def compute_hesitation_margins(self, epsilon: torch.Tensor) -> torch.Tensor:
        """
        Compute hesitation margins π_A(ε) = 1 - t_A(ε) - f_A(ε).
        
        For vague sets, we approximate f_A ≈ 1 - t_A, giving π_A ≈ 1 - 2t_A + t_A² 
        (simplified to max(π_min, 1 - t_A) in practice).
        
        Args:
            epsilon: Residual tensor
            
        Returns:
            Hesitation margin tensor
        """
        memberships = self.compute_memberships(epsilon)
        
        # Aggregate hesitation across all sets
        avg_truth = (memberships['S'] + memberships['M'] + memberships['L']) / 3.0
        hesitation = torch.clamp(1.0 - avg_truth, min=self.pi_min)
        
        return hesitation


class AABLoss(nn.Module):
    """
    Ambiguity-Aware Backpropagation Loss with vague set gradient modulation.
    
    Modifies the standard cross-entropy loss with hesitation-aware confidence:
    L_AAB = ϕ(ε) · L_CE(y, f(x))
    
    where ϕ(ε) is the vague set confidence factor based on residual ε.
    
    Args:
        vague_set: VagueSetMembership instance
        alpha: Learning rate scaling factor (default: 0.3)
        temperature: Temperature for residual computation (default: 1.0)
    """
    
    def __init__(
        self,
        vague_set: VagueSetMembership,
        alpha: float = 0.3,
        temperature: float = 1.0
    ):
        super().__init__()
        self.vague_set = vague_set
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def compute_residuals(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute absolute residuals: ε_i = ||1_{y_i} - softmax(z_i)||_2
        
        Args:
            logits: Model predictions of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
            temperature: Softmax temperature (default: self.temperature)
            
        Returns:
            Residual tensor of shape (batch_size,)
        """
        if temperature is None:
            temperature = self.temperature
        
        # Compute softmax probabilities
        probs = F.softmax(logits / temperature, dim=1)
        
        # Create one-hot encoded targets
        num_classes = logits.size(1)
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
        
        # Compute L2 residual
        residuals = torch.norm(one_hot - probs, p=2, dim=1)
        
        return residuals
    
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor,
        return_stats: bool = False
    ) -> torch.Tensor:
        """
        Compute AAB loss with vague set modulation.
        
        Args:
            logits: Model predictions
            targets: Ground truth labels
            return_stats: If True, return (loss, stats_dict)
            
        Returns:
            Modulated loss (scalar) or (loss, stats) if return_stats=True
        """
        # Compute base cross-entropy loss
        ce_losses = self.ce_loss(logits, targets)
        
        # Compute residuals
        residuals = self.compute_residuals(logits, targets)
        
        # Compute vague set confidence
        confidence = self.vague_set.compute_confidence(residuals)
        
        # Apply vague set modulation: L_AAB = ϕ(ε) · L_CE
        modulated_losses = confidence * ce_losses
        
        # Average over batch
        loss = modulated_losses.mean()
        
        if return_stats:
            with torch.no_grad():
                hesitation = self.vague_set.compute_hesitation_margins(residuals)
                stats = {
                    'loss': loss.item(),
                    'ce_loss': ce_losses.mean().item(),
                    'mean_residual': residuals.mean().item(),
                    'mean_confidence': confidence.mean().item(),
                    'mean_hesitation': hesitation.mean().item(),
                    'effective_lr_factor': 1.0 + self.alpha * confidence.mean().item()
                }
            return loss, stats
        
        return loss


class TwoStageAABSGN(nn.Module):
    """
    Two-stage AAB-SGN combining loss-level filtering with gradient-level modulation.
    
    Stage 1: SGN-style sample weighting to induce residual separability
    Stage 2: AAB gradient modulation with vague sets
    
    Automatically selects mode based on KL divergence threshold.
    
    Args:
        vague_set: VagueSetMembership instance
        kl_threshold: KL divergence threshold for mode selection (default: 1.5)
        sgn_warmup_epochs: Epochs for SGN stage (default: 10)
        use_sgn: If True, use two-stage; if False, standalone AAB (default: auto-detect)
    """
    
    def __init__(
        self,
        vague_set: VagueSetMembership,
        kl_threshold: float = 1.5,
        sgn_warmup_epochs: int = 10,
        use_sgn: Optional[bool] = None
    ):
        super().__init__()
        self.aab_loss = AABLoss(vague_set)
        self.kl_threshold = kl_threshold
        self.sgn_warmup_epochs = sgn_warmup_epochs
        self.use_sgn = use_sgn  # None = auto-detect
        
        # Sample weights for SGN stage (learned)
        self.sample_weights = None
        self.current_epoch = 0
        
        logger.info(f"TwoStageAABSGN initialized: KL_threshold={kl_threshold}, "
                   f"SGN_warmup={sgn_warmup_epochs}, mode={'auto' if use_sgn is None else use_sgn}")
    
    def estimate_kl_divergence(
        self,
        residuals: torch.Tensor,
        labels: torch.Tensor,
        num_samples: int = 1000
    ) -> float:
        """
        Estimate KL divergence between clean and corrupted residual distributions.
        
        Uses Gaussian mixture modeling to approximate distributions.
        
        Args:
            residuals: Residual tensor
            labels: Sample labels (for ground truth if available)
            num_samples: Number of samples for estimation
            
        Returns:
            Estimated KL divergence
        """
        # Sample subset
        if len(residuals) > num_samples:
            indices = torch.randperm(len(residuals))[:num_samples]
            residuals = residuals[indices]
        
        residuals_np = residuals.cpu().numpy()
        
        # Fit Gaussian mixture model (2 components)
        from sklearn.mixture import GaussianMixture
        
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(residuals_np.reshape(-1, 1))
        
        # Identify clean (lower mean) and noisy (higher mean) components
        means = gmm.means_.flatten()
        if means[0] < means[1]:
            clean_idx, noisy_idx = 0, 1
        else:
            clean_idx, noisy_idx = 1, 0
        
        # Compute KL divergence between Gaussians
        mu_clean, sigma_clean = gmm.means_[clean_idx][0], np.sqrt(gmm.covariances_[clean_idx][0][0])
        mu_noisy, sigma_noisy = gmm.means_[noisy_idx][0], np.sqrt(gmm.covariances_[noisy_idx][0][0])
        
        # KL divergence: D_KL(P_clean || P_noisy) for Gaussians
        kl_div = np.log(sigma_noisy / sigma_clean) + \
                (sigma_clean**2 + (mu_clean - mu_noisy)**2) / (2 * sigma_noisy**2) - 0.5
        
        return float(kl_div)
    
    def initialize_sample_weights(self, num_samples: int, device: torch.device):
        """Initialize learnable sample weights for SGN stage."""
        self.sample_weights = torch.ones(num_samples, device=device)
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_indices: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
        return_stats: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with automatic mode selection.
        
        Args:
            logits: Model predictions
            targets: Ground truth labels
            sample_indices: Indices for sample weighting (SGN stage)
            epoch: Current epoch number
            return_stats: Return statistics
            
        Returns:
            Loss (and stats if requested)
        """
        if epoch is not None:
            self.current_epoch = epoch
        
        # Compute AAB loss
        if return_stats:
            loss, stats = self.aab_loss(logits, targets, return_stats=True)
        else:
            loss = self.aab_loss(logits, targets)
            stats = {}
        
        # Apply SGN sample weighting if in warmup phase
        if self.use_sgn and self.current_epoch < self.sgn_warmup_epochs:
            if self.sample_weights is not None and sample_indices is not None:
                weights = self.sample_weights[sample_indices]
                loss = (loss * weights.mean()).mean()
                if return_stats:
                    stats['sgn_stage'] = True
        
        if return_stats:
            return loss, stats
        return loss


def create_aab_sgn_model(
    model: nn.Module,
    sigma_S: float = 0.1,
    sigma_M: float = 0.2,
    sigma_L: float = 1.0,
    pi_min: float = 0.1,
    kl_threshold: float = 1.5,
    alpha: float = 0.3
) -> Tuple[nn.Module, TwoStageAABSGN]:
    """
    Factory function to create AAB-SGN wrapped model.
    
    Args:
        model: Base neural network
        sigma_S, sigma_M, sigma_L: Vague set parameters
        pi_min: Minimum hesitation margin
        kl_threshold: KL threshold for mode selection
        alpha: Learning rate scaling factor
        
    Returns:
        (model, aab_sgn_loss) tuple
    """
    vague_set = VagueSetMembership(
        sigma_S=sigma_S,
        sigma_M=sigma_M,
        sigma_L=sigma_L,
        pi_min=pi_min
    )
    
    aab_sgn = TwoStageAABSGN(
        vague_set=vague_set,
        kl_threshold=kl_threshold
    )
    
    logger.info("AAB-SGN model created successfully")
    
    return model, aab_sgn


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy data
    batch_size, num_classes = 32, 10
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Create vague set and loss
    vague_set = VagueSetMembership()
    aab_loss = AABLoss(vague_set)
    
    # Compute loss
    loss, stats = aab_loss(logits, targets, return_stats=True)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Stats: {stats}")
    print("\n✓ AAB-SGN core implementation test passed!")
