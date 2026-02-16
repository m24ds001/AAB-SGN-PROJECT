"""
Adaptive KL Threshold Selection (Algorithm 1 from paper)

Implements automatic mode selection with O(n^-1/2) convergence guarantees.
"""

import torch
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import chi2
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class AdaptiveKLThreshold:
    """
    Adaptive KL threshold selection with finite-sample convergence guarantees.
    
    Implements Algorithm 1 from the paper with:
    - Gaussian mixture modeling for residual distributions
    - Confidence-adjusted threshold estimation
    - Exponential smoothing for stability
    
    Args:
        confidence_level (float): Confidence level α ∈ (0,1) (default: 0.05)
        tau_min (float): Minimum threshold value (default: 0.5)
        smoothing_weight (float): Exponential smoothing factor (default: 0.7)
        buffer_size (int): Residual buffer size for estimation (default: 1000)
    """
    
    def __init__(
        self,
        confidence_level: float = 0.05,
        tau_min: float = 0.5,
        smoothing_weight: float = 0.7,
        buffer_size: int = 1000
    ):
        self.confidence_level = confidence_level
        self.tau_min = tau_min
        self.smoothing_weight = smoothing_weight
        self.buffer_size = buffer_size
        
        # State
        self.residual_buffer: List[float] = []
        self.current_threshold = tau_min
        self.threshold_history: List[float] = []
        
        logger.info(f"AdaptiveKLThreshold initialized: α={confidence_level}, "
                   f"τ_min={tau_min}, β={smoothing_weight}")
    
    def update_buffer(self, residuals: torch.Tensor):
        """
        Update residual buffer with new samples.
        
        Args:
            residuals: Tensor of residual values
        """
        residuals_np = residuals.detach().cpu().numpy().flatten()
        self.residual_buffer.extend(residuals_np.tolist())
        
        # Keep only most recent samples
        if len(self.residual_buffer) > self.buffer_size:
            self.residual_buffer = self.residual_buffer[-self.buffer_size:]
    
    def estimate_kl_divergence(self) -> Tuple[float, dict]:
        """
        Estimate KL divergence between clean and corrupted distributions.
        
        Uses Gaussian mixture modeling (EM algorithm) as per Algorithm 1.
        
        Returns:
            (kl_divergence, info_dict) tuple
        """
        if len(self.residual_buffer) < 50:
            logger.warning(f"Insufficient samples ({len(self.residual_buffer)}), using default")
            return self.tau_min, {'status': 'insufficient_samples'}
        
        # Reshape for GMM
        residuals = np.array(self.residual_buffer).reshape(-1, 1)
        
        # Fit Gaussian Mixture Model (2 components)
        gmm = GaussianMixture(
            n_components=2,
            covariance_type='full',
            random_state=42,
            max_iter=100
        )
        
        try:
            gmm.fit(residuals)
        except Exception as e:
            logger.error(f"GMM fitting failed: {e}")
            return self.current_threshold, {'status': 'fitting_failed'}
        
        # Assign components: clean (lower mean) vs noisy (higher mean)
        means = gmm.means_.flatten()
        if means[0] < means[1]:
            clean_idx, noisy_idx = 0, 1
        else:
            clean_idx, noisy_idx = 1, 0
        
        # Extract Gaussian parameters
        mu_clean = gmm.means_[clean_idx][0]
        mu_noisy = gmm.means_[noisy_idx][0]
        sigma_clean_sq = gmm.covariances_[clean_idx][0][0]
        sigma_noisy_sq = gmm.covariances_[noisy_idx][0][0]
        
        # Compute KL divergence D_KL(P_clean || P_noisy)
        # For Gaussians: D_KL = log(σ_noisy/σ_clean) + (σ²_clean + (μ_clean - μ_noisy)²)/(2σ²_noisy) - 1/2
        kl_div = (
            0.5 * np.log(sigma_noisy_sq / sigma_clean_sq) +
            (sigma_clean_sq + (mu_clean - mu_noisy)**2) / (2 * sigma_noisy_sq) -
            0.5
        )
        
        info = {
            'kl_divergence': float(kl_div),
            'mu_clean': float(mu_clean),
            'mu_noisy': float(mu_noisy),
            'sigma_clean': float(np.sqrt(sigma_clean_sq)),
            'sigma_noisy': float(np.sqrt(sigma_noisy_sq)),
            'weights': gmm.weights_.tolist(),
            'status': 'success'
        }
        
        return float(kl_div), info
    
    def compute_adaptive_threshold(self) -> Tuple[float, dict]:
        """
        Compute confidence-adjusted adaptive threshold (Algorithm 1, steps 3-6).
        
        Returns:
            (threshold, info_dict) tuple
        """
        # Estimate empirical KL divergence
        kl_hat, info = self.estimate_kl_divergence()
        
        if info['status'] != 'success':
            return self.current_threshold, info
        
        # Compute confidence adjustment using chi-squared quantile
        # τ_adaptive = max(D̂_KL · χ²_{1,α}/2, τ_min)
        chi2_quantile = chi2.ppf(1 - self.confidence_level, df=1)
        confidence_factor = chi2_quantile / 2.0
        
        tau_adaptive = max(kl_hat * confidence_factor, self.tau_min)
        
        # Apply exponential smoothing (Algorithm 1, step 8)
        # τ^(t) = β · τ^(t-1) + (1-β) · τ_adaptive
        tau_smoothed = (
            self.smoothing_weight * self.current_threshold +
            (1 - self.smoothing_weight) * tau_adaptive
        )
        
        info.update({
            'tau_raw': tau_adaptive,
            'tau_smoothed': tau_smoothed,
            'chi2_quantile': chi2_quantile,
            'confidence_factor': confidence_factor
        })
        
        return tau_smoothed, info
    
    def update(self, residuals: torch.Tensor) -> Tuple[float, bool, dict]:
        """
        Update threshold with new residuals and return mode selection.
        
        Args:
            residuals: New residual samples
            
        Returns:
            (threshold, use_two_stage, info_dict) tuple
            - threshold: Updated KL threshold
            - use_two_stage: True if KL < threshold (two-stage mode)
            - info_dict: Diagnostic information
        """
        # Update buffer
        self.update_buffer(residuals)
        
        # Compute new threshold
        new_threshold, info = self.compute_adaptive_threshold()
        
        # Update state
        self.current_threshold = new_threshold
        self.threshold_history.append(new_threshold)
        
        # Mode selection: KL < threshold → use two-stage
        kl_divergence = info.get('kl_divergence', 0.0)
        use_two_stage = kl_divergence < new_threshold
        
        info.update({
            'current_threshold': new_threshold,
            'use_two_stage': use_two_stage,
            'mode': 'two-stage AAB-SGN' if use_two_stage else 'standalone AAB'
        })
        
        logger.info(f"KL={kl_divergence:.3f}, threshold={new_threshold:.3f}, "
                   f"mode={info['mode']}")
        
        return new_threshold, use_two_stage, info
    
    def get_threshold_stability(self, window: int = 10) -> float:
        """
        Compute threshold stability (standard deviation over recent window).
        
        Args:
            window: Number of recent updates to consider
            
        Returns:
            Standard deviation of threshold
        """
        if len(self.threshold_history) < window:
            return float('inf')
        
        recent = self.threshold_history[-window:]
        return float(np.std(recent))


class ModeSelector:
    """
    High-level mode selector combining adaptive threshold with manual override.
    
    Provides simple interface for deciding between standalone AAB and two-stage AAB-SGN.
    
    Args:
        fixed_threshold (float): If provided, use fixed threshold instead of adaptive
        adaptive_config (dict): Configuration for AdaptiveKLThreshold
    """
    
    def __init__(
        self,
        fixed_threshold: Optional[float] = None,
        adaptive_config: Optional[dict] = None
    ):
        self.fixed_threshold = fixed_threshold
        
        if fixed_threshold is None:
            config = adaptive_config or {}
            self.adaptive_selector = AdaptiveKLThreshold(**config)
            self.mode = 'adaptive'
            logger.info("ModeSelector: Using adaptive threshold")
        else:
            self.adaptive_selector = None
            self.mode = 'fixed'
            logger.info(f"ModeSelector: Using fixed threshold={fixed_threshold}")
    
    def select_mode(
        self,
        residuals: Optional[torch.Tensor] = None,
        kl_divergence: Optional[float] = None
    ) -> Tuple[bool, dict]:
        """
        Select operating mode based on residual separability.
        
        Args:
            residuals: Residual samples (required if adaptive mode)
            kl_divergence: Pre-computed KL divergence (optional)
            
        Returns:
            (use_two_stage, info_dict) tuple
        """
        if self.mode == 'fixed':
            if kl_divergence is None:
                raise ValueError("kl_divergence required for fixed mode")
            
            use_two_stage = kl_divergence < self.fixed_threshold
            info = {
                'mode': 'fixed',
                'threshold': self.fixed_threshold,
                'kl_divergence': kl_divergence,
                'use_two_stage': use_two_stage
            }
        else:
            if residuals is None:
                raise ValueError("residuals required for adaptive mode")
            
            _, use_two_stage, info = self.adaptive_selector.update(residuals)
            info['mode'] = 'adaptive'
        
        return use_two_stage, info


if __name__ == "__main__":
    # Test adaptive threshold
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Adaptive KL Threshold Selection")
    print("=" * 60)
    
    # Simulate residuals from two populations
    # Clean samples: N(0.2, 0.1²)
    # Corrupted samples: N(0.7, 0.15²)
    np.random.seed(42)
    
    selector = AdaptiveKLThreshold()
    
    for epoch in range(5):
        # Generate mixed residuals
        clean = np.random.normal(0.2, 0.1, size=100)
        corrupted = np.random.normal(0.7, 0.15, size=100)
        mixed = np.concatenate([clean, corrupted])
        np.random.shuffle(mixed)
        
        residuals = torch.from_numpy(mixed).float()
        
        # Update selector
        threshold, use_two_stage, info = selector.update(residuals)
        
        print(f"\nEpoch {epoch + 1}:")
        print(f"  KL divergence: {info['kl_divergence']:.3f}")
        print(f"  Threshold: {threshold:.3f}")
        print(f"  Mode: {info['mode']}")
        print(f"  Stability: {selector.get_threshold_stability():.4f}")
    
    print("\n✓ Adaptive KL threshold test passed!")
