"""
aab_sgn/kl_diagnostic.py
========================
Adaptive KL-threshold mode selector  [Paper Algorithm 1, Section 5.2]

Algorithm 1 (Adaptive KL Threshold Selection)
----------------------------------------------
Input:  residual buffer R = {ε₁,...,εₙ}, confidence level α ∈ (0,1)
Output: adaptive threshold τ^(t)_KL

1. Fit 2-component GMM via EM: P̂ = Σₖ wₖ N(μₖ, σ²ₖ)
2. Assign P̂_clean = N(μ_min, σ²_min)  where μ_min = argmin μₖ
3. Compute empirical KL: D̂_KL = ∫ p̂_clean(x) log(p̂_clean(x)/p̂_noisy(x)) dx
5-6. τ^adaptive_KL = max(D̂_KL · χ²_{1,α}/2,  τ_min = 0.5)
8-9. Smoothing (heuristic): τ^(t)_KL = 0.7·τ^(t-1)_KL + 0.3·τ^adaptive_KL

Notes from paper
----------------
· Threshold τ_KL = 1.5 was selected by grid-search on CIFAR-10 val (Limitation 3).
· Smoothing β₁ = 0.7 is a heuristic; stable ±0.15pp for β₁ ∈ [0.6, 0.8] (Table S22).
· Theorem 1 applies to the UNSMOOTHED τ^adaptive_KL (Limitation 4).
· The unsmoothed estimator achieves O_n(n^{-1/2}) via MLE delta method (Revised S12.2).
· Adaptive threshold achieves +0.23pp over fixed on 12 datasets (Table 5).
"""

import warnings
import numpy as np
from scipy.stats import chi2
from sklearn.mixture import GaussianMixture
from typing import Tuple


# ── Constants matching paper ──────────────────────────────────────────────────
FIXED_KL_THRESHOLD   = 1.5   # Grid-searched on CIFAR-10 val (Section 5.2, Limitation 3)
MIN_KL_THRESHOLD     = 0.5   # τ_min in Algorithm 1 line 6
SMOOTHING_BETA       = 0.7   # β₁ from Algorithm 1 line 9
MIN_SAMPLES_EM       = 50    # n ≥ max(50, 10/π) for EM convergence (Theorem 1)
RECOMMENDED_SAMPLES  = 500   # For Theorem 1 finite-sample bound (Supp. S12.2)


class AdaptiveKLThreshold:
    """
    Online adaptive KL threshold with exponential smoothing (Algorithm 1).

    Parameters
    ----------
    alpha          : float  Confidence level for χ² correction (default 0.05).
    smoothing_beta : float  β₁ ∈ [0.6, 0.8].  0.7 used in paper.
    fixed_threshold: float  Fallback when insufficient samples.  Default 1.5.
    """

    def __init__(
        self,
        alpha: float           = 0.05,
        smoothing_beta: float  = SMOOTHING_BETA,
        fixed_threshold: float = FIXED_KL_THRESHOLD,
    ):
        self.alpha           = alpha
        self.beta            = smoothing_beta
        self.fixed_threshold = fixed_threshold
        self._smoothed_tau   = None   # τ^(t)_KL running estimate

    # ── Closed-form Gaussian KL (Theorem 1 / Section 5.2.1) ─────────────────

    @staticmethod
    def gaussian_kl(mu1: float, s1: float, mu2: float, s2: float) -> float:
        """
        D_KL(N(μ₁,σ₁²) ‖ N(μ₂,σ₂²)) = log(σ₂/σ₁) + (σ₁²+(μ₁−μ₂)²)/(2σ₂²) − ½

        This C∞ functional of (μ₁,σ₁,μ₂,σ₂) is used in the delta-method proof
        of Theorem 1 convergence rate (Supplementary S12.2 / S19.2).
        """
        if s1 <= 0 or s2 <= 0:
            return 0.0
        return (
            np.log(s2 / s1)
            + (s1 ** 2 + (mu1 - mu2) ** 2) / (2.0 * s2 ** 2)
            - 0.5
        )

    # ── EM-based KL estimator ─────────────────────────────────────────────────

    def estimate_kl(
        self, residuals: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Algorithm 1, steps 1-4: fit 2-component GMM and compute D̂_KL.

        Returns
        -------
        kl_estimate : float
        mu_clean    : float  mean of lower (clean) component
        mu_noisy    : float  mean of upper (noisy) component
        """
        n = len(residuals)
        if n < MIN_SAMPLES_EM:
            warnings.warn(
                f"Only {n} residuals; need ≥ {MIN_SAMPLES_EM} for EM "
                f"(Theorem 1). Using fixed threshold {self.fixed_threshold}."
            )
            return self.fixed_threshold, 0.0, 1.0

        X = residuals.reshape(-1, 1)
        gmm = GaussianMixture(
            n_components=2, covariance_type="full",
            max_iter=300, random_state=42,
        )
        try:
            gmm.fit(X)
        except Exception as e:
            warnings.warn(f"GMM fitting failed ({e}). Using fixed threshold.")
            return self.fixed_threshold, 0.0, 1.0

        # Algorithm 1 step 2: assign P̂_clean = component with lower mean
        means = gmm.means_.flatten()
        stds  = np.sqrt(gmm.covariances_.flatten())
        c, no = (np.argmin(means), np.argmax(means))

        kl = max(self.gaussian_kl(means[c], stds[c], means[no], stds[no]), 0.0)
        return kl, means[c], means[no]

    # ── Algorithm 1 (full) ───────────────────────────────────────────────────

    def compute_adaptive_threshold(self, residuals: np.ndarray) -> float:
        """
        Algorithm 1 complete: estimate KL, apply χ² correction, smooth.

        Returns
        -------
        smoothed_threshold : float  τ^(t)_KL
        """
        kl, _, _ = self.estimate_kl(residuals)

        # Lines 5-6: χ² confidence-adjusted threshold
        chi2_corr = chi2.ppf(self.alpha, df=1) / 2.0
        adaptive  = max(kl * chi2_corr, MIN_KL_THRESHOLD)

        # Lines 8-9: exponential smoothing (heuristic, not covered by Theorem 1)
        if self._smoothed_tau is None:
            self._smoothed_tau = adaptive
        else:
            self._smoothed_tau = (
                self.beta * self._smoothed_tau
                + (1.0 - self.beta) * adaptive
            )
        return self._smoothed_tau

    # ── Primary API ──────────────────────────────────────────────────────────

    def fit_and_decide(
        self,
        residuals: np.ndarray,
        use_adaptive: bool = True,
    ) -> str:
        """
        Run KL diagnostic and return operating mode.

        Parameters
        ----------
        residuals    : 1-D array of ε values.
        use_adaptive : If True, use Algorithm 1 adaptive threshold.
                       If False, use fixed threshold 1.5.

        Returns
        -------
        mode : 'standalone_aab'  (KL > threshold)
             | 'two_stage'        (KL ≤ threshold)
        """
        if len(residuals) < MIN_SAMPLES_EM:
            warnings.warn(
                f"Only {len(residuals)} samples; defaulting to 'two_stage' (conservative)."
            )
            return "two_stage"

        kl, _, _ = self.estimate_kl(residuals)
        threshold = (
            self.compute_adaptive_threshold(residuals)
            if use_adaptive
            else self.fixed_threshold
        )
        return "standalone_aab" if kl > threshold else "two_stage"

    def report(self, residuals: np.ndarray) -> dict:
        """
        Full diagnostic report (matches Table 5 output format).

        Returns
        -------
        dict: mode, kl_estimate, threshold, mu_clean, mu_noisy,
              n_samples, recommendation.
        """
        kl, mu_c, mu_n = self.estimate_kl(residuals)
        threshold = self._smoothed_tau or self.fixed_threshold
        mode      = "standalone_aab" if kl > threshold else "two_stage"

        # Deployment guidance (Section 8.3, Table S26)
        if kl > 1.7:
            rec = "Strong separability — Standalone AAB (KL >> 1.5)."
        elif kl > threshold:
            rec = "Marginal separability — Standalone AAB; monitor performance."
        elif kl > 1.3:
            rec = "Borderline KL ∈ [1.3,1.5] — try standalone first (Section 6.9)."
        elif kl > 0.5:
            rec = "Moderate overlap — Two-stage AAB-SGN recommended."
        else:
            rec = "Severe overlap KL < 0.5 — consider domain-specific methods (Table S18)."

        return dict(
            mode=mode,
            kl_estimate=round(kl, 4),
            threshold=round(threshold, 4),
            mu_clean=round(mu_c, 4),
            mu_noisy=round(mu_n, 4),
            n_samples=len(residuals),
            recommendation=rec,
        )
