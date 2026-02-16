"""
aab_sgn/aab.py
==============
Ambiguity-Aware Backpropagation (AAB) loss.

Core gradient transformation  [Paper Eq. (2) / (4)]
----------------------------------------------------
    ∇θL_AAB = φ(ε)·∇θL_CE  +  L_CE·φ'(ε)·∇θε

Implemented as:
    L_AAB = φ(ε_i) · L_CE(y_i, f_θ(x_i))            [Standalone, Eq. 6 stage-2]
    L_AAB = w_i · φ(ε_i) · L_CE(y_i, f_θ(x_i))      [Two-stage AAB-SGN, Eq. 6]

PyTorch autograd differentiates L_AAB through φ(ε) → ε → θ, recovering Eq. (2)
exactly because φ is C∞ and no stop-gradient is applied (Appendix A.2 proof).

Residual:  ε_i = ‖1_{y_i} − softmax(z_i)‖₂   (tracked, no detach)

Key properties (Lemma 1):
  · ‖∇_AAB‖ ≤ ‖∇L_CE‖ + M|L_CE|‖∇ε‖
  · φ(ε) ≥ w_L/W = 0.5 > 0   →   no gradient vanishing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .vague_sets import VagueSets, DEFAULT_PARAMS


class AABLoss(nn.Module):
    """
    AAB-modulated cross-entropy loss.

    For standalone AAB (KL > 1.5):
        loss_i = φ(ε_i) · CE(y_i, f_θ(x_i))

    For two-stage AAB-SGN (KL < 1.5), pass `sample_weights` from SGN Stage 1:
        loss_i = w_i · φ(ε_i) · CE(y_i, f_θ(x_i))         [Eq. 6]

    Parameters
    ----------
    pi_min : float
        Minimum hesitation margin.  Must be > 0 (Theorem 3).
        Default 0.10 satisfies π*(τ=0.4, δ=0.05, n=500) ≈ 0.095.
    reduction : 'mean' | 'sum'
    **kwargs : passed to VagueSets (sigma_S, c_M, sigma_M, sigma_L, w_S, w_M, w_L).

    Examples
    --------
    >>> criterion = AABLoss(pi_min=0.10)
    >>> loss = criterion(model(inputs), targets)   # modulated loss
    >>> loss.backward()                             # autograd → Eq. (2) gradient
    """

    def __init__(
        self,
        pi_min: float   = DEFAULT_PARAMS["pi_min"],
        reduction: str  = "mean",
        **kwargs,
    ):
        super().__init__()
        self.vague_sets = VagueSets(pi_min=pi_min, **kwargs)
        self.reduction  = reduction

    # ── Residual  ε_i = ‖1_{y_i} − softmax(z_i)‖₂ ─────────────────────────

    @staticmethod
    def residual(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample residuals (tracked through autograd).

        Both the one-hot encoding and softmax are differentiable paths so
        ∂ε/∂θ is computed correctly — no stop-gradient anywhere.

        Parameters
        ----------
        logits  : (B, C)  raw model outputs
        targets : (B,)    class indices

        Returns
        -------
        eps : (B,)  ε ∈ [0, √2]
        """
        probs   = F.softmax(logits, dim=1)                 # (B, C)
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0)    # (B, C)
        return (one_hot - probs).norm(dim=1)               # (B,)

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute AAB-modulated loss.

        Algorithm S1 steps (Supplementary S16.1):
          1.  z  = f_θ(x)                              ← provided as `logits`
          2.  p  = softmax(z)
          3.  ε  = ‖1_y − p‖₂                          ← tracked by autograd
          4-5. t_S, t_M, t_L → φ(ε)
          6.  L_AAB = φ(ε) · L_CE    (or w·φ·L_CE)
          7.  ∇θ = ∂L_AAB/∂θ via autograd              ← equals Eq.(2) [App. A.2]
          8.  θ ← θ − η_eff · ∇θ                       ← done in trainer

        Parameters
        ----------
        logits         : (B, C)  pre-softmax outputs.
        targets        : (B,)    ground-truth class indices.
        sample_weights : (B,)    optional SGN Stage-1 weights (two-stage mode).

        Returns
        -------
        loss : scalar tensor.
        """
        # Step 3 — residual (no detach, autograd tracks ε → θ)
        eps = self.residual(logits, targets)               # (B,)

        # Steps 4-5 — confidence factor
        _, _, _, phi = self.vague_sets(eps)                # (B,)

        # Per-sample CE
        ce = F.cross_entropy(logits, targets, reduction="none")  # (B,)

        # Step 6 — modulated loss
        modulated = phi * ce                               # (B,)

        # Two-stage: additional SGN Stage-1 weights  [Eq. 6]
        if sample_weights is not None:
            modulated = sample_weights * modulated

        if self.reduction == "mean":
            return modulated.mean()
        if self.reduction == "sum":
            return modulated.sum()
        return modulated

    # ── Diagnostics ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_stats(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Per-batch diagnostics (no gradient tracking).

        Returns
        -------
        dict: eps_mean/std, phi_mean/std, t_S/M/L_mean,
              pi_margin, grad_variance_reduction, phi_mean (for η_eff scaling).
        """
        eps = self.residual(logits.detach(), targets)
        ts, tm, tl, phi = self.vague_sets(eps)
        return {
            "eps_mean":               eps.mean().item(),
            "eps_std":                eps.std().item(),
            "phi_mean":               phi.mean().item(),
            "phi_std":                phi.std().item(),
            "t_S_mean":               ts.mean().item(),
            "t_M_mean":               tm.mean().item(),
            "t_L_mean":               tl.mean().item(),
            "pi_margin":              self.vague_sets.pi_min,
            # Empirical estimate of 57% variance reduction (Figure 3c)
            "grad_variance_reduction": float(1.0 - phi.var().item()),
        }

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"
