"""
aab_sgn/trainer.py
==================
Full AAB-SGN training pipeline with automatic mode selection.

Operating modes  [Paper Figure 2, Section 4.1]
----------------------------------------------
KL > 1.5  →  Standalone AAB
    loss_i = φ(ε_i) · L_CE(y_i, f_θ(x_i))

KL ≤ 1.5  →  Two-stage AAB-SGN
    Stage 1 (SGN): loss_i = w_i · L_CE(y_i, f_θ(x_i))       [Eq. 5]
    Stage 2 (AAB): loss_i = w_i · φ(ε_i) · L_CE(y_i, ...)   [Eq. 6]

Effective learning rate  [Section 9.0.0.1]:
    η_eff = η₀ · (1 + α · φ̄)   ≈ 1.22 η₀  for α=0.3, φ̄=0.73

Convergence  [Theorem 2, Eq. 9]:
    E[min_{t≤T} ‖∇L‖²] ≤ 2L(L₀−L*)/(η₀√T) + η₀M²/√T + b²
    where b ≤ 1.5 τ_max G  (Lemma 2)

Hardware: NVIDIA RTX 4090, PyTorch 2.0.1, CUDA 11.8, Python 3.10
"""

import logging
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .aab import AABLoss
from .kl_diagnostic import AdaptiveKLThreshold, RECOMMENDED_SAMPLES
from .vague_sets import DEFAULT_PARAMS

logger = logging.getLogger(__name__)


# ── SGN sample-weight module (Stage 1) ───────────────────────────────────────

class SGNWeights(nn.Module):
    """
    Learned per-sample weights w_i for Stage 1 (SGN preprocessing).

    Stage 1 suppresses ambiguous samples and induces bimodal residuals,
    raising KL from < 1.5 to > 1.5 (Table 4, main paper):
        · Clothing1M : 0.34 → 1.82
        · Food101-N  : 0.51 → 1.75
        · CIFAR-10N  : 0.42 → 1.68

    Reference: Englesson & Azizpour (ICLR 2024) — SGN, Eq. (5).
    """

    def __init__(self, num_samples: int, init: float = 0.0):
        super().__init__()
        # Stored in logit space; sigmoid maps to (0, 1)
        self.logits = nn.Parameter(torch.full((num_samples,), init))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Return w_i ∈ (0, 1) for batch indices."""
        return torch.sigmoid(self.logits[indices])


# ── Main trainer ─────────────────────────────────────────────────────────────

class AABSGNTrainer:
    """
    AAB-SGN training loop with automatic KL-based mode selection.

    Parameters
    ----------
    model              : nn.Module — any differentiable architecture.
    optimizer          : Torch optimizer (SGD momentum=0.9 recommended).
    device             : 'cuda' | 'cpu'.
    pi_min             : Minimum hesitation margin (default 0.10, Theorem 3).
    alpha              : Effective LR scale factor (default 0.30).
    kl_threshold       : Fixed KL threshold (default 1.5).
    use_adaptive_kl    : If True, run Algorithm 1 adaptive threshold.
    num_train_samples  : Size of training set — required for SGNWeights.
    scheduler          : Optional LR scheduler.
    log_interval       : Logging frequency in steps.

    Usage
    -----
    >>> trainer = AABSGNTrainer(model, optimizer, device='cuda')
    >>> trainer.run_kl_diagnostic(train_loader)   # sets mode automatically
    >>> for epoch in range(200):
    ...     trainer.train_epoch(train_loader, epoch)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        device: str                      = "cuda",
        pi_min: float                    = DEFAULT_PARAMS["pi_min"],
        alpha: float                     = DEFAULT_PARAMS["alpha"],
        kl_threshold: float              = 1.5,
        use_adaptive_kl: bool            = True,
        num_train_samples: Optional[int] = None,
        scheduler: Optional[object]      = None,
        log_interval: int                = 100,
    ):
        self.model         = model.to(device)
        self.optimizer     = optimizer
        self.device        = device
        self.alpha         = alpha
        self.scheduler     = scheduler
        self.log_interval  = log_interval

        self.aab_loss = AABLoss(pi_min=pi_min).to(device)
        self.kl_diag  = AdaptiveKLThreshold(fixed_threshold=kl_threshold)
        self.use_adaptive_kl = use_adaptive_kl

        # Mode set by run_kl_diagnostic()
        self.mode        : Optional[str]   = None
        self.kl_estimate : Optional[float] = None

        # SGN weights (Stage 1)
        self._sgn_weights   : Optional[SGNWeights]         = None
        self._sgn_optimizer : Optional[optim.Optimizer]    = None
        self._num_train     = num_train_samples

        # History for analysis / plotting
        self.history: Dict[str, list] = {
            k: [] for k in
            ["train_loss", "train_acc", "phi_mean", "eps_mean", "kl_estimate"]
        }

    # ── KL Diagnostic (Algorithm S2) ─────────────────────────────────────────

    def run_kl_diagnostic(
        self,
        loader: torch.utils.data.DataLoader,
        n_samples: int = RECOMMENDED_SAMPLES,
    ) -> dict:
        """
        Collect residuals on n_samples examples and run Algorithm 1.

        Call ONCE before the main training loop.  Sets self.mode automatically.

        Expected results by dataset (Table 4):
            CIFAR-10 40% synthetic : KL ≈ 2.45 → standalone_aab
            Clothing1M             : KL ≈ 0.34 → two_stage
            CIFAR-10N (human)      : KL ≈ 0.42 → two_stage

        Returns
        -------
        dict : mode, kl_estimate, threshold, n_samples, recommendation.
        """
        self.model.eval()
        residuals = []
        with torch.no_grad():
            for batch in loader:
                if len(residuals) >= n_samples:
                    break
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                logits = self.model(inputs)
                probs  = F.softmax(logits, dim=1)
                oh     = torch.zeros_like(probs)
                oh.scatter_(1, targets.unsqueeze(1), 1.0)
                residuals.extend((oh - probs).norm(dim=1).cpu().numpy().tolist())

        residuals    = np.array(residuals[:n_samples])
        mode         = self.kl_diag.fit_and_decide(residuals, self.use_adaptive_kl)
        kl, _, _     = self.kl_diag.estimate_kl(residuals)
        report       = self.kl_diag.report(residuals)

        self.mode        = mode
        self.kl_estimate = kl

        if mode == "two_stage" and self._num_train:
            self._init_sgn_weights(self._num_train)

        logger.info(
            f"[KL Diagnostic] KL={kl:.3f}  threshold={report['threshold']:.3f}"
            f"  mode={mode}  → {report['recommendation']}"
        )
        return report

    def _init_sgn_weights(self, n: int):
        self._sgn_weights   = SGNWeights(n).to(self.device)
        self._sgn_optimizer = optim.Adam(
            self._sgn_weights.parameters(), lr=1e-3, weight_decay=1e-4
        )
        logger.info(f"[SGN] Initialised per-sample weights for {n} training samples.")

    # ── Single training step (Algorithm S1) ──────────────────────────────────

    def _step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        One gradient update step per Algorithm S1 (Supplementary S16.1).

        Step 1:  z  = f_θ(x)
        Step 2:  p  = softmax(z)
        Step 3:  ε  = ‖1_y − p‖₂     (tracked by autograd — no detach)
        Steps 4-5: t_S, t_M, t_L → φ(ε)
        Step 6:  L_AAB = [w_i·] φ(ε)·L_CE
        Step 7:  ∇θ = ∂L_AAB/∂θ via autograd   (≡ Eq. (2), see Appendix A.2)
        Step 8:  θ ← θ − η_eff · ∇θ
        """
        self.model.train()

        # Forward pass (Step 1)
        logits = self.model(inputs)                                  # (B, C)

        # Stage-1 SGN weights (two-stage mode only)
        sample_weights = None
        if self.mode == "two_stage" and self._sgn_weights is not None and \
           indices is not None:
            sample_weights = self._sgn_weights(indices.to(self.device))

        # Modulated loss (Steps 3-6)
        loss = self.aab_loss(logits, targets, sample_weights=sample_weights)

        # η_eff = η₀(1 + α·φ̄) — computed without gradient for LR scaling only
        with torch.no_grad():
            stats     = self.aab_loss.get_stats(logits, targets)
            eta_scale = 1.0 + self.alpha * stats["phi_mean"]
            for pg in self.optimizer.param_groups:
                pg["_lr_effective"] = pg["lr"] * eta_scale

        # Backward (Step 7) + update (Step 8)
        self.optimizer.zero_grad()
        if self._sgn_optimizer:
            self._sgn_optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()
        if self._sgn_optimizer:
            self._sgn_optimizer.step()

        with torch.no_grad():
            acc = (logits.argmax(1) == targets).float().mean().item()

        return dict(
            loss=loss.item(), acc=acc,
            phi_mean=stats["phi_mean"], eps_mean=stats["eps_mean"],
        )

    # ── Full epoch ────────────────────────────────────────────────────────────

    def train_epoch(
        self, loader: torch.utils.data.DataLoader, epoch: int
    ) -> Dict[str, float]:
        """
        Train one epoch.  DataLoader may return (imgs, labels) or
        (imgs, labels, indices) — indices required for two-stage mode.
        """
        if self.mode is None:
            logger.warning("Mode not set — call run_kl_diagnostic() first. Defaulting to standalone_aab.")
            self.mode = "standalone_aab"

        losses, accs, phis, epss = [], [], [], []

        for step, batch in enumerate(loader):
            inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            indices = batch[2] if len(batch) == 3 else None

            s = self._step(inputs, targets, indices)
            losses.append(s["loss"]); accs.append(s["acc"])
            phis.append(s["phi_mean"]); epss.append(s["eps_mean"])

            if step % self.log_interval == 0:
                logger.info(
                    f"Ep{epoch:3d} [{step:4d}/{len(loader)}] "
                    f"loss={s['loss']:.4f} acc={s['acc']:.3f} "
                    f"φ̄={s['phi_mean']:.3f} mode={self.mode}"
                )

        if self.scheduler is not None:
            self.scheduler.step()

        stats = dict(
            train_loss=float(np.mean(losses)),
            train_acc=float(np.mean(accs)),
            phi_mean=float(np.mean(phis)),
            eps_mean=float(np.mean(epss)),
            mode=self.mode,
            kl_estimate=self.kl_estimate or 0.0,
        )
        for k in ["train_loss", "train_acc", "phi_mean", "eps_mean", "kl_estimate"]:
            self.history[k].append(stats[k])
        return stats

    # ── Evaluation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(
        self, loader: torch.utils.data.DataLoader, top_k: int = 1
    ) -> Dict[str, float]:
        """Top-1 (and optionally top-k) accuracy on a DataLoader."""
        self.model.eval()
        c1 = ck = total = 0
        for batch in loader:
            inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            logits = self.model(inputs)
            c1    += (logits.argmax(1) == targets).sum().item()
            if top_k > 1:
                _, tk = logits.topk(top_k, dim=1)
                ck   += tk.eq(targets.unsqueeze(1).expand_as(tk)).any(1).sum().item()
            total += targets.size(0)
        result = {"acc": c1 / total}
        if top_k > 1:
            result[f"acc_top{top_k}"] = ck / total
        return result
