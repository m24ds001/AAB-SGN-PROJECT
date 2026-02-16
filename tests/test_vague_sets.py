"""
tests/test_vague_sets.py
========================
Unit tests verifying all mathematical properties stated in the paper.
Run with:  pytest tests/test_vague_sets.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, numpy as np, pytest
from aab_sgn.vague_sets import VagueSets, DEFAULT_PARAMS
from aab_sgn.aab import AABLoss
from aab_sgn.kl_diagnostic import AdaptiveKLThreshold


class TestVagueSets:
    def setup_method(self):
        self.vs  = VagueSets()
        self.eps = torch.linspace(0, 2.5, 500)

    def test_pi_min_zero_raises(self):
        """Theorem 3 necessity: π_min=0 must raise ValueError."""
        with pytest.raises(ValueError, match="pi_min must be strictly > 0"):
            VagueSets(pi_min=0.0)

    def test_pi_min_negative_raises(self):
        with pytest.raises(ValueError): VagueSets(pi_min=-0.05)

    def test_t_S_peaks_at_zero(self):
        """Minor set peaks at ε=0 (clean samples)."""
        ts = self.vs.t_S(self.eps)
        assert ts[0].item() == pytest.approx(1.0, abs=1e-5)

    def test_t_M_peaks_at_c_M(self):
        """Moderate set peaks at c_M=0.5."""
        e = torch.linspace(0, 1.5, 1000)
        peak_eps = e[self.vs.t_M(e).argmax()].item()
        assert abs(peak_eps - self.vs.c_M) < 0.01

    def test_t_L_monotone(self):
        """Severe set is monotone non-decreasing (corrupted labels)."""
        tl = self.vs.t_L(self.eps)
        assert (tl[1:] - tl[:-1] >= -1e-6).all()

    def test_phi_lower_bound(self):
        """φ(ε) ≥ w_L/W = 0.5 — prevents gradient vanishing [Lemma 1]."""
        phi = self.vs.confidence(self.eps)
        assert phi.min().item() >= self.vs.w_L / self.vs.W - 1e-5

    def test_phi_upper_bound(self):
        """φ(ε) ≤ 1.0."""
        phi = self.vs.confidence(self.eps)
        assert phi.max().item() <= 1.0 + 1e-5

    def test_W_default(self):
        """W = w_S + w_M + w_L = 3.0."""
        assert self.vs.W == pytest.approx(3.0)

    def test_pi_star_value(self):
        """π*(0.4, 0.05, 500) ≈ 0.095  [Theorem 3 Eq.11]."""
        v = self.vs.pi_star(0.4, 0.05, 500)
        assert abs(v - 0.095) < 0.003

    def test_pi_star_default_satisfied(self):
        """Default π_min=0.10 must satisfy theoretical threshold."""
        assert DEFAULT_PARAMS["pi_min"] >= self.vs.pi_star(0.4, 0.05, 500)

    def test_pi_star_decreases_with_n(self):
        """Larger n → smaller required margin (more data = better noise estimate)."""
        assert self.vs.pi_star(0.4, 0.05, 5000) < self.vs.pi_star(0.4, 0.05, 500)

    def test_phi_differentiable(self):
        """φ(ε) is C∞ — gradient must exist at all ε > 0."""
        e = torch.linspace(0.01, 2.0, 50).requires_grad_(True)
        self.vs.confidence(e).sum().backward()
        assert e.grad is not None and not torch.isnan(e.grad).any()

    def test_forward_four_outputs(self):
        """forward() returns (t_S, t_M, t_L, phi) all same shape."""
        ts, tm, tl, phi = self.vs(self.eps)
        assert ts.shape == tm.shape == tl.shape == phi.shape == self.eps.shape


class TestAABLoss:
    def setup_method(self):
        self.fn = AABLoss(pi_min=0.10)
        self.B, self.C = 16, 10

    def _batch(self):
        return torch.randn(self.B, self.C, requires_grad=True), \
               torch.randint(0, self.C, (self.B,))

    def test_scalar(self):
        l, t = self._batch()
        assert self.fn(l, t).shape == torch.Size([])

    def test_positive(self):
        l, t = self._batch()
        assert self.fn(l, t).item() > 0

    def test_gradients_exist(self):
        l, t = self._batch()
        self.fn(l, t).backward()
        assert l.grad is not None and not torch.isnan(l.grad).any()

    def test_residual_range(self):
        """ε ∈ [0, √2]."""
        l, t = self._batch()
        eps = AABLoss.residual(l, t)
        assert (eps >= 0).all() and (eps <= 2**0.5 + 1e-5).all()

    def test_sample_weights_scale(self):
        """w·φ·CE = ½·φ·CE when weights = 0.5."""
        l, t = self._batch()
        l0   = l.detach()
        loss1 = self.fn(l0, t, torch.ones(self.B)).item()
        loss2 = self.fn(l0, t, torch.full((self.B,), 0.5)).item()
        assert abs(loss2 - loss1 * 0.5) < 0.02 * loss1

    def test_stats_keys(self):
        l, t = self._batch()
        s = self.fn.get_stats(l, t)
        for k in ["eps_mean", "eps_std", "phi_mean", "phi_std",
                  "t_S_mean", "t_M_mean", "t_L_mean",
                  "pi_margin", "grad_variance_reduction"]:
            assert k in s

    def test_clean_higher_phi(self):
        """Clean samples (small ε) get higher φ than corrupted (large ε)."""
        vs = self.fn.vague_sets
        assert vs.confidence(torch.tensor([0.05])) > vs.confidence(torch.tensor([0.90]))


class TestKLDiagnostic:
    def setup_method(self):
        self.d = AdaptiveKLThreshold()

    def test_kl_symmetric(self):
        assert abs(self.d.gaussian_kl(0.0, 1.0, 0.0, 1.0)) < 1e-9

    def test_kl_positive(self):
        assert self.d.gaussian_kl(0.0, 0.5, 1.0, 0.8) >= 0

    def test_well_separated_standalone(self):
        """KL >> 1.5 → standalone_aab  (synthetic noise scenario)."""
        rng = np.random.RandomState(42)
        r   = np.concatenate([rng.normal(0.1, 0.05, 500), rng.normal(0.9, 0.10, 500)])
        assert self.d.fit_and_decide(r, use_adaptive=False) == "standalone_aab"

    def test_overlapping_two_stage(self):
        """KL << 1.5 → two_stage  (real-world noise scenario)."""
        rng = np.random.RandomState(7)
        r   = rng.normal(0.5, 0.30, 500)   # single overlapping cluster
        assert self.d.fit_and_decide(r, use_adaptive=False) == "two_stage"

    def test_insufficient_samples_warns(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.d.fit_and_decide(np.random.rand(10))
            assert len(w) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
