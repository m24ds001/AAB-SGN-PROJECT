"""
tests/test_autograd_equivalence.py
====================================
Numerical verification: PyTorch autograd on Algorithm S1 ≡ Equation (2).

From Paper Appendix A.2
-----------------------
When PyTorch differentiates L_AAB = φ_i · L_CE, it produces:
    ∇θL_AAB = φ_i · ∇θL_CE + L_CE · ∇θφ_i

Since φ_i = φ(ε_i) and ε_i = ‖1_{y_i} − softmax(z_i)‖₂, the chain rule gives:
    ∇θφ_i = φ'(ε_i) · (∂ε_i/∂z_i) · (∂z_i/∂θ)

This matches Equation (2) exactly.  Three conditions required (all satisfied):
  (1) φ is C∞ and differentiable w.r.t. ε
  (2) Autograd tracks ε → θ (no detach)
  (3) No stop-gradient applied to φ

Usage
-----
    python tests/test_autograd_equivalence.py

Expected output
---------------
    ALL TESTS PASSED (5/5)
"""

import sys, os, math
import torch
import torch.nn.functional as F
from torch.autograd import gradcheck

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aab_sgn.vague_sets import VagueSets
from aab_sgn.aab import AABLoss


class TinyLinear(torch.nn.Module):
    def __init__(self, d, c):
        super().__init__()
        self.fc = torch.nn.Linear(d, c, bias=False)
    def forward(self, x): return self.fc(x)


def autograd_grad(model, inputs, targets, loss_fn):
    """Algorithm S1, Steps 6-7: L_AAB = φ·L_CE, then autograd."""
    model.zero_grad()
    logits = model(inputs)
    loss   = loss_fn(logits, targets)
    loss.backward()
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def analytical_grad(model, inputs, targets, vs):
    """Equation (2): ∇θL_AAB = φ(ε)·∇θL_CE + L_CE·φ'(ε)·∇θε"""
    # Term 1: φ(ε)·∇θL_CE
    model.zero_grad()
    logits = model(inputs)
    probs  = F.softmax(logits, dim=1)
    oh     = torch.zeros_like(probs); oh.scatter_(1, targets.unsqueeze(1), 1.0)
    eps    = (oh - probs).norm(dim=1)
    phi    = vs.confidence(eps.detach())              # detach for term-1 only
    ce     = F.cross_entropy(logits, targets, reduction="none")
    (phi.detach() * ce).mean().backward()
    g1 = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    # Term 2: L_CE·φ'(ε)·∇θε
    model.zero_grad()
    logits = model(inputs)
    probs  = F.softmax(logits, dim=1)
    oh     = torch.zeros_like(probs); oh.scatter_(1, targets.unsqueeze(1), 1.0)
    eps    = (oh - probs).norm(dim=1)
    # φ'(ε) numerically
    e2 = eps.detach().requires_grad_(True)
    vs.confidence(e2).sum().backward()
    dphi = e2.grad.clone()
    ce   = F.cross_entropy(logits.detach(), targets, reduction="none")
    (ce * dphi * eps).mean().backward()
    g2 = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    return {n: g1[n] + g2.get(n, torch.zeros_like(g1[n])) for n in g1}


def rel_err(a, b):
    return (a - b).norm() / (b.norm() + 1e-12)


def test_grad_equivalence(tol=1e-4, n_trials=10):
    print("Test 1: Autograd ≡ Equation (2)  [Appendix A.2]")
    vs = VagueSets(); aab = AABLoss(pi_min=0.10)
    ok = True
    for t in range(n_trials):
        torch.manual_seed(t * 13)
        m = TinyLinear(16, 4)
        x = torch.randn(8, 16); y = torch.randint(0, 4, (8,))
        ag = autograd_grad(m, x.clone(), y, aab)
        an = analytical_grad(m, x.clone(), y, vs)
        errs = [rel_err(ag[k], an[k]).item() for k in ag]
        max_e = max(errs)
        if max_e > tol:
            print(f"  FAIL trial={t} max_rel_err={max_e:.2e}"); ok = False
    print(f"  {'PASS' if ok else 'FAIL'} (tol={tol})")
    return ok


def test_pi_min_enforcement():
    print("Test 2: π_min > 0 enforcement  [Theorem 3]")
    try:
        VagueSets(pi_min=0.0)
        print("  FAIL: should have raised ValueError"); return False
    except ValueError:
        print("  PASS: ValueError raised for pi_min=0"); return True


def test_phi_bounds():
    print("Test 3: φ(ε) ∈ [0.5, 1.0]  [Lemma 1]")
    vs  = VagueSets()
    eps = torch.linspace(0, 2.5, 2000)
    phi = vs.confidence(eps)
    lb  = vs.w_L / vs.W  # 0.5
    ok  = phi.min().item() >= lb - 1e-5 and phi.max().item() <= 1.0 + 1e-5
    print(f"  φ ∈ [{phi.min():.4f}, {phi.max():.4f}]  expected [{lb:.4f}, 1.0000]")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_pi_star():
    print("Test 4: π*(0.4,0.05,500) ≈ 0.095  [Theorem 3 Eq.11]")
    vs   = VagueSets()
    pi_s = vs.pi_star(0.4, 0.05, 500)
    ok   = abs(pi_s - 0.095) < 0.003 and 0.10 >= pi_s
    print(f"  π*(0.4,0.05,500)={pi_s:.5f}  default π_min=0.10 "
          f"{'satisfies' if 0.10 >= pi_s else 'VIOLATES'} threshold")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_gradcheck():
    print("Test 5: torch.autograd.gradcheck  [Appendix A.2]")
    aab = AABLoss(pi_min=0.10)
    m   = TinyLinear(8, 3).double()
    x   = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    y   = torch.randint(0, 3, (4,))
    try:
        ok = gradcheck(lambda z: aab(m(z), y).unsqueeze(0), (x,), eps=1e-6, atol=1e-4)
        print(f"  PASS: gradcheck={ok}")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


if __name__ == "__main__":
    results = [
        test_grad_equivalence(),
        test_pi_min_enforcement(),
        test_phi_bounds(),
        test_pi_star(),
        test_gradcheck(),
    ]
    n = sum(results)
    print(f"\n{'='*40}")
    print(f"{'ALL TESTS PASSED' if n == len(results) else 'SOME TESTS FAILED'} ({n}/{len(results)})")
    sys.exit(0 if n == len(results) else 1)
