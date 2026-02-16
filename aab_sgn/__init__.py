"""
AAB-SGN: Ambiguity-Aware Backpropagation with Symbolic Gradient Modulation
=========================================================================
Robust deep learning under label noise via vague set theory.

Paper: "AAB-SGN: Robust Learning Under Label Noise via
        Hesitation-Aware Gradient Modulation"
Authors: Ramsha Mehreen & Renikunta Ramesh
         Kakatiya Institute of Technology and Science, Warangal
Journal: Taylor & Francis — Journal of Experimental & Theoretical AI (JETAI)
"""

from .vague_sets    import VagueSets, DEFAULT_PARAMS
from .aab           import AABLoss
from .kl_diagnostic import AdaptiveKLThreshold
from .trainer       import AABSGNTrainer, SGNWeights

__version__ = "1.0.0"
__author__  = "Ramsha Mehreen, Renikunta Ramesh"
__email__   = "ramshamehreen2208@gmail.com"

__all__ = [
    "VagueSets", "DEFAULT_PARAMS",
    "AABLoss",
    "AdaptiveKLThreshold",
    "AABSGNTrainer", "SGNWeights",
]
