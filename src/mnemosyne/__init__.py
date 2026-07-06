"""
Mnemosyne: a reverse-complement-equivariant selective SSM with a persistent
associative motif memory (Hopfield-Mamba v2).

The token-mixing operator of a Mnemosyne layer is *semiseparable-plus-low-rank*:

    Y = (T_ssm + T_mem) V

where T_ssm is an L x L causal semiseparable matrix realised by a selective SSM
scan (rank <= d_state) and T_mem = softmax(beta Q K_M^T) is an L x L matrix of
rank <= P that factors through P *learned, persistent* memory slots. Both terms
cost O(L) in sequence length, so the layer is strictly linear-time yet can
retrieve any stored motif at a distance-independent O(P) cost.
"""

from .config import MnemosyneConfig
from .model import MnemosyneMAE, SSMBaseline
from .block import MnemosyneBlock
from .memory import AssociativeMemory
from .ssm import SelectiveSSM

__all__ = [
    "MnemosyneConfig",
    "MnemosyneMAE",
    "SSMBaseline",
    "MnemosyneBlock",
    "AssociativeMemory",
    "SelectiveSSM",
]
