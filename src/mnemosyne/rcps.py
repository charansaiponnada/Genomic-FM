"""
Reverse-complement (RC) symmetry for DNA.

DNA is double-stranded: the reverse complement of a sequence encodes the same
molecule. A good genomic model should be RC-*equivariant* at the reconstruction
level and RC-*invariant* at the pooled-representation level. We enforce this
exactly by symmetrisation (no weight tricks required, and it is unit-testable):

    f_equiv(x)   = 1/2 [ f(x) + S( f(R x) ) ]          (logit output)
    g_inv(x)     = 1/2 [ pool(x) + pool(R x) ]          (pooled features)

where R reverses+complements the input token ids and S reverses the length axis
and permutes the vocab channels by the base complement. Both R and S are
involutions, which gives f_equiv(R x) = S f_equiv(x) and g_inv(R x) = g_inv(x)
identically (see tests/test_core.py::test_rc_equivariance).
"""
from __future__ import annotations
import torch

# vocab: 0=PAD 1=MASK 2=A 3=C 4=G 5=T 6=N
COMPLEMENT_PERM = torch.tensor([0, 1, 5, 4, 3, 2, 6], dtype=torch.long)


def rc_input(ids: torch.Tensor) -> torch.Tensor:
    """R : reverse-complement a batch of token ids (b, l)."""
    comp = COMPLEMENT_PERM.to(ids.device)[ids]
    return torch.flip(comp, dims=[-1])


def rc_logits(logits: torch.Tensor) -> torch.Tensor:
    """S : RC transform on per-position vocab logits (b, l, V)."""
    perm = COMPLEMENT_PERM.to(logits.device)
    flipped = torch.flip(logits, dims=[1])           # reverse positions
    return flipped.index_select(-1, perm)            # complement base channels


def symmetrize_logits(forward_logits, rc_of_rc_input_logits):
    """f_equiv = 1/2 [ f(x) + S(f(Rx)) ]."""
    return 0.5 * (forward_logits + rc_logits(rc_of_rc_input_logits))
