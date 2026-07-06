"""
Associative Motif Memory (unified slot read).

Every token reads a set of `S = P_p + P_r` slots via a single Modern-Hopfield
retrieval (Ramsauer et al., 2021):

    q_i = W_q x_i ,   K = W_k Slots ,   V = W_v Slots
    a_i = softmax(beta * q_i K^T) in R^S ,   r_i = a_i V .

The slots come from two sources:

  * Persistent slots  M in R^{P_p x d}  -- learned parameters, shared across all
    sequences. A differentiable, content-addressable dictionary of *universal*
    genomic motifs (TATA boxes, splice consensus, ...). This is what
    Hopfield-Mamba v1 tried to build, but v1 wrote it per-token within a single
    sequence, so it could never accumulate universal structure.

  * Register slots  Reg in R^{P_r x d}  -- *gathered from the current sequence*
    by cross-attention from P_r learned latents (Perceiver-style bottleneck).
    Each register is a soft summary of the whole sequence, so a token can
    retrieve information bound anywhere in the sequence at O(P_r) cost,
    independent of distance. This is the engine for long-range associative
    recall that a fixed-width SSM state cannot provide.

Both the gather and the broadcast are single softmax attentions over a bottleneck
of size S << L, so the whole module is O(L * S * d): strictly linear in length.
The token-mixing matrix it induces has rank <= S (low-rank), complementing the
semiseparable SSM branch (Theorem in the paper).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AssociativeMemory(nn.Module):
    def __init__(self, d_model, n_persistent=256, n_registers=64, mem_dim=-1,
                 learn_beta=True, beta_init=-1.0, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_persistent = n_persistent
        self.n_registers = n_registers
        self.mem_dim = d_model if mem_dim < 0 else mem_dim
        beta0 = 1.0 / math.sqrt(self.mem_dim) if beta_init < 0 else beta_init

        # persistent learned slots
        if n_persistent > 0:
            self.M = nn.Parameter(torch.randn(n_persistent, d_model) / math.sqrt(d_model))
        else:
            self.register_parameter("M", None)

        # register gather: P_r learned latent queries pull a summary from the seq
        if n_registers > 0:
            self.reg_q = nn.Parameter(torch.randn(n_registers, self.mem_dim) / math.sqrt(self.mem_dim))
            self.W_kg = nn.Linear(d_model, self.mem_dim, bias=False)   # keys for gather
            self.W_vg = nn.Linear(d_model, d_model, bias=False)        # values for gather

        # token->slot broadcast read
        self.W_q = nn.Linear(d_model, self.mem_dim, bias=False)
        self.W_k = nn.Linear(d_model, self.mem_dim, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.log_beta = nn.Parameter(torch.tensor(math.log(beta0)), requires_grad=learn_beta)
        self.dropout = nn.Dropout(dropout)

    @property
    def beta(self):
        return torch.exp(self.log_beta)

    def _gather_registers(self, x):
        """x:(b,l,d) -> Reg:(b,P_r,d) : each register is a soft summary of the seq."""
        kg = self.W_kg(x)                                        # (b,l,dh)
        scores = self.beta * torch.einsum("rd,bld->brl", self.reg_q, kg)  # (b,P_r,l)
        attn = F.softmax(scores, dim=-1)
        return torch.einsum("brl,bld->brd", attn, self.W_vg(x))  # (b,P_r,d)

    def _slots(self, x):
        b = x.shape[0]
        slots = []
        if self.M is not None:
            slots.append(self.M.unsqueeze(0).expand(b, -1, -1))  # (b,P_p,d)
        if self.n_registers > 0:
            slots.append(self._gather_registers(x))             # (b,P_r,d)
        return torch.cat(slots, dim=1)                          # (b,S,d)

    def forward(self, x, return_attn=False):
        """x:(b,l,d) -> retrieved (b,l,d), [attn (b,l,S)]"""
        slots = self._slots(x)                                  # (b,S,d)
        q = self.W_q(x)                                         # (b,l,dh)
        k = self.W_k(slots)                                     # (b,S,dh)
        v = self.W_v(slots)                                     # (b,S,d)
        scores = self.beta * torch.einsum("bld,bsd->bls", q, k)
        attn = F.softmax(scores, dim=-1)                        # (b,l,S)
        r = torch.einsum("bls,bsd->bld", attn, v)
        r = self.dropout(r)
        return (r, attn) if return_attn else r

    @torch.no_grad()
    def energy(self, x):
        """Mean Hopfield energy over tokens: E(q) = -lse(beta,Kq)/beta + 0.5||q||^2.
        Deep (low) energy => queries land in well-separated pattern basins."""
        slots = self._slots(x)
        q = self.W_q(x)
        k = self.W_k(slots)
        beta = self.beta
        lse = torch.logsumexp(beta * torch.einsum("bld,bsd->bls", q, k), dim=-1) / beta
        return (-lse + 0.5 * q.pow(2).sum(-1)).mean()

    @torch.no_grad()
    def slot_usage(self, x):
        """Mean probability mass per slot (S,). Reveals dead / collapsed slots."""
        _, attn = self.forward(x, return_attn=True)
        return attn.mean(dim=(0, 1))
