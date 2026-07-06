"""
Selective state-space model (Mamba S6) block.

This implements the input-dependent (selective) SSM recurrence

    h_t = exp(dt_t * A) . h_{t-1} + (dt_t * B_t) . x_t ,   y_t = C_t . h_t + D . x_t

with a short causal depthwise conv and SiLU gating, following Gu & Dao (2023).

Two execution paths:
  * GPU  : if `mamba_ssm` (CUDA kernel) is importable *and* the tensor is on
           cuda, we use the fused `selective_scan_fn` for speed.
  * CPU  : a correct, fully-differentiable reference scan in pure PyTorch.
           O(L) sequential -- slow for long L, but exact, so the model is
           unit-testable on a laptop without a GPU.

The reference scan follows the minimal formulation of Gu & Dao / mamba-minimal.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # optional fast path
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _fused_scan
    _HAS_FUSED = True
except Exception:  # pragma: no cover - only available with CUDA build
    _HAS_FUSED = False


def reference_selective_scan(u, delta, A, B, C, D):
    """Exact O(L) reference scan.

    Shapes:
        u     : (b, l, d_in)      input (post conv + silu)
        delta : (b, l, d_in)      positive step sizes
        A     : (d_in, n)         (negative) state matrix, diagonal
        B     : (b, l, n)
        C     : (b, l, n)
        D     : (d_in,)
    Returns y : (b, l, d_in)
    """
    b, l, d_in = u.shape
    n = A.shape[1]
    # discretise
    deltaA = torch.exp(torch.einsum("bld,dn->bldn", delta, A))          # (b,l,d,n)
    deltaB_u = torch.einsum("bld,bln,bld->bldn", delta, B, u)           # (b,l,d,n)

    x = u.new_zeros((b, d_in, n))
    ys = []
    for t in range(l):
        x = deltaA[:, t] * x + deltaB_u[:, t]                          # (b,d,n)
        y = torch.einsum("bdn,bn->bd", x, C[:, t])                     # (b,d)
        ys.append(y)
    y = torch.stack(ys, dim=1)                                          # (b,l,d)
    return y + u * D


class SelectiveSSM(nn.Module):
    """A single Mamba-style selective SSM block (pre-norm handled by caller)."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank=-1, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = (d_model + 15) // 16 if dt_rank < 0 else dt_rank

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A is parameterised as -exp(A_log) so it stays negative (stable).
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (b, l, d_model) -> (b, l, d_model)"""
        b, l, _ = x.shape
        xz = self.in_proj(x)                                   # (b,l,2*d_inner)
        xin, z = xz.chunk(2, dim=-1)                           # each (b,l,d_inner)

        # short causal depthwise conv over time
        xin = xin.transpose(1, 2)                              # (b,d_inner,l)
        xin = self.conv1d(xin)[..., :l]                        # causal crop
        xin = xin.transpose(1, 2)
        xin = F.silu(xin)

        A = -torch.exp(self.A_log)                             # (d_inner, n)
        dbl = self.x_proj(xin)                                 # (b,l,dt_rank+2n)
        dt, B, C = torch.split(dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                      # (b,l,d_inner), > 0

        if _HAS_FUSED and xin.is_cuda:
            # fused kernel expects (b, d_inner, l)
            y = _fused_scan(
                xin.transpose(1, 2).contiguous(),
                dt.transpose(1, 2).contiguous(),
                A, B.transpose(1, 2).contiguous(), C.transpose(1, 2).contiguous(),
                self.D.float(), z=None, delta_softplus=False,
            ).transpose(1, 2)
        else:
            y = reference_selective_scan(xin, dt, A, B, C, self.D)

        y = y * F.silu(z)                                      # gated output
        return self.dropout(self.out_proj(y))
