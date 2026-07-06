"""Mnemosyne block: selective SSM in parallel with a persistent-memory read,
combined by a learned per-channel gate. Token-mixing = semiseparable + low-rank."""
from __future__ import annotations
import torch
import torch.nn as nn

from .ssm import SelectiveSSM
from .memory import AssociativeMemory


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class MnemosyneBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.norm = RMSNorm(cfg.d_model)
        self.ssm = SelectiveSSM(
            cfg.d_model, d_state=cfg.d_state, d_conv=cfg.d_conv,
            expand=cfg.expand, dt_rank=cfg.dt_rank, dropout=cfg.dropout,
        )
        self.use_memory = cfg.use_memory
        if self.use_memory:
            self.memory = AssociativeMemory(
                cfg.d_model, n_persistent=cfg.mem_persistent, n_registers=cfg.mem_registers,
                mem_dim=cfg.mem_dim, learn_beta=cfg.learn_beta,
                beta_init=cfg.beta_init, dropout=cfg.dropout,
            )
            # per-channel gate; bias init < 0 keeps memory mostly closed at start
            self.mem_gate = cfg.mem_gate
            if self.mem_gate:
                self.gate = nn.Linear(cfg.d_model, cfg.d_model)
                nn.init.zeros_(self.gate.weight)
                nn.init.constant_(self.gate.bias, cfg.gate_bias_init)
        self._last_gate = None  # diagnostic: mean gate activation

    def forward(self, x):
        h = self.norm(x)
        y = self.ssm(h)                                   # semiseparable branch
        if self.use_memory:
            r = self.memory(h)                            # low-rank global branch
            if self.mem_gate:
                g = torch.sigmoid(self.gate(h))           # (b,l,d) in (0,1)
                self._last_gate = g.detach().mean().item()
                y = y + g * r
            else:
                self._last_gate = 1.0
                y = y + r
        return x + y                                      # residual
