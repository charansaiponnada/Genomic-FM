"""Configuration for Mnemosyne models."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json
import math


@dataclass
class MnemosyneConfig:
    # ---- vocabulary / tokenisation ----
    # 0=PAD 1=MASK 2=A 3=C 4=G 5=T 6=N  (see data.tokenizer)
    vocab_size: int = 7
    pad_id: int = 0
    mask_id: int = 1

    # ---- backbone ----
    d_model: int = 256
    n_layers: int = 6
    d_state: int = 16          # SSM latent state size N
    d_conv: int = 4            # short causal conv width inside the SSM
    expand: int = 2            # inner = expand * d_model
    dt_rank: int = -1          # -1 => ceil(d_model / 16)

    # ---- associative memory (unified Hopfield read over slots) ----
    # The memory is a set of slots read by every token via one softmax retrieval.
    # Some slots are learned & persistent (universal motif dictionary); some are
    # gathered from the current sequence (Perceiver-style registers) to give
    # per-sequence, distance-independent long-range recall.
    use_memory: bool = True
    mem_persistent: int = 256  # P_p : learned persistent slots (universal motifs)
    mem_registers: int = 64    # P_r : per-sequence gathered slots (long-range recall)
    mem_dim: int = -1          # head dim for read/write; -1 => d_model
    learn_beta: bool = True    # learnable Hopfield inverse temperature
    beta_init: float = -1.0    # -1 => 1/sqrt(mem_dim)
    mem_gate: bool = True      # False => memory added ungated (ablation)
    gate_bias_init: float = -2.0  # sigmoid(-2) ~ 0.12 : memory starts mostly closed

    # ---- reverse-complement equivariance ----
    rc_equivariant: bool = True

    # ---- decoder (MAE) ----
    decoder_layers: int = 2

    # ---- training / masking ----
    max_seq_len: int = 1024
    mask_ratio: float = 0.30
    dropout: float = 0.1

    def resolved_dt_rank(self) -> int:
        return math.ceil(self.d_model / 16) if self.dt_rank < 0 else self.dt_rank

    def resolved_mem_dim(self) -> int:
        return self.d_model if self.mem_dim < 0 else self.mem_dim

    @property
    def mem_slots_total(self) -> int:
        return self.mem_persistent + self.mem_registers

    def resolved_beta_init(self) -> float:
        return 1.0 / math.sqrt(self.resolved_mem_dim()) if self.beta_init < 0 else self.beta_init

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "MnemosyneConfig":
        with open(path) as f:
            return cls(**json.load(f))
