"""Full Mnemosyne masked-autoencoder model, plus a pure-SSM baseline that shares
the same backbone with the memory switched off (for controlled comparisons)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MnemosyneConfig
from .block import MnemosyneBlock, RMSNorm
from .rcps import rc_input, symmetrize_logits


class _Backbone(nn.Module):
    def __init__(self, cfg: MnemosyneConfig, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([MnemosyneBlock(cfg) for _ in range(n_layers)])
        self.norm = RMSNorm(cfg.d_model)

    def forward(self, h):
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)


class MnemosyneMAE(nn.Module):
    """Masked-autoencoder over nucleotide tokens."""

    def __init__(self, cfg: MnemosyneConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.drop = nn.Dropout(cfg.dropout)
        self.encoder = _Backbone(cfg, cfg.n_layers)
        self.decoder = _Backbone(cfg, cfg.decoder_layers)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    # ---- core forward (no RC symmetrisation) ----
    def _logits(self, ids):
        h = self.drop(self.embed(ids))
        h = self.encoder(h)
        h = self.decoder(h)
        return self.head(h)                       # (b, l, V)

    def logits(self, ids):
        """RC-equivariant reconstruction logits."""
        fwd = self._logits(ids)
        if not self.cfg.rc_equivariant:
            return fwd
        rc = self._logits(rc_input(ids))
        return symmetrize_logits(fwd, rc)

    # ---- pooled features for linear probing (RC-invariant) ----
    def encode(self, ids):
        h = self.drop(self.embed(ids))
        return self.encoder(h)                    # (b, l, d)

    @torch.no_grad()
    def pooled_features(self, ids, attn_mask=None):
        def _pool(x):
            h = self.encode(x)
            if attn_mask is None:
                return h.mean(1)
            m = attn_mask.unsqueeze(-1).float()
            return (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        feat = _pool(ids)
        if self.cfg.rc_equivariant:
            feat = 0.5 * (feat + _pool(rc_input(ids)))
        return feat                               # (b, d)

    # ---- MAE loss ----
    def mae_loss(self, ids, mask_ratio=None):
        """Corrupt `mask_ratio` of positions with MASK, predict the originals.
        Returns (loss, accuracy_on_masked)."""
        cfg = self.cfg
        mr = cfg.mask_ratio if mask_ratio is None else mask_ratio
        valid = ids != cfg.pad_id
        rand = torch.rand_like(ids, dtype=torch.float)
        mask = (rand < mr) & valid                # positions to predict
        corrupted = torch.where(mask, torch.full_like(ids, cfg.mask_id), ids)

        logits = self.logits(corrupted)
        if mask.sum() == 0:
            return logits.sum() * 0.0, torch.tensor(0.0)
        sel = logits[mask]
        tgt = ids[mask]
        loss = F.cross_entropy(sel, tgt)
        acc = (sel.argmax(-1) == tgt).float().mean()
        return loss, acc

    # ---- diagnostics ----
    def gate_stats(self):
        gs = [l._last_gate for l in self.encoder.layers if l._last_gate is not None]
        return sum(gs) / len(gs) if gs else None

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def SSMBaseline(cfg: MnemosyneConfig) -> MnemosyneMAE:
    """Same backbone, memory disabled -- the controlled ablation baseline."""
    import copy
    c = copy.deepcopy(cfg)
    c.use_memory = False
    return MnemosyneMAE(c)
