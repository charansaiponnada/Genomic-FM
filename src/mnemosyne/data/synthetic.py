"""
Multi-Query Associative Recall (MQAR) -- the falsifiable long-range experiment.

MQAR (Arora et al., "Zoology", 2023) is the canonical probe that separates
attention/memory models from fixed-state SSMs: a sequence contains key->value
bindings early on, then later queries a key and the model must emit its value.
A fixed-width SSM state must compress *all* bindings into O(d_state) memory, so
its accuracy collapses as the recall distance (sequence length) grows; a
content-addressable memory retrieves the right value at O(#slots) cost,
independent of distance.

We use a dedicated tiny vocabulary so the signal is clean and the result is a
crisp, publishable "accuracy vs. sequence length" curve.

Vocab:  0 = PAD/blank filler,  1..V = key/value symbols.
Targets are -100 (ignored) except at query positions, where the target is the
value bound to the queried key.
"""
from __future__ import annotations
import numpy as np
import torch

BLANK = 0
IGNORE = -100


def mqar_vocab_size(n_symbols: int) -> int:
    return n_symbols + 1  # +1 for BLANK/PAD id 0


def make_mqar_batch(batch_size, seq_len, n_pairs=4, n_queries=4,
                    n_symbols=32, seed=None):
    """Returns (tokens (b,L) long, targets (b,L) long with IGNORE elsewhere)."""
    rng = np.random.default_rng(seed)
    assert 2 * n_pairs + n_queries <= seq_len, "sequence too short for this config"
    toks = np.full((batch_size, seq_len), BLANK, dtype=np.int64)
    tgts = np.full((batch_size, seq_len), IGNORE, dtype=np.int64)
    symbols = np.arange(1, n_symbols + 1)

    for b in range(batch_size):
        keys = rng.choice(symbols, size=n_pairs, replace=False)
        values = rng.choice(symbols, size=n_pairs, replace=True)
        # definitions: adjacent (key, value) pairs at the start
        for i in range(n_pairs):
            toks[b, 2 * i] = keys[i]
            toks[b, 2 * i + 1] = values[i]
        # queries: placed among the later positions, distance grows with seq_len
        region = np.arange(2 * n_pairs, seq_len)
        qpos = rng.choice(region, size=n_queries, replace=False)
        for qp in qpos:
            j = rng.integers(0, n_pairs)
            toks[b, qp] = keys[j]
            tgts[b, qp] = values[j]

    return torch.from_numpy(toks), torch.from_numpy(tgts)


def random_dna_batch(batch_size, seq_len, seed=None):
    """Random ACGT windows in the model's nucleotide vocab (ids 2..5)."""
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.integers(2, 6, size=(batch_size, seq_len)).astype(np.int64))
