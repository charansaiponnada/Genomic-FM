"""Single-nucleotide tokenizer. vocab: 0=PAD 1=MASK 2=A 3=C 4=G 5=T 6=N."""
from __future__ import annotations
import numpy as np
import torch

PAD, MASK = 0, 1
BASE2ID = {"A": 2, "C": 3, "G": 4, "T": 5, "N": 6, "a": 2, "c": 3, "g": 4, "t": 5, "n": 6}
ID2BASE = {2: "A", 3: "C", 4: "G", 5: "T", 6: "N"}
VOCAB_SIZE = 7


def encode(seq: str) -> torch.Tensor:
    return torch.tensor([BASE2ID.get(c, 6) for c in seq], dtype=torch.long)


def encode_batch(seqs, max_len=None, pad=PAD) -> torch.Tensor:
    ids = [[BASE2ID.get(c, 6) for c in s] for s in seqs]
    L = max_len or max(len(x) for x in ids)
    out = np.full((len(ids), L), pad, dtype=np.int64)
    for i, row in enumerate(ids):
        row = row[:L]
        out[i, : len(row)] = row
    return torch.from_numpy(out)


def decode(ids) -> str:
    return "".join(ID2BASE.get(int(i), "N") for i in ids if int(i) not in (PAD, MASK))
