"""
Long-range associative-recall experiment (MQAR).

Trains a small model *with* and *without* the associative memory at several
sequence lengths and reports query accuracy. The central, falsifiable prediction
of Mnemosyne:

    memory model  -> accuracy stays high as length (recall distance) grows;
    pure SSM      -> accuracy decays, because a fixed-width state cannot hold all
                     bindings.

Runs on CPU for small lengths (proof of concept) and on GPU for the full sweep.
Writes results to results/longrange.json for the paper to consume.
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from mnemosyne.config import MnemosyneConfig
from mnemosyne.block import MnemosyneBlock, RMSNorm
from data.synthetic import make_mqar_batch, mqar_vocab_size, IGNORE


class RecallModel(nn.Module):
    """Encoder-only model with a token-classification head for MQAR."""

    def __init__(self, cfg: MnemosyneConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([MnemosyneBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, ids):
        h = self.embed(ids)
        for layer in self.layers:
            h = layer(h)
        return self.head(self.norm(h))


def run_one(use_memory, seq_len, device, steps=800, batch=64, n_pairs=4,
            n_queries=4, n_symbols=32, d_model=64, n_layers=2, d_state=16,
            mem_persistent=0, mem_registers=64, learn_beta=True, use_gate=True,
            lr=3e-3, seed=0, eval_batches=8):
    torch.manual_seed(seed)
    V = mqar_vocab_size(n_symbols)
    cfg = MnemosyneConfig(
        vocab_size=V, d_model=d_model, n_layers=n_layers, d_state=d_state,
        expand=2, use_memory=use_memory, mem_persistent=mem_persistent,
        mem_registers=mem_registers, learn_beta=learn_beta, mem_gate=use_gate,
        rc_equivariant=False, dropout=0.0, max_seq_len=seq_len,
    )
    model = RecallModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss(ignore_index=IGNORE)

    model.train()
    for step in range(steps):
        toks, tgts = make_mqar_batch(batch, seq_len, n_pairs, n_queries, n_symbols,
                                     seed=1000 + step)
        toks, tgts = toks.to(device), tgts.to(device)
        logits = model(toks)
        loss = lossf(logits.reshape(-1, V), tgts.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # evaluation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(eval_batches):
            toks, tgts = make_mqar_batch(batch, seq_len, n_pairs, n_queries,
                                         n_symbols, seed=90000 + i)
            toks, tgts = toks.to(device), tgts.to(device)
            pred = model(toks).argmax(-1)
            m = tgts != IGNORE
            correct += (pred[m] == tgts[m]).sum().item()
            total += m.sum().item()
    return correct / max(total, 1), model.__class__ and sum(p.numel() for p in model.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_symbols", type=int, default=32)
    ap.add_argument("--n_pairs", type=int, default=4)
    ap.add_argument("--out", default="results/longrange.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    for L in args.lengths:
        for use_mem in (False, True):
            acc, nparams = run_one(
                use_mem, L, device, steps=args.steps, batch=args.batch,
                d_model=args.d_model, n_symbols=args.n_symbols, n_pairs=args.n_pairs)
            tag = "mnemosyne" if use_mem else "ssm"
            rows.append(dict(model=tag, seq_len=L, query_acc=acc, params=nparams))
            print(f"L={L:5d}  {tag:10s}  acc={acc:.3f}  params={nparams:,}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(device=device, config=vars(args), rows=rows), f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
