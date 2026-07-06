"""
Ablation sweeps on the associative memory, run on the MQAR long-range task
(fast, clean signal). Answers the reviewer questions the v1 paper could not:

  * P_r  (register slots)     : does per-sequence recall capacity scale accuracy?
  * P_p  (persistent slots)   : do universal-motif slots help per-sequence recall?
                                (expected: little, since MQAR bindings are unique)
  * beta (learnable temp)     : fixed vs learned retrieval sharpness
  * gate : learned gate vs always-on memory

Writes results/ablations.json for the paper.
"""
from __future__ import annotations
import os, sys, json, argparse
import torch

sys.path.insert(0, os.path.dirname(__file__))
from eval_longrange import run_one


def sweep(device, seq_len, steps, batch):
    rows = []
    common = dict(steps=steps, batch=batch, n_symbols=32, n_pairs=8, n_queries=8)

    # 1) register slots P_r  (0 => persistent-only memory)
    for pr in [0, 8, 16, 32, 64, 128]:
        acc, _ = run_one(True, seq_len, device, mem_persistent=0, mem_registers=pr, **common)
        rows.append(dict(ablation="registers_P_r", value=pr, acc=acc))
        print(f"P_r={pr:4d}  acc={acc:.3f}")

    # 2) persistent slots P_p (registers fixed at 32)
    for pp in [0, 32, 128, 256]:
        acc, _ = run_one(True, seq_len, device, mem_persistent=pp, mem_registers=32, **common)
        rows.append(dict(ablation="persistent_P_p", value=pp, acc=acc))
        print(f"P_p={pp:4d}  acc={acc:.3f}")

    # 3) learnable vs fixed beta
    for lb in [True, False]:
        acc, _ = run_one(True, seq_len, device, mem_persistent=0, mem_registers=64,
                        learn_beta=lb, **common)
        rows.append(dict(ablation="learn_beta", value=lb, acc=acc))
        print(f"learn_beta={lb}  acc={acc:.3f}")

    # 4) gate vs always-on
    for g in [True, False]:
        acc, _ = run_one(True, seq_len, device, mem_persistent=0, mem_registers=64,
                        use_gate=g, **common)
        rows.append(dict(ablation="gate", value=g, acc=acc))
        print(f"use_gate={g}  acc={acc:.3f}")

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="results/ablations.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = sweep(device, args.seq_len, args.steps, args.batch)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
