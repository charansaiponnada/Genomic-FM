"""
Masked-autoencoder pre-training for Mnemosyne (and the pure-SSM baseline).

Trains both models under an identical protocol so the comparison is controlled:
same backbone, same data, same steps, memory on/off. Logs loss, masked-token
accuracy, gate activation, and Hopfield energy so we can *prove* the memory is
active (the key failure of v1 was an inert memory that never moved the loss).

Usage (GPU box):
    python src/train_mae.py --steps 20000 --window 1024 --d_model 256 \
        --out results/pretrain
CPU smoke:
    python src/train_mae.py --smoke
"""
from __future__ import annotations
import os, sys, json, time, argparse, math
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from mnemosyne.config import MnemosyneConfig
from mnemosyne.model import MnemosyneMAE, SSMBaseline
from data import genomes


def cosine_lr(step, total, base, warmup):
    if step < warmup:
        return base * step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return 0.5 * base * (1 + math.cos(math.pi * p))


def train_model(name, model, tokens, args, device):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05,
                            betas=(0.9, 0.95))
    n = tokens.shape[0]
    log = []
    model.train()
    t0 = time.time()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, args.steps, args.lr, int(0.05 * args.steps))
        idx = torch.randint(0, n, (args.batch,))
        ids = tokens[idx].to(device)
        loss, acc = model.mae_loss(ids)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            gate = model.gate_stats()
            energy = None
            if getattr(model.cfg, "use_memory", False):
                with torch.no_grad():
                    energy = model.encoder.layers[0].memory.energy(model.encode(ids)).item()
            rec = dict(step=step, loss=round(loss.item(), 4), acc=round(acc.item(), 4),
                       gate=None if gate is None else round(gate, 4),
                       energy=None if energy is None else round(energy, 3),
                       lr=round(opt.param_groups[0]["lr"], 6))
            log.append(rec)
            print(f"[{name}] " + "  ".join(f"{k}={v}" for k, v in rec.items()))
    dt = time.time() - t0
    return dict(name=name, params=model.num_params(), minutes=round(dt / 60, 2), log=log), model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--d_state", type=int, default=16)
    ap.add_argument("--mem_persistent", type=int, default=256)
    ap.add_argument("--mem_registers", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mask_ratio", type=float, default=0.3)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--max_windows_per", type=int, default=20000)
    ap.add_argument("--out", default="results/pretrain")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    if args.smoke:
        args.steps, args.window, args.d_model, args.n_layers = 30, 128, 32, 2
        tokens, _ = genomes.random_windows(n=256, window=args.window)
    else:
        tokens, _ = genomes.build_windows(
            genomes.DEFAULT_URLS, window=args.window, stride=args.window // 2,
            max_windows_per=args.max_windows_per)

    cfg = MnemosyneConfig(
        d_model=args.d_model, n_layers=args.n_layers, d_state=args.d_state,
        mem_persistent=args.mem_persistent, mem_registers=args.mem_registers,
        max_seq_len=args.window, mask_ratio=args.mask_ratio, rc_equivariant=True,
    )

    results = {}
    # controlled pair: identical backbone, memory off vs on
    base_stats, base_model = train_model("ssm", SSMBaseline(cfg), tokens, args, device)
    mnem_stats, mnem_model = train_model("mnemosyne", MnemosyneMAE(cfg), tokens, args, device)
    results["ssm"], results["mnemosyne"] = base_stats, mnem_stats

    torch.save(base_model.state_dict(), os.path.join(args.out, "ssm.pt"))
    torch.save(mnem_model.state_dict(), os.path.join(args.out, "mnemosyne.pt"))
    cfg.to_json(os.path.join(args.out, "config.json"))
    with open(os.path.join(args.out, "pretrain_log.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nfinal loss  ssm={base_stats['log'][-1]['loss']}  "
          f"mnemosyne={mnem_stats['log'][-1]['loss']}")
    print(f"wrote {args.out}/pretrain_log.json")


if __name__ == "__main__":
    main()
