"""
Downstream evaluation on GUE with a *fair* protocol.

Key methodological fix over the v1 paper: we do NOT compare our frozen linear
probe against other papers' full fine-tuning as if it were a win. We report:
  (a) ssm-baseline vs mnemosyne under the *identical* frozen-probe protocol
      (this is the controlled claim about the memory module), and
  (b) published full-fine-tune baselines in a clearly separated column, labelled
      as such, for context only.

Both encoders are frozen; we mean-pool RC-invariant features and fit a linear
(logistic-regression) probe. Metrics: accuracy, MCC (the GUE convention), AUROC.

Requires `datasets` + `scikit-learn` on the GPU box. GUE tasks are loaded by
HF name; edit TASKS to match the hub dataset you use.
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from mnemosyne.config import MnemosyneConfig
from mnemosyne.model import MnemosyneMAE, SSMBaseline
from mnemosyne.data.tokenizer import encode_batch

# Map: friendly task name -> (hf_dataset, config, seq_column, label_column)
# These follow the GUE tasks packaged for DNABERT-2 / NT benchmarks; adjust to
# whichever mirror you load.
TASKS = {
    "prom_tata":    ("leannmlindsey/GUE", "prom_core_tata",    "sequence", "label"),
    "prom_notata":  ("leannmlindsey/GUE", "prom_core_notata",  "sequence", "label"),
    "splice":       ("leannmlindsey/GUE", "splice_reconstructed", "sequence", "label"),
    "h3k4me3":      ("leannmlindsey/GUE", "H3K4me3",           "sequence", "label"),
}

# Published FULL FINE-TUNE numbers for context only (do not compare to our probes).
PUBLISHED_FINETUNE = {  # MCC unless noted; fill from the cited papers
    "prom_tata":   {"DNABERT-2": None, "Caduceus": None, "HyenaDNA": None},
    "prom_notata": {"DNABERT-2": None, "Caduceus": None, "HyenaDNA": None},
    "splice":      {"DNABERT-2": None, "Caduceus": None, "HyenaDNA": None},
    "h3k4me3":     {"DNABERT-2": None, "Caduceus": None, "HyenaDNA": None},
}


def extract_features(model, seqs, window, device, batch=64):
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            ids = encode_batch(seqs[i:i + batch], max_len=window).to(device)
            feats.append(model.pooled_features(ids).cpu().numpy())
    return np.concatenate(feats)


def probe(model, task, window, device):
    from datasets import load_dataset
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score

    hf, cfg, scol, lcol = TASKS[task]
    dtr = load_dataset(hf, cfg, split="train")
    dte = load_dataset(hf, cfg, split="test")
    Xtr = extract_features(model, dtr[scol], window, device)
    Xte = extract_features(model, dte[scol], window, device)
    ytr, yte = np.array(dtr[lcol]), np.array(dte[lcol])

    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    out = dict(acc=accuracy_score(yte, pred), mcc=matthews_corrcoef(yte, pred))
    try:
        out["auroc"] = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    except Exception:
        out["auroc"] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="results/pretrain")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--out", default="results/gue.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = MnemosyneConfig.from_json(os.path.join(args.ckpt_dir, "config.json"))
    models = {}
    for name, ctor in [("ssm", SSMBaseline), ("mnemosyne", MnemosyneMAE)]:
        m = ctor(cfg)
        m.load_state_dict(torch.load(os.path.join(args.ckpt_dir, f"{name}.pt"),
                                     map_location=device))
        models[name] = m.to(device)

    results = {}
    for task in args.tasks:
        results[task] = {"protocol": "frozen linear probe (both models)"}
        for name, m in models.items():
            r = probe(m, task, args.window, device)
            results[task][name] = r
            print(f"{task:12s} {name:10s} acc={r['acc']:.4f} mcc={r['mcc']:.4f} "
                  f"auroc={r['auroc']}")
        results[task]["published_full_finetune_for_context"] = PUBLISHED_FINETUNE.get(task)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
