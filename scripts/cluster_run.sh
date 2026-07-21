#!/usr/bin/env bash
# ==========================================================================
# Mnemosyne — run on the college GPU cluster (L40S) from a JupyterLab terminal.
#
# Usage (paste into the JupyterLab Terminal):
#     bash <(curl -sL https://raw.githubusercontent.com/charansaiponnada/Genomic-FM/master/scripts/cluster_run.sh)
# or, if you already cloned:
#     bash scripts/cluster_run.sh
#
# Stage 1 (this script) runs the GATING experiment from the documentation's
# decision flow (Fig. 3): the MQAR long-range recall sweep. Only if the recall
# curve is flat for Mnemosyne and decays for the SSM baseline do we spend the
# cluster budget on full genome-scale pre-training.
# ==========================================================================
set -u

REPO_URL="https://github.com/charansaiponnada/Genomic-FM.git"
REPO_DIR="${REPO_DIR:-$HOME/Genomic-FM}"
STEPS="${STEPS:-3000}"
LENGTHS="${LENGTHS:-64 128 256 512 1024}"
BATCH="${BATCH:-32}"        # keep modest: the fallback scan holds an autograd graph per timestep

echo "=============================================="
echo " Mnemosyne cluster run — stage 1 (MQAR gate)"
echo "=============================================="

# ---------- 1. get the code ----------
if [ -d "$REPO_DIR/.git" ]; then
  echo "[1/5] repo exists -> pulling latest"
  git -C "$REPO_DIR" pull --ff-only || echo "  (pull failed; continuing with local copy)"
else
  echo "[1/5] cloning $REPO_URL -> $REPO_DIR"
  git clone --depth 1 "$REPO_URL" "$REPO_DIR" || { echo "FATAL: clone failed (no internet?)"; exit 1; }
fi
cd "$REPO_DIR" || exit 1

# ---------- 2. environment report ----------
echo "[2/5] environment"
python -c "import sys; print('  python', sys.version.split()[0])"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
  || echo "  nvidia-smi not available"
python - <<'PY'
try:
    import torch
    print(f"  torch {torch.__version__} | cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print("  device:", torch.cuda.get_device_name(0))
except Exception as e:
    print("  torch NOT importable:", e)
PY

# ---------- 3. dependencies ----------
echo "[3/5] dependencies"
python -c "import torch" 2>/dev/null || pip install --quiet torch
python -c "import numpy" 2>/dev/null || pip install --quiet numpy
# Fused Mamba kernel: big speedup, needs nvcc. Best-effort — the model falls back
# to the pure-PyTorch scan automatically if this fails.
if command -v nvcc >/dev/null 2>&1; then
  echo "  nvcc found ($(nvcc --version | tail -1)); attempting fused kernel install"
  pip install --quiet ninja packaging 2>/dev/null
  pip install --quiet causal-conv1d>=1.4.0 2>/dev/null || echo "  causal-conv1d failed (ok)"
  pip install --quiet mamba-ssm>=2.2.2   2>/dev/null || echo "  mamba-ssm failed (ok, using fallback scan)"
else
  echo "  no nvcc -> using pure-PyTorch scan (slower but correct)"
fi
python -c "
try:
    import mamba_ssm; print('  fused mamba-ssm kernel: AVAILABLE (fast path)')
except Exception:
    print('  fused mamba-ssm kernel: not available (reference scan)')
"

# ---------- 4. correctness gate ----------
echo "[4/5] correctness tests"
python tests/test_core.py || { echo "FATAL: tests failed — stop, do not train."; exit 1; }

# ---------- 5. THE experiment ----------
echo "[5/5] MQAR long-range sweep  (lengths: $LENGTHS | steps: $STEPS | batch: $BATCH)"
echo "      this is the falsifiable test: memory vs matched SSM as distance grows"
mkdir -p results
python -u src/eval_longrange.py \
    --lengths $LENGTHS \
    --steps "$STEPS" \
    --batch "$BATCH" \
    --out results/longrange.json 2>&1 | tee results/longrange_stdout.log

echo
echo "=============================================="
echo " DONE. Results: $REPO_DIR/results/longrange.json"
echo
echo " Next: share results/longrange.json (or push it):"
echo "   cd $REPO_DIR && git add -f results/longrange.json && \\"
echo "     git commit -m 'cluster: MQAR long-range results' && git push"
echo "=============================================="
