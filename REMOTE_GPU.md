# Running on a free remote GPU from VS Code (no Colab)

You develop and launch everything from your laptop's VS Code terminal; the
training runs on a cloud GPU. Three good free options, ranked for this repo.

---

## Option 1 — Modal  (recommended: matches `modal_app.py`)

Serverless GPUs. New accounts get free monthly compute credit (enough for the
long-range sweep and a small MAE run). You never leave VS Code.

```bash
pip install modal
modal setup                    # one browser click to link your account
# from the repo root:
modal run modal_app.py::smoke          # ~1 min: verifies CUDA + fused kernel
modal run modal_app.py::longrange      # the headline experiment
modal run modal_app.py::pretrain       # MAE: ssm vs mnemosyne (controlled)
modal run modal_app.py::gue            # downstream probes
modal run modal_app.py::pull           # copy results/ back to your laptop
```

- Code lives on your laptop; `add_local_dir` ships `src/` to the GPU each run.
- Results persist in a Modal **Volume** and come back with `::pull`.
- Change the GPU in `modal_app.py` (`GPU = "A10G"` → `"A100"` for big models).
- Cost control: it's serverless, so you're billed only while a function runs.

## Option 2 — Lightning AI Studios  (best for interactive dev)

Free monthly GPU credits (~22 GPU-hours, T4-class). Two ways to use it from
your laptop VS Code:

1. Open a Studio, then **"Connect with local VS Code"** — it wires your desktop
   VS Code to the cloud machine over SSH (Remote-SSH under the hood). You edit
   and run exactly as if the GPU were local.
2. Or use the in-browser VS Code the Studio ships with.

```bash
git clone https://github.com/charansaiponnada/Genomic-FM && cd Genomic-FM
pip install -r requirements-gpu.txt
python tests/test_core.py
python src/eval_longrange.py --lengths 64 128 256 512 1024 --steps 4000
```

## Option 3 — Kaggle Notebooks via the API  (the free workhorse for long runs)

30 GPU-hours/week free (T4×2 or P100) — the most raw free compute here. Not
VS Code-native, but you can drive it headless from your laptop:

```bash
pip install kaggle                       # put kaggle.json in ~/.kaggle/
# create kernel-metadata.json (script kernel, GPU on, internet on), then:
kaggle kernels push -p ./kaggle_job      # queues the run
kaggle kernels status <user>/<kernel>
kaggle kernels output <user>/<kernel> -p ./results
```

Develop and unit-test locally in VS Code (CPU), push the heavy job to Kaggle,
pull results back. Good when you need many hours and don't mind non-interactive.

---

## Also fine, not free-forever
- **Google Cloud $300 / 90-day credit + VS Code Remote-SSH**: the closest thing
  to "a real GPU box in my VS Code." Create an L4/T4 VM, install the
  *Remote - SSH* extension, connect, and work as if local. Watch the meter.
- **SageMaker Studio Lab** (free, JupyterLab not VS Code), **Paperspace Gradient
  free tier** (older GPUs). Both work but are notebook-first.
- **Vast.ai / RunPod**: not free, but cents/hour and SSH-able from VS Code if you
  outgrow the free tiers.

## What to run, in order
1. `tests/test_core.py` — sanity (also runs on your laptop, no GPU).
2. `src/eval_longrange.py` — the falsifiable long-range claim (cheapest signal).
3. `src/ablate.py` — memory ablations.
4. `src/train_mae.py` then `src/eval_gue.py` — pre-train + downstream (needs data
   download; heaviest).

> The model auto-uses the fused `mamba-ssm` CUDA kernel when present and falls
> back to the pure-PyTorch scan otherwise, so nothing breaks if a wheel won't
> build — it's just slower.
