"""
Run Mnemosyne training/eval on a cloud GPU straight from your laptop's VS Code
terminal -- no Colab, no notebook. Modal gives new accounts free monthly credit.

One-time:
    pip install modal
    modal setup                      # opens browser once to link your account

Then, from the repo root in VS Code:
    modal run modal_app.py::smoke                 # ~1 min, checks the GPU path
    modal run modal_app.py::pretrain              # MAE pre-train (ssm vs mnemosyne)
    modal run modal_app.py::longrange             # MQAR length sweep
    modal run modal_app.py::ablations
    modal run modal_app.py::pull                  # copy results/ back to your laptop

Everything runs on Modal's GPU; your laptop just orchestrates. Results are written
to a persistent Modal Volume and can be pulled down with `pull`.
"""
import modal

app = modal.App("mnemosyne-genomic-fm")

# Build the GPU image. mamba-ssm needs nvcc; if the wheel fails to build the
# model still runs via the pure-PyTorch scan, so we install it best-effort.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.4.0", "numpy", "datasets", "scikit-learn", "matplotlib", "tqdm",
    )
    .run_commands(
        "pip install ninja packaging",
        "pip install causal-conv1d>=1.4.0 || true",
        "pip install mamba-ssm>=2.2.2 || true",
    )
    .add_local_dir("src", "/root/src")
    .add_local_dir("tests", "/root/tests")
)

vol = modal.Volume.from_name("mnemosyne-results", create_if_missing=True)
GPU = "A10G"  # cheap + plenty for <=350M models; use "A100" for the large sweep


def _run(cmd: str):
    import subprocess, os
    os.chdir("/root")
    print("RUN:", cmd)
    subprocess.run(cmd, shell=True, check=True)
    vol.commit()


@app.function(image=image, gpu=GPU, volumes={"/root/results": vol}, timeout=600)
def smoke():
    import torch
    print("cuda available:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
    _run("python tests/test_core.py")
    _run("python src/train_mae.py --smoke --out results/smoke")


@app.function(image=image, gpu=GPU, volumes={"/root/results": vol}, timeout=60 * 60 * 6)
def pretrain(steps: int = 20000, window: int = 1024, d_model: int = 256, n_layers: int = 6):
    _run(f"python src/train_mae.py --steps {steps} --window {window} "
         f"--d_model {d_model} --n_layers {n_layers} --out results/pretrain")


@app.function(image=image, gpu=GPU, volumes={"/root/results": vol}, timeout=60 * 60 * 4)
def longrange(lengths: str = "64 128 256 512 1024", steps: int = 4000):
    _run(f"python src/eval_longrange.py --lengths {lengths} --steps {steps} "
         f"--out results/longrange.json")


@app.function(image=image, gpu=GPU, volumes={"/root/results": vol}, timeout=60 * 60 * 4)
def ablations(seq_len: int = 256, steps: int = 3000):
    _run(f"python src/ablate.py --seq_len {seq_len} --steps {steps} "
         f"--out results/ablations.json")


@app.function(image=image, gpu=GPU, volumes={"/root/results": vol}, timeout=60 * 60 * 3)
def gue(window: int = 512):
    _run(f"python src/eval_gue.py --ckpt_dir results/pretrain --window {window} "
         f"--out results/gue.json")


@app.function(volumes={"/root/results": vol})
def _list():
    import os
    for root, _, files in os.walk("/root/results"):
        for f in files:
            print(os.path.join(root, f))


@app.local_entrypoint()
def pull():
    """Copy the results volume down to ./results on your laptop."""
    import subprocess
    subprocess.run("modal volume get mnemosyne-results / ./results --force",
                   shell=True, check=False)
    print("pulled results/ -- wire them into the paper with src/../make_figures.py")
