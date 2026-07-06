# Mnemosyne — a genomic foundation model with a unified associative memory

**Author:** Charan Sai Ponnada

A linear-time genomic sequence model that augments a selective state-space (Mamba)
backbone with a **unified associative memory**, read by every position through a
single Modern-Hopfield step. The memory holds two kinds of slots:

- **persistent slots** — learned parameters forming a differentiable dictionary of
  *universal* motifs, and
- **register slots** — soft summaries gathered from the current sequence that give
  **per-sequence, distance-independent long-range recall**.

The effective token-mixing operator is *semiseparable-plus-low-rank*, so the layer
stays strictly `O(L)`; the model is exactly **reverse-complement equivariant**
(proved, and unit-tested to floating-point zero).

> This is the **v2 rewrite**. The original HopField-Mamba (v1) work is preserved in
> [`archive/`](archive/README.md). Why v2 supersedes it: [`docs/ASSESSMENT.md`](docs/ASSESSMENT.md).

![architecture](research_paper/figures/mnemosyne_architecture.png)

## Quickstart (laptop / CPU — no GPU needed)

```bash
pip install -r requirements.txt
python tests/test_core.py        # verify the architecture (shapes, RC-equivariance, grads)
```

## Run the experiments on a free GPU (from VS Code, no Colab)

See [`REMOTE_GPU.md`](REMOTE_GPU.md). Recommended path (Modal, free monthly credit):

```bash
pip install modal && modal setup
modal run modal_app.py::longrange     # headline: MQAR recall vs distance (memory vs SSM)
modal run modal_app.py::ablations     # sweep P_r, P_p, beta, gate
modal run modal_app.py::pretrain      # controlled MAE (ssm vs mnemosyne) + memory diagnostics
modal run modal_app.py::gue           # frozen-probe downstream (fair protocol)
modal run modal_app.py::pull          # copy results/ back to your laptop
python src/make_report.py             # fold real numbers into the paper
```

## Repository layout

| Path | Contents |
|------|----------|
| [`src/mnemosyne/`](src/mnemosyne/) | the model: `ssm`, `memory`, `block`, `model`, `rcps`, `config` |
| [`src/data/`](src/data/) | tokenizer, genome windowing, MQAR generator |
| `src/train_mae.py`, `src/eval_longrange.py`, `src/eval_gue.py`, `src/ablate.py` | experiments |
| `src/make_report.py` | turns `results/*.json` → LaTeX (no hand-typed numbers) |
| [`tests/`](tests/) | CPU correctness tests (all green) |
| [`research_paper/mnemosyne.tex`](research_paper/mnemosyne.tex) | the paper (theory + honest results wiring) |
| [`docs/`](docs/) | [`ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`ASSESSMENT.md`](docs/ASSESSMENT.md) |
| [`deck/`](deck/) | research presentation (`.pptx`) |
| `modal_app.py`, `REMOTE_GPU.md` | free remote-GPU execution |
| [`archive/`](archive/README.md) | v1 (HopField-Mamba) notebooks, paper, docs |

## Research values

Every number in the paper is generated from real run logs by `src/make_report.py`;
unfilled cells render as a visible red TODO, so a fabricated result cannot slip in.
Negative results are reported, not hidden. See [`docs/ASSESSMENT.md`](docs/ASSESSMENT.md)
for an honest account of what is verified, what is pending, and how this might fail.

## Key references
Gu & Dao 2023 (Mamba) · Dao & Gu 2024 (SSD / Mamba-2) · Ramsauer et al. 2021
(Modern Hopfield) · Arora et al. 2023 (Zoology / MQAR) · Schiff et al. 2024
(Caduceus) · Nguyen et al. 2024 (Evo) · Brixi et al. 2025 (Evo 2) · Avsec et al.
2025 (AlphaGenome). Full list in the paper and the deck.
