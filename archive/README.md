# Archive — v1 (HopField-Mamba) research record

This folder preserves the **original v1 work** so nothing is lost. The active
project is now **Mnemosyne** (`../src/`, `../research_paper/mnemosyne.tex`). Kept
here for provenance and reproducibility, not active development.

| Path | What it is |
|------|------------|
| `notebooks/` | Phase 1–4 Colab notebooks: Transformer MAE prototype, Mamba MAE, HopField-Mamba, GUE evaluation (with their execution outputs). The v1 experimental record. |
| `paper_v1/` | The v1 IEEE paper (`hopfield_mamba_genomics.tex/.pdf`), its GUE table, TikZ/draw.io figures, and the figures it embeds. |
| `docs_v1/` | v1 LaTeX reports: Phase-2 report and the HopField–Mamba synthesis. |
| `figures_v1/` | v1 pipeline diagram and dataset/diagnostic PNGs. |

## Why v1 was superseded
v1's associative memory was written per-token *within* a sequence, so it never
accumulated cross-sequence structure and did not move the pre-training loss
(27M vs 16M params, identical loss). It was also unidirectional and evaluated with
an unfair frozen-probe-vs-full-fine-tune comparison. Mnemosyne (v2) fixes all of
these — see `../research_paper/publication_roadmap.md` and `../docs/ASSESSMENT.md`.
