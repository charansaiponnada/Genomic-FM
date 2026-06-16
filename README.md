# HopField-Mamba: A Genomic Foundation Model with Associative Memory

**Author:** Charan Sai Ponnada

A research synthesis and implementation project integrating **Modern Hopfield Networks**
with **Mamba State Space Models** for multi-species genomic sequence modeling. The core
insight: Hopfield energy dynamics can guide the selection mechanism of SSMs, providing
exponentially large effective memory at $\mathcal{O}(N)$ inference cost.

## Project Structure

| Path | Description |
|------|-------------|
| [`GenomicFM_Phase1_Prototype.ipynb`](./GenomicFM_Phase1_Prototype.ipynb) | Phase 1: Transformer MAE prototype on Human Chr22 (T4 Colab) |
| [`GenomicFM_Phase2_L40S.ipynb`](./GenomicFM_Phase2_L40S.ipynb) | Phase 2: Mamba-based MAE (single GPU prototype) |
| [`docs/HopfieldMamba_Synthesis.tex`](./docs/HopfieldMamba_Synthesis.tex) | **Research synthesis paper** — Hopfield–Mamba hybrid architecture |
| [`docs/HopfieldMamba_Synthesis.pdf`](./docs/HopfieldMamba_Synthesis.pdf) | Compiled synthesis document (10 pages) |
| [`docs/GenomicFM_Phase2.tex`](./docs/GenomicFM_Phase2.tex) | Phase 2 LaTeX report |
| [`docs/GenomicFM_Phase2.pdf`](./docs/GenomicFM_Phase2.pdf) | Compiled Phase 2 report |

## Research Summary

### Motivation
Genomic sequences demand three properties no architecture simultaneously provides:
- **$\mathcal{O}(N)$ inference** for million-base-pair contexts
- **Exponential memory capacity** for rare motifs and distal regulatory elements
- **Cross-species generalization** via shared motif vocabularies

### The Untapped Duality

The State Space Duality (SSD) framework (Dao & Gu, 2024) proved Attention $\equiv$ SSMs.
Modern Hopfield Networks (Ramsauer et al., 2021) proved Hopfield $\equiv$ Attention.
By transitivity: **Hopfield $\equiv$ SSM** — yet no architecture exploits this connection.

### Proposed Architecture

HopField-Mamba augments the Mamba recurrence with:
1. **Hopfield Read:** Content-addressable retrieval from a learned pattern memory
2. **Augmented SSM Step:** $\bm{h}_t = \bar{\bm{A}}_t \bm{h}_{t-1} + \bar{\bm{B}}_t \bm{x}_t + \sigma(\cdot) \odot \bm{h}_t^{\text{mem}}$
3. **Hopfield Write:** Memory update via learned gated Hebbian rule

### Key Claims
- **First explicit Hopfield–SSM hybrid** grounded in energy-based dynamics
- Effective memory $C_{\text{eff}} = \Theta(\min(P, \exp(d/2)))$ vs. $C_{\text{ssm}} = \Theta(d_{\text{state}})$
- Only $\sim 2.5\times$ per-token cost over standard Mamba

## References

- Ramsauer et al. 2021 — *Hopfield Networks is All You Need* (ICLR)
- Gu & Dao 2023 — *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*
- Dao & Gu 2024 — *Transformers are SSMs* (ICML, SSD framework)
- Schiff et al. 2024 — *Caduceus* (ICML, Mamba-based genomic FM)
- He et al. 2022 — *Masked Autoencoders Are Scalable Vision Learners* (CVPR)
