# GenomicFM — Multi-Species Genomic Foundation Model

**Author:** Charan Sai Ponnada

A self-supervised deep learning model that learns DNA sequence representations across multiple organisms using a **Masked Autoencoder (MAE)** architecture adapted from vision Transformers.

---

## Work Completed — Phase 1 (Prototype)

A complete Phase 1 prototype implemented in [`GenomicFM_Phase1_Prototype.ipynb`](./GenomicFM_Phase1_Prototype.ipynb), designed to run on a **Google Colab T4 (16GB VRAM)**. Successfully executed as confirmed in commit `e2f0b25`.

### Pipeline

1. **DNATokenizer** — Character-level tokenizer with 10-token vocabulary (`[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`, `A`, `C`, `G`, `T`, `N`). Handles ambiguous IUPAC codes.

2. **Data Pipeline** — Downloads **Human Chromosome 22** (GRCh38.p14, ~50MB) from NCBI FTP, parses FASTA, applies N-filtering (rejects windows with >10% ambiguous bases), extracts non-overlapping 512-bp windows (~1MB clean sequence).

3. **GenomicMAE Architecture** (~4M parameters):
   - Embedding dim: 128
   - Encoder: 4 Transformer layers, 8 heads
   - Decoder: 2 Transformer layers, 4 heads
   - Mask ratio: 75%
   - Pre-LayerNorm, GELU, learned positional encodings
   - 8-species embedding slot (Phase 2 ready)

4. **Training**:
   - 3 epochs, LR 1e-4 (OneCycle, cosine decay, 5% warmup)
   - AdamW (β=(0.9, 0.95), weight_decay=0.01)
   - Gradient accumulation ×4 (effective batch size 64)
   - FP16 mixed precision

### Key Design Insight

MAE drops 75% of tokens *before* the encoder, reducing attention from O(N²) to O((0.25N)²) — a **16× reduction** vs. standard BERT MLM.

### Expected Outcomes

| Metric | Random Baseline | Target |
|--------|----------------|--------|
| Loss | 2.30 (log 10) | <1.8 |
| Reconstruction accuracy | 20% | >20% |

### Diagnostics

Loss curve, LR schedule, nucleotide distribution (GC content), per-epoch loss comparison, VRAM usage breakdown, and color-coded reconstruction tests.

---

## Roadmap — Phase 2+

| Step | Description |
|------|-------------|
| 1. Scale data | Add Mouse, Zebrafish, *Drosophila* with species embeddings |
| 2. Mamba backbone | Replace Transformer with Mamba for O(N) scaling (ref. Caduceus, Schiff et al. 2024) |
| 3. Scale model | d_model 128→256→512, layers 4→8→12, target ~100M params |
| 4. DDP multi-GPU | DistributedDataParallel on L40S cluster (2×48GB) |
| 5. Downstream eval | GUE benchmarks: promoters, splice sites, variant effects, cross-species transfer |

### References

- He et al. 2022 — *Masked Autoencoders Are Scalable Vision Learners* (CVPR)
- Safari et al. 2025 — DNA masking inefficiencies (motivation for MAE)
- Nguyen et al. 2023 — *HyenaDNA* (NeurIPS)
- Schiff et al. 2024 — *Caduceus* (ICML, Mamba-based genomic FM)
- Dalla-Torre et al. 2023 — *Nucleotide Transformer* (bioRxiv)
