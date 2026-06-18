# Publication Roadmap — HopField-Mamba

## Current Status
| Metric | Value |
|---|---|
| Mamba parameters | 16.06M |
| HopField-Mamba parameters | 27.09M |
| Memory slots (P) | 512 (paper incorrectly states 1024 — fix needed) |
| Pre-training data | 25,000 windows (5K × 5 species) |
| Training epochs | 10 |
| Downstream task | Synthetic promoter (100% AUROC — ceiling effect) |
| Cross-species range | HFM: 6.7 pp vs Mamba: 7.0 pp |
| Bidirectional | No |

---

## Critical Fixes Before Publication

### 1. Paper Discrepancy
- [ ] Fix `P=1024` → `P=512` in `hopfield_mamba_genomics.tex` Section 4.1
- [ ] Verify all parameter counts match the actual implementation

### 2. GUE Benchmarks (Replace Synthetic Data)
| Dataset | Task | Why It Matters |
|---|---|---|
| `gue_promoter` | Promoter detection | Real data, no ceiling effect |
| `gue_splice` | Splice-site classification | Tests exon/intron boundary detection |
| `gue_tf_binding` | Transcription factor binding | Tests regulatory motif understanding |
| `gue_variant` | Variant effect prediction | Most clinically relevant |

- [ ] Run linear probe on all GUE datasets
- [ ] Report accuracy, AUROC, MCC
- [ ] Compare against published HyenaDNA, Caduceus, DNABERT-2 numbers

### 3. Bidirectional Processing
- [ ] Implement reverse-complement SSM (Caduceus-style)
- [ ] Add `--bidirectional` flag to training script
- [ ] Ablate: bidirectional vs unidirectional for each model

### 4. Data Scaling
| Stage | Windows | Species | Total bases |
|---|---|---|---|
| Current | 25K | 5 | 0.1B |
| Target | 1M | 10+ | 4B |

- [ ] Add full chromosomes (not just one FASTA per species)
- [ ] Add more species (primate, bird, reptile, bacteria)
- [ ] Remove the 5,000-window cap per species
- [ ] Scale to millions of windows

### 5. Model Scaling
| Target | Params | d_model | Layers | P |
|---|---|---|---|---|
| Small (current) | 27M | 512 | 6 | 512 |
| Medium | 100M | 768 | 12 | 1024 |
| Large | 350M | 1024 | 24 | 2048 |
| XL | 1B+ | 1536 | 36 | 4096 |

- [ ] Train medium and large variants
- [ ] Ablate P: {256, 512, 1024, 2048}
- [ ] Ablate write rate λ: {0.001, 0.01, 0.1}

### 6. Architecture Refinements
- [ ] Fix memory write equation: `M ← M + λ · MLP(h) · (MLP(h) - M)` should be checked for gradient stability
- [ ] Consider replacing `softmax` with `entmax` for sparse memory attention
- [ ] Add residual connections around the Hopfield module
- [ ] Experiment with layer-specific memory vs shared memory across layers

### 7. Evaluation Hardening
| Benchmark | What it measures |
|---|---|
| GUE (full suite) | Genomic understanding across 9 tasks |
| Nucleotide Transformer benchmarks | Multi-species representation quality |
| Cross-species zero-shot transfer | Evolutionary generalization (already started) |
| Long-range dependence test | Measure retrieval of motifs at distance > 2K bp |

---

## Target Venues & Submission Timeline

| Venue | Deadline | Tier | Notes |
|---|---|---|---|
| NeurIPS Workshop (MLSB/AI4Science) | Sep 2026 | Workshop | Good for early feedback |
| ICML | Jan 2027 | A* | Needs full GUE + scaling |
| ISMB | Jan 2027 | A (bio) | Needs biological validation |
| ICLR | Sep 2026 | A* | Most competitive |
| Bioinformatics (Journal) | Rolling | Q1 | Needs rigorous benchmarks |

---

## Summary of Gaps
1. Data is ~1000× smaller than SOTA pre-training sets
2. Model is ~10× smaller than SOTA genomic models
3. No bidirectional processing (critical for DNA)
4. Only one downstream task (synthetic ceiling)
5. No comparison against existing published benchmarks
6. Paper has a `P=1024` vs `P=512` discrepancy
