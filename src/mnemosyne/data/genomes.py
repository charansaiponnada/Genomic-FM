"""
Genome windowing for MAE pre-training.

Downloads FASTA(s), slices them into fixed-length windows, tokenises to
single-nucleotide ids, and caches token arrays to disk. Designed to run on the
remote GPU box; falls back to a random-DNA generator for CPU smoke tests.

Example (multi-species, small):
    URLS = {
      "human": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz",
      ...
    }
    ds = build_windows(URLS, window=1024, stride=512, max_windows_per=20000)
"""
from __future__ import annotations
import os
import gzip
import io
import urllib.request
import numpy as np
import torch

from .tokenizer import BASE2ID

CACHE = os.environ.get("GENOME_CACHE", "./data/cache")


def _iter_fasta(fh):
    name, chunks = None, []
    for line in fh:
        line = line.decode() if isinstance(line, bytes) else line
        if line.startswith(">"):
            if name is not None:
                yield name, "".join(chunks)
            name, chunks = line[1:].strip(), []
        else:
            chunks.append(line.strip())
    if name is not None:
        yield name, "".join(chunks)


def _download(url: str) -> str:
    os.makedirs(CACHE, exist_ok=True)
    fname = os.path.join(CACHE, os.path.basename(url))
    if not os.path.exists(fname):
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, fname)
    return fname


def _seq_from_file(path: str) -> str:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rb") as f:
        return "".join(seq for _, seq in _iter_fasta(f))


def tokenize_windows(seq: str, window: int, stride: int, max_windows: int,
                     max_n_frac: float = 0.05) -> np.ndarray:
    ids = np.frombuffer(seq.upper().encode(), dtype=np.uint8)
    lut = np.full(256, 6, dtype=np.int64)  # default N
    for ch, i in BASE2ID.items():
        lut[ord(ch)] = i
    toks = lut[ids]
    out, n_id = [], 6
    for start in range(0, len(toks) - window + 1, stride):
        w = toks[start:start + window]
        if (w == n_id).mean() <= max_n_frac:
            out.append(w)
            if len(out) >= max_windows:
                break
    return np.stack(out) if out else np.empty((0, window), dtype=np.int64)


def build_windows(urls: dict, window=1024, stride=512, max_windows_per=20000,
                  species_map=None):
    """Returns (tokens LongTensor (N, window), species LongTensor (N,))."""
    species_map = species_map or {s: i for i, s in enumerate(urls)}
    all_tok, all_sp = [], []
    for sp, url in urls.items():
        cache = os.path.join(CACHE, f"{sp}_w{window}_s{stride}_m{max_windows_per}.npy")
        if os.path.exists(cache):
            arr = np.load(cache)
        else:
            seq = _seq_from_file(_download(url))
            arr = tokenize_windows(seq, window, stride, max_windows_per)
            os.makedirs(CACHE, exist_ok=True)
            np.save(cache, arr)
        print(f"{sp}: {len(arr)} windows")
        all_tok.append(arr)
        all_sp.append(np.full(len(arr), species_map[sp], dtype=np.int64))
    tok = np.concatenate(all_tok); sp = np.concatenate(all_sp)
    return torch.from_numpy(tok), torch.from_numpy(sp)


def random_windows(n=2000, window=1024, seed=0):
    """CPU smoke fallback: random ACGT windows."""
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.integers(2, 6, size=(n, window)).astype(np.int64)), \
        torch.zeros(n, dtype=torch.long)


# Small smoke-scale set (single small chromosomes) -- fast to download, for testing.
SMOKE_URLS = {
    "human": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr21.fa.gz",
    "mouse": "https://hgdownload.soe.ucsc.edu/goldenPath/mm39/chromosomes/chr19.fa.gz",
    "zebrafish": "https://hgdownload.soe.ucsc.edu/goldenPath/danRer11/chromosomes/chr25.fa.gz",
}

# The five-species pre-training corpus described in the project documentation
# (research_paper/mnemosyne_documentation.tex, Table 1). Cross-kingdom coverage:
# 3 vertebrates + 1 plant + 1 invertebrate. Actual bytes ingested are controlled
# by `max_windows_per`, so the corpus size is set by the training config, not by
# the raw download size.
DEFAULT_URLS = {
    "human":      "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr1.fa.gz",
    "mouse":      "https://hgdownload.soe.ucsc.edu/goldenPath/mm39/chromosomes/chr1.fa.gz",
    "zebrafish":  "https://hgdownload.soe.ucsc.edu/goldenPath/danRer11/chromosomes/chr1.fa.gz",
    "arabidopsis": "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-58/fasta/"
                   "arabidopsis_thaliana/dna/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz",
    "drosophila": "https://hgdownload.soe.ucsc.edu/goldenPath/dm6/bigZips/dm6.fa.gz",
}
