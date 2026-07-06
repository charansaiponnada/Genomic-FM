"""
Publication-quality Mnemosyne architecture figure (AIAYN-style two-panel).
Panel A: the RC-equivariant MAE stack.  Panel B: one Mnemosyne block's internals.
Outputs: research_paper/figures/mnemosyne_architecture.{png,svg}
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# palette
C_SSM   = "#2E8B7A"   # teal  - sequential / semiseparable
C_MEM   = "#3A6EA5"   # blue  - associative memory / low-rank
C_PERS  = "#5B8DEF"   # persistent slots
C_REG   = "#8FB8FF"   # register slots
C_GATE  = "#E08A2B"   # gate
C_IO    = "#4A4A5A"   # io boxes
C_NORM  = "#B0B0BC"   # norm/add
INK     = "#1B1B24"
GREY    = "#6A6A78"


def box(ax, x, y, w, h, text, fc, tc="white", fs=11, ec=None, style="round,pad=0.02", lw=1.2, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=lw,
                       edgecolor=ec or fc, facecolor=fc, alpha=alpha, mutation_scale=10)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tc, fontsize=fs, fontweight="medium", zorder=5)
    return (x + w / 2, y + h / 2)


def arrow(ax, p0, p1, color=GREY, lw=1.6, style="-|>", rad=0.0, ls="-"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13,
                        lw=lw, color=color, connectionstyle=f"arc3,rad={rad}",
                        linestyle=ls, zorder=1)
    ax.add_patch(a)


fig = plt.figure(figsize=(13.5, 9.2), dpi=220)
fig.patch.set_facecolor("white")
gsA = fig.add_axes([0.02, 0.02, 0.40, 0.96]); gsA.axis("off")
gsB = fig.add_axes([0.46, 0.02, 0.52, 0.96]); gsB.axis("off")
for ax in (gsA, gsB):
    ax.set_xlim(0, 10); ax.set_ylim(0, 20)

# ================= Panel A : MAE stack =================
ax = gsA
ax.text(5, 19.5, "Mnemosyne — RC-equivariant MAE", ha="center", fontsize=13.5,
        fontweight="bold", color=INK)
cx = 5.0; w = 5.6; xL = cx - w / 2

# input tokens with masking
ax.text(cx, 1.05, "DNA:  A C G [MASK] T G [MASK] C …", ha="center", fontsize=10.5,
        color=INK, family="monospace")
box(ax, xL, 1.7, w, 1.0, "Nucleotide embedding", C_IO, fs=11)
# RC twin note
box(ax, xL - 0.15, 3.05, w + 0.3, 0.85, "two strands  x  and  RC(x)   (shared weights)",
    "#EEF2F7", tc=INK, fs=9.5, ec=C_MEM)

yb = 4.35
box(ax, xL, yb, w, 2.3, "Encoder\nN × Mnemosyne block", C_SSM, fs=12)
box(ax, xL, yb + 2.7, w, 1.7, "Decoder\nM × Mnemosyne block", C_SSM, fs=11, alpha=0.85)
box(ax, xL, yb + 4.8, w, 1.0, "Linear head  →  per-base logits", C_IO, fs=10.5)
box(ax, xL - 0.15, yb + 6.15, w + 0.3, 1.05,
    r"RC symmetrize   $\frac{1}{2}[f(x)+S\,f(Rx)]$", C_MEM, fs=10.5)
box(ax, xL + 0.6, yb + 7.5, w - 1.2, 0.95, "Masked-token reconstruction loss",
    C_GATE, fs=10)

# arrows up the stack
xs = cx
ys = [2.2, 3.05, 3.9, yb, yb + 2.7, yb + 4.8, yb + 6.15, yb + 7.5]
for i in range(len(ys) - 1):
    arrow(ax, (xs, ys[i] + (1.0 if i == 0 else 0.85 if i == 1 else 0)), (xs, ys[i + 1]),
          color=GREY, lw=1.7)
arrow(ax, (cx, 1.25), (cx, 1.7))

ax.text(9.6, yb + 1.15, "O(L)", rotation=90, ha="center", va="center",
        fontsize=10, color=C_SSM, fontweight="bold")

# ================= Panel B : block internals =================
ax = gsB
ax.text(5, 19.5, "Mnemosyne block  (token-mixing = semiseparable + low-rank)",
        ha="center", fontsize=13, fontweight="bold", color=INK)

# input + norm
box(ax, 3.6, 1.4, 2.8, 0.95, "input  x", C_IO, fs=11)
box(ax, 3.6, 2.75, 2.8, 0.95, "RMSNorm", C_NORM, tc=INK, fs=10.5)
arrow(ax, (5.0, 2.35), (5.0, 2.75))

# split point
split_y = 4.1
ax.plot([5.0, 5.0], [3.7, split_y], color=GREY, lw=1.7)
ax.plot([1.9, 8.1], [split_y, split_y], color=GREY, lw=1.7)

# ---- left branch: Selective SSM ----
box(ax, 0.5, 4.7, 3.0, 1.6, "Selective SSM\n(Mamba scan)", C_SSM, fs=11.5)
ax.text(2.0, 6.55, "local · sequential\nN-semiseparable", ha="center", fontsize=8.6,
        color=C_SSM)
arrow(ax, (1.9, split_y), (2.0, 4.7), color=GREY)

# ---- right branch: associative memory ----
box(ax, 4.6, 9.05, 4.9, 1.05, "Associative memory  (Hopfield read)", C_MEM, fs=11)
ax.text(7.05, 8.55, r"$r_i=V\,\mathrm{softmax}(\beta\,q_i K^{\top})$    ·   low-rank, O(L)",
        ha="center", fontsize=8.6, color=C_MEM)
# two slot sources
box(ax, 4.6, 6.55, 2.3, 1.35, "Persistent\nslots  M\n(learned motifs)", C_PERS, fs=9.5)
box(ax, 7.2, 6.55, 2.3, 1.35, "Register slots\n(gathered from\nsequence)", C_REG, tc=INK, fs=9.5)
arrow(ax, (5.75, 7.9), (6.2, 9.05), color=C_MEM, rad=0.1)
arrow(ax, (8.35, 7.9), (7.9, 9.05), color=C_MEM, rad=-0.1)
arrow(ax, (8.1, split_y), (8.35, 6.55), color=GREY)  # seq feeds register gather
ax.text(9.9, 5.5, "seq →\nregisters", ha="center", fontsize=8.2, color=C_REG)

# gate on memory branch
box(ax, 5.4, 10.5, 3.3, 0.95, r"gate  $\sigma(\cdot)$  ⊙   (per-channel)", C_GATE, fs=10)
arrow(ax, (7.05, 10.1), (7.05, 10.5), color=C_MEM)

# combine
box(ax, 3.9, 12.1, 2.2, 0.95, "add  ⊕", C_NORM, tc=INK, fs=11)
arrow(ax, (2.0, 6.3), (4.4, 12.1), color=C_SSM, rad=0.18)     # ssm -> add
arrow(ax, (7.05, 11.45), (5.6, 12.1), color=C_GATE, rad=0.18) # gated mem -> add

# residual
box(ax, 3.9, 13.7, 2.2, 0.95, "residual  ⊕", C_NORM, tc=INK, fs=11)
arrow(ax, (5.0, 13.05), (5.0, 13.7), color=GREY)
arrow(ax, (3.6, 1.87), (3.2, 1.87), color=GREY)               # tap input
ax.plot([3.2, 3.2], [1.87, 14.17], color=GREY, lw=1.3, ls=(0, (4, 3)))
arrow(ax, (3.2, 14.17), (3.9, 14.17), color=GREY, ls="--")
box(ax, 3.9, 15.1, 2.2, 0.9, "output", C_IO, fs=11)
arrow(ax, (5.0, 14.65), (5.0, 15.1), color=GREY)

# legend
leg = [Line2D([0], [0], color=C_SSM, lw=8, label="SSM branch (semiseparable, local)"),
       Line2D([0], [0], color=C_MEM, lw=8, label="Associative memory (low-rank, global)"),
       Line2D([0], [0], color=C_GATE, lw=8, label="Learned per-channel gate")]
ax.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, -0.01),
          fontsize=8.8, frameon=False, ncol=1)

os.makedirs("research_paper/figures", exist_ok=True)
fig.savefig("research_paper/figures/mnemosyne_architecture.png", bbox_inches="tight", facecolor="white")
fig.savefig("research_paper/figures/mnemosyne_architecture.svg", bbox_inches="tight", facecolor="white")
print("wrote research_paper/figures/mnemosyne_architecture.png / .svg")
