"""Deck assets: a SCHEMATIC (hypothesis) long-range curve + a small data-flow icon.
The curve is explicitly labelled as a prediction, not measured data."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs(os.path.dirname(__file__) or ".", exist_ok=True)
TEAL, BLUE, AMBER, INK, GREY = "#2E8B7A", "#3A6EA5", "#E08A2B", "#1B1B24", "#8A8A96"

# ---- expected long-range curve (schematic) ----
fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
L = np.array([64, 128, 256, 512, 1024, 2048])
mnem = np.array([0.99, 0.98, 0.98, 0.97, 0.97, 0.96])
ssm = np.array([0.98, 0.90, 0.74, 0.55, 0.38, 0.26])
ax.plot(L, mnem, "-o", color=BLUE, lw=3, ms=7, label="Mnemosyne (memory)")
ax.plot(L, ssm, "-o", color=TEAL, lw=3, ms=7, label="SSM baseline (no memory)")
ax.set_xscale("log", base=2)
ax.set_xticks(L); ax.set_xticklabels(L)
ax.set_xlabel("sequence length  =  recall distance", fontsize=11)
ax.set_ylabel("MQAR query accuracy", fontsize=11)
ax.set_ylim(0.15, 1.03)
ax.legend(fontsize=10, frameon=False, loc="lower left")
ax.grid(alpha=0.25)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.text(0.5, 0.5, "SCHEMATIC — HYPOTHESIS\n(to be measured on GPU)",
        transform=ax.transAxes, ha="center", va="center", fontsize=15,
        color=GREY, alpha=0.30, rotation=18, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__), "expected_longrange.png"),
            bbox_inches="tight", facecolor="white")
print("wrote expected_longrange.png")
