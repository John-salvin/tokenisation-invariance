"""Figure of Section 7.2: agreement at every fragmentation level for the
original COMET-22 and the two adapters, averaged over the five held-out
languages. Written to results/figures/repair_curves.pdf.
"""
from __future__ import annotations

import pandas as pd

from .paths import DATA, FIGURES, TABLES

R = DATA / "derived" / "repair"
BUCKETS = [f"[{a:.2f},{a + 0.25:.2f})" for a in [1 + 0.25 * i for i in range(8)]]
COLOURS = {"original": "#8A8A8A", "A1": "#5B8C7A", "A2": "#2C5F7C"}   # the paper's palette


def repair_curves() -> pd.DataFrame:
    """Mean agreement over the five held-out languages: the original metric
    (lambda 0), A1 at full strength, and A2 at each fold's selected lambda.
    The first row is the canonical (unsplit) text, at ratio 1."""
    anch = pd.read_csv(R / "frontier_anchored.csv")
    plain = pd.read_csv(R / "frontier_plain.csv")
    lam = pd.read_csv(TABLES / "nested_lambda_selection.csv").set_index("held_out").lambda_star
    cols = ["rho_canonical"] + [f"rho_{b}" for b in BUCKETS]
    base = anch[anch["lambda"] == 0].set_index("held_out")[cols]
    a1 = plain[plain["lambda"] == 1].set_index("held_out")[cols]
    a2 = pd.concat([anch[(anch.held_out == h) & (anch["lambda"] == l)] for h, l in lam.items()]
                   ).set_index("held_out")[cols]
    out = pd.DataFrame({"original": base.mean(), "A1": a1.mean(), "A2": a2.mean()})
    out.insert(0, "ratio", [1.0] + [1.125 + 0.25 * i for i in range(8)])
    out.index = ["canonical"] + BUCKETS
    return out


def plot_repair(curves: pd.DataFrame, path=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shutil
    # typeset with LaTeX in Times when it is installed (as in the paper); else a Times-like font
    use_tex = shutil.which("latex") is not None
    plt.rcParams.update({
        "text.usetex": use_tex,
        "text.latex.preamble": r"\usepackage{mathptmx}",
        "font.family": "serif",
        "font.serif": ["Times", "TeX Gyre Termes", "Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "pdf.fonttype": 42, "axes.linewidth": 0.6,
    })
    fig, ax = plt.subplots(figsize=(3.05, 2.05))
    x = curves.ratio
    styles = [("original", "COMET-22 (original)", "o", "-", 1.2),
              ("A1", "A1 plain", "s", "--", 1.3),
              ("A2", "A2 anchored", "D", "-", 1.6)]
    for key, label, marker, ls, lw in styles:
        ax.plot(x, curves[key], ls, color=COLOURS[key], lw=lw, marker=marker, ms=3.2,
                markeredgewidth=0, label=label, zorder=3 if key == "A2" else 2)
    # repair at the most fragmented bucket, cost on canonical text
    xe = x.iloc[-1] + 0.07
    ax.annotate("", xy=(xe, curves.A2.iloc[-1]), xytext=(xe, curves.original.iloc[-1]),
                arrowprops=dict(arrowstyle="<->", color="#3A3A3A", lw=0.6, shrinkA=0, shrinkB=0))
    ax.text(xe + 0.04, (curves.A2.iloc[-1] + curves.original.iloc[-1]) / 2, "repair",
            rotation=90, va="center", ha="left", fontsize=7, color="#3A3A3A")
    # A1's cost: its drop on canonical text (ratio 1), the leftmost point
    xc = x.iloc[0] - 0.06
    ax.annotate("", xy=(xc, curves.A1.iloc[0]), xytext=(xc, curves.original.iloc[0]),
                arrowprops=dict(arrowstyle="<->", color="#3A3A3A", lw=0.6, shrinkA=0, shrinkB=0))
    ax.text(xc - 0.04, (curves.A1.iloc[0] + curves.original.iloc[0]) / 2, "A1 cost",
            rotation=90, va="center", ha="right", fontsize=7, color="#3A3A3A")
    ax.set_xlim(0.80, 3.08)
    ax.set_ylim(0.08, 0.6)
    ax.set_xticks([1.0, 1.5, 2.0, 2.5, 3.0])
    ax.set_xticklabels(["1.0", "1.5", "2.0", "2.5", "3.0"])
    ax.set_xlabel("Fragmentation ratio (tokens / canonical tokens)")
    ax.set_ylabel(r"Agreement with MQM (Spearman $\rho$)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E6E6E6", lw=0.5, zorder=0)
    ax.legend(loc="lower left", frameon=False, handlelength=2.2, borderaxespad=0.2)
    fig.tight_layout(pad=0.2)
    path = path or FIGURES / "repair_curves.pdf"
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"CreationDate": None})      # no timestamp in the file
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=300)
    plt.close(fig)


def run() -> None:
    c = repair_curves()
    c.to_csv(TABLES / "repair_curves.csv")
    plot_repair(c)


if __name__ == "__main__":
    run()
