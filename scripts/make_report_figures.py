"""Generate the three report figures from outputs/metrics.json. Palette
validated via the dataviz skill's validate_palette.js (blue/orange pair,
all checks pass, light mode)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#8a8a86"
TEXT = "#2b2b28"

plt.rcParams.update({
    "font.size": 10.5, "text.color": TEXT, "axes.edgecolor": "#c9c8c0",
    "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def fig_coherence(metrics):
    coh = metrics["coherence"]["2026"] if "2026" in metrics["coherence"] else metrics["coherence"][2026]
    levels = sorted(coh, key=lambda k: int(k))
    observed = [coh[l]["observed_coherence"] for l in levels]
    null_mean = [coh[l]["null_mean"] for l in levels]
    z = [coh[l]["z_score"] for l in levels]

    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = range(len(levels))
    w = 0.32
    ax.bar([i - w / 2 for i in x], observed, width=w, color=BLUE, label="Observed (TF-IDF)")
    ax.bar([i + w / 2 for i in x], null_mean, width=w, color=GRAY, label="Null (random labels, 30 trials)")
    for i, zi in enumerate(z):
        ax.text(i, observed[i] + 0.003, f"z≈{zi:.0f}", ha="center", fontsize=9, color=TEXT)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"level {l}" for l in levels])
    ax.set_ylabel("mean pairwise TF-IDF cosine similarity")
    ax.set_title("Coherence vs. random-labels null (2026 snapshot)", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_ylim(0, max(observed) * 1.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "coherence_vs_null.png", dpi=150)
    plt.close(fig)


def fig_stability(metrics):
    stab = metrics["stability"]
    cross = stab["cross_snapshot"]["by_level"]
    pert = stab["perturbation"]["by_level"]
    levels = sorted(cross, key=lambda k: int(k))

    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = range(len(levels))
    w = 0.32

    pert_mean = [pert[l]["mean_ari"] for l in levels]
    pert_err = [[pert[l]["mean_ari"] - pert[l]["ci95"][0] for l in levels],
                [pert[l]["ci95"][1] - pert[l]["mean_ari"] for l in levels]]
    cross_mean = [cross[l]["mean_ari"] for l in levels]
    cross_err = [[cross[l]["mean_ari"] - cross[l]["ci95"][0] for l in levels],
                 [cross[l]["ci95"][1] - cross[l]["mean_ari"] for l in levels]]

    ax.bar([i - w / 2 for i in x], pert_mean, width=w, color=BLUE, label="Perturbation (5 seeds, 10% edges removed)",
           yerr=pert_err, capsize=3, error_kw={"linewidth": 1, "ecolor": TEXT})
    ax.bar([i + w / 2 for i in x], cross_mean, width=w, color=ORANGE, label="Cross-snapshot (3 transitions)",
           yerr=cross_err, capsize=3, error_kw={"linewidth": 1, "ecolor": TEXT})

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"level {l}" for l in levels])
    ax.set_ylabel("Adjusted Rand Index")
    ax.set_title("Stability by level, mean ± 95% CI", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "stability_by_level.png", dpi=150)
    plt.close(fig)


def fig_extrinsic_sweep(metrics):
    ext = metrics["extrinsic"]
    sweep = sorted(ext["branching_factor_sweep"], key=lambda r: r["mean_candidates_scored"])
    flat_recall = ext["summary"]["flat_mean_recall"]
    flat_scored = ext["summary"]["flat_mean_candidates_scored"]

    fig, ax = plt.subplots(figsize=(6, 3.6))
    xs = [r["mean_candidates_scored"] for r in sweep]
    ys = [r["mean_recall"] for r in sweep]
    ax.plot(xs, ys, color=BLUE, linewidth=1.6, marker="o", markersize=5, label="Drill-down (b0, b1 swept)")
    labeled = {(3, 3), (5, 5)}
    for r in sweep:
        if (r["b0"], r["b1"]) in labeled:
            ax.annotate(f"({r['b0']},{r['b1']})", (r["mean_candidates_scored"], r["mean_recall"]),
                        textcoords="offset points", xytext=(6, -12), fontsize=8.5, color=TEXT)

    ax.scatter([flat_scored], [flat_recall], color=ORANGE, s=55, zorder=5, marker="D", label="Flat baseline (all 3104)")
    ax.axhline(flat_recall, color=ORANGE, linewidth=0.8, linestyle=":", alpha=0.6)

    ax.set_xlabel("mean candidates scored per question")
    ax.set_ylabel("mean recall@20")
    ax.set_title("Drill-down branching sweep vs. flat baseline (2026, 14 questions)", loc="left", fontsize=10.5)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "extrinsic_branching_sweep.png", dpi=150)
    plt.close(fig)


def main():
    metrics = json.loads((ROOT / "outputs" / "metrics.json").read_text(encoding="utf-8"))
    fig_coherence(metrics)
    fig_stability(metrics)
    fig_extrinsic_sweep(metrics)
    print(f"wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
