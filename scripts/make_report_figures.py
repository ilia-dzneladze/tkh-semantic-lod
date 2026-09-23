"""Generate the report figures from outputs/metrics.json. Palette
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
    "figure.facecolor": "white", "axes.facecolor": "white", "pdf.fonttype": 42,
})


def _save(fig, name):
    """PNG for quick viewing, vector PDF for the LaTeX report."""
    fig.savefig(OUT_DIR / f"{name}.png", dpi=150)
    fig.savefig(OUT_DIR / f"{name}.pdf")
    plt.close(fig)


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
    _save(fig, "coherence_vs_null")


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
    ax.bar([i + w / 2 for i in x], cross_mean, width=w, color=ORANGE, label="Cross-snapshot (3 transitions, half-sampling CI)",
           yerr=cross_err, capsize=3, error_kw={"linewidth": 1, "ecolor": TEXT})

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"level {l}" for l in levels])
    ax.set_ylabel("Adjusted Rand Index")
    ax.set_title("Stability by level, mean ± 95% CI", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    _save(fig, "stability_by_level")


def fig_extrinsic_sweep(metrics):
    ext = metrics["extrinsic"]
    sweep = sorted(ext["branching_factor_sweep"], key=lambda r: r["mean_candidates_scored"])
    flat_recall = ext["summary"]["flat_mean_recall"]
    flat_scored = ext["summary"]["flat_mean_candidates_scored"]
    n_q = ext["summary"]["n_questions_scored"]

    fig, ax = plt.subplots(figsize=(6, 3.6))
    xs = [r["mean_candidates_scored"] for r in sweep]
    ys = [r["mean_recall"] for r in sweep]
    ax.plot(xs, ys, color=BLUE, linewidth=1.6, marker="o", markersize=5, label="Drill-down (b0, b1 swept)")
    labeled = {(3, 3), (8, 8)}
    for r in sweep:
        if (r["b0"], r["b1"]) in labeled:
            ax.annotate(f"({r['b0']},{r['b1']})", (r["mean_candidates_scored"], r["mean_recall"]),
                        textcoords="offset points", xytext=(6, -12), fontsize=8.5, color=TEXT)

    ax.scatter([flat_scored], [flat_recall], color=ORANGE, s=55, zorder=5, marker="D", label=f"Flat baseline (all {flat_scored:.0f})")
    ax.axhline(flat_recall, color=ORANGE, linewidth=0.8, linestyle=":", alpha=0.6)

    ax.set_xlabel("mean candidates scored per question")
    ax.set_ylabel("mean recall@20")
    ax.set_title(f"Drill-down branching sweep vs. flat baseline (2026, {n_q} questions)",
                 loc="left", fontsize=10.5)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    _save(fig, "extrinsic_branching_sweep")


def fig_rerank_sweep(metrics):
    ext = metrics["extrinsic"]
    sweep = sorted(ext["rerank_beta_sweep"], key=lambda r: r["beta"])
    flat_recall = ext["summary"]["flat_mean_recall"]
    shipped_beta = ext["summary"]["branching"]["beta"]
    n_q = ext["summary"]["n_questions_scored"]

    fig, ax = plt.subplots(figsize=(6, 3.6))
    xs = [r["beta"] for r in sweep]
    ys = [r["mean_recall"] for r in sweep]
    ax.plot(xs, ys, color=BLUE, linewidth=1.6, marker="o", markersize=5,
            label="Drill-down (b0=8, b1=8), reranked")
    for r in sweep:
        if r["beta"] == shipped_beta:
            ax.scatter([r["beta"]], [r["mean_recall"]], color=ORANGE, s=65, zorder=5,
                       marker="D", label=f"Shipped (beta={shipped_beta})")

    ax.axhline(flat_recall, color=GRAY, linewidth=0.8, linestyle=":", alpha=0.8)
    ax.text(0.02, flat_recall + 0.0008, "flat baseline", fontsize=8.5, color=GRAY)

    ax.set_xlabel("beta (0 = own embedding only, 1 = ancestor label+gloss only)")
    ax.set_ylabel("mean recall@20")
    ax.set_title(f"Hierarchy-context reranking vs. flat baseline (2026, {n_q} questions)",
                 loc="left", fontsize=10.5)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    _save(fig, "rerank_beta_sweep")


def fig_routing(metrics):
    """Share of ground truth kept in the routed pool vs the pool's size.
    Error bars are the bootstrap 95% CI of the lift over chance (over
    questions), drawn around each point."""
    routing = metrics["extrinsic"]["routing"]
    n_q = metrics["extrinsic"]["summary"]["n_questions_scored"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, color, marker, text in (("label", BLUE, "o", "label+gloss routing"),
                                      ("centroid", ORANGE, "s", "centroid routing (no labels)")):
        rows = sorted(routing[name], key=lambda r: r["mean_pool_fraction"])
        xs = [r["mean_pool_fraction"] for r in rows]
        ys = [r["gt_in_pool_rate"] for r in rows]
        err = [[r["lift_over_chance"] - r["lift_ci95"][0] for r in rows],
               [r["lift_ci95"][1] - r["lift_over_chance"] for r in rows]]
        ax.errorbar(xs, ys, yerr=err, color=color, linewidth=1.6, marker=marker, markersize=5,
                    capsize=3, elinewidth=1, label=text)
        last = rows[-1]
        ax.annotate(f"({last['b0']},{last['b1']})", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(6, -3), fontsize=8.5, color=TEXT)
    lim = 0.3
    ax.plot([0, lim], [0, lim], color=GRAY, linewidth=0.9, linestyle="--", label="random pool (chance)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("share of candidates in the routed pool")
    ax.set_ylabel("share of ground-truth nodes kept")
    ax.set_title(f"Coarse-to-fine routing vs. chance (2026, {n_q} questions)", loc="left", fontsize=10.5)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    _save(fig, "routing_vs_chance")


def fig_tradeoff(metrics):
    """Structure-vs-meaning trade-off across alpha: held-out hyperedge lift
    (structure side) against TF-IDF coherence over its random null (meaning
    side), one panel per level, one line per holdout scheme."""
    sh = metrics["structural_holdout"]
    levels = sorted(sh["summary"]["edge"], key=lambda k: int(k))
    fig, axes = plt.subplots(1, len(levels), figsize=(9, 3.3))
    for ax, level in zip(axes, levels):
        for scheme, color, marker, text in (("edge", BLUE, "o", "random edges held out"),
                                            ("paper", ORANGE, "s", "whole papers held out")):
            # alpha=1.0 is degenerate (structure-only graph too sparse: ~1100
            # forced merges, one cluster), so it's left out of the plot
            rows = [r for r in sh["summary"][scheme][level]["by_alpha"] if r["forced_merges_mean"] == 0]
            xs = [r["tfidf_ratio"] for r in rows]
            ys = [r["heldout_lift"] for r in rows]
            ax.plot(xs, ys, color=color, linewidth=1.6, marker=marker, markersize=5, label=text)
            for r in rows:
                if r["alpha"] == sh["shipped_alpha"]:
                    ax.scatter([r["tfidf_ratio"]], [r["heldout_lift"]], s=120, facecolors="none",
                               edgecolors=TEXT, linewidths=1.2, zorder=5,
                               label=f"shipped α={r['alpha']:g}" if scheme == "edge" else None)
                if scheme == "edge" and r["alpha"] in (0.0, 0.7):
                    offset = (5, 4) if r["alpha"] else (12, 12)
                    ax.annotate(f"α={r['alpha']:g}", (r["tfidf_ratio"], r["heldout_lift"]),
                                textcoords="offset points", xytext=offset, fontsize=8, color=TEXT,
                                zorder=6, bbox=dict(fc="white", ec="none", pad=0.3))
        ax.margins(x=0.12, y=0.1)
        ax.set_title(f"level {level}", loc="left", fontsize=10)
        ax.set_xlabel("TF-IDF coherence / null")
    axes[0].set_ylabel("held-out edge lift over chance")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Structure vs. meaning across alpha (2026, 5 seeds, 20% held out)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, "structure_meaning_tradeoff")


def _rate_dot(ax, x, r, color, marker, label=None, side="right"):
    lo, hi = r["ci95"]
    ax.errorbar([x], [r["rate"]], yerr=[[r["rate"] - lo], [hi - r["rate"]]], color=color, marker=marker,
                markersize=7, linestyle="none", capsize=3, elinewidth=1.2, label=label, zorder=3)
    dx, ha = (8, "left") if side == "right" else (-8, "right")
    ax.annotate(f"{r['k']}/{r['n']}", (x, r["rate"]), textcoords="offset points", xytext=(dx, -3),
                ha=ha, fontsize=8.5, color=TEXT)


def fig_blind(metrics):
    """Blind intruder detection by level against chance and null items, and
    the blind gloss ratings for real vs control items. Rates with Wilson
    95% CIs. Real items blue circles, null/control gray squares."""
    b = metrics["blind_eval"]
    intr, gloss = b["intruder"], b["gloss"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.5))

    levels = sorted(intr["real_by_level"], key=int)
    for i, lvl in enumerate(levels):
        _rate_dot(ax1, i, intr["real_by_level"][lvl], BLUE, "o", "real items" if i == 0 else None)
    _rate_dot(ax1, len(levels), intr["null"], GRAY, "s", "null items")
    ax1.axhline(intr["chance"], color=GRAY, linewidth=0.9, linestyle="--")
    ax1.text(-0.45, intr["chance"] + 0.02, "chance (1 in 6)", fontsize=8.5, color=GRAY, ha="left")
    ax1.set_xticks(range(len(levels) + 1))
    ax1.set_xticklabels([f"level {l}" for l in levels] + ["null"])
    ax1.set_xlim(-0.5, len(levels) + 0.5)
    ax1.set_ylim(-0.04, 1.0)
    ax1.set_ylabel("intruder found")
    ax1.set_title("Intruder test, 2026", loc="left", fontsize=10.5)
    ax1.legend(frameon=False, loc="upper left", fontsize=8.5)

    cats = ["accurate", "vague", "wrong"]
    for j, cat in enumerate(cats):
        _rate_dot(ax2, j - 0.14, gloss["real"][cat], BLUE, "o", "real gloss" if j == 0 else None, side="left")
        _rate_dot(ax2, j + 0.14, gloss["control"][cat], GRAY, "s", "control (other cluster)" if j == 0 else None)
    ax2.set_xticks(range(len(cats)))
    ax2.set_xticklabels(cats)
    ax2.set_xlim(-0.5, len(cats) - 0.5)
    ax2.set_ylim(-0.04, 1.0)
    ax2.set_ylabel("share of items")
    ax2.set_title("Gloss rating, all snapshots", loc="left", fontsize=10.5)
    ax2.legend(frameon=False, loc="upper center", fontsize=8.5)

    for ax in (ax1, ax2):
        ax.grid(axis="y", color="#e4e3dd", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "blind_eval")


def main():
    metrics = json.loads((ROOT / "outputs" / "metrics.json").read_text(encoding="utf-8"))
    if "blind_eval" in metrics:
        fig_blind(metrics)
    fig_coherence(metrics)
    fig_stability(metrics)
    fig_extrinsic_sweep(metrics)
    if "rerank_beta_sweep" in metrics.get("extrinsic", {}):
        fig_rerank_sweep(metrics)
    if "routing" in metrics.get("extrinsic", {}):
        fig_routing(metrics)
    if "structural_holdout" in metrics:
        fig_tradeoff(metrics)
    print(f"wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
