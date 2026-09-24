"""Generate the report figures from outputs/metrics.json, plus the static
overview of levels 0-2 across snapshots from the shipped hierarchy.json
files. Palettes validated via the dataviz skill's validate_palette.js
(light mode, all checks pass; the eight-hue set warns on contrast, so
every block in the overview is labelled or numbered)."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT_DIR = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from tkh.io import SNAPSHOT_CUTOFFS  # noqa: E402

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#8a8a86"
TEXT = "#2b2b28"
MUTED = "#6b6a65"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER = "#a8a7a1"

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


def _pale(color, share=0.55):
    return tuple(c + (1 - c) * share for c in to_rgb(color))


def _ribbon(ax, x0, x1, src, dst, color, alpha):
    """Filled S-curve from the span src=(top, bottom) at x0 to dst at x1."""
    xm = (x0 + x1) / 2
    verts = [(x0, src[0]), (xm, src[0]), (xm, dst[0]), (x1, dst[0]),
             (x1, dst[1]), (xm, dst[1]), (xm, src[1]), (x0, src[1]), (x0, src[0])]
    codes = ([MplPath.MOVETO] + [MplPath.CURVE4] * 3 + [MplPath.LINETO]
             + [MplPath.CURVE4] * 3 + [MplPath.CLOSEPOLY])
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none",
                           alpha=alpha, zorder=1))


def _spread(ys, gap, lo, hi):
    """Push label heights (sorted top to bottom) at least gap apart inside [lo, hi]."""
    out = []
    for y in ys:
        out.append(min(y, out[-1] - gap if out else hi))
    for k in range(len(out) - 1, -1, -1):
        out[k] = max(out[k], out[k + 1] + gap if k + 1 < len(out) else lo)
    return out


def _short_id(i):
    return str(int(i.rsplit("S", 1)[1]))


def fig_hierarchy():
    """Levels 0-2 at every snapshot as nested bars (each child inside its
    parent's span, height = node count on one scale for all snapshots),
    with ribbons for the level-0 members each snapshot passes to the next."""
    years = SNAPSHOT_CUTOFFS
    nodes, children = {}, {}
    for y in years:
        h = json.loads((ROOT / "outputs" / "snapshots" / str(y) / "hierarchy.json").read_text(encoding="utf-8"))
        nodes[y] = {sn["id"]: sn for sn in h["super_nodes"]}
        children[y] = defaultdict(list)
        for sn in sorted(h["super_nodes"], key=lambda s: s["id"]):
            if sn["parent_id"]:
                children[y][sn["parent_id"]].append(sn)
    top = {y: {i: set(sn["member_ids"]) for i, sn in nodes[y].items() if sn["level"] == 0} for y in years}

    flows = {}  # year -> {(id at year, id at next year): shared members}
    for a, b in zip(years, years[1:]):
        flows[a] = {(i, j): len(mi & mj) for i, mi in top[a].items()
                    for j, mj in top[b].items() if mi & mj}

    # hue for the level-0 ids alive at 3+ snapshots, gray for the rest
    alive = Counter(i for y in years for i in top[y])
    hued = sorted(sorted((i for i in alive if alive[i] >= 3), key=lambda i: -alive[i])[:len(SERIES)])
    color = {i: SERIES[k] for k, i in enumerate(hued)}

    # vertical order: hued ids in hue order, each gray id at the
    # member-weighted mean position of the ids it exchanges members with
    key = {i: float(k) for k, i in enumerate(hued)}
    partners = defaultdict(Counter)
    for a in flows:
        for (i, j), n in flows[a].items():
            if i != j:
                partners[i][j] += n
                partners[j][i] += n
    for _ in range(5):
        for i in sorted(alive):
            known = {j: n for j, n in partners[i].items() if j in key}
            if i not in color and known:
                key[i] = sum(key[j] * n for j, n in known.items()) / sum(known.values())
    order = {y: sorted(top[y], key=lambda i: (key.get(i, len(hued)), i)) for y in years}

    pad = 60  # gap between level-0 blocks, in nodes
    widths, gap = (0.32, 0.16, 0.16), 0.04
    x0 = {y: 3.45 + 2.2 * c for c, y in enumerate(years)}

    def bar_x(y, level):
        return x0[y] + sum(widths[:level]) + gap * level

    right = {y: bar_x(y, 2) + widths[2] for y in years}
    height = {y: sum(map(len, top[y].values())) + pad * (len(top[y]) - 1) for y in years}
    hmax = max(height.values())
    span = {}
    for y in years:
        cursor = (hmax + height[y]) / 2
        for i in order[y]:
            span[(y, i)] = (cursor, cursor - len(top[y][i]))
            cursor -= len(top[y][i]) + pad

    fig_h, ax_frac = 9.6, 0.76
    fig = plt.figure(figsize=(14, fig_h))
    ax = fig.add_axes((0, 0.14, 1, ax_frac))
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.01 * hmax, 1.01 * hmax)
    ax.axis("off")
    per_pt = 1.02 * hmax / (fig_h * ax_frac * 72)

    def draw(y, sn, upper, base):
        prev = years[years.index(y) - 1] if y != years[0] else None
        fill = _pale(base) if prev and sn["id"] not in nodes[prev] else base
        lvl, n = sn["level"], sn["member_count"]
        ax.add_patch(Rectangle((bar_x(y, lvl), upper - n), widths[lvl], n, facecolor=fill,
                               edgecolor="white", linewidth=(0.9, 0.5, 0.2)[lvl], zorder=2))
        for ch in children[y][sn["id"]]:
            draw(y, ch, upper, base)
            upper -= ch["member_count"]

    for y in years:
        for i in order[y]:
            upper, lower = span[(y, i)]
            draw(y, nodes[y][i], upper, color.get(i, OTHER))
            if upper - lower + pad > 10 * per_pt:  # chip may spill into the gaps, not onto a neighbour
                ax.text(bar_x(y, 0) + widths[0] / 2, (upper + lower) / 2, _short_id(i), ha="center",
                        va="center", fontsize=6.5, color=TEXT, zorder=3,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
        ax.annotate(str(y), (x0[y] + (right[y] - x0[y]) / 2, 1), xycoords=("data", "axes fraction"),
                    xytext=(0, 26), textcoords="offset points", ha="center", fontsize=11,
                    fontweight="bold", color=TEXT)
        ax.annotate(f"{len(set().union(*top[y].values())):,} nodes", (x0[y] + (right[y] - x0[y]) / 2, 1),
                    xycoords=("data", "axes fraction"), xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=8.5, color=MUTED)
        for lvl in range(3):
            ax.annotate(f"L{lvl}", (bar_x(y, lvl) + widths[lvl] / 2, 1), xycoords=("data", "axes fraction"),
                        xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7, color=MUTED)

    for a, b in zip(years, years[1:]):
        rank_a = {i: k for k, i in enumerate(order[a])}
        rank_b = {j: k for k, j in enumerate(order[b])}
        out_top = {i: span[(a, i)][0] for i in order[a]}
        in_top = {j: span[(b, j)][0] for j in order[b]}
        src, dst = {}, {}
        for (i, j), n in sorted(flows[a].items(), key=lambda f: (rank_a[f[0][0]], rank_b[f[0][1]])):
            src[(i, j)] = (out_top[i], out_top[i] - n)
            out_top[i] -= n
        for (i, j), n in sorted(flows[a].items(), key=lambda f: (rank_b[f[0][1]], rank_a[f[0][0]])):
            dst[(i, j)] = (in_top[j], in_top[j] - n)
            in_top[j] -= n
        for i, j in flows[a]:
            _ribbon(ax, right[a], x0[b], src[(i, j)], dst[(i, j)], color.get(i, OTHER),
                    0.5 if i == j else 0.18)

    def side_labels(y, side):
        ids = order[y]
        centres = [sum(span[(y, i)]) / 2 for i in ids]
        placed = _spread(centres, 11 * per_pt, 0, hmax)
        edge = x0[y] if side == "left" else right[y]
        sign = -1 if side == "left" else 1
        for i, yc, yl in zip(ids, centres, placed):
            text = f"{_short_id(i)}  {nodes[y][i]['label']}"
            text = text if len(text) <= 52 else text[:51].rstrip() + "…"
            ax.plot([edge, edge + sign * 0.06, edge + sign * 0.16], [yc, yc, yl],
                    color=MUTED, linewidth=0.6, zorder=0)
            ax.text(edge + sign * 0.2, yl, text, ha="right" if side == "left" else "left",
                    va="center", fontsize=7.5, color=TEXT)

    side_labels(years[0], "left")
    side_labels(years[-1], "right")

    between = [i for i in sorted(alive) if i not in top[years[0]] and i not in top[years[-1]]]
    notes = []
    for i in between:
        seen = [y for y in years if i in top[y]]
        when = str(seen[0]) if len(seen) == 1 else f"{seen[0]}–{seen[-1]}"
        notes.append(f"{_short_id(i)} {nodes[seen[-1]][i]['label']} ({when})")

    legend_style = dict(loc="lower left", frameon=False, fontsize=8.5, title_fontsize=8.5, alignment="left",
                        handlelength=1.4, columnspacing=1.6)
    fig.legend([Patch(fc=BLUE), Patch(fc=_pale(BLUE)), Patch(fc=OTHER)],
               ["id carried over from the previous snapshot", "id born at this snapshot",
                "level-0 id alive at only one or two snapshots"],
               title="blocks (any level)", bbox_to_anchor=(0.012, 0.068), ncol=3, **legend_style)
    fig.legend([Patch(fc=BLUE, alpha=0.5), Patch(fc=BLUE, alpha=0.18)],
               ["staying in the same level-0 id", "moving to another level-0 id"],
               title="ribbons (level-0 members passed to the next snapshot)", bbox_to_anchor=(0.6, 0.068),
               ncol=2,
               **legend_style)
    fig.text(0.012, 0.985, "Levels 0–2 of the hierarchy across the four snapshots", fontsize=12.5,
             color=TEXT, va="top")
    fig.text(0.012, 0.02,
             "Each column is one snapshot: level 0 (12 super-nodes), level 1 (50) and level 2 (200) side by side, "
             "every child drawn inside its parent's span. Height is node count, on one scale for all four "
             "snapshots (nodes only accumulate).\nRibbons carry each snapshot's level-0 members into the next, "
             "and the rest of each block is new nodes. Numbers are level-0 ids (L0_S000nn) and labels are that "
             "snapshot's; nothing in 2020 counts as born.\nLevel-0 ids alive only in between: "
             + "; ".join(notes) + ".",
             fontsize=8, color=MUTED, va="bottom", linespacing=1.5)
    _save(fig, "hierarchy_across_snapshots")


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
    fig_hierarchy()
    print(f"wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
