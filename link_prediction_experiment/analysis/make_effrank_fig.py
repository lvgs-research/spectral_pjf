"""Effective-rank-by-stage grouped bars (Tech | Tianchi, HeteroSAGE).

Run:  python -m link_prediction_experiment.analysis.make_effrank_fig
Writes figures/effrank_stages.pdf (+ effrank_stages_preview.png).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUITE = Path(__file__).resolve().parents[2]
DATA = SUITE / "tables" / "data"
FIG_DIR = SUITE / "figures"

ARMS = ["p1", "p2", "p3", "p8"]
ARM_LABELS = {
    "p1": "content MLP",
    "p2": "content+graph (naive)",
    "p3": "graph only",
    "p8": "content+graph (decoupled)",
}
# categorical palette, fixed order
ARM_COLORS = {"p1": "#2a78d6", "p2": "#1baf7a", "p3": "#eda100", "p8": "#008300"}

STAGES = ["raw", "pre-decoder", "seeker->job", "job->seeker", "joint head"]
STAGE_LABELS = ["raw\ninput", "pre-\ndecoder",
                "seeker→job\npre-logit", "job→seeker\npre-logit",
                "joint\nhead"]
# capacity key per stage
STAGE_CAP_KEY = {"pre-decoder": "joint_predec", "seeker->job": "seeker2job_accept",
                 "job->seeker": "job2seeker_pass", "joint head": "joint_prod"}
PANELS = [("tech", "Tech"), ("tianchi", "Tianchi")]   # (data key, panel title)

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#d9d8d4"


def build_chart_data():
    """{ds: {arm: {stage: (mean, popSD)}}} from the two roy JSONs."""
    spectral = json.load(open(DATA / "effrank_spectral_roy.json"))
    raw = json.load(open(DATA / "effrank_raw_unique_roy.json"))
    data = {}
    for ds, _ in PANELS:
        data[ds] = {}
        for arm in ARMS:
            per_stage = {"raw": (float(raw[f"{ds}__{arm}"]["raw_edge"]["eff"]), 0.0)}
            recs = [r for r in spectral["records"]
                    if r["dataset"] == ds and r["experiment"] == arm and r["conv"] == "sage"
                    and r.get("encoder", "qwen3-0.6b") == "qwen3-0.6b"]
            for stage, cap_key in STAGE_CAP_KEY.items():
                vals = np.array([r["capacity"][cap_key]["eff"] for r in recs], dtype=np.float64)
                if vals.size == 0:
                    raise SystemExit(f"no sage records for {ds}/{arm} in {DATA / 'effrank_spectral_roy.json'}")
                per_stage[stage] = (float(vals.mean()), float(np.std(vals)))   # population SD (ddof=0)
            data[ds][arm] = per_stage
    return data


def main():
    data = build_chart_data()

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 7.0,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.8,
        "text.color": TEXT_PRIMARY,
        "axes.edgecolor": TEXT_SECONDARY,
        "axes.labelcolor": TEXT_PRIMARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.55), sharey=True)

    n_arms = len(ARMS)
    n_stages = len(STAGES)
    group_w = 0.78
    bw = group_w / n_arms
    x = np.arange(n_stages)

    for ax, (ds, title) in zip(axes, PANELS):
        for i, arm in enumerate(ARMS):
            means = np.array([data[ds][arm][s][0] for s in STAGES])
            sds = np.array([data[ds][arm][s][1] for s in STAGES])
            xpos = x - group_w / 2 + (i + 0.5) * bw
            ax.bar(xpos, means, width=bw * 0.92, color=ARM_COLORS[arm],
                   label=ARM_LABELS[arm] if ds == "tech" else None, zorder=3,
                   edgecolor="white", linewidth=0.4)   # surface gap
            has_sd = sds > 0
            if has_sd.any():
                ax.errorbar(xpos[has_sd], means[has_sd], yerr=sds[has_sd], fmt="none",
                            ecolor=TEXT_SECONDARY, elinewidth=0.6, capsize=1.2,
                            capthick=0.6, zorder=4)
            # value label on every bar
            for xp, m, sd in zip(xpos, means, sds):
                top = m + sd if sd > 0 else m
                txt = f"{m:.0f}" if m >= 100 else f"{m:.1f}"
                ax.annotate(txt, (xp, top), xytext=(0, 1.2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=4.8, color=TEXT_SECONDARY,
                            rotation=90, zorder=5)
        ax.set_yscale("log")
        ax.set_ylim(1, 4000)
        ax.set_title(title, pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels(STAGE_LABELS)
        ax.yaxis.grid(True, which="major", color=GRID, linewidth=0.5, zorder=0)
        ax.yaxis.grid(True, which="minor", color=GRID, linewidth=0.25, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=2, width=0.5)

    axes[0].set_ylabel("Effective rank (log scale)")
    fig.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.01),
               handlelength=1.0, handleheight=0.8, columnspacing=1.2)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.93))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "effrank_stages.pdf")
    fig.savefig(FIG_DIR / "effrank_stages_preview.png", dpi=220)
    print("wrote", FIG_DIR / "effrank_stages.pdf")


if __name__ == "__main__":
    main()
