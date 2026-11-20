"""Rank the gap metrics by how well they predict real-image segmentation performance.

For every metric: Spearman correlation between its gap score and the real mIoU
of the model trained on that sim condition. A good metric has a strongly
negative rho (bigger gap -> worse segmentation on real).

Two versions:
    by condition:          13 points (gap and mIoU averaged over cameras)
    by condition x camera: 52 points (per-camera gap vs per-camera mIoU)

The *_histmatch conditions (experiment 3) are left out of the ranking and
reported separately. If segment.py --color-aug was run, the color-augmented
models are compared to the plain ones, and the ranking is redone against them.

    python src/correlate.py

Needs outputs/gaps/*.csv (compute_gaps.py) and outputs/segmentation/results.csv (segment.py).
Writes outputs/analysis/spearman.csv, the summary tables and figures.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pairs_check import CAMERAS, DATA_DIR

OUT_DIR = DATA_DIR.parent / "outputs" / "analysis"
GAPS_DIR = DATA_DIR.parent / "outputs" / "gaps"
SEG_CSV = DATA_DIR.parent / "outputs" / "segmentation" / "results.csv"
COLORAUG_CSV = DATA_DIR.parent / "outputs" / "segmentation" / "results_coloraug.csv"

# metrics where higher means MORE similar; flipped so every score is a gap
SIMILARITIES = ["hist_corr"]

FAMILIES = {"hist": "color", "cdf": "color", "glcm": "texture", "gabor": "texture",
            "edge": "structure", "ssim": "structure", "table": "lighting", "shadow": "lighting",
            "light": "lighting"}
FAMILY_COLORS = {"color": "#d62728", "texture": "#2ca02c", "structure": "#1f77b4", "lighting": "#ff7f0e"}


def family(metric):
    """Metric family (color, texture, structure, lighting) from the metric name prefix."""
    return FAMILIES[metric.split("_")[0]]


def load_gaps(name):
    """Read a gap csv and flip the similarity metrics so every score is a gap."""
    df = pd.read_csv(GAPS_DIR / name)
    for m in SIMILARITIES:
        df[m] = -df[m]
    return df


def seg_summary(csv=SEG_CSV):
    """Mean real mIoU (and per class / per camera IoU) over seeds, plus its std, per condition."""
    seg = pd.read_csv(csv)
    cols = seg.columns.drop(["condition", "seed"])
    mean = seg.groupby("condition")[cols].mean()
    mean["miou_std"] = seg.groupby("condition")["miou"].std()
    mean["n_seeds"] = seg.groupby("condition")["seed"].count()
    return mean


def spearman_table(gaps, seg):
    """rho and p for every metric: by condition and by condition x camera."""
    by_cond = gaps["by_condition"].join(seg["miou"], how="inner")

    per_cam = gaps["by_camera"]
    seg_cam = seg[[f"miou_{c}" for c in CAMERAS]].reset_index().melt(
        id_vars="condition", var_name="camera", value_name="miou")
    seg_cam["camera"] = seg_cam["camera"].str.replace("miou_", "")
    per_cam = per_cam.merge(seg_cam, on=["condition", "camera"])

    rows = []
    for m in gaps["metrics"]:
        rho, p = spearmanr(by_cond[m], by_cond["miou"], nan_policy="omit")
        rho_cam, p_cam = spearmanr(per_cam[m], per_cam["miou"], nan_policy="omit")
        rows.append({"metric": m, "family": family(m), "rho": rho, "p": p,
                     "rho_by_camera": rho_cam, "p_by_camera": p_cam})
    table = pd.DataFrame(rows).sort_values("rho").reset_index(drop=True)
    return table, by_cond, per_cam


def plot_bars(table, out_path):
    """Bar chart of Spearman rho per metric, colored by family, with the per-camera rho as dots."""
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(table) + 1.2))
    y = np.arange(len(table))[::-1]
    ax.barh(y, table["rho"], color=[FAMILY_COLORS[f] for f in table["family"]])
    ax.scatter(table["rho_by_camera"], y, color="black", s=12, zorder=3, label="by condition x camera (n=52)")
    ax.set_yticks(y, table["metric"])
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman rho between gap and real mIoU (negative = good predictor)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLORS.values()]
    ax.legend(handles + [ax.collections[0]], list(FAMILY_COLORS) + ["by condition x camera (n=52)"],
              fontsize=8, loc="upper right")
    ax.set_title("Which gap metric predicts sim-to-real segmentation performance?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scatter(table, by_cond, out_path, n=3):
    """Best n metrics and the worst one: gap vs real mIoU, one point per condition."""
    picks = list(table["metric"][:n]) + [table["metric"].iloc[-1]]
    fig, axes = plt.subplots(1, len(picks), figsize=(4.2 * len(picks), 4))
    for ax, m in zip(axes, picks):
        ax.scatter(by_cond[m], by_cond["miou"], color=FAMILY_COLORS[family(m)])
        for cond, row in by_cond.iterrows():
            ax.annotate(cond.replace("sim_", ""), (row[m], row["miou"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        rho = table.set_index("metric").loc[m, "rho"]
        ax.set_title(f"{m}  (rho = {rho:.2f})")
        ax.set_xlabel(f"gap score (-{m}, flipped)" if m in SIMILARITIES else "gap score")
        ax.set_ylabel("real mIoU")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_coloraug(compare, out_path):
    """Real mIoU per condition without and with color augmentation (error bars = std over seeds)."""
    compare = compare.sort_values("miou")
    x = np.arange(len(compare))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, compare["miou"], 0.4, yerr=compare["miou_std"], capsize=2, color="#9aa5b1", label="no color augmentation")
    ax.bar(x + 0.2, compare["miou_coloraug"], 0.4, yerr=compare["miou_std_coloraug"], capsize=2,
           color="#1f77b4", label="with color augmentation")
    ax.set_xticks(x, [c.replace("sim_", "") for c in compare.index], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("real mIoU")
    ax.set_ylim(0, max(compare["miou"].max(), compare["miou_coloraug"].max()) + 0.18)
    ax.legend(fontsize=8, loc="upper center", ncol=2)
    ax.set_title("Does color augmentation close the sim-to-real gap?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    """Rank the metrics, then report experiment 3 (histogram matching) and the color-aug check."""
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()   # no options, just --help
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_condition = load_gaps("by_condition.csv").set_index("condition")
    by_camera = load_gaps("by_camera.csv")
    seg = seg_summary()

    histmatch = [c for c in seg.index if c.endswith("_histmatch")]
    keep = lambda conds: [c for c in conds if not c.endswith("_histmatch")]
    gaps = {"by_condition": by_condition.loc[keep(by_condition.index)],
            "by_camera": by_camera[by_camera["condition"].isin(keep(by_camera["condition"]))],
            "metrics": list(by_condition.columns)}
    seg_main = seg.loc[keep(seg.index)]

    table, by_cond, _ = spearman_table(gaps, seg_main)
    table.to_csv(OUT_DIR / "spearman.csv", index=False)
    plot_bars(table, OUT_DIR / "spearman_bars.png")
    plot_scatter(table, by_cond, OUT_DIR / "top_metrics_scatter.png")

    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 3)
    print(f"segmentation (real mIoU, mean over seeds):\n{seg_main[['miou', 'miou_std', 'n_seeds', 'sim_miou']].sort_values('miou')}\n")
    print(f"Spearman rho, gap vs real mIoU ({len(seg_main)} conditions):\n{table}\n")

    # experiment 3: does histogram matching close the gap?
    if histmatch:
        rows = []
        for cond in histmatch:
            base = cond.removesuffix("_histmatch")
            for c in (base, cond):
                rows.append({"condition": c, "miou": seg.loc[c, "miou"], "miou_std": seg.loc[c, "miou_std"],
                             **by_condition.loc[c, ["cdf_emd_gray", "hist_bhatt", "glcm_dist", "ssim",
                                                    "table_brightness"]].to_dict()})
        correction = pd.DataFrame(rows).set_index("condition")
        correction.to_csv(OUT_DIR / "histmatch.csv")
        print(f"histogram matching correction:\n{correction}\n")

    # extra check: with color augmentation, do the gaps still predict real performance?
    if COLORAUG_CSV.exists():
        aug = seg_summary(COLORAUG_CSV)
        aug = aug.loc[keep(aug.index)]
        compare = seg_main[["miou", "miou_std"]].join(aug[["miou", "miou_std", "n_seeds"]], rsuffix="_coloraug", how="inner")
        compare["change"] = compare["miou_coloraug"] - compare["miou"]
        compare.to_csv(OUT_DIR / "coloraug.csv")
        plot_coloraug(compare, OUT_DIR / "coloraug.png")
        aug_table, _, _ = spearman_table(gaps, aug)
        both = table[["metric", "family", "rho", "p"]].merge(
            aug_table[["metric", "rho", "p"]], on="metric", suffixes=("", "_coloraug"))
        both.to_csv(OUT_DIR / "spearman_coloraug.csv", index=False)
        print(f"color augmentation (real mIoU):\n{compare.sort_values('miou')}\n")
        print(f"Spearman rho without vs with color augmentation ({len(compare)} conditions):\n{both}\n")

    seg.to_csv(OUT_DIR / "segmentation_summary.csv")
    print(f"saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
