"""Check whether the metric ranking depends on the metric settings.

Every metric with a free parameter is recomputed with a few other settings
(histogram bins, Canny thresholds and match tolerance, SSIM window, GLCM gray
levels, Gabor wavelengths, shadow threshold) on the same 64 labeled pairs per
condition, and its Spearman rho against real mIoU is recomputed. If a metric's
rho flips sign or jumps around, its place in the ranking is not trustworthy.

    python src/sensitivity.py

Needs outputs/segmentation/results.csv (segment.py).
Writes outputs/analysis/sensitivity.csv and sensitivity.png.
"""
import argparse
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from correlate import FAMILY_COLORS, OUT_DIR, SIMILARITIES, family, seg_summary
from metrics_color import histogram_gap
from metrics_lighting import SHADOW_RATIO, lighting_gap
from metrics_structure import edge_gap, ssim_gap
from metrics_texture import gabor_bank, gabor_gap, glcm_gap
from pairs_check import CAMERAS, DATA_DIR, load_image, load_labels

GABOR_BANKS = {"2-4-8": gabor_bank((2, 4, 8)), "4-8-16": gabor_bank((4, 8, 16)), "8-16-32": gabor_bank((8, 16, 32))}

# (parameter name, values, default, function(real, sim, real_label, sim_label, value) -> gap dict)
SETTINGS = [
    ("bins", [16, 32, 64, 128, 256], 64, lambda r, s, rl, sl, v: histogram_gap(r, s, bins=v)),
    ("canny", [(25, 75), (50, 150), (100, 200)], (50, 150),
     lambda r, s, rl, sl, v: edge_gap(r, s, low=v[0], high=v[1])),
    ("edge_tol", [1, 3, 5], 3, lambda r, s, rl, sl, v: {"edge_f1": edge_gap(r, s, tol=v)["edge_f1"]}),
    ("ssim_win", [7, 15, 31], 7, lambda r, s, rl, sl, v: ssim_gap(r, s, win_size=v)),
    ("glcm_levels", [16, 32, 64], 32, lambda r, s, rl, sl, v: glcm_gap(r, s, levels=v)),
    ("gabor_wavelengths", list(GABOR_BANKS), "4-8-16", lambda r, s, rl, sl, v: gabor_gap(r, s, GABOR_BANKS[v])),
    ("shadow_ratio", [0.5, 0.6, 0.7, 0.8], SHADOW_RATIO,
     lambda r, s, rl, sl, v: {k: lighting_gap(r, s, rl, sl, ratio=v)[k] for k in ("shadow_frac", "shadow_depth")}),
]


def condition_scores(condition):
    """Mean of every (parameter, value, metric) gap over the labeled pairs of one condition."""
    rows = []
    for camera in CAMERAS:
        real_labels, sim_labels = load_labels("real", camera), load_labels(condition, camera)
        for frame in sorted(real_labels):
            real = load_image(DATA_DIR / "real" / "frames" / camera / f"{frame}.png")
            sim = load_image(DATA_DIR / condition / "frames" / camera / f"{frame}.png")
            for param, values, default, fn in SETTINGS:
                for v in values:
                    for metric, score in fn(real, sim, real_labels[frame], sim_labels[frame], v).items():
                        rows.append({"param": param, "value": str(v), "default": v == default,
                                     "metric": metric, "gap": -score if metric in SIMILARITIES else score})
    df = pd.DataFrame(rows).groupby(["param", "value", "default", "metric"], as_index=False)["gap"].mean()
    df["condition"] = condition
    print(f"done {condition}", flush=True)
    return df


def plot(table, out_path):
    """One row per (parameter, metric): rho for every setting, the default one circled."""
    keys = list(table.groupby(["param", "metric"], sort=False).groups)
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(keys) + 1.2))
    for i, (param, metric) in enumerate(keys):
        y = len(keys) - 1 - i
        t = table[(table["param"] == param) & (table["metric"] == metric)]
        color = FAMILY_COLORS[family(metric)]
        ax.plot([t["rho"].min(), t["rho"].max()], [y, y], color=color, lw=1, alpha=0.5)
        ax.scatter(t["rho"], [y] * len(t), color=color, s=18, zorder=3)
        d = t[t["default"]]
        ax.scatter(d["rho"], [y] * len(d), facecolor="none", edgecolor="black", s=70, zorder=4)
    ax.set_yticks(range(len(keys))[::-1], [f"{m}  ({p})" for p, m in keys], fontsize=8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman rho between gap and real mIoU (negative = good predictor)")
    ax.set_title("Metric settings vs. rho (dots = settings, circle = setting used)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    """Recompute the metrics under every setting, then rho against real mIoU for each."""
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()   # no options, just --help
    seg = seg_summary()
    conditions = [c for c in seg.index if not c.endswith("_histmatch")]
    with Pool(4) as pool:
        gaps = pd.concat(pool.map(condition_scores, conditions), ignore_index=True)

    rows = []
    for (param, value, default, metric), g in gaps.groupby(["param", "value", "default", "metric"], sort=False):
        rho, p = spearmanr(g["gap"], seg.loc[g["condition"], "miou"], nan_policy="omit")
        rows.append({"param": param, "value": value, "default": default, "metric": metric, "rho": rho, "p": p})
    order = {param: i for i, (param, *_) in enumerate(SETTINGS)}
    table = pd.DataFrame(rows).sort_values(["param", "metric"], key=lambda c: c.map(order) if c.name == "param" else c)
    table.to_csv(OUT_DIR / "sensitivity.csv", index=False)
    plot(table, OUT_DIR / "sensitivity.png")

    spread = table.groupby(["param", "metric"])["rho"].agg(["min", "max"])
    spread["used"] = table[table["default"]].set_index(["param", "metric"])["rho"]
    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 3)
    print(f"\nrho range over settings:\n{spread}\n\nsaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
