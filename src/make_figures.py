"""Figures for the presentation and report (saved to outputs/figures/).

    python src/make_figures.py            # everything (retrains 2 models for the prediction examples, ~5 min)
    python src/make_figures.py --no-seg   # skip the segmentation examples

Needs the outputs of compute_gaps.py, segment.py and correlate.py for the result plots.
"""
import argparse

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from label_real import CLASS_COLORS
from metrics_color import channel_cdf, channel_hist
from metrics_lighting import SHADOW_RATIO, TABLE
from metrics_structure import canny_edges
from metrics_texture import GABOR_BANK
from pairs_check import CAMERAS, DATA_DIR, load_image, load_labels

OUT_DIR = DATA_DIR.parent / "outputs" / "figures"
FRAME = "frame_00040"
CLASS_NAMES = ["background", "cube", "robot", "table", "frame"]
VARIATIONS = ["sim_baseline", "sim_neutral_light", "sim_dim_light", "sim_bright_light", "sim_cool_light",
              "sim_side_light", "sim_two_lights", "sim_shadows", "sim_gray_background",
              "sim_table_matched_color", "sim_blur", "sim_noise", "sim_combined"]


def frame(source, camera, name=FRAME):
    """One frame of a data source (real or a sim condition) as RGB."""
    return load_image(DATA_DIR / source / "frames" / camera / f"{name}.png")


def save(fig, name):
    """Save a figure to outputs/figures/ and close it."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


def grid(images, titles, ncols, name, cell=3.0):
    """Save a grid of images with titles."""
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell * ncols, cell * 0.9 * nrows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, img, title in zip(np.ravel(axes), images, titles):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    save(fig, name)


def class_legend(fig):
    """Add the 5 class colors as a legend under a figure."""
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[i] / 255) for i in range(5)]
    fig.legend(handles, CLASS_NAMES, loc="lower center", ncol=5, frameon=False, fontsize=10)


def fig_pairs():
    """Real vs sim baseline for all 4 cameras."""
    images = [frame("real", c) for c in CAMERAS] + [frame("sim_baseline", c) for c in CAMERAS]
    titles = [f"real  {c}" for c in CAMERAS] + [f"sim  {c}" for c in CAMERAS]
    grid(images, titles, 4, "pairs_real_vs_sim.png")


def fig_variations(camera="cam_high"):
    """Real frame next to the same frame in all 13 sim variations."""
    images = [frame("real", camera)] + [frame(v, camera) for v in VARIATIONS]
    titles = ["REAL"] + [v.replace("sim_", "") for v in VARIATIONS]
    grid(images, titles, 7, "variations.png", cell=2.6)


def fig_labels(camera="cam_right_wrist"):
    """Input image and its class map, real (hand labeled) and sim (from the renderer)."""
    images = [frame("real", camera), CLASS_COLORS[load_labels("real", camera)[FRAME]],
              frame("sim_baseline", camera), CLASS_COLORS[load_labels("sim_baseline", camera)[FRAME]]]
    titles = ["real image", "real label (SAM click tool)", "sim image", "sim label (rendered by Drake)"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    class_legend(fig)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "labels.png")


def gabor_energy(img):
    """Summed Gabor magnitude of the 4 orientations at 8 px wavelength (for display)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    total = np.zeros_like(gray)
    for even, odd in GABOR_BANK[4:8]:   # wavelength 8 px, 4 orientations
        total += cv2.magnitude(cv2.filter2D(gray, cv2.CV_32F, even), cv2.filter2D(gray, cv2.CV_32F, odd))
    return total


def shadow_overlay(img, label):
    """Table in blue, table pixels counted as shadow in magenta."""
    table = label == TABLE
    v = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 2].astype(np.float32)
    shadow = table & (v < SHADOW_RATIO * np.median(v[table]))
    out = (img * 0.5).astype(np.uint8)
    out[table & ~shadow] = (0.5 * img[table & ~shadow] + [0, 60, 120]).clip(0, 255)
    out[shadow] = [255, 0, 255]
    return out


def fig_metric_steps(camera="cam_high", sim="sim_shadows"):
    """Intermediate outputs of the metric stages for one real/sim pair."""
    real, simg = frame("real", camera), frame(sim, camera)
    rl, sl = load_labels("real", camera)[FRAME], load_labels(sim, camera)[FRAME]
    g_real, g_sim = gabor_energy(real), gabor_energy(simg)
    vmax = np.percentile(np.concatenate([g_real.ravel(), g_sim.ravel()]), 99)
    rows = [("real", real, rl, g_real), (sim.replace("sim_", "sim "), simg, sl, g_sim)]
    fig, axes = plt.subplots(2, 4, figsize=(15, 5.6))
    for r, (name, img, lab, g) in enumerate(rows):
        panels = [(img, None, f"{name}: input"), (canny_edges(img), "gray", "Canny edges"),
                  (g, "magma", "Gabor energy (8 px)"), (shadow_overlay(img, lab), None, "table (blue) / shadow (magenta)")]
        for c, (p, cmap, title) in enumerate(panels):
            axes[r, c].imshow(p, cmap=cmap, vmin=0 if cmap == "magma" else None, vmax=vmax if cmap == "magma" else None)
            axes[r, c].set_title(title, fontsize=10)
            axes[r, c].axis("off")
    fig.tight_layout()
    save(fig, "metric_steps.png")


def fig_cdf(camera="cam_high"):
    """Blue-channel and gray CDFs: real vs the best and worst sim."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    sources = [("real", "black", "-"), ("sim_baseline", "tab:orange", "--"),
               ("sim_bright_light", "tab:red", "--"), ("sim_combined", "tab:green", "--")]
    for ax, (ch, title) in zip(axes, [(2, "blue channel CDF"), (None, "gray CDF")]):
        for src, color, style in sources:
            img = frame(src, camera)
            x = img[:, :, ch] if ch is not None else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ax.plot(channel_cdf(x), style, color=color, label=src.replace("sim_", "sim "))
        ax.set_title(title)
        ax.set_xlabel("intensity")
        ax.set_ylabel("fraction of pixels <= intensity")
    axes[0].legend(fontsize=9)
    axes[0].text(0.55, 0.12, "EMD = area between curves", transform=axes[0].transAxes, fontsize=9)
    fig.tight_layout()
    save(fig, "cdf_example.png")


def fig_histograms(camera="cam_high", sims=("sim_baseline", "sim_combined", "sim_bright_light")):
    """RGB histograms (64 bins, as in the color metrics): real alone, then real vs each sim with its scores."""
    real = frame("real", camera)
    rows = [("real", real)] + [(s, frame(s, camera)) for s in sims]
    centers = (np.arange(64) + 0.5) * 4
    fig, axes = plt.subplots(len(rows), 4, figsize=(16, 3.1 * len(rows)),
                             gridspec_kw={"width_ratios": [1.1, 1, 1, 1]})
    for r, (name, img) in enumerate(rows):
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(name.replace("sim_", "sim "), fontsize=11)
        axes[r, 0].axis("off")
        for c, color in enumerate(["red", "green", "blue"]):
            ax = axes[r, c + 1]
            ax.fill_between(centers, channel_hist(real[:, :, c]).ravel(), step="mid", color="0.75",
                            label="real")
            if name != "real":
                ax.step(centers, channel_hist(img[:, :, c]).ravel(), where="mid", color=color, lw=1.6,
                        label=name.replace("sim_", "sim "))
                h_real, h_sim = channel_hist(real[:, :, c]), channel_hist(img[:, :, c])
                scores = (f"corr {cv2.compareHist(h_real, h_sim, cv2.HISTCMP_CORREL):.2f}\n"
                          f"chi2 {cv2.compareHist(h_real, h_sim, cv2.HISTCMP_CHISQR_ALT):.2f}\n"
                          f"Bhatt {cv2.compareHist(h_real, h_sim, cv2.HISTCMP_BHATTACHARYYA):.2f}")
                ax.text(0.03, 0.95, scores, transform=ax.transAxes, va="top", fontsize=9,
                        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"})
            ax.set_xlim(0, 256)
            ax.set_yscale("log")
            ax.set_ylim(1e-4, 1)
            if r == 0:
                ax.set_title(f"{color} channel", fontsize=11)
            if r == len(rows) - 1:
                ax.set_xlabel("intensity")
            if c == 0:
                ax.set_ylabel("fraction of pixels (log)")
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"RGB histograms, 64 bins ({camera}, {FRAME}); gray = real", fontsize=12)
    fig.tight_layout()
    save(fig, "histograms_rgb.png")


def fig_histmatch_steps(camera="cam_high", condition="sim_baseline", c=2):
    """How histogram matching works on one channel: histograms, CDFs, the mapping, the result."""
    from histmatch import match_to_cdfs, real_reference_cdfs
    refs, _ = real_reference_cdfs(camera)
    sim = frame(condition, camera)
    matched = match_to_cdfs(sim, refs)
    levels = np.arange(256)
    sim_cdf = channel_cdf(sim[:, :, c])
    lut = np.round(np.interp(sim_cdf, refs[c], levels))
    name = ["red", "green", "blue"][c]
    centers = (np.arange(64) + 0.5) * 4
    ref_hist = np.diff(np.concatenate([[0], refs[c]]))
    ref_hist64 = ref_hist.reshape(64, 4).sum(1)

    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
    ax = axes[0]
    ax.fill_between(centers, ref_hist64, step="mid", color="0.75", label="real (reference)")
    ax.step(centers, channel_hist(sim[:, :, c]).ravel(), where="mid", color="tab:orange", label="sim")
    ax.step(centers, channel_hist(matched[:, :, c]).ravel(), where="mid", color="tab:blue", label="sim matched")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.set_title(f"1. {name} histograms")
    ax.set_xlabel("intensity")
    ax.set_ylabel("fraction of pixels (log)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(levels, refs[c], color="black", label="real CDF")
    ax.plot(levels, sim_cdf, color="tab:orange", label="sim CDF")
    ax.plot(levels, channel_cdf(matched[:, :, c]), "--", color="tab:blue", label="matched CDF")
    v = 160   # one level, to show how it is mapped
    target = int(lut[v])
    ax.annotate("", xy=(target, sim_cdf[v]), xytext=(v, sim_cdf[v]),
                arrowprops={"arrowstyle": "->", "color": "tab:red", "lw": 1.5})
    ax.plot([v, target], [sim_cdf[v]] * 2, "o", color="tab:red", ms=4)
    ax.text((v + target) / 2, sim_cdf[v] - 0.08, f"{v} -> {target}", color="tab:red", fontsize=9, ha="center")
    ax.set_title("2. CDFs: same percentile -> new value")
    ax.set_xlabel("intensity")
    ax.set_ylabel("fraction of pixels <= intensity")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[2]
    ax.plot(levels, lut, color="tab:red")
    ax.plot(levels, levels, ":", color="0.6", label="no change")
    ax.set_title("3. mapping (lookup table)")
    ax.set_xlabel("sim intensity")
    ax.set_ylabel("new intensity")
    ax.legend(fontsize=8)

    ax = axes[3]
    ax.imshow(np.hstack([sim, matched]))
    ax.set_title("4. sim before | after")
    ax.axis("off")
    fig.suptitle(f"Histogram matching, step by step ({condition}, {camera}, {name} channel)", fontsize=12)
    fig.tight_layout()
    save(fig, "histmatch_steps.png")


def fig_histmatch(camera="cam_high"):
    """Real, sim and histogram-matched sim side by side (experiment 3 inputs/outputs)."""
    images = [frame("real", camera), frame("sim_baseline", camera), frame("sim_baseline_histmatch", camera),
              frame("sim_combined", camera), frame("sim_combined_histmatch", camera)]
    titles = ["real", "sim baseline", "baseline + hist. match", "sim combined", "combined + hist. match"]
    grid(images, titles, 5, "histmatch.png")


def fig_segmentation():
    """Train the best and worst condition once and show real predictions (best and worst frames)."""
    import torch
    from segment import iou_per_class, load_set, predict, train

    real_images, real_labels, real_cams = load_set("real")
    names = [f for c in CAMERAS for f in sorted(load_labels("real", c))]
    for condition in ["sim_combined", "sim_bright_light"]:
        torch.manual_seed(0)
        images, labels, _ = load_set(condition)
        model = train(images, labels, "cuda", epochs=15, batch_size=8, lr=1e-3)
        pred = predict(model, real_images, "cuda")
        scores = np.array([np.nanmean(iou_per_class(p, l)) for p, l in zip(pred, real_labels)])
        del model
        torch.cuda.empty_cache()

        picks = list(np.argsort(scores)[::-1][:2]) + list(np.argsort(scores)[:2])
        fig, axes = plt.subplots(3, 4, figsize=(14, 8))
        for col, i in enumerate(picks):
            kind = "success" if col < 2 else "failure"
            axes[0, col].imshow(real_images[i])
            axes[0, col].set_title(f"{kind}: {real_cams[i]} {names[i][-5:]}  mIoU {scores[i]:.2f}", fontsize=9)
            axes[1, col].imshow(CLASS_COLORS[real_labels[i]])
            axes[2, col].imshow(CLASS_COLORS[pred[i]])
        for ax in axes.ravel():
            ax.axis("off")
        axes[1, 0].text(-0.05, 0.5, "ground truth", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="right")
        axes[2, 0].text(-0.05, 0.5, "prediction", transform=axes[2, 0].transAxes, rotation=90, va="center", ha="right")
        fig.suptitle(f"model trained on {condition}, tested on real", fontsize=12)
        class_legend(fig)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        save(fig, f"seg_{condition}.png")


def fig_seg_bars():
    """Real mIoU per condition (bars) with the sim mIoU of the same models (ticks)."""
    import pandas as pd
    seg = pd.read_csv(DATA_DIR.parent / "outputs" / "analysis" / "segmentation_summary.csv", index_col=0)
    seg = seg.loc[[c for c in seg.index if not c.endswith("_histmatch")]].sort_values("miou")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(seg))
    ax.barh(y, seg["miou"], xerr=seg["miou_std"], color="tab:blue", label="tested on real")
    ax.scatter(seg["sim_miou"], y, color="black", marker="|", s=200, label="tested on sim")
    ax.set_yticks(y, [c.replace("sim_", "") for c in seg.index])
    ax.set_xlim(0, 1)
    ax.set_xlabel("mIoU (mean ± std over 3 seeds)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9, frameon=False)
    ax.set_title("Segmentation trained on each sim variation")
    fig.tight_layout()
    save(fig, "seg_by_condition.png")


def main():
    """Make every figure (the segmentation examples retrain 2 models unless --no-seg)."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--no-seg", action="store_true")
    args = p.parse_args()
    fig_pairs()
    fig_variations()
    fig_labels()
    fig_metric_steps()
    fig_cdf()
    fig_histograms()
    fig_histmatch_steps()
    fig_histmatch()
    fig_seg_bars()
    if not args.no_seg:
        fig_segmentation()


if __name__ == "__main__":
    main()
