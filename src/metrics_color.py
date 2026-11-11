"""Color / intensity gap metrics: histogram comparison and CDF comparison.

Every function takes two RGB uint8 images and returns a dict of scores,
so results from different metric modules can be merged into one row.

Run directly to check on a few baseline pairs
"""
import cv2
import numpy as np

from pairs_check import CAMERAS, DATA_DIR, get_pair_paths, iter_pairs, load_image


CHANNELS = ["r", "g", "b"]


def channel_hist(channel, bins=64):
    """Normalized histogram (sums to 1) of one uint8 channel, float32 for cv2.compareHist."""
    hist = cv2.calcHist([channel], [0], None, [bins], [0, 256])
    return (hist / hist.sum()).astype(np.float32)


def histogram_gap(real, sim, bins=64):
    """Compare per-channel RGB histograms, averaged over channels.

    hist_corr:  correlation, 1 = identical, lower = more different
    hist_chi2:  symmetric chi-square distance, 0 = identical
    hist_bhatt: Bhattacharyya distance, 0 = identical, 1 = no overlap
    """
    corr, chi2, bhatt = [], [], []
    for c in range(3):
        h_real = channel_hist(real[:, :, c], bins)
        h_sim = channel_hist(sim[:, :, c], bins)
        corr.append(cv2.compareHist(h_real, h_sim, cv2.HISTCMP_CORREL))
        chi2.append(cv2.compareHist(h_real, h_sim, cv2.HISTCMP_CHISQR_ALT))
        bhatt.append(cv2.compareHist(h_real, h_sim, cv2.HISTCMP_BHATTACHARYYA))
    return {"hist_corr": np.mean(corr), "hist_chi2": np.mean(chi2), "hist_bhatt": np.mean(bhatt)}


def channel_cdf(channel):
    """CDF over the 256 intensity levels of one uint8 channel."""
    hist = np.bincount(channel.ravel(), minlength=256)
    return np.cumsum(hist) / hist.sum()


def cdf_gap(real, sim):
    """Compare CDFs of each RGB channel and of grayscale intensity.

    cdf_ks_<ch>:  max vertical distance between CDFs (KS statistic), 0..1
    cdf_emd_<ch>: area between CDFs = 1D earth mover's distance, scaled to 0..1
                  (0.1 roughly means intensities are shifted by ~25 levels)
    """
    real_chs = [real[:, :, c] for c in range(3)] + [cv2.cvtColor(real, cv2.COLOR_RGB2GRAY)]
    sim_chs = [sim[:, :, c] for c in range(3)] + [cv2.cvtColor(sim, cv2.COLOR_RGB2GRAY)]

    scores = {}
    for name, ch_real, ch_sim in zip(CHANNELS + ["gray"], real_chs, sim_chs):
        diff = np.abs(channel_cdf(ch_real) - channel_cdf(ch_sim))
        scores[f"cdf_ks_{name}"] = diff.max()
        scores[f"cdf_emd_{name}"] = diff.sum() / 255
    return scores


def color_gap(real, sim):
    """All color metrics for one pair."""
    return {**histogram_gap(real, sim), **cdf_gap(real, sim)}


def plot_color_pair(real, sim, out_path, title=""):
    """Save a figure with the two images, their RGB histograms and their CDFs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].imshow(real)
    ax[0, 0].set_title("real")
    ax[0, 1].imshow(sim)
    ax[0, 1].set_title("sim")
    for a in ax[0]:
        a.axis("off")

    levels = np.arange(256)
    for c, color in enumerate(["red", "green", "blue"]):
        for img, style, label in [(real, "-", "real"), (sim, "--", "sim")]:
            hist = np.bincount(img[:, :, c].ravel(), minlength=256) / img[:, :, c].size
            ax[1, 0].plot(levels, hist, style, color=color, label=f"{label} {color[0]}")
            ax[1, 1].plot(levels, channel_cdf(img[:, :, c]), style, color=color, label=f"{label} {color[0]}")

    ax[1, 0].set_title("histogram (solid = real, dashed = sim)")
    ax[1, 0].set_xlabel("intensity")
    ax[1, 0].set_ylabel("fraction of pixels")
    ax[1, 1].set_title("CDF (solid = real, dashed = sim)")
    ax[1, 1].set_xlabel("intensity")
    ax[1, 1].set_ylabel("fraction of pixels <= intensity")
    ax[1, 1].legend(fontsize=8, ncol=2)

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def print_table(title, rows_by_camera):
    keys = list(next(iter(rows_by_camera.values()))[0].keys())
    print(f"\n{title}")
    print(f"{'camera':16s}" + "".join(f"{k:>14s}" for k in keys))
    for camera, rows in rows_by_camera.items():
        means = [np.mean([r[k] for r in rows]) for k in keys]
        print(f"{camera:16s}" + "".join(f"{m:14.3f}" for m in means))


if __name__ == "__main__":
    # real vs sim: should be far apart
    real_vs_sim = {camera: [] for camera in CAMERAS}
    for camera, frame, real, sim in iter_pairs("sim_baseline", step=5):
        real_vs_sim[camera].append(color_gap(real, sim))
    print_table("real vs sim_baseline (every 5th frame)", real_vs_sim)

    # real frame n vs real frame n+1: should be close to identical
    real_vs_real = {camera: [] for camera in CAMERAS}
    for camera in CAMERAS:
        real_paths = [r for r, _ in get_pair_paths("sim_baseline", camera)]
        for a, b in list(zip(real_paths, real_paths[1:]))[::5]:
            real_vs_real[camera].append(color_gap(load_image(a), load_image(b)))
    print_table("sanity check: real frame n vs real frame n+1", real_vs_real)

    # save histogram / CDF plots for a couple of frames per camera
    out_dir = DATA_DIR.parent / "outputs" / "color" / "sim_baseline"
    for camera, frame, real, sim in iter_pairs("sim_baseline"):
        if frame in ("frame_00000", "frame_00040"):
            plot_color_pair(real, sim, out_dir / f"{camera}_{frame}.png", f"sim_baseline  {camera}  {frame}")
    print(f"\nplots saved to {out_dir}")
