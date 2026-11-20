"""Experiment 3: histogram-match sim frames to the real camera's color distribution.

For each camera, the reference is the pooled RGB histogram of the real frames
that are NOT hand-labeled (so the real test frames are never looked at). Each
sim frame is then matched to it channel by channel with the classic CDF mapping.
The result is saved as a new condition, data/<condition>_histmatch/, with the
same labels, so compute_gaps.py and segment.py can run on it like any other.

    python src/histmatch.py                       # sim_baseline and sim_combined
    python src/histmatch.py --conditions sim_baseline
"""
import argparse
import shutil

import cv2
import numpy as np

from metrics_color import channel_cdf
from pairs_check import CAMERAS, DATA_DIR, load_image, load_labels


def real_reference_cdfs(camera):
    """Per-channel CDF of all unlabeled real frames of one camera pooled together."""
    labeled = set(load_labels("real", camera))
    paths = [p for p in sorted((DATA_DIR / "real" / "frames" / camera).glob("frame_*.png")) if p.stem not in labeled]
    images = np.stack([load_image(p) for p in paths])
    return [channel_cdf(images[..., c]) for c in range(3)], len(paths)


def match_to_cdfs(img, ref_cdfs):
    """Map each channel so its CDF follows the reference CDF."""
    out = np.empty_like(img)
    levels = np.arange(256)
    for c in range(3):
        lut = np.interp(channel_cdf(img[..., c]), ref_cdfs[c], levels)
        out[..., c] = np.round(lut).astype(np.uint8)[img[..., c]]
    return out


def main():
    """Write a histogram-matched copy of each chosen sim condition, per camera."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--conditions", nargs="+", default=["sim_baseline", "sim_combined"])
    args = p.parse_args()

    refs = {}
    for camera in CAMERAS:
        refs[camera], n = real_reference_cdfs(camera)
        print(f"{camera}: reference from {n} unlabeled real frames")

    for condition in args.conditions:
        src, dst = DATA_DIR / condition, DATA_DIR / f"{condition}_histmatch"
        shutil.rmtree(dst, ignore_errors=True)
        for camera in CAMERAS:
            out_dir = dst / "frames" / camera
            out_dir.mkdir(parents=True)
            for path in sorted((src / "frames" / camera).glob("frame_*.png")):
                matched = match_to_cdfs(load_image(path), refs[camera])
                cv2.imwrite(str(out_dir / path.name), cv2.cvtColor(matched, cv2.COLOR_RGB2BGR))
        shutil.copytree(src / "labels", dst / "labels")
        print(f"wrote {dst}")


if __name__ == "__main__":
    main()
