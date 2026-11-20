"""Compute every gap metric for every sim condition on the labeled real frames.

Uses the 16 labeled frames per camera (every 5th frame, 64 pairs per condition),
because the shadow metric needs the table masks.

    python src/compute_gaps.py                       # all conditions in data/
    python src/compute_gaps.py --only sim_baseline

Writes
    outputs/gaps/per_frame.csv     one row per (condition, camera, frame)
    outputs/gaps/by_camera.csv     mean per (condition, camera)
    outputs/gaps/by_condition.csv  mean per condition
"""
import argparse

import pandas as pd
from tqdm import tqdm

from metrics_color import color_gap
from metrics_lighting import lighting_gap
from metrics_structure import structure_gap
from metrics_texture import texture_gap
from pairs_check import CAMERAS, DATA_DIR, list_conditions, load_image, load_labels

OUT_DIR = DATA_DIR.parent / "outputs" / "gaps"


def all_gaps(real, sim, real_label, sim_label):
    """Every gap metric (color, texture, structure, lighting) for one real/sim pair."""
    return {**color_gap(real, sim), **texture_gap(real, sim), **structure_gap(real, sim),
            **lighting_gap(real, sim, real_label, sim_label)}


def condition_rows(condition):
    """One row of gap scores per labeled frame of every camera, for one sim condition."""
    rows = []
    for camera in CAMERAS:
        real_labels, sim_labels = load_labels("real", camera), load_labels(condition, camera)
        for frame in tqdm(sorted(real_labels), desc=f"{condition} {camera}", leave=False):
            real = load_image(DATA_DIR / "real" / "frames" / camera / f"{frame}.png")
            sim = load_image(DATA_DIR / condition / "frames" / camera / f"{frame}.png")
            gaps = all_gaps(real, sim, real_labels[frame], sim_labels[frame])
            rows.append({"condition": condition, "camera": camera, "frame": frame, **gaps})
    return rows


def main():
    """Compute the gaps for the chosen conditions and write the per-frame/camera/condition csvs."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", nargs="+", help="conditions to (re)compute, others are kept from the old csv")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_frame_csv = OUT_DIR / "per_frame.csv"
    conditions = args.only or list_conditions()

    old = pd.read_csv(per_frame_csv) if args.only and per_frame_csv.exists() else None
    rows = []
    for condition in conditions:
        rows += condition_rows(condition)
        print(f"done {condition}")
    df = pd.DataFrame(rows)
    if old is not None:
        df = pd.concat([old[~old["condition"].isin(conditions)], df], ignore_index=True)

    df.to_csv(per_frame_csv, index=False)
    metrics = df.columns.drop(["condition", "camera", "frame"])
    df.groupby(["condition", "camera"])[metrics].mean().to_csv(OUT_DIR / "by_camera.csv")
    by_condition = df.groupby("condition")[metrics].mean()
    by_condition.to_csv(OUT_DIR / "by_condition.csv")

    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.precision", 3):
        print(by_condition)
    print(f"\nsaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
