"""Lighting gap metrics: shadow analysis on the table.

The table is where the robot and cube cast their shadows, so we look at the
brightness (HSV value) of the table pixels only, using each image's own
label map. A pixel is "in shadow" if it is darker than SHADOW_RATIO times the
median table brightness.
"""
import cv2
import numpy as np


TABLE = 3
SHADOW_RATIO = 0.7


def table_lighting(img, label, ratio=SHADOW_RATIO):
    """Brightness statistics of the table region, or None if the table is barely visible."""
    table = label == TABLE
    if table.sum() < 500:
        return None
    v = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 2][table].astype(np.float32)
    median = np.median(v)
    shadow = v < ratio * median
    return {
        "brightness": median / 255,
        "shadow_frac": shadow.mean(),
        # how dark the shadows are compared to the lit table (1 = no shadows)
        "shadow_depth": np.median(v[shadow]) / median if shadow.any() else 1.0,
        # how uneven the light is across the table (coefficient of variation)
        "nonuniformity": v.std() / (v.mean() + 1e-6),
    }


def lighting_gap(real, sim, real_label, sim_label, ratio=SHADOW_RATIO):
    """table_brightness, shadow_frac, shadow_depth, light_nonuniformity:
    absolute real - sim difference of each table statistic (NaN if no table)
    """
    keys = {"brightness": "table_brightness", "shadow_frac": "shadow_frac",
            "shadow_depth": "shadow_depth", "nonuniformity": "light_nonuniformity"}
    f_real, f_sim = table_lighting(real, real_label, ratio), table_lighting(sim, sim_label, ratio)
    if f_real is None or f_sim is None:
        return {name: np.nan for name in keys.values()}
    return {name: abs(f_real[k] - f_sim[k]) for k, name in keys.items()}
