"""Edge / structure gap metrics: Canny edges and SSIM.

SSIM and the edge F1 compare pixel by pixel, so they also pick up the small
pose misalignment between real and sim (about 10-30 px). Edge density doesn't.
Every function returns a dict of gap scores (higher = more different).
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity


def canny_edges(img, low=50, high=150):
    """Fixed thresholds on purpose, so a less contrasty image really gets fewer edges."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.4), low, high) > 0


def edge_gap(real, sim, tol=3, low=50, high=150):
    """edge_density: |fraction of edge pixels in real - in sim|
    edge_f1:        1 - F1 of edge matching, where an edge counts as matched if
                    the other image has an edge within tol pixels
    """
    e_real, e_sim = canny_edges(real, low, high), canny_edges(sim, low, high)
    # distance from every pixel to the nearest edge pixel of each image
    dist_real = cv2.distanceTransform((~e_real).astype(np.uint8), cv2.DIST_L2, 3)
    dist_sim = cv2.distanceTransform((~e_sim).astype(np.uint8), cv2.DIST_L2, 3)
    precision = (dist_real[e_sim] <= tol).mean() if e_sim.any() else 0.0
    recall = (dist_sim[e_real] <= tol).mean() if e_real.any() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"edge_density": abs(e_real.mean() - e_sim.mean()), "edge_f1": 1 - f1}


def ssim_gap(real, sim, win_size=7):
    """ssim: 1 - SSIM of the gray images (0 = identical), over win_size x win_size windows"""
    gray_real = cv2.cvtColor(real, cv2.COLOR_RGB2GRAY)
    gray_sim = cv2.cvtColor(sim, cv2.COLOR_RGB2GRAY)
    return {"ssim": 1 - structural_similarity(gray_real, gray_sim, win_size=win_size, data_range=255)}


def structure_gap(real, sim):
    """All structure metrics for one pair."""
    return {**edge_gap(real, sim), **ssim_gap(real, sim)}
