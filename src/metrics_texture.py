"""Texture gap metrics: GLCM statistics and Gabor filter responses.

Both summarize the whole image, so unlike SSIM they don't need the real and
sim frames to be pixel aligned. Every function returns a dict of gap scores
(higher = more different).
"""
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


GLCM_PROPS = ["contrast", "homogeneity", "energy", "correlation"]


def glcm_features(img, levels=32):
    """GLCM properties of the gray image, averaged over 2 distances and 4 angles."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    quantized = (gray // (256 // levels)).astype(np.uint8)
    glcm = graycomatrix(quantized, distances=[1, 4], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=levels, symmetric=True, normed=True)
    return {p: graycoprops(glcm, p).mean() for p in GLCM_PROPS}


def glcm_gap(real, sim, levels=32):
    """glcm_<prop>: absolute difference of each GLCM property
    glcm_dist:     mean relative difference over the 4 properties (one combined number)
    """
    f_real, f_sim = glcm_features(real, levels), glcm_features(sim, levels)
    scores = {f"glcm_{p}": abs(f_real[p] - f_sim[p]) for p in GLCM_PROPS}
    scores["glcm_dist"] = np.mean([abs(f_real[p] - f_sim[p]) / (abs(f_real[p]) + 1e-6) for p in GLCM_PROPS])
    return scores


def gabor_bank(wavelengths=(4, 8, 16), n_orientations=4):
    """(even, odd) kernel pairs, so the response magnitude doesn't depend on edge phase."""
    bank = []
    for lam in wavelengths:
        ksize = 2 * lam + 1
        for k in range(n_orientations):
            theta = k * np.pi / n_orientations
            even = cv2.getGaborKernel((ksize, ksize), 0.56 * lam, theta, lam, 0.5, 0)
            odd = cv2.getGaborKernel((ksize, ksize), 0.56 * lam, theta, lam, 0.5, np.pi / 2)
            bank.append((even - even.mean(), odd))
    return bank


GABOR_BANK = gabor_bank()


def gabor_features(img, bank=GABOR_BANK):
    """Mean and std of the Gabor magnitude for each of the 12 filters (24 numbers)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    feats = []
    for even, odd in bank:
        mag = cv2.magnitude(cv2.filter2D(gray, cv2.CV_32F, even), cv2.filter2D(gray, cv2.CV_32F, odd))
        feats += [mag.mean(), mag.std()]
    return np.array(feats)


def gabor_gap(real, sim, bank=GABOR_BANK):
    """gabor_dist:  distance between the feature vectors, relative to the real one
                    (captures both how much texture there is and where it is in scale/orientation)
    gabor_shape: 1 - cosine similarity, only the scale/orientation mix, ignoring overall contrast
    """
    f_real, f_sim = gabor_features(real, bank), gabor_features(sim, bank)
    cos = f_real @ f_sim / (np.linalg.norm(f_real) * np.linalg.norm(f_sim) + 1e-12)
    return {"gabor_dist": np.linalg.norm(f_real - f_sim) / np.linalg.norm(f_real), "gabor_shape": 1 - cos}


def texture_gap(real, sim):
    """All texture metrics for one pair."""
    return {**glcm_gap(real, sim), **gabor_gap(real, sim)}
