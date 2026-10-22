# CSC481_Project
Sim2Real Gap Analysis Using Image Processing Techniques

CSC 481 (Intro to Image Processing) final project, DePaul.

Perception models trained in simulation usually do worse on the real robot because the sim images just don't look like the real ones. The idea here is to measure that visual gap with a bunch of classic image processing metrics on matched real/sim image pairs, and then check which of those metrics actually predict how badly a segmentation model trained in sim does on real images.

## Setup

- Real images: teleoperated pick and place (red cube) on the Trossen Stationary AI arms in the Robotics and Controls Lab at DePaul University.
- Sim images: the same scene and trajectory replayed in Drake.
- 4 cameras: `cam_high`, `cam_low`, `cam_left_wrist`, `cam_right_wrist`, 640x480.
- Real and sim frames are matched by camera and frame index.

## Data layout

Data is not in the repo (too big). Put it under `data/` like this:

```
data/
  real/frames/<camera>/frame_00000.png ...
  sim_baseline/frames/<camera>/frame_00000.png ...
  sim_<variation>/frames/<camera>/...      # lighting, shadows, texture, etc.
```

## Metrics

- Color / intensity: histogram comparison, CDF comparison
- Texture: GLCM features, Gabor filter responses
- Edges / structure: Canny edge comparison, SSIM
- Lighting: shadow analysis

## Experiments

1. Baseline gap: all metrics on real vs. the original sim.
2. Sim variations: fine-tune a segmentation model on each sim variation, test on real, then Spearman correlation between each metric's gap score and the segmentation score to rank the metrics.
3. Correction: histogram match sim to real and redo the gap scores and segmentation to see if it actually helps.

## Running

```
python -m venv .venv
.venv\Scripts\activate        # windows
pip install -r requirements.txt
```

## References

- Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli, "Image quality assessment: from error visibility to structural similarity," IEEE TIP, 2004.
- M. Yan, Q. Sun, I. Frosio, S. Tyree, J. Kautz, "How to Close Sim-Real Gap? Transfer with Segmentation!" arXiv:2005.07695.
