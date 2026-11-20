# Sim-to-Real Visual Gap Analysis for Robot Perception

CSC 481 (Intro to Image Processing) final project, DePaul University, Autumn 2026. Yamini Ramesh.

Perception models trained on simulated images usually do worse on a real robot, because simulated images don't look like real ones. This project measures that visual gap with classical image-processing metrics on matched real/sim image pairs, and checks which metrics actually **predict** how badly a segmentation model trained in sim does on real images.

Pipeline:

1. A real pick-and-place episode (4 cameras) is replayed in the Drake simulator and rendered under 13 visual variations (lighting, shadows, background, table color, blur, noise).
2. For every variation, color, texture, structure and lighting **gap metrics** are computed between matched real and sim frames.
3. A **segmentation model** (DeepLabV3) is trained on each variation and tested on hand-labeled real frames (mIoU).
4. The metrics are **ranked** by the Spearman correlation between their gap score and the real mIoU.
5. **Experiment 3:** sim images are histogram-matched to the real color distribution to see if that closes the gap.
6. Extra checks: 3 training seeds per variation, the ranking under different metric settings, and retraining with color augmentation.

Main results (details in the report):

| | |
|---|---|
| mIoU on sim / on real | about 0.93 / 0.40-0.58 |
| Best predictors (Spearman rho) | blue-channel CDF earth mover's distance -0.68 (p = 0.011), SSIM -0.61 (p = 0.027), histogram correlation -0.57 (p = 0.044) |
| Not predictive | shadow, Gabor and edge metrics; Bhattacharyya and chi-square saturate |
| Histogram matching | halves the color gap, but real mIoU does not improve (baseline 0.55 -> 0.57, combined 0.58 -> 0.51) |
| Color augmentation | real mIoU +0.15 on average (0.62-0.69) for every variation |

## Contents

```
README.md                 this file
requirements.txt          Python packages
src/                      all project code (see "Scripts" below)
sim/                      the Drake rendering script used to make the sim variations (record only, see below)
outputs/                  results produced by the code (csv tables and figures), included in the submission
  gaps/                     gap scores per frame / camera / variation
  segmentation/             mIoU per variation and seed (without and with color augmentation)
  analysis/                 metric ranking, histogram-matching and color-augmentation results, plots
  figures/                  figures with inputs, intermediate outputs and success/failure examples
  color/                    first exploratory histogram/CDF plots (sim baseline)
```

## Dataset

The dataset is my own (real robot recordings plus my renders and hand labels), so it is shared through a drive:

**Download: [ADD THE ONEDRIVE / GOOGLE DRIVE LINK HERE]**

Unzip it into the project folder so that there is a `data/` folder next to `src/`:

```
data/
  real/
    frames/<camera>/frame_00000.png ... frame_00075.png   real images, 640x480
    labels/<camera>.npz                                   hand labels of every 5th frame (16 per camera)
    labels_vis/<camera>/*.png                             the same labels as colored images
  sim_<variation>/                                         13 variations, e.g. sim_baseline, sim_bright_light
    frames/<camera>/frame_00000.png ...                   rendered images, same frame numbers as real
    labels/<camera>.npz                                   rendered labels of all frames
    labels_vis/<camera>/*.png
    variation_meta.json                                   the render settings of this variation
  sim_baseline_histmatch/, sim_combined_histmatch/         experiment 3 (can be regenerated with histmatch.py)
```

- **Cameras:** `cam_high`, `cam_low`, `cam_left_wrist`, `cam_right_wrist`.
- **Classes:** 0 background, 1 cube, 2 robot, 3 table, 4 frame.
- **Label files:** each `labels/<camera>.npz` holds `labels` (N x 480 x 640, uint8 class ids) and `frame_names`.

How the data was made:

- **Real:** one teleoperated pick-and-place of a red cube on the Trossen Stationary AI arms in the DePaul Robotics and Controls Lab. I kept every 10th frame of the 30 fps recording, which gives 76 frames per camera.
- **Sim:** the logged joint states of the same episode, replayed in Drake and rendered from the four camera poses with `sim/render_visual_variations.py`. Real frame k and sim frame k show the same robot pose (up to about 10-30 px of misalignment).
- **Real labels:** made by me with `src/label_real.py` (SAM click tool). These 64 frames are the real test set.

## Installation

Python 3.12. The image-processing and analysis scripts run on any machine (CPU). Training the segmentation model (`segment.py`, part of `make_figures.py`) and the labeling tool (`label_real.py`) need an NVIDIA GPU with CUDA.

```
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # pick the build for your CUDA, see pytorch.org
pip install -r requirements.txt
```

I ran everything in WSL (Ubuntu 24.04) on an RTX 3050 laptop GPU (6 GB), with Python 3.12, torch 2.12, torchvision 0.27, OpenCV 4, scikit-image 0.26, SciPy 1.17, pandas 3.0 and transformers 5.14.

## How to run

All scripts are in `src/` and are run from the project folder. Every script except the two quick checks prints its options with `--help`. Results go to `outputs/`, and running a step again overwrites its outputs.

**Only check the evaluation (no GPU, a few seconds).** The results in `outputs/` are already included. Re-rank the metrics from them:

```
python src/correlate.py
```

**Full pipeline, in order:**

| # | Command | What it does | Output | Time |
|---|---|---|---|---|
| 1 | `python src/pairs_check.py` | checks that every real frame has a matching sim frame of the same size | printed table | seconds |
| 2 | `python src/label_real.py --camera cam_high` (once per camera) | interactive SAM labeling of the real frames (already done, labels are in the dataset) | `data/real/labels/` | manual |
| 3 | `python src/histmatch.py` | experiment 3: histogram-matches `sim_baseline` and `sim_combined` to the real colors | `data/sim_*_histmatch/` | ~1 min |
| 4 | `python src/compute_gaps.py` | all gap metrics for every variation on the 64 labeled pairs | `outputs/gaps/` | a few min |
| 5 | `python src/segment.py --seeds 0 1 2` | trains DeepLabV3 on each variation, tests on the real frames | `outputs/segmentation/results.csv` | ~2.5 min per model (GPU) |
| 6 | `python src/segment.py --seeds 0 1 2 --color-aug` | the same with color augmentation | `outputs/segmentation/results_coloraug.csv` | ~2.5 min per model (GPU) |
| 7 | `python src/correlate.py` | Spearman ranking of the metrics, experiment 3 and color-aug summaries | `outputs/analysis/` | seconds |
| 8 | `python src/sensitivity.py` | recomputes the metrics with other settings and re-ranks them | `outputs/analysis/sensitivity.*` | ~15 min |
| 9 | `python src/make_figures.py` | all figures (retrains 2 models for the examples; `--no-seg` skips that) | `outputs/figures/` | ~5 min |

Useful options:

- `compute_gaps.py --only sim_baseline` and `segment.py --only sim_baseline sim_combined` run only some variations.
- `segment.py` skips (variation, seed) pairs that are already in its csv, so it can be stopped and resumed.
- `histmatch.py --conditions sim_baseline` matches only some variations.

## Scripts

| File | Purpose |
|---|---|
| `src/pairs_check.py` | Shared helpers: the list of cameras and variations, loading images and label files, matching real/sim frames. Run directly to check the data. |
| `src/metrics_color.py` | Color gap: per-channel 64-bin histograms compared with correlation, chi-square and Bhattacharyya; CDFs of R, G, B and gray compared with the Kolmogorov-Smirnov statistic and the earth mover's distance (area between the CDFs). Run directly for a quick check on the baseline that also includes a real-vs-real sanity check. |
| `src/metrics_texture.py` | Texture gap: GLCM contrast, homogeneity, energy and correlation (32 levels, offsets 1 and 4 px, 4 angles); a Gabor filter bank (3 wavelengths x 4 orientations, even/odd pairs) summarized as a 24-D vector. |
| `src/metrics_structure.py` | Structure gap: Canny edges with fixed thresholds (edge density and edge F1 with a 3 px tolerance, using distance transforms) and SSIM. |
| `src/metrics_lighting.py` | Lighting gap on the table pixels (from the labels): median brightness, shadow fraction and depth (shadow = darker than 0.7 x the median, a threshold rule, no model), non-uniformity. |
| `src/compute_gaps.py` | Runs all four metric modules on every labeled real/sim pair and averages per camera and per variation. |
| `src/segment.py` | Segmentation transfer: DeepLabV3-MobileNetV3 (COCO weights, new 5-class head) trained on one variation's 304 sim frames, tested on the 64 real frames. Reports mIoU, IoU per class, mIoU per camera, and the sim mIoU as a sanity check. `--color-aug` adds color jitter. |
| `src/correlate.py` | Evaluation: Spearman rho and p-value of every metric against real mIoU (over 13 variations and over 52 variation x camera pairs), the bar chart, the experiment 3 table and the color-augmentation comparison. |
| `src/sensitivity.py` | Robustness of the ranking: histogram bins, Canny thresholds, edge tolerance, SSIM window, GLCM levels, Gabor wavelengths and shadow threshold are varied and rho is recomputed. |
| `src/histmatch.py` | Experiment 3: classic histogram matching T(v) = F_real^-1(F_sim(v)) per RGB channel, with the real reference CDF pooled from the 60 real frames per camera that are *not* in the test set. |
| `src/label_real.py` | My labeling tool: Segment Anything (SAM ViT-L) proposes a mask from clicks or a box, a brush fixes it, and class priorities resolve overlaps. Controls are listed at the top of the file and printed at start. |
| `src/make_figures.py` | Figures for the report: input pairs, the 13 variations, labels, intermediate outputs (Canny, Gabor energy, shadow mask), histograms and CDFs, histogram-matching steps, and segmentation success/failure examples (best and worst real frames). |
| `sim/render_visual_variations.py` | How the 13 sim variations were rendered (lights, shadows, background and table colors, blur, noise) and how the sim labels were made. It needs Drake and the lab's `trossen_arm_drake` repository (robot model and logged states), which are not part of this submission, so it is included as a record and does not run on its own. |

## Evaluation

All the quantitative results are written as csv files, so they can be checked or re-plotted.

- **Segmentation quality:** `outputs/segmentation/results.csv` has one row per variation and seed. It includes:
  - real mIoU and pixel accuracy
  - IoU of each class
  - mIoU per camera
  - the mIoU of the same model on sim frames

  `outputs/analysis/segmentation_summary.csv` averages it over seeds.
- **Gap scores:**
  - `outputs/gaps/per_frame.csv`: every metric for every labeled pair
  - `by_camera.csv`: averaged per camera
  - `by_condition.csv`: averaged per variation
- **Metric ranking:**
  - `outputs/analysis/spearman.csv` and `spearman_bars.png`: rho and p for every metric
  - `top_metrics_scatter.png`: gap vs mIoU for the best and worst metrics
- **Experiment 3:** `outputs/analysis/histmatch.csv` gives the gap and mIoU before and after matching.
- **Robustness:**
  - `outputs/analysis/sensitivity.csv` and `.png`: metric settings
  - `coloraug.csv`, `coloraug.png` and `spearman_coloraug.csv`: color augmentation
- **Qualitative results:**
  - `outputs/figures/seg_sim_combined.png` and `seg_sim_bright_light.png`: the two best and two worst real frames for the best and worst variation (input, ground truth, prediction), i.e. success and failure cases
  - `metric_steps.png`: intermediate outputs of the metrics
  - `histmatch.png` and `histmatch_steps.png`: the correction

## Libraries and tools

| Library or tool | Used for |
|---|---|
| [OpenCV](https://opencv.org) | image I/O, color conversions, histograms and `compareHist`, Canny, Gabor kernels and filtering, distance transform, drawing |
| [scikit-image](https://scikit-image.org) | GLCM (`graycomatrix`, `graycoprops`) and SSIM (`structural_similarity`) |
| NumPy, [SciPy](https://scipy.org) | CDFs, histogram matching, Spearman correlation |
| pandas | result tables |
| Matplotlib | plots |
| [PyTorch / torchvision](https://pytorch.org) | DeepLabV3-MobileNetV3-Large model with COCO-pretrained weights |
| [Hugging Face Transformers](https://huggingface.co/docs/transformers) | the Segment Anything model (`facebook/sam-vit-large`) for the labeling tool |
| [Drake](https://drake.mit.edu) | robot simulation and rendering (outside this package) |

What I implemented myself:

- the matched real/sim data setup, and the 13 single-factor variations with a calibrated table color
- the SAM labeling tool
- the metric definitions and their combination:
  - CDF/EMD gaps
  - edge F1 with tolerance
  - Gabor feature vector
  - table-based shadow analysis
- histogram matching, using a reference built only from non-test frames
- the training/evaluation loop, the correlation analysis, the robustness checks and all figures

The libraries provide the low-level operations (filters, GLCM, SSIM, the network architecture).

## References

- Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli, "Image quality assessment: from error visibility to structural similarity," IEEE TIP, 2004.
- R. M. Haralick, K. Shanmugam, I. Dinstein, "Textural features for image classification," IEEE TSMC, 1973.
- J. Canny, "A computational approach to edge detection," IEEE TPAMI, 1986.
- L.-C. Chen, G. Papandreou, F. Schroff, H. Adam, "Rethinking atrous convolution for semantic image segmentation," arXiv:1706.05587, 2017.
- A. Kirillov et al., "Segment anything," ICCV 2023.
- S. James et al., "Sim-to-real via sim-to-sim: Data-efficient robotic grasping via randomized-to-canonical adaptation networks," CVPR 2019.
- Y. Yang, S. Soatto, "FDA: Fourier domain adaptation for semantic segmentation," CVPR 2020.
- K. Ding, K. Ma, S. Wang, E. P. Simoncelli, "Image quality assessment: Unifying structure and texture similarity," IEEE TPAMI, 2022.
