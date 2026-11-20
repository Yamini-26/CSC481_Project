# Sim-to-Real Visual Gap Analysis for Robot Perception

CSC 481 final project, DePaul University. Yamini Ramesh.

A segmentation model trained on simulated images does worse on real camera images, because the simulated images look different. This project measures that visual difference with image-processing metrics (color histograms and CDFs, GLCM, Gabor filters, Canny edges, SSIM, shadow analysis) and finds out which metrics predict how badly the model does on real images.

How it works:

1. A real pick-and-place episode (4 cameras) is replayed in the Drake simulator and rendered in 13 versions, each with one visual change (lighting, shadows, background, table color, blur, noise).
2. The gap metrics are computed between matched real and simulated frames for each version.
3. A DeepLabV3 segmentation model is trained on each version and tested on 64 hand-labeled real frames (mIoU).
4. The metrics are ranked by how well their gap score follows the real mIoU (Spearman correlation).
5. The simulated images are histogram-matched to the real colors to see if that closes the gap.

## Results and observations

- Every model scores about 0.93 mIoU on simulated images but only 0.40-0.58 on real ones.
- The **color distribution** gap (blue-channel CDF distance, rho = -0.68) and **SSIM** (rho = -0.61) predict real performance best. Shadow, Gabor and edge metrics don't. Bhattacharyya and chi-square give almost the same value for every version, so they can't rank anything.
- The best version is *combined* (shadows + gray background + matched table color, 0.58 mIoU) and the worst is *bright light* (0.40).
- **Histogram matching** halves the color gap but doesn't improve the model (0.55 -> 0.57, and 0.58 -> 0.51 for combined). It creates color bands and a purple sky, because one pixel value can only map to one new value.
- **Color augmentation** during training improves every version by about 0.15 mIoU (to 0.62-0.69). Color was the main cause of the drop.
- The model works on close-up wrist views and fails on table glare and the cluttered room, which the simulator never shows.

## Dataset

Download: **[CSC481_data.zip (Google Drive)](https://drive.google.com/file/d/1glIYD4KTIwz4iWHAI9hPvAd_TcK75b8C/view?usp=sharing)**

Unzip it so that `data/` is next to `src/`:

```
data/
  real/frames/<camera>/frame_00000.png ...   real images (76 per camera)
  real/labels/<camera>.npz                   my hand labels (every 5th frame, 16 per camera)
  sim_<version>/frames/<camera>/...          simulated images, same frame numbers
  sim_<version>/labels/<camera>.npz          simulated labels
```

- **Cameras:** `cam_high`, `cam_low`, `cam_left_wrist`, `cam_right_wrist`.
- **Classes:** background, cube, robot, table, frame.

## Setup

Python 3.12. Training (`segment.py`, `make_figures.py`) and the labeling tool need an NVIDIA GPU. Everything else runs on CPU.

```
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Libraries:

| Library | Used for |
|---|---|
| OpenCV | histograms, Canny, Gabor filters |
| scikit-image | GLCM, SSIM |
| NumPy, SciPy, pandas | numbers and Spearman correlation |
| Matplotlib | plots |
| PyTorch / torchvision | DeepLabV3 model |
| Hugging Face Transformers | Segment Anything (SAM) for the labeling tool |

## How to run

Run from the project folder, in this order. Each script shows its options with `--help`.

```
python src/pairs_check.py
```
Checks that every real frame has a matching simulated frame.

```
python src/label_real.py --camera cam_high
```
Opens the SAM click tool to label real frames (already done; the labels are in the dataset). The controls print when it starts.

```
python src/histmatch.py
```
Makes the histogram-matched versions `data/sim_baseline_histmatch` and `data/sim_combined_histmatch`.

```
python src/compute_gaps.py
```
Computes all gap metrics for every version and saves them to `outputs/gaps/`.

```
python src/segment.py --seeds 0 1 2
python src/segment.py --seeds 0 1 2 --color-aug
```
Trains the segmentation model on every version (3 seeds each, about 2.5 min per model) and tests it on the real frames. Results go to `outputs/segmentation/results.csv`; with `--color-aug` they go to `results_coloraug.csv`. Add `--only sim_baseline` to run one version.

```
python src/correlate.py
```
Ranks the metrics against real mIoU and summarizes histogram matching and color augmentation. Saves tables and plots to `outputs/analysis/`.

```
python src/sensitivity.py
```
Recomputes the metrics with different settings (bins, thresholds, window sizes) to check the ranking doesn't depend on them. Saves to `outputs/analysis/sensitivity.*`.

```
python src/make_figures.py
```
Makes all figures (inputs, intermediate outputs, histograms, success and failure examples) in `outputs/figures/`.

## Outputs

The `outputs/` folder is included with all results:

- `outputs/gaps/`: gap scores per frame, camera and version
- `outputs/segmentation/`: mIoU per version and seed, with IoU per class and per camera
- `outputs/analysis/`:
  - metric ranking (`spearman.csv`, `spearman_bars.png`)
  - histogram matching (`histmatch.csv`)
  - color augmentation (`coloraug.csv`, `coloraug.png`)
  - settings check (`sensitivity.csv`, `sensitivity.png`)
- `outputs/figures/`:
  - success/failure examples (`seg_sim_combined.png`, `seg_sim_bright_light.png`)
  - metric intermediate outputs (`metric_steps.png`)
  - histograms (`histograms_rgb.png`)
  - histogram matching (`histmatch_steps.png`)
