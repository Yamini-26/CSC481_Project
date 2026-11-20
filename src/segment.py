"""Train a segmentation model on one sim condition, then test it on the labeled real frames.

Model: DeepLabV3 with a MobileNetV3-Large backbone (torchvision), starting from
its COCO weights with a new 5-class head (background, cube, robot, table, frame).
Training uses all 76 frames x 4 cameras of the sim condition; testing uses the
16 x 4 hand-labeled real frames. No color augmentation by default on purpose: the
point is to see how the sim's own appearance transfers to real. --color-aug adds
random brightness/contrast/saturation/hue jitter, to check how much of the gap
plain augmentation hides.

    python src/segment.py --only sim_baseline sim_combined
    python src/segment.py                  # every condition in data/
    python src/segment.py --seeds 0 1 2    # repeat with different seeds
    python src/segment.py --color-aug      # same, with color jitter

Results go to outputs/segmentation/results.csv (results_coloraug.csv with
--color-aug), one row per (condition, seed).
Conditions/seeds already in the csv are skipped, so it can be stopped and resumed.
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.models.segmentation import (DeepLabV3_MobileNet_V3_Large_Weights,
                                             deeplabv3_mobilenet_v3_large)
from torchvision.transforms.v2 import functional as TF

from pairs_check import CAMERAS, DATA_DIR, list_conditions, load_image, load_labels

CLASS_NAMES = ["background", "cube", "robot", "table", "frame"]
SEG_DIR = DATA_DIR.parent / "outputs" / "segmentation"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_set(source, frames_only=None):
    """uint8 images (N, H, W, 3), labels (N, H, W) and the camera of each, for one data source."""
    images, labels, cams = [], [], []
    for camera in CAMERAS:
        lab = load_labels(source, camera)
        for frame in sorted(frames_only[camera] if frames_only else lab):
            images.append(load_image(DATA_DIR / source / "frames" / camera / f"{frame}.png"))
            labels.append(lab[frame])
            cams.append(camera)
    return np.stack(images), np.stack(labels), np.array(cams)


def to_float(images, device):
    """uint8 (N, H, W, 3) images -> float (N, 3, H, W) tensor in 0..1 on the device."""
    return torch.from_numpy(images).to(device).permute(0, 3, 1, 2).float() / 255


def normalize(x):
    """ImageNet mean/std normalization (what the pretrained backbone expects)."""
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def to_input(images, device):
    """uint8 images -> normalized network input."""
    return normalize(to_float(images, device))


def color_jitter(x):
    """Random brightness, contrast, saturation (x0.6-1.4) and hue (+-0.05) per image, x in 0..1."""
    out = []
    for img in x:
        f = torch.empty(3).uniform_(0.6, 1.4).tolist()
        img = TF.adjust_brightness(img, f[0])
        img = TF.adjust_contrast(img, f[1])
        img = TF.adjust_saturation(img, f[2])
        img = TF.adjust_hue(img, float(torch.empty(1).uniform_(-0.05, 0.05)))
        out.append(img)
    return torch.stack(out)


def augment(x, y):
    """Random horizontal flip and random scale/crop (geometry only, no color changes)."""
    if torch.rand(1) < 0.5:
        x, y = x.flip(-1), y.flip(-1)
    s = float(torch.empty(1).uniform_(1.0, 1.5))
    if s > 1.01:
        h, w = x.shape[-2:]
        x = F.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False)
        y = F.interpolate(y[:, None].float(), scale_factor=s, mode="nearest")[:, 0].long()
        top = int(torch.randint(0, x.shape[-2] - h + 1, (1,)))
        left = int(torch.randint(0, x.shape[-1] - w + 1, (1,)))
        x, y = x[..., top:top + h, left:left + w], y[..., top:top + h, left:left + w]
    return x, y


def build_model(device):
    """DeepLabV3-MobileNetV3 with COCO weights and a new 5-class output layer."""
    model = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1)
    model.classifier[4] = torch.nn.Conv2d(256, len(CLASS_NAMES), 1)
    model.aux_classifier = None
    return model.to(device)


def train(images, labels, device, epochs, batch_size, lr, color_aug=False):
    """Train a new model on (images, labels) with pixel-wise cross-entropy and return it."""
    model = build_model(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(images) / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(images))
        total = 0.0
        for i in range(0, len(images), batch_size):
            idx = order[i:i + batch_size].numpy()
            x = to_float(images[idx], device)
            if color_aug:
                x = color_jitter(x)
            x = normalize(x)
            y = torch.from_numpy(labels[idx]).to(device).long()
            x, y = augment(x, y)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(model(x)["out"], y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(idx)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch + 1}/{epochs}  loss {total / len(images):.3f}")
    return model


@torch.no_grad()
def predict(model, images, device, batch_size=8):
    """Predicted class map for every image."""
    model.eval()
    preds = []
    for i in range(0, len(images), batch_size):
        with torch.autocast("cuda", dtype=torch.float16):
            preds.append(model(to_input(images[i:i + batch_size], device))["out"].argmax(1).cpu().numpy())
    return np.concatenate(preds)


def iou_per_class(pred, label):
    """IoU of each class over all given pixels (NaN if the class is in neither)."""
    ious = []
    for c in range(len(CLASS_NAMES)):
        inter = np.logical_and(pred == c, label == c).sum()
        union = np.logical_or(pred == c, label == c).sum()
        ious.append(inter / union if union else np.nan)
    return np.array(ious)


def evaluate(pred, labels, cams):
    """mIoU over all real frames, IoU per class, and mIoU per camera."""
    ious = iou_per_class(pred, labels)
    row = {"miou": np.nanmean(ious), "pixel_acc": (pred == labels).mean()}
    row.update({f"iou_{name}": v for name, v in zip(CLASS_NAMES, ious)})
    for camera in CAMERAS:
        row[f"miou_{camera}"] = np.nanmean(iou_per_class(pred[cams == camera], labels[cams == camera]))
    return row


def main():
    """Train and test every (condition, seed) not yet in the results csv."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", nargs="+", help="conditions to run (default: all in data/)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--color-aug", action="store_true", help="add color jitter to the training images")
    args = p.parse_args()
    out_csv = SEG_DIR / ("results_coloraug.csv" if args.color_aug else "results.csv")

    device = "cuda"
    real_images, real_labels, real_cams = load_set("real")
    real_frames = {camera: list(load_labels("real", camera)) for camera in CAMERAS}
    print(f"real test set: {len(real_images)} frames")

    SEG_DIR.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(columns=["condition", "seed"])
    done_keys = set(zip(done["condition"], done["seed"]))

    for condition in args.only or list_conditions():
        todo = [s for s in args.seeds if (condition, s) not in done_keys]
        if not todo:
            continue
        images, labels, _ = load_set(condition)
        # the sim frames that match the labeled real ones, as a sim-on-sim sanity check (these are in training)
        sim_images, sim_labels, sim_cams = load_set(condition, real_frames)
        for seed in todo:
            print(f"{condition}  seed {seed}  ({len(images)} training frames)")
            torch.manual_seed(seed)
            np.random.seed(seed)
            t = time.time()
            model = train(images, labels, device, args.epochs, args.batch_size, args.lr, args.color_aug)
            row = {"condition": condition, "seed": seed, **evaluate(predict(model, real_images, device),
                                                                    real_labels, real_cams)}
            row["sim_miou"] = evaluate(predict(model, sim_images, device), sim_labels, sim_cams)["miou"]
            row["train_minutes"] = (time.time() - t) / 60
            print(f"  real mIoU {row['miou']:.3f}  (sim {row['sim_miou']:.3f})  "
                  + "  ".join(f"{c} {row[f'iou_{c}']:.2f}" for c in CLASS_NAMES)
                  + f"  [{row['train_minutes']:.1f} min]")
            new = pd.DataFrame([row])
            done = new if done.empty else pd.concat([done, new], ignore_index=True)
            done.to_csv(out_csv, index=False)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
