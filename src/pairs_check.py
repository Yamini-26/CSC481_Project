"""Load matched real/sim image pairs.

Expects data laid out as:
    data/real/frames/<camera>/frame_XXXXX.png
    data/<sim_condition>/frames/<camera>/frame_XXXXX.png

Run directly to check the data (includes all sim variations)
"""
from pathlib import Path

import cv2

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CAMERAS = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]


def list_conditions():
    """All sim conditions in data/"""
    return sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir() and p.name.startswith("sim_"))


def load_image(path):
    """Read an image as an RGB uint8 array (OpenCV reads BGR, so convert)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_pair_paths(condition, camera):
    """Matched (real_path, sim_path) list for one camera, matched by filename."""
    real_dir = DATA_DIR / "real" / "frames" / camera
    sim_dir = DATA_DIR / condition / "frames" / camera

    real = {p.name: p for p in real_dir.glob("*.png")}
    sim = {p.name: p for p in sim_dir.glob("*.png")}

    missing = set(real) ^ set(sim)
    if missing:
        print(f"[warn] {condition}/{camera}: {len(missing)} frames only in one side, skipping them")

    return [(real[name], sim[name]) for name in sorted(set(real) & set(sim))]


def iter_pairs(condition, cameras=None, step=1):
    """Yield (camera, frame_name, real_img, sim_img) for every matched pair.

    step > 1 skips frames, handy for quick tests (step=10 -> every 10th frame).
    """
    for camera in cameras or CAMERAS:
        for real_path, sim_path in get_pair_paths(condition, camera)[::step]:
            real, sim = load_image(real_path), load_image(sim_path)
            if real.shape != sim.shape:
                raise ValueError(f"shape mismatch {real.shape} vs {sim.shape}: {real_path.name} ({camera})")
            yield camera, real_path.stem, real, sim


if __name__ == "__main__":
    conditions = list_conditions()
    print(f"data dir: {DATA_DIR}")
    print(f"sim conditions: {conditions}\n")

    for condition in conditions:
        for camera in CAMERAS:
            pairs = get_pair_paths(condition, camera)
            if not pairs:
                print(f"{condition:20s} {camera:16s} no pairs found")
                continue
            real, sim = load_image(pairs[0][0]), load_image(pairs[0][1])
            ok = "ok" if real.shape == sim.shape else "SHAPE MISMATCH"
            print(f"{condition:20s} {camera:16s} {len(pairs):4d} pairs  {real.shape}  {ok}")
