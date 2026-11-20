"""Click-to-segment tool for making ground-truth masks on the real frames.

Uses Segment Anything (SAM): click on an object and SAM finds its outline,
then a brush can fix whatever SAM got wrong. Masks are saved in the same
format as the sim labels:
    data/real/labels/<camera>.npz        labels (n, H, W) uint8 class ids + frame_names
    data/real/labels_vis/<camera>/*.png  colorized, for viewing
    data/real/labels/_work/<camera>/     per-frame objects, so you can stop and resume

Needs torch + transformers + a GPU, so run it with the Drake env in WSL:
    cd /mnt/c/Users/ryami/Downloads/Intro_to_Image_Processing/Final_Project/CSC481_Project
    ~/trossen_arm_drake/t_arm_drake/bin/python src/label_real.py --camera cam_right_wrist

Controls (also printed when it starts):
    1 cube   2 robot   3 table   4 frame   0 erase        pick the class
    left click        add a point ON the object
    right click       add a point NOT on the object
    left drag         draw a box around the object
    m                 cycle SAM's 3 mask guesses (part / object / bigger)
    b                 toggle brush: left drag paints, right drag erases
    [ ]               brush size
    space / enter     accept the mask as an object of the current class
    z                 undo last click/box/stroke, or last accepted object
    c                 clear clicks, box and brush strokes
    v                 show / hide the accepted labels
    n / p             save, next / previous frame
    q / esc           save and quit

Overlaps don't depend on drawing order: cube > robot > frame > table,
and "erase" objects punch holes in everything.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CLASS_IDS = {"background": 0, "cube": 1, "robot": 2, "table": 3, "frame": 4}
CLASS_NAMES = {v: k for k, v in CLASS_IDS.items()}
CLASS_NAMES[0] = "erase"
# same RGB colors as the sim labels_vis
CLASS_COLORS = np.array([[40, 40, 40], [230, 30, 30], [40, 190, 60], [250, 200, 40], [40, 110, 230]],
                        dtype=np.uint8)
PAINT_ORDER = [3, 4, 2, 1]   # later wins: table < frame < robot < cube

HELP = __doc__[__doc__.index("Controls"):__doc__.index("Overlaps")]


class Sam:
    """SAM with the image embedding cached, so each new click only runs the small decoder."""

    def __init__(self, model_name, device):
        """Load the SAM model and its pre/post-processor from Hugging Face."""
        import torch
        from transformers import SamModel, SamProcessor
        self.torch = torch
        self.device = device
        self.processor = SamProcessor.from_pretrained(model_name)
        self.model = SamModel.from_pretrained(model_name).to(device).eval()

    def set_image(self, rgb):
        """Run the heavy image encoder once per frame and keep the embedding."""
        self.rgb = rgb
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            self.embedding = self.model.get_image_embeddings(inputs["pixel_values"])

    def predict(self, points, box):
        """points: [(x, y, 1 or 0)], box: (x0, y0, x1, y1) or None.
        Returns 3 candidate masks (bool, H x W) and their predicted quality."""
        kwargs = {}
        if points:
            kwargs["input_points"] = [[[x, y] for x, y, _ in points]]
            kwargs["input_labels"] = [[lab for _, _, lab in points]]
        if box is not None:
            kwargs["input_boxes"] = [[list(box)]]
        inputs = self.processor(images=self.rgb, return_tensors="pt", **kwargs).to(self.device)
        inputs.pop("pixel_values")
        with self.torch.no_grad():
            out = self.model(**inputs, image_embeddings=self.embedding, multimask_output=True)
        masks = self.processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        return masks[0][0].numpy().astype(bool), out.iou_scores[0, 0].cpu().numpy()


def compose(objects, shape):
    """Class-id map from the accepted (class_id, mask) objects."""
    label = np.zeros(shape, dtype=np.uint8)
    for cls in PAINT_ORDER:
        for c, m in objects:
            if c == cls:
                label[m] = cls
    for c, m in objects:
        if c == 0:
            label[m] = 0
    return label


class Labeler:
    """Interactive OpenCV window: SAM clicks/boxes + brush -> per-frame class masks."""
    def __init__(self, camera, frames, sam, scale):
        """Set up the window state; work files go to data/real/labels/_work/<camera>/."""
        self.camera = camera
        self.frames = frames
        self.sam = sam
        self.scale = scale
        self.work_dir = DATA_DIR / "real" / "labels" / "_work" / camera
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.cls = CLASS_IDS["table"]
        self.show_labels = True
        self.brush = False
        self.brush_radius = 6

    # ---------- per-frame state ----------

    def load_frame(self, i):
        """Show frame i, reload its saved objects (if any) and embed it with SAM."""
        self.i = i
        path = self.frames[i]
        self.rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        self.shape = self.rgb.shape[:2]
        self.objects = []
        work = self.work_dir / f"{path.stem}.npz"
        if work.exists():
            d = np.load(work)
            n = self.shape[0] * self.shape[1]
            self.objects = [(int(c), np.unpackbits(p, count=n).reshape(self.shape).astype(bool))
                            for c, p in zip(d["classes"], d["masks"])]
        self.clear_prompts()
        print(f"[{i + 1}/{len(self.frames)}] {self.camera}/{path.name}  ({len(self.objects)} objects)  embedding...",
              end="", flush=True)
        self.sam.set_image(self.rgb)
        print(" ready")

    def clear_prompts(self):
        """Drop the current clicks, box and brush strokes (not the accepted objects)."""
        self.points, self.box = [], None
        self.candidates, self.scores, self.pick = None, None, 0
        self.add = np.zeros(self.shape, bool)
        self.remove = np.zeros(self.shape, bool)
        self.history = []   # undo stack of "point" / "box" / "stroke" entries
        self.drag = None

    def run_sam(self):
        """Re-run the SAM decoder on the current clicks/box and pick its best guess."""
        if self.points or self.box is not None:
            self.candidates, self.scores = self.sam.predict(self.points, self.box)
            self.pick = int(np.argmax(self.scores))
        else:
            self.candidates = None

    def current_mask(self):
        """SAM's chosen mask with the brush additions/removals applied."""
        mask = self.candidates[self.pick].copy() if self.candidates is not None else np.zeros(self.shape, bool)
        return (mask | self.add) & ~self.remove

    def accept(self):
        """Keep the current mask as an object of the selected class."""
        mask = self.current_mask()
        if not mask.any():
            return
        self.objects.append((self.cls, mask))
        print(f"  + {CLASS_NAMES[self.cls]} ({mask.sum()} px)")
        self.clear_prompts()

    def undo(self):
        """Undo the last click/box/stroke, or the last accepted object if there are none."""
        if self.history:
            kind, data = self.history.pop()
            if kind == "point":
                self.points.pop()
            elif kind == "box":
                self.box = data
            elif kind == "stroke":
                self.add, self.remove = data
            self.run_sam()
        elif self.objects:
            c, m = self.objects.pop()
            print(f"  - undid {CLASS_NAMES[c]}")

    def save(self):
        """Save this frame's objects and rebuild the camera's label file."""
        stem = self.frames[self.i].stem
        work = self.work_dir / f"{stem}.npz"
        if not self.objects:
            work.unlink(missing_ok=True)
        else:
            np.savez_compressed(work, classes=np.array([c for c, _ in self.objects]),
                                masks=np.stack([np.packbits(m.ravel()) for _, m in self.objects]))
        write_outputs(self.camera, self.work_dir, self.shape)

    # ---------- drawing ----------

    def render(self):
        """Image with accepted labels, the current mask, prompts and a status bar on top."""
        img = self.rgb.astype(np.float32)
        if self.show_labels and self.objects:
            label = compose(self.objects, self.shape)
            on = label > 0
            img[on] = 0.55 * img[on] + 0.45 * CLASS_COLORS[label[on]]
        mask = self.current_mask()
        if mask.any():
            color = CLASS_COLORS[self.cls] if self.cls else np.array([255, 255, 255])
            img[mask] = 0.45 * img[mask] + 0.55 * color
        img = img.astype(np.uint8)
        if mask.any():
            cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cnts, -1, (255, 255, 255), 1)

        s = self.scale
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        for x, y, lab in self.points:
            cv2.circle(img, (int(x * s), int(y * s)), 6, (0, 255, 0) if lab else (255, 0, 0), -1)
            cv2.circle(img, (int(x * s), int(y * s)), 6, (0, 0, 0), 1)
        box = self.box
        if self.drag and self.drag[0] == "box" and self.drag[3]:
            box = (*self.drag[1], *self.drag[2])
        if box is not None:
            cv2.rectangle(img, (int(box[0] * s), int(box[1] * s)), (int(box[2] * s), int(box[3] * s)),
                          (255, 255, 0), 2)

        guess = f"  guess {self.pick + 1}/3 (q={self.scores[self.pick]:.2f})" if self.candidates is not None else ""
        mode = f"BRUSH r={self.brush_radius}" if self.brush else "SAM"
        hud = (f"{self.camera} {self.frames[self.i].stem} [{self.i + 1}/{len(self.frames)}]  "
               f"class: {CLASS_NAMES[self.cls].upper()}  {mode}{guess}  objects: {len(self.objects)}"
               f"{'' if self.show_labels else '  LABELS HIDDEN (v)'}")
        bar = np.zeros((28, img.shape[1], 3), np.uint8)
        cv2.putText(bar, hud, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return cv2.cvtColor(np.vstack([bar, img]), cv2.COLOR_RGB2BGR)

    # ---------- input ----------

    def on_mouse(self, event, x, y, flags, _):
        """Mouse handler: click = point, drag = box, right click = negative point; brush mode paints."""
        x = min(max(x / self.scale, 0), self.shape[1] - 1)
        y = min(max((y - 28) / self.scale, 0), self.shape[0] - 1)

        if self.brush:
            if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
                self.history.append(("stroke", (self.add.copy(), self.remove.copy())))
                self.drag = ("paint", event == cv2.EVENT_LBUTTONDOWN)
            if event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
                self.drag = None
            if self.drag and event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MOUSEMOVE):
                dot = np.zeros(self.shape, np.uint8)
                cv2.circle(dot, (int(x), int(y)), self.brush_radius, 1, -1)
                dot = dot.astype(bool)
                if self.drag[1]:
                    self.add |= dot
                    self.remove &= ~dot
                else:
                    self.remove |= dot
                    self.add &= ~dot
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag = ("box", (x, y), (x, y), False)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag:
            moved = self.drag[3] or abs(x - self.drag[1][0]) + abs(y - self.drag[1][1]) > 6 / self.scale
            self.drag = ("box", self.drag[1], (x, y), moved)
        elif event == cv2.EVENT_LBUTTONUP and self.drag:
            (x0, y0), moved = self.drag[1], self.drag[3]
            self.drag = None
            if moved:
                self.history.append(("box", self.box))
                self.box = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
            else:
                self.history.append(("point", None))
                self.points.append((x, y, 1))
            self.run_sam()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.history.append(("point", None))
            self.points.append((x, y, 0))
            self.run_sam()

    def run(self, start):
        """Main loop: draw the window and handle the keyboard shortcuts listed in HELP."""
        win = "label_real (h = help in terminal)"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, self.on_mouse)
        self.load_frame(start)
        while True:
            cv2.imshow(win, self.render())
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue
            ch = chr(key)
            if ch in "01234":
                self.cls = int(ch)
            elif key in (10, 13, 32):
                self.accept()
            elif ch == "m" and self.candidates is not None:
                self.pick = (self.pick + 1) % 3
            elif ch == "b":
                self.brush = not self.brush
            elif ch == "[":
                self.brush_radius = max(1, self.brush_radius - 2)
            elif ch == "]":
                self.brush_radius += 2
            elif ch == "z":
                self.undo()
            elif ch == "c":
                self.clear_prompts()
            elif ch == "v":
                self.show_labels = not self.show_labels
            elif ch == "h":
                print(HELP)
            elif ch in "np":
                if self.current_mask().any():
                    print("  (unaccepted mask on screen: press space to keep it, c to drop it)")
                    continue
                self.save()
                step = 1 if ch == "n" else -1
                self.load_frame(min(max(self.i + step, 0), len(self.frames) - 1))
            elif ch == "q" or key == 27:
                self.save()
                break
        cv2.destroyAllWindows()


def write_outputs(camera, work_dir, shape):
    """Rebuild labels/<camera>.npz and labels_vis/ from all labeled frames of this camera."""
    stems, labels = [], []
    n = shape[0] * shape[1]
    vis_dir = DATA_DIR / "real" / "labels_vis" / camera
    vis_dir.mkdir(parents=True, exist_ok=True)
    for work in sorted(work_dir.glob("frame_*.npz")):
        d = np.load(work)
        objects = [(int(c), np.unpackbits(p, count=n).reshape(shape).astype(bool))
                   for c, p in zip(d["classes"], d["masks"])]
        label = compose(objects, shape)
        stems.append(work.stem)
        labels.append(label)
        cv2.imwrite(str(vis_dir / f"{work.stem}.png"), cv2.cvtColor(CLASS_COLORS[label], cv2.COLOR_RGB2BGR))
    out = DATA_DIR / "real" / "labels" / f"{camera}.npz"
    if labels:
        np.savez_compressed(out, labels=np.stack(labels), frame_names=np.array(stems))
    print(f"  saved {len(labels)} labeled frames -> {out.relative_to(DATA_DIR.parent)}")


def main():
    """Parse arguments, load SAM and start labeling at the first unlabeled frame."""
    p = argparse.ArgumentParser(description="Label real frames with SAM")
    p.add_argument("--camera", required=True, choices=["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"])
    p.add_argument("--step", type=int, default=5, help="label every Nth frame (5 -> 16 frames per camera)")
    p.add_argument("--scale", type=float, default=1.5, help="window zoom")
    p.add_argument("--model", default="facebook/sam-vit-large")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    frames = sorted((DATA_DIR / "real" / "frames" / args.camera).glob("frame_*.png"))[::args.step]
    work_dir = DATA_DIR / "real" / "labels" / "_work" / args.camera
    done = {w.stem for w in work_dir.glob("frame_*.npz")}
    start = next((i for i, f in enumerate(frames) if f.stem not in done), 0)

    print(HELP)
    print(f"loading {args.model} ...")
    sam = Sam(args.model, args.device)
    Labeler(args.camera, frames, sam, args.scale).run(start)


if __name__ == "__main__":
    main()
