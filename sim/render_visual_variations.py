#!/usr/bin/env python

# Re-render an already simulated replay under different *visual* conditions
# (lighting, shadows, background, material colors, camera effects) for the
# sim-to-real visual gap project.
#
# Physics is NOT re-run: the joint + cube states logged by lerobot_simulation.py
# (trajectory_states.npz) are replayed frame by frame, so every variation shows
# exactly the same robot/cube poses and only the appearance changes.
#
# Copy of the script that lives in the trossen_arm_drake repo (it imports helpers
# from lerobot_simulation.py there, so run it from that repo, not from here).
#
# Usage (from the trossen_arm_drake repo root):
#   python render_visual_variations.py --list
#   python render_visual_variations.py                      # all variations
#   python render_visual_variations.py --only sim_baseline sim_shadows
#
# Output:
#   <out-root>/<variation>/frames/<camera>/frame_XXXXX.png       color image
#   <out-root>/<variation>/labels/<camera>.npz                   class masks, see below
#   <out-root>/<variation>/labels_vis/<camera>/frame_XXXXX.png   colorized masks, for viewing only
#
# labels/<camera>.npz holds
#   labels       uint8 (n_frames, H, W), pixel value = class id (CLASS_IDS)
#   frame_names  matching frame names, e.g. "frame_00040"
# Load with: d = np.load("labels/cam_high.npz"); mask = d["labels"][40]

from pathlib import Path

import argparse
import json
import re
import subprocess
import sys
import time

import cv2
import numpy as np

from pydrake.all import (
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    RigidTransform,
    RgbdSensor,
    CameraInfo,
    ClippingRange,
    DepthRange,
    DepthRenderCamera,
    ColorRenderCamera,
    RenderCameraCore,
    MakeRenderEngineVtk,
    RenderEngineVtkParams,
    LightParameter,
    Rgba,
    BodyIndex,
)

from lerobot_simulation import (
    CAM_WIDTH,
    CAM_HEIGHT,
    CAM_FOCAL,
    CAMERA_FRAME_MAP,
    CUBE_POSITIONS,
    LIGHT_VARIATIONS,
    load_cube_urdf,
)


# Same defaults lerobot_simulation.py used for replay_pick_warm (= sim_baseline)
CUBE_MASS = 0.05
CUBE_FRICTION = 0.8
CUBE_SIDE = 0.038

# Colors measured from the real pick_place_depth_3 frames (cam_high, mean RGB 0..1)
REAL_TABLE_RGB = (0.67, 0.53, 0.42)
REAL_BACKGROUND_RGB = (0.50, 0.53, 0.53)   # between cam_high (~0.58) and cam_low (~0.29) walls

# A pixel color is material color x lighting, so the real table color can't be
# used as the material directly (under the warm light it renders much lighter).
# Instead the maple material was scaled per channel by real / rendered table
# color (cam_high table pixels) until the *rendered* table under the baseline
# light came out at ~REAL_TABLE_RGB. Took two rounds because red clips at 255
# in sim_baseline.
TABLE_MATCHED_RGB = (0.380, 0.304, 0.271)

# URDF material colors we can swap per variation
URDF_MATERIAL_RGBA = {
    "maple_tabletop": "0.58 0.45 0.31 1.0",
    "trossen_black": "0.137 0.137 0.137 1.0",
}

# Segmentation classes written to labels/ (0 = background / empty)
CLASS_IDS = {"background": 0, "cube": 1, "robot": 2, "table": 3, "frame": 4}

# RGB color per class id for labels_vis/
CLASS_COLORS = np.array([
    [40, 40, 40],     # background  dark gray
    [230, 30, 30],    # cube        red
    [40, 190, 60],    # robot       green
    [250, 200, 40],   # table       yellow
    [40, 110, 230],   # frame       blue
], dtype=np.uint8)


def directional(direction, color=(1.0, 1.0, 1.0), intensity=1.0):
    return LightParameter(
        type="directional",
        direction=list(direction),
        color=Rgba(*color, 1.0),
        intensity=intensity,
        frame="world",
    )


WARM = LIGHT_VARIATIONS["warm"]
DRAKE_DEFAULT_BG = (204 / 255, 229 / 255, 255 / 255)   # the light blue "sky"

# Each variation changes one thing relative to sim_baseline, except
# sim_combined, which stacks three of the single changes.
# Keys:
#   lights        list of LightParameter
#   shadows       cast shadows from the lights
#   background    clear color (RGB 0..1)
#   table_rgb     tabletop diffuse color
#   cube_rgb      cube diffuse color
#   post          camera effects applied after rendering (see apply_post)
VARIATIONS = {
    # reproduces replay_pick_warm (sanity check against the existing sim_baseline)
    "sim_baseline": dict(lights=WARM),

    # lighting
    "sim_neutral_light": dict(lights=LIGHT_VARIATIONS["neutral"]),
    "sim_dim_light": dict(lights=[directional([0.5, -0.5, -1.0], (1.0, 0.95, 0.85), 1.0)]),
    "sim_bright_light": dict(lights=[directional([0.5, -0.5, -1.0], (1.0, 0.95, 0.85), 4.0)]),
    "sim_cool_light": dict(lights=[directional([0.5, -0.5, -1.0], (0.80, 0.88, 1.0), 2.5)]),
    "sim_side_light": dict(lights=[directional([1.0, -0.3, -0.35], (1.0, 0.95, 0.85), 2.5)]),
    "sim_two_lights": dict(lights=[
        directional([0.5, -0.5, -1.0], (1.0, 0.95, 0.85), 1.4),
        directional([-0.5, 0.5, -1.0], (1.0, 1.0, 1.0), 0.8),
    ]),

    # shadows
    "sim_shadows": dict(lights=WARM, shadows=True),

    # background / materials
    "sim_gray_background": dict(lights=WARM, background=REAL_BACKGROUND_RGB),
    "sim_table_matched_color": dict(lights=WARM, table_rgb=TABLE_MATCHED_RGB),

    # camera effects (Drake renders a perfect pinhole image)
    "sim_blur": dict(lights=WARM, post=dict(blur_sigma=2.0)),
    "sim_noise": dict(lights=WARM, post=dict(noise_std=8.0)),

    # sim_shadows + sim_gray_background + sim_table_matched_color, baseline light
    "sim_combined": dict(
        lights=WARM,
        shadows=True,
        background=REAL_BACKGROUND_RGB,
        table_rgb=TABLE_MATCHED_RGB,
    ),
}


def rgba_str(rgb):
    return f"{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} 1.0"


def write_robot_urdf(variation, urdf_dir):
    """Copy of stationary_ai.urdf with material colors swapped. Written next to
    the original so the relative ../meshes paths still resolve."""
    text = (urdf_dir / "stationary_ai.urdf").read_text()
    if "table_rgb" in variation:
        old = f'<color rgba="{URDF_MATERIAL_RGBA["maple_tabletop"]}"/>'
        if old not in text:
            raise ValueError("maple_tabletop color not found in stationary_ai.urdf")
        text = text.replace(old, f'<color rgba="{rgba_str(variation["table_rgb"])}"/>')
    path = urdf_dir / "_render_variation.urdf"
    path.write_text(text)
    return path


def cube_urdf_string(variation):
    text = load_cube_urdf(CUBE_MASS, CUBE_FRICTION, CUBE_SIDE)
    if "cube_rgb" in variation:
        text, n = re.subn(r'<color rgba="[^"]*"\s*/>',
                          f'<color rgba="{rgba_str(variation["cube_rgb"])}"/>', text)
        if n == 0:
            raise ValueError("no <color rgba=...> found in urdf/cube.urdf")
    return text


def build_scene(variation, urdf_dir):
    """Plant + scene graph + one RgbdSensor per camera, no simulator."""
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

    params = RenderEngineVtkParams()
    params.lights = variation["lights"]
    params.cast_shadows = variation.get("shadows", False)
    if params.cast_shadows:
        params.shadow_map_size = 1024
    if "background" in variation:
        params.default_clear_color = list(variation["background"])
    scene_graph.AddRenderer("renderer", MakeRenderEngineVtk(params))

    robot_urdf = write_robot_urdf(variation, urdf_dir)
    try:
        Parser(plant).AddModels(str(robot_urdf))
    finally:
        robot_urdf.unlink()
    Parser(plant).AddModelsFromString(cube_urdf_string(variation), "urdf")
    plant.Finalize()

    cam_info = CameraInfo(
        width=CAM_WIDTH, height=CAM_HEIGHT,
        focal_x=CAM_FOCAL, focal_y=CAM_FOCAL,
        center_x=CAM_WIDTH / 2.0, center_y=CAM_HEIGHT / 2.0,
    )
    core = RenderCameraCore("renderer", cam_info, ClippingRange(0.01, 10.0), RigidTransform())
    color_cam = ColorRenderCamera(core, show_window=False)
    depth_cam = DepthRenderCamera(core, DepthRange(0.01, 10.0))

    sensors = {}
    for cam_name, frame_name in CAMERA_FRAME_MAP.items():
        frame = plant.GetFrameByName(frame_name)
        sensor = builder.AddSystem(RgbdSensor(
            parent_id=plant.GetBodyFrameIdOrThrow(frame.body().index()),
            X_PB=frame.GetFixedPoseInBodyFrame(),
            color_camera=color_cam,
            depth_camera=depth_cam,
        ))
        sensor.set_name(f"rgbd_{cam_name}")
        builder.Connect(scene_graph.get_query_output_port(), sensor.query_object_input_port())
        sensors[cam_name] = sensor

    diagram = builder.Build()
    return diagram, plant, sensors


def body_class_lut(plant):
    """Map Drake render label -> class id. MultibodyPlant labels every visual
    geometry with its body index, so classify by body name."""
    lut = np.zeros(plant.num_bodies(), dtype=np.uint8)
    for i in range(plant.num_bodies()):
        name = plant.get_body(BodyIndex(i)).name()
        if name == "cube_link":
            cls = "cube"
        elif name == "tabletop_link":
            cls = "table"
        elif name.startswith("follower_") or name.startswith("leader_"):
            cls = "robot"
        elif name == "world":
            cls = "background"
        else:
            cls = "frame"   # aluminum frame, camera mounts, cameras
        lut[i] = CLASS_IDS[cls]
    return lut


def apply_post(rgb, post, rng):
    out = rgb.astype(np.float32)
    if post.get("blur_sigma"):
        out = cv2.GaussianBlur(out, (0, 0), post["blur_sigma"])
    if post.get("noise_std"):
        out = out + rng.normal(0.0, post["noise_std"], out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def frame_times(t_end, interval):
    """Same instants the ImageSaver in lerobot_simulation.py published at."""
    n = int(np.floor(t_end / interval + 1e-9)) + 1
    return np.arange(n) * interval


def render_frames(name, variation, states_path, out_root, urdf_dir, interval, seed, start, stop):
    """Render frames [start, stop) of one variation."""
    states = np.load(states_path)
    diagram, plant, sensors = build_scene(variation, urdf_dir)
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    lut = body_class_lut(plant)
    rng = np.random.default_rng([seed, start])   # same noise no matter how frames are chunked

    times, q_log = states["times"], states["q"]
    if q_log.shape[1] != plant.num_positions():
        raise ValueError(f"logged q has {q_log.shape[1]} positions, plant has {plant.num_positions()}")

    out_dir = out_root / name
    targets = frame_times(times[-1], interval)
    indices = range(start, min(stop, len(targets)))
    masks = {cam_name: [] for cam_name in sensors}
    for idx in indices:
        t = targets[idx]
        k = min(np.searchsorted(times, t), len(times) - 1)   # first logged state at/after t
        plant.SetPositions(plant_context, q_log[k])

        for cam_name, sensor in sensors.items():
            sensor_context = sensor.GetMyContextFromRoot(context)

            color = sensor.color_image_output_port().Eval(sensor_context)
            rgb = np.array(color.data, copy=True).reshape(CAM_HEIGHT, CAM_WIDTH, 4)[:, :, :3]
            if "post" in variation:
                rgb = apply_post(rgb, variation["post"], rng)

            label = sensor.label_image_output_port().Eval(sensor_context)
            label = np.array(label.data, copy=False).reshape(CAM_HEIGHT, CAM_WIDTH).astype(np.int32)
            valid = (label >= 0) & (label < len(lut))   # empty/don't-care labels are large reserved values
            mask = np.zeros(label.shape, dtype=np.uint8)
            mask[valid] = lut[label[valid]]
            masks[cam_name].append(mask)

            for sub, img in (("frames", rgb), ("labels_vis", CLASS_COLORS[mask])):
                d = out_dir / sub / cam_name
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"frame_{idx:05d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # this process only has a chunk of the frames; pack_labels() merges the chunks
    chunk_dir = out_dir / "labels" / "_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for cam_name, cam_masks in masks.items():
        np.savez(chunk_dir / f"{cam_name}_{start:05d}.npz",
                 labels=np.stack(cam_masks), indices=np.array(indices))


def pack_labels(name, out_root):
    """Merge the per-chunk mask files into labels/<camera>.npz."""
    label_dir = out_root / name / "labels"
    chunk_dir = label_dir / "_chunks"
    for cam_name in CAMERA_FRAME_MAP:
        chunks = [np.load(p) for p in sorted(chunk_dir.glob(f"{cam_name}_*.npz"))]
        indices = np.concatenate([c["indices"] for c in chunks])
        np.savez_compressed(
            label_dir / f"{cam_name}.npz",
            labels=np.concatenate([c["labels"] for c in chunks]),
            frame_names=np.array([f"frame_{i:05d}" for i in indices]),
        )
    for p in chunk_dir.iterdir():
        p.unlink()
    chunk_dir.rmdir()


def write_meta(name, variation, states_path, out_root, interval, n_frames):
    meta = {
        "variation": name,
        "source_states": states_path,
        "n_frames": n_frames,
        "interval_s": interval,
        "class_ids": CLASS_IDS,
        "class_colors_rgb": CLASS_COLORS.tolist(),
        "settings": describe(variation),
    }
    (out_root / name / "variation_meta.json").write_text(json.dumps(meta, indent=2))


def describe(variation):
    """JSON-friendly copy of a variation dict."""
    out = {}
    for k, v in variation.items():
        if k == "lights":
            out[k] = [dict(type=l.type, direction=list(l.direction),
                           color=[l.color.r(), l.color.g(), l.color.b()],
                           intensity=l.intensity) for l in v]
        else:
            out[k] = v
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Render visual variations of a logged replay")
    p.add_argument("--states", type=str,
                   default="simulation_frames/replay_pick_warm/trajectory_states.npz",
                   help="trajectory_states.npz written by lerobot_simulation.py")
    p.add_argument("--out-root", type=str, default="simulation_frames/visual_variations")
    p.add_argument("--only", nargs="*", default=None, help="variation names to render")
    p.add_argument("--list", action="store_true", help="print variation names and exit")
    p.add_argument("--real-fps", type=float, default=30.0)
    p.add_argument("--ffmpeg-stride", type=int, default=10)
    p.add_argument("--seed", type=int, default=0, help="seed for the noise variations")
    p.add_argument("--chunk-size", type=int, default=15,
                   help="frames rendered per subprocess (VTK leaks memory per render)")
    p.add_argument("--frames", type=int, nargs=2, default=None, metavar=("START", "STOP"),
                   help="internal: render only frames [START, STOP) of a single variation")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.list:
        print("\n".join(VARIATIONS))
        raise SystemExit

    names = args.only or list(VARIATIONS)
    unknown = set(names) - set(VARIATIONS)
    if unknown:
        raise SystemExit(f"unknown variations: {sorted(unknown)}")

    interval = args.ffmpeg_stride / args.real_fps
    out_root = Path(args.out_root)

    if args.frames:
        start, stop = args.frames
        render_frames(names[0], VARIATIONS[names[0]], args.states, out_root, Path("urdf"),
                      interval, args.seed, start, stop)
        raise SystemExit

    # Drake/VTK never gives back the memory it uses per render (~4 MB without
    # shadows, ~150 MB with), which kills WSL after a few dozen frames. So each
    # chunk of frames is rendered in its own short-lived process.
    times = np.load(args.states)["times"]
    n_frames = len(frame_times(times[-1], interval))
    print(f"states: {args.states}  ({len(times)} logged steps, {times[-1]:.2f}s) -> {n_frames} frames")
    print(f"output: {out_root}/")
    for name in names:
        t0 = time.time()
        for start in range(0, n_frames, args.chunk_size):
            subprocess.run([
                sys.executable, __file__, "--only", name,
                "--frames", str(start), str(start + args.chunk_size),
                "--states", args.states, "--out-root", args.out_root,
                "--real-fps", str(args.real_fps), "--ffmpeg-stride", str(args.ffmpeg_stride),
                "--seed", str(args.seed),
            ], check=True)
        pack_labels(name, out_root)
        write_meta(name, VARIATIONS[name], args.states, out_root, interval, n_frames)
        print(f"  {name:24s} {n_frames} frames x {len(CAMERA_FRAME_MAP)} cams  ({time.time() - t0:.0f}s)",
              flush=True)
