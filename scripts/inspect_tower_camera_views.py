#!/usr/bin/env python3
"""Render a physical-up camera sweep of the unchanged full Gaussian model."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from revision_experiments.dynamic_temporal_stability import (
    load_splats, build_auto_camera, build_intrinsics, render_rgb_alpha, save_png,
)
from workflow.coordinate_frames import physical_up_in_gaussian_frame


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runtime, out = args.runtime_dir.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    static = load_splats(runtime / "static.pt", device)
    moving = load_splats(runtime / "moving.pt", device)
    up = physical_up_in_gaussian_frame(runtime / "gs_to_fe.txt")
    groups = (static, moving)
    fields = [torch.cat([getattr(group, key) for group in groups]) for key in
              ("means", "quats", "scales", "opacities", "colors")]
    width, height = 512, 768
    intrinsics = build_intrinsics(width, height, 45)
    kmat = torch.from_numpy(intrinsics).float().to(device)
    white = torch.ones(3, device=device)
    panels, cameras = [], []
    for elevation in (15, -15):
        for azimuth in range(0, 360, 45):
            c2w, info = build_auto_camera(static, moving, "moving", azimuth, elevation, 3.0, None, up)
            view = torch.from_numpy(np.linalg.inv(c2w)).float().to(device)
            rgb, _, _ = render_rgb_alpha(*fields, moving.sh_degree, view, kmat, width, height, 0, .01, 1e10, white, False)
            name = f"az_{azimuth:03d}_el_{elevation:+03d}.png"
            save_png(out / name, rgb)
            tile = Image.new("RGB", (512, 800), "white")
            tile.paste(Image.open(out / name), (0, 32))
            ImageDraw.Draw(tile).text((12, 8), f"azimuth={azimuth}, elevation={elevation}", fill="black")
            panels.append(tile)
            cameras.append({"file": name, "c2w": c2w.tolist(), **info})
            print(name, flush=True)
    sheet = Image.new("RGB", (2048, 3200), "white")
    for i, tile in enumerate(panels):
        sheet.paste(tile, ((i % 4) * 512, (i // 4) * 800))
    sheet.save(out / "camera_sweep.png")
    (out / "camera_sweep.json").write_text(json.dumps({"model": "unchanged static + moving; zero displacement", "cameras": cameras}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
