#!/usr/bin/env python3
"""Render one image for each verified real-tower pipeline stage.

This script deliberately consumes the paper tower checkpoints and the
notebook-aligned Step II outputs.  It does not create synthetic input data.
Step III is visualized from the already registered FE displacement stream;
the wind-field generator and ANSYS solver remain upstream inputs.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revision_experiments.dynamic_temporal_stability import (
    build_auto_camera,
    build_intrinsics,
    load_splats,
    render_rgb_alpha,
    save_png,
)


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_count(path: Path) -> int:
    state = torch_load(path)
    splats = state["splats"] if isinstance(state, dict) and "splats" in state else state
    return int(splats["means"].shape[0])


def add_caption(path: Path, title: str, subtitle: str) -> None:
    """Add a small English caption without changing the rendered scene."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return
    image = Image.open(path).convert("RGB")
    band_height = max(64, image.height // 14)
    canvas = Image.new("RGB", (image.width, image.height + band_height), "white")
    canvas.paste(image, (0, band_height))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(18, image.width // 42))
        small = ImageFont.truetype("DejaVuSans.ttf", max(13, image.width // 62))
    except Exception:
        font = small = ImageFont.load_default()
    draw.text((18, 8), title, fill=(20, 20, 20), font=font)
    draw.text((18, band_height // 2 + 3), subtitle, fill=(80, 80, 80), font=small)
    canvas.save(path)


def render_checkpoint(path: Path, output: Path, title: str, subtitle: str,
                      device: torch.device, camera_source=None) -> dict[str, object]:
    splats = load_splats(path, device)
    if camera_source is None:
        camera_source = splats
    c2w, camera_info = build_auto_camera(
        camera_source, camera_source, "combined", 0.0, -5.0, 0.32, None
    )
    width, height = 768, 1152
    K_np = build_intrinsics(width, height, 45.0)
    viewmat = torch.from_numpy(np.linalg.inv(c2w)).float().to(device)
    Kmat = torch.from_numpy(K_np).float().to(device)
    white = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
    rgb, _, _ = render_rgb_alpha(
        splats.means, splats.quats, splats.scales, splats.opacities,
        splats.colors, splats.sh_degree, viewmat, Kmat, width, height,
        3.0, 0.01, 1e10, white, False,
    )
    save_png(output, rgb)
    add_caption(output, title, subtitle)
    result = {
        "path": str(output),
        "source": str(path),
        "count": splats.count,
        "resolution": [width, height + max(64, height // 14)],
        "camera": camera_info,
    }
    del splats, rgb, viewmat, Kmat
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result


def displacement_plot(source_root: Path, output: Path, frame: int) -> dict[str, object]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    moving_path = source_root / "results" / "lingshi2_moving.pt"
    vibration_path = source_root / "results" / "lingshi2_moving_vibration.dat"
    state = torch_load(moving_path)
    splats = state["splats"] if isinstance(state, dict) and "splats" in state else state
    points = splats["means"].detach().cpu().numpy()
    count = len(points)
    total = vibration_path.stat().st_size // (4 * count * 3)
    if frame < 0 or frame >= total:
        raise ValueError(f"frame must be in [0, {total})")
    displacement = np.memmap(vibration_path, dtype=np.float32, mode="r", shape=(total, count, 3))[frame]
    magnitude = np.linalg.norm(displacement, axis=1)
    # Keep the figure responsive while preserving the full registered field.
    step = max(1, math.ceil(count / 60000))
    points = points[::step]
    magnitude = magnitude[::step]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=150, gridspec_kw={"width_ratios": [2.2, 1]})
    scatter = axes[0].scatter(points[:, 0], -points[:, 2], c=magnitude, s=1.2, cmap="turbo", linewidths=0)
    axes[0].set_title(f"Registered FE displacement field, frame {frame}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("-z (up)")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(alpha=0.2)
    fig.colorbar(scatter, ax=axes[0], label="displacement magnitude")
    axes[1].hist(magnitude, bins=60, color="#2b6cb0", alpha=0.85)
    axes[1].set_title("Magnitude distribution")
    axes[1].set_xlabel("displacement magnitude")
    axes[1].set_ylabel("Gaussian count")
    axes[1].grid(alpha=0.2)
    fig.suptitle("Step III input handoff: FE-to-Gaussian registered stream", fontsize=14)
    fig.text(0.5, 0.01, f"399 registered frames; frame {frame}; max norm={float(magnitude.max()):.6g}; ANSYS/wind solve not rerun here", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return {
        "path": str(output),
        "source_moving_checkpoint": str(moving_path),
        "source_vibration": str(vibration_path),
        "frame": frame,
        "frame_count": int(total),
        "moving_count": count,
        "max_norm_subsample": float(magnitude.max()),
        "status": "registered FE input visualization; solver not rerun",
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="WSL source root containing the paper tower checkpoints")
    parser.add_argument("--step-ii-dir", type=Path, required=True,
                        help="notebook-aligned Step II output directory")
    parser.add_argument("--final-dir", type=Path, required=True,
                        help="real-tower final demo directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fe-frame", type=int, default=37)
    parser.add_argument("--reuse-existing", action="store_true",
                        help="reuse stage images already present in the output directory")
    return parser


def main() -> int:
    args = make_parser().parse_args()
    source = args.source_root.expanduser().resolve()
    step_ii = args.step_ii_dir.expanduser().resolve()
    final = args.final_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    stages: list[dict[str, object]] = []
    # Step I is an existing paper checkpoint, not a retraining run in this release.
    stages.append(render_checkpoint(
        source / "yuanshi.pt", output / "01_step_I_initial_yuanshi.png",
        "Step I | Existing initial 3DGS checkpoint",
        "yuanshi.pt | 1,489,286 Gaussians | upstream gsplat/COLMAP result",
        device,
    ))
    step_ii_models = [
        ("__source_tu1__", "02_step_II_input_tower.png", "Step II | Input tower GS", "272,300 Gaussians | paper checkpoint"),
        ("adaptive/tu1_percentage_filtered_terrain_adaptive.pt", "03_step_II_adaptive_terrain.png", "Step II | Adaptive terrain separation", "181,706 terrain Gaussians | notebook-aligned rerun"),
        ("adaptive/tu1_percentage_filtered_objects_adaptive.pt", "04_step_II_adaptive_objects.png", "Step II | Non-terrain object partition", "90,594 object Gaussians | notebook-aligned rerun"),
        ("tu1_percentage_filtered_terrain_adaptive_kriging.pt", "05_step_II_kriging_terrain.png", "Step II | Ordinary Kriging terrain", "15,675 Kriging Gaussians | notebook-aligned rerun"),
        ("tu1_percentage_filtered_expanded.pt", "06_step_II_expanded_model.png", "Step II | Constrained multi-layer expansion", "917,407 Gaussians | notebook-aligned rerun"),
    ]
    for relative, filename, title, subtitle in step_ii_models:
        model_path = (source / "results" / "filter_schemes" / "percentage" / "tu1_percentage_filtered.pt") if relative == "__source_tu1__" else (step_ii / relative)
        image_path = output / filename
        if args.reuse_existing and image_path.is_file():
            stages.append({"path": str(image_path), "source": str(model_path),
                           "title": title, "reused_existing_image": True})
        else:
            stages.append(render_checkpoint(model_path, image_path, title, subtitle, device))

    stages.append(displacement_plot(source, output / f"07_step_III_registered_FE_frame_{args.fe_frame:03d}.png", args.fe_frame))

    before = final / "artifacts" / "gs_before.pt"
    after = final / "artifacts" / "gs_after_frame_00000.pt"
    before_splats = load_splats(before, device)
    stages.append(render_checkpoint(
        before, output / "08_step_IV_before_frame_00000.png",
        "Step IV | Before deformation", "272,300 Gaussians | physical FE displacement scale=1",
        device, before_splats,
    ))
    stages.append(render_checkpoint(
        after, output / "09_step_IV_after_frame_00000.png",
        "Step IV | After physical FE displacement", "272,300 Gaussians | frame 0 | scale=1",
        device, before_splats,
    ))
    del before_splats
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # Copy the already verified dynamic frames into the stage set with explicit names.
    import shutil
    frame_dir = final / "dynamic_render" / "frames"
    dynamic_entries = []
    for index in range(4):
        source_frame = frame_dir / f"frame_{index:05d}.png"
        target_frame = output / f"10_step_IV_dynamic_frame_{index:05d}.png"
        shutil.copy2(source_frame, target_frame)
        dynamic_entries.append({"path": str(target_frame), "source": str(source_frame), "frame": index})

    manifest = {
        "status": "passed",
        "description": "Real Yunnan straight-line tower stage images; no synthetic model used.",
        "stage_images": stages,
        "dynamic_frames": dynamic_entries,
        "boundaries": {
            "step_I": "existing gsplat/COLMAP checkpoint visualized; photos and COLMAP retraining are external",
            "step_II": "real tower checkpoint and notebook-aligned terrain/Kriging/expansion outputs were executed",
            "step_III": "registered FE displacement stream visualized; wind-field generation and ANSYS solve were not rerun",
            "step_IV": "before/after checkpoints and dynamic frames use the verified moving-mean renderer branch; full polar/quaternion/scale/covariance chain is not rerun by this script",
        },
    }
    (output / "stage_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
