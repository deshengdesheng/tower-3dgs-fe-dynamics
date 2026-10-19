#!/usr/bin/env python3
"""Run the dynamic gsplat renderer from an exported runtime directory.

The real-tower pipeline writes ``artifacts/dynamic_gs/runtime_manifest.json``
alongside self-contained static/moving checkpoints and the FE displacement
stream.  This small entry point is intentionally separate from the validation
runner so a reviewer can rerender frames without revalidating the private
source tree.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
RENDERER = REPO_ROOT / "revision_experiments" / "dynamic_temporal_stability.py"


def _path_from_manifest(runtime_dir: Path, payload: dict[str, object], key: str, filename: str) -> Path:
    relative_key = f"{key}_relative"
    relative = payload.get(relative_key)
    if isinstance(relative, str) and relative:
        candidate = (runtime_dir / relative).resolve()
        if candidate.is_file():
            return candidate
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        return (runtime_dir / filename).resolve()
    candidate = Path(value)
    return candidate if candidate.is_file() else (runtime_dir / filename).resolve()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True,
                        help="directory containing runtime_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--resolution", default="768x1152")
    parser.add_argument("--scale-disp", type=float, default=50.0)
    parser.add_argument("--radius-clip", type=float, default=0.0,
                        help="skip projected Gaussians smaller than this pixel radius; 0 keeps fine tower members")
    parser.add_argument("--target-mode", choices=("moving", "static", "combined"), default="moving")
    # Historical view defaults remain available. The continuous-tower launcher
    # supplies explicit framing and its registered physical-up transform.
    parser.add_argument("--azimuth-deg", type=float, default=0.0)
    parser.add_argument("--elevation-deg", type=float, default=-5.0)
    parser.add_argument("--distance-scale", type=float, default=0.32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--renderer-python", type=Path, default=None,
                        help="optional Python executable for the gsplat renderer (useful when the training env lacks the compiled CUDA extension)")
    parser.add_argument("--gsplat-prebuilt", type=Path, default=None,
                        help="optional prebuilt gsplat_cuda.so loaded by the renderer process")
    parser.add_argument("--gs-to-fe-transform", type=Path, default=None,
                        help="optional Gaussian-to-FE transform; its FE +Z axis defines camera up")
    parser.add_argument("--up-vector", default="0,0,1",
                        help="camera up vector in Gaussian coordinates, used if no transform is supplied")
    parser.add_argument("--flip-vertical", action="store_true")
    parser.add_argument("--save-moving-frames", action="store_true",
                        help="also render the moving Gaussian subset with the same camera")
    parser.add_argument("--save-undeformed", action="store_true",
                        help="also render true zero-displacement reference images; archived frame 0 can be nonzero")
    return parser


def main() -> int:
    args = make_parser().parse_args()
    runtime_dir = args.runtime_dir.expanduser().resolve()
    manifest_path = runtime_dir / "runtime_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"runtime manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    static = _path_from_manifest(runtime_dir, payload, "static_checkpoint", "static.pt")
    moving = _path_from_manifest(runtime_dir, payload, "moving_checkpoint", "moving.pt")
    vibration = _path_from_manifest(runtime_dir, payload, "vibration_file", "moving_vibration.dat")
    for path in (static, moving, vibration):
        if not path.is_file():
            raise FileNotFoundError(f"runtime input not found: {path}")
    output = args.output_dir.expanduser().resolve() if args.output_dir else runtime_dir.parent.parent / "dynamic_render_rerun"
    renderer_python = args.renderer_python.expanduser().resolve() if args.renderer_python else Path(sys.executable)
    if not renderer_python.is_file():
        raise FileNotFoundError(f"renderer Python executable not found: {renderer_python}")
    command = [
        str(renderer_python),
        str(RENDERER),
        "--static_ckpt", str(static),
        "--moving_ckpt", str(moving),
        "--vib_path", str(vibration),
        "--scale_disp", str(args.scale_disp),
        "--radius_clip", str(args.radius_clip),
        "--resolution", args.resolution,
        "--frames", str(args.frames),
        "--frame_start", str(args.frame_start),
        "--frame_stride", str(args.frame_stride),
        "--target_mode", args.target_mode,
        "--azimuth_deg", str(args.azimuth_deg),
        "--elevation_deg", str(args.elevation_deg),
        "--distance_scale", str(args.distance_scale),
        "--save_frames",
        "--output_dir", str(output),
        "--device", args.device,
        "--up_vector", args.up_vector,
    ]
    if args.gs_to_fe_transform is not None:
        command.extend(["--gs_to_fe_transform", str(args.gs_to_fe_transform.expanduser().resolve())])
    if args.flip_vertical:
        command.append("--flip_vertical")
    if args.save_moving_frames:
        command.append("--save_moving_frames")
    if args.save_undeformed:
        command.append("--save_undeformed")
    print("[run]", " ".join(command), flush=True)
    environment = os.environ.copy()
    if args.gsplat_prebuilt is not None:
        extension = args.gsplat_prebuilt.expanduser().resolve()
        if not extension.is_file():
            raise FileNotFoundError(f"prebuilt gsplat extension not found: {extension}")
        environment["GSPLAT_PREBUILT"] = str(extension)
    completed = subprocess.run(command, cwd=REPO_ROOT, env=environment, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
