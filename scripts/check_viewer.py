#!/usr/bin/env python3
"""Check the real Viser service and viewer callback using synthetic inputs only.

Run scripts/run_synthetic_smoke.py first. --device cpu --no-render checks
initialization and HTTP serving without compiling or calling the CUDA renderer.
The default also renders several frames through the actual viewer callback.
This does not automate a browser or reproduce the manuscript's performance.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GSPLAT_3DGS_ONLY", "1")
os.environ.setdefault("MAX_JOBS", "2")

import numpy as np
import torch
import nerfview
from PIL import Image

import auto_play_highres as playback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "outputs/synthetic_smoke/inputs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/viewer_smoke")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    if not args.no_render and (args.device != "cuda" or not torch.cuda.is_available()):
        parser.error("Callback rendering requires CUDA; use --device cpu --no-render for API/HTTP checks.")
    inputs = args.input_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    moving_checkpoint = inputs / "initial.pt"
    candidate = ROOT / "outputs/synthetic_smoke/mapped.pt"
    if candidate.exists():
        moving_checkpoint = candidate
    playback.initialize_scene(inputs / "static.pt", moving_checkpoint,
                              inputs / "vibration.npy", scale_disp=1.0, device=args.device)
    if playback.T < 2:
        raise RuntimeError("The synthetic viewer smoke requires at least two vibration frames.")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server, viewer, fps, frame_text = playback.create_viewer(160, 120, port=port)
    summary = {
        "status": "passed", "scope": "synthetic viewer API/HTTP and optional callback check; no browser automation",
        "device": args.device, "rendered": False,
        "versions": {name: importlib.metadata.version(name) for name in ("viser", "nerfview", "torch")},
        "frames": int(playback.T), "moving_gaussians": int(playback.ms_m.shape[0]),
        "moving_checkpoint": str(moving_checkpoint),
    }
    try:
        viewer.update(5, 1)
        frame_text.value = "1"
        server.flush()
        url = f"http://127.0.0.1:{server.get_port()}/"
        with urllib.request.urlopen(url, timeout=10) as response:
            body = response.read()
            if response.status != 200 or b"<html" not in body.lower():
                raise RuntimeError("The real Viser server did not serve its HTML client.")
            summary["http_status"] = response.status
            summary["http_bytes"] = len(body)
        if not args.no_render:
            playback.bench_stats = playback.RenderBenchmark(str(output / "callback_timing.json"), window=3)
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, 3] = [0.5, -0.175, -1.2]
            camera = nerfview.CameraState(fov=0.8, aspect=160 / 120, c2w=c2w)
            # Exercise both API generations using the real callback and real CUDA.
            frame0 = viewer.render_fn(camera, (160, 120))
            viewer.render_tab_state.viewer_width = 128
            viewer.render_tab_state.viewer_height = 96
            smaller = viewer.render_fn(camera, viewer.render_tab_state)
            playback.cur = 1
            frame1 = viewer.render_fn(camera, (160, 120))
            for image, shape in ((frame0, (120, 160, 3)), (smaller, (96, 128, 3)),
                                 (frame1, (120, 160, 3))):
                if image.shape != shape or image.dtype != np.uint8 or not np.isfinite(image).all():
                    raise RuntimeError(f"Unexpected viewer image shape/dtype: {image.shape}, {image.dtype}.")
                if not np.any(image < 250):
                    raise RuntimeError("Viewer returned an empty white image.")
            changed = int(np.count_nonzero(frame0 != frame1))
            if not changed:
                raise RuntimeError("Non-zero synthetic vibration did not change the rendered frame.")
            Image.fromarray(frame0).save(output / "frame_0.png")
            Image.fromarray(frame1).save(output / "frame_1.png")
            Image.fromarray(smaller).save(output / "frame_small.png")
            summary.update(rendered=True, callback_sizes=[[160, 120], [128, 96]], changed_channels=changed,
                           non_white_pixels=int(np.count_nonzero(np.any(frame0 < 250, axis=2))))
    finally:
        server.stop()
        playback.bench_stats = None
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
