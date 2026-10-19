#!/usr/bin/env python3
"""Run the shipped pipeline with small, entirely synthetic inputs.

This checks executable training, deformation, SfM diagnostics and CUDA rendering.
It is a functionality check, not reproduction of the manuscript's measurements.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_step(name, command, output, environment):
    print(f"[start] {name}", flush=True)
    started = time.perf_counter()
    log = output / f"{name}.log"
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=1200,
            check=False,
        )
    if process.returncode:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-35:])
        raise RuntimeError(f"{name} failed (exit {process.returncode}); log: {log}\n{tail}")
    print(f"[pass] {name} ({time.perf_counter() - started:.1f}s)", flush=True)
    return {"name": name, "exit_code": process.returncode, "log": str(log.relative_to(output))}


def build_synthetic_inputs(output):
    import numpy as np
    import torch
    from PIL import Image
    import gsplat
    from gsplat.rendering import rasterization

    package_path = Path(gsplat.__file__).resolve()
    if not package_path.is_relative_to(ROOT):
        raise RuntimeError(f"Wrong gsplat source loaded: {package_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("The training/rendering smoke test requires CUDA.")
    torch.manual_seed(7)
    x, y, z = np.meshgrid(
        np.linspace(0.0, 1.0, 12),
        [-0.04, -0.015, 0.015, 0.04],
        [-0.025, 0.025],
        indexing="ij",
    )
    points = np.column_stack([x.ravel(), y.ravel(), z.ravel()]).astype(np.float32)
    n = len(points)
    colors = np.column_stack([
        0.2 + 0.6 * points[:, 0],
        0.3 + 0.4 * (points[:, 1] > 0),
        0.2 + 0.5 * (points[:, 2] > 0),
    ]).astype(np.float32)
    splats = {
        "means": torch.from_numpy(points.copy()),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        "scales": torch.log(torch.tensor([[0.04, 0.02, 0.02]]).repeat(n, 1)),
        "opacities": torch.logit(torch.full((n,), 0.8)),
        "sh0": torch.from_numpy((colors - 0.5) / 0.28209479177387814)[:, None, :],
        "shN": torch.zeros((n, 15, 3)),
    }
    input_dir = output / "inputs"
    sparse_dir = input_dir / "colmap" / "sparse" / "0"
    image_dir = input_dir / "colmap" / "images"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"step": 0, "splats": splats}, input_dir / "initial.pt")
    static_splats = {key: value.clone() for key, value in splats.items()}
    static_splats["means"][:, 1] -= 0.35
    torch.save({"splats": static_splats}, input_dir / "static.pt")
    vibration = np.zeros((4, n, 3), dtype=np.float32)
    vibration[:, :, 1] = np.array([0.0, 0.01, 0.0, -0.01])[:, None] * points[None, :, 0]
    np.save(input_dir / "vibration.npy", vibration)

    width = height = 64
    intrinsic = torch.tensor([[60.0, 0.0, 32.0], [0.0, 60.0, 32.0], [0.0, 0.0, 1.0]], device="cuda")
    gpu = {key: value.cuda() for key, value in splats.items()}
    camera_offsets = [(-0.3, -0.2), (-0.2, 0.2), (-0.1, -0.2), (0.0, 0.2),
                      (0.1, -0.2), (0.2, 0.2), (0.3, -0.2), (0.4, 0.2)]
    with (sparse_dir / "cameras.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", 1))
        stream.write(struct.pack("<iiQQ4d", 1, 1, width, height, 60.0, 60.0, 32.0, 32.0))
    with (sparse_dir / "images.bin").open("wb") as stream, torch.no_grad():
        stream.write(struct.pack("<Q", len(camera_offsets)))
        for image_id, (offset_x, offset_y) in enumerate(camera_offsets, 1):
            translation = np.array([-0.5 - offset_x, -offset_y, 2.0])
            view = torch.eye(4, device="cuda")
            view[:3, 3] = torch.tensor(translation, dtype=torch.float32, device="cuda")
            rgb, alpha, _ = rasterization(
                gpu["means"], gpu["quats"], gpu["scales"].exp(), gpu["opacities"].sigmoid(),
                gpu["sh0"], view[None], intrinsic[None], width, height,
                sh_degree=0, radius_clip=0.0,
            )
            image = ((rgb[0] + 1.0 - alpha[0]).clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
            name = f"frame_{image_id:03d}.png"
            Image.fromarray(image).save(image_dir / name)
            stream.write(struct.pack("<i4d3di", image_id, 1.0, 0.0, 0.0, 0.0, *translation, 1))
            stream.write(name.encode("utf-8") + b"\0")
            stream.write(struct.pack("<Q", n))
            camera_points = points + translation
            pixels = camera_points[:, :2] / camera_points[:, 2:3] * 60 + 32
            for point_index, pixel in enumerate(pixels):
                stream.write(struct.pack("<ddq", *pixel, point_index + 1))
    with (sparse_dir / "points3D.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", n))
        for point_index, (point, color) in enumerate(zip(points, colors)):
            stream.write(struct.pack("<Q3d3BdQ", point_index + 1, *point, *(color * 255).astype(np.uint8), 0.0, len(camera_offsets)))
            for image_id in range(1, len(camera_offsets) + 1):
                stream.write(struct.pack("<ii", image_id, point_index))

    # The merge script's edge endpoints are zero-based row indices, not FE IDs.
    beam_x = np.linspace(0.0, 1.0, 24)
    beam_points = np.zeros((24, 7))
    beam_points[:, 0] = np.arange(24)
    beam_points[:, 1] = beam_x
    beam_points[:, 4] = 0.01
    beam_points[:, 5] = 0.03 * beam_x
    beam_edges = np.column_stack([np.arange(23), np.arange(23), np.arange(1, 24), np.zeros(23)])
    np.savetxt(input_dir / "beam_points.txt", beam_points)
    np.savetxt(input_dir / "beam_edges.txt", beam_edges, fmt="%d")
    return input_dir, package_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "synthetic_smoke")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + str(ROOT / "examples")
    environment["MAX_JOBS"] = environment.get("MAX_JOBS", "2")
    environment["GSPLAT_3DGS_ONLY"] = "1"
    environment["CUDA_HOME"] = environment.get("CUDA_HOME", "/usr/local/cuda-11.8")
    environment["PATH"] = environment["CUDA_HOME"] + "/bin" + os.pathsep + environment.get("PATH", "")
    os.environ.update(environment)
    started = time.perf_counter()
    inputs, package_path = build_synthetic_inputs(output)
    steps = []
    python = sys.executable
    roi = output / "roi_cache.npz"
    fixtures = ROOT / "tests" / "fixtures"
    steps.append(run_step("roi_cache", [python, "examples/build_fe_irregular_roi_cache.py",
        "--node-file", str(fixtures / "nodes.txt"), "--elem-file", str(fixtures / "elements.txt"),
        "--section-file", str(fixtures / "sections.csv"), "--transform-file", str(fixtures / "identity_transform.txt"),
        "--out-ply", str(output / "roi.ply"), "--out-npz", str(output / "roi_points.npz"),
        "--out-cache", str(roi), "--roi-radius", "0.08"], output, environment))
    baseline_dir = output / "baseline_training"
    steps.append(run_step("baseline_training", [python, "examples/simple_trainer.py", "default",
        "--data-dir", str(inputs / "colmap"), "--data-factor", "1",
        "--result-dir", str(baseline_dir), "--max-steps", "2",
        "--save-steps", "2", "--eval-steps", "999999", "--disable-viewer"], output, environment))
    baseline_checkpoint = baseline_dir / "ckpts" / "ckpt_1_rank0.pt"
    training_environment = environment | {
        "CKPT_PATH": str(baseline_checkpoint), "DATA_DIR": str(inputs / "colmap"),
        "RESULT_DIR": str(output / "training"), "ROI_CACHE_PATH": str(roi),
        "DATA_FACTOR": "1", "NORMALIZE_WORLD_SPACE": "0", "MAX_STEPS": "8", "SAVE_STEPS": "8",
        "REFINE_START_ITER": "1", "REFINE_STOP_ITER": "8", "REFINE_EVERY": "2", "RESET_EVERY": "0",
        "LOAD_OPTIMIZER_STATE": "0", "LOAD_STRATEGY_STATE": "0",
    }
    steps.append(run_step("roi_training", [python, "examples/train_irregular_region_filter.py"], output, training_environment))
    steps.append(run_step("fe_deformation", [python, "merged_gaussian_transform.py",
        "--points-file", str(inputs / "beam_points.txt"), "--lines-file", str(inputs / "beam_edges.txt"),
        "--orig-model-out", str(output / "fe_original.pt"), "--deformed-model-out", str(output / "fe_deformed.pt"),
        "--transform-out", str(output / "fe_transform.npz"), "--device", "cpu"], output, environment))
    steps.append(run_step("anisotropic_mapping", [python, "apply_transform_with_anisotropic_kernel.py",
        "--filtered-pt-file", str(inputs / "initial.pt"), "--original-pt-file", str(output / "fe_original.pt"),
        "--transform-file", str(output / "fe_transform.npz"), "--output-pt-file", str(output / "mapped.pt"),
        "--output-npz-file", str(output / "mapped_transform.npz"), "--device", "cpu", "--no-keops",
        "--k-neighbors", "5", "--batch-size", "32"], output, environment))
    steps.append(run_step("sfm_diagnostics", [python, "revision_experiments/sfm_quality_group_diagnostics.py",
        "--image-root", str(inputs / "colmap" / "images"), "--sparse-dir", str(inputs / "colmap" / "sparse" / "0"),
        "--output-dir", str(output / "sfm"), "--source-stratified-min-images", "1"], output, environment))
    steps.append(run_step("dynamic_rendering", [python, "revision_experiments/dynamic_temporal_stability.py",
        "--static_ckpt", str(inputs / "static.pt"), "--moving_ckpt", str(output / "mapped.pt"),
        "--vib_path", str(inputs / "vibration.npy"), "--scale_disp", "1", "--resolution", "64x64",
        "--frames", "4", "--radius_clip", "0", "--device", "cuda", "--save_frames",
        "--output_dir", str(output / "dynamic")], output, environment))
    import numpy as np
    import torch
    trained_path = output / "training" / "ckpts" / "ckpt_7_rank0.pt"
    baseline = torch.load(baseline_checkpoint, map_location="cpu")
    trained = torch.load(trained_path, map_location="cpu")
    if baseline["step"] != 1:
        raise RuntimeError("Baseline 3DGS training did not reach its requested checkpoint step.")
    if trained["step"] != 7:
        raise RuntimeError("Training did not reach its requested final step.")
    for values in baseline["splats"].values():
        if not torch.isfinite(values).all():
            raise RuntimeError("Baseline 3DGS training produced non-finite model parameters.")
    for values in trained["splats"].values():
        if not torch.isfinite(values).all():
            raise RuntimeError("Training produced non-finite model parameters.")
    mapped = torch.load(output / "mapped.pt", map_location="cpu")["splats"]
    initial = torch.load(inputs / "initial.pt", map_location="cpu")["splats"]
    if torch.allclose(mapped["means"], initial["means"]):
        raise RuntimeError("Non-zero synthetic FE displacement did not move the Gaussians.")
    for values in mapped.values():
        if not torch.isfinite(values).all():
            raise RuntimeError("Mapping produced non-finite model parameters.")
    transform = np.load(output / "mapped_transform.npz")
    if not np.isfinite(transform["displacement_vectors"]).all():
        raise RuntimeError("Mapping produced non-finite displacement.")
    versions = {}
    for name in ("torch", "numpy", "scipy", "scikit-learn", "Pillow", "tyro", "torchmetrics", "pycolmap", "viser", "nerfview", "tensorly", "fused-ssim"):
        versions[name] = importlib.metadata.version(name)
    summary = {
        "status": "passed", "scope": "synthetic functionality smoke test; no paper metrics reproduced",
        "python": sys.version, "gpu": torch.cuda.get_device_name(0), "versions": versions,
        "gsplat_source": str(package_path), "steps": steps,
        "baseline_checkpoint": str(baseline_checkpoint.relative_to(output)),
        "baseline_final_step": int(baseline["step"]),
        "baseline_gaussian_count": len(baseline["splats"]["means"]),
        "trained_checkpoint": str(trained_path.relative_to(output)),
        "training_final_step": int(trained["step"]), "gaussian_count": len(trained["splats"]["means"]),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
