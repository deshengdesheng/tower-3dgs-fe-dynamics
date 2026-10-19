#!/usr/bin/env python3
"""Run the reproducible real-tower handoff from existing 3DGS checkpoints.

This entry point deliberately starts with the paper's tower checkpoint and
FE-registered ROI cache. The bundled ``gsplat`` implementation is used for
the existing checkpoint format and the dynamic off-screen renderer.

The default source layout is the layout used by the local tower case::

    <source-root>/yuanshi.pt
    <source-root>/results/filter_schemes/percentage/tu1_percentage_filtered.pt
    <source-root>/results/lingshi2.pt
    <source-root>/results/lingshi2_static.pt
    <source-root>/results/lingshi2_moving.pt
    <source-root>/results/lingshi2_moving_vibration.dat
    <source-root>/roi_fe_beam_exact_capsule_merged_cache_3dgs.npz

The ``lingshi2`` checkpoint is the already PLY/KD-tree-filtered tower subset;
the raw ``yuanshi`` checkpoint is only inventoried. ROI membership is measured
for the summary and is not used to filter rendered splats. The renderer gets
the static/moving split and applies the registered displacement frame-by-frame.
The single-frame deformed checkpoint written here is an audit artifact, not a
renderer input.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RENDER_SCRIPT = ROOT / "revision_experiments" / "dynamic_temporal_stability.py"


def _normalise_external_path(value: str | os.PathLike[str]) -> Path:
    """Accept Windows/WSL paths when this script runs on either side."""
    text = os.path.expandvars(os.path.expanduser(str(value).strip()))
    lower = text.lower().replace("\\", "/")
    prefixes = (
        "//wsl.localhost/ubuntu20045/",
        "//wsl$/ubuntu20045/",
    )
    if os.name != "nt":
        matched_wsl = False
        for prefix in prefixes:
            if lower.startswith(prefix):
                text = "/" + text.replace("\\", "/").split("/", 4)[-1]
                matched_wsl = True
                break
        if not matched_wsl and len(text) >= 3 and text[1] == ":" and text[2] in "/\\":
            text = "/mnt/" + text[0].lower() + "/" + text[3:].replace("\\", "/")
    return Path(text).expanduser().resolve()


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.6
        return torch.load(path, map_location="cpu")


def _splats_from_state(state: Any) -> dict[str, Any]:
    if isinstance(state, dict) and "splats" in state:
        state = state["splats"]
    if not isinstance(state, dict):
        raise ValueError("checkpoint must contain a mapping or a 'splats' mapping")
    required = ("means", "quats", "scales", "opacities", "sh0", "shN")
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"checkpoint is missing Gaussian fields: {missing}")
    return state


def _checkpoint_summary(path: Path, roi_mask: Callable[[torch.Tensor], torch.Tensor] | None = None) -> dict[str, Any]:
    """Validate one checkpoint and optionally calculate its ROI statistics."""
    state = _torch_load(path)
    splats = _splats_from_state(state)
    means = splats["means"].detach().cpu().float()
    count = int(means.shape[0])
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError(f"{path} has means with invalid shape {tuple(means.shape)}")
    finite = bool(torch.isfinite(means).all())
    for key in ("quats", "scales", "opacities", "sh0", "shN"):
        value = splats[key]
        if not torch.is_tensor(value) or value.shape[0] != count:
            raise ValueError(f"{path} field {key} is not aligned with means")
        finite = finite and bool(torch.isfinite(value.detach()).all())
    if not finite:
        raise ValueError(f"{path} contains non-finite Gaussian parameters")

    result: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "count": count,
        "parameter_shapes": {key: list(splats[key].shape) for key in ("means", "quats", "scales", "opacities", "sh0", "shN")},
        "finite": finite,
        "means_min": means.min(dim=0).values.tolist(),
        "means_max": means.max(dim=0).values.tolist(),
    }
    if roi_mask is not None:
        mask = roi_mask(means)
        if mask.shape != (count,):
            raise ValueError("ROI mask has an invalid shape")
        opacity = torch.sigmoid(splats["opacities"].detach().cpu().float().reshape(count, -1)[:, 0])
        result.update({
            "roi_count": int(mask.sum().item()),
            "roi_fraction": float(mask.float().mean().item()),
            "roi_opacity_sum": float(opacity[mask].sum().item()),
            "roi_opacity_ge_0.005": int((mask & (opacity >= 0.005)).sum().item()),
        })
    del state, splats, means
    gc.collect()
    return result


def _find_existing(root: Path, candidates: list[str], label: str, required: bool) -> Path | None:
    for relative in candidates:
        candidate = (root / relative).resolve()
        if candidate.is_file():
            return candidate
    if required:
        names = ", ".join(str(root / item) for item in candidates)
        raise FileNotFoundError(f"missing {label}; checked: {names}")
    return None


def _load_roi(roi_path: Path) -> tuple[dict[str, Any], np.lib.npyio.NpzFile]:
    data = np.load(roi_path, allow_pickle=False)
    required = ("roi_hash", "roi_origin", "roi_dims", "roi_radius", "roi_voxel_size")
    missing = [key for key in required if key not in data]
    if missing:
        data.close()
        raise ValueError(f"ROI cache is missing fields: {missing}")
    mode = str(np.asarray(data["roi_build_mode"]).reshape(()).item()) if "roi_build_mode" in data else "voxel"
    info: dict[str, Any] = {
        "path": str(roi_path),
        "size_bytes": roi_path.stat().st_size,
        "mode": mode,
        "hash_count": int(np.unique(np.asarray(data["roi_hash"], dtype=np.int64)).size),
        "roi_origin": np.asarray(data["roi_origin"], dtype=float).tolist(),
        "roi_dims": np.asarray(data["roi_dims"], dtype=int).tolist(),
        "roi_radius": float(np.asarray(data["roi_radius"]).reshape(())),
        "roi_voxel_size": float(np.asarray(data["roi_voxel_size"]).reshape(())),
    }
    if "source_min" in data:
        info["source_min"] = np.asarray(data["source_min"], dtype=float).tolist()
    if "source_max" in data:
        info["source_max"] = np.asarray(data["source_max"], dtype=float).tolist()
    if mode == "exact-capsule":
        info["capsule_segment_count"] = int(len(data["capsule_starts"]))
        info["capsule_cell_count"] = int(len(data["capsule_cell_keys"]))
    return info, data


def _voxel_roi_mask_factory(roi_data: np.lib.npyio.NpzFile):
    origin = torch.from_numpy(np.asarray(roi_data["roi_origin"], dtype=np.float32))
    dims = torch.from_numpy(np.asarray(roi_data["roi_dims"], dtype=np.int64))
    voxel = float(np.asarray(roi_data["roi_voxel_size"]).reshape(()))
    hashes = torch.from_numpy(np.unique(np.asarray(roi_data["roi_hash"], dtype=np.int64)))

    def mask(means: torch.Tensor) -> torch.Tensor:
        voxels = torch.floor((means - origin) / voxel).long()
        valid = torch.all((voxels >= 0) & (voxels < dims), dim=1)
        output = torch.zeros(len(means), dtype=torch.bool)
        if valid.any():
            keys = (voxels[valid, 0] * dims[1] + voxels[valid, 1]) * dims[2] + voxels[valid, 2]
            positions = torch.searchsorted(hashes, keys)
            clamped = positions.clamp(max=max(len(hashes) - 1, 0))
            output[torch.where(valid)[0]] = (positions < len(hashes)) & (hashes[clamped] == keys)
        return output

    return mask


def _exact_roi_mask_factory(roi_path: Path):
    """Use the same exact-capsule predicate as ROI training when available."""
    sys.path.insert(0, str(ROOT))
    from gsplat.strategy.irregular_region_strategy import IrregularRegionFilteredStrategy

    # The strategy's cache loader prints Unicode status symbols.  Suppress that
    # incidental output so Windows consoles using a legacy code page do not
    # make an otherwise valid validation run fail with UnicodeEncodeError.
    with contextlib.redirect_stdout(io.StringIO()):
        strategy = IrregularRegionFilteredStrategy.from_cache(str(roi_path), verbose_region_every=0)

    def mask(means: torch.Tensor) -> torch.Tensor:
        return strategy._get_region_mask(means)

    return mask


def _metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _apply_vibration_frame(
    moving_path: Path,
    vibration_path: Path,
    moving_count: int,
    frame_count: int,
    frame_index: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Apply one FE-derived displacement frame to the moving Gaussian model.

    The raw stream is the paper pipeline's already registered FE-to-Gaussian
    displacement field.  A memory map keeps the 399-frame file out of RAM and
    makes the generated checkpoint an auditable handoff to the renderer.
    """
    if not 0 <= frame_index < frame_count:
        raise ValueError(f"deformation frame {frame_index} is outside [0, {frame_count})")
    state = _torch_load(moving_path)
    splats = _splats_from_state(state)
    base_means = splats["means"].detach().cpu().float()
    if len(base_means) != moving_count:
        raise ValueError("moving checkpoint count changed during deformation stage")
    mapped = np.memmap(
        vibration_path,
        mode="r",
        dtype=np.float32,
        shape=(frame_count, moving_count, 3),
    )
    displacement = np.asarray(mapped[frame_index], dtype=np.float32)
    if not np.isfinite(displacement).all():
        raise ValueError("FE-derived vibration frame contains non-finite values")
    displacement_t = torch.from_numpy(displacement.copy())
    deformed = {key: value.detach().cpu().clone() if torch.is_tensor(value) else value for key, value in splats.items()}
    deformed["means"] = base_means + displacement_t
    deformation_dir = output_dir / "deformation"
    deformation_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = deformation_dir / f"moving_deformed_frame_{frame_index:05d}.pt"
    torch.save({"splats": deformed}, checkpoint_path)
    displacement_path = deformation_dir / f"frame_{frame_index:05d}_displacement.npy"
    np.save(displacement_path, displacement)
    norms = np.linalg.norm(displacement, axis=1)
    return {
        "status": "passed",
        "source_checkpoint": str(moving_path),
        "source_vibration": str(vibration_path),
        "frame_index": int(frame_index),
        "moving_count": int(moving_count),
        "displacement_shape": [int(x) for x in displacement.shape],
        "displacement_min": displacement.min(axis=0).astype(float).tolist(),
        "displacement_max": displacement.max(axis=0).astype(float).tolist(),
        "displacement_mean_norm": float(norms.mean()),
        "displacement_max_norm": float(norms.max()),
        "deformed_checkpoint": str(checkpoint_path),
        "displacement_npy": str(displacement_path),
        "mapping_provenance": "precomputed FE-to-Gaussian registration from the paper tower pipeline",
        "artifact_role": "single-frame audit snapshot; not consumed by the dynamic renderer",
        "used_by_renderer": False,
    }


def _load_cpu_splats(path: Path) -> dict[str, Any]:
    """Load a checkpoint into CPU tensors for deterministic artifact export."""
    state = _torch_load(path)
    splats = _splats_from_state(state)
    result: dict[str, Any] = {}
    for key, value in splats.items():
        result[key] = value.detach().cpu().clone() if torch.is_tensor(value) else value
    return result


def _save_combined_checkpoint(
    static_path: Path,
    moving_path: Path,
    output_path: Path,
    moving_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one standard gsplat checkpoint from static + moving splats.

    The notebook pipeline renders these two partitions together.  Exporting a
    combined before/after pair preserves a one-to-one ordering between the
    moving points in the two snapshots, which is more useful for inspection
    than exposing only the moving partition.
    """
    static = _load_cpu_splats(static_path)
    moving = moving_override if moving_override is not None else _load_cpu_splats(moving_path)
    keys = ("means", "quats", "scales", "opacities", "sh0", "shN")
    missing = [key for key in keys if key not in static or key not in moving]
    if missing:
        raise ValueError(f"cannot combine checkpoints; missing fields: {missing}")
    combined: dict[str, Any] = {}
    for key in keys:
        left, right = static[key], moving[key]
        if not torch.is_tensor(left) or not torch.is_tensor(right):
            raise ValueError(f"checkpoint field {key} is not a tensor")
        if left.ndim != right.ndim or left.shape[1:] != right.shape[1:]:
            raise ValueError(f"checkpoint field {key} has incompatible shapes: {left.shape}, {right.shape}")
        combined[key] = torch.cat((left, right), dim=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"splats": combined}, output_path)
    return {
        "path": str(output_path),
        "size_bytes": output_path.stat().st_size,
        "count": int(combined["means"].shape[0]),
        "static_count": int(static["means"].shape[0]),
        "moving_count": int(moving["means"].shape[0]),
        "format": "torch checkpoint with top-level splats mapping",
    }


def _write_dynamic_runtime_manifest(
    output_dir: Path,
    paths: dict[str, Path],
    vibration_info: dict[str, Any],
    render: dict[str, Any] | None,
    copy_inputs: bool = True,
) -> dict[str, Any]:
    """Export the dynamic GS runtime inputs and an executable command record.

    The vibration stream is intentionally copied byte-for-byte when
    ``copy_inputs`` is enabled, so the output directory can be handed off
    without access to the private WSL source tree.  A small manifest records
    the exact point count and frame shape needed to open the memmap.
    """
    runtime_dir = output_dir / "artifacts" / "dynamic_gs"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    dynamic_paths: dict[str, str] = {}
    for key, filename in (("static", "static.pt"), ("moving", "moving.pt"), ("vibration", "moving_vibration.dat")):
        source = paths[key]
        destination = runtime_dir / filename
        if copy_inputs:
            shutil.copy2(source, destination)
            dynamic_paths[key] = str(destination)
        else:
            dynamic_paths[key] = str(source)

    relative_paths = {
        "static_checkpoint_relative": "static.pt",
        "moving_checkpoint_relative": "moving.pt",
        "vibration_file_relative": "moving_vibration.dat",
    }

    runtime = {
        "format": "paper-tower-dynamic-gs-runtime-v1",
        "description": "Static and moving gsplat partitions plus the registered FE displacement stream.",
        "static_checkpoint": dynamic_paths["static"],
        "moving_checkpoint": dynamic_paths["moving"],
        "vibration_file": dynamic_paths["vibration"],
        **relative_paths,
        "vibration_dtype": "float32",
        "vibration_shape": vibration_info.get("shape"),
        "renderer": str(RENDER_SCRIPT),
        "runtime_runner": str(ROOT / "scripts" / "run_dynamic_gs.py"),
        "renderer_command": render.get("command") if render else None,
        "copy_inputs": bool(copy_inputs),
        "source_inputs": {key: str(paths[key]) for key in ("static", "moving", "vibration")},
        "notes": [
            "The renderer applies one vibration frame to moving means and concatenates static + moving splats.",
            "The raw yuanshi checkpoint and the ROI cache are not runtime renderer inputs.",
        ],
    }
    manifest_path = runtime_dir / "runtime_manifest.json"
    manifest_path.write_text(json.dumps(runtime, indent=2, ensure_ascii=False), encoding="utf-8")
    readme_path = runtime_dir / "README.md"
    try:
        runtime_argument = runtime_dir.relative_to(ROOT).as_posix()
    except ValueError:
        runtime_argument = str(runtime_dir).replace("\\", "/")
    readme_path.write_text(
        "# Dynamic GS runtime input package\n\n"
        "This directory contains the static and moving gsplat partitions and the "
        "registered FE displacement stream used by the paper-tower renderer.\n\n"
        "From the repository root, run:\n\n"
        "```bash\n"
        "GSPLAT_3DGS_ONLY=1 CUDA_HOME=/usr/local/cuda-11.8 \\\n"
        "python -B scripts/run_dynamic_gs.py \\\n"
        f"  --runtime-dir {runtime_argument} \\\n"
        "  --frames 4 --resolution 768x1152 --device cuda\n"
        "```\n\n"
        "The before/after full checkpoints are stored one directory above.\n",
        encoding="utf-8",
    )
    return {
        "directory": str(runtime_dir),
        "manifest": str(manifest_path),
        "readme": str(readme_path),
        "static_checkpoint": dynamic_paths["static"],
        "moving_checkpoint": dynamic_paths["moving"],
        "vibration_file": dynamic_paths["vibration"],
        "copied": bool(copy_inputs),
    }


def _parse_resolution(value: str) -> str:
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except Exception as exc:
        raise argparse.ArgumentTypeError("resolution must be WxH, e.g. 768x1152") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("resolution dimensions must be positive")
    return f"{width}x{height}"


def _run_renderer(args: argparse.Namespace, paths: dict[str, Path], output: Path) -> dict[str, Any]:
    render_output = output / "dynamic_render"
    render_output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(RENDER_SCRIPT),
        "--static_ckpt", str(paths["static"]),
        "--moving_ckpt", str(paths["moving"]),
        "--vib_path", str(paths["vibration"]),
        "--scale_disp", str(args.scale_disp),
        "--resolution", args.resolution,
        "--frames", str(args.frames),
        "--frame_start", str(args.frame_start),
        "--frame_stride", str(args.frame_stride),
        "--target_mode", args.target_mode,
        "--azimuth_deg", str(args.azimuth_deg),
        "--elevation_deg", str(args.elevation_deg),
        "--distance_scale", str(args.distance_scale),
        "--save_frames",
        "--output_dir", str(render_output),
        "--device", args.device,
    ]
    if args.flip_vertical:
        command.append("--flip_vertical")
    log_path = output / "dynamic_render.log"
    environment = os.environ.copy()
    if os.name != "nt":
        cuda_home = environment.get("CUDA_HOME") or "/usr/local/cuda-11.8"
        environment["CUDA_HOME"] = cuda_home
        environment["PATH"] = str(Path(cuda_home) / "bin") + os.pathsep + environment.get("PATH", "")
        # The bundled source is intentionally run in its 3DGS-only build mode;
        # inheriting an unrelated value (for example ``0``) can leave the
        # lazy CUDA extension unset in a fresh child process.
        environment["GSPLAT_3DGS_ONLY"] = "1"
        environment.setdefault("MAX_JOBS", "2")
    environment.setdefault("PYTHONUNBUFFERED", "1")
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
    if process.returncode:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
        raise RuntimeError(f"dynamic renderer failed with exit code {process.returncode}\n{tail}")
    metrics_path = render_output / "dynamic_temporal_stability.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    return {
        "exit_code": process.returncode,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "command": command,
        "log": str(log_path),
        "output_dir": str(render_output),
        "metrics_file": str(metrics_path),
        "frame_count": len(list((render_output / "frames").glob("*.png"))),
        "metrics_summary": metrics.get("summary", {}),
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=_normalise_external_path, required=True,
                        help="source-root containing the existing tower checkpoints and ROI cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "real_tower_pipeline")
    parser.add_argument("--yuanshi-checkpoint", type=Path, default=None)
    parser.add_argument("--tu1-checkpoint", type=Path, default=None)
    parser.add_argument("--lingshi2-checkpoint", type=Path, default=None)
    parser.add_argument("--static-checkpoint", type=Path, default=None)
    parser.add_argument("--moving-checkpoint", type=Path, default=None)
    parser.add_argument("--vibration", type=Path, default=None)
    parser.add_argument("--roi-cache", type=Path, default=None)
    parser.add_argument("--scale-disp", type=float, default=50.0)
    # These are the validated tower framing settings.  The old 256x256 default
    # was too small for the scene and made the tower appear as an unreadable
    # blur in the handoff output.
    parser.add_argument("--resolution", type=_parse_resolution, default="768x1152")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--target-mode", choices=("moving", "static", "combined"), default="moving")
    parser.add_argument("--azimuth-deg", type=float, default=-35.0)
    parser.add_argument("--elevation-deg", type=float, default=10.0)
    parser.add_argument("--distance-scale", type=float, default=0.32)
    parser.add_argument("--flip-vertical", action="store_true",
                        help="reproduce the legacy upside-down viewer orientation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--roi-stats-mode", choices=("auto", "exact", "voxel"), default="auto")
    parser.add_argument("--deformation-frame", type=int, default=0,
                        help="FE-derived vibration frame to materialize as a deformed moving checkpoint")
    parser.add_argument("--skip-deformation", action="store_true",
                        help="validate inputs without materializing a deformed checkpoint")
    parser.add_argument("--skip-render", action="store_true", help="validate and summarize inputs without rendering")
    parser.add_argument("--no-copy-dynamic-inputs", action="store_true",
                        help="write a runtime manifest pointing at source dynamic inputs instead of copying them")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.scale_disp < 0 or args.frames < 2 or args.frame_start < 0 or args.frame_stride < 1:
        raise SystemExit("scale-disp must be non-negative; frames >= 2; frame-start >= 0; frame-stride >= 1")
    source_root = _normalise_external_path(args.source_root)
    if not source_root.is_dir():
        raise FileNotFoundError(f"source-root does not exist: {source_root}")
    output = _normalise_external_path(args.output_dir)
    if output == source_root or source_root in output.parents:
        raise ValueError("output-dir must be outside source-root so source data remain read-only")
    output.mkdir(parents=True, exist_ok=True)
    if not RENDER_SCRIPT.is_file():
        raise FileNotFoundError(RENDER_SCRIPT)

    def resolve_override(value: Path | None, defaults: list[str], label: str, required: bool) -> Path | None:
        if value is not None:
            path = _normalise_external_path(value)
            if not path.is_file() and required:
                raise FileNotFoundError(f"missing {label}: {path}")
            return path if path.is_file() else None
        return _find_existing(source_root, defaults, label, required)

    paths: dict[str, Path] = {
        "yuanshi": resolve_override(args.yuanshi_checkpoint, ["yuanshi.pt"], "yuanshi checkpoint", True),
        "tu1": resolve_override(
            args.tu1_checkpoint,
            [
                "results/filter_schemes/percentage/tu1_percentage_filtered.pt",
                "results/tu1_filtered.pt",
                "results/tu1.pt",
                "tu1_filtered.pt",
                "tu1.pt",
            ],
            "tu1 checkpoint",
            False,
        ),
        "lingshi2": resolve_override(args.lingshi2_checkpoint, ["results/lingshi2.pt", "lingshi2.pt"],
                                      "lingshi2 checkpoint", True),
        "static": resolve_override(args.static_checkpoint, ["results/lingshi2_static.pt"], "static checkpoint", True),
        "moving": resolve_override(args.moving_checkpoint, ["results/lingshi2_moving.pt"], "moving checkpoint", True),
        "vibration": resolve_override(args.vibration, ["results/lingshi2_moving_vibration.dat"], "vibration data", True),
        "roi": resolve_override(args.roi_cache, ["roi_fe_beam_exact_capsule_merged_cache_3dgs.npz"], "ROI cache", True),
    }
    paths = {key: value for key, value in paths.items() if value is not None}

    roi_info, roi_data = _load_roi(paths["roi"])
    try:
        if args.roi_stats_mode == "exact" or (args.roi_stats_mode == "auto" and roi_info["mode"] == "exact-capsule"):
            try:
                roi_mask = _exact_roi_mask_factory(paths["roi"])
                roi_stats_mode = "exact-capsule"
            except Exception as exc:
                if args.roi_stats_mode == "exact":
                    raise
                print(f"[warn] exact ROI predicate unavailable; falling back to voxel hash: {exc}", file=sys.stderr)
                roi_mask = _voxel_roi_mask_factory(roi_data)
                roi_stats_mode = "voxel-hash-fallback"
        else:
            roi_mask = _voxel_roi_mask_factory(roi_data)
            roi_stats_mode = "voxel-hash"

        checkpoint_stats = {}
        for name in ("yuanshi", "tu1", "lingshi2", "static", "moving"):
            if name in paths:
                print(f"[check] {name}: {paths[name]}", flush=True)
                checkpoint_stats[name] = _checkpoint_summary(paths[name], roi_mask)
    finally:
        roi_data.close()

    moving_count = checkpoint_stats["moving"]["count"]
    static_count = checkpoint_stats["static"]["count"]
    rendered_checkpoint_count = static_count + moving_count
    lingshi2_count = checkpoint_stats["lingshi2"]["count"]
    if rendered_checkpoint_count != lingshi2_count:
        raise ValueError(
            "static plus moving Gaussian count does not equal lingshi2 count: "
            f"{static_count} + {moving_count} != {lingshi2_count}"
        )
    vibration_bytes = paths["vibration"].stat().st_size
    frame_bytes = moving_count * 3 * np.dtype(np.float32).itemsize
    if frame_bytes <= 0 or vibration_bytes % frame_bytes:
        raise ValueError(f"vibration size {vibration_bytes} is not divisible by moving_count*3*float32 ({frame_bytes})")
    vibration_frames = vibration_bytes // frame_bytes
    if vibration_frames < args.frame_start + (args.frames - 1) * args.frame_stride + 1:
        raise ValueError("vibration data do not contain the requested frame window")
    vibration_info = {**_metadata(paths["vibration"]), "moving_count": moving_count,
                      "frame_count": int(vibration_frames), "dtype": "float32", "shape": [int(vibration_frames), moving_count, 3]}

    deformation = None if args.skip_deformation else _apply_vibration_frame(
        paths["moving"], paths["vibration"], moving_count, int(vibration_frames),
        args.deformation_frame, output,
    )
    render = None if args.skip_render else _run_renderer(args, paths, output)
    artifacts: dict[str, Any] = {}
    if deformation is not None:
        # Export a matched full-model pair.  The legacy moving-only snapshot is
        # retained above for compatibility, while these files are the handoff
        # artifacts users can open side-by-side in any gsplat viewer.
        artifacts_dir = output / "artifacts"
        before_path = artifacts_dir / "gs_before.pt"
        after_path = artifacts_dir / f"gs_after_frame_{args.deformation_frame:05d}.pt"
        before = _save_combined_checkpoint(paths["static"], paths["moving"], before_path)
        deformed_moving = _load_cpu_splats(Path(deformation["deformed_checkpoint"]))
        after = _save_combined_checkpoint(paths["static"], paths["moving"], after_path, deformed_moving)
        # The notebook multiplies FE displacements by ``scale_disp`` only for
        # visual playback.  Export that view-space snapshot separately so the
        # physical (unscaled) checkpoint above remains unambiguous.
        render_scaled_moving = _load_cpu_splats(paths["moving"])
        raw_displacement = np.load(deformation["displacement_npy"], allow_pickle=False)
        render_scaled_moving["means"] = render_scaled_moving["means"] + torch.from_numpy(
            raw_displacement * float(args.scale_disp)
        )
        scaled_path = artifacts_dir / f"gs_after_frame_{args.deformation_frame:05d}_render_scaled{args.scale_disp:g}x.pt"
        after_render_scaled = _save_combined_checkpoint(
            paths["static"], paths["moving"], scaled_path, render_scaled_moving
        )
        artifacts["gs_before"] = before
        artifacts["gs_after"] = after
        artifacts["gs_after_render_scaled"] = {
            **after_render_scaled,
            "scale_disp": float(args.scale_disp),
            "source_physical_checkpoint": str(after_path),
            "artifact_role": "view-space snapshot matching the renderer's scaled displacement",
        }
        artifacts["deformation_pair_contract"] = {
            "before": "static + moving checkpoint means",
            "after": "same static means + moving means plus selected registered FE displacement",
            "frame_index": int(args.deformation_frame),
            "physical_after_scale": 1.0,
            "renderer_visualization_scale": float(args.scale_disp),
            "point_order": "static partition followed by moving partition in both files",
        }
    artifacts["dynamic_gs_runtime"] = _write_dynamic_runtime_manifest(
        output,
        paths,
        vibration_info,
        render,
        copy_inputs=not args.no_copy_dynamic_inputs,
    )
    summary = {
        "status": "validated" if args.skip_render else "passed",
        "scope": "real tower checkpoints with precomputed FE-registered ROI and vibration data",
        "gsplat_source": str(ROOT / "gsplat"),
        "source_root": str(source_root),
        "inputs": {name: _metadata(path) for name, path in paths.items() if name != "roi" and name != "vibration"},
        "roi_cache": {**roi_info, "stats_mode": roi_stats_mode},
        "vibration": vibration_info,
        "checkpoints": checkpoint_stats,
        "render_contract": {
            "static_gaussian_count": static_count,
            "moving_gaussian_count": moving_count,
            "total_gaussian_count_per_frame": rendered_checkpoint_count,
            "matches_lingshi2_count": rendered_checkpoint_count == lingshi2_count,
            "roi_usage": "statistics_only; no ROI mask is passed to the renderer",
            "raw_yuanshi_usage": "input inventory and validation only; not rendered",
            "frame_update": "moving means = base moving means + scale_disp * registered displacement; static means unchanged",
            "single_frame_deformation_checkpoint_used_by_renderer": False,
        },
        "deformation": deformation,
        "artifacts": artifacts,
        "dynamic_render": render,
        "parameters": {
            "scale_disp": args.scale_disp,
            "resolution": args.resolution,
            "frames": args.frames,
            "frame_start": args.frame_start,
            "frame_stride": args.frame_stride,
            "camera": {
                "target_mode": args.target_mode,
                "azimuth_deg": args.azimuth_deg,
                "elevation_deg": args.elevation_deg,
                "distance_scale": args.distance_scale,
            },
            "flip_vertical": bool(args.flip_vertical),
            "device": args.device,
        },
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
