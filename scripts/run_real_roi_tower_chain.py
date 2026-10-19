#!/usr/bin/env python3
"""Run the real tower chain continuously from an existing ROI-trained checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("[run]", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-checkpoint", type=Path, required=True,
                        help="Step-I checkpoint immediately before ROI-guided training")
    parser.add_argument("--roi-checkpoint", type=Path, required=True,
                        help="existing Step-IIa output produced by ROI-guided training")
    parser.add_argument("--gs-to-fe-transform", type=Path, required=True,
                        help="4x4 matrix mapping Gaussian coordinates to FE coordinates")
    parser.add_argument("--source-root", type=Path, required=True,
                        help="authorized source tree containing the ROI cache, lingshi2_moving.pt, and its vibration stream")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=37)
    parser.add_argument("--resolution", default="768x1152")
    parser.add_argument("--scale-disp", type=float, default=50.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-points", type=int, default=60000)
    parser.add_argument("--renderer-python", type=Path, default=None,
                        help="optional Python with a working compiled gsplat CUDA extension")
    parser.add_argument("--gsplat-prebuilt", type=Path, default=None,
                        help="optional gsplat_cuda.so to load when the renderer Python has no installed package extension")
    args = parser.parse_args()

    initial = args.initial_checkpoint.expanduser().resolve()
    roi_checkpoint = args.roi_checkpoint.expanduser().resolve()
    transform = args.gs_to_fe_transform.expanduser().resolve()
    source = args.source_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    for path in (initial, roi_checkpoint, transform):
        if not path.is_file():
            raise FileNotFoundError(path)
    moving_source = source / "results" / "lingshi2_moving.pt"
    vibration_source = source / "results" / "lingshi2_moving_vibration.dat"
    roi_cache = source / "roi_fe_beam_exact_capsule_merged_cache_3dgs.npz"
    for path in (moving_source, vibration_source, roi_cache):
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=False)

    step_ii = output / "step_ii"
    step_iii = output / "step_iii" / "runtime"
    step_iv = output / "step_iv" / "dynamic_render"
    figures = output / "stage_visuals"
    stem = roi_checkpoint.stem

    run([
        sys.executable, str(ROOT / "scripts" / "run_paper_step_ii.py"),
        "--input", str(roi_checkpoint), "--output-dir", str(step_ii),
        "--structure-roi-cache", str(roi_cache),
        "--cluster",
        "--downsample", "0.15", "--grid-size", "0.8", "--neighbors", "50",
        "--slope-threshold", "60", "--std-factor", "2.5",
        "--height-var-threshold", "1.0", "--cluster-eps", "3.0",
        "--kriging-grid-size", "0.2", "--kriging-downsample", "0.5",
        "--kriging-subsample", "3000", "--expand-layers", "1000",
        "--extension-grid-size", "0.3", "--observer-center", "0.5,-0.2,0.0",
        "--max-multiplier", "5.0", "--height-sigma", "1.0",
        "--top-80-percent", "0.8", "--bottom-82-percent", "0.82",
        "--min-distance", "5", "--max-candidates", "50",
        "--spatial-threshold", "0.5", "--max-candidates-detail", "20",
        "--early-stop-threshold", "0.9", "--k-neighbors", "6",
        "--search-radius", "8", "--max-neighbors", "12",
        "--similarity-threshold", "0.2",
    ])

    expanded = step_ii / f"{stem}_expanded.pt"
    terrain = step_ii / "adaptive" / f"{stem}_terrain_adaptive.pt"
    objects = step_ii / "adaptive" / f"{stem}_objects_adaptive.pt"
    kriging = step_ii / f"{stem}_terrain_adaptive_kriging.pt"
    for path in (expanded, terrain, objects, kriging):
        if not path.is_file():
            raise FileNotFoundError(f"Step II did not produce required output: {path}")

    run([
        sys.executable, str(ROOT / "scripts" / "build_roi_dynamic_runtime.py"),
        "--expanded", str(expanded), "--moving-source", str(moving_source),
        "--vibration", str(vibration_source), "--output-dir", str(step_iii),
        "--neighbors", "32", "--structure-roi-cache", str(roi_cache),
    ])
    shutil.copy2(transform, step_iii / "gs_to_fe.txt")
    run([
        sys.executable, str(ROOT / "scripts" / "export_deformed_expanded_checkpoint.py"),
        "--expanded", str(expanded), "--runtime-dir", str(step_iii),
        "--frame", str(args.frame_start + args.frame_stride),
        "--scale-disp", str(args.scale_disp),
        "--output", str(output / "step_iv" / f"expanded_deformed_frame_{args.frame_start + args.frame_stride:05d}.pt"),
    ])
    dynamic_command = [
        sys.executable, str(ROOT / "scripts" / "run_dynamic_gs.py"),
        "--runtime-dir", str(step_iii), "--output-dir", str(step_iv),
        "--frames", str(args.frames), "--frame-start", str(args.frame_start),
        "--frame-stride", str(args.frame_stride), "--resolution", args.resolution,
        "--scale-disp", str(args.scale_disp), "--target-mode", "moving",
        "--azimuth-deg", "90", "--elevation-deg", "15",
        "--distance-scale", "3.0", "--gs-to-fe-transform", str(transform),
        "--save-moving-frames", "--save-undeformed", "--device", args.device,
    ]
    if args.renderer_python is not None:
        dynamic_command.extend(["--renderer-python", str(args.renderer_python.expanduser().resolve())])
    if args.gsplat_prebuilt is not None:
        dynamic_command.extend(["--gsplat-prebuilt", str(args.gsplat_prebuilt.expanduser().resolve())])
    run(dynamic_command)
    run([
        sys.executable, str(ROOT / "scripts" / "generate_roi_chain_visuals.py"),
        "--chain-dir", str(output), "--output-dir", str(figures),
        "--initial", str(initial), "--roi-trained", str(roi_checkpoint),
        "--gs-to-fe-transform", str(transform), "--max-points", str(args.max_points),
        "--disp-scale", str(args.scale_disp), "--disp-frame", str(args.frame_start + args.frame_stride),
        "--terrain", str(terrain), "--objects", str(objects), "--kriging", str(kriging),
        "--expanded", str(expanded), "--static", str(step_iii / "static.pt"),
        "--moving", str(step_iii / "moving.pt"), "--vibration", str(step_iii / "moving_vibration.dat"),
        "--dynamic-render-dir", str(step_iv), "--roi-cache", str(roi_cache),
    ])

    step_ii_manifest = json.loads((step_ii / "step_ii_manifest.json").read_text(encoding="utf-8"))
    manifest_input = Path(step_ii_manifest["input"]).resolve()
    if manifest_input != roi_checkpoint:
        raise RuntimeError(f"Step IIb consumed {manifest_input}, expected ROI checkpoint {roi_checkpoint}")
    if step_ii_manifest.get("parameters", {}).get("z_normal"):
        raise RuntimeError("Step IIb used --z-normal, contrary to the tower notebook's default inverted-Z convention")

    runtime_manifest = json.loads((step_iii / "runtime_manifest.json").read_text(encoding="utf-8"))
    mapping_audit = json.loads((step_iii / "mapping_audit.json").read_text(encoding="utf-8"))
    runtime_parent = Path(runtime_manifest["source_expanded"]).resolve()
    if runtime_parent != expanded.resolve():
        raise RuntimeError(f"Step III was built from {runtime_parent}, expected Step IIb output {expanded.resolve()}")
    expanded_count = int(mapping_audit["expanded_count"])
    moving_count = int(mapping_audit["assigned_moving_count"])
    static_count = int(mapping_audit["static_count"])
    if moving_count + static_count != expanded_count:
        raise RuntimeError("Step III static/moving split does not partition the Step IIb expanded model")
    if int(mapping_audit.get("remaining_static_roi_count", -1)) != 0:
        raise RuntimeError("Step III left target structure ROI Gaussians in the static subset")
    if int(runtime_manifest["expanded_count"]) != expanded_count:
        raise RuntimeError("Step III runtime manifest Gaussian count disagrees with the mapping audit")

    frame_indices = [args.frame_start + i * args.frame_stride for i in range(args.frames)]
    total_frames = int(runtime_manifest["vibration_shape"][0])
    if any(index < 0 or index >= total_frames for index in frame_indices):
        raise ValueError(f"requested frame indices {frame_indices} exceed FE stream length {total_frames}")
    rendered_frames = [step_iv / "frames" / f"frame_{index:05d}.png" for index in frame_indices]
    missing_frames = [path for path in rendered_frames if not path.is_file()]
    if missing_frames:
        raise RuntimeError(f"gsplat did not produce requested dynamic frames: {missing_frames}")

    render_metrics_path = step_iv / "dynamic_temporal_stability.json"
    render_metrics = json.loads(render_metrics_path.read_text(encoding="utf-8")) if render_metrics_path.is_file() else None
    if render_metrics is None:
        raise RuntimeError(f"gsplat metrics file is missing: {render_metrics_path}")
    stage_visual_manifest = json.loads((figures / "stage_visual_manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "status": "completed",
        "reproduction_scope": "continuous demo: supplied Step I/IIa, rerun IIb with explicit structure protection, archived FE field, translation-only IV",
        "continuity": [
            {"stage": "I", "checkpoint": str(initial), "execution": "supplied_checkpoint"},
            {"stage": "IIa", "checkpoint": str(roi_checkpoint),
             "handoff_to_step_ii": str(roi_checkpoint), "execution": "supplied_roi_training_result"},
            {"stage": "IIb", "expanded_model": str(expanded),
             "terrain": str(terrain), "objects": str(objects), "kriging": str(kriging)},
            {"stage": "III", "runtime_dir": str(step_iii),
             "source_expanded": runtime_manifest.get("source_expanded"),
             "mapping_audit": str(step_iii / "mapping_audit.json")},
            {"stage": "IV", "renderer": "gsplat", "input_runtime_dir": str(step_iii),
             "output_dir": str(step_iv), "metrics": str(render_metrics_path)},
        ],
        "step_ii_parameters": step_ii_manifest["parameters"],
        "coordinate_convention": step_ii_manifest["coordinate_convention"],
        "gaussian_counts": {
            "initial_checkpoint": stage_visual_manifest.get("step_ia_counts", {}).get("initial"),
            "roi_trained_checkpoint": stage_visual_manifest.get("step_ia_counts", {}).get("roi_trained"),
            "expanded_step_iib": expanded_count,
            "static_step_iii": static_count,
            "moving_step_iii": moving_count,
        },
        "dynamic_render_inputs": {
            "parent_expanded_checkpoint": str(expanded.resolve()),
            "static_subset": str((step_iii / "static.pt").resolve()),
            "moving_subset": str((step_iii / "moving.pt").resolve()),
            "registered_displacement_stream": str((step_iii / "moving_vibration.dat").resolve()),
            "rendered_frame_indices": frame_indices,
            "full_model_checkpoint_before_deformation": str(expanded.resolve()),
            "full_model_checkpoint_after_deformation": str((output / "step_iv" / f"expanded_deformed_frame_{args.frame_start + args.frame_stride:05d}.pt").resolve()),
        },
        "mapping_audit_summary": {key: mapping_audit.get(key) for key in
                                   ("expanded_count", "assigned_moving_count", "static_count",
                                    "exact_roi_count", "remaining_static_roi_count",
                                    "distance_mean", "distance_p95", "distance_max")},
        "render_metrics": render_metrics,
        "important_limits": [
            "The initial and ROI-trained checkpoints are supplied results; photo/SfM and Step-IIa long training are not rerun by this launcher.",
            "Step IIa is represented by the authentic 399999-to-449999 ROI-guided training output and exact-capsule ROI mask. This launcher does not rerun the long training; it does not perform a synthetic opacity-ranked clone/prune replay.",
            "Step IIb uses the notebook terrain algorithm with an explicit exact-capsule structure exclusion guard, because its original parameters misclassified the tower when transferred to this ROI checkpoint. This is a new continuous demonstration, not the original notebook's tu1 model or numerical result.",
            "Step III replays the nearest archived moving sample's translation on every target exact-capsule ROI Gaussian. Source samples may be reused; target expanded indices are unique and the static/moving partition is complete. This is not an exact Gaussian-ID join or Eq.20 interpolation.",
            "The dynamic renderer applies registered translation to moving means. The full paper polar rotation, scale, quaternion and covariance update is not claimed.",
            "The view is fitted to the moving tower subset while gsplat renders static + moving Gaussian subsets partitioned from the same expanded model.",
        ],
    }
    (output / "continuous_chain_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    run([
        sys.executable, str(ROOT / "scripts" / "assemble_continuous_tower_review.py"),
        "--chain-dir", str(output),
    ])
    print(json.dumps({"output_dir": str(output), "stage_visual_count": len(list(figures.glob("*.png"))),
                      "runtime": manifest["mapping_audit_summary"],
                      "render_frames": [str(path) for path in rendered_frames],
                      "before_checkpoint": str(expanded.resolve()),
                      "after_checkpoint": manifest["dynamic_render_inputs"]["full_model_checkpoint_after_deformation"]},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
