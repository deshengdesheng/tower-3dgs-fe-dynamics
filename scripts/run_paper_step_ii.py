#!/usr/bin/env python3
"""Run the paper's Step II structure/terrain completion chain.

The chain is the one recorded in ``1_3alll_Zhixianta_dixingtuozhan.ipynb``:
adaptive terrain/object separation, Kriging terrain reconstruction, and
feature-constrained multi-layer expansion.  The command accepts either an
authorized tower checkpoint or ``--demo`` for a data-free smoke run.

This entry point does not hide the input boundary: a real tower run requires
the private checkpoint and its coordinate convention.  The demo is only an
interface test and is never presented as a paper result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "workflow" / "paper_stages"


def _run(command: list[str]) -> None:
    print(">>> " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="input Gaussian checkpoint (.pt)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--structure-roi-cache", type=Path, default=None,
        help="optional exact-capsule FE ROI cache; explicitly protects structural Gaussians from terrain classification (additional to the notebook baseline)",
    )
    parser.add_argument("--demo", action="store_true", help="generate a synthetic terrain checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--downsample", type=float, default=0.2)
    parser.add_argument("--grid-size", type=float, default=2.0)
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--cluster-eps", type=float, default=1.0)
    parser.add_argument(
        "--z-normal",
        action="store_true",
        help="override the notebook default for data whose ground is at low Z; leave disabled for the tower checkpoint, whose physical up is -Z",
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="remove isolated terrain points with DBSCAN after adaptive extraction",
    )
    parser.add_argument("--slope-threshold", type=float, default=30.0)
    parser.add_argument(
        "--std-factor",
        type=float,
        default=2.5,
        help="regional terrain-height standard-deviation factor (paper notebook default: 2.5)",
    )
    parser.add_argument(
        "--height-var-threshold",
        type=float,
        default=None,
        help="local terrain height threshold; demo defaults to 0.9, paper default to 0.5",
    )
    parser.add_argument("--kriging-grid-size", type=float, default=0.5)
    parser.add_argument("--kriging-downsample", type=float, default=1.0)
    parser.add_argument("--kriging-subsample", type=int, default=3000)
    parser.add_argument("--expand-layers", type=int, default=20)
    parser.add_argument("--extension-grid-size", type=float, default=0.3)
    parser.add_argument("--observer-center", default="0.5,-0.2,0.0")
    parser.add_argument("--max-multiplier", type=float, default=5.0)
    parser.add_argument("--height-sigma", type=float, default=1.0)
    parser.add_argument("--top-80-percent", type=float, default=0.8)
    parser.add_argument("--bottom-82-percent", type=float, default=0.82)
    parser.add_argument("--min-distance", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--spatial-threshold", type=float, default=0.5)
    parser.add_argument("--max-candidates-detail", type=int, default=20)
    parser.add_argument("--early-stop-threshold", type=float, default=0.9)
    parser.add_argument("--k-neighbors", type=int, default=6)
    parser.add_argument("--search-radius", type=int, default=8)
    parser.add_argument("--max-neighbors", type=int, default=12)
    parser.add_argument("--similarity-threshold", type=float, default=0.2)
    args = parser.parse_args()

    if args.demo and args.input:
        parser.error("--demo and --input are mutually exclusive")
    if not args.demo and not args.input:
        parser.error("provide --input or use --demo")

    output_dir = _path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.demo:
        input_path = output_dir / "demo.pt"
        _run([
            sys.executable,
            str(STAGE / "generate_demo_pt.py"),
            "--output", str(input_path),
            "--metadata", str(output_dir / "demo_metadata.txt"),
            "--size", "12",
            "--resolution", "0.4",
            "--height", "4",
            "--buildings", "2",
            "--trees", "8",
            "--seed", str(args.seed),
        ])
    else:
        input_path = _path(str(args.input)).resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"input checkpoint does not exist: {input_path}")

    # Keep a conservative generic threshold unless the caller supplies the
    # notebook's tower value (1.0 m in cell 22). The synthetic demo has
    # deliberately stronger terrain relief, so 0.9 m keeps enough ground
    # points to exercise the Kriging and expansion stages.
    height_var_threshold = args.height_var_threshold
    if height_var_threshold is None:
        height_var_threshold = 0.9 if args.demo else 0.5

    adaptive_dir = output_dir / "adaptive"
    # The adaptive extractor writes its two checkpoints directly into this
    # directory and does not create it itself on all environments.
    adaptive_dir.mkdir(parents=True, exist_ok=True)
    adaptive_command = [
        sys.executable,
        str(STAGE / "extract_terrain_adaptive.py"),
        "--input", str(input_path),
        "--output-dir", str(adaptive_dir),
        "--downsample", str(args.downsample),
        "--grid-size", str(args.grid_size),
        "--neighbors", str(args.neighbors),
        "--slope-threshold", str(args.slope_threshold),
        "--height-var-threshold", str(height_var_threshold),
        "--std-factor", str(args.std_factor),
        "--cluster-eps", str(args.cluster_eps),
    ]
    if args.cluster:
        adaptive_command.append("--cluster")
    if args.z_normal:
        adaptive_command.append("--z-normal")
    structure_roi_cache = None
    if args.structure_roi_cache is not None:
        structure_roi_cache = _path(str(args.structure_roi_cache)).resolve()
        if not structure_roi_cache.is_file():
            raise FileNotFoundError(f"structure ROI cache does not exist: {structure_roi_cache}")
        adaptive_command.extend(["--structure-roi-cache", str(structure_roi_cache)])
    _run(adaptive_command)
    terrain_path = adaptive_dir / f"{input_path.stem}_terrain_adaptive.pt"
    classification_audit_path = adaptive_dir / f"{input_path.stem}_terrain_classification_audit.json"
    classification_audit = json.loads(classification_audit_path.read_text(encoding="utf-8"))
    if structure_roi_cache is not None and classification_audit["counts"]["terrain_roi_overlap"] != 0:
        raise RuntimeError("Step II terrain includes protected structural ROI Gaussians")
    kriging_path = output_dir / f"{input_path.stem}_terrain_adaptive_kriging.pt"
    _run([
        sys.executable,
        str(STAGE / "extract_terrain_kriging.py"),
        "--input", str(terrain_path),
        "--output", str(kriging_path),
        "--grid-size", str(args.kriging_grid_size),
        "--downsample", str(args.kriging_downsample),
        "--subsample", str(args.kriging_subsample),
    ] + (["--z-normal"] if args.z_normal else []))
    expanded_path = output_dir / f"{input_path.stem}_expanded.pt"
    _run([
        sys.executable,
        str(ROOT / "extended_more_layers_param.py"),
        "--input", str(input_path),
        "--kriging", str(kriging_path),
        "--output", str(expanded_path),
        "--grid-size", str(args.extension_grid_size),
        "--expand-layers", str(args.expand_layers),
        "--observer-center", args.observer_center,
        "--max-multiplier", str(args.max_multiplier),
        "--height-sigma", str(args.height_sigma),
        "--top-80-percent", str(args.top_80_percent),
        "--bottom-82-percent", str(args.bottom_82_percent),
        "--min-distance", str(args.min_distance),
        "--max-candidates", str(args.max_candidates),
        "--spatial-threshold", str(args.spatial_threshold),
        "--max-candidates-detail", str(args.max_candidates_detail),
        "--early-stop-threshold", str(args.early_stop_threshold),
        "--k-neighbors", str(args.k_neighbors),
        "--search-radius", str(args.search_radius),
        "--max-neighbors", str(args.max_neighbors),
        "--similarity-threshold", str(args.similarity_threshold),
        "--no-multiprocessing",
    ])

    manifest = {
        "paper_step": "II",
        "synthetic_demo": bool(args.demo),
        "input": str(input_path),
        "outputs": {
            "adaptive_terrain": str(terrain_path),
            "kriging_terrain": str(kriging_path),
            "expanded_model": str(expanded_path),
            "terrain_classification_audit": str(classification_audit_path),
        },
        "parameters": {
            "downsample": args.downsample,
            "grid_size": args.grid_size,
            "neighbors": args.neighbors,
            "slope_threshold": args.slope_threshold,
            "std_factor": args.std_factor,
            "height_var_threshold": height_var_threshold,
            "cluster": args.cluster,
            "cluster_eps": args.cluster_eps,
            "z_normal": args.z_normal,
            "structure_roi_cache": str(structure_roi_cache) if structure_roi_cache else None,
            "kriging_grid_size": args.kriging_grid_size,
            "kriging_downsample": args.kriging_downsample,
            "kriging_subsample": args.kriging_subsample,
            "expand_layers": args.expand_layers,
            "extension_grid_size": args.extension_grid_size,
            "observer_center": args.observer_center,
            "max_multiplier": args.max_multiplier,
            "height_sigma": args.height_sigma,
            "top_80_percent": args.top_80_percent,
            "bottom_82_percent": args.bottom_82_percent,
            "min_distance": args.min_distance,
            "max_candidates": args.max_candidates,
            "spatial_threshold": args.spatial_threshold,
            "max_candidates_detail": args.max_candidates_detail,
            "early_stop_threshold": args.early_stop_threshold,
            "k_neighbors": args.k_neighbors,
            "search_radius": args.search_radius,
            "max_neighbors": args.max_neighbors,
            "similarity_threshold": args.similarity_threshold,
        },
        "source_notebook": "1_3alll_Zhixianta_dixingtuozhan.ipynb",
        "terrain_classification": {
            "variant": classification_audit["classification_variant"],
            "additional_structure_protection": classification_audit["structure_roi_protection"],
            "counts": classification_audit["counts"],
            "audit": str(classification_audit_path),
        },
        "coordinate_convention": "tower notebook default: inverted Gaussian Z; physical up is approximately -Z; --z-normal is disabled",
        "status": "interface_verified" if args.demo else "input_driven",
        "limitation": "A real tower run requires the authorized private checkpoint and coordinate-aligned terrain inputs.",
    }
    (output_dir / "step_ii_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
