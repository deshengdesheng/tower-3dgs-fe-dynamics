#!/usr/bin/env python3
"""Export a full expanded Gaussian checkpoint after one registered FE frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_checkpoint(path: Path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "splats" in state:
        return state, state["splats"], True
    if not isinstance(state, dict) or "means" not in state:
        raise ValueError(f"unsupported Gaussian checkpoint: {path}")
    return state, state, False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--scale-disp", type=float, default=50.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expanded = args.expanded.expanduser().resolve()
    runtime = args.runtime_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not expanded.is_file():
        raise FileNotFoundError(expanded)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint: {output}")
    manifest_path = runtime / "runtime_manifest.json"
    mapping_path = runtime / "runtime_index_mapping.npz"
    vibration_path = runtime / "moving_vibration.dat"
    for path in (manifest_path, mapping_path, vibration_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_expanded = Path(manifest["source_expanded"]).resolve()
    if source_expanded != expanded:
        raise ValueError(f"runtime parent {source_expanded} does not match --expanded {expanded}")
    state, splats, nested = load_checkpoint(expanded)
    n = int(splats["means"].shape[0])
    with np.load(mapping_path) as mapping:
        indices = np.asarray(mapping["moving_expanded_indices"], dtype=np.int64)
    if indices.ndim != 1 or len(np.unique(indices)) != len(indices):
        raise ValueError("runtime moving index map must be one-dimensional and unique")
    if len(indices) != int(manifest["moving_count"]) or np.any(indices < 0) or np.any(indices >= n):
        raise ValueError("runtime moving index map is inconsistent with the expanded checkpoint")
    frame_count = int(manifest["vibration_shape"][0])
    expected_shape = tuple(int(value) for value in manifest["vibration_shape"])
    if expected_shape != (frame_count, len(indices), 3):
        raise ValueError(f"unexpected vibration shape in manifest: {expected_shape}")
    if args.frame < 0 or args.frame >= frame_count:
        raise ValueError(f"frame must be within [0, {frame_count})")
    expected_bytes = int(np.prod(expected_shape)) * np.dtype(np.float32).itemsize
    if vibration_path.stat().st_size != expected_bytes:
        raise ValueError("vibration file size disagrees with runtime manifest")

    displacement = np.memmap(vibration_path, mode="r", dtype=np.float32, shape=expected_shape)[args.frame]
    means = splats["means"].detach().cpu().clone()
    index_tensor = torch.from_numpy(indices)
    delta = torch.from_numpy(np.asarray(displacement).copy()).to(dtype=means.dtype)
    means[index_tensor] = means[index_tensor] + delta * float(args.scale_disp)
    deformed_splats = dict(splats)
    deformed_splats["means"] = means
    if nested:
        output_state = dict(state)
        output_state["splats"] = deformed_splats
    else:
        output_state = deformed_splats

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_state, output)
    sidecar = {
        "format": "registered-fe-deformed-expanded-gaussians-v1",
        "source_expanded": str(expanded),
        "runtime_dir": str(runtime),
        "runtime_index_mapping": str(mapping_path),
        "vibration_file": str(vibration_path),
        "frame": int(args.frame),
        "scale_disp": float(args.scale_disp),
        "gaussian_count": n,
        "deformed_gaussian_count": int(len(indices)),
        "operation": "moving_means += scale_disp * registered_displacement; all other Gaussian parameters unchanged",
        "output": str(output),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(sidecar, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
