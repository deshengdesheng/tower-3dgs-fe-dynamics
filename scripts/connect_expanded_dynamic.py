#!/usr/bin/env python3
"""Connect the notebook-aligned expanded GS model to the registered FE stream.

The terrain/extension stage preserves the original input rows as a prefix and
appends copied Gaussians.  This connector uses that invariant explicitly:
original rows keep the existing static/moving split and registered FE stream;
newly synthesized extension rows are static because they have no independent
FE correspondence.  This avoids silently applying a displacement to the
wrong Gaussian.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch


def load(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "splats" in state:
        state = state["splats"]
    return state


def match_rows(base: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Match duplicate coordinates without losing repeated rows."""
    buckets: dict[bytes, list[int]] = {}
    for index, row in enumerate(base):
        buckets.setdefault(row.tobytes(), []).append(index)
    used: dict[bytes, int] = {}
    result: list[int] = []
    for row in query:
        key = row.tobytes()
        offset = used.get(key, 0)
        candidates = buckets.get(key, [])
        if offset >= len(candidates):
            raise ValueError("a static/moving row is not present in the expanded-model prefix")
        result.append(candidates[offset])
        used[key] = offset + 1
    return np.asarray(result, dtype=np.int64)


def subset(state: dict[str, torch.Tensor], indices: np.ndarray) -> dict[str, torch.Tensor]:
    idx = torch.from_numpy(indices)
    return {key: value.index_select(0, idx) if torch.is_tensor(value) and value.ndim else value
            for key, value in state.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True,
                        help="the 272,300-row lingshi2/tu1 checkpoint used by Step II")
    parser.add_argument("--static", type=Path, required=True,
                        help="original lingshi2_static checkpoint")
    parser.add_argument("--moving", type=Path, required=True,
                        help="original lingshi2_moving checkpoint")
    parser.add_argument("--vibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    expanded = load(args.expanded)
    base = load(args.base)
    old_static = load(args.static)
    old_moving = load(args.moving)
    for name, state in (("expanded", expanded), ("base", base), ("static", old_static), ("moving", old_moving)):
        if "means" not in state:
            raise ValueError(f"{name} checkpoint has no means tensor")

    base_count = int(base["means"].shape[0])
    expanded_count = int(expanded["means"].shape[0])
    if expanded_count < base_count:
        raise ValueError("expanded checkpoint is shorter than the Step-II input")
    prefix_error = float((expanded["means"][:base_count] - base["means"]).abs().max())
    if prefix_error > 1e-6:
        raise ValueError(f"expanded prefix is not the same model (max mean error {prefix_error:g})")

    static_rows = match_rows(base["means"].numpy(), old_static["means"].numpy())
    moving_rows = match_rows(base["means"].numpy(), old_moving["means"].numpy())
    if len(np.unique(np.concatenate([static_rows, moving_rows]))) != base_count:
        raise ValueError("static/moving split does not cover the Step-II input exactly")
    extension_rows = np.arange(base_count, expanded_count, dtype=np.int64)
    static_rows_expanded = np.concatenate([static_rows, extension_rows])

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    static_path = output / "static.pt"
    moving_path = output / "moving.pt"
    torch.save({"splats": subset(expanded, static_rows_expanded)}, static_path)
    torch.save({"splats": subset(expanded, moving_rows)}, moving_path)
    vibration_path = output / "moving_vibration.dat"
    shutil.copy2(args.vibration, vibration_path)

    frame_bytes = int(len(moving_rows) * 3 * np.dtype(np.float32).itemsize)
    vibration_bytes = vibration_path.stat().st_size
    if vibration_bytes % frame_bytes:
        raise ValueError("FE stream size is incompatible with the original moving Gaussian count")
    frame_count = vibration_bytes // frame_bytes
    manifest = {
        "format": "paper-tower-expanded-dynamic-gs-runtime-v1",
        "chain": ["Step II expanded checkpoint", "Step III registered FE stream", "Step IV gsplat dynamic rendering"],
        "static_checkpoint_relative": "static.pt",
        "moving_checkpoint_relative": "moving.pt",
        "vibration_file_relative": "moving_vibration.dat",
        "vibration_dtype": "float32",
        "vibration_shape": [int(frame_count), int(len(moving_rows)), 3],
        "expanded_count": expanded_count,
        "original_count": base_count,
        "original_static_count": int(len(static_rows)),
        "original_moving_count": int(len(moving_rows)),
        "new_extension_static_count": int(len(extension_rows)),
        "mapping_rule": "original Step-II prefix preserves lingshi2 static/moving split; appended expansion rows are static with zero FE displacement",
        "source_expanded": str(args.expanded.resolve()),
        "source_base": str(args.base.resolve()),
        "source_static": str(args.static.resolve()),
        "source_moving": str(args.moving.resolve()),
        "source_vibration": str(args.vibration.resolve()),
    }
    (output / "runtime_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "This runtime is connected to the notebook-aligned Step-II expanded model.\n"
        "The original 272,300 rows retain the registered FE mapping; appended terrain/extension rows are static because no independent FE correspondence was supplied.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output), **{k: manifest[k] for k in ("expanded_count", "original_static_count", "original_moving_count", "new_extension_static_count", "vibration_shape")}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
