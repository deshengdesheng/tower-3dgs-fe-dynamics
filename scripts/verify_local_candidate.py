#!/usr/bin/env python3
"""Audit the persisted real-tower local candidate without private source files.

This check validates the files that are actually distributed in the candidate
directory: checkpoint sizes/fields, the static-moving partition, FE stream
shape/finite values, rendered frame dimensions, and the deformed checkpoint's
translation-only contract.  It intentionally does not claim to validate the
upstream photo/SfM training or the private ANSYS solve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_splats(path: Path) -> dict:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    splats = state["splats"] if isinstance(state, dict) and "splats" in state else state
    if not isinstance(splats, dict) or "means" not in splats:
        raise ValueError(f"unsupported Gaussian checkpoint: {path}")
    return splats


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args()
    candidate = args.candidate_dir.expanduser().resolve()
    manifest_path = candidate / "candidate_manifest.json"
    require(manifest_path.is_file(), f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("scope", "").startswith("real Yunnan"), "candidate is not the real tower package")

    step_ii = candidate / "step_ii"
    runtime = candidate / "step_iii" / "runtime"
    step_iv = candidate / "step_iv"
    expanded = step_ii / "ckpt_449999_rank0_expanded.pt"
    static_path = runtime / "static.pt"
    moving_path = runtime / "moving.pt"
    deformed = step_iv / "expanded_deformed_frame_00037.pt"
    vibration = runtime / "moving_vibration.dat"
    for path in (expanded, static_path, moving_path, deformed, vibration):
        require(path.is_file(), f"missing candidate artifact: {path}")

    expanded_splats = load_splats(expanded)
    static_splats = load_splats(static_path)
    moving_splats = load_splats(moving_path)
    deformed_splats = load_splats(deformed)
    expanded_count = int(expanded_splats["means"].shape[0])
    static_count = int(static_splats["means"].shape[0])
    moving_count = int(moving_splats["means"].shape[0])
    require((expanded_count, static_count, moving_count) == (1740385, 967558, 772827),
            "candidate Gaussian counts differ from the persisted chain")
    require(set(expanded_splats) == set(static_splats) == set(moving_splats) == set(deformed_splats),
            "checkpoint tensor fields differ")
    require(static_count + moving_count == expanded_count, "static/moving counts do not partition expanded model")

    mapping_path = runtime / "runtime_index_mapping.npz"
    with np.load(mapping_path) as mapping:
        moving_indices = np.asarray(mapping["moving_expanded_indices"], dtype=np.int64)
        static_indices = np.asarray(mapping["static_expanded_indices"], dtype=np.int64)
    require(np.array_equal(moving_indices, np.sort(moving_indices)), "moving indices are not sorted")
    require(np.array_equal(static_indices, np.sort(static_indices)), "static indices are not sorted")
    seen = np.zeros(expanded_count, dtype=np.uint8)
    seen[moving_indices] += 1
    seen[static_indices] += 1
    require(np.all(seen == 1), "static/moving indices are not an exact complementary partition")

    expected_bytes = 399 * moving_count * 3 * np.dtype(np.float32).itemsize
    require(vibration.stat().st_size == expected_bytes, "FE stream byte size is inconsistent")
    stream = np.memmap(vibration, mode="r", dtype=np.float32, shape=(399, moving_count, 3))
    finite_frames = {}
    for frame in (0, 37, 74, 111):
        values = np.asarray(stream[frame])
        require(np.isfinite(values).all(), f"FE stream frame {frame} contains non-finite values")
        finite_frames[str(frame)] = {
            "max_abs": float(np.max(np.abs(values))),
            "nonzero_rows": int(np.any(values != 0, axis=1).sum()),
        }

    for key, value in expanded_splats.items():
        if not torch.is_tensor(value):
            continue
        actual = deformed_splats[key]
        require(torch.is_tensor(actual) and actual.dtype == value.dtype and actual.shape == value.shape,
                f"deformed field differs: {key}")
        if key != "means":
            require(torch.equal(actual, value), f"deformed field changed outside means: {key}")
    static_index_tensor = torch.tensor(static_indices.tolist(), dtype=torch.long)
    require(torch.equal(deformed_splats["means"].index_select(0, static_index_tensor),
                        expanded_splats["means"].index_select(0, static_index_tensor)),
            "static means changed in the deformed checkpoint")

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for PNG dimension validation") from exc
    frames_dir = step_iv / "dynamic_render" / "frames"
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    require([path.name for path in frame_files] == [
        "frame_00000.png", "frame_00037.png", "frame_00074.png", "frame_00111.png"
    ], "dynamic frame set is incomplete")
    dimensions = {}
    for path in frame_files:
        with Image.open(path) as image:
            dimensions[path.name] = list(image.size)
            require(image.size == (1024, 1536), f"unexpected frame dimensions: {path}")

    result = {
        "passed": True,
        "candidate": str(candidate),
        "counts": {"expanded": expanded_count, "static": static_count, "moving": moving_count},
        "vibration_shape": [399, moving_count, 3],
        "vibration_bytes": vibration.stat().st_size,
        "frame_dimensions": dimensions,
        "fe_frame_finite_checks": finite_frames,
        "partition_exact_complement": True,
        "deformed_contract": "moving means only; static means and all other fields unchanged",
        "scope": "persisted local artifacts only; not upstream training, wind/ANSYS solve, or full Eq.20 dynamics",
    }
    output = candidate / "candidate_local_verification.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
