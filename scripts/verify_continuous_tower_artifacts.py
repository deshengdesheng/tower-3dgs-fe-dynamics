#!/usr/bin/env python3
"""Independently audit persisted continuous tower artifacts on CPU.

This verifies artifact lineage and archived-translation replay. It does not
validate FE physics, the original training run, or the paper's Eq.20 update.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


CHUNK = 32768


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


def verify_subset(parent: dict, path: Path, indices: np.ndarray) -> dict:
    child = load_splats(path)
    require(set(child) == set(parent), f"tensor/attribute key set differs: {path}")
    parent_n = len(parent["means"])
    fields = {}
    for key, source in parent.items():
        target = child[key]
        if not torch.is_tensor(source):
            continue
        require(torch.is_tensor(target), f"{path.name}:{key} is not a tensor")
        require(source.dtype == target.dtype, f"{path.name}:{key} dtype differs")
        aligned = source.ndim > 0 and source.shape[0] == parent_n
        expected_shape = (len(indices), *source.shape[1:]) if aligned else tuple(source.shape)
        require(tuple(target.shape) == expected_shape, f"{path.name}:{key} shape differs")
        if aligned:
            for start in range(0, len(indices), CHUNK):
                stop = min(start + CHUNK, len(indices))
                selected = source.index_select(0, torch.from_numpy(indices[start:stop]))
                require(torch.equal(selected, target[start:stop]), f"{path.name}:{key} mismatch at row {start}")
        else:
            require(torch.equal(source, target), f"{path.name}:{key} non-row tensor differs")
        fields[key] = {"shape": list(target.shape), "dtype": str(target.dtype), "exact": True}
    del child
    gc.collect()
    return {"path": str(path), "gaussian_count": int(len(indices)), "tensor_fields": fields, "passed": True}


def stream_zero_summary(frame: np.ndarray) -> dict:
    max_abs = 0.0
    max_norm = 0.0
    nonzero_rows = 0
    for start in range(0, len(frame), CHUNK):
        rows = np.asarray(frame[start:start + CHUNK], dtype=np.float64)
        require(np.isfinite(rows).all(), "non-finite displacement in audited frame")
        max_abs = max(max_abs, float(np.max(np.abs(rows))))
        max_norm = max(max_norm, float(np.max(np.linalg.norm(rows, axis=1))))
        nonzero_rows += int(np.any(rows != 0, axis=1).sum())
    return {"max_absolute_component": max_abs, "max_vector_norm": max_norm,
            "nonzero_rows": nonzero_rows, "is_exactly_zero": nonzero_rows == 0}


def audit(chain: Path, expected_roi_count: int) -> dict:
    manifest_path = chain / "continuous_chain_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["status"] == "completed", "main chain manifest is not completed")
    inputs = manifest["dynamic_render_inputs"]
    parent_path = Path(inputs["parent_expanded_checkpoint"]).resolve()
    static_path = Path(inputs["static_subset"]).resolve()
    moving_path = Path(inputs["moving_subset"]).resolve()
    after_path = Path(inputs["full_model_checkpoint_after_deformation"]).resolve()
    require(Path(inputs["full_model_checkpoint_before_deformation"]).resolve() == parent_path,
            "before-deformation checkpoint differs from runtime parent")
    runtime = static_path.parent
    rm = json.loads((runtime / "runtime_manifest.json").read_text(encoding="utf-8"))
    require(Path(rm["source_expanded"]).resolve() == parent_path, "runtime has a different parent")
    require(moving_path.parent == runtime, "moving/static runtime folders differ")
    require(rm["mapping_mode"] == "all_target_exact_roi_nearest_archived_translation_replay",
            "runtime is not all-target ROI translation replay")
    with np.load(runtime / "runtime_index_mapping.npz") as mapping:
        moving_idx = np.array(mapping["moving_expanded_indices"], dtype=np.int64)
        static_idx = np.array(mapping["static_expanded_indices"], dtype=np.int64)
        source_idx = np.array(mapping["moving_source_indices"], dtype=np.int64)
    parent = load_splats(parent_path)
    n = len(parent["means"])
    require(len(moving_idx) + len(static_idx) == n, "partition cardinality differs")
    for label, indices in (("moving", moving_idx), ("static", static_idx)):
        require(indices.ndim == 1 and len(indices) > 0, f"{label} indices must be a nonempty vector")
        require(np.all(indices >= 0) and np.all(indices < n), f"{label} indices out of range")
        require(np.all(indices[1:] > indices[:-1]), f"{label} indices are not strictly increasing")
    seen = np.zeros(n, dtype=np.uint8)
    seen[moving_idx] += 1
    seen[static_idx] += 1
    require(np.all(seen == 1), "moving/static indices are not an exact complementary partition")
    del seen
    print(f"[audit] exact partition: {n:,} parent rows", flush=True)
    subsets = {
        "static": verify_subset(parent, static_path, static_idx),
        "moving": verify_subset(parent, moving_path, moving_idx),
    }
    print("[audit] every tensor field equals persisted parent-row selection", flush=True)

    repo = str(Path(__file__).resolve().parents[1])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from gsplat.strategy.irregular_region_strategy import IrregularRegionFilteredStrategy
    strategy = IrregularRegionFilteredStrategy.from_cache(rm["structure_roi_cache"])
    require(strategy.roi_mode == "exact-capsule", "structure cache is not exact-capsule")
    with torch.no_grad():
        roi = strategy._get_region_mask(parent["means"].detach().cpu()).numpy()
    roi_count = int(roi.sum())
    require(roi_count == expected_roi_count, f"ROI count {roi_count} differs from expected {expected_roi_count}")
    require(np.array_equal(np.flatnonzero(roi), moving_idx), "moving rows are not exactly all ROI rows")
    static_roi_count = int(roi[static_idx].sum())
    require(static_roi_count == 0, "ROI Gaussians remain in static subset")
    roi_result = {"independent_cpu_predicate": "IrregularRegionFilteredStrategy._get_region_mask (exact-capsule)",
                  "exact_roi_count": roi_count, "moving_roi_count": int(roi[moving_idx].sum()),
                  "static_roi_count": static_roi_count, "moving_equals_exact_roi": True, "passed": True}
    del strategy, roi
    gc.collect()
    print(f"[audit] all {roi_count:,} exact ROI Gaussians move; static ROI count zero", flush=True)

    source_geometry = load_splats(Path(rm["source_moving_geometry"]))
    source_n = len(source_geometry["means"])
    require(source_idx.shape == moving_idx.shape, "source-row map cardinality differs")
    require(np.all(source_idx >= 0) and np.all(source_idx < source_n), "source-row map out of bounds")
    from scipy.spatial import cKDTree
    distances, nearest = cKDTree(source_geometry["means"].numpy()).query(
        parent["means"].numpy()[moving_idx], k=1, workers=2)
    require(np.array_equal(nearest, source_idx), "persisted source rows are not the claimed nearest archive rows")
    nearest_result = {"exact_nearest_source_indices": True, "source_count": source_n,
                      "source_rows_used": int(len(np.unique(source_idx))),
                      "distance_max": float(distances.max()), "distance_p95": float(np.percentile(distances, 95)),
                      "distance_gt_0_05_count": int((distances > 0.05).sum()), "passed": True}
    del source_geometry, nearest, distances
    gc.collect()

    shape = tuple(int(value) for value in rm["vibration_shape"])
    require(shape[1:] == (len(moving_idx), 3), "transferred stream shape differs")
    target_file = Path(inputs["registered_displacement_stream"]).resolve()
    source_file = Path(rm["source_vibration"]).resolve()
    require(target_file.stat().st_size == int(np.prod(shape)) * 4, "target stream byte size differs")
    source_frame_bytes = source_n * 3 * 4
    require(source_file.stat().st_size == shape[0] * source_frame_bytes, "archive stream byte size/frame count differs")
    target = np.memmap(target_file, mode="r", dtype=np.float32, shape=shape)
    source = np.memmap(source_file, mode="r", dtype=np.float32, shape=(shape[0], source_n, 3))
    frames = (0, 37, 74, 111)
    frame_results = {}
    for frame in frames:
        require(frame < shape[0], f"requested verification frame {frame} missing")
        for start in range(0, len(source_idx), CHUNK):
            stop = min(start + CHUNK, len(source_idx))
            actual = target[frame, start:stop]
            expected = source[frame, source_idx[start:stop]]
            require(np.isfinite(actual).all(), f"frame {frame} has nonfinite values")
            require(np.array_equal(actual, expected), f"frame {frame} differs at target row {start}")
        frame_results[str(frame)] = {"exact_archive_row_selection": True, "rows_checked": len(source_idx)}
    frame0 = {"transferred_roi": stream_zero_summary(target[0]),
              "original_archive": stream_zero_summary(source[0])}
    print("[audit] frames 0/37/74/111 exactly match original archive at persisted source indices", flush=True)

    after = load_splats(after_path)
    sidecar = json.loads(after_path.with_suffix(after_path.suffix + ".json").read_text(encoding="utf-8"))
    require(sidecar["frame"] == 37 and sidecar["scale_disp"] == 50.0, "after-checkpoint is not frame37 x50")
    require(set(after) == set(parent), "after-checkpoint attributes differ")
    after_fields = {}
    for key, value in parent.items():
        if not torch.is_tensor(value):
            continue
        actual = after[key]
        require(torch.is_tensor(actual) and actual.dtype == value.dtype and actual.shape == value.shape,
                f"after-checkpoint {key} shape/dtype differs")
        if key != "means":
            require(torch.equal(actual, value), f"non-position Gaussian tensor {key} changed")
            after_fields[key] = {"exactly_unchanged": True}
    changed_rows = 0
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        expected = parent["means"][start:stop].clone()
        lo, hi = np.searchsorted(moving_idx, [start, stop])
        if hi > lo:
            local = torch.from_numpy(moving_idx[lo:hi] - start)
            delta = torch.from_numpy(np.array(target[37, lo:hi], copy=True)).to(dtype=expected.dtype)
            expected[local] = expected[local] + delta * 50.0
        require(torch.equal(after["means"][start:stop], expected), f"after means differ at row {start}")
        changed_rows += int(torch.any(after["means"][start:stop] != parent["means"][start:stop], dim=1).sum())
    for start in range(0, len(static_idx), CHUNK):
        ids = torch.from_numpy(static_idx[start:start + CHUNK])
        require(torch.equal(parent["means"].index_select(0, ids), after["means"].index_select(0, ids)),
                "static means changed in after-checkpoint")
    after_result = {"path": str(after_path), "frame": 37, "scale_disp": 50.0,
                    "means_exact_expected_float_tensor_operation": True, "static_means_exactly_unchanged": True,
                    "changed_mean_rows": changed_rows, "other_tensor_fields": after_fields, "passed": True}
    del after, parent, target, source
    gc.collect()
    print("[audit] after37 checkpoint equals parent plus registered frame37 x50 on moving means only", flush=True)
    return {"passed": True, "chain_manifest": str(manifest_path), "parent_checkpoint": str(parent_path),
            "gaussian_count": n, "moving_count": len(moving_idx), "static_count": len(static_idx),
            "partition_exact_complement": True, "subsets": subsets, "roi": roi_result,
            "nearest_source_mapping": nearest_result, "vibration_shape": list(shape),
            "vibration_frames": frame_results, "frame_zero_displacement": frame0,
            "after_checkpoint": after_result,
            "scope": "Exact CPU artifact-lineage and archived-translation-replay audit; not FE-physics, original training, or full Eq.20/covariance reproduction."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-roi-count", type=int, default=772827)
    args = parser.parse_args()
    chain = args.chain_dir.resolve()
    output = args.output.resolve() if args.output else chain / "independent_artifact_verification.json"
    torch.set_num_threads(2)
    try:
        result = audit(chain, args.expected_roi_count)
    except Exception as exc:
        result = {"passed": False, "error": str(exc), "traceback": traceback.format_exc()}
    result["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["verification_device"] = "cpu"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "output": str(output),
                      "frame_zero_displacement": result.get("frame_zero_displacement"),
                      "error": result.get("error")}, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
