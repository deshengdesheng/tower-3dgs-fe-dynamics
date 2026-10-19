#!/usr/bin/env python3
"""Build a provenance-safe prune-after-expansion tower candidate.

The supplied 272,300-point pruning result is matched to the original points
inside the 917,407-point expanded model. All expansion-only rows are retained
as static rows. FE rows removed by pruning are removed from the vibration
stream as well; no displacement is assigned to a newly synthesized row.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


FIELDS = ("quats", "scales", "opacities", "sh0", "shN")


def load(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "splats" in state:
        state = state["splats"]
    if not isinstance(state, dict) or "means" not in state:
        raise ValueError(f"invalid splats checkpoint: {path}")
    return {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in state.items()}


def row_groups(means: np.ndarray) -> dict[tuple[float, float, float], list[int]]:
    groups: dict[tuple[float, float, float], list[int]] = defaultdict(list)
    for index, row in enumerate(means):
        groups[(float(row[0]), float(row[1]), float(row[2]))].append(index)
    return groups


def feature_rows(splats: dict[str, torch.Tensor], indices: list[int] | np.ndarray) -> np.ndarray:
    idx = torch.as_tensor(indices, dtype=torch.long)
    parts = [splats[key].float().reshape(splats["means"].shape[0], -1).index_select(0, idx).numpy() for key in FIELDS]
    return np.concatenate(parts, axis=1)


def map_subset_to_base(query: dict[str, torch.Tensor], base: dict[str, torch.Tensor]) -> np.ndarray:
    """Map rows with exact centers and payload to unique base row IDs."""
    base_means = base["means"].float().numpy()
    query_means = query["means"].float().numpy()
    base_groups = row_groups(base_means)
    query_groups = row_groups(query_means)
    base_features = feature_rows(base, np.arange(len(base_means)))
    query_features = feature_rows(query, np.arange(len(query_means)))
    output = np.full(len(query_means), -1, dtype=np.int64)
    max_error = 0.0
    for key, query_ids in query_groups.items():
        base_ids = base_groups.get(key, [])
        if len(base_ids) < len(query_ids):
            raise ValueError(f"query has more rows than base at center {key}")
        if len(query_ids) == 1:
            costs = np.max(np.abs(query_features[query_ids[0]] - base_features[base_ids]), axis=1)
            chosen = base_ids[int(np.argmin(costs))]
            output[query_ids[0]] = chosen
            max_error = max(max_error, float(costs.min()))
            continue
        costs = np.max(np.abs(query_features[query_ids, None, :] - base_features[None, base_ids, :]), axis=2)
        rows, cols = linear_sum_assignment(costs)
        output[np.asarray(query_ids)[rows]] = np.asarray(base_ids)[cols]
        max_error = max(max_error, float(costs[rows, cols].max()))
    if np.any(output < 0) or len(np.unique(output)) != len(output):
        raise ValueError("subset-to-base mapping is incomplete or non-unique")
    if max_error > 1e-5:
        raise ValueError(f"payload mismatch while mapping subset to base: {max_error:g}")
    return output


def map_base_to_expanded(base: dict[str, torch.Tensor], expanded: dict[str, torch.Tensor]) -> np.ndarray:
    """Map every base row to an expanded row, resolving duplicate centers by payload."""
    base_means = base["means"].float().numpy()
    expanded_means = expanded["means"].float().numpy()
    base_groups = row_groups(base_means)
    expanded_groups = row_groups(expanded_means)
    base_features = feature_rows(base, np.arange(len(base_means)))
    expanded_features = feature_rows(expanded, np.arange(len(expanded_means)))
    output = np.full(len(base_means), -1, dtype=np.int64)
    max_error = 0.0
    for key, base_ids in base_groups.items():
        expanded_ids = expanded_groups.get(key, [])
        if len(expanded_ids) < len(base_ids):
            raise ValueError(f"base center is missing from expanded model: {key}")
        costs = np.max(np.abs(base_features[base_ids, None, :] - expanded_features[None, expanded_ids, :]), axis=2)
        rows, cols = linear_sum_assignment(costs)
        output[np.asarray(base_ids)[rows]] = np.asarray(expanded_ids)[cols]
        max_error = max(max_error, float(costs[rows, cols].max()))
    if np.any(output < 0) or len(np.unique(output)) != len(output):
        raise ValueError("base-to-expanded mapping is incomplete or non-unique")
    if max_error > 1e-5:
        raise ValueError(f"payload mismatch while mapping expanded model: {max_error:g}")
    return output


def subset(splats: dict[str, torch.Tensor], indices: np.ndarray) -> dict[str, torch.Tensor]:
    idx = torch.from_numpy(indices.astype(np.int64, copy=False))
    n = int(splats["means"].shape[0])
    return {key: value.index_select(0, idx).clone() if torch.is_tensor(value) and value.ndim and value.shape[0] == n else value for key, value in splats.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--pruned", type=Path, required=True)
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--source-moving", type=Path, required=True)
    parser.add_argument("--vibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    base = load(args.base.resolve())
    pruned = load(args.pruned.resolve())
    expanded = load(args.expanded.resolve())
    source_moving = load(args.source_moving.resolve())
    base_n = int(base["means"].shape[0])
    expanded_n = int(expanded["means"].shape[0])
    if int(pruned["means"].shape[0]) >= base_n:
        raise ValueError("pruned checkpoint must contain fewer rows than the base checkpoint")

    pruned_to_base = map_subset_to_base(pruned, base)
    moving_to_base = map_subset_to_base(source_moving, base)
    base_to_expanded = map_base_to_expanded(base, expanded)
    kept_base = np.zeros(base_n, dtype=bool)
    kept_base[pruned_to_base] = True
    dynamic_base = np.zeros(base_n, dtype=bool)
    dynamic_base[moving_to_base] = True
    surviving_dynamic_source = kept_base[moving_to_base]
    surviving_moving_base = moving_to_base[surviving_dynamic_source]
    surviving_moving_source_rows = np.flatnonzero(surviving_dynamic_source).astype(np.int64)

    original_expanded_rows = base_to_expanded
    extension_mask = np.ones(expanded_n, dtype=bool)
    extension_mask[original_expanded_rows] = False
    kept_expanded_rows = np.concatenate([
        original_expanded_rows[kept_base],
        np.flatnonzero(extension_mask).astype(np.int64),
    ])
    kept_expanded_rows = np.unique(kept_expanded_rows)
    dynamic_expanded_rows = base_to_expanded[surviving_moving_base]
    dynamic_set = set(dynamic_expanded_rows.tolist())
    static_expanded_rows = np.asarray([row for row in kept_expanded_rows if int(row) not in dynamic_set], dtype=np.int64)
    if len(static_expanded_rows) + len(dynamic_expanded_rows) != len(kept_expanded_rows):
        raise ValueError("static/moving partitions do not cover kept expanded model")

    output = args.output_dir.resolve()
    stages = output / "stages"
    runtime = output / "runtime"
    mappings = output / "mappings"
    for directory in (stages, runtime, mappings):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.base.resolve(), stages / "01_initial_3dgs.pt")
    shutil.copy2(args.pruned.resolve(), stages / "02_importance_region_pruned.pt")
    shutil.copy2(args.expanded.resolve(), stages / "03_terrain_expanded_parent.pt")
    torch.save({"splats": subset(expanded, static_expanded_rows), "pipeline_metadata": {"stage": "prune-after-expansion", "role": "static partition"}}, runtime / "static.pt")
    torch.save({"splats": subset(expanded, dynamic_expanded_rows), "pipeline_metadata": {"stage": "prune-after-expansion", "role": "FE moving partition"}}, runtime / "moving.pt")
    source_vibration = np.memmap(args.vibration.resolve(), mode="r", dtype=np.float32, shape=(399, int(source_moving["means"].shape[0]), 3))
    filtered_vibration = np.asarray(source_vibration[:, surviving_moving_source_rows, :], dtype=np.float32)
    filtered_vibration.tofile(runtime / "moving_vibration.dat")
    frame0 = {key: value.clone() if torch.is_tensor(value) else value for key, value in expanded.items()}
    frame0_means = frame0["means"].clone()
    frame0_means[torch.from_numpy(dynamic_expanded_rows)] += torch.from_numpy(filtered_vibration[0])
    frame0["means"] = frame0_means
    torch.save({"splats": subset(frame0, kept_expanded_rows), "pipeline_metadata": {"stage": "prune-after-expansion", "role": "FE frame 00000", "physical_displacement_scale": 1.0}}, stages / "04_fe_registered_frame_00000.pt")

    np.savez_compressed(mappings / "base_to_expanded.npz", base_to_expanded=base_to_expanded, pruned_to_base=pruned_to_base, moving_to_base=moving_to_base, surviving_moving_source_rows=surviving_moving_source_rows, kept_expanded_rows=kept_expanded_rows, dynamic_expanded_rows=dynamic_expanded_rows)
    metadata = {
        "operation": "prune-after-expansion",
        "base_checkpoint": str(args.base.resolve()),
        "pruned_checkpoint": str(args.pruned.resolve()),
        "expanded_parent_checkpoint": str(args.expanded.resolve()),
        "base_count": base_n,
        "pruned_base_count": int(kept_base.sum()),
        "expanded_parent_count": expanded_n,
        "extension_count": int(extension_mask.sum()),
        "kept_expanded_count": int(len(kept_expanded_rows)),
        "original_base_points_removed": int((~kept_base).sum()),
        "moving_source_count": int(len(moving_to_base)),
        "moving_surviving_count": int(len(surviving_moving_source_rows)),
        "moving_removed_by_pruning": int(len(moving_to_base) - len(surviving_moving_source_rows)),
        "mapping": "exact float32 means; duplicate centers disambiguated by quaternion/scale/opacity/SH payload",
        "fe_stream_original_shape": [399, int(len(moving_to_base)), 3],
        "fe_stream_filtered_shape": [399, int(len(surviving_moving_source_rows)), 3],
        "step_ii_reexecution": False,
        "warning": "This is a prune-after-expansion candidate. The terrain/Kriging expansion was generated from the unpruned base and was not rerun from the pruned checkpoint.",
    }
    (stages / "02_importance_region_pruned.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    runtime_manifest = {
        "format": "prune_after_expansion_runtime_v1",
        "chain": ["272300-point initial", "existing 5% opacity prune", "917407-point Step-II expansion parent", "filtered FE stream", "static+moving runtime"],
        "static_checkpoint": "static.pt",
        "moving_checkpoint": "moving.pt",
        "vibration_file": "moving_vibration.dat",
        "static_count": int(len(static_expanded_rows)),
        "moving_count": int(len(dynamic_expanded_rows)),
        "total_count": int(len(kept_expanded_rows)),
        "source_expanded_count": expanded_n,
        "expansion_only_static_count": int(extension_mask.sum()),
        "removed_base_count": int((~kept_base).sum()),
        "removed_moving_count": int(len(moving_to_base) - len(surviving_moving_source_rows)),
        "fe_mapping": {"center_max_error": 0.0, "matching": "source moving rows -> base IDs -> expanded IDs"},
        "fe_stream_original_shape": [399, int(len(moving_to_base)), 3],
        "fe_stream_filtered_shape": [399, int(len(surviving_moving_source_rows)), 3],
        "step_ii_reexecution": False,
        "warning": "Prune-after-expansion; not a rerun of Step II from the pruned checkpoint.",
    }
    (runtime / "runtime_manifest.json").write_text(json.dumps(runtime_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    chain = {"status": "passed", "reproduction_scope": "prune-after-expansion candidate", "stages": metadata, "runtime": runtime_manifest, "mapping_file": "mappings/base_to_expanded.npz"}
    (output / "continuous_chain_manifest.json").write_text(json.dumps(chain, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(chain, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
