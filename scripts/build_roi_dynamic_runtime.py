#!/usr/bin/env python3
"""Attach the registered FE stream to the continuous ROI/expanded model.

The ROI checkpoint does not contain the original notebook's per-Gaussian FE
IDs. With --structure-roi-cache, every target Gaussian inside the exact FE
capsule ROI receives the nearest archived moving sample's translation. This
explicit replay is not the paper's Eq.20 interpolation/covariance update.
Without that option, the historical one-to-one assignment mode is retained.
Only subsets of the supplied expanded Gaussian model are rendered.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree


def load(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    return state["splats"] if isinstance(state, dict) and "splats" in state else state


def subset(state: dict[str, torch.Tensor], indices: np.ndarray) -> dict[str, torch.Tensor]:
    idx = torch.from_numpy(indices.astype(np.int64))
    return {key: value.index_select(0, idx) if torch.is_tensor(value) and value.ndim and value.shape[0] == len(state["means"]) else value for key, value in state.items()}


def build_all_roi_translation_runtime(args, expanded, expanded_points, old_points, frame_count):
    """Transfer archived translations to all ROI targets, keeping GS rows intact."""
    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    from gsplat.strategy.irregular_region_strategy import IrregularRegionFilteredStrategy

    if not np.isfinite(args.max_transfer_distance) or args.max_transfer_distance <= 0:
        raise ValueError("--max-transfer-distance must be a finite positive model-coordinate distance")
    for label, points in (("expanded Gaussian means", expanded_points), ("archived moving means", old_points)):
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError(f"{label} must be a nonempty N x 3 array")
        if not np.isfinite(points).all():
            raise ValueError(f"{label} contain non-finite coordinates")
    if frame_count <= 0:
        raise ValueError("archived vibration stream contains no frames")

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"ROI replay requires a new or empty output directory: {output}")
    strategy = IrregularRegionFilteredStrategy.from_cache(str(args.structure_roi_cache))
    if strategy.roi_mode != "exact-capsule":
        raise ValueError("--structure-roi-cache requires an exact-capsule FE ROI cache")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        roi_mask = strategy._get_region_mask(torch.as_tensor(expanded_points, device=device)).cpu().numpy()
        source_roi_mask = strategy._get_region_mask(torch.as_tensor(old_points, device=device)).cpu().numpy()
    moving_indices = np.flatnonzero(roi_mask).astype(np.int64, copy=False)
    static_indices = np.flatnonzero(~roi_mask).astype(np.int64, copy=False)
    if not len(moving_indices):
        raise ValueError("the supplied exact-capsule cache contains no expanded Gaussian targets")
    errors, source_indices = cKDTree(old_points).query(expanded_points[moving_indices], k=1, workers=-1)
    source_indices = source_indices.astype(np.int64, copy=False)
    too_far = errors > args.max_transfer_distance
    if np.any(too_far):
        raise ValueError(
            f"{int(too_far.sum())} ROI targets exceed --max-transfer-distance "
            f"{args.max_transfer_distance}; maximum distance is {float(errors.max()):.8g}. "
            "No targets are silently left static."
        )
    if len(np.unique(moving_indices)) != len(moving_indices):
        raise RuntimeError("ROI target indices are not unique")

    source_stream = np.memmap(args.vibration, mode="r", dtype=np.float32, shape=(frame_count, len(old_points), 3))
    print(f"[ROI replay] checking finite archived displacements in {frame_count} frames", flush=True)
    for frame in range(frame_count):
        if not np.isfinite(source_stream[frame]).all():
            raise ValueError(f"archived vibration frame {frame} contains non-finite displacement values")

    output.mkdir(parents=True, exist_ok=True)
    torch.save({"splats": subset(expanded, static_indices)}, output / "static.pt")
    torch.save({"splats": subset(expanded, moving_indices)}, output / "moving.pt")
    transferred_shape = (frame_count, len(moving_indices), 3)
    target_stream = np.memmap(output / "moving_vibration.dat", mode="w+", dtype=np.float32, shape=transferred_shape)
    for frame in range(frame_count):
        target_stream[frame] = source_stream[frame, source_indices, :]
        if (frame + 1) % 50 == 0 or frame + 1 == frame_count:
            target_stream.flush()
            print(f"[ROI replay] wrote {frame + 1}/{frame_count} frames for {len(moving_indices):,} ROI Gaussians", flush=True)
    target_stream.flush()
    del target_stream, source_stream

    far_mask = errors > 0.05
    np.savez_compressed(
        output / "runtime_index_mapping.npz",
        moving_expanded_indices=moving_indices,
        static_expanded_indices=static_indices,
        moving_source_indices=source_indices,
        target_to_source_distance=errors.astype(np.float32),
        far_target_expanded_indices=moving_indices[far_mask],
        far_target_source_indices=source_indices[far_mask],
    )
    rule = "all exact-capsule target ROI Gaussians receive their nearest archived moving sample's translation; source rows may be reused"
    limitation = (
        "This is nearest-neighbour replay of archived translations, not original Gaussian-ID matching "
        "or the paper's Eq.20 anisotropic interpolation, polar rotation, scale/quaternion/covariance update. "
        "The ROI is a padded structural prior and can include nearby background; spatial coverage does not "
        "prove FE accuracy or prevent crossing between nearby members."
    )
    prior_issue = {
        "scope": "previous real_tower_roi_continuous_corrected_20261017_v3 candidate audit; not a count inferred for arbitrary inputs",
        "previous_roi_count": 772827,
        "previous_static_roi_count": 665207,
        "previous_static_roi_fraction": 665207 / 772827,
        "change": "all target ROI Gaussians are now moving; this removes the fixed static ROI duplicates left by one-to-one archived-row assignment, without claiming rendering quality or full-paper reproduction",
    }
    audit = {
        "source_expanded": str(args.expanded.resolve()),
        "source_archived_moving_geometry": str(args.moving_source.resolve()),
        "source_vibration": str(args.vibration.resolve()),
        "mapping_mode": "all_target_exact_roi_nearest_archived_translation_replay",
        "structure_roi_cache": str(args.structure_roi_cache.resolve()),
        "roi_predicate": "IrregularRegionFilteredStrategy._get_region_mask (exact-capsule)",
        "registration_rule": rule,
        "assigned_moving_count": int(len(moving_indices)),
        "expanded_count": int(len(expanded_points)),
        "static_count": int(len(static_indices)),
        "exact_roi_count": int(roi_mask.sum()),
        "moving_roi_count": int(roi_mask[moving_indices].sum()),
        "remaining_static_roi_count": int(roi_mask[static_indices].sum()),
        "archived_moving_count": int(len(old_points)),
        "archived_moving_roi_count": int(source_roi_mask.sum()),
        "archive_rows_used": int(len(np.unique(source_indices))),
        "target_nearest_source_outside_roi_count": int((~source_roi_mask[source_indices]).sum()),
        "distance_units": "original Gaussian model coordinates; not assumed to be metres",
        "distance_mean": float(errors.mean()),
        "distance_median": float(np.median(errors)),
        "distance_p95": float(np.percentile(errors, 95)),
        "distance_p99": float(np.percentile(errors, 99)),
        "distance_max": float(errors.max()),
        "distance_gt_0_05_count": int(far_mask.sum()),
        "distance_gt_0_05_fraction": float(far_mask.mean()),
        "max_transfer_distance": args.max_transfer_distance,
        "over_max_transfer_distance_count": int(too_far.sum()),
        "vibration_frames": int(frame_count),
        "transferred_vibration_bytes": int(np.prod(transferred_shape, dtype=np.int64) * np.dtype(np.float32).itemsize),
        "source_coordinates_and_all_displacement_frames_finite": True,
        "expanded_indices_unique_and_partition_complete": bool(
            len(moving_indices) + len(static_indices) == len(expanded_points)
            and not np.intersect1d(moving_indices, static_indices).size
        ),
        "gaussian_attributes": "all per-Gaussian tensor fields are selected unchanged from the same expanded checkpoint in persisted expanded-index order",
        "previous_candidate_issue": prior_issue,
        "limitation": limitation,
    }
    (output / "mapping_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "format": "paper-tower-roi-expanded-dynamic-gs-runtime-v3",
        "mapping_mode": audit["mapping_mode"],
        "chain": ["Step IIa ROI-guided training output", "Step IIb structure-protected terrain/Kriging/expansion", "Step III nearest archived translation replay to all target ROI", "Step IV static + moving gsplat rendering"],
        "static_checkpoint_relative": "static.pt",
        "moving_checkpoint_relative": "moving.pt",
        "vibration_file_relative": "moving_vibration.dat",
        "vibration_dtype": "float32",
        "vibration_shape": list(transferred_shape),
        "expanded_count": int(len(expanded_points)),
        "moving_count": int(len(moving_indices)),
        "static_count": int(len(static_indices)),
        "mapping_audit_relative": "mapping_audit.json",
        "index_mapping_relative": "runtime_index_mapping.npz",
        "source_expanded": str(args.expanded.resolve()),
        "parent_checkpoint": str(args.expanded.resolve()),
        "source_moving_geometry": str(args.moving_source.resolve()),
        "source_moving_geometry_role": "archived coordinate samples of the translation field, never rendered as another Gaussian model",
        "source_vibration": str(args.vibration.resolve()),
        "structure_roi_cache": str(args.structure_roi_cache.resolve()),
        "max_transfer_distance": args.max_transfer_distance,
        "remaining_static_roi_count": audit["remaining_static_roi_count"],
        "limitation": limitation,
    }
    (output / "runtime_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "README_CN.md").write_text(
        "moving.pt 与 static.pt 都是同一 Step IIb expanded 模型的互补子集，Gaussian 属性保持不变。\n"
        "本模式将 exact-capsule 结构 ROI 内的全部目标 Gaussian 设为动态；每个目标逐帧复用空间最近的档案 moving 行位移，源行允许重复。\n"
        "距离大于 max_transfer_distance 时直接失败，不会悄悄把目标留作静态；距离超过0.05的目标及源索引记录在 audit/NPZ 中。距离单位为 Gaussian 模型坐标。\n"
        "之前 v3 候选版的一对一映射使772827个ROI点中665207个（86.1%）仍静态，存在静动重影风险；本模式保证本次ROI静态残留为0，渲染质量仍须逐图核验。\n"
        "这是已有档案位移场的最近邻平移重放，不是 Gaussian ID 连接，也未实现论文 Eq.20 的各向异性核插值、极分解旋转、尺度/四元数/协方差更新。\n"
        "本次没有重新执行 ANSYS/FE 求解；ROI带padding，可能包含近旁背景，不能仅凭覆盖统计宣称FE一致性。\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output), **audit}, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--moving-source", type=Path, required=True)
    parser.add_argument("--vibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--structure-roi-cache", type=Path, default=None,
                        help="optional exact-capsule cache: replay archived nearest-source translation to every target ROI Gaussian")
    parser.add_argument("--max-transfer-distance", type=float, default=0.1,
                        help="in all-target ROI mode, fail if any nearest-source distance exceeds this model-coordinate limit")
    args = parser.parse_args()

    expanded = load(args.expanded)
    old_moving = load(args.moving_source)
    expanded_points = expanded["means"].detach().cpu().numpy().astype(np.float32)
    old_points = old_moving["means"].detach().cpu().numpy().astype(np.float32)
    if not args.vibration.is_file():
        raise FileNotFoundError(args.vibration)
    frame_bytes = old_points.shape[0] * 3 * np.dtype(np.float32).itemsize
    if args.vibration.stat().st_size % frame_bytes:
        raise ValueError("FE vibration stream is not divisible by archived moving Gaussian count")
    frame_count = args.vibration.stat().st_size // frame_bytes

    if args.structure_roi_cache is not None:
        return build_all_roi_translation_runtime(args, expanded, expanded_points, old_points, frame_count)

    tree = cKDTree(expanded_points)
    # A small fixed neighbour list can be exhausted when several archived FE
    # points lie in a dense Gaussian patch.  Start with the requested list,
    # then expand only the rows that still need an unused target.
    initial_k = min(max(args.neighbors, 16), len(expanded_points))
    distances, candidates = tree.query(old_points, k=initial_k, workers=-1)
    if candidates.ndim == 1:
        candidates = candidates[:, None]
        distances = distances[:, None]
    order = np.argsort(distances[:, 0], kind="stable")
    assigned = np.full(len(old_points), -1, dtype=np.int64)
    used: set[int] = set()
    collision_fallbacks = 0
    unresolved: list[int] = []
    for row in order:
        selected = -1
        for candidate in candidates[row]:
            candidate = int(candidate)
            if candidate not in used:
                selected = candidate
                break
        if selected < 0:
            unresolved.append(int(row))
            continue
        assigned[row] = selected
        used.add(selected)

    # Resolve dense-patch collisions with progressively larger local lists.
    # This keeps the mapping deterministic and unique without constructing a
    # prohibitive all-pairs distance matrix.
    for row in unresolved:
        selected = -1
        query_k = min(len(expanded_points), max(initial_k * 4, 1024))
        while selected < 0:
            extra_dist, extra_candidates = tree.query(old_points[row], k=query_k)
            extra_candidates = np.atleast_1d(extra_candidates)
            for candidate in extra_candidates:
                candidate = int(candidate)
                if candidate not in used:
                    selected = candidate
                    break
            if selected < 0 and query_k < len(expanded_points):
                query_k = min(len(expanded_points), query_k * 2)
            else:
                break
        if selected < 0:
            raise RuntimeError(f"could not find a unique expanded Gaussian for archived FE row {row}")
        assigned[row] = selected
        used.add(selected)
        collision_fallbacks += 1

    moving_indices = np.unique(assigned)
    if len(moving_indices) != len(assigned):
        raise RuntimeError("spatial registration did not produce unique moving rows")
    static_indices = np.setdiff1d(np.arange(len(expanded_points), dtype=np.int64), moving_indices, assume_unique=True)
    registered_points = expanded_points[assigned]
    errors = np.linalg.norm(registered_points - old_points, axis=1)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"splats": subset(expanded, static_indices)}, output / "static.pt")
    torch.save({"splats": subset(expanded, assigned)}, output / "moving.pt")
    shutil.copy2(args.vibration, output / "moving_vibration.dat")
    # Persist the exact row mapping so the runtime split is auditable and can
    # be reproduced without treating the archived moving geometry as a second
    # Gaussian model.  Both runtime checkpoints are subsets of expanded.pt.
    np.savez_compressed(
        output / "runtime_index_mapping.npz",
        moving_expanded_indices=assigned,
        static_expanded_indices=static_indices,
    )
    audit = {
        "source_expanded": str(args.expanded.resolve()),
        "source_archived_moving_geometry": str(args.moving_source.resolve()),
        "assigned_moving_count": int(len(assigned)),
        "expanded_count": int(len(expanded_points)),
        "static_count": int(len(static_indices)),
        "registration_rule": "greedy unique nearest-neighbour assignment from archived FE-moving coordinates to the continuous expanded ROI Gaussian model",
        "candidate_neighbors": args.neighbors,
        "collision_fallback_count": collision_fallbacks,
        "initial_candidate_neighbors": int(initial_k),
        "distance_mean": float(errors.mean()),
        "distance_median": float(np.median(errors)),
        "distance_p95": float(np.percentile(errors, 95)),
        "distance_max": float(errors.max()),
        "vibration_frames": int(frame_count),
        "limitation": "The ROI checkpoint has no original Gaussian parent/source IDs; this is an auditable spatial registration, not an exact training-time ID match.",
    }
    (output / "mapping_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "format": "paper-tower-roi-expanded-dynamic-gs-runtime-v2",
        "chain": [
            "Step IIa ROI importance densification/refinement and pruning",
            "Step IIb adaptive terrain separation",
            "Step IIb ordinary Kriging terrain reconstruction",
            "Step IIb constrained multi-layer expansion",
            "Step III registered FE displacement stream",
            "Step IV combined static+moving gsplat rendering",
        ],
        "static_checkpoint_relative": "static.pt",
        "moving_checkpoint_relative": "moving.pt",
        "vibration_file_relative": "moving_vibration.dat",
        "vibration_dtype": "float32",
        "vibration_shape": [int(frame_count), int(len(assigned)), 3],
        "expanded_count": int(len(expanded_points)),
        "moving_count": int(len(assigned)),
        "static_count": int(len(static_indices)),
        "mapping_audit_relative": "mapping_audit.json",
        "index_mapping_relative": "runtime_index_mapping.npz",
        "source_expanded": str(args.expanded.resolve()),
        "parent_checkpoint": str(args.expanded.resolve()),
        "source_moving_geometry": str(args.moving_source.resolve()),
        "source_moving_geometry_role": "archived FE-row reference used only for spatial registration; never rendered",
        "source_vibration": str(args.vibration.resolve()),
    }
    (output / "runtime_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "README_CN.md").write_text(
        "本运行目录由连续 ROI 高斯链生成：Step IIa 剪枝 -> Step IIb 地形/Kriging/扩展 -> Step III FE 注册 -> Step IV gsplat。\n"
        "moving.pt 与 static.pt 均来自同一个 expanded.pt；FE 点位采用可审计的空间最近邻注册，详见 mapping_audit.json 和 runtime_index_mapping.npz。\n"
        "source_moving_geometry 仅是论文已有 FE 行参考坐标，不作为渲染输入；动态帧只读取本目录 moving.pt/static.pt。\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output), **audit}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
