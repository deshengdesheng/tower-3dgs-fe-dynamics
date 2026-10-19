#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""地形扩展消融实验汇总脚本。

默认只统计已经存在的扩展结果，输出论文可用的 CSV/Markdown 表格。
如需补跑缺失变体，可加 --run-missing。
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import KDTree


@dataclass(frozen=True)
class Variant:
    method: str
    output: Path
    description: str
    command: tuple[str, ...] | None = None


def load_means(path: Path) -> np.ndarray:
    ckpt = torch.load(path, map_location="cpu")
    splats = ckpt.get("splats", ckpt)
    means = splats["means"].detach().cpu().numpy()
    return means.astype(np.float32, copy=False)


def occupied_cells(points: np.ndarray, origin_xy: np.ndarray, grid_size: float) -> set[tuple[int, int]]:
    indices = np.floor((points[:, :2] - origin_xy[None, :]) / grid_size).astype(np.int64)
    return set(map(tuple, indices))


def bbox_area(points: np.ndarray) -> float:
    x_range = float(points[:, 0].max() - points[:, 0].min())
    y_range = float(points[:, 1].max() - points[:, 1].min())
    return x_range * y_range


def hole_ratio(cells: set[tuple[int, int]]) -> float:
    if not cells:
        return float("nan")
    arr = np.asarray(list(cells), dtype=np.int64)
    width = int(arr[:, 0].max() - arr[:, 0].min() + 1)
    height = int(arr[:, 1].max() - arr[:, 1].min() + 1)
    if width <= 0 or height <= 0:
        return float("nan")
    return 1.0 - len(cells) / float(width * height)


def height_consistency(
    points: np.ndarray,
    kriging_points: np.ndarray,
    sample_size: int,
    seed: int,
) -> tuple[float, float]:
    if len(points) == 0:
        return float("nan"), float("nan")
    eval_points = points
    if sample_size > 0 and len(eval_points) > sample_size:
        rng = np.random.default_rng(seed)
        eval_points = eval_points[rng.choice(len(eval_points), sample_size, replace=False)]

    tree = KDTree(kriging_points[:, :2])
    _, nearest = tree.query(eval_points[:, :2], k=1)
    dz = eval_points[:, 2] - kriging_points[nearest[:, 0], 2]
    rmse = float(np.sqrt(np.mean(dz * dz)))
    mae = float(np.mean(np.abs(dz)))
    return rmse, mae


def run_command(command: tuple[str, ...]) -> None:
    print(">>> " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def build_variants(args: argparse.Namespace) -> list[Variant]:
    python = args.python
    base_cmd = (
        python,
        str(Path(__file__).resolve().parents[1] / "extended_more_layers_param.py"),
        "--input",
        str(args.input),
        "--grid-size",
        str(args.grid_size),
        "--expand-layers",
        str(args.expand_layers),
        "--observer-center",
        args.observer_center,
        "--max-multiplier",
        str(args.max_multiplier),
        "--height-sigma",
        str(args.height_sigma),
        "--top-80-percent",
        str(args.top_80_percent),
        "--bottom-82-percent",
        str(args.bottom_82_percent),
        "--min-distance",
        str(args.min_distance),
        "--max-candidates",
        str(args.max_candidates),
        "--spatial-threshold",
        str(args.spatial_threshold),
        "--max-candidates-detail",
        str(args.max_candidates_detail),
        "--early-stop-threshold",
        str(args.early_stop_threshold),
        "--k-neighbors",
        str(args.k_neighbors),
        "--search-radius",
        str(args.search_radius),
        "--max-neighbors",
        str(args.max_neighbors),
        "--similarity-threshold",
        str(args.similarity_threshold),
    )

    no_feature = args.output_dir / "terrain_ablation_no_feature_matching.pt"
    no_rotation = args.output_dir / "terrain_ablation_no_rotation_correction.pt"
    no_kriging_height = args.output_dir / "terrain_ablation_no_kriging_height_consistency.pt"
    full = args.output_dir / "terrain_ablation_full_proposed.pt"
    return [
        Variant(
            method="No extension",
            output=args.input,
            description="原始过滤后模型，不做边界扩展。",
            command=None,
        ),
        Variant(
            method="w/o feature matching",
            output=no_feature,
            description="保留多层扩展与 Kriging，但源网格不按特征匹配，改用最近源网格。",
            command=base_cmd
            + (
                "--kriging",
                str(args.kriging),
                "--disable-feature-matching",
                "--output",
                str(no_feature),
            ),
        ),
        Variant(
            method="w/o rotation correction",
            output=no_rotation,
            description="保留多层扩展、Kriging 和特征匹配，但不做 XY 旋转校正。",
            command=base_cmd
            + (
                "--kriging",
                str(args.kriging),
                "--disable-rotation-correction",
                "--output",
                str(no_rotation),
            ),
        ),
        Variant(
            method="w/o Kriging height consistency",
            output=no_kriging_height,
            description="保留 Kriging 覆盖范围、多层扩展和特征匹配，但不使用 Kriging 高程评分与高度对齐。",
            command=base_cmd
            + (
                "--kriging",
                str(args.kriging),
                "--disable-kriging-height-consistency",
                "--output",
                str(no_kriging_height),
            ),
        ),
        Variant(
            method="Full terrain extension",
            output=full,
            description="特征相似性复制 + 多层边界扩展 + Kriging 高程约束。",
            command=base_cmd + ("--kriging", str(args.kriging), "--output", str(full)),
        ),
    ]


def legacy_variants(args: argparse.Namespace) -> list[Variant]:
    return [
        Variant(
            method="No extension",
            output=args.input,
            description="原始过滤后模型，不做边界扩展。",
        ),
        Variant(
            method="Local height-only extension",
            output=Path("results/filter_schemes/percentage/tu1_percentage_filtered_expanded_height_mode.pt"),
            description="弱化的高度模式扩展，用于检验仅靠局部高度传播的效果。",
        ),
        Variant(
            method="Feature expansion w/o Kriging",
            output=Path("results/filter_schemes/percentage/tu1_percentage_filtered_expanded_no_kriging.pt"),
            description="去掉 Kriging 高程约束，只使用邻域/特征一致性扩展。",
        ),
        Variant(
            method="Parametric feature extension",
            output=Path("results/filter_schemes/percentage/tu1_percentage_filtered_expanded_param.pt"),
            description="参数化特征扩展，但覆盖层数较少。",
        ),
        Variant(
            method="Full terrain extension",
            output=Path("results/tu1_expanded_more_layers1.pt"),
            description="多层边界扩展 + Kriging 高程约束 + 特征相似性匹配。",
        ),
    ]


def format_float(value: float, digits: int = 4) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate terrain extension ablation table.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--kriging",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/terrain_extension_ablation"))
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--grid-size", type=float, default=0.3)
    parser.add_argument("--sample-size", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-missing", action="store_true")
    parser.add_argument(
        "--use-legacy-results",
        action="store_true",
        help="直接汇总当前仓库已有的历史扩展结果，不重新生成变体。",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--expand-layers", type=int, default=1000)
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_csv or args.output_dir / "terrain_extension_ablation.csv"
    output_md = args.output_md or args.output_dir / "terrain_extension_ablation.md"

    if not args.input.exists():
        raise FileNotFoundError(f"找不到输入模型: {args.input}")
    if not args.kriging.exists():
        raise FileNotFoundError(f"找不到 Kriging 地形参考: {args.kriging}")

    variants = legacy_variants(args) if args.use_legacy_results else build_variants(args)
    for variant in variants:
        if variant.output.exists():
            continue
        if args.run_missing and variant.command is not None:
            run_command(variant.command)
        else:
            raise FileNotFoundError(
                f"缺少变体结果: {variant.output}\n"
                f"可加 --run-missing 生成，或加 --use-legacy-results 汇总已有历史结果。"
            )

    base_points = load_means(args.input)
    kriging_points = load_means(args.kriging)
    base_count = len(base_points)
    origin_xy = base_points[:, :2].min(axis=0)
    base_cells = occupied_cells(base_points, origin_xy, args.grid_size)
    base_area = len(base_cells) * args.grid_size * args.grid_size
    base_bbox_area = bbox_area(base_points)

    rows: list[dict[str, object]] = []
    for variant in variants:
        points = load_means(variant.output)
        cells = occupied_cells(points, origin_xy, args.grid_size)
        coverage_area = len(cells) * args.grid_size * args.grid_size
        added_points = max(0, len(points) - base_count)
        added_eval_points = points[base_count:] if added_points > 0 else points
        height_rmse, height_mae = height_consistency(
            added_eval_points,
            kriging_points,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        rows.append(
            {
                "Method": variant.method,
                "Total Gaussians": len(points),
                "Added Gaussians": added_points,
                "Occupied cells": len(cells),
                "Coverage area": coverage_area,
                "Coverage gain (%)": (coverage_area / base_area - 1.0) * 100.0,
                "BBox area gain (%)": (bbox_area(points) / base_bbox_area - 1.0) * 100.0,
                "Height RMSE": height_rmse,
                "Height MAE": height_mae,
                "Hole ratio (%)": hole_ratio(cells) * 100.0,
                "Description": variant.description,
                "Output": str(variant.output),
            }
        )

    fieldnames = [
        "Method",
        "Total Gaussians",
        "Added Gaussians",
        "Occupied cells",
        "Coverage area",
        "Coverage gain (%)",
        "BBox area gain (%)",
        "Height RMSE",
        "Height MAE",
        "Hole ratio (%)",
        "Description",
        "Output",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with output_md.open("w", encoding="utf-8") as file:
        file.write("| " + " | ".join(fieldnames[:-2]) + " |\n")
        file.write("| " + " | ".join(["---"] * (len(fieldnames) - 2)) + " |\n")
        for row in rows:
            values = []
            for key in fieldnames[:-2]:
                value = row[key]
                if isinstance(value, float):
                    values.append(format_float(value))
                else:
                    values.append(str(value))
            file.write("| " + " | ".join(values) + " |\n")
        file.write("\n")
        file.write("说明：Coverage gain 基于 XY 网格占用面积统计；Height RMSE/MAE 基于扩展新增点到 Kriging 地形参考的最近邻高程差。\n")

    print(f"CSV: {output_csv}")
    print(f"Markdown: {output_md}")


if __name__ == "__main__":
    main()
