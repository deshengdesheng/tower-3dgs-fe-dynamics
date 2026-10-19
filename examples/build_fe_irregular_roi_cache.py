#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从有限元梁单元生成 3DGS 坐标系下的不规则 ROI 点云与体素缓存。

用途：
  - 不再依赖 sampled_cloud_peizhun.ply 的离散截面点云；
  - 直接用 FE 节点、梁单元和截面尺寸生成连续杆件中心线采样；
  - 使用用户提供的配准矩阵，将 FE 坐标转到 3DGS 坐标；
  - 输出训练用 ROI cache，供 train_irregular_region_filter.py 使用。

输入矩阵方向为：
    3DGS checkpoint 坐标 -> FE 坐标
本脚本使用其逆矩阵：
    FE 坐标 -> 3DGS checkpoint 坐标
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def normalize_path_for_wsl(path: str) -> str:
    """兼容 WSL 中粘贴的 Windows 路径。"""
    path = os.path.expanduser(str(path).strip())
    if len(path) >= 4 and path[0] in {"r", "R"} and re.match(r"^[A-Za-z]:[\\/]", path[1:]):
        path = path[1:]
    if re.match(r"^[A-Za-z]:[\\/]", path):
        drive = path[0].lower()
        rest = path[2:].replace("\\", "/").lstrip("/")
        path = f"/mnt/{drive}/{rest}"
    return path


def load_transform(path: str) -> np.ndarray:
    """Read a user-provided 4x4 matrix mapping 3DGS coordinates to FE coordinates."""
    transform = np.loadtxt(normalize_path_for_wsl(path)).astype(np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"变换矩阵必须是 4x4，当前为 {transform.shape}")
    return transform


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def load_node_coordinates(path: str) -> Dict[int, np.ndarray]:
    """读取节点坐标：id x y z。"""
    nodes: Dict[int, np.ndarray] = {}
    with open(normalize_path_for_wsl(path), "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                node_id = int(float(parts[0]))
                nodes[node_id] = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
            except ValueError:
                continue
    return nodes


def load_element_info(path: str) -> List[Tuple[int, int, int, int, int]]:
    """读取梁单元：id n1 n2 dir_node sec_id。"""
    elements: List[Tuple[int, int, int, int, int]] = []
    with open(normalize_path_for_wsl(path), "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                elements.append(tuple(int(float(value)) for value in parts[:5]))
            except ValueError:
                continue
    return elements


def load_section_table(path: str) -> Dict[int, Tuple[str, List[float]]]:
    """读取截面表：id,type,p1,p2,p3,p4。"""
    sections: Dict[int, Tuple[str, List[float]]] = {}
    with open(normalize_path_for_wsl(path), "r", newline="", encoding="utf-8", errors="ignore") as file:
        reader = csv.reader(file)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                section_id = int(float(row[0]))
                section_type = row[1].strip().upper()
                params = [float(value) for value in row[2:6]]
            except (ValueError, IndexError):
                continue
            sections[section_id] = (section_type, params)
    return sections


def section_radius(section_type: str, params: List[float], radius_scale: float, min_radius: float, max_radius: float) -> float:
    """把截面尺寸简化成中心线包络半径。"""
    section_type = section_type.upper()
    if section_type == "L":
        radius = float(np.hypot(params[0], params[1])) * 0.5
    elif section_type == "T":
        radius = float(np.hypot(params[0], params[1] + params[2])) * 0.5
    else:
        radius = float(max(params)) * 0.5
    radius = radius * radius_scale
    return float(np.clip(radius, min_radius, max_radius))


def sample_fe_centerlines(
    nodes: Dict[int, np.ndarray],
    elements: List[Tuple[int, int, int, int, int]],
    sections: Dict[int, Tuple[str, List[float]]],
    axial_step: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """沿 FE 梁中心线采样，并为每个采样点记录截面包络半径。"""
    point_chunks = []
    radius_chunks = []
    skipped = 0

    for _, node_1, node_2, _, section_id in elements:
        if node_1 not in nodes or node_2 not in nodes or section_id not in sections:
            skipped += 1
            continue
        p0 = nodes[node_1]
        p1 = nodes[node_2]
        vector = p1 - p0
        length = float(np.linalg.norm(vector))
        if length <= 1e-12:
            skipped += 1
            continue

        count = max(2, int(np.ceil(length / axial_step)) + 1)
        weights = np.linspace(0.0, 1.0, count, dtype=np.float64)
        samples = p0[None, :] * (1.0 - weights[:, None]) + p1[None, :] * weights[:, None]
        section_type, params = sections[section_id]
        radius = section_radius(section_type, params, radius_scale=1.0, min_radius=0.0, max_radius=np.inf)

        point_chunks.append(samples)
        radius_chunks.append(np.full(len(samples), radius, dtype=np.float64))

    if not point_chunks:
        raise RuntimeError("没有生成任何 FE 中心线采样点，请检查节点/单元/截面文件")

    points = np.vstack(point_chunks)
    radii = np.concatenate(radius_chunks)
    print(f"FE 中心线采样点: {len(points):,}, 跳过单元: {skipped:,}")
    return points, radii


def voxel_downsample(points: np.ndarray, voxel_size: float, max_points: int) -> np.ndarray:
    """按体素下采样点云。"""
    if voxel_size > 0:
        grid = np.floor(points / voxel_size).astype(np.int64)
        _, indices = np.unique(grid, axis=0, return_index=True)
        points = points[np.sort(indices)]
    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(42)
        indices = np.sort(rng.choice(len(points), size=max_points, replace=False))
        points = points[indices]
    return points


def build_roi_cache(points: np.ndarray, roi_radius: float, roi_voxel_size: float):
    """把点云膨胀成体素 ROI 哈希。"""
    bbox_min = points.min(axis=0) - roi_radius - roi_voxel_size
    bbox_max = points.max(axis=0) + roi_radius + roi_voxel_size
    origin = bbox_min.astype(np.float32)
    dims = np.ceil((bbox_max - origin) / roi_voxel_size).astype(np.int64) + 3
    dims = np.maximum(dims, 1)

    base = np.floor((points - origin) / roi_voxel_size).astype(np.int64)
    radius_vox = int(np.ceil(roi_radius / roi_voxel_size))
    offsets = []
    for dx in range(-radius_vox, radius_vox + 1):
        for dy in range(-radius_vox, radius_vox + 1):
            for dz in range(-radius_vox, radius_vox + 1):
                dist = np.linalg.norm(np.asarray([dx, dy, dz], dtype=np.float64) * roi_voxel_size)
                if dist <= roi_radius + 1e-12:
                    offsets.append((dx, dy, dz))
    offsets = np.asarray(offsets or [(0, 0, 0)], dtype=np.int64)

    voxels = (base[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    valid = np.all((voxels >= 0) & (voxels < dims), axis=1)
    voxels = voxels[valid]
    keys = (voxels[:, 0] * dims[1] + voxels[:, 1]) * dims[2] + voxels[:, 2]
    return np.unique(keys.astype(np.int64, copy=False)), origin, dims


def write_ascii_ply(path: str, points: np.ndarray) -> None:
    """写出只含 xyz 的 ASCII PLY。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="ascii") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("end_header\n")
        for point in points:
            file.write(f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="从 FE 梁中心线直接生成 3DGS 不规则 ROI cache")
    parser.add_argument("--node-file", default="nodes.txt")
    parser.add_argument("--elem-file", default="elements.txt")
    parser.add_argument("--section-file", default="sections.txt")
    parser.add_argument(
        "--transform-file",
        required=True,
        help="4x4 matrix txt mapping 3DGS checkpoint coordinates to FE coordinates",
    )
    parser.add_argument("--axial-step", type=float, default=0.05, help="FE 坐标下中心线采样步长")
    parser.add_argument("--roi-radius", type=float, default=0.04, help="3DGS 坐标下 ROI 膨胀半径")
    parser.add_argument("--roi-voxel-size", type=float, default=None)
    parser.add_argument("--point-voxel-size", type=float, default=0.008, help="输出预览点云下采样体素")
    parser.add_argument("--max-points", type=int, default=250_000)
    parser.add_argument("--preview-points", type=int, default=20_000)
    parser.add_argument("--out-ply", default="roi_fe_centerline_3dgs.ply")
    parser.add_argument("--out-npz", default="roi_fe_centerline_3dgs.npz")
    parser.add_argument("--out-cache", default="roi_fe_centerline_cache_3dgs.npz")
    args = parser.parse_args()

    nodes = load_node_coordinates(args.node_file)
    elements = load_element_info(args.elem_file)
    sections = load_section_table(args.section_file)
    print(f"数据加载完成: 节点 {len(nodes):,}, 梁 {len(elements):,}, 截面 {len(sections):,}")

    fe_points, _ = sample_fe_centerlines(nodes, elements, sections, axial_step=args.axial_step)
    transform_3dgs_to_fe = load_transform(args.transform_file)
    transform_fe_to_3dgs = np.linalg.inv(transform_3dgs_to_fe)
    points_3dgs = apply_transform(fe_points, transform_fe_to_3dgs).astype(np.float32)
    points_3dgs = voxel_downsample(points_3dgs, args.point_voxel_size, args.max_points)

    roi_voxel_size = args.roi_voxel_size or max(args.roi_radius / 2.0, 1e-4)
    roi_hash, roi_origin, roi_dims = build_roi_cache(points_3dgs, args.roi_radius, roi_voxel_size)
    preview = voxel_downsample(points_3dgs, args.point_voxel_size, args.preview_points)

    write_ascii_ply(args.out_ply, points_3dgs)
    np.savez_compressed(args.out_npz, points=points_3dgs)
    np.savez_compressed(
        args.out_cache,
        roi_hash=roi_hash,
        roi_origin=roi_origin.astype(np.float32),
        roi_dims=roi_dims.astype(np.int64),
        roi_radius=np.asarray(args.roi_radius, dtype=np.float32),
        roi_voxel_size=np.asarray(roi_voxel_size, dtype=np.float32),
        preview_points=preview.astype(np.float32),
        source_min=points_3dgs.min(axis=0).astype(np.float32),
        source_max=points_3dgs.max(axis=0).astype(np.float32),
    )

    print("✅ FE ROI 已生成")
    print(f"   ROI 点云 PLY: {args.out_ply}")
    print(f"   ROI 点云 NPZ: {args.out_npz}")
    print(f"   ROI 缓存: {args.out_cache}")
    print(f"   预览/训练点数: {len(points_3dgs):,}")
    print(f"   ROI 体素数: {len(roi_hash):,}")
    print(f"   min: {points_3dgs.min(axis=0).tolist()}")
    print(f"   max: {points_3dgs.max(axis=0).tolist()}")
    print(f"   roi_radius: {args.roi_radius}, roi_voxel_size: {roi_voxel_size}")


if __name__ == "__main__":
    main()
