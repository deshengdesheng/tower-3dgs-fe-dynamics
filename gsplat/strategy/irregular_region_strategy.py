from __future__ import annotations

import json
import os
import re
import struct
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from gsplat.strategy.default import DefaultStrategy
from gsplat.strategy.ops import remove, reset_opa


_PLY_DTYPE_MAP = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def _load_xyz_points(path: str) -> np.ndarray:
    """Load XYZ points from common lightweight formats.

    Supported:
        - .npy: array with shape [N, 3] or more columns
        - .npz: key "points", "xyz", "vertices", or the first array
        - .json: list, or dict with "points"/"xyz"/"vertices"
        - .pt/.pth: tensor, ndarray, or dict with "points"/"xyz"/"means"
        - .ply: ASCII or binary_little_endian PLY with vertex x/y/z
    """
    path = _normalize_path_for_wsl(path)
    suffix = os.path.splitext(path)[1].lower()

    if suffix == ".npy":
        points = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        if "roi_hash" in data:
            key = next((k for k in ("preview_points", "points", "xyz", "vertices", "means") if k in data), None)
            if key is None:
                raise ValueError(
                    f"{path} 是 ROI 体素缓存，但缺少 preview_points/points，无法作为点云显示"
                )
            points = data[key]
        else:
            key = next((k for k in ("points", "xyz", "vertices", "means") if k in data), None)
            if key is None:
                key = data.files[0]
            points = data[key]
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict):
            key = next((k for k in ("points", "xyz", "vertices", "means") if k in data), None)
            if key is None:
                raise ValueError(f"JSON 文件缺少 points/xyz/vertices/means 字段: {path}")
            data = data[key]
        points = np.asarray(data)
    elif suffix in {".pt", ".pth"}:
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            key = next((k for k in ("points", "xyz", "vertices", "means") if k in data), None)
            if key is None and "splats" in data and "means" in data["splats"]:
                key = ("splats", "means")
            if key is None:
                raise ValueError(f"PT 文件缺少 points/xyz/vertices/means/splats.means 字段: {path}")
            data = data[key[0]][key[1]] if isinstance(key, tuple) else data[key]
        points = data.detach().cpu().numpy() if torch.is_tensor(data) else np.asarray(data)
    elif suffix == ".ply":
        points = _load_ply_xyz(path)
    else:
        raise ValueError(f"不支持的 ROI 点云格式: {suffix}")

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"点云数组必须为 [N, >=3]，当前形状: {points.shape}")
    points = points[:, :3]
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise ValueError(f"ROI 点云为空或全部为 NaN/Inf: {path}")
    return points.astype(np.float32, copy=False)


def _normalize_path_for_wsl(path: str) -> str:
    """Accept common Windows path forms when running inside WSL/bash."""
    path = os.path.expanduser(str(path).strip())

    # Bash users sometimes paste Python raw-string syntax, e.g. r"H:\a\b.ply".
    if len(path) >= 4 and path[0] in {"r", "R"} and re.match(r"^[A-Za-z]:[\\/]", path[1:]):
        path = path[1:]

    # Convert H:\foo\bar.ply or H:/foo/bar.ply to /mnt/h/foo/bar.ply.
    if re.match(r"^[A-Za-z]:[\\/]", path):
        drive = path[0].lower()
        rest = path[2:].replace("\\", "/").lstrip("/")
        path = f"/mnt/{drive}/{rest}"

    return path


def _load_ply_xyz(path: str) -> np.ndarray:
    with open(path, "rb") as file:
        header_lines = []
        while True:
            line = file.readline()
            if not line:
                raise ValueError(f"PLY 文件头不完整: {path}")
            decoded = line.decode("ascii", errors="replace").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break

        if header_lines[0] != "ply":
            raise ValueError(f"不是有效 PLY 文件: {path}")

        fmt = None
        vertex_count = None
        vertex_props = []
        in_vertex = False
        for line in header_lines:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
                in_vertex = True
            elif parts[0] == "element":
                in_vertex = False
            elif in_vertex and parts[0] == "property":
                if parts[1] == "list":
                    raise ValueError("当前轻量 PLY 读取器不支持 vertex list property")
                vertex_props.append((parts[2], parts[1]))

        if fmt is None or vertex_count is None:
            raise ValueError(f"PLY 文件缺少 format 或 vertex element: {path}")
        prop_names = [name for name, _ in vertex_props]
        for axis in ("x", "y", "z"):
            if axis not in prop_names:
                raise ValueError(f"PLY vertex 缺少 {axis} 属性: {path}")

        if fmt == "ascii":
            rows = []
            for _ in range(vertex_count):
                row = file.readline().decode("ascii", errors="replace").split()
                rows.append([float(row[prop_names.index(axis)]) for axis in ("x", "y", "z")])
            return np.asarray(rows, dtype=np.float32)

        if fmt != "binary_little_endian":
            raise ValueError(f"暂只支持 ASCII 与 binary_little_endian PLY，当前: {fmt}")

        dtype_fields = []
        struct_fmt = "<"
        for name, prop_type in vertex_props:
            if prop_type not in _PLY_DTYPE_MAP:
                raise ValueError(f"不支持的 PLY 属性类型: {prop_type}")
            dtype_fields.append((name, _PLY_DTYPE_MAP[prop_type]))
            struct_fmt += {
                "char": "b",
                "int8": "b",
                "uchar": "B",
                "uint8": "B",
                "short": "h",
                "int16": "h",
                "ushort": "H",
                "uint16": "H",
                "int": "i",
                "int32": "i",
                "uint": "I",
                "uint32": "I",
                "float": "f",
                "float32": "f",
                "double": "d",
                "float64": "d",
            }[prop_type]

        item_size = struct.calcsize(struct_fmt)
        raw = file.read(vertex_count * item_size)
        array = np.frombuffer(raw, dtype=np.dtype(dtype_fields), count=vertex_count)
        return np.stack([array["x"], array["y"], array["z"]], axis=1).astype(np.float32)


def _voxel_downsample(points: np.ndarray, voxel_size: float, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points

    grid = np.floor(points / voxel_size).astype(np.int64)
    _, unique_indices = np.unique(grid, axis=0, return_index=True)
    points = points[np.sort(unique_indices)]

    if len(points) > max_points:
        rng = np.random.default_rng(42)
        indices = np.sort(rng.choice(len(points), size=max_points, replace=False))
        points = points[indices]
    return points


class IrregularRegionFilteredStrategy(DefaultStrategy):
    """Region-aware 3DGS densification using an irregular ROI point cloud.

    The key region is no longer an axis-aligned box. Instead, ROI voxels are
    generated from a selected point cloud, then dilated by ``roi_radius``. A
    Gaussian is regarded as inside the key region if its center falls in one of
    these dilated voxels. This better follows tower members and nearby selected
    structures than a regular bounding box.
    """

    def __init__(
        self,
        roi_pointcloud_path: Optional[str] = None,
        roi_points: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
        roi_radius: float = 0.03,
        roi_voxel_size: Optional[float] = None,
        roi_max_points: int = 200_000,
        aspect_ratio_percentile: int = 20,
        volume_percentile: int = 20,
        disable_pruning: bool = False,
        region_growth_limit: float = 4.0,
        global_growth_limit: float = 0.0,
        global_max_gaussians: int = 0,
        outside_conservative_factor: int = 4,
        outside_grad_multiplier: float = 2.0,
        region_prune_opa_multiplier: float = 1.0,
        outside_prune_opa_multiplier: float = 0.5,
        region_prune_scale3d_multiplier: float = 1.5,
        outside_prune_scale3d_multiplier: float = 2.0,
        outside_prune_scale2d_multiplier: float = 2.0,
        max_prune_fraction: float = 0.0,
        max_region_prune_fraction: float = 0.0,
        max_outside_prune_fraction: float = 0.0,
        outside_budget_prune_fraction: float = 0.0,
        min_growth_budget: int = 0,
        min_outside_point_ratio: float = 0.18,
        disable_refine: bool = False,
        verbose_region_every: int = 1000,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if roi_points is None and roi_pointcloud_path is None:
            raise ValueError("必须提供 roi_points 或 roi_pointcloud_path")

        self.roi_pointcloud_path = roi_pointcloud_path
        self.roi_radius = float(roi_radius)
        self.roi_voxel_size = float(roi_voxel_size or max(self.roi_radius / 2.0, 1e-4))
        self.roi_max_points = int(roi_max_points)
        self.aspect_ratio_percentile = int(aspect_ratio_percentile)
        self.volume_percentile = int(volume_percentile)
        self.disable_pruning = bool(disable_pruning)
        self.region_growth_limit = float(region_growth_limit)
        self.global_growth_limit = float(global_growth_limit)
        self.global_max_gaussians = int(global_max_gaussians)
        self.outside_conservative_factor = max(1, int(outside_conservative_factor))
        self.outside_grad_multiplier = float(outside_grad_multiplier)
        self.region_prune_opa_multiplier = float(region_prune_opa_multiplier)
        self.outside_prune_opa_multiplier = float(outside_prune_opa_multiplier)
        self.region_prune_scale3d_multiplier = float(region_prune_scale3d_multiplier)
        self.outside_prune_scale3d_multiplier = float(outside_prune_scale3d_multiplier)
        self.outside_prune_scale2d_multiplier = float(outside_prune_scale2d_multiplier)
        self.max_prune_fraction = float(max_prune_fraction)
        self.max_region_prune_fraction = float(max_region_prune_fraction)
        self.max_outside_prune_fraction = float(max_outside_prune_fraction)
        self.outside_budget_prune_fraction = float(outside_budget_prune_fraction)
        self.min_growth_budget = int(min_growth_budget)
        self.min_outside_point_ratio = float(min_outside_point_ratio)
        self.disable_refine = bool(disable_refine)
        self.verbose_region_every = int(verbose_region_every)

        self.initial_region_point_count = None
        self.max_region_point_count = None
        self.initial_total_point_count = None
        self.max_total_point_count = None

        if roi_points is None:
            roi_points_np = _load_xyz_points(roi_pointcloud_path)
        else:
            roi_points_np = (
                roi_points.detach().cpu().numpy() if torch.is_tensor(roi_points) else np.asarray(roi_points)
            )
            roi_points_np = roi_points_np[:, :3].astype(np.float32, copy=False)

        roi_points_np = _voxel_downsample(
            roi_points_np,
            voxel_size=max(self.roi_voxel_size, 1e-6),
            max_points=self.roi_max_points,
        )
        self._build_roi_hash(roi_points_np)

    @classmethod
    def from_cache(
        cls,
        roi_cache_path: str,
        aspect_ratio_percentile: int = 20,
        volume_percentile: int = 20,
        disable_pruning: bool = False,
        region_growth_limit: float = 4.0,
        global_growth_limit: float = 0.0,
        global_max_gaussians: int = 0,
        outside_conservative_factor: int = 4,
        outside_grad_multiplier: float = 2.0,
        region_prune_opa_multiplier: float = 1.0,
        outside_prune_opa_multiplier: float = 0.5,
        region_prune_scale3d_multiplier: float = 1.5,
        outside_prune_scale3d_multiplier: float = 2.0,
        outside_prune_scale2d_multiplier: float = 2.0,
        max_prune_fraction: float = 0.0,
        max_region_prune_fraction: float = 0.0,
        max_outside_prune_fraction: float = 0.0,
        outside_budget_prune_fraction: float = 0.0,
        min_growth_budget: int = 0,
        min_outside_point_ratio: float = 0.18,
        disable_refine: bool = False,
        verbose_region_every: int = 1000,
        **kwargs,
    ):
        obj = cls.__new__(cls)
        DefaultStrategy.__init__(obj, **kwargs)
        obj.roi_pointcloud_path = None
        obj.aspect_ratio_percentile = int(aspect_ratio_percentile)
        obj.volume_percentile = int(volume_percentile)
        obj.disable_pruning = bool(disable_pruning)
        obj.region_growth_limit = float(region_growth_limit)
        obj.global_growth_limit = float(global_growth_limit)
        obj.global_max_gaussians = int(global_max_gaussians)
        obj.outside_conservative_factor = max(1, int(outside_conservative_factor))
        obj.outside_grad_multiplier = float(outside_grad_multiplier)
        obj.region_prune_opa_multiplier = float(region_prune_opa_multiplier)
        obj.outside_prune_opa_multiplier = float(outside_prune_opa_multiplier)
        obj.region_prune_scale3d_multiplier = float(region_prune_scale3d_multiplier)
        obj.outside_prune_scale3d_multiplier = float(outside_prune_scale3d_multiplier)
        obj.outside_prune_scale2d_multiplier = float(outside_prune_scale2d_multiplier)
        obj.max_prune_fraction = float(max_prune_fraction)
        obj.max_region_prune_fraction = float(max_region_prune_fraction)
        obj.max_outside_prune_fraction = float(max_outside_prune_fraction)
        obj.outside_budget_prune_fraction = float(outside_budget_prune_fraction)
        obj.min_growth_budget = int(min_growth_budget)
        obj.min_outside_point_ratio = float(min_outside_point_ratio)
        obj.disable_refine = bool(disable_refine)
        obj.verbose_region_every = int(verbose_region_every)
        obj.initial_region_point_count = None
        obj.max_region_point_count = None
        obj.initial_total_point_count = None
        obj.max_total_point_count = None
        obj.roi_mode = "voxel"
        obj._load_roi_cache(roi_cache_path)
        return obj

    def _load_roi_cache(self, roi_cache_path: str):
        roi_cache_path = _normalize_path_for_wsl(roi_cache_path)
        data = np.load(roi_cache_path)
        required = ["roi_hash", "roi_origin", "roi_dims", "roi_radius", "roi_voxel_size"]
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"ROI 缓存缺少字段 {missing}: {roi_cache_path}")

        keys = np.asarray(data["roi_hash"], dtype=np.int64)
        keys = np.unique(keys)
        self.roi_radius = float(np.asarray(data["roi_radius"]).reshape(()))
        self.roi_voxel_size = float(np.asarray(data["roi_voxel_size"]).reshape(()))
        self.roi_origin_np = np.asarray(data["roi_origin"], dtype=np.float32)
        self.roi_dims_np = np.asarray(data["roi_dims"], dtype=np.int64)

        if "source_min" in data and "source_max" in data:
            self.roi_bbox_min_np = np.asarray(data["source_min"], dtype=np.float32) - self.roi_radius - self.roi_voxel_size
            self.roi_bbox_max_np = np.asarray(data["source_max"], dtype=np.float32) + self.roi_radius + self.roi_voxel_size
        else:
            self.roi_bbox_min_np = self.roi_origin_np
            self.roi_bbox_max_np = self.roi_origin_np + self.roi_dims_np.astype(np.float32) * self.roi_voxel_size

        self.roi_hash_sorted_cpu = torch.from_numpy(keys)
        self.roi_origin_cpu = torch.from_numpy(self.roi_origin_np)
        self.roi_bbox_min_cpu = torch.from_numpy(self.roi_bbox_min_np.astype(np.float32))
        self.roi_bbox_max_cpu = torch.from_numpy(self.roi_bbox_max_np.astype(np.float32))
        self.roi_dims_cpu = torch.from_numpy(self.roi_dims_np.astype(np.int64))
        self._roi_cache_device = None
        self.roi_mode = "voxel"

        if "roi_build_mode" in data:
            mode_value = np.asarray(data["roi_build_mode"])
            self.roi_mode = str(mode_value.item() if mode_value.shape == () else mode_value.tolist())

        if self.roi_mode == "exact-capsule":
            required_capsule = [
                "capsule_starts",
                "capsule_ends",
                "capsule_radii",
                "capsule_cell_keys",
                "capsule_cell_segment_indices",
                "capsule_grid_origin",
                "capsule_grid_dims",
                "capsule_grid_size",
            ]
            missing_capsule = [key for key in required_capsule if key not in data]
            if missing_capsule:
                raise ValueError(f"exact-capsule ROI 缓存缺少字段 {missing_capsule}: {roi_cache_path}")
            self.capsule_starts_cpu = torch.from_numpy(np.asarray(data["capsule_starts"], dtype=np.float32))
            self.capsule_ends_cpu = torch.from_numpy(np.asarray(data["capsule_ends"], dtype=np.float32))
            self.capsule_radii_cpu = torch.from_numpy(np.asarray(data["capsule_radii"], dtype=np.float32))
            self.capsule_cell_keys_cpu = torch.from_numpy(np.asarray(data["capsule_cell_keys"], dtype=np.int64))
            self.capsule_cell_segment_indices_cpu = torch.from_numpy(
                np.asarray(data["capsule_cell_segment_indices"], dtype=np.int64)
            )
            self.capsule_grid_origin_cpu = torch.from_numpy(np.asarray(data["capsule_grid_origin"], dtype=np.float32))
            self.capsule_grid_dims_cpu = torch.from_numpy(np.asarray(data["capsule_grid_dims"], dtype=np.int64))
            self.capsule_grid_size = float(np.asarray(data["capsule_grid_size"]).reshape(()))
            self.capsule_cell_unique_cpu, self.capsule_cell_start_cpu, self.capsule_cell_count_cpu = (
                self._build_cell_offsets_cpu(self.capsule_cell_keys_cpu)
            )

        print("✅ 不规则 ROI 体素缓存已加载")
        print(f"   缓存: {roi_cache_path}")
        print(f"   ROI 模式: {self.roi_mode}")
        print(f"   ROI 半径: {self.roi_radius}")
        print(f"   ROI 体素: {self.roi_voxel_size}")
        print(f"   ROI bbox min: {self.roi_bbox_min_np.tolist()}")
        print(f"   ROI bbox max: {self.roi_bbox_max_np.tolist()}")
        print(f"   ROI dilated voxels: {len(keys):,}")
        if self.roi_mode == "exact-capsule":
            print(f"   胶囊梁段数: {len(self.capsule_starts_cpu):,}")
            print(f"   胶囊粗网格 cell 条目: {len(self.capsule_cell_keys_cpu):,}")
            print(f"   胶囊粗网格唯一 cell: {len(self.capsule_cell_unique_cpu):,}")

    def _build_cell_offsets_cpu(self, sorted_keys: torch.Tensor):
        if sorted_keys.numel() == 0:
            empty = torch.empty(0, dtype=torch.long)
            return empty, empty, empty
        unique, counts = torch.unique_consecutive(sorted_keys, return_counts=True)
        starts = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long), counts[:-1]]), dim=0)
        return unique.long(), starts.long(), counts.long()

    def _build_roi_hash(self, roi_points: np.ndarray):
        self.roi_bbox_min_np = roi_points.min(axis=0) - self.roi_radius - self.roi_voxel_size
        self.roi_bbox_max_np = roi_points.max(axis=0) + self.roi_radius + self.roi_voxel_size
        self.roi_origin_np = self.roi_bbox_min_np.astype(np.float32)
        dims = np.ceil((self.roi_bbox_max_np - self.roi_origin_np) / self.roi_voxel_size).astype(np.int64) + 3
        self.roi_dims_np = np.maximum(dims, 1)

        base = np.floor((roi_points - self.roi_origin_np) / self.roi_voxel_size).astype(np.int64)
        radius_vox = int(np.ceil(self.roi_radius / self.roi_voxel_size))
        offsets = []
        for dx in range(-radius_vox, radius_vox + 1):
            for dy in range(-radius_vox, radius_vox + 1):
                for dz in range(-radius_vox, radius_vox + 1):
                    dist = np.linalg.norm(np.array([dx, dy, dz], dtype=np.float32) * self.roi_voxel_size)
                    if dist <= self.roi_radius + 1e-8:
                        offsets.append((dx, dy, dz))
        offsets = np.asarray(offsets or [(0, 0, 0)], dtype=np.int64)

        voxels = base[:, None, :] + offsets[None, :, :]
        voxels = voxels.reshape(-1, 3)
        valid = np.all((voxels >= 0) & (voxels < self.roi_dims_np), axis=1)
        voxels = voxels[valid]
        keys = self._hash_np(voxels)
        keys = np.unique(keys.astype(np.int64, copy=False))

        self.roi_hash_sorted_cpu = torch.from_numpy(keys)
        self.roi_origin_cpu = torch.from_numpy(self.roi_origin_np)
        self.roi_bbox_min_cpu = torch.from_numpy(self.roi_bbox_min_np.astype(np.float32))
        self.roi_bbox_max_cpu = torch.from_numpy(self.roi_bbox_max_np.astype(np.float32))
        self.roi_dims_cpu = torch.from_numpy(self.roi_dims_np.astype(np.int64))
        self._roi_cache_device = None
        self.roi_mode = "voxel"

        print("✅ 不规则 ROI 区域已构建")
        print(f"   ROI 点数: {len(roi_points):,}")
        print(f"   ROI 半径: {self.roi_radius}")
        print(f"   ROI 体素: {self.roi_voxel_size}")
        print(f"   ROI bbox min: {self.roi_bbox_min_np.tolist()}")
        print(f"   ROI bbox max: {self.roi_bbox_max_np.tolist()}")
        print(f"   ROI dilated voxels: {len(keys):,}")

    def _hash_np(self, voxels: np.ndarray) -> np.ndarray:
        return (voxels[:, 0] * self.roi_dims_np[1] + voxels[:, 1]) * self.roi_dims_np[2] + voxels[:, 2]

    def _ensure_roi_tensors(self, device: torch.device):
        if self._roi_cache_device == device:
            return
        self.roi_hash_sorted = self.roi_hash_sorted_cpu.to(device=device, dtype=torch.long)
        self.roi_origin = self.roi_origin_cpu.to(device=device)
        self.roi_bbox_min = self.roi_bbox_min_cpu.to(device=device)
        self.roi_bbox_max = self.roi_bbox_max_cpu.to(device=device)
        self.roi_dims = self.roi_dims_cpu.to(device=device, dtype=torch.long)
        if getattr(self, "roi_mode", "voxel") == "exact-capsule":
            self.capsule_starts = self.capsule_starts_cpu.to(device=device)
            self.capsule_ends = self.capsule_ends_cpu.to(device=device)
            self.capsule_radii = self.capsule_radii_cpu.to(device=device)
            self.capsule_cell_keys = self.capsule_cell_keys_cpu.to(device=device, dtype=torch.long)
            self.capsule_cell_segment_indices = self.capsule_cell_segment_indices_cpu.to(device=device, dtype=torch.long)
            self.capsule_grid_origin = self.capsule_grid_origin_cpu.to(device=device)
            self.capsule_grid_dims = self.capsule_grid_dims_cpu.to(device=device, dtype=torch.long)
            self.capsule_cell_unique = self.capsule_cell_unique_cpu.to(device=device, dtype=torch.long)
            self.capsule_cell_start = self.capsule_cell_start_cpu.to(device=device, dtype=torch.long)
            self.capsule_cell_count = self.capsule_cell_count_cpu.to(device=device, dtype=torch.long)
        self._roi_cache_device = device

    @torch.no_grad()
    def _get_region_mask(self, means: torch.Tensor) -> torch.Tensor:
        if getattr(self, "roi_mode", "voxel") == "exact-capsule":
            return self._get_region_mask_exact_capsule(means)

        device = means.device
        self._ensure_roi_tensors(device)

        in_bbox = torch.all((means >= self.roi_bbox_min) & (means <= self.roi_bbox_max), dim=1)
        region_mask = torch.zeros(len(means), dtype=torch.bool, device=device)
        if not in_bbox.any():
            return region_mask

        candidate_idx = torch.where(in_bbox)[0]
        candidate_means = means[candidate_idx]
        voxels = torch.floor((candidate_means - self.roi_origin) / self.roi_voxel_size).long()
        valid = torch.all((voxels >= 0) & (voxels < self.roi_dims), dim=1)
        if not valid.any():
            return region_mask

        valid_idx = candidate_idx[valid]
        voxels = voxels[valid]
        keys = (voxels[:, 0] * self.roi_dims[1] + voxels[:, 1]) * self.roi_dims[2] + voxels[:, 2]

        positions = torch.searchsorted(self.roi_hash_sorted, keys)
        positions_clamped = positions.clamp(max=len(self.roi_hash_sorted) - 1)
        hits = (positions < len(self.roi_hash_sorted)) & (self.roi_hash_sorted[positions_clamped] == keys)
        region_mask[valid_idx] = hits
        return region_mask

    @torch.no_grad()
    def _get_region_mask_exact_capsule(self, means: torch.Tensor) -> torch.Tensor:
        """点到有限梁段距离判断：distance(point, segment) <= radius。"""
        device = means.device
        self._ensure_roi_tensors(device)

        in_bbox = torch.all((means >= self.roi_bbox_min) & (means <= self.roi_bbox_max), dim=1)
        region_mask = torch.zeros(len(means), dtype=torch.bool, device=device)
        if not in_bbox.any():
            return region_mask

        candidate_indices = torch.where(in_bbox)[0]
        candidate_means = means[candidate_indices]
        cells = torch.floor((candidate_means - self.capsule_grid_origin) / self.capsule_grid_size).long()
        valid = torch.all((cells >= 0) & (cells < self.capsule_grid_dims), dim=1)
        if not valid.any():
            return region_mask

        valid_indices = candidate_indices[valid]
        valid_means = candidate_means[valid]
        cells = cells[valid]
        keys = (cells[:, 0] * self.capsule_grid_dims[1] + cells[:, 1]) * self.capsule_grid_dims[2] + cells[:, 2]

        positions = torch.searchsorted(self.capsule_cell_unique, keys)
        positions_clamped = positions.clamp(max=max(len(self.capsule_cell_unique) - 1, 0))
        has_cell = (positions < len(self.capsule_cell_unique)) & (self.capsule_cell_unique[positions_clamped] == keys)
        if not has_cell.any():
            return region_mask

        # 按粗网格 cell 分组批量处理；最终判断仍是点到有限梁段的精确距离。
        hit_valid_indices = valid_indices[has_cell]
        hit_means = valid_means[has_cell]
        hit_positions = positions_clamped[has_cell]
        unique_positions = torch.unique(hit_positions)
        max_points_per_batch = 4096
        for cell_position in unique_positions:
            start = int(self.capsule_cell_start[cell_position].item())
            count = int(self.capsule_cell_count[cell_position].item())
            if count <= 0:
                continue
            segment_ids = self.capsule_cell_segment_indices[start : start + count]
            seg_start = self.capsule_starts[segment_ids]
            seg_end = self.capsule_ends[segment_ids]
            seg_vec = seg_end - seg_start
            denom = (seg_vec * seg_vec).sum(dim=1).clamp_min(1e-12)

            point_mask = hit_positions == cell_position
            point_indices = hit_valid_indices[point_mask]
            points = hit_means[point_mask]
            for batch_start in range(0, len(points), max_points_per_batch):
                batch_points = points[batch_start : batch_start + max_points_per_batch]
                batch_indices = point_indices[batch_start : batch_start + max_points_per_batch]
                point_vec = batch_points[:, None, :] - seg_start[None, :, :]
                t = ((point_vec * seg_vec[None, :, :]).sum(dim=2) / denom[None, :]).clamp(0.0, 1.0)
                closest = seg_start[None, :, :] + t[:, :, None] * seg_vec[None, :, :]
                dist2 = ((batch_points[:, None, :] - closest) ** 2).sum(dim=2)
                hits = torch.any(dist2 <= self.capsule_radii[segment_ids][None, :] ** 2, dim=1)
                region_mask[batch_indices[hits]] = True
        return region_mask

    def _calculate_aspect_ratios(self, scales: torch.Tensor) -> torch.Tensor:
        actual_scales = torch.exp(scales)
        max_scale = torch.max(actual_scales, dim=1).values
        min_scale = torch.min(actual_scales, dim=1).values
        return max_scale / min_scale.clamp(min=1e-6)

    def _calculate_volumes(self, scales: torch.Tensor) -> torch.Tensor:
        return torch.exp(scales).prod(dim=1)

    @torch.no_grad()
    def _limit_prune_mask_by_score(
        self,
        mask: torch.Tensor,
        score: torch.Tensor,
        max_count: int,
    ) -> torch.Tensor:
        """按候选强度保留最多 max_count 个待剪枝点，防止单轮灾难性过剪。"""
        n_candidates = int(mask.sum().item())
        if max_count <= 0 or n_candidates <= max_count:
            return mask
        candidate_indices = torch.where(mask)[0]
        candidate_scores = score[candidate_indices]
        _, order = torch.topk(candidate_scores, k=max_count, largest=True, sorted=False)
        limited = torch.zeros_like(mask)
        limited[candidate_indices[order]] = True
        return limited

    @torch.no_grad()
    def _limit_grow_masks_by_budget(
        self,
        is_dupli: torch.Tensor,
        is_split: torch.Tensor,
        score: torch.Tensor,
        remaining_budget: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """按梯度强度裁剪增长候选，使复制+分裂的净增点数不超过预算。"""
        if remaining_budget <= 0:
            return torch.zeros_like(is_dupli), torch.zeros_like(is_split)
        grow_mask = is_dupli | is_split
        n_candidates = int(grow_mask.sum().item())
        if n_candidates <= remaining_budget:
            return is_dupli, is_split

        candidate_indices = torch.where(grow_mask)[0]
        candidate_scores = score[candidate_indices]
        _, order = torch.topk(candidate_scores, k=remaining_budget, largest=True, sorted=False)
        limited = torch.zeros_like(grow_mask)
        limited[candidate_indices[order]] = True
        return is_dupli & limited, is_split & limited

    @torch.no_grad()
    def _prune_outside_for_growth_budget(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        max_prune_count: int,
    ) -> int:
        """从 ROI 外低价值点释放预算，使总数不增加时 ROI 内仍可继续细化。"""
        if max_prune_count <= 0:
            return 0

        means = params["means"]
        outside_mask = ~self._get_region_mask(means)
        if not outside_mask.any():
            return 0
        if self.min_outside_point_ratio > 0 and self.initial_total_point_count:
            min_outside_points = int(self.initial_total_point_count * self.min_outside_point_ratio)
            outside_count = int(outside_mask.sum().item())
            if outside_count <= min_outside_points:
                if self.verbose:
                    print(
                        f"ROI 预算迁移跳过: ROI 外点数 {outside_count:,} 已到保底 {min_outside_points:,}"
                    )
                return 0
            max_prune_count = min(max_prune_count, outside_count - min_outside_points)
            if max_prune_count <= 0:
                return 0

        opacities = torch.sigmoid(params["opacities"].flatten())
        max_scales = torch.exp(params["scales"]).max(dim=-1).values
        prune_score = (1.0 - opacities).clamp_min(0.0) + max_scales / max(
            float(state["scene_scale"]),
            1e-8,
        )

        # 只从 ROI 外剪；优先低透明度或异常大尺度点，避免伤害目标区域。
        candidate_mask = outside_mask
        limited_mask = self._limit_prune_mask_by_score(candidate_mask, prune_score, max_prune_count)
        n_prune = int(limited_mask.sum().item())
        if n_prune <= 0:
            return 0

        remove(params=params, optimizers=optimizers, state=state, mask=limited_mask)
        if self.verbose:
            print(f"ROI 预算迁移: 先剪 ROI 外 {n_prune} 个低价值点，为 ROI 内分裂释放预算")
        return n_prune

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """ROI 优先的后向更新：可先从 ROI 外释放预算，再做 ROI 内增长。"""
        if self.disable_refine:
            return
        if step >= self.refine_stop_iter:
            return

        self._update_state(params, state, info, packed=packed)

        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and (self.reset_every <= 0 or step % self.reset_every >= self.pause_refine_after_reset)
        ):
            if (
                self.max_total_point_count is not None
                and self.outside_budget_prune_fraction > 0
                and self.min_growth_budget > 0
            ):
                current_total = len(params["means"])
                remaining_budget = self.max_total_point_count - current_total
                if remaining_budget < self.min_growth_budget:
                    need_budget = self.min_growth_budget - max(0, remaining_budget)
                    outside_count = int((~self._get_region_mask(params["means"])).sum().item())
                    outside_ratio = outside_count / max(current_total, 1)
                    # ROI 外越少，预算迁移越保守，避免把背景/上下文点抽干。
                    if outside_ratio < 0.25:
                        dynamic_growth_budget = max(300, int(self.min_growth_budget * 0.35))
                    elif outside_ratio < 0.30:
                        dynamic_growth_budget = max(600, int(self.min_growth_budget * 0.55))
                    else:
                        dynamic_growth_budget = self.min_growth_budget
                    need_budget = min(need_budget, dynamic_growth_budget)
                    max_budget_prune = max(1, int(current_total * self.outside_budget_prune_fraction))
                    self._prune_outside_for_growth_budget(
                        params,
                        optimizers,
                        state,
                        max_prune_count=min(need_budget, max_budget_prune),
                    )

            n_dupli, n_split = self._grow_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                    f"Now having {len(params['means'])} GSs."
                )

            n_prune = self._prune_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {n_prune} GSs pruned. "
                    f"Now having {len(params['means'])} GSs."
                )

            state["grad2d"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            torch.cuda.empty_cache()

        if self.reset_every > 0 and step % self.reset_every == 0:
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=self.prune_opa * 2.0,
            )

    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> Tuple[int, int]:
        means = params["means"]
        scales = params["scales"]
        device = means.device

        region_mask = self._get_region_mask(means)
        region_point_count = region_mask.sum().item()

        if self.initial_region_point_count is None:
            self.initial_region_point_count = region_point_count
            self.max_region_point_count = max(1, int(self.initial_region_point_count * self.region_growth_limit))
            self.initial_total_point_count = len(means)
            if self.global_max_gaussians > 0:
                # 分段反馈训练会反复从新 checkpoint 重启；这里使用固定绝对上限，
                # 避免某一轮过剪后，后续轮次的全局上限跟着坏 checkpoint 一起变小。
                self.max_total_point_count = self.global_max_gaussians
            elif self.global_growth_limit > 0:
                self.max_total_point_count = max(
                    1,
                    int(self.initial_total_point_count * self.global_growth_limit),
                )
            else:
                self.max_total_point_count = None
            print(f"初始不规则 ROI 内点数: {self.initial_region_point_count:,}")
            print(f"不规则 ROI 内允许最大点数: {self.max_region_point_count:,}")
            print(f"初始总点数: {self.initial_total_point_count:,}")
            if self.max_total_point_count is not None:
                print(f"总点数允许最大值: {self.max_total_point_count:,}")

        if region_point_count >= self.max_region_point_count:
            if self.verbose:
                print(f"警告：ROI 内点数({region_point_count:,})达到限制({self.max_region_point_count:,})，暂停 ROI 内分裂")
            region_mask = torch.zeros_like(region_mask)

        if self.max_total_point_count is not None and len(means) >= self.max_total_point_count:
            if self.verbose:
                print(f"警告：总点数({len(means):,})达到限制({self.max_total_point_count:,})，暂停所有分裂")
            return 0, 0

        aspect_ratios = self._calculate_aspect_ratios(scales)
        volumes = self._calculate_volumes(scales)

        if region_mask.any():
            aspect_threshold_inside = torch.quantile(
                aspect_ratios[region_mask],
                1.0 - self.aspect_ratio_percentile / 100.0,
            )
            volume_threshold_inside = torch.quantile(
                volumes[region_mask],
                1.0 - self.volume_percentile / 100.0,
            )
        else:
            aspect_threshold_inside = torch.tensor(float("inf"), device=device)
            volume_threshold_inside = torch.tensor(float("inf"), device=device)

        conservative_aspect_percentile = max(1, self.aspect_ratio_percentile // self.outside_conservative_factor)
        conservative_volume_percentile = max(1, self.volume_percentile // self.outside_conservative_factor)
        aspect_threshold_outside = torch.quantile(
            aspect_ratios,
            1.0 - conservative_aspect_percentile / 100.0,
        )
        volume_threshold_outside = torch.quantile(
            volumes,
            1.0 - conservative_volume_percentile / 100.0,
        )

        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)

        is_grad_high_inside = grads > self.grow_grad2d
        is_grad_high_outside = grads > (self.grow_grad2d * self.outside_grad_multiplier)
        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        is_large = ~is_small

        outside_mask = ~region_mask
        inside_shape_filter = region_mask & (
            (aspect_ratios > aspect_threshold_inside) | (volumes > volume_threshold_inside)
        )
        outside_shape_filter = outside_mask & (
            (aspect_ratios > aspect_threshold_outside) | (volumes > volume_threshold_outside)
        )

        is_dupli_inside = is_grad_high_inside & is_small & inside_shape_filter
        is_dupli_outside = is_grad_high_outside & is_small & outside_shape_filter
        is_dupli = is_dupli_inside | is_dupli_outside
        n_dupli = is_dupli.sum().item()

        is_split_inside = is_grad_high_inside & is_large & inside_shape_filter
        is_split_outside = is_grad_high_outside & is_large & outside_shape_filter
        is_split = is_split_inside | is_split_outside

        if step < self.refine_scale2d_stop_iter:
            is_split |= (state["radii"] > self.grow_scale2d) & inside_shape_filter
            is_split |= (state["radii"] > self.grow_scale2d * 1.5) & outside_shape_filter

        # 复制和分裂同时命中同一点时，净增点数会被低估；这里优先复制，避免预算控制失真。
        is_split = is_split & ~is_dupli

        if self.max_total_point_count is not None:
            remaining_budget = self.max_total_point_count - len(means)
            old_dupli = int(is_dupli.sum().item())
            old_split = int(is_split.sum().item())
            is_dupli, is_split = self._limit_grow_masks_by_budget(
                is_dupli,
                is_split,
                grads,
                remaining_budget,
            )
            if self.verbose and old_dupli + old_split > int((is_dupli | is_split).sum().item()):
                print(
                    f"增长预算保护已生效: 候选复制{old_dupli:,}个, 分裂{old_split:,}个, "
                    f"剩余预算{remaining_budget:,}个"
                )

            is_dupli_inside = is_dupli & region_mask
            is_dupli_outside = is_dupli & outside_mask
            is_split_inside = is_split & region_mask
            is_split_outside = is_split & outside_mask

        n_dupli = is_dupli.sum().item()
        n_split = is_split.sum().item()

        if n_dupli > 0:
            from gsplat.strategy.ops import duplicate

            duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)

        is_split = torch.cat(
            [is_split, torch.zeros(n_dupli, dtype=torch.bool, device=device)]
        )

        if n_split > 0:
            from gsplat.strategy.ops import split

            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                revised_opacity=self.revised_opacity,
            )

        if self.verbose and (n_dupli > 0 or n_split > 0 or step % max(1, self.verbose_region_every) == 0):
            print(f"不规则 ROI 内点数: {region_point_count:,}, ROI 外点数: {(~region_mask).sum().item():,}")
            print(
                f"ROI 内分裂: 复制{is_dupli_inside.sum().item()}个, 分裂{is_split_inside.sum().item()}个 "
                f"(长宽比前{self.aspect_ratio_percentile}%, 体积前{self.volume_percentile}%)"
            )
            print(
                f"ROI 外保守分裂: 复制{is_dupli_outside.sum().item()}个, 分裂{is_split_outside.sum().item()}个 "
                f"(长宽比前{conservative_aspect_percentile}%, 体积前{conservative_volume_percentile}%)"
            )

        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> int:
        if self.disable_pruning:
            if self.verbose:
                print("剪枝功能已禁用，跳过剪枝")
            return 0

        means = params["means"]
        region_mask = self._get_region_mask(means)
        outside_mask = ~region_mask
        opacities = torch.sigmoid(params["opacities"].flatten())

        region_opa_prune = (opacities < self.prune_opa * self.region_prune_opa_multiplier) & region_mask
        region_scale_prune = torch.zeros_like(region_mask)

        if self.reset_every <= 0 or step > self.reset_every:
            max_scales = torch.exp(params["scales"]).max(dim=-1).values
            region_scale_prune = (
                max_scales > (self.prune_scale3d * self.region_prune_scale3d_multiplier * state["scene_scale"])
            ) & region_mask

        outside_prune = (opacities < self.prune_opa * self.outside_prune_opa_multiplier) & outside_mask

        max_scales = None
        if self.reset_every <= 0 or step > self.reset_every:
            max_scales = torch.exp(params["scales"]).max(dim=-1).values
            outside_scale_prune = (
                max_scales > (self.prune_scale3d * self.outside_prune_scale3d_multiplier * state["scene_scale"])
            )
            if step < self.refine_scale2d_stop_iter:
                outside_scale_prune |= state["radii"] > self.prune_scale2d * self.outside_prune_scale2d_multiplier
            outside_prune |= outside_scale_prune & outside_mask

        if max_scales is None and (
            self.max_prune_fraction > 0
            or self.max_region_prune_fraction > 0
            or self.max_outside_prune_fraction > 0
        ):
            max_scales = torch.exp(params["scales"]).max(dim=-1).values

        if max_scales is not None:
            # 透明度越低、尺度越异常，剪枝优先级越高。
            prune_score = (1.0 - opacities).clamp_min(0.0) + max_scales / max(
                float(state["scene_scale"]),
                1e-8,
            )
        else:
            prune_score = (1.0 - opacities).clamp_min(0.0)

        original_region_prune = region_opa_prune | region_scale_prune
        original_outside_prune = outside_prune
        total_count = len(means)
        region_count = int(region_mask.sum().item())
        outside_count = total_count - region_count

        if self.max_region_prune_fraction > 0 and region_count > 0:
            max_region_prune = max(1, int(region_count * self.max_region_prune_fraction))
            limited_region_prune = self._limit_prune_mask_by_score(
                original_region_prune,
                prune_score,
                max_region_prune,
            )
            region_opa_prune = region_opa_prune & limited_region_prune
            region_scale_prune = region_scale_prune & limited_region_prune

        if self.max_outside_prune_fraction > 0 and outside_count > 0:
            max_outside_prune = max(1, int(outside_count * self.max_outside_prune_fraction))
            outside_prune = self._limit_prune_mask_by_score(
                original_outside_prune,
                prune_score,
                max_outside_prune,
            )

        if self.min_outside_point_ratio > 0 and outside_count > 0 and self.initial_total_point_count > 0:
            min_outside_points = int(self.initial_total_point_count * self.min_outside_point_ratio)
            remaining_outside_prune = max(0, outside_count - min_outside_points)
            outside_prune_count = int(outside_prune.sum().item())
            if remaining_outside_prune <= 0:
                if self.verbose and outside_prune_count > 0:
                    print(
                        f"ROI 外剪枝保护: ROI 外点数 {outside_count:,} 已低于保底 "
                        f"{min_outside_points:,}，跳过 ROI 外剪枝"
                    )
                outside_prune = torch.zeros_like(outside_prune)
            elif outside_prune_count > remaining_outside_prune:
                # ROI 外不能被普通剪枝继续剪破保底，只保留最高优先级的少量剪枝候选。
                outside_prune = self._limit_prune_mask_by_score(
                    outside_prune,
                    prune_score,
                    remaining_outside_prune,
                )

        final_prune = region_opa_prune | region_scale_prune | outside_prune
        if self.max_prune_fraction > 0:
            max_total_prune = max(1, int(total_count * self.max_prune_fraction))
            final_prune = self._limit_prune_mask_by_score(
                final_prune,
                prune_score,
                max_total_prune,
            )
        n_prune = final_prune.sum().item()

        if self.verbose and n_prune > 0:
            original_region_count = int(original_region_prune.sum().item())
            original_outside_count = int(original_outside_prune.sum().item())
            final_region_prune = final_prune & region_mask
            final_outside_prune = final_prune & outside_mask
            print(
                f"不规则 ROI 分层剪枝: 实际 ROI内{final_region_prune.sum().item()}个, "
                f"ROI外{final_outside_prune.sum().item()}个"
            )
            if original_region_count + original_outside_count > n_prune:
                print(
                    f"剪枝保护已生效: 原候选 ROI内{original_region_count:,}个, "
                    f"ROI外{original_outside_count:,}个, 实际剪枝{n_prune:,}个"
                )

        if n_prune > 0:
            from gsplat.strategy.ops import remove

            remove(params=params, optimizers=optimizers, state=state, mask=final_prune)

        return n_prune
