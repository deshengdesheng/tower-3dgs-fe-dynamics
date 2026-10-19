#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""动态渲染 temporal SSIM / flicker 离屏评估脚本。

该脚本复用 auto_play_highres.py 的输入文件：
静态 checkpoint、动态 checkpoint、moving vibration 位移文件和 scale_disp。
不同之处是它不启动 viewer，而是在固定相机下逐帧离屏渲染，并输出论文/回复信
可引用的 temporal SSIM 与背景 flicker 指标。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 允许从 revision_experiments 目录脚本直接导入仓库内 gsplat 包。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.rendering import rasterization
from workflow.coordinate_frames import physical_up_in_gaussian_frame

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:  # pragma: no cover - 仅在缺少 skimage 时触发
    skimage_ssim = None

try:
    import cv2
except Exception:  # pragma: no cover - 仅在缺少 OpenCV 时触发
    cv2 = None

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - 仅在缺少 imageio 时触发
    imageio = None


@dataclass
class Splats:
    means: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    colors: torch.Tensor
    sh_degree: int

    @property
    def count(self) -> int:
        return int(self.means.shape[0])


@dataclass
class VibrationField:
    data: np.ndarray
    total_frames: int
    moving_count: int
    scale: float

    def frame(self, frame_id: int) -> np.ndarray:
        # 只取当前帧，避免一次性把超大 vibration 文件全部读入内存。
        idx = int(frame_id) % self.total_frames
        return np.asarray(self.data[idx], dtype=np.float32) * self.scale


def parse_resolution(text: str) -> tuple[int, int]:
    try:
        width, height = text.lower().split("x", 1)
        return int(width), int(height)
    except Exception as exc:
        raise argparse.ArgumentTypeError("分辨率格式应为 WxH，例如 1920x1080") from exc


def parse_vec3(text: str) -> np.ndarray:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("三维向量格式应为 x,y,z")
    return np.asarray(values, dtype=np.float32)


def torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_splats(path: Path, device: torch.device) -> Splats:
    ckpt = torch_load(path)
    splats = ckpt["splats"] if isinstance(ckpt, dict) and "splats" in ckpt else ckpt

    means = splats["means"].detach().float()
    quats = F.normalize(splats["quats"].detach().float(), p=2, dim=-1)
    scales = torch.exp(splats["scales"].detach().float())
    opacities = torch.sigmoid(splats["opacities"].detach().float()).reshape(-1)
    sh0 = splats["sh0"].detach().float()
    shN = splats["shN"].detach().float()
    # Kriging-generated Gaussian checkpoints store SH coefficients as
    # (N, RGB, coefficient), while gsplat's renderer expects (N, coefficient, RGB).
    if sh0.ndim == 3 and sh0.shape[1:] == (3, 1):
        sh0 = sh0.transpose(1, 2).contiguous()
        if shN.ndim == 3 and shN.shape[1] == 3:
            shN = shN.transpose(1, 2).contiguous()
    if sh0.ndim != 3 or shN.ndim != 3 or sh0.shape[0] != shN.shape[0] or sh0.shape[2] != shN.shape[2]:
        raise ValueError(
            f"incompatible SH coefficient layouts: sh0={tuple(sh0.shape)}, shN={tuple(shN.shape)}"
        )
    colors = torch.cat([sh0, shN], dim=1)
    sh_degree = int(math.isqrt(colors.shape[1]) - 1)

    return Splats(
        means=means.to(device),
        quats=quats.to(device),
        scales=scales.to(device),
        opacities=opacities.to(device),
        colors=colors.to(device),
        sh_degree=sh_degree,
    )


def load_vibration(path: Path, moving_count: int, scale: float) -> VibrationField:
    if path.suffix.lower() == ".npy":
        raw = np.load(path, mmap_mode="r")
        if raw.ndim == 1:
            total = raw.size // (moving_count * 3)
            data = raw[: total * moving_count * 3].reshape(total, moving_count, 3)
        elif raw.ndim == 3 and raw.shape[1:] == (moving_count, 3):
            data = raw
            total = raw.shape[0]
        else:
            raise ValueError(f"无法识别 vibration npy 形状: {raw.shape}")
    else:
        scalar_count = path.stat().st_size // np.dtype(np.float32).itemsize
        total = scalar_count // (moving_count * 3)
        if total <= 1:
            raise ValueError("vibration 文件帧数不足，无法计算相邻帧稳定性。")
        data = np.memmap(path, dtype=np.float32, mode="r", shape=(total, moving_count, 3))

    if total <= 1:
        raise ValueError("vibration 文件帧数不足，无法计算相邻帧稳定性。")
    return VibrationField(data=data, total_frames=int(total), moving_count=moving_count, scale=float(scale))


def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    # OpenCV/gsplat camera coordinates: +Z forward, +Y down. The supplied
    # world-space `up` must project towards decreasing image rows.
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(vec2, up))
    vec1 = normalize(np.cross(vec2, vec0))
    return np.stack([vec0, vec1, vec2, position], axis=1).astype(np.float32)


def build_auto_camera(
    static_splats: Splats,
    moving_splats: Splats,
    target_mode: str,
    azimuth_deg: float,
    elevation_deg: float,
    distance_scale: float,
    target_override: np.ndarray | None,
    up_vector: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    static_means = static_splats.means.detach().cpu().numpy()
    moving_means = moving_splats.means.detach().cpu().numpy()
    combined_means = np.concatenate([static_means, moving_means], axis=0)

    combined_min = np.percentile(combined_means, 1.0, axis=0)
    combined_max = np.percentile(combined_means, 99.0, axis=0)
    center = 0.5 * (combined_min + combined_max)
    if target_override is not None:
        center = target_override.astype(np.float32)
        resolved_target_mode = "manual"
    else:
        if target_mode == "moving":
            target_means = moving_means
        elif target_mode == "static":
            target_means = static_means
        elif target_mode == "combined":
            target_means = combined_means
        else:
            raise ValueError(f"未知 target_mode: {target_mode}")
        # 用 1/99 分位中心，减少极端离群点对相机目标的影响。
        target_min = np.percentile(target_means, 1.0, axis=0)
        target_max = np.percentile(target_means, 99.0, axis=0)
        center = (0.5 * (target_min + target_max)).astype(np.float32)
        resolved_target_mode = target_mode
    # Fit the camera to the object being inspected.  The old implementation
    # always used the combined expanded terrain/object box, so a moving tower
    # occupied only a few pixels and the temporal audit falsely reported zero
    # visible moving Gaussians.  A manual target still uses the full combined
    # box because no target cloud is implied in that mode.
    if target_override is None and target_mode == "moving":
        fit_min, fit_max = target_min, target_max
    elif target_override is None and target_mode == "static":
        fit_min, fit_max = target_min, target_max
    elif target_override is None and target_mode == "combined":
        fit_min, fit_max = combined_min, combined_max
    else:
        fit_min, fit_max = combined_min, combined_max
    radius = float(np.linalg.norm(fit_max - fit_min) * 0.5)
    radius = max(radius, 1e-3)

    up = normalize(np.asarray(up_vector if up_vector is not None else [0.0, 0.0, 1.0], dtype=np.float32))
    reference = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(reference, up))) > 0.95:
        reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    horizontal_x = normalize(reference - np.dot(reference, up) * up)
    horizontal_y = normalize(np.cross(up, horizontal_x))
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    direction = normalize(
        math.cos(el) * (math.cos(az) * horizontal_x + math.sin(az) * horizontal_y)
        + math.sin(el) * up
    )
    eye = center + direction * radius * float(distance_scale)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :4] = viewmatrix(center - eye, up, eye)

    info = {
        "mode": "auto_look_at",
        "target_mode": resolved_target_mode,
        "center": center.tolist(),
        "eye": eye.tolist(),
        "azimuth_deg": float(azimuth_deg),
        "elevation_deg": float(elevation_deg),
        "distance_scale": float(distance_scale),
        "up_vector": up.tolist(),
        "bbox_percentile_1": fit_min.tolist(),
        "bbox_percentile_99": fit_max.tolist(),
        "combined_bbox_percentile_1": combined_min.tolist(),
        "combined_bbox_percentile_99": combined_max.tolist(),
    }
    return c2w, info


def load_camera_matrix(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "c2w" not in payload:
        raise KeyError("camera json 需要包含 c2w 字段。")
    c2w = np.asarray(payload["c2w"], dtype=np.float32)
    if c2w.shape == (3, 4):
        full = np.eye(4, dtype=np.float32)
        full[:3, :4] = c2w
        c2w = full
    if c2w.shape != (4, 4):
        raise ValueError("c2w 必须是 3x4 或 4x4 矩阵。")
    return c2w, {"mode": "json_c2w", "path": str(path)}


def build_intrinsics(width: int, height: int, fov_y_deg: float) -> np.ndarray:
    fy = 0.5 * height / math.tan(0.5 * math.radians(fov_y_deg))
    fx = fy
    return np.asarray(
        [[fx, 0.0, width * 0.5], [0.0, fy, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def build_frame_indices(total_frames: int, start: int, frames: int, stride: int) -> list[int]:
    if frames < 2:
        raise ValueError("--frames 至少需要为 2，才能计算相邻帧指标。")
    if stride < 1:
        raise ValueError("--frame_stride 至少需要为 1。")
    indices: list[int] = []
    cur = int(start)
    while len(indices) < frames and cur < total_frames:
        indices.append(cur)
        cur += stride
    if len(indices) < 2:
        raise ValueError("有效帧数不足，请检查 --frame_start / --frames / --frame_stride。")
    return indices


def render_rgb_alpha(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    sh_degree: int,
    viewmat: torch.Tensor,
    Kmat: torch.Tensor,
    width: int,
    height: int,
    radius_clip: float,
    near: float,
    far: float,
    white: torch.Tensor,
    flip_vertical: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rgb, alpha, meta = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors.unsqueeze(0),
        viewmat[None],
        Kmat[None],
        width,
        height,
        sh_degree=sh_degree,
        render_mode="RGB",
        radius_clip=radius_clip,
        near_plane=near,
        far_plane=far,
    )
    rgb_out = torch.clamp(rgb[0] + (1.0 - alpha[0]) * white, 0.0, 1.0)
    alpha_out = torch.clamp(alpha[0], 0.0, 1.0)
    if flip_vertical:
        rgb_out = rgb_out.flip(0)
        alpha_out = alpha_out.flip(0)
    return rgb_out.detach().cpu().numpy(), alpha_out.detach().cpu().numpy(), meta


def visible_gaussian_count(meta: dict[str, Any]) -> int | None:
    for key in ("radii", "radii2d", "radius"):
        value = meta.get(key)
        if value is not None and torch.is_tensor(value):
            return int((value > 0).sum().item())
    return None


def downsample_image(image: np.ndarray, factor: int, is_mask: bool = False) -> np.ndarray:
    if factor <= 1:
        return image
    height, width = image.shape[:2]
    new_size = (max(1, width // factor), max(1, height // factor))
    if cv2 is not None:
        interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
        return cv2.resize(image, new_size, interpolation=interpolation)
    return image[::factor, ::factor]


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask
    if cv2 is None:
        return mask
    kernel_size = pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def temporal_ssim(a: np.ndarray, b: np.ndarray) -> float:
    if skimage_ssim is None:
        raise RuntimeError("缺少 scikit-image，无法计算 temporal SSIM；请安装 scikit-image。")
    h, w = a.shape[:2]
    win_size = 7 if min(h, w) >= 7 else max(3, min(h, w) // 2 * 2 - 1)
    return float(skimage_ssim(a, b, channel_axis=2, data_range=1.0, win_size=win_size))


def masked_temporal_ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if ys.size < 64:
        return float("nan")
    pad = 8
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(mask.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(mask.shape[1], int(xs.max()) + pad + 1)

    crop_a = a[y0:y1, x0:x1].copy()
    crop_b = b[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]
    # 非背景区域在两帧中设为相同白色，避免动态构件影响背景 SSIM。
    crop_a[~crop_mask] = 1.0
    crop_b[~crop_mask] = 1.0
    return temporal_ssim(crop_a, crop_b)


def luma(image: np.ndarray) -> np.ndarray:
    return image[..., 0] * 0.2126 + image[..., 1] * 0.7152 + image[..., 2] * 0.0722


def pair_metrics(
    prev_rgb: np.ndarray,
    cur_rgb: np.ndarray,
    prev_moving_alpha: np.ndarray,
    cur_moving_alpha: np.ndarray,
    static_alpha: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, float | int]:
    metric_factor = max(1, int(args.metric_downsample))
    a = downsample_image(prev_rgb, metric_factor).astype(np.float32)
    b = downsample_image(cur_rgb, metric_factor).astype(np.float32)

    prev_ma = downsample_image(prev_moving_alpha[..., 0], metric_factor, is_mask=True)
    cur_ma = downsample_image(cur_moving_alpha[..., 0], metric_factor, is_mask=True)
    sta = downsample_image(static_alpha[..., 0], metric_factor, is_mask=True)

    moving_mask = (prev_ma > args.moving_alpha_threshold) | (cur_ma > args.moving_alpha_threshold)
    moving_mask = dilate_mask(moving_mask, max(0, int(round(args.mask_dilate / metric_factor))))
    static_mask = sta > args.static_alpha_threshold
    background_mask = static_mask & (~moving_mask)

    diff_rgb = np.abs(b - a)
    diff_pixel = diff_rgb.mean(axis=2)
    diff_luma = np.abs(luma(b) - luma(a))

    if int(background_mask.sum()) > 0:
        bg_pixel = diff_pixel[background_mask]
        bg_luma = diff_luma[background_mask]
        bg_flicker_mean = float(np.mean(bg_pixel))
        bg_flicker_p95 = float(np.percentile(bg_pixel, 95))
        bg_flicker_p99 = float(np.percentile(bg_pixel, 99))
        bg_luma_mean = float(np.mean(bg_luma))
        bg_luma_p95 = float(np.percentile(bg_luma, 95))
        bg_ssim = masked_temporal_ssim(a, b, background_mask)
    else:
        bg_flicker_mean = float("nan")
        bg_flicker_p95 = float("nan")
        bg_flicker_p99 = float("nan")
        bg_luma_mean = float("nan")
        bg_luma_p95 = float("nan")
        bg_ssim = float("nan")

    if int(moving_mask.sum()) > 0:
        mv_pixel = diff_pixel[moving_mask]
        moving_diff_mean = float(np.mean(mv_pixel))
        moving_diff_p95 = float(np.percentile(mv_pixel, 95))
        moving_diff_p99 = float(np.percentile(mv_pixel, 99))
    else:
        moving_diff_mean = float("nan")
        moving_diff_p95 = float("nan")
        moving_diff_p99 = float("nan")

    return {
        "temporal_ssim": temporal_ssim(a, b),
        "background_temporal_ssim": bg_ssim,
        "frame_diff_mean": float(np.mean(diff_pixel)),
        "frame_diff_p95": float(np.percentile(diff_pixel, 95)),
        "frame_diff_p99": float(np.percentile(diff_pixel, 99)),
        "background_flicker_mean": bg_flicker_mean,
        "background_flicker_p95": bg_flicker_p95,
        "background_flicker_p99": bg_flicker_p99,
        "background_luma_flicker_mean": bg_luma_mean,
        "background_luma_flicker_p95": bg_luma_p95,
        "moving_region_diff_mean": moving_diff_mean,
        "moving_region_diff_p95": moving_diff_p95,
        "moving_region_diff_p99": moving_diff_p99,
        "moving_pixel_fraction": float(np.mean(moving_mask)),
        "background_pixel_fraction": float(np.mean(background_mask)),
        "background_pixels": int(background_mask.sum()),
    }


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def summarize(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for key in keys:
        values = finite_values(rows, key)
        if not values:
            summary[key] = {"n": 0}
            continue
        summary[key] = {
            "n": len(values),
            "mean": float(statistics.fmean(values)),
            "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
            "min": float(min(values)),
            "p05": float(np.percentile(values, 5)),
            "median": float(statistics.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(max(values)),
        }
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 6) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path: Path, summary: dict[str, dict[str, float | int]], payload: dict[str, Any]) -> None:
    metric_names = [
        "temporal_ssim",
        "background_temporal_ssim",
        "background_flicker_mean",
        "background_flicker_p95",
        "background_flicker_p99",
        "moving_region_diff_mean",
        "moving_region_diff_p95",
        "moving_region_diff_p99",
        "frame_diff_mean",
        "frame_diff_p95",
        "frame_diff_p99",
    ]
    with path.open("w", encoding="utf-8") as file:
        file.write("# Dynamic Temporal Stability Metrics\n\n")
        file.write(f"- Case: {payload['case_name']}\n")
        file.write(f"- Resolution: {payload['resolution'][0]}x{payload['resolution'][1]}\n")
        file.write(f"- Evaluated frames: {len(payload['frame_indices'])}\n")
        file.write(f"- Frame stride: {payload['frame_stride']}\n")
        file.write(f"- Metric downsample: {payload['metric_downsample']}x\n\n")
        file.write("| Metric | N | Mean | Std | Min | Median | P95 | Max |\n")
        file.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for name in metric_names:
            row = summary.get(name, {"n": 0})
            file.write(
                "| "
                + " | ".join(
                    [
                        name,
                        fmt(row.get("n", 0), 0),
                        fmt(row.get("mean", float("nan"))),
                        fmt(row.get("std", float("nan"))),
                        fmt(row.get("min", float("nan"))),
                        fmt(row.get("median", float("nan"))),
                        fmt(row.get("p95", float("nan"))),
                        fmt(row.get("max", float("nan"))),
                    ]
                )
                + " |\n"
            )
        file.write("\n")
        file.write(
            "Interpretation: higher temporal SSIM and background temporal SSIM indicate better "
            "frame-to-frame consistency; lower background flicker indicates less non-moving-region "
            "brightness jitter after excluding projected moving components.\n"
        )


def save_png(path: Path, image: np.ndarray) -> None:
    if imageio is None:
        raise RuntimeError("缺少 imageio，无法保存 PNG；请安装 imageio 或关闭 --save_frames。")
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offscreen temporal SSIM / flicker evaluation for dynamic 3DGS.")
    parser.add_argument("--static_ckpt", type=Path, required=True)
    parser.add_argument("--moving_ckpt", type=Path, required=True)
    parser.add_argument("--vib_path", type=Path, required=True)
    parser.add_argument("--scale_disp", type=float, default=50.0)
    parser.add_argument("--resolution", type=parse_resolution, default=parse_resolution("1920x1080"))
    parser.add_argument("--port", type=int, default=None, help="兼容 auto_play_highres.py；离屏评估中不使用。")
    parser.add_argument("--frames", type=int, default=None, help="评估帧数；未指定时使用 --benchmark_window 或 30。")
    parser.add_argument("--benchmark_window", type=int, default=None, help="兼容 auto_play 命令；当 --frames 未设置时作为评估帧数。")
    parser.add_argument("--benchmark_log", type=Path, default=None, help="兼容 auto_play 命令；指定 JSON 输出路径。")
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--case_name", type=str, default="Registered translation playback")
    parser.add_argument("--output_dir", type=Path, default=Path("revision_experiments/results/dynamic_temporal_stability"))
    parser.add_argument("--output_prefix", type=str, default="dynamic_temporal_stability")
    parser.add_argument("--save_frames", action="store_true", help="保存被评估帧 PNG，便于人工核查。")
    parser.add_argument("--save_moving_frames", action="store_true",
                        help="另外保存同相机的移动 Gaussian 子集 GS 渲染，便于检查塔体遮挡。")
    parser.add_argument("--save_undeformed", action="store_true",
                        help="保存同相机的零位移完整场景和移动子集 GS 基准图；档案第0帧不一定是零位移。")
    parser.add_argument("--save_every", type=int, default=1, help="开启 --save_frames 后每隔多少帧保存一张。")

    parser.add_argument("--camera_json", type=Path, default=None, help="可选，包含 c2w 的相机 JSON。")
    parser.add_argument(
        "--target_mode",
        type=str,
        default="moving",
        choices=("moving", "static", "combined"),
        help="未提供 --target 时，自动相机看向哪个点集的分位中心。",
    )
    parser.add_argument("--target", type=parse_vec3, default=None, help="自动相机看向的点，格式 x,y,z。")
    parser.add_argument("--gs_to_fe_transform", type=Path, default=None,
                        help="可选 4x4 高斯坐标到 FE 坐标的变换，用 FE +Z 设置相机向上方向")
    parser.add_argument("--up_vector", type=parse_vec3, default=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
                        help="相机在高斯坐标系中的向上方向 x,y,z；提供变换时由变换覆盖")
    parser.add_argument("--azimuth_deg", type=float, default=-35.0)
    parser.add_argument("--elevation_deg", type=float, default=18.0)
    parser.add_argument("--distance_scale", type=float, default=1.0)
    parser.add_argument("--fov_y_deg", type=float, default=45.0)

    parser.add_argument("--moving_alpha_threshold", type=float, default=0.02)
    parser.add_argument("--static_alpha_threshold", type=float, default=0.02)
    parser.add_argument("--mask_dilate", type=int, default=3, help="moving mask 膨胀像素，避免边缘泄漏。")
    parser.add_argument("--metric_downsample", type=int, default=1, help="指标计算前下采样倍数，1 表示原分辨率。")

    parser.add_argument("--radius_clip", type=float, default=3.0)
    parser.add_argument("--near", type=float, default=0.01)
    parser.add_argument("--far", type=float, default=1e10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # 离屏 PNG 使用标准图像坐标（数组第 0 行显示在图像顶部）。
    # viewer 回调为了适配 nerfview 的坐标约定会翻转一次，但这里直接保存
    # rasterization 输出，因此默认不翻转；需要复现旧 viewer 方向时显式指定。
    parser.add_argument("--flip_vertical", action="store_true", help="上下翻转离屏输出（兼容旧 viewer 方向）。")
    parser.add_argument(
        "--no_flip_vertical",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.frames = int(args.frames or args.benchmark_window or 30)
    width, height = args.resolution
    device = torch.device(args.device)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] static={args.static_ckpt}")
    static_splats = load_splats(args.static_ckpt, device)
    print(f"[load] moving={args.moving_ckpt}")
    moving_splats = load_splats(args.moving_ckpt, device)
    vibration = load_vibration(args.vib_path, moving_splats.count, args.scale_disp)
    frame_indices = build_frame_indices(vibration.total_frames, args.frame_start, args.frames, args.frame_stride)

    if args.camera_json is not None:
        c2w_np, camera_info = load_camera_matrix(args.camera_json)
    else:
        camera_up = (physical_up_in_gaussian_frame(args.gs_to_fe_transform)
                     if args.gs_to_fe_transform is not None else args.up_vector)
        c2w_np, camera_info = build_auto_camera(
            static_splats,
            moving_splats,
            args.target_mode,
            args.azimuth_deg,
            args.elevation_deg,
            args.distance_scale,
            args.target,
            camera_up,
        )
    K_np = build_intrinsics(width, height, args.fov_y_deg)
    viewmat = torch.from_numpy(np.linalg.inv(c2w_np)).float().to(device)
    Kmat = torch.from_numpy(K_np).float().to(device)
    white = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
    # `--no_flip_vertical` remains a harmless backwards-compatible alias.  The
    # default is now the natural upright orientation for saved PNG frames.
    flip_vertical = bool(args.flip_vertical and not args.no_flip_vertical)

    print(f"[render] resolution={width}x{height}, frames={len(frame_indices)}, stride={args.frame_stride}")
    static_rgb, static_alpha, static_meta = render_rgb_alpha(
        static_splats.means,
        static_splats.quats,
        static_splats.scales,
        static_splats.opacities,
        static_splats.colors,
        static_splats.sh_degree,
        viewmat,
        Kmat,
        width,
        height,
        args.radius_clip,
        args.near,
        args.far,
        white,
        flip_vertical,
    )

    if args.save_undeformed:
        for label, groups in (("undeformed", (static_splats, moving_splats)),
                              ("moving_undeformed", (moving_splats,))):
            reference, _, _ = render_rgb_alpha(
                torch.cat([group.means for group in groups]),
                torch.cat([group.quats for group in groups]),
                torch.cat([group.scales for group in groups]),
                torch.cat([group.opacities for group in groups]),
                torch.cat([group.colors for group in groups]),
                moving_splats.sh_degree, viewmat, Kmat, width, height,
                args.radius_clip, args.near, args.far, white, flip_vertical,
            )
            save_png(output_dir / f"{label}.png", reference)

    frame_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    prev_rgb: np.ndarray | None = None
    prev_moving_alpha: np.ndarray | None = None
    prev_frame_id: int | None = None

    for local_idx, frame_id in enumerate(frame_indices):
        disp = torch.from_numpy(vibration.frame(frame_id)).float().to(device)
        pos_m = moving_splats.means + disp

        all_means = torch.cat([static_splats.means, pos_m], dim=0)
        all_quats = torch.cat([static_splats.quats, moving_splats.quats], dim=0)
        all_scales = torch.cat([static_splats.scales, moving_splats.scales], dim=0)
        all_opacities = torch.cat([static_splats.opacities, moving_splats.opacities], dim=0)
        all_colors = torch.cat([static_splats.colors, moving_splats.colors], dim=0)

        t0 = time.perf_counter()
        rgb, _, meta = render_rgb_alpha(
            all_means,
            all_quats,
            all_scales,
            all_opacities,
            all_colors,
            static_splats.sh_degree,
            viewmat,
            Kmat,
            width,
            height,
            args.radius_clip,
            args.near,
            args.far,
            white,
            flip_vertical,
        )
        moving_rgb, moving_alpha, moving_meta = render_rgb_alpha(
            pos_m,
            moving_splats.quats,
            moving_splats.scales,
            moving_splats.opacities,
            moving_splats.colors,
            moving_splats.sh_degree,
            viewmat,
            Kmat,
            width,
            height,
            args.radius_clip,
            args.near,
            args.far,
            white,
            flip_vertical,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        render_ms = (time.perf_counter() - t0) * 1000.0

        frame_rows.append(
            {
                "frame": int(frame_id),
                "render_ms": float(render_ms),
                "visible_gaussians_all": visible_gaussian_count(meta),
                "visible_gaussians_moving": visible_gaussian_count(moving_meta),
            }
        )

        if args.save_frames and local_idx % max(1, args.save_every) == 0:
            save_png(output_dir / "frames" / f"frame_{frame_id:05d}.png", rgb)
        if args.save_moving_frames and local_idx % max(1, args.save_every) == 0:
            save_png(output_dir / "moving_frames" / f"frame_{frame_id:05d}.png", moving_rgb)

        if prev_rgb is not None and prev_moving_alpha is not None and prev_frame_id is not None:
            metrics = pair_metrics(prev_rgb, rgb, prev_moving_alpha, moving_alpha, static_alpha, args)
            pair_rows.append(
                {
                    "frame_a": int(prev_frame_id),
                    "frame_b": int(frame_id),
                    **metrics,
                }
            )

        prev_rgb = rgb
        prev_moving_alpha = moving_alpha
        prev_frame_id = int(frame_id)
        print(f"[frame] {local_idx + 1:03d}/{len(frame_indices):03d} id={frame_id} render_ms={render_ms:.2f}")

    if not pair_rows:
        raise RuntimeError("没有生成相邻帧指标，请至少评估 2 帧。")

    metric_keys = [
        "temporal_ssim",
        "background_temporal_ssim",
        "frame_diff_mean",
        "frame_diff_p95",
        "frame_diff_p99",
        "background_flicker_mean",
        "background_flicker_p95",
        "background_flicker_p99",
        "background_luma_flicker_mean",
        "background_luma_flicker_p95",
        "moving_region_diff_mean",
        "moving_region_diff_p95",
        "moving_region_diff_p99",
        "moving_pixel_fraction",
        "background_pixel_fraction",
    ]
    summary = summarize(pair_rows, metric_keys)
    render_summary = summarize(frame_rows, ["render_ms"])

    payload = {
        "case_name": args.case_name,
        "command": shlex.join([sys.executable, *sys.argv]),
        "static_ckpt": str(args.static_ckpt),
        "moving_ckpt": str(args.moving_ckpt),
        "vib_path": str(args.vib_path),
        "scale_disp": float(args.scale_disp),
        "resolution": [width, height],
        "frame_indices": frame_indices,
        "frame_stride": int(args.frame_stride),
        "metric_downsample": int(args.metric_downsample),
        "moving_alpha_threshold": float(args.moving_alpha_threshold),
        "static_alpha_threshold": float(args.static_alpha_threshold),
        "mask_dilate": int(args.mask_dilate),
        "flip_vertical": bool(flip_vertical),
        "camera": camera_info,
        "intrinsics": K_np.tolist(),
        "device": str(device),
        "static_visible_gaussians": visible_gaussian_count(static_meta),
        "summary": summary,
        "render_summary": render_summary,
        "frames": frame_rows,
        "pairs": pair_rows,
    }

    if args.benchmark_log is not None:
        json_path = args.benchmark_log
        json_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path = json_path.with_suffix(".csv")
        md_path = json_path.with_suffix(".md")
    else:
        json_path = output_dir / f"{args.output_prefix}.json"
        csv_path = output_dir / f"{args.output_prefix}.csv"
        md_path = output_dir / f"{args.output_prefix}.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(csv_path, pair_rows)
    write_markdown(md_path, summary, payload)

    print("\n[done] outputs:")
    print(f"  JSON: {json_path}")
    print(f"  CSV : {csv_path}")
    print(f"  MD  : {md_path}")
    print("[summary]")
    for key in (
        "temporal_ssim",
        "background_temporal_ssim",
        "background_flicker_mean",
        "background_flicker_p95",
        "moving_region_diff_mean",
    ):
        row = summary.get(key, {})
        print(f"  {key}: mean={fmt(row.get('mean', float('nan')))}, p95={fmt(row.get('p95', float('nan')))}")


if __name__ == "__main__":
    main()
