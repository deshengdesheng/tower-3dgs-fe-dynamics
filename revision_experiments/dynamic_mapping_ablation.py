#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""动态映射核心模块消融实验表生成脚本。

该脚本不重新渲染图像，而是在 FE 离散杆件上生成均匀探针点，
以线性插值的 FE 位移/旋转/尺度增量作为参考，比较不同 sparse-to-dense
映射策略的位移误差、旋转误差、尺度误差和跨杆件串扰。

注意：这是动态映射层消融，不是图像渲染质量消融。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


IDENTITY_Q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


@dataclass
class VariantMetrics:
    method: str
    removed_component: str
    probe_points: int
    disp_rmse: float
    disp_mae: float
    polar_rotation_consistency_deg: float
    polar_stretch_consistency: float
    cross_member_leakage: float


def q_normalize(q: np.ndarray) -> np.ndarray:
    return q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)


def q_inverse(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out / (np.sum(q * q, axis=-1, keepdims=True) + 1e-12)


def q_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    out = np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )
    return q_normalize(out)


def q_slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    q0 = q_normalize(q0)
    q1 = q_normalize(q1)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)
    dot = np.clip(dot, -1.0 + 1e-6, 1.0 - 1e-6)
    omega = np.arccos(dot)
    sin_omega = np.sin(omega)
    t = t.reshape(-1, 1)
    out = (
        np.sin((1.0 - t) * omega) / sin_omega * q0
        + np.sin(t * omega) / sin_omega * q1
    )
    linear = (1.0 - t) * q0 + t * q1
    out = np.where(dot > 1.0 - 1e-5, linear, out)
    return q_normalize(out)


def q_angle_error_deg(q_pred: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    dq = q_multiply(q_pred, q_inverse(q_ref))
    w = np.clip(np.abs(dq[:, 0]), -1.0, 1.0)
    return 2.0 * np.arccos(w) * 180.0 / np.pi


def shortest_rotation_quat(v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    """计算从 v0 到 v1 的最短旋转四元数。"""
    v0 = v0 / (np.linalg.norm(v0, axis=-1, keepdims=True) + 1e-12)
    v1 = v1 / (np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-12)
    dot = np.sum(v0 * v1, axis=-1)
    cross = np.cross(v0, v1)
    q = np.concatenate([(1.0 + dot)[:, None], cross], axis=1).astype(np.float32)

    # 处理近 180 度反向的退化情况。
    bad = np.linalg.norm(q, axis=1) < 1e-8
    if np.any(bad):
        fallback = np.zeros((int(np.sum(bad)), 4), dtype=np.float32)
        basis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        alt = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        axes = np.cross(v0[bad], basis[None, :])
        small = np.linalg.norm(axes, axis=1) < 1e-8
        axes[small] = np.cross(v0[bad][small], alt[None, :])
        axes = axes / (np.linalg.norm(axes, axis=1, keepdims=True) + 1e-12)
        fallback[:, 1:] = axes
        q[bad] = fallback
    return q_normalize(q)


def weighted_quaternion_average(q_nodes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # 用第一项作为符号参考，避免四元数正负等价导致抵消。
    ref = q_nodes[:, 0:1, :]
    q_aligned = np.where(
        np.sum(q_nodes * ref, axis=-1, keepdims=True) < 0.0,
        -q_nodes,
        q_nodes,
    )
    return q_normalize(np.sum(q_aligned * weights[:, :, None], axis=1))


def infer_node_groups(lines: np.ndarray, node_count: int) -> np.ndarray:
    groups = np.zeros(node_count, dtype=np.int32)
    for _, a, b, group in lines:
        groups[a] = group
        groups[b] = group
    return groups


def estimate_node_axes(points: np.ndarray, lines: np.ndarray) -> np.ndarray:
    axes = np.zeros_like(points, dtype=np.float32)
    counts = np.zeros(points.shape[0], dtype=np.int32)
    for _, a, b, _ in lines:
        direction = points[b] - points[a]
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            continue
        direction = direction / norm
        for idx in (a, b):
            aligned = direction
            if counts[idx] > 0 and float(np.dot(axes[idx], direction)) < 0.0:
                aligned = -direction
            axes[idx] += aligned
            counts[idx] += 1
    axes[counts == 0] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return axes / (np.linalg.norm(axes, axis=1, keepdims=True) + 1e-12)


def derive_node_motion_from_lines(
    points: np.ndarray, displacements: np.ndarray, lines: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """从 FE 杆件两端节点位移派生节点旋转和轴向拉伸参考场。"""
    node_count = points.shape[0]
    q_sum = np.zeros((node_count, 4), dtype=np.float64)
    log_scale_sum = np.zeros((node_count, 3), dtype=np.float64)
    counts = np.zeros(node_count, dtype=np.int32)

    a = lines[:, 1].astype(np.int64)
    b = lines[:, 2].astype(np.int64)
    valid = (a >= 0) & (b >= 0) & (a < node_count) & (b < node_count) & (a != b)
    a = a[valid]
    b = b[valid]
    if a.size == 0:
        return (
            np.broadcast_to(IDENTITY_Q, (node_count, 4)).copy(),
            np.ones((node_count, 3), dtype=np.float32),
        )

    edge0 = points[b] - points[a]
    edge1 = points[b] + displacements[b] - points[a] - displacements[a]
    edge0_len = np.linalg.norm(edge0, axis=1)
    edge1_len = np.linalg.norm(edge1, axis=1)
    good = edge0_len > 1e-12
    a = a[good]
    b = b[good]
    edge0 = edge0[good]
    edge1 = edge1[good]
    edge0_len = edge0_len[good]
    edge1_len = edge1_len[good]

    edge_q = shortest_rotation_quat(edge0, edge1)
    axial_log_stretch = np.log(np.clip(edge1_len / (edge0_len + 1e-12), 1e-6, None))
    edge_log_scale = np.stack(
        [axial_log_stretch, np.zeros_like(axial_log_stretch), np.zeros_like(axial_log_stretch)],
        axis=1,
    )

    for idx, (u, v) in enumerate(zip(a, b)):
        for node in (int(u), int(v)):
            q = edge_q[idx]
            if counts[node] > 0 and float(np.dot(q_sum[node], q)) < 0.0:
                q = -q
            q_sum[node] += q
            log_scale_sum[node] += edge_log_scale[idx]
            counts[node] += 1

    q_delta = np.broadcast_to(IDENTITY_Q, (node_count, 4)).copy()
    scale_factors = np.ones((node_count, 3), dtype=np.float32)
    used = counts > 0
    q_delta[used] = q_normalize(q_sum[used].astype(np.float32))
    scale_factors[used] = np.exp((log_scale_sum[used] / counts[used, None]).astype(np.float32))
    return q_delta.astype(np.float32), scale_factors.astype(np.float32)


def select_lines_by_motion(
    points: np.ndarray, displacements: np.ndarray, lines: np.ndarray, args: argparse.Namespace
) -> np.ndarray:
    """按节点位移幅值选择高响应杆件组或线段。"""
    if args.top_groups <= 0 and args.top_lines <= 0:
        return lines

    node_count = points.shape[0]
    valid = (
        (lines[:, 1] >= 0)
        & (lines[:, 2] >= 0)
        & (lines[:, 1] < node_count)
        & (lines[:, 2] < node_count)
    )
    lines = lines[valid]
    mag = np.linalg.norm(displacements, axis=1)
    line_scores = 0.5 * (mag[lines[:, 1]] + mag[lines[:, 2]])

    if args.top_groups > 0:
        groups = lines[:, 3].astype(np.int64)
        group_rows: list[tuple[float, int, int]] = []
        for group in np.unique(groups):
            mask = groups == group
            count = int(np.sum(mask))
            if count < args.min_lines_per_group:
                continue
            if args.group_score == "max":
                score = float(np.max(line_scores[mask]))
            else:
                score = float(np.mean(line_scores[mask]))
            group_rows.append((score, count, int(group)))
        group_rows.sort(reverse=True)
        keep_groups = {group for _, _, group in group_rows[: args.top_groups]}
        selected = lines[np.isin(groups, list(keep_groups))]
    else:
        order = np.argsort(-line_scores)
        selected = lines[order[: args.top_lines]]

    if selected.size == 0:
        raise ValueError("高位移线段选择结果为空，请降低 --min-lines-per-group 或增大 --top-groups。")
    return selected


def compact_to_selected_lines(
    points: np.ndarray, displacements: np.ndarray, lines: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """只保留被选中线段涉及的节点，并重映射线段端点索引。"""
    selected_nodes = np.unique(lines[:, 1:3].reshape(-1).astype(np.int64))
    old_to_new = np.full(points.shape[0], -1, dtype=np.int64)
    old_to_new[selected_nodes] = np.arange(selected_nodes.shape[0], dtype=np.int64)
    compact_lines = lines.copy()
    compact_lines[:, 1] = old_to_new[compact_lines[:, 1]]
    compact_lines[:, 2] = old_to_new[compact_lines[:, 2]]
    return (
        points[selected_nodes],
        displacements[selected_nodes],
        compact_lines,
        selected_nodes,
    )


def build_probe_points(
    points: np.ndarray,
    displacements: np.ndarray,
    q_delta: np.ndarray,
    scale_factors: np.ndarray,
    lines: np.ndarray,
    samples_per_segment: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probe_points: list[np.ndarray] = []
    ref_disp: list[np.ndarray] = []
    ref_q: list[np.ndarray] = []
    ref_log_scale: list[np.ndarray] = []
    ref_group: list[int] = []
    ref_axis: list[np.ndarray] = []
    t_values = np.linspace(0.0, 1.0, samples_per_segment, dtype=np.float32)
    log_scales = np.log(np.clip(scale_factors, 1e-6, None))

    for _, a, b, group in lines:
        axis = points[b] - points[a]
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        for t in t_values:
            probe_points.append(points[a] * (1.0 - t) + points[b] * t)
            ref_disp.append(displacements[a] * (1.0 - t) + displacements[b] * t)
            ref_q.append(q_slerp(q_delta[a : a + 1], q_delta[b : b + 1], np.array([t]))[0])
            ref_log_scale.append(log_scales[a] * (1.0 - t) + log_scales[b] * t)
            ref_group.append(int(group))
            ref_axis.append(axis)

    return (
        np.asarray(probe_points, dtype=np.float32),
        np.asarray(ref_disp, dtype=np.float32),
        np.asarray(ref_q, dtype=np.float32),
        np.asarray(ref_log_scale, dtype=np.float32),
        np.asarray(ref_group, dtype=np.int32),
        np.asarray(ref_axis, dtype=np.float32),
    )


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    sums = np.sum(weights, axis=1, keepdims=True)
    sums[sums[:, 0] <= 1e-12] = 1.0
    return weights / sums


def make_metrics(
    method: str,
    removed_component: str,
    pred_disp: np.ndarray,
    pred_q: np.ndarray,
    pred_log_scale: np.ndarray,
    cross_leakage: np.ndarray,
    ref_disp: np.ndarray,
    ref_q: np.ndarray,
    ref_log_scale: np.ndarray,
) -> VariantMetrics:
    disp_error = np.linalg.norm(pred_disp - ref_disp, axis=1)
    return VariantMetrics(
        method=method,
        removed_component=removed_component,
        probe_points=int(ref_disp.shape[0]),
        disp_rmse=float(np.sqrt(np.mean(disp_error * disp_error))),
        disp_mae=float(np.mean(disp_error)),
        polar_rotation_consistency_deg=float(np.mean(q_angle_error_deg(pred_q, ref_q))),
        polar_stretch_consistency=float(np.sqrt(np.mean((pred_log_scale - ref_log_scale) ** 2))),
        cross_member_leakage=float(np.mean(cross_leakage) * 100.0),
    )


def compute_ablation(args: argparse.Namespace) -> list[VariantMetrics]:
    points_raw = np.loadtxt(args.points_file, comments="#")
    lines = np.loadtxt(args.lines_file, comments="#", dtype=np.int64)

    points = points_raw[:, 1:4].astype(np.float32)
    displacements = points_raw[:, 4:7].astype(np.float32)

    lines = select_lines_by_motion(points, displacements, lines, args)
    selected_node_ids = None
    if args.compact_selected_nodes:
        points, displacements, lines, selected_node_ids = compact_to_selected_lines(
            points, displacements, lines
        )

    if args.derive_node_motion or args.transform_file is None:
        q_delta, scale_factors = derive_node_motion_from_lines(points, displacements, lines)
        node_groups = infer_node_groups(lines, points.shape[0])
    else:
        transform = np.load(args.transform_file, allow_pickle=True)
        if selected_node_ids is not None and transform["rotation_delta"].shape[0] >= int(selected_node_ids.max()) + 1:
            q_delta = transform["rotation_delta"].astype(np.float32)[selected_node_ids]
            scale_factors = transform["scale_factors"].astype(np.float32)[selected_node_ids]
            node_groups = (
                transform["point_groups"].astype(np.int32)[selected_node_ids]
                if "point_groups" in transform.files
                else infer_node_groups(lines, points.shape[0])
            )
        elif transform["rotation_delta"].shape[0] != points.shape[0]:
            raise ValueError(
                f"transform 节点数 {transform['rotation_delta'].shape[0]} 与 points 节点数 {points.shape[0]} 不一致；"
                "请使用 --derive-node-motion 从当前 FE 线段直接派生。"
            )
        else:
            q_delta = transform["rotation_delta"].astype(np.float32)
            scale_factors = transform["scale_factors"].astype(np.float32)
            node_groups = (
                transform["point_groups"].astype(np.int32)
                if "point_groups" in transform.files
                else infer_node_groups(lines, points.shape[0])
            )
    if len(np.unique(node_groups)) <= 1:
        node_groups = infer_node_groups(lines, points.shape[0])

    node_axes = estimate_node_axes(points, lines)
    node_log_scales = np.log(np.clip(scale_factors, 1e-6, None))

    (
        probe_points,
        ref_disp,
        ref_q,
        ref_log_scale,
        ref_group,
        ref_axis,
    ) = build_probe_points(
        points,
        displacements,
        q_delta,
        scale_factors,
        lines,
        samples_per_segment=args.samples_per_segment,
    )

    diff = probe_points[:, None, :] - points[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    dist = np.sqrt(dist2)
    valid = dist <= args.radius
    nearest = np.argmin(dist, axis=1)
    nearest_weight = np.exp(-dist[np.arange(probe_points.shape[0]), nearest] / args.radius)

    identity_q = np.broadcast_to(IDENTITY_Q, ref_q.shape).copy()
    zero_scale = np.zeros_like(ref_log_scale)
    all_q = np.broadcast_to(q_delta, (probe_points.shape[0], q_delta.shape[0], 4))
    cross_nearest = (node_groups[nearest] != ref_group).astype(np.float32)

    variants: list[VariantMetrics] = []

    # 基线1：只传播平移，不传播局部右极分解得到的旋转和尺度。
    variants.append(
        make_metrics(
            "w/o local right polar-decomposition model",
            "Section 4.1 local right polar-decomposition model",
            displacements[nearest] * nearest_weight[:, None],
            identity_q,
            zero_scale,
            cross_nearest,
            ref_disp,
            ref_q,
            ref_log_scale,
        )
    )

    # 基线2：使用最近节点的极分解旋转/尺度，不执行 Eq. (21) 的加权插值。
    nearest_q = q_normalize(
        (1.0 - nearest_weight[:, None]) * IDENTITY_Q[None, :]
        + nearest_weight[:, None] * q_delta[nearest]
    )
    variants.append(
        make_metrics(
            "w/o weighted interpolation in Eq. (21)",
            "Weighted aggregation in Eq. (21)",
            displacements[nearest] * nearest_weight[:, None],
            nearest_q,
            node_log_scales[nearest] * nearest_weight[:, None],
            cross_nearest,
            ref_disp,
            ref_q,
            ref_log_scale,
        )
    )

    # 基线3：改用各向同性权重，去掉 Eq. (20) 的杆件轴向约束。
    sigma_perp = args.radius / 2.0
    isotropic_weights = normalize_weights(np.exp(-0.5 * dist2 / (sigma_perp**2)) * valid)
    variants.append(
        make_metrics(
            "w/o member-axis-constrained anisotropic kernel",
            "Member-axis-constrained anisotropic kernel in Eq. (20)",
            isotropic_weights @ displacements,
            weighted_quaternion_average(all_q, isotropic_weights),
            isotropic_weights @ node_log_scales,
            np.sum(isotropic_weights * (node_groups[None, :] != ref_group[:, None]), axis=1),
            ref_disp,
            ref_q,
            ref_log_scale,
        )
    )

    # 基线4：保留轴向/径向各向异性，但去掉 Eq. (20) 的方向一致性项。
    d_para = np.sum(diff * node_axes[None, :, :], axis=2)
    d_perp = np.maximum(dist2 - d_para * d_para, 0.0)
    sigma_para = sigma_perp * args.sigma_ratio
    anisotropic_spatial = (
        np.exp(-0.5 * (d_para * d_para / (sigma_para**2) + d_perp / (sigma_perp**2)))
        * valid
    )
    no_direction_weights = normalize_weights(anisotropic_spatial)
    variants.append(
        make_metrics(
            "w/o direction-consistency term",
            "Direction-consistency term in Eq. (20)",
            no_direction_weights @ displacements,
            weighted_quaternion_average(all_q, no_direction_weights),
            no_direction_weights @ node_log_scales,
            np.sum(no_direction_weights * (node_groups[None, :] != ref_group[:, None]), axis=1),
            ref_disp,
            ref_q,
            ref_log_scale,
        )
    )

    # 完整方法：极分解增量 + 轴向各向异性 + 方向一致性。
    dir_cos = np.abs(np.sum(ref_axis[:, None, :] * node_axes[None, :, :], axis=2))
    full_weights = normalize_weights(
        anisotropic_spatial * np.exp(-args.beta * (1.0 - dir_cos))
    )
    variants.append(
        make_metrics(
            "Full proposed",
            "None",
            full_weights @ displacements,
            weighted_quaternion_average(all_q, full_weights),
            full_weights @ node_log_scales,
            np.sum(full_weights * (node_groups[None, :] != ref_group[:, None]), axis=1),
            ref_disp,
            ref_q,
            ref_log_scale,
        )
    )

    return variants


def write_csv(path: Path, rows: list[VariantMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Method",
        "Removed / weakened component",
        "Probe points",
        "FE displacement consistency RMSE",
        "FE displacement consistency MAE",
        "Polar rotation consistency (deg)",
        "Polar stretch consistency (log-RMSE)",
        "Cross-member leakage (%)",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Method": row.method,
                    "Removed / weakened component": row.removed_component,
                    "Probe points": row.probe_points,
                    "FE displacement consistency RMSE": row.disp_rmse,
                    "FE displacement consistency MAE": row.disp_mae,
                    "Polar rotation consistency (deg)": row.polar_rotation_consistency_deg,
                    "Polar stretch consistency (log-RMSE)": row.polar_stretch_consistency,
                    "Cross-member leakage (%)": row.cross_member_leakage,
                }
            )


def fmt(value: float) -> str:
    return f"{value:.4f}"


def write_md(path: Path, rows: list[VariantMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Method",
        "Removed / weakened component",
        "Probe points",
        "FE displacement consistency RMSE",
        "FE displacement consistency MAE",
        "Polar rotation consistency (deg)",
        "Polar stretch consistency (log-RMSE)",
        "Cross-member leakage (%)",
    ]
    with path.open("w", encoding="utf-8") as file:
        file.write("| " + " | ".join(headers) + " |\n")
        file.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            values = [
                row.method,
                row.removed_component,
                str(row.probe_points),
                fmt(row.disp_rmse),
                fmt(row.disp_mae),
                fmt(row.polar_rotation_consistency_deg),
                fmt(row.polar_stretch_consistency),
                fmt(row.cross_member_leakage),
            ]
            file.write("| " + " | ".join(values) + " |\n")
        file.write("\n")
        file.write(
            "Note: Probe points are uniformly sampled on FE members. "
            "The displacement columns measure consistency with FE member interpolation. "
            "The rotation/stretch columns measure consistency with the FE-derived local polar-decomposition field, not external measured ground truth. "
            "Lower cross-member leakage indicates less deformation mixing across members.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dynamic mapping ablation table.")
    parser.add_argument("--points-file", type=Path, required=True)
    parser.add_argument("--lines-file", type=Path, required=True)
    parser.add_argument("--transform-file", type=Path, default=None)
    parser.add_argument("--derive-node-motion", action="store_true", help="从当前 FE 线段直接派生节点旋转/拉伸，不读取 transform npz。")
    parser.add_argument("--output-csv", type=Path, default=Path("results/dynamic_mapping_ablation/dynamic_mapping_ablation.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("results/dynamic_mapping_ablation/dynamic_mapping_ablation.md"))
    parser.add_argument("--samples-per-segment", type=int, default=21)
    parser.add_argument("--radius", type=float, default=0.02)
    parser.add_argument("--sigma-ratio", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=8.0)
    parser.add_argument("--top-groups", type=int, default=0, help="按杆件组平均/最大位移选择前 K 个高响应杆件组。")
    parser.add_argument("--top-lines", type=int, default=0, help="按线段两端平均位移选择前 K 条高响应线段。")
    parser.add_argument("--min-lines-per-group", type=int, default=1, help="选择 top-groups 时保留的最小组内线段数。")
    parser.add_argument("--group-score", choices=["mean", "max"], default="mean", help="top-groups 的位移评分方式。")
    parser.add_argument("--compact-selected-nodes", action=argparse.BooleanOptionalAction, default=True, help="只用被选中线段涉及的节点作为插值支持集。")
    args = parser.parse_args()

    rows = compute_ablation(args)
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows)
    print(f"CSV: {args.output_csv}")
    print(f"Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
