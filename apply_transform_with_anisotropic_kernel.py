"""apply_transform_with_anisotropic_kernel.py

将 *filtered* 高斯模型根据已计算的全局变形，通过各向异性高斯核与方向一致性权重
映射到对应离散点并应用平移、切线/SCGD 旋转及缩放。

该脚本是对apply_transform_to_filtered_model_2.py的增强版本，引入了：
1. 基于主轴方向的各向异性高斯核
2. 方向一致性权重
3. 抑制跨杆件耦合的优化算法

CLI 参数：
  --filtered-pt-file       待变形的 *.pt 文件（含 splats）
  --original-pt-file       merged_original_model.pt，用于 KD-Tree 耦合
  --transform-file         merged_transform_relation.npz（或其 superset）
  --output-pt-file         变形后 *.pt 保存路径
  --output-npz-file        保存筛选模型各点对应的变形信息
  --sigma-ratio            轴向/径向方差比例，默认10.0（越大杆件方向性越强）
  --beta                   方向一致性权重系数，默认5.0（越大方向约束越强）
  --radius                 变形影响半径，默认1.5

示例：
python apply_transform_with_anisotropic_kernel.py \
  --filtered-pt-file  ./results/test_filtered.pt \
  --original-pt-file  ./results/merged_original_model.pt \
  --transform-file    ./results/merged_transform_relation.npz \
  --output-pt-file    ./results/test_filtered_transformed.pt \
  --output-npz-file   ./results/test_filtered_transform_relation.npz \
  --sigma-ratio 15.0 \
  --beta 8.0 \
  --radius 1.5
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import argparse
import time
from anisotropic_kernel import (
    estimate_principal_directions, align_directions,
    compute_anisotropic_weights, compute_anisotropic_weights_keops,
    find_nearest_points_kd
)

# ────────────────── 四元数操作函数 ──────────────────
# 四元数乘法 (x y z w 格式)
def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternions.

    参数:
        q1,q2: (...,4)  按 (x y z w) 排列并已归一化
    返回:
        (...,4) same layout, 归一化结果
    """
    x1,y1,z1,w1 = q1.unbind(-1)
    x2,y2,z2,w2 = q2.unbind(-1)

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    out = torch.stack([x,y,z,w], dim=-1)
    return F.normalize(out, dim=-1)

def quaternion_slerp(q1: torch.Tensor, q2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """四元数球面线性插值 (SLERP)

    参数:
        q1: 起始四元数 (...,4) 格式 (x,y,z,w)，已归一化
        q2: 结束四元数 (...,4) 格式 (x,y,z,w)，已归一化
        t: 插值参数 (...,1) 范围 [0,1]

    返回:
        插值后的四元数 (...,4)，已归一化
    """
    # 确保四元数已归一化
    q1 = F.normalize(q1, dim=-1)
    q2 = F.normalize(q2, dim=-1)

    # 计算点积（余弦角度）
    dot = (q1 * q2).sum(-1, keepdim=True)

    # 如果 dot < 0，翻转 q2，确保走最短路径
    q2 = torch.where(dot < 0, -q2, q2)
    dot = torch.abs(dot)

    # 若角度很小，退化为线性插值避免数值问题
    eps = 1e-6
    linear_mask = dot > (1 - eps)
    omega = torch.acos(torch.clamp(dot, -1 + eps, 1 - eps))  # 角度 θ
    sin_omega = torch.sin(omega)

    # 确保 t 与 dot 形状匹配
    t_expanded = t.expand_as(dot) if t.shape != dot.shape else t

    # SLERP公式: q' = (sin((1-t)θ)/sin(θ)) * q1 + (sin(tθ)/sin(θ)) * q2
    s1 = torch.where(linear_mask, 1.0 - t_expanded, torch.sin((1.0 - t_expanded) * omega) / sin_omega)
    s2 = torch.where(linear_mask, t_expanded, torch.sin(t_expanded * omega) / sin_omega)

    q_out = s1 * q1 + s2 * q2

    # 归一化输出
    return F.normalize(q_out, dim=-1)

# ────────────────── 查找最近点函数 ──────────────────
# 内存优化版本的最近点搜索
def find_nearest_points(filtered_points, original_points, radius=1.5, batch_size=1000, use_kdtree=True):
    """
    为过滤后的每个点找到原始模型中最近的点索引

    参数:
        filtered_points: 过滤后的点坐标 (N,3)
        original_points: 原始模型的点坐标 (M,3)
        radius: 搜索半径
        batch_size: 批处理大小
        use_kdtree: 是否使用KDTree优化

    返回:
        nearest_indices: 每个过滤点对应的最近原始点索引 (N,)
        weights: 每个对应点的权重 (N,)
    """
    # 优先使用KDTree优化版本
    if use_kdtree:
        try:
            return find_nearest_points_kd(filtered_points, original_points, radius, batch_size)
        except ImportError:
            print("未找到scikit-learn，回退到标准方法")

    print(f"使用标准方法查找最近点 (半径: {radius})...")
    N = filtered_points.shape[0]
    M = original_points.shape[0]
    device = filtered_points.device

    # 限制批处理大小以防止内存溢出
    batch_size = min(batch_size, 1000)
    print(f"使用批处理大小: {batch_size}")

    # 初始化结果
    nearest_indices = torch.zeros(N, dtype=torch.long, device=device)
    weights = torch.zeros(N, dtype=torch.float32, device=device)

    start_time = time.time()

    # 分批处理
    for i in range(0, N, batch_size):
        end_idx = min(i + batch_size, N)
        batch_points = filtered_points[i:end_idx]
        batch_size_actual = end_idx - i

        # 初始化该批次的最小距离和索引
        batch_min_dists = torch.full((batch_size_actual,), float('inf'), device='cpu')
        batch_min_indices = torch.zeros((batch_size_actual,), dtype=torch.long, device='cpu')

        # 分段计算与所有原始点的距离以节省内存
        seg_size = 2000  # 每次处理的原始点数
        for s in range(0, M, seg_size):
            s_end = min(s + seg_size, M)
            seg_points = original_points[s:s_end]

            # 使用CPU计算距离以节省GPU内存
            batch_points_cpu = batch_points.cpu()
            seg_points_cpu = seg_points.cpu()
            distances = torch.cdist(batch_points_cpu, seg_points_cpu)

            # 找出每个点在当前段中的最小距离和索引
            seg_min_dists, seg_min_indices = torch.min(distances, dim=1)

            # 调整索引为全局索引
            seg_min_indices += s

            # 更新全局最小距离和索引
            update_mask = seg_min_dists < batch_min_dists
            batch_min_dists[update_mask] = seg_min_dists[update_mask]
            batch_min_indices[update_mask] = seg_min_indices[update_mask]

            # 清理内存
            del distances, seg_min_dists, seg_min_indices

        # 计算权重（基于距离的衰减）并移动到设备
        batch_min_dists = batch_min_dists.to(device)
        batch_min_indices = batch_min_indices.to(device)
        batch_weights = torch.exp(-batch_min_dists / radius)

        # 保存结果
        nearest_indices[i:end_idx] = batch_min_indices
        weights[i:end_idx] = batch_weights

        # 显示进度
        if (i + batch_size) % (5 * batch_size) == 0 or end_idx == N:
            elapsed = time.time() - start_time
            progress = end_idx / N
            eta = elapsed / progress - elapsed if progress > 0 else 0
            valid_count = torch.sum(batch_min_dists < radius).item()
            print(f"处理点: {end_idx}/{N} ({100.0*progress:.1f}%), 有效点数: {valid_count}/{batch_size_actual}, "
                  f"已用时: {elapsed:.1f}秒, 估计剩余: {eta:.1f}秒")

        # 清理内存
        del batch_min_dists, batch_min_indices, batch_weights
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 统计有效映射点（在半径内）
    valid_count = torch.sum(weights > 0.1).item()
    print(f"总有效映射点数: {valid_count}/{N} ({100.0*valid_count/N:.1f}%)")

    return nearest_indices, weights

def compute_anisotropic_interpolation(filtered_points, original_points, source_dirs, displacements,
                                      rotation_values, scale_factors, radius=1.5,
                                      sigma_ratio=10.0, beta=5.0, batch_size=10000):
    """使用各向异性核和方向一致性权重做多源加权传播。

    这里真正使用 sigma_ratio 与 beta：
    - sigma_ratio 控制轴向/横向核宽度比例；
    - beta 控制源方向与目标参考方向不一致时的惩罚强度。
    """
    N = filtered_points.shape[0]
    device = filtered_points.device
    sigma_perp = max(float(radius) / 2.0, 1e-12)
    sigma_para = max(sigma_perp * float(sigma_ratio), 1e-12)

    out_disp = torch.zeros((N, 3), dtype=torch.float32, device=device)
    out_rot = torch.zeros((N, 4), dtype=torch.float32, device=device)
    out_scale = torch.ones((N, 3), dtype=torch.float32, device=device)
    out_dirs = torch.zeros((N, 3), dtype=torch.float32, device=device)
    identity_q = torch.zeros((N, 4), dtype=torch.float32, device=device)
    identity_q[:, 3] = 1.0
    log_scale = torch.log(torch.clamp(scale_factors, min=1e-6))

    source_dirs = F.normalize(source_dirs, dim=1)
    rotation_values = F.normalize(rotation_values, dim=1)

    start_time = time.time()
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        target = filtered_points[start:end]

        # 目标点到所有源点的相对向量；旧链路源点规模很小，分批全量计算更稳。
        diff = target[:, None, :] - original_points[None, :, :]
        dist2 = torch.sum(diff * diff, dim=2)
        dist = torch.sqrt(torch.clamp(dist2, min=1e-12))
        nearest_dist, nearest_indices = torch.min(dist, dim=1)

        src_dirs = source_dirs[None, :, :]
        d_para = torch.sum(diff * src_dirs, dim=2)
        d_perp2 = torch.clamp(dist2 - d_para * d_para, min=0.0)
        within_radius = dist <= radius

        # 用最近源点方向作为目标参考方向，避免为大规模 3DGS 点再做昂贵 PCA。
        ref_dirs = source_dirs[nearest_indices][:, None, :]
        dir_cos = torch.abs(torch.sum(ref_dirs * src_dirs, dim=2)).clamp(0.0, 1.0)
        w_spatial = torch.exp(-0.5 * (d_para * d_para / (sigma_para * sigma_para)
                                      + d_perp2 / (sigma_perp * sigma_perp)))
        w_direction = torch.exp(-float(beta) * (1.0 - dir_cos))
        raw_weights = w_spatial * w_direction * within_radius.float()

        # 如果半径内没有源点，退化为旧链路的最近点距离衰减，避免全零变形。
        row_sum = torch.sum(raw_weights, dim=1)
        no_support = row_sum <= 1e-12
        if torch.any(no_support):
            fallback = torch.exp(-nearest_dist[no_support] / max(float(radius), 1e-12))
            raw_weights[no_support] = 0.0
            raw_weights[no_support, nearest_indices[no_support]] = fallback
            row_sum = torch.sum(raw_weights, dim=1)

        influence = torch.clamp(row_sum, 0.0, 1.0)
        weights = raw_weights / (row_sum[:, None] + 1e-12)

        out_disp[start:end] = (weights @ displacements) * influence[:, None]

        # 多源四元数加权：先做符号一致化，再归一化，并按 influence 从单位四元数插值。
        q_src = rotation_values[None, :, :]
        q_ref = rotation_values[nearest_indices][:, None, :]
        q_sign = torch.where(torch.sum(q_src * q_ref, dim=2, keepdim=True) < 0.0, -1.0, 1.0)
        q_weighted = torch.sum(weights[:, :, None] * q_src * q_sign, dim=1)
        q_weighted = F.normalize(q_weighted, dim=1)
        out_rot[start:end] = quaternion_slerp(identity_q[start:end], q_weighted, influence[:, None])

        log_interp = weights @ log_scale
        out_scale[start:end] = torch.exp(log_interp * influence[:, None])
        out_dirs[start:end] = F.normalize(weights @ source_dirs, dim=1)

        if end == N or (end // batch_size) % 5 == 0:
            elapsed = time.time() - start_time
            progress = end / N
            eta = elapsed / progress - elapsed if progress > 0 else 0
            valid_count = int(torch.sum(row_sum > 1e-12).item())
            print(f"各向异性传播: {end}/{N} ({100.0 * progress:.1f}%), "
                  f"当前批有效点 {valid_count}/{end-start}, 已用时 {elapsed:.1f}s, 剩余 {eta:.1f}s")

    return out_disp, out_rot, out_scale, out_dirs

# ────────────────── 应用变换到过滤模型 ──────────────────
def apply_transform_with_anisotropic_kernel(filtered_model_path, original_model_path, transform_path,
                                          output_model_path, output_transform_path, radius=1.5,
                                          sigma_ratio=10.0, beta=5.0, device=torch.device("cpu"),
                                          use_keops=True, k_neighbors=20, batch_size=10000,
                                          use_kdtree=True):
    """
    将变换关系应用到过滤后的模型，使用各向异性高斯核

    参数:
        filtered_model_path: 过滤后的模型路径
        original_model_path: 原始模型路径
        transform_path: 变换关系路径
        output_model_path: 输出变形后的过滤模型路径
        output_transform_path: 输出过滤后的变换关系路径
        radius: 变形影响半径
        sigma_ratio: 轴向/径向方差比例
        beta: 方向一致性权重系数
        device: 计算设备
        use_keops: 是否使用PyKeOps优化
        k_neighbors: 估计主轴方向的近邻点数量
        batch_size: 批处理大小
        use_kdtree: 是否使用KDTree优化
    """
    total_start_time = time.time()
    print(f"开始应用各向异性高斯核变换...")

    # 加载过滤后的模型
    print(f"加载过滤模型: {filtered_model_path}")
    filtered_model = torch.load(filtered_model_path, map_location=device)
    filtered_splats = filtered_model["splats"]

    # 加载原始模型
    print(f"加载原始模型: {original_model_path}")
    original_model = torch.load(original_model_path, map_location=device)
    original_splats = original_model["splats"]

    # 加载变换关系
    print(f"加载变换关系: {transform_path}")
    transform_data = np.load(transform_path)
    displacements = torch.tensor(transform_data["displacement_vectors"], dtype=torch.float32, device=device)

    # 获取原始点云和过滤点云坐标
    original_points = original_splats["means"].to(device)
    filtered_points = filtered_splats["means"].to(device)

    # 显示基本信息
    print(f"原始点云大小: {original_points.shape[0]}")
    print(f"过滤点云大小: {filtered_points.shape[0]}")

    # 1. 估计原始点云的主轴方向
    print(f"估计点云主轴方向 (k邻近点数: {k_neighbors}, 批次大小: {batch_size})...")
    principal_dirs = estimate_principal_directions(original_points, k_neighbors, batch_size)

    # 2. 确保方向一致性
    print("对齐主轴方向...")
    aligned_dirs = align_directions(principal_dirs)

    # 根据变换关系模式选择不同的处理逻辑
    deform_mode = transform_data.get("deformation_mode", "tangent_only")
    print(f"变换模式: {deform_mode}")

    # 复制过滤模型数据
    transformed_splats = {k: v.clone() for k, v in filtered_splats.items()}

    if deform_mode == "delta":
        # 使用四元数差值模式（新版）
        print("应用四元数差值模式变换...")
        q_delta = torch.tensor(transform_data["rotation_delta"], dtype=torch.float32, device=device)
        q_delta = q_delta[:, [1,2,3,0]]  # wxyz → xyzw，修正四元数顺序
        scale_factors = torch.tensor(transform_data["scale_factors"], dtype=torch.float32, device=device)

        print("建立各向异性多源映射并应用变换...")
        filtered_displacements, interp_q_delta, interp_scale_factors, interp_dirs = compute_anisotropic_interpolation(
            filtered_points, original_points, aligned_dirs, displacements,
            q_delta, scale_factors, radius, sigma_ratio, beta, batch_size
        )

        # 应用位移
        transformed_splats["means"] = filtered_splats["means"].to(device) + filtered_displacements
        transformed_splats["means"] = transformed_splats["means"].cpu()  # 移回CPU

        # 转换四元数顺序并应用差值
        q_orig = filtered_splats["quats"].clone()
        q_orig_xyzw = q_orig[:, [1, 2, 3, 0]].to(device)  # wxyz -> xyzw

        # 应用四元数差值
        q_new_xyzw = quaternion_multiply(interp_q_delta, q_orig_xyzw)

        # 转回wxyz顺序并移至CPU
        q_new_wxyz = q_new_xyzw[:, [3, 0, 1, 2]].cpu()  # xyzw -> wxyz

        # 保证四元数符号一致性 (w >= 0)
        transformed_splats["quats"] = torch.where(q_new_wxyz[:,0:1] < 0, -q_new_wxyz, q_new_wxyz)

        # 应用缩放
        scales_log = torch.log(torch.exp(filtered_splats["scales"].to(device)) * interp_scale_factors)
        transformed_splats["scales"] = scales_log.cpu()

        # 保存过滤后的变换关系 - 确保所有数据都在CPU上
        interp_q_delta_wxyz = interp_q_delta[:, [3, 0, 1, 2]].cpu()
        np.savez(output_transform_path,
                displacement_vectors=filtered_displacements.cpu().numpy(),
                rotation_delta=interp_q_delta_wxyz.numpy(),
                scale_factors=interp_scale_factors.cpu().numpy(),
                principal_directions=interp_dirs.cpu().numpy(),
                deformation_mode="delta")
    else:
        # 兼容旧版切线旋转模式
        print("应用切线旋转模式变换...")
        q_tangent = torch.tensor(transform_data["rotation_tangent"], dtype=torch.float32, device=device)
        q_tangent = q_tangent[:, [1,2,3,0]]  # wxyz → xyzw，修正四元数顺序
        scale_factors = torch.tensor(transform_data["scale_factors"], dtype=torch.float32, device=device)

        print("建立各向异性多源映射并应用变换...")
        filtered_displacements, interp_q_tangent, interp_scale_factors, interp_dirs = compute_anisotropic_interpolation(
            filtered_points, original_points, aligned_dirs, displacements,
            q_tangent, scale_factors, radius, sigma_ratio, beta, batch_size
        )

        # 应用位移
        transformed_splats["means"] = filtered_splats["means"].to(device) + filtered_displacements
        transformed_splats["means"] = transformed_splats["means"].cpu()  # 移回CPU

        # 转换四元数顺序
        q_orig = filtered_splats["quats"].clone()
        q_orig_xyzw = q_orig[:, [1, 2, 3, 0]].to(device)  # wxyz -> xyzw

        # 应用切线旋转
        q_new_xyzw = quaternion_multiply(interp_q_tangent, q_orig_xyzw)

        # 转回wxyz顺序并移至CPU
        q_new_wxyz = q_new_xyzw[:, [3, 0, 1, 2]].cpu()  # xyzw -> wxyz

        # 保证四元数符号一致性 (w >= 0)
        transformed_splats["quats"] = torch.where(q_new_wxyz[:,0:1] < 0, -q_new_wxyz, q_new_wxyz)

        # 应用缩放
        scales_log = torch.log(torch.exp(filtered_splats["scales"].to(device)) * interp_scale_factors)
        transformed_splats["scales"] = scales_log.cpu()

        # 保存过滤后的变换关系 - 确保所有数据都在CPU上
        interp_q_tangent_wxyz = interp_q_tangent[:, [3, 0, 1, 2]].cpu()
        np.savez(output_transform_path,
                displacement_vectors=filtered_displacements.cpu().numpy(),
                rotation_tangent=interp_q_tangent_wxyz.numpy(),
                scale_factors=interp_scale_factors.cpu().numpy(),
                principal_directions=interp_dirs.cpu().numpy(),
                deformation_mode="tangent_only")

    # 保存变形后的过滤模型
    torch.save({"splats": transformed_splats}, output_model_path)

    total_time = time.time() - total_start_time
    print(f"变形后的过滤模型已保存至 {output_model_path}")
    print(f"过滤后的变换关系已保存至 {output_transform_path}")
    print(f"总点数: {transformed_splats['means'].shape[0]}")
    print(f"处理耗时: {total_time:.2f} 秒")

# ────────────────── 主函数 ──────────────────
def main():
    parser = argparse.ArgumentParser(
        description="将变换关系应用到过滤后的高斯点云模型，使用各向异性高斯核"
    )
    parser.add_argument("--filtered-pt-file", type=str, required=True, help="过滤后的模型文件路径")
    parser.add_argument("--original-pt-file", type=str, required=True, help="原始模型文件路径")
    parser.add_argument("--transform-file", type=str, required=True, help="变换关系文件路径")
    parser.add_argument("--output-pt-file", type=str, required=True, help="输出变形后的过滤模型路径")
    parser.add_argument("--output-npz-file", type=str, required=True, help="输出过滤后的变换关系路径")
    parser.add_argument("--radius", type=float, default=1.5, help="变形影响半径")
    parser.add_argument("--sigma-ratio", type=float, default=10.0, help="轴向/径向方差比例（越大杆件方向性越强）")
    parser.add_argument("--beta", type=float, default=5.0, help="方向一致性权重系数（越大方向约束越强）")
    parser.add_argument(
        "--device", type=str, choices=["cuda", "cpu", "auto"], default="auto",
        help="计算设备, auto 表示优先使用 CUDA, 若不可用则退回 CPU")
    parser.add_argument("--no-keops", action="store_true", help="禁用PyKeOps优化（如遇内存问题）")
    parser.add_argument("--k-neighbors", type=int, default=20, help="计算主轴方向时的近邻点数量")
    parser.add_argument("--batch-size", type=int, default=10000, help="批处理大小，用于内存优化")
    parser.add_argument("--use-cpu", action="store_true", help="强制使用CPU进行计算（可能较慢）")
    parser.add_argument("--no-kdtree", action="store_true", help="禁用KDTree优化")
    args = parser.parse_args()

    if args.use_cpu:
        device = torch.device("cpu")
        print("强制使用CPU")
    elif args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"使用设备: {device}")
    torch.set_grad_enabled(False)

    # 确保输出目录存在
    Path(args.output_pt_file).parent.mkdir(exist_ok=True, parents=True)
    Path(args.output_npz_file).parent.mkdir(exist_ok=True, parents=True)

    # 打印参数信息
    print(f"半径参数: {args.radius}")
    print(f"轴向/径向方差比例: {args.sigma_ratio}")
    print(f"方向一致性系数: {args.beta}")
    print(f"使用KDTree优化: {not args.no_kdtree}")

    # 应用变换
    apply_transform_with_anisotropic_kernel(
        args.filtered_pt_file,
        args.original_pt_file,
        args.transform_file,
        args.output_pt_file,
        args.output_npz_file,
        args.radius,
        args.sigma_ratio,
        args.beta,
        device=device,
        use_keops=not args.no_keops,
        k_neighbors=args.k_neighbors,
        batch_size=args.batch_size,
        use_kdtree=not args.no_kdtree
    )

    print("变换应用完成!")

if __name__ == "__main__":
    main()
