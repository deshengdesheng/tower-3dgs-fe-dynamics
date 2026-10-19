"""anisotropic_kernel.py

各向异性高斯核与方向一致性权重计算模块
用于抑制相互平行或交叉杆件间的耦合

主要功能:
1. 估计点云中每个点的主轴方向
2. 构造各向异性高斯核
3. 计算方向一致性权重
4. 基于双权重计算归一化的最终权重
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import KDTree
import time

def estimate_principal_directions_kd(points, k_neighbors=20, batch_size=10000):
    """估计点云中每个点的主轴方向 - 使用KDTree加速版本

    参数:
        points: 点云坐标 (N,3)
        k_neighbors: k-近邻数量
        batch_size: 批处理大小，避免内存溢出

    返回:
        principal_dirs: 每个点的主轴方向 (N,3)，已归一化
    """
    print("使用KDTree优化版本计算主轴方向")
    start_time = time.time()

    # 将点云移到CPU
    points_cpu = points.cpu().numpy()
    N = points.shape[0]
    device = points.device

    # 构建KD树
    print("构建KDTree...")
    tree_build_start = time.time()
    tree = KDTree(points_cpu)
    print(f"KDTree构建完成，耗时 {time.time() - tree_build_start:.2f} 秒")

    # 初始化主方向
    principal_dirs = torch.zeros((N, 3), device='cpu')

    # 分批处理
    print(f"使用批处理大小: {batch_size}")
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        batch_points = points_cpu[i:end_i]
        batch_size_actual = end_i - i

        # 查询k+1个最近邻（包括点自身）
        dists, indices = tree.query(batch_points, k=k_neighbors+1)

        # 去掉第一个邻居（点自身）
        indices = indices[:, 1:]
        dists = dists[:, 1:]

        # 为每个点计算主轴方向
        for b_idx in range(batch_size_actual):
            # 获取邻居点
            neighbor_indices = indices[b_idx]
            neighbors = points_cpu[neighbor_indices]

            # 计算局部中心点
            local_mean = np.mean(neighbors, axis=0)

            # 计算去中心化的坐标
            centered_coords = neighbors - local_mean

            # 计算协方差矩阵 C = X^T * X / n
            cov = np.matmul(centered_coords.T, centered_coords) / neighbors.shape[0]

            # 使用SVD找到最大特征值对应的特征向量
            try:
                U, S, Vh = np.linalg.svd(cov)
                principal_dir = U[:, 0]  # 最大特征值对应的特征向量
                principal_dirs[i + b_idx] = torch.tensor(principal_dir, dtype=torch.float32)
            except:
                # 如果SVD失败，使用随机方向
                principal_dirs[i + b_idx] = F.normalize(torch.randn(3), dim=0)

        # 显示进度
        if (i + batch_size) % (5 * batch_size) == 0 or end_i == N:
            elapsed = time.time() - start_time
            progress = end_i / N
            eta = elapsed / progress - elapsed if progress > 0 else 0
            print(f"主轴方向计算进度: {end_i}/{N} ({100.0*progress:.1f}%), "
                  f"已用时: {elapsed:.1f}秒, 估计剩余: {eta:.1f}秒")

    # 确保所有方向向量都已归一化并移回原设备
    principal_dirs = F.normalize(principal_dirs, dim=1).to(device)

    total_time = time.time() - start_time
    print(f"主轴方向计算完成，总耗时: {total_time:.2f} 秒")
    return principal_dirs

def estimate_principal_directions(points, k_neighbors=20, batch_size=10000):
    """估计点云中每个点的主轴方向 - 内存优化版本

    参数:
        points: 点云坐标 (N,3)
        k_neighbors: k-近邻数量
        batch_size: 批处理大小，避免内存溢出

    返回:
        principal_dirs: 每个点的主轴方向 (N,3)，已归一化
    """
    # 优先使用KDTree优化版本
    try:
        return estimate_principal_directions_kd(points, k_neighbors, batch_size)
    except ImportError:
        print("未找到scikit-learn，回退到标准方法")

    N = points.shape[0]
    device = points.device

    # 初始化主方向
    principal_dirs = torch.zeros((N, 3), device=device)

    # 使用更小的批处理大小以防止内存溢出
    batch_size = min(batch_size, 1000)  # 限制最大批处理大小
    sub_batch_size = min(100, batch_size)  # 距离计算的子批次大小
    print(f"使用批处理大小: {batch_size}, 子批处理大小: {sub_batch_size}")

    # 分批处理
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        batch_points = points[i:end_i]
        batch_size_actual = end_i - i

        # 使用更低效但更省内存的方式计算k近邻
        indices = []
        min_distances = []

        for j in range(0, batch_size_actual, sub_batch_size):
            j_end = min(j + sub_batch_size, batch_size_actual)
            sub_batch_points = batch_points[j:j_end]

            # 初始化该子批次的最小距离和索引
            sub_batch_min_dists = torch.full((j_end - j, k_neighbors+1), float('inf'), device=device)
            sub_batch_min_indices = torch.zeros((j_end - j, k_neighbors+1), dtype=torch.long, device=device)

            # 分段计算与所有点的距离
            seg_size = 1000  # 每次处理的原始点数
            for s in range(0, N, seg_size):
                s_end = min(s + seg_size, N)
                seg_points = points[s:s_end]

                # 计算当前子批次与当前段的距离
                # 使用CPU计算以节省GPU内存
                sub_batch_points_cpu = sub_batch_points.cpu()
                seg_points_cpu = seg_points.cpu()
                distances = torch.cdist(sub_batch_points_cpu, seg_points_cpu)

                # 找出这段中的最小k+1个距离
                if k_neighbors + 1 <= s_end - s:
                    # 如果这个段有足够的点，找出局部top-k
                    local_dists, local_indices = torch.topk(distances, k=min(k_neighbors+1, s_end-s),
                                                          dim=1, largest=False)
                    # 调整局部索引为全局索引
                    local_indices += s
                else:
                    # 如果这个段点数不足k+1，使用所有点
                    local_dists = distances
                    local_indices = torch.arange(s, s_end, device='cpu').expand(j_end - j, -1)

                # 合并当前结果与之前的结果
                combined_dists = torch.cat([sub_batch_min_dists.cpu(), local_dists], dim=1)
                combined_indices = torch.cat([sub_batch_min_indices.cpu(), local_indices], dim=1)

                # 重新选择最小的k+1个
                sub_batch_min_dists_cpu, indices_of_indices = torch.topk(combined_dists,
                                                                      k=min(k_neighbors+1, combined_dists.shape[1]),
                                                                      dim=1, largest=False)

                # 根据indices_of_indices选择对应的索引
                sub_batch_min_indices_cpu = torch.gather(combined_indices, 1, indices_of_indices)

                # 更新到GPU
                sub_batch_min_dists = sub_batch_min_dists_cpu.to(device)
                sub_batch_min_indices = sub_batch_min_indices_cpu.to(device)

                # 清理内存
                del distances, local_dists, local_indices, combined_dists, combined_indices
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # 添加到批次结果
            indices.append(sub_batch_min_indices)
            min_distances.append(sub_batch_min_dists)

        # 合并子批次结果
        batch_indices = torch.cat(indices, dim=0)
        batch_min_dists = torch.cat(min_distances, dim=0)

        # 处理可能自己与自己比较的情况
        for b_idx in range(batch_size_actual):
            for k_idx in range(min(k_neighbors+1, batch_indices.shape[1])):
                if batch_indices[b_idx, k_idx] == i + b_idx:
                    # 如果找到自己，将该索引置为-1
                    batch_indices[b_idx, k_idx] = -1

        # 为每个点计算主轴方向
        for b_idx in range(batch_size_actual):
            # 获取有效邻居点索引（排除自身）
            valid_indices = batch_indices[b_idx]
            valid_indices = valid_indices[valid_indices != -1]

            if valid_indices.shape[0] < 3:  # 确保有足够的邻居
                principal_dirs[i + b_idx] = torch.tensor([1.0, 0.0, 0.0], device=device)
                continue

            # 获取邻居点
            neighbors = points[valid_indices]

            # 计算局部中心点
            local_mean = torch.mean(neighbors, dim=0)

            # 计算去中心化的坐标
            centered_coords = neighbors - local_mean

            # 计算协方差矩阵 C = X^T * X / n
            cov = torch.matmul(centered_coords.T, centered_coords) / neighbors.shape[0]

            # 使用SVD找到最大特征值对应的特征向量
            try:
                U, S, V = torch.svd(cov)
                principal_dir = U[:, 0]  # 最大特征值对应的特征向量
                principal_dirs[i + b_idx] = principal_dir
            except:
                # 如果SVD失败，使用随机方向
                principal_dirs[i + b_idx] = F.normalize(torch.randn(3, device=device), dim=0)

        # 清理内存
        del batch_indices, batch_min_dists, batch_points
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # 显示进度
        if (i + batch_size) % (10 * batch_size) == 0 or end_i == N:
            print(f"主轴方向计算进度: {end_i}/{N} ({100.0*end_i/N:.1f}%)")

    # 确保所有方向向量都已归一化
    principal_dirs = F.normalize(principal_dirs, dim=1)

    return principal_dirs

def find_nearest_points_kd(filtered_points, original_points, radius=1.5, batch_size=1000):
    """为过滤后的每个点找到原始模型中最近的点索引 - KDTree加速版本

    参数:
        filtered_points: 过滤后的点坐标 (N,3)
        original_points: 原始模型的点坐标 (M,3)
        radius: 搜索半径
        batch_size: 批处理大小

    返回:
        nearest_indices: 每个过滤点对应的最近原始点索引 (N,)
        weights: 每个对应点的权重 (N,)
    """
    print(f"使用KDTree搜索最近点 (半径: {radius})...")
    start_time = time.time()

    # 将点云移到CPU用于KDTree构建
    filtered_cpu = filtered_points.cpu().numpy()
    original_cpu = original_points.cpu().numpy()

    N = filtered_points.shape[0]
    device = filtered_points.device

    # 构建KDTree
    print("构建KDTree...")
    tree_build_start = time.time()
    tree = KDTree(original_cpu)
    print(f"KDTree构建完成，耗时 {time.time() - tree_build_start:.2f} 秒")

    # 初始化结果
    nearest_indices = np.zeros(N, dtype=np.int64)
    distances = np.ones(N, dtype=np.float32) * np.inf

    # 分批处理
    print(f"使用批处理大小: {batch_size}")
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        batch_points = filtered_cpu[i:end_i]

        # 查找每个点的最近邻
        batch_dists, batch_indices = tree.query(batch_points, k=1, return_distance=True)

        # 保存结果
        nearest_indices[i:end_i] = batch_indices.flatten()
        distances[i:end_i] = batch_dists.flatten()

        # 显示进度
        if (i + batch_size) % (5 * batch_size) == 0 or end_i == N:
            elapsed = time.time() - start_time
            progress = end_i / N
            eta = elapsed / progress - elapsed if progress > 0 else 0
            valid_count = np.sum(distances[i:end_i] < radius)
            batch_size_actual = end_i - i
            print(f"处理点: {end_i}/{N} ({100.0*progress:.1f}%), 有效点数: {valid_count}/{batch_size_actual}, "
                  f"已用时: {elapsed:.1f}秒, 估计剩余: {eta:.1f}秒")

    # 计算权重（基于距离的衰减）
    weights = np.exp(-distances / radius)

    # 转换为torch张量并移到原始设备上
    nearest_indices_torch = torch.tensor(nearest_indices, dtype=torch.long, device=device)
    weights_torch = torch.tensor(weights, dtype=torch.float32, device=device)

    # 统计有效映射点（在半径内）
    valid_count = np.sum(distances < radius)
    print(f"总有效映射点数: {valid_count}/{N} ({100.0*valid_count/N:.1f}%)")

    total_time = time.time() - start_time
    print(f"最近点搜索完成，总耗时: {total_time:.2f} 秒")

    return nearest_indices_torch, weights_torch

def align_directions(principal_dirs, groups=None):
    """确保方向一致性（避免方向翻转导致平均抵消）

    参数:
        principal_dirs: 每个点的主轴方向 (N,3)
        groups: 点分组信息，可选

    返回:
        aligned_dirs: 调整后的方向 (N,3)
    """
    if groups is None:
        # 如果没有分组信息，使用第一个点作为参考
        ref_dir = principal_dirs[0]
        # 与参考方向点乘，判断方向是否一致
        dots = torch.sum(principal_dirs * ref_dir, dim=1)
        # 对点乘为负的方向取反
        aligned_dirs = torch.where(dots.unsqueeze(1) < 0,
                                  -principal_dirs,
                                   principal_dirs)
    else:
        # 按组处理
        aligned_dirs = principal_dirs.clone()
        unique_groups = torch.unique(groups)

        for g in unique_groups:
            mask = (groups == g)
            if not torch.any(mask):
                continue

            # 使用组内第一个点作为参考
            group_indices = torch.where(mask)[0]
            ref_idx = group_indices[0]
            ref_dir = principal_dirs[ref_idx]

            # 组内方向一致化
            dots = torch.sum(principal_dirs[mask] * ref_dir, dim=1)
            aligned_dirs[mask] = torch.where(dots.unsqueeze(1) < 0,
                                           -principal_dirs[mask],
                                            principal_dirs[mask])

    return aligned_dirs

def compute_rotation_matrices(principal_dirs):
    """计算将主轴方向旋转到z轴的旋转矩阵

    参数:
        principal_dirs: 每个点的主轴方向 (N,3)

    返回:
        rotation_matrices: 旋转矩阵 (N,3,3)
    """
    N = principal_dirs.shape[0]
    device = principal_dirs.device

    # 确保方向向量已归一化
    a_i = F.normalize(principal_dirs, dim=1)

    # 初始化旋转矩阵
    R = torch.zeros((N, 3, 3), device=device)

    # 生成正交基
    # 寻找与主方向不平行的向量作为辅助向量
    aux = torch.ones((N, 3), device=device)
    # 处理与(1,1,1)平行的特殊情况
    parallel_mask = torch.abs(torch.sum(a_i * aux, dim=1)) > 0.99 * torch.sqrt(torch.sum(aux**2, dim=1))
    aux[parallel_mask, 0] = 0  # 对平行的情况使用(0,1,1)

    # 计算第一个正交向量b_i（垂直于a_i）
    b_i = torch.cross(a_i, aux)
    b_i = F.normalize(b_i, dim=1)

    # 计算第二个正交向量c_i（垂直于a_i和b_i）
    c_i = torch.cross(a_i, b_i)
    c_i = F.normalize(c_i, dim=1)

    # 构建旋转矩阵R_i = [b_i, c_i, a_i]
    R[:, :, 0] = b_i
    R[:, :, 1] = c_i
    R[:, :, 2] = a_i

    return R

def compute_anisotropic_weights(points, principal_dirs, sigma_para=1.0, sigma_perp=0.1, beta=5.0, radius=None):
    """计算各向异性高斯核权重与方向一致性权重 - 内存优化版本

    参数:
        points: 点云坐标 (N,3)
        principal_dirs: 每个点的主轴方向 (N,3)，已归一化
        sigma_para: 轴向标准差
        sigma_perp: 径向标准差
        beta: 方向一致性权重系数
        radius: 搜索半径，只考虑半径内的点，显著减少内存占用

    返回:
        weights: 归一化权重矩阵 (N,N) 或稀疏矩阵
    """
    N = points.shape[0]
    device = points.device

    # 如果指定了半径，则使用稀疏计算
    if radius is not None:
        print(f"使用局部搜索半径 {radius} 以节省内存...")
        # 分批处理以减少内存占用
        batch_size = 1000
        weights_list = []
        indices_list = []

        for i in range(0, N, batch_size):
            end_i = min(i + batch_size, N)
            print(f"处理点 {i} 到 {end_i} (共 {N} 点)")

            # 提取批次点
            batch_points = points[i:end_i]
            batch_dirs = principal_dirs[i:end_i]

            # 计算距离
            distances = torch.cdist(batch_points, points)

            # 只考虑半径内的点
            radius_mask = distances < radius

            # 创建行索引和列索引
            batch_size_actual = end_i - i
            row_indices = torch.arange(batch_size_actual, device=device).unsqueeze(1).expand(-1, N)
            col_indices = torch.arange(N, device=device).unsqueeze(0).expand(batch_size_actual, -1)

            # 应用半径掩码
            valid_row_indices = row_indices[radius_mask]
            valid_col_indices = col_indices[radius_mask]

            # 计算有效点对的距离向量
            valid_points_i = batch_points[valid_row_indices]
            valid_points_j = points[valid_col_indices]
            diff = valid_points_j - valid_points_i

            # 计算方向一致性权重
            valid_dirs_i = batch_dirs[valid_row_indices]
            valid_dirs_j = principal_dirs[valid_col_indices]
            dir_cos = torch.abs(torch.sum(valid_dirs_i * valid_dirs_j, dim=1))
            w_dir = torch.exp(-beta * (1.0 - dir_cos))

            # 计算各向异性空间权重
            # 轴向分量
            d_para = torch.sum(diff * valid_dirs_i, dim=1)
            # 径向分量
            d_perp = torch.sum(diff**2, dim=1) - d_para**2
            # 各向异性高斯核
            w_spat = torch.exp(-0.5 * (d_para**2 / sigma_para**2 + d_perp / sigma_perp**2))

            # 组合权重
            w_combined = w_spat * w_dir

            # 保存有效权重和对应索引
            weights_list.append(w_combined)
            # 调整行索引以考虑批次偏移
            indices_list.append(torch.stack([valid_row_indices + i, valid_col_indices]))

            # 清理内存
            del distances, radius_mask, diff, w_dir, w_spat, w_combined
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # 合并所有批次的结果
        all_weights = torch.cat(weights_list)
        all_indices = torch.cat(indices_list, dim=1)

        # 构建稀疏矩阵
        sparse_weights = torch.sparse.FloatTensor(
            all_indices, all_weights, torch.Size([N, N])
        )

        # 行归一化
        # 转为密集矩阵计算行和（仅用于归一化）
        row_sums = torch.sparse.sum(sparse_weights, dim=1).to_dense().unsqueeze(1)
        # 将权重除以行和
        normalized_weights = sparse_weights.to_dense() / (row_sums + 1e-12)

        return normalized_weights
    else:
        # 原始完整版本实现（仅适用于中小规模点云）
        print("警告：未指定搜索半径，将计算完整距离矩阵（可能消耗大量内存）")

        # 尝试使用分批处理来减少内存消耗
        batch_size = 1000
        weights = torch.zeros((N, N), device=device)

        for i in range(0, N, batch_size):
            end_i = min(i + batch_size, N)

            for j in range(0, N, batch_size):
                end_j = min(j + batch_size, N)

                # 提取批次数据
                points_i = points[i:end_i].unsqueeze(1)  # (batch_i, 1, 3)
                points_j = points[j:end_j].unsqueeze(0)  # (1, batch_j, 3)
                dirs_i = principal_dirs[i:end_i].unsqueeze(1)  # (batch_i, 1, 3)
                dirs_j = principal_dirs[j:end_j].unsqueeze(0)  # (1, batch_j, 3)

                # 计算空间距离向量
                diff = points_j - points_i  # (batch_i, batch_j, 3)

                # 计算方向一致性权重: exp(-β(1-|a_i·a_j|))
                dir_cos = torch.abs(torch.sum(dirs_i * dirs_j, dim=2))  # (batch_i, batch_j)
                w_dir = torch.exp(-beta * (1.0 - dir_cos))

                # 计算各向异性空间权重
                # 1. 将距离向量投影到主方向上（轴向分量）
                d_para = torch.sum(diff * dirs_i, dim=2)  # (batch_i, batch_j)
                # 2. 计算径向分量（垂直于主方向）
                d_perp = torch.sum(diff**2, dim=2) - d_para**2  # 毕达哥拉斯定理
                # 3. 应用各向异性高斯核
                w_spat = torch.exp(-0.5 * (d_para**2 / sigma_para**2 + d_perp / sigma_perp**2))

                # 计算组合权重
                w_combined = w_spat * w_dir
                weights[i:end_i, j:end_j] = w_combined

        # 行归一化
        row_sums = weights.sum(dim=1, keepdim=True)
        norm_weights = weights / (row_sums + 1e-12)

        return norm_weights

# 支持PyKeOps的优化版本（如果可用）
def compute_anisotropic_weights_keops(points, principal_dirs, sigma_para=1.0, sigma_perp=0.1, beta=5.0, radius=None):
    """使用PyKeOps计算各向异性高斯核权重（内存高效版）

    参数:
        points: 点云坐标 (N,3)
        principal_dirs: 每个点的主轴方向 (N,3)，已归一化
        sigma_para: 轴向标准差
        sigma_perp: 径向标准差
        beta: 方向一致性权重系数
        radius: 搜索半径，只考虑半径内的点

    返回:
        weights: 归一化权重矩阵
    """
    try:
        import pykeops.torch as pk

        # 如果设置了半径，使用KNN搜索
        if radius is not None:
            print(f"使用PyKeOps的带半径搜索功能 (radius={radius})...")

            x_i = pk.LazyTensor(points[:, None, :])  # (N,1,3)
            x_j = pk.LazyTensor(points[None, :, :])  # (1,N,3)
            a_i = pk.LazyTensor(principal_dirs[:, None, :])  # (N,1,3)
            a_j = pk.LazyTensor(principal_dirs[None, :, :])  # (1,N,3)

            # 计算距离
            sqdist = ((x_i - x_j) ** 2).sum(dim=2)  # (N,N)
            # 只考虑半径内的点
            mask = sqdist <= radius**2

            # 计算方向一致性
            cos = (a_i * a_j).sum(dim=2).abs()  # (N,N)
            w_dir = (-beta * (1.0 - cos)).exp()

            # 计算空间距离
            diff = x_j - x_i  # (N,N,3)
            # 轴向分量
            d_para = (diff * a_i).sum(dim=2)  # (N,N)
            # 径向分量
            d_perp = (diff**2).sum(dim=2) - d_para**2
            # 各向异性高斯核
            w_spat = (-0.5 * (d_para**2 / sigma_para**2 + d_perp / sigma_perp**2)).exp()

            # 组合权重并应用掩码
            W = w_dir * w_spat * mask  # (N,N)
            # 行归一化
            return W / (W.sum(dim=1) + 1e-12)

        else:
            print("使用PyKeOps全局计算权重...")

            x_i = pk.LazyTensor(points[:, None, :])  # (N,1,3)
            x_j = pk.LazyTensor(points[None, :, :])  # (1,N,3)
            a_i = pk.LazyTensor(principal_dirs[:, None, :])  # (N,1,3)
            a_j = pk.LazyTensor(principal_dirs[None, :, :])  # (1,N,3)

            # 计算方向一致性
            cos = (a_i * a_j).sum(dim=2).abs()  # (N,N)
            w_dir = (-beta * (1.0 - cos)).exp()

            # 计算空间距离
            diff = x_j - x_i  # (N,N,3)
            # 轴向分量
            d_para = (diff * a_i).sum(dim=2)  # (N,N)
            # 径向分量
            d_perp = (diff**2).sum(dim=2) - d_para**2
            # 各向异性高斯核
            w_spat = (-0.5 * (d_para**2 / sigma_para**2 + d_perp / sigma_perp**2)).exp()

            # 组合权重
            W = w_dir * w_spat  # (N,N)
            # 行归一化
            return W / (W.sum(dim=1) + 1e-12)

    except ImportError:
        print("PyKeOps未安装，使用标准PyTorch实现")
        return compute_anisotropic_weights(points, principal_dirs, sigma_para, sigma_perp, beta, radius)

def apply_anisotropic_weights(model, transform_data, radius=1.0, sigma_ratio=10.0, beta=5.0):
    """应用各向异性权重到模型变换

    参数:
        model: 模型数据，包含点云坐标
        transform_data: 变换数据
        radius: 基础搜索半径
        sigma_ratio: 轴向与径向标准差比例
        beta: 方向一致性权重系数

    返回:
        weighted_transforms: 应用各向异性权重后的变换
    """
    # 提取点云坐标
    points = model["splats"]["means"]

    # 1. 估计主轴方向
    principal_dirs = estimate_principal_directions(points)

    # 2. 确保方向一致性
    aligned_dirs = align_directions(principal_dirs)

    # 3. 计算各向异性高斯核参数
    sigma_perp = radius / 2.0
    sigma_para = sigma_perp * sigma_ratio  # 典型值 10-50

    # 4. 计算权重
    try:
        # 尝试使用PyKeOps（如果可用）
        weights = compute_anisotropic_weights_keops(points, aligned_dirs,
                                                   sigma_para, sigma_perp, beta, radius)
    except Exception:
        # 回退到普通PyTorch实现
        weights = compute_anisotropic_weights(points, aligned_dirs,
                                             sigma_para, sigma_perp, beta, radius)

    # 返回权重与主轴方向
    return {
        "weights": weights,
        "principal_directions": aligned_dirs,
        "sigma_para": sigma_para,
        "sigma_perp": sigma_perp
    }