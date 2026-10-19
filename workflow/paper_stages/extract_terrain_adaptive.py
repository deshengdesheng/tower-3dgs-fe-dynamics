#!/usr/bin/env python
"""
3D高斯点云自适应地形提取工具

针对复杂起伏地形的自适应提取算法
使用局部高度分析、坡度特征和区域连续性，适合处理变化起伏的地形
支持Z坐标颠倒情况（地面点Z值大，非地面点Z值小）
"""

import torch
import numpy as np
# Plotly is only needed for the optional interactive HTML diagnostic.  Keep
# the numerical extraction runnable in the lean gsplat environment.
try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - optional visualization dependency
    go = None
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
import argparse
import json
import os
from pathlib import Path
import sys
import time
from tqdm import tqdm

def load_gaussian_model(model_path):
    """加载高斯模型并返回点云坐标"""
    print(f"加载模型: {model_path}")
    start_time = time.time()
    ckpt = torch.load(model_path, map_location='cpu')
    splats = ckpt["splats"]
    points = splats["means"].numpy()
    print(f"加载完成，共 {len(points):,} 个点，耗时: {time.time() - start_time:.2f}秒")
    return points, splats

def load_structure_roi_mask(points, cache_path):
    """Use the training strategy's exact predicate without changing GS coordinates."""
    repository_root = str(Path(__file__).resolve().parents[2])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    from gsplat.strategy.irregular_region_strategy import IrregularRegionFilteredStrategy

    strategy = IrregularRegionFilteredStrategy.from_cache(cache_path)
    if strategy.roi_mode != "exact-capsule":
        raise ValueError("--structure-roi-cache requires an exact-capsule FE ROI cache")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        mask = strategy._get_region_mask(torch.as_tensor(points, device=device))
    return mask.cpu().numpy().astype(bool, copy=False), {
        "cache": str(Path(cache_path).resolve()),
        "cache_mode": strategy.roi_mode,
        "predicate": "IrregularRegionFilteredStrategy._get_region_mask (exact-capsule)",
        "device": str(device),
    }

def analyze_height_distribution(points):
    """分析点云的高度分布"""
    print("分析全局高度分布...")
    z_values = points[:, 2]
    z_min = np.min(z_values)
    z_max = np.max(z_values)
    z_range = z_max - z_min
    z_mean = np.mean(z_values)
    z_median = np.median(z_values)
    z_std = np.std(z_values)
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    z_percentiles = {p: np.percentile(z_values, p) for p in percentiles}

    print(f"高度范围: {z_min:.2f} 到 {z_max:.2f} (范围: {z_range:.2f})")
    print(f"平均高度: {z_mean:.2f}，中位高度: {z_median:.2f}，标准差: {z_std:.2f}")
    print("百分位值:")
    for p, v in z_percentiles.items():
        print(f"  {p}%: {v:.2f}")

    return {
        'min': z_min,
        'max': z_max,
        'range': z_range,
        'mean': z_mean,
        'median': z_median,
        'std': z_std,
        'percentiles': z_percentiles
    }

def downsample_points_grid(points, grid_size=0.2, z_inverted=True):
    """
    使用网格下采样方法减少点数

    Args:
        points: 原始点云 (N, 3)
        grid_size: 网格大小
        z_inverted: Z坐标是否颠倒（True表示地面点Z值大）

    Returns:
        downsampled_points: 下采样后的点云
        original_indices: 下采样点在原始点云中的索引
    """
    print(f"使用网格下采样（网格大小: {grid_size}）...")

    # 计算网格索引
    grid_indices = np.floor(points[:, :3] / grid_size).astype(int)

    # 创建唯一网格标识
    grid_ids = grid_indices[:, 0] + grid_indices[:, 1] * 1000 + grid_indices[:, 2] * 1000000

    # 获取唯一网格及其第一个点的索引
    unique_grid_ids = np.unique(grid_ids)

    # 为每个唯一网格选择一个代表点
    unique_indices = []
    for grid_id in unique_grid_ids:
        # 获取当前网格中的所有点
        grid_point_indices = np.where(grid_ids == grid_id)[0]

        # 根据Z轴方向选择代表点
        if z_inverted:
            # Z轴颠倒：选择Z值最大的点（可能是地面）
            rep_idx = grid_point_indices[np.argmax(points[grid_point_indices, 2])]
        else:
            # 正常Z轴：选择Z值最小的点（可能是地面）
            rep_idx = grid_point_indices[np.argmin(points[grid_point_indices, 2])]

        unique_indices.append(rep_idx)

    # 提取下采样点
    downsampled_points = points[unique_indices]

    print(f"下采样后点数: {len(downsampled_points):,} ({len(downsampled_points)/len(points)*100:.1f}%)")

    return downsampled_points, np.array(unique_indices)

def calculate_local_features(points, z_inverted=True, k=30):
    """
    计算点云的局部特征，包括局部高度差、局部平面拟合、坡度等

    Args:
        points: 点云坐标 (N, 3)
        z_inverted: Z坐标是否颠倒
        k: 邻域点数量

    Returns:
        features: 包含各种特征的字典
    """
    print(f"计算局部特征 (使用 {k} 个邻域点)...")

    # 构建KD树用于快速邻域搜索
    tree = KDTree(points)

    # 初始化特征数组
    n_points = len(points)
    local_height_ranges = np.zeros(n_points)
    local_height_stds = np.zeros(n_points)
    slope_values = np.zeros(n_points)
    plane_residuals = np.zeros(n_points)
    normal_verticality = np.zeros(n_points)

    # 对每个点计算特征
    for i in tqdm(range(n_points), desc="计算局部特征"):
        # 查找邻域点
        indices = tree.query(points[i:i+1], k=k, return_distance=False)[0]
        neighbors = points[indices]

        # 计算局部高度特征
        z_values = neighbors[:, 2]
        local_height_ranges[i] = np.max(z_values) - np.min(z_values)
        local_height_stds[i] = np.std(z_values)

        # 计算局部平面拟合和法向量
        try:
            # 将邻域点转换为相对坐标
            centroid = np.mean(neighbors, axis=0)
            centered = neighbors - centroid

            # 使用SVD计算主方向
            u, s, vh = np.linalg.svd(centered, full_matrices=False)
            normal = vh[2]  # 最小特征值对应的特征向量

            # 确保法向量方向一致
            if (z_inverted and normal[2] < 0) or (not z_inverted and normal[2] > 0):
                normal = -normal

            # 计算点到平面的距离
            distances = np.abs(np.dot(centered, normal))
            plane_residuals[i] = np.mean(distances)

            # 计算法向量的垂直度 (0表示垂直于地面，1表示平行于地面)
            normal_verticality[i] = np.abs(normal[2])

            # 计算坡度（法向量与垂直轴的夹角）
            z_axis = np.array([0, 0, 1])
            slope_values[i] = np.arccos(np.abs(np.dot(normal, z_axis))) * 180 / np.pi
        except:
            # 如果SVD失败，使用默认值
            plane_residuals[i] = 1.0
            normal_verticality[i] = 0.0
            slope_values[i] = 90.0

    print("局部特征计算完成")

    # 返回所有特征
    return {
        'local_height_ranges': local_height_ranges,
        'local_height_stds': local_height_stds,
        'slope_values': slope_values,
        'plane_residuals': plane_residuals,
        'normal_verticality': normal_verticality
    }

def analyze_local_regions(points, grid_size=2.0):
    """
    分析点云中的局部区域，为每个区域建立统计模型

    Args:
        points: 点云坐标
        grid_size: 网格大小

    Returns:
        regions: 网格区域统计信息
    """
    print(f"分析局部区域 (网格大小: {grid_size})...")

    # 计算点云边界
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

    # 计算网格数量
    x_grids = max(1, int((x_max - x_min) / grid_size))
    y_grids = max(1, int((y_max - y_min) / grid_size))

    print(f"划分为 {x_grids} × {y_grids} = {x_grids * y_grids} 个网格")

    # 初始化区域信息字典
    regions = {}

    # 按网格处理
    for i in range(x_grids):
        for j in range(y_grids):
            # 计算网格边界
            x_start = x_min + i * grid_size
            x_end = x_start + grid_size
            y_start = y_min + j * grid_size
            y_end = y_start + grid_size

            # 获取网格内的点
            grid_mask = (points[:, 0] >= x_start) & (points[:, 0] < x_end) & \
                        (points[:, 1] >= y_start) & (points[:, 1] < y_end)

            grid_indices = np.where(grid_mask)[0]

            # 如果网格有足够的点，则计算统计信息
            if len(grid_indices) > 10:
                grid_z = points[grid_indices, 2]

                region_info = {
                    'indices': grid_indices,
                    'center': np.array([x_start + grid_size/2, y_start + grid_size/2]),
                    'z_min': np.min(grid_z),
                    'z_max': np.max(grid_z),
                    'z_mean': np.mean(grid_z),
                    'z_median': np.median(grid_z),
                    'z_std': np.std(grid_z),
                    'count': len(grid_indices)
                }

                regions[(i, j)] = region_info

    print(f"有效区域网格数量: {len(regions)}")

    return regions

def get_initial_terrain_mask(points, features, regions,
                           slope_threshold=30.0,
                           height_var_threshold=0.5,
                           z_inverted=True):
    """
    使用多特征组合获取初步地形掩码

    Args:
        points: 点云坐标
        features: 局部特征
        regions: 区域统计信息
        slope_threshold: 坡度阈值
        height_var_threshold: 高度变化阈值
        z_inverted: Z轴是否颠倒

    Returns:
        terrain_mask: 初步地形掩码
    """
    print("获取初步地形掩码...")
    start_time = time.time()

    # 初始化地形掩码
    terrain_mask = np.ones(len(points), dtype=bool)

    # 根据坡度过滤（坡度低的可能是地形）
    slope_mask = features['slope_values'] < slope_threshold
    terrain_mask = terrain_mask & slope_mask
    print(f"坡度过滤后点数: {np.sum(terrain_mask):,}")

    # 根据局部高度变化过滤（变化小的可能是地形）
    height_var_mask = features['local_height_ranges'] < height_var_threshold
    terrain_mask = terrain_mask & height_var_mask
    print(f"局部高度变化过滤后点数: {np.sum(terrain_mask):,}")

    # 根据法向量垂直度过滤（垂直于地面的可能是地形）
    vertical_threshold = 0.7  # 接近1表示法向量与Z轴平行
    vertical_mask = features['normal_verticality'] > vertical_threshold
    terrain_mask = terrain_mask & vertical_mask
    print(f"法向量垂直度过滤后点数: {np.sum(terrain_mask):,}")

    print(f"初步地形掩码准备完成，耗时: {time.time() - start_time:.2f}秒")

    return terrain_mask

def refine_terrain_regions(points, terrain_mask, regions, std_factor=1.5, z_inverted=True):
    """
    使用区域统计优化地形掩码

    Args:
        points: 点云坐标
        terrain_mask: 初步地形掩码
        regions: 区域统计信息
        std_factor: 标准差系数
        z_inverted: Z轴是否颠倒

    Returns:
        refined_mask: 优化后的地形掩码
    """
    print(f"使用区域统计优化地形掩码 (标准差系数: {std_factor})...")

    # 复制初步掩码
    refined_mask = terrain_mask.copy()

    # 对每个区域进行统计分析
    for region_idx, region_info in regions.items():
        # 获取区域内的点
        region_indices = region_info['indices']

        # 获取区域内初步标记为地形的点
        region_terrain_indices = region_indices[terrain_mask[region_indices]]

        # 如果区域内有足够的地形点，则进行高度统计分析
        if len(region_terrain_indices) > 5:
            # 计算区域内地形点的高度统计
            region_terrain_z = points[region_terrain_indices, 2]
            mean_z = np.mean(region_terrain_z)
            std_z = np.std(region_terrain_z)

            # 设置高度范围，根据Z轴方向调整
            if z_inverted:
                # Z轴颠倒：地形点Z值大
                lower_bound = mean_z - std_factor * std_z
                upper_bound = float('inf')  # 不限制上界

                # 更新区域内所有点的掩码
                for idx in region_indices:
                    if points[idx, 2] >= lower_bound:
                        refined_mask[idx] = True
                    else:
                        refined_mask[idx] = False
            else:
                # 正常Z轴：地形点Z值小
                lower_bound = float('-inf')  # 不限制下界
                upper_bound = mean_z + std_factor * std_z

                # 更新区域内所有点的掩码
                for idx in region_indices:
                    if points[idx, 2] <= upper_bound:
                        refined_mask[idx] = True
                    else:
                        refined_mask[idx] = False

    # 统计优化效果
    initial_terrain_count = np.sum(terrain_mask)
    refined_terrain_count = np.sum(refined_mask)

    print(f"区域统计优化前地形点数: {initial_terrain_count:,}")
    print(f"区域统计优化后地形点数: {refined_terrain_count:,}")
    print(f"变化: {refined_terrain_count - initial_terrain_count:+,} 点")

    return refined_mask

def clean_terrain_clusters(points, terrain_mask, eps=1.0, min_samples=10):
    """
    使用聚类清理孤立点

    Args:
        points: 点云坐标
        terrain_mask: 地形掩码
        eps: 聚类距离阈值
        min_samples: 最小聚类样本数

    Returns:
        cleaned_mask: 清理后的地形掩码
    """
    print(f"使用聚类清理孤立点 (距离阈值: {eps}, 最小样本数: {min_samples})...")

    # 获取地形点
    terrain_indices = np.where(terrain_mask)[0]

    if len(terrain_indices) < min_samples * 2:
        print("地形点数量太少，跳过聚类清理")
        return terrain_mask

    # 创建清理后的掩码，初始化为全False
    cleaned_mask = np.zeros(len(points), dtype=bool)

    try:
        # 提取地形点坐标
        terrain_points = points[terrain_indices]

        # 使用DBSCAN聚类
        print("执行DBSCAN聚类...")
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(terrain_points)
        labels = clustering.labels_

        # 统计每个聚类的点数
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        print(f"共发现 {n_clusters} 个聚类，噪声点数量: {np.sum(labels == -1)}")

        # 如果有多个聚类，仅保留最大的聚类
        if n_clusters > 0:
            # 找出最大聚类
            valid_labels = unique_labels[unique_labels != -1]
            counts = [np.sum(labels == label) for label in valid_labels]

            if len(counts) > 0:
                largest_cluster = valid_labels[np.argmax(counts)]
                largest_size = np.max(counts)

                print(f"最大聚类包含 {largest_size} 个点")

                # 标记最大聚类的点为地形
                largest_cluster_mask = (labels == largest_cluster)
                largest_cluster_indices = terrain_indices[largest_cluster_mask]

                # 更新掩码
                cleaned_mask[largest_cluster_indices] = True

                # 对于其他较大的聚类，如果大小超过最大聚类的30%，也保留
                for label in valid_labels:
                    if label != largest_cluster:
                        cluster_size = np.sum(labels == label)
                        if cluster_size > largest_size * 0.3:
                            print(f"保留次级聚类，包含 {cluster_size} 个点")
                            cluster_mask = (labels == label)
                            cluster_indices = terrain_indices[cluster_mask]
                            cleaned_mask[cluster_indices] = True
            else:
                # 没有有效聚类，使用原始掩码
                cleaned_mask = terrain_mask
        else:
            # 没有聚类，使用原始掩码
            cleaned_mask = terrain_mask
    except Exception as e:
        print(f"聚类过程中发生错误: {str(e)}")
        # 出现错误时使用原始掩码
        cleaned_mask = terrain_mask

    # 统计清理效果
    initial_count = np.sum(terrain_mask)
    cleaned_count = np.sum(cleaned_mask)
    removed = initial_count - cleaned_count

    print(f"聚类清理移除了 {removed:,} 个孤立点 ({removed/initial_count*100:.1f}%)")

    return cleaned_mask

def save_terrain_model(original_splats, terrain_mask, output_path):
    """
    保存地形模型

    Args:
        original_splats: 原始高斯模型的splats字典
        terrain_mask: 地形掩码
        output_path: 输出路径
    """
    print(f"保存地形模型...")
    start_time = time.time()

    # 获取地形点索引
    terrain_indices = np.where(terrain_mask)[0]

    # 创建新的splats字典
    terrain_splats = {}

    # 复制每个属性
    for key, value in original_splats.items():
        if isinstance(value, torch.Tensor) and value.shape[0] == len(original_splats["means"]):
            terrain_splats[key] = value[terrain_indices]
        else:
            terrain_splats[key] = value

    # 保存模型
    torch.save({"splats": terrain_splats}, output_path)

    print(f"已保存地形模型 ({len(terrain_indices):,} 个点)，耗时: {time.time() - start_time:.2f}秒")

def visualize_terrain(points, terrain_mask, output_path=None, point_size=2, max_points=10000, open_browser=False):
    """
    可视化地形提取结果

    Args:
        points: 点云坐标
        terrain_mask: 地形掩码
        output_path: 输出HTML路径
        point_size: 点大小
        max_points: 最大显示点数
        open_browser: 是否自动打开浏览器
    """
    if go is None:
        raise RuntimeError("interactive terrain visualization requires plotly; rerun without --visualize")
    print(f"生成可视化结果...")

    # 提取地形点和非地形点
    terrain_points = points[terrain_mask]
    non_terrain_points = points[~terrain_mask]

    # 如果点数超过最大限制，进行下采样
    if len(terrain_points) > max_points // 2:
        step = len(terrain_points) // (max_points // 2) + 1
        terrain_points = terrain_points[::step]
        print(f"地形点已下采样: {len(points[terrain_mask]):,} → {len(terrain_points):,}")

    if len(non_terrain_points) > max_points // 2:
        step = len(non_terrain_points) // (max_points // 2) + 1
        non_terrain_points = non_terrain_points[::step]
        print(f"非地形点已下采样: {len(points[~terrain_mask]):,} → {len(non_terrain_points):,}")

    # 创建3D散点图
    fig = go.Figure()

    # 添加地形点（绿色）
    fig.add_trace(go.Scatter3d(
        x=terrain_points[:, 0],
        y=terrain_points[:, 1],
        z=terrain_points[:, 2],
        mode='markers',
        marker=dict(
            size=point_size,
            color='green',
            opacity=0.7
        ),
        name='地形'
    ))

    # 添加非地形点（红色）
    fig.add_trace(go.Scatter3d(
        x=non_terrain_points[:, 0],
        y=non_terrain_points[:, 1],
        z=non_terrain_points[:, 2],
        mode='markers',
        marker=dict(
            size=point_size,
            color='red',
            opacity=0.5
        ),
        name='非地形'
    ))

    # 配置布局
    fig.update_layout(
        title="自适应地形提取结果",
        width=1000,
        height=800,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # 保存HTML文件
    if output_path:
        fig.write_html(output_path)
        print(f"已保存可视化结果至: {output_path}")

    # 显示图形
    if open_browser:
        fig.show(renderer="browser")

def create_output_paths(input_path, output_dir=None):
    """创建输出路径"""
    # 获取输入文件名（不含扩展名）
    input_basename = os.path.basename(input_path)
    input_name = os.path.splitext(input_basename)[0]

    # 如果没有指定输出目录，使用输入文件所在目录
    if output_dir is None:
        output_dir = os.path.dirname(input_path)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 创建输出路径
    output_paths = {
        'terrain_model': os.path.join(output_dir, f"{input_name}_terrain_adaptive.pt"),
        'objects_model': os.path.join(output_dir, f"{input_name}_objects_adaptive.pt"),
        'visualization': os.path.join(output_dir, f"{input_name}_terrain_adaptive.html"),
        'classification_audit': os.path.join(output_dir, f"{input_name}_terrain_classification_audit.json"),
        'classification_indices': os.path.join(output_dir, f"{input_name}_terrain_classification_indices.npz"),
    }

    return output_paths

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='从3D高斯点云中提取起伏地形（自适应算法）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--input', '-i', type=str, required=True,
                       help='输入PT文件路径')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--downsample', '-d', type=float, default=0.2,
                       help='点云下采样网格大小（设为0禁用下采样）')
    parser.add_argument('--grid-size', '-g', type=float, default=2.0,
                       help='区域分析网格大小')
    parser.add_argument('--slope-threshold', '-s', type=float, default=30.0,
                       help='坡度阈值（度）')
    parser.add_argument('--height-var-threshold', '-t', type=float, default=0.5,
                       help='局部高度变化阈值')
    parser.add_argument('--std-factor', '-f', type=float, default=1.5,
                       help='区域高度标准差系数')
    parser.add_argument('--neighbors', '-k', type=int, default=30,
                       help='局部特征计算的邻居点数')
    parser.add_argument('--cluster', '-c', action='store_true',
                       help='使用聚类清理')
    parser.add_argument('--cluster-eps', type=float, default=1.0,
                       help='聚类距离阈值')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='可视化结果')
    parser.add_argument('--open-browser', '-b', action='store_true',
                       help='自动打开浏览器查看结果')
    parser.add_argument('--z-normal', action='store_true',
                       help='Z坐标为正常方向（地面Z值小）。默认假设Z坐标颠倒（地面Z值大）')
    parser.add_argument('--structure-roi-cache', type=str, default=None,
                       help='可选 exact-capsule FE ROI 缓存；结构点先排除出地形分析并始终保留为非地形。这是对原Notebook算法的显式保护规则')

    # 如果没有参数，显示帮助信息
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        print("\n例如: python extract_terrain_adaptive.py -i ./model.pt -v", file=sys.stderr)
        sys.exit(1)

    return parser.parse_args()

def main():
    # 解析参数
    args = parse_arguments()

    print("\n===== 3D高斯点云自适应地形提取工具 =====")
    print(f"Z轴方向: {'正常（地面Z值小）' if args.z_normal else '颠倒（地面Z值大）'}")
    total_start_time = time.time()

    # 创建输出路径
    output_paths = create_output_paths(args.input, args.output_dir)

    # 1. 加载模型
    print("\n[步骤 1/10] 加载模型...")
    points, splats = load_gaussian_model(args.input)

    # This is an opt-in protection rule, not part of the unmodified notebook
    # classifier. Keep all calculations in the original checkpoint coordinates
    # and carry the original row indices through the candidate subset.
    structure_mask = np.zeros(len(points), dtype=bool)
    structure_roi = None
    if args.structure_roi_cache:
        structure_mask, structure_roi = load_structure_roi_mask(points, args.structure_roi_cache)
        print(f"结构ROI保护: {np.sum(structure_mask):,} 个点排除出地形统计并保留为非地形")
    candidate_indices = np.flatnonzero(~structure_mask)
    candidate_points = points[candidate_indices] if args.structure_roi_cache else points
    if len(candidate_points) < args.neighbors:
        raise ValueError(
            f"Only {len(candidate_points)} non-structure candidates remain; "
            f"at least --neighbors={args.neighbors} are needed for terrain analysis"
        )

    # 2. 分析高度分布
    print("\n[步骤 2/10] 分析高度分布...")
    height_stats = analyze_height_distribution(candidate_points)

    # 3. 下采样（可选）
    print("\n[步骤 3/10] 执行点云下采样...")
    if args.downsample > 0:
        working_points, candidate_sample_indices = downsample_points_grid(
            candidate_points, grid_size=args.downsample, z_inverted=not args.z_normal
        )
        sample_indices = candidate_indices[candidate_sample_indices]
        use_sampled = True
    else:
        working_points = candidate_points
        sample_indices = candidate_indices
        use_sampled = False
    if len(working_points) < args.neighbors:
        raise ValueError(
            f"Only {len(working_points)} sampled candidates remain; reduce --downsample "
            f"or --neighbors={args.neighbors}"
        )

    # 4. 计算局部特征
    print("\n[步骤 4/10] 计算局部特征...")
    local_features = calculate_local_features(
        working_points, z_inverted=not args.z_normal, k=args.neighbors
    )

    # 5. 分析局部区域
    print("\n[步骤 5/10] 分析局部区域...")
    regions = analyze_local_regions(working_points, grid_size=args.grid_size)

    # 6. 获取初步地形掩码
    print("\n[步骤 6/10] 获取初步地形掩码...")
    terrain_mask = get_initial_terrain_mask(
        working_points, local_features, regions,
        slope_threshold=args.slope_threshold,
        height_var_threshold=args.height_var_threshold,
        z_inverted=not args.z_normal
    )

    # 7. 使用区域统计优化地形掩码
    print("\n[步骤 7/10] 使用区域统计优化地形掩码...")
    refined_mask = refine_terrain_regions(
        working_points, terrain_mask, regions,
        std_factor=args.std_factor,
        z_inverted=not args.z_normal
    )

    # 8. 可选：使用聚类清理孤立点
    print("\n[步骤 8/10] 使用聚类清理孤立点...")
    if args.cluster:
        final_mask = clean_terrain_clusters(
            working_points, refined_mask,
            eps=args.cluster_eps,
            min_samples=10
        )
    else:
        print("跳过聚类清理（未启用）")
        final_mask = refined_mask

    # 9. 如果使用了下采样，将掩码映射回原始点云
    print("\n[步骤 9/10] 将掩码映射回原始点云...")
    if use_sampled:
        # 初始化原始点云的掩码
        full_terrain_mask = np.zeros(len(points), dtype=bool)
        full_terrain_mask[sample_indices[final_mask]] = True

        # 为没有被采样的点进行最近邻判断
        unsampled_indices = np.setdiff1d(candidate_indices, sample_indices, assume_unique=True)

        if len(unsampled_indices) > 0 and np.sum(final_mask) > 0:
            print(f"使用最近邻方法处理 {len(unsampled_indices):,} 个未采样点...")

            # 创建KD树
            terrain_sampled_indices = sample_indices[final_mask]
            tree = KDTree(points[terrain_sampled_indices])

            # 批量处理
            batch_size = 10000
            n_batches = (len(unsampled_indices) - 1) // batch_size + 1
            for i in tqdm(range(0, len(unsampled_indices), batch_size), total=n_batches, desc="处理未采样点"):
                batch_indices = unsampled_indices[i:i+batch_size]
                batch_points = points[batch_indices]

                # 查找最近的采样地形点
                dists, _ = tree.query(batch_points, k=1)

                # 如果距离小于网格大小，则认为也是地形
                threshold = args.downsample * 1.5
                close_mask = dists.flatten() < threshold
                full_terrain_mask[batch_indices[close_mask]] = True
    else:
        print("未使用下采样，直接使用原始掩码")
        full_terrain_mask = np.zeros(len(points), dtype=bool)
        full_terrain_mask[candidate_indices] = final_mask

    overlap_before_guard = int(np.sum(full_terrain_mask & structure_mask))
    full_terrain_mask[structure_mask] = False
    terrain_roi_overlap = int(np.sum(full_terrain_mask & structure_mask))
    if terrain_roi_overlap:
        raise RuntimeError("Protected structural ROI points entered the terrain output")

    # 10. 保存模型和可视化
    print("\n[步骤 10/10] 保存模型和可视化...")
    save_terrain_model(splats, full_terrain_mask, output_paths['terrain_model'])
    save_terrain_model(splats, ~full_terrain_mask, output_paths['objects_model'])
    terrain_indices = np.flatnonzero(full_terrain_mask)
    objects_indices = np.flatnonzero(~full_terrain_mask)
    np.savez_compressed(output_paths['classification_indices'],
                        terrain_original_indices=terrain_indices,
                        objects_original_indices=objects_indices,
                        protected_structure_original_indices=np.flatnonzero(structure_mask))
    classification_audit = {
        "format": "adaptive-terrain-classification-audit-v1",
        "input_checkpoint": str(Path(args.input).resolve()),
        "baseline_algorithm": "1_3alll_Zhixianta_dixingtuozhan.ipynb / extract_terrain_adaptive.py",
        "classification_variant": (
            "notebook_adaptive_with_explicit_exact_capsule_structure_protection"
            if args.structure_roi_cache else "original_notebook_adaptive"
        ),
        "structure_roi_protection": {
            "enabled": bool(args.structure_roi_cache),
            "roi": structure_roi,
            "rule": (
                "Exclude exact-capsule structural ROI points before height statistics, "
                "downsampling, local features, regional refinement and clustering; "
                "exclude them from nearest-neighbour terrain propagation and keep them "
                "in the objects output. Apply the same original-index guard once more before saving."
                if args.structure_roi_cache else "Disabled; original notebook classification logic is used."
            ),
            "new_rule_not_in_original_notebook_classifier": bool(args.structure_roi_cache),
        },
        "counts": {
            "input": len(points),
            "protected_structure": int(np.sum(structure_mask)),
            "terrain_analysis_candidates": len(candidate_indices),
            "sampled_candidates": len(working_points),
            "sampled_initial_terrain": int(np.sum(terrain_mask)),
            "sampled_refined_terrain": int(np.sum(refined_mask)),
            "sampled_final_terrain": int(np.sum(final_mask)),
            "terrain": len(terrain_indices),
            "objects": len(objects_indices),
            "terrain_roi_overlap_before_final_guard": overlap_before_guard,
            "terrain_roi_overlap": terrain_roi_overlap,
            "objects_roi_overlap": int(np.sum(~full_terrain_mask & structure_mask)),
        },
        "terrain_fraction": float(np.mean(full_terrain_mask)),
        "partition_checks": {
            "terrain_plus_objects_equals_input": len(terrain_indices) + len(objects_indices) == len(points),
            "terrain_roi_overlap_zero": terrain_roi_overlap == 0,
            "original_order_and_gaussian_attributes_preserved": True,
        },
        "parameters": {
            "downsample": args.downsample, "grid_size": args.grid_size,
            "neighbors": args.neighbors, "slope_threshold": args.slope_threshold,
            "height_var_threshold": args.height_var_threshold, "std_factor": args.std_factor,
            "cluster": args.cluster, "cluster_eps": args.cluster_eps, "z_normal": args.z_normal,
        },
        "outputs": {key: str(Path(value).resolve()) for key, value in output_paths.items()},
        "limitation": (
            "ROI protection enforces the supplied structural prior; it does not establish "
            "classification accuracy outside that ROI or introduce a new terrain classifier."
        ),
    }
    with open(output_paths['classification_audit'], 'w', encoding='utf-8') as audit_file:
        json.dump(classification_audit, audit_file, indent=2, ensure_ascii=False)
        audit_file.write('\n')
    print(f"分类审计: {output_paths['classification_audit']}")

    # 11. 可视化
    if args.visualize:
        print("\n正在生成可视化...")
        visualize_terrain(
            points, full_terrain_mask,
            output_path=output_paths['visualization'],
            open_browser=args.open_browser
        )

    # 完成
    total_time = time.time() - total_start_time
    print("\n===== 处理完成 =====")
    print(f"总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print(f"地形点数: {np.sum(full_terrain_mask):,} ({np.sum(full_terrain_mask)/len(points)*100:.1f}%)")
    print(f"非地形点数: {len(points) - np.sum(full_terrain_mask):,}")
    print(f"地形模型: {output_paths['terrain_model']}")
    print(f"非地形模型: {output_paths['objects_model']}")

    print("\n提示: 要查看提取的地形，可使用visualize_terrain.py工具:")
    print(f"python visualize_terrain.py -i {output_paths['terrain_model']} --view 3d")

if __name__ == "__main__":
    main()
