#!/usr/bin/env python
"""
扩展更多层版边缘迭代扩展算法
继续向外扩展更多层
增加旋转功能：使复制的网格从观察中心点看起来朝向一致
增加kriging地形参考：使用kriging插值地形文件来指导扩展区域的高程
"""

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import argparse
from pathlib import Path
from sklearn.neighbors import KDTree
import time
import multiprocessing as mp
from functools import lru_cache
import concurrent.futures

# 添加一个LRU缓存装饰器到kriging查询函数，避免重复查询相同位置
@lru_cache(maxsize=100000)  # 缓存最近100000次查询结果
def cached_kriging_query(x, y, k=4, krig_tree_id=None):
    """带缓存的kriging高度查询，用于避免重复计算"""
    # krig_tree_id只是为了确保不同的KDTree有不同的缓存
    # 实际查询在调用此函数的外层函数中进行
    return None  # 实际返回值由调用者设置

# 用于存储查询结果的简单缓存字典
_kriging_cache = {}

# 优化后的query_kriging_height函数
def query_kriging_height(x, y, krig_tree, krig_points, k=4):
    """查询指定位置的kriging高度"""
    # 四舍五入坐标减少浮点误差影响
    x_rounded = round(x, 4)
    y_rounded = round(y, 4)
    
    # 生成缓存键
    cache_key = (x_rounded, y_rounded, k)
    
    # 检查缓存
    if cache_key in _kriging_cache:
        return _kriging_cache[cache_key]

    # 缓存未命中，执行查询
    distances, indices = krig_tree.query([[x, y]], k=k)
    
    if k > 1:
        # 使用IDW(反距离加权)插值
        weights = 1.0 / (distances + 1e-6)
        weights = weights / np.sum(weights)
        z_vals = krig_points[indices[0], 2]
        z = float(np.sum(weights * z_vals))  # 确保返回标量
    else:
        # 最近邻
        z = float(krig_points[indices[0][0], 2])  # 确保返回标量
    
    # 更新缓存
    if len(_kriging_cache) > 500000:  # 如果缓存过大，清空
        _kriging_cache.clear()
    _kriging_cache[cache_key] = z
    
    return z

def rotate_points_around_center(points, center, angle):
    """
    将点云绕中心点在XY平面内旋转指定角度
    Args:
        points: (N, 3) 点云坐标
        center: (3,) 旋转中心
        angle: 旋转角度（弧度）
    Returns:
        rotated_points: (N, 3) 旋转后的点云
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    # 平移到原点
    translated = points - center
    
    # 2D旋转矩阵 (只在XY平面旋转)
    rotated = translated.copy()
    rotated[:, 0] = cos_a * translated[:, 0] - sin_a * translated[:, 1]
    rotated[:, 1] = sin_a * translated[:, 0] + cos_a * translated[:, 1]
    # Z坐标保持不变
    
    # 平移回去
    return rotated + center

def calculate_rotation_for_consistent_facing(source_pos, target_pos, observer_center):
    """
    计算旋转角度，使源网格和目标网格都朝向观察中心
    
    Args:
        source_pos: 源网格位置 (x, y)
        target_pos: 目标网格位置 (x, y)
        observer_center: 观察中心 (x, y)
    
    Returns:
        rotation_angle: 旋转角度（弧度）
    """
    # 源网格到观察中心的向量
    source_to_obs_x = observer_center[0] - source_pos[0]
    source_to_obs_y = observer_center[1] - source_pos[1]
    source_angle = np.arctan2(source_to_obs_y, source_to_obs_x)
    
    # 目标位置到观察中心的向量
    target_to_obs_x = observer_center[0] - target_pos[0]
    target_to_obs_y = observer_center[1] - target_pos[1]
    target_angle = np.arctan2(target_to_obs_y, target_to_obs_x)
    
    # 计算源网格需要旋转的角度，使其与目标网格朝向一致
    rotation_angle = target_angle - source_angle
    
    # 规范化角度到 [-pi, pi] 区间
    if rotation_angle > np.pi:
        rotation_angle -= 2 * np.pi
    elif rotation_angle < -np.pi:
        rotation_angle += 2 * np.pi
        
    return rotation_angle

def collect_neighborhood_features(pos, current_grids, grid_features, original_grids, original_count, search_radius=4, max_neighbors=8):
    """收集位置周围已有网格的特征和相对位置"""
    neighbors = []
    
    # 扫描周围search_radius个网格距离范围内的所有已填充网格
    for dx in range(-search_radius, search_radius + 1):
        for dy in range(-search_radius, search_radius + 1):
            if dx == 0 and dy == 0:
                continue  # 跳过中心位置
                
            neighbor_pos = (pos[0] + dx, pos[1] + dy)
            distance = np.sqrt(dx*dx + dy*dy)  # 欧氏距离
            
            # 如果邻居位置有网格且在搜索半径内
            if neighbor_pos in current_grids and distance <= search_radius:
                feature = None
                points_count = len(current_grids[neighbor_pos])
                
                # 获取特征
                if neighbor_pos in grid_features:
                    feature = grid_features[neighbor_pos]
                else:
                    # 新扩展的网格，追溯其源特征
                    points = current_grids[neighbor_pos]
                    if len(points) > 0:
                        first_point = points[0]
                        if first_point < original_count:
                            for orig_grid, orig_points in original_grids.items():
                                if first_point in orig_points:
                                    feature = grid_features[orig_grid]
                                    break
                
                if feature is not None:
                    # 记录相对位置、特征、距离和点数量（作为质量指标）
                    neighbors.append(((dx, dy), feature, distance, points_count))
            
    # 按质量和距离排序（先考虑点数量，再考虑距离）
    neighbors.sort(key=lambda x: (-x[3], x[2]))  # 按点数量降序，距离升序
    return neighbors[:max_neighbors]

def collect_corresponding_features(source_pos, target_relative_positions, current_grids, grid_features, original_grids, original_count):
    """
    收集源位置周围与目标位置严格对应的网格特征
    
    Args:
        source_pos: 源网格位置
        target_relative_positions: 目标网格周围的相对位置列表 [(dx, dy), ...]
        current_grids: 当前网格集合
        grid_features: 网格特征
        original_grids: 原始网格
        original_count: 原始点数量
        
    Returns:
        corresponding_features: 与目标位置严格对应的源网格特征列表
    """
    corresponding_features = []
    
    # 对每个目标相对位置，查找源网格中对应的位置
    for (dx, dy), _, _, _ in target_relative_positions:
        corresponding_pos = (source_pos[0] + dx, source_pos[1] + dy)
        
        # 检查对应位置是否存在网格
        if corresponding_pos in current_grids:
            feature = None
            points_count = len(current_grids[corresponding_pos])
            
            # 获取特征
            if corresponding_pos in grid_features:
                feature = grid_features[corresponding_pos]
            else:
                # 新扩展的网格，追溯其源特征
                points = current_grids[corresponding_pos]
                if len(points) > 0:
                    first_point = points[0]
                    if first_point < original_count:
                        for orig_grid, orig_points in original_grids.items():
                            if first_point in orig_points:
                                feature = grid_features[orig_grid]
                                break
            
            if feature is not None:
                # 记录相对位置、特征、点数量
                corresponding_features.append(((dx, dy), feature, points_count))
    
    return corresponding_features

def spatial_alignment_score_strict_position(target_pos, source_pos, current_grids, grid_features, original_grids, original_count, radius=4, search_radius=4, max_neighbors=8, similarity_threshold=0.3):
    """
    计算严格位置对应的空间相似度评分
    
    使用目标网格周围的高质量参考网格，
    在源网格严格对应的位置寻找网格，进行相似度比较
    
    Args:
        target_pos: 目标位置
        source_pos: 源位置
        current_grids: 当前网格集合
        grid_features: 网格特征
        original_grids: 原始网格
        original_count: 原始点数量
        radius: 搜索半径
        
    Returns:
        alignment_score: 位置对应的相似度评分
    """
    # 1. 收集目标位置周围的高质量邻居网格
    target_neighbors = collect_neighborhood_features(
        target_pos, current_grids, grid_features, original_grids, original_count, search_radius=search_radius, max_neighbors=max_neighbors
    )
    
    if not target_neighbors:
        return 0.0  # 如果目标位置没有邻居，相似度为0
    
    # 2. 收集源位置严格对应的网格特征
    source_corresponding = collect_corresponding_features(
        source_pos, target_neighbors, current_grids, grid_features, original_grids, original_count
    )
    
    if not source_corresponding:
        return 0.0  # 如果源位置没有对应特征，相似度为0
    
    # 3. 创建位置到特征的映射
    target_feature_map = {(dx, dy): feature for (dx, dy), feature, _, _ in target_neighbors}
    source_feature_map = {(dx, dy): feature for (dx, dy), feature, _ in source_corresponding}
    
    # 4. 计算相似度得分
    total_score = 0.0
    match_count = 0
    
    # 仅比较两个位置都存在网格的相对位置
    common_positions = set(target_feature_map.keys()) & set(source_feature_map.keys())
    
    for pos in common_positions:
        target_feature = target_feature_map[pos]
        source_feature = source_feature_map[pos]
        
        # 计算特征向量余弦相似度
        similarity = np.dot(target_feature, source_feature) / (
            np.linalg.norm(target_feature) * np.linalg.norm(source_feature) + 1e-8)
        
        if similarity >= similarity_threshold:
            total_score += similarity
            match_count += 1
    
    # 5. 返回平均相似度
    if match_count > 0:
        # 同时考虑匹配数量和平均相似度
        avg_similarity = total_score / match_count
        position_coverage = match_count / len(target_neighbors)
        return avg_similarity * position_coverage
    else:
        return 0.0

def get_orientation_vector(pos, observer_center):
    """
    计算从位置指向观察中心的单位向量
    Args:
        pos: (x, y, z) 位置坐标
        observer_center: (x, y) 观察中心
    Returns:
        norm_vec: 归一化的朝向向量
    """
    dx = observer_center[0] - pos[0]
    dy = observer_center[1] - pos[1]
    length = np.sqrt(dx*dx + dy*dy)
    if length < 1e-8:
        return np.array([0, 0])
    return np.array([dx/length, dy/length])

def rotate_quaternion_around_z(quat, angle):
    """
    绕Z轴旋转四元数
    
    Args:
        quat: 原始四元数 [w, x, y, z]
        angle: 旋转角度（弧度）
    
    Returns:
        rotated_quat: 旋转后的四元数
    """
    # Z轴旋转四元数 (绕Z轴旋转angle弧度)
    half_angle = angle / 2
    cos_half = np.cos(half_angle)
    sin_half = np.sin(half_angle)
    
    z_quat = np.array([cos_half, 0, 0, sin_half])
    
    # 四元数乘法: z_quat * quat
    w1, x1, y1, z1 = z_quat
    w2, x2, y2, z2 = quat
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])

def calculate_3d_direction_vector(position, observer_center):
    """
    计算从位置到观察中心的3D方向向量（考虑Z轴高度）
    
    Args:
        position: [x, y, z] 坐标
        observer_center: [x, y, z] 观察中心坐标
    
    Returns:
        direction: 归一化的3D方向向量
    """
    # 确保observer_center是3D坐标
    if len(observer_center) == 2:
        # 如果只给了XY坐标，使用position的Z坐标作为默认值
        observer_3d = np.array([observer_center[0], observer_center[1], position[2]])
    else:
        observer_3d = np.array(observer_center)
    
    # 计算方向向量
    direction = observer_3d - position
    
    # 归一化
    norm = np.linalg.norm(direction)
    if norm > 1e-8:
        return direction / norm
    else:
        return np.array([0, 0, 1])  # 如果位置与观察中心重合，默认朝上

def rotation_from_two_vectors(v1, v2):
    """
    计算从向量v1旋转到向量v2的旋转四元数
    
    Args:
        v1: 源方向向量（归一化）
        v2: 目标方向向量（归一化）
    
    Returns:
        quaternion: [w, x, y, z] 旋转四元数
    """
    # 确保输入是单位向量
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)
    
    # 计算叉乘和点积
    cross_product = np.cross(v1, v2)
    dot_product = np.dot(v1, v2)
    
    # 特殊情况处理：如果向量平行
    if np.allclose(cross_product, 0) and dot_product > 0:
        return np.array([1, 0, 0, 0])  # 单位四元数（无旋转）
    
    # 特殊情况处理：如果向量反方向平行
    if np.allclose(cross_product, 0) and dot_product < 0:
        # 找一个垂直于v1的向量
        if abs(v1[0]) < abs(v1[1]):
            perpendicular = np.array([0, v1[2], -v1[1]])
        else:
            perpendicular = np.array([v1[2], 0, -v1[0]])
        perpendicular = perpendicular / np.linalg.norm(perpendicular)
        return np.array([0, perpendicular[0], perpendicular[1], perpendicular[2]])
    
    # 一般情况：构造旋转四元数
    s = np.sqrt((1 + dot_product) * 2)
    invs = 1 / s
    
    quaternion = np.array([
        s * 0.5,
        cross_product[0] * invs,
        cross_product[1] * invs,
        cross_product[2] * invs
    ])
    
    return quaternion / np.linalg.norm(quaternion)  # 确保四元数归一化

def rotate_point_by_quaternion(point, quaternion):
    """
    使用四元数旋转点坐标
    
    Args:
        point: [x, y, z] 点坐标
        quaternion: [w, x, y, z] 四元数
    
    Returns:
        rotated_point: 旋转后的点坐标
    """
    # 将点转换为纯四元数 [0, x, y, z]
    point_quaternion = np.array([0, point[0], point[1], point[2]])
    
    # 计算共轭四元数
    q_conjugate = np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]])
    
    # 执行旋转: q * p * q^*
    temp = quaternion_multiply(quaternion, point_quaternion)
    result = quaternion_multiply(temp, q_conjugate)
    
    # 提取旋转后的点坐标
    return result[1:4]

def quaternion_multiply(q1, q2):
    """
    四元数乘法: q1 * q2
    
    Args:
        q1: [w, x, y, z] 第一个四元数
        q2: [w, x, y, z] 第二个四元数
    
    Returns:
        q: 乘积四元数
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])

# 优化的collect_neighborhood_heights函数，使用批量查询
def collect_neighborhood_heights(pos, current_grids, grid_size, x_min, y_min, krig_tree, krig_points, search_radius=4):
    """
    在指定位置周围收集邻域网格的Kriging高度 - 优化版
    
    Args:
        pos: (gx, gy) 网格坐标
        current_grids: 当前网格字典
        grid_size: 网格大小
        x_min, y_min: 网格原点坐标
        krig_tree: Kriging KD树
        krig_points: Kriging点云
        search_radius: 搜索半径
        
    Returns:
        neighbors: [(dx, dy, h, dist, pts), ...] 相对位置、高度、距离和点数
    """
    neighbors = []
    xy_coords = []  # 收集所有需要查询的坐标
    valid_neighbors = []  # 临时存储有效邻居

    # 第一步：收集所有有效邻居位置
    for dx in range(-search_radius, search_radius + 1):
        for dy in range(-search_radius, search_radius + 1):
            if dx == 0 and dy == 0:  # 忽略自身
                continue
            
            grid = (pos[0] + dx, pos[1] + dy)
            if grid in current_grids:
                # 转换到实际坐标
                nx = x_min + grid[0] * grid_size + grid_size/2
                ny = y_min + grid[1] * grid_size + grid_size/2
                
                # 存储信息
                pts = len(current_grids[grid])
                dist = np.sqrt(dx*dx + dy*dy)
                
                xy_coords.append((nx, ny))
                valid_neighbors.append((dx, dy, dist, pts, len(xy_coords)-1))  # 最后一个值是坐标索引

    # 如果没有邻居，直接返回
    if not xy_coords:
        return neighbors
    
    # 第二步：批量查询kriging高度
    heights = []
    for x, y in xy_coords:
        heights.append(query_kriging_height(x, y, krig_tree, krig_points))
    
    # 第三步：构建返回值并排序
    for dx, dy, dist, pts, idx in valid_neighbors:
        h = heights[idx]
        neighbors.append((dx, dy, h, dist, pts))
    
    # 按点数降序、距离升序排序
    neighbors.sort(key=lambda t: (-t[4], t[3]))
    return neighbors[:8]  # 最多返回8个邻居

def build_height_diff_vec(entries):
    """
    构建高度差向量
    
    Args:
        entries: [(dx, dy, h, dist, pts), ...] 高度条目列表
        
    Returns:
        diff_vec: 高度差向量，长度为C(n,2)
    """
    hs = [e[2] for e in entries]
    pairs = []
    for i in range(len(hs)):
        for j in range(i+1, len(hs)):
            pairs.append(hs[i] - hs[j])
    return np.array(pairs, dtype=np.float32)

def compute_height_consistency(target_pos, source_grid, current_grids, grid_size, x_min, y_min, krig_tree, krig_points, height_sigma):
    """
    计算高度一致性评分 - 优化版
    """
    # 获取目标位置周围的高度邻域
    target_h_neigh = collect_neighborhood_heights(target_pos, current_grids, grid_size, x_min, y_min, krig_tree, krig_points)
    
    # 如果邻居数量不足以形成差向量，返回0
    if len(target_h_neigh) < 2:
        return 0.0
    
    # 构建目标位置的高度差向量
    H_t = build_height_diff_vec(target_h_neigh)
    
    # 收集有效的相对位置，避免重复查询不存在的网格
    dx_dy_pairs = [(dx, dy) for (dx, dy, _, _, _) in target_h_neigh]
    existing_pairs = []
    for dx, dy in dx_dy_pairs:
        g = (source_grid[0] + dx, source_grid[1] + dy)
        if g in current_grids:
            existing_pairs.append((dx, dy, g))
    
    # 如果公共邻居数量不足以形成差向量，返回0
    if len(existing_pairs) < 2:
        return 0.0
    
    # 批量查询kriging高度
    source_h_neigh = []
    for dx, dy, g in existing_pairs:
        x = x_min + g[0] * grid_size + grid_size/2
        y = y_min + g[1] * grid_size + grid_size/2
        h = query_kriging_height(x, y, krig_tree, krig_points)
        pts = len(current_grids[g])
        dist = np.sqrt(dx*dx + dy*dy)
        source_h_neigh.append((dx, dy, h, dist, pts))
    
    # 构建源位置的高度差向量
    H_s = build_height_diff_vec(source_h_neigh)
    
    # 如果向量长度不一致，无法比较，返回0
    if len(H_t) != len(H_s):
        return 0.0
    
    # 计算差向量的L2距离，使用向量化操作
    diff_norm2 = np.sum((H_t - H_s)**2)
    
    # 使用高斯衰减作为高度一致性评分
    height_consistency = np.exp(-diff_norm2 / (2 * height_sigma**2))
    
    return height_consistency

# 多进程计算评分的工作函数
def _compute_score_worker(args):
    """
    计算单个候选的评分，用于多进程并行
    
    Args:
        args: (candidate_pos, source_grid, current_grids, grid_features, original_grids, 
               original_count, grid_size, x_min, y_min, krig_tree, krig_points, height_sigma)
    
    Returns:
        (source_grid, spatial_score, height_consistency, final_score)
    """
    (candidate_pos, source_grid, current_grids, grid_features, original_grids, 
     original_count, grid_size, x_min, y_min, krig_tree, krig_points, height_sigma, use_kriging,
     search_radius, max_neighbors, similarity_threshold) = args
    
    # 计算空间对齐评分
    spatial_score = spatial_alignment_score_strict_position(
        candidate_pos, source_grid, current_grids, grid_features, original_grids, original_count,
        search_radius=search_radius, max_neighbors=max_neighbors, similarity_threshold=similarity_threshold
    )
    
    # 如果使用kriging，计算高度一致性评分
    if use_kriging and spatial_score > similarity_threshold:  # 只有空间评分超过阈值时计算高度一致性
        height_consistency = compute_height_consistency(
            candidate_pos, source_grid, current_grids, grid_size, x_min, y_min, 
            krig_tree, krig_points, height_sigma
        )
        final_score = spatial_score * height_consistency
    else:
        height_consistency = 0.0
        final_score = spatial_score
    
    return (source_grid, spatial_score, height_consistency, final_score)

def extended_more_layers_expansion(grid_size=0.4, expand_layers=8, observer_center=(0.5, -0.2, 0.0), 
                                  debug=False, debug_count=5, input_pt=None, kriging_pt=None, 
                                  output_pt=None, max_multiplier=5.0, height_sigma=1.0, fast=False,
                                  no_multiprocessing=False,
                                  top_80_percent=0.8, bottom_82_percent=0.82, min_distance=5, max_candidates=50,
                                  spatial_threshold=0.5, max_candidates_detail=20, early_stop_threshold=0.9,
                                  k_neighbors=4, search_radius=4, max_neighbors=8, similarity_threshold=0.3,
                                  source_prefilter_k=0, source_prefilter_radius=0,
                                  disable_feature_matching=False,
                                  disable_rotation_correction=False,
                                  disable_kriging_height_consistency=False):
    """
    稀疏精细边缘迭代扩展（严格位置对应版）- 性能优化版：
    1. 识别边缘网格
    2. 候选位置稀疏化：确保候选位置之间至少距离2个网格，每层最多50个候选
    3. 严格位置对应打分：
       - 在目标位置周围找到优质网格
       - 在源网格的严格对应位置查找网格
       - 仅比较相同相对位置的网格特征相似度
    4. 从点数前80%且高度后82%的网格交集中选择最高空间对应评分的源网格
    5. 完全复制源网格所有点，进行XY平面旋转和高度平滑调整
    6. 迭代扩展更多层，直到达到5倍原始点数或指定层数
    7. 使用kriging地形作为高程参考，确保扩展点高度与地形一致
    """
    # 记录开始时间
    start_time = time.time()
    
    # 设置默认路径
    if input_pt is None:
        raise ValueError("input_pt must identify a user-provided Gaussian checkpoint")
    
    # 根据输入文件自动设置输出路径
    if output_pt is None:
        input_name = os.path.basename(input_pt)
        base_name = os.path.splitext(input_name)[0]
        output_dir = os.path.dirname(input_pt)
        output_pt = os.path.join(output_dir, f"{base_name}_expanded_param.pt")
    
    # 确保输出目录存在
    Path(output_pt).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"=== 稀疏精细边缘迭代扩展算法（性能优化版）===")
    print(f"网格边长: {grid_size}")
    print(f"扩展层数: {expand_layers}")
    print(f"观察中心: ({observer_center[0]}, {observer_center[1]})")
    print(f"输入文件: {input_pt}")
    print(f"输出文件: {output_pt}")
    if fast:
        print("✅ 已启用快速模式，部分质量换取更高速度")
    if no_multiprocessing:
        print("⚠️ 已禁用多进程加速")
    if disable_feature_matching:
        print("⚠️ 消融模式：禁用特征相似性匹配，改用最近源网格")
    if disable_rotation_correction:
        print("⚠️ 消融模式：禁用 XY 旋转校正")
    if disable_kriging_height_consistency:
        print("⚠️ 消融模式：禁用 Kriging 高程一致性评分与高度对齐，仅保留 Kriging 范围约束")
    
    # 加载模型
    print("加载输入模型...")
    ckpt = torch.load(input_pt, map_location='cpu')
    splats = ckpt["splats"]
    
    # 加载kriging地形文件(如果提供)
    krig_points = None
    krig_tree = None
    krig_x_min, krig_y_min, krig_x_max, krig_y_max = None, None, None, None
    use_kriging = False
    
    if kriging_pt:
        print(f"加载Kriging地形文件: {kriging_pt}")
        krig_ckpt = torch.load(kriging_pt, map_location='cpu')
        krig_points = krig_ckpt['splats']['means'].detach().cpu().numpy()
        
        # 创建KDTree，只使用XY坐标
        krig_tree = KDTree(krig_points[:, :2])
        
        # 计算kriging地形范围
        krig_x_min, krig_y_min = krig_points[:, 0].min(), krig_points[:, 1].min()
        krig_x_max, krig_y_max = krig_points[:, 0].max(), krig_points[:, 1].max()
        
        print(f"Kriging地形覆盖范围: X({krig_x_min:.2f}-{krig_x_max:.2f}), Y({krig_y_min:.2f}-{krig_y_max:.2f})")
        print(f"Kriging地形点数: {len(krig_points):,}")
        use_kriging = True
    use_kriging_height_consistency = use_kriging and not disable_kriging_height_consistency
    
    means = splats["means"].numpy()
    quats = splats["quats"].numpy()
    scales = splats["scales"].numpy()
    opacities = splats["opacities"].numpy()
    sh0 = splats["sh0"].numpy()
    shN = None
    if "shN" in splats:
        shN = splats["shN"].numpy()
    
    original_count = len(means)
    print(f"原始点数: {original_count:,}")
    
    # 检查点是否在kriging范围内
    def in_kriging_range(x, y, margin=grid_size*0.5):
        """检查点是否在kriging范围内"""
        return (krig_x_min-margin <= x <= krig_x_max+margin) and (krig_y_min-margin <= y <= krig_y_max+margin)
    
    # 网格索引转实际坐标
    def grid_to_xy(grid_pos):
        """网格索引转XY实际坐标"""
        gx, gy = grid_pos
        return (x_min + gx * grid_size + grid_size/2, y_min + gy * grid_size + grid_size/2)
    
    # 2. 分析区域
    x_min, x_max = means[:, 0].min(), means[:, 0].max()
    y_min, y_max = means[:, 1].min(), means[:, 1].max()
    z_min, z_max = means[:, 2].min(), means[:, 2].max()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    print(f"原始区域: X[{x_min:.1f}, {x_max:.1f}], Y[{y_min:.1f}, {y_max:.1f}], Z[{z_min:.1f}, {z_max:.1f}]")
    
    # 3. 网格划分
    print("执行网格划分...")
    
    x_grids = int(np.ceil(x_range / grid_size))
    y_grids = int(np.ceil(y_range / grid_size))
    
    print(f"原始网格: {x_grids} x {y_grids} = {x_grids * y_grids:,}")
    
    # 构建网格索引
    grid_points = {}
    
    for idx, pos in enumerate(means):
        x, y, z = pos
        
        gx = int((x - x_min) / grid_size)
        gy = int((y - y_min) / grid_size)
        
        gx = min(gx, x_grids - 1)
        gy = min(gy, y_grids - 1)
        
        if (gx, gy) not in grid_points:
            grid_points[(gx, gy)] = []
        grid_points[(gx, gy)].append(idx)
    
    # 只保留有点的网格
    original_grids = {k: v for k, v in grid_points.items() if len(v) >= 1}
    print(f"原始有效网格: {len(original_grids):,}")
    
    # 4. 筛选点数前80%且高度后82%的网格交集
    print("筛选点数前80%且高度后82%的网格交集...")
    
    # 4.1 按点数筛选前80%
    grid_point_counts = [(grid, len(points)) for grid, points in original_grids.items()]
    grid_point_counts.sort(key=lambda x: x[1], reverse=True)  # 按点数降序排列
    
    top_80_percent_count = int(len(grid_point_counts) * top_80_percent)
    top_80_percent_grids = set([grid for grid, _ in grid_point_counts[:top_80_percent_count]])
    
    print(f"点数前{int(top_80_percent*100)}%网格: {len(top_80_percent_grids):,}")
    
    # 计算每个网格的高度（前10%z值中位数减去后10%z值中位数）
    grid_heights = []
    for grid, points in original_grids.items():
        grid_means = means[points]
        z_coords = grid_means[:, 2]
        z_coords.sort()  # 按z坐标排序
        
        # 如果点数太少，使用最小和最大值的差作为高度
        if len(z_coords) < 10:
            height = np.max(z_coords) - np.min(z_coords)
        else:
            # 计算前10%的点的z中位数
            top_10_percent_idx = max(1, int(len(z_coords) * 0.1))
            top_10_percent_z = z_coords[-top_10_percent_idx:]
            top_median = np.median(top_10_percent_z)
            
            # 计算后10%的点的z中位数
            bottom_10_percent_idx = max(1, int(len(z_coords) * 0.1))
            bottom_10_percent_z = z_coords[:bottom_10_percent_idx]
            bottom_median = np.median(bottom_10_percent_z)
            
            # 高度 = 前10%中位数 - 后10%中位数
            height = top_median - bottom_median
        
        grid_heights.append((grid, height))

    # 按高度降序排列
    grid_heights.sort(key=lambda x: x[1], reverse=True)  # 按高度从高到低排序

    # 排除最高的(1-bottom_82_percent)比例，保留剩余bottom_82_percent
    top_x_percent_count = int(len(grid_heights) * (1 - bottom_82_percent))
    bottom_x_percent_height_grids = set([grid for grid, _ in grid_heights[top_x_percent_count:]])
    
    print(f"高度后{int(bottom_82_percent*100)}%网格: {len(bottom_x_percent_height_grids):,}")
    
    # 4.3 取交集作为候选源网格
    candidate_grids_set = top_80_percent_grids & bottom_x_percent_height_grids
    candidate_grids = {grid: len(original_grids[grid]) for grid in candidate_grids_set}
    
    print(f"候选源网格（点数前{int(top_80_percent*100)}% ∩ 高度后{int(bottom_82_percent*100)}%）: {len(candidate_grids):,}")
    candidate_grid_list = list(candidate_grids.keys())
    candidate_grid_xy = np.asarray(candidate_grid_list, dtype=np.float32)
    if source_prefilter_k > 0 or source_prefilter_radius > 0:
        print(
            "启用源网格预筛选: "
            f"top_k={source_prefilter_k}, radius={source_prefilter_radius} 网格"
        )
    
    # 统计候选网格点数分布
    candidate_point_counts = list(candidate_grids.values())
    print(f"候选网格点数统计: 平均={np.mean(candidate_point_counts):.1f}, 最小={np.min(candidate_point_counts)}, 最大={np.max(candidate_point_counts)}")
    
    # 统计候选网格高度分布
    candidate_heights = [height for grid, height in grid_heights if grid in candidate_grids_set]
    print(f"候选网格高度统计: 平均={np.mean(candidate_heights):.1f}, 最小={np.min(candidate_heights):.1f}, 最大={np.max(candidate_heights):.1f}")
    
    # 5. 计算简化网格特征（仅考虑高斯点内在属性）
    print("计算简化网格特征（仅考虑quats、scales、opacities、sh0、shN）...")
    
    grid_features = {}
    for (gx, gy), point_indices in original_grids.items():
        grid_quats = quats[point_indices]
        grid_scales = scales[point_indices]
        grid_opacities = opacities[point_indices]
        grid_sh0 = sh0[point_indices]
        grid_shN_values = None
        if shN is not None:
            grid_shN_values = shN[point_indices]
        
        # 高斯点内在属性特征向量
        features = []
        
        # 四元数特征 (4维)
        features.extend([
            np.mean(grid_quats[:, 0]),      # quat_w
            np.mean(grid_quats[:, 1]),      # quat_x
            np.mean(grid_quats[:, 2]),      # quat_y
            np.mean(grid_quats[:, 3]),      # quat_z
        ])
        
        # 尺度特征 (3维)
        features.extend([
            np.mean(grid_scales[:, 0]),     # scale_x
            np.mean(grid_scales[:, 1]),     # scale_y
            np.mean(grid_scales[:, 2]),     # scale_z
        ])
        
        # 透明度特征 (1维)
        features.append(np.mean(grid_opacities))
        
        # 球谐函数 sh0 特征 (RGB)。gsplat checkpoints in the paper use
        # (N, 1, 3), while the historical terrain demo uses (N, 3, 1).
        # Normalize only this feature view; preserve the original tensor
        # layout when writing new splats below.
        if grid_sh0.ndim != 3:
            raise ValueError(f"sh0 must be a 3-D tensor, got {grid_sh0.shape}")
        if grid_sh0.shape[1] == 1 and grid_sh0.shape[2] >= 3:
            rgb = grid_sh0[:, 0, :3]
        elif grid_sh0.shape[2] == 1 and grid_sh0.shape[1] >= 3:
            rgb = grid_sh0[:, :3, 0]
        else:
            raise ValueError(
                "sh0 must have RGB in either the last or middle dimension; "
                f"got {grid_sh0.shape}"
            )
        features.extend(np.mean(rgb, axis=0).tolist())
        
        # 球谐函数shN特征 (高阶球谐) - 取前6维避免特征向量过长
        if grid_shN_values is not None:
            if grid_shN_values.ndim != 3:
                raise ValueError(f"shN must be a 3-D tensor, got {grid_shN_values.shape}")
            if grid_shN_values.shape[1] == 3:
                # Historical terrain demo: (N, RGB, coefficients).
                higher = grid_shN_values[:, 0, :]
            elif grid_shN_values.shape[2] == 3:
                # gsplat convention: (N, coefficients, RGB).
                higher = grid_shN_values[:, :, 0]
            else:
                raise ValueError(
                    "shN must have RGB in either the middle or last dimension; "
                    f"got {grid_shN_values.shape}"
                )
            for i in range(min(6, higher.shape[1])):
                features.append(float(np.mean(higher[:, i])))
        else:
            # 如果没有高阶球谐，用零填充
            features.extend([0.0] * 6)
        
        grid_features[(gx, gy)] = np.array(features)
    
    feature_dim = len(next(iter(grid_features.values())))
    print(f"网格特征维度: {feature_dim}")
    
    # 6. 准备严格空间位置对应打分系统
    print("准备严格空间位置对应打分系统...")
    print(f"网格特征维度: {feature_dim}")
    print(f"候选源网格数量: {len(candidate_grids):,}")
    print(f"将使用4邻域空间对应模式进行相似度评估 (search_radius={search_radius}, max_neighbors={max_neighbors}, similarity_threshold={similarity_threshold})")
    
    # 7. 初始化扩展数据
    print("初始化扩展数据...")
    # 准备扩展属性存储
    extended_attrs = {
        "means": means.tolist(),          # ← 每行 [x,y,z]
        "quats": quats.tolist(),
        "scales": scales.tolist(),
        "opacities": opacities.tolist(),
        "sh0": sh0.tolist()
    }
    # Keep the provenance of every expanded Gaussian.  The first
    # ``original_count`` entries are the input rows; each later entry is a
    # copied/rotated Gaussian whose FE displacement must be inherited from
    # this source row when the dynamic chain is connected.
    parent_indices = np.arange(original_count, dtype=np.int64).tolist()
    if shN is not None:
        extended_attrs["shN"] = shN.tolist()
    
    # 当前网格状态
    current_grids = original_grids.copy()
    total_filled_count = 0
    
    # 8. 逐层扩展
    print("开始逐层扩展...")
    current_count = original_count
    max_count = original_count * max_multiplier  # 最多扩展到max_multiplier倍点数
    layer_filled = 0
    
    # 定义物理邻域方向（16邻域：8个距离1的邻域 + 8个距离2的邻域）
    neighbor_16 = [
        # 距离1的8个邻域（3x3去掉中心）
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
        # 距离2的8个主要方向邻域
        (-2, 0), (2, 0), (0, -2), (0, 2),
        (-2, -2), (-2, 2), (2, -2), (2, 2)
    ]
    
    # 改成8邻域：3×3 去掉中心
    neighbor_8 = [(-1, -1), (-1, 0), (-1, 1),
                  ( 0, -1),           ( 0, 1),
                  ( 1, -1), ( 1, 0),  ( 1, 1)]
    
    for layer in range(expand_layers):
        print(f"\n=== 第 {layer+1} 层扩展 ===")
        
        # 8.1 找出边缘网格
        edge_grids = []
        for grid_pos in current_grids:
            # 检查网格是否有空缺邻居（说明是边缘）
            neighbors = [(grid_pos[0]+dx, grid_pos[1]+dy) for dx, dy in neighbor_8]
            if any(neighbor not in current_grids for neighbor in neighbors):
                edge_grids.append(grid_pos)
        
        print(f"找到边缘网格: {len(edge_grids):,}")
        
        if len(edge_grids) == 0:
            print("没有边缘网格，扩展结束")
            break
        
        # 8.2 找出候选位置
        candidate_positions = []
        for grid_pos in edge_grids:
            # 检查邻居位置
            neighbors = [(grid_pos[0]+dx, grid_pos[1]+dy) for dx, dy in neighbor_8]
            for neighbor in neighbors:
                if neighbor not in current_grids:
                    candidate_positions.append(neighbor)
        
        # 去重
        candidate_positions = list(set(candidate_positions))
        print(f"候选位置: {len(candidate_positions)}")
        
        # 稀疏化候选位置
        sparse_candidate_positions = []
        # min_distance, max_candidates参数化
        for pos in candidate_positions:
            # 确保至少有min_distance个网格的距离
            if not any((abs(pos[0] - p[0]) + abs(pos[1] - p[1])) < min_distance for p in sparse_candidate_positions):
                sparse_candidate_positions.append(pos)
            # 限制每层最多max_candidates个候选
            if len(sparse_candidate_positions) >= max_candidates:
                break
        
        print(f"稀疏化后候选位置: {len(sparse_candidate_positions):,}")
        
        # 如果使用kriging，只保留kriging范围内的候选位置
        if use_kriging:
            valid_candidates = []
            for pos in sparse_candidate_positions:
                # 网格坐标转回实际坐标
                x, y = grid_to_xy(pos)
                
                # 检查是否在kriging范围内
                if in_kriging_range(x, y):
                    valid_candidates.append(pos)
            
            print(f"Kriging范围内的候选位置: {len(valid_candidates):,}/{len(sparse_candidate_positions):,}")
            sparse_candidate_positions = valid_candidates
            
            # 如果没有有效候选，结束扩展
            if len(sparse_candidate_positions) == 0:
                print("没有找到Kriging范围内的有效候选位置，扩展结束")
                break
                
        # 8.3 对每个候选位置打分并选择最佳源网格
        added_count = 0
        
        for i, candidate_pos in enumerate(sparse_candidate_positions):
            # 如果已经达到最大点数，不再添加
            if current_count >= max_count:
                print(f"达到最大点数 {max_count:,}，停止扩展")
                break
            
            # 候选位置的实际坐标
            candidate_x = x_min + candidate_pos[0] * grid_size + grid_size/2
            candidate_y = y_min + candidate_pos[1] * grid_size + grid_size/2
            
            # 计算从此位置到观察中心的方向向量
            direction_vector = np.array([
                observer_center[0] - candidate_x,
                observer_center[1] - candidate_y,
                observer_center[2] - 0  # 假设z为0
            ])
            direction_vector = direction_vector / np.linalg.norm(direction_vector)
            
            # 8.4 为此位置寻找最佳源网格
            best_score = -1
            best_source = None
            active_candidate_grids = candidate_grid_list
            if (source_prefilter_k > 0 or source_prefilter_radius > 0) and len(candidate_grid_list) > 0:
                # 加速：只对距离目标位置较近的源网格做特征/高程评分，避免每次全量扫描。
                delta = candidate_grid_xy - np.asarray(candidate_pos, dtype=np.float32)
                dist2 = np.einsum("ij,ij->i", delta, delta)
                valid_idx = np.arange(len(candidate_grid_list))
                if source_prefilter_radius > 0:
                    radius2 = float(source_prefilter_radius * source_prefilter_radius)
                    valid_idx = valid_idx[dist2 <= radius2]
                if len(valid_idx) == 0:
                    valid_idx = np.arange(len(candidate_grid_list))
                if source_prefilter_k > 0 and len(valid_idx) > source_prefilter_k:
                    local_order = np.argpartition(dist2[valid_idx], source_prefilter_k - 1)[:source_prefilter_k]
                    valid_idx = valid_idx[local_order]
                active_candidate_grids = [candidate_grid_list[int(idx)] for idx in valid_idx]
            
            # 性能优化：先进行快速粗筛选
            candidate_scores = []
            # spatial_threshold, max_candidates_detail参数化
            use_multiprocessing = (
                not debug
                and len(active_candidate_grids) > 50
                and not no_multiprocessing
                and not disable_feature_matching
            )
            
            if disable_feature_matching:
                # 消融：不比较颜色/尺度/透明度等特征，直接选择空间最近源网格。
                best_source = min(
                    active_candidate_grids,
                    key=lambda source_grid: (source_grid[0] - candidate_pos[0]) ** 2
                    + (source_grid[1] - candidate_pos[1]) ** 2,
                )
                best_score = 1.0
            elif use_multiprocessing and not fast:  # 在非快速模式下使用并行处理
                # 准备工作参数
                work_args = []
                for source_grid in active_candidate_grids:
                    args = (candidate_pos, source_grid, current_grids, grid_features, 
                            original_grids, original_count, grid_size, x_min, y_min, 
                            krig_tree, krig_points, height_sigma, use_kriging_height_consistency,
                            search_radius, max_neighbors, similarity_threshold)
                    work_args.append(args)
                
                # 使用线程池并行计算评分
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    results = list(executor.map(_compute_score_worker, work_args))
                
                # 处理结果
                for source_grid, spatial_score, height_consistency, score in results:
                    if score > best_score:
                        best_score = score
                        best_source = source_grid
                        
                        if debug and i < debug_count:
                            print(f"    空间评分: {spatial_score:.3f}, 高度一致性评分: {height_consistency:.3f}, 总评分: {score:.3f}")
                        
                        # 提前终止：如果评分已经很高，就不再继续搜索
                        if score > 0.9:
                            break
            else:  # 使用单线程处理
                for source_grid in active_candidate_grids:
                    # 基础空间对齐评分
                    spatial_score = spatial_alignment_score_strict_position(
                        candidate_pos, source_grid, 
                        current_grids, grid_features, original_grids, original_count,
                        search_radius=search_radius, max_neighbors=max_neighbors, similarity_threshold=similarity_threshold
                    )
                    
                    if spatial_score > spatial_threshold:  # 只考虑空间评分较高的候选
                        candidate_scores.append((source_grid, spatial_score))
                
                # 按空间评分排序，只保留前N个候选进行详细评分
                candidate_scores.sort(key=lambda x: x[1], reverse=True)
                top_candidates = candidate_scores[:min(max_candidates_detail, len(candidate_scores))]
                
                # 对筛选后的候选进行详细评分
                for source_grid, spatial_score in top_candidates:
                    # 如果启用Kriging高程一致性，加入高度一致性评分
                    if use_kriging_height_consistency:
                        # 计算改进版高度一致性评分
                        height_consistency = compute_height_consistency(
                            candidate_pos, source_grid, 
                            current_grids, grid_size, x_min, y_min, 
                            krig_tree, krig_points, height_sigma
                        )
                        
                        # 将空间评分与高度一致性评分相结合
                        score = spatial_score * height_consistency
                        
                        if debug and i < debug_count and score > best_score:
                            print(f"    空间评分: {spatial_score:.3f}, 高度一致性评分: {height_consistency:.3f}, 总评分: {score:.3f}")
                    else:
                        score = spatial_score
                    
                    if score > best_score:
                        best_score = score
                        best_source = source_grid
                        
                        # 提前终止：如果评分已经很高，就不再继续搜索
                        # early_stop_threshold参数化
                        if score > early_stop_threshold:
                            break

            # 如果找到了合适的源网格
            if best_source is not None:
                source_points = original_grids[best_source]
                source_coords = means[source_points]
                
                # 计算平均高度
                source_avg_z = float(np.mean(source_coords[:, 2]))
                
                # 修改高度调整逻辑，使用Kriging高度差作为整体平移量
                height_offset = 0.0
                if use_kriging_height_consistency:
                    # 计算源网格中心的Kriging高度
                    source_x = x_min + best_source[0] * grid_size + grid_size/2
                    source_y = y_min + best_source[1] * grid_size + grid_size/2
                    krig_src = query_kriging_height(source_x, source_y, krig_tree, krig_points)
                    
                    # 计算目标位置的Kriging高度
                    krig_tgt = query_kriging_height(candidate_x, candidate_y, krig_tree, krig_points)
                    
                    # 计算高度偏移 = 目标Kriging高度 - 源Kriging高度
                    height_offset = krig_tgt - krig_src
                    
                    if debug and i < debug_count:
                        print(f"    高度调整: 源Kriging高度={krig_src:.2f}, 目标Kriging高度={krig_tgt:.2f}, 偏移={height_offset:.2f}")

                # 目标位置的Z坐标初步取源网格的平均高度，如果有kriging则调整
                target_center_pos_3d = np.array([candidate_x, candidate_y, source_avg_z + height_offset])
                
                # 计算从源网格中心到观察中心的方向
                source_x = x_min + best_source[0] * grid_size + grid_size/2
                source_y = y_min + best_source[1] * grid_size + grid_size/2
                source_center = np.array([source_x, source_y, 0])
                
                source_direction = np.array([
                    observer_center[0] - source_x,
                    observer_center[1] - source_y,
                    0  # 不考虑Z方向
                ])
                source_direction = source_direction / np.linalg.norm(source_direction)
                
                # 计算需要的旋转角度
                target_direction = np.array([
                    observer_center[0] - candidate_x,
                    observer_center[1] - candidate_y,
                    0
                ])
                target_direction = target_direction / np.linalg.norm(target_direction)
                
                # 叉积判断旋转方向
                cross_product = np.cross(source_direction[:2], target_direction[:2])
                dot_product = np.dot(source_direction[:2], target_direction[:2])
                
                rotation_angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
                if cross_product < 0:
                    rotation_angle = -rotation_angle
                if disable_rotation_correction:
                    rotation_angle = 0.0
                
                if debug and i < debug_count:
                    print(f"    候选位置 {i+1}: ({candidate_x:.2f}, {candidate_y:.2f})")
                    print(f"    最佳源网格: {best_source} (评分: {best_score:.4f})")
                    print(f"    旋转角度: {rotation_angle*180/np.pi:.2f}°")
                
                if layer_filled == 0 and i == 0:
                    print(f"    使用XY平面旋转+Kriging高度基准调整")
                
                # 计算邻域网格的平均高度（用于没有Kriging高程约束时的高度调整）- 使用16邻域
                if not use_kriging_height_consistency:
                    neighbor_heights = []
                    neighbor_count = 0
                    for dx, dy in neighbor_16:
                        neighbor = (candidate_pos[0] + dx, candidate_pos[1] + dy)
                        if neighbor in current_grids:
                            # 获取邻域网格的点云
                            neighbor_points = current_grids[neighbor]
                            if len(neighbor_points) > 0:
                                neighbor_count += 1
                                # 计算邻域网格的平均高度
                                if neighbor_points[0] < original_count:
                                    # 原始网格点
                                    neighbor_z_coords = means[neighbor_points, 2]
                                else:
                                    # 扩展网格点 - 修正索引方式（Bug A修复）
                                    # 直接使用全局索引，不再减去original_count
                                    neighbor_z_coords = [extended_attrs["means"][point_idx][2] for point_idx in neighbor_points]
                                neighbor_heights.extend(neighbor_z_coords)
                    
                    # 计算目标高度
                    if len(neighbor_heights) > 0:
                        target_avg_height = np.mean(neighbor_heights)
                        height_offset = target_avg_height - source_avg_z
                    else:
                        # 如果没有邻域，保持原高度
                        height_offset = 0.0

                # 完全复制整个网格的所有点（带XY旋转和高度调整）
                N = len(source_points)
                source_coords = means[source_points]  # (N, 3)
                offsets = source_coords - np.array([source_x, source_y, source_avg_z])
                # 批量旋转XY
                if abs(rotation_angle) > 1e-6:
                    rotated_offsets = rotate_points_around_center(offsets[:, :2], np.zeros(2), rotation_angle)
                else:
                    rotated_offsets = offsets[:, :2]
                new_xy = np.array([candidate_x, candidate_y]) + rotated_offsets
                # z值批量调整
                if use_kriging_height_consistency:
                    new_z = krig_tgt + offsets[:, 2]
                else:
                    new_z = source_coords[:, 2] + height_offset
                new_points = np.column_stack([new_xy, new_z])
                extended_attrs["means"].extend(new_points.tolist())

                # 批量处理 quats
                if 'quats' in splats:
                    old_quats = quats[source_points]
                    if abs(rotation_angle) > 1e-6:
                        new_quats = np.array([rotate_quaternion_around_z(q, rotation_angle) for q in old_quats])
                    else:
                        new_quats = old_quats
                    extended_attrs["quats"].extend(new_quats.tolist())

                # 其他属性批量复制
                extended_attrs["scales"].extend(scales[source_points].tolist())
                extended_attrs["opacities"].extend(opacities[source_points].tolist())
                extended_attrs["sh0"].extend(sh0[source_points].tolist())
                if "shN" in extended_attrs and shN is not None:
                    extended_attrs["shN"].extend(shN[source_points].tolist())

                parent_indices.extend(np.asarray(source_points, dtype=np.int64).tolist())

                # 新索引批量生成
                new_point_indices = list(range(len(extended_attrs["means"]) - N, len(extended_attrs["means"])))
                total_filled_count += N
                
                # 添加到当前网格
                current_count += len(source_points)
                added_count += len(source_points)
                current_grids[candidate_pos] = new_point_indices
                # 为新网格添加特征（复制源网格的特征）
                grid_features[candidate_pos] = grid_features[best_source].copy()
        
        # 更新层填充计数
        layer_filled += 1
        print(f"第 {layer+1} 层: 添加 {added_count:,} 点, 当前总点数: {current_count:,}/{max_count:,}")
        
        # 检查是否达到最大点数
        if current_count >= max_count:
            print(f"达到最大点数 {max_count:,}，停止扩展")
            break
            
        # 如果这一层没有添加任何点，则结束扩展
        if added_count == 0:
            print("本层没有添加任何点，扩展结束")
            break
    
    # 计算用时
    elapsed = time.time() - start_time
    print(f"\n扩展完成，总用时: {elapsed:.1f}秒")
    print(f"总计填充点数: {total_filled_count:,}")
    
    # 9. 保存结果
    print("保存结果...")
    expanded_splats = {}
    
    # 直接将extended_attrs转为tensor，不需要reshape
    for key, val_list in extended_attrs.items():
        expanded_splats[key] = torch.tensor(np.asarray(val_list), dtype=torch.float32)
    
    # 添加原始splats中存在但extended_attrs中不存在的属性
    for key, value in splats.items():
        if key not in expanded_splats:
            expanded_splats[key] = value
    
    torch.save({"splats": expanded_splats}, output_pt)

    final_count = len(expanded_splats["means"])
    parent_index_path = Path(output_pt).with_suffix(".parent_index.npy")
    parent_index_array = np.asarray(parent_indices, dtype=np.int64)
    if len(parent_index_array) != final_count:
        raise RuntimeError(
            f"parent-index length mismatch: {len(parent_index_array)} != {final_count}"
        )
    if not np.array_equal(parent_index_array[:original_count], np.arange(original_count)):
        raise RuntimeError("expanded checkpoint prefix is not aligned with the input rows")
    if np.any(parent_index_array < 0) or np.any(parent_index_array >= original_count):
        raise RuntimeError("expanded checkpoint contains an invalid parent index")
    np.save(parent_index_path, parent_index_array)
    print(f"父Gaussian映射: {parent_index_path}")
    
    # 验证
    print(f"\n✅ 扩展更多层版边缘迭代扩展完成!")
    print(f"原始: {original_count:,} 点")
    print(f"最终: {final_count:,} 点")
    print(f"新增: {final_count - original_count:,} 点")
    print(f"增长: {final_count / original_count:.2f}x")
    
    print(f"\n算法统计:")
    print(f"网格边长: {grid_size}")
    print(f"扩展层数: {expand_layers}")
    print(f"观察中心: ({observer_center[0]}, {observer_center[1]})")
    print(f"特征维度: {feature_dim} (高斯点内在属性)")
    print(f"原始网格: {len(original_grids):,}")
    print(f"候选网格: {len(candidate_grids):,} (点数前{int(top_80_percent*100)}% ∩ 高度后{int(bottom_82_percent*100)}%)")
    print(f"最终网格: {len(current_grids):,}")
    
    # 计算扩展区域
    final_means = expanded_splats["means"].numpy()
    final_x_min, final_y_min = final_means[:, 0].min(), final_means[:, 1].min()
    final_x_max, final_y_max = final_means[:, 0].max(), final_means[:, 1].max()
    final_x_range = final_x_max - final_x_min
    final_y_range = final_y_max - final_y_min
    
    print(f"\n区域扩展:")
    print(f"原始区域: {x_range:.1f} x {y_range:.1f}")
    print(f"最终区域: {final_x_range:.1f} x {final_y_range:.1f}")
    print(f"区域增长: X方向 {final_x_range/x_range:.2f}x, Y方向 {final_y_range/y_range:.2f}x")
    
    if use_kriging_height_consistency:
        print(f"\n使用了Kriging地形文件: {kriging_pt}")
        print(f"Kriging范围: X({krig_x_min:.2f}-{krig_x_max:.2f}), Y({krig_y_min:.2f}-{krig_y_max:.2f})")
        print(f"高度一致性sigma: {height_sigma:.2f}")
    elif use_kriging:
        print(f"\n使用了Kriging范围约束，但禁用了Kriging高程一致性评分与高度对齐: {kriging_pt}")
    else:
        print("\n未使用Kriging地形文件，使用16邻域平均高度")

def parse_args():
    parser = argparse.ArgumentParser(description='扩展3D高斯飞溅点云')
    parser.add_argument('--input', type=str, required=True,
                        help='输入PT文件路径')
    parser.add_argument('--kriging', type=str, default=None,
                        help='kriging地形PT文件路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出PT文件路径(默认根据输入自动生成)')
    parser.add_argument('--grid-size', type=float, default=0.4,
                        help='网格大小')
    parser.add_argument('--expand-layers', type=int, default=50,
                        help='最大扩展层数')
    parser.add_argument('--observer-center', type=str, default="0.5,-0.2,0.0",
                        help='观察中心坐标，格式为"x,y,z"')
    parser.add_argument('--max-multiplier', type=float, default=5.0,
                        help='最大扩展倍数')
    parser.add_argument('--height-sigma', type=float, default=1.0,
                        help='高度一致性评分的sigma值')
    parser.add_argument('--debug', action='store_true',
                        help='开启调试输出')
    parser.add_argument('--debug-count', type=int, default=5,
                        help='调试时输出的网格数量')
    parser.add_argument('--fast', action='store_true',
                        help='启用快速模式，以牺牲部分质量为代价加快速度')
    parser.add_argument('--no-multiprocessing', action='store_true',
                        help='禁用多进程加速')
    # 新增参数
    parser.add_argument('--top-80-percent', type=float, default=0.8, help='点数前百分比，候选网格筛选，默认0.8')
    parser.add_argument('--bottom-82-percent', type=float, default=0.82, help='高度后百分比，候选网格筛选，默认0.82')
    parser.add_argument('--min-distance', type=int, default=5, help='候选位置稀疏化的最小距离，默认5')
    parser.add_argument('--max-candidates', type=int, default=50, help='每层最大候选位置数，默认50')
    parser.add_argument('--spatial-threshold', type=float, default=0.5, help='空间评分阈值，默认0.5')
    parser.add_argument('--max-candidates-detail', type=int, default=20, help='详细评分时最多考虑的候选数，默认20')
    parser.add_argument('--early-stop-threshold', type=float, default=0.9, help='详细评分提前终止阈值，默认0.9')
    parser.add_argument('--k-neighbors', type=int, default=4, help='kriging最近邻数，默认4')
    parser.add_argument('--search-radius', type=int, default=4, help='邻域搜索半径，默认4')
    parser.add_argument('--max-neighbors', type=int, default=8, help='邻域最大数量，默认8')
    parser.add_argument('--similarity-threshold', type=float, default=0.3, help='空间/特征相似度阈值，默认0.3')
    parser.add_argument('--source-prefilter-k', type=int, default=0,
                        help='加速：每个目标候选只评分最近的K个源网格，0表示关闭')
    parser.add_argument('--source-prefilter-radius', type=int, default=0,
                        help='加速：每个目标候选只评分指定网格半径内的源网格，0表示关闭')
    parser.add_argument('--disable-feature-matching', action='store_true',
                        help='消融：禁用特征相似性匹配，源网格改为最近网格')
    parser.add_argument('--disable-rotation-correction', action='store_true',
                        help='消融：禁用XY平面旋转校正')
    parser.add_argument('--disable-kriging-height-consistency', action='store_true',
                        help='消融：禁用Kriging高程一致性评分与高度对齐，仅保留Kriging范围约束')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # 解析观察中心坐标
    observer_center = tuple(map(float, args.observer_center.split(',')))
    
    # 调用主函数
    extended_more_layers_expansion(
        grid_size=args.grid_size,
        expand_layers=args.expand_layers,
        observer_center=observer_center,
        debug=args.debug,
        debug_count=args.debug_count,
        input_pt=args.input,
        kriging_pt=args.kriging,
        output_pt=args.output,
        max_multiplier=args.max_multiplier,
        height_sigma=args.height_sigma,
        fast=args.fast,
        no_multiprocessing=args.no_multiprocessing,
        top_80_percent=args.top_80_percent,
        bottom_82_percent=args.bottom_82_percent,
        min_distance=args.min_distance,
        max_candidates=args.max_candidates,
        spatial_threshold=args.spatial_threshold,
        max_candidates_detail=args.max_candidates_detail,
        early_stop_threshold=args.early_stop_threshold,
        k_neighbors=args.k_neighbors,
        search_radius=args.search_radius,
        max_neighbors=args.max_neighbors,
        similarity_threshold=args.similarity_threshold,
        source_prefilter_k=args.source_prefilter_k,
        source_prefilter_radius=args.source_prefilter_radius,
        disable_feature_matching=args.disable_feature_matching,
        disable_rotation_correction=args.disable_rotation_correction,
        disable_kriging_height_consistency=args.disable_kriging_height_consistency
    ) 
