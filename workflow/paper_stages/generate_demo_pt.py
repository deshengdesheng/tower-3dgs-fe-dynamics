#!/usr/bin/env python
"""
生成3D高斯点云示例数据文件(demo.pt)
用于演示地形提取、克里金插值和多层边缘扩展算法

特点：
1. 包含起伏山丘(约70%)和低矮建筑/植被(约30%)
2. 边缘有对称缺口用于展示扩展算法
3. Z轴方向：地面Z值大(向下增长)
4. 格式：PyTorch .pt格式，包含必要字段
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import time
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import RegularGridInterpolator

def create_terrain(size=30.0, resolution=0.2, height_range=10.0,
                   z_invert=True, seed=42):
    """
    创建起伏地形

    Args:
        size: 地形边长(米)
        resolution: 每个网格大小(米)
        height_range: 高度范围(米)
        z_invert: Z轴是否颠倒(地形点Z值大)
        seed: 随机数种子

    Returns:
        positions: 地形点的位置坐标 (N, 3)
        heights_normalized: 归一化的高度值 [0,1]
    """
    np.random.seed(seed)

    # 创建网格坐标
    grid_size = int(size / resolution)
    x = np.linspace(0, size, grid_size)
    y = np.linspace(0, size, grid_size)
    xx, yy = np.meshgrid(x, y)

    # 生成多频率噪声地形
    heights = np.zeros((grid_size, grid_size))

    # 生成多个不同频率的正弦波并叠加
    frequencies = [1, 2, 4, 8, 16]
    amplitudes = [1.0, 0.5, 0.25, 0.125, 0.0625]

    for freq, amp in zip(frequencies, amplitudes):
        phase_x = np.random.rand() * 2 * np.pi
        phase_y = np.random.rand() * 2 * np.pi

        heights += amp * np.sin(freq * np.pi * xx / size + phase_x)
        heights += amp * np.sin(freq * np.pi * yy / size + phase_y)

    # 加入一些随机波动
    noise = np.random.normal(0, 0.1, (grid_size, grid_size))
    heights += noise

    # 归一化高度到[0,1]
    heights = (heights - heights.min()) / (heights.max() - heights.min())

    # 缩放到所需高度范围
    heights = heights * height_range

    if z_invert:
        # 如果Z轴颠倒，高度值大表示低海拔
        heights = height_range - heights

    # 创建点云
    positions = []
    heights_normalized = []

    for i in range(grid_size):
        for j in range(grid_size):
            x_val = xx[i, j]
            y_val = yy[i, j]
            z_val = heights[i, j]

            # 为边缘创建对称缺口
            # 左侧缺口
            left_notch = (x_val < size/6) and (y_val > size/3) and (y_val < size*2/3)
            # 右侧缺口
            right_notch = (x_val > size*5/6) and (y_val > size/3) and (y_val < size*2/3)

            if not (left_notch or right_notch):
                positions.append([x_val, y_val, z_val])
                heights_normalized.append((heights[i, j] - heights.min()) / (heights.max() - heights.min()))

    return np.array(positions), np.array(heights_normalized)

def create_buildings(terrain_positions, building_count=10, min_height=10.0, max_height=15.0,
                     min_size=2.0, max_size=4.0, z_invert=True, seed=42):
    """
    在地形上创建简单的矩形建筑物

    Args:
        terrain_positions: 地形点的位置坐标 (N, 3)
        building_count: 建筑物数量
        min_height, max_height: 建筑物高度范围
        min_size, max_size: 建筑物尺寸范围
        z_invert: Z轴是否颠倒(True表示地面Z值大)
        seed: 随机数种子

    Returns:
        building_positions: 建筑物点的位置坐标 (M, 3)
    """
    np.random.seed(seed)

    # 获取地形范围
    x_min, y_min = np.min(terrain_positions[:, 0]), np.min(terrain_positions[:, 1])
    x_max, y_max = np.max(terrain_positions[:, 0]), np.max(terrain_positions[:, 1])

    # 缩小一点，避免建筑物超出边界
    x_range = (x_min + 2.0, x_max - 2.0)
    y_range = (y_min + 2.0, y_max - 2.0)

    building_positions = []

    for _ in range(building_count):
        # 随机选择建筑物中心位置
        center_x = np.random.uniform(x_range[0], x_range[1])
        center_y = np.random.uniform(y_range[0], y_range[1])

        # 建筑物尺寸
        building_size_x = np.random.uniform(min_size, max_size)
        building_size_y = np.random.uniform(min_size, max_size)

        # 建筑物高度
        building_height = np.random.uniform(min_height, max_height)

        # 找到最近的地形点，获取地面高度
        dists = np.sqrt((terrain_positions[:, 0] - center_x)**2 +
                        (terrain_positions[:, 1] - center_y)**2)
        nearest_idx = np.argmin(dists)
        ground_z = terrain_positions[nearest_idx, 2]

        # 建筑物顶部Z坐标
        if z_invert:
            # Z轴颠倒，建筑物的Z值比地面小
            roof_z = ground_z - building_height
        else:
            # 正常Z轴，建筑物的Z值比地面大
            roof_z = ground_z + building_height

        # 创建建筑物顶面点
        density = 5  # 每边的点数
        for i in range(density):
            for j in range(density):
                rx = (i / (density-1) - 0.5) * building_size_x
                ry = (j / (density-1) - 0.5) * building_size_y

                x = center_x + rx
                y = center_y + ry
                z = roof_z

                building_positions.append([x, y, z])

        # 添加更多墙面点
        sides = 10  # 每边的点数
        wall_points = 10  # 墙面高度方向的点数

        for i in range(sides):
            # 插值因子
            t = i / (sides-1)

            # 四条边
            edges = [
                [center_x - building_size_x/2, center_y - building_size_y/2 + t*building_size_y],
                [center_x + building_size_x/2, center_y - building_size_y/2 + t*building_size_y],
                [center_x - building_size_x/2 + t*building_size_x, center_y - building_size_y/2],
                [center_x - building_size_x/2 + t*building_size_x, center_y + building_size_y/2]
            ]

            for edge_x, edge_y in edges:
                # 找到最近的地形点，获取地面高度
                dists = np.sqrt((terrain_positions[:, 0] - edge_x)**2 +
                               (terrain_positions[:, 1] - edge_y)**2)
                nearest_idx = np.argmin(dists)
                local_ground_z = terrain_positions[nearest_idx, 2]

                # 在墙边添加更多点，使z方向更密集
                for k in range(wall_points):
                    w_t = (k+1) / (wall_points+1)  # 避免完全在地面或屋顶

                    if z_invert:
                        wall_z = local_ground_z - building_height * w_t
                    else:
                        wall_z = local_ground_z + building_height * w_t

                    building_positions.append([edge_x, edge_y, wall_z])

    return np.array(building_positions)

def create_vegetation(terrain_positions, count=100, min_height=2.0, max_height=4.0,
                      min_radius=0.3, max_radius=0.8, z_invert=True, seed=42):
    """
    在地形上创建简单的球形植被

    Args:
        terrain_positions: 地形点的位置坐标 (N, 3)
        count: 植被数量
        min_height, max_height: 植被高度范围
        min_radius, max_radius: 植被半径范围
        z_invert: Z轴是否颠倒(True表示地面Z值大)
        seed: 随机数种子

    Returns:
        vegetation_positions: 植被点的位置坐标 (M, 3)
    """
    np.random.seed(seed+100)  # 不同的种子

    # 获取地形范围
    x_min, y_min = np.min(terrain_positions[:, 0]), np.min(terrain_positions[:, 1])
    x_max, y_max = np.max(terrain_positions[:, 0]), np.max(terrain_positions[:, 1])

    # 缩小一点，避免植被超出边界
    x_range = (x_min + 2.0, x_max - 2.0)
    y_range = (y_min + 2.0, y_max - 2.0)

    vegetation_positions = []

    for _ in range(count):
        # 随机选择植被中心位置
        center_x = np.random.uniform(x_range[0], x_range[1])
        center_y = np.random.uniform(y_range[0], y_range[1])

        # 找到最近的地形点，获取地面高度
        dists = np.sqrt((terrain_positions[:, 0] - center_x)**2 +
                       (terrain_positions[:, 1] - center_y)**2)
        nearest_idx = np.argmin(dists)
        ground_z = terrain_positions[nearest_idx, 2]

        # 植被高度和半径
        height = np.random.uniform(min_height, max_height)
        radius = np.random.uniform(min_radius, max_radius)

        # 植被中心Z坐标
        if z_invert:
            center_z = ground_z - height/2
        else:
            center_z = ground_z + height/2

        # 创建球形植被点云
        n_points = int(20 * radius**2)  # 点数与体积成正比

        for _ in range(n_points):
            # 均匀分布在球体内
            while True:
                # 在半径为1的球内均匀采样
                x_offset = np.random.uniform(-1, 1)
                y_offset = np.random.uniform(-1, 1)
                z_offset = np.random.uniform(-1, 1)

                # 检查是否在球内
                if x_offset**2 + y_offset**2 + z_offset**2 <= 1:
                    break

            # 调整为椭球体形状(稍微扁平一些)
            x = center_x + x_offset * radius
            y = center_y + y_offset * radius
            z = center_z + z_offset * height/2

            vegetation_positions.append([x, y, z])

    return np.array(vegetation_positions)

def create_gaussian_attributes(positions, seed=42):
    """
    为点云创建高斯属性

    Args:
        positions: 点的位置坐标 (N, 3)
        seed: 随机数种子

    Returns:
        属性字典，包含means、quats、scales、opacities、sh0
    """
    np.random.seed(seed)
    n_points = len(positions)

    # 位置(means)已经给定
    means = positions

    # 四元数(quats)，表示旋转
    # 默认方向为 [1,0,0,0]（无旋转）
    # 对地形点，我们保持默认方向
    # 对非地形点，添加一些随机旋转
    quats = np.zeros((n_points, 4))
    quats[:, 0] = 1.0  # w分量为1，表示无旋转

    # 随机旋转(仅用于建筑和植被)，使用scipy的Rotation
    r = R.random(n_points, random_state=seed)
    random_quats = r.as_quat()  # 返回(x,y,z,w)格式

    # 转换为(w,x,y,z)格式
    random_quats_wxyz = np.zeros_like(random_quats)
    random_quats_wxyz[:, 0] = random_quats[:, 3]  # w
    random_quats_wxyz[:, 1:] = random_quats[:, :3]  # x,y,z

    # 对非地形点应用随机旋转（这里简单地假设高于一定高度的点是非地形）
    # 由于我们在后面会将点分为地形、建筑和植被，这里的逻辑可以调整
    quats = random_quats_wxyz

    # 尺度(scales)，采用对数尺度
    scales_min = np.log(0.05)  # 约0.05米
    scales_max = np.log(0.2)   # 约0.2米

    # 为地形点创建较大尺度，为非地形点创建较小尺度
    scales = np.random.uniform(scales_min, scales_max, (n_points, 3))

    # 不透明度(opacities)，对数尺度
    # 1.0表示完全不透明，0.0表示完全透明
    opacities = np.ones((n_points, 1)) * 10.0  # 10.0表示在对数尺度上完全不透明

    # 颜色(sh0)，基础球谐函数
    # 地形为绿色，建筑为蓝色，植被为红色
    sh0 = np.zeros((n_points, 3, 1))

    # 这里先创建随机颜色，后面会根据点的类型覆盖
    sh0[:, 0, 0] = np.random.uniform(0.3, 0.7, n_points)  # R
    sh0[:, 1, 0] = np.random.uniform(0.3, 0.7, n_points)  # G
    sh0[:, 2, 0] = np.random.uniform(0.3, 0.7, n_points)  # B

    # 创建一些随机的高阶球谐系数(shN)
    degree = 3  # 3阶球谐，共9个系数(不含0阶)
    coefficients = degree**2 - 1  # 9
    shN = np.zeros((n_points, 3, coefficients))

    # 为每个颜色通道生成随机球谐系数
    for i in range(3):  # R, G, B
        shN[:, i] = np.random.normal(0, 0.01, (n_points, coefficients))

    return {
        "means": torch.tensor(means, dtype=torch.float32),
        "quats": torch.tensor(quats, dtype=torch.float32),
        "scales": torch.tensor(scales, dtype=torch.float32),
        "opacities": torch.tensor(opacities, dtype=torch.float32),
        "sh0": torch.tensor(sh0, dtype=torch.float32),
        "shN": torch.tensor(shN, dtype=torch.float32)
    }

def assign_colors(attributes, terrain_indices, building_indices, vegetation_indices):
    """
    为不同类型的点分配颜色

    Args:
        attributes: 高斯点属性字典
        terrain_indices: 地形点索引
        building_indices: 建筑点索引
        vegetation_indices: 植被点索引

    Returns:
        更新后的属性字典
    """
    sh0 = attributes["sh0"].numpy()

    # 地形为绿色
    if len(terrain_indices) > 0:
        sh0[terrain_indices, 0, 0] = 0.2  # R
        sh0[terrain_indices, 1, 0] = 0.8  # G
        sh0[terrain_indices, 2, 0] = 0.2  # B

    # 建筑为蓝色
    if len(building_indices) > 0:
        sh0[building_indices, 0, 0] = 0.2  # R
        sh0[building_indices, 1, 0] = 0.2  # G
        sh0[building_indices, 2, 0] = 0.8  # B

    # 植被为红色
    if len(vegetation_indices) > 0:
        sh0[vegetation_indices, 0, 0] = 0.8  # R
        sh0[vegetation_indices, 1, 0] = 0.2  # G
        sh0[vegetation_indices, 2, 0] = 0.2  # B

    attributes["sh0"] = torch.tensor(sh0, dtype=torch.float32)
    return attributes

def save_metadata(output_file, terrain_count, building_count, vegetation_count,
                 x_min, x_max, y_min, y_max, z_min, z_max,
                 observer_center):
    """保存元数据信息到文本文件"""

    with open(output_file, 'w') as f:
        f.write("# 3D高斯点云示例数据元数据\n\n")

        f.write("## 1. 点数统计\n")
        total_count = terrain_count + building_count + vegetation_count
        f.write(f"总点数: {total_count:,}\n")
        f.write(f"地形点: {terrain_count:,} ({terrain_count/total_count*100:.1f}%)\n")
        f.write(f"建筑点: {building_count:,} ({building_count/total_count*100:.1f}%)\n")
        f.write(f"植被点: {vegetation_count:,} ({vegetation_count/total_count*100:.1f}%)\n\n")

        f.write("## 2. 空间范围\n")
        f.write(f"X范围: [{x_min:.2f}, {x_max:.2f}] 米\n")
        f.write(f"Y范围: [{y_min:.2f}, {y_max:.2f}] 米\n")
        f.write(f"Z范围: [{z_min:.2f}, {z_max:.2f}] 米\n")
        f.write(f"XY尺寸: {x_max-x_min:.2f} × {y_max-y_min:.2f} 米\n")
        f.write(f"Z起伏: {abs(z_max-z_min):.2f} 米\n\n")

        f.write("## 3. 坐标系设置\n")
        f.write("Z轴方向: 颠倒 (地面Z值大，向下增长)\n")
        f.write(f"观察中心: ({observer_center[0]:.1f}, {observer_center[1]:.1f}, {observer_center[2]:.1f})\n\n")

        f.write("## 4. 文件格式\n")
        f.write("格式: PyTorch .pt\n")
        f.write("包含字段: means, quats, scales, opacities, sh0, shN\n")
        f.write("兼容性: 使用旧版序列化格式 (_use_new_zipfile_serialization=False)\n\n")

        f.write("## 5. 特殊设计特征\n")
        f.write("- 左右两侧有对称缺口，用于演示边缘扩展算法\n")
        f.write("- 建筑屋顶平坦，便于算法通过坡度区分地形与建筑\n")
        f.write("- 地形有起伏变化，植被分散分布，用于测试克里金插值效果\n")

def create_demo_pt(output_pt="demo.pt", metadata_file="demo_metadata.txt",
                  terrain_size=30.0, terrain_resolution=0.2,
                  terrain_height=10.0, building_count=10,
                  vegetation_count=100, z_invert=True, seed=42,
                  building_height=10.0, building_size=3.0):
    """
    创建演示用的3D高斯点云数据文件

    Args:
        output_pt: 输出的PT文件路径
        metadata_file: 元数据文件路径
        terrain_size: 地形大小(米)
        terrain_resolution: 地形分辨率(米)
        terrain_height: 地形高度范围(米)
        building_count: 建筑物数量
        vegetation_count: 植被数量
        z_invert: Z轴是否颠倒(True表示地面Z值大)
        seed: 随机数种子
        building_height: 建筑物高度(米)
        building_size: 建筑物尺寸(米)
    """
    print(f"开始生成演示数据: {output_pt}")
    start_time = time.time()

    # 1. 创建地形
    print("创建起伏地形...")
    terrain_positions, terrain_heights = create_terrain(
        size=terrain_size,
        resolution=terrain_resolution,
        height_range=terrain_height,
        z_invert=z_invert,
        seed=seed
    )
    print(f"地形点数: {len(terrain_positions):,}")

    # 2. 创建建筑物
    print("创建建筑物...")
    building_positions = create_buildings(
        terrain_positions,
        building_count=building_count,
        min_height=building_height,
        max_height=building_height,
        min_size=building_size,
        max_size=building_size,
        z_invert=z_invert,
        seed=seed
    )
    print(f"建筑物点数: {len(building_positions):,}")

    # 3. 创建植被
    print("创建植被...")
    vegetation_positions = create_vegetation(
        terrain_positions,
        count=vegetation_count,
        z_invert=z_invert,
        seed=seed
    )
    print(f"植被点数: {len(vegetation_positions):,}")

    # 4. 合并所有点
    print("合并点云并生成高斯属性...")
    terrain_count = len(terrain_positions)
    building_count = len(building_positions)
    vegetation_count = len(vegetation_positions)

    print(f"  - 地形点数: {terrain_count}")
    print(f"  - 建筑点数: {building_count}")
    print(f"  - 植被点数: {vegetation_count}")

    # 记录不同类型点的索引范围
    terrain_indices = list(range(0, terrain_count))
    building_indices = list(range(terrain_count, terrain_count + building_count))
    vegetation_indices = list(range(terrain_count + building_count,
                                   terrain_count + building_count + vegetation_count))

    # 合并点云
    # 修复植被数量为0时的问题
    if vegetation_count > 0:
        print("  合并所有三种点...")
        all_positions = np.vstack([terrain_positions, building_positions, vegetation_positions])
    else:
        print("  植被点数为0，只合并地形和建筑点...")
        if building_count > 0:
            all_positions = np.vstack([terrain_positions, building_positions])
        else:
            all_positions = terrain_positions

    print(f"  合并后总点数: {len(all_positions)}")

    # 5. 创建高斯属性
    attributes = create_gaussian_attributes(all_positions, seed=seed)

    # 6. 为不同类型点分配颜色
    attributes = assign_colors(attributes, terrain_indices, building_indices, vegetation_indices)

    # 7. 保存为PT文件
    splats = attributes
    ckpt = {"splats": splats}

    print(f"保存文件: {output_pt}")
    torch.save(ckpt, output_pt, _use_new_zipfile_serialization=False)

    # 获取坐标范围
    x_min, x_max = float(all_positions[:, 0].min()), float(all_positions[:, 0].max())
    y_min, y_max = float(all_positions[:, 1].min()), float(all_positions[:, 1].max())
    z_min, z_max = float(all_positions[:, 2].min()), float(all_positions[:, 2].max())

    # 8. 保存元数据
    observer_center = (0.5, -0.2, 0.0)  # 默认观察中心
    print(f"保存元数据: {metadata_file}")
    save_metadata(
        metadata_file,
        terrain_count, building_count, vegetation_count,
        x_min, x_max, y_min, y_max, z_min, z_max,
        observer_center
    )

    elapsed = time.time() - start_time
    print(f"数据生成完成，耗时: {elapsed:.2f} 秒")
    print(f"总点数: {len(all_positions):,}")
    print(f"空间范围: X[{x_min:.2f}, {x_max:.2f}], Y[{y_min:.2f}, {y_max:.2f}], Z[{z_min:.2f}, {z_max:.2f}]")

def create_preview_script(output_file="view_demo.py"):
    """创建3D预览脚本"""
    script_content = """#!/usr/bin/env python
\"\"\"
3D高斯点云预览脚本
使用Plotly绘制交互式3D散点图
\"\"\"

import torch
import numpy as np
import plotly.graph_objects as go
import argparse

def load_gaussian_model(model_path):
    \"\"\"加载高斯模型并返回点云坐标\"\"\"
    print(f"加载模型: {model_path}")
    ckpt = torch.load(model_path, map_location='cpu')
    splats = ckpt["splats"]
    points = splats["means"].numpy()
    print(f"加载完成，共 {len(points):,} 个点")
    return points, splats

def visualize_point_cloud(points, sh0, max_points=10000, point_size=2):
    \"\"\"
    可视化点云

    Args:
        points: 点云坐标 (N, 3)
        sh0: 颜色 (N, 3, 1)
        max_points: 最大可视化点数
        point_size: 点大小
    \"\"\"
    # 如果点数超过最大值，进行下采样
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        vis_points = points[indices]
        vis_colors = sh0[indices]
    else:
        vis_points = points
        vis_colors = sh0

    # 将颜色从(N, 3, 1)转换为(N, 3)
    colors = vis_colors.reshape(-1, 3)

    # 创建RGB颜色字符串
    color_strings = [f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'
                    for r, g, b in colors]

    # 创建3D散点图
    fig = go.Figure(data=[go.Scatter3d(
        x=vis_points[:, 0],
        y=vis_points[:, 1],
        z=vis_points[:, 2],
        mode='markers',
        marker=dict(
            size=point_size,
            color=color_strings,
            opacity=0.8
        )
    )])

    # 配置布局
    fig.update_layout(
        title="3D高斯点云预览",
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

    # 显示图形
    fig.show()

def main():
    parser = argparse.ArgumentParser(description='3D高斯点云预览工具')
    parser.add_argument('-i', '--input', default='demo.pt', help='输入PT文件路径')
    parser.add_argument('-p', '--points', type=int, default=20000, help='最大显示点数')
    parser.add_argument('-s', '--size', type=int, default=2, help='点大小')
    args = parser.parse_args()

    # 加载模型
    points, splats = load_gaussian_model(args.input)

    # 提取颜色
    sh0 = splats["sh0"].numpy()

    # 可视化
    visualize_point_cloud(points, sh0, max_points=args.points, point_size=args.size)

if __name__ == "__main__":
    main()
"""

    with open(output_file, 'w') as f:
        f.write(script_content)

    print(f"已创建预览脚本: {output_file}")
    # 设置可执行权限
    try:
        import os
        os.chmod(output_file, 0o755)
    except:
        pass

def main():
    parser = argparse.ArgumentParser(description='生成3D高斯点云示例数据')
    parser.add_argument('-o', '--output', default='demo.pt', help='输出PT文件路径')
    parser.add_argument('-m', '--metadata', default='demo_metadata.txt', help='输出元数据文件路径')
    parser.add_argument('-v', '--view', action='store_true', help='创建可视化预览脚本')
    parser.add_argument('-s', '--size', type=float, default=30.0, help='地形大小(米)')
    parser.add_argument('-r', '--resolution', type=float, default=0.2, help='地形分辨率(米)')
    parser.add_argument('-z', '--height', type=float, default=10.0, help='地形高度范围(米)')
    parser.add_argument('-b', '--buildings', type=int, default=10, help='建筑物数量')
    parser.add_argument('-t', '--trees', type=int, default=100, help='植被数量')
    parser.add_argument('--seed', type=int, default=42, help='随机数种子')
    parser.add_argument('--debug', action='store_true', help='显示详细调试信息')
    parser.add_argument('--building-height', type=float, default=10.0, help='建筑物高度(米)，默认为10.0米')
    parser.add_argument('--building-size', type=float, default=3.0, help='建筑物尺寸(米)，默认为3.0米')
    args = parser.parse_args()

    # 设置是否输出详细信息
    verbose = args.debug
    if verbose:
        print("\n===== 调试模式 =====")
        print(f"输出文件: {args.output}")
        print(f"元数据文件: {args.metadata}")
        print(f"地形大小: {args.size}米, 分辨率: {args.resolution}米")
        print(f"建筑数量: {args.buildings}, 植被数量: {args.trees}")
        print(f"建筑高度: {args.building_height}米, 建筑尺寸: {args.building_size}米")

    try:
        # 创建示例数据
        create_demo_pt(
            output_pt=args.output,
            metadata_file=args.metadata,
            terrain_size=args.size,
            terrain_resolution=args.resolution,
            terrain_height=args.height,
            building_count=args.buildings,
            vegetation_count=args.trees,
            z_invert=True,  # 固定Z轴颠倒
            seed=args.seed,
            building_height=args.building_height,
            building_size=args.building_size
        )

        # 创建预览脚本
        if args.view:
            create_preview_script("view_demo.py")

        if verbose:
            print(f"\n✅ 数据生成成功: {args.output}")
            # 检查文件是否存在
            import os
            if os.path.exists(args.output):
                file_size = os.path.getsize(args.output) / 1024.0  # KB
                print(f"文件大小: {file_size:.1f} KB")
            else:
                print(f"警告: 文件 {args.output} 未成功创建!")

    except Exception as e:
        print(f"\n❌ 错误: 生成数据时出错: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()