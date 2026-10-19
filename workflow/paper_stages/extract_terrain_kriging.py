#!/usr/bin/env python
"""
3D高斯点云地形重建 - 克里金插值方法

使用克里金插值(Kriging)方法对地形点云进行重建和插值
相比简单网格法，克里金插值考虑了空间相关性，可提供更精确的地形表面
支持各向异性和不同变异函数模型
"""

import numpy as np
import torch
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from scipy.spatial import Delaunay
import time
from pykrige.ok import OrdinaryKriging
import warnings
warnings.filterwarnings("ignore")

def load_model(model_path):
    """加载模型"""
    print(f"加载模型: {model_path}")
    ckpt = torch.load(model_path, map_location='cpu')
    splats = ckpt['splats']
    points = splats['means'].detach().cpu().numpy()
    return points, splats

def downsample_points(points, grid_size=0.2, z_inverted=True):
    """
    使用网格下采样减少点数
    对克里金插值，大量点会导致计算缓慢，所以需要先下采样
    """
    print(f"下采样点云 (网格大小: {grid_size})...")

    # 计算网格索引
    grid_indices = np.floor(points[:, :3] / grid_size).astype(int)

    # 创建唯一网格标识
    grid_ids = grid_indices[:, 0] + grid_indices[:, 1] * 1000 + grid_indices[:, 2] * 1000000

    # 获取唯一网格及其点
    unique_grid_ids = np.unique(grid_ids)

    # 为每个唯一网格选择一个代表点
    downsampled_points = []
    for grid_id in tqdm(unique_grid_ids, desc="下采样"):
        # 获取当前网格中的所有点
        grid_point_indices = np.where(grid_ids == grid_id)[0]

        # 根据Z轴方向选择代表点
        if z_inverted:
            # Z轴颠倒：选择Z值最大的点（可能是地面）
            rep_idx = grid_point_indices[np.argmax(points[grid_point_indices, 2])]
        else:
            # 正常Z轴：选择Z值最小的点（可能是地面）
            rep_idx = grid_point_indices[np.argmin(points[grid_point_indices, 2])]

        downsampled_points.append(points[rep_idx])

    downsampled_points = np.array(downsampled_points)
    print(f"下采样后点数: {len(downsampled_points):,} ({len(downsampled_points)/len(points)*100:.1f}%)")

    return downsampled_points

def extract_terrain_kriging(points, grid_size=0.2, subsample_size=5000, variogram_model='spherical'):
    """
    使用克里金插值提取地形平面

    Args:
        points: 点云坐标
        grid_size: 输出网格大小
        subsample_size: 克里金计算用的子采样大小（全量计算太慢）
        variogram_model: 变异函数模型，可选 'linear', 'power', 'gaussian', 'spherical', 'exponential'

    Returns:
        插值后的地形点云
    """
    print(f"使用克里金插值提取地形平面 (网格大小: {grid_size})...")

    # 计算点云范围
    x_min, y_min = points[:, 0].min(), points[:, 1].min()
    x_max, y_max = points[:, 0].max(), points[:, 1].max()

    # 创建网格
    x_grid = np.arange(x_min, x_max + grid_size, grid_size)
    y_grid = np.arange(y_min, y_max + grid_size, grid_size)

    print(f"网格范围: X({x_min:.2f}-{x_max:.2f}), Y({y_min:.2f}-{y_max:.2f})")
    print(f"网格大小: {len(x_grid)}x{len(y_grid)} = {len(x_grid)*len(y_grid):,} 个点")

    # 对原始点进行子采样，提高克里金计算效率
    print("对点云进行子采样以加速克里金计算...")
    if len(points) > subsample_size:
        # 随机采样
        indices = np.random.choice(len(points), subsample_size, replace=False)
        sample_points = points[indices]
        print(f"子采样: {len(points):,} → {len(sample_points):,} 点")
    else:
        sample_points = points

    # 准备克里金插值的数据
    x = sample_points[:, 0]
    y = sample_points[:, 1]
    z = sample_points[:, 2]

    # 执行克里金插值
    print(f"执行克里金插值 (变异函数模型: {variogram_model})...")
    start_time = time.time()

    try:
        # 创建克里金模型
        print("构建克里金模型...")
        ok = OrdinaryKriging(
            x, y, z,
            variogram_model=variogram_model,
            verbose=False,
            enable_plotting=False
        )

        # 执行插值
        print("执行克里金插值计算（可能需要一些时间）...")
        z_kriged, _ = ok.execute('grid', x_grid, y_grid)

        elapsed = time.time() - start_time
        print(f"克里金插值完成，耗时: {elapsed:.1f}秒")

        # 创建插值后的点云
        print("创建插值后的点云...")
        kriged_points = []

        for i in range(len(y_grid)):
            for j in range(len(x_grid)):
                if not np.isnan(z_kriged[i, j]):
                    x_val = x_grid[j]
                    y_val = y_grid[i]
                    z_val = z_kriged[i, j]
                    kriged_points.append([x_val, y_val, z_val])

        kriged_points = np.array(kriged_points)
        print(f"克里金插值生成点数: {len(kriged_points):,}")

    except Exception as e:
        print(f"克里金插值出错: {str(e)}")
        print("尝试退回到简单平面拟合...")

        # 如果克里金失败，退回到简单平面拟合
        A = np.column_stack([sample_points[:, 0], sample_points[:, 1], np.ones(len(sample_points))])
        b = sample_points[:, 2]
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # 创建简单平面
        kriged_points = []
        for x_val in x_grid:
            for y_val in y_grid:
                z_val = coeffs[0] * x_val + coeffs[1] * y_val + coeffs[2]
                kriged_points.append([x_val, y_val, z_val])

        kriged_points = np.array(kriged_points)
        print(f"简单平面拟合生成点数: {len(kriged_points):,}")

    return kriged_points

def save_model(plane_points, original_splats, output_path):
    """保存模型"""
    print(f"保存模型: {output_path}")

    # 创建新的splats字典
    new_splats = {}

    # 转换点坐标
    means = torch.tensor(plane_points, dtype=torch.float32)
    new_splats['means'] = means

    # 复制其他必要字段
    n_points = len(plane_points)
    device = original_splats['means'].device

    # 四元数
    if 'quats' in original_splats:
        quats = torch.zeros((n_points, 4), device=device)
        quats[:, 0] = 1.0  # 默认为不旋转
        new_splats['quats'] = quats

    # 尺度
    if 'scales' in original_splats:
        log_scale = torch.log(torch.tensor([0.1], device=device))
        scales = torch.ones((n_points, 3), device=device) * log_scale
        new_splats['scales'] = scales

    # 不透明度
    if 'opacities' in original_splats:
        opacities = torch.ones((n_points, 1), device=device) * 10  # 完全不透明
        new_splats['opacities'] = opacities

    # 颜色 (使用灰色)
    if 'sh0' in original_splats:
        sh0 = torch.ones((n_points, 3, 1), device=device) * 0.5
        new_splats['sh0'] = sh0

    # 高阶球谐系数
    if 'shN' in original_splats and original_splats['shN'] is not None:
        sh_dim = original_splats['shN'].shape[2]
        shN = torch.zeros((n_points, 3, sh_dim), device=device)
        new_splats['shN'] = shN

    # 保存
    ckpt = {'splats': new_splats}
    torch.save(ckpt, output_path)
    print(f"模型已保存")

def visualize_terrain(points, kriged_points, output_path):
    """可视化地形"""
    print(f"生成可视化...")

    # 创建3D图形
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制原始点云（蓝色，透明）
    if len(points) > 5000:
        # 随机采样以加快可视化
        indices = np.random.choice(len(points), 5000, replace=False)
        vis_points = points[indices]
    else:
        vis_points = points

    ax.scatter(vis_points[:, 0], vis_points[:, 1], vis_points[:, 2],
               s=1, c='blue', alpha=0.3, label='原始地形点')

    # 绘制克里金插值点云（红色）
    if len(kriged_points) > 5000:
        # 随机采样以加快可视化
        indices = np.random.choice(len(kriged_points), 5000, replace=False)
        vis_kriged = kriged_points[indices]
    else:
        vis_kriged = kriged_points

    ax.scatter(vis_kriged[:, 0], vis_kriged[:, 1], vis_kriged[:, 2],
               s=1, c='red', alpha=0.7, label='克里金插值')

    # 设置标题和轴标签
    ax.set_title("克里金插值地形重建")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # 保存图像
    plt.savefig(output_path)
    print(f"可视化已保存: {output_path}")

def visualize_3d_interactive(points, kriged_points, output_path):
    """使用Plotly生成交互式3D可视化"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        print("生成交互式3D可视化...")

        # 如果点数太多，进行下采样
        if len(points) > 5000:
            indices = np.random.choice(len(points), 5000, replace=False)
            vis_points = points[indices]
        else:
            vis_points = points

        if len(kriged_points) > 5000:
            indices = np.random.choice(len(kriged_points), 5000, replace=False)
            vis_kriged = kriged_points[indices]
        else:
            vis_kriged = kriged_points

        # 创建图表
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])

        # 添加原始点云
        fig.add_trace(
            go.Scatter3d(
                x=vis_points[:, 0],
                y=vis_points[:, 1],
                z=vis_points[:, 2],
                mode='markers',
                name='原始地形点',
                marker=dict(
                    size=2,
                    color='blue',
                    opacity=0.5
                )
            )
        )

        # 添加克里金插值点云
        fig.add_trace(
            go.Scatter3d(
                x=vis_kriged[:, 0],
                y=vis_kriged[:, 1],
                z=vis_kriged[:, 2],
                mode='markers',
                name='克里金插值',
                marker=dict(
                    size=2,
                    color='red',
                    opacity=0.8
                )
            )
        )

        # 更新布局
        fig.update_layout(
            title="克里金插值地形重建 (交互式)",
            width=1000,
            height=800,
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode='data'
            )
        )

        # 保存为HTML
        fig.write_html(output_path)
        print(f"交互式可视化已保存: {output_path}")
    except ImportError:
        print("生成交互式可视化需要安装plotly。请运行 'pip install plotly'")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='克里金插值地形重建工具')
    parser.add_argument('-i', '--input', required=True, help='输入模型路径')
    parser.add_argument('-o', '--output', help='输出模型路径')
    parser.add_argument('-g', '--grid-size', type=float, default=0.2, help='网格大小')
    parser.add_argument('-d', '--downsample', type=float, default=0.5, help='下采样网格大小')
    parser.add_argument('-s', '--subsample', type=int, default=5000, help='克里金计算子采样大小')
    parser.add_argument('-m', '--variogram-model', default='spherical',
                      choices=['linear', 'power', 'gaussian', 'spherical', 'exponential'],
                      help='变异函数模型')
    parser.add_argument('-v', '--visualize', action='store_true', help='生成可视化')
    parser.add_argument('-p', '--plotly', action='store_true', help='生成交互式可视化（需要plotly）')
    parser.add_argument('--z-normal', action='store_true', help='Z轴为正常方向（地面Z值小，默认假设Z轴颠倒）')
    args = parser.parse_args()

    # 设置输出路径
    if not args.output:
        input_dir = os.path.dirname(args.input)
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(input_dir, f"{base_name}_kriging.pt")

    # 可视化输出路径
    vis_path = os.path.splitext(args.output)[0] + ".png"
    interactive_vis_path = os.path.splitext(args.output)[0] + ".html"

    # 加载模型
    points, splats = load_model(args.input)

    # 下采样（可选）
    if args.downsample > 0:
        points = downsample_points(points, grid_size=args.downsample, z_inverted=not args.z_normal)

    # 提取地形平面
    kriged_points = extract_terrain_kriging(
        points,
        grid_size=args.grid_size,
        subsample_size=args.subsample,
        variogram_model=args.variogram_model
    )

    # 保存模型
    save_model(kriged_points, splats, args.output)

    # 可视化
    if args.visualize:
        visualize_terrain(points, kriged_points, vis_path)

    # 交互式可视化
    if args.plotly:
        visualize_3d_interactive(points, kriged_points, interactive_vis_path)

    print("处理完成")
    print(f"输入模型: {args.input}")
    print(f"输出模型: {args.output}")
    print(f"原始点数: {len(points):,}")
    print(f"插值后点数: {len(kriged_points):,}")

if __name__ == "__main__":
    main()