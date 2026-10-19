#!/usr/bin/env python
"""
简化版地形提取后处理工具
用于将孤立的非地形点重新归类为地形点
"""

import torch
import numpy as np
import argparse
import time
import os
from sklearn.neighbors import NearestNeighbors

def main():
    parser = argparse.ArgumentParser(description='简化版地形提取后处理工具')
    parser.add_argument('--terrain', '-t', type=str, required=True, help='地形模型路径')
    parser.add_argument('--objects', '-o', type=str, required=True, help='非地形模型路径')
    parser.add_argument('--output', '-p', type=str, default=None, help='输出文件前缀')
    parser.add_argument('--radius', '-r', type=float, default=1.5, help='孤立点检测半径')
    parser.add_argument('--neighbors', '-n', type=int, default=10, help='最小邻居数量')

    args = parser.parse_args()

    print("\n===== 简化版地形提取后处理工具 =====")

    # 设置输出前缀
    if args.output is None:
        args.output = os.path.splitext(args.terrain)[0].replace("_terrain_adaptive", "") + "_refined"

    # 加载地形和非地形模型
    print(f"加载地形模型: {args.terrain}")
    terrain_ckpt = torch.load(args.terrain, map_location='cpu')
    terrain_splats = terrain_ckpt["splats"]
    terrain_points = terrain_splats["means"].numpy()

    print(f"加载非地形模型: {args.objects}")
    objects_ckpt = torch.load(args.objects, map_location='cpu')
    objects_splats = objects_ckpt["splats"]
    objects_points = objects_splats["means"].numpy()

    print(f"地形点数: {len(terrain_points):,}")
    print(f"非地形点数: {len(objects_points):,}")

    # 构建邻域查找器
    print("构建KD树...")
    nn = NearestNeighbors(n_neighbors=min(50, len(objects_points)), algorithm='kd_tree').fit(objects_points)

    # 找出孤立的点
    print(f"查找孤立点 (半径: {args.radius}, 最小邻居数: {args.neighbors})...")

    # 为每个点计算邻域内的点数
    neighbors_within_radius = []
    batch_size = 1000  # 分批处理以减少内存使用

    for i in range(0, len(objects_points), batch_size):
        batch_end = min(i + batch_size, len(objects_points))
        batch = objects_points[i:batch_end]

        # 使用radius_neighbors查找指定半径内的邻居
        indices = nn.radius_neighbors(batch, radius=args.radius, return_distance=False)

        # 计算每个点的邻居数（不包括自己）
        neighbor_counts = np.array([len(idx) - 1 for idx in indices])  # -1是因为每个点自己也算作邻居
        neighbors_within_radius.extend(neighbor_counts)

        print(f"处理进度: {batch_end}/{len(objects_points)} ({batch_end/len(objects_points)*100:.1f}%)")

    neighbors_within_radius = np.array(neighbors_within_radius)

    # 找出孤立的点（邻居数少于阈值）
    isolated_mask = neighbors_within_radius < args.neighbors
    isolated_count = np.sum(isolated_mask)
    print(f"找到 {isolated_count} 个孤立点 ({isolated_count/len(objects_points)*100:.2f}%)")

    # 将非孤立点保留为非地形点
    keep_mask = ~isolated_mask
    new_objects_splats = {}
    for key in objects_splats:
        new_objects_splats[key] = objects_splats[key][keep_mask]

    # 将孤立点添加到地形点
    new_terrain_splats = {}
    for key in terrain_splats:
        new_terrain_splats[key] = torch.cat([terrain_splats[key], objects_splats[key][isolated_mask]], dim=0)

    # 保存结果
    terrain_output = args.output + "_terrain.pt"
    objects_output = args.output + "_objects.pt"

    print(f"保存优化后的地形模型: {terrain_output}")
    torch.save({"splats": new_terrain_splats}, terrain_output)

    print(f"保存优化后的非地形模型: {objects_output}")
    torch.save({"splats": new_objects_splats}, objects_output)

    # 输出统计信息
    print("\n===== 处理结果 =====")
    print(f"原始地形点数: {len(terrain_points):,}")
    print(f"原始非地形点数: {len(objects_points):,}")
    print(f"重分类点数: {isolated_count:,}")
    print(f"新地形点数: {len(new_terrain_splats['means']):,} ({len(new_terrain_splats['means'])/float(len(terrain_points) + len(objects_points))*100:.1f}%)")
    print(f"新非地形点数: {len(new_objects_splats['means']):,} ({len(new_objects_splats['means'])/float(len(terrain_points) + len(objects_points))*100:.1f}%)")

    print(f"\n处理完成! 优化后的文件已保存:")
    print(f"- 地形模型: {terrain_output}")
    print(f"- 非地形模型: {objects_output}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f} 秒")