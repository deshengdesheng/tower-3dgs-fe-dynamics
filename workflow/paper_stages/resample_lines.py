#!/usr/bin/env python3
# -------------------------------------------------
# 功能：
#   1) 读取points_with_displacement_merge.txt和lines_merge.txt文件
#   2) 按照指定的固定距离对线段进行重采样
#   3) 插值生成新的点，保持与所在线段相同的分组
#   4) 导出新的点和线段信息到points_with_displacement.txt和lines.txt文件
# -------------------------------------------------

import numpy as np
from pathlib import Path
import argparse
import os

def calculate_distance(p1, p2):
    """计算两点之间的欧氏距离"""
    return np.linalg.norm(p1 - p2)

def interpolate(p1, p2, t):
    """线性插值，t为[0,1]之间的值"""
    return p1 * (1 - t) + p2 * t

def resample_line_segments(input_points_file, input_lines_file, output_points_file, output_lines_file, spacing=0.01):
    """
    读取点和线段文件，按固定间距重采样线段，生成新的点云和线段文件

    参数：
    input_points_file -- 输入的点坐标文件
    input_lines_file -- 输入的线段文件
    output_points_file -- 输出的点坐标文件
    output_lines_file -- 输出的线段文件
    spacing -- 重采样间距，默认0.01
    """
    print(f"读取点文件: {input_points_file}")
    print(f"读取线段文件: {input_lines_file}")

    # 读取点坐标数据
    points_data = []
    with open(input_points_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 7:  # idx x y z ux uy uz
                idx = int(parts[0])
                coords = [float(p) for p in parts[1:4]]
                displacements = [float(d) for d in parts[4:7]]
                points_data.append((idx, coords, displacements))

    # 构建点索引到坐标和位移的映射
    point_idx_to_coords = {}
    point_idx_to_disp = {}
    for idx, coords, disps in points_data:
        point_idx_to_coords[idx] = coords
        point_idx_to_disp[idx] = disps

    print(f"读取了 {len(points_data)} 个点")

    # 读取线段数据
    lines_data = []
    with open(input_lines_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 4:  # line_idx p1_idx p2_idx merged_group
                line_idx = int(parts[0])
                p1_idx = int(parts[1])
                p2_idx = int(parts[2])
                group = int(parts[3])
                lines_data.append((line_idx, p1_idx, p2_idx, group))

    print(f"读取了 {len(lines_data)} 条线段")

    # 重采样所有线段
    new_points = []  # [(x, y, z, ux, uy, uz), ...]
    new_lines = []   # [(p1_idx, p2_idx, group), ...]

    point_idx_counter = 0
    point_idx_map = {}  # 原始点索引 -> 新点索引

    for line_idx, p1_idx, p2_idx, group in lines_data:
        if p1_idx not in point_idx_to_coords or p2_idx not in point_idx_to_coords:
            print(f"警告：线段 {line_idx} 的点 {p1_idx} 或 {p2_idx} 不存在，跳过")
            continue

        p1_coords = np.array(point_idx_to_coords[p1_idx])
        p2_coords = np.array(point_idx_to_coords[p2_idx])
        p1_disp = np.array(point_idx_to_disp[p1_idx])
        p2_disp = np.array(point_idx_to_disp[p2_idx])

        # 计算原始线段长度
        line_length = calculate_distance(p1_coords, p2_coords)

        # 计算需要多少个点
        num_segments = max(1, int(line_length / spacing))

        # 创建或映射起点
        if p1_idx not in point_idx_map:
            new_p1_idx = point_idx_counter
            point_idx_map[p1_idx] = new_p1_idx
            new_points.append((p1_coords[0], p1_coords[1], p1_coords[2],
                               p1_disp[0], p1_disp[1], p1_disp[2]))
            point_idx_counter += 1
        else:
            new_p1_idx = point_idx_map[p1_idx]

        # 根据间距创建新的点
        segment_points = [new_p1_idx]

        for i in range(1, num_segments):
            t = i / num_segments
            # 坐标插值
            new_coord = interpolate(p1_coords, p2_coords, t)
            # 位移插值
            new_disp = interpolate(p1_disp, p2_disp, t)

            # 添加新点
            new_idx = point_idx_counter
            segment_points.append(new_idx)
            new_points.append((new_coord[0], new_coord[1], new_coord[2],
                              new_disp[0], new_disp[1], new_disp[2]))
            point_idx_counter += 1

        # 创建或映射终点
        if p2_idx not in point_idx_map:
            new_p2_idx = point_idx_counter
            point_idx_map[p2_idx] = new_p2_idx
            new_points.append((p2_coords[0], p2_coords[1], p2_coords[2],
                               p2_disp[0], p2_disp[1], p2_disp[2]))
            point_idx_counter += 1
        else:
            new_p2_idx = point_idx_map[p2_idx]

        segment_points.append(new_p2_idx)

        # 创建新的线段
        for i in range(len(segment_points) - 1):
            new_lines.append((segment_points[i], segment_points[i+1], group))

    print(f"重采样后生成了 {len(new_points)} 个点")
    print(f"重采样后生成了 {len(new_lines)} 条线段")

    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(os.path.abspath(output_points_file)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_lines_file)), exist_ok=True)

    # 写入新的点文件
    with open(output_points_file, 'w') as f:
        f.write("# idx x y z ux uy uz\n")
        for idx, (x, y, z, ux, uy, uz) in enumerate(new_points):
            f.write(f"{idx} {x:.6f} {y:.6f} {z:.6f} {ux:.6f} {uy:.6f} {uz:.6f}\n")

    # 写入新的线段文件
    with open(output_lines_file, 'w') as f:
        f.write("# line_idx p1_idx p2_idx merged_group\n")
        for idx, (p1, p2, group) in enumerate(new_lines):
            f.write(f"{idx} {p1} {p2} {group}\n")

    print(f"✓ 已写入点与位移: {output_points_file}")
    print(f"✓ 已写入线段信息: {output_lines_file}")

    return len(new_points), len(new_lines)

def main():
    parser = argparse.ArgumentParser(description="按固定间距重采样线段")

    # 输入文件参数
    parser.add_argument('-i', '--input_dir', type=str, default='discrete_data',
                        help='输入文件目录')
    parser.add_argument('-ip', '--input_points', type=str, default='points_with_displacement_merge.txt',
                        help='输入的点坐标文件名')
    parser.add_argument('-il', '--input_lines', type=str, default='lines_merge.txt',
                        help='输入的线段文件名')

    # 输出文件参数
    parser.add_argument('-o', '--output_dir', type=str, default='discrete_data',
                        help='输出文件目录')
    parser.add_argument('-op', '--output_points', type=str, default='points_with_displacement.txt',
                        help='输出的点坐标文件名')
    parser.add_argument('-ol', '--output_lines', type=str, default='lines.txt',
                        help='输出的线段文件名')

    # 重采样参数
    parser.add_argument('-s', '--spacing', type=float, default=0.01,
                        help='重采样间距，默认0.01')

    # 解析命令行参数
    args = parser.parse_args()

    # 构建完整的文件路径
    input_points_path = os.path.join(args.input_dir, args.input_points)
    input_lines_path = os.path.join(args.input_dir, args.input_lines)
    output_points_path = os.path.join(args.output_dir, args.output_points)
    output_lines_path = os.path.join(args.output_dir, args.output_lines)

    # 执行重采样
    num_points, num_lines = resample_line_segments(
        input_points_path,
        input_lines_path,
        output_points_path,
        output_lines_path,
        args.spacing
    )

    print(f"总结：按间距 {args.spacing} 重采样完成")
    print(f"  - 输入: {input_points_path}, {input_lines_path}")
    print(f"  - 输出: {output_points_path}, {output_lines_path}")
    print(f"  - 结果: {num_points} 个点, {num_lines} 条线段")

if __name__ == "__main__":
    main()