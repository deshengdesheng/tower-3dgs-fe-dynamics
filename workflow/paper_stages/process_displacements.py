#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import argparse

def read_point_file(filename):
    """
    读取点云文件，文件格式为：
    # idx x y z ux uy uz
    """
    data = []

    if not os.path.exists(filename):
        print(f"错误: 文件不存在 {filename}")
        return data

    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            try:
                parts = line.split()
                if len(parts) >= 4:  # 至少需要idx, x, y, z
                    idx, x, y, z = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                    data.append((idx, x, y, z))
            except Exception as e:
                print(f"错误: 在 {filename} 第 {line_num} 行: {e}")
                print(f"行内容: {line}")

    print(f"从 {filename} 读取了 {len(data)} 个点.")
    return data

def process_points(input_file1, input_file2, output_file):
    """
    处理点云数据，计算位移向量
    """
    # 当前工作目录
    print(f"当前工作目录: {os.getcwd()}")

    # 检查文件路径
    for filepath in [input_file1, input_file2]:
        if not os.path.exists(filepath):
            print(f"错误: 文件不存在 {filepath}")
            return False

    # 读取两个点云文件
    print(f"读取文件: {input_file1}")
    points1 = read_point_file(input_file1)
    if not points1:
        return False

    print(f"读取文件: {input_file2}")
    points2 = read_point_file(input_file2)
    if not points2:
        return False

    def index_by_id(points, label):
        indexed = {}
        for point in points:
            point_id = point[0]
            if point_id in indexed:
                raise ValueError(f"{label} 中节点 ID 重复: {point_id}")
            indexed[point_id] = point
        return indexed

    points2_by_id = index_by_id(points2, "第二个输入文件")
    points1_by_id = index_by_id(points1, "第一个输入文件")
    ids1 = set(points1_by_id)
    ids2 = set(points2_by_id)
    if ids1 != ids2:
        missing = sorted(ids1 - ids2)
        extra = sorted(ids2 - ids1)
        raise ValueError(
            "两个输入文件的节点 ID 集不一致: "
            f"第一个文件有 {len(missing)} 个 ID 在第二个文件缺失 "
            f"(示例 {missing[:10]}), 第二个文件有 {len(extra)} 个额外 ID "
            f"(示例 {extra[:10]})"
        )

    print(f"计算位移向量...")
    result = []
    for i in range(len(points1)):
        if i % 1000 == 0:
            print(f"处理: {i}/{len(points1)}")

        idx1, x1, y1, z1 = points1[i]
        _, x2, y2, z2 = points2_by_id[idx1]

        # 计算位移向量 (ux, uy, uz) = (x2-x1, y2-y1, z2-z1)
        ux = x2 - x1
        uy = y2 - y1
        uz = z2 - z1

        # 存储结果: idx x y z ux uy uz
        result.append((idx1, x1, y1, z1, ux, uy, uz))

    # 写入结果到输出文件
    print(f"写入结果到: {output_file}")
    with open(output_file, 'w') as f:
        f.write("# idx x y z ux uy uz\n")
        for idx, x, y, z, ux, uy, uz in result:
            f.write(f"{idx} {x:.6f} {y:.6f} {z:.6f} {ux:.6f} {uy:.6f} {uz:.6f}\n")

    print(f"处理完成, 共处理 {len(result)} 个点.")

    # 验证输出文件
    if os.path.exists(output_file):
        print(f"输出文件创建成功，大小: {os.path.getsize(output_file)/1024:.2f} KB")
        return True
    else:
        print("错误: 输出文件未创建成功")
        return False

def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='计算点云之间的位移向量')
    parser.add_argument('-i1', '--input1', required=True, help='第一个输入文件路径')
    parser.add_argument('-i2', '--input2', required=True, help='第二个输入文件路径')
    parser.add_argument('-o', '--output', required=True, help='输出文件路径')

    args = parser.parse_args()

    # 处理点云数据
    if not process_points(args.input1, args.input2, args.output):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
