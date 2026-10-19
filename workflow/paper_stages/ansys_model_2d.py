"""
ansys_model_2d.py — 二维矢量线图（正视图）
------------------------------------------------
参考 ansys_model.py，生成梁结构的正视图（XZ平面投影）

功能：
- 加载节点和单元数据
- 生成二维矢量线图（SVG/PDF/PNG格式）
- 支持多种视图：正视图(XZ)、侧视图(YZ)、俯视图(XY)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from pathlib import Path
from typing import List, Tuple, Dict
import argparse

# =============== 默认参数 ===============
NODE_FILE = "node_coord.txt"
ELEM_FILE = "elem_info.txt"

# =============== 数据加载 ===============

def load_node_coordinates(fname: str) -> Dict[int, Tuple[float, float, float]]:
    """加载节点坐标"""
    nodes = {}
    with open(fname, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            parts = ln.split()
            if len(parts) < 4:
                continue
            try:
                nid = int(float(parts[0]))
                nodes[nid] = tuple(map(float, parts[1:4]))
            except ValueError:
                pass
    return nodes


def load_element_info(fname: str) -> List[Tuple[int, int, int, int, int]]:
    """加载单元信息"""
    elems = []
    with open(fname, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            parts = ln.split()
            if len(parts) < 3:  # 至少需要 id, n1, n2
                continue
            try:
                # 兼容不同格式：可能有3-5列
                elem_data = [int(float(x)) for x in parts[:min(5, len(parts))]]
                # 补齐到5列
                while len(elem_data) < 5:
                    elem_data.append(0)
                elems.append(tuple(elem_data))
            except ValueError:
                pass
    return elems


# =============== 线段合并 ===============

def merge_collinear_lines(lines: List, tolerance: float = 1e-6) -> List:
    """
    合并同一直线上的连续线段
    只合并共享端点且方向严格平行的线段
    """
    if not lines:
        return lines

    # 转换为numpy数组，保留原始顺序
    segments = []
    for line in lines:
        p0 = np.array(line[0])
        p1 = np.array(line[1])
        segments.append((p0, p1))

    n = len(segments)

    # 量化坐标用于哈希（更大的容差）
    def quantize(p, tol=tolerance):
        return (round(p[0] / tol), round(p[1] / tol))

    # 构建端点到线段索引的映射
    endpoint_map = {}
    for i, (p0, p1) in enumerate(segments):
        for ep_idx, p in enumerate([p0, p1]):
            key = quantize(p)
            if key not in endpoint_map:
                endpoint_map[key] = []
            endpoint_map[key].append((i, ep_idx))

    def get_direction(p0, p1):
        """获取归一化方向向量"""
        d = p1 - p0
        length = np.linalg.norm(d)
        if length < 1e-10:
            return None, 0
        return d / length, length

    def directions_parallel(d1, d2):
        """检查两个方向向量是否平行（严格，角度 < 1度）"""
        if d1 is None or d2 is None:
            return False
        dot = abs(np.dot(d1, d2))
        return dot > 0.9998  # cos(1°) ≈ 0.9998

    # 使用并查集思想，但不合并，只标记可合并的组
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # 检查每个端点，如果两条线段共享端点且方向平行，则合并
    for key, seg_list in endpoint_map.items():
        if len(seg_list) < 2:
            continue
        # 检查所有配对
        for i in range(len(seg_list)):
            for j in range(i + 1, len(seg_list)):
                idx_i, ep_i = seg_list[i]
                idx_j, ep_j = seg_list[j]

                # 获取两条线段的方向
                d_i, len_i = get_direction(segments[idx_i][0], segments[idx_i][1])
                d_j, len_j = get_direction(segments[idx_j][0], segments[idx_j][1])

                if directions_parallel(d_i, d_j):
                    union(idx_i, idx_j)

    # 按组收集线段
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    # 对每个组，合并为一条线（取所有端点的极值）
    result = []
    for root, indices in groups.items():
        if len(indices) == 1:
            # 单条线段，直接保留
            p0, p1 = segments[indices[0]]
            result.append([(p0[0], p0[1]), (p1[0], p1[1])])
        else:
            # 多条线段，找主方向，然后投影取极值
            all_points = []
            for idx in indices:
                all_points.append(segments[idx][0])
                all_points.append(segments[idx][1])

            # 用第一条线段的方向作为主方向
            d, _ = get_direction(segments[indices[0]][0], segments[indices[0]][1])
            if d is None:
                # 退化情况，保留所有原始线段
                for idx in indices:
                    p0, p1 = segments[idx]
                    result.append([(p0[0], p0[1]), (p1[0], p1[1])])
                continue

            # 投影所有点到主方向
            projections = [(np.dot(p, d), p) for p in all_points]
            projections.sort(key=lambda x: x[0])

            # 取投影最小和最大的点
            p_min = projections[0][1]
            p_max = projections[-1][1]

            result.append([(p_min[0], p_min[1]), (p_max[0], p_max[1])])

    return result


# =============== 二维投影 ===============

def project_to_2d(nodes: Dict, elements: List, view: str = "front") -> Tuple[np.ndarray, np.ndarray]:
    """
    将3D节点和单元投影到2D平面

    view:
        'front'  - 正视图 (X-Z平面, Y方向看)
        'side'   - 侧视图 (Y-Z平面, X方向看)
        'top'    - 俯视图 (X-Y平面, Z方向看)
    """
    # 构建线段列表
    lines = []
    for elem in elements:
        eid, n1, n2 = elem[:3]
        if n1 in nodes and n2 in nodes:
            p1 = np.array(nodes[n1])
            p2 = np.array(nodes[n2])

            if view == "front":  # XZ平面
                line = [(p1[0], p1[2]), (p2[0], p2[2])]
            elif view == "side":  # YZ平面
                line = [(p1[1], p1[2]), (p2[1], p2[2])]
            else:  # top, XY平面
                line = [(p1[0], p1[1]), (p2[0], p2[1])]

            lines.append(line)

    # 构建节点2D坐标
    node_coords_2d = {}
    for nid, (x, y, z) in nodes.items():
        if view == "front":
            node_coords_2d[nid] = (x, z)
        elif view == "side":
            node_coords_2d[nid] = (y, z)
        else:
            node_coords_2d[nid] = (x, y)

    return lines, node_coords_2d


def plot_2d_structure(nodes: Dict, elements: List,
                      view: str = "front",
                      line_width: float = 1.0,
                      line_color: str = "black",
                      node_size: float = 0,
                      figsize: Tuple[float, float] = (10, 10),
                      dpi: int = 300,
                      title_font: int = 14,
                      tick_font: int = 12,
                      show_grid: bool = True,
                      equal_aspect: bool = True,
                      separate_lines: bool = True,
                      merge_collinear: bool = False) -> plt.Figure:
    """
    绘制二维矢量线图

    separate_lines: 如果为True，每条线独立绘制（SVG中每条线是独立元素）
    merge_collinear: 如果为True，合并同一直线上的连续线段
    """
    # 投影到2D
    lines, node_coords_2d = project_to_2d(nodes, elements, view)

    # 合并共线线段
    if merge_collinear:
        original_count = len(lines)
        lines = merge_collinear_lines(lines)
        print(f"📐 共线合并: {original_count} → {len(lines)} 条线段")

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 绘制线段
    if separate_lines:
        # 每条线独立绘制 - SVG中每条线是独立的 <path> 元素
        for i, line in enumerate(lines):
            xs = [line[0][0], line[1][0]]
            ys = [line[0][1], line[1][1]]
            ax.plot(xs, ys, linewidth=line_width, color=line_color,
                   solid_capstyle='round', marker='', linestyle='-')
    else:
        # 使用 LineCollection - 所有线合并为一个元素
        lc = LineCollection(lines, linewidths=line_width, colors=line_color)
        ax.add_collection(lc)

    # 绘制节点（可选）
    if node_size > 0:
        xs = [coord[0] for coord in node_coords_2d.values()]
        ys = [coord[1] for coord in node_coords_2d.values()]
        ax.scatter(xs, ys, s=node_size, c='red', zorder=5)

    # 设置坐标轴
    view_labels = {
        "front": ("X (m)", "Z (m)", "正视图 (XZ平面)"),
        "side":  ("Y (m)", "Z (m)", "侧视图 (YZ平面)"),
        "top":   ("X (m)", "Y (m)", "俯视图 (XY平面)")
    }
    xlabel, ylabel, title = view_labels.get(view, ("X", "Y", "2D View"))

    ax.set_xlabel(xlabel, fontsize=title_font, fontfamily='Times New Roman', style='italic')
    ax.set_ylabel(ylabel, fontsize=title_font, fontfamily='Times New Roman', style='italic')

    # 刻度字体
    ax.tick_params(axis='both', labelsize=tick_font)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')

    # 网格
    if show_grid:
        ax.grid(True, linestyle='--', alpha=0.5, color='#E0E0E0')

    # 等比例
    if equal_aspect:
        ax.set_aspect('equal')

    # 自动调整范围
    ax.autoscale()
    ax.margins(0.05)

    # 白色背景
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    plt.tight_layout()

    return fig, ax


def save_2d_figure(fig: plt.Figure, output_path: str,
                   width: int = 10000, height: int = 10000,
                   format: str = "svg"):
    """
    保存二维图形

    format: 'svg', 'pdf', 'png', 'eps'
    """
    # 确保目录存在
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 根据格式保存
    if format.lower() in ['svg', 'pdf', 'eps']:
        # 矢量格式
        fig.savefig(output_path, format=format, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    else:
        # 栅格格式 (PNG)
        # 计算 DPI 以达到目标分辨率
        fig_width, fig_height = fig.get_size_inches()
        dpi = max(width / fig_width, height / fig_height)
        fig.savefig(output_path, format='png', dpi=dpi,
                   bbox_inches='tight', facecolor='white', edgecolor='none')

    print(f"✅ 已保存: {output_path}")


def main():
    """主程序"""
    parser = argparse.ArgumentParser(description="梁结构二维矢量线图生成工具")

    # 输入文件
    parser.add_argument('--nodes', default=NODE_FILE, help='节点坐标文件')
    parser.add_argument('--elems', default=ELEM_FILE, help='单元信息文件')

    # 视图选择
    parser.add_argument('--view', choices=['front', 'side', 'top'], default='front',
                       help='视图类型: front(正视图XZ), side(侧视图YZ), top(俯视图XY)')

    # 输出设置
    parser.add_argument('--output', '-o', default='screenshots/structure_2d.svg',
                       help='输出文件路径 (支持 .svg, .pdf, .png, .eps)')
    parser.add_argument('--width', type=int, default=10000, help='图像宽度(px)')
    parser.add_argument('--height', type=int, default=10000, help='图像高度(px)')

    # 样式设置
    parser.add_argument('--line-width', type=float, default=1.0, help='线宽')
    parser.add_argument('--line-color', default='black', help='线颜色')
    parser.add_argument('--node-size', type=float, default=0, help='节点大小(0=不显示)')
    parser.add_argument('--title-font', type=int, default=14, help='轴标题字号')
    parser.add_argument('--tick-font', type=int, default=12, help='刻度字号')
    parser.add_argument('--figsize', type=float, nargs=2, default=[10, 10],
                       help='图形尺寸(英寸)')
    parser.add_argument('--dpi', type=int, default=300, help='DPI')
    parser.add_argument('--no-grid', action='store_true', help='不显示网格')
    parser.add_argument('--merge-lines', action='store_true',
                       help='合并所有线条为一个SVG元素（默认每条线独立）')
    parser.add_argument('--merge-collinear', action='store_true',
                       help='合并同一直线上的连续线段（如a-b, b-c共线则合并为a-c）')

    # 显示控制
    parser.add_argument('--show', action='store_true', help='显示图形')
    parser.add_argument('--no-save', action='store_true', help='不保存图形')

    args = parser.parse_args()

    # 加载数据
    print(f"📂 加载数据...")
    print(f"   节点文件: {args.nodes}")
    print(f"   单元文件: {args.elems}")

    try:
        nodes = load_node_coordinates(args.nodes)
        elements = load_element_info(args.elems)
        print(f"✅ 加载完成: {len(nodes)} 节点, {len(elements)} 单元")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    # 绘制
    print(f"🎨 生成 {args.view} 视图...")
    fig, ax = plot_2d_structure(
        nodes, elements,
        view=args.view,
        line_width=args.line_width,
        line_color=args.line_color,
        node_size=args.node_size,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        title_font=args.title_font,
        tick_font=args.tick_font,
        show_grid=not args.no_grid,
        separate_lines=not args.merge_lines,
        merge_collinear=args.merge_collinear
    )

    # 保存
    if not args.no_save:
        output_path = Path(args.output)
        fmt = output_path.suffix.lstrip('.').lower()
        if fmt not in ['svg', 'pdf', 'png', 'eps']:
            fmt = 'svg'
        save_2d_figure(fig, args.output, args.width, args.height, fmt)

    # 显示
    if args.show:
        plt.show()
    else:
        plt.close(fig)

    print("✨ 完成!")


if __name__ == "__main__":
    main()
