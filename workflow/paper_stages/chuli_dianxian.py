#!/usr/bin/env python3
# -------------------------------------------------
# 功能：
#   1) 读取梁结构节点、单元以及位移数据
#   2) 沿每根梁线按固定间距离散成点云（支持重复点去重）
#   3) **按线性插值方式** 让内部点随两端节点同步振动，避免"杂点不动"现象
#   4) 导出初始点云坐标到文本文件
#   5) 导出点云位移数据
#   6) 合并共线线段分组
# -------------------------------------------------

import numpy as np
import math
from pathlib import Path
import argparse
from collections import defaultdict

# ------------------ 可调参数 ------------------
SPACING       = 0.01           # 沿梁线分点间距（单位同坐标系）
# ------------------------------------------------
NODE_FILE      = "node_coord.txt"            ##ansys整合的
ELEM_FILE      = "elem_info.txt"             ##ansys整合的
DISP_FILE      = "node_coord_weiyi.txt"      # 节点位移文件
OUTPUT_DIR = Path("discrete_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 输出文件，格式参照 generate_two_lines_point_displacement_copy.py
PTS_FILE   = OUTPUT_DIR / "points_with_displacement_merge.txt"  # idx x y z ux uy uz
LINES_FILE = OUTPUT_DIR / "lines_old.txt"                      # line_idx p1_idx p2_idx group
MERGED_LINES_FILE = OUTPUT_DIR / "lines_merge.txt"         # line_idx p1_idx p2_idx merged_group

# -------------------------------------------------

class UnionFind:
    """并查集实现，用于合并线段组"""
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return

        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        else:
            self.parent[root_y] = root_x
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_x] += 1

def normalize_vector(vec):
    """标准化向量并使首个非零分量为正"""
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.zeros_like(vec)

    direction = vec / norm
    # 标准化方向（使首个非零分量为正）
    for i in range(len(direction)):
        if abs(direction[i]) > 1e-8:
            if direction[i] < 0:
                direction = -direction
            break

    return direction

def is_same_direction(dir1, dir2, tol=1e-6):
    """检查两个方向向量是否相同或相反"""
    dot_product = np.abs(np.dot(dir1, dir2))
    return abs(dot_product - 1.0) < tol

def are_points_collinear(p1, p2, p3, tol=1e-6):
    """判断三点是否共线"""
    v1 = p2 - p1
    v2 = p3 - p1

    # 计算叉乘
    cross = np.cross(v1, v2)
    cross_norm = np.linalg.norm(cross)

    # 如果叉乘接近于零，则三点共线
    return cross_norm < tol

def point_to_line_distance(point, line_start, line_end):
    """计算点到线段的距离"""
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-8:
        return np.linalg.norm(point - line_start)

    line_unit_vec = line_vec / line_len
    point_vec = point - line_start

    # 计算投影长度
    proj_len = np.dot(point_vec, line_unit_vec)

    # 如果投影在线段外部，返回到端点的距离
    if proj_len < 0:
        return np.linalg.norm(point - line_start)
    elif proj_len > line_len:
        return np.linalg.norm(point - line_end)

    # 计算垂直距离
    proj_point = line_start + proj_len * line_unit_vec
    return np.linalg.norm(point - proj_point)

def generate_point_cloud():
    """根据节点和单元直接生成点云（仅节点端点），并按“共线 + 共享端点”规则合并线段组。

    逻辑概述：
    1. 读取节点、单元、位移数据。
    2. 点云 = 所有参与单元的节点坐标（不再插值补点）。
    3. 线段 = 单元两端节点构成的线段。
    4. 若两条线段共线且在某端点处相连，则视为同一“大线段” -> 使用并查集合并分组。
    5. 输出 points_with_displacement.txt 与 lines.txt（含 merged_group）。
    """

    # -------- 文件检查 --------
    for f in (NODE_FILE, ELEM_FILE, DISP_FILE):
        if not Path(f).is_file():
            raise FileNotFoundError(f"缺少数据文件: {f}")

    # -------- 读取节点坐标 --------
    nodes_raw = np.loadtxt(NODE_FILE, dtype=float)
    if nodes_raw.ndim == 1:
        nodes_raw = nodes_raw.reshape(1, -1)

    node_ids   = nodes_raw[:, 0].astype(int)      # 节点 ID (N,)
    coords_arr = nodes_raw[:, 1:4].astype(float)  # 坐标 (N,3)
    id_to_idx  = {nid: i for i, nid in enumerate(node_ids)}

    # -------- 读取节点位移 --------
    disp_raw = np.loadtxt(DISP_FILE)
    disp_node_ids = disp_raw[:, 0].astype(int)
    node_disps = disp_raw[:, 1:]

    disp_arr = np.zeros((len(node_ids), 3))
    disp_id_to_idx = {nid: i for i, nid in enumerate(disp_node_ids)}
    for i, nid in enumerate(node_ids):
        if nid in disp_id_to_idx:
            disp_arr[i] = node_disps[disp_id_to_idx[nid]]

    # -------- 读取单元两端节点 --------
    elem_raw = np.loadtxt(ELEM_FILE)
    if elem_raw.shape[1] >= 3:
        elems = elem_raw[:, 1:3].astype(int)
    else:
        elems = np.loadtxt(ELEM_FILE, usecols=(1, 2), dtype=int)

    # 过滤有效单元（其节点必须存在于节点坐标文件）
    valid_elems = []
    for n1, n2 in elems:
        if n1 in id_to_idx and n2 in id_to_idx and n1 != n2:
            valid_elems.append((n1, n2))

    print(f"读取了 {len(valid_elems)} 个有效单元")

    # -------- 点云与位移（仅端点） --------
    point_cloud = coords_arr.copy()
    point_disps = disp_arr.copy()

    # -------- 创建线段数据 --------
    lines_data = []            # (line_idx, p1_idx, p2_idx)
    line_directions = {}       # line_idx -> 方向向量 (单位向量，首个非零分量为正)
    point_to_lines = defaultdict(list)  # 点索引 -> List[line_idx]

    for line_idx, (n1, n2) in enumerate(valid_elems):
        p_idx1 = id_to_idx[n1]
        p_idx2 = id_to_idx[n2]
        lines_data.append((line_idx, p_idx1, p_idx2))

        # 方向向量（标准化方向向量，使方向一致）
        v = coords_arr[p_idx2] - coords_arr[p_idx1]
        line_directions[line_idx] = normalize_vector(v)

        # 更新点到线段映射
        point_to_lines[p_idx1].append(line_idx)
        point_to_lines[p_idx2].append(line_idx)

    # -------- 合并线段组（共线 + 共享端点） --------
    uf = UnionFind()

    for pt_idx, lines in point_to_lines.items():
        if len(lines) < 2:
            continue
        # 任意两条线段共享此端点，若方向共线，则合并
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                li, lj = lines[i], lines[j]
                if is_same_direction(line_directions[li], line_directions[lj]):
                    uf.union(li, lj)

    # -------- 生成组ID --------
    root_to_group = {}
    line_to_group = {}
    next_gid = 1
    for line_idx in range(len(lines_data)):
        root = uf.find(line_idx)
        if root not in root_to_group:
            root_to_group[root] = next_gid
            next_gid += 1
        line_to_group[line_idx] = root_to_group[root]

    # -------- 过滤未在线上的孤立点 --------
    used_points = set()
    for _, p1, p2 in lines_data:
        used_points.add(p1)
        used_points.add(p2)

    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(used_points))}

    # -------- 写出点文件（仅使用的新索引） --------
    with open(PTS_FILE, "w") as f:
        f.write("# idx x y z ux uy uz\n")
        for old_idx in sorted(used_points):
            new_idx = old_to_new[old_idx]
            p = point_cloud[old_idx]
            u = point_disps[old_idx]
            f.write(f"{new_idx} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {u[0]:.6f} {u[1]:.6f} {u[2]:.6f}\n")
    print(f"✓ 已写入点与位移: {PTS_FILE} (仅保留在线段上的 {len(used_points)} 个点)")

    # -------- 写出线段文件（使用新索引） --------
    with open(MERGED_LINES_FILE, "w") as f:
        f.write("# line_idx p1_idx p2_idx merged_group\n")
        for line_idx, (orig_idx, p1, p2) in enumerate(lines_data):
            f.write(f"{orig_idx} {old_to_new[p1]} {old_to_new[p2]} {line_to_group[line_idx]}\n")

    print(f"✓ 已写入线段信息: {MERGED_LINES_FILE}")
    print(f"共识别 {next_gid - 1} 条“大线段” (merged_group)。")

    return point_cloud, point_disps

def process_beam_data(
    node_file: str = "node_coord.txt",
    elem_file: str = "elem_info.txt",
    disp_file: str = "node_coord_weiyi.txt",
    spacing: float = 0.01,
    output_dir: str | Path = "discrete_data",
):
    """按给定参数生成点云及位移数据。

    参数
    -----
    node_file : str
        节点坐标文件路径。
    elem_file : str
        单元信息文件路径 (仅需 n1 n2 两列)。
    disp_file : str
        节点位移文件路径。
    spacing : float
        沿梁线分点间距。
    output_dir : str | Path
        输出目录。
    返回
    -----
    point_cloud : np.ndarray (P,3)
    point_disps : np.ndarray (P,3)
    """

    global SPACING, NODE_FILE, ELEM_FILE, DISP_FILE, OUTPUT_DIR, PTS_FILE, LINES_FILE, MERGED_LINES_FILE

    SPACING = spacing
    NODE_FILE = node_file
    ELEM_FILE = elem_file
    DISP_FILE = disp_file
    OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    PTS_FILE = OUTPUT_DIR / "points_with_displacement_merge.txt"
    # 不再产生旧版 lines_old.txt
    LINES_FILE = OUTPUT_DIR / "lines_old.txt"  # 占位
    MERGED_LINES_FILE = OUTPUT_DIR / "lines_merge.txt"

    print("读取节点坐标: ", NODE_FILE)
    print("读取单元信息: ", ELEM_FILE)
    print("读取节点位移: ", DISP_FILE)

    point_cloud, point_disps = generate_point_cloud()

    print("写出点与位移: ", PTS_FILE)
    print("写出合并线段: ", MERGED_LINES_FILE)

    return point_cloud, point_disps

def load_node_set(node_file, tol=1e-6):
    """把节点坐标读入 set，元素是 (round(x/tol), round(y/tol), round(z/tol))。"""
    node_set = set()
    with open(node_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            # 如果首列是 id，则坐标从 parts[1:]
            coords = list(map(float, parts[-3:]))
            key = tuple(int(round(c / tol)) for c in coords)
            node_set.add(key)
    return node_set

def filter_offline_points(pt_file, node_set, tol=1e-6):
    """过滤掉离线点，只保留属于node_set的点。"""
    offline_pts = []
    with open(pt_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            idx = int(parts[0])
            coords = list(map(float, parts[1:4]))
            key = tuple(int(round(c / tol)) for c in coords)
            if key not in node_set:
                offline_pts.append(idx)
    return offline_pts


def main():
    parser = argparse.ArgumentParser(description="用于梁结构分析的点云离散化工具")
    parser.add_argument('--node_file', type=str, default='node_coord.txt', help='节点坐标文件')
    parser.add_argument('--elem_file', type=str, default='elem_info.txt', help='单元信息文件')
    parser.add_argument('--disp_file', type=str, default='node_coord_weiyi.txt', help='节点位移文件')
    parser.add_argument('--spacing', type=float, default=0.01, help='沿梁线分点间距')
    parser.add_argument('--output_dir', type=str, default='discrete_data', help='输出目录')
    args = parser.parse_args()

    process_beam_data(
        node_file=args.node_file,
        elem_file=args.elem_file,
        disp_file=args.disp_file,
        spacing=args.spacing,
        output_dir=args.output_dir,
    )

if __name__ == "__main__":
    main()

# 示例外部调用代码
# -------------------------------------------------
# from chuli_dianxian import process_beam_data
# point_cloud, disps = process_beam_data(
#     node_file="node_coord.txt",
#     elem_file="elem_info.txt",
#     disp_file="node_coord_weiyi.txt",
#     spacing=0.01,
#     output_dir="discrete_data",
# )
# -------------------------------------------------



