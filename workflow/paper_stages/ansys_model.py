import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import csv
import time
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# =============== 输入输出参数配置 ===============
# 文件路径
NODE_FILE = "node_coord.txt"      # 节点坐标 (id x y z)
ELEM_FILE = "elem_info.txt"       # 梁单元 (id n1 n2 dir_node sec_id)
SECTION_FILE = "sections.txt"     # 截面表格 (id,type,p1,p2,p3,p4)
OUTPUT_CLOUD_FILE = "point_cloud.txt"  # 点云输出文件

# 几何参数
EDGE_SAMPLE_RATIO = 0.3  # 轮廓描线抽样比例
STEP = 0.1               # 轴向采样步长 (m)
PT_SPACING = 0.01        # 截面内点云网格间距 (m)
EPS = 1e-8               # 几何计算精度阈值

"""
visualize_plotly.py — v4.0  (2025‑05‑10)
---------------------------------------
* **按截面编号区分 L / T 型，并按各自尺寸挤出实体**
* 截面参数来源：section_table.csv (列: id,type,p1,p2,p3,p4)
    * **L**: p1=leg_w1, p2=leg_w2, p3=thk1, p4=thk2
    * **T**: p1=bf, p2=hw, p3=tf,   p4=tw
* elem_info.txt 第 5 列 = section_id；脚本自动查表。
* 方向点逻辑 & 左手系依旧。
* 轮廓描线抽样 EDGE_SAMPLE_RATIO。

文件要求
────────
1. node_coord.txt   – 节点坐标 (id x y z)
2. elem_info.txt    – 梁单元 (id n1 n2 dir_node sec_id)
3. section_table.csv – 截面表格 (id,type,p1,p2,p3,p4)

若你已有截面列表 txt，可另存为 csv，或修改 load_section_table() 中解析逻辑。
"""

# =============== 数据加载 ===============


def load_node_coordinates(fname: str = NODE_FILE) -> Dict[int, Tuple[float, float, float]]:
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


def load_element_info(fname: str = ELEM_FILE) -> List[Tuple[int, int, int, int, int]]:
    elems = []
    with open(fname, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            parts = ln.split()
            if len(parts) < 5:
                continue
            try:
                elems.append(tuple(int(float(x)) for x in parts[:5]))
            except ValueError:
                pass
    return elems


def load_section_table(fname: str = SECTION_FILE) -> Dict[int, Tuple[str, List[float]]]:
    """返回 {sec_id: (sec_type, [p1..p4])}"""
    sec_map = {}
    if not Path(fname).exists():
        raise FileNotFoundError(f"截面表 {fname} 不存在！")
    with open(fname, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            try:
                sid = int(float(row[0]))
                typ = row[1].strip().upper()
                params = [float(x) for x in row[2:6]]
                sec_map[sid] = (typ, params)
            except (ValueError, IndexError):
                continue
    return sec_map

# =============== 截面多边形生成 ===============


def poly_L(w1: float, w2: float, t1: float, t2: float) -> np.ndarray:
    return np.array([
        [0,  0],
        [w1, 0],
        [w1, t1],
        [t2, t1],
        [t2, w2],
        [0,  w2]
    ], float)


def poly_T(bf: float, hw: float, tf: float, tw: float) -> np.ndarray:
    half_gap = (bf - tw) / 2
    return np.array([
    [0.0,            0.0        ],   # 1 翼缘左下 (起点)
    [bf,             0.0        ],   # 2 翼缘右下
    [bf,             tf         ],   # 3 翼缘右上
    [half_gap + tw,  tf         ],   # 4 腰板右下
    [half_gap + tw,  tf + hw    ],   # 5 腰板右上
    [half_gap,       tf + hw    ],   # 6 腰板左上
    [half_gap,       tf         ],   # 7 腰板左下
    [0.0,            tf         ],   # 8 翼缘左上
    ], float)

# =============== 几何辅助 ===============


def _stable_perp(axis: np.ndarray):
    ref = np.array([0, 0, 1]) if abs(axis[2]) < 0.9 else np.array([0, 1, 0])
    n2 = np.cross(ref, axis)
    n2 /= np.linalg.norm(n2)
    n1 = np.cross(n2, axis)
    n1 /= np.linalg.norm(n1)
    return n1, n2


def local_axes(p0: np.ndarray, p1: np.ndarray, dir_vec: Optional[np.ndarray]):
    axis = p1 - p0
    L = np.linalg.norm(axis)
    axis /= L
    if dir_vec is not None:
        proj = dir_vec - np.dot(dir_vec, axis) * axis
        n2_len = np.linalg.norm(proj)
        if n2_len > EPS:
            n2 = proj / n2_len
            n1 = np.cross(n2, axis)
            n1 /= np.linalg.norm(n1)
            return axis, n1, n2, L
    n1, n2 = _stable_perp(axis)
    return axis, n1, n2, L


def extrude_profile(p0: np.ndarray, p1: np.ndarray, poly2d: np.ndarray, dir_vec: Optional[np.ndarray]):
    axis, n1, n2, L = local_axes(p0, p1, dir_vec)
    front = p0 + n1[None,:]*poly2d[:,0:1] + n2[None,:]*poly2d[:,1:2]
    back  = front + axis[None,:]*L
    verts = np.vstack([front, back])
    n = len(poly2d)
    faces = []
    for i in range(n):
        j = (i+1)%n
        a,b,c,d = i,j,j+n,i+n
        faces.extend([[a,b,c],[a,c,d]])
    for k in range(1,n-1):
        faces.append([0,k,k+1])
        faces.append([n,k+n,(k+1)+n])
    return verts, np.array(faces,int)

# =============== Mesh 构建 ===============


def build_mesh(nodes, elements, sec_table):
    vx,vy,vz,I,J,K = [],[],[],[],[],[]
    v_off = 0
    for _,n1,n2,dir_node,sec_id in elements:
        if n1 not in nodes or n2 not in nodes or sec_id not in sec_table:
            continue
        p0 = np.array(nodes[n1]); p1 = np.array(nodes[n2])
        dir_vec = None
        if dir_node>0 and dir_node in nodes:
            dir_vec = np.array(nodes[dir_node]) - p0
        sec_type, params = sec_table[sec_id]
        if sec_type == 'L':
            poly = poly_L(*params)
        elif sec_type == 'T':
            poly = poly_T(*params)
        else:
            continue  # 未识别
        verts, faces = extrude_profile(p0,p1,poly,dir_vec)
        vx.extend(verts[:,0]); vy.extend(verts[:,1]); vz.extend(verts[:,2])
        I.extend(faces[:,0]+v_off); J.extend(faces[:,1]+v_off); K.extend(faces[:,2]+v_off)
        v_off += len(verts)
    return np.array(vx),np.array(vy),np.array(vz),np.array(I),np.array(J),np.array(K)

# =============== 轮廓边抽样 ===============

def sample_edges(I,J,K):
    edges=set(); rng=random.random
    for a,b in zip(I,J):
        if rng()<EDGE_SAMPLE_RATIO: edges.add(tuple(sorted((a,b))))
    for b,c in zip(J,K):
        if rng()<EDGE_SAMPLE_RATIO: edges.add(tuple(sorted((b,c))))
    for c,a in zip(K,I):
        if rng()<EDGE_SAMPLE_RATIO: edges.add(tuple(sorted((c,a))))
    return list(edges)

# =============== 可视化 ===============

def plot_structure(nodes,elements,sec_table):
    print("生成 Mesh…")
    t0=time.time()
    x,y,z,I,J,K = build_mesh(nodes,elements,sec_table)
    print(f"完成 (顶点 {len(x)}, 三角形 {len(I)}), 耗时 {time.time()-t0:.2f}s")

    # 高级渲染效果的梁实体
    mesh = go.Mesh3d(
        x=x, y=y, z=z, i=I, j=J, k=K,
        color="#0068D9",          # 更深、更鲜明的蓝色
        opacity=0.82,             # 略高的不透明度，增强质感
        flatshading=False,        # 关闭平面着色，使表面更光滑
        lighting=dict(
            ambient=0.55,          # 提高环境光
            diffuse=0.9,           # 增强漫反射
            roughness=0.5,         # 降低粗糙度
            specular=0.4,          # 增加高光反射
            fresnel=0.8            # 菲涅尔效应，边缘更亮
        ),
        lightposition=dict(x=100, y=200, z=150),  # 光源位置
        colorscale=[[0, "#0068D9"], [1, "#5B9CF6"]],  # 蓝色渐变
        intensity=np.ones(len(x)),  # 颜色强度向量
        showscale=False            # 不显示颜色条
    )

    # 增强线框效果
    edges=sample_edges(I,J,K)
    lx=[];ly=[];lz=[]
    for a,b in edges:
        lx+=[x[a],x[b],None]; ly+=[y[a],y[b],None]; lz+=[z[a],z[b],None]
    lines=go.Scatter3d(
        x=lx, y=ly, z=lz,
        mode="lines",
        line=dict(
            color="#2A2A2A",      # 更深的线条颜色，提高对比度
            width=1.2             # 稍微加粗
        ),
        opacity=0.7,              # 适当透明度
        hoverinfo="skip",
        showlegend=False
    )

    fig = go.Figure(data=[mesh, lines])

    # 应用科研论文级别的视觉风格
    fig.update_layout(
        # 移除标题
        scene=dict(
            aspectmode="data",
            bgcolor="white",
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=0.8),  # 设置更好的初始视角
                up=dict(x=0, y=0, z=1)
            ),
            xaxis=dict(
                title=dict(text="<i>X</i> (m)", font=dict(size=12)),  # 使用斜体
                tickfont=dict(size=12),  # 刻度字体
                showbackground=True,
                backgroundcolor="rgb(250, 250, 252)",
                gridcolor="#E0E0E0",
                zerolinecolor="#BDBDBD",
                showspikes=False  # 移除拾取线
            ),
            yaxis=dict(
                title=dict(text="<i>Y</i> (m)", font=dict(size=12)),  # 使用斜体
                tickfont=dict(size=12),  # 刻度字体
                showbackground=True,
                backgroundcolor="rgb(250, 250, 252)",
                gridcolor="#E0E0E0",
                zerolinecolor="#BDBDBD",
                showspikes=False
            ),
            zaxis=dict(
                title=dict(text="<i>Z</i> (m)", font=dict(size=12)),  # 使用斜体
                tickfont=dict(size=12),  # 刻度字体
                showbackground=True,
                backgroundcolor="rgb(250, 250, 252)",
                gridcolor="#E0E0E0",
                zerolinecolor="#BDBDBD",
                showspikes=False
            ),
        ),
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=10, b=0),  # 减少顶部边距，因为没有标题
        font=dict(family="Times New Roman")  # 更改为 Times New Roman 字体
    )

    # 创建screenshots文件夹(如果不存在)
    screenshots_dir = Path("screenshots")
    screenshots_dir.mkdir(exist_ok=True)

    # 添加字体大小控制菜单
    font_sizes = [12, 16, 20, 24, 30, 36, 42, 48]  # 扩充字号选项到48pt
    buttons = []

    # 生成按钮（分离轴标题和刻度数字字号）
    for size in font_sizes:
        buttons.append(
            dict(
                label=f"标题 {size}px / 刻度 {int(size*0.75)}px",
                method="relayout",
                args=[{
                    # -------- 轴标题 font --------
                    "scene.xaxis.title.font.size": size,
                    "scene.yaxis.title.font.size": size,
                    "scene.zaxis.title.font.size": size,

                    # -------- 刻度数字 font（75% 规则） --------
                    "scene.xaxis.tickfont.size": int(size*0.75),
                    "scene.yaxis.tickfont.size": int(size*0.75),
                    "scene.zaxis.tickfont.size": int(size*0.75),

                    # 论文里常把全局字体也调大一点
                    "font.size": int(size*0.8),

                    # 为了防止 <i></i> 被覆盖，再写回一次
                    "scene.xaxis.title.text": "<i>X</i> (m)",
                    "scene.yaxis.title.text": "<i>Y</i> (m)",
                    "scene.zaxis.title.text": "<i>Z</i> (m)"
                }]
            )
        )

    # 添加菜单到图形中
    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.05,
                xanchor="left",
                y=0.98,
                yanchor="top",
                bgcolor="white",
                bordercolor="#c7c7c7",
                font=dict(family="Times New Roman", size=12, color="black"),
                borderwidth=1,
                name="字体大小"
            )
        ]
    )

    # 配置高清截图按钮 - 提高到浏览器可处理的最高分辨率
    config = dict(
        toImageButtonOptions=dict(
            format="png",
            filename="BeamStructure_HighRes",
            height=4000,                # 升级到 4K 级别
            width=4000,
            scale=3                     # 高缩放比例提高锐利度
        ),
        displaylogo=False,
        modeBarButtonsToAdd=[
            "toImage",
            "resetCameraLastSave3d",
            "orbitRotation",
            "zoom3d",                   # 添加缩放控制
            "pan3d",                    # 添加平移控制
            "hoverClosest3d"            # 添加悬停信息
        ],
        responsive=True,
        displayModeBar=True,            # 始终显示工具栏
        modeBarButtonsToRemove=[]       # 保留所有按钮
    )

    # 显示交互式图形，让用户自己选择视角
    print("显示 3D 模型，请旋转至合适角度后使用以下功能:")
    print("1. 左上角下拉菜单可调整字体大小")
    print("2. 使用截图按钮可直接保存当前视图")
    print(f"3. 所有图像将保存到: {screenshots_dir.absolute()}")

    # 显示图形
    fig.show(config=config)

    print("\n提示: 调整字体大小后再使用截图按钮，保存的图片将保留当前字体大小")
    print("如需更高清晰度(10K-15K)与更大字体，请使用命令行: python 1.py --save-hires 36 28 10000")

# ---------- 几何工具 ----------
def pts_in_poly(poly: np.ndarray, spacing: float) -> np.ndarray:
    """
    给定 2D 多边形 (N×2)，返回其内部点云 (M×2)
    使用射线法，纯 numpy 实现，无外部依赖
    """
    minx, miny = poly.min(axis=0)
    maxx, maxy = poly.max(axis=0)
    xs = np.arange(minx, maxx + spacing*0.5, spacing)
    ys = np.arange(miny, maxy + spacing*0.5, spacing)
    X, Y = np.meshgrid(xs, ys)
    P  = np.c_[X.ravel(), Y.ravel()]

    # 射线法：统计多边形边与水平射线的交点个数
    n = len(poly)
    x0, y0 = P[:, 0], P[:, 1]
    inside = np.zeros(len(P), dtype=bool)

    for i in range(n):
        j = (i + 1) % n
        xi, yi = poly[i]
        xj, yj = poly[j]
        # 判断边 (i,j) 是否与水平射线相交
        cond = ((yi > y0) != (yj > y0)) & \
               (x0 < (xj - xi) * (y0 - yi) / (yj - yi + 1e-12) + xi)
        inside ^= cond  # 交点奇偶规则

    return P[inside]

def beam_point_cloud(p0: np.ndarray, p1: np.ndarray,
                     dir_vec: Optional[np.ndarray],
                     poly2d: np.ndarray,
                     step: float = STEP,
                     spacing: float = PT_SPACING) -> np.ndarray:
    """
    沿梁轴线离散，并在每个截面内生成 3D 点云
    返回 (K×3) numpy 数组
    """
    axis, n1, n2, L = local_axes(p0, p1, dir_vec)
    t_vals = np.arange(0.0, L + step*0.5, step)
    sec_pts = pts_in_poly(poly2d, spacing)           # 先算一次截面内部点
    pts_3d  = []
    for t in t_vals:
        o = p0 + axis * t                            # 本截面原点
        pts  = o + n1[None, :] * sec_pts[:, 0:1] +   \
                   n2[None, :] * sec_pts[:, 1:2]
        pts_3d.append(pts)
    return np.vstack(pts_3d)

def elements_to_point_cloud(nodes, elements, sec_table,
                            step=STEP, spacing=PT_SPACING) -> np.ndarray:
    """
    遍历全部梁单元，聚合点云
    """
    clouds = []
    for _, n1, n2, dir_node, sec_id in elements:
        if n1 not in nodes or n2 not in nodes or sec_id not in sec_table:
            continue
        p0 = np.array(nodes[n1])
        p1 = np.array(nodes[n2])
        dir_vec = None
        if dir_node > 0 and dir_node in nodes:
            dir_vec = np.array(nodes[dir_node]) - p0
        sec_type, params = sec_table[sec_id]
        if sec_type == 'L':
            poly = poly_L(*params)
        elif sec_type == 'T':
            poly = poly_T(*params)
        else:
            continue
        cloud = beam_point_cloud(p0, p1, dir_vec, poly, step, spacing)
        clouds.append(cloud)
    return np.vstack(clouds) if clouds else np.empty((0, 3))

def write_xyz(points: np.ndarray, fname: str = OUTPUT_CLOUD_FILE):
    """x y z ASCII"""
    with open(fname, "w") as f:
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
    print(f"已输出点云 {len(points)} 点 → {fname}")

# =============== 主入口 ===============
# =============== 超高清导出 ===============
def save_high_res(nodes, elements, sec_table,
                 title_size=36, tick_size=28,
                 width=10000, height=10000, scale=3,
                 filename="BeamStructure_HighRes.png"):
    """
    生成超高清图像 (10K-15K)，可分别控制标题和刻度字体大小

    参数:
        nodes (dict): 节点坐标字典 {节点ID: (x,y,z)}
        elements (list): 梁单元列表 [(id,n1,n2,dir_node,sec_id),...]
        sec_table (dict): 截面信息字典 {截面ID: (类型,参数列表)}
        title_size (int): 轴标题字体大小(px)，建议论文用36-48px
        tick_size (int): 刻度字体大小(px)，建议为标题的60-75%
        width (int): 图像宽度(px)，推荐8000-15000
        height (int): 图像高度(px)，推荐8000-15000
        scale (int): 缩放因子，用于抗锯齿，通常为2-3
        filename (str): 输出文件名，默认包含分辨率和字体信息

    内存要求:
        内存估算公式: (width×height×4 × scale²) / 2²⁰ ≈ RAM(GB)
        常见配置:
        - 10K²×3 ≈ 1.2GB
        - 12K²×3 ≈ 1.7GB
        - 15K²×3 ≈ 2.6GB

    返回:
        bool: 图像导出是否成功
    """
    print(f"正在生成超高清图像...")
    print(f"- 标题字体: {title_size}px")
    print(f"- 刻度字体: {tick_size}px")
    print(f"- 分辨率: {width}x{height}, 缩放: {scale}")

    # 创建一个新的图形对象
    x, y, z, I, J, K = build_mesh(nodes, elements, sec_table)

    mesh = go.Mesh3d(
        x=x, y=y, z=z, i=I, j=J, k=K,
        color="#0068D9",
        opacity=0.82,
        flatshading=False,
        lighting=dict(
            ambient=0.55, diffuse=0.9,
            roughness=0.5, specular=0.4,
            fresnel=0.8
        ),
        lightposition=dict(x=100, y=200, z=150),
        colorscale=[[0, "#0068D9"], [1, "#5B9CF6"]],
        intensity=np.ones(len(x)),
        showscale=False
    )

    edges = sample_edges(I, J, K)
    lx=[]; ly=[]; lz=[]
    for a, b in edges:
        lx+=[x[a], x[b], None]
        ly+=[y[a], y[b], None]
        lz+=[z[a], z[b], None]

    lines = go.Scatter3d(
        x=lx, y=ly, z=lz,
        mode="lines",
        line=dict(color="#2A2A2A", width=1.2),
        opacity=0.7,
        hoverinfo="skip",
        showlegend=False
    )

    custom_fig = go.Figure(data=[mesh, lines])

    # 坐标轴标签
    x_label = "<i>X</i> (m)"
    y_label = "<i>Y</i> (m)"
    z_label = "<i>Z</i> (m)"

    # 应用指定的字体大小和其他样式
    custom_fig.update_layout(
        scene=dict(
            aspectmode="data",
            bgcolor="white",
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=0.8),
                up=dict(x=0, y=0, z=1)
            ),
            xaxis=dict(
                title=dict(text=x_label, font=dict(size=title_size, family="Times New Roman")),
                tickfont=dict(size=tick_size, family="Times New Roman"),
                showbackground=True,
                backgroundcolor="rgb(250, 250, 252)",
                gridcolor="#E0E0E0",
                zerolinecolor="#BDBDBD",
                showspikes=False
            ),
            yaxis=dict(
                title=dict(text=y_label, font=dict(size=title_size, family="Times New Roman")),
                tickfont=dict(size=tick_size, family="Times New Roman"),
                showbackground=True,
                backgroundcolor="rgb(250, 250, 252)",
                gridcolor="#E0E0E0",
                zerolinecolor="#BDBDBD",
                showspikes=False
            ),
            zaxis=dict(
                title=dict(text=z_label, font=dict(size=title_size, family="Times New Roman")),
                tickfont=dict(size=tick_size, family="Times New Roman"),
                showbackground=True,
                backgroundcolor="rgb(250, 250, 252)",
                gridcolor="#E0E0E0",
                zerolinecolor="#BDBDBD",
                showspikes=False
            ),
        ),
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=10, b=0),
        font=dict(family="Times New Roman", size=int(title_size*0.8))
    )

    # 确保screenshots文件夹存在
    screenshots_dir = Path("screenshots")
    screenshots_dir.mkdir(exist_ok=True)
    output_path = screenshots_dir / filename

    try:
        # 尝试使用kaleido引擎保存超高分辨率图像
        custom_fig.write_image(str(output_path), width=width, height=height, scale=scale)
        print(f"已成功保存超高清图像到: {output_path}")
        return True
    except Exception as e:
        print(f"保存图像失败: {str(e)}")
        print(f"请确保已安装最新版kaleido库: pip install -U kaleido")
        return False

def main():
    """主程序入口，处理命令行参数并执行相应功能"""
    # 声明全局变量(必须在使用前声明)
    global STEP, PT_SPACING, EDGE_SAMPLE_RATIO

    import argparse

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="梁结构可视化与点云生成工具")

    # === 输入文件参数 ===
    parser.add_argument('--nodes', default=NODE_FILE, help='节点坐标文件路径 (默认: node_coord.txt)')
    parser.add_argument('--elems', default=ELEM_FILE, help='梁单元信息文件路径 (默认: elem_info.txt)')
    parser.add_argument('--sections', default=SECTION_FILE, help='截面定义文件路径 (默认: sections.txt)')

    # === 输出相关参数 ===
    parser.add_argument('--cloud-file', default=OUTPUT_CLOUD_FILE, help='点云输出文件路径 (默认: point_cloud.txt)')
    parser.add_argument('--image-dir', default="screenshots", help='图像输出目录 (默认: screenshots)')

    # === 可视化控制参数 ===
    parser.add_argument('--no-gui', action='store_true', help='不显示交互式界面')
    parser.add_argument('--no-cloud', action='store_true', help='不生成点云文件')

    # === 图像导出参数 ===
    parser.add_argument('--export', action='store_true', help='导出高分辨率图像')
    parser.add_argument('--title-font', type=int, default=36, help='轴标题字体大小(px) (默认: 36)')
    parser.add_argument('--tick-font', type=int, default=0, help='刻度字体大小(px) (默认: 标题字体×0.75)')
    parser.add_argument('--image-width', type=int, default=10000, help='图像宽度(px) (默认: 10000)')
    parser.add_argument('--image-height', type=int, default=10000, help='图像高度(px) (默认: 与宽度相同)')
    parser.add_argument('--scale', type=int, default=3, help='图像缩放因子 (默认: 3)')
    parser.add_argument('--filename', help='图像文件名 (默认: 自动生成)')

    # === 几何参数 ===
    parser.add_argument('--step', type=float, default=STEP, help='轴向采样步长(m) (默认: 0.1)')
    parser.add_argument('--spacing', type=float, default=PT_SPACING, help='截面内部点云间距(m) (默认: 0.01)')
    parser.add_argument('--edge-ratio', type=float, default=EDGE_SAMPLE_RATIO, help='轮廓边抽样比例 (默认: 0.3)')

    # 解析命令行参数
    args = parser.parse_args()

    # 更新全局参数
    STEP = args.step
    PT_SPACING = args.spacing
    EDGE_SAMPLE_RATIO = args.edge_ratio

    # 加载数据
    print(f"正在从以下文件加载数据:")
    print(f"- 节点文件: {args.nodes}")
    print(f"- 单元文件: {args.elems}")
    print(f"- 截面文件: {args.sections}")

    try:
        nodes = load_node_coordinates(args.nodes)
        elements = load_element_info(args.elems)
        sec_table = load_section_table(args.sections)
        print(f"数据加载完成: 节点 {len(nodes)}, 梁 {len(elements)}, 截面 {len(sec_table)}")
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        return

    # 处理刻度字体大小
    if args.tick_font <= 0:
        args.tick_font = int(args.title_font * 0.75)

    # 处理图像高度
    if args.image_height <= 0:
        args.image_height = args.image_width

    # 处理文件名
    if not args.filename:
        args.filename = f"BeamStructure_{args.image_width}x{args.image_height}_{args.title_font}px.png"

    # 是否导出高分辨率图像
    if args.export:
        save_high_res(nodes, elements, sec_table,
                     title_size=args.title_font,
                     tick_size=args.tick_font,
                     width=args.image_width,
                     height=args.image_height,
                     scale=args.scale,
                     filename=args.filename)

    # 是否显示交互式界面
    if not args.no_gui:
        plot_structure(nodes, elements, sec_table)

    # 是否生成点云
    if not args.no_cloud:
        print(f"\n生成点云...")
        print(f"- 轴向采样步长: {STEP} m")
        print(f"- 截面内网格间距: {PT_SPACING} m")
        pts = elements_to_point_cloud(nodes, elements, sec_table, step=STEP, spacing=PT_SPACING)
        write_xyz(pts, args.cloud_file)
        print(f"处理完成!")

    # 显示使用提示
    if not args.export and not args.no_gui:
        print("\n要生成超高清图像(10K-15K)，可使用以下命令:")
        print("python 1.py --export --title-font 36 --tick-font 28 --image-width 10000")
        print("\n参数完整列表:")
        print("  输入文件:")
        print("  --nodes FILE     : 节点坐标文件路径 (默认: node_coord.txt)")
        print("  --elems FILE     : 梁单元信息文件路径 (默认: elem_info.txt)")
        print("  --sections FILE  : 截面定义文件路径 (默认: sections.txt)")
        print("  输出设置:")
        print("  --cloud-file FILE: 点云输出文件路径 (默认: point_cloud.txt)")
        print("  --image-dir DIR  : 图像输出目录 (默认: screenshots)")
        print("  --filename FILE  : 图像文件名 (默认: 自动生成)")
        print("  可视化控制:")
        print("  --export         : 导出高分辨率图像")
        print("  --no-gui         : 不显示交互界面，仅导出图像")
        print("  --no-cloud       : 不生成点云文件")
        print("  图像参数:")
        print("  --title-font N   : 轴标题字体大小(px)，建议论文用36-48")
        print("  --tick-font N    : 刻度字体大小(px)，默认为标题的75%")
        print("  --image-width N  : 图像宽度(px)，建议10000-15000")
        print("  --image-height N : 图像高度(px)，默认与宽度相同")
        print("  --scale N        : 图像缩放因子，默认为3")
        print("  几何参数:")
        print("  --step FLOAT     : 轴向采样步长(m)，默认0.1")
        print("  --spacing FLOAT  : 截面内部点云间距(m)，默认0.01")
        print("  --edge-ratio FLOAT: 轮廓边抽样比例，默认0.3")
        print("\n完整示例:")
        print("  python 1.py --nodes node_coord.txt --elems elem_info.txt --export \\")
        print("    --title-font 36 --image-width 12000 --scale 3 --filename output.png")
        print("更多帮助请使用: python 1.py --help")

# ==========================================================================
# 完整参数列表及示例 - 可直接复制使用:
# ==========================================================================
# python 1.py \
#   --nodes node_coord.txt \                # 节点坐标文件路径
#   --elems elem_info.txt \                 # 梁单元信息文件路径
#   --sections sections.txt \               # 截面定义文件路径
#   --cloud-file point_cloud.txt \          # 点云输出文件路径
#   --image-dir screenshots \               # 图像输出目录
#   --export \                              # 导出高分辨率图像
#   --title-font 36 \                       # 轴标题字体大小(px)
#   --tick-font 27 \                        # 刻度字体大小(px)
#   --image-width 10000 \                   # 图像宽度(px)
#   --image-height 10000 \                  # 图像高度(px)
#   --scale 3 \                             # 图像缩放因子
#   --filename BeamStructure_Custom.png \   # 图像文件名
#   --step 0.1 \                            # 轴向采样步长(m)
#   --spacing 0.01 \                        # 截面内部点云间距(m)
#   --edge-ratio 0.3                        # 轮廓边抽样比例
#
# # 控制选项 (添加到上面命令):
# #   --no-gui                              # 不显示交互式界面
# #   --no-cloud                            # 不生成点云文件
#
# # 简化使用场景:
# # 1. 交互式界面: python 1.py
# # 2. 只生成点云: python 1.py --no-gui
# # 3. 导出高清图像: python 1.py --export --title-font 36 --image-width 10000
# # 4. 完整流程并自定义参数: python 1.py --export --title-font 48 --tick-font 30 --image-width 12000 --scale 3
# # 5. 批处理模式: python 1.py --export --no-gui --filename custom_output.png

if __name__ == "__main__":
    main()
