import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import argparse
import contextlib

SAVE_GROUP_FILES = False  # 默认不保存分组文件
ONLY_TRANSFORM = False
ONLY_APPLY = False
LAMBDA_REG = 1e-6  # 使用默认值

# ────────────────── 四元数和旋转工具函数 ──────────────────
def quaternion_from_vectors(v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    """计算从v0到v1的旋转四元数 (wxyz格式)"""
    v0 = v0 / (torch.norm(v0)+1e-12)
    v1 = v1 / (torch.norm(v1)+1e-12)
    dot = torch.clamp(torch.dot(v0,v1), -1.0, 1.0)

    if dot > 0.999999:
        return torch.tensor([1.0,0,0,0], device=v0.device, dtype=v0.dtype)  # 单位四元数，w=1

    if dot < -0.999999:
        axis = torch.cross(v0, torch.tensor([1.0,0,0], device=v0.device))
        if torch.norm(axis) < 1e-6:
            axis = torch.cross(v0, torch.tensor([0,1.0,0], device=v0.device))
        axis = axis/torch.norm(axis)
        return torch.tensor([0.0, *axis], device=v0.device)  # w=0, 180度旋转

    axis = torch.cross(v0,v1)
    s = torch.sqrt((1+dot)*2)
    invs = 1.0/s

    # wxyz格式
    w = s*0.5
    xyz = axis*invs
    q = torch.cat([w.unsqueeze(0), xyz])

    return torch.nn.functional.normalize(q, dim=0)

def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """将旋转矩阵转换为四元数 (w,x,y,z)

    参数:
        R: 旋转矩阵 (...,3,3)
    返回:
        四元数 (...,4) 格式为 (w,x,y,z)
    """
    batch_dim = R.shape[:-2]
    device, dtype = R.device, R.dtype

    # 按 Sarabandi 方法计算四元数
    # https://arc.aiaa.org/doi/10.2514/1.G000108

    # 计算旋转矩阵的迹
    trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(dim=-1)

    # 初始化结果数组
    q = torch.empty(batch_dim + (4,), device=device, dtype=dtype)

    # 根据迹的值选择不同的计算方法，以保证数值稳定性
    cond1 = trace > 0
    cond2 = ~cond1 & (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2])
    cond3 = ~cond1 & ~cond2 & (R[..., 1, 1] > R[..., 2, 2])
    cond4 = ~cond1 & ~cond2 & ~cond3

    # Case 1: 迹 > 0
    if cond1.any():
        indices = cond1.nonzero(as_tuple=True)
        R_1 = R[indices]
        S = torch.sqrt(trace[indices] + 1.0) * 2  # S = 4*qw
        qw = 0.25 * S
        qx = (R_1[..., 2, 1] - R_1[..., 1, 2]) / S
        qy = (R_1[..., 0, 2] - R_1[..., 2, 0]) / S
        qz = (R_1[..., 1, 0] - R_1[..., 0, 1]) / S
        q[indices] = torch.stack([qw, qx, qy, qz], dim=-1)

    # Case 2: R[0,0] 最大
    if cond2.any():
        indices = cond2.nonzero(as_tuple=True)
        R_2 = R[indices]
        S = torch.sqrt(1.0 + R_2[..., 0, 0] - R_2[..., 1, 1] - R_2[..., 2, 2]) * 2  # S = 4*qx
        qw = (R_2[..., 2, 1] - R_2[..., 1, 2]) / S
        qx = 0.25 * S
        qy = (R_2[..., 0, 1] + R_2[..., 1, 0]) / S
        qz = (R_2[..., 0, 2] + R_2[..., 2, 0]) / S
        q[indices] = torch.stack([qw, qx, qy, qz], dim=-1)

    # Case 3: R[1,1] 最大
    if cond3.any():
        indices = cond3.nonzero(as_tuple=True)
        R_3 = R[indices]
        S = torch.sqrt(1.0 + R_3[..., 1, 1] - R_3[..., 0, 0] - R_3[..., 2, 2]) * 2  # S = 4*qy
        qw = (R_3[..., 0, 2] - R_3[..., 2, 0]) / S
        qx = (R_3[..., 0, 1] + R_3[..., 1, 0]) / S
        qy = 0.25 * S
        qz = (R_3[..., 1, 2] + R_3[..., 2, 1]) / S
        q[indices] = torch.stack([qw, qx, qy, qz], dim=-1)

    # Case 4: R[2,2] 最大
    if cond4.any():
        indices = cond4.nonzero(as_tuple=True)
        R_4 = R[indices]
        S = torch.sqrt(1.0 + R_4[..., 2, 2] - R_4[..., 0, 0] - R_4[..., 1, 1]) * 2  # S = 4*qz
        qw = (R_4[..., 1, 0] - R_4[..., 0, 1]) / S
        qx = (R_4[..., 0, 2] + R_4[..., 2, 0]) / S
        qy = (R_4[..., 1, 2] + R_4[..., 2, 1]) / S
        qz = 0.25 * S
        q[indices] = torch.stack([qw, qx, qy, qz], dim=-1)

    # 归一化四元数，确保单位长度
    q = F.normalize(q, dim=-1)

    return q

# 四元数乘法 (w x y z 格式)
def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternions.

    参数:
        q1,q2: (...,4)  按 (w,x,y,z) 排列并已归一化
    返回:
        (...,4) same layout, 归一化结果
    """
    w1,x1,y1,z1 = q1.unbind(-1)
    w2,x2,y2,z2 = q2.unbind(-1)

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    out = torch.stack([w,x,y,z], dim=-1)
    return F.normalize(out, dim=-1)

# ────────────────── SO(3) 指数 / 对数映射 ──────────────────
def quat_log(q: torch.Tensor) -> torch.Tensor:
    """四元数 -> 李代数向量 omega (批量支持)"""
    assert q.shape[-1] == 4, "q shape (...,4)"
    q = F.normalize(q, dim=-1)
    w = q[..., 0]  # 标量部分
    xyz = q[..., 1:4]  # 向量部分
    norm_xyz = torch.norm(xyz, dim=-1, keepdim=True) + 1e-12
    theta = 2.0 * torch.atan2(norm_xyz, w.unsqueeze(-1))  # (...,1)
    axis = xyz / norm_xyz
    omega = theta * axis
    omega = torch.where(norm_xyz < 1e-6, xyz * 2.0, omega)  # 小角度近似
    return omega.squeeze(-1) if omega.shape[-1]==1 else omega


def quat_exp(omega: torch.Tensor) -> torch.Tensor:
    """李代数向量 omega -> 四元数 (批量支持)"""
    theta = torch.norm(omega, dim=-1, keepdim=True)
    half_theta = 0.5 * theta
    k = torch.where(theta < 1e-6, 0.5 - theta**2 / 48.0, torch.sin(half_theta)/theta)
    xyz = k * omega
    w = torch.cos(half_theta)
    q = torch.cat([w, xyz], dim=-1)  # wxyz格式
    return F.normalize(q, dim=-1)

# ────────────────── 变形矩阵构造方法 ──────────────────
# 极分解辅助函数
def polar_decompose(F_lin: torch.Tensor):
    """对 F_lin 执行右手系极分解, 返回旋转矩阵 R 与伸缩张量 S。"""
    U, S_diag, Vh = torch.linalg.svd(F_lin)

    # Kabsch 修正, 保证 det(R)=+1
    det = torch.linalg.det(torch.bmm(U, Vh))            # (n,)
    det = det.view(-1, 1, 1)                             # (n,1,1)

    # 若 det<0, 翻转 Vh 第三列即可消除反射
    Vh_fix = torch.where(det < 0, torch.cat([Vh[:, :2, :], -Vh[:, 2:, :]], dim=1), Vh)

    R = torch.bmm(U, Vh_fix)                             # (n,3,3)
    S_tilde = torch.bmm(R.transpose(-1, -2), F_lin)      # (n,3,3)
    return R, S_tilde

# ────────────────── 协方差矩阵构造方法 ──────────────────
def build_covariance_from_quats_scales(quats, scales):
    """使用四元数旋转和缩放构建协方差矩阵

    参数：
        quats: 四元数 (N,4) 格式 (w,x,y,z)
        scales: 对数缩放 (N,3)
    返回：
        cov: 协方差矩阵 (N,3,3)
    """
    N = quats.shape[0]
    device, dtype = quats.device, quats.dtype

    # 构建旋转矩阵 R
    R = torch.zeros(N, 3, 3, device=device, dtype=dtype)

    # 提取四元数分量
    w, x, y, z = quats.unbind(-1)

    # 计算旋转矩阵元素
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - z*w)
    R[:, 0, 2] = 2 * (x*z + y*w)
    R[:, 1, 0] = 2 * (x*y + z*w)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - x*w)
    R[:, 2, 0] = 2 * (x*z - y*w)
    R[:, 2, 1] = 2 * (y*z + x*w)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)

    # 构建对角缩放矩阵 S
    scales_exp = torch.exp(scales)
    S = torch.diag_embed(scales_exp)

    # 计算协方差 Cov = R * S^2 * R^T
    cov = torch.bmm(R, torch.bmm(S * S, R.transpose(1, 2)))

    return cov

# 从协方差矩阵分解出旋转和缩放
def decompose_covariance_to_rotation_scale(cov):
    """从协方差矩阵分解出旋转矩阵和缩放

    参数:
        cov: 协方差矩阵 (N,3,3)
    返回:
        R: 旋转矩阵 (N,3,3)
        scales: 对数缩放值 (N,3)
    """
    # 特征值分解
    eigvals, eigvecs = torch.linalg.eigh(cov)

    # 确保特征值为正
    eigvals = torch.clamp(eigvals, min=1e-8)

    # 对数缩放值
    scales = 0.5 * torch.log(eigvals)

    # 旋转矩阵是特征向量的矩阵
    R = eigvecs

    # 确保正交性和右手系
    det = torch.linalg.det(R)
    fix_reflection = torch.where(
        det < 0,
        torch.tensor([-1.0, -1.0, 1.0], device=cov.device),
        torch.tensor([1.0, 1.0, 1.0], device=cov.device)
    )
    R = torch.bmm(R, torch.diag_embed(fix_reflection))

    return R, scales

# 基于极分解构造 F, 同时返回 R 与对角缩放 (log 空间更新用)
def build_deformation_matrix_polar(J: torch.Tensor):
    """
    使用极分解方法构建变形矩阵F

    参数:
        J: 雅可比矩阵 (n,3,3)

    返回:
        F: 变形矩阵 (n,3,3)
        R: 旋转矩阵 (n,3,3)
        s_diag: 对角缩放 (n,3)
    """
    I = torch.eye(3, device=J.device).expand_as(J)
    F_linear = I + J                                 # 线性近似

    R, S_tilde = polar_decompose(F_linear)           # 极分解
    F_mat = torch.bmm(R, S_tilde)                    # 重组得到 F

    # 提取对角缩放, 确保其正值 (S_tilde 应为对称正定)
    s_diag = torch.diagonal(S_tilde, dim1=-2, dim2=-1)  # (n,3)

    return F_mat, R, s_diag

# ────────────────── 局部雅可比估计函数 ──────────────────
@torch.no_grad()
def estimate_jacobian_batch(points: torch.Tensor,
                            disp: torch.Tensor,
                            edges: torch.Tensor,
                            lambda_reg: float = 1e-6) -> torch.Tensor:
    """使用加权最小二乘 (与旧实现一致) 批量求解 Jacobian。

    参数:
        points: (N,3)  原始坐标
        disp:   (N,3)  位移
        edges:  (M,2)  点索引对 (局部索引)
        lambda_reg: 正则化强度
    返回:
        J: (N,3,3)  每点 Jacobian
    """
    device = points.device
    N = points.shape[0]

    # 生成有向边：每条无向边拆成两个有向边
    src = torch.cat([edges[:,0], edges[:,1]])  # (2M,)
    dst = torch.cat([edges[:,1], edges[:,0]])  # (2M,)

    p_src = points[src]               # (2M,3)
    p_dst = points[dst]

    p_src_def = p_src + disp[src]
    p_dst_def = p_dst + disp[dst]

    dp = p_dst - p_src               # (2M,3)
    dq = p_dst_def - p_src_def       # (2M,3)

    # 行向量权重
    w = torch.norm(dp, dim=1, keepdim=True).clamp_min(1e-6)  # (2M,1)
    P_w = dp * torch.sqrt(w)          # (2M,3)
    Y_w = (dq - dp) * torch.sqrt(w)   # (2M,3)

    # 计算每条边的 A = p^T p  和  B = p^T y
    A_edges = P_w.unsqueeze(2) * P_w.unsqueeze(1)  # (2M,3,3)
    B_edges = P_w.unsqueeze(2) * Y_w.unsqueeze(1)  # (2M,3,3)

    # 按源点 scatter 累加
    J_A = torch.zeros((N,3,3), device=device, dtype=points.dtype)
    J_B = torch.zeros((N,3,3), device=device, dtype=points.dtype)

    J_A.index_add_(0, src, A_edges)
    J_B.index_add_(0, src, B_edges)

    # 加正则项并求解 3x3 线性方程组
    I = torch.eye(3, device=device, dtype=points.dtype).expand(N,3,3)
    J_A_reg = J_A + lambda_reg * I

    # torch.linalg.solve 需要 A 非奇异；对无邻边点保持零 Jacobian
    mask = (torch.diagonal(J_A, dim1=-2, dim2=-1).sum(dim=-1) > 0)
    if mask.any():
        J_sol = torch.zeros_like(J_B)
        J_sol[mask] = torch.linalg.solve(J_A_reg[mask], J_B[mask])
        return J_sol.transpose(1,2)  # 结果与旧版一致 (N,3,3)
    else:
        return torch.zeros((N,3,3), device=device, dtype=points.dtype)

# ────────────────── 保存高斯模型函数 ──────────────────
def save_gaussian_splatting_checkpoint(path, means, quats, scales, opacities, sh0, shN):
    """保存高斯点云模型到checkpoint文件

    参数:
        path: 保存路径
        means: 点坐标 (N,3)
        quats: 四元数 (N,4) 格式 (w,x,y,z)
        scales: 对数缩放 (N,3)
        opacities: 不透明度logit值 (N,1)
        sh0: 球谐系数 DC项 (N,3,1)
        shN: 球谐系数 高阶项 (N,3,15)
    """
    # 四元数已经是wxyz格式，无需转换
    # 处理tensor维度
    opacities_1d = opacities.squeeze(-1)
    sh0_t = sh0.permute(0, 2, 1)
    shN_t = shN.permute(0, 2, 1)

    # 创建模型字典
    checkpoint = {
        "splats": {
            "means": means.cpu(),
            "quats": quats.cpu(),
            "scales": scales.cpu(),
            "opacities": opacities_1d.cpu(),
            "sh0": sh0_t.cpu(),
            "shN": shN_t.cpu(),
        }
    }
    torch.save(checkpoint, path)

# ────────────────── 应用变换函数 ──────────────────
def quaternion_inverse(q: torch.Tensor) -> torch.Tensor:
    """计算四元数的逆

    参数:
        q: 四元数 (...,4) 格式为 (w,x,y,z)
    返回:
        q_inv: 四元数的逆 (...,4)
    """
    # 四元数共轭
    q_conj = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)  # (w,-x,-y,-z)
    # 四元数的模
    norm_sq = torch.sum(q * q, dim=-1, keepdim=True)
    # 归一化，避免数值飘逸
    return F.normalize(q_conj / (norm_sq + 1e-12), dim=-1)

def apply_transformation_to_model(orig_model_path, transform_relation_path, output_path, device=torch.device("cpu")):
    """
    将变换关系应用到原始模型，生成变形后的模型

    参数:
        orig_model_path: 原始模型路径
        transform_relation_path: 变换关系路径
        output_path: 输出模型路径
        device: 计算设备
    """
    print(f"应用变换关系到原始模型...")

    # 加载原始模型
    orig_model = torch.load(orig_model_path, map_location=device)
    orig_splats = orig_model["splats"]

    # 加载变换关系
    transform_data = np.load(transform_relation_path)
    displacements = torch.tensor(transform_data["displacement_vectors"], dtype=torch.float32, device=device)

    # 根据变换关系模式选择不同的处理逻辑
    deform_mode = transform_data.get("deformation_mode", "tangent_only")

    if deform_mode == "delta":
        # 使用四元数差值模式（新版）
        q_delta = torch.tensor(transform_data["rotation_delta"], dtype=torch.float32, device=device)
        scale_factors = torch.tensor(transform_data["scale_factors"], dtype=torch.float32, device=device)

        # 复制原始模型数据
        deformed_splats = {k: v.clone() for k, v in orig_splats.items()}

        # 应用位移
        deformed_splats["means"] = orig_splats["means"] + displacements

        # 应用差值四元数 (已经是WXYZ格式)
        q_orig = orig_splats["quats"].clone()
        q_new = quaternion_multiply(q_delta, q_orig)
        deformed_splats["quats"] = q_new

        # 应用缩放
        scales_log = torch.log(torch.exp(orig_splats["scales"]) * scale_factors)
        deformed_splats["scales"] = scales_log
    else:
        # 兼容旧版切线旋转模式
        q_tangent = torch.tensor(transform_data["rotation_tangent"], dtype=torch.float32, device=device)
        scale_factors = torch.tensor(transform_data["scale_factors"], dtype=torch.float32, device=device)

        # 复制原始模型数据
        deformed_splats = {k: v.clone() for k, v in orig_splats.items()}

        # 应用位移
        deformed_splats["means"] = orig_splats["means"] + displacements

        # 应用切线旋转 (已经是WXYZ格式)
        q_orig = orig_splats["quats"].clone()
        q_new = quaternion_multiply(q_tangent, q_orig)
        deformed_splats["quats"] = q_new

        # 应用缩放
        scales_log = torch.log(torch.exp(orig_splats["scales"]) * scale_factors)
        deformed_splats["scales"] = scales_log

    # 保存变形后的模型
    torch.save({"splats": deformed_splats}, output_path)
    print(f"变形后的模型已保存至 {output_path}")

def main():
    global args, DEVICE

    parser = argparse.ArgumentParser(
        description="处理高斯点云分组、合并与应用变换的一体化工具"
    )
    parser.add_argument("--points-file", type=str, required=True, help="点云数据文件路径")
    parser.add_argument("--lines-file", type=str, required=True, help="线段数据文件路径")
    parser.add_argument("--orig-model-out", type=str, default="results/merged_original_model.pt", help="原始模型输出路径")
    parser.add_argument("--deformed-model-out", type=str, default="results/merged_deformed_model.pt", help="变形模型输出路径")
    parser.add_argument("--transform-out", type=str, default="results/merged_transform_relation.npz", help="变换关系输出路径")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="计算设备, auto 表示优先使用 CUDA, 若不可用则退回 CPU",
    )
    args = parser.parse_args()

    Path(args.orig_model_out).parent.mkdir(exist_ok=True, parents=True)
    Path(args.deformed_model_out).parent.mkdir(exist_ok=True, parents=True)
    Path(args.transform_out).parent.mkdir(exist_ok=True, parents=True)

    if args.device == "cuda":
        DEVICE = torch.device("cuda")
    elif args.device == "cpu":
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {DEVICE}")
    torch.set_grad_enabled(False)

    # 以下为生成变换关系模式
    print("使用变形构造方法: 极分解(polar decomposition)")

    # 检查输入文件是否存在
    points_file_path = Path(args.points_file)
    lines_file_path = Path(args.lines_file)

    if not points_file_path.exists():
        print(f"错误: 找不到点云数据文件 {points_file_path}")
        return

    if not lines_file_path.exists():
        print(f"错误: 找不到线段数据文件 {lines_file_path}")
        return

    print(f"加载点云数据: {points_file_path}")
    print(f"加载线段数据: {lines_file_path}")

    pts_raw = np.loadtxt(points_file_path, comments="#")
    lines_raw = np.loadtxt(lines_file_path, comments="#", dtype=int)

    points_all = torch.tensor(pts_raw[:, 1:4], dtype=torch.float32, device=DEVICE)
    disp_all = torch.tensor(pts_raw[:, 4:7], dtype=torch.float32, device=DEVICE)

    # 位移后的点坐标
    points_deformed = points_all + disp_all

    # group 信息在 lines.txt 第 3 列
    unique_groups = np.unique(lines_raw[:, 3])
    print("检测到 group 数:", len(unique_groups))

    # 构建点的邻接关系 (用于后面计算切线旋转)
    neighbors = {}
    for line in lines_raw:
        p1, p2 = line[1], line[2]  # 线段的两端点索引
        if p1 not in neighbors: neighbors[p1] = []
        if p2 not in neighbors: neighbors[p2] = []
        neighbors[p1].append(p2)
        neighbors[p2].append(p1)

    # 在遍历分组前创建全局容器，按原始点索引顺序预分配
    num_points_total = pts_raw.shape[0]

    # 初始化全局模型张量
    means_global = torch.zeros((num_points_total, 3), dtype=torch.float32, device=DEVICE)
    quats_global = torch.zeros((num_points_total, 4), dtype=torch.float32, device=DEVICE)
    quats_global[:, 0] = 1.0  # 初始化为单位四元数 (w=1)
    scales_log_global = torch.zeros((num_points_total, 3), dtype=torch.float32, device=DEVICE)
    op_logit_global = torch.zeros((num_points_total, 1), dtype=torch.float32, device=DEVICE)
    sh0_global = torch.zeros((num_points_total, 3, 1), dtype=torch.float32, device=DEVICE)
    shN_global = torch.zeros((num_points_total, 3, 15), dtype=torch.float32, device=DEVICE)

    # 初始化变形后模型张量
    means_def_global = torch.zeros((num_points_total, 3), dtype=torch.float32, device=DEVICE)
    quats_def_global = torch.zeros((num_points_total, 4), dtype=torch.float32, device=DEVICE)
    quats_def_global[:, 0] = 1.0  # 初始化为单位四元数 (w=1)

    # 变换关系全局容器
    q_tangent_global = torch.zeros((num_points_total, 4), dtype=torch.float32, device=DEVICE)
    q_tangent_global[:, 0] = 1.0  # 初始化为单位四元数 (w=1)

    q_delta_global = torch.zeros((num_points_total, 4), dtype=torch.float32, device=DEVICE)
    q_delta_global[:, 0] = 1.0  # 初始化为单位四元数 (w=1)

    # 对角缩放因子，初始化为 1
    scale_factors_global = torch.ones((num_points_total, 3), dtype=torch.float32, device=DEVICE)

    # 添加每个点所属的组信息数组
    point_groups = np.zeros(num_points_total, dtype=np.int32)  # 默认为0

    # 收集变换关系用于导出
    F_col, J_col, D_col = [], [], []

    # ────────────────── 遍历分组 ──────────────────
    # 使用 tqdm 进度条显示当前处理到的分组索引
    for g in tqdm(unique_groups, desc="Processing groups"):
        mask = lines_raw[:, 3] == g
        group_lines_np = lines_raw[mask][:, 1:3]
        group_pts_idx = np.unique(group_lines_np)

        # 建索引映射
        idx_map = {old: new for new, old in enumerate(group_pts_idx)}
        group_lines = torch.tensor([[idx_map[a], idx_map[b]] for a, b in group_lines_np], dtype=torch.long, device=DEVICE)
        points = points_all[group_pts_idx]
        disp = disp_all[group_pts_idx]

        N = points.shape[0]

        # ---------- 根据线段方向计算方向 ----------
        # 统计组内所有线段方向, 计算平均方向
        dirs = points_all[group_lines_np[:,1]] - points_all[group_lines_np[:,0]]  # (M,3)
        dirs = F.normalize(dirs, dim=-1)

        # 改进：将所有方向映射到同一半球后再平均，避免正反方向抵消
        ref_dir = dirs[0]  # 取第一条线段的方向作为参考
        dirs_aligned = torch.where(
            (dirs @ ref_dir.unsqueeze(-1)) < 0,  # 点积小于0表示与参考方向相反
            -dirs,  # 翻转方向
            dirs    # 保持原方向
        )
        dir_mean = F.normalize(dirs_aligned.mean(dim=0, keepdim=True), dim=-1)  # (1,3)

        # 参考轴
        ref_axis = torch.tensor([1.0,0.0,0.0], device=DEVICE)

        # ── 1) 原始方向：逐点计算局部切线方向四元数 ──
        quats_list = []
        for idx, orig_idx in enumerate(group_pts_idx):
            nbrs = neighbors.get(orig_idx, [])
            if len(nbrs) == 0:
                # 孤立点：保持单位四元数
                quats_list.append(torch.tensor([1, 0, 0, 0], device=DEVICE, dtype=torch.float32))
                continue

            # 收集本点邻接边方向
            dirs_pt = points_all[nbrs] - points[idx]          # (k,3)
            dirs_pt = F.normalize(dirs_pt, dim=-1)

            # 半球对齐
            if len(dirs_pt) > 0:
                ref = dirs_pt[0]
                dirs_pt = torch.where((dirs_pt @ ref).unsqueeze(-1) < 0, -dirs_pt, dirs_pt)
                tangent = F.normalize(dirs_pt.mean(dim=0), dim=-1)
                quat_i = quaternion_from_vectors(ref_axis, tangent)
            else:
                # 极少数情况：邻接点距离为零
                quat_i = torch.tensor([1, 0, 0, 0], device=DEVICE, dtype=torch.float32)

            quats_list.append(F.normalize(quat_i, dim=0))

        quats = torch.stack(quats_list, dim=0)                # (N,4) 逐点四元数

        # 设置各向异性 scale: 第一轴最长 (沿 dir_mean)
        scale_vals = torch.tensor([[0.03, 0.005, 0.005]], device=DEVICE)          # 长轴, 短轴, 短轴
        scales_log = torch.log(scale_vals).repeat(N,1)
        op_logit = torch.logit(torch.tensor([[0.8]], device=DEVICE).repeat(N,1))
        sh0 = torch.full((N,3,1),0.5,device=DEVICE)
        shN = torch.zeros(N,3,15,device=DEVICE)

        # --- 使用向量化方法估计 Jacobian ---
        J_u = estimate_jacobian_batch(points, disp, group_lines, lambda_reg=LAMBDA_REG)

        F_mat, R_scgd, s_diag = build_deformation_matrix_polar(J_u)

        means_new = points + disp

        # ---------- 先计算切线平均旋转 ----------
        q_tangent_list = []
        for idx, orig_idx in enumerate(group_pts_idx):
            nbrs = neighbors.get(orig_idx, [])
            if len(nbrs) == 0:
                q_tangent_list.append(torch.tensor([1, 0, 0, 0], device=DEVICE, dtype=torch.float32))
                continue

            omega_accum = torch.zeros(3, device=DEVICE, dtype=torch.float32)
            cnt = 0
            for nb in nbrs:
                v0 = points_all[nb] - points[idx]               # 原始方向
                v1 = points_deformed[nb] - means_new[idx]       # 变形后方向

                if torch.norm(v0) < 1e-8 or torch.norm(v1) < 1e-8:
                    continue

                q_nb = quaternion_from_vectors(v0, v1)
                omega_accum += quat_log(q_nb)
                cnt += 1

            if cnt == 0:
                q_tangent_list.append(torch.tensor([1, 0, 0, 0], device=DEVICE, dtype=torch.float32))
            else:
                omega_avg = omega_accum / cnt
                q_avg = quat_exp(omega_avg.unsqueeze(0))[0]
                q_tangent_list.append(q_avg)

        q_tangent = torch.stack(q_tangent_list)                      # (N,4)

        # ── 2) 用计算好的q_tangent更新四元数 ──
        quats_def = quaternion_multiply(q_tangent, quats)  # (N,4)
        q_delta = q_tangent                                # 差值就是切线旋转本身

        # ─ 保存本组（可选） ─
        if SAVE_GROUP_FILES:
            out_dir = Path(args.orig_model_out).parent / f"group_{g}"
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez(out_dir/"transform_relation.npz",
                    transformation_matrix_F=F_mat.cpu().numpy(),
                    jacobian_matrix=J_u.cpu().numpy(),
                    displacement_vectors=disp.cpu().numpy())
            save_gaussian_splatting_checkpoint(out_dir/"original_model.pt", points, quats, scales_log, op_logit, sh0, shN)

        # 收集变换关系数据用于导出
        F_col.append(F_mat.cpu().numpy())
        J_col.append(J_u.cpu().numpy())
        D_col.append(disp.cpu().numpy())

        # --- 直接写入全局模型和变换容器，使用原始索引 ---
        for orig_idx, new_idx in idx_map.items():
            # 原始模型数据
            means_global[orig_idx] = points[new_idx]
            quats_global[orig_idx] = quats[new_idx]
            scales_log_global[orig_idx] = scales_log[new_idx]
            op_logit_global[orig_idx] = op_logit[new_idx]
            sh0_global[orig_idx] = sh0[new_idx]
            shN_global[orig_idx] = shN[new_idx]

            # 变形后模型数据
            means_def_global[orig_idx] = means_new[new_idx]
            quats_def_global[orig_idx] = quats_def[new_idx]

            # 变换关系
            q_tangent_global[orig_idx] = q_tangent[new_idx]
            q_delta_global[orig_idx] = q_delta[new_idx]
            scale_factors_global[orig_idx] = torch.abs(s_diag[new_idx])  # 或用clamp: torch.clamp(s_diag[new_idx], min=1e-6)

            # 组信息
            point_groups[orig_idx] = g

    # ────────────────── 处理和保存模型 ──────────────────
    # 准备原始模型数据
    orig_splats = {
        "means": means_global.cpu(),
        "quats": quats_global.cpu(),
        "scales": scales_log_global.cpu(),
        "opacities": op_logit_global.squeeze(-1).cpu(),
        "sh0": sh0_global.permute(0, 2, 1).cpu(),
        "shN": shN_global.permute(0, 2, 1).cpu(),
        "groups": torch.tensor(point_groups, dtype=torch.int64)
    }

    # 保存原始模型
    torch.save({"splats": orig_splats}, Path(args.orig_model_out))

    # 准备变形后模型数据
    # 计算变形后对数缩放值
    scales_log_def = torch.log(torch.exp(scales_log_global) * scale_factors_global)

    def_splats = {
        "means": means_def_global.cpu(),
        "quats": quats_def_global.cpu(),  # 直接使用计算好的变形后四元数
        "scales": scales_log_def.cpu(),  # 应用缩放因子
        "opacities": op_logit_global.squeeze(-1).cpu(),
        "sh0": sh0_global.permute(0, 2, 1).cpu(),
        "shN": shN_global.permute(0, 2, 1).cpu(),
        "groups": torch.tensor(point_groups, dtype=torch.int64)
    }

    # 保存变形后模型
    torch.save({"splats": def_splats}, Path(args.deformed_model_out))

    # 保存包含四元数差值、缩放和位移的变换关系
    np.savez(Path(args.transform_out),
            displacement_vectors=disp_all.cpu().numpy(),
            rotation_delta=q_delta_global.cpu().numpy(),
            scale_factors=scale_factors_global.cpu().numpy(),
            point_groups=point_groups,
            deformation_mode="delta")
    print("✓ 原始模型、变形模型和变换关系保存完成，总点数:", orig_splats["means"].shape[0])

    print(f"原始模型: {args.orig_model_out}")
    print(f"变形模型: {args.deformed_model_out}")
    print(f"变换关系: {args.transform_out}")

if __name__ == "__main__":
    main()
