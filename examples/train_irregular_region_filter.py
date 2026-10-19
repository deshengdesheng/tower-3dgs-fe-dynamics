#!/usr/bin/env python3
"""
不规则 ROI 区域筛选训练脚本。

用途：
    代替规则 bbox，把“输电塔及其附近结构”的点云作为 ROI，形成贴合结构的不规则三维区域。
    原始 examples/train_region_filter.py 不会被修改。

运行示例：
    CKPT_PATH=/path/to/checkpoint.pt \
    DATA_DIR=/path/to/colmap_dataset \
    RESULT_DIR=/path/to/output \
    ROI_POINTCLOUD_PATH=/path/to/registered_roi.ply \
    python examples/train_irregular_region_filter.py

重要：
    ROI 点云必须与 3DGS checkpoint 处于同一坐标系。
    如果使用 ContextCapture/外部点云，需要先完成到 3DGS 坐标系的配准/相似变换。
"""

from __future__ import annotations

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

from examples.resume_training import ResumeConfig, main
from gsplat.distributed import cli
from gsplat.strategy.irregular_region_strategy import IrregularRegionFilteredStrategy


def _getenv_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _getenv_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _getenv_int_list(name: str, default: list[int]) -> list[int]:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    return [int(item) for item in value.replace(",", " ").split()]


def _getenv_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "y", "on"}


CKPT_PATH = os.environ.get(
    "CKPT_PATH",
    "",
)
DATA_DIR = os.environ.get(
    "DATA_DIR",
    "",
)
RESULT_DIR = os.environ.get(
    "RESULT_DIR",
    "",
)
ROI_POINTCLOUD_PATH = os.environ.get(
    "ROI_POINTCLOUD_PATH",
    "",
)
ROI_CACHE_PATH = os.environ.get("ROI_CACHE_PATH", "")

DATA_FACTOR = _getenv_int("DATA_FACTOR", 4)
MAX_STEPS = _getenv_int("MAX_STEPS", 1_000_000)
REFINE_START_ITER = _getenv_int("REFINE_START_ITER", 402_200)
REFINE_STOP_ITER = _getenv_int("REFINE_STOP_ITER", 650_000)
SAVE_STEPS = _getenv_int_list(
    "SAVE_STEPS",
    [431_000, 500_000, 600_000, 700_000, 800_000, 900_000, 1_000_000],
)
EVAL_STEPS = _getenv_int_list("EVAL_STEPS", [])
LOAD_OPTIMIZER_STATE = _getenv_bool("LOAD_OPTIMIZER_STATE", True)
LOAD_STRATEGY_STATE = _getenv_bool("LOAD_STRATEGY_STATE", True)

ROI_RADIUS = _getenv_float("ROI_RADIUS", 0.03)
ROI_VOXEL_SIZE = _getenv_float("ROI_VOXEL_SIZE", max(ROI_RADIUS / 2.0, 1e-4))
ROI_MAX_POINTS = _getenv_int("ROI_MAX_POINTS", 200_000)

ASPECT_RATIO_PERCENTILE = _getenv_int("ASPECT_RATIO_PERCENTILE", 20)
VOLUME_PERCENTILE = _getenv_int("VOLUME_PERCENTILE", 20)
REGION_GROWTH_LIMIT = _getenv_float("REGION_GROWTH_LIMIT", 4.0)
GLOBAL_GROWTH_LIMIT = _getenv_float("GLOBAL_GROWTH_LIMIT", 0.0)
GLOBAL_MAX_GAUSSIANS = _getenv_int("GLOBAL_MAX_GAUSSIANS", 0)
OUTSIDE_CONSERVATIVE_FACTOR = _getenv_int("OUTSIDE_CONSERVATIVE_FACTOR", 4)
OUTSIDE_GRAD_MULTIPLIER = _getenv_float("OUTSIDE_GRAD_MULTIPLIER", 2.0)

GROW_GRAD2D = _getenv_float("GROW_GRAD2D", 0.0002)
GROW_SCALE3D = _getenv_float("GROW_SCALE3D", 0.01)
GROW_SCALE2D = _getenv_float("GROW_SCALE2D", 0.05)
PRUNE_OPA = _getenv_float("PRUNE_OPA", 0.003)
PRUNE_SCALE3D = _getenv_float("PRUNE_SCALE3D", 0.1)
PRUNE_SCALE2D = _getenv_float("PRUNE_SCALE2D", 0.15)
REFINE_EVERY = _getenv_int("REFINE_EVERY", 100)
# 后期 ROI 微调继续使用 opacity reset 容易把成熟模型打透明；0 表示禁用。
RESET_EVERY = _getenv_int("RESET_EVERY", 0)
REFINE_SCALE2D_STOP_ITER = _getenv_int("REFINE_SCALE2D_STOP_ITER", 0)

REGION_PRUNE_OPA_MULTIPLIER = _getenv_float("REGION_PRUNE_OPA_MULTIPLIER", 1.0)
OUTSIDE_PRUNE_OPA_MULTIPLIER = _getenv_float("OUTSIDE_PRUNE_OPA_MULTIPLIER", 0.5)
REGION_PRUNE_SCALE3D_MULTIPLIER = _getenv_float("REGION_PRUNE_SCALE3D_MULTIPLIER", 1.5)
OUTSIDE_PRUNE_SCALE3D_MULTIPLIER = _getenv_float("OUTSIDE_PRUNE_SCALE3D_MULTIPLIER", 2.0)
OUTSIDE_PRUNE_SCALE2D_MULTIPLIER = _getenv_float("OUTSIDE_PRUNE_SCALE2D_MULTIPLIER", 2.0)
MAX_PRUNE_FRACTION = _getenv_float("MAX_PRUNE_FRACTION", 0.0)
MAX_REGION_PRUNE_FRACTION = _getenv_float("MAX_REGION_PRUNE_FRACTION", 0.0)
MAX_OUTSIDE_PRUNE_FRACTION = _getenv_float("MAX_OUTSIDE_PRUNE_FRACTION", 0.0)
OUTSIDE_BUDGET_PRUNE_FRACTION = _getenv_float("OUTSIDE_BUDGET_PRUNE_FRACTION", 0.0)
MIN_GROWTH_BUDGET = _getenv_int("MIN_GROWTH_BUDGET", 0)
MIN_OUTSIDE_POINT_RATIO = _getenv_float("MIN_OUTSIDE_POINT_RATIO", 0.18)
ROI_FIXED_FINETUNE = _getenv_bool("ROI_FIXED_FINETUNE", False)
BBOX_FILE = os.environ.get("BBOX_FILE", "")
BBOX_LOSS_WEIGHT = _getenv_float("BBOX_LOSS_WEIGHT", 0.0)
BBOX_LOSS_PAD_PIXELS = _getenv_int("BBOX_LOSS_PAD_PIXELS", 20)
BBOX_LOSS_MIN_SIZE = _getenv_int("BBOX_LOSS_MIN_SIZE", 32)
BBOX_EDGE_LOSS_WEIGHT = _getenv_float("BBOX_EDGE_LOSS_WEIGHT", 0.0)
BBOX_SSIM_LOSS_WEIGHT = _getenv_float("BBOX_SSIM_LOSS_WEIGHT", 0.0)
ROI_SCALE_REG_WEIGHT = _getenv_float("ROI_SCALE_REG_WEIGHT", 0.0)
BBOX_LOSS_INTERVAL = _getenv_int("BBOX_LOSS_INTERVAL", 1)
BBOX_STRUCTURE_LOSS_INTERVAL = _getenv_int("BBOX_STRUCTURE_LOSS_INTERVAL", 1)
ROI_SCALE_REG_INTERVAL = _getenv_int("ROI_SCALE_REG_INTERVAL", 1)
ROI_SCALE_REG_STATIC_MASK = _getenv_bool("ROI_SCALE_REG_STATIC_MASK", False)
NORMALIZE_WORLD_SPACE = _getenv_bool("NORMALIZE_WORLD_SPACE", True)
USE_BILATERAL_GRID = _getenv_bool("USE_BILATERAL_GRID", True)
PATCH_SIZE = os.environ.get("PATCH_SIZE", "").strip()


if __name__ == "__main__":
    required_paths = {
        "CKPT_PATH": CKPT_PATH,
        "DATA_DIR": DATA_DIR,
        "RESULT_DIR": RESULT_DIR,
    }
    missing = [name for name, value in required_paths.items() if not value]
    if missing:
        raise SystemExit(
            "Set the required paths with environment variables: " + ", ".join(missing)
        )
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"找不到初始 checkpoint: {CKPT_PATH}")
    if ROI_CACHE_PATH:
        roi_input_exists = os.path.exists(ROI_CACHE_PATH)
        roi_input_label = ROI_CACHE_PATH
    else:
        roi_input_exists = os.path.exists(ROI_POINTCLOUD_PATH)
        roi_input_label = ROI_POINTCLOUD_PATH

    if not roi_input_exists:
        raise FileNotFoundError(
            "找不到不规则 ROI 输入文件: "
            f"{roi_input_label}\n"
            "请设置 ROI_CACHE_PATH 指向 ROI 体素缓存，或设置 ROI_POINTCLOUD_PATH 指向已配准到 3DGS 坐标系的点云。"
        )

    cfg = ResumeConfig()
    cfg.resume_ckpt = CKPT_PATH
    cfg.data_dir = DATA_DIR
    cfg.result_dir = RESULT_DIR
    cfg.load_optimizer_state = LOAD_OPTIMIZER_STATE
    cfg.load_strategy_state = LOAD_STRATEGY_STATE
    cfg.eval_steps = EVAL_STEPS
    cfg.save_steps = SAVE_STEPS
    cfg.max_steps = MAX_STEPS
    cfg.disable_viewer = True
    cfg.antialiased = False
    cfg.random_bkgd = True
    cfg.normalize_world_space = NORMALIZE_WORLD_SPACE
    cfg.use_bilateral_grid = USE_BILATERAL_GRID
    cfg.patch_size = int(PATCH_SIZE) if PATCH_SIZE else None
    cfg.data_factor = DATA_FACTOR
    cfg.bbox_file = BBOX_FILE
    cfg.bbox_loss_weight = BBOX_LOSS_WEIGHT
    cfg.bbox_loss_pad_pixels = BBOX_LOSS_PAD_PIXELS
    cfg.bbox_loss_min_size = BBOX_LOSS_MIN_SIZE
    cfg.bbox_edge_loss_weight = BBOX_EDGE_LOSS_WEIGHT
    cfg.bbox_ssim_loss_weight = BBOX_SSIM_LOSS_WEIGHT
    cfg.roi_scale_reg_weight = ROI_SCALE_REG_WEIGHT
    cfg.bbox_loss_interval = BBOX_LOSS_INTERVAL
    cfg.bbox_structure_loss_interval = BBOX_STRUCTURE_LOSS_INTERVAL
    cfg.roi_scale_reg_interval = ROI_SCALE_REG_INTERVAL
    cfg.roi_scale_reg_static_mask = ROI_SCALE_REG_STATIC_MASK

    strategy_kwargs = dict(
        aspect_ratio_percentile=ASPECT_RATIO_PERCENTILE,
        volume_percentile=VOLUME_PERCENTILE,
        refine_start_iter=REFINE_START_ITER,
        refine_stop_iter=REFINE_STOP_ITER,
        refine_every=REFINE_EVERY,
        reset_every=RESET_EVERY,
        refine_scale2d_stop_iter=REFINE_SCALE2D_STOP_ITER,
        disable_pruning=False,
        region_growth_limit=REGION_GROWTH_LIMIT,
        global_growth_limit=GLOBAL_GROWTH_LIMIT,
        global_max_gaussians=GLOBAL_MAX_GAUSSIANS,
        outside_conservative_factor=OUTSIDE_CONSERVATIVE_FACTOR,
        outside_grad_multiplier=OUTSIDE_GRAD_MULTIPLIER,
        region_prune_opa_multiplier=REGION_PRUNE_OPA_MULTIPLIER,
        outside_prune_opa_multiplier=OUTSIDE_PRUNE_OPA_MULTIPLIER,
        region_prune_scale3d_multiplier=REGION_PRUNE_SCALE3D_MULTIPLIER,
        outside_prune_scale3d_multiplier=OUTSIDE_PRUNE_SCALE3D_MULTIPLIER,
        outside_prune_scale2d_multiplier=OUTSIDE_PRUNE_SCALE2D_MULTIPLIER,
        max_prune_fraction=MAX_PRUNE_FRACTION,
        max_region_prune_fraction=MAX_REGION_PRUNE_FRACTION,
        max_outside_prune_fraction=MAX_OUTSIDE_PRUNE_FRACTION,
        outside_budget_prune_fraction=OUTSIDE_BUDGET_PRUNE_FRACTION,
        min_growth_budget=MIN_GROWTH_BUDGET,
        min_outside_point_ratio=MIN_OUTSIDE_POINT_RATIO,
        disable_refine=ROI_FIXED_FINETUNE,
        grow_grad2d=GROW_GRAD2D,
        grow_scale3d=GROW_SCALE3D,
        grow_scale2d=GROW_SCALE2D,
        prune_opa=PRUNE_OPA,
        prune_scale3d=PRUNE_SCALE3D,
        prune_scale2d=PRUNE_SCALE2D,
        verbose=True,
    )
    if ROI_CACHE_PATH:
        cfg.strategy = IrregularRegionFilteredStrategy.from_cache(
            roi_cache_path=ROI_CACHE_PATH,
            **strategy_kwargs,
        )
    else:
        cfg.strategy = IrregularRegionFilteredStrategy(
            roi_pointcloud_path=ROI_POINTCLOUD_PATH,
            roi_radius=ROI_RADIUS,
            roi_voxel_size=ROI_VOXEL_SIZE,
            roi_max_points=ROI_MAX_POINTS,
            **strategy_kwargs,
        )

    print("✅ 使用不规则 ROI 区域筛选策略")
    print(f"   初始 checkpoint: {CKPT_PATH}")
    if ROI_CACHE_PATH:
        print(f"   ROI 缓存: {ROI_CACHE_PATH}")
    else:
        print(f"   ROI 点云: {ROI_POINTCLOUD_PATH}")
    print(f"   输出目录: {RESULT_DIR}")
    print(f"   ROI 半径: {ROI_RADIUS}, ROI 体素: {ROI_VOXEL_SIZE}")
    print(f"   ROI 内触发: 长宽比前 {ASPECT_RATIO_PERCENTILE}% + 体积前 {VOLUME_PERCENTILE}%")
    print(f"   ROI 外保守系数: {OUTSIDE_CONSERVATIVE_FACTOR}")
    print(f"   ROI 外梯度倍率: {OUTSIDE_GRAD_MULTIPLIER}")
    print(f"   ROI 增长上限: 初始点数 x {REGION_GROWTH_LIMIT}")
    print(f"   全局增长上限: 初始总点数 x {GLOBAL_GROWTH_LIMIT if GLOBAL_GROWTH_LIMIT > 0 else '不限制'}")
    if GLOBAL_MAX_GAUSSIANS > 0:
        print(f"   全局绝对点数上限: {GLOBAL_MAX_GAUSSIANS:,}")
    print(f"   细化区间: {REFINE_START_ITER} → {REFINE_STOP_ITER}")
    print(f"   保存步数: {SAVE_STEPS}")
    print(f"   评估步数: {EVAL_STEPS}")
    print(f"   加载优化器状态: {LOAD_OPTIMIZER_STATE}")
    print(f"   加载策略状态: {LOAD_STRATEGY_STATE}")
    print(f"   ROI固定点数精修: {'启用' if ROI_FIXED_FINETUNE else '禁用'}")
    print(f"   opacity reset: {'禁用' if RESET_EVERY <= 0 else f'每 {RESET_EVERY} step'}")
    print(f"   grow_grad2d={GROW_GRAD2D}, prune_opa={PRUNE_OPA}, refine_every={REFINE_EVERY}")
    print(
        f"   剪枝倍率: ROI内opa x{REGION_PRUNE_OPA_MULTIPLIER}, "
        f"ROI外opa x{OUTSIDE_PRUNE_OPA_MULTIPLIER}, ROI外scale3d x{OUTSIDE_PRUNE_SCALE3D_MULTIPLIER}"
    )
    print(
        f"   单轮剪枝保护: total≤{MAX_PRUNE_FRACTION:.4g}, "
        f"ROI内≤{MAX_REGION_PRUNE_FRACTION:.4g}, ROI外≤{MAX_OUTSIDE_PRUNE_FRACTION:.4g}"
    )
    print(
        f"   ROI预算迁移: 每轮最多剪ROI外≤{OUTSIDE_BUDGET_PRUNE_FRACTION:.4g}, "
        f"最小增长预算={MIN_GROWTH_BUDGET}, ROI外保底比例={MIN_OUTSIDE_POINT_RATIO:.3g}"
    )
    if BBOX_LOSS_WEIGHT > 0.0:
        print(
            f"   BBox局部loss: weight={BBOX_LOSS_WEIGHT}, "
            f"edge={BBOX_EDGE_LOSS_WEIGHT}, ssim={BBOX_SSIM_LOSS_WEIGHT}, "
            f"interval={BBOX_LOSS_INTERVAL}, structure_interval={BBOX_STRUCTURE_LOSS_INTERVAL}, "
            f"pad={BBOX_LOSS_PAD_PIXELS}, min_size={BBOX_LOSS_MIN_SIZE}, bbox={BBOX_FILE}"
        )
    if ROI_SCALE_REG_WEIGHT > 0.0:
        print(
            f"   ROI尺度正则: weight={ROI_SCALE_REG_WEIGHT}, "
            f"interval={ROI_SCALE_REG_INTERVAL}, static_mask={ROI_SCALE_REG_STATIC_MASK}"
        )
    print("🚀 开始训练")

    cli(main, cfg, verbose=True)
