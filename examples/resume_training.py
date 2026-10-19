import os
import sys
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import argparse
import json
import math
import types
import tyro
from typing_extensions import Literal, assert_never
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from simple_trainer import Config, Runner
from gsplat.strategy import DefaultStrategy, MCMCStrategy, RegionFilteredStrategy
from gsplat.distributed import cli

# 继承Config类，添加检查点相关参数
@dataclass
class ResumeConfig(Config):
    # 检查点路径，从这个检查点继续训练
    resume_ckpt: str = ""
    # 起始步骤，如果为0则从检查点的步骤+1继续
    init_step: int = 0
    # 是否加载优化器状态，如果False则只加载模型参数
    load_optimizer_state: bool = True
    # 是否加载策略状态，如果False则重新初始化策略状态
    load_strategy_state: bool = True
    # BBox 局部损失权重；0 表示关闭
    bbox_loss_weight: float = 0.0
    # 3D bbox 信息文件，格式同 evaluate_static_comparison.py 使用的 bbox_info.json
    bbox_file: str = ""
    # 投影到图像后的 bbox 扩展像素
    bbox_loss_pad_pixels: int = 20
    # 过小 bbox crop 的最小尺寸
    bbox_loss_min_size: int = 32
    # BBox 局部边缘损失权重；用于强化结构边界和高频细节
    bbox_edge_loss_weight: float = 0.0
    # BBox 局部 SSIM 损失权重；用于提升局部结构相似度
    bbox_ssim_loss_weight: float = 0.0
    # ROI 内尺度正则权重；固定点数精修时抑制局部过大 Gaussian 造成的糊块
    roi_scale_reg_weight: float = 0.0
    # BBox L1 局部损失计算间隔；1 表示每步计算
    bbox_loss_interval: int = 1
    # BBox Edge/SSIM 局部结构损失计算间隔；1 表示每步计算
    bbox_structure_loss_interval: int = 1
    # ROI 尺度正则计算间隔；1 表示每步计算
    roi_scale_reg_interval: int = 1
    # 固定点数精修时缓存 ROI mask，避免每步重算百万级点的 ROI 归属
    roi_scale_reg_static_mask: bool = False

# 保存全局初始步骤
init_step = 0

# Sobel 卷积核按 device/dtype/channel 复用，避免每步重复创建小张量。
_SOBEL_KERNEL_CACHE: Dict[Tuple[str, Optional[int], torch.dtype, int], Tuple[torch.Tensor, torch.Tensor]] = {}


def load_bbox_corners(path: str) -> torch.Tensor:
    """读取 3D bbox，并生成 8 个角点。"""
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    bbox_min = torch.tensor(data["bbox_min"], dtype=torch.float32)
    bbox_max = torch.tensor(data["bbox_max"], dtype=torch.float32)
    xs = [bbox_min[0], bbox_max[0]]
    ys = [bbox_min[1], bbox_max[1]]
    zs = [bbox_min[2], bbox_max[2]]
    corners = [[x, y, z] for x in xs for y in ys for z in zs]
    return torch.tensor(corners, dtype=torch.float32)


@torch.no_grad()
def project_bbox_crop(
    corners: torch.Tensor,
    camtoworld: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    *,
    pad_pixels: int,
    min_size: int,
) -> Optional[Tuple[int, int, int, int]]:
    """把 3D bbox 投影到当前训练图像，返回 [x0,y0,x1,y1)。"""
    device = camtoworld.device
    dtype = camtoworld.dtype
    corners = corners.to(device=device, dtype=dtype)
    worldtocam = torch.linalg.inv(camtoworld)
    homogeneous = torch.cat(
        [corners, torch.ones((corners.shape[0], 1), device=device, dtype=dtype)],
        dim=1,
    )
    cam = (worldtocam @ homogeneous.T).T[:, :3]
    valid_depth = cam[:, 2] > 1e-6
    if int(valid_depth.sum().item()) < 2:
        return None

    cam = cam[valid_depth]
    projected = (K.to(dtype=dtype) @ cam.T).T
    uv = projected[:, :2] / projected[:, 2:3]
    finite = torch.isfinite(uv).all(dim=1)
    uv = uv[finite]
    if uv.shape[0] < 2:
        return None

    x0 = torch.floor(uv[:, 0].min()).item() - pad_pixels
    y0 = torch.floor(uv[:, 1].min()).item() - pad_pixels
    x1 = torch.ceil(uv[:, 0].max()).item() + pad_pixels
    y1 = torch.ceil(uv[:, 1].max()).item() + pad_pixels

    # 小投影框扩大到最小尺寸，避免局部 loss 只看极少像素。
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    box_w = max(x1 - x0, min_size)
    box_h = max(y1 - y0, min_size)
    x0 = cx - box_w / 2
    x1 = cx + box_w / 2
    y0 = cy - box_h / 2
    y1 = cy + box_h / 2

    x0_i = max(0, int(round(x0)))
    y0_i = max(0, int(round(y0)))
    x1_i = min(width, int(round(x1)))
    y1_i = min(height, int(round(y1)))
    if x1_i - x0_i < min_size or y1_i - y0_i < min_size:
        return None
    return x0_i, y0_i, x1_i, y1_i


def compute_sobel_magnitude(images: torch.Tensor) -> torch.Tensor:
    """计算图像 Sobel 梯度幅值，输入输出均为 [B,H,W,C]。"""
    channels = images.shape[-1]
    images_chw = images.permute(0, 3, 1, 2)
    cache_key = (images.device.type, images.device.index, images.dtype, channels)
    kernels = _SOBEL_KERNEL_CACHE.get(cache_key)
    if kernels is None:
        kernel_x = images.new_tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        kernel_y = images.new_tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        kernels = (
            kernel_x.repeat(channels, 1, 1, 1),
            kernel_y.repeat(channels, 1, 1, 1),
        )
        _SOBEL_KERNEL_CACHE[cache_key] = kernels
    kernel_x, kernel_y = kernels
    grad_x = F.conv2d(images_chw, kernel_x, padding=1, groups=channels)
    grad_y = F.conv2d(images_chw, kernel_y, padding=1, groups=channels)
    return torch.sqrt(grad_x.square() + grad_y.square() + 1e-8).permute(0, 2, 3, 1)


def compute_bbox_local_losses(
    colors: torch.Tensor,
    pixels: torch.Tensor,
    corners: torch.Tensor,
    camtoworlds: torch.Tensor,
    Ks: torch.Tensor,
    *,
    pad_pixels: int,
    min_size: int,
    use_edge_loss: bool,
    use_ssim_loss: bool,
    fused_ssim_fn,
) -> Tuple[Optional[Dict[str, torch.Tensor]], int]:
    """计算当前 batch 中可见 bbox crop 的局部重建/结构/边缘损失。"""
    height, width = pixels.shape[1:3]
    l1_losses = []
    edge_losses = []
    ssim_losses = []
    for batch_index in range(colors.shape[0]):
        crop = project_bbox_crop(
            corners,
            camtoworlds[batch_index],
            Ks[batch_index],
            width,
            height,
            pad_pixels=pad_pixels,
            min_size=min_size,
        )
        if crop is None:
            continue
        x0, y0, x1, y1 = crop
        # 局部 crop 保持梯度，只让 bbox 区域的图像误差直接驱动优化。
        crop_colors = colors[batch_index : batch_index + 1, y0:y1, x0:x1, :]
        crop_pixels = pixels[batch_index : batch_index + 1, y0:y1, x0:x1, :]
        l1_losses.append(torch.abs(crop_colors - crop_pixels).mean())
        if use_edge_loss:
            edge_losses.append(
                torch.abs(compute_sobel_magnitude(crop_colors) - compute_sobel_magnitude(crop_pixels)).mean()
            )
        if use_ssim_loss and crop_colors.shape[1] >= 11 and crop_colors.shape[2] >= 11:
            ssim_losses.append(
                1.0
                - fused_ssim_fn(
                    crop_colors.permute(0, 3, 1, 2),
                    crop_pixels.permute(0, 3, 1, 2),
                    padding="valid",
                )
            )

    if not l1_losses:
        return None, 0

    losses = {"l1": torch.stack(l1_losses).mean()}
    if edge_losses:
        losses["edge"] = torch.stack(edge_losses).mean()
    if ssim_losses:
        losses["ssim"] = torch.stack(ssim_losses).mean()
    return losses, len(l1_losses)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='预训练模型路径')
    parser.add_argument('--data_dir', type=str, required=True, help='用户提供的COLMAP数据目录')
    parser.add_argument('--result_dir', type=str, required=True, help='结果保存目录')
    parser.add_argument('--init_step', type=int, default=0, help='起始步骤')
    return parser.parse_args()

def set_config(args):
    """创建训练配置"""
    cfg = Config()
    
    # 基本设置
    cfg.data_dir = args.data_dir
    cfg.result_dir = args.result_dir
    
    # 高级参数配置，与原始训练脚本一致
    cfg.eval_steps = [15000, 30000, 60000, 90000]
    cfg.save_steps = [15000, 30000, 60000, 90000, 120000, 150000, 180000]
    cfg.max_steps = 200000
    cfg.disable_viewer = True
    cfg.antialiased = False
    cfg.random_bkgd = True
    cfg.use_bilateral_grid = True
    cfg.bilateral_grid_shape = (16, 16, 8)
    cfg.ssim_lambda = 0.5
    cfg.init_num_pts = 80000
    cfg.data_factor = 4
    
    # 设置策略参数
    cfg.strategy = DefaultStrategy(
        refine_start_iter=100,
        refine_stop_iter=60000,
        grow_scale2d=0.04,
        grow_scale3d=0.01,
        prune_opa=0.01,
        prune_scale3d=0.25,
        prune_scale2d=0.2,
        verbose=True
    )
    
    return cfg

# Monkey patch the train function to start from init_step
original_train = Runner.train

def new_train(self):
    cfg = self.cfg
    device = self.device
    world_rank = self.world_rank
    world_size = self.world_size
    bbox_loss_weight = float(getattr(cfg, "bbox_loss_weight", 0.0) or 0.0)
    bbox_loss_corners = getattr(self, "bbox_loss_corners", None)
    
    # Dump cfg.
    if world_rank == 0:
        with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
            import yaml
            yaml.dump(vars(cfg), f)
    
    max_steps = cfg.max_steps
    global init_step
    
    schedulers = [
        # means has a learning rate schedule, that end at 0.01 of the initial value
        torch.optim.lr_scheduler.ExponentialLR(
            self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
        ),
    ]
    if cfg.pose_opt:
        # pose optimization has a learning rate schedule
        schedulers.append(
            torch.optim.lr_scheduler.ExponentialLR(
                self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
            )
        )
    if cfg.use_bilateral_grid:
        # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
        schedulers.append(
            torch.optim.lr_scheduler.ChainedScheduler(
                [
                    torch.optim.lr_scheduler.LinearLR(
                        self.bil_grid_optimizers[0],
                        start_factor=0.01,
                        total_iters=1000,
                    ),
                    torch.optim.lr_scheduler.ExponentialLR(
                        self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                    ),
                ]
            )
        )
    
    import time
    import tqdm
    from fused_ssim import fused_ssim
    
    trainloader = torch.utils.data.DataLoader(
        self.trainset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )
    trainloader_iter = iter(trainloader)
    
    # Training loop.
    global_tic = time.time()
    pbar = tqdm.tqdm(range(init_step, max_steps))
    for step in pbar:
        if not cfg.disable_viewer:
            while self.viewer.state.status == "paused":
                time.sleep(0.01)
            self.viewer.lock.acquire()
            tic = time.time()

        try:
            data = next(trainloader_iter)
        except StopIteration:
            trainloader_iter = iter(trainloader)
            data = next(trainloader_iter)

        camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
        Ks = data["K"].to(device)  # [1, 3, 3]
        pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
        num_train_rays_per_step = (
            pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
        )
        image_ids = data["image_id"].to(device)
        masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
        if cfg.depth_loss:
            points = data["points"].to(device)  # [1, M, 2]
            depths_gt = data["depths"].to(device)  # [1, M]

        height, width = pixels.shape[1:3]

        if cfg.pose_noise:
            camtoworlds = self.pose_perturb(camtoworlds, image_ids)

        if cfg.pose_opt:
            camtoworlds = self.pose_adjust(camtoworlds, image_ids)

        # sh schedule
        sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

        # forward
        renders, alphas, info = self.rasterize_splats(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=sh_degree_to_use,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            image_ids=image_ids,
            render_mode="RGB+ED" if cfg.depth_loss else "RGB",
            masks=masks,
        )
        if renders.shape[-1] == 4:
            colors, depths = renders[..., 0:3], renders[..., 3:4]
        else:
            colors, depths = renders, None

        if cfg.use_bilateral_grid:
            grid_y, grid_x = torch.meshgrid(
                (torch.arange(height, device=self.device) + 0.5) / height,
                (torch.arange(width, device=self.device) + 0.5) / width,
                indexing="ij",
            )
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
            from lib_bilagrid import slice
            colors = slice(self.bil_grids, grid_xy, colors, image_ids)["rgb"]

        if cfg.random_bkgd:
            bkgd = torch.rand(1, 3, device=device)
            colors = colors + bkgd * (1.0 - alphas)

        self.cfg.strategy.step_pre_backward(
            params=self.splats,
            optimizers=self.optimizers,
            state=self.strategy_state,
            step=step,
            info=info,
        )

        # loss
        l1loss = F.l1_loss(colors, pixels)
        ssimloss = 1.0 - fused_ssim(
            colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
        )
        loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
        bbox_l1loss = None
        bbox_edgeloss = None
        bbox_ssimloss = None
        roi_scaleloss = None
        bbox_visible_count = 0
        bbox_loss_interval = max(1, int(getattr(cfg, "bbox_loss_interval", 1) or 1))
        bbox_structure_loss_interval = max(1, int(getattr(cfg, "bbox_structure_loss_interval", 1) or 1))
        compute_bbox_l1 = step % bbox_loss_interval == 0
        compute_bbox_structure = step % bbox_structure_loss_interval == 0
        if bbox_loss_weight > 0.0 and bbox_loss_corners is not None and compute_bbox_l1:
            bbox_losses, bbox_visible_count = compute_bbox_local_losses(
                colors,
                pixels,
                bbox_loss_corners,
                camtoworlds,
                Ks,
                pad_pixels=int(getattr(cfg, "bbox_loss_pad_pixels", 20)),
                min_size=int(getattr(cfg, "bbox_loss_min_size", 32)),
                # Edge/SSIM 比 L1 更耗时，允许低频计算以提高吞吐。
                use_edge_loss=(
                    compute_bbox_structure
                    and float(getattr(cfg, "bbox_edge_loss_weight", 0.0) or 0.0) > 0.0
                ),
                use_ssim_loss=(
                    compute_bbox_structure
                    and float(getattr(cfg, "bbox_ssim_loss_weight", 0.0) or 0.0) > 0.0
                ),
                fused_ssim_fn=fused_ssim,
            )
            if bbox_losses is not None:
                bbox_l1loss = bbox_losses["l1"]
                loss = loss + bbox_loss_weight * bbox_l1loss
                bbox_edgeloss = bbox_losses.get("edge")
                if bbox_edgeloss is not None:
                    loss = loss + float(getattr(cfg, "bbox_edge_loss_weight", 0.0) or 0.0) * bbox_edgeloss
                bbox_ssimloss = bbox_losses.get("ssim")
                if bbox_ssimloss is not None:
                    loss = loss + float(getattr(cfg, "bbox_ssim_loss_weight", 0.0) or 0.0) * bbox_ssimloss
        if cfg.depth_loss:
            # query depths from depth map
            points = torch.stack(
                [
                    points[:, :, 0] / (width - 1) * 2 - 1,
                    points[:, :, 1] / (height - 1) * 2 - 1,
                ],
                dim=-1,
            )  # normalize to [-1, 1]
            grid = points.unsqueeze(2)  # [1, M, 1, 2]
            depths = F.grid_sample(
                depths.permute(0, 3, 1, 2), grid, align_corners=True
            )  # [1, 1, M, 1]
            depths = depths.squeeze(3).squeeze(1)  # [1, M]
            # calculate loss in disparity space
            disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
            disp_gt = 1.0 / depths_gt  # [1, M]
            depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
            loss += depthloss * cfg.depth_lambda
        if cfg.use_bilateral_grid:
            from lib_bilagrid import total_variation_loss
            tvloss = 10 * total_variation_loss(self.bil_grids.grids)
            loss += tvloss

        # regularizations
        if cfg.opacity_reg > 0.0:
            loss = (
                loss
                + cfg.opacity_reg
                * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
            )
        if cfg.scale_reg > 0.0:
            loss = (
                loss
                + cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()
            )
        roi_scale_reg_weight = float(getattr(cfg, "roi_scale_reg_weight", 0.0) or 0.0)
        roi_scale_reg_interval = max(1, int(getattr(cfg, "roi_scale_reg_interval", 1) or 1))
        if (
            roi_scale_reg_weight > 0.0
            and step % roi_scale_reg_interval == 0
            and hasattr(self.cfg.strategy, "_get_region_mask")
        ):
            static_roi_mask = bool(getattr(cfg, "roi_scale_reg_static_mask", False))
            region_mask = getattr(self, "_cached_roi_scale_reg_mask", None) if static_roi_mask else None
            if region_mask is None or region_mask.shape[0] != len(self.splats["means"]):
                # 固定点数精修不增删点时，ROI 归属变化很小，可缓存 mask 降低每步开销。
                region_mask = self.cfg.strategy._get_region_mask(self.splats["means"])
                if static_roi_mask:
                    self._cached_roi_scale_reg_mask = region_mask.detach()
            if bool(region_mask.any().item()):
                # 固定点数精修时只压 ROI 内过大尺度，减少局部糊块，不改变点数。
                roi_scaleloss = torch.exp(self.splats["scales"][region_mask]).max(dim=-1).values.mean()
                loss = loss + roi_scale_reg_weight * roi_scaleloss

        loss.backward()

        desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
        if bbox_l1loss is not None:
            desc += f"bbox_l1={bbox_l1loss.item():.4f}| "
        if bbox_edgeloss is not None:
            desc += f"bbox_edge={bbox_edgeloss.item():.4f}| "
        if bbox_ssimloss is not None:
            desc += f"bbox_ssim={bbox_ssimloss.item():.4f}| "
        if roi_scaleloss is not None:
            desc += f"roi_scale={roi_scaleloss.item():.4f}| "
        if cfg.depth_loss:
            desc += f"depth loss={depthloss.item():.6f}| "
        if cfg.pose_opt and cfg.pose_noise:
            # monitor the pose error if we inject noise
            pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
            desc += f"pose err={pose_err.item():.6f}| "
        pbar.set_description(desc)

        # write images (gt and render)
        # if world_rank == 0 and step % 800 == 0:
        #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
        #     canvas = canvas.reshape(-1, *canvas.shape[2:])
        #     imageio.imwrite(
        #         f"{self.render_dir}/train_rank{self.world_rank}.png",
        #         (canvas * 255).astype(np.uint8),
        #     )

        if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
            mem = torch.cuda.max_memory_allocated() / 1024**3
            self.writer.add_scalar("train/loss", loss.item(), step)
            self.writer.add_scalar("train/l1loss", l1loss.item(), step)
            self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
            self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
            self.writer.add_scalar("train/mem", mem, step)
            if bbox_l1loss is not None:
                self.writer.add_scalar("train/bbox_l1loss", bbox_l1loss.item(), step)
                self.writer.add_scalar("train/bbox_visible_count", bbox_visible_count, step)
            if bbox_edgeloss is not None:
                self.writer.add_scalar("train/bbox_edgeloss", bbox_edgeloss.item(), step)
            if bbox_ssimloss is not None:
                self.writer.add_scalar("train/bbox_ssimloss", bbox_ssimloss.item(), step)
            if roi_scaleloss is not None:
                self.writer.add_scalar("train/roi_scaleloss", roi_scaleloss.item(), step)
            if cfg.depth_loss:
                self.writer.add_scalar("train/depthloss", depthloss.item(), step)
            if cfg.use_bilateral_grid:
                self.writer.add_scalar("train/tvloss", tvloss.item(), step)
            if cfg.tb_save_image:
                canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                canvas = canvas.reshape(-1, *canvas.shape[2:])
                self.writer.add_image("train/render", canvas, step)
            self.writer.flush()

        # save checkpoint before updating the model
        if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
            mem = torch.cuda.max_memory_allocated() / 1024**3
            stats = {
                "mem": mem,
                "ellipse_time": time.time() - global_tic,
                "num_GS": len(self.splats["means"]),
            }
            print("Step: ", step, stats)
            import json
            with open(
                f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                "w",
            ) as f:
                json.dump(stats, f)
            
            # 保存优化器状态
            optimizer_state = {
                'optimizers': {name: opt.state_dict() for name, opt in self.optimizers.items()},
                'pose_optimizers': [opt.state_dict() for opt in self.pose_optimizers] if hasattr(self, 'pose_optimizers') else [],
                'app_optimizers': [opt.state_dict() for opt in self.app_optimizers] if hasattr(self, 'app_optimizers') else [],
                'bil_grid_optimizers': [opt.state_dict() for opt in self.bil_grid_optimizers] if hasattr(self, 'bil_grid_optimizers') else [],
                'schedulers': [scheduler.state_dict() for scheduler in schedulers] if schedulers else [],
            }
            
            data = {
                "step": step, 
                "splats": self.splats.state_dict(),
                "strategy_state": self.strategy_state,
                "optimizer_state": optimizer_state,
            }
            
            if cfg.pose_opt:
                if world_size > 1:
                    data["pose_adjust"] = self.pose_adjust.module.state_dict()
                else:
                    data["pose_adjust"] = self.pose_adjust.state_dict()
            if cfg.app_opt:
                if world_size > 1:
                    data["app_module"] = self.app_module.module.state_dict()
                else:
                    data["app_module"] = self.app_module.state_dict()
            torch.save(
                data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
            )

        # Turn Gradients into Sparse Tensor before running optimizer
        if cfg.sparse_grad:
            assert cfg.packed, "Sparse gradients only work with packed mode."
            gaussian_ids = info["gaussian_ids"]
            for k in self.splats.keys():
                grad = self.splats[k].grad
                if grad is None or grad.is_sparse:
                    continue
                self.splats[k].grad = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],  # [1, nnz]
                    values=grad[gaussian_ids],  # [nnz, ...]
                    size=self.splats[k].size(),  # [N, ...]
                    is_coalesced=len(Ks) == 1,
                )

        if cfg.visible_adam:
            gaussian_cnt = self.splats.means.shape[0]
            if cfg.packed:
                visibility_mask = torch.zeros_like(
                    self.splats["opacities"], dtype=bool
                )
                visibility_mask.scatter_(0, info["gaussian_ids"], 1)
            else:
                visibility_mask = (info["radii"] > 0).any(0)

        # optimize
        for optimizer in self.optimizers.values():
            if cfg.visible_adam:
                optimizer.step(visibility_mask)
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for optimizer in self.pose_optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for optimizer in self.app_optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for optimizer in self.bil_grid_optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for scheduler in schedulers:
            scheduler.step()

        # Run post-backward steps after backward and optimizer
        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.cfg.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                packed=cfg.packed,
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.cfg.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                lr=schedulers[0].get_last_lr()[0],
            )
        else:
            assert_never(self.cfg.strategy)

        # eval the full set
        if step in [i - 1 for i in cfg.eval_steps]:
            self.eval(step)
            self.render_traj(step)

        # run compression
        if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
            self.run_compression(step=step)

        if not cfg.disable_viewer:
            self.viewer.lock.release()
            num_train_steps_per_sec = 1.0 / (time.time() - tic)
            num_train_rays_per_sec = (
                num_train_rays_per_step * num_train_steps_per_sec
            )
            # Update the viewer state.
            self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
            # Update the scene.
            self.viewer.update(step, num_train_rays_per_step)

# Replace the original train function
Runner.train = new_train

def main(local_rank: int, world_rank: int, world_size: int, cfg: ResumeConfig):
    global init_step
    
    # 设置初始步骤
    init_step = cfg.init_step
    
    # 检查检查点路径
    if not cfg.resume_ckpt:
        raise ValueError("必须提供检查点路径 (--resume-ckpt)")
    
    # 创建结果目录
    os.makedirs(cfg.result_dir, exist_ok=True)
    os.makedirs(f"{cfg.result_dir}/ckpts", exist_ok=True)
    os.makedirs(f"{cfg.result_dir}/stats", exist_ok=True)
    os.makedirs(f"{cfg.result_dir}/renders", exist_ok=True)
    os.makedirs(f"{cfg.result_dir}/tb", exist_ok=True)
    
    # 加载checkpoint
    ckpt_data = torch.load(cfg.resume_ckpt, map_location='cpu')
    
    # 打印checkpoint信息
    if 'step' in ckpt_data:
        print(f"Loaded checkpoint from step {ckpt_data['step']}")
        if cfg.init_step == 0:
            # 如果用户没有指定init_step，我们从checkpoint的步骤继续
            init_step = ckpt_data['step'] + 1
            print(f"Continuing from step {init_step}")
    
    # 检查是否有优化器状态
    has_optimizer_state = 'optimizer_state' in ckpt_data
    if has_optimizer_state:
        if cfg.load_optimizer_state:
            print("检测到优化器状态，将尝试加载")
        else:
            print("检测到优化器状态，但基于用户设置将不加载")
    else:
        print("警告：检查点中没有优化器状态，继续训练可能不稳定")
    
    # 处理策略类型
    if hasattr(cfg, "strategy") and hasattr(cfg.strategy, "type"):
        strategy_type = cfg.strategy.type
        strategy_params = {k: v for k, v in vars(cfg.strategy).items() if k != "type"}
        
        if strategy_type == "RegionFilteredStrategy":
            # 处理区域参数
            if hasattr(cfg.strategy, "region_center") and isinstance(cfg.strategy.region_center, str):
                region_center = [float(x) for x in cfg.strategy.region_center.split(",")]
                strategy_params["region_center"] = region_center
            
            if hasattr(cfg.strategy, "region_radius"):
                strategy_params["region_radius"] = float(cfg.strategy.region_radius)
                
            # 创建区域筛选策略
            cfg.strategy = RegionFilteredStrategy(**strategy_params)
            print(f"使用区域筛选策略，中心点: {region_center}，半径: {strategy_params['region_radius']}")
    
    # 手动部分初始化Runner，跳过create_splats_with_optimizers
    device = f"cuda:{local_rank}"
    
    # 创建一个自定义的Runner实例，但不调用__init__方法
    from utils import set_random_seed
    
    # 设置随机种子
    set_random_seed(42 + local_rank)
    
    # 初始化Runner实例但不调用__init__
    runner = Runner.__new__(Runner)
    
    # 设置基本属性
    runner.cfg = cfg
    runner.world_rank = world_rank
    runner.local_rank = local_rank
    runner.world_size = world_size
    runner.device = device
    
    # 设置输出目录
    runner.ckpt_dir = f"{cfg.result_dir}/ckpts"
    runner.stats_dir = f"{cfg.result_dir}/stats"
    runner.render_dir = f"{cfg.result_dir}/renders"
    
    # 初始化tensorboard
    from torch.utils.tensorboard import SummaryWriter
    runner.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")
    
    # 加载数据集
    from datasets.colmap import Dataset, Parser
    runner.parser = Parser(
        data_dir=cfg.data_dir,
        factor=cfg.data_factor,
        normalize=cfg.normalize_world_space,
        test_every=cfg.test_every,
    )
    runner.trainset = Dataset(
        runner.parser,
        split="train",
        patch_size=cfg.patch_size,
        load_depths=cfg.depth_loss,
    )
    runner.valset = Dataset(
        runner.parser, 
        split="val"
    )
    runner.scene_scale = runner.parser.scene_scale * 1.1 * cfg.global_scale
    print("Scene scale:", runner.scene_scale)
    runner.bbox_loss_corners = None
    if float(getattr(cfg, "bbox_loss_weight", 0.0) or 0.0) > 0.0:
        if not cfg.bbox_file:
            raise ValueError("启用 bbox_loss_weight 时必须提供 bbox_file")
        if not os.path.exists(cfg.bbox_file):
            raise FileNotFoundError(f"找不到 bbox_file: {cfg.bbox_file}")
        runner.bbox_loss_corners = load_bbox_corners(cfg.bbox_file).to(device)
        print(
            f"BBox 局部 loss 已启用: weight={cfg.bbox_loss_weight}, "
            f"edge={cfg.bbox_edge_loss_weight}, ssim={cfg.bbox_ssim_loss_weight}, "
            f"pad={cfg.bbox_loss_pad_pixels}, min_size={cfg.bbox_loss_min_size}"
        )
    
    # 初始化评估指标
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    
    runner.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    runner.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    
    if cfg.lpips_net == "alex":
        runner.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(device)
    elif cfg.lpips_net == "vgg":
        runner.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=False
        ).to(device)
    else:
        raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")
    
    # 创建空的参数字典和优化器字典
    runner.splats = torch.nn.ParameterDict().to(device)
    runner.optimizers = {}
    
    # 初始化其他组件
    runner.pose_optimizers = []
    if cfg.pose_opt:
        from utils import CameraOptModule
        runner.pose_adjust = CameraOptModule(len(runner.trainset)).to(device)
        runner.pose_adjust.zero_init()
        runner.pose_optimizers = [
            torch.optim.Adam(
                runner.pose_adjust.parameters(),
                lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                weight_decay=cfg.pose_opt_reg,
            )
        ]
        if world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            runner.pose_adjust = DDP(runner.pose_adjust)
    
    if cfg.pose_noise > 0.0:
        from utils import CameraOptModule
        runner.pose_perturb = CameraOptModule(len(runner.trainset)).to(device)
        runner.pose_perturb.random_init(cfg.pose_noise)
        if world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            runner.pose_perturb = DDP(runner.pose_perturb)
    
    runner.app_optimizers = []
    feature_dim = 32 if cfg.app_opt else None
    if cfg.app_opt:
        from utils import AppearanceOptModule
        runner.app_module = AppearanceOptModule(
            len(runner.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
        ).to(device)
        torch.nn.init.zeros_(runner.app_module.color_head[-1].weight)
        torch.nn.init.zeros_(runner.app_module.color_head[-1].bias)
        runner.app_optimizers = [
            torch.optim.Adam(
                runner.app_module.embeds.parameters(),
                lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                weight_decay=cfg.app_opt_reg,
            ),
            torch.optim.Adam(
                runner.app_module.color_head.parameters(),
                lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
            ),
        ]
        if world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            runner.app_module = DDP(runner.app_module)
    
    runner.bil_grid_optimizers = []
    if cfg.use_bilateral_grid:
        from lib_bilagrid import BilateralGrid
        runner.bil_grids = BilateralGrid(
            len(runner.trainset),
            grid_X=cfg.bilateral_grid_shape[0],
            grid_Y=cfg.bilateral_grid_shape[1],
            grid_W=cfg.bilateral_grid_shape[2],
        ).to(device)
        runner.bil_grid_optimizers = [
            torch.optim.Adam(
                runner.bil_grids.parameters(),
                lr=2e-3 * math.sqrt(cfg.batch_size),
                eps=1e-15,
            ),
        ]
    
    # 禁用查看器
    runner.viewer = None
    
    # 直接从检查点加载splats
    if 'splats' in ckpt_data:
        for k, v in ckpt_data['splats'].items():
            runner.splats[k] = torch.nn.Parameter(v.to(device))
        print(f"Loaded splats parameters from checkpoint: {len(runner.splats['means'])} Gaussians")
    
    # 重新创建优化器
    from gsplat.optimizers import SelectiveAdam
    
    # 设置参数学习率
    param_names = ['means', 'scales', 'quats', 'opacities', 'sh0', 'shN']
    param_lrs = [1.6e-4 * runner.scene_scale, 5e-3, 1e-3, 5e-2, 2.5e-3, 2.5e-3 / 20]
    
    # 创建优化器
    for name, lr in zip(param_names, param_lrs):
        if name in runner.splats:
            optimizer_class = torch.optim.Adam if not cfg.visible_adam else SelectiveAdam
            runner.optimizers[name] = optimizer_class(
                [{"params": runner.splats[name], "lr": lr * math.sqrt(cfg.batch_size), "name": name}],
                eps=1e-15 / math.sqrt(cfg.batch_size),
                betas=(1 - cfg.batch_size * (1 - 0.9), 1 - cfg.batch_size * (1 - 0.999)),
            )
            
    # 尝试加载优化器状态
    if has_optimizer_state and cfg.load_optimizer_state and 'optimizers' in ckpt_data['optimizer_state']:
        opt_state = ckpt_data['optimizer_state']['optimizers']
        for name, opt in runner.optimizers.items():
            if name in opt_state:
                try:
                    opt.load_state_dict(opt_state[name])
                    print(f"成功加载'{name}'优化器状态")
                except Exception as e:
                    print(f"无法加载'{name}'优化器状态: {e}")
    
    # 加载姿态优化器状态（如果存在）
    if cfg.pose_opt and 'pose_adjust' in ckpt_data:
        if hasattr(runner.pose_adjust, 'module'):
            runner.pose_adjust.module.load_state_dict(ckpt_data['pose_adjust'])
        else:
            runner.pose_adjust.load_state_dict(ckpt_data['pose_adjust'])
        print("Loaded pose adjustment parameters from checkpoint")
        
        # 尝试加载姿态优化器状态
        if has_optimizer_state and cfg.load_optimizer_state and 'pose_optimizers' in ckpt_data['optimizer_state']:
            for i, opt in enumerate(runner.pose_optimizers):
                if i < len(ckpt_data['optimizer_state']['pose_optimizers']):
                    try:
                        opt.load_state_dict(ckpt_data['optimizer_state']['pose_optimizers'][i])
                        print(f"成功加载姿态优化器{i}状态")
                    except Exception as e:
                        print(f"无法加载姿态优化器{i}状态: {e}")
    
    # 加载外观模块状态（如果存在）
    if cfg.app_opt and 'app_module' in ckpt_data:
        if hasattr(runner.app_module, 'module'):
            runner.app_module.module.load_state_dict(ckpt_data['app_module'])
        else:
            runner.app_module.load_state_dict(ckpt_data['app_module'])
        print("Loaded appearance module parameters from checkpoint")
        
        # 尝试加载外观优化器状态
        if has_optimizer_state and cfg.load_optimizer_state and 'app_optimizers' in ckpt_data['optimizer_state']:
            for i, opt in enumerate(runner.app_optimizers):
                if i < len(ckpt_data['optimizer_state']['app_optimizers']):
                    try:
                        opt.load_state_dict(ckpt_data['optimizer_state']['app_optimizers'][i])
                        print(f"成功加载外观优化器{i}状态")
                    except Exception as e:
                        print(f"无法加载外观优化器{i}状态: {e}")
    
    # 尝试加载双边网格优化器状态
    if cfg.use_bilateral_grid and has_optimizer_state and cfg.load_optimizer_state and 'bil_grid_optimizers' in ckpt_data['optimizer_state']:
        for i, opt in enumerate(runner.bil_grid_optimizers):
            if i < len(ckpt_data['optimizer_state']['bil_grid_optimizers']):
                try:
                    opt.load_state_dict(ckpt_data['optimizer_state']['bil_grid_optimizers'][i])
                    print(f"成功加载双边网格优化器{i}状态")
                except Exception as e:
                    print(f"无法加载双边网格优化器{i}状态: {e}")
    
    # 尝试加载策略状态
    if 'strategy_state' in ckpt_data and cfg.load_strategy_state:
        try:
            if isinstance(cfg.strategy, DefaultStrategy):
                runner.strategy_state = ckpt_data['strategy_state']
                
                # 将策略状态中的张量移动到正确的设备
                for key, value in runner.strategy_state.items():
                    if torch.is_tensor(value):
                        runner.strategy_state[key] = value.to(device)
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if torch.is_tensor(sub_value):
                                value[sub_key] = sub_value.to(device)
                
                print("成功加载策略状态")
            else:
                print("策略类型不匹配，重新初始化策略状态")
                if isinstance(cfg.strategy, DefaultStrategy):
                    runner.strategy_state = cfg.strategy.initialize_state(
                        scene_scale=runner.scene_scale
                    )
                elif isinstance(cfg.strategy, MCMCStrategy):
                    runner.strategy_state = cfg.strategy.initialize_state()
        except Exception as e:
            print(f"无法加载策略状态: {e}")
            # 初始化新的策略状态
            if isinstance(cfg.strategy, DefaultStrategy):
                runner.strategy_state = cfg.strategy.initialize_state(
                    scene_scale=runner.scene_scale
                )
            elif isinstance(cfg.strategy, MCMCStrategy):
                runner.strategy_state = cfg.strategy.initialize_state()
    else:
        # 初始化策略状态
        if isinstance(cfg.strategy, DefaultStrategy):
            runner.strategy_state = cfg.strategy.initialize_state(
                scene_scale=runner.scene_scale
            )
        elif isinstance(cfg.strategy, MCMCStrategy):
            runner.strategy_state = cfg.strategy.initialize_state()
    
    # 添加rasterize_splats方法到runner
    def rasterize_splats(
        self,
        camtoworlds,
        Ks,
        width,
        height,
        masks=None,
        **kwargs
    ):
        means = self.splats["means"]  # [N, 3]
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        
        from gsplat.rendering import rasterization
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info
    
    runner.rasterize_splats = types.MethodType(rasterize_splats, runner)
    
    # 使用monkey-patched版本的train方法
    # 开始训练
    runner.train()

if __name__ == "__main__":
    """
    使用方法示例：
    
    ```bash
    # 从检查点继续训练，使用默认配置
    CUDA_VISIBLE_DEVICES=0 python resume_training.py default --resume-ckpt /path/to/checkpoint.pt --result_dir ./results/continued_training
    
    # 从检查点继续训练，同时修改训练参数
    CUDA_VISIBLE_DEVICES=0 python resume_training.py default --resume-ckpt /path/to/checkpoint.pt --data_dir data/new_dataset --max_steps 150000 --eval_steps 15000 30000 60000 90000 --save_steps 15000 30000 60000 90000 120000 150000 --ssim_lambda 0.5
    
    # 从检查点继续训练，但不加载优化器状态
    CUDA_VISIBLE_DEVICES=0 python resume_training.py default --resume-ckpt /path/to/checkpoint.pt --result_dir ./results/continued_training --load_optimizer_state=False
    
    # 从检查点继续训练，使用多GPU
    CUDA_VISIBLE_DEVICES=0,1,2,3 python resume_training.py default --resume-ckpt /path/to/checkpoint.pt --result_dir ./results/continued_training --steps_scaler 0.25
    ```
    """
    
    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "继续训练高斯点云，加载检查点，使用原始训练脚本的默认参数。",
            ResumeConfig(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "继续训练高斯点云，加载检查点，使用MCMC策略。",
            ResumeConfig(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)
    
    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )
    
    cli(main, cfg, verbose=True)
