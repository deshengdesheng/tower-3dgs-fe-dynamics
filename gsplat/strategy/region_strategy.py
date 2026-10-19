from typing import Any, Dict, List, Tuple, Union
import torch
import numpy as np
from gsplat.strategy.default import DefaultStrategy

class RegionFilteredStrategy(DefaultStrategy):
    """在DefaultStrategy基础上添加了区域筛选功能的策略类
    
    可以指定只对特定区域内的高斯点进行分裂/复制操作，并且只分裂长宽比前5%的点云和大的点云
    """
    
    def __init__(
        self,
        region_center=(0.0, 0.0, 0.0),  # 筛选区域的中心点
        region_radius=1.0,              # 筛选区域的半径
        region_points=None,             # 筛选区域的点列表，如果提供则优先使用
        extend_factor=1.1,              # 包围盒扩展因子
        aspect_ratio_percentile=5,      # 长宽比百分比，只分裂长宽比前X%的细长高斯点
        volume_percentile=5,            # 体积百分比，只分裂体积前X%的大高斯点
        disable_pruning=False,          # 是否禁用剪枝功能
        region_growth_limit=3.0,        # 包围盒内点数量增长限制，达到初始值的N倍后停止分裂
        outside_conservative_factor=5,  # 包围盒外保守系数，数值越大越保守
        **kwargs                        # 传递给DefaultStrategy的其他参数
    ):
        # 分离特定于RegionFilteredStrategy的参数，只将其他参数传递给父类
        super().__init__(**kwargs)
        
        # 初始化RegionFilteredStrategy特有的参数
        self.region_center = torch.tensor(region_center)
        self.region_radius = region_radius
        self.region_points = None
        self.bbox_min = None
        self.bbox_max = None
        self.extend_factor = extend_factor
        
        # 分裂条件设置
        self.aspect_ratio_percentile = aspect_ratio_percentile
        self.volume_percentile = volume_percentile
        
        # 禁用剪枝标志
        self.disable_pruning = disable_pruning
        
        # 包围盒内点数量增长限制
        self.region_growth_limit = region_growth_limit
        self.initial_region_point_count = None  # 初始包围盒内点数量
        self.max_region_point_count = None      # 最大允许的包围盒内点数量
        
        # 包围盒外保守系数
        self.outside_conservative_factor = outside_conservative_factor
        
        # 如果提供了点列表，则构建包围盒
        if region_points is not None:
            self.region_points = torch.tensor(region_points)
            self._build_bounding_box()
    
    def _build_bounding_box(self):
        """根据点列表构建包围盒"""
        if self.region_points is None:
            return
            
        # 计算包围盒的最小和最大坐标
        self.bbox_min = torch.min(self.region_points, dim=0)[0]
        self.bbox_max = torch.max(self.region_points, dim=0)[0]
        
        # 扩展包围盒
        bbox_center = (self.bbox_min + self.bbox_max) / 2
        bbox_extent = (self.bbox_max - self.bbox_min) / 2
        self.bbox_min = bbox_center - bbox_extent * self.extend_factor
        self.bbox_max = bbox_center + bbox_extent * self.extend_factor
        
        print(f"构建包围盒: 最小坐标={self.bbox_min}, 最大坐标={self.bbox_max}")
    
    def _calculate_aspect_ratios(self, scales):
        """计算高斯点的长宽比"""
        actual_scales = torch.exp(scales)  # [N, 3]
        max_scale, _ = torch.max(actual_scales, dim=1)  # [N]
        min_scale, _ = torch.min(actual_scales, dim=1)  # [N]
        aspect_ratios = max_scale / min_scale.clamp(min=1e-6)  # [N]
        return aspect_ratios
    
    def _calculate_volumes(self, scales):
        """计算高斯点的体积"""
        actual_scales = torch.exp(scales)  # [N, 3]
        volumes = actual_scales.prod(dim=1)  # [N]
        return volumes
    
    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> Tuple[int, int]:
        """重写_grow_gs方法，添加区域筛选功能和长宽比/体积筛选"""
        
        # 获取高斯点中心坐标
        means = params["means"]
        scales = params["scales"]
        device = means.device
        
        # 创建区域掩码
        if self.bbox_min is not None and self.bbox_max is not None:
            # 使用包围盒进行筛选
            self.bbox_min = self.bbox_min.to(device)
            self.bbox_max = self.bbox_max.to(device)
            
            # 检查每个点是否在包围盒内
            inside_x = (means[:, 0] >= self.bbox_min[0]) & (means[:, 0] <= self.bbox_max[0])
            inside_y = (means[:, 1] >= self.bbox_min[1]) & (means[:, 1] <= self.bbox_max[1])
            inside_z = (means[:, 2] >= self.bbox_min[2]) & (means[:, 2] <= self.bbox_max[2])
            region_mask = inside_x & inside_y & inside_z
        else:
            # 使用中心点和半径进行筛选
            if not isinstance(self.region_center, torch.Tensor):
                self.region_center = torch.tensor(self.region_center)
            self.region_center = self.region_center.to(device)
            
            # 计算每个高斯点与中心的距离
            distances = torch.norm(means - self.region_center.unsqueeze(0), dim=1)
            
            # 创建区域掩码
            region_mask = distances < self.region_radius
        
        # 获取包围盒内点的数量
        region_point_count = region_mask.sum().item()
        
        # 初始化包围盒内点的初始数量和最大数量
        if self.initial_region_point_count is None:
            self.initial_region_point_count = region_point_count
            self.max_region_point_count = int(self.initial_region_point_count * self.region_growth_limit)
            print(f"初始包围盒内点数: {self.initial_region_point_count}")
            print(f"包围盒内允许的最大点数: {self.max_region_point_count}")
        
        # 检查是否达到包围盒内点数量限制
        should_stop_growth = region_point_count >= self.max_region_point_count
        if should_stop_growth:
            print(f"警告：包围盒内点数({region_point_count})已达到或超过限制({self.max_region_point_count})，将停止区域内分裂")
            # 只对包围盒内的点设置掩码，防止分裂
            region_mask = torch.zeros_like(region_mask)
        
        # 计算包围盒内所有点的长宽比和体积
        aspect_ratios = self._calculate_aspect_ratios(scales)
        volumes = self._calculate_volumes(scales)
        
        # 只在包围盒内的点上计算百分位数
        if region_mask.sum() > 0:
            box_aspect_ratios = aspect_ratios[region_mask]
            box_volumes = volumes[region_mask]
            
            # 计算长宽比和体积的阈值（只在包围盒内的点中计算）
            aspect_threshold_inside = torch.quantile(
                box_aspect_ratios, 
                1.0 - self.aspect_ratio_percentile / 100.0
            )
            volume_threshold_inside = torch.quantile(
                box_volumes,
                1.0 - self.volume_percentile / 100.0
            )
        else:
            # 如果包围盒内没有点，使用默认阈值
            aspect_threshold_inside = float('inf')
            volume_threshold_inside = float('inf')
        
        # 为包围盒外的点计算更保守的阈值（基于所有点）
        conservative_aspect_percentile = max(1, self.aspect_ratio_percentile // self.outside_conservative_factor)  # 使用配置的保守系数
        conservative_volume_percentile = max(1, self.volume_percentile // self.outside_conservative_factor)        # 使用配置的保守系数
        
        aspect_threshold_outside = torch.quantile(
            aspect_ratios, 
            1.0 - conservative_aspect_percentile / 100.0
        )
        volume_threshold_outside = torch.quantile(
            volumes,
            1.0 - conservative_volume_percentile / 100.0
        )
        
        # 使用父类方法获取基本的分裂和复制条件
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        
        is_grad_high = grads > self.grow_grad2d
        # 包围盒外使用更严格的梯度条件
        is_grad_high_conservative = grads > (self.grow_grad2d * 2.0)
        
        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        
        # 计算长宽比和体积掩码
        # 包围盒内：使用激进阈值
        is_elongated_inside = (aspect_ratios > aspect_threshold_inside) & region_mask
        is_large_volume_inside = (volumes > volume_threshold_inside) & region_mask
        
        # 包围盒外：使用保守阈值
        outside_mask = ~region_mask
        is_elongated_outside = (aspect_ratios > aspect_threshold_outside) & outside_mask
        is_large_volume_outside = (volumes > volume_threshold_outside) & outside_mask
        
        # 组合分裂条件
        # 包围盒内：激进分裂
        inside_shape_filter = is_elongated_inside | is_large_volume_inside
        inside_final_filter = region_mask & inside_shape_filter
        
        # 包围盒外：保守分裂
        outside_shape_filter = is_elongated_outside | is_large_volume_outside
        outside_final_filter = outside_mask & outside_shape_filter
        
        # 应用到复制和分裂掩码
        # 包围盒内使用标准梯度条件，包围盒外使用更严格梯度条件
        is_dupli_inside = is_grad_high & is_small & inside_final_filter
        is_dupli_outside = is_grad_high_conservative & is_small & outside_final_filter
        is_dupli = is_dupli_inside | is_dupli_outside
        n_dupli = is_dupli.sum().item()
        
        is_large = ~is_small
        is_split_inside = is_grad_high & is_large & inside_final_filter
        is_split_outside = is_grad_high_conservative & is_large & outside_final_filter
        is_split = is_split_inside | is_split_outside
        
        if step < self.refine_scale2d_stop_iter:
            is_split_2d_inside = (state["radii"] > self.grow_scale2d) & inside_final_filter
            is_split_2d_outside = (state["radii"] > self.grow_scale2d * 1.5) & outside_final_filter  # 更严格的2D条件
            is_split = is_split | is_split_2d_inside | is_split_2d_outside
        
        n_split = is_split.sum().item()
        
        # 执行复制
        if n_dupli > 0:
            from gsplat.strategy.ops import duplicate
            duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)
            
        # 新添加的高斯点不会被分裂
        is_split = torch.cat(
            [
                is_split,
                torch.zeros(n_dupli, dtype=torch.bool, device=device),
            ]
        )
        
        # 执行分裂
        if n_split > 0:
            from gsplat.strategy.ops import split
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                revised_opacity=self.revised_opacity,
            )
        
        # 计算并打印统计信息
        if n_dupli > 0 or n_split > 0:
            # 统计包围盒内外的点数和操作
            n_dupli_inside = is_dupli_inside.sum().item()
            n_dupli_outside = is_dupli_outside.sum().item()
            n_split_inside = is_split_inside.sum().item()
            n_split_outside = is_split_outside.sum().item()
            
            print(f"区域内总点数: {region_mask.sum().item()}, 区域外总点数: {outside_mask.sum().item()}")
            print(f"包围盒内分裂: 复制{n_dupli_inside}个, 分裂{n_split_inside}个 (阈值: 长宽比前{self.aspect_ratio_percentile}%, 体积前{self.volume_percentile}%)")
            print(f"包围盒外保守分裂: 复制{n_dupli_outside}个, 分裂{n_split_outside}个 (阈值: 长宽比前{conservative_aspect_percentile}%, 体积前{conservative_volume_percentile}%)")
            print(f"总计: 复制{n_dupli}个, 分裂{n_split}个")
            
        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> int:
        """重写_prune_gs方法，实现保护包围盒内点的智能剪枝"""
        if self.disable_pruning:
            print("剪枝功能已禁用，跳过剪枝步骤")
            return 0
        
        # 获取所有高斯点的位置
        means = params["means"]
        device = means.device
        
        # 创建区域掩码，识别包围盒内的点
        region_mask = torch.zeros(len(means), dtype=torch.bool, device=device)
        
        if self.bbox_min is not None and self.bbox_max is not None:
            # 使用包围盒进行筛选
            self.bbox_min = self.bbox_min.to(device)
            self.bbox_max = self.bbox_max.to(device)
            
            # 检查每个点是否在包围盒内
            inside_x = (means[:, 0] >= self.bbox_min[0]) & (means[:, 0] <= self.bbox_max[0])
            inside_y = (means[:, 1] >= self.bbox_min[1]) & (means[:, 1] <= self.bbox_max[1])
            inside_z = (means[:, 2] >= self.bbox_min[2]) & (means[:, 2] <= self.bbox_max[2])
            region_mask = inside_x & inside_y & inside_z
        
        # 使用父类方法计算基本剪枝掩码
        is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
        if step > self.reset_every:
            is_too_big = (
                torch.exp(params["scales"]).max(dim=-1).values
                > self.prune_scale3d * state["scene_scale"]
            )
            # 屏幕空间剪枝（如果启用）
            if step < self.refine_scale2d_stop_iter:
                is_too_big |= state["radii"] > self.prune_scale2d

            is_prune = is_prune | is_too_big
        
        # **分层剪枝策略** - 包围盒内保护性，包围盒外标准
        region_points_count = region_mask.sum().item()
        outside_points_count = (~region_mask).sum().item()
        
        if region_points_count > 0 or outside_points_count > 0:
            # 包围盒内：适度剪枝（删除透明度低的点和极大的点）
            inside_opacities = torch.sigmoid(params["opacities"].flatten())
            
            # 透明度剪枝
            region_opa_prune = (inside_opacities < self.prune_opa) & region_mask
            
            # 尺度剪枝（对包围盒内部的过大点也要剪枝，但阈值更宽松）
            region_scale_prune = torch.zeros_like(region_mask)
            if step > self.reset_every:
                inside_scales = torch.exp(params["scales"]).max(dim=-1).values
                region_scale_prune = (
                    inside_scales > (self.prune_scale3d * 1.5 * state["scene_scale"])
                ) & region_mask
            
            # 合并包围盒内剪枝条件
            region_prune = region_opa_prune | region_scale_prune
            
            # 包围盒外：保守剪枝（比标准剪枝更宽松，但比保护性剪枝略严格）
            conservative_prune_opa = self.prune_opa * 0.5  # 比标准剪枝(0.005)更宽松，只删透明度<0.0025的点
            conservative_prune_scale3d = self.prune_scale3d * 2.0  # 比标准剪枝更宽松，允许更大的尺度
            
            # 包围盒外的保守剪枝条件
            outside_opa_prune = torch.sigmoid(params["opacities"].flatten()) < conservative_prune_opa
            outside_scale_prune = False
            
            if step > self.reset_every:
                outside_scale_prune = (
                    torch.exp(params["scales"]).max(dim=-1).values
                    > conservative_prune_scale3d * state["scene_scale"]
                )
                # 屏幕空间剪枝（如果启用）也更保守
                if step < self.refine_scale2d_stop_iter:
                    conservative_prune_scale2d = self.prune_scale2d * 2.0  # 更宽松的2D阈值
                    outside_scale_prune |= state["radii"] > conservative_prune_scale2d
            
            outside_prune = (outside_opa_prune | outside_scale_prune) & ~region_mask
            
            # 合并剪枝掩码
            final_prune = region_prune | outside_prune
            
            # 统计信息
            region_opa_pruned = region_opa_prune.sum().item()
            region_scale_pruned = region_scale_prune.sum().item()
            outside_pruned = outside_prune.sum().item()
            
            if self.verbose and (region_opa_pruned + region_scale_pruned + outside_pruned > 0):
                # 详细显示不同类型的剪枝数量
                print(f"分层剪枝: 包围盒内删除{region_opa_pruned}个透明度过低点(<{self.prune_opa})")
                if region_scale_pruned > 0:
                    print(f"         包围盒内删除{region_scale_pruned}个尺度过大点(3D尺度>{self.prune_scale3d * 1.5:.2f})")
                print(f"         包围盒外删除{outside_pruned}个点(保守剪枝: 透明度<{conservative_prune_opa:.4f}, 3D尺度>{conservative_prune_scale3d:.2f})")
                print(f"剩余点数: 包围盒内{region_points_count - (region_opa_pruned + region_scale_pruned)}个, 包围盒外{outside_points_count - outside_pruned}个")
        else:
            # 如果没有点，不进行任何剪枝
            final_prune = torch.zeros_like(is_prune)
        
        n_prune = final_prune.sum().item()
        if n_prune > 0:
            from gsplat.strategy.ops import remove
            remove(params=params, optimizers=optimizers, state=state, mask=final_prune)

        return n_prune 