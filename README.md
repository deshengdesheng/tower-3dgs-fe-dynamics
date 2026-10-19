# 真实塔体重建与动态高斯可视化：本地候选版

本目录面向论文 PHOTO-D-26-00762R2，尚未公开。作者研究代码采用 MIT；随附 gsplat 等第三方组件保留各自许可证，见 `THIRD_PARTY_NOTICES.md`、`PROVENANCE.md` 和 `LICENSE-gsplat`。

## 当前主入口

`outputs/real_tower_roi_continuous_corrected_20261017_v4/` 是本次实际运行完成的连续演示目录。四个采样帧和15张阶段图已生成；`independent_artifact_verification.json` 已独立核验同模型字段、完整分区、位移映射和变形后 checkpoint。完整场景仍有明显静态背景遮挡，不能宣称图像质量已全部通过。

```text
Step I 已有 ckpt_399999
  → Step IIa 已有真实 ROI 训练结果 ckpt_449999
  → Step IIb 结构 ROI 保护下的地形分离 → Kriging → 多层扩展
  → 同一 expanded 模型分成 static / moving，并注册历史 FE 位移流
  → Step IV gsplat 动态渲染
```

Step IIa 没有在本次演示中重新训练，也没有用人为复制/透明度排序代替训练。两个既有 checkpoint 的精确梁胶囊 ROI 内点数从 **238,354 增至 772,827**，总点数分别为 1,687,725 和 1,687,649。这是两个训练结果的区域统计，不能据此确定每个 Gaussian 的 clone、split 或 prune 历史。

Step IIb 直接消费 449999，而非切回旧 `lingshi2.pt`。原地形 notebook 针对 `tu1_percentage_filtered.pt` 的参数直接搬到 449999 后会把大量塔体误分为地形，因此本次增加明确的 `--structure-roi-cache` 保护：精确 ROI 内点排除出地形分类，保留为结构。这是新的连续演示规则，不等同于原 notebook 的历史输入或数值结果。

`static.pt` 和 `moving.pt` 的均值、四元数、对数尺度、不透明度及球谐系数均按索引取自同一个 Step IIb `expanded.pt`。全模型1,740,385个点精确拆为967,558个静态点和772,827个动态点，全部精确 ROI 点进入动态部分，静态 ROI 残留为0。历史 `lingshi2_moving.pt` 只提供位移场参考位置，不作为最终渲染几何；每个目标 ROI 点接收最近历史源行的平移，源行允许复用。这不是训练时 Gaussian ID 的精确连接，也不是论文 Eq.20 插值。

最终动态图使用真实 gsplat Gaussian 光栅化。中间的 Gaussian 中心投影、FE 位移图和地形剖面只是诊断图；其方向由注册变换中的物理 FE +Z 定义，不能用普通散点图代替 GS 渲染。

## 运行连续演示

在候选目录根目录、已安装依赖且 CUDA 光栅器可用的 Linux/WSL Python 环境中执行。设置的三个路径必须指向获授权的真实文件；输出目录必须尚不存在。

```bash
SOURCE=/path/to/authorized/gsplat_assets
GS_TO_FE=/path/to/registered_gs_to_fe_4x4.txt
GSPLAT_3DGS_ONLY=1 python -B scripts/run_real_roi_tower_chain.py \
  --initial-checkpoint "$SOURCE/results/yunnan_zhixianta_dense3/ckpts/ckpt_399999_rank0.pt" \
  --roi-checkpoint "$SOURCE/results/yunnan_zhixianta_dense3_baowei_bboxloss_v1/ckpts/ckpt_449999_rank0.pt" \
  --gs-to-fe-transform "$GS_TO_FE" --source-root "$SOURCE" \
  --output-dir outputs/real_tower_roi_continuous_new \
  --frames 4 --frame-start 0 --frame-stride 37 \
  --resolution 1024x1536 --scale-disp 50 --device cuda
```

源目录还需包含 `roi_fe_beam_exact_capsule_merged_cache_3dgs.npz`、`results/lingshi2_moving.pt` 和与其行数相符的 `results/lingshi2_moving_vibration.dat`。训练、FE 注册变换和流文件均须使用相容坐标系。若光栅器使用另一个环境，可传 `--renderer-python` 和 `--gsplat-prebuilt`；环境边界见 `ENVIRONMENT.md`。

输出索引、变形前后 GS 和动态运行文件见 `REAL_TOWER_DEMO_INDEX.md`。本次执行范围及核查依据见 `PIPELINE_DEMONSTRATION.md`。

直接查看本机结果：`outputs/real_tower_roi_continuous_corrected_20261017_v4/stage_visuals_topdown_20261019/index.html`。该中文图集含ROI训练对照、方向正确的各阶段诊断图、同范围俯视地形扩展前后图、真实零位移/变形GS对照和40帧子集动画。注意地形扩展前的449999输入本身已经含有地形；俯视对照中的红色仅表示本次追加的Gaussian中心。Windows动态重渲染入口为 `scripts/Start-ContinuousTowerDemo.ps1`，已实际运行成功。

## 与论文流程的对应及剩余范围

论文四步包括：I，SfM/3DGS 初始化和 FE 注册；II，ROI 重要区域加密/剪枝及地形补全；III，风场、风载荷和非线性 FE 求解；IV，局部右极分解、均值/四元数/尺度更新、各向异性稀疏到稠密传播及动态渲染。

当前连续演示使用已有 I/IIa 结果，重新执行 IIb，并复用历史已注册 FE 位移。动态渲染实际更新 moving 的均值；完整 polar rotation、quaternion、log-scale 和协方差更新尚未在该连续链中重放。风场和 ANSYS/FE 求解也未重跑，因此不能称为论文全部步骤的完整复现。

候选包含 `merged_gaussian_transform.py`、`apply_transform_with_anisotropic_kernel.py` 等单帧变形源码，但文件存在不等于全参数动态链已验证。原始照片、COLMAP、FE 输入与 checkpoint 需另行提供；本地 `outputs/` 是执行证据，不应自动视为可公开数据包。

## 初始化与代码来源

初始 GS 使用 [gsplat](https://github.com/nerfstudio-project/gsplat) 从 [COLMAP](https://github.com/colmap/colmap) SfM 相机和稀疏点云训练。候选的 `examples/simple_trainer.py` 读取 COLMAP 结果；它不替代 COLMAP 的特征匹配和稀疏重建。ROI 入口为 `examples/train_irregular_region_filter.py`，策略在 `gsplat/strategy/irregular_region_strategy.py`；须提供同一数据集、初始 checkpoint 和已注册 ROI。

`workflow/paper_workflow.json` 保留早期四步范围说明，其旧输出引用不替代本次主入口。notebook 对应见 `NOTEBOOK_TRACEABILITY.md`；环境安装见 `ENVIRONMENT.md`、`requirements-tested.txt`。独立干净环境安装尚未完成验证。

## 历史结果与投稿状态

`outputs/real_tower_demo_final_20261017/` 是旧 272,300 点 `lingshi2` 分支，仅保留作历史对照，不是当前 ROI→补全→动态连续链。v2 未加结构保护；v3 的一对一匹配使665,207个 ROI 点仍为静态；本次以 v4 为主入口。

代码尚无公开 URL，不能把本地路径或上游 gsplat 地址作为本论文源码链接填入摘要。当前与审稿意见的差距见 `REVIEWER_COMPLIANCE.md`。
