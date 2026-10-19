# 连续真实塔体演示的执行范围

主入口为 `outputs/real_tower_roi_continuous_corrected_20261017_v4/`。连续链已实际完成，独立 CPU 逐字段和位移核验通过；完整场景仍有明显静态背景遮挡，塔体子集已正立可见，不能把子集图当完整场景质量验收。

## 论文步骤与本次执行

| 论文阶段 | 本次行为 | 可核对证据 |
| --- | --- | --- |
| I：SfM/3DGS 初始化及注册 | 读取已有真实 `ckpt_399999_rank0.pt`，未重建照片/SfM 或重训 | 主清单的 initial 路径、初始 GS 诊断图 |
| IIa：ROI 重要区域加密/剪枝 | 展示已有真实 `399999→449999` 训练结果，未重新训练 | `1_1_20260520.ipynb` 的训练分支、两个 checkpoint 的精确 ROI mask |
| IIb：地形分离、Kriging、多层扩展 | 从 449999 重新执行，并增加精确结构 ROI 保护 | `step_ii/step_ii_manifest.json`、分类审计及 expanded checkpoint |
| III：风场与 FE 响应 | 复用历史已注册位移流，每个目标 ROI 向最近历史源点采样平移；未重跑风场/ANSYS | `step_iii/runtime/mapping_audit.json`、目标到源索引及流形状 |
| IV：变形与 GS 渲染 | 在同一 expanded 的 static/moving 子集上实际更新 moving 均值并调用 gsplat | runtime、前后 checkpoint、渲染帧、主清单 |
| IV 的完整旋转/尺度部分 | 尚未在连续动态链重放 polar rotation、quaternion、log-scale 与协方差更新 | 当前 renderer 只执行均值位移 |

IIa 的总点数为 1,687,725→1,687,649；精确梁胶囊 ROI 内点数为 238,354→772,827。统计说明容量向 ROI 分配的训练结果，不提供逐 Gaussian 的 clone/split/prune 谱系；449999 是真实已有训练结果，不声明它是全部论文实验唯一最终 checkpoint。

## 为何增加结构保护

地形 notebook 原用 `tu1_percentage_filtered.pt`，不是这里的 449999。直接迁移原参数时，大量塔体被划入 terrain。当前流程因此在分类前排除精确结构 ROI，并记录保护审计；它是新的连续演示修正规则，不应包装成原 notebook 的原样实验重现。未提供 `--structure-roi-cache` 时，地形脚本仍沿原分类逻辑。

## 数据如何一环扣一环

`run_real_roi_tower_chain.py` 将 449999 直接传入 IIb，再把其 `ckpt_449999_rank0_expanded.pt` 传给 `build_roi_dynamic_runtime.py`。后者按保存的索引拆分 expanded 的全部 Gaussian 字段。旧 moving 模型仅用作 FE 行参考坐标；最终渲染载入的是新 static/moving，不是旧 `lingshi2` 几何。

v3的一对一映射只移动107,844个点，使ROI中665,207个点仍静态。本次改为全部772,827个目标ROI点向最近历史moving行采样399帧平移，源行可复用；静态ROI残留为0。完整1,740,385个点分成772,827 moving与967,558 static，全部Gaussian字段按父模型索引严格相等。映射距离P95为0.0265625、最大0.0985143，单位为原模型坐标；1,385个目标距离超过0.05。它仍不是Gaussian ID连接或Eq.20插值，空间覆盖不证明FE精度，ROI padding也可能包含近旁背景。

最终渲染调用 gsplat，沿用模型的四元数、尺度、不透明度和球谐系数。中间 Gaussian 中心图、地形高程图与 FE 位移图是诊断图，不是照片级 GS 结果。诊断图由 FE +Z 推导物理向上方向，GS 相机也使用注册变换；不能随意改点坐标掩盖方向问题。

## 检查及复用

完整命令见 `README.md`；逐文件索引与 runtime 再运行命令见 `REAL_TOWER_DEMO_INDEX.md`。运行器检查 IIb 输入路径、expanded parent、static+moving数量、静态ROI为0、流帧范围、实际渲染帧和统计文件。`independent_artifact_verification.json` 额外独立验证全部子集张量字段、互补索引、精确ROI、最近源索引、0/37/74/111帧逐元素位移及完整after37模型。检查通过只证明这条平移演示的文件衔接正确。

`--scale-disp 50` 是展示放大倍数；默认导出的 `expanded_deformed_frame_00037.pt` 也采用该倍数，不能当未放大的物理变形模型。

档案第0帧已有非零位移，不能标为“未变形”。真实未变形模型为IIb expanded，图像基准使用同相机零位移单独渲染；模型和位移数据不为调整图片方向而翻转。

旧 `outputs/real_tower_demo_final_20261017/` 的 272,300 点分支与更早 smoke 输出仅为历史记录。本次主入口不宣称重新执行全部论文步骤，也不宣称已完成编辑要求或公开源码。
