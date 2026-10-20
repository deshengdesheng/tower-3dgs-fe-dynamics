# 公开代码与最新编辑意见的对应范围

本文件说明公开代码对最新编辑意见的支持范围，不代替论文稿件本身的排版核查。公开仓库为 <https://github.com/deshengdesheng/tower-3dgs-fe-dynamics>。当前主入口为 `outputs/real_tower_roi_continuous_corrected_20261017_v4/`；该本机演示目录及其真实数据未随仓库公开。

| 编辑要求 | 当前状态 | 仍需完成 |
| --- | --- | --- |
| 摘要加入用于 benchmarking/reproducibility 的源码链接 | 已完成 | 摘要与 R2 回复信使用作者公开仓库 URL；该链接不是上游 gsplat 地址 |
| Section 1 区分 introduction / related work / contributions | 需稿件检查 | 核对 Clean 和 Marked 稿的实际小节；代码目录不能证明完成 |
| 公式末尾按上下文加逗号或句号 | 需稿件检查 | 逐条核对最终稿和导出 PDF |
| 检查、提高低质量图及压缩问题 | 尚待最终核查 | 当前连续演示的逐图检查和论文全部图的最终导出检查 |
| 支持复用与比较 | 部分准备 | 提供可运行连续链、真实 GS 前后模型/runtime、坐标与匹配审计；补足执行范围和数据获取说明，并验证独立环境 |

## 代码候选实际覆盖

已整理的连续入口以已有真实 399999→449999 ROI 训练结果为起点，重新执行 IIb 的地形分离/Kriging/扩展，并在同一 expanded 模型上分出全部字段一致的 static/moving 子集，接入历史 FE 流后调用 gsplat。

ROI 内点数 238,354→772,827 是两个既有 checkpoint 的精确 mask 统计，不表示本次重训或确定的个体 clone 数。IIb 的结构 ROI 保护是为修复原 `tu1` 参数迁移到 449999 后的误分类而新增，不能视为原 notebook 数值结果。

FE 接入采用全部目标ROI向最近历史源采样平移，源行允许复用；772,827个ROI点全部动态、静态ROI残留为0，不能称为 Gaussian ID 精确连接或论文Eq.20插值。当前动态链仅更新均值位移；完整 polar rotation / quaternion / log-scale / 协方差更新，以及原始照片/SfM、长训练、风场和 ANSYS 求解尚未在该链重跑。

因此当前只能说明公开仓库包含的代码与连续演示范围，不能声明全部论文流程已复现。论文格式、公式标点和图片质量属于稿件层面的核查，不由代码仓库本身证明。

## 发布说明

仓库不包含受限的塔体影像、训练 checkpoint、FE 输入或私有演示输出。运行真实案例前须另行准备有权使用的输入，并按 `ENVIRONMENT.md` 配置环境。引用代码时应同时遵循作者代码与所用第三方组件各自的许可。

详细执行证据见 `PIPELINE_DEMONSTRATION.md`；文件索引见 `REAL_TOWER_DEMO_INDEX.md`。
