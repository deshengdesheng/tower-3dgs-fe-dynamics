# 真实塔体演示索引

当前主目录：`outputs/real_tower_roi_continuous_corrected_20261017_v4/`。连续链实际运行完成，独立核验通过；完整场景仍存在静态背景遮挡，塔体子集须与完整图分别查看。

## 当前连续链产物

| 相对主目录的路径 | 含义 |
| --- | --- |
| `step_ii/ckpt_449999_rank0_expanded.pt` | 真实 ROI 训练结果经地形处理/扩展后的全模型，也是变形前 GS |
| `step_ii/step_ii_manifest.json` | IIb 输入、参数及结构 ROI 保护记录 |
| `step_ii/adaptive/` | 地形/结构分类、原始索引与结构保护审计 |
| `step_iii/runtime/static.pt`、`moving.pt` | 同一 expanded 模型的静态和动态子集，全部 Gaussian 字段沿用 |
| `step_iii/runtime/moving_vibration.dat` | 按 moving 行顺序接入的历史 float32 FE 位移流 |
| `step_iii/runtime/runtime_index_mapping.npz` | 子集到 expanded 的实际索引 |
| `step_iii/runtime/mapping_audit.json` | 全部目标 ROI 向最近历史位移源采样的距离审计；源行可重复，不是 Gaussian ID 连接 |
| `step_iii/runtime/gs_to_fe.txt` | 真实注册矩阵副本，用于独立重渲染的物理朝向 |
| `step_iv/expanded_deformed_frame_00037.pt` | 默认帧 37 的全模型变形后 GS；位移放大 50 倍，不能当物理幅值结果 |
| `step_iv/dynamic_render/frames/` | static + moving 的真实 gsplat 渲染帧 |
| `step_iv/dynamic_render/moving_frames/` | 同相机的 moving 子集 GS 渲染，用于观察静态环境遮挡 |
| `stage_visuals_topdown_20261019/` | 本次最新逐步图集：诊断图、真实 GS 渲染和俯视地形扩展对照 |
| `stage_visuals_topdown_20261019/stage_visual_manifest.json` | 各图来源、坐标方向、执行范围和限制 |
| `continuous_chain_manifest.json` | 端到端文件传递、数量及执行边界 |
| `independent_artifact_verification.json` | 独立 CPU 全字段/索引分区/位移/变形后模型检查 |
| `stage_visuals_topdown_20261019/index.html` | 中文逐步图集，区分诊断投影和真实GS光栅化 |
| `stage_visuals_topdown_20261019/overview.png` | 动态网格总览，包含 ROI 结果、地形扩展俯视对照和完整场景渲染 |
| `stage_visuals_topdown_20261019/04b_step_IIb_terrain_expansion_before_after_topdown.png` | 449999 输入与 IIb 输出的同范围俯视对照；源模型共 1,687,649 点，图中左右两栏使用相同的 60,000 点抽样底图，红色仅标识右侧追加的 52,736 个 Gaussian 中心。父模型索引前缀已核验；这是中心投影诊断图，不是GS光栅化 |
| `stage_visuals/moving_before_after_gs.png` | 同相机、同一塔体Gaussian子集：真实零位移与第37帧对照 |
| `stage_visuals/moving_dynamic.gif` | 40张实际GS渲染帧构成的塔体子集动画，无合成插帧 |
| `step_iv/dynamic_playback_20261017_231404/` | 运行PowerShell入口实际生成的40个采样帧（0至390、间隔10），含完整场景/子集/零位移图 |
| `camera_inspection/camera_sweep.png` | 同一全模型的16个真实相机视角检查，零位移，未过滤背景 |
| `step_iv/dynamic_playback_20261017_231800/` | 45°方位、-15°仰角的另4帧视角检查；静态背景仍有明显遮挡 |

默认采样帧为 0、37、74、111。前后 GS 的差异仅包括772,827个 ROI 动态点的位移；完整旋转/尺度更新尚未重放。档案第0帧并非未变形：最大位移模长约0.003336（原模型坐标），放大50倍后约0.1668；零位移基准必须单独渲染。俯视图显示扩展操作是在已有地形场景上继续添加 Gaussian，不代表从无地形合成完整地形。

## 动态 GS 再运行

在根目录使用已验证的 WSL/CUDA 环境；输入 runtime 必须已经生成：

```bash
GSPLAT_3DGS_ONLY=1 python -B scripts/run_dynamic_gs.py \
  --runtime-dir outputs/real_tower_roi_continuous_corrected_20261017_v4/step_iii/runtime \
  --output-dir outputs/real_tower_roi_dynamic_rerun \
  --gs-to-fe-transform /path/to/registered_gs_to_fe_4x4.txt \
  --frames 4 --frame-start 0 --frame-stride 37 \
  --resolution 1024x1536 --scale-disp 50 --device cuda \
  --azimuth-deg 90 --elevation-deg 15 --distance-scale 3.0 \
  --save-moving-frames --save-undeformed --radius-clip 0
```

完整连续链命令在 `README.md`。方向使用同一真实注册矩阵；不能以任意翻转模型坐标修正视图。

Windows/WSL 环境可运行 `scripts/Start-ContinuousTowerDemo.ps1`；它从同一 v4 runtime 渲染 40 个采样帧到新建的时间戳目录，并保留零位移基准、完整场景和塔体子集。请通过参数或环境变量配置：`GSPLAT_WSL_DISTRIBUTION`（默认 `Ubuntu`）、`GSPLAT_PROCESSING_PYTHON`、`GSPLAT_RENDERING_PYTHON`；后两者以及可选的 `GSPLAT_PREBUILT` 必须使用 WSL 内的 Linux 路径。脚本不再包含作者机器的 WSL 用户目录或 Python 安装路径。

该PowerShell入口已实际执行成功，生成上表两个时间戳目录。GIF每张停留100ms仅用于浏览，不等同于原始FE物理时间步；位移流保存全部399帧。图集由 `scripts/assemble_continuous_tower_review.py` 从实际图像汇总，没有重新着色模型或用生成式图像补图。

## 历史项

`outputs/real_tower_demo_final_20261017/`：旧 `lingshi2` 272,300 点分支，含早期前后 GS 和 runtime，保留作对照。其后续渲染没有消费本次 449999→IIb 的 expanded，因此不能作为新连续链结果。

`outputs/real_tower_roi_continuous_corrected_20261017_v2/`：结构保护前版；v3保留了大量静态ROI点；本次 v4 为主入口。更早的 `real_tower_pipeline_final_20261013` 256×256 图不作为最终演示。
