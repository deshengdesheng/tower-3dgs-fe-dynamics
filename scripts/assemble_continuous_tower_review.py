#!/usr/bin/env python3
"""Assemble existing verified renders into a review gallery, without rerendering."""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def font(size: int):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def panel(path: Path, label: str, size=(700, 470)) -> Image.Image:
    tile = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(tile)
    draw.text((18, 12), label, font=font(19), fill="#1f2937")
    with Image.open(path) as source:
        image = ImageOps.contain(source.convert("RGB"), (size[0] - 24, size[1] - 62))
    tile.paste(image, ((size[0] - image.width) // 2, 52 + (size[1] - 62 - image.height) // 2))
    return tile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-dir", type=Path, required=True)
    parser.add_argument("--animation-dir", type=Path)
    parser.add_argument("--visuals-dir", type=Path,
                        help="optional separate gallery directory; defaults to CHAIN_DIR/stage_visuals")
    args = parser.parse_args()
    chain = args.chain_dir.resolve()
    out = args.visuals_dir.resolve() if args.visuals_dir else chain / "stage_visuals"
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((chain / "continuous_chain_manifest.json").read_text(encoding="utf-8"))
    ids = manifest["dynamic_render_inputs"]["rendered_frame_indices"]
    frame_a, frame_b = ids[:2]
    animation = args.animation_dir.resolve() if args.animation_dir else chain / "step_iv" / "dynamic_render"
    moving_a = animation / "moving_undeformed.png"
    if not moving_a.is_file():
        raise FileNotFoundError("true zero-displacement reference missing; rerender with --save-undeformed")
    (out / "09_step_IV_gsplat_zero_displacement.png").write_bytes(moving_a.read_bytes())
    moving_b = out / f"08_step_IV_gsplat_moving_subset_frame_{frame_b:05d}.png"
    before_after = Image.new("RGB", (1400, 1080), "white")
    before_after.paste(panel(moving_a, "Real gsplat / same ROI subset / zero displacement", (700, 1020)), (0, 60))
    before_after.paste(panel(moving_b, f"Same Gaussians / frame {frame_b} / displacement x50", (700, 1020)), (700, 60))
    ImageDraw.Draw(before_after).text((22, 16), "Registered translation replay; static environment hidden", font=font(25), fill="#1f2937")
    before_after.save(out / "moving_before_after_gs.png")

    entries = [
        ("00_step_I_initial_checkpoint.png", "I: existing initial 3DGS", "初始真实 GS checkpoint（中心投影诊断图）"),
        ("01_step_IIa_roi_refinement_pruning.png", "IIa: real ROI training result", "重要区域加密与剪枝：既有真实训练结果对照"),
        ("02_step_IIb_terrain_object.png", "IIb: terrain / objects", "地形和非地形分离；结构 ROI 受保护"),
        ("03_step_IIb_kriging_terrain.png", "IIb: Kriging terrain", "Kriging 地形重建"),
        ("04_step_IIb_multilayer_expanded.png", "IIb: expanded same GS model", "同一 GS 模型的多层扩展"),
        ("04b_step_IIb_terrain_expansion_before_after_topdown.png", "IIb: top-down terrain extension", "俯视前后对照：原始已有地形，红色为新增 Gaussian 中心"),
        ("05_step_III_registered_FE_displacement.png", "III: archived FE field replay", "历史 FE 位移接入；本次未重跑 FE 求解"),
        ("06_step_IV_deformation_before_after.png", "IV: same moving Gaussian centers", "同一动态高斯子集的中心变形诊断图"),
        ("moving_before_after_gs.png", "IV: real Gaussian rasterization", "真实 GS 光栅化帧对照；静态环境隐藏"),
        (f"07_step_IV_gsplat_dynamic_frame_{frame_a:05d}.png", "IV: complete scene / unresolved haze", "同一扩展模型的完整 GS 场景：仍有明显雾状遮挡，清晰度未通过"),
    ]
    columns = 3
    rows = math.ceil(len(entries) / columns)
    overview = Image.new("RGB", (700 * columns, 75 + 470 * rows), "#e5e7eb")
    ImageDraw.Draw(overview).text((20, 14), "Continuous real tower chain | diagnostic projections + real GS renders", font=font(27), fill="#111827")
    for i, (name, label, _) in enumerate(entries):
        overview.paste(panel(out / name, label), ((i % columns) * 700, 75 + (i // columns) * 470))
    overview.save(out / "overview.png")

    frame_paths = sorted((animation / "moving_frames").glob("frame_*.png"))
    images = []
    for path in frame_paths:
        with Image.open(path) as im:
            images.append(ImageOps.contain(im.convert("RGB"), (512, 768)).quantize(colors=128))
    if len(images) < 2:
        raise ValueError("at least two real gsplat moving-subset frames are required for animation")
    images[0].save(out / "moving_dynamic.gif", save_all=True, append_images=images[1:], duration=100, loop=0, disposal=2)
    blocks = []
    for name, _, caption in entries:
        blocks.append(f'<section><h2>{html.escape(caption)}</h2><a href="{html.escape(name)}"><img src="{html.escape(name)}"></a></section>')
    (out / "index.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>真实塔体连续高斯流程</title>'
        '<style>body{max-width:1400px;margin:30px auto;padding:0 24px;font:17px/1.7 sans-serif;color:#20252b;background:#f7f8fa}section{background:white;padding:20px;margin:24px 0}img{display:block;max-width:100%;max-height:1100px;margin:auto}h2{font-size:22px}</style>'
        '<h1>真实塔体：连续高斯流程与逐步结果</h1>'
        '<p>已有初始 3DGS → 已有重要区域加密/剪枝训练结果 → 本次地形处理与扩展 → 同一模型动态子集接入历史 FE 位移 → 真实 gsplat 渲染。</p>'
        '<p>诊断图按物理 FE +Z 向上显示；原模型与位移坐标未翻转。最终 GS 图由相同模型的均值、四元数、尺度、不透明度和球谐系数实际光栅化生成。</p>'
        '<p>本次未重跑长训练或 ANSYS 求解。动态部分仅重放平移位移，展示放大 50 倍；未完成论文 Eq.20 及旋转、尺度、协方差的全参数动态重放。完整场景与隐藏静态环境的塔体子集分别展示，不能把子集图当完整场景质量证明。</p>'
        '<p>档案第 0 帧已有非零位移；前后对照左图单独使用零位移渲染，右图使用第 37 帧，并非把第 0 帧当作未变形。</p>'
        '<section><h2>动态高斯子集播放</h2><p>40张真实GS帧；100ms仅为浏览播放间隔，不代表FE物理时间步。</p><img src="moving_dynamic.gif"></section>'
        + ''.join(blocks) + '</html>\n', encoding="utf-8")
    (out / "gallery_manifest.json").write_text(json.dumps({
        "source_chain_manifest": str(chain / "continuous_chain_manifest.json"),
        "images": [entry[0] for entry in entries], "animation_source_frames": [str(p) for p in frame_paths],
        "animation": "moving_dynamic.gif", "animation_frame_duration_ms": 100,
        "note": "Image assembly only; no synthetic frames, inpainting, recolouring or model changes.",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"gallery": str(out / "index.html"), "animation_frames": len(images)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
