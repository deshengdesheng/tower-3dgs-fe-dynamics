#!/usr/bin/env python3
"""SfM initialization diagnostics grouped by image texture and illumination.

The script evaluates the existing COLMAP sparse model against an original image
folder. It does not rerun SfM or 3DGS; it quantifies whether low-texture or
low/high-illumination images are less supported in the available SfM result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import struct
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF"}


@dataclass
class RegisteredImage:
    image_id: int
    camera_id: int
    name: str
    observations: int


@dataclass
class ImageQuality:
    rel_path: str
    file_name: str
    source: str
    texture_laplacian_var: float
    mean_gray: float
    dark_ratio: float
    bright_ratio: float
    registered: bool
    observations: int
    matched_colmap_name: str


def read_c_string(f) -> str:
    chars: list[bytes] = []
    while True:
        c = f.read(1)
        if not c:
            raise EOFError("Unexpected EOF while reading image name")
        if c == b"\x00":
            return b"".join(chars).decode("utf-8", errors="replace")
        chars.append(c)


def read_images_binary(path: Path) -> list[RegisteredImage]:
    images: list[RegisteredImage] = []
    with path.open("rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<i", f.read(4))[0]
            f.seek(32 + 24, os.SEEK_CUR)  # qvec + tvec
            camera_id = struct.unpack("<i", f.read(4))[0]
            name = read_c_string(f)
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            valid_observations = 0
            # POINT2D: x(double), y(double), point3D_id(int64). Only point3D_id != -1
            # 是真正连到稀疏三维点的 2D-3D 观测。
            for _point_idx in range(num_points2d):
                f.seek(16, os.SEEK_CUR)
                point3d_id = struct.unpack("<q", f.read(8))[0]
                if point3d_id != -1:
                    valid_observations += 1
            images.append(RegisteredImage(image_id, camera_id, name, valid_observations))
    return images


def read_points3d_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        return int(struct.unpack("<Q", f.read(8))[0])


def iter_images(image_root: Path) -> list[Path]:
    return sorted(p for p in image_root.rglob("*") if p.is_file() and p.suffix in IMAGE_EXTS)


def source_key(rel_path: str, camera_prefix: str | None = None) -> str:
    parts = Path(rel_path).parts
    if len(parts) > 1:
        if has_camera_prefix(parts[0], camera_prefix):
            return "_".join(parts[:2])
        return parts[0]
    name = Path(rel_path).name
    # Keep numeric source identifiers grouped when directory structure was flattened.
    stem_parts = name.split("_")
    if len(stem_parts) >= 2 and stem_parts[0].isdigit():
        return "_".join(stem_parts[:2])
    if len(stem_parts) >= 2 and has_camera_prefix(stem_parts[0], camera_prefix):
        return "_".join(stem_parts[:2])
    return "unknown"


def flat_image_key(rel_path: str) -> str:
    p = Path(rel_path.replace("\\", "/"))
    if len(p.parts) > 1:
        return "_".join(p.parts)
    return p.name


def has_camera_prefix(key: str, camera_prefix: str | None) -> bool:
    return bool(camera_prefix) and key.lower().startswith(camera_prefix.lower())


def parse_cc_export_image_keys(
    xml_path: Path,
    dataset_marker: str | None = None,
    camera_prefix: str | None = None,
) -> tuple[set[str], dict[str, int]]:
    text = xml_path.read_text(errors="ignore")
    matches = re.findall(r"[^<>\"']+\.(?:JPG|JPEG|PNG|TIF|TIFF)", text, flags=re.I)
    keys: set[str] = set()
    counts = {"xml_image_paths": len(matches), "xml_camera_prefix_paths": 0, "xml_other_paths": 0}
    for raw in matches:
        path = raw.replace("\\", "/")
        marker = dataset_marker.replace("\\", "/").strip("/") if dataset_marker else ""
        marker_with_separator = f"{marker}/"
        marker_index = path.lower().find(marker_with_separator.lower()) if marker else -1
        if marker_index >= 0:
            rel = path[marker_index + len(marker_with_separator):]
        else:
            rel = Path(path).name
        key = flat_image_key(rel)
        keys.add(key)
        if has_camera_prefix(key, camera_prefix):
            counts["xml_camera_prefix_paths"] += 1
        else:
            counts["xml_other_paths"] += 1
    counts["xml_unique_flat_keys"] = len(keys)
    return keys, counts


def make_name_keys(rel_path: str) -> set[str]:
    p = Path(rel_path)
    keys = {rel_path.replace("\\", "/"), p.name, flat_image_key(rel_path)}
    if len(p.parts) > 1:
        keys.add("_".join(p.parts))
        keys.add("_".join(p.parts[-2:]))
        keys.add("/".join(p.parts[-2:]))
    return keys


def build_registration_lookup(registered: Iterable[RegisteredImage]) -> dict[str, RegisteredImage]:
    lookup: dict[str, RegisteredImage] = {}
    for item in registered:
        name = item.name.replace("\\", "/")
        p = Path(name)
        keys = {name, p.name, flat_image_key(name)}
        if len(p.parts) > 1:
            keys.add("_".join(p.parts))
            keys.add("_".join(p.parts[-2:]))
            keys.add("/".join(p.parts[-2:]))
        for key in keys:
            lookup.setdefault(key, item)
    return lookup


def image_quality(path: Path, max_side: int) -> tuple[float, float, float, float]:
    with Image.open(path) as img:
        # JPEG draft 模式可避免从 H 盘读取并解码完整大图，分组诊断不需要原始分辨率。
        try:
            img.draft("L", (max_side, max_side))
        except Exception:
            pass
        gray = img.convert("L")
        w, h = gray.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
            gray = gray.resize((max(1, int(w * scale)), max(1, int(h * scale))), resample)
        arr = np.asarray(gray, dtype=np.float32)

    mean_gray = float(arr.mean())
    dark_ratio = float((arr < 30.0).mean())
    bright_ratio = float((arr > 225.0).mean())

    # 拉普拉斯方差作为纹理/边缘丰富度代理，低值通常对应低纹理或模糊图像。
    padded = np.pad(arr, 1, mode="edge")
    lap = (
        -4.0 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    texture = float(lap.var())
    return texture, mean_gray, dark_ratio, bright_ratio


def tercile_labels(values: list[float], low_is_challenging: bool = True) -> list[str]:
    q1, q2 = np.quantile(np.asarray(values, dtype=np.float64), [1 / 3, 2 / 3])
    labels = []
    for v in values:
        if v <= q1:
            labels.append("low")
        elif v <= q2:
            labels.append("medium")
        else:
            labels.append("high")
    return labels


def brightness_condition(values: list[float]) -> list[str]:
    # 亮度的挑战性通常是两端：过暗或过亮。这里仍按 low/medium/high 灰度分组。
    return tercile_labels(values)


def summarize(rows: list[ImageQuality], labels: list[str], group_name: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for group in ["low", "medium", "high"]:
        group_rows = [r for r, lab in zip(rows, labels) if lab == group]
        reg_rows = [r for r in group_rows if r.registered]
        obs = [r.observations for r in reg_rows]
        out.append(
            {
                "grouping": group_name,
                "group": group,
                "images": len(group_rows),
                "registered_images": len(reg_rows),
                "registration_rate_percent": round(100.0 * len(reg_rows) / len(group_rows), 2) if group_rows else 0.0,
                "mean_observations": round(mean(obs), 1) if obs else 0.0,
                "median_observations": round(median(obs), 1) if obs else 0.0,
                "min_observations": min(obs) if obs else 0,
                "obs_lt_500": sum(r.observations < 500 for r in reg_rows),
                "obs_lt_100": sum(r.observations < 100 for r in reg_rows),
                "mean_texture": round(mean([r.texture_laplacian_var for r in group_rows]), 2) if group_rows else 0.0,
                "mean_gray": round(mean([r.mean_gray for r in group_rows]), 2) if group_rows else 0.0,
                "mean_dark_ratio_percent": round(100.0 * mean([r.dark_ratio for r in group_rows]), 2) if group_rows else 0.0,
                "mean_bright_ratio_percent": round(100.0 * mean([r.bright_ratio for r in group_rows]), 2) if group_rows else 0.0,
            }
        )
    return out


def source_stratified_summary(
    rows: list[ImageQuality],
    value_getter,
    group_name: str,
    min_source_images: int,
) -> list[dict[str, object]]:
    source_to_rows: dict[str, list[ImageQuality]] = {}
    for row in rows:
        source_to_rows.setdefault(row.source, []).append(row)

    grouped_rows = {"low": [], "medium": [], "high": []}
    used_sources = 0
    for source_rows in source_to_rows.values():
        if len(source_rows) < min_source_images:
            continue
        used_sources += 1
        labels = tercile_labels([value_getter(r) for r in source_rows])
        for row, label in zip(source_rows, labels):
            grouped_rows[label].append(row)

    out: list[dict[str, object]] = []
    for group in ["low", "medium", "high"]:
        group_rows = grouped_rows[group]
        reg_rows = [r for r in group_rows if r.registered]
        obs = [r.observations for r in reg_rows]
        out.append(
            {
                "grouping": group_name,
                "group": group,
                "images": len(group_rows),
                "registered_images": len(reg_rows),
                "registration_rate_percent": round(100.0 * len(reg_rows) / len(group_rows), 2) if group_rows else 0.0,
                "mean_observations": round(mean(obs), 1) if obs else 0.0,
                "median_observations": round(median(obs), 1) if obs else 0.0,
                "min_observations": min(obs) if obs else 0,
                "obs_lt_500": sum(r.observations < 500 for r in reg_rows),
                "obs_lt_100": sum(r.observations < 100 for r in reg_rows),
                "used_sources": used_sources,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--quality-image-root", type=Path, default=None)
    parser.add_argument("--sparse-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--cc-export-xml", type=Path, default=None)
    parser.add_argument("--dataset-marker", default=None, help="Optional path component used to locate image names in the XML export.")
    parser.add_argument("--camera-prefix", default=None, help="Optional image-name prefix used for XML matching statistics.")
    parser.add_argument(
        "--require-camera-prefix-in-xml",
        action="store_true",
        help="Exclude images with --camera-prefix when they are absent from the XML export.",
    )
    parser.add_argument("--source-stratified-min-images", type=int, default=30)
    parser.add_argument(
        "--exclude-roots",
        nargs="*",
        default=[],
        help="Top-level directories to exclude from the image set.",
    )
    args = parser.parse_args()
    if args.require_camera_prefix_in_xml and not args.camera_prefix:
        parser.error("--require-camera-prefix-in-xml requires --camera-prefix")
    if args.require_camera_prefix_in_xml and args.cc_export_xml is None:
        parser.error("--require-camera-prefix-in-xml requires --cc-export-xml")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    registered = read_images_binary(args.sparse_dir / "images.bin")
    points3d_count = read_points3d_count(args.sparse_dir / "points3D.bin")
    lookup = build_registration_lookup(registered)
    quality_root = args.quality_image_root or args.image_root
    xml_keys: set[str] = set()
    xml_counts: dict[str, int] = {}
    if args.cc_export_xml is not None:
        xml_keys, xml_counts = parse_cc_export_image_keys(
            args.cc_export_xml,
            dataset_marker=args.dataset_marker,
            camera_prefix=args.camera_prefix,
        )

    excluded_roots = set(args.exclude_roots)
    image_paths: list[Path] = []
    candidate_counts = {
        "scanned_images": 0,
        "included_images": 0,
        "excluded_by_root": 0,
        "excluded_camera_prefix_not_in_xml": 0,
        "included_camera_prefix_from_xml": 0,
        "included_without_camera_prefix": 0,
    }
    for p in iter_images(quality_root):
        rel = p.relative_to(quality_root).as_posix()
        candidate_counts["scanned_images"] += 1
        top = Path(rel).parts[0] if Path(rel).parts else Path(rel).name
        if top in excluded_roots:
            candidate_counts["excluded_by_root"] += 1
            continue
        key = flat_image_key(rel)
        if args.require_camera_prefix_in_xml and has_camera_prefix(key, args.camera_prefix):
            if key not in xml_keys:
                candidate_counts["excluded_camera_prefix_not_in_xml"] += 1
                continue
            candidate_counts["included_camera_prefix_from_xml"] += 1
        else:
            candidate_counts["included_without_camera_prefix"] += 1
        image_paths.append(p)
    candidate_counts["included_images"] = len(image_paths)

    rows: list[ImageQuality] = []
    unmatched_registered = set(item.name for item in registered)
    for idx, path in enumerate(image_paths, 1):
        rel = path.relative_to(quality_root).as_posix()
        matched = None
        matched_key = ""
        for key in make_name_keys(rel):
            if key in lookup:
                matched = lookup[key]
                matched_key = lookup[key].name
                break
        texture, mean_gray, dark_ratio, bright_ratio = image_quality(path, args.max_side)
        if matched is not None:
            unmatched_registered.discard(matched.name)
        rows.append(
            ImageQuality(
                rel_path=rel,
                file_name=path.name,
                source=source_key(rel, args.camera_prefix),
                texture_laplacian_var=texture,
                mean_gray=mean_gray,
                dark_ratio=dark_ratio,
                bright_ratio=bright_ratio,
                registered=matched is not None,
                observations=matched.observations if matched is not None else 0,
                matched_colmap_name=matched_key,
            )
        )
        if idx % 500 == 0:
            print(f"processed {idx}/{len(image_paths)} images", flush=True)

    texture_labels = tercile_labels([r.texture_laplacian_var for r in rows])
    brightness_labels = brightness_condition([r.mean_gray for r in rows])
    texture_summary = summarize(rows, texture_labels, "texture_laplacian_variance")
    brightness_summary = summarize(rows, brightness_labels, "mean_brightness")
    summary_rows = texture_summary + brightness_summary
    source_summary_rows = source_stratified_summary(
        rows,
        lambda r: r.texture_laplacian_var,
        "texture_within_source_group",
        args.source_stratified_min_images,
    ) + source_stratified_summary(
        rows,
        lambda r: r.mean_gray,
        "brightness_within_source_group",
        args.source_stratified_min_images,
    )

    global_obs = [item.observations for item in registered]
    global_summary = {
        "image_root": str(args.image_root),
        "quality_image_root": str(quality_root),
        "cc_export_xml": str(args.cc_export_xml) if args.cc_export_xml else None,
        "candidate_rule": f"images beginning with {args.camera_prefix!r} are restricted to the XML export; other images are kept"
        if args.require_camera_prefix_in_xml
        else "all images after excluded roots",
        **xml_counts,
        **candidate_counts,
        "total_images_found": len(rows),
        "registered_images_in_colmap": len(registered),
        "registered_images_matched_to_image_root": sum(r.registered for r in rows),
        "unmatched_registered_names": len(unmatched_registered),
        "sparse_points3D": points3d_count,
        "global_registration_rate_percent_vs_image_root": round(100.0 * sum(r.registered for r in rows) / len(rows), 2) if rows else 0.0,
        "mean_observations_registered": round(mean(global_obs), 1) if global_obs else 0.0,
        "median_observations_registered": round(median(global_obs), 1) if global_obs else 0.0,
        "min_observations_registered": min(global_obs) if global_obs else 0,
        "registered_obs_lt_500": sum(o < 500 for o in global_obs),
        "registered_obs_lt_100": sum(o < 100 for o in global_obs),
    }

    write_csv(args.output_dir / "sfm_quality_group_summary.csv", summary_rows)
    write_csv(args.output_dir / "sfm_quality_source_stratified_summary.csv", source_summary_rows)
    with (args.output_dir / "sfm_quality_per_image.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(asdict(rows[0]).keys()) + ["texture_group", "brightness_group"] if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, t_lab, b_lab in zip(rows, texture_labels, brightness_labels):
            item = asdict(row)
            item["texture_group"] = t_lab
            item["brightness_group"] = b_lab
            writer.writerow(item)

    weak_rows = sorted([r for r in rows if r.registered and r.observations < 500], key=lambda r: r.observations)[:50]
    unregistered_low_texture = [
        (r, lab) for r, lab in zip(rows, texture_labels) if (not r.registered and lab == "low")
    ][:50]
    write_csv(args.output_dir / "sfm_weak_registered_examples.csv", [asdict(r) for r in weak_rows])
    write_csv(args.output_dir / "sfm_unregistered_low_texture_examples.csv", [asdict(r) for r, _ in unregistered_low_texture])
    (args.output_dir / "sfm_global_summary.json").write_text(json.dumps(global_summary, indent=2), encoding="utf-8")
    (args.output_dir / "unmatched_registered_names.json").write_text(
        json.dumps(sorted(unmatched_registered)[:500], indent=2), encoding="utf-8"
    )

    columns = [
        "grouping",
        "group",
        "images",
        "registered_images",
        "registration_rate_percent",
        "mean_observations",
        "median_observations",
        "min_observations",
        "obs_lt_500",
        "obs_lt_100",
        "mean_texture",
        "mean_gray",
        "mean_dark_ratio_percent",
        "mean_bright_ratio_percent",
    ]
    md = []
    md.append("# SfM initialization quality grouped by image texture and brightness\n")
    md.append("## Global summary\n")
    for k, v in global_summary.items():
        md.append(f"- {k}: {v}")
    md.append("\n## Grouped summary\n")
    md.append(markdown_table(summary_rows, columns))
    md.append("\n## Source-stratified summary\n")
    source_columns = [
        "grouping",
        "group",
        "images",
        "registered_images",
        "registration_rate_percent",
        "mean_observations",
        "median_observations",
        "min_observations",
        "obs_lt_500",
        "obs_lt_100",
        "used_sources",
    ]
    md.append(markdown_table(source_summary_rows, source_columns))
    md.append("\n## Notes\n")
    md.append(
        "- Texture is measured by grayscale Laplacian variance after image down-sampling. "
        "Lower values indicate weaker local texture or blur."
    )
    md.append("- Brightness is measured by mean grayscale intensity; low/high groups may indicate under-/over-illumination conditions.")
    md.append("- Observations are the number of COLMAP 2D keypoints in a registered image with associated point3D identifiers.")
    md.append("- This diagnostic uses the existing COLMAP/SfM result and does not rerun 3DGS.")
    (args.output_dir / "sfm_quality_group_summary.md").write_text("\n".join(md), encoding="utf-8")

    paper_md = []
    paper_md.append("# Paper-ready SfM initialization robustness diagnostics\n")
    paper_md.append("## Recommended table for manuscript\n")
    if args.require_camera_prefix_in_xml:
        image_selection_note = (
            f"Images beginning with `{args.camera_prefix}` are restricted to names present in the supplied XML export; "
            "other images are kept. "
        )
    else:
        image_selection_note = "All images are considered except those under explicitly excluded roots. "
    paper_md.append(
        image_selection_note
        + "Grouping is performed within each acquisition source/channel before aggregation.\n"
    )
    paper_md.append(markdown_table(source_summary_rows, source_columns))
    paper_md.append("\n## Global diagnostic table\n")
    paper_md.append(markdown_table(summary_rows, columns))
    paper_md.append("\n## Key interpretation\n")
    paper_md.append(
        f"- The evaluated SfM candidate set contains {len(rows)} images; "
        f"{sum(r.registered for r in rows)} images are registered in the COLMAP model, "
        f"with {points3d_count} sparse 3D points."
    )
    paper_md.append(
        f"- Registered images have {global_summary['mean_observations_registered']} 2D-3D observations per image on average "
        f"and a median of {global_summary['median_observations_registered']} observations."
    )
    paper_md.append(
        "- Texture and brightness grouping quantify the sensitivity of SfM support to low-texture "
        "and illumination conditions without rerunning 3DGS."
    )
    paper_md.append(
        "- The results support the use of the current SfM initialization for this acquisition, "
        "while the low-quality groups identify conditions where registration becomes less robust."
    )
    (args.output_dir / "sfm_initialization_robustness_paper_ready.md").write_text("\n".join(paper_md), encoding="utf-8")

    print(json.dumps(global_summary, indent=2))
    print(markdown_table(summary_rows, columns))
    print(markdown_table(source_summary_rows, source_columns))
    print(f"saved to {args.output_dir}")


if __name__ == "__main__":
    main()
