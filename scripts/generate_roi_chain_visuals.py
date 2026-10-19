#!/usr/bin/env python3
"""Render auditable figures for the continuous real-tower Gaussian chain.

The Gaussian and FE data remain unchanged. Only the camera/diagnostic basis is
rotated so its vertical axis follows FE +Z through the supplied registration
matrix. The Step-IIa panel compares the existing pre-ROI and ROI-trained
checkpoints; later stages consume the ROI-trained model's Step-II outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow.coordinate_frames import display_basis, physical_up_in_gaussian_frame
from gsplat.strategy.irregular_region_strategy import IrregularRegionFilteredStrategy


def load(path: Path) -> dict[str, torch.Tensor]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if isinstance(value, dict) and "splats" in value:
        value = value["splats"]
    return {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in value.items()}


def xyz(state: dict[str, torch.Tensor], limit: int | None = None, seed: int = 0) -> np.ndarray:
    """Return a plotting sample without relying on torch's NumPy bridge.

    Some of the author environments use a PyTorch build whose NumPy bridge is
    unavailable even though NumPy itself can plot normally.  Sampling with
    torch first keeps this diagnostic generator portable and avoids converting
    the 1.6M point checkpoint to a Python list.
    """
    values = state["means"].float().cpu()
    if limit is not None and values.shape[0] > limit:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        indices = torch.randperm(values.shape[0], generator=generator)[:limit]
        values = values[indices]
    return np.asarray(values.tolist(), dtype=np.float32)


def display_xyz(points: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project coordinates into a local frame whose vertical axis is physical up."""
    return np.asarray(points, dtype=np.float32) @ basis


def sample(points: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if len(points) <= limit:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), size=limit, replace=False)]


def sample_roi_checkpoint(state: dict[str, torch.Tensor], strategy, limit: int,
                          seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    means = state["means"].float().cpu()
    mask = strategy._get_region_mask(means).cpu()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randperm(len(means), generator=generator)[:min(len(means), limit)]
    points = np.asarray(means.index_select(0, indices).tolist(), dtype=np.float32)
    return points, mask.index_select(0, indices).numpy(), int(mask.sum().item())


def save_roi_refinement(path: Path, initial: dict[str, torch.Tensor], refined: dict[str, torch.Tensor],
                        basis: np.ndarray, limit: int, roi_cache: Path) -> dict[str, int]:
    strategy = IrregularRegionFilteredStrategy.from_cache(str(roi_cache))
    samples = []
    stats = {}
    for label, state, seed in (("399999", initial, 7), ("449999", refined, 8)):
        points, roi_sample, roi_count = sample_roi_checkpoint(state, strategy, limit, seed)
        count = int(state["means"].shape[0])
        samples.append((label, points, display_xyz(points, basis), roi_sample))
        stats[label] = {"gaussian_count": count, "exact_capsule_roi_count": roi_count,
                        "exact_capsule_roi_fraction": roi_count / max(count, 1),
                        "sampled_roi_count": int(roi_sample.sum())}
    all_sampled = np.concatenate([entry[2] for entry in samples], axis=0)
    lo = np.percentile(all_sampled, 0.5, axis=0)
    hi = np.percentile(all_sampled, 99.5, axis=0)
    pad = np.maximum((hi - lo) * 0.04, 1e-3)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.6), dpi=180)
    fig.subplots_adjust(top=0.84, bottom=0.07, left=0.07, right=0.98, hspace=0.28, wspace=0.18)
    for row, (label, _points, shown, roi_mask) in enumerate(samples):
        outside = ~roi_mask
        count = stats[label]["gaussian_count"]
        roi_count = stats[label]["exact_capsule_roi_count"]
        fraction = 100 * stats[label]["exact_capsule_roi_fraction"]
        for col in range(2):
            ax = axes[row, col]
            horizontal = col
            ax.scatter(shown[outside, horizontal], shown[outside, 2], color="#9da4ad",
                       s=0.48, alpha=0.32, linewidths=0, rasterized=True)
            ax.scatter(shown[roi_mask, horizontal], shown[roi_mask, 2], color="#d14f3d",
                       s=0.72, alpha=0.88, linewidths=0, rasterized=True)
            ax.set_xlim(float(lo[horizontal] - pad[horizontal]), float(hi[horizontal] + pad[horizontal]))
            ax.set_ylim(float(lo[2] - pad[2]), float(hi[2] + pad[2]))
            stage_label = "Before ROI training" if row == 0 else "After ROI training"
            ax.set_title(f"{stage_label} ({label}) | {'front' if col == 0 else 'side'}", fontsize=10)
            ax.set_xlabel("Horizontal X'" if col == 0 else "Horizontal Y'")
            ax.set_ylabel("Physical elevation H (up)")
            ax.grid(True, alpha=0.2)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.tick_params(labelsize=8)
            if col == 0:
                ax.text(0.02, 0.98, f"N={count:,}\nexact ROI={roi_count:,} ({fraction:.2f}%)",
                        transform=ax.transAxes, va="top", ha="left", fontsize=8,
                        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#9da4ad",
                     markersize=6, label="Outside exact-capsule ROI"),
              Line2D([0], [0], marker="o", color="w", markerfacecolor="#d14f3d",
                     markersize=6, label="Inside exact-capsule ROI")]
    fig.legend(handles=legend, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.905))
    fig.suptitle("Paper Step IIa: ROI-guided training result (399999 → 449999)\n"
                 "Exact capsule ROI; Gaussian-center diagnostic; archived training result", fontsize=12, y=0.98)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    save_single(path.with_name("00_step_I_initial_checkpoint.png"), samples[0][1], samples[0][1],
                f"Step I / initial tower 3DGS checkpoint (step 399999; N={stats['399999']['gaussian_count']:,})",
                basis, limit, 7)
    return stats


def bounds(all_points: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shown = display_xyz(all_points, basis)
    lo = np.percentile(shown, 0.5, axis=0)
    hi = np.percentile(shown, 99.5, axis=0)
    span = np.maximum(hi - lo, 1e-3)
    pad = span * 0.05
    return lo - pad, hi + pad


def projection_limits(points: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shown = display_xyz(points, basis)
    lo = np.percentile(shown, 0.1, axis=0)
    hi = np.percentile(shown, 99.9, axis=0)
    pad = np.maximum((hi - lo) * 0.08, 1e-3)
    return lo - pad, hi + pad


def profile(ax, points: np.ndarray, horizontal_axis: int,
            limits: tuple[np.ndarray, np.ndarray], title: str,
            basis: np.ndarray, color: str, size: float = 0.55):
    shown = display_xyz(points, basis)
    lo, hi = limits
    horizontal_name = "Horizontal X'" if horizontal_axis == 0 else "Horizontal Y'"
    ax.scatter(shown[:, horizontal_axis], shown[:, 2], color=color,
               s=size, alpha=0.58, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(horizontal_name)
    ax.set_ylabel("Physical elevation H (up)")
    ax.set_xlim(float(lo[horizontal_axis]), float(hi[horizontal_axis]))
    ax.set_ylim(float(lo[2]), float(hi[2]))
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(labelsize=8)


def save_map(path: Path, points: np.ndarray, basis: np.ndarray, title: str,
             limit: int, seed: int) -> None:
    lo, hi = projection_limits(points, basis)
    shown = display_xyz(sample(points, limit, seed), basis)
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5), dpi=180, constrained_layout=True)
    color = axes[0].scatter(shown[:, 0], shown[:, 1], c=shown[:, 2], cmap="terrain",
                            s=1.0, alpha=0.75, linewidths=0, rasterized=True)
    axes[0].set_title("Plan view; color = physical elevation")
    axes[0].set_xlabel("Horizontal X'")
    axes[0].set_ylabel("Horizontal Y'")
    axes[0].set_xlim(float(lo[0]), float(hi[0]))
    axes[0].set_ylim(float(lo[1]), float(hi[1]))
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(True, alpha=0.18)
    fig.colorbar(color, ax=axes[0], shrink=0.8, label="Physical elevation H")
    profile(axes[1], sample(points, limit, seed + 1), 0, (lo, hi),
            "Elevation profile", basis, "#2c7a62", size=1.0)
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_topdown_expansion(path: Path, source_means: torch.Tensor,
                           expanded_state: dict[str, torch.Tensor], basis: np.ndarray,
                           limit: int, seed: int, parent_index_path: Path) -> dict[str, int]:
    """Compare the existing terrain-inclusive model with its appended centers."""
    base = source_means.detach().cpu().float().contiguous()
    expanded = expanded_state["means"].detach().cpu().float().contiguous()
    base_count = int(base.shape[0])
    expanded_count = int(expanded.shape[0])
    if expanded_count < base_count:
        raise ValueError("expanded model has fewer Gaussians than the Step-IIb input")
    if not torch.equal(expanded[:base_count], base):
        raise ValueError("Step-IIb output does not preserve the source checkpoint's Gaussian-center prefix")

    parent_indices = np.load(parent_index_path, mmap_mode="r")
    if len(parent_indices) != expanded_count:
        raise ValueError("Step-IIb parent-index map length does not match the expanded model")
    if not np.array_equal(parent_indices[:base_count], np.arange(base_count, dtype=parent_indices.dtype)):
        raise ValueError("Step-IIb parent-index map does not preserve source rows")
    added_parent_indices = np.asarray(parent_indices[base_count:])
    if np.any(added_parent_indices < 0) or np.any(added_parent_indices >= base_count):
        raise ValueError("Step-IIb added Gaussian has an invalid source-parent index")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    sampled_indices = torch.randperm(base_count, generator=generator)[:min(base_count, limit)]
    base_points = np.asarray(base.index_select(0, sampled_indices).tolist(), dtype=np.float32)
    added_points = np.asarray(expanded[base_count:].tolist(), dtype=np.float32)
    base_shown = display_xyz(base_points, basis)
    added_shown = display_xyz(added_points, basis)
    all_shown = np.concatenate([base_shown, added_shown], axis=0)
    lo = np.percentile(all_shown[:, :2], 0.1, axis=0)
    hi = np.percentile(all_shown[:, :2], 99.9, axis=0)
    pad = np.maximum((hi - lo) * 0.05, 1e-3)
    lo, hi = lo - pad, hi + pad
    elevation = np.concatenate([base_shown[:, 2], added_shown[:, 2]])
    vmin, vmax = np.percentile(elevation, [1, 99]) if len(elevation) else (-1, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8), dpi=180, constrained_layout=True)
    for ax in axes:
        ax.set_xlim(float(lo[0]), float(hi[0]))
        ax.set_ylim(float(lo[1]), float(hi[1]))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Horizontal X' (physical plan view)")
        ax.set_ylabel("Horizontal Y' (physical plan view)")
        ax.grid(True, alpha=0.18)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.tick_params(labelsize=8)

    color = axes[0].scatter(base_shown[:, 0], base_shown[:, 1], c=base_shown[:, 2],
                            cmap="terrain", vmin=vmin, vmax=vmax, s=1.0, alpha=0.78,
                            linewidths=0, rasterized=True)
    axes[0].set_title(
        f"Before IIb: existing 449999 checkpoint\n"
        f"{base_count:,} total centers; plotting {len(base_points):,} sampled; terrain present"
    )
    axes[1].scatter(base_shown[:, 0], base_shown[:, 1], c=base_shown[:, 2],
                    cmap="terrain", vmin=vmin, vmax=vmax, s=1.0, alpha=0.52,
                    linewidths=0, rasterized=True)
    axes[1].scatter(added_shown[:, 0], added_shown[:, 1], color="#c43b30",
                    s=1.2, alpha=0.84, linewidths=0, rasterized=True)
    axes[1].set_title(
        f"After IIb: same {len(base_points):,} source sample + "
        f"{expanded_count - base_count:,} appended centers"
    )
    axes[1].legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#c43b30",
               markersize=6, label="Newly appended Gaussian centers"),
    ], loc="best", frameon=True, fontsize=8)
    fig.colorbar(color, ax=axes, shrink=0.78, label="Physical elevation H (up)")
    fig.suptitle(
        "Step IIb terrain-extension footprint | identical top-down camera and axis limits\n"
        "Gaussian-center diagnostic, not a rendered image; red marks only the newly appended centers",
        fontsize=12,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {
        "source_count": base_count,
        "expanded_count": expanded_count,
        "appended_count": expanded_count - base_count,
        "source_prefix_exact": True,
        "parent_index_map_valid": True,
        "base_centers_sampled": int(len(base_points)),
        "appended_centers_shown": int(len(added_points)),
    }


def save_pair(path: Path, left: np.ndarray, right: np.ndarray, all_points: np.ndarray,
              titles: tuple[str, str], colors: tuple[str, str], suptitle: str,
              basis: np.ndarray, limit: int, seed: int, shared_bounds: bool = True,
              paired_samples: bool = False) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=180, constrained_layout=True)
    for row, (points, title, color) in enumerate(zip((left, right), titles, colors)):
        sample_seed = seed + 1 if paired_samples or row == 0 else seed + 2
        panel_points = sample(points, limit, sample_seed)
        limits = projection_limits(all_points if shared_bounds else panel_points, basis)
        profile(axes[row, 0], panel_points, 0, limits, f"{title} | front", basis, color)
        profile(axes[row, 1], panel_points, 1, limits, f"{title} | side", basis, color)
    fig.suptitle(suptitle + "\nGaussian-center diagnostic; physical elevation increases upward", fontsize=12)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_single(path: Path, points: np.ndarray, all_points: np.ndarray, title: str,
                basis: np.ndarray, limit: int, seed: int, cmap="viridis",
                map_view: bool = False) -> None:
    if map_view:
        save_map(path, points, basis, title, limit, seed)
        return
    lo, hi = projection_limits(all_points, basis)
    q = sample(points, limit, seed)
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5), dpi=180, constrained_layout=True)
    profile(axes[0], q, 0, (lo, hi), "Front elevation", basis, "#2f6f9f", size=0.8)
    profile(axes[1], q, 1, (lo, hi), "Side elevation", basis, "#b94f43", size=0.8)
    fig.suptitle(title + "\nGaussian-center diagnostic; physical elevation increases upward", fontsize=12)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gs-to-fe-transform", type=Path, required=True,
                        help="4x4 matrix mapping Gaussian coordinates to FE coordinates")
    parser.add_argument("--max-points", type=int, default=60000)
    parser.add_argument("--disp-scale", type=float, default=50.0)
    parser.add_argument("--disp-frame", type=int, default=37)
    parser.add_argument("--initial", type=Path, required=True,
                        help="real checkpoint immediately before ROI-guided training")
    parser.add_argument("--roi-trained", type=Path, required=True,
                        help="real checkpoint after ROI-guided training/densification/pruning")
    parser.add_argument("--terrain", type=Path, default=None)
    parser.add_argument("--objects", type=Path, default=None)
    parser.add_argument("--kriging", type=Path, default=None)
    parser.add_argument("--expanded", type=Path, default=None)
    parser.add_argument("--static", type=Path, default=None)
    parser.add_argument("--moving", type=Path, default=None)
    parser.add_argument("--vibration", type=Path, default=None)
    parser.add_argument("--dynamic-render-dir", type=Path, default=None,
                        help="optional gsplat render directory containing frames/frame_*.png")
    parser.add_argument("--roi-cache", type=Path, default=None,
                        help="paper FE-ROI cache used by the supplied training run; recorded as provenance")
    args = parser.parse_args()
    chain = args.chain_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    roi_cache_path = args.roi_cache.expanduser().resolve() if args.roi_cache else None
    if roi_cache_path is not None and not roi_cache_path.is_file():
        raise FileNotFoundError(roi_cache_path)
    if roi_cache_path is None:
        raise ValueError("--roi-cache is required to visualize the exact-capsule importance region")
    up = physical_up_in_gaussian_frame(args.gs_to_fe_transform)
    basis = display_basis(up)

    def selected(path: Path | None, fallback: Path) -> Path:
        return (path if path is not None else fallback).expanduser().resolve()

    initial_path = args.initial.expanduser().resolve()
    roi_path = args.roi_trained.expanduser().resolve()
    if not initial_path.is_file() or not roi_path.is_file():
        raise FileNotFoundError("both --initial and --roi-trained must point to real checkpoints")
    raw_source = load(initial_path)
    initial_count = int(raw_source["means"].shape[0])
    roi_state = load(roi_path)
    roi_count = int(roi_state["means"].shape[0])
    roi_means = roi_state["means"].detach().cpu().float()
    records: list[dict[str, object]] = []
    roi_stats = save_roi_refinement(out / "01_step_IIa_roi_refinement_pruning.png",
                                    raw_source, roi_state, basis, args.max_points, roi_cache_path)
    records.append({"file": "00_step_I_initial_checkpoint.png", "stage": "I",
                    "source": str(initial_path),
                    "interpretation": "existing tower 3DGS checkpoint at step 399999; upstream image/SfM/initial training are not rerun"})
    records.append({"file": "01_step_IIa_roi_refinement_pruning.png", "stage": "IIa",
                    "source": [str(initial_path), str(roi_path)],
                    "interpretation": "authentic 399999-to-449999 ROI-guided training output; exact capsule ROI mask shown, no synthetic opacity-only clone/prune replay",
                    "roi_counts": roi_stats})
    del raw_source, roi_state
    terrain_path = selected(args.terrain, chain / "02b_adaptive" / "02a_roi_pruned_terrain_adaptive.pt")
    objects_path = selected(args.objects, chain / "02b_adaptive" / "02a_roi_pruned_objects_adaptive.pt")
    kriging_path = selected(args.kriging, chain / "02c_kriging.pt")
    expanded_path = selected(args.expanded, chain / "02d_expanded.pt")
    static_path = selected(args.static, chain / "03_runtime" / "static.pt")
    moving_path = selected(args.moving, chain / "03_runtime" / "moving.pt")
    vibration_path = selected(args.vibration, chain / "03_runtime" / "moving_vibration.dat")
    terrain = load(terrain_path)
    objects = load(objects_path)
    kriging = load(kriging_path)
    expanded = load(expanded_path)
    static = load(static_path)
    moving = load(moving_path)

    expansion_parent_index = expanded_path.with_suffix(".parent_index.npy")
    if not expansion_parent_index.is_file():
        raise FileNotFoundError(f"Step-IIb parent-index map is missing: {expansion_parent_index}")
    expansion_counts = save_topdown_expansion(
        out / "04b_step_IIb_terrain_expansion_before_after_topdown.png",
        roi_means, expanded, basis, args.max_points, 41, expansion_parent_index,
    )
    records.append({
        "file": "04b_step_IIb_terrain_expansion_before_after_topdown.png",
        "stage": "IIb",
        "source": [str(roi_path), str(expanded_path), str(expansion_parent_index)],
        "counts": expansion_counts,
        "interpretation": "Existing scene already includes terrain; added centers are shown separately in red under a shared plan-view camera and axis range.",
    })

    expanded_xyz = xyz(expanded, args.max_points, 29)
    # The sampled expanded cloud supplies the plotting extent for the
    # expansion panel.  It is deliberately not used as a renderer input.
    all_points = expanded_xyz
    terrain_xyz = xyz(terrain, args.max_points, 17)
    object_xyz = xyz(objects, args.max_points, 18)
    save_pair(out / "02_step_IIb_terrain_object.png", terrain_xyz, object_xyz,
              np.concatenate([terrain_xyz, object_xyz], axis=0),
              ("Step IIb / adaptive terrain Gaussians", "Step IIb / non-terrain Gaussians (includes protected tower ROI)"),
              ("#7a5732", "#2873a6"), "Adaptive terrain/object separation with explicit structure protection", basis, args.max_points, 17,
              shared_bounds=False)
    records.append({"file": "02_step_IIb_terrain_object.png", "stage": "IIb",
                    "source": [str(terrain_path), str(objects_path)]})

    kriging_xyz = xyz(kriging, args.max_points, 23)
    save_single(out / "03_step_IIb_kriging_terrain.png", kriging_xyz, kriging_xyz,
                "Step IIb / Ordinary Kriging terrain reconstruction", basis, args.max_points, 23, "terrain",
                map_view=True)
    save_single(out / "04_step_IIb_multilayer_expanded.png", expanded_xyz, expanded_xyz,
                "Step IIb / constrained multi-layer Gaussian expansion", basis, args.max_points, 29)
    records.extend([
        {"file": "03_step_IIb_kriging_terrain.png", "stage": "IIb", "source": str(kriging_path)},
        {"file": "04_step_IIb_multilayer_expanded.png", "stage": "IIb", "source": str(expanded_path)},
    ])

    vibration_manifest = vibration_path.parent / "runtime_manifest.json"
    frame_count = None
    if vibration_manifest.exists():
        frame_count = int(json.loads(vibration_manifest.read_text(encoding="utf-8"))["vibration_shape"][0])
    if frame_count is None:
        bytes_per_frame = len(moving["means"]) * 3 * np.dtype(np.float32).itemsize
        if vibration_path.stat().st_size % bytes_per_frame:
            raise ValueError(f"cannot infer frame count from displacement stream: {vibration_path}")
        frame_count = vibration_path.stat().st_size // bytes_per_frame
    vib = np.memmap(vibration_path, mode="r", dtype=np.float32,
                    shape=(frame_count, len(moving["means"]), 3))
    frame = min(max(args.disp_frame, 0), vib.shape[0] - 1)
    moving_before = xyz(moving, args.max_points, 37)
    # The plot uses the same deterministic sampled moving Gaussians as the
    # before/after panel; the full vibration array remains the renderer input.
    moving_count = len(moving["means"])
    sample_generator = torch.Generator(device="cpu").manual_seed(37)
    sample_indices = torch.randperm(moving_count, generator=sample_generator)[:min(moving_count, args.max_points)].tolist()
    moving_after = moving_before + np.asarray(vib[frame][sample_indices], dtype=np.float32) * float(args.disp_scale)
    save_pair(out / "05_step_III_registered_FE_displacement.png", moving_before, moving_after,
              np.concatenate([moving_before, moving_after], axis=0),
              ("Step III / processed Gaussian reference\n(archived FE field mapped to these centers)",
               f"Step III / frame {frame} mapped displacement × {args.disp_scale:g}"),
              ("#3a6ea5", "#d24e3e"), "Archived FE displacement mapped to processed Gaussian centers", basis, args.max_points, 37,
              paired_samples=True)
    records.append({"file": "05_step_III_registered_FE_displacement.png", "stage": "III",
                    "source": str(vibration_path),
                    "frame": frame, "display_scale": args.disp_scale})

    # Reassemble the complete model from the same runtime split used by the
    # dynamic renderer.  This is the key continuity check for before/after.
    # This panel is intentionally a tower-moving subset view.  The complete
    # static scene is still the input to the renderer; showing its 1.5M
    # terrain Gaussians here would hide the deformation signal.
    save_pair(out / "06_step_IV_deformation_before_after.png", moving_before, moving_after,
              np.concatenate([moving_before, moving_after], axis=0),
              ("Step IV / processed moving Gaussian subset before deformation", "Step IV / same subset after FE deformation"),
              ("#3a6ea5", "#d24e3e"), "Dynamic deformation on the processed Gaussian model", basis,
              args.max_points, 37, paired_samples=True)
    records.append({"file": "06_step_IV_deformation_before_after.png", "stage": "IV",
                    "source": [str(expanded_path), str(vibration_path)],
                    "frame": frame, "display_scale": args.disp_scale})

    if args.dynamic_render_dir is not None:
        render_dir = args.dynamic_render_dir.expanduser().resolve()
        render_metrics = render_dir / "dynamic_temporal_stability.json"
        if not render_metrics.is_file():
            raise FileNotFoundError(f"gsplat render metrics are missing: {render_metrics}")
        rendered_ids = json.loads(render_metrics.read_text(encoding="utf-8")).get("frame_indices", [])
        for frame_id in sorted(set(int(value) for value in rendered_ids)):
            source_frame = render_dir / "frames" / f"frame_{frame_id:05d}.png"
            if not source_frame.is_file():
                raise FileNotFoundError(f"requested gsplat frame is missing: {source_frame}")
            target_frame = out / f"07_step_IV_gsplat_dynamic_frame_{frame_id:05d}.png"
            target_frame.write_bytes(source_frame.read_bytes())
            records.append({"file": target_frame.name, "stage": "IV", "source": str(source_frame),
                            "renderer_input": [str(static_path), str(moving_path), str(vibration_path)],
                            "frame": frame_id,
                            "rendered_model": "static + moving Gaussian subsets from the same Step IIb expanded checkpoint"})
            moving_frame = render_dir / "moving_frames" / f"frame_{frame_id:05d}.png"
            if moving_frame.is_file():
                target_moving = out / f"08_step_IV_gsplat_moving_subset_frame_{frame_id:05d}.png"
                target_moving.write_bytes(moving_frame.read_bytes())
                records.append({"file": target_moving.name, "stage": "IV", "source": str(moving_frame),
                                "frame": frame_id, "renderer_input": [str(moving_path), str(vibration_path)],
                                "rendered_model": "moving subset from the same expanded model, static scene hidden, same camera as full scene"})

    (out / "stage_visual_manifest.json").write_text(json.dumps({
        "format": "real-tower-continuous-roi-chain-figures-v1",
        "coordinate_convention": "diagnostic vertical is physical FE +Z mapped back into Gaussian coordinates; model points and displacement values are not altered",
        "display_transform": {"gs_to_fe_transform": str(args.gs_to_fe_transform.resolve()),
                              "physical_up_in_gaussian_frame": up.tolist(),
                              "basis_columns": basis.tolist()},
        "projection": "orthographic front/side profiles; physical elevation is the vertical plot axis and increases upward",
        "step_ia_counts": {"initial": initial_count, "roi_trained": roi_count},
        "roi_cache": str(roi_cache_path) if roi_cache_path else None,
        "chain_dir": str(chain), "figures": records,
        "limitations": [
            "Step I training/COLMAP is represented by the supplied real checkpoint; it is not retrained here.",
            "Step IIa is represented by the real 399999-to-449999 FE-ROI training branch; the training run is not repeated here, and these center plots do not track Gaussian IDs across training.",
            "Step III uses the archived registered FE stream; ANSYS/FE solving was not rerun.",
            "All exact-capsule target ROI Gaussians receive nearest archived-sample translations; source rows may be reused. This is an auditable translation replay, not Eq.20 interpolation.",
            "Step IV uses a translation-only FE displacement renderer; the full paper right-polar rotation, scale, quaternion and covariance updates are not reproduced here.",
        ],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "figure_count": len(records), "frame": frame}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
