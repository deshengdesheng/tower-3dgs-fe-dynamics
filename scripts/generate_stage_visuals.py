#!/usr/bin/env python3
"""Generate auditable stage images for the real Yunnan straight-tower case.

The figures are point-cloud diagnostics, not a replacement for the gsplat
renderer.  They are generated from the real checkpoints and the registered
FE displacement stream used by the local candidate release.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from workflow.coordinate_frames import display_basis, physical_up_in_gaussian_frame


def load(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "splats" in state:
        state = state["splats"]
    return {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in state.items()}


def points(state: dict[str, torch.Tensor], max_points: int, seed: int = 7):
    xyz = state["means"].float().numpy()
    n = len(xyz)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(n, max_points), replace=False)
    return xyz[idx], idx


def display_coordinates(xyz: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project model coordinates into a frame whose vertical follows FE +Z."""
    return np.asarray(xyz, dtype=np.float32) @ basis


def style(ax, title: str, xlabel="Horizontal X'", ylabel="Horizontal Y'", zlabel="Physical elevation H (up)"):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_zlabel(zlabel)
    ax.view_init(elev=18, azim=-58)
    ax.grid(True, alpha=.22)


def save_scatter(path: Path, xyz: np.ndarray, basis: np.ndarray, title: str, color=None, cmap="viridis", size=1.0):
    fig = plt.figure(figsize=(8.5, 7.0), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    if color is None: color = xyz @ basis[:, 2]
    plotted = display_coordinates(xyz, basis)
    sc = ax.scatter(plotted[:, 0], plotted[:, 1], plotted[:, 2], c=color, s=size, cmap=cmap, alpha=.72, linewidths=0)
    style(ax, title)
    fig.colorbar(sc, ax=ax, shrink=.62, pad=.10, label="Physical elevation H / diagnostic value")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def save_compare(path: Path, before: np.ndarray, after: np.ndarray, basis: np.ndarray, title: str):
    fig = plt.figure(figsize=(12, 6.2), dpi=180)
    for i, (xyz, label, color) in enumerate(((before, "Before deformation", "#3b6fb6"), (after, "After FE displacement (50x display)", "#d24a3a")), 1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        plotted = display_coordinates(xyz, basis)
        ax.scatter(plotted[:, 0], plotted[:, 1], plotted[:, 2], s=1.0, c=color, alpha=.65, linewidths=0)
        style(ax, label)
    fig.suptitle(title, fontsize=12); fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def save_disp(path: Path, base: np.ndarray, disp: np.ndarray, basis: np.ndarray, title: str):
    mag = np.linalg.norm(disp, axis=1)
    fig = plt.figure(figsize=(12, 6.2), dpi=180)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    plotted = display_coordinates(base, basis)
    sc = ax.scatter(plotted[:, 0], plotted[:, 1], plotted[:, 2], c=mag, s=1.4, cmap="magma", alpha=.8, linewidths=0)
    style(ax, "Registered FE displacement magnitude")
    fig.colorbar(sc, ax=ax, shrink=.62, pad=.08, label="|u| (physical units)")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(mag, bins=50, color="#b64b4b", alpha=.85)
    ax2.set_title("Moving-Gaussian displacement distribution")
    ax2.set_xlabel("|u|"); ax2.set_ylabel("Gaussian count"); ax2.grid(True, alpha=.22)
    fig.suptitle(title, fontsize=12); fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True,
                   help="authorized local source tree containing the paper checkpoints and FE artifacts")
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gs-to-fe-transform", type=Path, required=True)
    p.add_argument("--max-points", type=int, default=50000)
    args = p.parse_args()
    src = args.source_root; out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    basis = display_basis(physical_up_in_gaussian_frame(args.gs_to_fe_transform))
    step2 = args.repo_root / "outputs" / "step_ii_real_tower_notebook_aligned_20261017"
    demo = args.repo_root / "outputs" / "real_tower_demo_final_20261017"
    before_path = demo / "artifacts" / "gs_before.pt"
    after_path = demo / "artifacts" / "gs_after_frame_00000_render_scaled50x.pt"
    tu1_path = src / "results" / "filter_schemes" / "percentage" / "tu1_percentage_filtered.pt"
    roi_refined_path = src / "results" / "yunnan_zhixianta_dense3_baowei_bboxloss_v1" / "ckpts" / "ckpt_449999_rank0.pt"
    terrain_path = step2 / "adaptive" / "tu1_percentage_filtered_terrain_adaptive.pt"
    objects_path = step2 / "adaptive" / "tu1_percentage_filtered_objects_adaptive.pt"
    kriging_path = step2 / "tu1_percentage_filtered_terrain_adaptive_kriging.pt"
    expanded_path = step2 / "tu1_percentage_filtered_expanded.pt"
    moving_path = demo / "artifacts" / "dynamic_gs" / "moving.pt"
    vibration_path = demo / "artifacts" / "dynamic_gs" / "moving_vibration.dat"

    records = []
    def rec(name, stage, path, source, regenerated=True, note=""):
        records.append({"file": name, "paper_stage": stage, "path": str(path), "source": source,
                        "regenerated_in_this_run": regenerated, "note": note})

    # I: existing real tower GS checkpoint (the upstream COLMAP/gsplat result).
    tu1 = load(tu1_path); xyz, _ = points(tu1, args.max_points)
    save_scatter(out / "01_initial_real_tower_gs.png", xyz, basis, "Step I — Initial real-tower 3DGS checkpoint", size=1.0)
    rec("01_initial_real_tower_gs.png", "I", tu1_path, "1_1chuli_yunan_zhixianta.ipynb",
        note="Visualization of the existing real-tower checkpoint; photo/COLMAP and long training are upstream inputs.")

    # IIa: verified importance-region refinement/pruning checkpoint. This is
    # shown as its own paper-stage result; the current IIb and IV artifacts
    # below still descend from tu1/lingshi2, so we do not imply false lineage.
    roi_checkpoint = torch.load(roi_refined_path, map_location="cpu", weights_only=False)
    roi_state = roi_checkpoint["splats"] if isinstance(roi_checkpoint, dict) and "splats" in roi_checkpoint else roi_checkpoint
    roi_xyz, _ = points(roi_state, args.max_points)
    roi_step = roi_checkpoint.get("step", "unknown") if isinstance(roi_checkpoint, dict) else "unknown"
    save_scatter(
        out / "01b_importance_region_refinement.png",
        roi_xyz, basis,
        f"Step IIa — ROI Gaussian refinement/pruning (step {roi_step}; N={len(roi_state['means']):,})",
        color=roi_xyz @ basis[:, 2], cmap="viridis", size=1.0,
    )
    rec(
        "01b_importance_region_refinement.png", "IIa", roi_refined_path,
        "1_1chuli_yunan_zhixianta.ipynb; checkpoint cfg.yml",
        note=("Verified training output at step 449999 with 1,687,649 Gaussians. "
              "Its checkpoint has no per-Gaussian parent IDs. Current Step IIb terrain artifacts and Step IV dynamic artifacts "
              "still descend from the separate 272,300-point tu1/lingshi2 model; this image is not evidence that those stages "
              "use this ROI-refined checkpoint."),
    )

    # II: terrain separation, Kriging and multi-layer expansion.
    terrain = load(terrain_path); objects = load(objects_path); kriging = load(kriging_path); expanded = load(expanded_path)
    txyz, _ = points(terrain, args.max_points); oxyz, _ = points(objects, args.max_points)
    kxyz, _ = points(kriging, args.max_points); exyz, _ = points(expanded, args.max_points)
    save_scatter(out / "02_terrain_separation.png", txyz, basis, "Step II — Adaptive terrain extraction", color=txyz @ basis[:, 2], cmap="terrain", size=1.1)
    fig = plt.figure(figsize=(12, 6.2), dpi=180)
    for i, (q, title, color) in enumerate(((txyz, "Terrain", "#7b5b38"), (oxyz, "Non-terrain / tower objects", "#2c73a8")), 1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        plotted = display_coordinates(q, basis)
        ax.scatter(plotted[:, 0], plotted[:, 1], plotted[:, 2], s=1, c=color, alpha=.7, linewidths=0)
        style(ax, title)
    fig.suptitle("Step II — Terrain/object separation", fontsize=12); fig.tight_layout(); fig.savefig(out / "03_terrain_object_separation.png", bbox_inches="tight"); plt.close(fig)
    save_scatter(out / "04_kriging_terrain.png", kxyz, basis, "Step II — Kriging terrain reconstruction", color=kxyz @ basis[:, 2], cmap="terrain", size=1.1)
    save_scatter(out / "05_multilayer_expansion.png", exyz, basis, "Step II — Feature-constrained multi-layer expansion", color=exyz @ basis[:, 2], cmap="viridis", size=1.0)
    rec("02_terrain_separation.png", "II", terrain_path, "1_3alll_Zhixianta_dixingtuozhan.ipynb")
    rec("03_terrain_object_separation.png", "II", f"{terrain_path}; {objects_path}", "1_3alll_Zhixianta_dixingtuozhan.ipynb")
    rec("04_kriging_terrain.png", "II", kriging_path, "1_3alll_Zhixianta_dixingtuozhan.ipynb")
    rec("05_multilayer_expansion.png", "II", expanded_path, "1_3alll_Zhixianta_dixingtuozhan.ipynb")

    # III: FE registration/displacement input (not an ANSYS solve in this release).
    moving = load(moving_path); mxyz, _ = points(moving, args.max_points)
    vib = np.memmap(vibration_path, mode="r", dtype=np.float32, shape=(399, len(moving["means"]), 3))[0]
    rng = np.random.default_rng(11); sel = rng.choice(len(moving["means"]), size=min(len(moving["means"]), args.max_points), replace=False)
    save_disp(out / "06_fe_registered_displacement.png", moving["means"].numpy()[sel], vib[sel], basis, "Step III — Registered FE-to-Gaussian displacement input")
    rec("06_fe_registered_displacement.png", "III", vibration_path, "1_3all_zhixian_ansys copy.ipynb; 1_3_all_Zhixinta_wushijian_jianhua_copy.ipynb",
        note="Uses the precomputed registered FE displacement stream; this candidate does not rerun ANSYS/nonlinear FE solving.")

    # IV: dynamic deformation and rendering.
    bxyz, _ = points(load(before_path), args.max_points, seed=11); axyz, _ = points(load(after_path), args.max_points, seed=11)
    save_compare(out / "07_deformation_before_after.png", bxyz, axyz, basis, "Step IV — Real-tower GS before/after FE deformation")
    rec("07_deformation_before_after.png", "IV", f"{before_path}; {after_path}", "1_3all_Zhixianta_weiyidaoru_feng.ipynb")
    render_dir = demo / "dynamic_render" / "frames"
    for frame in sorted(render_dir.glob("frame_*.png")):
        target = out / f"08_{frame.name}"
        target.write_bytes(frame.read_bytes())
        rec(target.name, "IV", frame, "1_3all_Zhixianta_weiyidaoru_feng.ipynb; gsplat dynamic renderer", regenerated=False,
            note="Copied from the verified real-tower dynamic render; rerunnable with scripts/run_dynamic_gs.py.")

    manifest = {"case": "real Yunnan straight tower", "generated_at": "2026-10-17", "figures": records,
                "scope": "real checkpoint and registered FE input; no synthetic point cloud",
                "limitations": ["Step I photo/COLMAP/long training is upstream and not rerun here.",
                                "Step III ANSYS/FE solve is not rerun; the registered displacement stream is visualized.",
                                "The dynamic renderer updates moving means; full covariance/right-polar update chain is not claimed."]}
    (out / "stage_visual_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "figure_count": len(records), "manifest": str(out / "stage_visual_manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
