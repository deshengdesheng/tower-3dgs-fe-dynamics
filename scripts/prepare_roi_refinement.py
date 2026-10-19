#!/usr/bin/env python3
"""Run an optional opacity-ranked split/prune diagnostic.

This helper does not read the FE-beam ROI cache and does not implement the
paper's training-time Step IIa strategy. Use the existing ROI-trained
checkpoint as the Step IIa output in a paper-chain reproduction.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np
import torch


def load(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    return state["splats"] if isinstance(state, dict) and "splats" in state else state


def _local_axis(quats: torch.Tensor) -> torch.Tensor:
    """Return the first local Gaussian axis for wxyz or xyzw quaternions.

    The exact convention only changes the direction of the small split offset;
    both conventions are normalized and produce a finite, deterministic axis.
    """
    q = torch.nn.functional.normalize(quats.float(), dim=1)
    w, x, y, z = q.unbind(dim=1)
    axis = torch.stack(
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
        dim=1,
    )
    return torch.nn.functional.normalize(axis, dim=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--densify-fraction", type=float, default=0.05,
        help="fraction of high-opacity Gaussians to split once (diagnostic only; not paper Step IIa)",
    )
    parser.add_argument("--prune-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not 0.0 <= args.prune_fraction < 1.0:
        raise ValueError("prune fraction must be in [0, 1)")
    if not 0.0 <= args.densify_fraction < 1.0:
        raise ValueError("densify fraction must be in [0, 1)")
    splats = load(args.input)
    n = int(splats["means"].shape[0])
    for key in ("opacities", "quats", "scales"):
        if key not in splats:
            raise ValueError(f"the checkpoint has no {key} field")
    opacity = splats["opacities"].detach().float().reshape(n)
    contribution = torch.sigmoid(opacity)
    densify_count = int(round(n * args.densify_fraction))
    importance_order = torch.argsort(contribution, descending=True, stable=True)
    parent_indices = importance_order[:densify_count]
    # Clone/split high-importance points along their local first axis.  The
    # offset is bounded by the stored Gaussian scale, which is log encoded.
    parent_means = splats["means"].detach().float().index_select(0, parent_indices)
    parent_scales = splats["scales"].detach().float().index_select(0, parent_indices)
    axis = _local_axis(splats["quats"].detach().float().index_select(0, parent_indices))
    radius = parent_scales[:, 0].exp().clamp(min=1e-5, max=0.25) * 0.35
    child_means = parent_means + axis * radius[:, None]
    child_payload: dict[str, torch.Tensor] = {}
    for key, value in splats.items():
        if not torch.is_tensor(value) or value.ndim == 0 or value.shape[0] != n:
            continue
        child_payload[key] = value.index_select(0, parent_indices).clone()
    child_payload["means"] = child_means.to(splats["means"].dtype)
    # Split a Gaussian's opacity mass between parent and child in logit space.
    child_payload["opacities"] = child_payload["opacities"] - float(np.log(2.0))
    child_payload["scales"] = child_payload["scales"] - float(np.log(2.0))
    expanded: dict[str, torch.Tensor] = {}
    for key, value in splats.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == n:
            expanded[key] = torch.cat((value, child_payload[key]), dim=0)
        else:
            expanded[key] = value
    expanded_n = n + densify_count
    expanded_contribution = torch.sigmoid(expanded["opacities"].detach().float().reshape(expanded_n))

    # Prune only low-importance original points.  Children and their parents
    # are protected so the densification remains observable in the result.
    prune_count = int(round(n * args.prune_fraction))
    protected = torch.zeros(expanded_n, dtype=torch.bool)
    protected[parent_indices] = True
    protected[n:] = True
    candidate = torch.nonzero(~protected, as_tuple=False).reshape(-1)
    candidate_order = candidate[torch.argsort(expanded_contribution[candidate], stable=True)]
    prune_indices = candidate_order[:min(prune_count, len(candidate_order))]
    keep = torch.ones(expanded_n, dtype=torch.bool)
    keep[prune_indices] = False
    indices = torch.nonzero(keep, as_tuple=False).reshape(-1)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    filtered: dict[str, torch.Tensor] = {}
    for key, value in expanded.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == expanded_n:
            # ``indices`` addresses the expanded tensor (parents + children),
            # so indexing the original ``splats`` here would either fail or
            # silently discard the newly densified children.
            filtered[key] = value.index_select(0, indices)
        else:
            filtered[key] = value
    # PyTorch's Windows zip writer can fail on non-ASCII parent directories on
    # some builds.  Save through an ASCII temp path, then use Python's Unicode-
    # aware file copy for the requested workspace destination.
    temporary_checkpoint = Path(tempfile.gettempdir()) / f"roi_refined_{uuid.uuid4().hex}.pt"
    torch.save({"splats": filtered}, temporary_checkpoint)
    shutil.copyfile(temporary_checkpoint, output)
    temporary_checkpoint.unlink()
    payload = {
        "stage": "Optional opacity-ranked split/prune diagnostic (not paper Step IIa)",
        "source_checkpoint": str(args.input.resolve()),
        "output_checkpoint": str(output),
        "source_gaussian_count": n,
        "densification_method": "diagnostic clone/split of top-opacity Gaussians; FE-ROI membership is not used",
        "importance_score": "sigmoid(opacities_logit)",
        "densified_parent_count": int(densify_count),
        "count_after_densification": int(expanded_n),
        "retained_gaussian_count": int(indices.numel()),
        "pruned_gaussian_count": int(prune_indices.numel()),
        "densification_gain": int(densify_count),
        "protected_densified_children": int(densify_count),
        "protected_parent_count": int(densify_count),
        "output_parent_index": str(output.with_suffix(".parent_index.npy")),
        "prune_fraction": args.prune_fraction,
        "opacity_contribution": "sigmoid(opacities_logit)",
        "minimum_retained_contribution": float(expanded_contribution[indices].min()),
        "maximum_pruned_contribution": float(expanded_contribution[prune_indices].max()) if prune_indices.numel() else None,
        "note": "This is an opacity-only diagnostic transformation and must not be cited as a reproduction of the paper's FE-ROI-guided training-time densification/pruning.",
    }
    # The author environment may ship a PyTorch build whose NumPy bridge is
    # unavailable even though NumPy itself is importable.  Build the small
    # audit vector from Python integers so the checkpoint export remains
    # runnable in that environment.
    np.save(output.with_suffix(".parent_index.npy"), np.asarray(indices.tolist(), dtype=np.int64))
    manifest = args.manifest.resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
