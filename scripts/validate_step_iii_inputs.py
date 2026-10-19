#!/usr/bin/env python3
"""Validate external Step III FE inputs before Gaussian deformation.

The listed author notebooks import ANSYS nodes/elements/sections and consume
the solved three-component nodal displacement history.  They do not contain a
wind-field generator or an ANSYS nonlinear solver.  This utility therefore
checks the handoff contract and writes a machine-readable audit manifest; it
does not claim to replace the external wind/FE solve.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import struct

import numpy as np


def _count_rows(path: Path, minimum_columns: int) -> int:
    count = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields = line.split()
        if len(fields) >= minimum_columns:
            try:
                [float(value) for value in fields[:minimum_columns]]
            except ValueError:
                continue
            count += 1
    return count


def _count_sections(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as stream:
        for raw in stream:
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            row = next(csv.reader([raw])) if "," in raw else raw.split()
            if len(row) < 2:
                continue
            try:
                float(row[0])
            except ValueError:
                continue
            count += 1
    return count


def _load_displacement(path: Path, frames: int | None, points: int | None) -> tuple[tuple[int, ...], str]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, mmap_mode="r")
        shape = tuple(int(value) for value in array.shape)
        if len(shape) != 3 or shape[-1] != 3:
            raise ValueError(f"expected displacement shape (frames, points, 3), got {shape}")
        if not np.isfinite(np.asarray(array[: min(shape[0], 1)])).all():
            raise ValueError("displacement array contains non-finite values")
        return shape, "npy"
    if suffix == ".dat":
        if frames is None or points is None:
            raise ValueError("raw float32 .dat requires --frames and --points")
        expected = int(frames) * int(points) * 3
        actual = path.stat().st_size // struct.calcsize("f")
        if path.stat().st_size % struct.calcsize("f") or actual != expected:
            raise ValueError(f"raw .dat has {actual} float32 values; expected {expected}")
        values = np.memmap(path, dtype=np.float32, mode="r", shape=(frames, points, 3))
        if not np.isfinite(np.asarray(values[:1])).all():
            raise ValueError("displacement stream contains non-finite values")
        return (int(frames), int(points), 3), "raw-float32-dat"
    array = np.loadtxt(path, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.shape[-1] != 3:
        raise ValueError(f"text displacement must have three columns, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("displacement text contains non-finite values")
    return (int(array.shape[0]), 1, 3), "text-node-displacements"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-file", type=Path, required=True)
    parser.add_argument("--elem-file", type=Path, required=True)
    parser.add_argument("--section-file", type=Path, required=True)
    parser.add_argument("--displacement-file", type=Path, required=True)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--points", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("node_file", "elem_file", "section_file", "displacement_file"):
        path = getattr(args, name).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
        setattr(args, name, path)
    node_count = _count_rows(args.node_file, 4)
    element_count = _count_rows(args.elem_file, 5)
    section_count = _count_sections(args.section_file)
    displacement_shape, displacement_format = _load_displacement(
        args.displacement_file, args.frames, args.points
    )
    report = {
        "paper_step": "III-input-handoff",
        "nodes": node_count,
        "elements": element_count,
        "sections": section_count,
        "displacement_shape": displacement_shape,
        "displacement_format": displacement_format,
        "source_notebooks": [
            "1_3all_zhixian_ansys copy.ipynb",
            "1_3all_Zhixianta_weiyidaoru_feng.ipynb",
            "1_3_all_Zhixinta_wushijian_jianhua_copy.ipynb",
        ],
        "status": "input_contract_validated",
        "limitation": "Wind synthesis and ANSYS/FE nonlinear solve are external upstream stages; this utility does not perform them.",
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
