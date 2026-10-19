#!/usr/bin/env python3
"""Run the repository renderer while reusing the installed matching gsplat build.

The source checkout and the runtime checkpoints remain unchanged.  Importing the
installed package first is intentional: it contains ``gsplat.csrc`` compiled for
the active PyTorch/CUDA environment, whereas the stale JIT cache does not expose
the packed-projection symbols used by the current Python wrapper.
"""

from __future__ import annotations

import runpy
import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / "revision_experiments" / "dynamic_temporal_stability.py"


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--installed-gsplat-root",
        type=Path,
        default=os.environ.get("GSPLAT_INSTALLED_ROOT"),
        help="optional installed gsplat package root containing its compiled CUDA extension",
    )
    args, remaining = parser.parse_known_args()

    if args.installed_gsplat_root is not None:
        installed_root = args.installed_gsplat_root.expanduser().resolve()
        if not installed_root.is_dir():
            raise FileNotFoundError(f"installed gsplat root not found: {installed_root}")
        # Load the matching binary-backed package before the repository script
        # can place its source checkout at sys.path[0].
        sys.path.insert(0, str(installed_root))
        import gsplat  # noqa: F401,E402

    sys.path.insert(1, str(REPO_ROOT))
    sys.argv = [str(TARGET), *remaining]
    runpy.run_path(str(TARGET), run_name="__main__")


if __name__ == "__main__":
    main()
