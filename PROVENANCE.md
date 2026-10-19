# Source provenance

This release candidate was assembled from the authors' research working tree and selected supporting sources. It is not a pristine snapshot of a single upstream commit, and the local source HEAD alone does not reproduce its complete contents.

## gsplat ancestry

The original working tree's remote was https://github.com/nerfstudio-project/gsplat.git. Its local HEAD was `4eee57eddb33d62343c7e5128c5b93a2867ef16d`. The common ancestor of that HEAD and the locally recorded `origin/main` was `795161945b37747709d4da965b226a19fdf87d3f` (`use mask in 2dgs (#497)`). Two additional local commits concerned ignore rules (`724d3fa` and `4eee57e`). The locally recorded remote branch was not refreshed as part of provenance collection, so this is an ancestry observation, not a claim about the current upstream tip or an upstream published release.

The original checkout had tracked working-tree modifications to `README.md`, `examples/benchmarks/basic.sh`, `examples/requirements.txt`, `gsplat/cuda/_backend.py`, `gsplat/cuda/csrc/ext.cpp`, `gsplat/strategy/__init__.py`, and `setup.py`, plus untracked research additions. Only selected relevant sources were copied into this candidate. Its source distribution does not include the original checkout's complete history, datasets, checkpoints, experiment outputs, or every local modification.

## Research and release changes

The candidate includes the authors' ROI-aware strategies and training/resume entry points, FE-to-Gaussian deformation mapping, anisotropic interpolation, dynamic visualization, terrain-extension code, and diagnostics as identified in `README.md`. Supporting gsplat/training sources retain their third-party licensing.

Release preparation removed author-machine input defaults and the embedded project-specific registration matrix, moved reusable CLI parsing and execution effects into main entry points, supplied development-only synthetic tests, and added runtime/setup documentation. The connected paper-tower validation was run separately against the private source worktree and is recorded under `outputs/real_tower_pipeline_20261013/`; those private checkpoints and FE files are not redistributed. These changes belong to this candidate; checking out the original local HEAD will not supply them or all the untracked research code.

The bundled `GSPLAT_3DGS_ONLY=1` option in `gsplat/cuda/_backend.py` and `gsplat/cuda/csrc/ext.cpp` is a local compilation compatibility option. It selects a separate 3DGS extension name and omits 2DGS compilation/bindings. It does not establish full 2DGS compatibility or identical behavior to a current upstream gsplat release. Keep it disabled when validating the full upstream feature set.

## Source-built dependencies

The locally installed rmbrualla pycolmap checkout had HEAD `cc7ea4b7301720ac29287dbe450952511b32125e`, with no tracked modifications (an untracked build directory was present). The locally installed rahul-goel fused-ssim checkout had HEAD `d99e3d27513fa3563d98f74fcd40fd429e9e9b0e` and a clean status. Their remotes and installation instructions are recorded in `ENVIRONMENT.md`. A local checkout's commit identification is not verification that a fresh online install works; clean-environment installation remains a separate check.

See `THIRD_PARTY_NOTICES.md` for the bundled third-party source licenses and `RELEASE_STATUS.md` for validation status. Private project measurements are intentionally absent; synthetic test success is not reproduction of paper results.
