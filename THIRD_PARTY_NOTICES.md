# Third-party source notices

The MIT license in `LICENSE` covers the authors' research code only. It does not relicense the third-party sources below.

## gsplat

This distribution includes the `gsplat` Python package and CUDA sources, and supporting training files from Nerfstudio's gsplat project, https://github.com/nerfstudio-project/gsplat. Its Apache License 2.0 is preserved in `LICENSE-gsplat`. The source working checkout identified itself as commit `4eee57eddb33d62343c7e5128c5b93a2867ef16d`; research scripts and working-tree changes were additional to that commit.

The bundled working-tree changes include custom `RegionFilteredStrategy` and `IrregularRegionFilteredStrategy` exports in `gsplat/strategy/__init__.py`, and the `GSPLAT_3DGS_ONLY` compatibility option in `gsplat/cuda/_backend.py` and `gsplat/cuda/csrc/ext.cpp`. The local candidate also adds opt-in loading of a prebuilt CUDA extension through `GSPLAT_PREBUILT` in `gsplat/cuda/_backend.py`. These changes are described here because they are based on Apache-2.0-licensed gsplat source; this notice does not replace any upstream file-level notices. The package contains source rather than a machine-specific compiled `csrc.so`.

Training support code is included in `examples/simple_trainer.py`, `examples/utils.py`, and `examples/datasets/`. `examples/resume_training.py` and the ROI entry point provide the authors' training extension. Third-party attribution in source files is retained.

## Bilateral Guided Radiance Field Processing

`examples/lib_bilagrid.py` preserves its existing attribution to Yuehao Wang (2024), https://github.com/yuehaowang, and https://bilarfpro.github.io/. It is distributed under Apache License 2.0; see its source header and `LICENSE-gsplat` for the license text.

## MultiNeRF camera utilities

`examples/datasets/traj.py` preserves its source attribution to Google Research MultiNeRF's `internal/camera_utils.py` at commit `5b4d4f64608ec8077222c52fdf814d40acc10bc1`: https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/internal/camera_utils.py. MultiNeRF is distributed under Apache License 2.0. The attribution in the source file and the Apache license text in `LICENSE-gsplat` are retained; this code is not relicensed under the authors' MIT license.

## OpenGL Mathematics (GLM)

CUDA compilation uses the vendored GLM headers in `gsplat/cuda/csrc/third_party/glm/glm/`. The original copyright and dual licensing terms are preserved in `gsplat/cuda/csrc/third_party/glm/copying.txt`. This distribution relies on the MIT licensing option stated there. The vendored tree currently also contains GLM documentation, utility code, workflow metadata, and tests; these are retained with their upstream notices and license terms, although they are not needed by the runtime.

## Runtime dependencies

PyTorch, NumPy, scikit-learn, pycolmap, fused-ssim, viser, nerfview, and other installed dependencies retain their own licenses. Dependency installation does not transfer their ownership or licensing to the authors.
