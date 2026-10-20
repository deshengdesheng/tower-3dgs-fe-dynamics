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

The packages below are dependencies or external tools; they are not relicensed by this repository. Consult each upstream project for its current license and citation requirements.

- [COLMAP](https://github.com/colmap/colmap) performs the upstream Structure-from-Motion reconstruction that supplies camera poses and sparse points. The repository's training path consumes those outputs; COLMAP is not bundled here. Follow COLMAP's own license and citation instructions when using that external tool.
- [PyTorch](https://github.com/pytorch/pytorch) provides tensor operations and model-training infrastructure.
- [pycolmap](https://github.com/rmbrualla/pycolmap/tree/cc7ea4b7301720ac29287dbe450952511b32125e) provides Python access to COLMAP reconstruction data; the pinned fork and commit used by the documented environment are recorded in `ENVIRONMENT.md`. Its MIT license is available in that source revision's [`LICENSE.txt`](https://github.com/rmbrualla/pycolmap/blob/cc7ea4b7301720ac29287dbe450952511b32125e/LICENSE.txt).
- [viser 0.2.7](https://github.com/viser-project/viser) and [nerfview 0.1.3](https://github.com/nerfstudio-project/nerfview) provide the interactive camera/training inspection interface used by `examples/simple_trainer.py` and `scripts/check_viewer.py`. Both projects use Apache-2.0; see their upstream [viser license](https://github.com/viser-project/viser/blob/main/LICENSE) and [nerfview license](https://github.com/nerfstudio-project/nerfview/blob/main/LICENSE).
- [Matplotlib 3.10.0](https://github.com/matplotlib/matplotlib) generates static diagnostic and analysis figures. See its upstream [license file](https://github.com/matplotlib/matplotlib/blob/v3.10.0/LICENSE/LICENSE); it is not covered by this repository's MIT license.
- [Plotly.py 6.2.0](https://github.com/plotly/plotly.py) is used by optional interactive terrain/FE inspection scripts and is distributed under MIT; see its upstream [`LICENSE.txt`](https://github.com/plotly/plotly.py/blob/v6.2.0/LICENSE.txt).
- NumPy, scikit-learn, fused-ssim, and other installed packages retain their respective upstream licenses; see `requirements-tested.txt` and `ENVIRONMENT.md` for the documented environment.

These tools provide SfM, computation, viewer, and plotting infrastructure. The final 3D Gaussian image rasterization is performed by gsplat. Their use does not make these libraries part of the authors' method or transfer their ownership to the authors.
