# Environment and installation status

## Observed environment

The authors' available runtime is Ubuntu under WSL, Python 3.10.15, PyTorch 2.0.1+cu118, torchvision 0.15.2+cu118, and a CUDA 11.8 PyTorch build. CUDA device discovery succeeded on an NVIDIA GeForce RTX 4070 Ti SUPER (16,376 MiB) with driver 591.86. Driver compatibility does not establish which local CUDA compiler is selected: check `nvcc --version` before compiling extensions.

The environment contains gsplat 1.4.0 and Nerfstudio 1.1.5. The installed gsplat was an editable checkout of the authors' working source, so successful imports in that environment alone do not establish that an independent checkout is complete. Use the bundled gsplat source in this distribution; do not overwrite it with an unrelated PyPI release. The upstream base and local changes are described in `THIRD_PARTY_NOTICES.md`.

`requirements-tested.txt` records selected dependency versions observed from installed package metadata. This is not a full environment lockfile or a claim that every dependency combination has been verified. The original environment contains unrelated packages and some duplicate distribution metadata, so its entire package listing has intentionally not been copied.

## Source-built dependencies

The training parser imports `SceneManager` from the rmbrualla pycolmap fork, not the modern PyPI COLMAP bindings. Installed metadata reported pycolmap 0.0.1 from a local source checkout whose remote is https://github.com/rmbrualla/pycolmap.git and whose HEAD is `cc7ea4b7301720ac29287dbe450952511b32125e`. No tracked changes were found; only an untracked `build/` directory was present.

Installed fused-ssim metadata reported version 0.0.0 from a local source checkout whose remote is https://github.com/rahul-goel/fused-ssim.git and whose HEAD is `d99e3d27513fa3563d98f74fcd40fd429e9e9b0e`. That checkout was clean. Its setup uses PyTorch `CUDAExtension`, so it must be built with PyTorch already installed and a compatible CUDA compiler. The source commits were verified locally; availability of those exact commits from a fresh remote clone has not been checked here.

## Proposed clean installation

The following Linux/WSL recipe is provided for validation; it has not yet been completed in a new environment. It assumes Python 3.10, Git, a C++ build toolchain, and a CUDA toolkit compatible with the selected PyTorch build. Run it from the repository root, outside the authors' original working tree:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install pip==24.2 setuptools==75.1.0 wheel==0.44.0
python -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-tested.txt
python -m pip install --no-build-isolation 'fused-ssim @ git+https://github.com/rahul-goel/fused-ssim.git@d99e3d27513fa3563d98f74fcd40fd429e9e9b0e'
```

The bundled gsplat can be resolved from the repository root and by the provided scripts' repository-path setup. A standalone wheel/editable installation requires an appropriate package build configuration; do not assume `pip install -e .` works unless that configuration is present and tested. The gsplat CUDA backend may compile on first use. Set `GSPLAT_3DGS_ONLY=1` only when using the bundled compatibility option and the 3DGS path; the 2DGS path is then intentionally unavailable.

After setup, verify the actual import origin and run the synthetic tests and smoke runner documented in `README.md`. A successful smoke test establishes the tested synthetic paths, not reproduction of the paper's experiments. Full training requires separately supplied, authorized image/COLMAP, FE, displacement, and registration inputs. No private data paths or data contents are required by this installation recipe.

The viewer additionally depends on a working CUDA rasterizer for practical performance. Optional KeOps acceleration uses pykeops 2.3 in the observed environment; the ordinary implementation does not require it. Nerfstudio is not listed as a mandatory dependency merely because it was installed in the original environment.
