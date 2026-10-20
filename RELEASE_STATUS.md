# Release Checklist

This checklist records release preparation and validation boundaries. It is not evidence that the code reproduces every paper result.

- [x] Confirm the user has authority to release the included code and selected MIT.
- [x] Add an MIT license for the authors' code.
- [x] Bundle the required `gsplat` source and preserve its Apache-2.0 license.
- [x] Confirm which scripts implement methods/results reported in the manuscript; keep unrelated experiments out (`NOTEBOOK_TRACEABILITY.md`).
- [x] Review every included file for local paths, private coordinates, credentials, and unintended metadata. No credentials or author-machine defaults remain in the launcher; private tower assets are explicit external inputs.
- [x] Record the exact upstream `gsplat` ancestry and local source provenance.
- [x] Record observed Python, PyTorch, CUDA, and package versions.
- [x] Keep command-line parsing and output-directory creation inside script entry points rather than module import.
- [x] Add a small synthetic ROI example that contains no project data.
- [x] Add and run CPU tests for ROI selection, rigid transforms, zero displacement, rotations, and anisotropic weights.
- [x] Run synthetic tests in the observed WSL environment and document the results.
- [x] Verify the direct ROI-checkpoint -> FE mapping -> dynamic-render handoff on synthetic inputs.
- [x] Validate both `.npy` and raw float32 `.dat` vibration inputs.
- [x] Document the external photo-to-SfM step, bundled COLMAP-to-3DGS training path, and upstream project references.
- [x] Package the Step II adaptive-terrain -> Kriging -> multi-layer expansion chain and run its synthetic smoke path.
- [x] Package a Step III ANSYS/FE input-contract validator for node/element/section tables and registered displacement streams.
- [x] Record that the listed notebooks do not contain the wind-field generator or nonlinear ANSYS/FE solver.
- [ ] Verify that the README commands work on a clean checkout without access to author-only paths.
- [x] Create the public repository: <https://github.com/deshengdesheng/tower-3dgs-fe-dynamics>.
- [x] Add the repository URL to the manuscript abstract and R2 response letter.
- [ ] Verify README commands on a clean checkout without access to author-only paths.
