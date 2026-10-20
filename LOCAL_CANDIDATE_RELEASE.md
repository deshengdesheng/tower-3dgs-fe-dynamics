# Reproducibility and release notes

The public source repository for the tower workflow is
<https://github.com/deshengdesheng/tower-3dgs-fe-dynamics>. This repository
contains research code and selected upstream sources, not the private
author-supplied images, checkpoints, FE inputs, or generated outputs used in
the verified local run; those are intentionally not included in Git.

## Verified v4 handoff

The current demonstration is
`outputs/real_tower_roi_continuous_corrected_20261017_v4/`. It connects the
following existing inputs and newly executed stages:

1. **Step I:** Start from the supplied 399999-step tower checkpoint. The
   photographs, COLMAP reconstruction, 3DGS training, and coordinate
   registration were not rerun.
2. **Step IIa:** Use the supplied 449999-step ROI-trained checkpoint. It is an
   existing training result, not a retraining performed by the release runner.
   The exact-capsule ROI count changes from 238,354 to 772,827 across the two
   supplied checkpoints; this does not establish per-Gaussian clone/split/prune
   lineage.
3. **Step IIb:** Run adaptive terrain/object separation, Kriging, and
   multi-layer terrain extension from the 449999 checkpoint. An explicit
   exact-capsule structure guard was added because the original notebook's
   terrain thresholds misclassified structural points on this checkpoint.
   This is a guarded transfer of the notebook method, not its original
   unmodified numerical result.
4. **Step III:** Reuse the archived registered 399-frame FE displacement
   stream. Nearest-position mapping transfers 107,844 historical moving rows
   to 772,827 target ROI Gaussians; source rows may be reused. The wind-field
   generator and ANSYS/FE solve were not run.
5. **Step IV:** Split the same 1,740,385-Gaussian expanded model into 967,558
   static and 772,827 moving points, then render frames 0, 37, 74, and 111 with
   gsplat. The renderer updates means only. `scale_disp=50` is a display
   amplification; the full polar/quaternion/scale/covariance update chain was
   not exercised by this run.

The local manifests and independent artifact audit are in the v4 output
directory. Full-scene images show static-background occlusion; the tower-only
view is a diagnostic subset, not evidence that the whole scene is unobstructed.

## Rerunning with authorized inputs

Use the full command and external-file contract in `README.md`. Required
checkpoints, ROI cache, registered displacement files, and the Gaussian-to-FE
transform must belong to the same coordinate system. The launcher requires
these paths explicitly and refuses to overwrite an existing output directory.

The default output, checkpoints, FE data, images, and notebooks are excluded by
`.gitignore`. They are retained locally as execution evidence, not bundled in
the source release. A clean checkout therefore requires authorized external
inputs before the real-tower chain can run; the repository is not a
self-contained data release.

## Publication state

The author's research code is licensed under MIT as stated in `LICENSE`;
bundled third-party code retains its own license and source notices. The public
repository URL above is the source-code link used for this paper. Review
`PROVENANCE.md`, `THIRD_PARTY_NOTICES.md`, and the applicable license files
when reusing or redistributing any component.
