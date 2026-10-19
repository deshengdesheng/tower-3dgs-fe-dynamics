# Notebook traceability

The six author notebooks named in the workflow are private working notebooks
and are not included in this source candidate. They contain author-machine
paths, private inputs, saved outputs, and a large embedded ANSYS notebook. The
table below maps their relevant methods to the release and distinguishes code
coverage from stages actually executed in the verified v4 run.

| Notebook | Method represented | v4 execution boundary |
| --- | --- | --- |
| `1_3all_Zhixianta_weiyidaoru_feng.ipynb` | Select the tower Gaussian subset, align an archived vibration stream, split static/moving points, render dynamic GS frames. | Reuses the archived registered displacement field; nearest-position transfer is audited, not an exact Gaussian-ID join. Renderer updates moving means only. |
| `1_3alll_Zhixianta_dixingtuozhan.ipynb` | Adaptive terrain/object separation, Kriging DEM, constrained multi-layer extension. | Rerun from the existing ROI-trained checkpoint with an additional exact-capsule structure guard. This guard is not part of the original classifier. |
| `1_3_all_Zhixinta_wushijian_jianhua_copy.ipynb` | FE beam-line processing, displacement interpolation, anisotropic Gaussian deformation utilities. | Existing displacement stream is remapped and replayed. Full polar/quaternion/log-scale/covariance update is not run by the v4 renderer. |
| `1_3all_zhixian_ansys copy.ipynb` | Import ANSYS nodes/elements/sections and preprocess beam geometry. | Source utilities are represented; no ANSYS solve, wind synthesis, or raw solver-output regeneration is included. |
| `1_3alll_Zhixianta_tu.ipynb` | Paper figures, model/viewer diagnostics, FE-beam ROI cache and training examples. | Existing 449999-step ROI result is supplied to the chain; long ROI training is not rerun. Stage figures distinguish point-cloud diagnostics from gsplat renders. |
| `1_1_20260520.ipynb` | Reviewer/SfM diagnostics and ROI refinement examples. | Relevant diagnostics and training interfaces are documented; private images, COLMAP data, and FE assets remain external. |

The initial photo/SfM and baseline gsplat training are upstream inputs to the
listed notebooks. The related initialization notebook `1_1chuli_yunan_zhixianta.ipynb`
is also not bundled; its COLMAP/gsplat entry points are documented in
`README.md`.

## Verified v4 lineage

```text
supplied 399999 checkpoint
  -> supplied 449999 ROI-trained checkpoint
  -> rerun terrain/object separation + Kriging + guarded terrain expansion
  -> split the same expanded checkpoint into static/moving partitions
  -> map archived FE displacement samples onto target ROI Gaussians
  -> gsplat mean-translation replay and rendered frames
```

The audited v4 output is
`outputs/real_tower_roi_continuous_corrected_20261017_v4/`. Its
`continuous_chain_manifest.json`, `step_ii/step_ii_manifest.json`,
`step_iii/runtime/mapping_audit.json`, and
`independent_artifact_verification.json` record input lineage, counts,
parameters, mapping distances, frame indices, and the exact deformation
contract. The output directory is local and ignored by Git.

The earlier `outputs/real_tower_demo_final_20261017/` branch starts from a
separate 272,300-point `lingshi2` model. It is historical comparison material,
not the v4 continuous chain.
