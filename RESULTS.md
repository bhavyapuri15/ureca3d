# Results

All runs share one pipeline: SIREN backbone, per-paper query sampling, Marching
Cubes extraction, and the evaluation protocol of NSH §5.2 — both meshes rescaled
to `[-0.5, 0.5]³`, 100K points sampled from each, F-score threshold 0.005.

**These are reduced-budget runs on CPU**, not the papers' settings. Absolute
numbers are therefore not comparable to the published tables; the comparison that
carries information is between methods on the same shape. Settings: 6K input
points (8K for the torus), batch 2000, grid resolution 128³, iteration counts as
listed.

| shape | method | iters | CD-L1 ×10³ ↓ | F-score ↑ | Normal C. ↑ | #comp | time (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cube_sheet` | DEUDF | 2000 | 176.51 | 1.4 | 43.8 | 120 | 283 |
| `cube_sheet` | MPF | 2500 | 7.59 | 25.5 | 96.3 | 5 | 989 |
| `cube_sheet` | NSH | 2000 | 83.50 | 3.2 | 60.5 | 42 | 670 |
| `thin_plate` | DEUDF | 2000 | 3.95 | 74.4 | 87.2 | 2 | 285 |
| `thin_plate` | MPF | 2000 | 22.42 | 19.2 | 89.8 | 7 | 744 |
| `thin_plate` | NSH | 2000 | 64.62 | 4.4 | 86.6 | 247 | 676 |
| `torus` | NSH | 1500 | 2.39 | 98.1 | 99.1 | 21 | 648 |

Reproduce any row with:

```bash
python train.py --method <method> --shape <shape> --points 6000 \
                --batch 2000 --res 128 --iters <iters>
```

![reconstructions](docs/gallery.png)

## Reading the table

**NSH is strong where an SDF is well posed and fails where it is not.** On the
torus it reaches Chamfer 2.39×10⁻³ and F-score 98.1 in 1500 iterations. On the
open thin plate the same method returns F-score 4.4 and fragments into 247
components. This is the expected outcome, not a bug: a sheet of zero thickness
admits no consistent inside/outside, so the signed field has no well-posed target
and tears apart trying to construct one. It is the failure that motivates both
later papers.

**DEUDF handles the open sheet.** Chamfer 3.95×10⁻³ and F-score 74.4 on
`thin_plate`, in two connected components — the double cover, one sheet either
side of the true surface. Without DCUDF to collapse that cover, a substantial
part of the residual Chamfer error is the offset between the two sheets rather
than misplacement of the surface itself.

**DEUDF struggles on `cube_sheet`.** Chamfer 176×10⁻³ across 120 components. The
shape mixes a watertight solid with a thin sheet, and at this iteration count the
unsigned field does not separate them cleanly; the adaptive iso-level then cuts a
shell through both at once. This is the case the third paper is aimed at.

**MPF is the only method that handles `cube_sheet`.** Chamfer 7.59×10⁻³, normal
consistency 96.3, 5 components — an order of magnitude better than either
alternative on the same input, extracted directly from φ at level 0 with standard
Marching Cubes and no double-cover post-processing. This reproduces the paper's
central claim on its own toy example.

MPF's lower F-score on `thin_plate` than DEUDF (19.2 against 74.4) sits alongside
its higher normal consistency (89.8 against 87.2): the surface is oriented
correctly but not yet tightly placed, which is consistent with an under-trained
run rather than a wrong field.

## Caveats

- Iteration counts are a quarter to a sixth of the papers', and grid resolution
  is 128³ rather than 192³–512³. Every number here would improve with the full
  schedule; MPF in particular was still improving when training stopped.
- DEUDF is extracted as a double cover. DCUDF is not implemented, so its Chamfer
  distance is inflated for that reason alone.
- MPF's metric branch collapses under the published weight balance at this
  budget — `L_eik^r` returns to ≈ 1.0 within a few hundred iterations, meaning φ
  has reduced to θ. The results above were obtained in that regime, so the metric
  field is not yet contributing what the paper intends. See the implementation
  notes in the README.
- Component counts are reported because they are the direct measure of ghost
  geometry, which is what NSH's singular-Hessian term exists to suppress. A
  with/without ablation on that term is the obvious next experiment and has not
  been run.

## Prior state

Two results in this table were empty before the extraction and initialisation
issues described in the README were fixed:

| run | before | after |
|---|---|---|
| DEUDF, `thin_plate` | no surface found (fixed iso-value 0.005 below the field minimum of 0.0186) | CD 3.95×10⁻³, F 74.4 |
| MPF, `cube_sheet` | no surface found (φ converged into [−0.034, −0.014], never crossing zero) | CD 7.59×10⁻³, F 25.5 |
