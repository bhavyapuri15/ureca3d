# ureca3d — reimplementing NSH, DEUDF and MPF

Three neural implicit surface-reconstruction methods, one shared pipeline, so the
comparison is apples-to-apples.

| paper | venue | what it changes |
|---|---|---|
| **NSH** — Neural-Singular-Hessian | TOG / SIGGRAPH Asia 2023 | regularises an SDF by forcing `det(Hessian) = 0` near the surface |
| **DEUDF** — Details Enhanced UDF | AAAI 2025 | fixes three failure modes of UDF learning (non-negativity, normals, Eikonal weighting) |
| **MPF** — Metric–Phase Fields | ICML 2026 | splits the field into a distance part `r` and a sign part `θ` |

## Install

```bash
pip install -r requirements.txt
```

## Your job

Everything is written except the maths. Search for `TODO(you)`:

```
common/diffops.py    gradient, hessian, det3x3, laplacian      (4 functions)
methods/nsh.py       tau_schedule + 4 loss terms               (5 functions)
methods/deudf.py     xi_schedule + 5 loss terms                (6 functions)
methods/mpf.py       compose + 7 loss terms                    (8 functions)
```

Do `common/diffops.py` first — nothing runs without it.

## Loop

```bash
python -m pytest tests -q            # test YOUR code
python -m pytest tests -q --sol      # test the reference (should always pass)

python train.py --method nsh   --shape torus      --iters 10000
python train.py --method deudf --shape thin_plate --iters 10000
python train.py --method mpf   --shape cube_sheet --iters 10000
```

Add `--solutions` to run with reference implementations — use it to check that a
bad reconstruction is your maths and not your setup. Add `--iters 2000 --res 96`
for a fast smoke test.

Results land in `out/<shape>_<method>/`: `recon.ply`, `input.ply`,
`metrics.json`, `loss.png`.

## Shapes

| name | why it is in the list |
|---|---|
| `sphere` | sanity check — if this fails, the bug is not subtle |
| `torus` | watertight with a hole; tests topology and ghost geometry |
| `thin_plate` | open sheet, zero thickness — no SDF exists; NSH must fail here |
| `cube_sheet` | MPF's own Fig. 3 toy: watertight cube + non-manifold thin sheet |

Any `.ply` / `.obj` / `.off` path also works: `--shape scan.ply`.

Useful flags: `--points 1000` (sparse input — NSH claims robustness),
`--noise 0.0025` (DEUDF's noise setting), `--single_sided` (simulates a scanner
that saw a thin wall from one side only — MPF's hardest case).

## Known limitations, state them in your report

- **UDF extraction is a double cover.** DEUDF's field has no sign change, so
  Marching Cubes at level 0 finds nothing. We march at a small positive iso-value
  instead, which gives two sheets straddling the true surface. The paper uses
  DCUDF (Hou et al. 2023) to collapse it; that is a separate optimisation and is
  out of scope here. Expect DEUDF's Chamfer distance to look worse than published
  **for this reason alone**, not because the field is wrong.
- **Synthetic shapes, not ShapeNet/Thingi10K.** Numbers are not comparable to the
  published tables. The *relative ordering across methods on the same shape* is
  what is meaningful.
- **MPF's λ_far.** The paper lists six weight names and five values (Sec. 5).
  We assumed 0.1. Worth asking Prof. He.
