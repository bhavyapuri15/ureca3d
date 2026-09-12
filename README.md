# ureca3d

Reimplementations of three neural implicit surface-reconstruction methods on a
single shared pipeline, so that differences between them are attributable to the
methods rather than to three separate codebases.

| method | venue | contribution |
|---|---|---|
| **NSH** — Neural-Singular-Hessian | ACM TOG 42(6), SIGGRAPH Asia 2023 | regularises an SDF by enforcing `det(H) = 0` in the thin shell around the surface |
| **DEUDF** — Details Enhanced UDF | AAAI 2025 | addresses three failure modes of UDF learning: output-layer non-negativity, unused normals, uniformly applied Eikonal |
| **MPF** — Metric–Phase Fields | ICML 2026 | decouples proximity (`r`) from sign (`θ`) into two fields and recombines them as `φ = r·tanh(βθ) + θ` |

Everything is shared except the loss: same SIREN backbone, same query sampling
strategy per paper, same Marching Cubes extraction, same evaluation protocol.

## Results

See [RESULTS.md](RESULTS.md) for the full table and figures.

![reconstructions](docs/gallery.png)

## Install

```bash
pip install -r requirements.txt
python -m pytest tests -q        # 18 property tests, ~1 min
```

Runs on CUDA, Apple MPS or CPU; the device is selected automatically.

## Usage

```bash
python train.py --method nsh   --shape torus      --iters 10000
python train.py --method deudf --shape thin_plate --iters 10000
python train.py --method mpf   --shape cube_sheet --iters 10000
```

Outputs land in `out/<shape>_<method>/`: `recon.ply`, `input.ply`,
`metrics.json`, `loss.png`.

Useful flags:

| flag | effect |
|---|---|
| `--shape scan.ply` | any `.ply` / `.obj` / `.off` instead of a built-in shape |
| `--points 1000` | sparse input, for NSH's robustness claim |
| `--noise 0.0025` | Gaussian noise, DEUDF's noise setting |
| `--single_sided` | keeps only points whose normal faces one way, simulating a scanner that saw a thin wall from one side |
| `--level auto` | iso-value chosen relative to the field's own minimum (default for DEUDF) |
| `--no_sphere_init` | MPF ablation: disable the geometric initialisation of the phase head |
| `--iters 2000 --res 96` | fast smoke test |

## Test shapes

| name | why it is included |
|---|---|
| `sphere` | sanity check |
| `torus` | watertight with non-trivial topology; exercises ghost-geometry suppression |
| `thin_plate` | open sheet of zero thickness — no SDF exists, so NSH is expected to fail |
| `cube_sheet` | the toy from MPF Fig. 3: watertight cube with an attached thin sheet, giving a non-manifold junction and a boundary edge in one shape |

## Layout

```
common/diffops.py    gradient, hessian, det3x3, laplacian (autograd)
common/siren.py      SIREN backbone; MPF's two-head network
common/sampling.py   per-paper query samplers (Q_far, Q_near, N_delta, DEUDF pairs)
common/extract.py    Marching Cubes, adaptive iso-level, component filtering
common/metrics.py    Chamfer-L1, F-score, normal consistency
common/data.py       synthetic shapes, mesh loading, PCA normals
methods/nsh.py       tau schedule + 4 loss terms
methods/deudf.py     xi schedule + 5 loss terms
methods/mpf.py       compose, sphere init + 7 loss terms
train.py             unified driver
tests/test_math.py   property tests
```

## Implementation notes

Points where the papers were ambiguous, silent, or where a practical adaptation
was required. These are the decisions worth reviewing.

**UDF extraction needs an adaptive iso-level.** A UDF has no sign change, so
Marching Cubes at level 0 finds nothing; the usual remedy is to march at a small
positive value and obtain a double cover. But the learned field bottoms out at a
small positive minimum that varies per shape and per run, and on a finite grid
only points *near* that minimum are ever sampled. A fixed iso-value therefore
either misses the surface entirely or inflates the shell. Here the level is set
to `field_min + 0.75 × grid_spacing` — the thinnest closed shell the grid can
represent. With a fixed 0.005 the field minimum was 0.0186 and no surface was
extracted at all.

**MPF's alignment term needs a relative activation threshold.** `L_align` is
switched on after a fixed 3000 iterations, tuned for the paper's 10K schedule. On
a shorter run the threshold is never reached, the term silently contributes zero,
and the method being run is not MPF. It is treated here as a fraction of total
iterations instead.

**MPF's metric branch collapses under the published weight balance.** `r =
softplus(g_r(·))` with sharpness 100 is numerically flat for pre-activations
below about −0.05, so a head initialised there has `r ≡ 0` with vanishing
gradient, and `L_eik^r` sits pinned at exactly 1.0 (`|0 − 1|`) from the first
iteration. Initialising the metric head's final bias to a small positive value
fixes the initialisation (`L_eik^r` starts at 0.36 rather than 1.0), but does not
fix the equilibrium: `L_zero` drives `r → 0` on the surface at weight 1.0 while
`L_eik^r` opposes it at 0.007, and within a few hundred iterations `r` collapses
again and `L_eik^r` returns to ≈ 1. `φ` then reduces to `θ` and the metric branch
contributes nothing.

This is an open discrepancy, not a solved one. It may resolve at the full 10K
schedule with the paper's batch size, or it may indicate that one of the assumed
weights is wrong — `λ_far` is the value missing from Sec. 5. Watch `eik_r` in the
training log: if it is pinned at 1.0, the metric branch is dead regardless of
what the other terms are doing.

**MPF's phase field has no symmetry breaking.** `L_zero` drives `θ` to zero on
the data; `L_phase` and `L_far` act on magnitudes and are sign-agnostic. A field
that is negative everywhere satisfies all terms reasonably while never crossing
zero, and extraction returns an empty mesh — observed directly, with `φ`
converging into `[−0.034, −0.014]`. The fix is the geometric initialisation
standard in this literature (SAL, IGR, DiGS): pre-fit `θ` to a sphere SDF for a
few hundred iterations so it starts negative inside and positive outside. The
paper's optional negative phase-head bias serves the same purpose more weakly.
Disable with `--no_sphere_init` to reproduce the failure.

**DEUDF's attenuation polarity.** `δ(d) = (1 + (d/ξ)⁴)⁻¹` equals 1 at `d = 0` and
decays as `|d|` grows, so as written it weights the Eikonal constraint *toward*
the near-surface region. The surrounding text describes the intent as reducing
the contribution of points close to the surface, which would require `1 − δ`.
Both were tried; the published formula is what is implemented here.

**Reductions are means, not sums.** The papers write several losses as sums over
a batch. Means are used throughout so the weights do not depend on batch size.
This rescales the published weights uniformly and does not change their relative
balance.

**MPF's λ_far.** Sec. 5 names six loss weights and lists five values. `λ_far =
0.1` is assumed.

## Limitations

- **No DCUDF.** DEUDF's published pipeline extracts with DCUDF (Hou et al. 2023),
  which shrinks the double cover onto the minimal-distance surface. That is a
  separate optimisation and is not implemented; DEUDF's Chamfer distance here is
  inflated for that reason alone, not because the field is wrong.
- **Synthetic shapes.** Evaluation uses the four shapes above rather than
  ShapeNet / Thingi10K / ABC, so absolute numbers are not comparable to the
  published tables. The meaningful comparison is between methods on the same
  shape under identical settings.
- **Reduced training budget.** Results in `RESULTS.md` were produced at fewer
  iterations and lower grid resolution than the papers use. See that file for the
  exact settings per run.

## References

- Wang, Zhang, Xu, Zhang, Wang, Chen, Xin, Wang, Tu. *Neural-Singular-Hessian: Implicit Neural Representation of Unoriented Point Clouds by Enforcing Singular Hessian.* ACM TOG 42(6), 2023.
- Xu, Hou, Wang, Qin, Zhang, He. *Details Enhancement in Unsigned Distance Field Learning for High-fidelity 3D Surface Reconstruction.* AAAI 2025.
- Kong, Chen, Zong, Hou, Hou, Wang, He. *Metric–Phase Fields: Decoupling Distance and Sign for Thin-Structure Reconstruction from Unoriented Point Clouds.* ICML 2026.
- Sitzmann, Martel, Bergman, Lindell, Wetzstein. *Implicit Neural Representations with Periodic Activation Functions.* NeurIPS 2020.
- Hou, Chen, Wang, Qin, He. *Robust Zero Level-Set Extraction from Unsigned Distance Fields Based on Double Covering.* ACM TOG 42(6), 2023.
