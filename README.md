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

## Status (24 Oct 2026)

Numbers: [RESULTS.md](RESULTS.md) (F-score at 0.005, ground-truth frame, res 256).

- **NSH — reproduced on the torus.** Main surface: F 98.4, CD 2.20×10⁻³ (10K
  iterations); full mesh: F 51.6. The gap is ghost sheets outside NSH's sampled
  far-field region: our setup samples NSH's free space only in the padded point-cloud
  box, while DEUDF and MPF now sample [−1, 1]³, and every ghost component lies outside
  that box. It fails on the open and non-manifold shapes, as expected for an SDF:
  `thin_plate` main surface F 54.8, and on `cube_sheet` it reconstructs the cube but
  recovers none of the sheet. **Next step:** rerun NSH with Ω = [−1, 1]³ for a
  like-for-like comparison.
- **DEUDF — working** after aligning with the official code
  ([GiliAI/DEUDF](https://github.com/GiliAI/DEUDF)): F 95.6 / 98.0 / 95.4 on torus /
  thin_plate / cube_sheet (3K-iteration short runs; full 30K runs pending). Root cause
  of the earlier failure: `L_positive` was applied at the near-surface pair points as
  well as in Ω, which lifted the field ≈ 0.02 off the data, so extraction missed the
  surface.
- **Bugs fixed:** MPF loss weights were shifted by one position (MPF Sec. 5); DEUDF's
  Eikonal attenuation was inverted (DEUDF Eq. 4).
- **Evaluation fixed:** meshes are scored in the ground truth's frame (previously each
  mesh was normalised by its own bounding box), and every component ≥ 5% of the largest
  is kept for every method (previously only the largest).
- **Deviations from the papers:**
  - DEUDF normal term: paper Eq. 3 is the default; the official code's
    `1 − |cos(∇f(q¹), ∇f(q²))|` did worse in short runs (`normal_form="pairs"`).
  - DEUDF settings the paper leaves unstated follow the official code (batch, LR/ξ
    schedule, Adam betas, ω).
  - DEUDF surfaces are Marching Cubes double covers at a data-anchored iso-level, not
    DCUDF.
  - MPF samples free space in [−1, 1]³, not the point cloud's bounding box (Sec. 5).
  - MPF uses a sphere initialisation of θ and a +0.05 metric-head bias (not in the
    paper).
  - Losses are means, not sums.
- **MPF — in progress.** Spurious surfaces are fixed by sampling over [−1, 1]³, but
  thin plates still come out as ≈ 0.3-thick slabs instead of thin shells, and the
  metric branch r stays near zero until `L_align` starts. Several questions are
  open with the MPF authors.

## Results

See [RESULTS.md](RESULTS.md) for the full table and figures.

![reconstructions](docs/gallery.png)

## Install

```bash
pip install -r requirements.txt
python -m pytest tests -q        # 26 property tests, ~10 s
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
| `--level data` | UDF iso-value just above the median field value at the input points (default for DEUDF) |
| `--level auto` | UDF iso-value relative to the field's global minimum (the older rule) |
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
methods/deudf.py     LR/xi schedule + 4 loss terms (Eq. 1-5)
methods/mpf.py       compose, sphere init + 7 loss terms
train.py             unified driver
reextract.py         re-extract + re-evaluate a finished run from ckpt.pt
tests/test_math.py   property tests
```

## Implementation notes

Points where the papers were ambiguous, silent, or where a practical adaptation
was required. These are the decisions worth reviewing.

**UDF extraction needs an iso-level anchored at the data.** A UDF has no sign
change, so Marching Cubes at level 0 finds nothing; the usual remedy is to march
at a small positive value and obtain a double cover. A fixed iso-value either
misses the surface or inflates the shell, so the level has to be adaptive. The
earlier rule, `field_min + 0.75 × grid_spacing`, is relative to the field's
global minimum, which need not be at the data: in one Colab run the level was
0.0101 while f at the input points was ≈ 0.0206, and almost nothing was
extracted. The default (`--level data`) is now `median f(input points) + 0.75 ×
grid_spacing`; 0.75 cells corresponds to DCUDF's 0.003 threshold at 512³ in the
official DEUDF setup. The F-score threshold (0.005 after rescaling to
[−0.5, 0.5]³) is ≈ 0.009 in our coordinates, and a double cover sits ≈ level
away from the surface on each side, so DEUDF's scores depend strongly on the
grid resolution: use res ≥ 256.

**DEUDF follows the authors' released code where it differs from the paper.**
The paper (arXiv 2406.00346v2) leaves batch size, iteration count, Adam
settings and several point sets unstated. The released code
([GiliAI/DEUDF](https://github.com/GiliAI/DEUDF), the config its README uses)
fills these in, and `methods/deudf.py` reproduces its behaviour (reimplemented
here, not copied): 4096 surface + 4096 uniform points per iteration, with the
normal pairs built from the surface points; `L_positive` (Eq. 2) on uniform
samples in Ω = [−1, 1]³ only; an Eikonal term (Eq. 4) with an unweighted
far-field part in addition to the δ-weighted parts; LR 5e-5 and ξ = 0.01 held
for 40% of training and then cosine-annealed (to 0 and 0.002); Adam betas
(0.1, 0.9); no gradient clipping; ω = 60 in every SIREN layer; ~30K iterations.
One quirk of the released sampler — it evaluates part of the Eikonal term at the
constant point (−1, −1, −1) — is a bug and is not reproduced.

The exception is the normal term. The released code uses
`1 − |cos(∇f(q¹), ∇f(q²))|`, which parallel gradients satisfy as well as
anti-parallel ones, so it does not pin the field's minimum between q¹ and q²
(i.e. at the data). The default here is the paper's Eq. 3, which compares each
gradient with the PCA normal; the released form is available as
`normal_form="pairs"`. In 3000-iteration runs with everything else identical:

| shape (F at res 256) | `pairs` (released code) | `paper` (Eq. 3, default) |
|---|---|---|
| thin_plate | 48.6 (\|f\| at data ≤ 0.013) | **80.1** (≤ 0.001) |
| torus | 49.0 (≤ 0.014) | **93.1** (≤ 0.003) |
| cube_sheet | 43.8 (≤ 0.013) | **45.6** (≤ 0.009) |

The other changes were ablated the same way on thin_plate (one at a time):
default Adam betas were worse (F 33.2 vs 48.6); batch 15K, hidden ω = 30 and
dropping the 40% hold made no difference at this budget. The released form may
behave better at the official budget (~30K iterations, 300K points); that has
not been tested.

An earlier version of this repo applied `L_positive` to the near-surface pairs
as well as to the uniform samples. Since the UDF is flat near its minimum, the
penalty on small f within 0.003 of the data lifted f at the data itself: `dist`
plateaued at ≈ 0.0206 (predicted from the gradient balance: ≈ 0.021).
Restricting it to Ω as Eq. 2 states brings f at the data to ≈ 0 — this single
change fixed the plateau (isolated A/B: pre-alignment code with only this change, thin_plate, 3000 iterations, seed 0: f at the data −0.0007 / 0.0000 / 0.0004 (min / median / max) against 0.0200 / 0.0207 / 0.0213 without it (2000 iterations); F 72.9 / 81.6 at res 128 / 256. On thin_plate at this budget, this change alone scores about as well as the full alignment below (F 80.1 at res 256)).

**Component filtering keeps every large component.** After Marching Cubes,
components with fewer than 5% of the largest one's faces are dropped as ghost
geometry, for every method. An earlier version kept only the single largest
component, which threw away the sheet of `cube_sheet` when its double cover
split. `metrics.json` reports both counts (`n_components`, `n_kept`).
`reextract.py <run_dir>` re-extracts and re-evaluates a finished run from its
`ckpt.pt` without retraining.

**MPF's alignment term needs a relative activation threshold.** `L_align` is
switched on after a fixed 3000 iterations, tuned for the paper's 10K schedule. On
a shorter run the threshold is never reached, the term silently contributes zero,
and the method being run is not MPF. It is treated here as a fraction of total
iterations instead.

**MPF's metric branch collapsed in earlier runs — to be re-checked.** `r =
softplus(g_r(·))` with sharpness 100 is numerically flat for pre-activations
below about −0.05, so a head initialised there has `r ≡ 0` with vanishing
gradient, and `L_eik^r` sits pinned at exactly 1.0 (`|0 − 1|`) from the first
iteration. Initialising the metric head's final bias to a small positive value
fixes the initialisation (`L_eik^r` starts at 0.36 rather than 1.0), but does not
fix the equilibrium: `L_zero` drives `r → 0` on the surface at weight 1.0 while
`L_eik^r` opposes it at 0.007, and within a few hundred iterations `r` collapses
again and `L_eik^r` returns to ≈ 1. `φ` then reduces to `θ` and the metric branch
contributes nothing.

Re-checked with the corrected weights (Sec. 5) in 2000-iteration runs: `eik_r` is
still pinned at 1.0 until `L_align` switches on (at 30% of training), then settles
near 0.7, and r stays small away from the data rather than growing like a distance.
By Proposition 3.1 the surface is θ's zero set, so this does not move the surface,
but the metric branch is not yet doing what the paper describes. How the reference
code keeps r alive is an open question for the authors. Watch `eik_r` in
the training log: pinned at 1.0 means the metric branch is dead.

**MPF samples free space in [−1, 1]³ — a deviation from Sec. 5.** The paper
draws far-field samples (for `L_phase`, `L_far`) "uniformly from the bounding box Ω
of the input point cloud". For flat shapes that box is a slab — |z| < 0.34 for the
torus, padded — while Marching Cubes covers [−1, 1]³, so the field is unconstrained
in most of the extraction volume and spurious zero level sets appear there. Here
the far-field and ambient (`L_lap`) samples come from [−1, 1]³ (`uniform_bound=1.0`
in `methods/mpf.py`; `None` restores the paper's box). Torus, 2000 iterations:
35 components with the box, 1 with the cube; F 38.7 → 65.0 in the ground truth's
frame.

**MPF's sphere initialisation of θ is kept, for its results.** It is not in the
paper. An earlier note here said MPF without it converges to an all-negative φ and
an empty mesh; that was observed with the shifted loss weights. With the correct
`λ_eik-φ`, `L_eik^φ` asks for |∇θ| = 1 at the data, which forces θ to cross zero
there, and runs without the initialisation do produce surfaces. In 2000-iteration
runs (F, ground-truth frame) sphere initialisation still scored best on all three
shapes — torus / thin_plate / cube_sheet: 65.0 / 55.9 / 87.1, against 51.9 / 49.3 /
59.6 with no initialisation and 57.9 / 50.4 / 56.1 with the paper's optional
negative phase-head bias (−0.1). It imposes a genus-0 sign layout, though: on the
torus it leaves a membrane across the hole. `--no_sphere_init` disables it.

**DEUDF's attenuation.** Eq. 4 weights the Eikonal term by `δ(d) = 1 / (1 +
(ξ/d)⁴)`, which is ≈ 0 near the surface (the constraint is off at the UDF's cusp)
and ≈ 1 far from it. It is computed as `d⁴ / (d⁴ + ξ⁴)` to avoid dividing by zero
at `d = 0`. An earlier version of this repo used the inverse, `1 / (1 + (d/ξ)⁴)`;
DEUDF results produced before that fix used the wrong weighting.

**Metrics are computed in the ground truth's frame.** Both meshes are
normalised to [−0.5, 0.5]³ with the ground truth's centre and scale before
sampling (`common/metrics.py`, `frame="gt"`). An earlier version normalised each
mesh by its own bounding box, so any stray geometry rescaled and shifted the whole
reconstruction: a sphere plus one small blob near the domain edge scored F 1.3
instead of 91.7. Numbers reported before this change are not comparable;
`metrics.json` keeps the old-frame values (`f_score_each_bbox`) for comparison, and
`reextract.py` re-scores a finished run from its checkpoint.

**Reductions are means, not sums.** The papers write several losses as sums over
a batch. Means are used throughout so the weights do not depend on batch size.
This rescales the published weights uniformly and does not change their relative
balance.

**MPF's loss weights.** Sec. 5 lists six weights in the order (`λ_align`,
`λ_eik-r`, `λ_eik-φ`, `λ_lap`, `λ_phase`, `λ_far`) = (0.7, 0.007, 0.007, 0.0004,
0.002, 0.1). An earlier version of this repo read only five values and shifted
three of them; results produced before that fix used the wrong weights.

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
