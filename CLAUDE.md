# ureca3d — project context for Claude Code

URECA research project (NTU, supervisor Prof. Ying He). Reimplementations of three
neural implicit surface-reconstruction methods on one shared pipeline:

| method | paper | field |
|---|---|---|
| NSH | Neural-Singular-Hessian, ACM TOG 42(6), SIGGRAPH Asia 2023 | SDF |
| DEUDF | Details Enhancement in UDF Learning, AAAI 2025 | UDF |
| MPF | Metric–Phase Fields, ICML 2026 (arXiv 2605.25503), Prof. He's group | metric + phase |

## What the supervisor asked for (12 Oct 2026)

1. Solid implementations of **one SDF method (NSH)** and **one UDF method (DEUDF)**.
2. Explain why MPF gives poor results on the toy shapes; "for these simple toy models
   MPF should deal with them without any issues". Compare against Jiayi Kong's (MPF
   first author) implementation once available.
3. Possible follow-up research: extend MPF to 4D (moving objects).

## Current status (24 Oct 2026)

Full numbers in RESULTS.md. All scored in the ground truth's frame (`metrics.py`
`frame="gt"`), components ≥ 5% of the largest kept for every method, res 256.

| method | run | torus | thin_plate | cube_sheet |
|---|---|---|---|---|
| NSH | Colab 10K, re-scored from checkpoints | F 51.6 (main surface 98.4) | F 42.0 (54.8) | F 47.6 (93.9, but 0% of the sheet) |
| DEUDF | local short run, 3K iters (30K pending) | F 95.6 | F 98.0 | F 95.4 |
| MPF | local 2K-iter debugging runs only | F 65.0 | F 55.9 | F 87.1 |

- **NSH:** main surface reproduced (torus F 98.4 / CD 2.20e-3). Its extra components
  are ghost sheets entirely outside the far-field sampling box (padded point bbox,
  |z| ≤ 0.34 for the torus); they cost about half the F-score once kept. NSH sampling
  was deliberately not changed (supervisor report, no method changes).
- **DEUDF:** working (Stage 1 below). Full 30K Colab runs never done yet.
- **MPF:** paused for the supervisor report. See Stage 2 below.
- **Old numbers** (Colab at 0846712, CPU runs from September) are invalid: old metric
  and/or pre-fix code. Kept in RESULTS.md under "Old results".

### DEUDF debugging (Stage 1, done in short local runs, 23 Oct 2026)

Compared against the paper (arXiv 2406.00346v2) and the official code
(github.com/GiliAI/DEUDF, cloned read-only to ../DEUDF-official — never copy code
from it, GPL-3).

- **Cause of the plateau:** `L_positive` (Eq. 2) was applied to the near-surface
  pairs Q as well as to Ω. Eq. 2 and the official code use Ω only. The gradient
  balance predicts f ≈ 0.021 at the data; observed 0.0206. Restricting it to Ω alone
  fixes it (isolated A/B on the pre-alignment code, thin_plate, 3000 iters, seed 0: f at data min/median/max −0.0007 / 0.0000 / 0.0004 vs 0.0200 / 0.0207 / 0.0213 without it (2000 iters); F 72.9 / 81.6 at res 128 / 256 — about the same as the full alignment on thin_plate at this budget (80.1)).
- **Aligned with the official code:** Ω = [−1,1]³; Eikonal = unweighted on Ω +
  δ-weighted on Q + ½ δ-weighted on Ω (their sampler's (−1,−1,−1) bug dropped);
  4096 surface + 4096 uniform points, pairs from the same points; LR 5e-5 and ξ held
  to 40% then cosine; Adam betas (0.1, 0.9); no grad clipping; ω = 60 in all layers.
- **Exception — normal loss:** the official `1 − |cos(∇f(q1), ∇f(q2))|` left the
  field minimum off the data (F 48.6 / 49.0 / 43.8 on thin_plate / torus /
  cube_sheet at res 256, 3000 iters); paper Eq. 3 gave 80.1 / 93.1 / 45.6. Default
  is `normal_form="paper"`; `"pairs"` kept as an option.
- **Extraction:** iso-level = median f(input points) + 0.75 grid cells
  (`--level data`, DEUDF default). F at res 128 is capped by the double-cover
  offset (F threshold ≈ 0.009 in our units), so judge DEUDF at res ≥ 256.
- **Component filter:** keep every component ≥ 5% of the largest (all methods);
  keeping only the largest dropped cube_sheet's sheet. NSH runs are re-extracted
  from their checkpoints with `reextract.py` (no retraining).
- The Stage 1 F-scores above were computed with the old per-mesh-bbox metric. In the
  ground truth's frame the 3K-iteration runs score F 95.6 / 98.0 / 95.4
  (torus / thin_plate / cube_sheet).

### MPF debugging (Stage 2, short local runs, 24 Oct 2026; paused)

- Ω = [−1,1]³ for far-field and L_lap samples (`uniform_bound=1.0`, deviation from
  Sec. 5's point-cloud box): torus 35 → 1 components.
- Sphere init imposes a genus-0 sign layout (membrane across the torus hole) but still
  scored best on all three shapes vs no init and the paper's −0.1 phase-head bias.
- β drift / rising L_phase come from L_lap; L_lap on far samples only reduces the drift
  but not F (not adopted).
- thin_plate comes out as a ≈ 0.3-thick slab (two zero sheets), not a thin shell.
- r is dead (eik_r = 1.0) until L_align starts, then ≈ 0.7.
- Next when resumed: MPF Colab runs at 10K iters, res 256; questions for Jiayi below.

Next: rerun NSH with Ω = [−1,1]³ for a like-for-like comparison (its ghost sheets lie
outside its padded-box sampling region); full DEUDF 30K runs on Colab; then MPF.

## Known bugs (fixed in b24a9af and 3a69d7a)

**Bug 1 — MPF loss weights shifted (`methods/mpf.py`, `DEFAULTS`).** Paper Sec. 5 lists
SIX weights in this order: (λ_align, λ_eik-r, λ_eik-φ, λ_lap, λ_phase, λ_far) =
(0.7, 0.007, 0.007, 0.0004, 0.002, 0.1). The repo assumed five and shifted them:

| key | wrong (current) | correct |
|---|---|---|
| lam_eik_phi | 0.0004 | 0.007 |
| lam_lap | 0.002 | 0.0004 |
| lam_phase | 0.1 | 0.002 |
| lam_far | 0.1 | 0.1 (stated in paper, not assumed) |

Also update the comments/README that call λ_far "missing" or "assumed", and the README
note about the metric branch collapsing "under the published weight balance" — it was
observed with the wrong weights and must be re-checked.

**Bug 2 — DEUDF attenuation inverted (`methods/deudf.py`, `attenuation`).** Paper Eq. 4
and the AI6131 lecture slide both give δ(d) = 1 / (1 + (ξ/d)^4): ≈0 near the surface
(Eikonal OFF at the cusp), ≈1 far away. The repo computes 1 / (1 + (d/ξ)^4), the reverse.
Implement the safe equivalent `d4 = f**4; return d4 / (d4 + xi**4)` (no division by
zero at d = 0). Remove the "polarity is an open question" text from the docstring and
README.

Caution on claims: MPF Table 4 shows a single weight at 10× changes Chamfer only
1.99 → ≤2.57, so Bug 1 may not explain everything. Do not claim it is "the" cause until
reruns confirm it.

## Other gaps to investigate (after the bug fixes)

- Run budget: papers use 10K iterations, ~15K points per batch per point set, GPU, and
  256³+ grids. Existing results were CPU, 2–2.5K iters, batch 2K, 128³.
- MPF Eq. 7 writes L_zero as a SUM over input points; the repo uses means everywhere.
  Leave as is but flag it as a question for Jiayi.
- Not in the MPF paper: sphere initialisation of θ (`sphere_init`) and the +0.05 r-head
  bias (`common/siren.py`). After Bug 1 is fixed, test with `--no_sphere_init`; keep them
  only if still needed and document why.
- MPF represents thin sheets as very thin closed shells (paper Appendix B), so
  `thin_plate` may need a finer Marching Cubes grid (try 256³).
- In MPF logs, `eik_r` pinned at ≈1.0 means r has collapsed to 0 (dead metric branch).

## Questions for Jiayi Kong (MPF first author)

Collected during Stage 2 (short local runs, 2000 iters). Ask before claiming any of
these as bugs in the paper.

1. **Keeping r alive before L_align.** In our runs `eik_r` is pinned at 1.0 (r ≡ 0 in
   the band) until L_align switches on, then ~0.7. Before that only L_eik^r (0.007)
   rewards a non-zero r, against L_zero (weight 1) pushing r(p) → 0. How does the
   reference code keep r alive early — a metric-head initialisation (we use a +0.05
   final bias, not in the paper), a different softplus slope, or something else?
2. **Sum vs mean in L_zero (Eq. 7).** Eq. 7 is a sum over X; the other terms are
   expectations. Taken literally L_zero is |X| times stronger, which would collapse r
   further. Is it a mean in the code?
3. **Ω: point-cloud box or the full cube?** Sec. 5 draws far-field samples from the
   bounding box of the input cloud; for flat shapes that leaves most of the Marching
   Cubes volume unconstrained, and we got spurious surfaces there (torus: 35 → 1
   components with [−1,1]³). Is Ω the padded box, [−1,1]³, or something else? How
   much padding? Where is L_lap sampled ("sparse samples in the ambient space")?
4. **δ and the band jitter.** What δ is used for N_δ, and how are band samples
   generated — Gaussian jitter with what σ (fixed, or per-point like NSH's k-NN
   distance)? We use δ = 0.05 and NSH-style σ.
5. **Initialisation of θ.** Sphere initialisation (ours, not in the paper) fixes a
   genus-0 sign layout: it helps cube_sheet but leaves a membrane across the torus
   hole. What does the reference code do for θ — the optional negative phase-head
   bias (what value?), or standard SIREN init only?

## Rerun plan

| method | shapes | settings |
|---|---|---|
| NSH | torus, thin_plate, cube_sheet | 10K iters, res 256 |
| DEUDF (fixed δ) | torus, thin_plate, cube_sheet | 10K iters, adaptive iso |
| MPF (fixed weights) | torus, thin_plate, cube_sheet | 10K iters, res 256, with and without sphere init |

Runs go on Google Colab (T4 GPU) via `notebooks/ureca_colab.ipynb`; local machine is a
MacBook Air (MPS/CPU) — use it only for smoke tests (`--iters 300 --res 64`).
After reruns: update `RESULTS.md`, regenerate `docs/gallery.png`, update README notes.

## Working conventions

- Bhavya writes the core maths himself where it is a learning exercise; explain changes
  to loss terms with the paper equation number they implement.
- Run `python -m pytest tests -q` after any change to `common/` or `methods/`.
- Keep one shared pipeline: method differences live only in `methods/*.py`.
- Commit each logical change separately with a clear message (e.g. "Fix MPF loss
  weights to match paper Sec. 5"). Do not push without asking.
- Paper equations: cite as "NSH Eq. 13", "DEUDF Eq. 4", "MPF Eq. 7".
