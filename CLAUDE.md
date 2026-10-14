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

## Known bugs — fix these first

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
