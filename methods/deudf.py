"""DEUDF -- Details Enhanced UDF learning (Xu et al., AAAI 2025).

Fits an unsigned distance field, which -- unlike an SDF -- can represent open
surfaces, thin sheets and shapes with internal structure, since it requires no
globally consistent inside/outside. UDFs reconstruct visibly worse than SDFs in
practice, and this paper is a diagnosis of why, in three parts.

1. Non-negativity enforced by the output layer. abs() folds the field, producing
   a W-shaped profile with several local minima near the surface, so the zero set
   is ambiguous and gradient directions oscillate. softplus is nearly flat for
   negative inputs, so the gradient vanishes over a wide band around the surface
   and optimisation stalls exactly where precision is needed. DEUDF uses an
   unconditioned MLP -- no abs, no softplus -- and allows the field to dip
   slightly negative, recovering a clean V-shaped profile. The surface is then
   the local minimum of |f| rather than an exact zero set, and non-negativity is
   imposed softly (L_positive).

2. Normal directions unused. A UDF's gradient vanishes at the surface, so it
   cannot be constrained there. DEUDF constrains it just off the surface, on
   pairs of points straddling it along the (unoriented, PCA) normal.

3. Uniformly applied Eikonal constraint. ||grad u|| = 1 holds away from the
   surface but not near it, where the field genuinely bends. DEUDF attenuates the
   constraint as a function of the field value, so it switches itself off close
   to the surface.

Paper vs released code. Where the paper (arXiv 2406.00346v2) is silent or
differs from the authors' released code (github.com/GiliAI/DEUDF, the config the
README uses), this file follows the code, since that is what produced the
published results. Each place is marked "Official code:" below. In summary:
L_positive and part of the Eikonal term use uniform samples in [-1,1]^3 only;
the Eikonal term has an unweighted far-field part; LR and xi are held for 40% of
training and then cosine-annealed; Adam betas (0.1, 0.9); no gradient clipping;
omega 60 in every sine layer; 4096 surface + 4096 uniform points per iteration.

One exception: the normal term. The official code compares the gradients at q1
and q2 with each other (normal_form="pairs"); the default here is the paper's
Eq. 3, which compares each with the PCA normal (normal_form="paper"). In 3000-
iteration runs (seed 0, otherwise identical settings) the pair form left the
field minimum off the data (|f| at the input points up to 0.013-0.016) and gave
F 48.6 / 49.0 / 43.8 on thin_plate / torus / cube_sheet at res 256; Eq. 3 kept
|f| at the data below 0.001 / 0.003 / 0.009 and gave F 80.1 / 93.1 / 45.6. The
pair form may behave better at the official budget (~30K iterations, 300K
points); it has not been tested there.
"""
import math

import torch

from common.diffops import gradient, make_input


# Paper, "UDF learning" paragraph; the official code where the paper is silent
DEFAULTS = dict(
    lam_dist=400.0,        # lambda_1, Eq. 1
    lam_positive=50.0,     # lambda_2, Eq. 2
    lam_normal=40.0,       # lambda_3, Eq. 3
    lam_eikonal=10.0,      # lambda_4, Eq. 4
    normal_form="paper",   # "paper" (Eq. 3) or "pairs" (official code), see loss_normal*
    xi_start=0.01,
    xi_end=0.002,
    lam_max=0.003,         # q = p +/- lam*n with lam in (0, 0.003]
    lr=5e-5,               # official code: 1e-4 x phase factor 0.5
    hold=0.4,              # LR and xi constant for the first 40%, then cosine
    adam_betas=(0.1, 0.9), # official code; the paper only says "Adam"
    grad_clip=None,        # official code does not clip
    iters=30000,           # official code: 61 epochs x 489 iterations ~ 29.8K
    batch=4096,            # official code: 4 x 2048 = 4096 surface + 4096 uniform
    uniform_bound=1.0,     # Omega = [-1,1]^3 (paper Sec. 3; official sampler)
    omega_0=60.0,          # 60 for clean clouds, 30 for noisy (their footnote 1)
    omega_hidden=60.0,     # official code uses the same omega in every sine layer
    n_layers=5,
    knn_pca=16,
)


def phase_factor(it, total, hold=0.4):
    """1 for the first `hold` fraction of training, then cosine-annealed to 0.

    Official code: the LR factor is 0.5 up to 40% progress and then cosine to 0,
    and xi's factor is 1.0 up to 40% and then cosine to 0.2, i.e. both have this
    shape. The paper says the LR "decays to zero following a cosine annealing
    schedule" and that xi decreases "following the learning rate". The official
    code steps these once per epoch; here they are per iteration.
    """
    t = it / max(total - 1, 1)
    if t <= hold:
        return 1.0
    return 0.5 * (1.0 + math.cos(math.pi * (t - hold) / (1.0 - hold)))


def lr_lambda(it, total, cfg):
    """Multiplier on cfg["lr"] at iteration `it` (used by train.py's LambdaLR)."""
    return phase_factor(it, total, cfg["hold"])


def xi_schedule(it, total, xi_start=0.01, xi_end=0.002, hold=0.4):
    """Attenuation bandwidth xi of Eq. 4: 0.01 -> 0.002 on the LR's schedule."""
    return xi_end + (xi_start - xi_end) * phase_factor(it, total, hold)


def loss_dist(f_surface):
    """Eq. 1: |f(p)| at the input points. Mean rather than sum, so the weights
    are independent of batch size; this rescales the published weights uniformly
    and does not change their relative balance."""
    return f_surface.abs().mean()


def loss_positive(f_uniform):
    """Eq. 2: mean exp(-100 f(x)) over uniform samples x in Omega = [-1,1]^3.

    Only space samples -- never the near-surface pairs Q. Applying it to Q (as an
    earlier version of this repo did) penalises small f within 0.003 of the data,
    and since the UDF is flat near its minimum that lifts f at the data too: f
    plateaued at ~0.02 instead of 0. Official code: the same set (off-surface
    rows of the batch only).

    Note the absence of an absolute value -- this is deliberate and the paper
    devotes a Remark to it. exp(-100 f) diverges for negative f and is negligible
    for positive f, making it a one-sided penalty: stay positive, with room to
    dip slightly negative near the surface. The alternative exp(-100|f|) used by
    NeuralUDF penalises near-zero values from both sides, which suppresses small
    disconnected components.

    The exponent is clamped: without it a strongly negative f overflows to inf
    and training collapses to NaN.
    """
    return torch.exp(torch.clamp(-100.0 * f_uniform, max=20.0)).mean()


def loss_normal(grad_q1, grad_q2, normals, eps=1e-8):
    """Eq. 3 as written in the paper (normal_form="paper", the default):
    sum over k in {1,2} of 1 + (-1)^k <grad f(q^k), n> / (|grad| |n|).

    q^1 = p + lam*n lies on the +n side and q^2 = p - lam*n on the -n side. A
    distance field increases away from the surface on both sides, so at the
    optimum cos(grad f(q^1), n) = +1 and cos(grad f(q^2), n) = -1, and both terms
    vanish.

    Flipping n swaps the roles of q^1 and q^2 and leaves the sum unchanged, which
    is what makes unoriented PCA normals usable. No absolute value is taken: the
    pairing already handles orientation, and abs() would discard the information
    distinguishing the two sides.
    """
    n = normals / (normals.norm(dim=-1, keepdim=True) + eps)
    u1 = grad_q1 / (grad_q1.norm(dim=-1, keepdim=True) + eps)
    u2 = grad_q2 / (grad_q2.norm(dim=-1, keepdim=True) + eps)
    return ((1.0 - (u1 * n).sum(-1)) + (1.0 + (u2 * n).sum(-1))).mean()


def loss_normal_pairs(grad_q1, grad_q2, eps=1e-8):
    """Normal alignment as the official code computes it (normal_form="pairs"):
    mean 1 - |cos(grad f(q1), grad f(q2))|.

    Differs from the paper's Eq. 3: the PCA normal is only used to place q1 and
    q2, not compared against. The term asks the gradients on the two sides to be
    collinear -- but parallel gradients satisfy it as well as anti-parallel ones,
    so it does not force the field's minimum to lie between q1 and q2, i.e. at the
    data. That is the likely reason it did worse in short runs (see the module
    docstring). It is invariant to the sign of n by construction. Range [0, 1],
    against [0, 4] for Eq. 3.
    """
    u1 = grad_q1 / (grad_q1.norm(dim=-1, keepdim=True) + eps)
    u2 = grad_q2 / (grad_q2.norm(dim=-1, keepdim=True) + eps)
    return (1.0 - (u1 * u2).sum(-1).abs()).mean()


def attenuation(f, xi):
    """delta(d) = 1 / (1 + (xi/d)^4), the weight in Eq. 4 (from IDF, Wang et al.
    2022).

    delta ~ 0 near the surface, so the Eikonal constraint is switched off at the
    cusp of the UDF where ||grad u|| = 1 cannot hold, and delta ~ 1 far from it.
    delta = 0.5 at |d| = xi. Computed as d^4 / (d^4 + xi^4), which is the same
    function without the division by zero at d = 0.
    """
    d4 = f ** 4
    return d4 / (d4 + xi ** 4)


def eikonal_residual(grad):
    return (torch.linalg.norm(grad, dim=-1) - 1.0).abs()


def loss_eikonal_weighted(f_query, grad_query, xi):
    """mean delta(f(x)) |‖grad f(x)‖ - 1|, the summand of Eq. 4.

    The weight is detached. Left attached, the network can reduce the loss by
    pushing |f| towards zero to shrink delta rather than by satisfying the
    constraint.
    """
    w = attenuation(f_query, xi).detach().reshape(-1)
    return (w * eikonal_residual(grad_query)).mean()


def loss_eikonal(f_q, g_q, f_u, g_u, xi):
    """Eq. 4 as the official code applies it, on Q = {q1, q2} and uniform Omega:

        mean_Omega |‖grad f‖ - 1|                 (unweighted, "base_grad")
      + mean_Q     delta(f) |‖grad f‖ - 1|        ("base_grad_lr", surface rows)
      + 1/2 mean_Omega delta(f) |‖grad f‖ - 1|    ("base_grad_lr", uniform rows)

    The paper writes a single delta-weighted sum over Q u Omega. The unweighted
    far-field part is only in the code; far from the surface delta ~ 1, so the
    two agree there, but the code's version also constrains uniform points that
    happen to lie near the surface.

    Official code bug, dropped here: for the uniform rows its sampler sets q_r to
    the normals tensor, which is (-1,-1,-1) for off-surface points, so the last
    term is also evaluated at that one constant corner point.
    """
    return (eikonal_residual(g_u).mean()
            + loss_eikonal_weighted(f_q, g_q, xi)
            + 0.5 * loss_eikonal_weighted(f_u, g_u, xi))


def compute_loss(net, pts_surface, q1, q2, normals, pts_uniform, it, total, cfg):
    """Eq. 5. q1, q2 are built from pts_surface (same points, +/- lam*n).
    Returns (total_loss, component_dict).

    With per-set means and the paper's lambdas this is exactly twice the official
    code's loss (which averages every term over the combined surface + uniform
    batch, with zeros where a term does not apply). A uniform factor does not
    change Adam's updates.
    """
    xs, a, b, xu = map(make_input, (pts_surface, q1, q2, pts_uniform))

    f_s = net(xs)
    f_a, f_b = net(a), net(b)
    g_a, g_b = gradient(f_a, a), gradient(f_b, b)
    f_u = net(xu)
    g_u = gradient(f_u, xu)

    xi = xi_schedule(it, total, cfg["xi_start"], cfg["xi_end"], cfg["hold"])

    L_dist = loss_dist(f_s)
    L_pos = loss_positive(f_u)
    if cfg["normal_form"] == "pairs":
        L_nrm = loss_normal_pairs(g_a, g_b)
    else:
        L_nrm = loss_normal(g_a, g_b, normals)
    L_eik = loss_eikonal(torch.cat([f_a, f_b], 0), torch.cat([g_a, g_b], 0),
                         f_u, g_u, xi)

    total_loss = (cfg["lam_dist"] * L_dist + cfg["lam_positive"] * L_pos
                  + cfg["lam_normal"] * L_nrm + cfg["lam_eikonal"] * L_eik)
    return total_loss, {"dist": L_dist.item(), "positive": L_pos.item(),
                        "normal": L_nrm.item(), "eikonal": L_eik.item(), "xi": xi}
