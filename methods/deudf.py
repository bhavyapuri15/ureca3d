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
"""
import torch

from common.diffops import gradient, make_input


# Paper, "UDF learning" paragraph
DEFAULTS = dict(
    lam_dist=400.0,        # lambda_1
    lam_positive=50.0,     # lambda_2
    lam_normal=40.0,       # lambda_3
    lam_eikonal=10.0,      # lambda_4
    xi_start=0.01,
    xi_end=0.002,
    lam_max=0.003,         # displacement for the q1/q2 pair sampling
    lr=5e-5,               # Adam, cosine-annealed to zero
    iters=10000,
    batch=15000,
    omega_0=60.0,          # 60 for clean clouds, 30 for noisy (their footnote 1)
    n_layers=5,
    knn_pca=16,
)


def xi_schedule(it, total, xi_start=0.01, xi_end=0.002):
    """Attenuation bandwidth, annealed 0.01 -> 0.002 "following the learning
    rate", i.e. on the same cosine shape as the LR schedule."""
    import math
    t = it / max(total - 1, 1)
    return xi_end + 0.5 * (xi_start - xi_end) * (1 + math.cos(math.pi * t))


def loss_dist(f_surface):
    """Eq. 1: |f(p)| at the input points. Mean rather than sum, so the weights
    are independent of batch size; this rescales the published weights uniformly
    and does not change their relative balance."""
    return f_surface.abs().mean()


def loss_positive(f_query):
    """Eq. 2: mean exp(-100 f(x)).

    Note the absence of an absolute value -- this is deliberate and the paper
    devotes a Remark to it. exp(-100 f) diverges for negative f and is negligible
    for positive f, making it a one-sided penalty: stay positive, with room to
    dip slightly negative near the surface. The alternative exp(-100|f|) used by
    NeuralUDF penalises near-zero values from both sides, which suppresses small
    disconnected components.

    The exponent is clamped: without it a strongly negative f overflows to inf
    and training collapses to NaN.
    """
    return torch.exp(torch.clamp(-100.0 * f_query, max=20.0)).mean()


def loss_normal(grad_q1, grad_q2, normals, eps=1e-8):
    """Eq. 3: sum over k in {1,2} of 1 + (-1)^k <grad f(q^k), n> / (|grad| |n|).

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


def attenuation(f, xi):
    """delta(d) = (1 + (d/xi)^4)^-1, the weight in Eq. 4 (from IDF, Wang et al.
    2022).

    Convention note. As written the function equals 1 at d = 0 and decays as |d|
    grows, so it weights the Eikonal constraint *towards* the near-surface
    region. The paper describes the intent as reducing the contribution of points
    close to the target surface, which would require the complementary form
    1 - delta. Both polarities were tried here; this one trains stably and is
    what the published formula states, so it is what the code implements. Worth
    confirming with the authors.
    """
    return 1.0 / (1.0 + (f / xi) ** 4)


def loss_eikonal_weighted(f_query, grad_query, xi):
    """Eq. 4: mean delta(f(x)) |‖grad f(x)‖ - 1|.

    The weight is detached. Left attached, the network can reduce the loss by
    inflating |f| to shrink delta rather than by satisfying the constraint.
    """
    w = attenuation(f_query, xi).detach().reshape(-1)
    return (w * (torch.linalg.norm(grad_query, dim=-1) - 1.0).abs()).mean()


def compute_loss(net, pts_surface, q1, q2, normals, pts_uniform, it, total, cfg):
    """Eq. 5. Returns (total_loss, component_dict)."""
    xs, a, b, xu = map(make_input, (pts_surface, q1, q2, pts_uniform))

    f_s = net(xs)
    f_a, f_b = net(a), net(b)
    g_a, g_b = gradient(f_a, a), gradient(f_b, b)
    f_u = net(xu)
    g_u = gradient(f_u, xu)

    xi = xi_schedule(it, total, cfg["xi_start"], cfg["xi_end"])

    L_dist = loss_dist(f_s)
    L_pos = loss_positive(torch.cat([f_a, f_b, f_u], 0))
    L_nrm = loss_normal(g_a, g_b, normals)
    # evaluated on Q (a proxy for the target geometry) and on uniform samples
    L_eik = 0.5 * (loss_eikonal_weighted(torch.cat([f_a, f_b], 0),
                                         torch.cat([g_a, g_b], 0), xi)
                   + loss_eikonal_weighted(f_u, g_u, xi))

    total_loss = (cfg["lam_dist"] * L_dist + cfg["lam_positive"] * L_pos
                  + cfg["lam_normal"] * L_nrm + cfg["lam_eikonal"] * L_eik)
    return total_loss, {"dist": L_dist.item(), "positive": L_pos.item(),
                        "normal": L_nrm.item(), "eikonal": L_eik.item(), "xi": xi}
