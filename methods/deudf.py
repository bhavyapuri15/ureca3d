"""DEUDF -- Details Enhanced UDF learning (Xu et al., AAAI 2025).

=========================== THE IDEA IN ONE PARAGRAPH ==========================
An SDF needs a consistent inside/outside, so it cannot represent an open sheet
or a shape with an internal wall. A UDF (unsigned distance field) drops the sign
and just stores distance-to-surface, so it handles anything -- but UDFs are hard
to learn well, for three concrete reasons, and DEUDF fixes each one:

(1) NON-NEGATIVITY IS ENFORCED BADLY. Everyone puts abs() or softplus() on the
    network output to guarantee u >= 0.
      - abs() folds the function, creating a "W"-shaped profile with several
        local minima near the surface -- so the zero set is ambiguous.
      - softplus() is nearly flat for negative inputs, so the gradient VANISHES
        over a wide band around the surface and training stalls there.
    DEUDF's fix: use a plain unconditioned MLP -- no abs, no softplus -- and
    accept that values may go slightly negative. The learned field then has a
    clean "V" shape (their Fig. 1g). They still call it a UDF; the surface is
    the *local minimum of |f|*, not exactly the zero set.
    Instead of a hard constraint they use a soft one: L_positive.

(2) NORMAL DIRECTIONS ARE UNUSED. UDF gradients vanish AT the surface, so you
    cannot constrain them there. DEUDF instead samples PAIRS of points
    q1 = p + lam*n and q2 = p - lam*n straddling the surface along the (PCA,
    unoriented) normal, and constrains the gradient at those. Crucially the UDF
    gradient points AWAY from the surface on both sides, so it agrees with +n on
    one side and -n on the other -- hence the (-1)^k factor in Eq. 3.

(3) THE EIKONAL CONSTRAINT IS APPLIED UNIFORMLY. ||grad u|| = 1 is true for a
    UDF away from the surface, but near the surface the gradient genuinely
    shrinks/flips, so forcing it there drags the surface away from the data.
    DEUDF weights the Eikonal term by delta(f) = (1 + (f/xi)^4)^-1: ~1 when
    |f| >> xi, ~0 when |f| < xi. So the constraint switches itself off near the
    surface. xi anneals 0.01 -> 0.002, tightening the exempt band over training.

================================= WHAT TO DO ==================================
Four loss terms again. The subtlety is in (2) and (3) -- read the docstrings.
"""
import torch

from common.diffops import gradient


# Paper "UDF learning" paragraph
DEFAULTS = dict(
    lam_dist=400.0,       # lambda_1
    lam_positive=50.0,    # lambda_2
    lam_normal=40.0,      # lambda_3
    lam_eikonal=10.0,     # lambda_4
    xi_start=0.01,        # attenuation bandwidth, start
    xi_end=0.002,         # ... and end
    lam_max=0.003,        # displacement for the q1/q2 pair sampling
    lr=5e-5,              # Adam, cosine-annealed to 0
    iters=10000,
    batch=15000,
    omega_0=60.0,         # 60 for clean clouds, 30 for noisy (their footnote 1)
    n_layers=5,           # "5-layer SIREN, 256 units per layer"
    knn_pca=16,
)


def xi_schedule(it, total, xi_start=0.01, xi_end=0.002):
    """Anneal the Eikonal attenuation bandwidth. Paper: "initially set xi to 0.01
    and gradually decrease xi to 0.002 over the course of the learning process,
    following the learning rate" -- i.e. cosine, same shape as the LR schedule.

    TODO(you): cosine interpolation from xi_start to xi_end.
        t = it / total ;  xi = xi_end + 0.5*(xi_start-xi_end)*(1+cos(pi*t))
    """
    raise NotImplementedError


def loss_dist(f_surface):
    """Eq. 1:  L_dist = sum |f(p_i)| over input points.
    Same as NSH's L_manifold. TODO(you): one line (use mean, not sum -- keeps
    the weights scale-independent of batch size; note this in your report).
    """
    raise NotImplementedError


def loss_positive(f_query):
    """Eq. 2:  L_positive = sum exp(-100 * f(x)).

    *** LOOK AT THE SIGN. There is NO absolute value. ***  This is not a typo and
    the paper spends a whole Remark on it. Compare:
      - NeuralUDF-style exp(-100|f|) punishes any value near zero, in both
        directions, which shrinks the set of near-zero points and destroys small
        disconnected components.
      - DEUDF's exp(-100 f) punishes NEGATIVE values heavily (exp of a large
        positive number) and is ~0 for positive f. So it is a soft
        non-negativity constraint: "please stay positive", with room to dip
        slightly negative near the surface. That dip is what gives the clean V.

    Numerical warning: if f goes very negative, exp(-100f) overflows to inf and
    your training dies with NaN. Clamp the exponent, e.g.
        torch.exp(torch.clamp(-100.0 * f, max=20.0))
    and say you did so -- it is an implementation detail the paper omits.

    Args: f_query (M,1) over Q (the paired samples) and/or uniform samples
    TODO(you).
    """
    raise NotImplementedError


def loss_normal(grad_q1, grad_q2, normals, eps=1e-8):
    """Eq. 3:  L_normal = sum_{k in {1,2}} ( 1 + (-1)^k * <grad f(q^k), n> /
                                             (||grad f(q^k)|| * ||n||) ).

    Unpack it. q^1 = p + lam*n is on the +n side, q^2 = p - lam*n on the -n side.
    For a true UDF the gradient points away from the surface, so
        grad f(q^1) ~ +n   ->  cos = +1
        grad f(q^2) ~ -n   ->  cos = -1
    The factor (-1)^k is -1 for k=1 and +1 for k=2, so both terms become
    (1 - 1) = 0 at the optimum. Term k=1 is  1 - cos1 ; term k=2 is  1 + cos2.

    Why this is safe with unoriented PCA normals: flip n -> -n and q1/q2 swap
    roles, so the sum is unchanged. Verify that invariance in a test -- it is
    the property that lets you use PCA normals at all.

    Args:  grad_q1, grad_q2 (M,3); normals (M,3)
    TODO(you): normalise both vectors, take the dot products, build the two
    terms. Do NOT use abs() here -- the pairing already handles orientation, and
    abs() would throw away the information that distinguishes the two sides.
    """
    raise NotImplementedError


def attenuation(f, xi):
    """delta(d) = (1 + (d/xi)^4)^-1  (from IDF, Wang et al. 2022).

    A smooth U-shaped gate: ~0 for |d| << xi, ~1 for |d| >> xi. This is the
    "adaptively weighted" part of the adaptively-weighted Eikonal constraint.
    Plot it once for xi=0.01 -- it belongs in your slides, it explains the whole
    section in one picture.

    Args: f (M,1) field value; xi scalar
    TODO(you): one line. Careful: f may be negative, and (d/xi)^4 handles that
    fine, but use f not |f| and the 4th power keeps it even anyway.
    """
    raise NotImplementedError


def loss_eikonal_weighted(f_query, grad_query, xi):
    """Eq. 4:  L_eikonal = sum delta(f(x)) * | ||grad f(x)||_2 - 1 |.

    Note it is |.| around the whole bracket (L1 on the Eikonal residual), and
    the weight delta depends on the *field value*, not on the distance to the
    cloud. So the network decides where the constraint applies -- that is what
    makes it adaptive.

    Evaluated on Q (the paired samples, "a proxy of the target geometry") AND on
    uniformly random points in Omega.
    TODO(you): detach xi? No -- xi is a scalar schedule, not a parameter.
    Should you detach the delta weight? The paper does not say. Try both; not
    detaching lets the network reduce the loss by inflating |f|, which is a real
    failure mode. Recommend detaching and reporting it.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------

def compute_loss(net, pts_surface, q1, q2, normals, pts_uniform, it, total, cfg):
    """Assembles Eq. 5. Complete -- calls your functions above."""
    from common.diffops import make_input

    xs = make_input(pts_surface)
    a = make_input(q1)
    b = make_input(q2)
    xu = make_input(pts_uniform)

    f_s = net(xs)
    f_a, f_b = net(a), net(b)
    g_a, g_b = gradient(f_a, a), gradient(f_b, b)
    f_u = net(xu)
    g_u = gradient(f_u, xu)

    xi = xi_schedule(it, total, cfg["xi_start"], cfg["xi_end"])

    L_dist = loss_dist(f_s)
    L_pos = loss_positive(torch.cat([f_a, f_b, f_u], 0))
    L_nrm = loss_normal(g_a, g_b, normals)
    L_eik = 0.5 * (loss_eikonal_weighted(torch.cat([f_a, f_b], 0),
                                         torch.cat([g_a, g_b], 0), xi)
                   + loss_eikonal_weighted(f_u, g_u, xi))

    total_loss = (cfg["lam_dist"] * L_dist + cfg["lam_positive"] * L_pos
                  + cfg["lam_normal"] * L_nrm + cfg["lam_eikonal"] * L_eik)
    return total_loss, {"dist": L_dist.item(), "positive": L_pos.item(),
                        "normal": L_nrm.item(), "eikonal": L_eik.item(), "xi": xi}
