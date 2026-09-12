"""NSH -- Neural-Singular-Hessian (Wang et al., ACM TOG 42(6), 2023).

Fits a signed distance function to an unoriented point cloud.

Idea. "f = 0 on the points" plus "||grad f|| = 1" does not determine f: many
wiggly fields satisfy both, and they appear as ghost geometry floating in free
space. Earlier work suppressed this with a smoothness penalty (Laplacian or
Hessian energy), which also erases genuine detail.

NSH instead enforces a property the true SDF provably has. For a point x near
the surface with closest surface point x', the Hessian of the true SDF at x has
two eigenvectors aligned with the principal curvature directions at x' and a
third aligned with the surface normal -- whose eigenvalue is exactly zero, since
the SDF varies linearly along the normal. A zero eigenvalue makes the matrix
singular, so det(H) = 0 throughout the thin shell around the surface.

This is a strictly weaker constraint than smoothness: Hessian energy ||H||^2 = 0
forces all nine entries to zero and reduces f to an affine function, whereas
det(H) = 0 removes a single degree of freedom and leaves curvature intact. By
Morse theory it also removes critical points in pairs, eliminating the spurious
ones responsible for ghost geometry while preserving those the topology requires.
"""
import torch
import torch.nn.functional as F

from common.diffops import gradient, hessian, det3x3, make_input


# Paper Sec. 5.1 and Eq. 13
DEFAULTS = dict(
    lam_manifold=7000.0,
    lam_nonmanifold=600.0,
    lam_eikonal=50.0,
    lam_singularH=1.0,     # the paper folds this into tau
    alpha=100.0,
    sigma_min=0.8,         # relaxed Eikonal floor; their Table 6 sweeps this
    lr=5e-5,
    iters=10000,
    batch=15000,
    omega_0=30.0,
    n_layers=6,            # 4 hidden layers + input + output
)


def tau_schedule(it, total):
    """Annealing factor tau of Eq. 13.

    Paper: tau stays at 1 for the first 20% of iterations, decreases linearly to
    3e-4 over 20%-40%, and reaches 3e-5 at termination.

    This schedule is the method's coarse-to-fine behaviour. While tau is large
    the singular-Hessian term dominates and the field snaps to a crude shape with
    the correct topology; as it decays the data terms recover detail. Holding tau
    constant gives either an over-smoothed result or ghost geometry, depending on
    the value.
    """
    t = it / max(total - 1, 1)
    if t < 0.2:
        return 1.0
    if t < 0.4:
        return 1.0 + ((t - 0.2) / 0.2) * (3e-4 - 1.0)
    return 3e-4 + ((t - 0.4) / 0.6) * (3e-5 - 3e-4)


def loss_manifold(f_surface):
    """Eq. 2: mean |f(p)| over the input points. L1 rather than L2, so a few
    badly fit points are not averaged away."""
    return f_surface.abs().mean()


def loss_nonmanifold(f_far, alpha=100.0):
    """Eq. 3: mean exp(-alpha |f(x)|) over Q_far.

    Away from the data the field must not be near zero. Without this the network
    can satisfy L_manifold perfectly by outputting zero everywhere. The exponential
    is ~1 at f = 0 and negligible once |f| > 5/alpha, so it stops constraining
    points already comfortably far from the level set.
    """
    return torch.exp(-alpha * f_far.abs()).mean()


def loss_eikonal_relaxed(grad_surface, sigma_min=0.8):
    """Eq. 12: mean ReLU(-(||grad f(p)|| - sigma_min)).

    A one-sided hinge -- zero once the gradient norm reaches sigma_min -- rather
    than the usual two-sided |‖∇f‖ - 1|. Surface extraction needs a
    well-conditioned zero crossing, not a metrically exact distance, and relaxing
    the constraint frees capacity for the other terms.

    Enforced only on the input points (paper: "the condition is enforced
    exclusively for points belonging to the input point cloud").
    """
    return F.relu(-(torch.linalg.norm(grad_surface, dim=-1) - sigma_min)).mean()


def loss_singular_hessian(H_near):
    """Eq. 9: mean |det H(x)| over Q_near.

    Evaluated on near-surface samples only. The zero-determinant property holds
    in the thin shell around the surface and is false in free space, so enforcing
    it everywhere would fight the other terms.
    """
    return det3x3(H_near).abs().mean()


def compute_loss(net, pts_surface, pts_near, pts_far, it, total, cfg):
    """Eq. 13. Returns (total_loss, component_dict).

    Each term uses a different point set, and that is load-bearing -- see the
    per-term docstrings above.
    """
    xs, xn, xf = map(make_input, (pts_surface, pts_near, pts_far))

    f_s = net(xs)
    g_s = gradient(f_s, xs)

    f_n = net(xn)
    g_n = gradient(f_n, xn)
    H_n = hessian(f_n, xn, grad_y=g_n)

    f_f = net(xf)

    L_man = loss_manifold(f_s)
    L_non = loss_nonmanifold(f_f, cfg["alpha"])
    L_eik = loss_eikonal_relaxed(g_s, cfg["sigma_min"])
    L_sh = loss_singular_hessian(H_n)
    tau = tau_schedule(it, total)

    total_loss = (cfg["lam_manifold"] * L_man
                  + cfg["lam_nonmanifold"] * L_non
                  + cfg["lam_eikonal"] * L_eik
                  + cfg["lam_singularH"] * tau * L_sh)
    return total_loss, {"manifold": L_man.item(), "nonmanifold": L_non.item(),
                        "eikonal": L_eik.item(), "singularH": L_sh.item(),
                        "tau": tau}
