"""MPF -- Metric-Phase Fields (Kong, Chen, Zong, Hou, Hou, Wang, He; ICML 2026).

This is Prof. He's own paper. He will know it better than anyone, so be precise.

=========================== THE IDEA IN ONE PARAGRAPH ==========================
An SDF asks one scalar function to do two jobs at once: report the DISTANCE to
the surface, and carry a globally consistent SIGN. On a watertight blob that is
fine. On a thin sheet, an open boundary, or a non-manifold junction, the sign is
ill-defined -- and forcing one function to satisfy both jobs makes the problem
ill-conditioned: you get inflated thickness, sheets that get sealed shut, layers
that merge. A UDF drops the sign entirely, which fixes representation but breaks
extraction (no sign change => Marching Cubes has nothing to bracket, and the
field is non-differentiable at its zero set, a cusp).

MPF decouples the two jobs into two networks heads:
    r(x)      >= 0   the METRIC field. UDF-like. Just distance. (Eq. 16, softplus)
    theta(x)  in R   the PHASE field. Carries sign preference where one is
                     meaningful. Smooth, no hard switching. (Eq. 17)
From theta they build a bounded soft sign indicator
    P(x) = tanh(beta * theta(x))          in (-1, 1)         (Eq. 2)
with beta LEARNABLE -- the network itself decides how sharp the inside/outside
transition should be for this shape.

And then the composite ("gated-metric formulation with residual phase injection"):
    phi(x) = r(x) * P(x) + theta(x)                            (Eq. 3)

Why this exact form? Two propositions in the paper:
 * Prop 3.1 (sign consistency): since r >= 0 and beta > 0, sign(phi) = sign(theta)
   and phi(x) = 0  <=>  theta(x) = 0. So the zero set of phi is the zero set of
   theta -- the phase field alone decides where the surface is, and the metric
   field cannot move it. It only shapes phi AWAY from the interface.
 * Prop 3.2 (gradient agreement): at a training point where r(p)=0 and phi(p)=0,
   grad phi(p) = grad theta(p). So the composite's normal at the surface is the
   phase field's normal -- well-defined and stable, unlike a UDF's cusp. The
   "+theta" residual term is what guarantees a live gradient path to theta even
   where the gated term r*P goes inactive.

Net effect: phi is signed and smooth enough that you extract the mesh with plain
Marching Cubes at level 0 -- no DCUDF, no double-cover post-processing -- while
around thin sheets phi forms a narrow two-sided shell (a UDF-like double cover
that still crosses zero), so those survive too.

================================= WHAT TO DO ==================================
compose() first -- it is 3 lines and everything else depends on it. Then the
seven loss terms. This is the biggest of the three; if you run short of time,
implement L_zero, L_eik^phi, L_align and L_lap (the core) and set the others to
zero, and SAY SO in your report. A partial MPF with an honest ablation is worth
more than a complete one you cannot explain.
"""
import torch

from common.diffops import gradient, laplacian, safe_normalize


# Paper Sec. 5 "Implementation details"
DEFAULTS = dict(
    lam_zero=1.0,
    lam_align=0.7,
    lam_eik_r=0.007,
    lam_eik_phi=0.0004,
    lam_lap=0.002,
    lam_phase=0.1,
    lam_far=0.1,      # <-- SEE NOTE BELOW
    lam_normal=0.15,  # only when PCA normals are used, else 0
    alpha_align=10.0,   # w(r) = exp(-alpha r) concentration
    alpha_far=100.0,    # far-field repulsion sharpness (paper: "we set alpha = 100")
    delta=0.05,         # near-surface band width
    beta_init=50.0,
    lr=1e-4,
    iters=10000,
    batch=15000,
    align_start=3000,   # "we activate L_align only after 3,000 iterations"
    omega_0=30.0,
)
# NOTE / open question to raise with Prof. He:
# Sec. 5 lists SIX weight names (lam_align, lam_eik-r, lam_eik-phi, lam_lap,
# lam_phase, lam_far) but only FIVE values: (0.7, 0.007, 0.0004, 0.002, 0.1).
# One value is missing in the published text. Above I assumed the list is
# truncated and reused 0.1 for lam_far. Flag this when you present -- spotting it
# is a genuinely good sign that you read the paper properly, and he can tell you
# the real number in one sentence.


def compose(r, theta, beta):
    """Eq. 2 + Eq. 3.  Returns (phi, P), both (N,1).

        P   = tanh(beta * theta)
        phi = r * P + theta

    TODO(you): three lines. beta is a tensor scalar (net.beta), so this stays
    differentiable w.r.t. beta and it gets learned.

    Sanity checks to convince yourself (and to put in tests/):
      - r >= 0 and beta > 0  =>  sign(phi) == sign(theta)   (Prop. 3.1)
      - theta == 0           =>  P == 0 and phi == 0        (Prop. 3.1)
      - if r == 0 and phi == 0 then d(phi) == d(theta)      (Prop. 3.2)
    """
    raise NotImplementedError("implement compose()")


def loss_zero(r_surface, phi_surface):
    """Eq. 7:  L_zero = sum_{p in X} ( r(p) + |phi(p)| ).

    Two constraints, one term. phi(p)=0 puts the extracted surface on the data.
    r(p)=0 additionally anchors the METRIC branch to the data -- the paper calls
    this preventing "metric drift". Without it, r could settle at some nonzero
    offset on the surface and Prop. 3.2 (which assumes r(p)=0) would not hold,
    so your surface normals would degrade.

    Note r >= 0 by construction so it needs no abs; phi is signed so it does.
    TODO(you): one line.
    """
    raise NotImplementedError


def loss_eik_r(grad_r_band):
    """Eq. 10:  L^r_eik = E_{x in N_delta} [ | ||grad r||_2 - 1 | ].

    Standard Eikonal, but ONLY on the band N_delta = {0 < r(x) < delta}
    (Eq. 9) -- i.e. near the surface but NOT on it. Excluding r ~ 0 avoids the
    cusp where a true UDF is not differentiable; excluding far points avoids the
    medial axis where it also breaks down.

    Args: grad_r_band (M,3).  TODO(you): one line.
    """
    raise NotImplementedError


def loss_eik_phi(grad_phi_surface, grad_theta_surface):
    """Eq. 11:  L^phi_eik = E_{p in X}[ | ||grad phi(p)|| - 1 | ]
                          + E_{p in X}[ | ||grad theta(p)|| - 1 | ].

    Called the "topological Eikonal constraint". Applied ON the input points,
    where by Prop. 3.2 the two gradients coincide -- constraining both is
    redundant in theory but stabilises optimisation before convergence, when
    they do not yet agree.

    TODO(you): two norms, two L1 residuals, add.
    """
    raise NotImplementedError


def loss_align(r_band, grad_r_band, grad_theta_band, alpha=10.0, eps=1e-8):
    """Eq. 8:  L_align = E_x [ w(r(x)) * ( 1 - | <n_r, n_theta> | ) ]
        with  n_r = grad r / (||grad r|| + eps),  n_theta likewise,
              w(r) = exp(-alpha * r).

    Meaning: near the surface the metric and phase fields should vary along the
    SAME direction -- the phase transition should happen across the surface, not
    at some angle to it. w(r) concentrates the constraint into a narrow band
    (it decays as you move away). The ABS on the inner product makes it
    orientation-invariant: we do not care whether theta increases inward or
    outward, only that the two fields agree up to sign. That is exactly what
    "unoriented point cloud" demands.

    Activated only after `align_start` iterations -- early on the gradients are
    noise and aligning them locks in a bad configuration.

    Args: r_band (M,1), grad_r_band (M,3), grad_theta_band (M,3)
    TODO(you): use safe_normalize from common.diffops; keepdim carefully so the
    weight (M,1) multiplies the residual (M,1).
    """
    raise NotImplementedError


def loss_lap(phi_lap_samples):
    """Eq. 12:  L_lap = E_{x in Omega} [ |Laplacian phi(x)| ].

    Borrowed from DiGS. Pushes phi toward being harmonic in free space, i.e. no
    spurious oscillation between the data. Note it is applied to the COMPOSITE
    phi and on AMBIENT samples (not near-surface), so it regularises global
    behaviour without smoothing away thin structures -- that separation is the
    point (compare NSH, where the analogous term must be annealed instead).

    Args: phi_lap_samples (M,1) -- already the Laplacian values
    TODO(you): one line.
    """
    raise NotImplementedError


def loss_phase(P_far):
    """Eq. 13:  L_phase = E_{x in Omega \\ N_delta} [ (1 - |P(x)|)^2 ].

    Far from the surface, the soft sign P = tanh(beta theta) should be saturated
    to +1 or -1, i.e. "definitely inside" or "definitely outside". If P hovers
    near 0 in free space, the transition band thickens and blurs and you get
    fat or ambiguous surfaces. Squared, so it is gentle.

    Args: P_far (M,1) in (-1,1).  TODO(you): one line.
    """
    raise NotImplementedError


def loss_far(r_far, phi_far, alpha=100.0):
    """Eq. 14:  L_far = E_{x in Omega \\ N_delta}[ exp(-alpha * (r(x) + |phi(x)|)) ].

    Repulsion: kills spurious zero level sets away from the data. It fires only
    when BOTH r and |phi| are near zero -- exactly the signature of a ghost
    sheet -- and decays to nothing otherwise, so it does not constrain the field
    where it has no business doing so.

    Same overflow warning as DEUDF's L_positive: clamp the exponent.
    Args: r_far (M,1), phi_far (M,1).  TODO(you).
    """
    raise NotImplementedError


def loss_normal_align(grad_phi_surface, normals, eps=1e-8):
    """Eq. 15 (OPTIONAL):  E_p[ 1 - | <grad phi / ||grad phi||, n(p)> | ].

    Only when PCA normals are reliable (dense, low-noise input). abs() again for
    orientation invariance. Set lam_normal = 0 to disable; report both.
    TODO(you) -- or skip and return a zero tensor if you are short on time.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------

def compute_loss(net, pts_surface, pts_band, pts_far, pts_ambient,
                 normals, it, cfg):
    """Assembles Eq. 18. Complete -- calls your functions above.

    Four different point sets, each with its own role:
        pts_surface : the input cloud X            -> L_zero, L_eik^phi, L_normal
        pts_band    : N_delta, 0 < d < delta       -> L_eik^r, L_align
        pts_far     : Omega \\ N_delta              -> L_phase, L_far
        pts_ambient : uniform over Omega           -> L_lap
    """
    from common.diffops import make_input

    xs, xb, xf, xa = map(make_input, (pts_surface, pts_band, pts_far, pts_ambient))

    phi_s, r_s, th_s, _ = net(xs)
    g_phi_s = gradient(phi_s, xs)
    g_th_s = gradient(th_s, xs)

    phi_b, r_b, th_b, _ = net(xb)
    g_r_b = gradient(r_b, xb)
    g_th_b = gradient(th_b, xb)

    phi_f, r_f, _, P_f = net(xf)

    phi_a, _, _, _ = net(xa)
    lap_a = laplacian(phi_a, xa)

    L = {}
    L["zero"] = loss_zero(r_s, phi_s)
    L["eik_r"] = loss_eik_r(g_r_b)
    L["eik_phi"] = loss_eik_phi(g_phi_s, g_th_s)
    L["lap"] = loss_lap(lap_a)
    L["phase"] = loss_phase(P_f)
    L["far"] = loss_far(r_f, phi_f, cfg["alpha_far"])
    zero = torch.zeros((), device=phi_s.device)
    L["align"] = (loss_align(r_b, g_r_b, g_th_b, cfg["alpha_align"])
                  if it >= cfg["align_start"] else zero)
    L["normal"] = (loss_normal_align(g_phi_s, normals)
                   if (cfg["lam_normal"] > 0 and normals is not None) else zero)

    total_loss = (cfg["lam_zero"] * L["zero"] + cfg["lam_align"] * L["align"]
                  + cfg["lam_eik_r"] * L["eik_r"] + cfg["lam_eik_phi"] * L["eik_phi"]
                  + cfg["lam_lap"] * L["lap"] + cfg["lam_far"] * L["far"]
                  + cfg["lam_phase"] * L["phase"] + cfg["lam_normal"] * L["normal"])
    log = {k: (v.item() if torch.is_tensor(v) else v) for k, v in L.items()}
    log["beta"] = net.beta.item()
    return total_loss, log
