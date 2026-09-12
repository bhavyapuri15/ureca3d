"""MPF -- Metric-Phase Fields (Kong, Chen, Zong, Hou, Hou, Wang, He; ICML 2026).

A signed distance function asks one scalar field to report proximity to the
surface and to carry a globally consistent sign. Those two demands are compatible
on a watertight manifold. On a thin sheet, an open boundary or a non-manifold
junction the sign is ill-defined, and forcing a single field to satisfy an
ill-posed constraint makes the problem ill-conditioned: thickness inflates,
sheets are sealed, nearby layers merge. A UDF avoids this by discarding the sign,
at the cost of a cusp at the zero set and no sign change for Marching Cubes.

MPF decouples the two roles across two heads on a shared backbone:

    r(x) >= 0   metric field, UDF-like, proximity only            (Eq. 16)
    theta(x)    phase field, carries sign where sign is meaningful (Eq. 17)
    P(x) = tanh(beta theta(x)) in (-1,1), beta learnable           (Eq. 2)
    phi(x) = r(x) P(x) + theta(x)                                  (Eq. 3)

Two properties justify the composition.

Proposition 3.1 (sign consistency). r >= 0 and beta > 0 give sign(P) = sign(theta),
hence sign(phi) = sign(theta) and phi = 0 iff theta = 0. The phase field alone
determines the surface location; the metric field shapes phi away from it but
cannot move it.

Proposition 3.2 (gradient agreement). Where r(p) = 0 and phi(p) = 0, differentiating
Eq. 3 gives grad phi(p) = grad theta(p). The composite's normal at the surface is
the phase field's normal -- smooth and well defined, unlike a UDF's cusp. The
additive theta residual is what keeps a gradient path to theta open where the
gated term goes inactive.

Consequence: phi genuinely changes sign, so the surface extracts with standard
Marching Cubes at level 0. Around a thin sheet phi takes opposite signs on the two
faces, separated by a narrow gap, so the sheet survives rather than being filled.
"""
import torch

from common.diffops import gradient, laplacian, safe_normalize, make_input


# Paper Sec. 5, "Implementation details"
DEFAULTS = dict(
    lam_zero=1.0,
    lam_align=0.7,
    lam_eik_r=0.007,
    lam_eik_phi=0.0004,
    lam_lap=0.002,
    lam_phase=0.1,
    lam_far=0.1,
    lam_normal=0.15,     # only when PCA normals are reliable, else 0
    alpha_align=10.0,
    alpha_far=100.0,
    delta=0.05,          # near-surface band width
    beta_init=50.0,
    lr=1e-4,
    iters=10000,
    batch=15000,
    align_start=3000,
    omega_0=30.0,
    sphere_init_iters=400,   # see sphere_init() below; 0 disables
)
# Sec. 5 names six loss weights -- lam_align, lam_eik-r, lam_eik-phi, lam_lap,
# lam_phase, lam_far -- and lists five values: (0.7, 0.007, 0.0004, 0.002, 0.1).
# One value is missing from the published text; lam_far = 0.1 is assumed here.


def sphere_init(net, iters=400, radius=0.5, bound=1.0, lr=1e-3, device="cpu",
                verbose=True):
    """Geometric initialisation of the phase head: pre-fit theta to a sphere SDF.

    Why this is needed. Nothing in the loss set breaks the sign symmetry of
    theta. L_zero drives theta to 0 on the data; L_phase drives |P| to 1 and
    L_far drives |phi| away from 0, but both are sign-agnostic. A field that is
    negative everywhere therefore satisfies every term reasonably well while
    never crossing zero -- and Marching Cubes at level 0 returns nothing. This is
    not a hypothetical: without this step, MPF on the cube-plus-sheet input
    converges to phi in [-0.034, -0.014] and extracts an empty mesh.

    The fix is the geometric initialisation standard in this literature (SAL,
    IGR, DiGS): start theta close to the SDF of a sphere, so it is negative
    inside and positive outside before the data terms take over. The paper's own
    optional negative phase-head bias serves the same purpose less strongly.

    Runs in a few seconds and is discarded from the optimiser state afterwards.
    """
    import torch as _t
    opt = _t.optim.Adam(net.parameters(), lr=lr)
    for i in range(iters):
        x = (_t.rand(4096, 3, device=device) * 2 - 1) * bound
        target = x.norm(dim=-1, keepdim=True) - radius
        feat = net.backbone(x)
        theta = net.head_t(feat)
        loss = ((theta - target) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if verbose:
        print(f"  [mpf] sphere init done, residual {loss.item():.5f}")
    return net


def compose(r, theta, beta):
    """Eq. 2 and Eq. 3. Returns (phi, P), both (N,1).

    beta is a tensor so that it remains differentiable and is learned.
    """
    P = torch.tanh(beta * theta)
    return r * P + theta, P


def loss_zero(r_surface, phi_surface):
    """Eq. 7: mean over input points of r(p) + |phi(p)|.

    phi(p) = 0 places the extracted surface on the data. r(p) = 0 additionally
    anchors the metric branch -- the paper calls this preventing metric drift.
    Proposition 3.2 assumes r(p) = 0, so without it the surface normals degrade.
    r is non-negative by construction and needs no absolute value; phi is signed.
    """
    return (r_surface + phi_surface.abs()).mean()


def loss_eik_r(grad_r_band):
    """Eq. 10: mean |‖grad r‖ - 1| on the band N_delta = {0 < r < delta}.

    Points with r ~ 0 are excluded to avoid the cusp where a true UDF is not
    differentiable; far points are excluded to avoid the medial axis, where the
    unit-gradient property also fails.
    """
    return (torch.linalg.norm(grad_r_band, dim=-1) - 1.0).abs().mean()


def loss_eik_phi(grad_phi_surface, grad_theta_surface):
    """Eq. 11, the topological Eikonal constraint, on the input points.

    By Proposition 3.2 the two gradients coincide at convergence, so constraining
    both is redundant in the limit but stabilises optimisation before it.
    """
    return ((torch.linalg.norm(grad_phi_surface, dim=-1) - 1.0).abs().mean()
            + (torch.linalg.norm(grad_theta_surface, dim=-1) - 1.0).abs().mean())


def loss_align(r_band, grad_r_band, grad_theta_band, alpha=10.0, eps=1e-8):
    """Eq. 8: mean w(r) (1 - |<n_r, n_theta>|), with w(r) = exp(-alpha r).

    Near the surface the metric and phase fields should vary along a common
    direction, so that the phase transition happens across the surface rather
    than at an angle to it. w(r) concentrates the constraint into a narrow band.
    The absolute value makes it orientation-invariant -- whether theta increases
    inward or outward is irrelevant, which is what unoriented input requires.

    Activated only after `align_start` iterations; earlier the gradients are noise
    and aligning them locks in a poor configuration.
    """
    nr = safe_normalize(grad_r_band, eps)
    nt = safe_normalize(grad_theta_band, eps)
    dot = (nr * nt).sum(-1, keepdim=True).abs()
    return (torch.exp(-alpha * r_band) * (1.0 - dot)).mean()


def loss_lap(phi_lap_samples):
    """Eq. 12: mean |Laplacian phi| on ambient samples.

    Encourages phi to be approximately harmonic in free space, suppressing
    oscillation between the data points. Applied to the composite field and away
    from the surface, so it regularises global behaviour without smoothing thin
    structures.
    """
    return phi_lap_samples.abs().mean()


def loss_phase(P_far):
    """Eq. 13: mean (1 - |P|)^2 on samples outside the near-surface band.

    Far from the surface the soft sign should saturate to +-1. If P lingers near
    zero in free space the transition band thickens and the surface blurs.
    """
    return ((1.0 - P_far.abs()) ** 2).mean()


def loss_far(r_far, phi_far, alpha=100.0):
    """Eq. 14: mean exp(-alpha (r + |phi|)) outside the band.

    Repulsion against spurious zero level sets away from the data. It is
    significant only when both r and |phi| are near zero -- the signature of a
    ghost sheet -- and decays to nothing otherwise. Exponent clamped as in
    DEUDF's L_positive.
    """
    return torch.exp(torch.clamp(-alpha * (r_far + phi_far.abs()), max=20.0)).mean()


def loss_normal_align(grad_phi_surface, normals, eps=1e-8):
    """Eq. 15, optional: mean 1 - |<grad phi / ‖grad phi‖, n>|.

    Enabled only where PCA normals are reliable (dense, low-noise input).
    Orientation-invariant for the same reason as L_align.
    """
    n = safe_normalize(normals, eps)
    u = safe_normalize(grad_phi_surface, eps)
    return (1.0 - (u * n).sum(-1).abs()).mean()


def compute_loss(net, pts_surface, pts_band, pts_far, pts_ambient,
                 normals, it, cfg):
    """Eq. 18. Returns (total_loss, component_dict).

    Four point sets, each with a distinct role:
        pts_surface  input cloud X          -> L_zero, L_eik^phi, L_normal
        pts_band     N_delta, 0 < d < delta -> L_eik^r, L_align
        pts_far      outside the band       -> L_phase, L_far
        pts_ambient  uniform over Omega     -> L_lap
    """
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

    zero = torch.zeros((), device=phi_s.device)
    L = {
        "zero": loss_zero(r_s, phi_s),
        "eik_r": loss_eik_r(g_r_b),
        "eik_phi": loss_eik_phi(g_phi_s, g_th_s),
        "lap": loss_lap(lap_a),
        "phase": loss_phase(P_f),
        "far": loss_far(r_f, phi_f, cfg["alpha_far"]),
        "align": (loss_align(r_b, g_r_b, g_th_b, cfg["alpha_align"])
                  if it >= cfg["align_start"] else zero),
        "normal": (loss_normal_align(g_phi_s, normals)
                   if (cfg["lam_normal"] > 0 and normals is not None) else zero),
    }

    total_loss = (cfg["lam_zero"] * L["zero"] + cfg["lam_align"] * L["align"]
                  + cfg["lam_eik_r"] * L["eik_r"] + cfg["lam_eik_phi"] * L["eik_phi"]
                  + cfg["lam_lap"] * L["lap"] + cfg["lam_far"] * L["far"]
                  + cfg["lam_phase"] * L["phase"] + cfg["lam_normal"] * L["normal"])
    log = {k: (v.item() if torch.is_tensor(v) else v) for k, v in L.items()}
    log["beta"] = net.beta.item()
    return total_loss, log
