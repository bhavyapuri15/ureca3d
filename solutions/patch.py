"""REFERENCE SOLUTIONS -- read only after you have tried yourself.

Why this file exists: you have three days and a professor waiting for pictures.
If you get stuck on one term at hour 8, you should not lose the whole
deliverable. Run with `--solutions` to swap in working implementations, confirm
the pipeline produces a mesh, then go back and fix your own version.

How it works: `apply()` monkeypatches the functions into the modules, so nothing
in your own files is overwritten.

If you present results produced with --solutions, say so. Prof. He will respect
"I implemented six of the seven MPF terms, the seventh came from my reference
implementation" far more than a silent gap.
"""
import math

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------- diffops
def gradient(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                               create_graph=True, retain_graph=True)[0]


def hessian(y, x, grad_y=None):
    g = grad_y if grad_y is not None else gradient(y, x)
    rows = []
    for j in range(x.shape[-1]):
        gj = g[:, j]
        rows.append(torch.autograd.grad(gj, x, grad_outputs=torch.ones_like(gj),
                                        create_graph=True, retain_graph=True)[0])
    return torch.stack(rows, dim=1)


def det3x3(H):
    a, b, c = H[:, 0, 0], H[:, 0, 1], H[:, 0, 2]
    d, e, f = H[:, 1, 0], H[:, 1, 1], H[:, 1, 2]
    g, h, i = H[:, 2, 0], H[:, 2, 1], H[:, 2, 2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def laplacian(y, x, grad_y=None):
    g = grad_y if grad_y is not None else gradient(y, x)
    out = 0.0
    for j in range(x.shape[-1]):
        gj = g[:, j]
        hj = torch.autograd.grad(gj, x, grad_outputs=torch.ones_like(gj),
                                 create_graph=True, retain_graph=True)[0]
        out = out + hj[:, j]
    return out.unsqueeze(-1)


# --------------------------------------------------------------------- NSH
def tau_schedule(it, total):
    t = it / max(total - 1, 1)
    if t < 0.2:
        return 1.0
    if t < 0.4:
        u = (t - 0.2) / 0.2
        return 1.0 + u * (3e-4 - 1.0)
    u = (t - 0.4) / 0.6
    return 3e-4 + u * (3e-5 - 3e-4)


def nsh_manifold(f_s):
    return f_s.abs().mean()


def nsh_nonmanifold(f_far, alpha=100.0):
    return torch.exp(-alpha * f_far.abs()).mean()


def nsh_eikonal_relaxed(g_s, sigma_min=0.8):
    n = torch.linalg.norm(g_s, dim=-1)
    return F.relu(-(n - sigma_min)).mean()


def nsh_singular_hessian(H):
    return det3x3(H).abs().mean()


# ------------------------------------------------------------------- DEUDF
def xi_schedule(it, total, xi_start=0.01, xi_end=0.002):
    t = it / max(total - 1, 1)
    return xi_end + 0.5 * (xi_start - xi_end) * (1 + math.cos(math.pi * t))


def deudf_dist(f_s):
    return f_s.abs().mean()


def deudf_positive(f_q):
    return torch.exp(torch.clamp(-100.0 * f_q, max=20.0)).mean()


def deudf_normal(g1, g2, n, eps=1e-8):
    n = n / (n.norm(dim=-1, keepdim=True) + eps)
    u1 = g1 / (g1.norm(dim=-1, keepdim=True) + eps)
    u2 = g2 / (g2.norm(dim=-1, keepdim=True) + eps)
    c1 = (u1 * n).sum(-1)
    c2 = (u2 * n).sum(-1)
    return ((1.0 - c1) + (1.0 + c2)).mean()


def attenuation(f, xi):
    return 1.0 / (1.0 + (f / xi) ** 4)


def deudf_eikonal_weighted(f_q, g_q, xi):
    w = attenuation(f_q, xi).detach().reshape(-1)
    res = (torch.linalg.norm(g_q, dim=-1) - 1.0).abs()
    return (w * res).mean()


# --------------------------------------------------------------------- MPF
def compose(r, theta, beta):
    P = torch.tanh(beta * theta)
    phi = r * P + theta
    return phi, P


def mpf_zero(r_s, phi_s):
    return (r_s + phi_s.abs()).mean()


def mpf_eik_r(g_r):
    return (torch.linalg.norm(g_r, dim=-1) - 1.0).abs().mean()


def mpf_eik_phi(g_phi, g_th):
    return ((torch.linalg.norm(g_phi, dim=-1) - 1.0).abs().mean()
            + (torch.linalg.norm(g_th, dim=-1) - 1.0).abs().mean())


def mpf_align(r_b, g_r, g_th, alpha=10.0, eps=1e-8):
    nr = g_r / (g_r.norm(dim=-1, keepdim=True) + eps)
    nt = g_th / (g_th.norm(dim=-1, keepdim=True) + eps)
    dot = (nr * nt).sum(-1, keepdim=True).abs()
    w = torch.exp(-alpha * r_b)
    return (w * (1.0 - dot)).mean()


def mpf_lap(lap):
    return lap.abs().mean()


def mpf_phase(P_far):
    return ((1.0 - P_far.abs()) ** 2).mean()


def mpf_far(r_f, phi_f, alpha=100.0):
    return torch.exp(torch.clamp(-alpha * (r_f + phi_f.abs()), max=20.0)).mean()


def mpf_normal_align(g_phi, n, eps=1e-8):
    n = n / (n.norm(dim=-1, keepdim=True) + eps)
    u = g_phi / (g_phi.norm(dim=-1, keepdim=True) + eps)
    return (1.0 - (u * n).sum(-1).abs()).mean()


# --------------------------------------------------------------------------
def apply(which=("diffops", "nsh", "deudf", "mpf")):
    """Monkeypatch reference implementations into the real modules."""
    if "diffops" in which:
        import common.diffops as d
        d.gradient, d.hessian, d.det3x3, d.laplacian = gradient, hessian, det3x3, laplacian
    if "nsh" in which:
        import methods.nsh as m
        m.gradient, m.hessian, m.det3x3 = gradient, hessian, det3x3
        m.tau_schedule = tau_schedule
        m.loss_manifold = nsh_manifold
        m.loss_nonmanifold = nsh_nonmanifold
        m.loss_eikonal_relaxed = nsh_eikonal_relaxed
        m.loss_singular_hessian = nsh_singular_hessian
    if "deudf" in which:
        import methods.deudf as m
        m.gradient = gradient
        m.xi_schedule = xi_schedule
        m.loss_dist = deudf_dist
        m.loss_positive = deudf_positive
        m.loss_normal = deudf_normal
        m.attenuation = attenuation
        m.loss_eikonal_weighted = deudf_eikonal_weighted
    if "mpf" in which:
        import methods.mpf as m
        m.gradient, m.laplacian = gradient, laplacian
        m.compose = compose
        m.loss_zero = mpf_zero
        m.loss_eik_r = mpf_eik_r
        m.loss_eik_phi = mpf_eik_phi
        m.loss_align = mpf_align
        m.loss_lap = mpf_lap
        m.loss_phase = mpf_phase
        m.loss_far = mpf_far
        m.loss_normal_align = mpf_normal_align
