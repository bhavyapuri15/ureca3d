"""Unit tests for YOUR implementations.

    cd ureca3d && python -m pytest tests -q
    python -m pytest tests -q --sol      # same tests against the reference

These check mathematical properties, not exact numbers, so they catch the
mistakes that actually happen: a missing create_graph, a sign flip, a mean over
the wrong axis, abs() in the wrong place. Run them BEFORE you launch a 40-minute
training job.
"""
import math
import sys

import pytest
import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])

# ---------------------------------------------------------------- fixtures
def quadratic_field(n=64):
    """f(x,y,z) = x^2 + 2y^2 + 3z^2  -- analytic grad and Hessian.
    grad = (2x, 4y, 6z);  H = diag(2,4,6);  det H = 48;  lap = 12."""
    x = torch.randn(n, 3, dtype=torch.float64, requires_grad=True)
    y = (x[:, 0:1] ** 2 + 2 * x[:, 1:2] ** 2 + 3 * x[:, 2:3] ** 2)
    return x, y


def sphere_sdf(n=64, r=0.7):
    """A true SDF: f = ||x|| - r. Its Hessian is singular everywhere (det = 0).
    This is THE test for NSH: if your det3x3 + hessian are right, feeding a real
    SDF must give you ~0."""
    x = (torch.randn(n, 3, dtype=torch.float64) * 0.5).requires_grad_(True)
    y = (x.norm(dim=-1, keepdim=True) - r)
    return x, y


# ------------------------------------------------------------------ diffops
def test_gradient_matches_analytic():
    from common.diffops import gradient
    x, y = quadratic_field()
    g = gradient(y, x)
    expect = torch.stack([2 * x[:, 0], 4 * x[:, 1], 6 * x[:, 2]], 1)
    assert torch.allclose(g, expect, atol=1e-8)


def test_gradient_keeps_graph():
    """create_graph=True is not optional -- without it the Hessian silently
    fails and your network never learns from gradient-based terms."""
    from common.diffops import gradient
    x, y = quadratic_field()
    g = gradient(y, x)
    assert g.grad_fn is not None, "gradient() must be differentiable (create_graph=True)"


def test_hessian_is_symmetric_and_correct():
    from common.diffops import hessian
    x, y = quadratic_field()
    H = hessian(y, x)
    assert H.shape == (x.shape[0], 3, 3)
    assert torch.allclose(H, H.transpose(1, 2), atol=1e-8), "Hessian must be symmetric"
    expect = torch.diag(torch.tensor([2., 4., 6.], dtype=torch.float64))
    assert torch.allclose(H, expect.expand_as(H), atol=1e-8)


def test_det3x3():
    from common.diffops import det3x3
    M = torch.randn(32, 3, 3, dtype=torch.float64)
    assert torch.allclose(det3x3(M), torch.linalg.det(M), atol=1e-9)


def test_laplacian():
    from common.diffops import laplacian
    x, y = quadratic_field()
    lap = laplacian(y, x)
    assert torch.allclose(lap, torch.full_like(lap, 12.0), atol=1e-8)


def test_true_sdf_has_singular_hessian():
    """The paper's core claim, as a test. det H of ||x||-r must be ~0."""
    from common.diffops import hessian, det3x3
    x, y = sphere_sdf()
    d = det3x3(hessian(y, x))
    assert d.abs().max() < 1e-6, f"det H of a true SDF should vanish, got {d.abs().max()}"


# ---------------------------------------------------------------------- NSH
def test_tau_schedule_shape():
    from methods.nsh import tau_schedule
    T = 1000
    assert tau_schedule(0, T) == pytest.approx(1.0)
    assert tau_schedule(int(0.1 * T), T) == pytest.approx(1.0)
    mid = tau_schedule(int(0.4 * T), T)
    assert mid == pytest.approx(3e-4, rel=0.1)
    end = tau_schedule(T - 1, T)
    assert end == pytest.approx(3e-5, rel=0.1)
    # monotone non-increasing
    vals = [tau_schedule(i, T) for i in range(0, T, 10)]
    assert all(b <= a + 1e-12 for a, b in zip(vals, vals[1:]))


def test_nsh_relaxed_eikonal_is_one_sided():
    """Zero when the gradient is long enough, positive only when too short."""
    from methods.nsh import loss_eikonal_relaxed
    big = torch.tensor([[1.0, 0, 0], [2.0, 0, 0]])
    assert loss_eikonal_relaxed(big, 0.8).item() == pytest.approx(0.0)
    small = torch.tensor([[0.3, 0, 0]])
    assert loss_eikonal_relaxed(small, 0.8).item() == pytest.approx(0.5, abs=1e-6)


def test_nsh_nonmanifold_decays():
    from methods.nsh import loss_nonmanifold
    near = torch.zeros(8, 1)
    far = torch.full((8, 1), 0.5)
    assert loss_nonmanifold(near).item() == pytest.approx(1.0)
    assert loss_nonmanifold(far).item() < 1e-10
    # must use |f|: a large NEGATIVE value is also far away
    neg = torch.full((8, 1), -0.5)
    assert loss_nonmanifold(neg).item() < 1e-10, "L_non-manifold needs abs()"


# -------------------------------------------------------------------- DEUDF
def test_deudf_positive_is_one_sided():
    """Unlike NSH's term, this one is asymmetric on purpose: it punishes
    negative values and ignores positive ones."""
    from methods.deudf import loss_positive
    pos = torch.full((8, 1), 0.2)
    neg = torch.full((8, 1), -0.02)
    assert loss_positive(pos).item() < 1e-6
    assert loss_positive(neg).item() > 1.0, \
        "loss_positive must NOT use abs() -- see the Remark in the paper"


def test_deudf_attenuation_gate():
    from methods.deudf import attenuation
    xi = 0.01
    assert attenuation(torch.tensor([0.0]), xi).item() == pytest.approx(1.0)
    # NOTE the convention: delta(0)=1 by the formula (1+(d/xi)^4)^-1.
    # The paper wants the constraint OFF near the surface, so check the shape:
    near = attenuation(torch.tensor([0.0]), xi).item()
    far = attenuation(torch.tensor([0.1]), xi).item()
    assert far < near, "attenuation must decay as |f| grows (that is the formula)"


def test_deudf_normal_loss_is_orientation_invariant():
    """Flip the PCA normal AND swap the two straddling points -> same loss.
    This is what makes unoriented normals usable."""
    from methods.deudf import loss_normal
    g1 = torch.randn(16, 3); g2 = torch.randn(16, 3); n = torch.randn(16, 3)
    a = loss_normal(g1, g2, n)
    b = loss_normal(g2, g1, -n)
    assert torch.allclose(a, b, atol=1e-5)


def test_deudf_normal_loss_zero_at_optimum():
    n = torch.nn.functional.normalize(torch.randn(16, 3), dim=-1)
    from methods.deudf import loss_normal
    assert loss_normal(n * 1.0, -n * 1.0, n).item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------- MPF
def test_mpf_compose_sign_consistency():
    """Proposition 3.1: sign(phi) == sign(theta), and phi==0 iff theta==0."""
    from methods.mpf import compose
    r = torch.rand(256, 1) * 2.0
    th = torch.randn(256, 1)
    beta = torch.tensor(50.0)
    phi, P = compose(r, th, beta)
    assert torch.all(torch.sign(phi) == torch.sign(th))
    z = torch.zeros(16, 1)
    phi0, P0 = compose(torch.rand(16, 1), z, beta)
    assert torch.allclose(phi0, z, atol=1e-7)
    assert torch.allclose(P0, z, atol=1e-7)
    # P in (-1,1). With beta=50 tanh saturates to exactly 1.0 in float32, so
    # check boundedness with a gentler beta.
    _, P_soft = compose(r, th, torch.tensor(0.5))
    assert P_soft.abs().max() < 1.0
    assert P.abs().max() <= 1.0


def test_mpf_compose_gradient_agreement():
    """Proposition 3.2: where r=0 and phi=0, grad phi == grad theta."""
    from methods.mpf import compose
    x = torch.randn(64, 3, dtype=torch.float64, requires_grad=True)
    # construct r(x) and theta(x) that both vanish on the plane x0 = 0
    r = (x[:, 0:1] ** 2)
    th = x[:, 0:1] * 1.0
    beta = torch.tensor(50.0, dtype=torch.float64)
    phi, _ = compose(r, th, beta)
    gp = torch.autograd.grad(phi, x, torch.ones_like(phi), create_graph=True)[0]
    gt = torch.autograd.grad(th, x, torch.ones_like(th), create_graph=True)[0]
    on = x[:, 0].abs() < 1e-3
    if on.any():
        assert torch.allclose(gp[on], gt[on], atol=1e-3)


def test_mpf_phase_saturation():
    from methods.mpf import loss_phase
    sat = torch.full((8, 1), 0.999)
    amb = torch.zeros(8, 1)
    assert loss_phase(sat).item() < 1e-4
    assert loss_phase(amb).item() == pytest.approx(1.0)


def test_mpf_align_orientation_invariant():
    from methods.mpf import loss_align
    r = torch.rand(64, 1) * 0.05
    a = torch.randn(64, 3); b = torch.randn(64, 3)
    assert torch.allclose(loss_align(r, a, b), loss_align(r, a, -b), atol=1e-6)
    # perfectly aligned -> zero
    v = torch.randn(64, 3)
    assert loss_align(torch.zeros(64, 1), v, v).item() == pytest.approx(0.0, abs=1e-6)


def test_mpf_far_fires_only_near_zero():
    from methods.mpf import loss_far
    z = torch.zeros(8, 1)
    big = torch.full((8, 1), 0.3)
    assert loss_far(z, z).item() == pytest.approx(1.0)
    assert loss_far(big, big).item() < 1e-10
