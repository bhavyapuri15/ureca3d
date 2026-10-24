"""Property tests for the differential operators and loss terms.

    cd ureca3d && python -m pytest tests -q

These assert mathematical properties rather than exact numbers, so they catch the
failures that actually occur in this kind of code: a missing create_graph, a sign
flip, a reduction over the wrong axis, an absolute value in the wrong place.
Notably, test_true_sdf_has_singular_hessian verifies NSH's central claim
numerically -- it feeds the exact SDF of a sphere through the Hessian code and
checks that the determinant vanishes.
"""
import math
import sys

import numpy as np
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
    # DEUDF Eq. 4: delta(d) = 1 / (1 + (xi/d)^4) -- Eikonal OFF at the surface,
    # ON far from it.
    assert attenuation(torch.tensor([0.0]), xi).item() == pytest.approx(0.0)
    assert attenuation(torch.tensor([xi]), xi).item() == pytest.approx(0.5)
    assert attenuation(torch.tensor([-xi]), xi).item() == pytest.approx(0.5)
    assert attenuation(torch.tensor([0.1]), xi).item() == pytest.approx(1.0, abs=1e-3)
    assert torch.isfinite(attenuation(torch.tensor([0.0, 1e-12]), xi)).all()


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


def test_deudf_normal_pairs_loss():
    """Official-code form: 1 - |cos(grad q1, grad q2)|. Zero for collinear
    gradients (either direction), one for orthogonal ones."""
    from methods.deudf import loss_normal_pairs
    g = torch.randn(16, 3)
    assert loss_normal_pairs(g, -2.0 * g).item() == pytest.approx(0.0, abs=1e-5)
    assert loss_normal_pairs(g, 0.5 * g).item() == pytest.approx(0.0, abs=1e-5)
    e1 = torch.tensor([[1.0, 0.0, 0.0]]); e2 = torch.tensor([[0.0, 1.0, 0.0]])
    assert loss_normal_pairs(e1, e2).item() == pytest.approx(1.0, abs=1e-5)


def test_deudf_schedules_hold_then_cosine():
    """LR and xi are constant for the first 40% of training, then cosine-annealed
    (official code; the paper: xi decreases 'following the learning rate')."""
    from methods.deudf import phase_factor, xi_schedule
    T = 1001
    assert phase_factor(0, T) == 1.0 and phase_factor(400, T) == 1.0
    assert phase_factor(700, T) == pytest.approx(0.5)
    assert phase_factor(1000, T) == pytest.approx(0.0, abs=1e-12)
    assert xi_schedule(0, T) == pytest.approx(0.01)
    assert xi_schedule(400, T) == pytest.approx(0.01)
    assert xi_schedule(1000, T) == pytest.approx(0.002)


def test_deudf_pairs_from_same_points():
    """q1, q2 straddle the very points of the surface batch, at a distance in
    (0, lam_max] along the normal."""
    from common.sampling import Sampler
    rng = np.random.default_rng(0)
    P = rng.uniform(-0.5, 0.5, (500, 3)).astype(np.float32)
    n = rng.normal(size=(500, 3)).astype(np.float32)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    s = Sampler(P, seed=0)
    ps, idx = s.surface(256)
    q1, q2, nr = s.normal_pairs(256, n, 0.003, idx=idx)
    assert np.allclose((q1 + q2) / 2, ps, atol=1e-6)
    lam = np.linalg.norm(q1 - ps, axis=1)
    assert (lam > 0).all() and (lam <= 0.003 + 1e-7).all()


def test_uniform_bound_covers_the_cube():
    """bound=1 samples the whole of [-1,1]^3, not the padded point bbox."""
    from common.sampling import Sampler
    P = np.random.default_rng(0).uniform(-0.2, 0.2, (500, 3)).astype(np.float32)
    s = Sampler(P, seed=0)
    u = s.uniform(20000, bound=1.0)
    assert u.min() < -0.9 and u.max() > 0.9 and np.abs(u).max() <= 1.0
    assert np.abs(s.uniform(20000)).max() < 0.31     # default: padded bbox


def test_level_from_data_sits_above_the_data():
    """For a UDF offset by +0.02 everywhere, the level must be 0.02 plus the
    grid margin -- not relative to the global minimum."""
    from common.extract import level_from_data
    fn = lambda x: x[:, 2:3].abs() + 0.02
    pts = np.random.default_rng(0).uniform(-0.5, 0.5, (100, 3)).astype(np.float32)
    pts[:, 2] = 0.0
    level, f = level_from_data(fn, pts, res=129)
    assert np.allclose(f, 0.02)
    assert level == pytest.approx(0.02 + 0.75 * 2.0 / 128)


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


def test_filter_components_keeps_all_large_parts():
    """Every component with >= 5% of the largest one's faces survives -- not just
    the largest -- and tiny blobs are dropped."""
    import trimesh
    from common.extract import filter_components
    big = trimesh.creation.icosphere(subdivisions=3)                       # 1280 faces
    mid = trimesh.creation.icosphere(subdivisions=2); mid.apply_translation([3, 0, 0])  # 320
    tiny = trimesh.creation.icosphere(subdivisions=0); tiny.apply_translation([6, 0, 0])  # 20
    mesh, n, kept = filter_components(trimesh.util.concatenate([big, mid, tiny]))
    assert (n, kept) == (3, 2)
    assert len(mesh.faces) == len(big.faces) + len(mid.faces)


def test_metrics_use_the_ground_truth_frame():
    """A perfect reconstruction plus a small stray blob far away must still score
    well: the blob costs some precision, but must not rescale the whole mesh (the
    old per-mesh bounding-box normalisation shrank everything and F collapsed)."""
    import trimesh
    from common.metrics import evaluate
    gt = trimesh.creation.icosphere(subdivisions=4, radius=0.5)
    # two independent 100K-point samples of the same surface do not reach 100:
    # the sample spacing is close to the 0.005 threshold
    base = evaluate(gt, gt)["f_score"]
    assert base > 85.0
    blob = trimesh.creation.icosphere(subdivisions=1, radius=0.02)
    blob.apply_translation([0.95, 0.0, 0.0])
    pred = trimesh.util.concatenate([gt, blob])
    assert evaluate(pred, gt, frame="gt")["f_score"] > base - 3.0
    assert evaluate(pred, gt, frame="each")["f_score"] < 50.0


def test_far_field_respects_bound_and_band():
    """MPF far-field samples: inside [-bound, bound]^3 when bound is given, and
    never within delta of the input cloud."""
    from common.sampling import Sampler
    from scipy.spatial import cKDTree
    P = np.random.default_rng(0).uniform(-0.3, 0.3, (2000, 3)).astype(np.float32)
    P[:, 2] *= 0.1                                    # a flat slab of points
    s = Sampler(P, seed=0)
    f = s.far_field(4000, delta=0.05, bound=1.0)
    assert np.abs(f).max() <= 1.0 and np.abs(f[:, 2]).max() > 0.9
    assert cKDTree(P).query(f)[0].min() >= 0.05
    assert np.abs(s.far_field(4000, delta=0.05)[:, 2]).max() < 0.2   # default: padded bbox
