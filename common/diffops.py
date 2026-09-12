"""Differential operators via autograd.

All three methods are built from four quantities:

    gradient(f)   first derivatives,  (N,3)
    hessian(f)    second derivatives, (N,3,3)
    det3x3(H)     NSH's key quantity
    laplacian(f)  trace of the Hessian, used by MPF

`create_graph=True` is required everywhere: without it the returned derivative is
detached from the graph, so a second differentiation (the Hessian) fails and no
gradient reaches the network weights from any derivative-based loss term.
"""
import torch


def gradient(y, x):
    """First derivative of a scalar field.

    Args:
        y: (N,1) network output, produced from x (x must require grad)
        x: (N,3) input coordinates
    Returns:
        (N,3), row i is grad f(x_i)
    """
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                               create_graph=True, retain_graph=True)[0]


def hessian(y, x, grad_y=None):
    """Full Hessian of a scalar field.

    The Hessian is the Jacobian of the gradient. There is no batched-Jacobian
    primitive that is faster here, so it is built one row at a time: three extra
    backward passes. This is why NSH is the slowest of the three methods.

    Args:
        y: (N,1); x: (N,D); grad_y: optional precomputed (N,D) gradient
    Returns:
        (N,D,D), H[i,j,k] = d^2 f / dx_j dx_k at point i. Symmetric up to
        floating-point error, since mixed partials commute for a C^2 function.
    """
    g = grad_y if grad_y is not None else gradient(y, x)
    rows = []
    for j in range(x.shape[-1]):
        gj = g[:, j]
        rows.append(torch.autograd.grad(gj, x, grad_outputs=torch.ones_like(gj),
                                        create_graph=True, retain_graph=True)[0])
    return torch.stack(rows, dim=1)


def det3x3(H):
    """Determinant of a batch of 3x3 matrices, by the rule of Sarrus.

    Closed form rather than torch.linalg.det: this runs on ~15K points every
    iteration, and the explicit expression avoids an LU factorisation per point.

    Args:   H: (N,3,3)
    Returns: (N,)
    """
    a, b, c = H[:, 0, 0], H[:, 0, 1], H[:, 0, 2]
    d, e, f = H[:, 1, 0], H[:, 1, 1], H[:, 1, 2]
    g, h, i = H[:, 2, 0], H[:, 2, 1], H[:, 2, 2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def laplacian(y, x, grad_y=None):
    """Laplacian = trace of the Hessian = f_xx + f_yy + f_zz. Returns (N,1).

    Only the diagonal entries are needed, so the full Hessian is not built.
    """
    g = grad_y if grad_y is not None else gradient(y, x)
    out = 0.0
    for j in range(x.shape[-1]):
        gj = g[:, j]
        hj = torch.autograd.grad(gj, x, grad_outputs=torch.ones_like(gj),
                                 create_graph=True, retain_graph=True)[0]
        out = out + hj[:, j]
    return out.unsqueeze(-1)


# ----------------------------------------------------------------------------

def make_input(x):
    """Detach and require grad, so autograd can differentiate w.r.t. coordinates.
    Must be called BEFORE the forward pass."""
    return x.clone().detach().requires_grad_(True)


def safe_normalize(v, eps=1e-8):
    return v / (v.norm(dim=-1, keepdim=True) + eps)
