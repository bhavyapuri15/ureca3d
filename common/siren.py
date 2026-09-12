"""SIREN network (Sitzmann et al., NeurIPS 2020).

All three papers use SIREN as the backbone. Reason: a ReLU MLP has a zero second
derivative almost everywhere, so you *cannot* compute a meaningful Hessian from
it (NSH needs one) and it is biased toward low-frequency output (loses detail).
sin() is smooth and infinitely differentiable, so every derivative you need for
the loss terms exists.

This file is COMPLETE. You do not need to edit it.
"""
import numpy as np
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    """y = sin(omega_0 * (Wx + b)).

    omega_0 scales the frequency. The original SIREN paper uses 30. NSH uses 30.
    DEUDF uses 60 for clean point clouds and 30 for noisy ones (their footnote 1).
    """

    def __init__(self, in_f, out_f, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_f = in_f
        self.linear = nn.Linear(in_f, out_f)
        self.init_weights()

    def init_weights(self):
        # SIREN's principled init: keeps the pre-activation distribution stable
        # through depth so sin() does not saturate. Do not replace with default
        # PyTorch init -- SIREN genuinely does not train without this.
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_f, 1 / self.in_f)
            else:
                b = np.sqrt(6 / self.in_f) / self.omega_0
                self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class Siren(nn.Module):
    """SIREN MLP: R^in -> R^out.

    Used as:
      * NSH   : Siren(3 -> 1), output is the (relaxed) SDF f(x).
      * DEUDF : Siren(3 -> 1), output is the UDF -- note NO abs/softplus at the
                end. That "unconditioned MLP" is one of DEUDF's contributions.
      * MPF   : Siren(3 -> 256) backbone, then two small heads (see methods/mpf.py).
    """

    def __init__(self, in_f=3, hidden=256, out_f=1, n_layers=5,
                 omega_0=30.0, omega_hidden=30.0, final_linear=True):
        super().__init__()
        layers = [SineLayer(in_f, hidden, is_first=True, omega_0=omega_0)]
        for _ in range(n_layers - 2):
            layers.append(SineLayer(hidden, hidden, is_first=False, omega_0=omega_hidden))
        if final_linear:
            final = nn.Linear(hidden, out_f)
            with torch.no_grad():
                b = np.sqrt(6 / hidden) / omega_hidden
                final.weight.uniform_(-b, b)
            layers.append(final)
        else:
            layers.append(SineLayer(hidden, out_f, is_first=False, omega_0=omega_hidden))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MPFNet(nn.Module):
    """MPF's two-head architecture (paper Sec. 3.5).

    backbone : 3 -> 256 -> 256   (SIREN, 2 sine layers)
    r-head   : 256 -> 256 -> 1   then softplus  => metric field r(x) >= 0
    th-head  : 256 -> 256 -> 1   (unbounded)    => phase field theta(x)
    beta     : a single learnable scalar, init 50.

    Composite signed field:  phi(x) = r(x) * tanh(beta * theta(x)) + theta(x)
    """

    def __init__(self, hidden=256, omega_0=30.0, softplus_beta=100.0,
                 beta_init=50.0, phase_bias=None, r_bias=0.05):
        super().__init__()
        self.backbone = nn.Sequential(
            SineLayer(3, hidden, is_first=True, omega_0=omega_0),
            SineLayer(hidden, hidden, omega_0=omega_0),
        )
        self.head_r = nn.Sequential(SineLayer(hidden, hidden, omega_0=omega_0),
                                    nn.Linear(hidden, 1))
        self.head_t = nn.Sequential(SineLayer(hidden, hidden, omega_0=omega_0),
                                    nn.Linear(hidden, 1))
        self.softplus = nn.Softplus(beta=softplus_beta)
        # beta is learnable and must stay > 0; we store log(beta) so it can never
        # go negative during optimisation.
        self.log_beta = nn.Parameter(torch.tensor(float(np.log(beta_init))))
        for head in (self.head_r, self.head_t):
            with torch.no_grad():
                b = np.sqrt(6 / hidden) / omega_0
                head[-1].weight.uniform_(-b, b)
        with torch.no_grad():
            # Start the metric head inside softplus's active region. With
            # beta = 100, softplus is numerically flat for pre-activations below
            # about -0.05: r saturates to 0, its gradient vanishes, and the
            # metric branch dies -- L_eik^r then sits pinned at exactly 1.0
            # (= |0 - 1|) for the whole run while phi degenerates to theta alone.
            # A small positive bias keeps the head in the linear region at init.
            self.head_r[-1].bias.fill_(r_bias)
        if phase_bias is not None:
            # Optional: small negative init bias on the phase head, suggested in
            # the paper for shapes you expect to be closed/watertight.
            with torch.no_grad():
                self.head_t[-1].bias.fill_(float(phase_bias))

    @property
    def beta(self):
        return self.log_beta.exp()

    def forward(self, x):
        """Returns (phi, r, theta, P) all of shape (N, 1).

        You will implement the composition in methods/mpf.py::compose.
        """
        from methods.mpf import compose  # local import to keep the TODO in one place
        feat = self.backbone(x)
        r = self.softplus(self.head_r(feat))          # Eq. 16, r >= 0 by construction
        theta = self.head_t(feat)                     # Eq. 17, unbounded
        phi, P = compose(r, theta, self.beta)         # Eq. 2 + Eq. 3  <-- YOUR CODE
        return phi, r, theta, P
