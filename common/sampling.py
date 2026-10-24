"""Query-point samplers.

None of these methods only look at the input points. They all need *space*
samples, because the losses that shape the field (Eikonal, Hessian, Laplacian,
repulsion) are integrals over a region, approximated by Monte-Carlo sampling.
Each paper samples differently, and the difference matters:

  NSH   (Sec 5.1) : Q_far  = uniform in the bounding box
                    Q_near = one Gaussian sample per input point, sigma = the
                             distance from that point to its 50th neighbour.
                             So Q_near follows the local point density: dense
                             regions get a tight band, sparse regions a wide one.
  DEUDF (Eq. 3)   : Q = {p + lam*n, p - lam*n}, lam ~ U(0, 0.003). Pairs of
                    points straddling the surface along the PCA normal. This is
                    what makes the normal-alignment loss possible.
  MPF   (Sec 5)   : jitter input points to build a tubular band N_delta, keep
                    only samples with 0 < d_X(x) < delta (KD-tree distance to
                    the input cloud); far-field samples drawn uniformly in the
                    bbox and *rejected* if they fall in the band.

This file is COMPLETE.
"""
import numpy as np
import torch
from scipy.spatial import cKDTree


class Sampler:
    def __init__(self, points, bbox_pad=0.1, knn_sigma=50, seed=0):
        self.P = np.asarray(points, np.float32)
        self.rng = np.random.default_rng(seed)
        self.tree = cKDTree(self.P)
        lo, hi = self.P.min(0), self.P.max(0)
        c, e = (lo + hi) / 2, (hi - lo) / 2 + bbox_pad
        self.lo, self.hi = c - e, c + e
        # per-point sigma = distance to the knn_sigma-th neighbour (NSH Sec 5.1)
        k = min(knn_sigma + 1, len(self.P))
        d, _ = self.tree.query(self.P, k=k)
        self.sigma = np.maximum(d[:, -1], 1e-4).astype(np.float32)

    # ------------------------------------------------------------- basic
    def surface(self, n):
        """n random input points (the 'manifold'/on-surface batch)."""
        i = self.rng.integers(0, len(self.P), n)
        return self.P[i], i

    def uniform(self, n, bound=None):
        """Q_far: uniform in the padded bounding box, or in [-bound, bound]^3 if
        bound is given (DEUDF's Omega = [-1,1]^3). The padded box can be much
        smaller than the extraction domain -- e.g. |z| < 0.34 for the torus -- and
        the field is unconstrained outside it."""
        if bound is not None:
            return self.rng.uniform(-bound, bound, (n, 3)).astype(np.float32)
        return self.rng.uniform(self.lo, self.hi, (n, 3)).astype(np.float32)

    def near(self, n):
        """Q_near, NSH style: Gaussian around input points with density-adaptive
        sigma. One sample per chosen point."""
        i = self.rng.integers(0, len(self.P), n)
        return (self.P[i] + self.rng.normal(0, 1, (n, 3)).astype(np.float32)
                * self.sigma[i, None]).astype(np.float32)

    # ------------------------------------------------------------- MPF
    def band(self, n, delta=0.05, oversample=4):
        """N_delta: points with 0 < dist_to_cloud < delta (MPF Eq. 9)."""
        cand = self.near(n * oversample)
        d, _ = self.tree.query(cand)
        keep = cand[(d > 1e-5) & (d < delta)]
        if len(keep) < n:                      # top up with more jitter
            extra = self.near(n * oversample * 2)
            de, _ = self.tree.query(extra)
            keep = np.concatenate([keep, extra[(de > 1e-5) & (de < delta)]])
        idx = self.rng.integers(0, max(len(keep), 1), n)
        return keep[idx] if len(keep) else self.near(n)

    def far_field(self, n, delta=0.05, oversample=3, bound=None):
        """Omega \\ N_delta: uniform samples that are NOT near the surface
        (MPF Eq. 14). Used for the repulsion term that kills spurious sheets.
        bound: as in uniform() -- None for the padded bbox, else [-bound, bound]^3."""
        cand = self.uniform(n * oversample, bound)
        d, _ = self.tree.query(cand)
        keep = cand[d >= delta]
        if len(keep) < n:
            keep = np.concatenate([keep, self.uniform(n, bound)])
        idx = self.rng.integers(0, len(keep), n)
        return keep[idx]

    # ------------------------------------------------------------- DEUDF
    def normal_pairs(self, n, normals, lam_max=0.003, idx=None):
        """Q = {q1 = p + lam*n, q2 = p - lam*n}, lam in (0, lam_max] (DEUDF Eq. 3
        text).

        Returns q1, q2, nrm -- each (n,3). These straddle the surface, so the
        field must have the same |gradient direction| at both, but the gradient
        FLIPS sign across the surface for a UDF. That is the (-1)^k in their loss.
        Pass idx (from surface()) to build the pairs from the same input points,
        as the official DEUDF code does.
        """
        i = self.rng.integers(0, len(self.P), n) if idx is None else idx
        p, nr = self.P[i], normals[i]
        lam = (lam_max * (1.0 - self.rng.uniform(0, 1, (len(i), 1)))).astype(np.float32)
        return (p + lam * nr).astype(np.float32), (p - lam * nr).astype(np.float32), nr

    def dist_to_cloud(self, x):
        """KD-tree unsigned distance from arbitrary points to the input cloud.
        MPF uses this as an approximation of the true UDF for its band test."""
        d, _ = self.tree.query(np.asarray(x, np.float32))
        return d.astype(np.float32)


def to_t(a, device):
    return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
