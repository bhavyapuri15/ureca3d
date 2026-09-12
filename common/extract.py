"""Surface extraction: turn the trained field into a triangle mesh.

SDF / MPF : Marching Cubes at level 0. Clean, because the field changes sign.
UDF       : there is NO sign change -- the field just touches zero and bounces
            back up. Plain MC at level 0 finds nothing. The standard hack is to
            march at a small POSITIVE iso-value eps, which gives you a
            "double cover": a thin closed shell wrapping the real surface, two
            sheets eps away on each side.
            DEUDF solves this properly with DCUDF (Hou et al. 2023), which
            shrinks the double cover onto the minimal-distance surface. DCUDF is
            a separate optimisation and is too much to implement in 3 days --
            so we extract the double cover and say so. That is an honest,
            defensible limitation to put on your slide.

This file is COMPLETE.
"""
import numpy as np
import torch
import trimesh
from skimage import measure


@torch.no_grad()
def eval_grid(fn, res=192, bound=1.0, chunk=65536, device="cpu"):
    """Evaluate a scalar field on a res^3 grid. fn: (N,3) tensor -> (N,) tensor."""
    t = np.linspace(-bound, bound, res, dtype=np.float32)
    X, Y, Z = np.meshgrid(t, t, t, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    out = np.empty(len(pts), np.float32)
    for i in range(0, len(pts), chunk):
        b = torch.as_tensor(pts[i:i + chunk], device=device)
        out[i:i + chunk] = fn(b).detach().cpu().numpy().reshape(-1)
    return out.reshape(res, res, res), t


def field_to_mesh(fn, res=192, bound=1.0, level=0.0, device="cpu",
                  needs_grad=False, chunk=65536):
    """Marching cubes on the field. Returns a trimesh (or None if empty).

    needs_grad=True runs the field WITH autograd enabled (MPF's phi needs the
    graph only if you compute derivatives -- for plain extraction you do not).
    """
    ctx = torch.enable_grad() if needs_grad else torch.no_grad()
    with ctx:
        vol, t = eval_grid(fn, res, bound, chunk, device)
    lo, hi = float(vol.min()), float(vol.max())

    if level == "auto":
        # UDF double-cover extraction. The learned field never actually reaches
        # zero -- it bottoms out at some small positive value, and on a finite
        # grid you only ever sample near that minimum, not at it. A fixed
        # iso-value therefore either finds nothing (too low) or bloats the shell
        # (too high). Pick it relative to the field's own minimum instead:
        # one grid cell above the floor is the thinnest closed shell the grid
        # can actually represent.
        spacing = 2.0 * bound / (res - 1)
        level = lo + max(0.75 * spacing, 0.002)
        print(f"  [extract] auto level = {level:.5f}  (field min {lo:.5f}, "
              f"grid spacing {spacing:.5f})")

    if not (lo < level < hi):
        print(f"  [extract] level {level} outside field range "
              f"[{lo:.4f}, {hi:.4f}] -- no surface found")
        return None
    v, f, n, _ = measure.marching_cubes(vol, level=level,
                                        spacing=(t[1] - t[0],) * 3)
    v = v + np.array([-bound, -bound, -bound], np.float32)
    m = trimesh.Trimesh(vertices=v, faces=f, vertex_normals=n, process=False)
    return m


def keep_largest_components(mesh, k=1, min_frac=0.05):
    """Drop tiny floating blobs (ghost geometry). Report how many you dropped --
    the COUNT is itself a result: NSH's whole pitch is that it removes them."""
    if mesh is None:
        return None, 0
    comps = mesh.split(only_watertight=False)
    if len(comps) <= 1:
        return mesh, len(comps)
    comps = sorted(comps, key=lambda c: len(c.faces), reverse=True)
    big = [c for c in comps if len(c.faces) >= min_frac * len(comps[0].faces)][:max(k, 1)]
    return trimesh.util.concatenate(big), len(comps)
