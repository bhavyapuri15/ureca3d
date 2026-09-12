"""Evaluation metrics -- the numbers that go in your table.

The three papers report:
  Chamfer distance (L1, scaled x10^3)   lower is better  -- fitting tightness
  F-score at threshold tau              higher is better -- precision/recall harmonic mean
  Normal consistency (%)                higher is better -- do the surface
                                        orientations agree

Protocol (NSH Sec 5.2, matched by the others): normalise everything to
[-0.5,0.5]^3, sample 100K points from both the reconstructed and the GT mesh,
F-score threshold 0.005. DEUDF uses 0.005 and 0.0025. Use 0.005 as your headline.

This file is COMPLETE. Use it unmodified so your numbers are comparable to the
published tables.
"""
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def _rescale_unit(mesh):
    m = mesh.copy()
    c = (m.bounds[0] + m.bounds[1]) / 2
    m.vertices -= c
    m.vertices /= (np.abs(m.vertices).max() * 2.0)   # -> [-0.5, 0.5]
    return m


def sample_with_normals(mesh, n=100000, seed=0):
    rs = np.random.RandomState(seed)
    pts, fid = trimesh.sample.sample_surface(mesh, n, seed=seed) \
        if "seed" in trimesh.sample.sample_surface.__code__.co_varnames \
        else trimesh.sample.sample_surface(mesh, n)
    return np.asarray(pts), mesh.face_normals[fid]


def evaluate(pred_mesh, gt_mesh, n=100000, f_thresh=0.005, seed=0,
             rescale=True):
    """Returns dict: chamfer_l1_x1e3, f_score, precision, recall, normal_consistency."""
    if pred_mesh is None or len(pred_mesh.faces) == 0:
        return {"chamfer_l1_x1e3": float("nan"), "f_score": 0.0,
                "precision": 0.0, "recall": 0.0, "normal_consistency": float("nan")}
    A, B = (_rescale_unit(pred_mesh), _rescale_unit(gt_mesh)) if rescale \
        else (pred_mesh, gt_mesh)
    pa, na = sample_with_normals(A, n, seed)
    pb, nb = sample_with_normals(B, n, seed + 1)

    ta, tb = cKDTree(pa), cKDTree(pb)
    d_ab, i_ab = tb.query(pa)      # pred -> gt   (accuracy)
    d_ba, i_ba = ta.query(pb)      # gt   -> pred (completeness)

    chamfer = 0.5 * (d_ab.mean() + d_ba.mean())

    precision = float((d_ab < f_thresh).mean())
    recall = float((d_ba < f_thresh).mean())
    f = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    # unoriented normal agreement: |cos| , because reconstructions from
    # unoriented input may have globally flipped orientation and that is fine
    nc = 0.5 * (np.abs((na * nb[i_ab]).sum(1)).mean()
                + np.abs((nb * na[i_ba]).sum(1)).mean())

    return {"chamfer_l1_x1e3": float(chamfer * 1e3),
            "f_score": 100 * f,
            "precision": 100 * precision,
            "recall": 100 * recall,
            "normal_consistency": 100 * float(nc)}


def format_table(rows):
    """rows: list of dicts with 'shape','method' + metric keys -> markdown table."""
    cols = ["shape", "method", "chamfer_l1_x1e3", "f_score",
            "normal_consistency", "n_components", "sec"]
    head = {"chamfer_l1_x1e3": "CD-L1 x1e3 v", "f_score": "F-score ^",
            "normal_consistency": "Normal C. ^", "n_components": "#comp",
            "sec": "time(s)", "shape": "shape", "method": "method"}
    out = ["| " + " | ".join(head[c] for c in cols) + " |",
           "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)
