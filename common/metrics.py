"""Evaluation metrics -- the numbers that go in your table.

The three papers report:
  Chamfer distance (L1, scaled x10^3)   lower is better  -- fitting tightness
  F-score at threshold tau              higher is better -- precision/recall harmonic mean
  Normal consistency (%)                higher is better -- do the surface
                                        orientations agree

Protocol (NSH Sec 5.2, matched by the others): normalise everything to
[-0.5,0.5]^3, sample 100K points from both the reconstructed and the GT mesh,
F-score threshold 0.005. DEUDF uses 0.005 and 0.0025. Use 0.005 as your headline.

Normalisation frame. The reconstruction and the ground truth live in the same
coordinate frame, so both are normalised with the GROUND TRUTH's centre and
scale (frame="gt", the default). An earlier version normalised each mesh by its
own bounding box (frame="each"): any stray geometry then rescaled and shifted
the whole reconstruction relative to the ground truth -- e.g. an MPF torus with
thin sheets reaching the domain edge scored F 9.6 instead of 65.0 -- and a
double cover could be slid into range by the recentring (DEUDF thin_plate at
res 128: F 28.1 instead of 1.8). "each" is kept only to compare with old numbers.
"""
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def _unit_transform(mesh):
    """Centre and scale that map `mesh` into [-0.5, 0.5]^3."""
    c = (mesh.bounds[0] + mesh.bounds[1]) / 2
    return c, np.abs(mesh.vertices - c).max() * 2.0


def _apply(mesh, c, s):
    m = mesh.copy()
    m.vertices = (m.vertices - c) / s
    return m


def sample_with_normals(mesh, n=100000, seed=0):
    rs = np.random.RandomState(seed)
    pts, fid = trimesh.sample.sample_surface(mesh, n, seed=seed) \
        if "seed" in trimesh.sample.sample_surface.__code__.co_varnames \
        else trimesh.sample.sample_surface(mesh, n)
    return np.asarray(pts), mesh.face_normals[fid]


def evaluate(pred_mesh, gt_mesh, n=100000, f_thresh=0.005, seed=0, frame="gt"):
    """Returns dict: chamfer_l1_x1e3, f_score, precision, recall, normal_consistency.

    frame: "gt"   -- normalise both meshes with the ground truth's transform
           "each" -- normalise each mesh by its own bounding box (old behaviour)
           None   -- use the meshes as given
    """
    if pred_mesh is None or len(pred_mesh.faces) == 0:
        return {"chamfer_l1_x1e3": float("nan"), "f_score": 0.0,
                "precision": 0.0, "recall": 0.0, "normal_consistency": float("nan")}
    if frame == "gt":
        c, s = _unit_transform(gt_mesh)
        A, B = _apply(pred_mesh, c, s), _apply(gt_mesh, c, s)
    elif frame == "each":
        A, B = _apply(pred_mesh, *_unit_transform(pred_mesh)), _apply(gt_mesh, *_unit_transform(gt_mesh))
    elif frame is None:
        A, B = pred_mesh, gt_mesh
    else:
        raise ValueError(f"frame must be 'gt', 'each' or None, not {frame!r}")
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
