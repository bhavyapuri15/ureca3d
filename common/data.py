"""Test shapes and point-cloud sampling.

The professor asked for *reconstruction results*, so you need input point clouds
and (for numbers) ground-truth meshes. Downloading ShapeNet/Thingi10K is a waste
of your 3 days. These synthetic shapes are chosen to expose exactly the
differences the three papers are about:

    sphere       watertight, trivial          -> sanity check, everything works
    torus        watertight, has a hole       -> topology / ghost geometry check
    thin_plate   open sheet, zero thickness   -> SDF fails, UDF/MPF should win
    cube_sheet   MPF's own Fig. 3 toy shape   -> watertight cube + attached thin
                                                 sheet, non-manifold junction.
                                                 THE money figure for MPF.
    sparse_*     any of the above with 1K pts -> NSH's sparse-input claim

This file is COMPLETE.
"""
import numpy as np
import trimesh


# ---------------------------------------------------------------- shape zoo

def make_sphere(r=0.6, subdiv=4):
    return trimesh.creation.icosphere(subdivisions=subdiv, radius=r)


def make_torus(R=0.45, r=0.16, sections=96):
    return trimesh.creation.torus(major_radius=R, minor_radius=r,
                                  major_sections=sections, minor_sections=48)


def make_thin_plate(size=0.9, n=80):
    """A single open sheet with zero thickness, gently curved.
    A true SDF does not exist for this (no inside/outside). This is the case
    DEUDF and MPF are built for, and where NSH must fail -- show that."""
    u = np.linspace(-size / 2, size / 2, n)
    X, Y = np.meshgrid(u, u, indexing="ij")
    Z = 0.18 * np.sin(3.0 * X) * np.cos(3.0 * Y)
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b = i * n + j, i * n + j + 1
            c, d = (i + 1) * n + j, (i + 1) * n + j + 1
            F += [[a, b, d], [a, d, c]]
    return trimesh.Trimesh(vertices=V, faces=np.array(F), process=False)


def make_cube_sheet(cube=0.5, sheet_h=0.45, sheet_w=0.36):
    """MPF Figure 3: a watertight cube with a thin sheet sticking out of the top.

    The sheet's attachment line is non-manifold (3 faces meet an edge) and its
    free end is a boundary. Result:
      - SDF methods (NSH) give a clean sign on the cube but thicken/close the sheet
      - UDF methods (DEUDF) represent the sheet but lose the cube's inside/outside
      - MPF should get both right
    Reproduce this one figure and you have essentially reproduced the MPF paper's
    core claim."""
    box = trimesh.creation.box(extents=[cube, cube, cube])
    n = 60
    xs = np.linspace(-sheet_w / 2, sheet_w / 2, n)
    zs = np.linspace(cube / 2, cube / 2 + sheet_h, n)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    V = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1)
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b = i * n + j, i * n + j + 1
            c, d = (i + 1) * n + j, (i + 1) * n + j + 1
            F += [[a, b, d], [a, d, c]]
    sheet = trimesh.Trimesh(vertices=V, faces=np.array(F), process=False)
    return trimesh.util.concatenate([box, sheet])


SHAPES = {
    "sphere": make_sphere,
    "torus": make_torus,
    "thin_plate": make_thin_plate,
    "cube_sheet": make_cube_sheet,
}


# ---------------------------------------------------------------- loading

def normalize_mesh(mesh, scale=0.9):
    """Center at origin and fit inside [-scale, scale]^3.

    All three papers normalise to roughly [-1,1]^3 (NSH Sec 5.1, DEUDF 'cube
    domain Omega=[-1,1]^3', MPF Sec 5). Metrics are only comparable across
    methods if every method sees the same normalisation, so do it once, here."""
    mesh = mesh.copy()
    c = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    mesh.vertices -= c
    s = np.abs(mesh.vertices).max()
    mesh.vertices *= (scale / s)
    return mesh


def load_shape(name_or_path, n_points=20000, noise=0.0, seed=0, single_sided=False):
    """Returns dict with keys: points (N,3), normals (N,3), mesh (trimesh or None).

    Args:
        name_or_path : one of SHAPES, or a path to .obj/.ply/.off
        n_points     : how many points to sample from the surface
        noise        : std of Gaussian noise added to the points, in normalised
                       units (DEUDF tests 0.0025 and 0.005 -- use those)
        single_sided : keep only points whose normal faces +y. Simulates a
                       scanner that saw a thin wall from one side only -- MPF
                       calls this a "single-layer point cloud" and it is the
                       pathological case for SDFs.
    NOTE: normals are returned for EVALUATION ONLY (normal-consistency metric,
    and DEUDF's L_normal which uses *unoriented* PCA normals). None of the three
    methods may use the ground-truth orientation as supervision -- that would be
    cheating, since all three claim to work on *unoriented* point clouds.
    """
    rng = np.random.default_rng(seed)
    if name_or_path in SHAPES:
        mesh = SHAPES[name_or_path]()
    else:
        mesh = trimesh.load(name_or_path, process=False, force="mesh")
    mesh = normalize_mesh(mesh)

    pts, fid = trimesh.sample.sample_surface(mesh, n_points)
    nrm = mesh.face_normals[fid]

    if single_sided:
        keep = nrm[:, 1] > 0
        pts, nrm = pts[keep], nrm[keep]

    if noise > 0:
        pts = pts + rng.normal(0, noise, pts.shape)

    return {"points": np.asarray(pts, np.float32),
            "normals": np.asarray(nrm, np.float32),
            "mesh": mesh}


def pca_normals(points, k=16):
    """Unoriented normals by local PCA (Hoppe et al. 1992).

    DEUDF's L_normal (Eq. 3) and MPF's optional L_normal (Eq. 15) both use these.
    Crucially the SIGN is arbitrary -- that is why both papers use an
    orientation-invariant loss (|dot| or the (-1)^k trick). Never fix the signs.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k)
    nbrs = points[idx]                              # (N,k,3)
    nbrs = nbrs - nbrs.mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nbrs, nbrs) / k  # (N,3,3)
    w, v = np.linalg.eigh(cov)
    return np.ascontiguousarray(v[:, :, 0], dtype=np.float32)  # smallest eigenvector
