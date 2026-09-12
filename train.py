"""Unified training driver for NSH / DEUDF / MPF.

    python train.py --method nsh   --shape torus      --iters 5000
    python train.py --method deudf --shape thin_plate --iters 5000
    python train.py --method mpf   --shape cube_sheet --iters 5000
    python train.py --method nsh   --shape my_scan.ply --points 20000

Outputs into out/<shape>_<method>/:
    recon.ply     reconstructed mesh
    input.ply     the input point cloud (for figures)
    metrics.json  CD / F-score / normal consistency
    loss.png      loss curves
This file is COMPLETE.
"""
import argparse, json, os, time

import numpy as np
import torch
import trimesh

from common import data as D
from common.sampling import Sampler, to_t
from common.siren import Siren, MPFNet
from common.extract import field_to_mesh, keep_largest_components
from common.metrics import evaluate
from methods import nsh, deudf, mpf

METHODS = {"nsh": nsh, "deudf": deudf, "mpf": mpf}


def pick_device(arg):
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build(method, cfg, device):
    if method == "mpf":
        return MPFNet(omega_0=cfg["omega_0"], beta_init=cfg["beta_init"]).to(device)
    return Siren(3, 256, 1, n_layers=cfg["n_layers"], omega_0=cfg["omega_0"],
                 omega_hidden=30.0).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--shape", default="torus")
    ap.add_argument("--points", type=int, default=20000)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--single_sided", action="store_true")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--res", type=int, default=192)
    ap.add_argument("--level", type=float, default=None,
                    help="MC iso-level. Default 0 for nsh/mpf, 0.005 for deudf "
                         "(UDF has no sign change -- see common/extract.py)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out")
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--solutions", action="store_true",
                    help="use reference implementations instead of yours "
                         "(see solutions/patch.py) -- for unblocking only")
    a = ap.parse_args()

    if a.solutions:
        from solutions import patch
        patch.apply()
        print("  [!] running with REFERENCE implementations, not yours")

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = pick_device(a.device)
    mod = METHODS[a.method]
    cfg = dict(mod.DEFAULTS)
    for k in ("iters", "batch", "lr"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)

    tag = f"{os.path.basename(a.shape).split('.')[0]}_{a.method}"
    odir = os.path.join(a.out, tag); os.makedirs(odir, exist_ok=True)
    print(f"[{tag}] device={device}  iters={cfg['iters']}  batch={cfg['batch']}")

    # ---------------------------------------------------------------- data
    S = D.load_shape(a.shape, a.points, a.noise, a.seed, a.single_sided)
    P, gt_mesh = S["points"], S["mesh"]
    trimesh.PointCloud(P).export(os.path.join(odir, "input.ply"))
    sampler = Sampler(P, seed=a.seed)
    nrm = D.pca_normals(P, cfg.get("knn_pca", 16))   # UNORIENTED, eval/loss only

    net = build(a.method, cfg, device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["iters"], 0.0) \
        if a.method == "deudf" else None

    # ---------------------------------------------------------------- train
    hist, t0 = [], time.time()
    B = cfg["batch"]
    for it in range(cfg["iters"]):
        ps, idx = sampler.surface(B)
        if a.method == "nsh":
            loss, log = mod.compute_loss(
                net, to_t(ps, device), to_t(sampler.near(B), device),
                to_t(sampler.uniform(B), device), it, cfg["iters"], cfg)
        elif a.method == "deudf":
            q1, q2, n_pair = sampler.normal_pairs(B, nrm, cfg["lam_max"])
            loss, log = mod.compute_loss(
                net, to_t(ps, device), to_t(q1, device), to_t(q2, device),
                to_t(n_pair, device), to_t(sampler.uniform(B), device),
                it, cfg["iters"], cfg)
        else:  # mpf
            loss, log = mod.compute_loss(
                net, to_t(ps, device), to_t(sampler.band(B, cfg["delta"]), device),
                to_t(sampler.far_field(B, cfg["delta"]), device),
                to_t(sampler.uniform(B), device), to_t(nrm[idx], device), it, cfg)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        if sched: sched.step()

        log["total"] = loss.item(); log["it"] = it
        hist.append(log)
        if it % a.log_every == 0 or it == cfg["iters"] - 1:
            parts = " ".join(f"{k}={v:.4g}" for k, v in log.items()
                             if k not in ("it",))
            print(f"  it {it:6d}  {parts}")
        if not np.isfinite(loss.item()):
            print("  !! loss became non-finite -- stopping. "
                  "Check for exp() overflow in L_positive / L_far.")
            break

    train_s = time.time() - t0

    # ---------------------------------------------------------------- extract
    if a.method == "mpf":
        fn = lambda x: net(x)[0]
    else:
        fn = lambda x: net(x)
    level = a.level if a.level is not None else (0.005 if a.method == "deudf" else 0.0)
    mesh = field_to_mesh(fn, res=a.res, bound=1.0, level=level, device=device)
    mesh, ncomp = keep_largest_components(mesh, k=1)
    if mesh is not None:
        mesh.export(os.path.join(odir, "recon.ply"))

    m = evaluate(mesh, gt_mesh) if gt_mesh is not None else {}
    m.update({"shape": a.shape, "method": a.method, "n_components": ncomp,
              "sec": round(train_s, 1), "iters": cfg["iters"],
              "points": len(P), "level": level})
    json.dump(m, open(os.path.join(odir, "metrics.json"), "w"), indent=2)
    print(json.dumps(m, indent=2))

    # ---------------------------------------------------------------- curves
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        keys = [k for k in hist[0] if k not in ("it", "tau", "xi", "beta")]
        fig, ax = plt.subplots(figsize=(7, 4))
        for k in keys:
            ax.plot([h["it"] for h in hist], [max(h[k], 1e-12) for h in hist],
                    label=k, lw=1)
        ax.set_yscale("log"); ax.set_xlabel("iteration"); ax.legend(fontsize=7)
        ax.set_title(f"{tag} losses")
        fig.tight_layout(); fig.savefig(os.path.join(odir, "loss.png"), dpi=130)
    except Exception as e:
        print("loss plot skipped:", e)


if __name__ == "__main__":
    main()
