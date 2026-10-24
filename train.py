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
    ckpt.pt       training state, only with --ckpt_every N

With --ckpt_every N --resume, an interrupted run picks up from its last
checkpoint instead of starting over:

    python train.py --method mpf --shape torus --ckpt_every 1000 --resume
This file is COMPLETE.
"""
import argparse, hashlib, json, os, time

import numpy as np
import torch
import trimesh

from common import data as D
from common.sampling import Sampler, to_t
from common.siren import Siren, MPFNet
from common.extract import field_to_mesh, filter_components, level_from_data
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
                 omega_hidden=cfg.get("omega_hidden", 30.0)).to(device)


def extract_and_evaluate(net, method, P, gt_mesh, res, level, device, odir):
    """Marching Cubes on the trained field, component filtering, metrics.
    Writes <odir>/recon.ply and returns the metrics dict (without training info).
    Shared by train.py and reextract.py."""
    fn = (lambda x: net(x)[0]) if method == "mpf" else (lambda x: net(x))
    if level is None:
        level = "data" if method == "deudf" else 0.0
    elif level not in ("auto", "data"):
        level = float(level)
    level_rule, f_stats = level, {}
    if level == "data":
        level, fP = level_from_data(fn, P, res, bound=1.0, device=device)
        f_stats = {"f_data_min": float(fP.min()), "f_data_median": float(np.median(fP)),
                   "f_data_max": float(fP.max())}
        print(f"  [extract] data level = {level:.5f}  (f at input points: min "
              f"{fP.min():.5f} median {np.median(fP):.5f} max {fP.max():.5f}; "
              f"{100 * (fP < level).mean():.1f}% below the level)")
    mesh = field_to_mesh(fn, res=res, bound=1.0, level=level, device=device)
    mesh, ncomp, nkept = filter_components(mesh)
    if mesh is not None:
        mesh.export(os.path.join(odir, "recon.ply"))
    m = {}
    if gt_mesh is not None:
        m = evaluate(mesh, gt_mesh, frame="gt")
        old = evaluate(mesh, gt_mesh, frame="each")   # for comparison with old numbers
        m.update({"metric_frame": "gt", "f_score_each_bbox": old["f_score"],
                  "chamfer_l1_x1e3_each_bbox": old["chamfer_l1_x1e3"]})
    m.update({"n_components": ncomp, "n_kept": nkept, "res": res,
              "level": level, "level_rule": level_rule, **f_stats})
    return m


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
    ap.add_argument("--level", default=None,
                    help="MC iso-level: a number, 'data' or 'auto'. Default 0 for "
                         "nsh/mpf, 'data' for deudf (a UDF has no sign change, so "
                         "the level is picked just above the median field value "
                         "at the input points; 'auto' uses the field minimum "
                         "instead -- see common/extract.py)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out")
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--no_sphere_init", action="store_true",
                    help="MPF only: skip the geometric initialisation of the "
                         "phase head (ablation -- see methods/mpf.py)")
    ap.add_argument("--ckpt_every", type=int, default=0,
                    help="save <out>/<tag>/ckpt.pt every N iterations and at the "
                         "end of training (0 = never)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>/<tag>/ckpt.pt if it exists")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = pick_device(a.device)
    mod = METHODS[a.method]
    cfg = dict(mod.DEFAULTS)
    for k in ("iters", "batch", "lr"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)

    # MPF activates L_align after a fixed 3000 iterations (paper Sec. 5, tuned
    # for their 10K-iteration schedule). On a shorter run that threshold can
    # never be reached, so the alignment term silently never fires and you are
    # not running the method at all. Treat it as a fraction of training instead.
    if a.method == "mpf":
        cfg["align_start"] = min(cfg["align_start"], int(0.3 * cfg["iters"]))

    tag = f"{os.path.basename(a.shape).split('.')[0]}_{a.method}"
    odir = os.path.join(a.out, tag); os.makedirs(odir, exist_ok=True)
    print(f"[{tag}] device={device}  iters={cfg['iters']}  batch={cfg['batch']}")

    # ---------------------------------------------------------------- data
    S = D.load_shape(a.shape, a.points, a.noise, a.seed, a.single_sided)
    P, gt_mesh = S["points"], S["mesh"]
    trimesh.PointCloud(P).export(os.path.join(odir, "input.ply"))
    sampler = Sampler(P, seed=a.seed)
    nrm = D.pca_normals(P, cfg.get("knn_pca", 16))   # UNORIENTED, eval/loss only

    # A checkpoint is only valid for the run that wrote it.
    run_id = {k: getattr(a, k) for k in ("method", "shape", "points", "noise",
                                         "single_sided", "seed", "no_sphere_init")}
    run_id.update({k: cfg[k] for k in ("iters", "batch", "lr")})
    run_id["points_md5"] = hashlib.md5(np.ascontiguousarray(P).tobytes()).hexdigest()
    ckpt_path = os.path.join(odir, "ckpt.pt")
    ckpt = None
    if a.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        if ckpt["run_id"] != run_id:
            raise SystemExit(f"{ckpt_path} was written by a different run:\n"
                             f"  checkpoint: {ckpt['run_id']}\n  this run:   {run_id}\n"
                             "Delete it or use a different --out.")

    net = build(a.method, cfg, device)
    if (ckpt is None and a.method == "mpf" and not a.no_sphere_init
            and cfg["sphere_init_iters"] > 0):
        mpf.sphere_init(net, iters=cfg["sphere_init_iters"], device=device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"],
                           betas=cfg.get("adam_betas", (0.9, 0.999)))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: mod.lr_lambda(i, cfg["iters"], cfg)) \
        if hasattr(mod, "lr_lambda") else None
    grad_clip = cfg.get("grad_clip", 10.0)

    # All schedules (NSH tau, DEUDF xi, MPF align_start) are functions of `it`,
    # so restarting the loop at the saved iteration continues them exactly. The
    # sampler RNG is restored too, so the batches match an uninterrupted run.
    start_it, hist, prev_s = 0, [], 0.0
    if ckpt is not None:
        net.load_state_dict(ckpt["net"])
        opt.load_state_dict(ckpt["opt"])
        if sched: sched.load_state_dict(ckpt["sched"])
        sampler.rng.bit_generator.state = ckpt["sampler_rng"]
        start_it, hist, prev_s = ckpt["it"], ckpt["hist"], ckpt["train_s"]
        print(f"[{tag}] resumed from {ckpt_path} at iteration {start_it}")

    def save_ckpt(next_it):
        state = {"run_id": run_id, "cfg": cfg, "it": next_it, "net": net.state_dict(),
                 "opt": opt.state_dict(),
                 "sched": sched.state_dict() if sched else None,
                 "sampler_rng": sampler.rng.bit_generator.state, "hist": hist,
                 "train_s": prev_s + time.time() - t0}
        torch.save(state, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)   # never leave a half-written file

    # ---------------------------------------------------------------- train
    t0 = time.time()
    B = cfg["batch"]
    for it in range(start_it, cfg["iters"]):
        ps, idx = sampler.surface(B)
        if a.method == "nsh":
            loss, log = mod.compute_loss(
                net, to_t(ps, device), to_t(sampler.near(B), device),
                to_t(sampler.uniform(B), device), it, cfg["iters"], cfg)
        elif a.method == "deudf":
            q1, q2, n_pair = sampler.normal_pairs(B, nrm, cfg["lam_max"], idx=idx)
            loss, log = mod.compute_loss(
                net, to_t(ps, device), to_t(q1, device), to_t(q2, device),
                to_t(n_pair, device),
                to_t(sampler.uniform(B, cfg.get("uniform_bound")), device),
                it, cfg["iters"], cfg)
        else:  # mpf
            loss, log = mod.compute_loss(
                net, to_t(ps, device), to_t(sampler.band(B, cfg["delta"]), device),
                to_t(sampler.far_field(B, cfg["delta"], bound=cfg.get("uniform_bound")), device),
                to_t(sampler.uniform(B, cfg.get("uniform_bound")), device),
                to_t(nrm[idx], device), it, cfg)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
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
        if a.ckpt_every and ((it + 1) % a.ckpt_every == 0 or it + 1 == cfg["iters"]):
            save_ckpt(it + 1)

    train_s = prev_s + time.time() - t0

    # ---------------------------------------------------------------- extract
    m = extract_and_evaluate(net, a.method, P, gt_mesh, a.res, a.level, device, odir)
    m.update({"shape": a.shape, "method": a.method, "sec": round(train_s, 1),
              "iters": cfg["iters"], "points": len(P), "resumed_from_it": start_it})
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
