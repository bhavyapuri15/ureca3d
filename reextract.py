"""Re-extract and re-evaluate a finished run from its checkpoint (no retraining).

    python reextract.py out/torus_nsh                 # same res as before
    python reextract.py out/torus_nsh --res 256 --level 0

Needs <run_dir>/ckpt.pt, i.e. a run trained with --ckpt_every. The input points
are regenerated from the checkpoint's run_id (shape, points, noise, seed) and
checked against the stored hash, so the evaluation sees exactly the training data.

Overwrites recon.ply and metrics.json. The first time, the originals are kept as
recon_orig.ply and metrics_orig.json; training fields (sec, iters, ...) are
carried over from metrics.json.
"""
import argparse, hashlib, json, os, shutil

import numpy as np
import torch

from common import data as D
from train import METHODS, build, extract_and_evaluate, pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--res", type=int, default=None,
                    help="default: the res recorded in metrics.json, else 256")
    ap.add_argument("--level", default=None,
                    help="as in train.py; default 0 for nsh/mpf, 'data' for deudf")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    device = pick_device(a.device)
    ckpt = torch.load(os.path.join(a.run_dir, "ckpt.pt"), map_location=device)
    rid = ckpt["run_id"]
    if ckpt["it"] != rid["iters"]:
        raise SystemExit(f"checkpoint is at iteration {ckpt['it']} of {rid['iters']}: "
                         "training did not finish")
    # Checkpoints from before the config was saved fall back to the defaults.
    cfg = ckpt.get("cfg") or dict(METHODS[rid["method"]].DEFAULTS)

    S = D.load_shape(rid["shape"], rid["points"], rid["noise"], rid["seed"],
                     rid["single_sided"])
    P = S["points"]
    if hashlib.md5(np.ascontiguousarray(P).tobytes()).hexdigest() != rid["points_md5"]:
        raise SystemExit("regenerated input points do not match the checkpoint")

    net = build(rid["method"], cfg, device)
    net.load_state_dict(ckpt["net"])
    net.eval()

    mpath = os.path.join(a.run_dir, "metrics.json")
    old = json.load(open(mpath)) if os.path.exists(mpath) else {}
    for name in ("metrics.json", "recon.ply"):
        src = os.path.join(a.run_dir, name)
        dst = os.path.join(a.run_dir, name.replace(".", "_orig.", 1))
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    res = a.res or old.get("res") or 256
    print(f"[reextract] {a.run_dir}: {rid['method']} on {rid['shape']}, res {res}")
    m = extract_and_evaluate(net, rid["method"], P, S["mesh"], res, a.level,
                             device, a.run_dir)
    for k in ("sec", "iters", "points", "resumed_from_it"):
        if k in old:
            m[k] = old[k]
    m.update({"shape": rid["shape"], "method": rid["method"], "reextracted": True})
    json.dump(m, open(mpath, "w"), indent=2)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
