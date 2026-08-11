"""Train the BC policy on a demo corpus.

    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
    PYTHONPATH=. .venv/bin/python -m rower_soccer.bc.train \
        runs_v2/bc/ant_2v2_v1.npz -o runs_v2/bc/ant_action_v1 --epochs 200

    PYTHONPATH=. .venv/bin/python -m rower_soccer.bc.train demos/*.demo.npz \
        -o runs_v2/bc/ant_action_v1        # builds the dataset first

Writes to the output directory:

    best.pt        lowest validation action MSE seen (this is the one to use)
    final.pt       the last epoch, for resuming or for comparing to `best`
    config.json    every argument, the corpus fingerprint, the final metrics
    metrics.jsonl  one row per epoch

The split comes from the dataset and is by MATCH (`dataset.split_of_match`), so
validation is a match the trainer never saw — not a tick 25 ms away from one it
did. Everything reported here is on held-out matches.

Two filters are on by default and both are about honesty rather than taste
-------------------------------------------------------------------------

**`--contract registry` (default).** The corpus spans a change of drill
checkpoints. The three 2026-08-08 demos were played with `follow_ant_v1` and
`dribble_ant_v1`, whose `decoder`/`action_net` weights differ from the published
frozen decoder by up to 0.96 / 0.20 in absolute value — a *different motor
controller*. Their actions are therefore not reproducible by the frozen decoder
at any z, so including them would ask the expert head to chase 13,637 targets
that provably do not exist and would pollute every z target it does learn from.
The dataset already keys each sample to a layout id; the default keeps the
layouts whose field tuple matches the live registry, which is exactly the
shared-decoder set (34,261 of 47,898 samples). Pass `--contract all` to train on
everything and watch the val MSE floor rise.

**Mirror augmentation is OFF by default**, which is the opposite of what
`augment.py`'s existence suggests, so here is the measurement. The mirror itself
is correct — `test_mirror_physics` proves it against MuJoCo to 1e-15. What is
wrong is the assumption that it buys the FROZEN decoder anything. The decoder is
a learned net whose weights carry no symmetry, so a mirrored (proprio, action)
pair need not be realisable by it for any z. Two measurements, in increasing
order of authority:

  * optimising z directly against 2,048 sampled rows for 4,000 Adam steps
    reaches a median per-sample action MSE of 0.51 on mirrored rows against
    0.015 on unmirrored ones from the same initialisation — mirrored targets are
    largely out of the decoder's reach;
  * training it: `--mirror` doubles the corpus to 43,200 train rows and scores
    0.19862 on the UNMIRRORED validation rows, against 0.19798 without it. A
    wash, for 7x the wall clock (283 s vs 38 s). Its own mirrored val rows score
    0.2296, i.e. it does not fit them either.

So: no harm, no gain, real cost. `--mirror` is one flag away and worth
re-running the moment the decoder stops being frozen (`--train-decoder`), where
the symmetry argument does apply to weights that can actually move.

Note that `--mirror` changes what the headline `val action MSE` means, because
the mirrored validation rows are in it. `metrics.jsonl` and `config.json` carry
`unmirrored_action_mse` for exactly this comparison; use that one.
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import math
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from rower_soccer.bc.dataset import (BCDataset, SPLIT_TRAIN, SPLIT_VAL,
                                     build_dataset)
from rower_soccer.bc.model import BCConfig, BCPolicy

__all__ = ["train", "prepare", "select_corpus", "evaluate",
           "registry_layouts", "load_corpus", "Batches"]


# --- corpus selection ------------------------------------------------------

def registry_layouts(ds: BCDataset) -> List[int]:
    """Layout ids whose recorded field tuple matches the LIVE registry spec.

    A layout is `(skill, fields, width)` as the demo recorded it. The registry's
    current checkpoints are the ones that share `_decoder_ant_final.pt`, so
    "matches the registry contract" and "was produced by the frozen decoder" are
    the same set today. If that ever stops being true this function is where it
    breaks, loudly, rather than in a val curve.
    """
    from rower_soccer.skills import registry as R
    keep = []
    for lay in ds.meta["layouts"]:
        try:
            spec = R.get_spec(lay["skill"])
        except Exception:                                   # noqa: BLE001
            continue
        if tuple(spec.fields) == tuple(lay["fields"]):
            keep.append(int(lay["id"]))
    return sorted(keep)


def select_corpus(ds: BCDataset, *, contract: str = "registry",
                  layouts: Optional[Sequence[int]] = None,
                  controllers: Optional[Sequence[str]] = None,
                  verbose: bool = True) -> BCDataset:
    """Apply the layout / controller filters, reporting what each one cost."""
    n0 = len(ds)
    if layouts is not None:
        keep = [int(x) for x in layouts]
    elif contract == "registry":
        keep = registry_layouts(ds)
    elif contract == "all":
        keep = None
    else:
        raise ValueError(f"--contract must be 'registry' or 'all', got {contract!r}")
    if keep is not None:
        ds = ds.select(np.isin(ds.arrays["layout"], keep))
        if verbose:
            names = {int(l["id"]): f'{l["skill"]}/{l["obs_dim"]}'
                     for l in ds.meta["layouts"]}
            print(f"[bc] layouts {[names.get(k, k) for k in keep]}: "
                  f"{len(ds)}/{n0} samples kept", flush=True)
    if controllers:
        n1 = len(ds)
        ds = ds.with_controller(*controllers)
        if verbose:
            print(f"[bc] controllers {list(controllers)}: {len(ds)}/{n1} kept",
                  flush=True)
    if len(ds) == 0:
        raise ValueError("every sample was filtered out")
    return ds


# --- tensors ---------------------------------------------------------------

class Batches:
    """Shuffled minibatches over GPU-resident tensors.

    The whole corpus is ~35 MB in float32, so there is no reason for a DataLoader,
    worker processes, or a single host->device copy per batch. Permutation is
    done on the device with a seeded generator, so a run is reproducible.
    """

    def __init__(self, tensors: Dict[str, torch.Tensor], batch_size: int,
                 generator: Optional[torch.Generator] = None, shuffle: bool = True):
        self.t = tensors
        self.n = int(next(iter(tensors.values())).shape[0])
        self.bs = int(batch_size)
        self.g = generator
        self.shuffle = shuffle

    def __len__(self):
        return max(1, math.ceil(self.n / self.bs))

    def __iter__(self):
        dev = next(iter(self.t.values())).device
        order = torch.randperm(self.n, generator=self.g, device=dev) if self.shuffle \
            else torch.arange(self.n, device=dev)
        for i in range(0, self.n, self.bs):
            idx = order[i:i + self.bs]
            yield {k: v.index_select(0, idx) for k, v in self.t.items()}


def _to_tensors(ds: BCDataset, pol: BCPolicy, device: str,
                controller_weight: Optional[Dict[str, float]] = None):
    a = ds.arrays
    obs = torch.as_tensor(np.ascontiguousarray(a["obs"]), dtype=torch.float32,
                          device=device)
    obs = pol.take_obs(obs).contiguous()
    z = torch.as_tensor(np.ascontiguousarray(a["z"]), dtype=torch.float32,
                        device=device)
    out = dict(
        obs=obs,
        action=torch.as_tensor(np.ascontiguousarray(a["action"]),
                               dtype=torch.float32, device=device),
        z=torch.nan_to_num(z, nan=0.0),
        z_mask=torch.isfinite(z).all(-1),
        controller=torch.as_tensor(a["controller"].astype(np.int64), device=device),
        skill=torch.as_tensor(a["skill"].astype(np.int64), device=device),
        mirrored=torch.as_tensor(a["mirrored"].astype(np.int64), device=device),
    )
    if controller_weight:
        w = np.ones(len(ds), np.float32)
        for name, val in controller_weight.items():
            w[a["controller"] == ds.controller_vocab.index(name)] = float(val)
        out["weight"] = torch.as_tensor(w, device=device)
    return out


# --- evaluation during training -------------------------------------------

@torch.no_grad()
def evaluate(pol: BCPolicy, t: Dict[str, torch.Tensor], vocab, chunk: int = 8192) -> dict:
    """Action / latent agreement on a tensor block, plus per-slice breakdowns."""
    pol.eval()
    n = t["obs"].shape[0]
    se = torch.zeros(t["action"].shape[1], device=t["obs"].device)
    zse = torch.zeros((), device=t["obs"].device)
    zn = torch.zeros((), device=t["obs"].device)
    per_row = torch.empty(n, device=t["obs"].device)
    for i in range(0, n, chunk):
        o = t["obs"][i:i + chunk]
        a = t["action"][i:i + chunk]
        pred, zp = pol(o)
        d = (pred.clamp(-1.0, 1.0) - a) ** 2
        se += d.sum(0)
        per_row[i:i + chunk] = d.mean(1)
        if zp is not None:
            m = t["z_mask"][i:i + chunk]
            if m.any():
                zse += ((zp[m] - t["z"][i:i + chunk][m]) ** 2).sum()
                zn += m.sum() * zp.shape[1]
    pol.train()
    mse = float(per_row.mean())
    var = float(t["action"].var(0, unbiased=False).mean())
    out = dict(n=int(n), action_mse=mse, action_rmse=mse ** 0.5,
               per_actuator_mse=[float(x) for x in (se / max(n, 1)).cpu()],
               explained=1.0 - mse / var if var > 0 else float("nan"),
               latent_mse=float(zse / zn) if float(zn) > 0 else None)
    for key, names in (("controller", vocab["controller"]), ("skill", vocab["skill"])):
        col = t[key]
        d = {}
        for i, name in enumerate(names):
            m = col == i
            if bool(m.any()):
                d[name] = dict(n=int(m.sum()), action_mse=float(per_row[m].mean()))
        out[f"by_{key}"] = d
    m = t["mirrored"] == 0
    if bool(m.any()) and not bool(m.all()):
        out["unmirrored_action_mse"] = float(per_row[m].mean())
        out["mirrored_action_mse"] = float(per_row[~m].mean())
    return out


# --- the training run ------------------------------------------------------

def prepare(ds: BCDataset, args) -> tuple:
    """(policy, train tensors, val tensors, vocab) — everything but the loop."""
    device = args.device
    cfg = BCConfig.from_dataset(
        ds, drop_keys=list(args.drop_keys), arch=args.arch, loss=args.loss,
        z_loss_weight=args.z_loss_weight,
        freeze_decoder=(args.arch == "latent" and not args.train_decoder),
        decoder_path=(args.decoder if args.arch == "latent" else ""))
    pol = BCPolicy(cfg, device=device)

    tr = ds.select(ds.arrays["split"] == SPLIT_TRAIN)
    va = ds.select(ds.arrays["split"] == SPLIT_VAL)
    if len(tr) == 0 or len(va) == 0:
        raise ValueError(f"split is degenerate: {len(tr)} train / {len(va)} val. "
                         "The corpus needs at least two matches.")
    cw = dict(kv.split("=") for kv in args.controller_weight) if args.controller_weight \
        else None
    cw = {k: float(v) for k, v in cw.items()} if cw else None
    t_tr = _to_tensors(tr, pol, device, cw)
    t_va = _to_tensors(va, pol, device, None)
    # Whitening is fit on TRAIN only — fitting it on the whole corpus leaks the
    # validation matches' statistics into the model, which is a small leak but a
    # free one to avoid.
    pol.set_normalization(t_tr["obs"])
    vocab = dict(controller=list(ds.controller_vocab), skill=list(ds.skill_vocab))
    return pol, t_tr, t_va, vocab, tr, va


def train(ds: BCDataset, args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pol, t_tr, t_va, vocab, tr, va = prepare(ds, args)
    device = args.device
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    opt = torch.optim.AdamW(pol.trainable_parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs), eta_min=args.lr * args.lr_final_frac)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    batches = Batches({k: v for k, v in t_tr.items()}, args.batch_size, gen)

    print(f"[bc] {len(tr)} train / {len(va)} val samples   obs {pol.obs_dim} "
          f"(proprio {len(pol.p_idx)} task {len(pol.t_idx)})   arch {args.arch} "
          f"loss {args.loss}   trainable {pol.n_trainable():,} params", flush=True)
    if pol.decoder_source:
        print(f"[bc] decoder {'FROZEN' if pol.cfg.freeze_decoder else 'trainable'} "
              f"from {pol.decoder_source}", flush=True)

    hist_path = os.path.join(out_dir, "metrics.jsonl")
    hist = open(hist_path, "w")
    best = dict(epoch=-1, action_mse=float("inf"))
    bad = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        pol.train()
        tot, nb = 0.0, 0
        parts: Dict[str, float] = {}
        for b in batches:
            loss = pol.losses(b["obs"], b["action"], b["z"], b["z_mask"],
                              b.get("weight"))
            opt.zero_grad(set_to_none=True)
            loss["total"].backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(pol.trainable_parameters(),
                                               args.max_grad_norm)
            opt.step()
            tot += float(loss["total"])
            for k in ("action", "latent"):
                if k in loss:
                    parts[k] = parts.get(k, 0.0) + float(loss[k])
            nb += 1
        sched.step()

        val = evaluate(pol, t_va, vocab)
        trn = evaluate(pol, t_tr, vocab) if args.eval_train else None
        row = dict(epoch=epoch, lr=float(sched.get_last_lr()[0]),
                   train_loss=tot / max(nb, 1),
                   train_parts={k: v / max(nb, 1) for k, v in parts.items()},
                   val=val, train_eval=trn, seconds=time.time() - t0)
        hist.write(json.dumps(row, default=float) + "\n")
        hist.flush()

        improved = val["action_mse"] < best["action_mse"] - args.min_delta
        if improved:
            best = dict(epoch=epoch, **{k: v for k, v in val.items()})
            bad = 0
            pol.export(os.path.join(out_dir, "best.pt"),
                       extra=dict(epoch=epoch, val=val, args=vars(args)))
        else:
            bad += 1
        if epoch % args.log_every == 0 or improved or epoch == args.epochs - 1:
            lat = "" if val["latent_mse"] is None else f" val_z {val['latent_mse']:.4f}"
            print(f"  ep {epoch:4d}  train {tot / max(nb, 1):.5f}  "
                  f"val_a {val['action_mse']:.5f}  expl {val['explained']:+.3f}"
                  f"{lat}  lr {sched.get_last_lr()[0]:.2e}"
                  f"{'  *' if improved else ''}", flush=True)
        if bad >= args.patience:
            print(f"[bc] early stop at epoch {epoch}: no val improvement in "
                  f"{args.patience} epochs (best epoch {best['epoch']}, "
                  f"MSE {best['action_mse']:.5f})", flush=True)
            break
    hist.close()

    # Calibrate log_std on the VAL residual, then write final.pt. The mean is
    # unchanged by this; only the sampled policy is, which is what a later
    # KL-anchored fine-tune will measure against.
    rms = pol.calibrate_log_std(t_va["obs"], t_va["action"])
    final = evaluate(pol, t_va, vocab)
    pol.export(os.path.join(out_dir, "final.pt"),
               extra=dict(epoch="final", val=final, args=vars(args),
                          action_std=[float(x) for x in rms.cpu()]))

    summary = dict(
        out=out_dir, args=vars(args), seconds=time.time() - t0,
        corpus=dict(total=len(ds), train=len(tr), val=len(va),
                    train_matches=sorted({d["file"] for d in ds.meta["demos"]
                                          if d.get("split") == "train"}),
                    val_matches=sorted({d["file"] for d in ds.meta["demos"]
                                        if d.get("split") == "val"}),
                    layouts=[f'{l["id"]}:{l["skill"]}/{l["obs_dim"]}'
                             for l in ds.meta["layouts"]],
                    dataset_created=ds.meta.get("created_utc"),
                    augmentation=ds.meta.get("augmentation")),
        best=best, final=final,
        action_std=[float(x) for x in rms.cpu()],
        decoder=pol.decoder_source, frozen=list(pol.frozen_parameter_names))
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print(f"[bc] best epoch {best['epoch']}  val action MSE {best['action_mse']:.5f}"
          f"  (RMSE {best['action_mse'] ** 0.5:.4f}, explained "
          f"{best['explained']:+.3f})", flush=True)
    print(f"[bc] wrote {out_dir}/best.pt final.pt config.json metrics.jsonl",
          flush=True)
    return summary


# --- CLI -------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", nargs="+",
                   help="a built dataset .npz, or demo files/globs to build one from")
    p.add_argument("-o", "--out", required=True, help="output directory")
    p.add_argument("--arch", default="latent", choices=["latent", "plain"],
                   help="'latent': expert head -> z -> frozen shared decoder. "
                        "'plain': an unstructured MLP, the control arm.")
    p.add_argument("--loss", default="action", choices=["action", "latent", "both"])
    p.add_argument("--z-loss-weight", type=float, default=1.0)
    p.add_argument("--decoder", default="runs_v2/_decoder_ant_final.pt")
    p.add_argument("--train-decoder", action="store_true",
                   help="unfreeze the shared decoder (then the mirror argument in "
                        "the module docstring no longer applies — try --mirror)")
    p.add_argument("--contract", default="registry", choices=["registry", "all"])
    p.add_argument("--layouts", default=None,
                   help="explicit comma-separated layout ids, overriding --contract")
    p.add_argument("--controllers", default=None,
                   help="keep only these controllers, e.g. 'human'")
    p.add_argument("--controller-weight", action="append", default=[],
                   metavar="NAME=W", help="loss weight per controller, e.g. human=2")
    p.add_argument("--drop-keys", default="",
                   help="observation keys to drop, e.g. 'prev_action'")
    p.add_argument("--mirror", action="store_true",
                   help="append the pitch-mirrored corpus (see the module docstring "
                        "for why this is off by default)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-final-frac", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--min-delta", type=float, default=1e-5)
    p.add_argument("--eval-train", action="store_true",
                   help="also score the training split each epoch (the "
                        "train/val gap is the overfitting read)")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--val-fraction", type=float, default=0.25,
                   help="only used when building a dataset from demo files")
    return p


def load_corpus(paths: Sequence[str], args, verbose: bool = True) -> BCDataset:
    files: List[str] = []
    for pat in paths:
        files.extend(sorted(_glob.glob(pat)) or [pat])
    if len(files) == 1 and not files[0].endswith(".demo.npz"):
        ds = BCDataset.load(files[0])
    else:
        ds = build_dataset(files, val_fraction=args.val_fraction, verbose=verbose)
    ds = select_corpus(ds, contract=args.contract,
                       layouts=([int(x) for x in args.layouts.split(",")]
                                if args.layouts else None),
                       controllers=(args.controllers.split(",")
                                    if args.controllers else None),
                       verbose=verbose)
    if args.mirror:
        from rower_soccer.bc.augment import mirror_dataset
        n0 = len(ds)
        ds = mirror_dataset(ds, append=True)
        if verbose:
            print(f"[bc] mirror augmentation: {n0} -> {len(ds)} samples", flush=True)
    return ds


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.drop_keys = [k for k in args.drop_keys.split(",") if k]
    ds = load_corpus(args.data, args)
    train(ds, args)


if __name__ == "__main__":
    main()
