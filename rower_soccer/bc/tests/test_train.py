"""The trainer: corpus selection, the match-level split, and the loop.

The split test is the one that matters. Consecutive demo ticks are 25 ms apart
and nearly identical, so a tick-level split measures memorisation and reports it
as generalisation — a mistake that would make every number in the final report
a lie. `dataset.py` splits by match; this checks the trainer never undoes that.
"""

import json
import os

import numpy as np
import pytest
import torch

from rower_soccer.bc import train as T
from rower_soccer.bc.dataset import SPLIT_TRAIN, SPLIT_VAL, build_dataset
from rower_soccer.bc.model import BCRunner, load_bc_checkpoint
from rower_soccer.bc.tests import fixtures as F
from rower_soccer.bc.tests import synth


def _corpus(tmp_path, n=4, ticks=40, **kw):
    return build_dataset(F.make_corpus(tmp_path, n_matches=n, n_ticks=ticks, **kw),
                         verbose=False)


def _args(tmp_path, ds_path, **kw):
    kw.setdefault("decoder", F.make_frozen_decoder(tmp_path / "dec.pt"))
    kw.setdefault("contract", "all")     # the synth skills are not registry skills
    out = kw.pop("out", str(tmp_path / "out"))
    return F.train_args(ds_path, out, **kw)


# --- corpus selection ------------------------------------------------------

def test_registry_layouts_drops_a_stale_contract(tmp_path):
    """A demo recorded before `follow` moved to the v3 field tuple must not be
    mixed in: it was played by a different checkpoint with a different decoder,
    so its actions are unreachable through the frozen one."""
    paths = [F.make_game_demo(tmp_path / "new.demo.npz", match_id="new", seed=0,
                              follow_fields=synth.FOLLOW_V3),
             F.make_game_demo(tmp_path / "old.demo.npz", match_id="old", seed=1,
                              follow_fields=synth.FOLLOW_V1)]
    ds = build_dataset(paths, verbose=False)
    widths = {int(l["obs_dim"]) for l in ds.meta["layouts"]}
    assert widths == {69, 71}, widths
    keep = T.registry_layouts(ds)
    kept = ds.select(np.isin(ds.arrays["layout"], keep))
    assert len(kept) > 0
    kept_widths = {int(l["obs_dim"]) for l in ds.meta["layouts"]
                   if int(l["id"]) in keep}
    assert kept_widths == {71}, "the 69-wide v1 contract should have been dropped"
    # ...and it is the same thing select_corpus does by default
    same = T.select_corpus(ds, contract="registry", verbose=False)
    assert len(same) == len(kept)
    assert len(T.select_corpus(ds, contract="all", verbose=False)) == len(ds)


def test_select_corpus_filters_controllers(tmp_path):
    ds = _corpus(tmp_path, n=2)
    got = T.select_corpus(ds, contract="all", controllers=["human"], verbose=False)
    assert len(got) < len(ds)
    assert set(np.unique(got.arrays["controller"])) == {
        ds.controller_vocab.index("human")}


def test_select_corpus_refuses_an_empty_result(tmp_path):
    ds = _corpus(tmp_path, n=2)
    with pytest.raises(ValueError, match="filtered out"):
        T.select_corpus(ds, layouts=[999], verbose=False)


def test_unknown_contract_is_refused(tmp_path):
    ds = _corpus(tmp_path, n=2)
    with pytest.raises(ValueError, match="registry"):
        T.select_corpus(ds, contract="whatever", verbose=False)


# --- batching --------------------------------------------------------------

def test_batches_cover_every_sample_exactly_once():
    n = 101
    t = dict(x=torch.arange(n).float()[:, None])
    g = torch.Generator()
    g.manual_seed(0)
    b = T.Batches(t, batch_size=16, generator=g)
    seen = torch.cat([bb["x"][:, 0] for bb in b])
    assert len(b) == 7
    assert sorted(seen.tolist()) == list(range(n))
    # and a second pass is a different order (it is shuffled, not cached)
    again = torch.cat([bb["x"][:, 0] for bb in b])
    assert not torch.equal(seen, again)


# --- the split -------------------------------------------------------------

def test_validation_matches_are_disjoint_from_training_ones(tmp_path):
    ds = _corpus(tmp_path, n=4)
    args = _args(tmp_path, tmp_path / "unused")
    pol, t_tr, t_va, vocab, tr, va = T.prepare(ds, args)
    tr_demos = set(np.unique(tr.arrays["demo"]).tolist())
    va_demos = set(np.unique(va.arrays["demo"]).tolist())
    assert tr_demos and va_demos and not (tr_demos & va_demos)
    # no tick appears on both sides
    key = lambda d: {(int(a), int(b), int(c)) for a, b, c in
                     zip(d.arrays["demo"], d.arrays["tick"], d.arrays["player"])}
    assert not (key(tr) & key(va))


def test_degenerate_split_is_refused(tmp_path):
    ds = _corpus(tmp_path, n=1)
    with pytest.raises(ValueError, match="degenerate"):
        T.prepare(ds, _args(tmp_path, tmp_path / "unused"))


def test_normalization_is_fit_on_train_only(tmp_path):
    ds = _corpus(tmp_path, n=4, ticks=60)
    pol, t_tr, t_va, _, tr, va = T.prepare(ds, _args(tmp_path, tmp_path / "u"))
    ex = pol.ac.mlp_extractor
    got = torch.cat([ex.p_mean[ex.p_idx.argsort()], ex.t_mean])  # order-free check
    want_tr = t_tr["obs"].mean(0)
    want_all = torch.cat([t_tr["obs"], t_va["obs"]]).mean(0)
    d_tr = float((ex.p_mean - want_tr[ex.p_idx]).abs().max())
    d_all = float((ex.p_mean - want_all[ex.p_idx]).abs().max())
    assert d_tr < 1e-4 < d_all or d_tr < d_all
    assert got.shape[0] == pol.obs_dim


# --- the loop --------------------------------------------------------------

def test_train_end_to_end_writes_a_loadable_checkpoint(tmp_path):
    ds = _corpus(tmp_path, n=4, ticks=60)
    out = tmp_path / "run"
    args = _args(tmp_path, tmp_path / "u", out=str(out), epochs=4)
    summary = T.train(ds, args)

    for name in ("best.pt", "final.pt", "config.json", "metrics.jsonl"):
        assert os.path.exists(out / name), name
    rows = [json.loads(l) for l in open(out / "metrics.jsonl")]
    assert len(rows) == 4
    assert all("val" in r and r["val"]["n"] > 0 for r in rows)
    assert summary["best"]["epoch"] >= 0
    assert summary["corpus"]["train"] > 0 and summary["corpus"]["val"] > 0

    ac, meta = load_bc_checkpoint(out / "best.pt")
    assert meta["config"]["arch"] == "latent"
    assert meta["decoder_source"].endswith("dec.pt")
    assert meta["critic_trained"] is False
    runner = BCRunner(out / "best.pt")
    assert runner.obs_dim == sum(F.game_obs_layout()[1])
    # final.pt's log_std is the calibrated residual, not the decoder's
    assert len(summary["action_std"]) == 8
    ac_f, _ = load_bc_checkpoint(out / "final.pt")
    assert torch.allclose(ac_f.log_std.exp(),
                          torch.tensor(summary["action_std"]), atol=1e-5)


def test_training_actually_reduces_the_loss(tmp_path):
    ds = _corpus(tmp_path, n=4, ticks=80)
    out = tmp_path / "run"
    args = _args(tmp_path, tmp_path / "u", out=str(out), epochs=12, lr=3e-3)
    T.train(ds, args)
    rows = [json.loads(l) for l in open(out / "metrics.jsonl")]
    assert rows[-1]["train_loss"] < rows[0]["train_loss"], \
        "the trainer is not learning anything at all on synthetic data"


def test_early_stopping_fires(tmp_path):
    ds = _corpus(tmp_path, n=4, ticks=60)
    out = tmp_path / "run"
    # patience 0 with min_delta huge: nothing can ever "improve", so it must
    # stop on the first epoch after the first.
    args = _args(tmp_path, tmp_path / "u", out=str(out), epochs=50,
                 patience=1, min_delta=1e9)
    T.train(ds, args)
    rows = [json.loads(l) for l in open(out / "metrics.jsonl")]
    assert len(rows) < 50


def test_best_checkpoint_is_the_best_epoch(tmp_path):
    ds = _corpus(tmp_path, n=4, ticks=60)
    out = tmp_path / "run"
    summary = T.train(ds, _args(tmp_path, tmp_path / "u", out=str(out), epochs=8))
    rows = [json.loads(l) for l in open(out / "metrics.jsonl")]
    best_row = min(rows, key=lambda r: r["val"]["action_mse"])
    assert summary["best"]["epoch"] == best_row["epoch"]
    _, meta = load_bc_checkpoint(out / "best.pt")
    assert meta["epoch"] == best_row["epoch"]
    assert meta["val"]["action_mse"] == pytest.approx(
        best_row["val"]["action_mse"], rel=1e-9)


def test_loss_latent_mode_runs(tmp_path):
    ds = _corpus(tmp_path, n=4, ticks=40)
    out = tmp_path / "run"
    T.train(ds, _args(tmp_path, tmp_path / "u", out=str(out), epochs=2,
                      loss="latent"))
    rows = [json.loads(l) for l in open(out / "metrics.jsonl")]
    assert rows[0]["train_parts"]["latent"] > 0


def test_plain_arch_runs(tmp_path):
    ds = _corpus(tmp_path, n=4, ticks=40)
    out = tmp_path / "run"
    T.train(ds, _args(tmp_path, tmp_path / "u", out=str(out), epochs=2,
                      arch="plain"))
    ac, meta = load_bc_checkpoint(out / "best.pt")
    assert meta["config"]["arch"] == "plain"


def test_mirror_augmentation_doubles_the_corpus_and_masks_z(tmp_path):
    ds = _corpus(tmp_path, n=4, ticks=40)
    ds_path = tmp_path / "ds.npz"
    ds.save(ds_path)
    args = _args(tmp_path, ds_path, mirror=True)
    big = T.load_corpus([str(ds_path)], args, verbose=False)
    assert len(big) == 2 * len(ds)
    m = big.arrays["mirrored"] == 1
    assert np.isnan(big.arrays["z"][m]).all()
    assert np.isfinite(big.arrays["z"][~m]).all()
    # the mirrored rows keep their split, so a val match cannot leak into train
    for v in (SPLIT_TRAIN, SPLIT_VAL):
        assert int((big.arrays["split"] == v).sum()) == \
            2 * int((ds.arrays["split"] == v).sum())
    pol, t_tr, t_va, vocab, tr, va = T.prepare(big, args)
    L = pol.losses(t_tr["obs"][:64], t_tr["action"][:64], t_tr["z"][:64],
                   t_tr["z_mask"][:64])
    assert torch.isfinite(L["total"])


def test_controller_weight_is_applied(tmp_path):
    ds = _corpus(tmp_path, n=4, ticks=40)
    args = _args(tmp_path, tmp_path / "u", controller_weight=["scripted=0"])
    pol, t_tr, *_ = T.prepare(ds, args)
    w = t_tr["weight"]
    sc = t_tr["controller"] == ds.controller_vocab.index("scripted")
    assert float(w[sc].max()) == 0.0 and float(w[~sc].min()) == 1.0


def test_evaluate_reports_slices(tmp_path):
    ds = _corpus(tmp_path, n=4, ticks=40)
    pol, t_tr, t_va, vocab, *_ = T.prepare(ds, _args(tmp_path, tmp_path / "u"))
    m = T.evaluate(pol, t_va, vocab)
    assert m["n"] == t_va["obs"].shape[0]
    assert len(m["per_actuator_mse"]) == 8
    assert m["latent_mse"] is not None
    assert set(m["by_controller"]) <= set(ds.controller_vocab)
    assert sum(v["n"] for v in m["by_skill"].values()) == m["n"]
