"""The eval harness, arithmetic half. The rollout lives in `test_eval_rollout`.

`test_agreement_is_zero_for_a_self_consistent_corpus` is the calibration: build
a corpus whose recorded action IS what the policy emits, and every agreement
number must come out perfect. Any bookkeeping error — a wrong column mapping, a
missing clamp, a transposed slice — moves it off perfect, and this is far easier
to debug than a plausible-looking 0.2 on real data.
"""

import os

import numpy as np
import pytest
import torch

from rower_soccer.bc import eval as E
from rower_soccer.bc.dataset import build_dataset
from rower_soccer.bc.model import BCPolicy, BCConfig, BCRunner
from rower_soccer.bc.tests import fixtures as F


def _policy(tmp_path, **kw):
    keys, sizes = F.game_obs_layout()
    cfg = BCConfig(obs_keys=keys, obs_sizes=sizes, act_dim=8, z_dim=16,
                   decoder_path=F.make_frozen_decoder(tmp_path / "dec.pt"), **kw)
    pol = BCPolicy(cfg)
    pol.set_normalization(torch.randn(64, pol.obs_dim))
    return pol, pol.export(tmp_path / "bc.pt")


def _corpus(tmp_path, n=3, ticks=30):
    return build_dataset(F.make_corpus(tmp_path, n_matches=n, n_ticks=ticks),
                         verbose=False)


# --- agreement -------------------------------------------------------------

def test_agreement_is_zero_for_a_self_consistent_corpus(tmp_path):
    pol, path = _policy(tmp_path)
    ds = _corpus(tmp_path)
    runner = BCRunner(path)
    cols = E._dataset_columns(runner, ds)
    with torch.no_grad():
        o = torch.as_tensor(ds.arrays["obs"][:, cols], dtype=torch.float32)
        pred = runner.ac.dist(o).mean.clamp(-1, 1).numpy()
        z = runner.ac.z(o).numpy()
    ds.arrays["action"] = pred.astype(np.float32)
    ds.arrays["z"] = z.astype(np.float32)

    a = E.agreement(runner, ds)
    assert a["n"] == len(ds)
    assert a["action_mse"] < 1e-10
    assert a["action_mae"] < 1e-5
    assert a["explained"] == pytest.approx(1.0, abs=1e-6)
    assert a["sign_agree"] == pytest.approx(1.0)
    assert a["latent_mse"] < 1e-10
    assert a["latent_n"] == len(ds)
    for v in a["by_controller"].values():
        assert v["action_mse"] < 1e-9


def test_agreement_of_a_zero_policy_explains_nothing(tmp_path):
    """A policy that always emits 0 has explained ~ -(mean^2/var) <= 0: the
    sanity floor every real number has to beat."""
    pol, path = _policy(tmp_path)
    with torch.no_grad():
        for p in pol.ac.parameters():
            p.zero_()
    path = pol.export(tmp_path / "zero.pt")
    ds = _corpus(tmp_path)
    a = E.agreement(BCRunner(path), ds)
    assert a["explained"] <= 0.05
    assert a["pred_saturated"] == 0.0


def test_agreement_slices_partition_the_corpus(tmp_path):
    pol, path = _policy(tmp_path)
    ds = _corpus(tmp_path)
    a = E.agreement(BCRunner(path), ds)
    for key in ("by_controller", "by_skill", "by_split"):
        assert sum(v["n"] for v in a[key].values()) == a["n"], key
    assert len(a["per_actuator_mse"]) == 8
    assert 0.0 <= a["sign_agree"] <= 1.0
    txt = E.format_agreement(a)
    assert "action MSE" in txt and "per actuator" in txt


def test_agreement_maps_columns_by_key(tmp_path):
    """The policy may have been trained on a SUBSET of the observation; the
    scorer has to pick those columns out by name, not assume a prefix."""
    pol, path = _policy(tmp_path, drop_keys=["prev_action"])
    ds = _corpus(tmp_path)
    runner = BCRunner(path)
    cols = E._dataset_columns(runner, ds)
    assert cols.size == pol.obs_dim < sum(F.game_obs_layout()[1])
    off, i = {}, 0
    for k, n in zip(ds.meta["obs_keys"], ds.meta["obs_sizes"]):
        off[k] = (i, i + n)
        i += n
    assert not set(range(*off["prev_action"])) & set(cols.tolist())
    a = E.agreement(runner, ds)
    assert np.isfinite(a["action_mse"])


def test_agreement_rejects_a_mismatched_dataset(tmp_path):
    pol, path = _policy(tmp_path)
    ds = _corpus(tmp_path)
    runner = BCRunner(path)
    runner.obs_keys = runner.obs_keys + ["not_a_key"]
    runner.obs_sizes = runner.obs_sizes + [3]
    with pytest.raises(ValueError, match="no observation keys"):
        E.agreement(runner, ds)


def test_agreement_reports_the_mirror_split(tmp_path):
    from rower_soccer.bc.augment import mirror_dataset
    pol, path = _policy(tmp_path)
    ds = mirror_dataset(_corpus(tmp_path), append=True)
    a = E.agreement(BCRunner(path), ds)
    assert set(a["by_mirrored"]) == {"original", "mirrored"}
    assert a["by_mirrored"]["original"]["n"] == a["by_mirrored"]["mirrored"]["n"]
    # z is NaN on mirrored rows, so the latent slice must cover half the corpus
    assert a["latent_n"] == a["n"] // 2


def test_agreement_refuses_an_empty_dataset(tmp_path):
    pol, path = _policy(tmp_path)
    ds = _corpus(tmp_path)
    with pytest.raises(ValueError, match="empty"):
        E.agreement(BCRunner(path), ds.select(np.zeros(len(ds), bool)))
