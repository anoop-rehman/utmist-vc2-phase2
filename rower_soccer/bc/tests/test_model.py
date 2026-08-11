"""The BC network: layout derivation, the frozen decoder, and the export fold.

The two tests worth reading twice are `test_fold_is_exact` and
`test_export_loads_through_skills_policy`. Between them they are the guarantee
that the thing written to disk is the thing that was trained: the observation
whitening is folded into the weights (so no deploy-time normaliser can be
forgotten) and the result satisfies `skills.policy.load_policy`'s p_idx/t_idx
check (so it cannot be loaded into the wrong body or the wrong layout in
silence). This repo has lost two runs to checkpoints that loaded "successfully"
into the wrong thing; these are the tests that stop the third.
"""

import os

import numpy as np
import pytest
import torch

from rower_soccer.bc import model as M
from rower_soccer.bc.dataset import build_dataset
from rower_soccer.bc.tests import fixtures as F
from rower_soccer.skills.registry import PROPRIO_V1

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def _cfg(tmp_path, **kw):
    keys, sizes = F.game_obs_layout()
    dec = kw.pop("decoder_path", None)
    if dec is None:
        dec = F.make_frozen_decoder(tmp_path / "dec.pt")
    return M.BCConfig(obs_keys=keys, obs_sizes=sizes, act_dim=8, z_dim=16,
                      decoder_path=dec, **kw)


# --- observation layout ----------------------------------------------------

def test_split_indices_puts_proprio_in_contract_order():
    keys, sizes = F.game_obs_layout()
    p, t = M.split_indices(keys, sizes)
    assert len(p) == F.PROPRIO_DIM
    assert len(p) + len(t) == sum(sizes)
    assert set(p).isdisjoint(t)
    # p must be PROPRIO_V1 order, NOT the observation's sorted-key order: the
    # decoder receives obs.index_select(-1, p_idx) and was trained on that
    # concatenation. The two orders differ for the ant (`bodies_pos` sorts
    # first either way, but `world_zaxis` sorts after `touch_sensors` and
    # `sensors_*` do not sort where the contract puts them).
    off, i = {}, 0
    for k, n in zip(keys, sizes):
        off[k] = i
        i += n
    want = []
    for f in PROPRIO_V1:
        want.extend(range(off[f], off[f] + F.PROPRIO_SIZES[f]))
    assert p == want
    assert t == sorted(t), "task indices should be in observation order"


def test_obs_layout_drop_keys():
    keys, sizes = F.game_obs_layout()
    k2, s2, cols = M.obs_layout(keys, sizes, drop_keys=["prev_action"])
    assert "prev_action" not in k2
    assert sum(s2) == sum(sizes) - 8
    assert cols.size == sum(s2)
    # the kept columns must be the original positions, in order
    assert np.all(np.diff(cols) > 0)
    with pytest.raises(ValueError, match="does not have"):
        M.obs_layout(keys, sizes, drop_keys=["no_such_key"])


def test_missing_proprio_is_refused():
    keys, sizes = F.game_obs_layout()
    with pytest.raises(ValueError, match="missing proprio fields"):
        M.split_indices([k for k in keys if k != "joints_pos"],
                        [s for k, s in zip(keys, sizes) if k != "joints_pos"])


# --- the frozen decoder ----------------------------------------------------

def test_decoder_is_loaded_and_stays_frozen(tmp_path):
    cfg = _cfg(tmp_path)
    pol = M.BCPolicy(cfg)
    dec, act, _, _ = M.load_frozen_decoder(cfg.decoder_path)
    for k, v in pol.ac.mlp_extractor.decoder.state_dict().items():
        assert torch.equal(v, dec[k]), f"decoder.{k} was not loaded"
    assert "mlp_extractor.decoder.0.weight" in pol.frozen_parameter_names
    assert "action_net.weight" in pol.frozen_parameter_names

    before = {k: v.detach().clone() for k, v in pol.ac.named_parameters()}
    opt = torch.optim.Adam(pol.trainable_parameters(), lr=1e-2)
    obs = torch.randn(16, pol.obs_dim)
    tgt = torch.randn(16, 8).clamp(-1, 1)
    for _ in range(3):
        opt.zero_grad()
        pol.losses(obs, tgt)["total"].backward()
        opt.step()
    for k, v in pol.ac.named_parameters():
        moved = not torch.equal(v.detach(), before[k])
        if k.startswith(("mlp_extractor.decoder", "action_net")) or k == "log_std":
            assert not moved, f"{k} moved but should be frozen"
    assert any(not torch.equal(pol.ac.state_dict()[k], before[k])
               for k in ("mlp_extractor.z_proj.weight",
                         "mlp_extractor.task_enc.0.weight"))


def test_train_decoder_flag_unfreezes(tmp_path):
    pol = M.BCPolicy(_cfg(tmp_path, freeze_decoder=False))
    assert "mlp_extractor.decoder.0.weight" not in pol.frozen_parameter_names


def test_freeze_without_decoder_is_refused():
    keys, sizes = F.game_obs_layout()
    cfg = M.BCConfig(obs_keys=keys, obs_sizes=sizes, decoder_path="",
                     freeze_decoder=True)
    with pytest.raises(ValueError, match="freeze random weights"):
        M.BCPolicy(cfg)


# --- normalisation and the export fold -------------------------------------

def test_fold_is_exact(tmp_path):
    """The exported (raw-observation) weights must reproduce the trained,
    whitened module. This is what lets the checkpoint be a plain ActorCritic."""
    torch.manual_seed(0)
    pol = M.BCPolicy(_cfg(tmp_path))
    fit = torch.randn(512, pol.obs_dim) * 7.0 + 3.0
    pol.set_normalization(fit)
    # perturb the encoders so the fold is not tested on the identity
    with torch.no_grad():
        for p in pol.trainable_parameters():
            p.add_(torch.randn_like(p) * 0.05)

    path = pol.export(tmp_path / "bc.pt")
    ac, meta = M.load_bc_checkpoint(path)
    obs = torch.randn(64, pol.obs_dim) * 7.0 + 3.0
    with torch.no_grad():
        a0, z0 = pol(obs)
        a1, z1 = ac.dist(obs).mean, ac.z(obs)
    assert torch.allclose(a0, a1, atol=1e-4), float((a0 - a1).abs().max())
    assert torch.allclose(z0, z1, atol=1e-4), float((z0 - z1).abs().max())
    assert meta["config"]["arch"] == "latent"
    assert meta["proprio_indices"] == pol.p_idx


def test_normalization_whitens_encoder_input_but_not_the_decoder(tmp_path):
    """The decoder is frozen and was trained on RAW proprio; whitening its input
    would be a silent 1e2 rescale of the motor controller's world."""
    pol = M.BCPolicy(_cfg(tmp_path))
    fit = torch.randn(256, pol.obs_dim) * 5.0 + 2.0
    pol.set_normalization(fit)
    ex = pol.ac.mlp_extractor
    prop, task = ex.split(fit)
    n_prop, n_task = ex._norm(prop, task)
    assert abs(float(n_prop.mean())) < 0.1 and abs(float(n_prop.std()) - 1) < 0.2
    assert abs(float(n_task.mean())) < 0.1 and abs(float(n_task.std()) - 1) < 0.2
    # forward_actor must hand the decoder the raw proprio
    obs = fit[:8]
    with torch.no_grad():
        z = ex.z(obs)
        want = ex.decoder(torch.cat([ex.split(obs)[0], z], -1))
        got = ex.forward_actor(obs)
    assert torch.allclose(want, got)


def test_constant_columns_are_not_amplified(tmp_path):
    pol = M.BCPolicy(_cfg(tmp_path))
    fit = torch.randn(128, pol.obs_dim)
    fit[:, 0] = 4.0                       # a dead column
    pol.set_normalization(fit)
    ex = pol.ac.mlp_extractor
    j = int((ex.p_idx == 0).nonzero()[0, 0]) if bool((ex.p_idx == 0).any()) else None
    scale = ex.p_scale[j] if j is not None else ex.t_scale[int((ex.t_idx == 0).nonzero()[0, 0])]
    assert float(scale) == 1.0


# --- the checkpoint contract ----------------------------------------------

def test_export_loads_through_skills_policy(tmp_path):
    """`skills.policy.load_policy` must accept a BC checkpoint given the layout
    from `bc_meta` — and reject it when the layout is wrong."""
    from rower_soccer.skills.api import CheckpointMismatch
    from rower_soccer.skills.policy import clear_policy_cache, load_policy

    pol = M.BCPolicy(_cfg(tmp_path))
    pol.set_normalization(torch.randn(64, pol.obs_dim))
    path = pol.export(tmp_path / "bc.pt")
    ac, meta = M.load_bc_checkpoint(path)

    clear_policy_cache()
    expert = load_policy(path, **M.load_policy_kwargs(meta))
    obs = np.random.randn(pol.obs_dim).astype(np.float32)
    a_expert, z_expert = expert.act(obs)
    with torch.no_grad():
        a_direct = ac.dist(torch.as_tensor(obs)[None]).mean.clamp(-1, 1)[0].numpy()
    assert np.allclose(a_expert, a_direct, atol=1e-5)
    assert z_expert.shape == (16,)

    kw = M.load_policy_kwargs(meta)
    kw["proprio_indices"] = list(range(len(kw["proprio_indices"])))
    clear_policy_cache()
    with pytest.raises(CheckpointMismatch, match="proprio"):
        load_policy(path, **kw)
    clear_policy_cache()


def test_checkpoint_rejects_foreign_files(tmp_path):
    dec = F.make_frozen_decoder(tmp_path / "dec.pt")
    with pytest.raises(ValueError, match="not a BC checkpoint"):
        M.load_bc_checkpoint(dec)


def test_export_is_atomic(tmp_path):
    pol = M.BCPolicy(_cfg(tmp_path))
    path = pol.export(tmp_path / "sub" / "bc.pt")
    assert os.path.exists(path) and not os.path.exists(path + ".tmp")


# --- losses ----------------------------------------------------------------

def test_latent_loss_ignores_nan_rows(tmp_path):
    """Mirrored rows carry z = NaN by design; the latent loss must skip them
    rather than propagate a NaN into every weight."""
    pol = M.BCPolicy(_cfg(tmp_path, loss="latent"))
    obs = torch.randn(32, pol.obs_dim)
    act = torch.randn(32, 8).clamp(-1, 1)
    z = torch.randn(32, 16)
    mask = torch.zeros(32, dtype=torch.bool)
    mask[:10] = True
    z[~mask] = float("nan")
    loss = pol.losses(obs, act, torch.nan_to_num(z), mask)
    assert torch.isfinite(loss["total"])
    loss["total"].backward()
    assert all(torch.isfinite(p.grad).all() for p in pol.trainable_parameters()
               if p.grad is not None)
    assert float(loss["latent_frac"]) == pytest.approx(10 / 32, abs=1e-6)


def test_loss_modes_select_the_right_term(tmp_path):
    for mode in ("action", "latent", "both"):
        pol = M.BCPolicy(_cfg(tmp_path, loss=mode))
        o = torch.randn(16, pol.obs_dim)
        a = torch.randn(16, 8).clamp(-1, 1)
        zz = torch.randn(16, 16)
        L = pol.losses(o, a, zz)
        if mode == "action":
            assert torch.equal(L["total"], L["action"])
        elif mode == "latent":
            assert torch.equal(L["total"], L["latent"])
        else:
            assert torch.allclose(L["total"], L["action"] + L["latent"])


def test_sample_weights_reweight_the_action_loss(tmp_path):
    pol = M.BCPolicy(_cfg(tmp_path))
    obs = torch.randn(8, pol.obs_dim)
    act = torch.randn(8, 8).clamp(-1, 1)
    w = torch.zeros(8)
    w[:4] = 1.0
    got = pol.losses(obs, act, weight=w)["action"]
    want = pol.losses(obs[:4], act[:4])["action"]
    assert torch.allclose(got, want, atol=1e-6)


def test_calibrate_log_std_matches_the_residual(tmp_path):
    pol = M.BCPolicy(_cfg(tmp_path))
    obs = torch.randn(256, pol.obs_dim)
    act = torch.randn(256, 8).clamp(-1, 1)
    rms = pol.calibrate_log_std(obs, act)
    assert torch.allclose(pol.ac.log_std.exp(), rms, atol=1e-6)
    with torch.no_grad():
        resid = (pol.ac.dist(obs).mean - act).pow(2).mean(0).sqrt()
    assert torch.allclose(rms, resid, atol=1e-5)


# --- the plain control arm -------------------------------------------------

def test_plain_arch_roundtrip(tmp_path):
    keys, sizes = F.game_obs_layout()
    cfg = M.BCConfig(obs_keys=keys, obs_sizes=sizes, arch="plain",
                     decoder_path="")
    assert cfg.freeze_decoder is False
    pol = M.BCPolicy(cfg)
    pol.set_normalization(torch.randn(128, pol.obs_dim) * 3 + 1)
    with torch.no_grad():
        for p in pol.trainable_parameters():
            p.add_(torch.randn_like(p) * 0.02)
    path = pol.export(tmp_path / "plain.pt")
    ac, meta = M.load_bc_checkpoint(path)
    obs = torch.randn(32, pol.obs_dim) * 3 + 1
    with torch.no_grad():
        assert torch.allclose(pol(obs)[0], ac.dist(obs).mean, atol=1e-4)
    assert pol(obs)[1] is None
    # a plain policy has no bottleneck, so SkillController must refuse it
    from rower_soccer.skills.api import CheckpointMismatch
    from rower_soccer.skills.policy import load_policy
    with pytest.raises(CheckpointMismatch, match="SimpleActorCritic"):
        load_policy(path, proprio_indices=[0], task_indices=[1], act_dim=8)


def test_plain_arch_rejects_latent_loss():
    keys, sizes = F.game_obs_layout()
    with pytest.raises(ValueError, match="no latent bottleneck"):
        M.BCConfig(obs_keys=keys, obs_sizes=sizes, arch="plain", loss="latent")


# --- BCRunner (the deploy side) --------------------------------------------

def test_runner_assembles_by_key_not_by_position(tmp_path):
    pol = M.BCPolicy(_cfg(tmp_path))
    pol.set_normalization(torch.randn(64, pol.obs_dim))
    path = pol.export(tmp_path / "bc.pt")
    runner = M.BCRunner(path)
    keys, sizes = F.game_obs_layout()
    rng = np.random.default_rng(0)
    # a live dm_soccer obs dict: unordered, values with the leading singleton
    # buffer dim the soccer env keeps
    d = {k: rng.normal(size=(1, n)).astype(np.float32) for k, n in zip(keys, sizes)}
    shuffled = dict(sorted(d.items(), key=lambda kv: kv[0][::-1]))
    v = runner.obs_vector(shuffled)
    assert v.shape == (pol.obs_dim,)
    flat = np.concatenate([np.asarray(d[k], np.float32).ravel() for k in runner.obs_keys])
    assert np.array_equal(v, flat)
    a = runner.action(shuffled)
    assert a.shape == (8,) and np.all(np.abs(a) <= 1.0)
    assert runner.z(shuffled).shape == (16,)

    del shuffled["joints_pos"]
    with pytest.raises(ValueError, match="missing keys"):
        runner.obs_vector(shuffled)


def test_runner_honours_drop_keys(tmp_path):
    pol = M.BCPolicy(_cfg(tmp_path, drop_keys=["prev_action"]))
    assert "prev_action" not in pol.obs_keys
    assert pol.obs_dim == sum(F.game_obs_layout()[1]) - 8
    path = pol.export(tmp_path / "bc.pt")
    runner = M.BCRunner(path)
    assert "prev_action" not in runner.obs_keys
    keys, sizes = F.game_obs_layout()
    d = {k: np.zeros((1, n), np.float32) for k, n in zip(keys, sizes)}
    assert runner.obs_vector(d).shape == (pol.obs_dim,)


# --- the facts the whole design rests on -----------------------------------
#
# These read the real checkpoints and the real corpus. They are the reason
# `model.py` freezes the decoder and regresses the game observation onto z; if
# either stops holding, the architecture choice needs revisiting, and a silent
# pass is worse than a skip. They skip (loudly) when the artefacts are absent,
# e.g. in a fresh worktree with no runs_v2/.

_V3_CHECKPOINTS = ("runs_v2/follow_ant_final_frozen/best.pt",
                   "runs_v2/dribble_ant_v3/best.pt",
                   "runs_v2/kick_ant_v3/best.pt",
                   "runs_v2/shoot_ant_v3/best.pt")


def _repo(path):
    return os.path.join(REPO, path)


def test_registry_checkpoints_share_the_frozen_decoder():
    ref_path = _repo(M.DEFAULT_DECODER)
    have = [p for p in _V3_CHECKPOINTS if os.path.exists(_repo(p))]
    if not os.path.exists(ref_path) or not have:
        print("  [skip] runs_v2 checkpoints not present")
        return
    ref_dec, ref_act, _, _ = M.load_frozen_decoder(ref_path)
    for p in have:
        dec, act, _, _ = M.load_frozen_decoder(_repo(p))
        dd = max(float((dec[k] - ref_dec[k]).abs().max()) for k in ref_dec)
        da = max(float((act[k] - ref_act[k]).abs().max()) for k in ref_act)
        assert dd == 0.0 and da == 0.0, (
            f"{p} no longer shares _decoder_ant_final.pt (decoder {dd}, "
            f"action_net {da}). bc/train.py's --contract registry filter assumes "
            "'matches the registry field tuple' == 'was produced by the frozen "
            "decoder'; that assumption just broke.")


def test_frozen_decoder_reproduces_recorded_actions():
    """`clamp(action_net(decoder([proprio, z]))) == recorded action`, exactly.

    This is the fact that makes BC here a 186 -> 16 regression with an
    achievable target instead of a 186 -> 8 one that must re-learn locomotion.
    """
    import glob

    ref_path = _repo(M.DEFAULT_DECODER)
    demos = sorted(glob.glob(os.path.join(REPO, "demos", "*.demo.npz")))
    if not os.path.exists(ref_path) or not demos:
        print("  [skip] demos/ or the frozen decoder are not present")
        return
    from rower_soccer.bc.train import registry_layouts

    ds = build_dataset(demos[-1:], verbose=False)
    keep = registry_layouts(ds)
    ds = ds.select(np.isin(ds.arrays["layout"], keep))
    if len(ds) == 0:
        print("  [skip] no registry-contract samples in this demo")
        return

    off = ds.obs_offsets()
    cols = np.concatenate([np.arange(off[k].start, off[k].stop) for k in PROPRIO_V1])
    dec_sd, act_sd, _, _ = M.load_frozen_decoder(ref_path)
    dec = torch.nn.Sequential(torch.nn.Linear(65 + 16, 256), torch.nn.ELU(),
                              torch.nn.Linear(256, 256), torch.nn.ELU(),
                              torch.nn.Linear(256, 256), torch.nn.ELU())
    dec.load_state_dict(dec_sd)
    head = torch.nn.Linear(256, 8)
    head.load_state_dict(act_sd)

    n = min(4000, len(ds))
    prop = torch.as_tensor(ds.arrays["obs"][:n][:, cols], dtype=torch.float32)
    z = torch.as_tensor(ds.arrays["z"][:n], dtype=torch.float32)
    want = torch.as_tensor(ds.arrays["action"][:n], dtype=torch.float32)
    with torch.no_grad():
        got = head(dec(torch.cat([prop, z], -1))).clamp(-1, 1)
    err = float((got - want).abs().max())
    assert err < 1e-4, f"the frozen decoder no longer reproduces the demos: {err}"

    # ...and the game observation's proprio block IS the expert's, bit for bit.
    e = np.abs(ds.arrays["obs"][:n][:, cols] - ds.arrays["expert_obs"][:n, :65])
    assert float(np.nanmax(e)) == 0.0
