"""The rollout half of the eval harness — needs MuJoCo, so it lives in --slow.

What is under test is the HARNESS, not the policy: that the env is the same 2v2,
r=0.15-drill-ball, pinned-pitch, 40 Hz setup the demos were recorded in; that a
BC slot and a scripted slot both drive their player; that every behavioural
metric comes out in range; and that the video actually gets written. Whether a
particular checkpoint plays well is a judgement call made by watching that video,
not an assertion.
"""

import os

import numpy as np
import pytest
import torch

from rower_soccer.bc import eval as E
from rower_soccer.bc.model import BCPolicy, BCConfig
from rower_soccer.bc.tests import fixtures as F


def _policy(tmp_path, **kw):
    keys, sizes = F.game_obs_layout()
    cfg = BCConfig(obs_keys=keys, obs_sizes=sizes, act_dim=8, z_dim=16,
                   decoder_path=F.make_frozen_decoder(tmp_path / "dec.pt"), **kw)
    pol = BCPolicy(cfg)
    pol.set_normalization(torch.randn(64, pol.obs_dim))
    return pol, pol.export(tmp_path / "bc.pt")


def test_rollout_measures_behaviour(tmp_path):
    """A short 2v2 rollout with a real BC checkpoint against the scripted chase.

    Deliberately tiny (2 s), because what is being tested is the harness — that
    the env is the r=0.15 drill ball 2v2 the demos were recorded in, that both
    kinds of agent drive their slot, and that every metric is in range. Whether
    the POLICY is good is not a unit test; that is `--rollout --video`.
    """
    pol, path = _policy(tmp_path)
    out = E.rollout(checkpoint=path, home=("bc", "bc"),
                    away=(E.BASELINE_SKILL, E.BASELINE_SKILL),
                    seconds=2.0, seed=0)
    assert out["steps"] == int(round(2.0 * out["hz"]))
    assert out["hz"] == 40
    for side in ("home", "away"):
        b = out[side]
        assert 0.0 <= b["possession"] <= 1.0
        assert 0.0 <= b["close_possession"] <= b["possession"] + 1e-9
        assert 0.0 <= b["upright_frac"] <= 1.0
        assert b["touches"] >= 0 and b["walked_m"] >= 0.0
        assert b["mean_ball_distance"] > 0.0
    assert out["home"]["possession"] + out["away"]["possession"] == pytest.approx(1.0)
    assert out["home"]["agents"] == ["bc", "bc"]
    assert out["away"]["agents"] == [E.BASELINE_SKILL] * 2
    assert len(out["per_player"]["touches"]) == 4


def test_rollout_env_is_the_drill_ball_pitch():
    env = E.make_eval_env("ant", seed=0)
    assert len(env.task.players) == 4
    assert env.task.control_timestep == pytest.approx(0.025)
    r = float(env.physics.model.geom(
        env.physics.bind(env.task.ball.geom).element_id).size[0])
    assert r == pytest.approx(0.15, abs=1e-6), \
        "the eval env must use the r=0.15 drill ball every checkpoint trained on"
    arena = env.task.arena
    assert tuple(arena._min_size) == tuple(arena._max_size) == (15.0, 11.0)


def test_rollout_writes_a_video(tmp_path):
    pol, path = _policy(tmp_path)
    vid = tmp_path / "roll.mp4"
    out = E.rollout(checkpoint=path, seconds=1.0, seed=0, video=str(vid),
                    render_size=(160, 120), fps=10)
    assert out["video"] == str(vid)
    assert os.path.exists(vid) and os.path.getsize(vid) > 0


def test_compare_runs_both_arms(tmp_path):
    pol, path = _policy(tmp_path)
    cmp = E.compare(path, episodes=1, seconds=1.0)
    assert set(cmp) == {"bc", "baseline"}
    assert cmp["bc"]["episodes"][0]["home"]["agents"] == ["bc", "bc"]
    assert cmp["baseline"]["episodes"][0]["home"]["agents"] == [E.BASELINE_SKILL] * 2
    txt = E.format_rollout(cmp)
    assert "rollout" in txt and "baseline" in txt


def test_rollout_needs_a_checkpoint_for_a_bc_slot():
    with pytest.raises(ValueError, match="needs --checkpoint"):
        E.rollout(checkpoint=None, home=("bc", "bc"), seconds=0.1)
