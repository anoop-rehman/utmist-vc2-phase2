"""Shared fixtures for the trainer/eval tests: a game-shaped synthetic demo.

`synth.make_demo` writes a cut-down observation (root pose + the six pitch
landmarks) because that is all `dataset` and `augment` read. The BC model reads
more: it needs the whole `PROPRIO_V1` block, because that block is the frozen
decoder's input contract. So this module widens the synthetic observation to
include it, in the same sorted-key order a real demo has, and builds a throwaway
frozen decoder so no test depends on `runs_v2/` being present.
"""

import numpy as np

from rower_soccer.bc.tests import synth
from rower_soccer.skills.registry import PROPRIO_V1

#: The ant's real per-field widths (from a recorded demo's obs_sizes).
PROPRIO_SIZES = {
    "bodies_pos": 27, "body_height": 1, "joints_pos": 8, "joints_vel": 8,
    "sensors_accelerometer": 3, "sensors_gyro": 3, "sensors_velocimeter": 3,
    "touch_sensors": 9, "world_zaxis": 3,
}
PROPRIO_DIM = sum(PROPRIO_SIZES.values())          # 65

#: A couple of task-ish keys so the task block is not empty and looks like the
#: real thing (the real game obs has 121 task columns).
EXTRA_KEYS = {"ball_ego_position": 3, "prev_action": 8, "stats_vel_to_ball": 1}


def game_obs_layout():
    """(keys, sizes) sorted exactly as `recording.obs_layout` sorts them."""
    d = dict(zip(synth.OBS_KEYS, synth.OBS_SIZES))
    d.update(PROPRIO_SIZES)
    d.update(EXTRA_KEYS)
    keys = sorted(d)
    return keys, [d[k] for k in keys]


def make_game_demo(path, **kw):
    """`synth.make_demo` with the full proprio block in the observation."""
    keys, sizes = game_obs_layout()
    kw.setdefault("obs_keys", keys)
    kw.setdefault("obs_sizes", sizes)
    return synth.make_demo(path, **kw)


def make_corpus(tmp_path, n_matches=4, n_ticks=40, **kw):
    """`n_matches` synthetic demos, distinct match ids, in `tmp_path`."""
    return [make_game_demo(tmp_path / f"m{i}.demo.npz", match_id=f"match-{i}",
                           seed=i, n_ticks=n_ticks, **kw)
            for i in range(n_matches)]


def make_frozen_decoder(path, z_dim=16, act_dim=8, proprio_dim=PROPRIO_DIM,
                        task_dim=6, seed=0):
    """A throwaway `_decoder_ant_final.pt`-shaped export, so tests are hermetic."""
    import torch

    from rower_soccer.warp_port.ppo import ActorCritic, export_sb3_compatible

    torch.manual_seed(seed)
    ac = ActorCritic(obs_dim=proprio_dim + task_dim, act_dim=act_dim,
                     proprio_indices=list(range(proprio_dim)),
                     task_indices=list(range(proprio_dim, proprio_dim + task_dim)),
                     z_dim=z_dim)
    export_sb3_compatible(ac, str(path))
    return str(path)


def train_args(data, out, **over):
    """A real parsed `bc.train` argument namespace, then overridden.

    Going through `build_parser` on purpose: a test that hand-rolls a namespace
    stops catching the case where a new flag is added and `train()` starts
    reading it.
    """
    from rower_soccer.bc.train import build_parser

    a = build_parser().parse_args([str(data), "-o", str(out)])
    a.device = "cpu"
    a.epochs = 3
    a.batch_size = 64
    a.log_every = 10**6
    for k, v in over.items():
        setattr(a, k, v)
    if isinstance(a.drop_keys, str):
        a.drop_keys = [k for k in a.drop_keys.split(",") if k]
    return a
