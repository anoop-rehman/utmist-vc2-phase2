"""Behaviour cloning: recorded 2v2 demos -> a policy prior, and its evaluation.

Six modules, deliberately layered so each one imports only what it needs:

    dataset.py   demos (`game/recording.py` .npz) -> one consolidated `BCDataset`
    augment.py   the pitch-mirror transform (obs, expert obs, action, world state)
    stats.py     `python -m rower_soccer.bc.stats` — what is actually in a dataset
    model.py     the policy: game obs -> z -> the FROZEN shared decoder -> action
    train.py     `python -m rower_soccer.bc.train` — the loop, split, early stop
    eval.py      `python -m rower_soccer.bc.eval` — held-out agreement AND a
                 2v2 rollout against the scripted baseline, with video

`dataset` and `stats` import numpy only. `augment` additionally parses the
creature MJCF with the stdlib XML parser (no mujoco, no dm_control, no torch),
so building and augmenting a corpus never touches a simulator. `model` and
`train` add torch; only `eval --rollout` needs a simulator.

The one fact the model layer rests on, measured on the current corpus and
re-checked by `tests/test_model.py`: every v3 drill shares
`runs_v2/_decoder_ant_final.pt` byte for byte, the game observation contains
that decoder's entire 65-wide input, and therefore
``clamp(action_net(decoder([proprio, z])))`` reproduces the recorded action to
1.6e-6. Behaviour cloning here is a 186 -> 16 regression onto an achievable
target, not a 186 -> 8 one that has to re-learn locomotion.

The demo format is FROZEN (see `game/recording.py`); everything here adapts to
it. In particular each demo carries its OWN `meta.skill_obs[skill]["fields"]`,
so a corpus that spans a change to a skill's observation contract loads without
complaint — the differing layouts get separate ids and a consumer selects one.
"""

from rower_soccer.bc.dataset import (BCDataset, build_dataset, load_dataset,
                                     SPLIT_TRAIN, SPLIT_VAL)

__all__ = ["BCDataset", "build_dataset", "load_dataset", "SPLIT_TRAIN", "SPLIT_VAL"]
