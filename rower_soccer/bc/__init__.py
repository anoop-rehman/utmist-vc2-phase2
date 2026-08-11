"""Behavior-cloning data layer: recorded 2v2 demos -> training tensors.

Three modules, deliberately layered so a trainer can import the cheap one:

    dataset.py   demos (`game/recording.py` .npz) -> one consolidated `BCDataset`
    augment.py   the pitch-mirror transform (obs, expert obs, action, world state)
    stats.py     `python -m rower_soccer.bc.stats` — what is actually in a dataset

`dataset` and `stats` import numpy only. `augment` additionally parses the
creature MJCF with the stdlib XML parser (no mujoco, no dm_control, no torch),
so building and augmenting a corpus never touches a simulator.

The demo format is FROZEN (see `game/recording.py`); everything here adapts to
it. In particular each demo carries its OWN `meta.skill_obs[skill]["fields"]`,
so a corpus that spans a change to a skill's observation contract loads without
complaint — the differing layouts get separate ids and a consumer selects one.
"""

from rower_soccer.bc.dataset import (BCDataset, build_dataset, load_dataset,
                                     SPLIT_TRAIN, SPLIT_VAL)

__all__ = ["BCDataset", "build_dataset", "load_dataset", "SPLIT_TRAIN", "SPLIT_VAL"]
