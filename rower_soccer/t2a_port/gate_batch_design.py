"""Gate for `agent_specs.batch_design` -- the reference's stage-sorted minibatch.

    PYTHONPATH=. .venv/bin/python -m rower_soccer.t2a_port.gate_batch_design

Their `update_policy` (`design_opt/agents/transform2act_agent.py:272-297`)
shuffles the whole batch each optimisation epoch and then, when
`agent_specs.batch_design` is set -- which every hopper cfg they ship does --
re-sorts it by stage with `get_perm_batch_design`. Minibatches are consecutive
slices of that array, so each one is stage-PURE except at the two stage
boundaries.

`train_t2a.py` sliced a plain `randperm` instead, which makes every minibatch
stage-MIXED. That is not a wash. It changes, per optimisation epoch:

  * how many Adam steps each stage's tower takes (one per minibatch that
    contains any of its rows), and
  * how many rows each of those steps averages over.

Adam rescales a gradient by its own running RMS, so a tower that sees a
seventeenth of the rows does not take a seventeenth of a step -- it takes a
full-sized step from a seventeenth of the data, once per minibatch instead of
twice per epoch.

No GPU, no MuJoCo: this gates the permutation and the arithmetic only.
"""

import numpy as np
import torch

from rower_soccer.t2a_port.train_t2a import STAGE_RANK, stage_sorted_perm

PASS = FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{('   ' + detail) if detail else ''}")
    if ok:
        PASS += 1
    else:
        FAIL += 1


def their_perm(row_stage_np, rng):
    """`get_perm_batch_design` transcribed, to compare against ours."""
    shuffled = rng.permutation(row_stage_np.shape[0])
    inds = [[], [], []]
    for i in shuffled:
        inds[int(row_stage_np[i])].append(i)
    return np.array(inds[0] + inds[1] + inds[2])


def minibatch_stages(perm, row_stage, mini):
    n_mb = perm.shape[0] // mini
    return [set(row_stage[perm[i * mini:(i + 1) * mini]].tolist())
            for i in range(n_mb)]


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    mini = 2048

    # A real epoch-400 batch of `port_s1`, read off its JSON log line:
    # n_train_eps 568 -> 5 skeleton + 1 attribute row per episode, and
    # batch_steps 58,500 execution rows. `minibatches` in that same line is 30,
    # which is floor(61,908 / 2,048) -- so this composition is the trainer's
    # own, not a reconstruction.
    n_eps, n_exec = 568, 58500
    stages = np.concatenate([np.zeros(5 * n_eps), np.ones(n_eps),
                             np.full(n_exec, 2)]).astype(np.int64)
    rng.shuffle(stages)
    row_stage = torch.as_tensor(stages)
    n = row_stage.shape[0]
    check("batch composition matches the logged minibatch count",
          n // mini == 30, f"{n} rows -> {n // mini} minibatches, log says 30")

    perm = stage_sorted_perm(torch.randperm(n), row_stage)
    check("stage-sorted permutation is a permutation",
          torch.equal(perm.sort().values, torch.arange(n)))

    mixed = [s for s in minibatch_stages(perm, row_stage, mini) if len(s) > 1]
    check("at most two minibatches straddle a stage boundary",
          len(mixed) <= 2, f"{len(mixed)} mixed of {n // mini}")

    ours = minibatch_stages(perm, row_stage, mini)
    design_mb = [i for i, s in enumerate(ours) if s & {0, 1}]
    check("the design towers see only the first few minibatches",
          len(design_mb) == 2, f"{len(design_mb)} of {n // mini} minibatches "
          f"carry a design row")

    # Negative control: the behaviour this fix replaces.
    plain = minibatch_stages(torch.randperm(n), row_stage, mini)
    plain_design = [i for i, s in enumerate(plain) if s & {0, 1}]
    check("control: an unsorted permutation puts design rows in EVERY "
          "minibatch", len(plain_design) == len(plain),
          f"{len(plain_design)} of {len(plain)} -- {len(plain_design) / max(len(design_mb), 1):.0f}x "
          f"more Adam steps on the design towers per optimisation epoch")

    # Their construction, ours, same stage profile row for row.
    theirs = their_perm(stages, np.random.default_rng(1))
    a = row_stage[torch.as_tensor(theirs)]
    b = row_stage[stage_sorted_perm(torch.randperm(n), row_stage)]
    check("stage profile is identical to their get_perm_batch_design",
          torch.equal(a, b))

    # Within a stage the order must still be shuffled, or the minibatches
    # would be correlated across optimisation epochs.
    p1 = stage_sorted_perm(torch.randperm(n), row_stage)
    p2 = stage_sorted_perm(torch.randperm(n), row_stage)
    check("within a stage the order is still shuffled",
          not torch.equal(p1, p2)
          and torch.equal(row_stage[p1], row_stage[p2]))

    # And a control on the sort key itself.
    bad = stage_sorted_perm(torch.randperm(n), torch.zeros_like(row_stage))
    check("control: a constant sort key does NOT purify the minibatches",
          len([s for s in minibatch_stages(bad, row_stage, mini)
               if len(s) > 1]) == n // mini)

    check("stage ranks match hopper.if_use_transform_action's order",
          [STAGE_RANK[k] for k in ("skel_trans", "attr_trans", "execution")]
          == [0, 1, 2])

    print(f"\n{PASS}/{PASS + FAIL} checks passed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
