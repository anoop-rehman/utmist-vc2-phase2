"""D3: how many DISTINCT morphologies does one epoch of sampling produce?

`TRANSFORM2ACT_PORT_MAP.md` section 7 offers two ways to run a batch whose worlds
have different skeletons:

  A. compile one superset model and mask the inactive bodies -- fast, but a
     "disabled" body that still has mass or contact geometry silently changes
     the physics of a body plan that should not contain it;
  B. group worlds by topology and compile one model per distinct topology --
     exact by construction, but the cost scales with how many distinct
     topologies there actually are.

The port map says to measure before choosing, because if a batch contains tens
of topologies rather than thousands, B is both simpler and safer. This measures
it, on the live run's own checkpoint.

It also answers the second open question from section 10: the real number of
distinct `body_index` values in a batch, which decides whether `IndexLinear`'s
loop should be batched (the sweep found the loop WINS at few indices and loses
badly at many).

Only the design stages are run -- they involve no physics, so this is cheap.

    cd /workspace/Transform2Act && source env-gpu.sh
    .venv-gpu/bin/python /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/topology_census.py \
        --cfg hopper_gpu --epoch 400 --episodes 200
"""

import argparse
import collections
import hashlib
import os
import sys

sys.path.append("/workspace/Transform2Act")
os.chdir("/workspace/Transform2Act")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from design_opt.agents.transform2act_agent import Transform2ActAgent  # noqa: E402
from design_opt.utils.config import Config  # noqa: E402
from khrylib.utils.torch import *  # noqa: E402,F403


def tensorfy(np_list, device=torch.device("cpu")):
    """`transform2act_agent.tensorfy`, copied so this file does not depend on
    the agent module exporting it."""
    if isinstance(np_list[0], list):
        return [[torch.tensor(x).to(device) for x in y] for y in np_list]
    return [torch.tensor(y).to(device) for y in np_list]


def topo_key(env):
    """A topology is the set of body names, which encode tree position. The XML
    string is NOT the key -- it also carries the continuous attributes, which
    differ every episode and would make every topology look unique."""
    names = tuple(sorted(b.name for b in env.robot.bodies))
    return hashlib.md5("|".join(names).encode()).hexdigest()[:12], names


def census(agent, episodes, mean_action):
    env = agent.env
    topos = collections.Counter()
    n_bodies = collections.Counter()
    per_episode_indices = []
    all_indices = set()
    # D3 M3 E0 needs the NAMES behind a topology hash, not only its count, to
    # say what the winning body plan actually is. Filled here so E0's analysis
    # reuses this function rather than reimplementing the sampling loop.
    names_of = {}

    for _ in range(episodes):
        state = env.reset()
        # Design stages only: `skel_transform_nsteps` skeleton steps then one
        # attribute step. Stepping into execution would cost physics for nothing.
        for _ in range(agent.cfg.skel_transform_nsteps + 1):
            with torch.no_grad():
                action = agent.policy_net.select_action(
                    tensorfy([state]), mean_action).numpy().astype(np.float64)
            state, _, done, info = env.step(action)
            if done:
                break
            if info.get("stage") == "execution":
                break
        key, names = topo_key(env)
        names_of[key] = names
        topos[key] += 1
        n_bodies[len(names)] += 1
        idx = set(env.get_body_index().tolist())
        per_episode_indices.append(len(idx))
        all_indices |= idx

    return topos, n_bodies, per_episode_indices, all_indices, names_of


def report(tag, episodes, topos, n_bodies, per_ep_idx, all_idx, names_of=None):
    print(f"\n=== {tag}: {episodes} designs ===")
    print(f"distinct topologies      {len(topos):4d}  "
          f"({100 * len(topos) / episodes:.1f}% of designs are unique)")
    top = topos.most_common(5)
    print("  most common:            " +
          ", ".join(f"{c} x{n}" for c, n in top))
    if names_of is not None and top:
        print("  most common body names: " + ",".join(names_of[top[0][0]]))
    print("  body count histogram:   " +
          ", ".join(f"{k} bodies: {v}" for k, v in sorted(n_bodies.items())))
    print(f"distinct body_index values, whole batch  {len(all_idx):4d}"
          f"   (JSMLP max_index is 256)")
    print(f"  per design: mean {np.mean(per_ep_idx):.1f}  max {max(per_ep_idx)}")
    print(f"  range {min(all_idx)}..{max(all_idx)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="hopper_gpu")
    p.add_argument("--epoch", default="400")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--untrained", action="store_true",
                   help="skip the checkpoint and census a freshly initialised "
                        "policy. The trained checkpoints available here start "
                        "at epoch 100, by which point the skeleton has already "
                        "converged; an untrained policy upper-bounds the "
                        "diversity a port would have to survive early on.")
    args = p.parse_args()

    torch.set_default_dtype(torch.float64)
    cfg = Config(args.cfg, tmp=False)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    epoch = 0 if args.untrained else (int(args.epoch) if args.epoch.isnumeric()
                                      else args.epoch)
    agent = Transform2ActAgent(cfg=cfg, dtype=torch.float64,
                               device=torch.device("cpu"), seed=cfg.seed,
                               num_threads=1, training=False, checkpoint=epoch)

    print(f"cfg {args.cfg}  checkpoint {'UNTRAINED' if args.untrained else epoch}  "
          f"skel_transform_nsteps {cfg.skel_transform_nsteps}  "
          f"max_body_depth {cfg.max_body_depth}  "
          f"enable_remove {cfg.enable_remove}")

    # Sampling is what a training epoch does; mean_action is what eval does. The
    # port has to survive the diversity of the former.
    for tag, ma in (("SAMPLED (as in training)", False),
                    ("MEAN ACTION (as in eval)", True)):
        report(tag, args.episodes, *census(agent, args.episodes, ma))


if __name__ == "__main__":
    main()
