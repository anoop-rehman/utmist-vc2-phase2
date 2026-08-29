"""D3 M3 E0: does the skeleton stage explore when it starts from a good body?

Two measurements per checkpoint, on THEIR ant (`design_opt/cfg/ant.yml`,
5 bodies / 4 motors -- NOT the 13-body DeepMind ant D1 and D2 use):

  1. **Sampled diversity.** `topology_census.census` -- reused, not
     reimplemented -- draws N designs the way a training epoch draws them and
     counts distinct topologies. A topology is the SET OF BODY NAMES, which is
     a complete tree identifier: `xml_robot.py:317-321` names the root '0' and
     every child `str(sibling_index) + parent_name`, so the name is the path
     from the root and the name set is the tree.

  2. **The mean-action design**, which is what eval runs and what a render
     shows. Deterministic given the weights: the design-stage observation does
     not depend on the reset noise (the skeleton and attribute stages touch no
     physics), so re-running it at the same checkpoint gives the same body.
     Recorded as a topology plus the demapped attribute genome, so consecutive
     epochs and different seeds can be differenced.

Writes one JSON per (cfg, epoch) so the cross-seed comparison in `--compare`
never has to re-run sampling.

    cd /workspace/Transform2Act && source env-gpu.sh
    .venv-gpu/bin/python .../t2a_port/e0_analyse.py \
        --cfg ant_e0_s1 --epochs 10,20,30,40,50,60,70,80,90,100 --episodes 200
    .venv-gpu/bin/python .../t2a_port/e0_analyse.py --compare \
        --cfgs ant_e0_s1,ant_e0_s2,ant_e0_s3 --epoch 100

CPU only -- it opens no CUDA context, so it is safe beside live MPS clients.
"""

import argparse
import json
import os
import sys

sys.path.append("/workspace/Transform2Act")
sys.path.append("/workspace/utmist-vc2-phase2")
os.chdir("/workspace/Transform2Act")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rower_soccer.t2a_port.topology_census import census, topo_key  # noqa: E402

OUT = "/workspace/utmist-vc2-phase2/runs/d3_e0_ant/census"

# The five attribute-genome columns, in the order `Body.get_params` emits them
# for a non-root body on ant.yml (`offset_x, offset_y, gear, size, ext_start`),
# each demapped to roughly [-1, 1] by the `sin` param_mapping.
GENOME_COLS = ["offset_x", "offset_y", "gear", "size", "ext_start"]


def build_agent(cfg_id, epoch):
    from design_opt.agents.transform2act_agent import Transform2ActAgent
    from design_opt.utils.config import Config
    cfg = Config(cfg_id, tmp=False)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    return Transform2ActAgent(cfg=cfg, dtype=torch.float64,
                              device=torch.device("cpu"), seed=cfg.seed,
                              num_threads=1, training=False,
                              checkpoint=(0 if epoch == 0 else int(epoch)))


def tensorfy(np_list):
    if isinstance(np_list[0], list):
        return [[torch.tensor(x) for x in y] for y in np_list]
    return [torch.tensor(y) for y in np_list]


def mean_action_design(agent):
    """Run ONE episode's design stages with mean_action=True and describe the
    body that comes out, in both genome and physical units."""
    env = agent.env
    state = env.reset()
    for _ in range(agent.cfg.skel_transform_nsteps + 1):
        with torch.no_grad():
            a = agent.policy_net.select_action(
                tensorfy([state]), True).numpy().astype(np.float64)
        state, _, done, info = env.step(a)
        if done or info.get("stage") == "execution":
            break
    key, names = topo_key(env)
    genome = {b.name: env.get_attr_design()[i].tolist()
              for i, b in enumerate(env.robot.bodies)}
    # Physical units, read off the robot rather than recomputed: capsule radius
    # and length in metres, actuator gear in the XML's own units.
    phys = {}
    for b in env.robot.bodies:
        g = b.geoms[0]
        # The root's geom is a SPHERE (`ant.xml`, `<geom size="0.25"
        # type="sphere">`) and carries no `fromto`, so `Geom.start`/`.end`
        # exist only on capsules. Every limb is a capsule.
        length = (float(np.linalg.norm(np.asarray(g.end) - np.asarray(g.start)))
                  if g.type == "capsule" else 0.0)
        gear = (float(b.joints[0].actuator.gear)
                if b.joints and b.joints[0].actuator is not None else None)
        phys[b.name] = {"radius": float(np.asarray(g.size).reshape(-1)[0]),
                        "length": length, "gear": gear,
                        "depth": int(b.depth),
                        "type": str(g.type)}
    return {"topo": key, "names": list(names), "genome": genome, "phys": phys,
            "n_bodies": len(names), "n_motors": int(env.model.nu)}


def analyse(cfg_id, epochs, episodes):
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for ep in epochs:
        agent = build_agent(cfg_id, ep)
        topos, n_bodies, per_ep, all_idx, names_of = census(
            agent, episodes, mean_action=False)
        top_key, top_n = topos.most_common(1)[0]
        ma = mean_action_design(agent)
        row = {
            "cfg": cfg_id, "epoch": ep, "episodes": episodes,
            "distinct_topologies": len(topos),
            "top_topology": top_key,
            "top_topology_count": top_n,
            "top_topology_names": list(names_of[top_key]),
            "top5": [[k, c] for k, c in topos.most_common(5)],
            "body_count_hist": {str(k): v for k, v in sorted(n_bodies.items())},
            "mean_action": ma,
        }
        rows.append(row)
        json.dump(row, open(f"{OUT}/{cfg_id}_e{ep:04d}.json", "w"), indent=1)
        print(f"{cfg_id} epoch {ep:4d}  distinct {len(topos):4d}/{episodes}  "
              f"top {top_n:4d} ({100*top_n/episodes:.1f}%)  "
              f"mean-action topo {ma['topo']} "
              f"{ma['n_bodies']} bodies / {ma['n_motors']} motors  "
              f"[{','.join(ma['names'])}]", flush=True)
    return rows


def load(cfg_id, epoch):
    return json.load(open(f"{OUT}/{cfg_id}_e{epoch:04d}.json"))


def genome_matrix(rows_by_seed):
    """Pooled per-column std over every body of every seed's mean-action
    design, used to standardise the cross-seed difference. Standardising by a
    pooled std rather than by the parameter's own bound is what makes a
    difference in `gear` comparable to a difference in `size`."""
    vals = []
    for r in rows_by_seed:
        for name, g in r["mean_action"]["genome"].items():
            if name != "0":            # the root has no gear and pads zeros
                vals.append(g)
    v = np.asarray(vals)
    return v.std(axis=0, ddof=1) if len(v) > 1 else np.ones(v.shape[-1])


def compare(cfgs, epoch):
    rows = [load(c, epoch) for c in cfgs]
    print(f"\n=== cross-seed comparison at epoch {epoch} ===")
    for c, r in zip(cfgs, rows):
        ma = r["mean_action"]
        print(f"  {c}: topo {ma['topo']}  {ma['n_bodies']} bodies "
              f"{ma['n_motors']} motors  names [{','.join(ma['names'])}]")
    sd = genome_matrix(rows)
    sd = np.where(sd > 0, sd, 1.0)
    print(f"\n  pooled per-column std (standardiser): " +
          ", ".join(f"{n} {s:.3f}" for n, s in zip(GENOME_COLS, sd)))

    n = len(cfgs)
    print("\n  topology identity matrix (SAME / DIFF):")
    for i in range(n):
        print("   " + " ".join(
            ("SAME" if rows[i]["mean_action"]["topo"]
             == rows[j]["mean_action"]["topo"] else "DIFF") for j in range(n)))

    print("\n  attribute distance = mean over SHARED bodies and 5 columns of "
          "|delta| / pooled_std")
    print("   (n_shared bodies in brackets; '-' where no body name is shared)")
    for i in range(n):
        cells = []
        for j in range(n):
            gi = rows[i]["mean_action"]["genome"]
            gj = rows[j]["mean_action"]["genome"]
            shared = [k for k in gi if k in gj and k != "0"]
            if not shared:
                cells.append("    -    ")
                continue
            d = np.abs(np.asarray([gi[k] for k in shared])
                       - np.asarray([gj[k] for k in shared])) / sd
            cells.append(f"{d.mean():.3f}[{len(shared)}]")
        print(f"   {cfgs[i]:<12} " + " ".join(f"{c:>10}" for c in cells))

    print("\n  physical summary of each mean-action design:")
    for c, r in zip(cfgs, rows):
        ph = r["mean_action"]["phys"]
        limbs = {k: v for k, v in ph.items() if k != "0"}
        rad = [v["radius"] for v in limbs.values()]
        ln = [v["length"] for v in limbs.values()]
        gr = [v["gear"] for v in limbs.values() if v["gear"] is not None]
        dep = [v["depth"] for v in limbs.values()]
        print(f"   {c}: {len(limbs)} limbs, depths {sorted(set(dep))}  "
              f"radius {min(rad):.3f}-{max(rad):.3f} m  "
              f"length {min(ln):.3f}-{max(ln):.3f} m  "
              f"gear {min(gr):.0f}-{max(gr):.0f} (bounds 20-400)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg")
    p.add_argument("--epochs", default="10,20,30,40,50,60,70,80,90,100")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--compare", action="store_true")
    p.add_argument("--cfgs", default="ant_e0_s1,ant_e0_s2,ant_e0_s3")
    p.add_argument("--epoch", type=int, default=100)
    args = p.parse_args()
    torch.set_default_dtype(torch.float64)
    if args.compare:
        compare(args.cfgs.split(","), args.epoch)
    else:
        analyse(args.cfg, [int(e) for e in args.epochs.split(",")],
                args.episodes)


if __name__ == "__main__":
    main()
