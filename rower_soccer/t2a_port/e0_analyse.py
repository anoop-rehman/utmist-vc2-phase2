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
        topos, n_bodies, per_ep, all_idx, names_of, genomes = census(
            agent, episodes, mean_action=False)
        # Pool every body of every sampled design. The root row is dropped: it
        # pads three of the five columns with zeros and would deflate them.
        pop = np.concatenate([g[1:] for g in genomes if len(g) > 1], axis=0)
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
            "sampled_genome_std": pop.std(axis=0, ddof=1).tolist(),
            "sampled_genome_rows": int(pop.shape[0]),
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


def genome_std(rows_by_seed):
    """The standardiser for the cross-seed design distance: the per-column
    standard deviation of the attribute genome over the SAMPLED population --
    every non-root body of every one of the ~200 designs each seed drew at this
    epoch, pooled across seeds by averaging variances.

    Not the std of the three mean-action designs. That was the first version
    and it is degenerate: at epoch 0 the attribute stage has barely moved
    `gear`, so its std across three designs is 1.6e-4 and a 2e-4 numerical
    difference between two seeds' identical limbs reads as a distance of 1.25.
    The sampled population is a real distribution (thousands of body rows) and
    is what "how different, in units of how much this search varies" means.

    Floored at 1e-3 -- one twentieth of a percent of a column's [-1, 1] range --
    so a column the search genuinely never touches cannot divide by zero."""
    var = np.mean([np.square(r["sampled_genome_std"]) for r in rows_by_seed],
                  axis=0)
    return np.maximum(np.sqrt(var), 1e-3)


def compare(cfgs, epoch):
    rows = [load(c, epoch) for c in cfgs]
    print(f"\n=== cross-seed comparison at epoch {epoch} ===")
    for c, r in zip(cfgs, rows):
        ma = r["mean_action"]
        print(f"  {c}: topo {ma['topo']}  {ma['n_bodies']} bodies "
              f"{ma['n_motors']} motors  names [{','.join(ma['names'])}]")
    sd = genome_std(rows)
    print(f"\n  standardiser = per-column std of the SAMPLED genome "
          f"population ({rows[0]['sampled_genome_rows']} body rows/seed): " +
          ", ".join(f"{n} {s:.4f}" for n, s in zip(GENOME_COLS, sd)))

    n = len(cfgs)
    print("\n  topology identity matrix (SAME / DIFF):")
    for i in range(n):
        print("   " + " ".join(
            ("SAME" if rows[i]["mean_action"]["topo"]
             == rows[j]["mean_action"]["topo"] else "DIFF") for j in range(n)))

    print("\n  attribute distance, TWO definitions, both over the body names "
          "the two designs SHARE:")
    print("   SMD   = mean |delta| / sampled-population std   (how different, "
          "in units of how much this search varies)")
    print("   range = mean |delta| / 2                        (fraction of a "
          "column's full [-1, 1] demapped range; never degenerate)")
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
            raw = np.abs(np.asarray([gi[k] for k in shared])
                         - np.asarray([gj[k] for k in shared]))
            cells.append(f"{(raw / sd).mean():.2f}/{(raw / 2).mean():.3f}"
                         f"[{len(shared)}]")
        print(f"   {cfgs[i]:<12} " + " ".join(f"{c:>16}" for c in cells))

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
