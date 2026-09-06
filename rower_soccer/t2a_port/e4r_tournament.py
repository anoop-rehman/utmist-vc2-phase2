"""D3 M3 E4R: the win-rate MATRIX over archived selves, and non-transitivity.

Why a matrix and not "did it improve?"
--------------------------------------
Self-play is specifically prone to a failure a scalar cannot see:
**non-transitivity** -- iteration 30 beats 20, 20 beats 10, and 10 beats 30.
Every pairwise comparison looks like progress and there is no progress at all.
A monotone ratchet is a triangular matrix; a cycle shows up as a triple that
closes on itself.

Both slot orientations are played (i in slot 0 vs j in slot 1, then swapped)
and averaged, because the learner always trains in slot 0 and a slot advantage
would otherwise masquerade as skill. The gap between the two orientations is
reported as a diagnostic -- gate 3 says the pi-z rotation is exact, so it
should be small, and if it is not that is a finding.

Thresholds are pre-registered in `D3_E4R_SHARED.md` and calibrated by
simulation, not chosen:

  NON-TRANSITIVE   cyclic-triple fraction > 0.10
                   A perfectly transitive ladder of 12 players at 20 episodes
                   per ordered pair, with ADJACENT pairs at a near-tied 0.55
                   (the hardest case for binomial noise), produces cycles at
                   mean 0.013 / p95 0.036 / p99 0.055. A tournament with no
                   real ordering at all produces 0.136. 0.10 is ~1.8x the
                   noise ceiling and well under chance.
"""
import argparse, itertools, json, os, pickle, sys
import numpy as np
import torch

sys.path.insert(0, "/workspace/Transform2Act")
sys.path.insert(0, "/workspace/utmist-vc2-phase2")
torch.set_default_dtype(torch.float64)


def cyclic_fraction(beats):
    n = beats.shape[0]
    tot = cyc = 0
    for i, j, k in itertools.combinations(range(n), 3):
        tot += 1
        for a, b, c in ((i, j, k), (i, k, j)):
            if beats[a, b] and beats[b, c] and beats[c, a]:
                cyc += 1
                break
    return (cyc / tot) if tot else 0.0, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--n-ckpt", type=int, default=12)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from design_opt.agents.transform2act_agent import Transform2ActAgent
    from design_opt.utils.config import Config
    from design_opt.models.transform2act_policy import Transform2ActPolicy
    from khrylib.robot.xml_robot import Robot
    from khrylib.utils.torch import to_cpu, to_test
    from rower_soccer.t2a_port import e2_eval, rtg_scene
    from rower_soccer.t2a_port import e4r_ring as R

    # tmp=False: tmp=True points cfg_dir at /tmp/design_opt/<cfg>, where no ring
    # exists. The trainer writes the ring under results/<cfg>/ring via
    # Config(..., tmp=False), so the tournament must resolve the same path.
    cfg = Config(a.cfg, tmp=False)
    agent = Transform2ActAgent(cfg=cfg, dtype=torch.float64,
                               device=torch.device("cpu"), seed=0,
                               num_threads=1, training=False, checkpoint=0)
    env = agent.env
    ring_dir = os.path.join(cfg.cfg_dir, "ring")
    eps = sorted(int(f.split("_")[1].split(".")[0])
                 for f in os.listdir(ring_dir) if f.startswith("policy_"))
    eps = [e for e in eps if e >= 0]
    if len(eps) > a.n_ckpt:
        idx = np.linspace(0, len(eps) - 1, a.n_ckpt)
        eps = sorted({eps[int(round(i))] for i in idx})
    print("tournament over %d checkpoints: %s" % (len(eps), eps), flush=True)

    def mk():
        return Transform2ActPolicy(cfg.policy_specs, agent)

    mem, pol = {}, {}
    for e in eps:
        body = os.path.join(ring_dir, "body_%04d.xml" % e)
        merged = os.path.join(ring_dir, "scene_%04d.xml" % e)
        mem[e] = dict(merged_path=merged, body_path=body,
                      robot=Robot(cfg.robot_cfg, xml=body),
                      policy=None)
        p = mk()
        p.load_state_dict(pickle.load(open(
            os.path.join(ring_dir, "policy_%04d.p" % e), "rb"))["policy_dict"])
        p.eval()
        pol[e] = p
        mem[e]["policy"] = p

    n = len(eps)
    S = np.full((n, n), np.nan)          # score of row vs column, slot-averaged
    raw = {}
    env.ring_epoch = None                # never let the ring redraw here
    for ii, i in enumerate(eps):
        for jj, j in enumerate(eps):
            if i == j:
                continue
            R._install(env, mem[j])
            agent.policy_net.load_state_dict(pol[i].state_dict())
            with to_cpu(agent.policy_net), to_test(agent.policy_net):
                o = R._play(env, agent, e2_eval, a.episodes,
                            seed_base=70000 + 1000 * ii + jj)
            raw[f"{i}v{j}"] = o
            S[ii, jj] = o["score"]
            print("  %4d vs %4d: score %.3f win %.2f mutual %.2f stale %.2f"
                  % (i, j, o["score"], o["win_rate"], o["mutual_rate"],
                     o["stalemate_rate"]), flush=True)

    # slot-average: M[i][j] = (S[i][j] + (1 - S[j][i])) / 2
    M = np.full((n, n), np.nan)
    asym = []
    for x in range(n):
        for y in range(n):
            if x == y or np.isnan(S[x, y]) or np.isnan(S[y, x]):
                continue
            M[x, y] = 0.5 * (S[x, y] + (1.0 - S[y, x]))
            asym.append(abs(S[x, y] - (1.0 - S[y, x])))
    beats = np.zeros((n, n), dtype=bool)
    for x in range(n):
        for y in range(n):
            if x != y and not np.isnan(M[x, y]):
                beats[x, y] = M[x, y] > 0.5
    frac, ntri = cyclic_fraction(beats)

    # monotonicity: does score rise with the opponent's age?
    rows = []
    for x, e in enumerate(eps):
        older = [(e - eps[y], M[x, y]) for y in range(n)
                 if eps[y] < e and not np.isnan(M[x, y])]
        if older:
            rows.append(dict(epoch=e, mean_score_vs_older=float(
                np.mean([v for _, v in older])), n=len(older)))
    out = dict(cfg=a.cfg, checkpoints=eps, episodes=a.episodes,
               matrix_slot_averaged=[[None if np.isnan(v) else float(v)
                                      for v in r] for r in M],
               cyclic_triple_fraction=float(frac), n_triples=int(ntri),
               NON_TRANSITIVE=bool(frac > 0.10),
               slot_asymmetry_mean=float(np.mean(asym)) if asym else None,
               slot_asymmetry_max=float(np.max(asym)) if asym else None,
               score_vs_older=rows, raw=raw)
    p = a.out or os.path.join("/workspace/utmist-vc2-phase2/rower_soccer/docs/"
                              "t2a/e4r", f"tournament_{a.cfg}.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"), indent=1)
    print("\n  cyclic triples %.3f of %d  -> %s (threshold 0.10)"
          % (frac, ntri, "NON-TRANSITIVE" if frac > 0.10 else "transitive"))
    # `x or -1` turns a genuine 0.0 into the -1 sentinel, because 0.0 is falsy
    # -- so a PERFECT symmetry printed as an error value. Check for None.
    am, ax = out["slot_asymmetry_mean"], out["slot_asymmetry_max"]
    print("  slot asymmetry mean %s max %s (gate 3 says the rotation is exact, "
          "so this should be ~0)"
          % ("n/a" if am is None else "%.4f" % am,
             "n/a" if ax is None else "%.4f" % ax))
    print("  ->", p)


if __name__ == "__main__":
    main()
