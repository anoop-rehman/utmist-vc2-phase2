"""D3 E4B: one clip, four panels -- the current creature racing its own past
selves at increasing age gaps.

The point is to SHOW the ratchet gradient the report gives as numbers: a
near-coin-flip against a recent self, a dominant win against an old one.

What makes it honest rather than decorative:

  * THE SAME EPISODE SEEDS IN ALL FOUR PANELS. The only thing that differs
    between panels is the opponent's age. If each panel rolled its own seeds
    the viewer would be comparing four unrelated episodes and the visual claim
    would be unsupported.
  * The MEDIAN episode by return is shown, not the best, so a panel cannot be
    won by cherry-picking.
  * Each label carries the age gap, the outcome, and BOTH displacements
    (`dx` and `oppdx`). The opponent's displacement is what lets a reader
    confirm the opponent is real and moving -- an inert body reads ~0.35 m.
"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, "/workspace/Transform2Act")
sys.path.insert(0, "/workspace/utmist-vc2-phase2")
torch.set_default_dtype(torch.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="rtg_e4r_s2")
    p.add_argument("--ckpt", default=None,
                   help="learner checkpoint BASENAME in models/, e.g. epoch_0200")
    p.add_argument("--learner-ring-epoch", type=int, default=None,
                   help="load the LEARNER from ring/policy_XXXX.p instead. "
                        "models/ only keeps a few checkpoints (the rest are "
                        "archived to GCS and pruned), so the ring holds the "
                        "most recent available self -- 280 against a current "
                        "289, versus 200 in models/.")
    p.add_argument("--current-epoch", type=int, required=True,
                   help="epoch the learner is at, for the age-gap labels")
    p.add_argument("--opponents", default="280,240,200,0")
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=7)
    p.add_argument("--seed-base", type=int, default=6100)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--width", type=int, default=340)
    p.add_argument("--height", type=int, default=190)
    p.add_argument("--max-frames", type=int, default=170)
    p.add_argument("--stride", type=int, default=3)
    a = p.parse_args()

    from design_opt.agents.transform2act_agent import Transform2ActAgent
    from design_opt.utils.config import Config
    from design_opt.models.transform2act_policy import Transform2ActPolicy
    from khrylib.robot.xml_robot import Robot
    from khrylib.utils.torch import to_test
    from rower_soccer.t2a_port import e2_eval
    from rower_soccer.t2a_port import e4r_ring as R
    import pickle

    cfg = Config(a.cfg, tmp=False)
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    ag = Transform2ActAgent(cfg=cfg, dtype=torch.float64,
                            device=torch.device("cpu"), seed=cfg.seed,
                            num_threads=1, training=False,
                            checkpoint=str(a.ckpt) if a.ckpt else 0)
    import pickle as _pk
    if a.learner_ring_epoch is not None:
        _rp = os.path.join(cfg.cfg_dir, "ring",
                           "policy_%04d.p" % a.learner_ring_epoch)
        ag.policy_net.load_state_dict(
            _pk.load(open(_rp, "rb"))["policy_dict"], strict=True)
        print("  learner <- ring/policy_%04d.p" % a.learner_ring_epoch)
    to_test(ag.policy_net)
    env = ag.env
    env.ring_epoch = None                      # fixed opponent per panel
    rd = os.path.join(cfg.cfg_dir, "ring")
    act, wrap = e2_eval.gnn_actor(ag.policy_net, ag.running_state, True)

    panels, outcomes = [], []
    for opp in [int(x) for x in a.opponents.split(",")]:
        body = os.path.join(rd, "body_%04d.xml" % opp)
        scene = os.path.join(rd, "scene_%04d.xml" % opp)
        pol = Transform2ActPolicy(cfg.policy_specs, ag)
        pol.load_state_dict(pickle.load(
            open(os.path.join(rd, "policy_%04d.p" % opp), "rb"))["policy_dict"])
        pol.eval()
        for q in pol.parameters():
            q.requires_grad_(False)
        R._install(env, dict(merged_path=scene, body_path=body,
                             robot=Robot(cfg.robot_cfg, xml=body), policy=pol))
        env.set_opponent_policy(pol)
        # Score both sides at their mean action, as the tournament does: a
        # stochastic opponent would hand slot 0 a systematic edge.
        keep = env.opp_mean_action
        env.opp_mean_action = True
        try:
            # pass 1: rank on the SHARED seeds, no rendering
            stats = [e2_eval.roll(env, act, wrap, a.seed_base + i,
                                  max_steps=env.max_nsteps + 5)
                     for i in range(a.episodes)]
            keep_idx = [i for i, s in enumerate(stats) if s is not None]
            rs = [stats[i] for i in keep_idx]
            med = keep_idx[int(np.argsort([s["R"] for s in rs])[len(rs) // 2])]
            # The gradient this clip exists to show lives in the WIN RATE, not
            # in any one episode: a single median episode against a coin-flip
            # opponent is still a win half the time, and the first render of
            # this figure showed WIN on all four panels for exactly that
            # reason. The panel shows a representative episode; the label
            # states the statistic.
            wr = float(np.mean([bool(x["reached"]) and not bool(x["opp_reached"])
                                for x in rs]))
            # pass 2: replay the median episode WITH rendering
            e = e2_eval.roll(env, act, wrap, a.seed_base + med,
                             max_steps=env.max_nsteps + 5, render=True,
                             camera="pitch", width=a.width, height=a.height,
                             max_frames=a.max_frames, stride=a.stride)
        finally:
            env.opp_mean_action = keep
        gap = a.current_epoch - opp
        odx = (abs(float(e["opp_com_x"]) - 1.0)
               if e.get("opp_com_x") is not None else float("nan"))
        tag = ("WIN" if e["reached"] else
               ("LOSS" if e["opp_reached"] else
                ("fell" if e["fell"] else "no finish")))
        txt = ("gap %d ep   win rate %.2f (n=%d)   this ep: %s  dx=%.2f  "
               "oppdx=%.2f" % (gap, wr, len(rs), tag, e["net_dx"], odx))
        panels.append([e2_eval.label(f, txt) for f in e["frames"]])
        outcomes.append((opp, gap, tag, e["net_dx"], odx, med, wr))
        print("  opponent e%-4d gap %3d  WIN RATE %.2f (n=%d)  | shown: "
              "seed %d median, %-9s dx %.2f m  oppdx %.2f m"
              % (opp, gap, wr, len(rs), a.seed_base + med, tag, e["net_dx"],
                 odx), flush=True)

    T = max(len(t) for t in panels)
    panels = [t + [t[-1]] * (T - len(t)) for t in panels]
    # 2x2 so the clip sits beside a chart rather than dominating the section
    frames = [np.concatenate(
        [np.concatenate([panels[0][j], panels[1][j]], axis=1),
         np.concatenate([panels[2][j], panels[3][j]], axis=1)], axis=0)
        for j in range(T)]
    import imageio
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    imageio.mimwrite(a.out, np.stack(frames), fps=a.fps,
                     macro_block_size=1, quality=6)
    print("\n  -> %s  (%.1f MB, %d frames, all panels on seeds %d..%d)"
          % (a.out, os.path.getsize(a.out) / 2**20, T,
             a.seed_base, a.seed_base + a.episodes - 1))


if __name__ == "__main__":
    main()
