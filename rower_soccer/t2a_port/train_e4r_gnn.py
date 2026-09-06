"""D3 M3 E4R: SHARED-WEIGHT self-play for Transform2Act on run-to-goal.

ONE agent -- one design head, one controller -- plays both sides of the 1v1.
Its opponent is a past self drawn from a ring of archived checkpoints
(`e4r_ring.py`), redrawn at every episode reset.

Derived from `train_e4_gnn.py` (two independent lineages), which is archived
under `docs/t2a/e4_twolineage_archive/` and was not found to be wrong -- the
redirect is to a cleaner question, not away from a broken one. E3.1's finding
that the **design head is blind** (skeleton and attribute stages see only
`attr_fixed ++ attr_design`, never simulation state) makes observation-
conditioned specialisation impossible by construction, which handicaps a
divergence study but is irrelevant to a shared-weight ratchet -- and this is
half the compute.

THE SUCCESS CRITERION AND ITS TRAP
----------------------------------
Each iteration should beat all past iterations, and be roughly tied against
its current self. Those two halves CONFLICT at equilibrium: if current ties
current, both reach the line on the same step, `n_reached == 2`, and
`run_to_goal.py` scores `parse = 0`. **The +/-1000 sparse term switches itself
off exactly at the point we are trying to reach.** That is why the training
opponent is a strictly-past self and never the current one -- beating a weaker
past self is where the +/-1000 pays.

It is also why "tied" has to be split three ways in the mirror match. A 0-0
stalemate and a 1-1 race both read as "tied" on any scalar:

    DECISIVE   exactly one reached          -> parse = +/-1000
    MUTUAL     both reached, same step      -> parse = 0, and this is the GOOD tie
    STALEMATE  neither reached (timeout)    -> parse = 0, and this is DEGENERATE

`mirror_match` reports all three plus forward progress, so the two cannot be
confused.

The learner is always agent 0; the past self plays slot 1 through the pi-z
rotation gated in `gate_e4.py` (11/11, observation max|delta| 0.000e+00).
"""


import argparse
import json
import os
import subprocess
import sys
import time

# wandb lives beside `.venv-gpu` (which has none) and needs protobuf >= 4,
# while the venv pins 3.x and tensorboardX's generated code needs 3.x's C
# extension. Both are settled BEFORE any import that could pull protobuf in.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
if os.path.isdir("/workspace/t2a_pylibs"):
    sys.path.insert(0, "/workspace/t2a_pylibs")
sys.path.append("/workspace/Transform2Act")
sys.path.append("/workspace/utmist-vc2-phase2")
os.chdir("/workspace/Transform2Act")

import numpy as np  # noqa: E402
import torch  # noqa: E402

RENDER_DIR = "/workspace/utmist-vc2-phase2/runs/d3_e3_adversarial/renders"
# D3 M3 E3.1 renders go to their own directory. E3's render dir already mixes
# design-ON and frozen-control clips that differ by one letter of cfg name and
# misled a reader once (see runs/d3_e3_adversarial/renders/INDEX.md); a third
# family in the same directory would be worse.
RENDER_DIR_E31 = "/workspace/utmist-vc2-phase2/runs/d3_e31_fix/renders"
# ...and E4 gets its own for the same reason: rtg_e4_s1a vs rtg_e31_s1 differ
# by two characters, and E4's clips show TWO moving creatures where E3's show
# one moving and one sliding. Mixing them is the exact confusion INDEX.md
# records.
RENDER_DIR_E4 = "/workspace/utmist-vc2-phase2/runs/d3_e4r_ring/renders"
VIDEO_CKPT = "_video_tmp"


# ------------------------------------------------------------ curriculum --
def alpha_at(epoch, curriculum_steps, batch):
    """CompetEvo's dense-weight, bit-identical to `train_e11_mlp.alpha`.

    `(cs - done)/cs`, NOT `1 - done/cs`: algebraically the same and not the
    same in float64. `gate_e21.py` phase 2 caught the difference at 1.1e-16 and
    the code was changed rather than the gate, so E3's alpha trajectory is the
    same float64 sequence E2.1's `d2rep` arm was trained on.

    Returns None -- not 1.0 -- when the curriculum is OFF, and that None is
    what leaves `custom_reward` unset and the flat env reward in the buffer.
    """
    if not curriculum_steps:
        return None
    return max((curriculum_steps - epoch * batch) / curriculum_steps, 0.0)


def make_custom_reward(agent):
    """khrylib's `custom_reward(env, state, action, env_reward, info)` hook.

    Returns the CURRICULUM reward, which `sample_worker` puts in the PPO
    buffer. `LoggerRL.step` is handed `env_reward` separately and logs that, so
    every statistic this trainer and `e2_eval` report stays the raw env return
    in both conditions -- a curriculum arm is measured on exactly the
    instrument a flat arm is.

    The design stages return reward 0 and carry no `dense`/`parse`, and the
    exception path in `RunToGoalEnv.step` returns 0 as well; `dense + parse ==
    reward` on every path, so the defaults below make those steps 0 under any
    alpha.
    """
    zero_info = np.array([0.0])

    def custom_reward(env, state, action, env_reward, info):
        a = agent.cur_alpha
        if a is None:
            return env_reward, zero_info
        dn = float(info.get("dense", env_reward))
        pa = float(info.get("parse", 0.0))
        return a * dn + (1.0 - a) * pa, zero_info
    return custom_reward


# --------------------------------------------------------------- archive --
def archive(model_dir, bucket, tag, log):
    """Push checkpoints to GCS and prune locally. 157 MB per GNN checkpoint on
    a 13 GB disk is the constraint; `epoch_0100`-multiples and `best.p` are
    uploaded, and everything but the two most recent and `best.p` is removed
    locally ONLY after `gsutil cp` reports success and the remote size matches.
    """
    if not bucket:
        return
    try:
        cps = sorted(f for f in os.listdir(model_dir)
                     if f.startswith("epoch_") and f.endswith(".p"))
        keep = set(cps[-2:]) | {"best.p"}
        for f in cps + (["best.p"] if os.path.exists(
                os.path.join(model_dir, "best.p")) else []):
            src = os.path.join(model_dir, f)
            dst = f"{bucket.rstrip('/')}/{tag}/{f}"
            r = subprocess.run(["gsutil", "-q", "cp", src, dst],
                               capture_output=True, text=True, timeout=1800)
            if r.returncode != 0:
                log(f"  archive FAILED {f}: {r.stderr.strip()[-200:]}")
                continue
            st = subprocess.run(["gsutil", "stat", dst], capture_output=True,
                                text=True, timeout=300)
            size = None
            for line in st.stdout.splitlines():
                if "Content-Length" in line:
                    size = int(line.split(":")[1].strip())
            if size != os.path.getsize(src):
                log(f"  archive SIZE MISMATCH {f}: {size} vs "
                    f"{os.path.getsize(src)} -- kept locally")
                continue
            if f not in keep:
                os.remove(src)
                log(f"  archived+pruned {f}")
    except Exception as e:
        log(f"  archive FAILED ({e!r})")


# ------------------------------------------------------------------ main --
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    p.add_argument("--num-threads", type=int, default=10)
    p.add_argument("--torch-threads", type=int, default=0,
                   help="torch.set_num_threads for THIS process. 0 leaves it "
                        "alone. It matters only for a CPU-only arm: khrylib's "
                        "`agent.py` sets OMP_NUM_THREADS=1 at import and "
                        "`env-gpu.sh` sets it again, so the PPO update runs "
                        "single-threaded -- measured at >700 s per epoch on "
                        "this box against 150 s for the same update on the "
                        "GPU. Setting the env var after torch is imported does "
                        "not move torch's pool; this does. The samplers are "
                        "unaffected: they are separate processes.")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--epoch", default="0", help="resume from this checkpoint")
    p.add_argument("--max-epoch", type=int, default=0,
                   help="override cfg.max_epoch_num; 0 = use the cfg")
    p.add_argument("--curriculum-steps", type=int, default=0,
                   help="CompetEvo's exploration curriculum: the PPO buffer "
                        "gets alpha*dense + (1-alpha)*parse with "
                        "alpha = max((cs - epoch*batch)/cs, 0). 0 = OFF, the "
                        "raw env reward. E2.1's d2rep value is 130208333.")
    p.add_argument("--stop-file", default=None)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-project", default="creature-soccer")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--morph-every", type=int, default=1)
    p.add_argument("--morph-episodes", type=int, default=20)
    p.add_argument("--video-every", type=int, default=6)
    p.add_argument("--video-episodes", type=int, default=9)
    p.add_argument("--archive-every", type=int, default=50)
    p.add_argument("--archive-bucket",
                   default="gs://vc2-2026-checkpoints/_t2a_archive")
    p.add_argument("--opp-refresh", type=int, default=10,
                   help="epochs between opponent snapshot exchanges. 10 is "
                        "justified in D3_E4_PREREQ.md: E3.1's within-lineage "
                        "drift over a 10-epoch lag is SMD 0.185, so the "
                        "opponent is meaningfully stale but not a different "
                        "body (40-epoch lag 0.305; between-seed null 0.89).")
    p.add_argument("--ring-every", type=int, default=10,
                   help="archive the current self into the ring every N "
                        "epochs. 40 members over 400 epochs.")
    p.add_argument("--ring-persist-every", type=int, default=4,
                   help="persist every Nth ARCHIVE to disk (not every Nth "
                        "epoch). Each policy is 148 MB, so all 41 members x 3 "
                        "arms would be 17.8 GB. The in-memory ring still holds "
                        "every member, so the experiment is unchanged; this "
                        "only thins what the post-hoc tournament can read, and "
                        "it subsamples to ~12 anyway.")
    p.add_argument("--ring-delta", type=float, default=0.0,
                   help="CompetEvo's WINDOW parameter, not a mixing "
                        "probability: the opponent is uniform on "
                        "[max(1, floor(delta*epoch)), epoch-1]. 0 = the whole "
                        "history, which is what 'beat ALL past selves' asks "
                        "for; 0.5 = the most recent half (their dev setting).")
    p.add_argument("--snapshot-root",
                   default="/workspace/Transform2Act/results/_e4_snapshots")
    p.add_argument("--restart-check-epoch", type=int, default=150,
                   help="pre-registered PBT substitute: if goal rate is still "
                        "0.00 here, this seed drew a dead controller. E3.1's "
                        "s3 was detectable this way by epoch ~140 while its "
                        "solvers were already climbing.")
    p.add_argument("--mirror-episodes", type=int, default=20)
    p.add_argument("--ladder-episodes", type=int, default=10)
    p.add_argument("--ladder-k", type=int, default=5,
                   help="how many past selves to score each eval; the FULL "
                        "matrix is a post-hoc job (e4r_tournament.py)")
    p.add_argument("--pool-evals", type=int, default=5,
                   help="how many recent evaluations the pooled correlation "
                        "aggregates. r over 10 episodes is noise; this "
                        "project's rule is to aggregate before comparing.")
    args = p.parse_args()

    from design_opt.agents.transform2act_agent import Transform2ActAgent
    from design_opt.utils.config import Config
    from khrylib.utils.torch import to_cpu, to_test
    from rower_soccer.t2a_port import e2_eval, e3_morph, rtg_scene
    from rower_soccer.t2a_port import e4_selfplay as sp
    from rower_soccer.t2a_port import e4r_ring
    from design_opt.models.transform2act_policy import Transform2ActPolicy
    from khrylib.robot.xml_robot import Robot
    from rower_soccer.t2a_port.e2_wandb import Run

    dtype = torch.float64
    torch.set_default_dtype(dtype)
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    cfg = Config(args.cfg, tmp=False)
    if args.max_epoch:
        cfg.max_epoch_num = args.max_epoch
    device = (torch.device("cuda", index=args.gpu_index)
              if torch.cuda.is_available() else torch.device("cpu"))
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    start_epoch = int(args.epoch) if args.epoch.isnumeric() else args.epoch
    agent = Transform2ActAgent(cfg=cfg, dtype=dtype, device=device,
                               seed=cfg.seed, num_threads=args.num_threads,
                               training=True, checkpoint=start_epoch)
    L = agent.logger.info
    env = agent.env
    design_on = not env.env_specs.get("force_identity_design", False)
    agent.cur_alpha = alpha_at(0, args.curriculum_steps, cfg.min_batch_size)
    if args.curriculum_steps:
        agent.custom_reward = make_custom_reward(agent)
    L(f"E4R GNN shared-weight ring self-play: cfg {args.cfg} seed {cfg.seed} threads {args.num_threads} "
      f"torch_threads {torch.get_num_threads()} "
      f"DESIGN {'ON' if design_on else 'OFF (frozen-body control)'} "
      f"max_epoch {cfg.max_epoch_num} batch {cfg.min_batch_size} "
      f"curriculum_steps {args.curriculum_steps} alpha0 {agent.cur_alpha} "
      f"opponent_mode {env.opp_mode} ring_every {args.ring_every} "
      f"ring_delta {args.ring_delta} opp_mean_action {env.opp_mean_action} dt {env.dt} "
      f"max_nsteps {env.max_nsteps} stop_file {args.stop_file}")

    os.makedirs(RENDER_DIR_E4, exist_ok=True)
    jsonl = os.path.join(cfg.cfg_dir, "e4r_epochs.jsonl")

    wb = Run(args.wandb_name or f"d3_e4r_{args.cfg}",
             project=args.wandb_project, enabled=args.wandb, log=L,
             tags=["d3", "e4", "gnn", "run-to-goal", "self-play", "shared-weight", "ring",
                   "design-on" if design_on else "design-off"],
             config=dict(arm="gnn_selfplay", cfg=args.cfg, seed=cfg.seed,
                         ring_every=args.ring_every, ring_delta=args.ring_delta,
                         design_on=design_on,
                         curriculum_steps=args.curriculum_steps,
                         batch=cfg.min_batch_size,
                         mini_batch=cfg.mini_batch_size,
                         max_epoch=cfg.max_epoch_num,
                         opponent_speed=env.opp_speed, dt=env.dt,
                         max_nsteps=env.max_nsteps,
                         policy_lr=cfg.policy_lr, value_lr=cfg.value_lr))

    total_steps = 0
    # ---------------------------------------------------------- E4 -----
    BASE_SRC = os.path.join("assets", "mujoco_envs", "ant_competevo.xml")
    merged_out = os.path.join(cfg.cfg_dir, "opponent_scene.xml")
    opp_meta = {"epoch": None, "cfg": None}

    def make_policy():
        return Transform2ActPolicy(cfg.policy_specs, agent)

    ring = e4r_ring.OpponentRing(
        cfg, BASE_SRC, rtg_scene.build, make_policy, Robot,
        delta=args.ring_delta, seed=cfg.seed, log=L)
    env.set_ring(ring)

    def archive_self(epoch):
        """Put the current self into the ring. Same to_cpu/to_test wrapper as
        the morphology census: the env hands CPU tensors, the learner lives on
        cuda, and eval mode stops the probe updating the policy's own
        running-norm statistics."""
        with to_cpu(agent.policy_net), to_test(agent.policy_net):
            body = sp.dump_mean_action_body(env, agent.policy_net, e3_morph)
        if not body:
            L(f"  [ring] epoch {epoch}: design stages failed, not archived")
            return
        sd = {k: v.detach().cpu()
              for k, v in agent.policy_net.state_dict().items()}
        keep = (epoch % (args.ring_every * max(1, args.ring_persist_every))) == 0
        nmem = ring.add(epoch, sd, body, persist=keep)
        L(f"  [ring] epoch {epoch}: archived, ring now holds {nmem}"
          f"{'' if keep else ' (in memory only; not persisted)'}")

    # ---- pre-registered dead-controller check (PBT substitute) ---------
    # DETECTS and flags; it does NOT restart by itself. An automatic
    # re-initialisation mid-run would change the run's semantics silently, and
    # on this box a restart is an operator action anyway (MPS is active, so
    # arms stop by stop-file). The marker file is what the watcher reports.
    restart_flagged = [False]

    def check_dead_controller(epoch, ev_hist):
        if restart_flagged[0] or epoch < args.restart_check_epoch:
            return
        recent = [e["eval"]["goal_rate"] for e in ev_hist
                  if e["epoch"] >= args.restart_check_epoch - 50]
        if len(recent) < 3 or max(recent) > 0.0:
            return
        restart_flagged[0] = True
        marker = os.path.join(cfg.cfg_dir, "RESTART_RECOMMENDED")
        with open(marker, "w") as f:
            json.dump(dict(cfg=args.cfg, epoch=epoch, n_evals=len(recent),
                           max_goal_rate=max(recent),
                           rule=("pre-registered: goal rate still 0.00 at "
                                 "epoch %d" % args.restart_check_epoch)), f)
        L(f"  *** RESTART RECOMMENDED: goal rate 0.00 across {len(recent)} "
          f"evals through epoch {epoch}. Pre-registered dead-controller rule "
          f"(D3 E4B section 6). Marker: {marker}")

    eval_history = []
    recent_evals = []
    for epoch in range(start_epoch, cfg.max_epoch_num):
        # archive BEFORE training so epoch 0's self is in the ring, then let
        # the env draw a strictly-past member at every episode reset
        if args.ring_every and epoch % args.ring_every == 0:
            archive_self(epoch)
        env.ring_epoch = epoch
        env.ring_chosen = []
        agent.cur_alpha = alpha_at(epoch, args.curriculum_steps,
                                   cfg.min_batch_size)
        info = agent.optimize_policy(epoch)
        agent.log_optimize_policy(epoch, info)
        agent.save_checkpoint(epoch)
        torch.cuda.empty_cache()

        log, log_eval = info["log"], info["log_eval"]
        total_steps += int(log.num_steps)
        # KEY NAMES CARRY THEIR PROTOCOL -- E1.1's near-miss was reading
        # Transform2Act's `exec_R_eps` (a MEAN-ACTION evaluation pass) against
        # the MLP trainer's (a STOCHASTIC training return) as one statistic.
        # Neither is the E3 result; the comparable curve is `e3/eval_*`, from
        # the shared instrument `e2_eval.evaluate`.
        payload = {
            "e4r/train_R_STOCHASTIC": float(log.avg_reward),
            "e4r/train_R_eps_STOCHASTIC": float(log.avg_episode_reward),
            "e4r/exec_R_MEANACTION_eval": float(log_eval.avg_exec_reward),
            "e4r/exec_R_eps_MEANACTION_eval":
                float(log_eval.avg_exec_episode_reward),
            "e4r/ep_len": float(log.avg_episode_len),
            "e4r/num_episodes": int(log.num_episodes),
            "e4r/total_steps": total_steps,
            "e4r/T_sample": info["T_sample"], "e4r/T_update": info["T_update"],
            "e4r/T_eval": info["T_eval"],
        }
        if agent.cur_alpha is not None:
            payload["e4r/alpha"] = float(agent.cur_alpha)
            # the fall-dodge's weight in the objective THIS epoch: the sparse
            # term is worth (1-alpha)*1000, against E2.1's measured critical
            # value of 261 points (a_crit = 0.739).
            payload["e4r/dodge_worth"] = float((1.0 - agent.cur_alpha) * 1000.0)

        # ---- morphology, every epoch, from epoch 0 -----------------------
        # D3 M3 E3.1: `control_log_std` is a LEARNED parameter, so
        # `log_std_crit` (D3_E3_ADVERSARIAL.md 3f) is a BASIN BOUNDARY rather
        # than a precision requirement -- below it the gradient is
        # self-reinforcing, above it the same gradient runs toward deleting
        # actuators. That makes "sigma actually goes down" load-bearing and it
        # was unmonitored in E3: the trajectory had to be reconstructed from
        # checkpoints afterwards. Logged per epoch, to disk, from epoch 0.
        sd = agent.policy_net.state_dict()
        cls = sd.get("control_action_log_std")
        als = sd.get("attr_action_log_std")
        cls = float(cls.mean().item()) if cls is not None else None
        als = float(als.mean().item()) if als is not None else None
        row = {"epoch": epoch, "total_steps": total_steps,
               "alpha": agent.cur_alpha,
               "control_log_std": cls, "attr_log_std": als}
        if cls is not None:
            payload["policy/control_log_std"] = cls
            payload["policy/control_sigma"] = float(np.exp(cls))
            # the quantity 3f's threshold is stated in
            payload["policy/ctrl_cost_per_step_pred"] = float(
                0.5 * 8 * np.exp(2.0 * cls))
        if als is not None:
            payload["policy/attr_log_std"] = als
        # which snapshot this epoch was trained against -- provenance for
        # EVERY row, not only the ones carrying an eval
        row["ring"] = dict(size=len(ring.epochs()),
                           chosen=list(env.ring_chosen))
        payload.update(e4r_ring.ring_stats(env.ring_chosen, epoch))
        if args.morph_every and epoch % args.morph_every == 0:
            t0 = time.time()
            with e3_morph.rng_guard(env), to_cpu(agent.policy_net), \
                    to_test(agent.policy_net):
                ok = e3_morph.run_design_stages(env, agent.policy_net, True,
                                                agent.running_state)
                ma = e3_morph.body_summary(env) if ok else {}
                cen = e3_morph.census(env, agent.policy_net,
                                      args.morph_episodes, False,
                                      agent.running_state)
            row["mean_action_design"] = ma
            row["census"] = cen
            if ma:
                payload.update({
                    "morph/n_bodies": ma["n_bodies"],
                    "morph/n_limbs": ma["n_limbs"],
                    "morph/n_motors": ma["model_nu_ours"],
                    "morph/mass": ma["model_mass_ours"],
                    "morph/limb_len_mean": ma["limb_length"]["mean"],
                    "morph/limb_len_max": ma["limb_length"]["max"],
                    "morph/limb_len_sum": ma["limb_length"]["sum"],
                    "morph/limb_radius_mean": ma["limb_radius"]["mean"],
                    "morph/gear_mean": ma["gear"]["mean"],
                    "morph/max_depth": max(int(d) for d in ma["depth_hist"]),
                    "morph/n_opp_bodies": ma["n_opp_bodies"],
                })
            payload.update({
                "morph/distinct_topologies": cen["distinct_topologies"],
                "morph/top_topology_share": cen["top_topology_share"],
                "morph/sampled_bodies_mean": cen["bodies_mean"],
                "morph/sampled_bodies_min": cen["bodies_min"],
                "morph/sampled_bodies_max": cen["bodies_max"],
                "morph/design_fail_rate":
                    cen["design_failed"] / max(1, args.morph_episodes),
                "morph/T_morph": time.time() - t0,
            })

        # ---- the shared instrument, and the fall-dodge correlation -------
        if args.eval_every and (epoch + 1) % args.eval_every == 0:
            t0 = time.time()
            with e3_morph.rng_guard(env), to_cpu(agent.policy_net), \
                    to_test(agent.policy_net):
                act, wrap = e2_eval.gnn_actor(agent.policy_net,
                                              agent.running_state, True)
                ev = e2_eval.evaluate(env, act, wrap,
                                      episodes=args.eval_episodes,
                                      seed_base=1000,
                                      max_steps=env.max_nsteps + 5)
            eps = ev.pop("episodes", [])
            payload.update({f"e4r/eval_{k}": v for k, v in ev.items()})
            recent_evals.append(eps)
            recent_evals[:] = recent_evals[-args.pool_evals:]
            d_now = e3_morph.dodge_stats(eps)
            d_pool = e3_morph.pooled_dodge(recent_evals)
            row["eval"] = ev
            row["eval_episodes"] = eps
            row["dodge"] = d_now
            row["dodge_pooled"] = d_pool
            # E4's two pre-registered degeneracy guards. A high draw rate
            # means the coupled channel is OFF for those episodes
            # (n_reached == 2 -> parse = 0), so the divergence number is
            # untestable rather than null.
            rs = sp.race_stats(eps, env.goal_x)
            row["race"] = rs
            # the two things the user's criterion actually needs
            with e3_morph.rng_guard(env), to_cpu(agent.policy_net), \
                    to_test(agent.policy_net):
                mm = e4r_ring.mirror_match(env, agent, ring, e2_eval,
                                           episodes=args.mirror_episodes)
                lad = e4r_ring.ladder(env, agent, ring, e2_eval,
                                      episodes=args.ladder_episodes,
                                      k=args.ladder_k)
            eval_history.append(dict(epoch=epoch, eval=ev))
            check_dead_controller(epoch, eval_history)
            row["restart_recommended"] = restart_flagged[0]
            row["mirror"] = mm
            row["ladder"] = lad
            for k2, v2 in mm.items():
                if isinstance(v2, (int, float)):
                    payload[f"mirror/{k2}"] = float(v2)
            if lad.get("mean_win") is not None:
                payload["ladder/mean_win"] = float(lad["mean_win"])
                payload["ladder/spearman_age_win"] = (
                    float(lad["spearman"]) if lad.get("spearman") is not None
                    else 0.0)
            L(f"  mirror: decisive {mm.get('decisive_rate')} mutual "
              f"{mm.get('mutual_rate')} STALEMATE {mm.get('stalemate_rate')} "
              f"fwd {mm.get('fwd_mean')} | ladder mean_win {lad.get('mean_win')} "
              f"rho {lad.get('spearman')} over {lad.get('n_opponents')} past selves")
            for k, v in rs.items():
                payload[f"race/{k}"] = float(v)
            for k, v in d_now.items():
                if v is not None:
                    payload[f"dodge/{k}"] = float(v)
            for k, v in d_pool.items():
                if v is not None:
                    payload[f"dodge/pooled_{k}"] = float(v)
            L(f"  eval@{epoch}: R {ev['R_mean']:.1f} goal {ev['goal_rate']:.2f} "
              f"lost {ev['loss_rate']:.2f} fell {ev['fall_rate']:.2f} "
              f"fwd {ev['max_fwd']:.2f} m dx {ev['net_dx']:.2f} m "
              f"speed {ev['speed']:.3f} m/s nb {ev['bodies_exec']:.1f} "
              f"designfail {ev['design_fail_rate']:.2f} "
              f"| r(fall,R) {d_pool['r_fall_return']} "
              f"r(fwd,R) {d_pool['r_fwd_return']} (pooled n={d_pool['pooled_n']}) "
              f"({time.time() - t0:.0f}s)")

        with open(jsonl, "a") as f:
            f.write(json.dumps(row) + "\n")

        # ---- video, off a TRANSIENT checkpoint ---------------------------
        video = None
        want_video = args.video_every and (epoch == start_epoch
                                           or (epoch + 1) % args.video_every == 0)
        if want_video:
            tmp = "%s/%s.p" % (cfg.model_dir, VIDEO_CKPT)
            try:
                import pickle
                with to_cpu(agent.policy_net, agent.value_net):
                    pickle.dump({"policy_dict": agent.policy_net.state_dict(),
                                 "value_dict": agent.value_net.state_dict(),
                                 "running_state": agent.running_state,
                                 "loss_iter": agent.loss_iter,
                                 "best_rewards": agent.best_rewards,
                                 "epoch": epoch}, open(tmp, "wb"))
                # The cfg names differ by one letter (`rtg_e3_s1` design ON
                # vs `rtg_e3c_s1` frozen control) and produce completely
                # different experiments. A reader sorting the render directory
                # by date sees only controls, because the design-ON arms stop
                # early -- which did mislead a reader into concluding the
                # morphology was not changing. The tag makes the clip
                # self-describing.
                tag = "DESIGN-ON" if design_on else "FROZEN-CONTROL"
                rd = (RENDER_DIR_E4 if args.cfg.startswith("rtg_e4r")
                      else RENDER_DIR_E31 if args.cfg.startswith("rtg_e31")
                      else RENDER_DIR)
                os.makedirs(rd, exist_ok=True)
                out = f"{rd}/{tag}_{args.cfg}_e{epoch + 1:04d}_bmw.mp4"
                env2 = dict(os.environ, MUJOCO_GL="osmesa",
                            CUDA_VISIBLE_DEVICES="")
                cmd = [sys.executable,
                       "/workspace/utmist-vc2-phase2/rower_soccer/t2a_port/"
                       "e3_video.py", "--cfg", args.cfg, "--ckpt", VIDEO_CKPT,
                       "--out", out, "--json",
                       "--episodes", str(args.video_episodes)]
                r = subprocess.run(cmd, env=env2, capture_output=True,
                                   text=True, timeout=3600)
                line = [ln for ln in r.stdout.splitlines()
                        if ln.startswith("E3VIDEO ")]
                if line:
                    d = json.loads(line[0][len("E3VIDEO "):])
                    video = d["mp4"]
                    payload.update(d["scalars"])
                    L(f"  video {video}")
                else:
                    L(f"  video FAILED rc={r.returncode} "
                      f"{r.stderr.strip()[-400:]}")
            except Exception as e:
                L(f"  video FAILED ({e!r})")
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

        wb.log_epoch(epoch, payload, video=video)

        if args.archive_every and (epoch + 1) % args.archive_every == 0:
            archive(cfg.model_dir, args.archive_bucket, args.cfg, L)

        if args.stop_file and os.path.exists(args.stop_file):
            L(f"stop file {args.stop_file} present -- stopping after {epoch}")
            cp = "%s/epoch_%04d.p" % (cfg.model_dir, epoch + 1)
            if not os.path.exists(cp):
                import pickle
                with to_cpu(agent.policy_net, agent.value_net):
                    pickle.dump({"policy_dict": agent.policy_net.state_dict(),
                                 "value_dict": agent.value_net.state_dict(),
                                 "running_state": agent.running_state,
                                 "loss_iter": agent.loss_iter,
                                 "best_rewards": agent.best_rewards,
                                 "epoch": epoch}, open(cp, "wb"))
                L(f"saved {cp}")
            break
    else:
        L("training done!")
    archive(cfg.model_dir, args.archive_bucket, args.cfg, L)
    wb.finish()


if __name__ == "__main__":
    main()
