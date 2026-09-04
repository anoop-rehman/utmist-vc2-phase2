"""D3 M3 E3: Transform2Act's design+control loop on an adversarial task.

`train_e2_gnn.py` is left untouched -- E2's four arms have to stay
reproducible from the file that produced them -- and this is its successor
with the four things E3 needs and E2 did not have:

  1. **The d2rep reward regime** (`--curriculum-steps`). E2.1 established that
     run-to-goal against this scripted opponent is solvable on this body only
     when the sparse +/-1000 stays under ~26% of the objective for the WHOLE
     run: `d2rep` (alpha 1.000 -> 0.847) reaches goal 0.95/1.00 with zero falls
     in 40 episodes, while the flat reward at 4x the budget reaches 0.25 and
     CompetEvo's nominal anneal reaches 0.10. E3 runs `d2rep`.

     The mix enters through `Agent.custom_reward` -- khrylib's own hook,
     already wired into `sample_worker` -- so the PPO buffer gets
     `alpha*dense + (1-alpha)*parse` while `LoggerRL.step` keeps logging the
     RAW env reward. That is exactly E2.1's invariant ("the curriculum touches
     the buffer and nothing else"), obtained here without editing a
     Transform2Act file at all.

  2. **A per-epoch morphology summary**, from epoch 0. `D3_E2_RTG.md` 6
     measured a fall-dodge worth ~+826 that CompetEvo's own termination rule
     creates, and `D3_E21_CURRICULUM.md` 7 records that `d2rep` AVOIDS it
     rather than removing it. Morphology is a far wider channel for finding
     that optimum than control alone -- an agent that can reshape its body
     could evolve toward falling reliably -- so limb lengths, gears, topology
     and body count are logged beside the fall rate every epoch rather than
     reconstructed afterwards.

  3. **The E2/E2.1 correlation instrument, computed as the run proceeds.**
     E2 measured `r(fall rate, return) = +0.989` and
     `r(forward progress, return) = +0.019`; E2.1's d2rep inverted it to
     roughly -0.94 / +0.95 over the trained arms. If E3's drifts back toward
     E2's structure the dodge has reopened through morphology, and that is
     worth knowing while it is happening.

  4. **Video decoupled from checkpointing.** A GNN checkpoint here is 157 MB,
     so saving one every few epochs to feed the renderer would cost 6 GB per
     seed on a disk with 13 GB free. The clip is rendered off a TRANSIENT
     checkpoint that is written, rendered from, and deleted, so the video
     cadence (~15 min wall clock) and the archival cadence (epoch 100) are
     independent.

The design stages are ON or OFF purely by cfg (`env_specs.force_identity_design`),
so the E3 arms and the frozen-body GNN control run through this identical file
and identical instrument -- which is what makes an E3 null interpretable.

    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \\
           CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
    cd /workspace/Transform2Act && source env-gpu.sh
    setsid nohup .venv-gpu/bin/python .../t2a_port/train_e3_gnn.py \\
      --cfg rtg_e3_s1 --curriculum-steps 130208333 --num-threads 10 \\
      --wandb --wandb-name d3_e3_gnn_s1 --stop-file /tmp/stop_e3_s1 &
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
    p.add_argument("--pool-evals", type=int, default=5,
                   help="how many recent evaluations the pooled correlation "
                        "aggregates. r over 10 episodes is noise; this "
                        "project's rule is to aggregate before comparing.")
    args = p.parse_args()

    from design_opt.agents.transform2act_agent import Transform2ActAgent
    from design_opt.utils.config import Config
    from khrylib.utils.torch import to_cpu, to_test
    from rower_soccer.t2a_port import e2_eval, e3_morph
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
    L(f"E3 GNN: cfg {args.cfg} seed {cfg.seed} threads {args.num_threads} "
      f"torch_threads {torch.get_num_threads()} "
      f"DESIGN {'ON' if design_on else 'OFF (frozen-body control)'} "
      f"max_epoch {cfg.max_epoch_num} batch {cfg.min_batch_size} "
      f"curriculum_steps {args.curriculum_steps} alpha0 {agent.cur_alpha} "
      f"opponent_speed {env.opp_speed} dt {env.dt} "
      f"max_nsteps {env.max_nsteps} stop_file {args.stop_file}")

    os.makedirs(RENDER_DIR, exist_ok=True)
    jsonl = os.path.join(cfg.cfg_dir, "e3_epochs.jsonl")

    wb = Run(args.wandb_name or f"d3_e3_{args.cfg}",
             project=args.wandb_project, enabled=args.wandb, log=L,
             tags=["d3", "e3", "gnn", "run-to-goal",
                   "design-on" if design_on else "design-off"],
             config=dict(arm="gnn", cfg=args.cfg, seed=cfg.seed,
                         design_on=design_on,
                         curriculum_steps=args.curriculum_steps,
                         batch=cfg.min_batch_size,
                         mini_batch=cfg.mini_batch_size,
                         max_epoch=cfg.max_epoch_num,
                         opponent_speed=env.opp_speed, dt=env.dt,
                         max_nsteps=env.max_nsteps,
                         policy_lr=cfg.policy_lr, value_lr=cfg.value_lr))

    total_steps = 0
    recent_evals = []
    for epoch in range(start_epoch, cfg.max_epoch_num):
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
            "e3/train_R_STOCHASTIC": float(log.avg_reward),
            "e3/train_R_eps_STOCHASTIC": float(log.avg_episode_reward),
            "e3/exec_R_MEANACTION_eval": float(log_eval.avg_exec_reward),
            "e3/exec_R_eps_MEANACTION_eval":
                float(log_eval.avg_exec_episode_reward),
            "e3/ep_len": float(log.avg_episode_len),
            "e3/num_episodes": int(log.num_episodes),
            "e3/total_steps": total_steps,
            "e3/T_sample": info["T_sample"], "e3/T_update": info["T_update"],
            "e3/T_eval": info["T_eval"],
        }
        if agent.cur_alpha is not None:
            payload["e3/alpha"] = float(agent.cur_alpha)
            # the fall-dodge's weight in the objective THIS epoch: the sparse
            # term is worth (1-alpha)*1000, against E2.1's measured critical
            # value of 261 points (a_crit = 0.739).
            payload["e3/dodge_worth"] = float((1.0 - agent.cur_alpha) * 1000.0)

        # ---- morphology, every epoch, from epoch 0 -----------------------
        row = {"epoch": epoch, "total_steps": total_steps,
               "alpha": agent.cur_alpha}
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
            payload.update({f"e3/eval_{k}": v for k, v in ev.items()})
            recent_evals.append(eps)
            recent_evals[:] = recent_evals[-args.pool_evals:]
            d_now = e3_morph.dodge_stats(eps)
            d_pool = e3_morph.pooled_dodge(recent_evals)
            row["eval"] = ev
            row["eval_episodes"] = eps
            row["dodge"] = d_now
            row["dodge_pooled"] = d_pool
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
                out = f"{RENDER_DIR}/{args.cfg}_e{epoch + 1:04d}_bmw.mp4"
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
