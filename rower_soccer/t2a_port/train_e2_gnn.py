"""D3 M3 E2: Transform2Act's GNN on CompetEvo's 1v1 run-to-goal.

`train_their_ant.py`'s loop (theirs, plus a stop file, because under NVIDIA MPS
killing a CUDA client can corrupt the live survivors) with three additions E2
needs and E0/E1 did not have:

  * **inline wandb**, metrics and video in ONE run, logged from this process as
    each epoch finishes -- see `e2_wandb.py` for why that removes the reason
    E0/E1 needed separate `_media` runs;
  * **an inline mean-action evaluation** with the task's own success metric
    (goal rate / loss rate / fall rate), through `e2_eval.evaluate` -- the SAME
    function the MLP arm and the post-hoc table call, because E1.1's headline
    number was nearly wrong from reading two different statistics side by side;
  * **an inline best/median/worst clip**, rendered in a SUBPROCESS off a saved
    checkpoint (this process holds a CUDA context and forks sampler workers;
    an offscreen GL context here is a way to lose the run) and then logged into
    THIS run in the same `wandb.log` call as that epoch's metrics.

    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \\
           CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
    cd /workspace/Transform2Act && source env-gpu.sh
    setsid nohup .venv-gpu/bin/python \\
      /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/train_e2_gnn.py \\
      --cfg rtg_gnn_s1 --num_threads 15 --wandb --wandb-name d3_e2_gnn_s1 \\
      --stop-file /tmp/stop_e2_gnn_s1 &
"""

import argparse
import json
import os
import subprocess
import sys
import time

# wandb lives beside `.venv-gpu` (which has none) and needs protobuf >= 4,
# while the venv pins 3.x and tensorboardX's generated code needs 3.x's C
# extension. Both are settled BEFORE any import that could pull protobuf in:
# the path goes first so protobuf 6 wins, and the pure-Python protobuf
# implementation is forced so tensorboardX still loads. Setting either after
# `google.protobuf` is imported is a no-op -- measured, it silently disabled
# wandb on the first E2 GNN smoke.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
if os.path.isdir("/workspace/t2a_pylibs"):
    sys.path.insert(0, "/workspace/t2a_pylibs")
sys.path.append("/workspace/Transform2Act")
sys.path.append("/workspace/utmist-vc2-phase2")
os.chdir("/workspace/Transform2Act")

import numpy as np  # noqa: E402
import torch  # noqa: E402

RENDER_DIR = "/workspace/utmist-vc2-phase2/runs/d3_e2_rtg/renders"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    p.add_argument("--num_threads", type=int, default=15)
    p.add_argument("--gpu_index", type=int, default=0)
    p.add_argument("--epoch", default="0")
    p.add_argument("--stop-file", default=None)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-project", default="creature-soccer")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--video-every", type=int, default=10,
                   help="epochs between clips; 0 disables. Must be a multiple "
                        "of save_model_interval, because the renderer runs off "
                        "a saved checkpoint.")
    p.add_argument("--video-episodes", type=int, default=9)
    args = p.parse_args()

    from design_opt.agents.transform2act_agent import Transform2ActAgent
    from design_opt.utils.config import Config
    from khrylib.utils.torch import to_cpu
    from rower_soccer.t2a_port import e2_eval
    from rower_soccer.t2a_port.e2_wandb import Run

    dtype = torch.float64
    torch.set_default_dtype(dtype)
    cfg = Config(args.cfg, tmp=False)
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
    L(f"E2 GNN: cfg {args.cfg} seed {cfg.seed} threads {args.num_threads} "
      f"max_epoch {cfg.max_epoch_num} batch {cfg.min_batch_size} "
      f"opponent_speed {agent.env.opp_speed} dt {agent.env.dt} "
      f"max_nsteps {agent.env.max_nsteps} stop_file {args.stop_file}")

    wb = Run(args.wandb_name or f"d3_e2_{args.cfg}",
             project=args.wandb_project, enabled=args.wandb, log=L,
             tags=["d3", "e2", "gnn", "run-to-goal"],
             config=dict(arm="gnn", cfg=args.cfg, seed=cfg.seed,
                         batch=cfg.min_batch_size,
                         mini_batch=cfg.mini_batch_size,
                         opponent_speed=agent.env.opp_speed,
                         dt=agent.env.dt, max_nsteps=agent.env.max_nsteps,
                         policy_lr=cfg.policy_lr, value_lr=cfg.value_lr))

    total_steps = 0
    for epoch in range(start_epoch, cfg.max_epoch_num):
        info = agent.optimize_policy(epoch)
        agent.log_optimize_policy(epoch, info)
        agent.save_checkpoint(epoch)
        torch.cuda.empty_cache()

        log, log_eval = info["log"], info["log_eval"]
        total_steps += int(log.num_steps)
        payload = {
            "e2/train_R": float(log.avg_reward),
            "e2/train_R_eps": float(log.avg_episode_reward),
            "e2/exec_R": float(log_eval.avg_exec_reward),
            "e2/exec_R_eps": float(log_eval.avg_exec_episode_reward),
            "e2/ep_len": float(log.avg_episode_len),
            "e2/num_episodes": int(log.num_episodes),
            "e2/total_steps": total_steps,
            "e2/T_sample": info["T_sample"], "e2/T_update": info["T_update"],
            "e2/T_eval": info["T_eval"],
        }

        if args.eval_every and (epoch + 1) % args.eval_every == 0:
            t0 = time.time()
            with to_cpu(agent.policy_net):
                agent.policy_net.eval()
                act, wrap = e2_eval.gnn_actor(agent.policy_net,
                                              agent.running_state, True)
                ev = e2_eval.evaluate(agent.env, act, wrap,
                                      episodes=args.eval_episodes,
                                      seed_base=1000,
                                      max_steps=agent.env.max_nsteps + 5)
                agent.policy_net.train()
            payload.update({f"e2/eval_{k}": v for k, v in ev.items()
                            if k != "episodes"})
            L(f"  eval@{epoch}: R {ev['R_mean']:.1f} goal {ev['goal_rate']:.2f} "
              f"lost {ev['loss_rate']:.2f} fell {ev['fall_rate']:.2f} "
              f"dx {ev['net_dx']:.2f} m speed {ev['speed']:.3f} m/s "
              f"({time.time() - t0:.0f}s)")

        video = None
        if args.video_every and (epoch + 1) % args.video_every == 0:
            cp = "%s/epoch_%04d.p" % (cfg.model_dir, epoch + 1)
            if os.path.exists(cp):
                out = f"{RENDER_DIR}/{args.cfg}_e{epoch + 1:04d}_bmw.mp4"
                env2 = dict(os.environ, MUJOCO_GL="osmesa",
                            CUDA_VISIBLE_DEVICES="")
                cmd = [sys.executable,
                       "/workspace/utmist-vc2-phase2/rower_soccer/t2a_port/"
                       "e2_video.py", "--arm", "gnn", "--cfg", args.cfg,
                       "--epoch", str(epoch + 1), "--out", out, "--json",
                       "--episodes", str(args.video_episodes)]
                try:
                    r = subprocess.run(cmd, env=env2, capture_output=True,
                                       text=True, timeout=3600)
                    line = [l for l in r.stdout.splitlines()
                            if l.startswith("E2VIDEO ")]
                    if line:
                        d = json.loads(line[0][len("E2VIDEO "):])
                        video = d["mp4"]
                        payload.update(d["scalars"])
                        L(f"  video {video}")
                    else:
                        L(f"  video FAILED rc={r.returncode} "
                          f"{r.stderr.strip()[-400:]}")
                except Exception as e:
                    L(f"  video FAILED ({e!r})")
            else:
                L(f"  video skipped: no checkpoint {cp}")

        wb.log_epoch(epoch, payload, video=video)

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
    wb.finish()


if __name__ == "__main__":
    main()
