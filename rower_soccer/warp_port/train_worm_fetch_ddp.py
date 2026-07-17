"""Distributed data-parallel PPO (DD-PPO) for the worm-fetch workload.

One process per GPU. Each rank runs its OWN WarpWormFetchEnv (its own worlds)
and a policy replica; gradients are all-reduced (averaged) every minibatch step
so the replicas stay identical. Effective parallel envs = world_size x
--worlds, which is how we push past a single GPU's ~30k-world SM ceiling toward
the 100k-1M regime.

Device handling: we set CUDA_VISIBLE_DEVICES to this rank's local GPU *before
importing torch/warp*, so every library sees exactly one GPU as cuda:0. That
sidesteps pointing Warp at a non-zero device (its default-device model makes
that fragile) -- each process simply owns one card. NCCL still communicates
across processes via the rendezvous store.

Launch (single node, N GPUs):
    torchrun --standalone --nproc_per_node=N \
        -m rower_soccer.warp_port.train_worm_fetch_ddp --worlds 8192 --scene arena
"""
import os

# MUST run before torch/warp import a CUDA context: pin this rank to one GPU.
# DDP_SHARE_GPU=1 lets N ranks share one physical GPU (correctness testing on a
# single-GPU box -- NCCL supports it; throughput obviously doesn't scale).
_LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
os.environ.setdefault("MUJOCO_GL", "egl")
if os.environ.get("DDP_SHARE_GPU") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_LOCAL_RANK)

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402


def is_dist():
    return int(os.environ.get("WORLD_SIZE", 1)) > 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="ddp_worm_fetch")
    p.add_argument("--worlds", type=int, default=8192, help="worlds PER GPU")
    p.add_argument("--steps", type=int, default=20_000_000_000,
                   help="GLOBAL env-step budget (summed across ranks)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rollout", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--ent-floor", type=float, default=-1.2)
    p.add_argument("--ent-ceil", type=float, default=0.0)
    p.add_argument("--ent-anneal-steps", type=int, default=400_000_000)
    p.add_argument("--z-dim", type=int, default=16)
    p.add_argument("--scene", choices=["arena", "pitch"], default="arena")
    p.add_argument("--floor-half", type=float, default=5.0)
    p.add_argument("--creature-xml", default="creature_configs/three_seg_worm.xml")
    p.add_argument("--up-axis-json",
                   default="creature_configs/three_seg_worm_up_axis.json")
    p.add_argument("--max-hours", type=float, default=1.0)
    p.add_argument("--bench-iters", type=int, default=0,
                   help=">0: run this many train_iters, report throughput, exit "
                        "(validation mode -- no wandb/ckpt)")
    p.add_argument("--wandb-project", default="creature-soccer")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = rank == 0
    if is_dist():
        # NCCL refuses two ranks on one physical GPU; the shared-GPU
        # correctness test (DDP_SHARE_GPU=1) uses gloo, which all-reduces CUDA
        # tensors fine (slower -- irrelevant for correctness). Real multi-GPU
        # nodes use NCCL.
        backend = "gloo" if os.environ.get("DDP_SHARE_GPU") == "1" else "nccl"
        dist.init_process_group(backend=backend)
    torch.cuda.set_device(0)   # the one GPU this rank sees

    from rower_soccer.warp_port.worm_fetch_env import WarpWormFetchEnv
    from rower_soccer.warp_port.ppo import ActorCritic, PPOTrainer

    # Distinct seed per rank so the ranks explore different env states (more
    # diverse experience), but the policy INIT must be identical -> broadcast.
    env = WarpWormFetchEnv(num_worlds=args.worlds, seed=args.seed + 1000 * rank,
                           scene=args.scene, floor_half=args.floor_half,
                           creature_xml=args.creature_xml,
                           up_axis_json=args.up_axis_json)
    ac = ActorCritic(env.obs_dim, env.act_dim,
                     proprio_indices=env.proprio_indices.tolist(),
                     task_indices=env.task_indices.tolist(), z_dim=args.z_dim)
    trainer = PPOTrainer(env, ac, lr=args.lr, rollout_len=args.rollout,
                         minibatches=args.minibatches, epochs=args.epochs,
                         ent_coef=args.ent_coef, ent_floor=args.ent_floor,
                         ent_ceil=args.ent_ceil,
                         ent_anneal_steps=args.ent_anneal_steps,
                         distributed=is_dist())

    # Make every replica start bit-identical.
    if is_dist():
        for pth in trainer.ac.parameters():
            dist.broadcast(pth.data, src=0)
        dist.barrier()

    global_worlds = args.worlds * world_size
    if is_main:
        print(f"[setup] ranks={world_size} worlds/gpu={args.worlds} "
              f"GLOBAL_worlds={global_worlds:,} obs={env.obs_dim} act={env.act_dim} "
              f"steps/iter={trainer.T * global_worlds:,}", flush=True)

    def all_mean(x):
        t = torch.tensor([x], device="cuda")
        if is_dist():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)   # AVG is nccl-only
            t /= world_size
        return float(t)

    # -- validation/benchmark mode: time N iters, report global throughput ----
    if args.bench_iters > 0:
        for _ in range(3):
            trainer.train_iter()               # warmup (JIT, graph, caches)
        torch.cuda.synchronize()
        if is_dist():
            dist.barrier()
        t0 = time.perf_counter()
        for _ in range(args.bench_iters):
            trainer.train_iter()
        torch.cuda.synchronize()
        if is_dist():
            dist.barrier()
        dt = time.perf_counter() - t0
        per_rank_sps = trainer.T * args.worlds * args.bench_iters / dt
        global_sps = per_rank_sps * world_size
        # The DDP invariant: after identical averaged gradients, every replica
        # must hold identical weights. Compare a full-parameter checksum.
        if is_dist():
            flat = torch.cat([pp.data.flatten() for pp in trainer.ac.parameters()])
            checks = [torch.zeros_like(flat) for _ in range(world_size)]
            dist.all_gather(checks, flat)
            drift = max(float((c - flat).abs().max()) for c in checks)
        else:
            drift = 0.0
        if is_main:
            print(f"[bench] ranks={world_size} global_worlds={global_worlds:,} "
                  f"iters={args.bench_iters} time={dt:.1f}s "
                  f"per_gpu_steps/s={per_rank_sps:,.0f} "
                  f"GLOBAL_steps/s={global_sps:,.0f} "
                  f"replica_drift={drift:.2e} (must be 0)", flush=True)
        if is_dist():
            dist.destroy_process_group()
        return

    # -- real training loop ----------------------------------------------------
    use_wandb = (not args.no_wandb) and is_main
    if use_wandb:
        import wandb
        git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True).stdout.strip()
        wandb.init(project=args.wandb_project, name=args.run_name,
                   id=args.run_name, resume="allow",
                   config={**vars(args), "world_size": world_size,
                           "global_worlds": global_worlds, "git_sha": git_sha})
        wandb.define_metric("env_step")
        wandb.define_metric("*", step_metric="env_step")

    t0 = time.perf_counter()
    deadline = t0 + args.max_hours * 3600.0
    it = 0
    global_steps = 0
    while global_steps < args.steps and time.perf_counter() < deadline:
        stats = trainer.train_iter()
        it += 1
        global_steps = trainer.total_steps * world_size
        if it % 5 == 0:
            now = time.perf_counter()
            gsps = global_steps / (now - t0)
            fit = all_mean(float(env.fitness().mean()))
            rew = all_mean(stats["ep_rew_env_mean"])
            if is_main:
                print(f"[monitor] gstep={global_steps:,} "
                      f"global_fps={gsps:,.0f} eta={(deadline-now)/60:.1f}min "
                      f"ep_rew={rew:.1f} reward_now={fit:.3f} "
                      f"std={stats['std']:.3f} div={trainer.n_diverged:,}",
                      flush=True)
                if use_wandb:
                    import wandb
                    wandb.log({"env_step": global_steps, "monitor/global_fps": gsps,
                               "train/ep_rew": rew, "train/reward_now": fit,
                               "train/std": stats["std"]})

    if is_main:
        print(f"[setup] done in {(time.perf_counter()-t0)/60:.1f}min "
              f"({global_steps:,} global steps)", flush=True)
    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
