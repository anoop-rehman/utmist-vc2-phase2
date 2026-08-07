"""Simulator-scaling probe: env-steps/s vs #envs for the worm-fetch workload.

Law 1 of the scaling story (steps/s vs parallelism). For each world count it
builds the env, captures the CUDA graph (so we time the SAME stepping path
training uses), runs a warmup then a timed burst of random-action steps, and
records steps/s + peak VRAM. OOM is caught and recorded as the ceiling for that
GPU, not a crash. Results go to stdout AND wandb (a table + per-point metrics),
which is the whole monitoring surface for the remote fanout.

    MUJOCO_GL=egl .venv/bin/python -m rower_soccer.warp_port.probe_throughput \
        --worlds 1024 2048 4096 8192 16384 32768 --gpu-tag h100
"""
import argparse
import time

import numpy as np
import torch


def probe_one(worlds, steps, warmup, seed=0):
    """Build a worm-fetch env at `worlds`, time `steps` graph-launched steps.
    Returns (steps_per_sec, vram_peak_mb) or raises on OOM."""
    from rower_soccer.warp_port.worm_fetch_env import WarpWormFetchEnv
    free0, total = torch.cuda.mem_get_info()      # device-level, so it counts
    env = WarpWormFetchEnv(num_worlds=worlds, scene="arena", use_graph=True,
                           seed=seed)              # Warp allocs (torch's own
    act = torch.zeros(worlds, env.act_dim, device="cuda")   # counter would miss)
    env.reset()
    gen = torch.Generator(device="cuda").manual_seed(seed)
    for _ in range(warmup):                       # warm caches / graph
        act.uniform_(-1.0, 1.0, generator=gen)
        env.step(act)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        act.uniform_(-1.0, 1.0, generator=gen)
        env.step(act)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    sps = worlds * steps / dt
    free1, _ = torch.cuda.mem_get_info()
    vram = (free0 - free1) / 1e6                  # device memory this env took
    del env, act
    torch.cuda.empty_cache()
    return sps, vram


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", type=int, nargs="+",
                   default=[1024, 2048, 4096, 8192, 16384, 32768])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=40)
    p.add_argument("--gpu-tag", default="gpu")
    p.add_argument("--wandb-project", default="creature-soccer")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=f"scale_probe_{args.gpu_tag}",
                   id=f"scale_probe_{args.gpu_tag}", resume="allow",
                   config={"gpu": args.gpu_tag, "steps": args.steps,
                           "phase": "throughput"})
        wandb.define_metric("worlds")
        wandb.define_metric("*", step_metric="worlds")
        table = wandb.Table(columns=["worlds", "steps_per_sec", "vram_mb"])

    gpu = torch.cuda.get_device_name(0)
    print(f"[probe] GPU={gpu} tag={args.gpu_tag}  worlds={args.worlds}", flush=True)
    print(f"{'worlds':>8} {'steps/s':>12} {'vram_MB':>9}", flush=True)
    for w in args.worlds:
        try:
            sps, vram = probe_one(w, args.steps, args.warmup)
            print(f"{w:>8} {sps:>12,.0f} {vram:>9,.0f}", flush=True)
            if use_wandb:
                wandb.log({"worlds": w, "steps_per_sec": sps, "vram_mb": vram})
                table.add_data(w, round(sps), round(vram))
        except Exception as e:                     # OOM or any Warp alloc failure
            msg = f"{type(e).__name__}: {str(e)[:80]}"
            print(f"{w:>8} {'FAILED':>12} -> {msg}", flush=True)
            if use_wandb:
                wandb.log({"worlds": w, "steps_per_sec": 0.0, "oom": 1})
            torch.cuda.empty_cache()
            break                                  # first failure = the ceiling
    if use_wandb:
        wandb.log({"throughput_table": table})
        wandb.finish()


if __name__ == "__main__":
    main()
