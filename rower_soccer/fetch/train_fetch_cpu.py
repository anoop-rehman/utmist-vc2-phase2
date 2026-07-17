"""Faithful dm_control quadruped-fetch reproduction, CPU physics + SB3 SAC.

The task is dm_control's own `suite.load('quadruped', 'fetch')` -- byte-faithful
physics, spawns (ball dropped from z=2 with 5*randn velocity), and reward
(upright * reach * (0.5 + 0.5*fetch), all linear tolerances). Nothing is
re-implemented; this file is only a gymnasium adapter + SAC training loop.

Algorithm: SAC (SB3). The dm_control paper benchmarks fetch with distributed
D4PG/DMPO, which need actor fleets we don't have; SAC is the standard
single-machine model-free baseline on dm_control and its off-policy sample
efficiency is what a ~1k-fps CPU env actually needs. (The Warp port trains PPO
at ~50k fps in parallel -- see warp_port/fetch_env.py.)

    MUJOCO_GL=egl .venv/bin/python -m rower_soccer.fetch.train_fetch_cpu \
        --run-name fetch_cpu_sac --procs 6 --steps 5000000
"""
import argparse
import json
import os
import subprocess
import time

import gymnasium as gym
import numpy as np


class DmcFetchGym(gym.Env):
    """suite.load('quadruped','fetch') -> flat-obs gymnasium env.

    Obs = concat of the task's OrderedDict in ITS OWN order (egocentric_state,
    torso_velocity, torso_upright, imu, force_torque, ball_state,
    target_position) -- 90 dims. Action = the 12 actuator ctrls, [-1, 1.1/0.8]
    ranges as the model defines them.
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, seed=None):
        from dm_control import suite
        self._env = suite.load("quadruped", "fetch",
                               task_kwargs={"random": seed})
        spec = self._env.action_spec()
        self.action_space = gym.spaces.Box(spec.minimum.astype(np.float32),
                                           spec.maximum.astype(np.float32))
        ts = self._env.reset()
        flat = self._flatten(ts.observation)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                                shape=flat.shape,
                                                dtype=np.float32)

    @staticmethod
    def _flatten(obs_dict):
        return np.concatenate([np.asarray(v, dtype=np.float32).ravel()
                               for v in obs_dict.values()])

    def reset(self, seed=None, options=None):
        ts = self._env.reset()
        return self._flatten(ts.observation), {}

    def step(self, action):
        ts = self._env.step(action)
        obs = self._flatten(ts.observation)
        # dm_control episodes end only by time limit (20 s / 0.02 = 1000 steps).
        return obs, float(ts.reward or 0.0), False, ts.last(), {}

    def render(self):
        return self._env.physics.render(camera_id=0, height=240, width=320)


def make_env(seed):
    def _thunk():
        from stable_baselines3.common.monitor import Monitor
        # Monitor is what feeds SB3's ep_info_buffer -- without it the episode
        # returns in the console/wandb logs are silently nan.
        return Monitor(DmcFetchGym(seed=seed))
    return _thunk


def eval_episode(env, model, video_path=None, fps=50):
    """One deterministic episode on a fresh env; returns total reward."""
    frames, total = [], 0.0
    obs, _ = env.reset()
    done = False
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, r, _, done, _ = env.step(act)
        total += r
        if video_path is not None:
            frames.append(env.render())
    if video_path is not None:
        import imageio
        with imageio.get_writer(video_path, fps=fps, quality=7) as w:
            for f in frames:
                w.append_data(f)
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="fetch_cpu_sac")
    p.add_argument("--steps", type=int, default=5_000_000)
    p.add_argument("--procs", type=int, default=6,
                   help="parallel physics envs (pod throttles beyond ~8)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--buffer", type=int, default=1_000_000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--max-hours", type=float, default=10.0)
    p.add_argument("--video-secs", type=float, default=1200.0)
    p.add_argument("--first-video-secs", type=float, default=120.0)
    p.add_argument("--ckpt-secs", type=float, default=1800.0)
    p.add_argument("--gcs-bucket", default="vc2-2026-checkpoints")
    p.add_argument("--wandb-project", default="creature-soccer")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import SubprocVecEnv

    run_dir = os.path.join("runs_v2", args.run_name)
    os.makedirs(os.path.join(run_dir, "videos"), exist_ok=True)
    git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    cfg = {**vars(args), "algo": "SAC", "backend": "dm_control_cpu",
           "task": "quadruped_fetch", "git_sha": git_sha}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.run_name, id=args.run_name,
                   resume="allow", config=cfg)
        wandb.define_metric("env_step")
        wandb.define_metric("*", step_metric="env_step")

    venv = SubprocVecEnv([make_env(args.seed + i) for i in range(args.procs)])
    model = SAC("MlpPolicy", venv, learning_rate=args.lr,
                buffer_size=args.buffer, batch_size=args.batch,
                seed=args.seed, device="cuda" if torch.cuda.is_available() else "cpu",
                verbose=0)
    eval_env = DmcFetchGym(seed=10_000)

    t0 = time.perf_counter()
    deadline = t0 + args.max_hours * 3600
    last_video = t0 - max(0.0, args.video_secs - args.first_video_secs)
    last_ckpt = t0
    best = -np.inf
    chunk = 20_000  # env-steps between housekeeping passes
    ckpt_path = os.path.join(run_dir, "sac_latest.zip")
    best_path = os.path.join(run_dir, "sac_best.zip")

    print(f"[setup] procs={args.procs} obs={venv.observation_space.shape} "
          f"act={venv.action_space.shape} device={model.device}", flush=True)
    while model.num_timesteps < args.steps and time.perf_counter() < deadline:
        model.learn(total_timesteps=chunk, reset_num_timesteps=False,
                    progress_bar=False, log_interval=None)
        now = time.perf_counter()
        fps = model.num_timesteps / (now - t0)
        ep_rew = float(np.mean([e["r"] for e in model.ep_info_buffer])) \
            if model.ep_info_buffer else float("nan")
        print(f"[monitor] step={model.num_timesteps:,}/{args.steps:,} "
              f"fps={fps:,.0f} eta={(deadline-now)/60:.0f}min "
              f"ep_rew={ep_rew:.1f} (max 1000)", flush=True)
        if use_wandb:
            import wandb
            wandb.log({"env_step": model.num_timesteps, "monitor/fps": fps,
                       "train/ep_rew": ep_rew})

        if args.video_secs > 0 and now - last_video >= args.video_secs:
            last_video = now
            vpath = os.path.join(run_dir, "videos",
                                 f"eval_step_{model.num_timesteps:010d}.mp4")
            score = eval_episode(eval_env, model, vpath)
            print(f"[monitor] video: {vpath} (eval ep_rew={score:.1f})", flush=True)
            if score > best:
                best = score
                model.save(best_path)
                print(f"[monitor] new BEST eval {best:.1f} -> {best_path}", flush=True)
                if args.gcs_bucket:
                    from rower_soccer.warp_port.gcs import sync_async
                    sync_async(best_path, args.gcs_bucket, args.run_name)
            if use_wandb:
                import wandb
                wandb.log({"env_step": model.num_timesteps,
                           "eval/video": wandb.Video(vpath, format="mp4"),
                           "eval/ep_rew": score})

        if now - last_ckpt >= args.ckpt_secs:
            last_ckpt = now
            model.save(ckpt_path)
            print(f"[monitor] checkpoint -> {ckpt_path}", flush=True)
            if args.gcs_bucket:
                from rower_soccer.warp_port.gcs import sync_async
                sync_async(ckpt_path, args.gcs_bucket, args.run_name)
                sync_async(os.path.join(run_dir, "config.json"),
                           args.gcs_bucket, args.run_name)

    model.save(ckpt_path)
    if args.gcs_bucket:
        from rower_soccer.warp_port.gcs import sync_async
        sync_async(ckpt_path, args.gcs_bucket, args.run_name)
    print(f"[setup] done in {(time.perf_counter()-t0)/60:.1f}min; saved "
          f"{ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
