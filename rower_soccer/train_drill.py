"""Stage-1+ drill training entry point (PPO behind a swappable interface).

Usage:
    MUJOCO_GL=egl .venv/bin/python -m rower_soccer.train_drill \
        --drill follow --creature worm --steps 10000000 --procs 8 \
        --run-name follow_v1
"""

import argparse
import json
import os
import subprocess

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from rower_soccer.drills.gym_wrap import DrillGymEnv
from rower_soccer.models.latent_policy import LatentActorCriticPolicy
from rower_soccer.monitor import ProgressMonitor

DRILLS = {}


def _register_drills():
    from rower_soccer.drills.dribble import make_dribble_env
    from rower_soccer.drills.follow import make_follow_env
    DRILLS["follow"] = make_follow_env
    DRILLS["dribble"] = make_dribble_env


def make_gym_env(drill, creature, seed):
    def factory(random_state=None):
        return DRILLS[drill](random_state=random_state, creature_kind=creature)
    return DrillGymEnv(factory, seed=seed)


def main():
    _register_drills()
    p = argparse.ArgumentParser()
    p.add_argument("--drill", default="follow", choices=list(DRILLS))
    p.add_argument("--creature", default="worm")
    p.add_argument("--steps", type=int, default=10_000_000)
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--z-dim", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--ent-coef", type=float, default=0.003)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--run-name", required=True)
    p.add_argument("--video-secs", type=float, default=300.0,
                   help="wallclock seconds between eval videos (0 disables)")
    p.add_argument("--load", default=None, help="checkpoint to resume from")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--wandb-project", default="creature-soccer")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    run_dir = os.path.join("runs_v2", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    config = {**vars(args), "git_sha": git_sha}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=1)

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.run_name,
                   config=config, sync_tensorboard=True, dir=run_dir)

    # probe layout once
    probe = make_gym_env(args.drill, args.creature, seed=0)
    layout = dict(proprio_indices=probe.proprio_indices.tolist(),
                  task_indices=probe.task_indices.tolist())
    print(f"[setup] obs_dim={probe.obs_dim} proprio={len(probe.proprio_indices)} "
          f"task={len(probe.task_indices)} act={probe.action_space.shape}", flush=True)

    venv = SubprocVecEnv(
        [lambda i=i: make_gym_env(args.drill, args.creature, seed=1000 + i)
         for i in range(args.procs)], start_method="forkserver")
    venv = VecMonitor(venv)

    policy_kwargs = dict(extractor_kwargs=dict(
        proprio_indices=layout["proprio_indices"],
        task_indices=layout["task_indices"],
        z_dim=args.z_dim))

    if args.load:
        model = PPO.load(args.load, env=venv, tensorboard_log=run_dir)
    else:
        model = PPO(
            LatentActorCriticPolicy, venv,
            learning_rate=args.lr, n_steps=args.n_steps,
            batch_size=args.batch_size, n_epochs=10,
            gamma=args.gamma, gae_lambda=0.95, ent_coef=args.ent_coef,
            policy_kwargs=policy_kwargs, tensorboard_log=run_dir,
            verbose=1, device=args.device)

    monitor = ProgressMonitor(
        total_timesteps=args.steps, run_dir=run_dir,
        eval_env_factory=(lambda: make_gym_env(args.drill, args.creature, seed=7))
        if args.video_secs > 0 else None,
        video_every_seconds=args.video_secs,
        use_wandb=use_wandb)

    try:
        model.learn(total_timesteps=args.steps, callback=monitor,
                    progress_bar=False, reset_num_timesteps=not bool(args.load))
    finally:
        path = os.path.join(run_dir, "final_model.zip")
        model.save(path)
        print(f"[setup] saved {path}", flush=True)


if __name__ == "__main__":
    main()
