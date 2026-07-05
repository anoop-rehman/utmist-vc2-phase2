"""Training monitoring: throughput, ETA, and periodic eval-video logging.

Console lines are grep-friendly:
    [monitor] step=1,234,567/10,000,000 (12.3%) fps=8,412 eta=17.4min ep_rew=0.312
Videos land in <run_dir>/videos/eval_step_<N>.mp4 (also logged to TensorBoard
as scalars; videos on disk to keep TB light).
"""

import os
import time

import imageio
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ProgressMonitor(BaseCallback):
    def __init__(self, total_timesteps, run_dir, eval_env_factory=None,
                 video_every_seconds=300.0, video_episodes=1, verbose=1,
                 use_wandb=False):
        super().__init__(verbose)
        self.total = total_timesteps
        self.run_dir = run_dir
        self.eval_env_factory = eval_env_factory
        self.video_every_seconds = video_every_seconds
        self.video_episodes = video_episodes
        self.use_wandb = use_wandb
        self._t0 = None
        self._last_video_time = None
        os.makedirs(os.path.join(run_dir, "videos"), exist_ok=True)

    def _on_training_start(self):
        self._t0 = time.perf_counter()
        self._step0 = self.num_timesteps
        if self.eval_env_factory is not None:
            self._record_video()  # step-0 baseline video

    def _on_rollout_end(self):
        elapsed = time.perf_counter() - self._t0
        done = self.num_timesteps - self._step0
        fps = done / max(elapsed, 1e-9)
        remaining = max(self.total - self.num_timesteps, 0)
        eta_min = remaining / max(fps, 1e-9) / 60
        buf = self.model.ep_info_buffer
        ep_rew = np.mean([e["r"] for e in buf]) if buf else float("nan")
        print(f"[monitor] step={self.num_timesteps:,}/{self.total:,} "
              f"({100*self.num_timesteps/self.total:.1f}%) fps={fps:,.0f} "
              f"eta={eta_min:.1f}min ep_rew={ep_rew:.3f}", flush=True)
        self.logger.record("monitor/fps_measured", fps)
        self.logger.record("monitor/eta_minutes", eta_min)

    def _on_step(self):
        if (self.eval_env_factory is not None and
                time.perf_counter() - self._last_video_time >= self.video_every_seconds):
            self._record_video()
        return True

    def _record_video(self):
        self._last_video_time = time.perf_counter()
        env = getattr(self, "_eval_env", None)
        if env is None:
            env = self._eval_env = self.eval_env_factory()
        path = os.path.join(self.run_dir, "videos",
                            f"eval_step_{self.num_timesteps:010d}.mp4")
        fps_video = 40
        with imageio.get_writer(path, fps=fps_video, quality=7) as w:
            for _ in range(self.video_episodes):
                obs, _ = env.reset()
                done = False
                ep_rew, steps = 0.0, 0
                while not done:
                    if self.model is not None and self.num_timesteps > 0:
                        action, _ = self.model.predict(obs, deterministic=True)
                    else:
                        action = env.action_space.sample()
                    obs, r, term, trunc, _ = env.step(action)
                    done = term or trunc
                    ep_rew += r
                    steps += 1
                    w.append_data(env.render())
        print(f"[monitor] video: {path} (eval ep_rew={ep_rew:.2f} over {steps} steps)",
              flush=True)
        if self.use_wandb:
            import wandb
            wandb.log({"eval/video": wandb.Video(path, format="mp4"),
                       "eval/ep_rew": ep_rew}, step=self.num_timesteps)
