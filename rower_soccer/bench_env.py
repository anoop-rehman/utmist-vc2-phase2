"""Benchmark creature soccer env stepping throughput (no rendering).

Usage: python -m rower_soccer.bench_env [--steps 1000]
"""

import argparse
import time

import numpy as np


def bench(n_home, n_away, steps):
    from rower_soccer.envs.build import make_soccer_env

    env = make_soccer_env(n_home=n_home, n_away=n_away, time_limit=45.0)
    specs = env.action_spec()

    def random_actions():
        return [np.random.uniform(s.minimum, s.maximum, size=s.shape) for s in specs]

    timestep = env.reset()
    for _ in range(20):
        timestep = env.step(random_actions())
        if timestep.last():
            timestep = env.reset()
    t0 = time.perf_counter()
    for _ in range(steps):
        timestep = env.step(random_actions())
        if timestep.last():
            timestep = env.reset()
    dt = time.perf_counter() - t0
    control_dt = env.control_timestep()
    print(f"{n_home}v{n_away}: {steps/dt:,.0f} env steps/s "
          f"({steps/dt*control_dt:.1f}x realtime, control dt {control_dt*1000:.1f} ms)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    bench(1, 0, args.steps)
    bench(2, 2, args.steps)


if __name__ == "__main__":
    main()
