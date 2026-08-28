"""Sampled (NOT mean-action) episode census in THEIR codebase, plus a weight export.

    cd /workspace/Transform2Act && source env-gpu.sh
    .venv-gpu/bin/python /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/\
their_sampled_probe.py --episodes 200 --seed 1 --npz /tmp/t2a_init_s1.npz

Why this exists: `D3_HANDOFF.md` 2026-08-28 promoted "their untrained sampled
`train_R` is 0.830 and ours is 1.000" to the leading suspect, reading the 0.17
gap as backwards drift their hopper has and ours does not. `train_R` is
`LoggerRL.avg_reward`, i.e. total reward over ALL logged steps, and
`agent.sample_worker` logs the 5 skeleton steps and the 1 attribute step too --
each worth reward 0. So their denominator is 6 larger per episode than the
port's, which counts execution steps only. This probe measures the two halves
separately and prints them side by side so the arithmetic is not the argument.

It also writes the freshly-initialised `policy_dict` to an npz that
`end_probe.py --their-npz` loads, which is what makes the port-side comparison
a comparison at IDENTICAL weights rather than at two different random inits.
"""
import argparse, os, sys
import numpy as np

sys.path.append('/workspace/Transform2Act')
os.chdir('/workspace/Transform2Act')

import torch
from design_opt.utils.config import Config
from design_opt.agents.transform2act_agent import Transform2ActAgent
from khrylib.utils.torch import tensor


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', default='hopper_gpu_s2')
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--npz', default='')
    p.add_argument('--mean-action', action='store_true')
    a = p.parse_args()

    torch.set_default_dtype(torch.float64)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    cfg = Config(a.cfg, tmp=True)
    agent = Transform2ActAgent(cfg=cfg, dtype=torch.float64,
                               device=torch.device('cpu'), seed=a.seed,
                               num_threads=1, training=True, checkpoint=0)
    if a.npz:
        sd = agent.policy_net.state_dict()
        np.savez(a.npz, **{f'policy_dict|{k}': v.cpu().numpy()
                           for k, v in sd.items()})
        print(f'wrote {a.npz}: {len(sd)} tensors')

    env, pol = agent.env, agent.policy_net
    from design_opt.agents.transform2act_agent import tensorfy
    rows = []
    with torch.no_grad():
        for ep in range(a.episodes):
            state = env.reset()
            n_design = n_exec = 0
            exec_ret = 0.0
            x0 = None
            for t in range(10000):
                act = pol.select_action(tensorfy([state]),
                                        a.mean_action).numpy().astype(np.float64)
                if env.stage == 'execution' and x0 is None:
                    x0 = float(env.sim.data.qpos[0])
                state, r, done, info = env.step(act)
                if info['stage'] == 'execution':
                    n_exec += 1
                    exec_ret += r
                else:
                    n_design += 1
                if done:
                    break
            x1 = float(env.sim.data.qpos[0])
            rows.append((n_design, n_exec, exec_ret, x1 - (x0 or 0.0),
                         len(env.robot.bodies)))
    d = np.array([r[0] for r in rows], float)
    e = np.array([r[1] for r in rows], float)
    R = np.array([r[2] for r in rows], float)
    dx = np.array([r[3] for r in rows], float)
    nb = np.array([r[4] for r in rows], float)
    dt = env.dt
    print(f'\nTHEIR env, mean_action={a.mean_action}, {a.episodes} episodes, '
          f'seed {a.seed}, dt={dt}')
    print(f'  design steps/ep      {d.mean():.2f}  (all reward 0, all logged '
          f'by LoggerRL.step)')
    print(f'  EXEC steps/ep        {e.mean():.2f}   median {np.median(e):.0f}  '
          f'max {e.max():.0f}')
    print(f'  total steps/ep       {(d + e).mean():.2f}  <- their '
          f'avg_episode_len')
    print(f'  exec return/ep       {R.mean():.2f}')
    print(f'  R per EXEC step      {R.sum() / e.sum():.4f}')
    print(f'  R per TOTAL step     {R.sum() / (d + e).sum():.4f}  <- their '
          f'train_R')
    print(f'  mean dx per episode  {dx.mean():+.4f} m   '
          f'(std {dx.std():.4f}, n={len(dx)})')
    print(f'  mean dx/dt per step  {dx.sum() / (e.sum() * dt):+.4f} m/s')
    print(f'  bodies/design        {nb.mean():.2f}  min {nb.min():.0f}  '
          f'max {nb.max():.0f}')
    q = np.percentile(e, [10, 25, 50, 75, 90])
    print(f'  exec-len pct 10/25/50/75/90  '
          f'{q[0]:.0f} {q[1]:.0f} {q[2]:.0f} {q[3]:.0f} {q[4]:.0f}')
    np.save('/tmp/their_exec_lens.npy', e)


if __name__ == '__main__':
    main()
