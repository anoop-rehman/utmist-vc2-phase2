"""D3 M3 E0: does the evolved ant exploit the contact model?

`D3_HANDOFF.md` ("And the thing the video could not show") records that the
reference hopper's limbs go 0.24-0.41 m through the floor against capsule radii
of 0.03-0.08 m, and that the tracking camera hides it -- it showed up only in a
contact probe. That finding is on the HOPPER, whose XML sets `solref=".02 1"`
and geom `density="1000"`. `assets/mujoco_envs/ant.xml` sets neither: its
`<default>` geom carries `density="5.0"` and no solref/solimp override, so the
ant's contact regime is a different one and the hopper's number must not be
carried across. This measures it on the ant instead.

Per execution step, over one mean-action episode:

  * the lowest point of every non-floor geom -- for a capsule, centre minus
    half-length times |z component of its axis| minus radius; for a sphere,
    centre minus radius -- and how far the deepest of them is below z = 0;
  * the number of contacts involving the floor;
  * net x displacement and path length, so "did it locomote" is answerable
    beside "did it cheat" (`displacement_probe.py`'s net/path, same definition).

    cd /workspace/Transform2Act && source env-gpu.sh
    .venv-gpu/bin/python .../t2a_port/e0_body_probe.py --cfg ant_e0_s1 \
        --epoch 100

CPU only -- no CUDA context, safe beside live MPS clients.
"""

import argparse
import os
import sys

sys.path.append("/workspace/Transform2Act")
sys.path.append("/workspace/utmist-vc2-phase2")
os.chdir("/workspace/Transform2Act")

import numpy as np  # noqa: E402
import torch  # noqa: E402


def tensorfy(np_list):
    if isinstance(np_list[0], list):
        return [[torch.tensor(x) for x in y] for y in np_list]
    return [torch.tensor(y) for y in np_list]


def lowest_points(model, data):
    """Lowest world-z of each non-floor geom, exactly as `D3_HANDOFF.md`
    defines it: capsule centre minus half-length |z-axis| minus radius."""
    out = []
    for g in range(model.ngeom):
        name = model.geom_id2name(g)
        if name == "floor":
            continue
        gtype = model.geom_type[g]
        pos = data.geom_xpos[g]
        size = model.geom_size[g]
        if gtype == 3:                      # capsule: size = (radius, halflen)
            zax = data.geom_xmat[g].reshape(3, 3)[:, 2]
            low = pos[2] - size[1] * abs(zax[2]) - size[0]
        else:                               # sphere and everything else
            low = pos[2] - size[0]
        out.append(low)
    return np.asarray(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    p.add_argument("--epoch", default="100")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--untrained", action="store_true",
                   help="no checkpoint: a freshly initialised policy.")
    p.add_argument("--initial-body", action="store_true",
                   help="zero design action, so the body is exactly the task's "
                        "starting XML. Same flag as render_checkpoint.py.")
    args = p.parse_args()

    torch.set_default_dtype(torch.float64)
    from design_opt.agents.transform2act_agent import Transform2ActAgent
    from design_opt.utils.config import Config

    cfg = Config(args.cfg, tmp=False)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    epoch = 0 if args.untrained else (int(args.epoch) if args.epoch.isnumeric()
                                      else args.epoch)
    agent = Transform2ActAgent(cfg=cfg, dtype=torch.float64,
                               device=torch.device("cpu"), seed=cfg.seed,
                               num_threads=1, training=False, checkpoint=epoch)
    env, policy = agent.env, agent.policy_net
    policy.eval()

    state = env.reset()
    if agent.running_state is not None:
        state = agent.running_state(state)
    depths, ncon_floor, xs = [], [], []
    steps = 0
    floor_id = env.model.geom_name2id("floor")
    with torch.no_grad():
        for _ in range(args.max_steps + cfg.skel_transform_nsteps + 1):
            if args.initial_body and env.if_use_transform_action() != 2:
                a = np.zeros((len(env.robot.bodies), env.attr_design_dim + 2))
            else:
                a = policy.select_action(tensorfy([state]),
                                         True).numpy().astype(np.float64)
            state, _, done, info = env.step(a)
            if agent.running_state is not None:
                state = agent.running_state(state)
            if info.get("stage") == "execution":
                steps += 1
                low = lowest_points(env.model, env.data)
                depths.append(-low.min())          # positive = below the floor
                ncon_floor.append(sum(
                    1 for i in range(env.data.ncon)
                    if env.data.contact[i].geom1 == floor_id
                    or env.data.contact[i].geom2 == floor_id))
                xs.append(env.get_body_com("0")[0])
            if done:
                break

    d = np.asarray(depths)
    c = np.asarray(ncon_floor)
    x = np.asarray(xs)
    net = abs(x[-1] - x[0])
    path = float(np.abs(np.diff(x)).sum())
    radii = [env.model.geom_size[g][0] for g in range(env.model.ngeom)
             if env.model.geom_id2name(g) != "floor"]

    print(f"\n{args.cfg} epoch {epoch}: {steps} execution steps, "
          f"{len(env.robot.bodies)} bodies, {env.model.nu} motors")
    print(f"  deepest point below the floor   {max(d.max(), 0):.4f} m")
    print(f"  mean depth below the floor      "
          f"{np.clip(d, 0, None).mean():.4f} m")
    print(f"  steps with >2 cm penetration    "
          f"{100 * (d > 0.02).mean():.1f}%   >10 cm {100 * (d > 0.10).mean():.1f}%")
    print(f"  capsule/sphere radii            "
          f"{min(radii):.3f}-{max(radii):.3f} m")
    print(f"  floor contacts per step         {c.mean():.2f}   "
          f"airborne (zero contacts) on {100 * (c == 0).mean():.1f}% of steps")
    print(f"  net |dx| {net:.2f} m   path {path:.2f} m   "
          f"net/path {net / path if path else float('nan'):.3f}")


if __name__ == "__main__":
    main()
