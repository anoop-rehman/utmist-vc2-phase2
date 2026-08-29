"""How hard does the optimizer press on the design bounds?

`D3_HANDOFF.md` ("The optimizer presses on its bounds") records that 32% of the
reference hopper's capsules sit at the minimum radius and 18% of its gears at the
maximum, over 40 SAMPLED epoch-1000 designs -- "with no energy cost in the
reward, the optimum is as light and as strongly actuated as the bounds allow".
Whether their ant does the same decides whether that design space is safe to put
on a soccer pitch, and it has to be measured the same way to be comparable: 40
sampled designs, not one mean-action design, which is a single draw and can miss
a mode entirely.

"At the bound" is within 1% of the parameter's range. A tighter equality test
(1e-6) reports 2.5% where this reports 44%, because the projected parameters
land near but not exactly on the bound; 1% of the range is what "sits at the
bound" means physically.

    cd /workspace/Transform2Act && source env-gpu.sh
    .venv-gpu/bin/python .../t2a_port/e0_bounds_probe.py ant_e0_s1 100 40
    .venv-gpu/bin/python .../t2a_port/e0_bounds_probe.py hopper_gpu 1000 40

CPU only -- no CUDA context, safe beside live MPS clients.
"""
import os, sys, collections
sys.path.append("/workspace/Transform2Act"); sys.path.append("/workspace/utmist-vc2-phase2")
os.chdir("/workspace/Transform2Act")
import numpy as np, torch
torch.set_default_dtype(torch.float64)
from design_opt.agents.transform2act_agent import Transform2ActAgent
from design_opt.utils.config import Config
from khrylib.utils.torch import to_test

def tf(l):
    return [[torch.tensor(x) for x in y] for y in l] if isinstance(l[0], list) else [torch.tensor(y) for y in l]

cfg_id, epoch, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cfg = Config(cfg_id, tmp=False)
np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
ag = Transform2ActAgent(cfg=cfg, dtype=torch.float64, device=torch.device("cpu"),
                        seed=cfg.seed, num_threads=1, training=False, checkpoint=epoch)
env, pol = ag.env, ag.policy_net
to_test(pol)
gp = cfg.robot_cfg["geom_params"]; ap = cfg.robot_cfg["actuator_params"]
rlb, rub = gp["size"]["lb"], gp["size"]["ub"]
glb, gub = ap["gear"]["lb"], ap["gear"]["ub"]
rad, gear = [], []
for _ in range(n):
    s = env.reset()
    for _ in range(cfg.skel_transform_nsteps + 1):
        with torch.no_grad():
            a = pol.select_action(tf([s]), False).numpy().astype(np.float64)
        s, _, d, i = env.step(a)
        if d or i.get("stage") == "execution": break
    for b in env.robot.bodies:
        g = b.geoms[0]
        if g.type == "capsule":
            rad.append(float(np.asarray(g.size).reshape(-1)[0]))
        if b.joints and b.joints[0].actuator is not None:
            gear.append(float(b.joints[0].actuator.gear))
rad, gear = np.array(rad), np.array(gear)
tol_frac = 0.01   # "at the bound" = within 1% of the range of it
print(f"{cfg_id} epoch {epoch}: {n} sampled designs, {len(rad)} capsules, {len(gear)} actuators")
for nm, v, lb, ub, fmt in (("radius", rad, rlb, rub, ".4f"),
                           ("gear  ", gear, glb, gub, ".1f")):
    t = tol_frac * (ub - lb)
    print(f"  {nm}  range [{lb}, {ub}]  observed "
          f"{format(v.min(), fmt)}-{format(v.max(), fmt)}  "
          f"at MIN {100*(v <= lb + t).mean():.1f}%  "
          f"at MAX {100*(v >= ub - t).mean():.1f}%   (within 1% of range)")
