import os, sys
sys.path.append("/workspace/Transform2Act"); sys.path.append("/workspace/utmist-vc2-phase2")
os.chdir("/workspace/Transform2Act")
import numpy as np, torch
torch.set_default_dtype(torch.float64)
from design_opt.utils.config import Config
from design_opt.envs import env_dict
P=F=0
def chk(n, ok, d=""):
    global P,F
    if ok: P+=1
    else: F+=1
    print(f"  [{'OK  ' if ok else 'FAIL'}] {n}   {d}")

# 1. log_std at init
from design_opt.agents.transform2act_agent import Transform2ActAgent
for cid, want in (("rtg_e31_s1",-1.5),("rtg_e31f_s1",-1.5),("rtg_e3_s1",0.0),("rtg_e3c_s1",0.0)):
    cfg=Config(cid, tmp=True); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    ag=Transform2ActAgent(cfg=cfg,dtype=torch.float64,device=torch.device("cpu"),
                          seed=cfg.seed,num_threads=1,training=False,checkpoint=0)
    ls=float(ag.policy_net.state_dict()["control_action_log_std"].mean().item())
    sig=np.exp(ls); cost=0.5*8*sig*sig
    chk(f"{cid}: control_log_std == {want}", abs(ls-want)<1e-9,
        f"log_std {ls:+.4f} sigma {sig:.4f} cost/step {cost:.4f}"
        + ("  BELOW the 1.0 survive bonus" if cost<1 else "  above"))
    del ag

# 2. the floor binds
def strip(cid, min_motors=None):
    cfg=Config(cid, tmp=True)
    if min_motors is not None:
        cfg.env_specs=dict(cfg.env_specs); cfg.env_specs["min_motors"]=min_motors
    return cfg, env_dict[cfg.env_name](cfg, agent=None)

for cid, lo in (("rtg_e31f_s1",4),("rtg_e31_s1",0)):
    cfg,env=strip(cid)
    W=env.control_action_dim+env.attr_design_dim+1
    mins=[]
    for ep in range(12):
        rng=np.random.RandomState(ep); env.seed(ep); env.reset()
        for step in range(cfg.skel_transform_nsteps):
            n=len(env.robot.bodies); a=np.zeros((n,W))
            a[:,-1]=2                       # tell EVERY body to remove
            env.step(a)
        mins.append(env.n_actuators())
    got=min(mins)
    if lo:
        chk(f"{cid}: floor holds under all-remove ({cfg.env_specs.get('min_motors')} motors)",
            got>=lo, f"min actuators over 12 destructive episodes = {got} (floor {lo}); counts {sorted(set(mins))}")
    else:
        chk(f"NEG {cid}: WITHOUT the floor the same actions go below 4",
            got<4, f"min actuators = {got}; counts {sorted(set(mins))}")

# 3. E3's own arms unaffected
cfg,env=strip("rtg_e3_s1")
chk("E3's cfg has no min_motors (floor off, E3 reproducible)",
    int(env.env_specs.get("min_motors",0))==0)
print(f"\n=== {P} passed, {F} failed ===")
sys.exit(1 if F else 0)
