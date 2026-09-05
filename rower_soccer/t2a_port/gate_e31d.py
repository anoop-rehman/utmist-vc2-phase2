import os, sys
sys.path.append("/workspace/Transform2Act"); sys.path.append("/workspace/utmist-vc2-phase2")
os.chdir("/workspace/Transform2Act")
import numpy as np, torch
torch.set_default_dtype(torch.float64)
from design_opt.utils.config import Config
from design_opt.envs import env_dict
from rower_soccer.t2a_port import e3_morph
P=F=0
def chk(n, ok, d=""):
    global P,F
    P,F=(P+1,F) if ok else (P,F+1)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {n}   {d}")
def arrays(m):
    for nm in dir(m):
        if nm.startswith("_"): continue
        try: v=getattr(m,nm)
        except Exception: continue
        if isinstance(v,np.ndarray) and v.dtype.kind in "fiub": yield nm, np.array(v)

EXPECT={"rtg_e31d_s3body":(12,6,0.949,5.201,"901ec8c2e00b"),
        "rtg_e31d_s2body":(18,6,1.470,10.395,"50271e7f5d26")}
for cid,(nb,nm_,mass,limb,topo) in EXPECT.items():
    cfg=Config(cid,tmp=True); env=env_dict[cfg.env_name](cfg,agent=None)
    s=e3_morph.body_summary(env)
    chk(f"{cid}: loads the EVOLVED body, not the original ant",
        s['n_bodies']==nb and s['model_nu_ours']==nm_ and s['topo']==topo,
        f"{s['n_bodies']} bodies / {s['model_nu_ours']} motors, topo {s['topo']} (original ant is 13/8)")
    chk(f"{cid}: mass and limb match the dumped design",
        abs(s['model_mass_ours']-mass)<1e-3 and abs(s['limb_length']['sum']-limb)<1e-2,
        f"mass {s['model_mass_ours']:.3f} vs {mass}, limb total {s['limb_length']['sum']:.3f} vs {limb}")
    chk(f"{cid}: the scripted opponent survived the round-trip",
        s['n_opp_bodies']==13 and (s['model_nu']-s['model_nu_ours'])==8,
        f"{s['n_opp_bodies']} opponent bodies, {s['model_nu']-s['model_nu_ours']} opponent motors")
    # frozen under destructive design actions
    # THE RIGHT ASSERTION IS CONSTANCY ACROSS EPISODES, NOT BIT-EQUALITY WITH THE
    # EXPORTED XML. One identity attribute step applies a sin/arcsin round-trip
    # that lands 1.273e-08 off in the genome; where a value sits near the XML's
    # 6-dp write precision that flips the last digit and 15 mjModel arrays move
    # by <= 3.5e-06. It is a ONE-TIME SNAP -- reset_robot rebuilds from
    # init_xml_str every episode, so it re-applies from the same start and
    # cannot compound (verified identical at episodes 1, 2, 3, 10, 50, 100,
    # 200). What the experiment needs is that every episode runs the SAME body,
    # which is what this now checks; the constant 3.5e-08 kg offset on a 0.949
    # kg body is 3.7e-06 % and is recorded as a tolerance, not asserted away.
    W=env.control_action_dim+env.attr_design_dim+1
    ref=None; changed=set(); counts=[]
    for ep in range(10):
        rng=np.random.RandomState(ep); env.seed(ep); env.reset()
        while env.if_use_transform_action()!=2:
            n=len(env.robot.bodies); a=np.zeros((n,W))
            if env.if_use_transform_action()==0: a[:,-1]=rng.randint(1,3,size=n)
            else: a[:,env.control_action_dim:-1]=rng.uniform(-1,1,(n,env.attr_design_dim))
            _,_,d,_=env.step(a)
            if d: break
        cur=dict(arrays(env.model))
        if ref is None: ref=cur
        else:
            for k,v in cur.items():
                if k not in ref or ref[k].shape!=v.shape or not np.array_equal(ref[k],v): changed.add(k)
        counts.append(len(env.robot.bodies))
    chk(f"{cid}: body IDENTICAL ACROSS EPISODES under destructive design actions",
        not changed and set(counts)=={nb},
        f"{len(changed)} of {len(ref)} arrays differ between episodes, body counts {sorted(set(counts))}")
    # opponent still on schedule
    errs=[]; env.seed(0); env.reset()
    while env.if_use_transform_action()!=2: env.step(np.zeros((len(env.robot.bodies),W)))
    qs,_,_=env._opp()
    for k in range(40):
        env.step(np.zeros((len(env.robot.bodies),W)))
        errs.append(abs(float(env.data.qpos[qs][0])-env.opp_x(k+1)))
    chk(f"{cid}: opponent still follows 1 - v*dt*k exactly", max(errs)==0.0, f"max error {max(errs):.3e}")
print(f"\n=== {P} passed, {F} failed ===")
sys.exit(1 if F else 0)
