"""D3 M3 E1.1 post-hoc check: was the body frozen for the WHOLE trained run,
and how does each arm actually move?

`gate_e11_identity.py` gates the mechanism with destructive RANDOM design
actions before training. This is the complementary check afterwards: drive the
design stages with the **trained policy's own actions** and confirm every
mjModel array is still identical to the initial body. A gate that only ever saw
random actions could in principle miss a policy that learned to exploit some
path; this closes that.

It also measures the behaviour of either arm with one common instrument, so the
GNN and MLP numbers in the write-up come from the same code.

    .venv-gpu/bin/python .../t2a_port/e11_posthoc_check.py --arm gnn --cfg ant_e11_gnn_s1 --epoch 100
    .venv-gpu/bin/python .../t2a_port/e11_posthoc_check.py --arm mlp --cfg ant_e11_mlp_s1 --epoch 99
"""
import argparse, os, sys
sys.path.append("/workspace/Transform2Act"); sys.path.append("/workspace/utmist-vc2-phase2")
os.chdir("/workspace/Transform2Act")
import numpy as np, torch

def arrays(m):
    for nm in dir(m):
        if nm.startswith("_"): continue
        try: v = getattr(m, nm)
        except Exception: continue
        if isinstance(v, np.ndarray) and v.dtype.kind in "fiub": yield nm, np.array(v)

def lowest(model, data):
    out=[]
    for g in range(model.ngeom):
        if model.geom_id2name(g)=="floor": continue
        p=data.geom_xpos[g]; s=model.geom_size[g]
        if model.geom_type[g]==3:
            z=data.geom_xmat[g].reshape(3,3)[:,2]; out.append(p[2]-s[1]*abs(z[2])-s[0])
        else: out.append(p[2]-s[0])
    return np.asarray(out)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--arm",choices=["gnn","mlp"],required=True)
    p.add_argument("--cfg",required=True); p.add_argument("--tag",default=None)
    p.add_argument("--epoch",default="100"); p.add_argument("--episodes",type=int,default=5)
    p.add_argument("--seed-base",type=int,default=1000)
    p.add_argument("--stochastic",action="store_true",help="sample actions instead of taking the mean -- the protocol the TRAINING log reports, which is not comparable across arms with different learned action noise")
    a=p.parse_args()
    torch.set_default_dtype(torch.float64)
    from design_opt.utils.config import Config
    from design_opt.envs.ant import AntEnv
    cfg=Config(a.cfg,tmp=False)
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)

    if a.arm=="gnn":
        from design_opt.agents.transform2act_agent import Transform2ActAgent
        from khrylib.utils.torch import to_test
        ag=Transform2ActAgent(cfg=cfg,dtype=torch.float64,device=torch.device("cpu"),
                              seed=cfg.seed,num_threads=1,training=False,checkpoint=int(a.epoch))
        env,pol,rs=ag.env,ag.policy_net,ag.running_state; to_test(pol)
        def tf(l): return [[torch.tensor(x) for x in y] for y in l] if isinstance(l[0],list) else [torch.tensor(y) for y in l]
    else:
        from rower_soccer.t2a_port.train_e11_mlp import Actor,RunningNorm,flat_obs
        env=AntEnv(cfg,agent=None)
        d=f"/workspace/Transform2Act/results/{a.cfg}"+(f"_{a.tag}" if a.tag else "")
        blob=torch.load(os.path.join(d,f"epoch_{int(a.epoch):04d}.p"),map_location="cpu")
        names=list(env.model.actuator_names)
        rows=[i for i,b in enumerate(env.robot.bodies) if i>0 and b.get_actuator_name() in names]
        od=flat_obs(env.reset()).shape[0]
        actor=Actor(od,env.model.nu,[64,64],0.0); actor.load_state_dict(blob["actor"]); actor.eval()
        norm=RunningNorm(od); norm.load(blob["norm"])

    ref=dict(arrays(env.model)); nref=len(ref); changed=set()
    W=env.control_action_dim+env.attr_design_dim+1
    stats=[]
    for ep in range(a.episodes):
        np.random.seed(a.seed_base+ep); torch.manual_seed(a.seed_base+ep); env.np_random.seed(a.seed_base+ep)
        state=env.reset()
        if a.arm=="gnn" and rs is not None: state=rs(state)
        while env.if_use_transform_action()!=2:
            if a.arm=="gnn":
                with torch.no_grad(): act=pol.select_action(tf([state]),True).numpy().astype(np.float64)
            else: act=np.zeros((len(env.robot.bodies),W))
            state,_,done,_=env.step(act)
            if a.arm=="gnn" and rs is not None: state=rs(state)
            for nm,v in arrays(env.model):
                if nm not in ref or ref[nm].shape!=v.shape or not np.array_equal(ref[nm],v): changed.add(nm)
            if done: break
        R=0.0;n=0;xs=[];con=[];dep=[]
        fid=env.model.geom_name2id("floor")
        while n<1000:
            if a.arm=="gnn":
                with torch.no_grad(): act=pol.select_action(tf([state]),not a.stochastic).numpy().astype(np.float64)
            else:
                o=norm(flat_obs(state))
                with torch.no_grad(): mu,_=actor.select_action(torch.as_tensor(o).unsqueeze(0),not a.stochastic)
                act=np.zeros((len(env.robot.bodies),W)); act[rows,0]=mu.numpy()[0]
            state,r,done,_=env.step(act)
            if a.arm=="gnn" and rs is not None: state=rs(state)
            R+=r;n+=1; xs.append(env.get_body_com("0")[0])
            dep.append(-lowest(env.model,env.data).min())
            con.append(sum(1 for i in range(env.data.ncon) if env.data.contact[i].geom1==fid or env.data.contact[i].geom2==fid))
            if done: break
        for nm,v in arrays(env.model):
            if nm not in ref or ref[nm].shape!=v.shape or not np.array_equal(ref[nm],v): changed.add(nm)
        x=np.asarray(xs); net=abs(x[-1]-x[0]); path=float(np.abs(np.diff(x)).sum())
        stats.append((R,n,net,net/path if path else 0.0,100*np.mean(np.asarray(con)==0),max(max(dep),0)))
    print(f"\n{a.arm.upper()} {a.cfg}{'_'+a.tag if a.tag else ''} epoch {a.epoch}: "
          f"{len(env.robot.bodies)} bodies / {env.model.nu} motors")
    print(f"  BODY FROZEN across {a.episodes} episodes driven by the TRAINED policy: "
          f"{'YES -- '+str(nref)+' arrays identical' if not changed else 'NO -- CHANGED: '+str(sorted(changed))}")
    R,n,net,npx,air,dp=map(lambda i:[s[i] for s in stats],range(6))
    print(f"  exec return  mean {np.mean(R):8.1f}   per-episode {[f'{v:.0f}' for v in R]}")
    print(f"  ep length    mean {np.mean(n):8.1f}   net |dx| mean {np.mean(net):.1f} m  net/path {np.mean(npx):.3f}")
    print(f"  airborne     mean {np.mean(air):8.1f}%  deepest below floor {max(dp):.4f} m")
    if a.arm=="mlp": std=float(actor.log_std.exp().mean().item())
    else:
        ls=[v for k,v in pol.state_dict().items() if k=="control_action_log_std"]
        std=float(ls[0].exp().mean().item()) if ls else float("nan")
    print(f"  learned action std {std:.4f}   (protocol: {'STOCHASTIC' if a.stochastic else 'mean-action'})")

if __name__=="__main__": main()
