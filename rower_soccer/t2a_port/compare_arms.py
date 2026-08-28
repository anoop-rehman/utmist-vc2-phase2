"""Training-curve table across the port's arms and their two reference seeds.

    cd /workspace/utmist-vc2-phase2
    .venv/bin/python rower_soccer/t2a_port/compare_arms.py 0 5 10 15 20 25

The one thing this file exists to get right: **their `train_R` and the port's
are not the same quantity.** Theirs is `LoggerRL.avg_reward`, total reward over
every logged step, and `khrylib/rl/agents/agent.py:70` logs the 5 skeleton and
1 attribute step too (reward 0 each). So `train_R_eps / train_R` on a reference
log is `exec_steps + 6`, and the `- 6` below is not a fudge. The port logs its
own `train_len` in the JSON sidecar and needs no correction. `exec_R` is
execution-only on both sides (`design_opt/utils/logger.py:22`), so the eval
column is directly comparable as printed.

See D3_HANDOFF.md, "Update 2026-08-28 (second)".
"""
import re, json, sys
def port(p):
    out={}
    try: fh=open(p)
    except OSError: return out
    for l in fh:
        m=re.match(r'^(\d+)\t.*train_R_eps ([\d.eE+-]+)\texec_R ([\d.eE+-]+)\texec_R_eps ([\d.eE+-]+)', l)
        if m:
            e=int(m.group(1)); out.setdefault(e,{}).update(R_eps=float(m.group(2)), ev=float(m.group(4)))
        if l.strip().startswith('{'):
            d=json.loads(l.strip()); out.setdefault(d['epoch'],{})['len']=d['train_len']
    return out
def ref(p):
    out={}
    for l in open(p):
        m=re.match(r'^(\d+)\t.*train_R ([\d.eE+-]+)\ttrain_R_eps ([\d.eE+-]+)\texec_R ([\d.eE+-]+)\texec_R_eps ([\d.eE+-]+)', l)
        if m:
            e=int(m.group(1)); R=float(m.group(2)); Re=float(m.group(3))
            out[e]={'R_eps':Re,'len':Re/R-6,'ev':float(m.group(5))}
    return out
A={'port_s1_init (fixed)':port('runs/t2a_port/port_s1_init/log_train.txt'),
   'port_s1 (bug, bd off)':port('runs/t2a_port/port_s1/log_train.txt'),
   'port_s1_bd (bug, bd on)':port('runs/t2a_port/port_s1_bd/log_train.txt'),
   'ref seed 1':ref('/workspace/Transform2Act/results_hopper_gpu.log'),
   'ref seed 2':ref('/workspace/Transform2Act/results_hopper_gpu_s2.log')}
eps=[int(x) for x in sys.argv[1:]] or [0,5,10,15,20,25]
hdr=f"{'arm':<24}"+''.join(f"{'ep '+str(e):>18}" for e in eps)
print(hdr); print('-'*len(hdr))
for k,d in A.items():
    print(f"{k:<24}"+''.join(
        (f"{d[e].get('len',0):8.1f}/{d[e].get('R_eps',0):9.1f}" if e in d else f"{'-':>18}")
        for e in eps))
print("\n(train exec-episode length / train_R_eps)\n")
print(f"{'arm':<24}"+''.join(f"{'ep '+str(e):>12}" for e in eps))
for k,d in A.items():
    print(f"{k:<24}"+''.join((f"{d[e].get('ev',0):12.1f}" if e in d else f"{'-':>12}") for e in eps))
print("\n(eval exec_R_eps)")
