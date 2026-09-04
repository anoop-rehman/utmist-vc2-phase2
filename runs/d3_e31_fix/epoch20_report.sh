#!/bin/bash
# D3 M3 E3.1: the epoch-20 verdict. Reports BOTH falsifiers, Outcome C's
# classification, and -- required by D3_E31_FIX.md's falsifier section --
# forward progress and goal rate beside the morphology columns, because a body
# that is growing AND walking and one that is only growing are different
# results the falsifiers cannot separate.
set -uo pipefail
D=/workspace/utmist-vc2-phase2/runs/d3_e31_fix
until [ "$(grep -cE '^[0-9]+	T_sample' $D/logs/train_p_s2.log)" -ge 21 ] \
   && [ "$(grep -cE '^[0-9]+	T_sample' $D/logs/train_p_s3.log)" -ge 21 ]; do sleep 120; done
echo "======== EPOCH 20: the pre-registered falsifier window closes ========"
/workspace/utmist-vc2-phase2/.venv/bin/python - <<'PY'
import csv
CEIL, CRIT = 29, -0.9645
for s in (2, 3):
    f=f"/workspace/utmist-vc2-phase2/runs/d3_e31_fix/census/rtg_e31_s{s}_morph.csv"
    rows=[r for r in csv.DictReader(open(f)) if r.get("epoch")]
    print(f"\n--- rtg_e31_s{s} ---")
    print(f"  {'ep':>3}{'log_std':>10}{'bodies':>8}{'motors':>8}{'passive':>9}"
          f"{'mass':>8}{'pop_mean':>10}{'p_act4':>8}{'fwd m':>8}{'goal':>7}{'fell':>7}")
    for r in rows:
        g=lambda k: r.get(k) or ""
        fwd=f"{float(g('eval_max_fwd')):.2f}" if g('eval_max_fwd') else "-"
        goal=f"{float(g('eval_goal_rate')):.2f}" if g('eval_goal_rate') else "-"
        fell=f"{float(g('eval_fall_rate')):.2f}" if g('eval_fall_rate') else "-"
        pas=int(g('n_bodies'))-1-int(g('n_motors')) if g('n_bodies') and g('n_motors') else ""
        print(f"  {r['epoch']:>3}{float(g('control_log_std')):>10.4f}{g('n_bodies'):>8}"
              f"{g('n_motors'):>8}{pas:>9}{float(g('mass')):>8.3f}"
              f"{float(g('sampled_bodies_mean')):>10.2f}"
              f"{g('p_act4'):>8}{fwd:>8}{goal:>7}{fell:>7}")
    w=[r for r in rows if int(r["epoch"])<=20]
    f1=[r for r in w if r.get("control_log_std") and float(r["control_log_std"])>CRIT]
    f2=[r for r in w if r.get("p_act4") and float(r["p_act4"])==0.0]
    print(f"\n  FALSIFIER 1 (log_std > {CRIT} in first 20): "
          f"{'FIRED at epoch '+f1[0]['epoch'] if f1 else 'NOT FIRED'}")
    print(f"  FALSIFIER 2 (p_act4 -> 0 by epoch 20):      "
          f"{'FIRED at epoch '+f2[0]['epoch'] if f2 else 'NOT FIRED'}")
    # Outcome C: >=27 bodies sustained >=5 epochs with p_act4>=0.9
    run=0; sat=False
    for r in rows:
        if r.get("sampled_bodies_mean") and float(r["sampled_bodies_mean"])>=26 and float(r.get("p_act4") or 0)>=0.9:
            run+=1
            if run>=5: sat=True
        else: run=0
    last=rows[-1]
    fwdv=[float(r["eval_max_fwd"]) for r in rows if r.get("eval_max_fwd")]
    print(f"  OUTCOME C (population >=26 for >=5 epochs, p_act4>=0.9): "
          f"{'SATURATED' if sat else 'not saturated'}   "
          f"(latest population {last.get('sampled_bodies_mean')}, readout "
          f"{last.get('n_bodies')}, ceiling {CEIL})")
    print(f"  mass: {float(rows[0]['mass']):.3f} -> {float(last['mass']):.3f} kg "
          f"(original ant 0.879) -- check against forward progress before proposing a size cost")
    if fwdv:
        print(f"  forward progress: {fwdv[0]:.2f} -> {fwdv[-1]:.2f} m over "
              f"{len(fwdv)} evals   [C1 needs it RISING toward ~1.0 m by ~epoch 84;"
              f" the controls were at 0.21/0.14 m at epoch 9]")
PY
