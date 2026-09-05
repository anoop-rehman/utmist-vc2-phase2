# D3 M3 E4 two-lineage self-play — ARCHIVED, not abandoned

Built, gated 11/11, launched, and stopped by stop-file at epoch ~14/400 when
the user redirected to a **shared-weight ring** setup (E4R). Roughly one hour
of compute was spent and ~60 h freed.

**Nothing here was found to be wrong.** The machinery works: the pi-z rotation
is exact (observation max|delta| 0.000e+00; a snapshot ran 4.657 m/s in slot 1
against 4.891 trained in slot 0), the snapshot exchange self-synchronises, and
both arms were logging clean instruments when stopped.

Kept because **the divergence question may come back**. What is here:

* `train_e4_gnn.py`      two-lineage trainer with the snapshot exchange
* `e4_selfplay.py`       atomic publish/load, race guards
* `e4_divergence.py`     the pre-registered Delta(e) = D_self - D_null statistic
* `launch_e4.sh`         one seed pair per wave
* `autolaunch_e4_wave1.sh` detached launcher that refuses rather than squeezes

Still live in the tree and shared with E4R (do NOT treat as archived):
`design_opt/envs/run_to_goal_sp.py`, `rtg_scene.build(opponent_src=...)`,
`gate_e4.py`, `e4_null_traj.py`, `e4_compset.py`, and the null/comparison-set
JSONs — E4R reuses all of them.

Why the redirect is well-founded: E3.1 established that **the design head is
blind** (skeleton and attribute stages see only `attr_fixed ++ attr_design`,
never simulation state), so observation-conditioned specialisation between two
lineages is impossible by construction. That handicaps the divergence question
but is irrelevant to a shared-weight ratchet, which is also half the compute.
