#!/usr/bin/env bash
# Stage-2 dribble (wide-cone steering): 4 DIVERSE IDEAS x 3 hyperparams = 12 arms.
#
# cur2 failed because it jumped cone 0deg -> 90deg in one step: the colinear-push
# prior lands at the flat spawn floor (0.23, vs 0.19 bare) with no gradient toward
# steering, and topples (1.3M diverged) flailing at unreachable lateral targets.
# The shared fix is an ANNEALED cone (15deg -> 360deg) so the reward signal never
# goes flat. On top of that backbone, each idea attacks a different failure mode:
#
#   A pace    -- HOW FAST to widen (the curriculum-step size that broke cur2)
#   B shape   -- fill the flat desert with a dense ball->target progress reward
#   C explore -- keep exploration alive INTO the wide-cone regime (entropy floor)
#   D reg     -- stop the toppling that ate cur2's signal (CAPS smooth + energy)
#
# All warm-start from cur1 (the colinear expert) and run ~10h on the shared A4000.
# Staggered 8s; setsid+nohup so a disconnect can't kill them; GCS-synced to resume.
set -u
cd /workspace/utmist-vc2-phase2
mkdir -p logs
BUCKET=vc2-2026-checkpoints
INIT=runs_v2/_init_cur1_ws.pt

# Shared backbone: warm-start cur1, colinear stage-1 geometry, annealed cone
# 15deg (0.26 rad) -> 360deg (pi). 1024 worlds so 12 fit on one 16GB GPU.
BASE="--worlds 1024 --steps 20000000000 --max-hours 10 \
  --init-from $INIT --fixed-start \
  --ball-spawn 0.8 0.8 --target-speed 0 0 --target-dist 2.0 5.0 --reward-coef 0.5 \
  --cone-start 0.26 --cone-max 3.14159 \
  --ent-floor -1.2 --ent-ceil 0.0 \
  --first-video-secs 90 --video-secs 1200 --ckpt-secs 1800 --gcs-bucket $BUCKET"

# Curriculum-pacing default for the ideas that don't vary it (B, C, D): medium.
CONE_MED="--cone-anneal-steps 1000000000"
ENT_DEFAULT="--ent-anneal-steps 400000000"   # cur1's schedule (C overrides this)

launch () {  # run_name extra_args...
  local name=$1; shift
  setsid nohup env MUJOCO_GL=egl .venv/bin/python \
    -m rower_soccer.warp_port.train_dribble_warp \
    --run-name "$name" $BASE "$@" \
    > "logs/${name}.log" 2>&1 < /dev/null &
  echo "launched $name (pid $!)"
  sleep 8
}

# --- Idea A: curriculum pacing (vary cone-anneal speed) --------------------
launch s2_pace_fast  --cone-anneal-steps 400000000  $ENT_DEFAULT
launch s2_pace_med   --cone-anneal-steps 1000000000 $ENT_DEFAULT   # == the reference
launch s2_pace_slow  --cone-anneal-steps 2000000000 $ENT_DEFAULT

# --- Idea B: dense reward shaping (fill the desert; vary progress-scale) ----
launch s2_shape_lo   $CONE_MED $ENT_DEFAULT --reward-mode progress --progress-scale 1.0 --approach-scale 0.5
launch s2_shape_md   $CONE_MED $ENT_DEFAULT --reward-mode progress --progress-scale 2.0 --approach-scale 0.5
launch s2_shape_hi   $CONE_MED $ENT_DEFAULT --reward-mode progress --progress-scale 4.0 --approach-scale 0.5

# --- Idea C: sustained exploration (no ent anneal; vary the entropy floor) --
launch s2_expl_lo    $CONE_MED --ent-anneal-steps 0 --ent-floor -1.0
launch s2_expl_md    $CONE_MED --ent-anneal-steps 0 --ent-floor -0.5
launch s2_expl_hi    $CONE_MED --ent-anneal-steps 0 --ent-floor 0.0

# --- Idea D: anti-topple regularization (CAPS smoothness + energy) ----------
launch s2_reg_lo     $CONE_MED $ENT_DEFAULT --smooth-coef 0.05
launch s2_reg_md     $CONE_MED $ENT_DEFAULT --smooth-coef 0.1  --energy-coef 0.02
launch s2_reg_hi     $CONE_MED $ENT_DEFAULT --smooth-coef 0.2  --energy-coef 0.05

echo "all 12 stage-2 idea arms launched."
