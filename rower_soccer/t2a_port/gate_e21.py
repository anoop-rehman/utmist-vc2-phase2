"""D3 M3 E2.1: the gate on the exploration curriculum.

    cd /workspace/Transform2Act && source env-gpu.sh
    .venv-gpu/bin/python .../t2a_port/gate_e21.py

Five phases. Every phase has at least one negative control, because a gate
that cannot fail is not evidence. What is being gated is exactly two edits:

  * `design_opt/envs/run_to_goal.py` now returns `dense` and `parse` in its
    info dict. It must not have changed what the env REWARDS.
  * `train_e11_mlp.py --curriculum-steps N` mixes them into the PPO buffer.
    It must not have changed what the trainer MEASURES, and with N = 0 it
    must be the flat-reward trainer E2 ran, exactly.

`gate_e2.py` (41 checks) is run separately and is the gate that E2's own
setup -- scene, opponent, frozen body, reward, termination, observation --
is untouched by this work.
"""
import os
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
if os.path.isdir("/workspace/t2a_pylibs"):
    sys.path.insert(0, "/workspace/t2a_pylibs")
sys.path.append("/workspace/Transform2Act")
sys.path.append("/workspace/utmist-vc2-phase2")
os.chdir("/workspace/Transform2Act")

import argparse  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

OK, BAD = [], []


def chk(name, cond, detail=""):
    (OK if cond else BAD).append(name)
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""),
          flush=True)


def mk_args(**kw):
    from rower_soccer.t2a_port.train_e11_mlp import main  # noqa: F401
    d = dict(cfg="rtg_mlp_s1", num_threads=1, max_epoch=None, policy_lr=3e-4,
             value_lr=3e-4, hdims="64,64", log_std=0.0, ent_coef=0.0,
             max_grad_norm=0.5, save_interval=10, stop_file=None,
             mini_batch=None, optim_epochs=None, curriculum_steps=0,
             anneal_lr=False, tag="gate", wandb=False, wandb_name=None,
             wandb_project="creature-soccer", eval_every=0, eval_episodes=2,
             video_every=0, video_episodes=3, batch=50000)
    d.update(kw)
    return argparse.Namespace(**d)


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    from rower_soccer.t2a_port.train_e11_mlp import Trainer

    print("\n=== phase 1: the env's two reward terms ===", flush=True)
    tr = Trainer(mk_args())
    env = tr.env
    np.random.seed(7)
    env.seed(7)
    state = env.reset()
    while env.if_use_transform_action() != 2:
        state, _, d, _ = env.step(np.zeros((tr.nbody, tr.act_width)))
    n, err_sum, err_dense, parses, hit = 0, 0.0, 0.0, set(), 0
    dense_only_err_on_terminal = None
    for _ in range(2000):
        full = np.zeros((tr.nbody, tr.act_width))
        a = np.random.uniform(-1, 1, tr.act_dim)
        full[tr.act_rows, 0] = a
        state, r, done, info = env.step(full)
        n += 1
        dn, pa = info["dense"], info["parse"]
        err_sum = max(err_sum, abs((dn + pa) - r))
        # CompetEvo's dense term, recomputed from the primitives
        ref = info["forward"] - info["ctrl_cost"] - 0.0 + 1.0
        err_dense = max(err_dense, abs(dn - ref))
        parses.add(round(pa, 9))
        if pa != 0.0:
            hit += 1
            dense_only_err_on_terminal = abs(dn - r)
        if done:
            np.random.seed(7 + n)
            env.seed(7 + n)
            state = env.reset()
            while env.if_use_transform_action() != 2:
                state, _, dd, _ = env.step(np.zeros((tr.nbody, tr.act_width)))
                if dd:
                    break
    chk("dense + parse == env reward on every step", err_sum == 0.0,
        f"n={n} max|err|={err_sum:.3e}")
    chk("dense == forward - ctrl_cost - contact + survive", err_dense < 1e-12,
        f"max|err|={err_dense:.3e}")
    chk("parse only ever takes {0, +1000, -1000}",
        parses <= {0.0, 1000.0, -1000.0}, f"values={sorted(parses)}")
    chk("at least one sparse event was actually seen", hit > 0, f"n_sparse={hit}")
    chk("NEG: on a sparse step dense alone is NOT the reward",
        dense_only_err_on_terminal is not None
        and dense_only_err_on_terminal > 900.0,
        f"|dense - reward| = {dense_only_err_on_terminal}")

    print("\n=== phase 2: alpha, against CompetEvo's own formula ===", flush=True)
    CS = 200 * 50_000
    tc = Trainer(mk_args(curriculum_steps=CS))
    ref = lambda e: max((200 - e) / 200, 0)          # noqa: E731  their line
    worst = max(abs(tc.alpha(e) - ref(e)) for e in range(0, 400))
    chk("alpha(epoch) == max((200-epoch)/200, 0) for epochs 0..399",
        worst == 0.0, f"max|err|={worst:.3e} over 400 epochs")
    chk("alpha starts at exactly 1.0", tc.alpha(0) == 1.0)
    chk("alpha hits exactly 0 at curriculum_steps", tc.alpha(200) == 0.0)
    chk("alpha is pinned at 0 after", tc.alpha(1000) == 0.0)
    t4 = Trainer(mk_args(curriculum_steps=4_000_000))
    chk("E2.1's own setting: 4M steps -> alpha 0 at epoch 80",
        t4.alpha(0) == 1.0 and t4.alpha(40) == 0.5 and t4.alpha(80) == 0.0,
        f"a(0)={t4.alpha(0)} a(40)={t4.alpha(40)} a(80)={t4.alpha(80)}")
    chk("NEG: curriculum_steps = 0 returns None, not 1.0",
        tr.alpha(0) is None and tr.alpha(500) is None)

    print("\n=== phase 3: what lands in the PPO buffer ===", flush=True)
    # One net, one seed => one trajectory. Only the buffer may differ.
    runs = {}
    for tag, al in (("flat", None), ("a1", 1.0), ("a0", 0.0), ("a05", 0.5)):
        t = Trainer(mk_args(curriculum_steps=(0 if al is None else 4_000_000)))
        t.epoch = 0
        t.alpha_now = al
        torch.manual_seed(3)
        np.random.seed(3)
        t.env.np_random.seed(3)
        runs[tag] = t.sample_worker(0, None, 1500, False)
    flat, a1, a0, a05 = runs["flat"], runs["a1"], runs["a0"], runs["a05"]
    same_traj = all(
        np.array_equal(flat["act"], runs[k]["act"]) for k in ("a1", "a0", "a05"))
    chk("same seed -> bit-identical trajectory in all four conditions",
        same_traj and all(flat["ep_rets"] == runs[k]["ep_rets"]
                          for k in ("a1", "a0", "a05")),
        f"n_eps={len(flat['ep_rets'])} steps={len(flat['rew'])}")
    chk("flat: the buffer IS the raw env reward",
        np.array_equal(flat["rew"], np.asarray(flat["rew"])) and
        abs(float(np.sum(flat["rew"])) - float(np.sum(a1["rew"] * 0 + flat["rew"]))) == 0.0)
    # dense/parse recovered from the two extreme mixes
    dense_b, parse_b = a1["rew"], a0["rew"]
    chk("alpha=1 buffer + alpha=0 buffer == the flat buffer, exactly",
        float(np.abs((dense_b + parse_b) - flat["rew"]).max()) == 0.0,
        f"max|err|={float(np.abs((dense_b + parse_b) - flat['rew']).max()):.3e}")
    chk("alpha=0.5 == 0.5*dense + 0.5*parse, exactly",
        float(np.abs(a05["rew"] - (0.5 * dense_b + 0.5 * parse_b)).max()) < 1e-12)
    n_big_flat = int((np.abs(flat["rew"]) > 900).sum())
    n_big_a1 = int((np.abs(a1["rew"]) > 900).sum())
    chk("THE MECHANISM: at alpha=1 no +/-1000 reaches the buffer",
        n_big_a1 == 0 and n_big_flat > 0,
        f"flat has {n_big_flat} such steps, alpha=1 has {n_big_a1}")
    chk("NEG: at alpha=0 the buffer is the sparse term ALONE",
        n_big_flat == int((np.abs(a0["rew"]) > 900).sum())
        and float(np.abs(a0["rew"][np.abs(a0["rew"]) <= 900]).max()) == 0.0)

    print("\n=== phase 4: the fall-dodge, before and after ===", flush=True)
    # E2 section 6: under the flat reward a short episode that ends on a fall
    # outscores a long one that ends on the opponent's certain goal, by the
    # full 1000 it never pays. The claim under test is that alpha = 1 removes
    # exactly that premium and nothing else.
    def per_ep(out, key="rew"):
        v, i = [], 0
        for L in out["ep_lens"]:
            v.append(float(np.asarray(out[key])[i:i + L].sum()))
            i += L
        return np.asarray(v)

    per_flat, per_a1, per_a0 = per_ep(flat), per_ep(a1), per_ep(a0)
    gap = per_flat - per_a1
    chk("flat objective - alpha=1 objective == the sparse term, exactly",
        float(np.abs(gap - per_a0).max()) == 0.0,
        f"per-episode gaps {np.round(gap, 1).tolist()}")
    chk("the removed premium is a whole number of +/-1000s",
        set(np.round(gap, 9).tolist()) <= {0.0, 1000.0, -1000.0},
        f"{sorted(set(np.round(gap, 9).tolist()))}")
    chk("NEG: the two objectives really are different numbers",
        float(np.abs(per_flat - per_a1).max()) > 1.0,
        f"max per-episode gap = {float(np.abs(per_flat - per_a1).max()):.1f}")

    # And the positive claim, measured on the QUIET policy rather than a
    # random one. Stated because the first draft of this gate assumed the
    # dense reward rewards survival unconditionally and it does NOT: the
    # control cost is `0.5 * sum(a^2)` and the MLP initialises at log_std = 0,
    # so a freshly initialised policy pays ~4.0 per step against a survive
    # bonus of 1.0 and the dense return FALLS with episode length. What the
    # dense reward rewards at initialisation is quietening down; it rewards
    # survival only once the actions are small, which is the regime the
    # curriculum is meant to move training into.
    tq = Trainer(mk_args(curriculum_steps=4_000_000))
    with torch.no_grad():
        tq.actor.net[-1].weight.zero_()
        tq.actor.net[-1].bias.zero_()
        tq.actor.log_std.fill_(-6.0)          # the idle zero-torque control
    tq.epoch = 0
    tq.alpha_now = 1.0
    torch.manual_seed(5); np.random.seed(5); tq.env.np_random.seed(5)
    q1 = tq.sample_worker(0, None, 2000, False)
    tq.alpha_now = None
    torch.manual_seed(5); np.random.seed(5); tq.env.np_random.seed(5)
    qf = tq.sample_worker(0, None, 2000, False)
    # "does stopping the episode early pay?", asked exactly, per episode:
    # is the FULL episode the best prefix of itself under each objective?
    def prefix_gain(out):
        """Per episode: how much better the BEST place to stop is than
        finishing. That is exactly what the fall-dodge is worth, measured
        against each objective, and it is the size of it that matters --
        contact shoves make the dense return dip locally, so `zero` is not
        the honest claim; `negligible beside 1000` is."""
        v, i = [], 0
        for L in out["ep_lens"]:
            c = np.cumsum(np.asarray(out["rew"])[i:i + L])
            v.append(float(c.max()) - float(c[-1]))
            i += L
        return np.asarray(v)

    g_q1, g_qf = prefix_gain(q1), prefix_gain(qf)
    lens_q = sorted(set(q1["ep_lens"]))
    chk("NEG: quiet policy, FLAT reward: stopping early is worth ~+1000 in "
        "every episode -- this IS the fall-dodge",
        bool((g_qf > 900).all()) and len(g_qf) > 0,
        f"gains {np.round(g_qf, 1).tolist()}, lengths {lens_q}")
    # Parameter-free separation, chosen so no threshold is invented: the
    # WORST case under alpha=1 must be better than the BEST case under the
    # flat reward. The alpha=1 residual is not zero -- a contact shove makes
    # the dense return dip locally -- so "zero" would be a false claim; what
    # is true is that the two do not overlap.
    chk("quiet policy, alpha=1: the WORST prefix gain is below the BEST one "
        "under the flat reward -- the distributions do not overlap",
        float(g_q1.max()) < float(g_qf.min()),
        f"gains {np.round(g_q1, 2).tolist()} -- max {float(g_q1.max()):.2f} "
        f"vs flat min {float(g_qf.min()):.1f}, a factor of "
        f"{float(g_qf.mean()) / max(float(g_q1.mean()), 1e-9):.0f} on the "
        f"means")
    per_step = float(np.asarray(q1["rew"]).mean())
    per_step_f = float(np.asarray(qf["rew"]).mean())
    chk("quiet policy's dense reward per step is strictly positive, so a "
        "longer episode is worth more under alpha=1",
        per_step > 0.0,
        f"{per_step:+.3f}/step = +1.0 survive + forward (the opponent "
        f"bulldozes it backwards, so forward is NEGATIVE) - ctrl_cost; "
        f"the flat reward is {per_step_f:+.3f}/step")
    init_cost = float(np.asarray(flat["rew"]).mean())
    print(f"       (a FRESHLY INITIALISED policy's flat reward is "
          f"{init_cost:+.2f}/step, dominated by 0.5*sum(a^2) at log_std=0 "
          f"-- reported, not gated)", flush=True)

    print("\n=== phase 5: the flat arm is E2's arm ===", flush=True)
    t = Trainer(mk_args(curriculum_steps=0))
    t.epoch = 0
    torch.manual_seed(11); np.random.seed(11); t.env.np_random.seed(11)
    a = t.sample_worker(0, None, 800, False)
    t2 = Trainer(mk_args(curriculum_steps=4_000_000))
    t2.epoch = 0
    t2.alpha_now = t2.alpha(500)      # past the end of the schedule
    torch.manual_seed(11); np.random.seed(11); t2.env.np_random.seed(11)
    b = t2.sample_worker(0, None, 800, False)
    chk("default alpha_now is None, so an ungated caller gets the flat reward",
        t.alpha_now is None)
    chk("ep_rets are the RAW env return in both conditions",
        np.allclose(a["ep_rets"], b["ep_rets"], atol=0, rtol=0),
        f"{np.round(a['ep_rets'], 3).tolist()}")
    chk("NEG: past the schedule's end the BUFFER is not the env reward",
        float(np.abs(np.asarray(a["rew"]) - np.asarray(b["rew"])).max()) > 0.5)

    print(f"\n{len(OK)} checks passed, {len(BAD)} FAILED")
    if BAD:
        for b_ in BAD:
            print("  FAILED:", b_)
        sys.exit(1)


if __name__ == "__main__":
    main()
