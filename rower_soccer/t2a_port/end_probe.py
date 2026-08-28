"""Why does an episode end, and whose policy is it? A termination census.

    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
           CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
    cd /workspace/utmist-vc2-phase2

    # the port's own checkpoint
    PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python -m rower_soccer.t2a_port.end_probe \
        --ckpt runs/t2a_port/port_s1/models/epoch_0400.p --worlds 64 --seed 7

    # THEIR checkpoint, inside the port's pipeline (see --their-npz below)
    PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python -m rower_soccer.t2a_port.end_probe \
        --their-npz their_e1000.npz --worlds 32 --seed 23 --mean-action

`exec_R_eps` is a return, and a return is a rate times a length. When the two
move in opposite directions -- which is exactly what `port_s1` does against
their reference -- the aggregate hides the mechanism. This prints the split,
plus WHY each episode ended (`batched_exec_env.END_*`) and the height and root
angle at the moment it did, so "it learned a worse gait" and "it never learns
to stay up" stop looking alike.

`--their-npz` takes their `epoch_XXXX.p` re-exported as an npz, because their
checkpoints are pickles written by torch 1.8 and do not load in this venv:

    cd /workspace/Transform2Act && source env-gpu.sh
    .venv-gpu/bin/python -c "
    import pickle, numpy as np
    d = pickle.load(open('results/hopper_gpu_s2/models/epoch_1000.p','rb'))
    np.savez('their_e1000.npz', **{f'policy_dict|{k}': v.cpu().numpy()
                                   for k, v in d['policy_dict'].items()})"

Running THEIR policy through the port is the cheapest test there is of whether
the port's env, physics, design stage, reward and done condition are faithful:
their score is known from their own log, and it is a number the port cannot
fake. `--backend cpu` swaps mujoco_warp (float32-only) for CPU MuJoCo in
float64, which turns the same probe into a measurement of what fp32 physics
costs.
"""
import argparse, collections, types, numpy as np, torch

from rower_soccer.t2a_port.train_t2a import Trainer
from rower_soccer.t2a_port.two_stage_pipeline import iter_groups


TRACE = {"on": False}


def probe(tr, n_worlds, mean_action, max_steps):
    worlds = tr.design_phase(n_worlds, None, mean_action, world_offset=0)
    seed = int(torch.randint(0, 2 ** 30, (1,), generator=tr.gen,
                             device=tr.device).item())
    it = iter_groups(worlds, tr.spec, backend=tr.args.backend,
                     done_condition=tr.cfg.get("done_condition"),
                     reward_specs=tr.cfg.get("reward_specs", {}),
                     clip_qvel=tr.cfg["obs_specs"].get("clip_qvel", False),
                     seed=seed)
    codes, lens, rets, hgt, ang, dxs = [], [], [], [], [], []
    ngroups = 0
    while True:
        try:
            gi, g, idx = next(it)
        except StopIteration:
            break
        ngroups += 1
        K = g.n
        adj, ind = g.adj(), g.ind()
        alive = torch.ones(K, dtype=torch.bool, device=tr.device)
        code = torch.zeros(K, dtype=torch.long, device=tr.device)
        ret = torch.zeros(K, device=tr.device, dtype=tr.dtype)
        ln = torch.zeros(K, device=tr.device, dtype=tr.dtype)
        h_end = torch.zeros(K, device=tr.device, dtype=tr.dtype)
        a_end = torch.zeros(K, device=tr.device, dtype=tr.dtype)
        obs = g.env.reset()
        # Root x at the first EXECUTION step, so `dx` is the same quantity
        # `hopper.py:150`'s `posbefore` starts from and the same one
        # `their_sampled_probe.py` reports. Frozen per world at death.
        x0 = g.env.backend.qpos[:, 0].clone().to(tr.dtype)
        x_end = x0.clone()
        for t in range(max_steps):
            with torch.no_grad():
                act, lp = tr.policy.act("execution", obs, adj, ind,
                                        mean_action=mean_action,
                                        generator=tr.gen)
            nobs, r, done, info = g.env.step(act, auto_reset=False)
            live = alive.to(tr.dtype)
            ret += r * live
            ln += live
            newly = alive & done
            if bool(newly.any()):
                code = torch.where(newly, info["last_end"], code)
                h_end = torch.where(newly, g.env.backend.qpos[:, 1].to(tr.dtype), h_end)
                a_end = torch.where(newly, g.env.backend.qpos[:, 2].to(tr.dtype), a_end)
                x_end = torch.where(newly, g.env.backend.qpos[:, 0].to(tr.dtype), x_end)
            if TRACE["on"] and ngroups == 1 and (t % 10 == 0 or t < 5):
                q = g.env.backend.qpos
                used, cap = g.env.check_contact_capacity() if hasattr(g.env, "check_contact_capacity") else (None, None)
                print(f"    t={t:4d} h={q[0,1].item():7.4f} ang={np.degrees(q[0,2].item()):7.2f} "
                      f"x={q[0,0].item():8.3f} alive={int(alive.sum())}/{K}")
            alive = alive & (~done)
            obs = nobs
            if not bool(alive.any()):
                break
            if (t + 1) % 16 == 0:
                dead = (~alive).nonzero(as_tuple=True)[0]
                if int(dead.numel()):
                    g.env._write_initial(dead, add_noise=False)
                    g.env.backend.forward()
                    obs = g.env.obs()
        still = alive.nonzero(as_tuple=True)[0]
        if int(still.numel()):
            code[still] = -1  # never ended inside max_steps
        x_end = torch.where(alive, g.env.backend.qpos[:, 0].to(tr.dtype), x_end)
        codes += code.tolist(); lens += ln.tolist(); rets += ret.tolist()
        hgt += h_end.tolist(); ang += a_end.tolist()
        dxs += (x_end - x0).tolist()
        del g, adj, ind, obs
    return codes, lens, rets, hgt, ang, ngroups, worlds, dxs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="")
    p.add_argument("--their-npz", default="")
    p.add_argument("--cfg", default="hopper_gpu_s2")
    p.add_argument("--worlds", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--mean-action", action="store_true")
    p.add_argument("--backend", default="warp", choices=["warp", "cpu"])
    p.add_argument("--trace", action="store_true")
    a0 = p.parse_args()

    args = types.SimpleNamespace(
        cfg=a0.cfg, run="probe_tmp", outdir="/tmp",
        seed=a0.seed, batch_steps=57344, min_worlds=32, max_worlds=2048,
        eval_worlds=16, epochs=0, device="cuda" if a0.backend=="warp" else "cpu",
        backend=a0.backend, fp32=(a0.backend == "warp"),
        save_interval=1000, mempool_mb=256, stop_file="", batch_design=None)
    tr = Trainer(args)
    if a0.their_npz:
        z = np.load(a0.their_npz)
        sd = {k.split("|", 1)[1]: torch.as_tensor(z[k]).to(tr.device, tr.dtype)
              for k in z.files if k.startswith("policy_dict|")}
        missing, unexpected = tr.policy.load_their_state_dict(sd, strict=True)
        assert not missing and not unexpected, (missing, unexpected)
        ck = {"epoch": "THEIRS"}
        print(f"loaded THEIR policy: {len(sd)} tensors, 0 missing, 0 unexpected")
    else:
        ck = torch.load(a0.ckpt, map_location=tr.device, weights_only=False)
        tr.policy.load_state_dict(ck["policy"]); tr.value.load_state_dict(ck["value"])
    tr.policy.eval(); tr.value.eval()

    TRACE["on"] = a0.trace
    codes, lens, rets, hgt, ang, ng, worlds, dxs = probe(
        tr, a0.worlds, a0.mean_action, a0.max_steps)
    names = {-1: "still_alive", 0: "running", 1: "FELL", 2: "NONFINITE",
             3: "TIMEOUT"}
    c = collections.Counter(codes)
    print(f"ckpt {a0.ckpt}  epoch {ck['epoch']}  mean_action={a0.mean_action} "
          f"worlds={a0.worlds} groups={ng}")
    print(f"  mean len {np.mean(lens):.1f}  median {np.median(lens):.0f}  "
          f"max {np.max(lens):.0f}  mean ret {np.mean(rets):.1f}  "
          f"R/step {np.mean(rets)/max(np.mean(lens),1):.2f}")
    for k, v in sorted(c.items()):
        sel = [i for i, x in enumerate(codes) if x == k]
        print(f"  {names[k]:>12}: {v:4d} ({100*v/len(codes):5.1f}%)  "
              f"len {np.mean([lens[i] for i in sel]):7.1f}  "
              f"height@end {np.mean([hgt[i] for i in sel]):6.3f}  "
              f"ang@end(deg) {np.degrees(np.mean([ang[i] for i in sel])):7.2f}")
    # The three lines `their_sampled_probe.py` prints, so the two outputs can
    # be read against each other without arithmetic in between.
    q = np.percentile(lens, [10, 25, 50, 75, 90])
    dt = 0.008
    print(f"  exec-len pct 10/25/50/75/90  "
          f"{q[0]:.0f} {q[1]:.0f} {q[2]:.0f} {q[3]:.0f} {q[4]:.0f}")
    print(f"  mean dx per episode  {np.mean(dxs):+.4f} m   "
          f"(std {np.std(dxs):.4f}, n={len(dxs)})")
    print(f"  mean dx/dt per step  "
          f"{np.sum(dxs) / (max(np.sum(lens), 1) * dt):+.4f} m/s")
    print("  per-world (height, ang_deg, len) at termination:")
    for i in range(min(len(hgt), 24)):
        why = []
        if hgt[i] <= 0.7: why.append("low")
        if hgt[i] >= 2.0: why.append("high")
        if abs(np.degrees(ang[i])) >= 20: why.append("tilt")
        print(f"    w{i:3d} h={hgt[i]:8.3f} ang={np.degrees(ang[i]):8.2f} "
              f"len={lens[i]:6.0f} -> {','.join(why) or 'NONE?!'}")
    nb = [len(w.robot.bodies) for w in worlds]
    print(f"  bodies per design: mean {np.mean(nb):.2f} min {min(nb)} max {max(nb)}")


if __name__ == "__main__":
    main()
