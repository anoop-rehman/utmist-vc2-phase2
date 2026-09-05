"""D3 M3 E4: the snapshot exchange between two co-evolving lineages.

Each arm trains one lineage and, every `--opp-refresh` epochs:

  1. PUBLISHES its own current state -- policy weights and the XML of its
     current mean-action body -- into `<root>/<cfg>/`;
  2. LOADS whatever its partner last published and installs it as the opponent.

Publish-then-load in that order is what makes the pair self-synchronising: at
epoch 0 neither has published yet, so each publishes, then finds the other's
files on the same boundary (or, if it got there first, one refresh later). No
barrier, no lockstep, and either arm can be restarted independently.

Every write is to a temporary file followed by `os.replace`, which is atomic
within a filesystem. Without it a partner can read a half-written checkpoint
at exactly the moment it is being rewritten -- a rare, unreproducible crash
mid-run, which is the worst kind.

The staleness this introduces is deliberate and is stated as a limit on E4's
resolution in `D3_E4_PREREQ.md`: each lineage always faces an opponent up to
`--opp-refresh` epochs old, so the design bounds the RATE of divergence that
can be resolved, not whether divergence happens.
"""
import json, os, pickle, shutil, tempfile
import numpy as np
import torch


def pub_dir(root, cfg_id):
    d = os.path.join(root, cfg_id)
    os.makedirs(d, exist_ok=True)
    return d


def _atomic(path, write):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def publish(root, cfg_id, epoch, policy_net, body_xml_str, meta=None):
    """Write this lineage's current policy and body where the partner can see
    them. `body_xml_str` is a MERGED scene (our body plus whatever opponent it
    happened to be facing); the partner takes only its first <body>."""
    d = pub_dir(root, cfg_id)
    sd = {k: v.detach().cpu() for k, v in policy_net.state_dict().items()}
    _atomic(os.path.join(d, "latest_policy.p"),
            lambda p: pickle.dump({"policy_dict": sd, "epoch": epoch},
                                  open(p, "wb")))
    _atomic(os.path.join(d, "latest_body.xml"),
            lambda p: open(p, "w").write(body_xml_str))
    _atomic(os.path.join(d, "latest_meta.json"),
            lambda p: json.dump(dict(epoch=epoch, cfg=cfg_id, **(meta or {})),
                                open(p, "w")))
    return d


def read_partner(root, partner_cfg):
    """(policy_dict, body_xml_path, meta) or None if the partner has not
    published yet."""
    d = os.path.join(root, partner_cfg)
    pol = os.path.join(d, "latest_policy.p")
    body = os.path.join(d, "latest_body.xml")
    meta = os.path.join(d, "latest_meta.json")
    if not (os.path.exists(pol) and os.path.exists(body)):
        return None
    try:
        blob = pickle.load(open(pol, "rb"))
        m = json.load(open(meta)) if os.path.exists(meta) else {}
    except Exception:
        return None            # mid-write or truncated: try again next refresh
    return blob["policy_dict"], body, m


def dump_mean_action_body(env, policy, e3_morph):
    """Run the design stages at the mean action and return the resulting
    merged scene as an XML string -- this lineage's CURRENT body.

    Wrapped in `rng_guard` so publishing cannot perturb training. That guard is
    the same one the per-epoch morphology census uses, and it exists because a
    probe that advances the RNG changes the run it is measuring.
    """
    with e3_morph.rng_guard(env):
        ok = e3_morph.run_design_stages(env, policy, True)
        xml = env.cur_xml_str if ok else None
    return xml


def install_opponent(env, agent_cfg, policy_dict, body_xml_path,
                     merged_out, build_scene, base_src, make_policy):
    """Compile a scene whose opp_* sibling is the partner's body, swap it into
    the running env, and install the partner's controller."""
    from lxml import etree
    tree = build_scene(base_src, body_xml_path)
    _atomic(merged_out,
            lambda p: open(p, "wb").write(etree.tostring(tree,
                                                         pretty_print=True)))
    env.swap_opponent(merged_out, body_xml_path)
    pol = make_policy()
    pol.load_state_dict(policy_dict)
    pol.eval()
    for p in pol.parameters():
        p.requires_grad_(False)
    env.set_opponent_policy(pol)
    return pol


def race_stats(eps, goal_x):
    """Draw rate and race margin, the two pre-registered degeneracy guards.

    A DRAW is `n_reached == 2` in one physics step, where `run_to_goal` scores
    `parse = 0`: the coupled channel is switched off for that episode. The
    scene is mirror-symmetric and both agents run the same 5 m, so equal-speed
    lineages arrive together -- a high draw rate means E4's divergence number
    is UNTESTABLE rather than null, and it is pre-registered that way.

    `margin_m` is how much further the loser still had to run when the episode
    ended, positive when we are ahead. Zero is a dead heat.
    """
    if not eps:
        return {}
    def g(k, d=0.0):
        return np.array([e.get(k, d) for e in eps], dtype=float)
    reached, opp = g("reached"), g("opp_reached")
    draws = np.logical_and(reached > 0, opp > 0)
    out = {"draw_rate": float(draws.mean()),
           "decisive_rate": float(np.logical_xor(reached > 0, opp > 0).mean())}
    if any("com_x" in e for e in eps):
        ours_left = goal_x - g("com_x")
        theirs_left = goal_x + g("opp_com_x")
        out["margin_m"] = float(np.mean(theirs_left - ours_left))
        out["abs_margin_m"] = float(np.mean(np.abs(theirs_left - ours_left)))
    return out
