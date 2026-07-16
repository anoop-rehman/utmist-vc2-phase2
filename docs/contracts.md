---
title: Interface contracts
---

# Interface contracts

The pipeline stages are built by independent workstreams (low-level control, data
collection + BC, RL fine-tune). They stay integrable because they agree on a set
of **frozen interface contracts** — command spec, controller protocol, obs
layouts, trajectory format, checkpoint format, env factory. These are the
integration boundaries; proxies (BoxHead walkers) implement the same interfaces,
which is what makes proxy-built infrastructure transfer to the real creatures
unchanged.

!!! warning "Frozen — changes require agreement"
    Don't change dims, orders, or dtypes without a group decision: BC data
    already collected would be invalidated. Evolution is additive (append fields,
    never reorder), every checkpoint/dataset carries a `git_sha` and refuses to
    load on mismatch, and proxy weights never ship.

The authoritative contract document is below, rendered verbatim from
`rower_soccer/docs/CONTRACTS.md`.

---

{% include-markdown "../rower_soccer/docs/CONTRACTS.md" %}
