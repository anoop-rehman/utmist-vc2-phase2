# D3 M3 E3 renders — read this before watching any of them

**The two families differ by ONE LETTER in the cfg name and are completely
different experiments.** Sorting this directory by date shows nothing but
unchanged ants, because the design-ON arms were stopped at epoch ~19 and the
frozen controls have been running ever since.

| prefix | what it is | body | clips |
|---|---|---|---|
| `rtg_e3_s{1,2,3}` → **`DESIGN-ON_*`** | **the E3 experiment.** Skeleton + attribute stages LIVE | **evolves** — ends as a 5-body, 0-motor stump | `e0001`, `e0006`, `e0012`, `e0018` (12 clips, stopped at epoch ~19) |
| `rtg_e3c_s{1,2}` → **`FROZEN-CONTROL_*`** | the frozen-body GNN control | **cannot change** — 13-body / 8-motor ant, always | `e0001` … `e0060`+ (still running) |

> **The single most important visual in this experiment is
> `DESIGN-ON_evolved-body_seed*_epoch0006/0012/0018`.** Those show the actuators
> disappearing. Every clip newer than them is a control whose body is frozen by
> construction and *cannot* change — if you watch those and conclude "the
> morphology isn't changing", that is correct and expected, and it is not E3.

`DESIGN-ON_*` and `FROZEN-CONTROL_*` are symlinks to the cfg-named files; both
names point at the same mp4.

**What to look for in the design-ON clips**: at `e0001` the mean-action design is
already a ~6-body stump; by `e0018` it is 5 bodies with **zero motors**, and
each panel is labelled `nb=` with its own body count. The creature does not
walk because it has no actuators to walk with — see `D3_E3_ADVERSARIAL.md` §3e.
