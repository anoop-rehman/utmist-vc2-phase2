# Google Cloud Storage in this project — and how to replicate it

*Written 2026-08-14. Every number and every quoted error message in this
document was measured or reproduced on the live pod unless the text explicitly
says otherwise. Where a claim could not be verified without authenticating as
someone else or writing to a bucket, it is marked **[unverified]** rather than
guessed — see [Section 10](#10-what-i-could-not-verify).*

This project uses GCS for **two unrelated things**. They share a cloud project
and a `gcloud` install and nothing else: different buckets, different regions,
different triggers, different code, different failure behaviour, different
security posture. Read them as two systems.

| | **1. Chat-state persistence** | **2. Checkpoint sync** |
|---|---|---|
| Protects against | losing Claude Code transcripts + memory when the pod dies | losing trained weights when a run or a spot instance dies |
| Code | `scripts/persistence/{sync,pull}.sh` (bash) | `rower_soccer/warp_port/gcs.py` (python) + `scripts/gcs_pull_run.sh` |
| Trigger | git `post-commit` hook | inside the trainer loop, on a wall-clock timer |
| Bucket here | `gs://vc2-2026-claude-code-state` | `gs://vc2-2026-checkpoints` |
| Layout | `<prefix>/config/...` mirroring `~/.claude` | `<run_name>/<file>` |
| Blocks the caller? | no — detached background process | no — daemon threads, except the final flush |
| Live size here | 120 MiB (1.24 GiB with old versions) | 884 MiB across 119 runs |

Both are built on the `gcloud storage` CLI shelling out from bash or Python.
There is deliberately no `google-cloud-storage` Python dependency anywhere: the
only requirement is that `gcloud` is installed and a credential exists.

Throughout, replace these with your own:

| Placeholder | This project's value |
|---|---|
| `PROJECT_ID` | `vc2-2026` |
| `STATE_BUCKET` | `vc2-2026-claude-code-state` |
| `CKPT_BUCKET` | `vc2-2026-checkpoints` |
| `PREFIX` | `claude-code-state` |

---

## 1. Prerequisites

```bash
# gcloud. On this pod it is unpacked at /workspace/google-cloud-sdk and both
# sync.sh and gcs.py hardcode that path as a fallback, so a non-login shell
# and a subprocess-spawned trainer both find it.
gcloud --version        # measured here: Google Cloud SDK 579.0.0

# Auth. You do this yourself, interactively, once per machine.
gcloud auth login
```

Two `gcloud` facts that bite immediately, both reproduced here:

**A default project is not optional.** Every call in this repo passes
`--project=...` explicitly, and that is why. Without it:

```
ERROR: (gcloud.storage.ls) The required property [project] is not currently set.
It can be set on a per-command basis by re-running your command with the [--project] flag.
```

For GCS the project is the *quota and billing* project, not the bucket's owner —
so it must be set even when the bucket name alone is globally unique.

**Nothing in either mechanism uses a service-account key file.** Both run as
whatever principal `gcloud auth login` established. That is a real advantage for
a single-operator research box: there is no JSON key to leak, rotate, or
accidentally commit. If you replicate this into CI or onto an unattended fleet
you will need a service account, and at that point the credential-handling rules
in [Section 4](#4-what-is-deliberately-not-synced-and-why) apply to *that* key
with full force.

---

## 2. Mechanism 1 — Claude Code chat-state persistence

### 2.1 The problem it solves

Claude Code writes each session's full transcript to
`$CLAUDE_CONFIG_DIR/projects/<cwd-slug>/<session-uuid>.jsonl`, defaulting to
`~/.claude`. On this machine that is the container overlay: it is reset to the
base image when the pod stops. `/workspace` is local NVMe, not a network volume,
so it dies too. GCS is the only durable copy of the transcripts and of
`projects/<slug>/memory/`, which is where cross-session memory lives.

Losing that is not like losing a log file. The transcripts are the project's
working memory — every decision, dead end, and measured number that has not been
written into a doc yet.

### 2.2 The files

All of them live in `scripts/persistence/` and are tracked in git:

| File | Role |
|---|---|
| `claude-gcs.env` | the only thing you edit — four variables |
| `sync.sh` | push local → GCS |
| `pull.sh` | pull GCS → local, additive |
| `post-commit.hook` | the trigger; copy it to `.git/hooks/post-commit` |
| `CLAUDE-PERSISTENCE.md`, `REMOTE-WORK.md` | the original terse notes |

Two more scripts, `claude-bootstrap.sh` and `claude-snapshot.sh`, are referenced
by `sync.sh` and `pull.sh` but are **not in the repo** — they exist only in
`/workspace/claude-persistence/` on this pod. They are RunPod network-volume
helpers from an earlier era. Both call sites guard with `[ -x ... ]`, so their
absence is a no-op. You do not need them; see
[Section 9](#9-things-in-the-current-setup-that-look-wrong-or-fragile).

### 2.3 The config file

`scripts/persistence/claude-gcs.env` is tracked in git and **contains no
secrets** — I read the whole file to confirm before writing this. It is four
variables and two comments:

```bash
GCS_PROJECT="PROJECT_ID"
GCS_BUCKET="STATE_BUCKET"     # bucket name only, no gs:// and no trailing /
GCS_PREFIX="claude-code-state"  # folder inside the bucket
INCLUDE_CREDENTIALS=0
```

A bucket name is not a secret — it is an identifier, and access is governed by
IAM, not by obscurity. Keeping this file in git is what lets a fresh clone push
to the right place with zero setup. `INCLUDE_CREDENTIALS` is discussed at length
in [Section 4](#4-what-is-deliberately-not-synced-and-why); the short version is
that `0` is the correct value and you should not change it.

Both scripts hard-fail if `GCS_BUCKET` is still the literal string `REPLACE_ME`,
which is the intended checked-in default for a fresh copy.

### 2.4 What triggers a push

`.git/hooks/post-commit`, whose content is `post-commit.hook`. It does three
things, and the reasoning for each is worth copying:

```bash
SYNC=/workspace/claude-persistence/sync.sh
[ -x "$SYNC" ] || exit 0
setsid bash -c '
  exec 9>/tmp/claude-chat-sync.lock
  flock -n 9 || exit 0
  bash '"$SYNC"' >> /workspace/claude-persistence/sync.log 2>&1
' < /dev/null > /dev/null 2>&1 &
exit 0
```

1. **`setsid` + `&` + closed stdio.** The commit returns immediately; the sync
   runs detached in its own session, so it also survives the terminal that
   launched the commit going away.
2. **`flock -n`.** A burst of commits produces one sync, not a pile of
   concurrent `rsync`s racing to write the same objects. `-n` means a second
   commit during a sync *skips*, it does not queue — correct here, because the
   next commit will pick up everything anyway.
3. **`exit 0` unconditionally, output to a log.** A backup hook that can fail a
   commit is worse than no backup hook. Failures are silent by design; the log
   is the only place they surface.

Git hooks are not cloned, so this is a per-checkout install step. One useful
property, verified here: **a git worktree shares the main checkout's hooks
directory** (`git rev-parse --git-path hooks` inside
`.claude/worktrees/<name>/` returns `/workspace/utmist-vc2-phase2/.git/hooks`),
so commits made from agent worktrees fire the same hook. You install it once.

You can also run `bash sync.sh` by hand at any time. It is idempotent.

### 2.5 What gets pushed — an allowlist, on purpose

```bash
SYNC_DIRS=(projects)
SYNC_FILES=(history.jsonl settings.json .claude.json)
[ "${INCLUDE_CREDENTIALS:-0}" = "1" ] && SYNC_FILES+=(.credentials.json)
```

`projects/` carries all session transcripts *and* the `memory/` directory. That
is the payload; the rest is small.

The header of `sync.sh` explains why this is an allowlist rather than
`rsync --exclude`, and it is the single most important line in the script:

> `gcloud storage rsync --exclude` silently honors only part of a `"|"`-joined
> regex (verified — a lone pattern excluded 395 files, the same pattern OR'd
> with a second excluded 1), so a denylist here would quietly ship everything.

I did not re-verify that `--exclude` bug myself **[unverified]**, but the design
conclusion holds regardless of the bug's current status: with an allowlist, a
new directory appearing under `~/.claude` is excluded by default. With a
denylist, it is *included* by default and you find out later. For a directory
that accumulates caches, uploads, shell snapshots and credentials, fail-closed
is the only defensible default.

### 2.6 Remote layout

```
gs://STATE_BUCKET/
└── PREFIX/
    ├── CLAUDE-PERSISTENCE.md      ← the "kit": copies of the scripts themselves,
    ├── claude-bootstrap.sh           so a bare machine with only gcloud can
    ├── claude-gcs.env                bootstrap without cloning the repo first
    ├── claude-snapshot.sh
    ├── pull.sh
    ├── sync.sh
    └── config/
        ├── .claude.json
        ├── history.jsonl
        ├── settings.json
        └── projects/
            └── -workspace/                       ← cwd slug: "/" → "-"
                ├── <session-uuid>.jsonl
                ├── <session-uuid>/subagents/agent-<id>.jsonl
                ├── <session-uuid>/subagents/agent-<id>.meta.json
                ├── <session-uuid>/tool-results/<file>
                └── memory/MEMORY.md
```

The slug matters for restore: Claude derives it from the cwd the session ran
from, so `/workspace` becomes `-workspace`. `claude --resume` only lists
sessions whose slug matches your *current* cwd. Restore to the same absolute
paths or your history appears to be missing.

Shipping the scripts into the bucket alongside the data is a small idea that
pays off badly-needed dividends at 3am: recovery needs `gcloud` and nothing
else, not a working git remote and an SSH key.

### 2.7 Failure behaviour

`sync.sh` runs `set -euo pipefail` and preflights hard, with its own messages:

```
ERROR: set GCS_BUCKET in <dir>/claude-gcs.env
ERROR: gcloud not on PATH
ERROR: not authenticated. Run:  gcloud auth login
ERROR: no config dir at <path>
```

Past the preflight, both loops end in `|| true` and the `cp` calls redirect to
`/dev/null`, so a single failed object does not abort the run — and, because
the hook discards the exit status, nothing ever propagates to git. **A silent
sync failure looks exactly like a working sync.** The only detector is the log:

```bash
tail -40 /workspace/claude-persistence/sync.log
```

A healthy run ends with a line like `Pushed 36 session transcript(s). Remote
size: 122841 KB`. Build the habit of checking that number is growing.

`rsync` here never deletes: `--delete-unmatched-destination-objects` is not
passed. The remote is a superset of the local. That is why a stale
`projects/-workspace-utmist-vc2-phase2/` prefix from an older era still sits in
the bucket, harmlessly.

`rsync` decides what changed by modification time, falling back to hashes when
mtime is unavailable (`--checksums-only` forces hashes). It has no delta
encoding: an append-only JSONL that grew by one line is re-uploaded whole. That
is the root of the versioning cost in [Section 8](#8-cost-and-lifecycle).

### 2.8 Restoring onto a fresh machine

```bash
gcloud auth login

gcloud storage cp gs://STATE_BUCKET/PREFIX/pull.sh . --project=PROJECT_ID
gcloud storage cp gs://STATE_BUCKET/PREFIX/claude-gcs.env . --project=PROJECT_ID
bash pull.sh

# Recreate the world the transcripts talk about, at the SAME absolute paths.
git clone <your repo> /workspace/<same-dir-name>

cd /workspace && claude --resume
```

`pull.sh` is additive — it never deletes local sessions — so running it on a
machine that already has history is safe. Its destination is chosen in this
order:

1. `$CLAUDE_PULL_DEST` if set (use this to restore into a scratch dir and
   inspect before committing to it),
2. `/workspace/.claude-persistent` if `/workspace` exists and is writable,
3. `$CLAUDE_CONFIG_DIR`, else `~/.claude`.

Note rule 2: on any box with a writable `/workspace`, `pull.sh` restores to
`/workspace/.claude-persistent`, **not** `~/.claude`, and then tells you to
`export CLAUDE_CONFIG_DIR=/workspace/.claude-persistent`. If you skip that
export, Claude reads the empty default dir and your restore looks like it did
nothing. (`CLAUDE-PERSISTENCE.md` says pull restores "into `~/.claude`"; that
is wrong for this machine shape — see
[Section 9](#9-things-in-the-current-setup-that-look-wrong-or-fragile).)

`pull.sh` requests `.credentials.json` unconditionally and `chmod 600`s it —
both no-ops when, as here, no such object exists.

---

## 3. Mechanism 2 — training checkpoint sync

### 3.1 The problem it solves

A drill run is hours of GPU time producing a few megabytes of weights. Three
ways to lose them, all of which have happened to this project:

- the pod is stopped or the container is recycled;
- a SkyPilot managed-spot job is preempted and relaunched on a fresh instance
  with an empty `runs_v2/`;
- the run finishes and the process exits before the last upload lands.

`NPMP_SMP_POSTMORTEM.md` records the cost of getting this wrong: the
`npmp_rower_v2` weights (94.9M steps) were lost with the pod because that run
had `gcs_bucket: null`. Only the wandb metrics survived. The module's design is
a direct response.

### 3.2 The API

`rower_soccer/warp_port/gcs.py`, ~95 lines, no dependencies beyond the stdlib
and `gcloud` on `PATH`:

```python
sync_async(local_path, bucket, run_name)     # fire-and-forget, background thread
sync_blocking(local_path, bucket, run_name)  # upload on the calling thread
wait_all(timeout=600) -> bool                # join outstanding threads
```

Destination is computed by one function, and this is the whole naming scheme:

```python
def _dest(local_path, bucket, run_name):
    base = os.path.basename(local_path)
    return f"gs://{bucket.removeprefix('gs://')}/{run_name}/{base}"
```

So the remote layout is flat, one prefix per run, basename preserved:

```
gs://CKPT_BUCKET/
├── kick_ant_v12_v3_unfrozen/
│   ├── best.pt            1,423,580 B   best-fitness policy, export format
│   ├── checkpoint.pt      4,278,978 B   full resumable trainer state
│   ├── checkpoint_mid.pt  4,278,978 B   one-shot mid-run rollback copy
│   ├── config.json            1,642 B   the args the run actually used
│   └── latest.pt          1,423,646 B   most recent policy
└── ... 118 more run prefixes
```

`bucket.removeprefix('gs://')` means `--gcs-bucket vc2-2026-checkpoints` and
`--gcs-bucket gs://vc2-2026-checkpoints` both work. `run_name` is the only
namespacing, so **two concurrent runs sharing a `--run-name` will silently
overwrite each other's checkpoints.** There is no guard against this. Unique run
names are load-bearing.

### 3.3 What triggers an upload

Every trainer (`train_kick_warp.py`, `train_follow_warp.py`,
`train_shoot_warp.py`, `train_dribble_warp.py`, `train_track_warp.py`,
`train_fetch_warp.py`, `train_worm_fetch_warp.py`, `rower_soccer/fetch/train_fetch_cpu.py`)
takes `--gcs-bucket` and follows the same three-point shape. From
`train_kick_warp.py`:

**On a new best score** — pushes just `best.pt`:

```python
if sel is not None and sel > best_score:
    best_score = sel
    export_sb3_compatible(ac, best_path)
    if args.gcs_bucket:
        sync_async(best_path, args.gcs_bucket, args.run_name)
```

**On the checkpoint timer** — `--ckpt-secs`, default `1800.0`, i.e. every 30
minutes of wall clock, not every N steps:

```python
if now - last_ckpt >= args.ckpt_secs:
    save_checkpoint(trainer, ckpt_path)
    export_sb3_compatible(ac, latest_path)
    ...
    for path in (ckpt_path, cfg_path, latest_path):
        sync_async(path, args.gcs_bucket, args.run_name)
    if wrote_mid:
        sync_async(mid_path, args.gcs_bucket, args.run_name)
```

**At end of run** — and this block is the one to copy verbatim:

```python
if args.gcs_bucket:
    wait_all()
    for path in (cfg_path, ckpt_path, latest_path, final_path):
        sync_blocking(path, args.gcs_bucket, args.run_name)
```

Local files are overwritten in place, so the remote object count per run stays
constant at 5–6 no matter how long the run goes.

Two of the trainers (`train_fetch_warp.py`, `train_worm_fetch_warp.py`,
`train_fetch_cpu.py`) default `--gcs-bucket` to the bucket name rather than
`None`; the rest default to `None` and rely on the launch script passing it.
Defaulting to the bucket is the safer choice — see
[Section 9](#9-things-in-the-current-setup-that-look-wrong-or-fragile).

### 3.4 The two rules that make the remote copy trustworthy

Both are documented in the module docstring, and both encode a bug that was
actually hit.

**One in-flight upload per destination.** `sync_async` keeps an `_inflight` set
under a lock and *skips* rather than queues:

```python
with _lock:
    if dest in _inflight:
        print(f"[gcs] skip ...: previous upload still in flight", flush=True)
        return
    _inflight.add(dest)
```

Without this, a 30-minute checkpoint interval and a 35-minute upload on a slow
pod produce two concurrent writes to the same object — and GCS gives you no
ordering guarantee, so the *older* checkpoint can land last. Skipping is right:
the next interval carries newer bytes anyway. Seeing `[gcs] skip` in a training
log is a signal that your upload is slower than your checkpoint interval, not
an error.

**The final flush drains first, then blocks.** Upload threads are daemons, so
the interpreter kills them at exit. Without `wait_all()`, the end-of-run
checkpoint — the one most worth keeping — is silently dropped. `wait_all()`
joins with a 600 s deadline and warns rather than hangs:

```
[gcs] WARN N upload(s) unfinished after 600s; they will be killed at exit
```

### 3.5 Failure behaviour

Everything is wrapped:

```python
except Exception as e:  # noqa: BLE001 - never let sync crash training
    print(f"[gcs] WARN sync failed ({os.path.basename(local_path)}): {e}", flush=True)
```

`subprocess.run(..., check=True, capture_output=True, timeout=300)` means a
hung `gcloud` is killed at 5 minutes and a non-zero exit raises — both land in
the same warning. The rule is absolute: **a sync failure never kills training.**
Grep training logs for `[gcs] WARN` and `[gcs] skip`; a successful upload prints
`[gcs] synced <file> -> gs://...`.

Note the trade-off: `capture_output=True` means gcloud's own stderr is swallowed
into the exception's repr. You get the fact of the failure and usually the
return code, but a truncated view of gcloud's diagnosis.

### 3.6 The pull side: surviving preemption

`scripts/gcs_pull_run.sh <bucket> <run_name>` runs *before* the trainer and is
what makes spot instances safe:

```bash
for f in checkpoint.pt config.json best.pt checkpoint_mid.pt latest.pt; do
  if gcloud storage cp "${SRC}/${f}" "${DEST}/${f}" 2>/dev/null; then
    echo "[gcs_pull]   pulled ${f}"
  fi
done
```

`--resume` reads the *local* `runs_v2/<run>/checkpoint.pt`. On a relaunched spot
instance that directory is empty, so without this pull `--resume` silently
starts from scratch and you lose the run without any error. The `2>/dev/null`
plus per-file `if` makes a first launch (empty prefix) a clean no-op. It prints
which branch it took:

```
[gcs_pull] resume artifact present -> job will --resume
[gcs_pull] no prior checkpoint -> fresh start
```

Separately, `rower_soccer/skills/policy.py::resolve_checkpoint` accepts a
`gs://` URI anywhere a checkpoint path is expected, fetching to
`$VC2_CHECKPOINT_CACHE` (default `~/.cache/vc2-checkpoints`) and caching on
disk. So `--init-from gs://CKPT_BUCKET/follow_ant_v1/best.pt` works directly
from any machine with `gcloud`. The cache is keyed by URI with no
invalidation — if the remote object is overwritten, a machine that already
fetched it keeps the stale copy forever.

---

## 4. What is deliberately NOT synced, and why

### 4.1 `.credentials.json` — the security-critical exclusion

`INCLUDE_CREDENTIALS` defaults to `0`, and that default is a **security
property, not a preference or a convenience trade-off.**

`~/.claude/.credentials.json` holds a live OAuth token for the operator's Claude
account. Setting `INCLUDE_CREDENTIALS=1` copies a working bearer credential into
object storage, where it then sits at rest, is replicated by object versioning,
and is readable by every principal with `storage.objects.get` on the bucket —
a set that is easy to widen by accident and hard to audit after the fact.

The cost of *not* syncing it is that you type `claude` and log in on the new
machine, roughly twenty seconds, once per machine. The cost of syncing it is
that the blast radius of any read access to this bucket escalates from "can read
my chat transcripts" to "can act as me". Those are not comparable, and no
recovery-time argument bridges the gap.

Verified on the live bucket: `gcloud storage ls gs://vc2-2026-claude-code-state/claude-code-state/config/`
returns exactly `.claude.json`, `history.jsonl`, `settings.json`, and the
`projects/` prefix. There is no `.credentials.json` object. The default is
holding.

**Leave it at 0.** If you think you need `1`, what you actually need is a
service account for the machine, not a copy of a human's token.

### 4.2 Everything else under `~/.claude`

Measured on this pod: `~/.claude` is 150 MB total, of which `projects/` is
119 MB. The allowlist ships `projects/` and three small files; the remaining
~31 MB is excluded as regenerable or machine-local:

| Excluded | Size here | Why |
|---|---|---|
| `jobs/` | 14 MB | per-machine job state |
| `uploads/` | 6.9 MB | already referenced from transcripts |
| `plugins/` | 6.4 MB | reinstallable |
| `file-history/` | 3.9 MB | edit undo state, machine-local |
| `cache/`, `backups/`, `shell-snapshots/`, `sessions/`, `daemon*`, `telemetry/`, `downloads/`, `tasks/` | < 1 MB total | caches and runtime state |

`shell-snapshots/` deserves a specific mention: it captures shell environment,
which on a research box is exactly where an API key ends up as an exported
variable. Excluding it is not just a size decision.

### 4.3 Not in the state bucket at all

Checkpoints (bucket 2), code (git), and this repo's `.env` — which holds the
wandb key, is matched by `.gitignore:125`, lives in the repo directory and not
in `~/.claude`, and is therefore outside both mechanisms' reach. That is the
correct arrangement and worth an explicit check when you replicate: **confirm
your secret files are outside the synced tree, rather than relying on them being
excluded from it.**

### 4.4 The honest caveat

The transcripts *themselves* are the sensitive payload. Anything an agent read,
printed, or pasted during a session is in `projects/*.jsonl` — including any
secret a tool happened to echo. Two consequences:

1. **Public Access Prevention and Uniform Bucket-Level Access are load-bearing
   on the state bucket, not hygiene.** Verified enforced here.
2. `settings.json` and `.claude.json` are synced and *can*, in general, contain
   MCP server configurations with embedded credentials. I inspected both on this
   machine by key name and type without printing values: `settings.json` is four
   UI keys and no secret-shaped field; `.claude.json` is feature flags and
   project state. Clean here — but check yours before you trust the default, and
   re-check if you add MCP servers.

---

## 5. Quickstart: replicating this in a new project

Substitute your own values throughout. **Nothing below is run in this document —
these are the commands you run yourself, and every one of them needs an
authenticated `gcloud`.**

### 5.1 Create the buckets

```bash
PROJECT_ID=my-project
STATE_BUCKET=${PROJECT_ID}-claude-code-state
CKPT_BUCKET=${PROJECT_ID}-checkpoints
REGION=us-central1          # co-locate with your compute; egress is the cost

# Chat state: private, versioned. Versioning is the safety net for the one
# failure this mechanism cannot otherwise survive -- a corrupted or truncated
# transcript being rsynced over a good one.
gcloud storage buckets create gs://${STATE_BUCKET} \
    --project=${PROJECT_ID} \
    --location=${REGION} \
    --uniform-bucket-level-access \
    --public-access-prevention \
    --versioning

# Checkpoints: private, no versioning. Checkpoints are overwritten by design
# and the run itself keeps best/mid/latest copies, so versions add cost without
# adding recoverability.
gcloud storage buckets create gs://${CKPT_BUCKET} \
    --project=${PROJECT_ID} \
    --location=${REGION} \
    --uniform-bucket-level-access \
    --public-access-prevention
```

Both buckets in this project have UBLA on and a 7-day soft-delete window (the
current GCS default). Soft delete is not versioning — it protects against
deletion, not against overwrite.

### 5.2 Minimum IAM

Grant at the **bucket**, not the project.

| Principal | Role | Covers |
|---|---|---|
| machine that pushes | `roles/storage.objectUser` | `objects.{get,list,create,delete}` |
| machine that only restores | `roles/storage.objectViewer` | `objects.{get,list}` |
| the human who creates buckets | `roles/storage.admin`, once | bucket create + IAM |

```bash
gcloud storage buckets add-iam-policy-binding gs://${STATE_BUCKET} \
    --member="user:you@example.com" \
    --role="roles/storage.objectUser" \
    --project=${PROJECT_ID}
```

The gotcha: **`roles/storage.objectCreator` is not enough.** Overwriting an
existing object requires `storage.objects.delete`, which `objectCreator` lacks,
so both `rsync` and the checkpoint overwrite fail on the *second* run while the
first appears to work. `objectUser` (or the older `objectAdmin`) is the floor.

For reference on what *not* to copy: in this project both human collaborators
hold `roles/owner` at the project level. That is convenient for a two-person
research project and is not least privilege. If you are replicating, start with
the bucket-scoped roles above.

### 5.3 Install the chat-state kit

```bash
mkdir -p scripts/persistence
# copy sync.sh, pull.sh, claude-gcs.env, post-commit.hook from this repo
chmod +x scripts/persistence/{sync,pull}.sh
```

Edit `scripts/persistence/claude-gcs.env`:

```bash
GCS_PROJECT="my-project"
GCS_BUCKET="my-project-claude-code-state"
GCS_PREFIX="claude-code-state"
INCLUDE_CREDENTIALS=0
```

Then decide where the *live* copy of the scripts lives. This project runs them
from `/workspace/claude-persistence/`, outside the repo, so that a checkout of a
different branch cannot change the backup mechanism mid-flight. That is
defensible, but it creates two copies that can drift
([Section 9](#9-things-in-the-current-setup-that-look-wrong-or-fragile)). If you
prefer one copy, point the hook at the in-repo path instead:

```bash
# Option A: mirror this project — live copy outside the repo
mkdir -p /workspace/claude-persistence
cp scripts/persistence/{sync,pull}.sh scripts/persistence/claude-gcs.env \
   /workspace/claude-persistence/
cp scripts/persistence/post-commit.hook .git/hooks/post-commit

# Option B: single copy, hook points into the repo (edit SYNC= in the hook first)
sed 's#^SYNC=.*#SYNC='"$PWD"'/scripts/persistence/sync.sh#' \
    scripts/persistence/post-commit.hook > .git/hooks/post-commit
```

Either way:

```bash
chmod +x .git/hooks/post-commit
mkdir -p "$(dirname /workspace/claude-persistence/sync.log)"
```

If the hook's log directory does not exist, the redirect fails and the sync
never runs — silently, because the hook discards everything.

### 5.4 Wire up checkpoint sync

Copy `rower_soccer/warp_port/gcs.py` and `scripts/gcs_pull_run.sh`. In your
trainer:

```python
p.add_argument("--gcs-bucket", default="my-project-checkpoints")

# on a new best
if args.gcs_bucket:
    from your_pkg.gcs import sync_async
    sync_async(best_path, args.gcs_bucket, args.run_name)

# on the checkpoint timer
if args.gcs_bucket:
    from your_pkg.gcs import sync_async
    for path in (ckpt_path, cfg_path, latest_path):
        sync_async(path, args.gcs_bucket, args.run_name)

# at the very end -- drain, THEN block
if args.gcs_bucket:
    from your_pkg.gcs import sync_blocking, wait_all
    wait_all()
    for path in (cfg_path, ckpt_path, latest_path, final_path):
        sync_blocking(path, args.gcs_bucket, args.run_name)
```

And in the launch script, before the trainer:

```bash
bash scripts/gcs_pull_run.sh "${CKPT_BUCKET}" "${RUN_NAME}"
python -m your_pkg.train --run-name "${RUN_NAME}" \
    --gcs-bucket "${CKPT_BUCKET}" --ckpt-secs 1800 --resume
```

### 5.5 Verify it worked

```bash
# 1. Read-only dry run first. This writes nothing. On this pod the listing and
#    diff phase takes 2.3 s against 38 transcripts / 119 MB.
gcloud storage rsync ~/.claude/projects \
    gs://${STATE_BUCKET}/claude-code-state/config/projects \
    --recursive --dry-run --project=${PROJECT_ID}
# expect lines of the form:
#   Would copy file:///root/.claude/projects/... to gs://.../...

# 2. Real push.
bash scripts/persistence/sync.sh
# expect a final line like:
#   Pushed 36 session transcript(s). Remote size: 122841 KB
#   (credentials excluded - run 'claude' and log in on the new machine)

# 3. Confirm the credential is NOT there. This should list exactly three
#    objects and a projects/ prefix, and no .credentials.json.
gcloud storage ls gs://${STATE_BUCKET}/claude-code-state/config/ \
    --project=${PROJECT_ID}

# 4. Confirm the hook fires. Commit anything, then:
tail -20 /workspace/claude-persistence/sync.log

# 5. Restore rehearsal, into a scratch dir so nothing is overwritten.
CLAUDE_PULL_DEST=/tmp/restore-test bash scripts/persistence/pull.sh
find /tmp/restore-test/projects -name '*.jsonl' | wc -l   # matches step 2's count?

# 6. Checkpoint side, after a run has been going for one --ckpt-secs interval:
gcloud storage ls -l gs://${CKPT_BUCKET}/${RUN_NAME}/ --project=${PROJECT_ID}
```

Step 5 is the one people skip and the one that matters. A backup you have never
restored is a hypothesis.

---

## 6. Failure modes and how to recognise them

Error text below was reproduced with read-only commands on this pod unless
marked otherwise. Account identifiers are replaced with `you@example.com`.

**No project set** — the most common first-run failure, because the scripts pass
`--project` but ad-hoc commands you type do not:

```
ERROR: (gcloud.storage.ls) The required property [project] is not currently set.
```
Fix: `--project=PROJECT_ID`, or `gcloud config set project PROJECT_ID`.

**Not authenticated** — `sync.sh` and `pull.sh` catch this in preflight via
`gcloud auth print-access-token` and print their own message:

```
ERROR: not authenticated. Run:  gcloud auth login
```

**Bucket does not exist** (typo in `GCS_BUCKET`):

```
ERROR: (gcloud.storage.ls) gs://my-typo-bucket not found: 404.
```

**Bucket exists, you lack access** — note GCS deliberately conflates 403 and
404 here, so "or it may not exist" does *not* mean your name is wrong:

```
ERROR: (gcloud.storage.ls) [you@example.com] does not have permission to access
b instance [test] (or it may not exist): you@example.com does not have
storage.objects.list access to the Google Cloud Storage bucket.
Permission 'storage.objects.list' denied on resource
'//storage.googleapis.com/projects/_/buckets/test' (or it may not exist).
```

The write-side analogue — having `list`/`get` but not `delete`, so overwrites
fail while the first upload succeeded — is the `objectCreator` trap in
[Section 5.2](#52-minimum-iam). I could not capture its exact text without
writing to a bucket **[unverified]**; expect the same shape with
`storage.objects.delete` named.

**Missing object on pull** — benign in `gcs_pull_run.sh`, which suppresses it:

```
ERROR: (gcloud.storage.cp) The following URLs matched no objects or files:
gs://vc2-2026-checkpoints/nope_run/checkpoint.pt
```

**Uploads slower than the checkpoint interval** — in the training log:

```
[gcs] skip checkpoint.pt: previous upload still in flight
```
Not an error. If it is *every* interval, raise `--ckpt-secs`.

**Final flush timed out** — the one that actually loses weights:

```
[gcs] WARN 2 upload(s) unfinished after 600s; they will be killed at exit
```

**Any other upload failure** — gcloud's own stderr is captured into the
exception, so this line is the whole diagnosis you get:

```
[gcs] WARN sync failed (checkpoint.pt): <exception repr>
```

**The silent ones.** Worth listing separately because there is no error text at
all:

- Hook never installed, or its log directory missing → no sync, no message. Detect by the log's mtime.
- Two runs sharing a `--run-name` → last writer wins, no warning.
- Restore to the wrong cwd → `claude --resume` shows an empty list, as if nothing was restored.
- `pull.sh` restored to `/workspace/.claude-persistent` and you did not export `CLAUDE_CONFIG_DIR` → same symptom.
- `--resume` without `gcs_pull_run.sh` on a fresh spot instance → run restarts from zero, and only the step counter in wandb betrays it.

---

## 7. Object counts and sizes this project actually produces

All measured 2026-08-14 with `gcloud storage du` / `ls -l`.

**`gs://vc2-2026-checkpoints`** — `us-central1`, STANDARD, UBLA on, versioning
off, no lifecycle rule, `public_access_prevention: inherited`.

| | |
|---|---|
| Live objects | 500 |
| Top-level run prefixes | 119 |
| Live size | 927,344,579 B (884 MiB) |
| Per run | 5–6 objects; 10.9–15.1 MiB for a fully-populated recent run, 7.8 MiB averaged bucket-wide (older prefixes hold fewer files) |

Per-file, measured across all 37 local run dirs that have a checkpoint, and
confirmed against the bucket:

| File | Size |
|---|---|
| `checkpoint.pt` (full trainer state) | 2,997,415 – 4,343,618 B (3.00–4.34 MB) |
| `checkpoint_mid.pt` | byte-identical to `checkpoint.pt` at the moment it was copied |
| `best.pt` / `latest.pt` / `final.pt` (policy export) | 1,411,228 – 1,445,278 B (1.41–1.45 MB) |
| `config.json` | 1.0–1.7 KB |

A handful of run prefixes also carry hand-uploaded `videos/*.mp4` (e.g.
`follow_ant_v1/videos/`, 79 KB and 3.0 MB). No script uploads these — they were
pushed with ad-hoc `gcloud storage cp`, which is why the count is inconsistent
across runs.

**Locally**, `runs_v2/` is 6.7 GB across 153 entries — roughly 7× the bucket,
because wandb dirs, per-run videos, and logs stay on disk. Only the five
checkpoint files per run go up.

**Different order of magnitude: Transform2Act.** `results/*/models/*.p` are
**157,116,291 B (157 MB) each**, one per saved epoch plus `best.p`, and
`hopper_gpu/` alone holds `epoch_0100` through `epoch_0400` plus `best`. A
`grep` for `gs://` and `gcs` across `/workspace/Transform2Act` returns nothing:
**these are not synced anywhere.** If you replicate this pattern with
T2A-sized checkpoints, syncing every epoch is ~157 MB per upload and the
30-minute interval assumption stops being free.

**`gs://vc2-2026-claude-code-state`** — `northamerica-northeast1`, STANDARD,
UBLA on, PAP **enforced**, versioning **on**, one lifecycle rule.

| | |
|---|---|
| Live objects | 114 |
| Live size | 125,802,202 B (120 MiB) |
| **All versions** | **1,335,790,673 B (1.24 GiB)** |
| Amplification | **10.6×** |
| Syncs run to date | 75 (`grep -c '^source :' sync.log`) |
| Local source | `~/.claude` 150 MB, `projects/` 119 MB, 38 transcripts |

That 10.6× is the headline number and it is a direct, predictable consequence of
the design: transcripts are append-only JSONL, `rsync` has no delta encoding, so
every commit re-uploads each touched transcript in full, and versioning keeps
every one of those copies.

---

## 8. Cost and lifecycle

### 8.1 The cost

At roughly $0.020–0.023 per GB-month for regional Standard storage, and Class A
(write/list) operations around $0.005 per 1,000 **[unverified — list prices from
memory; check the current GCS pricing page]**:

| | Billable | ≈ /month |
|---|---|---|
| Checkpoints | 0.93 GB | ~$0.02 |
| Chat state (all versions) | 1.24 GiB | ~$0.03 |
| Operations: 75 syncs, each listing ~200 objects and writing only what changed | order 10k Class A | cents, cumulative — not per month |

**Total is a few cents a month.** State this plainly so nobody optimises the
wrong thing: cost is not a reason to touch this setup. The reason to add a
lifecycle rule is *bounded growth* — both buckets currently accumulate
monotonically, and "a few cents" is a statement about August 2026, not about a
year of unattended running.

### 8.2 Lifecycle rules

**State bucket — a rule exists and has not fired yet.** Verified config:

```yaml
lifecycle_config:
  rule:
  - action: {type: Delete}
    condition:
      daysSinceNoncurrentTime: 60
      numNewerVersions: 10
```

Lifecycle conditions are ANDed: a version is deleted only once it has been
noncurrent for 60 days *and* 10 newer versions exist. The bucket's soft-delete
policy took effect 2026-07-26, so on 2026-08-14 nothing satisfies the 60-day
clause — which is exactly why all 1.24 GiB is still there. The rule is correct
in shape and simply has not had time to act. If the 10.6× ratio bothers you,
`numNewerVersions: 3` with `daysSinceNoncurrentTime: 14` reclaims most of it
while still keeping a fortnight of undo.

**Checkpoint bucket — no rule, and one is advisable.** 119 run prefixes, most
of them abandoned experiments, growing by ~11 MB per run forever. Two options:

```bash
# Age out cold checkpoints to Nearline (cheaper at rest, retrieval fee).
# 119 prefixes at ~11 MB is not urgent; this is about the next 500.
cat > /tmp/ckpt-lifecycle.json <<'EOF'
{"lifecycle": {"rule": [
  {"action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
   "condition": {"age": 90}}
]}}
EOF
gcloud storage buckets update gs://${CKPT_BUCKET} \
    --lifecycle-file=/tmp/ckpt-lifecycle.json --project=${PROJECT_ID}
```

Do **not** put a `Delete` rule on the checkpoint bucket without thinking hard.
These are the only surviving copies of trained policies — `runs_v2/` is
gitignored and local disk is ephemeral — and the postmortem in this repo exists
precisely because weights were lost once already. An age-based transition is
safe; an age-based delete is a second version of the same mistake.

### 8.3 Region

The two buckets are in different regions (`us-central1` and
`northamerica-northeast1`), which is almost certainly historical rather than
intended. Cross-region reads cost egress and add latency. When you replicate,
put both in one region, co-located with your compute.

---

## 9. Things in the current setup that look wrong or fragile

Found while writing this. None is on fire; all would bite a replicator who
copied the setup verbatim.

1. **`.claude.json` is never synced, and the copy in the bucket is stale.**
   `sync.sh` looks for `$SRC/.claude.json` = `/root/.claude/.claude.json`, which
   does not exist. The real file is `/root/.claude.json` — *beside* the config
   dir, not inside it, which is where Claude Code puts it when
   `CLAUDE_CONFIG_DIR` is unset. `claude-snapshot.sh` gets this right and has a
   comment saying so; `sync.sh` does not. Consequence: the local file was
   modified today (43,310 B) while the bucket's copy is dated **2026-07-26** and
   has not moved in 75 syncs. Low impact — it is project state, not transcripts
   — but it is a silent no-op, and `pull.sh` will happily restore the July copy
   over a newer one. Fix: have `sync.sh` fall back to `$HOME/.claude.json`.

2. **Two copies of the scripts, no mechanism keeping them in sync.** The hook
   runs `/workspace/claude-persistence/sync.sh`; git tracks
   `scripts/persistence/sync.sh`. I diffed all three shared files today and they
   are identical — but that is luck, not enforcement. Editing the tracked copy
   changes nothing about what actually runs, and there is no test or check that
   would notice.

3. **`CLAUDE-PERSISTENCE.md` documents the wrong restore path.** It says
   `bash pull.sh # restores into ~/.claude`. On any box with a writable
   `/workspace` — including this one — `pull.sh` restores into
   `/workspace/.claude-persistent` and requires a `CLAUDE_CONFIG_DIR` export.
   Following the doc literally during a real recovery produces a restore that
   looks like it did nothing.

4. **The repo ships an incomplete kit.** `sync.sh` and `pull.sh` reference
   `claude-bootstrap.sh` and `claude-snapshot.sh`, which are not in
   `scripts/persistence/`. Guarded, so harmless — but a fresh clone gives you a
   kit whose own source references files it does not contain, and the missing
   ones happen to be the two that handle the `.claude.json` location correctly.

5. **The checkpoint bucket has `public_access_prevention: inherited`, not
   `enforced`.** With UBLA on and no `allUsers` binding it is private today.
   "Inherited" only means nothing stops someone from making it public later.
   The state bucket is `enforced`; the checkpoint bucket should match.

6. **Inconsistent `--gcs-bucket` defaults.** `train_fetch_warp.py`,
   `train_worm_fetch_warp.py` and `train_fetch_cpu.py` default to
   `"vc2-2026-checkpoints"`; `train_kick_warp.py`, `train_follow_warp.py`,
   `train_shoot_warp.py`, `train_track_warp.py` default to `None`. A run
   launched by hand without the flag on one of the latter group silently gets no
   backup — which is precisely how `npmp_rower_v2` was lost
   (`gcs_bucket: null`). Defaulting to the bucket everywhere makes the safe
   behaviour the default and the unsafe one explicit.

7. **`run_name` collisions overwrite silently.** No timestamp, no generation
   check, no guard. Two runs, one name, last writer wins.

8. **Transform2Act's 157 MB checkpoints are not backed up at all.** Only local
   NVMe. Whether that is intentional (they are large and the run is
   reproducible) or an oversight, it is not written down anywhere, and a live
   run is producing them right now.

9. **`_fetch_gcs` caches by URI with no invalidation.** Since
   `gs://.../best.pt` is overwritten in place by training, a machine that
   fetched it once will keep serving the old weights from
   `~/.cache/vc2-checkpoints` indefinitely.

10. **The state bucket contains full chat transcripts and both project members
    hold `roles/owner`.** Appropriate to note rather than to fix here, but a
    replicator should not copy the IAM shape.

---

## 10. What I could not verify

- **Real push wall-clock time.** `post-commit.hook`'s comment says "the ~50 s
  rsync". I could not time an actual push without writing to the bucket, which
  was out of scope for this document. What I did measure: the read-only
  `--dry-run` listing and diff phase completes in **2.3 s** against 38
  transcripts / 119 MB. The remainder is upload time and scales with how much
  changed.
- **The `gcloud storage rsync --exclude` bug** described in `sync.sh`'s header
  (395 files vs 1). Not re-tested. The allowlist design is right on
  fail-closed grounds independently of whether the bug still reproduces on
  gcloud 579.0.0.
- **The exact error text for an overwrite denied by missing
  `storage.objects.delete`.** Reproducing it requires a write attempt.
- **Current GCS list prices.** The per-GB and per-operation figures in
  [Section 8](#8-cost-and-lifecycle) are from memory and should be checked
  against the pricing page. The measured byte counts and object counts they are
  multiplied by are exact.
- **Whether Transform2Act's checkpoints are deliberately excluded from GCS or
  simply never wired up.** No code, comment, or doc states either.

---

## Appendix: file map

| Path | What it is |
|---|---|
| `scripts/persistence/claude-gcs.env` | chat-sync config — the only file you edit |
| `scripts/persistence/sync.sh` | push `~/.claude` → GCS |
| `scripts/persistence/pull.sh` | pull GCS → `~/.claude`, additive |
| `scripts/persistence/post-commit.hook` | trigger; copy to `.git/hooks/post-commit` |
| `scripts/persistence/CLAUDE-PERSISTENCE.md` | original notes (see fragility 3) |
| `scripts/persistence/REMOTE-WORK.md` | what survives an SSH drop vs a pod stop |
| `rower_soccer/warp_port/gcs.py` | checkpoint upload: async / blocking / drain |
| `scripts/gcs_pull_run.sh` | pre-launch restore, makes `--resume` spot-safe |
| `rower_soccer/skills/policy.py` | `resolve_checkpoint()` — accepts `gs://` URIs |
| `rower_soccer/warp_port/train_*_warp.py` | the `--gcs-bucket` call sites |
| `POD_SETUP.md` | pod-side launch conventions |
