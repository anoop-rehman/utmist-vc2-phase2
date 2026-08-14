# Persist Claude Code chat state to GCS — portable setup

Self-contained. Nothing here is specific to the project it came from: pick a
bucket, edit four variables, copy five files. ~15 minutes.

Everything below was verified against a working installation. Where something
could not be verified it says so.

---

## What this protects

Claude Code writes each session's full transcript to

```
$CLAUDE_CONFIG_DIR/projects/<cwd-slug>/<session-uuid>.jsonl     # default ~/.claude
```

and cross-session memory to `projects/<cwd-slug>/memory/`.

On an ephemeral box — a container, a cloud VM, a pod — that directory dies with
the machine. Losing it is not like losing a log: the transcripts are the working
memory of the project. Every decision, dead end and measured number that has not
yet been written into a doc lives only there.

This pushes that state to a private GCS bucket after every git commit, and pulls
it back on a new machine.

**Not a general backup.** It syncs Claude's config directory, nothing else. Your
code is in git; your large artifacts belong somewhere else.

---

## The kit

Five files. Copy them into `scripts/persistence/` in your repo.

| file | role |
|---|---|
| `claude-gcs.env` | the only thing you edit — four variables |
| `sync.sh` | push local → GCS |
| `pull.sh` | pull GCS → local |
| `post-commit.hook` | the trigger |
| `claude-snapshot.sh` | optional; rescues state when Claude writes to a container overlay |

---

## Quickstart

### 1. Bucket

```bash
gcloud auth login                     # you, interactively, once per machine
gcloud config set project YOUR_PROJECT

gcloud storage buckets create gs://YOUR-STATE-BUCKET \
  --location=US-CENTRAL1 \
  --uniform-bucket-level-access \
  --public-access-prevention

# Versioning is the safety net for the one failure this cannot otherwise
# survive: a truncated or corrupted transcript rsynced over a good one.
gcloud storage buckets update gs://YOUR-STATE-BUCKET --versioning
```

Transcripts are append-only JSONL and rsync has no delta encoding, so **every
sync stores a whole new version of every changed file**. Expect total versioned
bytes to run ~10× live bytes. Add a lifecycle rule up front:

```bash
cat > /tmp/lc.json <<'JSON'
{"rule":[{"action":{"type":"Delete"},
          "condition":{"daysSinceNoncurrentTime":60,"numNewerVersions":10}}]}
JSON
gcloud storage buckets update gs://YOUR-STATE-BUCKET --lifecycle-file=/tmp/lc.json
```

That deletes only *non-current versions* older than 60 days when at least 10
newer ones exist. It can never touch a live object.

### 2. Minimum IAM

`roles/storage.objectAdmin` on the bucket. Object create + delete + read;
`delete` is required because rsync replaces objects. No project-level role.

### 3. Config

`scripts/persistence/claude-gcs.env` — safe to commit, holds no secrets. A
bucket name is an identifier, not a credential; access is governed by IAM.

```bash
GCS_PROJECT="YOUR_PROJECT"
GCS_BUCKET="YOUR-STATE-BUCKET"     # name only: no gs://, no trailing slash
GCS_PREFIX="claude-code-state"     # folder inside the bucket
INCLUDE_CREDENTIALS=0              # leave at 0 — see below
```

Ship it with `GCS_BUCKET="REPLACE_ME"`; both scripts hard-fail on that value so
an unconfigured clone cannot silently push nowhere.

### 4. Install the hook

```bash
cp scripts/persistence/post-commit.hook .git/hooks/post-commit
chmod +x .git/hooks/post-commit
```

The hook backgrounds itself with `setsid`, takes a `flock` so concurrent commits
collapse into one sync, and appends to a log. A commit never waits on the
network and never fails because the bucket is unreachable.

**If you keep the scripts anywhere other than the repo, symlink — never copy.**
See gotcha 1.

### 5. Verify, in this order

```bash
# a. dry run — writes nothing
DRY_RUN=1 bash scripts/persistence/sync.sh

# b. real push
bash scripts/persistence/sync.sh
#    expect: "Pushed N session transcript(s). Remote size: ... KB"
#            "(credentials excluded - run 'claude' and log in on the new machine)"

# c. confirm the credential is NOT up there — this must print nothing
gcloud storage ls "gs://YOUR-STATE-BUCKET/claude-code-state/config/.credentials.json"

# d. confirm the hook fires
git commit --allow-empty -m "sync test" && sleep 20 && tail -5 <your sync log>

# e. rehearse the restore into a scratch dir, so nothing is overwritten
CLAUDE_PULL_DEST=/tmp/restore-test bash scripts/persistence/pull.sh
ls /tmp/restore-test/projects/
```

Step (c) is not optional. Do it before you trust the setup.

---

## Restoring on a new machine

```bash
gcloud auth login

gcloud storage cp gs://YOUR-STATE-BUCKET/claude-code-state/pull.sh .
gcloud storage cp gs://YOUR-STATE-BUCKET/claude-code-state/claude-gcs.env .
bash pull.sh
```

Then two things that are easy to get wrong:

**Recreate the world at the same absolute paths.** Transcripts reference
absolute paths. Clone your repos to exactly the paths the old machine used, or
the restored context refers to files that are not there.

**`cd` to the same directory the session ran from.** The project slug is derived
from the working directory — `/workspace` becomes `-workspace`. Resume from the
wrong cwd and Claude looks in a slug that does not exist and finds no sessions.

```bash
cd /the/same/cwd && claude --resume
```

**Where `pull.sh` actually writes** — it takes the first of these:

| condition | destination | then you must |
|---|---|---|
| `CLAUDE_PULL_DEST` set | that path | nothing (inspect it) |
| `/workspace/.claude-persistent` exists | that dir | `export CLAUDE_CONFIG_DIR=<that dir>` |
| otherwise | `${CLAUDE_CONFIG_DIR:-$HOME/.claude}` | nothing |

The middle row surprises people: Claude will not see the restored state until
`CLAUDE_CONFIG_DIR` is exported. `pull.sh` prints the export line; do what it
says.

**`pull.sh` overwrites.** No merge, no newer-file check. Restoring onto a
machine whose local state is newer destroys it. Use `CLAUDE_PULL_DEST` first
whenever you are not certain which side is fresher.

---

## What is excluded, and why

`INCLUDE_CREDENTIALS=0` excludes `.credentials.json`, which holds the OAuth
access and refresh tokens for your Claude account.

Leave it at 0. A bucket is a different trust boundary from a machine: anyone
with read access to the bucket, now or later, would get a working credential,
and it survives there until someone thinks to delete it. Logging in again on a
new machine takes ten seconds. The scripts print a reminder that they skipped
it, so the omission is never silent.

**`.claude.json` IS synced and that is fine** — verified. It holds per-project
`allowedTools`, `hasTrustDialogAccepted`, MCP server lists, and account identity
(email, org, tier). No access token and no refresh token; those are only in
`.credentials.json`. Restoring it is what saves you re-approving every trust
dialog and tool permission.

Two things to know about it anyway: it carries PII (email, org, machine ID) into
the bucket, and restoring it grants standing tool approvals for those paths —
which is what you want on a rebuilt machine and may not be on a different one.

---

## Gotchas, each learned the hard way

**1. Symlink the scripts; never keep a second copy.** If your hook runs a copy
outside the repo, edits to the repo copy do nothing — silently. We shipped a fix
that appeared to work (the manual test used the repo copy) while every automatic
sync kept running the broken one. Point the hook at the repo, or symlink:

```bash
ln -s "$REPO/scripts/persistence/sync.sh" /wherever/the/hook/points/sync.sh
```

**2. `.claude.json` is NOT inside the config directory.** Claude Code writes it
to `$HOME/.claude.json`, *beside* `$HOME/.claude/`. A sync loop that looks for
`$CONFIG_DIR/.claude.json` finds nothing and — if it skips missing files
quietly — says nothing. Ours did that for 75 consecutive syncs; the bucket copy
was three weeks stale and the restore would have overwritten a current file with
it. Resolve from an explicit candidate list and **warn loudly** when it finds
nothing.

**3. A silent skip is the enemy.** Every `[ -f "$f" ] || continue` is a place a
bug can hide for months. Make absence noisy.

**4. Verify what the hook runs, not what you edited.** After any change, run the
hook's own path and check the timestamp in the bucket actually moved.

---

## Failure modes

| symptom | cause | fix |
|---|---|---|
| `ERROR: not authenticated` | no active gcloud credential | `gcloud auth login` |
| `ERROR: GCS_BUCKET is REPLACE_ME` | config never edited | edit `claude-gcs.env` |
| `AccessDeniedException: 403` on push | missing `storage.objects.delete` | grant `roles/storage.objectAdmin`; `objectCreator` alone cannot replace objects |
| commit succeeds, nothing in the bucket | hook not executable, or points at a path that does not exist | `chmod +x .git/hooks/post-commit`; check the `SYNC=` line |
| restore leaves Claude with no sessions | wrong cwd, so wrong slug | `cd` to the original directory before `claude --resume` |
| restored state not picked up at all | `pull.sh` wrote to `/workspace/.claude-persistent` | `export CLAUDE_CONFIG_DIR=<that dir>` |
| bucket far larger than live data | versioning + append-only JSONL | expected ~10×; add the lifecycle rule above |

---

## Not verified here

Real end-to-end push wall-clock on a cold cache (the dry-run listing phase
measured 2.3 s against 38 transcripts / 119 MB, which is not the same thing);
the exact 403 text for a delete-permission failure, which is reproduced from
documentation rather than triggered; and current GCS list prices.
