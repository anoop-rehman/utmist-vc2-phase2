# Claude Code chat persistence (updated 2026-08-10)

## The problem

Claude Code writes chats to `/root/.claude/projects/<slug>/*.jsonl`. On this
machine that is the **container overlay — it dies with the pod**, and unlike the
old RunPod setup, `/workspace` here is local NVMe, not a network volume. So GCS
is the ONLY durable copy. Bucket:

    gs://vc2-2026-claude-code-state     (project vc2-2026, private, versioned)

## Day-to-day: you rarely need to do anything

A git `post-commit` hook in `/workspace/utmist-vc2-phase2` runs `sync.sh` in the
background after every commit (measured: delta push takes seconds, costs
nothing noticeable). Manual push any time:

    bash /workspace/claude-persistence/sync.sh

What gets pushed (allowlist — see sync.sh header for why not `--exclude`):
`projects/` (ALL session transcripts + the memory/ dir), `history.jsonl`,
`settings.json`, `.claude.json`. Credentials are excluded by default
(INCLUDE_CREDENTIALS=0 in claude-gcs.env): on a new machine you just log in.

## Resuming the chat on a brand-new machine

```bash
# 1. auth
gcloud auth login          # (and gh auth login, wandb login for the project)

# 2. fetch the kit from the bucket, restore the chat state
gcloud storage cp gs://vc2-2026-claude-code-state/claude-code-state/pull.sh .
gcloud storage cp gs://vc2-2026-claude-code-state/claude-code-state/claude-gcs.env .
bash pull.sh               # destination depends -- see below

# 3. recreate the WORLD the transcripts talk about, at the SAME paths
#    (transcripts reference absolute paths; matching them keeps context valid)
git clone git@github.com:anoop-rehman/utmist-vc2-phase2 /workspace/utmist-vc2-phase2
git clone https://github.com/KJaebye/competevo /workspace/competevo
git clone https://github.com/Khrylx/Transform2Act /workspace/Transform2Act

# 4. resume — MUST cd to the same cwd the session ran from, the project slug
#    is derived from it (current sessions: /workspace -> slug "-workspace")
cd /workspace && claude --resume
```

### Where pull.sh actually restores to

It picks the FIRST of these, which is not always `~/.claude`:

| condition | destination | you must then |
|---|---|---|
| `CLAUDE_PULL_DEST` is set | that path | nothing (inspect it) |
| `/workspace/.claude-persistent` exists | that dir | `export CLAUDE_CONFIG_DIR=/workspace/.claude-persistent` |
| otherwise | `${CLAUDE_CONFIG_DIR:-$HOME/.claude}` | nothing |

The middle row is the one that surprises people: on a pod that already has the
persistent dir, `pull.sh` does NOT write to `~/.claude`, and Claude will not see
the restored state until `CLAUDE_CONFIG_DIR` is exported. `pull.sh` prints the
export line it wants; do what it says.

`CLAUDE_PULL_DEST=/some/dir bash pull.sh` restores to a scratch dir if you want
to inspect before overwriting.

**pull.sh overwrites.** It has no merge and no newer-file check, so restoring
onto a machine whose local state is newer loses that state. Use
`CLAUDE_PULL_DEST` first if you are unsure which side is fresher.

### The scripts live in the repo

`/workspace/claude-persistence/*.sh` are SYMLINKS into
`scripts/persistence/` in this repo, and the post-commit hook runs the former.
Before 2026-08-14 they were independent copies, which is how a fix to the repo
copy could -- and did -- fail to take effect. Keep them symlinked; if you ever
replace one with a real file, edits to the repo stop mattering silently.

## Notes

- The bucket also holds a pre-2026-08-08 era snapshot under
  `config/projects/-workspace-utmist-vc2-phase2/` (sessions run from inside the
  repo dir). Harmless; rsync never deletes it.
- Big things are NOT in the bucket: checkpoints live in
  `gs://vc2-2026-checkpoints/<run>/`, code in git. This bucket is chat + memory
  only (~tens of MB).
- Do not run two machines resuming the SAME session simultaneously.
- Old-era files (claude-bootstrap.sh / claude-snapshot.sh) are RunPod
  network-volume helpers; harmless but unused on this machine.
