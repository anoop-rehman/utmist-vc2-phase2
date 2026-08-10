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
bash pull.sh               # restores into ~/.claude

# 3. recreate the WORLD the transcripts talk about, at the SAME paths
#    (transcripts reference absolute paths; matching them keeps context valid)
git clone git@github.com:anoop-rehman/utmist-vc2-phase2 /workspace/utmist-vc2-phase2
git clone https://github.com/KJaebye/competevo /workspace/competevo
git clone https://github.com/Khrylx/Transform2Act /workspace/Transform2Act

# 4. resume — MUST cd to the same cwd the session ran from, the project slug
#    is derived from it (current sessions: /workspace -> slug "-workspace")
cd /workspace && claude --resume
```

`CLAUDE_PULL_DEST=/some/dir bash pull.sh` restores to a scratch dir if you want
to inspect before overwriting.

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
