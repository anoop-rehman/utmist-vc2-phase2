# Working while the laptop is closed

*2026-08-10. Companion to CLAUDE-PERSISTENCE.md (which covers the pod DYING;
this covers the SSH connection dying).*

## What survives an SSH drop, verified on this pod

| thing | survives? | why |
|---|---|---|
| Background Claude sessions (the `&`/bg kind, incl. Remote Control chats) | YES | run under `claude daemon` — PPID 1, no TTY, zero SSH dependency |
| Training runs / nohup'd scripts (drills, competevo sanity) | YES | nohup + detached, parented to init |
| Subagents spawned by a background session | YES | children of the daemon tree |
| An interactive `claude` typed into an IDE/SSH terminal | **NO** | attached to the pts — dies with the connection |
| Anything typed into a bare SSH shell without nohup/tmux | **NO** | same reason |

## The two habits

**1. Long-horizon autonomous work → background sessions.** Give the task, close
the laptop. The session keeps executing its plan; talk to it any time from
Remote Control (phone / claude.ai). This is how the CompeteEvo/Transform2Act
milestones are being run.

**2. Interactive terminal work → tmux.** tmux 3.2a is installed. Start every
interactive session inside it:

```bash
tmux new -As work        # attaches if it exists, creates if not
claude                   # or anything else
# laptop closes -> tmux detaches, everything keeps running
tmux attach -t work      # next morning
```

`Ctrl-b d` detaches on purpose; `tmux ls` lists sessions.

## Caveats

- A **pod stop** kills daemon, tmux, everything. That's what the GCS chat backup
  (post-commit hook + `sync.sh`) and checkpoint GCS syncs are for: transcripts,
  memory, and weights all survive; resume per CLAUDE-PERSISTENCE.md.
- A background session that finishes its task goes idle until you next message
  it — "paused" in that sense is just "done and waiting", not lost.
- Don't run the same chat interactively AND via Remote Control expecting two
  parallel conversations; it is one session.
