# Checkpointing, persistence and recovery

**Summary.** What a checkpoint contains, the write → verify → promote procedure, cadence and
retention, graceful shutdown (signals and a stop file, including Windows), the compatibility guard,
how to choose and use an external store, heartbeat and interruption accounting, and the mandatory
resilience test (a–f). Read it in W3 (CLI), W4, W6, W7, and whenever recovery or progress safety is in question.

## Checkpoint contents

One checkpoint = one directory `runs/<run_id>/ckpt/<global_step>/` containing:
- policy weights; optimizer/algorithm state (e.g. SB3 `.zip`);
- normalization statistics in a separate file (e.g. `VecNormalize` pickle or JSON);
- counters: global step, episodes, segments, wall-clock trained, eval history pointer;
- RNG states where practical (Python, NumPy, Torch). Accept that vectorized envs reset on resume,
  and say so in the spec;
- full resolved config; world-spec version; code commit hash (plus a dirty flag); dependency versions;
- off-policy only: an explicit decision whether the replay buffer is persisted (default: no for
  buffers > 200 MB, because of the RAM and upload cost; record the impact);
- `manifest.json` (schema in contracts.md) — written last.

Tiny models mean a checkpoint is usually well under a few MB. Measure and record the size.

## Write → verify → promote

1. Write each file to `<name>.tmp`, flush, `fsync`, then `os.replace` to the final name (atomic on the
   same volume, on POSIX and Windows). On POSIX also fsync the directory.
2. Compute SHA-256 and the size of each file.
3. **Load-test:** in the same process, deserialize into a fresh policy object and run one forward pass on a
   recorded observation; compare with the live policy's output (exact or ≤ 1e-6).
4. Write `manifest.json` (via temp + replace) with `verified: true`.
5. Update `runs/<run_id>/LAST_GOOD` (a one-line pointer, temp + replace). Only now is it "latest".
A crash at any step leaves the previous `LAST_GOOD` intact.

**Cadence:** by wall time *or* steps, whichever comes first. Default every 10 min, plus an early first
checkpoint (~60 s, for launch confirmation), plus at every segment end. Measure checkpoint overhead and
keep it < 2–3 % of training time.
**Retention:** keep the last 3 verified + all milestones (`step0`, `early`, `middle`, `best`, `final`) +
anything an eval report references. Prune only after a newer checkpoint is verified.

## Graceful shutdown

- POSIX: trap SIGTERM/SIGINT. Windows: handle `CTRL_BREAK_EVENT`/SIGBREAK and SIGINT (SIGTERM cannot be
  caught on Windows).
- **Portable stop file:** the trainer checks `runs/<run_id>/STOP` every iteration. This is the
  preferred stop path for `stop` commands, scheduled jobs and the resource watchdog.
- On any of these: checkpoint now → write the clean-exit marker `runs/<run_id>/CLEAN_EXIT` (end reason,
  step, time) → publish if time allows → exit 0.

## Compatibility guard

`resume` refuses (exit 4) if the observation space, action space, network architecture or world-spec
version differs from the checkpoint. `--fork` starts a new lineage: new run id, `parent` = source
checkpoint id, recorded in the manifest and the decisions log. Never silently reinterpret old weights.

## External persistence

Never bloat main-branch history with frequently changing binaries. Otherwise choose freely after
**detecting** what exists (W1). Compare candidates on availability here, reliability, simplicity,
permissions, size limits and overhead:

| Channel | Notes |
|---|---|
| Orphan `checkpoints` branch, latest N, force-pushed | Works over plain git/SSH (no API token needed); bounded history; easy fetch in a fresh clone. Strong default when git push works |
| GitHub Release assets | Needs an API token; clean per-file URLs; good for milestone/demo bundles |
| Hugging Face Hub model repo | Needs `HF_TOKEN`; generous size; check egress |
| S3-compatible (S3, R2, B2) | Needs keys; most robust; may cost money → ask user |
| CI artifacts | Expire (retention days); acceptable only as a secondary copy |

Creating a new remote repository, using a paid service, or a first push to a public location is an
escalation item: ask once with a recommended default. Before a first push, audit what is being pushed (no
secrets, no logs with environment data, no large files on main) and follow the host's publish rules.

**Remote cadence** may be lower than local (default every 30–60 min and at segment end). State the
trade-off in status: *max progress lost on a hard kill = time since the last remote publish.*

**Upload protocol:** upload the checkpoint files under a temporary name/prefix → read back or query the
remote checksum and compare → upload the heartbeat snapshot → **update the remote manifest last** (it is
the commit point). An interrupted upload never becomes "latest". Retry with backoff a bounded number of
times; a failure goes into status and `diagnose`, and training continues.

**No external store available:** every `status` output prints
`Cross-environment recovery NOT guaranteed: no external checkpoint store configured.`
Ask the user once for the credential or choice (record it under "Open problems"), and continue.

**Secrets:** read only from environment variables or the platform's secret mechanism. Never print, log
or commit them; redact them in error messages.

## Heartbeat, session history, interruption accounting

- **Heartbeat** `runs/<run_id>/heartbeat.json` every 30–60 s (temp + replace): fields in contracts.md.
  It is published remotely with the manifest at publish cadence.
- **Session history** `runs/<run_id>/history.jsonl`, append-only, one compact line per segment: start,
  end, end reason, steps, host id, peak RSS. Published with the manifest.
- **On every start**, the program computes and records:
  - previous end reason: **confirmed** (clean-exit marker present) or **inferred** (no marker; last
    heartbeat age);
  - estimated gap duration where measurable (last heartbeat → now);
  - **training time lost** = wall time between the last *published* checkpoint and the last heartbeat
    (local resume: last *local* checkpoint);
  - cumulative interruption count and lost time.
- A terminated machine cannot announce its own death. Immediate alerts would need an independent
  external monitor (deferred, and only if trivially supported). The default is a clear report at the next start.

## Resilience test (hard gate for MVP)

Run it with a short config (minutes, small step budget), within the resource budget, and record each
result in session history and the state file:

- **(a)** Start training; let it publish at least one verified remote checkpoint.
- **(b)** Hard-kill it (`kill -9` / `taskkill /F /PID`), with no graceful path.
- **(c)** In a **fresh working directory with no local run data** (a fresh clone, or a new
  environment), fetch from the remote store and run `resume`.
- **(d)** Confirm: the global step continues from the checkpoint's step; the recovery is recorded as
  `inferred` with the lost time; a short post-resume evaluation is within noise (CI overlap) of the pre-kill evaluation.
- **(e)** Graceful stop (stop file, plus SIGTERM on POSIX / CTRL_BREAK on Windows) → `CLEAN_EXIT`
  present, end reason `confirmed`, nothing lost beyond the in-flight iteration.
- **(f)** Corrupt or truncate the latest checkpoint (local and, in a scratch copy, remote) → `resume`
  detects the checksum or load-test failure, falls back to the last known good checkpoint, and says so.

A checkpoint system that has not passed (a)–(f) is incomplete.
