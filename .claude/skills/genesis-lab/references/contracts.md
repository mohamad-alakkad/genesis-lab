# Contracts: CLI, manifest, status, heartbeat, history, eval result, web bundle

**Summary.** The minimum interfaces that let a brand-new session understand and operate GENESIS with
no conversational memory. Field names below are the defaults; extend freely, but keep the required
fields (`scripts/verify_manifest.py` checks the manifest ones). The project state file format is
[../templates/PROJECT_STATE.md](../templates/PROJECT_STATE.md). Read it when implementing the CLI (W3)
or any tool that reads or writes these files.

## CLI contract

One entry point (default `python -m genesis <cmd>`; record the real one in `PROJECT_STATE.md` →
Commands). Names may change; the capabilities may not. All commands are non-interactive, quiet by
default (`-v` for more), idempotent where sensible, and cap their console output.

| Command | Does | Key flags |
|---|---|---|
| `train` | Start a new run from a config (detached by default; the launcher confirms the first heartbeat and verified checkpoint) | `--config`, `--seed`, `--foreground` |
| `resume` | Find the latest verified checkpoint (local `LAST_GOOD`, then remote manifest), restore, continue. Refuses if none exists | `--new`, `--fork`, `--segments N`, `--until`, `--foreground` |
| `stop` | Create the stop file; the trainer checkpoints and exits cleanly | `--run` |
| `status` | ≤ 25-line summary; refresh `runs/status.json`. Never prints logs | `--json` |
| `evaluate` | Held-out suite vs baselines for a checkpoint → JSON + Markdown in `docs/reports/` | `--ckpt`, `--families`, `--episodes` |
| `export` | Web bundle(s) + milestone brains, with checksums | `--ckpt`, `--milestones` |
| `diagnose` | Health checks (training-and-evaluation.md); prints findings only | — |
| `publish` / `fetch` | Push the latest verified checkpoint to, or pull it from, the external store (also called automatically) | `--verify-only` |

**Exit codes:** 0 ok · 1 error · 2 usage · 3 no checkpoint found (resume without `--new`) · 4
incompatible checkpoint (spaces/spec/arch differ; use `--fork`) · 5 integrity failure (no valid
fallback) · 6 resource guard refused or stopped the run · 10 diagnose warnings · 11 diagnose critical.

## Checkpoint manifest (`manifest.json`, one per checkpoint; the remote copy is the commit point)

```json
{
  "schema": "genesis.manifest/1",
  "run_id": "r20261118-a1",
  "checkpoint_id": "r20261118-a1/000512000",
  "parent": null,
  "global_step": 512000,
  "wall_clock_trained_s": 5400,
  "created_at": "2026-11-18T10:00:00Z",
  "files": [{"path": "model.zip", "sha256": "…", "size": 412345},
            {"path": "vecnormalize.pkl", "sha256": "…", "size": 2345}],
  "config_hash": "sha256:…",
  "world_spec_version": "1.2.0",
  "code_commit": "abc1234", "code_dirty": false,
  "dependencies": {"python": "3.11.9", "torch": "2.x", "stable-baselines3": "2.x", "gymnasium": "1.x"},
  "verified": true,
  "last_known_good": "r20261118-a1/000448000",
  "milestones": ["middle"],
  "eval_summary": {"suite": "heldout-v1", "score_mean": 0.0, "score_ci95": [0.0, 0.0], "episodes": 100}
}
```
Required: `schema, run_id, checkpoint_id, global_step, created_at, files[].path/sha256/size,
world_spec_version, code_commit, verified`. The **remote store** also holds `latest.json`:
`{"checkpoint_id", "manifest_sha256", "published_at"}` (updated last).

## Status file (`runs/status.json`) and the `status` summary

```json
{
  "schema": "genesis.status/1",
  "state": "running",
  "run_id": "r20261118-a1",
  "global_step": 512000, "steps_per_sec": 2400,
  "heartbeat_age_s": 40, "last_local_ckpt_age_s": 300, "last_remote_publish_age_s": 1500,
  "eval_latest": {"score_mean": 0.0, "ci95": [0.0, 0.0]},
  "eval_best": {"score_mean": 0.0, "checkpoint_id": "…"},
  "baselines": {"random": 0.0, "heuristic": 0.0},
  "interruptions": {"count": 1, "lost_time_s": 600, "last_end_reason": "inferred"},
  "resources": {"peak_rss_mb": 820, "limit_mb": 1800, "mechanism": "windows-job-object"},
  "persistence_warning": null,
  "recommended_next": "nothing — training healthy; check again next session"
}
```
`state` ∈ `running | paused | interrupted | completed | plateaued | budget_exhausted | failed`.
`running` with a heartbeat older than 3× its interval is reported as `interrupted (inferred)`.

The summary prints, in order: state + run + step + steps/sec; heartbeat/checkpoint/publish ages; max
progress at risk (time since the last remote publish); latest and best held-out score vs baselines;
interruptions and lost time; peak RAM vs limit; the persistence warning (verbatim, if any); the
recommended next command. ≤ 25 lines, no log excerpts.

## Heartbeat (`runs/<run_id>/heartbeat.json`, every 30–60 s)

`{run_id, host_id, session_id, pid, global_step, wall_time_s, updated_at, last_local_ckpt,
last_remote_publish, steps_per_sec, rss_mb, segment_index}`

## Session history (`runs/<run_id>/history.jsonl`, append-only)

One line per segment, written at start and completed at the end:
`{"segment": 7, "host_id": "…", "start": "…", "end": "…", "end_reason": "segment_done|stop_file|signal|resource_limit|host_pressure|crash|inferred_kill", "end_confirmed": true, "steps_start": 0, "steps_end": 0, "lost_time_s": 0, "peak_rss_mb": 0, "recovered_from": "checkpoint_id|null"}`

## Evaluation result (`docs/reports/eval-<checkpoint>.json` + `.md`)

`{schema: "genesis.eval/1", checkpoint_id, suite, seeds_range, episodes_per_family, deterministic:
true, families: {name: {policy: {metric: {mean, ci95}}, random: {…}, heuristic: {…}}}, verdict:
"learned|partially|not_learned", notes}`

## Web policy bundle (`policy-<milestone>.json`, plus optional binary weights)

```json
{
  "schema": "genesis.policy/1",
  "world_spec_version": "1.2.0",
  "milestone": "final",
  "source_checkpoint_id": "r20261118-a1/000512000",
  "training_step": 512000,
  "observation": {"size": 64, "layout": "see WORLD_SPEC §Observations"},
  "actions": ["forward", "forward_left", "forward_right", "turn_left", "turn_right", "stay"],
  "normalization": {"mean": [], "var": [], "clip": 10.0, "epsilon": 1e-8},
  "network": {"layers": [{"w": [], "b": [], "activation": "tanh"}], "output": "logits"},
  "selection": "argmax",
  "checksum": "sha256 over the canonical JSON without this field"
}
```
Versioning: bump `world_spec_version` on any change to the rules, observations or actions; the
trainer, the export and the JS loader refuse mismatched versions.
