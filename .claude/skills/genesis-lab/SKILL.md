---
name: genesis-lab
description: Operating manual for project GENESIS, a small digital creature that learns survival behavior from scratch with CPU-only reinforcement learning in a compact 2D world, keeps its training progress in verified, remotely published checkpoints, and is shown in a cinematic 3D web demo. Use when starting or continuing GENESIS work; training, resuming, stopping or checking the creature's training; checkpoint, persistence or cross-session recovery questions; evaluation, baselines, reward-hacking audits or progress reports; or building, exporting a policy for, or deploying the GENESIS demo.
---

# GENESIS Lab

"We created a digital creature, gave it a world, and let it learn how to survive."
You (Claude Code) are the **Execution Engineer**: you design, implement, validate, launch,
diagnose and report. Plain Python processes do the training. A browser demo shows the result.

## 1. The layer rule (MUST)

**You are intelligence, programs are compute.** You write and launch programs; programs simulate,
train, checkpoint, publish, evaluate and write status. Training MUST progress with no LLM
involved: no Claude session, Claude API call or recursive Claude invocation may act as the loop
that keeps training alive or makes training decisions. A human or a plain scheduler running one
`resume` command is fine. Layer detail: [architecture-and-world.md](references/architecture-and-world.md).

## 2. Decision priority order (resolves most conflicts)

1. Correctness of actual learned behavior
2. Reliable preservation of training progress
3. Low token consumption
4. Simplicity of implementation and operation
5. CPU and memory efficiency
6. Reproducibility and measurable results
7. Visual impact
8. Extensibility

## 3. Invariants (MUST — never trade these away)

- **Authentic learning.** Every meaningful behavior comes from a policy trained from random
  initialization. No pretrained weights or models of any kind. Scripted paths, pathfinding or
  heuristics never stand in for the policy. Physics, collisions, rendering and world rules may be scripted.
- **Heuristic baselines are evaluation-only** and structurally unable to reach the web bundle.
- **Verified checkpoints only.** Nothing is "latest" until it is atomically written, checksummed and load-tested.
- **Resilience test is a hard gate** before MVP (hard kill, fresh-directory recovery, SIGTERM, corrupted checkpoint).
- **No external store → say so every time:** `status` prints
  `Cross-environment recovery NOT guaranteed: no external checkpoint store configured.`
- **Honesty in the demo.** Visuals consume decisions; they never choose or override them. Failure is reported as failure.
- **Escalation boundaries** in §8 are respected.
- **Resource budget (user-authorized, hard).** On the shared production server, GENESIS may use
  at most **2 GB RAM in total** (all its processes combined: trainer, env workers, evaluation,
  builds, browsers). Server stability and existing services always come first. Before any
  training or other heavy process: set a *hard OS-enforced* memory limit below 2 GB, cap threads
  and env workers, lower CPU priority, and run at most one heavy GENESIS process at a time. The
  program's own watchdog checkpoints and stops itself when nearing the limit. Procedure and
  thresholds: [architecture-and-world.md § Resource budget](references/architecture-and-world.md#resource-budget-and-enforcement).
  Exceeding the budget is never acceptable, not even briefly.
- **Host rules win** for operational actions: if the machine has its own rules (a user/global
  `CLAUDE.md`, infrastructure docs, deploy windows), follow them over this Skill.

## 4. Project layout conventions (defaults; record changes in the decisions log)

| What | Path |
|---|---|
| Project state (read first, ≤ 60 lines) | `PROJECT_STATE.md` (repo root) |
| Decisions log (one line per decision) | `docs/DECISIONS.md` |
| World specification (source of truth) | `docs/WORLD_SPEC.md` |
| Reports (eval, audit, progress) | `docs/reports/` |
| User visual references (authoritative) | `design/` (add a `design/README.md` if missing) |
| Run data: checkpoints, heartbeat, status, logs | `runs/` — git-ignored, never committed |

Templates for the first three: [templates/](templates/) — `PROJECT_STATE.md`, `world-spec.md`, `report.md`.

## 5. Session start (Workflow 0) — do this every session, in this order

1. Read `PROJECT_STATE.md` only. Do **not** read old conversations, full logs or metric files.
   If it does not exist → copy [templates/PROJECT_STATE.md](templates/PROJECT_STATE.md) to the repo root and go to Workflow 1.
2. If a CLI exists (state file → "Commands"), run `status` once. If `diagnose` is flagged there, run it once.
3. Pick the workflow from the table, do it, then update `PROJECT_STATE.md` (phase, gate evidence, next action).

| Situation (from state + status) | Next |
|---|---|
| No state file, or resources unmeasured | W1 |
| Resources known, no decisions log | W2 |
| No env/CLI, or Stage A failing | W3 |
| Stage A passed, Stages B/C not evidenced | W4 |
| Pipeline validated, no run active | W5 |
| `running`, heartbeat fresh | Nothing to do for training — work on W8/W9 or end the session |
| `running` but heartbeat stale, or `interrupted` | W7 |
| Persistence warning present | W6 (ask user once, per §8) |
| `completed` / `plateaued` / `budget_exhausted` | W8 |
| `failed` | `diagnose` → fix → W4 |
| Evaluated policy passed Stage E | W9 → W10 |

Cold-start target: orientation in a few hundred tokens of reading.

## 6. Workflows (goal → steps → exit gate)

Every workflow starts with §5 and ends by updating `PROJECT_STATE.md`.

**W1 Initial assessment** — replace planning assumptions (≈4 vCPU, small ephemeral disk, no GPU,
interruptions likely; **RAM budget fixed at 2 GB for GENESIS regardless of machine size**) with
facts. Steps: measure CPU count, total and *currently free* RAM, free disk, OS, Python
version; probe egress to GitHub, PyPI, Hugging Face, S3/R2 (one cheap request each, read the error
body); detect credentials by presence only (`gh auth status`, env var names — never values);
launch the detached **heartbeat-only session-lifetime probe** (architecture-and-world.md) and
check it in a later session; build and test the memory-limit wrapper (must kill a
deliberate over-allocation test process). **Gate:** a "Resources" block in the state file with
measured numbers, the enforced limit mechanism, and the probe result (or "pending, check next session").

**W2 Architecture selection** — choose behavior set, observations, actions, algorithm, library,
persistence channel, runtime/scheduler, demo stack. One line of rationale each in `docs/DECISIONS.md`.
Read [architecture-and-world.md](references/architecture-and-world.md) and
[training-and-evaluation.md](references/training-and-evaluation.md). **Gate:** decisions log
complete; `docs/WORLD_SPEC.md` drafted from the template with a version number.

**W3 Environment and policy implementation** — implement the world spec (Python reference),
Gymnasium-compatible env, policy definition, the CLI ([contracts.md](references/contracts.md)),
heuristic + random baselines in an evaluation-only module. **Gate:** Stage A tests pass; throughput recorded.

**W4 Pipeline validation** — short run (minutes) proving a learning signal, then the full
checkpoint → publish → restore loop. **Gate:** Stage B curve above step-0 and Stage C baseline
numbers recorded; one remote round-trip with matching checksum (or the persistence warning active).

**W5 Autonomous training** — launch a segmented run detached from the session via the CLI, always
inside the memory-limit wrapper with the measured peak from W4 well under budget; the
launcher itself confirms the first heartbeat and first verified checkpoint and exits non-zero
otherwise. Then **end the session**. **Gate:** a later `status` shows a full segment of progress
with no session activity.

**W6 Checkpoint publishing** — verified local checkpoint → upload under a temp name → verify remote
checksum → update remote manifest last. Read [checkpointing-and-recovery.md](references/checkpointing-and-recovery.md).
**Gate:** a remote copy whose SHA-256 matches the manifest (verify with `scripts/verify_manifest.py`).

**W7 Cross-session recovery** — fresh environment → fetch latest verified remote checkpoint →
`resume` → confirm continuity and the recorded interruption. **Gate:** resilience test (a–f) passed
and recorded in session history.

**W8 Evaluation and reporting** — held-out suite, baselines, generalization (Stage E), reward-hacking
audit, plain-language report from [templates/report.md](templates/report.md). **Gate:** report with
means, confidence intervals and an explicit verdict: learned / partially / not learned.

**W9 Visual application** — JS world port + exported policy + presentation. May start against a
random/early policy once `WORLD_SPEC.md` is stable; ships only a policy that passed W8. Read
[visual-experience.md](references/visual-experience.md). **Gate:** world and policy parity tests
pass; smooth on desktop and a mid-range phone.

**W10 Public demo readiness** — honesty labels, milestone brains, performance, accessibility,
deployment plan. Publication needs user permission. **Gate:** every item in
[acceptance-checklist.md](references/acceptance-checklist.md) has evidence.

## 7. Token-efficiency rules

Do: design, implement, test, run short validations, launch, read compact summaries, diagnose real
failures, report briefly.

Never:
- poll training, `tail -f` logs, or sleep/wait inside a session for progress;
- read full logs or metric files — use `status`, `diagnose`, `evaluate` outputs;
- narrate individual episodes;
- build an LLM-in-the-loop controller or call Claude to steer training;
- keep a session open merely to wait — end it; next session runs `status` once.

Also: install quietly (`pip -q`, `npm --silent`); run tests quietly (`pytest -q`, failures only);
every script caps its own console output; reference large artifacts by path, never print them;
launch-time confirmation is the launcher's job, not yours.

## 8. Autonomy and escalation

Decide routine technical matters yourself without asking: file layout, algorithms,
hyperparameters, libraries, world simplifications, visual implementation. Record them in the decisions log.

Ask the user **only** for: credentials/access you cannot obtain; external services or anything
that costs money; enabling CI schedulers or changing repository settings; public publication;
major product-direction changes (what the creature fundamentally is). Ask one concise question with
a recommended default, record it under "Open problems", and continue all unblocked work.

Scope: one creature, one compact world, one primary policy, a small behavior set, one training
pipeline, one checkpoint mechanism, one evaluation suite, one demo. No Kubernetes, distributed
training, databases, microservices, orchestration frameworks, heavy 3D physics in training, image
observations or large dependency stacks. Defer everything else until MVP gates pass.

## 9. Reference map — read only what the current task needs

| When you are… | Read |
|---|---|
| Designing the world, runtime, segments, scheduler, Python↔JS parity | [references/architecture-and-world.md](references/architecture-and-world.md) |
| Choosing the algorithm, rewards, staged plan, metrics, audit, stopping | [references/training-and-evaluation.md](references/training-and-evaluation.md) |
| Checkpointing, persistence channel, heartbeat, recovery, resilience test | [references/checkpointing-and-recovery.md](references/checkpointing-and-recovery.md) |
| Building the demo, policy bridge, interactions, honesty layer | [references/visual-experience.md](references/visual-experience.md) |
| Implementing the CLI, manifest, status, heartbeat, web bundle schemas | [references/contracts.md](references/contracts.md) |
| Checking MVP readiness or declaring done | [references/acceptance-checklist.md](references/acceptance-checklist.md) |

## 10. Bundled helper

`python .claude/skills/genesis-lab/scripts/verify_manifest.py <manifest.json> [--dir <files_dir>]`
— stdlib only. Checks required manifest fields, file sizes and SHA-256. Prints findings only; exit
0 = valid, 1 = invalid, 2 = usage/unreadable. Use it in tests and after every remote fetch.

## 11. Progress reports to the user

≤ 15 lines, plain language, no logs: what the creature can do now (vs random and heuristic
baselines, with numbers), training state, whether progress is safe outside this machine, open
questions for the user, next step.
