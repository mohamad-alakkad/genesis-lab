# GENESIS — Project state

**Updated:** 2026-11-18 · **Phase:** W4→W5 (pipeline validated locally; first Claude Cloud cycle) · **Next action:** check the cloud probe run (RemoteTrigger list_runs) and `git ls-remote origin 'genesis-store-*'`

## Commands
- CLI: `python -m genesis <train|resume|stop|status|evaluate|export|diagnose|publish|fetch>`
- Tests: `python -m pytest -q` (15 tests: world, PPO gradient check, checkpoints, export isolation)
- Cloud cycle: `bash cloud/cycle.sh <minutes> <seeds…>` then `bash cloud/wait.sh` until `ALL_DONE` (see `cloud/BRIEF.md`)
- Memory-limit wrapper (this server only): `python -m genesis.jobguard -- <cmd>`; the detached launch path uses it automatically

## Resources (measured, W1)
- Server: Windows Server 2019 · 4 logical CPUs (GENESIS allowance 2) · RAM 7.24 GB total / 3.7–4.4 GB free · Disk free 34.9 GB · Python 3.12.3, numpy 2.5.3, psutil 7.2.2, no torch
- **GENESIS RAM budget: 2 GB hard. OS limit: 1800 MB via Windows Job Object (+BELOW_NORMAL, affinity 2 cores); over-allocation test PASSED 2026-11-18 (2.5 GB attempt stopped at 1.7 GB, MemoryError)**
- Egress: GitHub y (SSH auth ok), PyPI y, HF y, S3 y · Credentials: no HF/AWS/GH tokens (names only checked)
- Background survival on the server: probe pid 7828 started 2026-11-18 14:17Z (`runs/probe/heartbeat.json`): pending
- Throughput (server, 1 core, 128 envs incl. PPO updates): 3.4k steps/s · Peak RSS 54 MB
- **Compute policy (user, 2026-11-18): training in Claude Code Cloud only; no GitHub Actions compute; no server training without explicit authorization.**
- Cloud sandbox capabilities: pending the probe (cpus, RAM, session length, push to `genesis-store-*`)

## Persistence
- Channel: branch `genesis-store-s<seed>` on github.com/mohamad-alakkad/genesis-lab (**public repo**), one orphan commit force-pushed, verified by remote hash and SHA-256 on fetch
- Local round-trip verified against a bare test repo (2026-11-18). GitHub round-trip: pending the first cloud cycle

## Training
- Active run: none locally. Validation run `t-short` (server, 400k steps, before the cloud directive): held-out food 0.13 → 6.63 [4.87, 8.40]
- Baselines (standard, 100 episodes): random food 0.26, ticks 175 · heuristic food 3.38, ticks 356

## Gates (evidence pointers)
- [x] Stage A — `pytest -q` 15 passed; 3.4k steps/s
- [x] Stage B — t-short eval_history: step 0 → 0.13 food, 393k → 6.63 (CIs disjoint)
- [x] Stage C — random and heuristic, 4 families × 100 episodes (`docs/reports/baselines.json` written by the first `evaluate`)
- [ ] Pipeline round-trip to GitHub (checkpoint → publish → fetch in a fresh clone) — cloud cycle 1
- [ ] Detached segment without session — cloud cycles
- [ ] Resilience test a–f — 
- [ ] Stage D/E eval + reward-hacking audit — 
- [ ] World + policy parity — 
- [ ] MVP checklist — 

## Cloud runs in flight
- (none yet)

## Open problems / questions for the user
- Publishing the visual pack in the public repo (default: keep `design/` local-only; the web build will include only the assets it needs, after your approval) — 2026-11-18

## Pointers
- Decisions: `docs/DECISIONS.md` · World spec: `docs/WORLD_SPEC.md` (v1.0.0) · Reports: `docs/reports/` · Design: `design/GENESIS-VISUAL-PACK/` (local)
