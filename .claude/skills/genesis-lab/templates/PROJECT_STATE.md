# GENESIS — Project state
<!-- Copy to the repo root. Keep it ≤ 60 lines and current. Update at the end of every workflow. -->

**Updated:** YYYY-MM-DD · **Phase:** W1 Initial assessment · **Next action:** <one line, with the exact command>

## Commands
- CLI: `python -m genesis <train|resume|stop|status|evaluate|export|diagnose|publish|fetch>`
- Tests: `pytest -q` · Parity: `<cmd>` · Demo: `<cmd>`
- Memory-limit wrapper: `<cmd>` (always used for heavy processes)

## Resources (measured, W1)
- Host: <os> · CPU: <n> logical (GENESIS allowance <n>) · RAM total/free: <x>/<y> GB · Disk free: <z> GB
- **GENESIS RAM budget: 2 GB hard (user-authorized). OS limit: 1800 MB via <mechanism>; over-allocation test: <pass/fail/date>**
- Python: <ver> · Egress: GitHub <y/n>, PyPI <y/n>, HF <y/n>, S3 <y/n> · Credentials present: <names only>
- Background processes survive session end: <yes/no/pending> (probe started <date>)
- Throughput: <steps/s> · Peak RSS observed: <mb>

## Persistence
- Channel: <none | orphan branch | releases | HF | S3> · Status: <ok / WARNING: Cross-environment recovery NOT guaranteed>
- Last remote publish: <checkpoint id, date>

## Training
- Active run: <run_id or none> · Last known state: <state> at step <n> (<date>)
- Best held-out: <score ± ci> vs random <x>, heuristic <y>

## Gates (evidence pointers)
- [ ] Stage A — <path/number>
- [ ] Stage B — 
- [ ] Stage C — 
- [ ] Pipeline round-trip (checkpoint → publish → restore) — 
- [ ] Detached segment without session — 
- [ ] Resilience test a–f — 
- [ ] Stage D/E eval + reward-hacking audit — 
- [ ] World + policy parity — 
- [ ] MVP checklist (see Skill acceptance-checklist.md) — 

## Open problems / questions for the user
- <one line each; include the recommended default and the date asked>

## Pointers
- Decisions log: `docs/DECISIONS.md` · World spec: `docs/WORLD_SPEC.md` (v<x>) · Reports: `docs/reports/`
- Design references: `design/`
