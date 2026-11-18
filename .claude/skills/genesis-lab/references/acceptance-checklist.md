# Acceptance checklist (MVP) and deferred scope

**Summary.** The 13 MVP items, each with the evidence that proves it, and the list of enhancements
that are explicitly out of the MVP. The MVP is done only when every item has its evidence recorded
(path or number) in `PROJECT_STATE.md` → Gates. Read it in W10, or whenever someone asks "are we done?".

## MVP — all required

| # | Item | Evidence |
|---|---|---|
| 1 | Policy initializes from random weights; no pretrained weights anywhere | Code path of policy construction (no load at init); step-0 eval ≈ random baseline (CI overlap) |
| 2 | Simulation generates experience autonomously; Stage A passes | `pytest -q` output summary; throughput number |
| 3 | Training ran detached, with no LLM involvement, for ≥ 1 full segment | `history.jsonl` segment spanning a period with no session activity; heartbeat continuity |
| 4 | Trained policy beats random and heuristic baselines on held-out configurations, or the report states clearly that it does not | Eval report with means, 95 % CI, episodes, verdict |
| 5 | Checkpoints are atomic, checksummed, load-tested; last known good retained | Tests for the write → verify → promote path; `verify_manifest.py` pass; `LAST_GOOD` pointer |
| 6 | Checkpoints persist outside the disposable runtime, or status prominently warns and the user was asked | Remote manifest + matching checksum, or the warning text in status + the question recorded in Open problems |
| 7 | Resilience test (a)–(f) passed in a fresh working directory | Entries in `history.jsonl` + a short results table in the state file or a report |
| 8 | Measured CPU/RAM within limits: **GENESIS peak RSS ≤ 2 GB**, under an OS-enforced limit, with no host-pressure incident | `peak_rss_mb` per segment; enforcement mechanism + over-allocation test result |
| 9 | Reward-hacking audit completed and documented | Audit section in the eval report with behavior statistics |
| 10 | Python/JS world parity and policy parity tests pass | Test output summary (fixtures, steps, max error, action agreement) |
| 11 | Demo shows the real policy acting, with milestone brains and a plain "What's real here"; visuals never override decisions | Demo build loading bundles from Stage-E-passed checkpoints; honesty text; heuristic absent from the bundle (check result) |
| 12 | A new session reconstructs status from `PROJECT_STATE.md` + `status` alone | State file ≤ 60 lines, current; `status` output covers state, risk and next command |
| 13 | The user can get a plain progress report (≤ ~15 lines) without logs | An example report in `docs/reports/` |

## Deferred enhancements (explicitly out of MVP)

Multiple creatures or species; evolution/population training; richer behaviors; curriculum
systems; external push alerts with an independent monitor; hosted leaderboards; analytics;
multi-seed hyperparameter sweeps beyond basic robustness; sound design; custom art pipelines.
