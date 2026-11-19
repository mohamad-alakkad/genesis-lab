# GENESIS — Project state

**Updated:** 2026-11-19 13:05Z · **Phase:** GENESIS 1.0 track — world 3.0.0 live with gen6 life-reward brain; report covers gen1–gen6
**Preview:** https://genesis-preview-3b416cc0d8.dentexa-com.workers.dev (world 3.0.0, gen6-life-s1 best) · staging: https://genesis-staging-3b9a70614d.dentexa-com.workers.dev · deploy: `bash web/deploy.sh <worker> web/public/policies`
**Next action:** next research question = can shelter use emerge where stealth fails (e.g. scent tracking)? candidates in docs/DECISIONS.md; experience polish (sound listening by the user, creature anatomy close-up camera)

## Commands
- CLI: `python -m genesis <train|resume|stop|status [--remote branches]|evaluate|export|diagnose|publish|fetch>`
- Tests: `python -m pytest -q` (18) · JS parity: `cd web && node --test test/parity.test.mjs test/policy.test.mjs` (fixtures: `python -m genesis.fixtures --policy <ckpt_dir>`)
- Cloud cycle: `bash cloud/cycle.sh <min> <seeds>` → `bash cloud/wait.sh` until ALL_DONE (`cloud/BRIEF.md`)
- Server memory guard (only if the user authorizes server compute): `python -m genesis.jobguard -- <cmd>`

## Resources (measured)
- Server: Win Server 2019, 4 CPU, 7.24 GB RAM (3.7–4.4 free), 34.9 GB disk. **Job Object 1800 MB verified** (2.5 GB alloc stopped at 1.7 GB). Probe pid 7828 (heartbeat survival) started 14:17Z: check `runs/probe/heartbeat.json`
- **Compute policy (user): Claude Code Cloud only; no GitHub Actions; no server training without explicit authorization.**
- Cloud sandbox (measured cycle-01): 4 CPU, 16 GB RAM, 30 GB disk, Python 3.11, git push to any non-main branch OK, survived ≥ 20 min; v1 ≈ 2.1–3.3k steps/s per process with 2 in parallel

## Persistence
- Store branches on github.com/mohamad-alakkad/genesis-lab (**public**): `genesis-store-<lineage>-s<seed>` (gen1 = `genesis-store-s1/s2`). One orphan commit, verified by remote hash, SHA-256 on fetch
- GitHub round-trip from the cloud: verified (cycle-01 published every 8 min)

## Training
- **gen1 (world v1.0.0, tag `v1-baseline`)**: 3.28M steps per seed in a 20-min cloud cycle. Held-out food: s1 0.08 → 27.66, s2 0.30 → 29.05 (heuristic 3.38, random 0.26). Learned strongly; kept as the reproducible baseline
- **gen2 (world v2.0.0, living world)**: hourly routine `trig_01WshMDVj5EGSv1gd6EpsKfE` (cron :57), 45-min cycles, seeds 1–4, 60M steps/seed, first run 14:57Z
- gen2 baselines (standard, 40 episodes, survival ticks/caught): random 296/0.25 · forager 394/0.80 · survivor 711/0.40. No predator: forager survives 1196 ticks (55% to the cap)

## Gates
- [x] Stage A: pytest 18 passed (v2 world, PPO gradient check, checkpoint fallback, trainer smoke)
- [x] Stage B: v1 t-short (0.13 → 6.63) and gen1 cloud (0.08 → 27.7)
- [x] Stage C: v2 baselines random/forager/survivor (`evalonly/suite.py`)
- [x] Pipeline round-trip to GitHub from the cloud: gen1 store branches
- [x] Detached segment without session + cross-environment resume: gen2 cycle 2 (fresh sandbox, 15:57Z) resumed at 32,669,696 from the store, prev end confirmed, 0 s lost
- [ ] Resilience test a–f: run in the cloud (kill / fresh-clone fetch / corrupt checkpoint)
- [ ] Stage D/E eval + reward-hacking audit (gen2)
- [ ] World + policy parity: v1 PASSED; v2 world PASSED (20 fixtures × 900 steps incl. visitor events); v2 policy parity test to regenerate with the final brain
- [ ] MVP checklist

## Cloud runs in flight
- None. Training triggers disabled (trig_01WshMDVj5EGSv1gd6EpsKfE, trig_017M19mXuACfto4d9UDVraiK); one-shot evaluator trig_0188ABQ8EoN8dC9o8WLooKft (re-point prompt + run for the next generation).
- Results: gen4 docs/reports/gen4-results.*, gen5 docs/reports/gen5-families.*, gen6 docs/reports/gen6-families.* (fresh seeds, 10 families). Experiment registry: cloud/experiments/gen*.json.

## Open problems / questions for the user
- gen2 learning flattening at ~750 survival ticks (survivor baseline 711), caught ≈ 0.7: diagnose after 60M (next-change ladder: observation/reward audit before hyperparameters)
- Visual/audio: desktop verified in real Chrome (scene, Research Console: overview/live/training/experiments); phone layout NOT verified (window resize unavailable); audio not listened to; 5-critic review queued for after training
- gen2-s1 (38.0M) and gen2-s2 (34.0M) ended by the programmed plateau criterion (clean exit: plateau), confirming the plateau finding; s3/s4 still finishing their cycle
- gen3 (world 2.1.0) starts with the 16:57Z hourly cycle; refresh `python -m genesis.research` + re-export + redeploy when gen3 brains exist (site must switch to the 2.1.0 sim files at the same time)
- Publishing the visual pack in the public repo (default: keep `design/` local; the web build ships only the assets it needs) — 2026-11-18

## Pointers
- `docs/DECISIONS.md` · `docs/WORLD_SPEC.md` (v2.0.0) · `docs/reports/` · `design/GENESIS-VISUAL-PACK/` (local) · `cloud/`
