# GENESIS evaluation report — <checkpoint id> — YYYY-MM-DD
<!-- Copy to docs/reports/. Numbers come from `evaluate` JSON; never paste logs. -->

## Verdict (one line)
**Learned | Partially learned | Not learned** — <plain-language sentence>

## Plain-language summary (≤ 8 lines, for non-technical readers)
What the creature can do now, compared with a random mover and a simple scripted rule.

## Setup
- Checkpoint: <id>, step <n>, <h> of training, world spec v<x>, seed(s) <…> (single seed? say so)
- Suite: <name>, held-out seed range <…>, <n> episodes per family, deterministic (argmax) actions

## Results (mean ± 95 % CI)
| family | metric | policy | random | heuristic |
|---|---|---|---|---|
| standard | | | | |
| unseen layouts | | | | |
| visitor perturbations | | | | |
| harder configuration | | | | |

Step-0 (untrained) score: <x> (≈ random → confirms random init)

## Reward-hacking audit
- Behavior statistics: <action histogram, displacement, wall contacts, food-near-without-eating rate, early-termination rate>
- Sampled trajectories (n = <k>): <one line each>
- Exploits found → fix → status: <…> (or "none found", with the statistics above)

## Limitations
<single seed, small eval, untested conditions, …>

## Next change (if not fully learned)
<the next step on the ladder, with a reason>
