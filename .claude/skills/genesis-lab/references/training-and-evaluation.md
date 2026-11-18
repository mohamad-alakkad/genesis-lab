# Training and evaluation

**Summary.** What "from scratch" means here, the default algorithm and CPU settings, reward design
pitfalls, the staged plan A–F with exit evidence, baselines and how to keep them from shipping, the
held-out evaluation protocol and metrics, the reward-hacking audit, stopping criteria evaluated by
code, `diagnose` checks, and how to report failure honestly. Read it in W2, W3, W4 and W8.

## From scratch: the three-way distinction

1. Define a new neural policy and train it from random initialization — **required**.
2. Use an established library for the optimization (Stable-Baselines3, or a small, well-tested
   PyTorch implementation) — **encouraged**.
3. Load pretrained weights or pretrained models of any kind — **not part of the experiment; never**.

Record the initialization seed. Prove random init: the step-0 checkpoint's held-out score ≈ the
random-agent baseline (within the confidence interval). Keep that step-0 checkpoint as a milestone.

## Algorithm and CPU defaults

- **Default:** PPO with an MLP policy (e.g. 2 hidden layers of 64–128 units) on a discrete action
  space. SB3 is fine. A small hand-written PPO is fine if it is simpler for the batch-vectorized env.
  DQN is a reasonable alternative for a small discrete action set (state the replay-buffer
  persistence decision and its memory cost). Justify any other choice in the decisions log.
- **Vectorization:** one process stepping N envs as arrays (see the resource budget in
  architecture-and-world.md). Prefer `n_envs` of 16–64 inside that process over extra processes.
- **Threads:** set PyTorch, OMP and MKL thread counts explicitly to the GENESIS core allowance (at most
  half the measured cores). Never oversubscribe.
- **Normalization:** observation normalization (running mean/var) and optionally reward normalization.
  Persist the statistics with every checkpoint. Freeze them for evaluation and export.
- **Hyperparameters:** start from library defaults for small control tasks. Tune only when Stage B
  shows a specific failure (see the next-change ladder). No sweeps in the MVP.
- **Seeds:** train ≥ 2–3 seeds if the budget allows (sequentially, within the RAM budget). Otherwise
  state "single seed" as a limitation.

## Reward design

Prefer a reward that is mostly the true objective: `+` on eating (scaled to the energy gained),
`−` on hazard contact, a small survival term, optional potential-based shaping (difference of a
potential function, which is policy-invariant). Pitfalls to check up front:
- **Negative per-step reward** teaches early death. Keep net per-step reward ≥ 0 for a creature
  that is merely alive, or make termination itself costly.
- **Proximity rewards** without potential form teach circling near food without eating.
- **Movement cost too high** teaches freezing. Too low means energy does not matter.
- Reward only what the world spec says, so the JS world needs no reward code.

## Staged plan (each stage has objective exit evidence)

| Stage | What | Exit evidence |
|---|---|---|
| A Env verification | Unit/property tests: action effects, observation bounds and layout, collisions (no tunnelling through walls/obstacles at max speed), energy accounting, reward per event, termination; deterministic replay from seed; throughput benchmark | `pytest -q` green; steps/sec recorded in state file |
| B Short run | Small budget (≈ 5–15 min). Learning curve vs step-0 | Held-out score at end > step-0 score, beyond CI overlap; curve summary (≤ 10 numbers) in report |
| C Baselines | Random agent + ≥ 1 scripted heuristic on the same held-out suite | Baseline JSON committed under `docs/reports/` |
| D Main training | Autonomous, segmented; scheduled eval every K steps on a **fixed** held-out seed set; best-so-far tracked separately from latest | Status shows eval history; `best` pointer in manifest |
| E Generalization | Unseen layouts, unseen seeds, visitor-style perturbations, ≥ 1 deliberately harder configuration (more obstacles / scarcer food) | Per-family results with CI in the eval report |
| F Demo | Export only a checkpoint that passed Stage E | Bundle references a checkpoint id listed as Stage-E-passed |

## Baselines (evaluation-only, MUST NOT ship)

- **Random:** uniform over actions.
- **Heuristic (greedy):** e.g. steer toward the nearest sensed food ray and away from sensed hazards.
  It uses the same observations as the policy, so the comparison is fair.
- Isolation: baselines live in an evaluation-only module that the export path and the web build never
  import. Add a test that fails if the export code imports it, and a check that the web bundle
  contains no heuristic code (e.g. grep the built bundle for its marker string).

## Evaluation protocol

- Fixed held-out seed set from the reserved evaluation range, recorded in the spec. Never train on it.
- Deterministic policy (argmax) for evaluation, because the demo uses argmax. Optionally also report
  stochastic.
- Episodes: enough for useful CIs (default ≥ 100 per scenario family). Report mean ± 95 % CI
  (bootstrap or t-interval) and the number of episodes.
- Scenario families: standard, unseen layouts, visitor-style perturbations (food moved, obstacles
  inserted mid-episode within demo bounds), harder configuration.
- Output: compact JSON + Markdown per evaluation (schema in contracts.md). Never print per-episode rows.

**Metrics** (choose only those relevant to the implemented behaviors): survival time (ticks), food
eaten per episode, energy efficiency (food per distance or per energy spent), hazard contacts per
episode, success rate (survived to the step cap), and the improvement over each baseline.

## Reward-hacking audit (required before declaring success)

Compute behavior statistics on held-out episodes and inspect 3–5 sampled trajectories as compact
numeric summaries (per 50-tick windows: distance moved, food eaten, nearest-food distance, wall
contacts, action histogram) or a short render. Check for:
- circling or hovering near food without eating;
- freezing or spinning to save energy (action-histogram dominance, near-zero displacement);
- wall hugging or boundary exploits; clipping through obstacles; respawn or edge-case glitches;
- deliberately ending episodes early because the per-step reward is negative;
- the hazard being ignored because its penalty is too small to matter.
Each exploit found → fix reward or dynamics → document it in the report and decisions log →
retrain (new lineage if spaces or the spec changed). Record "audit passed" only with the statistics attached.

## Stopping criteria (evaluated by the program, never by you)

The trainer writes a terminal state into status:
- `completed` — target met (e.g. held-out score ≥ target for 2 consecutive evals);
- `plateaued` — best held-out score not improved by > ε for M consecutive evals (default M = 5);
- `budget_exhausted` — step or wall-time budget used up;
- `failed` — unrecoverable error or a `diagnose`-critical condition persisting across a resume.

## `diagnose` checks (print findings only)

NaN/inf in losses, weights or observations; entropy collapse (entropy < 5 % of its initial value before
the target is reached); flat reward (no held-out improvement over the last N evals while still early);
value-loss explosion (> 10× its running median); throughput drop (> 40 % below the recorded benchmark);
disk pressure (< 1 GB free or `runs/` above its cap); memory near budget (peak RSS > 1.2 GB) or a
`resource_limit`/`host_pressure` exit; stale heartbeat (> 3× its interval while state is `running`);
failed or overdue uploads; manifest/checksum mismatch. Exit 0 = no findings, 10 = warnings,
11 = critical.

## Honest failure and the next-change ladder

If the policy does not beat both baselines on held-out data, the report says so plainly. Never
compensate for a broken design with more training time. Try, in order, and record each change:
1. reward bug or exploit (audit) → fix;
2. observation missing needed information (e.g. hazard type not in rays) → add it;
3. world too hard → simplify (fewer obstacles, more food, longer energy budget);
4. hyperparameters (learning rate, entropy coefficient, `n_steps`, network width);
5. algorithm change.
