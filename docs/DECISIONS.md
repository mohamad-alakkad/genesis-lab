# GENESIS decisions log (one line each, newest last)

- 2026-11-18 Behavior set: eat seeds for energy, avoid active vents, route around slates, under an energy budget (Skill default; it matches the visual pack's seed/vent/slate props).
- 2026-11-18 World: continuous 16×10 arena (the design tokens' aspect), circles only, kinematic creature, 10 Hz, cap 1000 ticks — exact and cheap to port.
- 2026-11-18 Observations: 11 rays × 3 proximity channels (solid / seed / vent), not first-hit one-hot, so a vent never hides the seed behind it; no absolute coordinates.
- 2026-11-18 Actions: 6 discrete (argmax in the demo, trivial parity).
- 2026-11-18 Algorithm: PPO written in NumPy (manual backprop, gradient-checked in tests), 2×64 tanh MLPs for policy and value, orthogonal init. No torch: about 50 MB RSS instead of 400+, and float64 weights port to JS exactly.
- 2026-11-18 Hyperparameters: 128 envs × 128 steps, 4 epochs, minibatch 2048, lr 3e-4 linear decay to 5%, γ 0.99, λ 0.95, clip 0.2, ent 0.01. Library-style defaults, no sweeps.
- 2026-11-18 Env: one process, batch-vectorized NumPy world; per-env mulberry32 streams make each env independent of batch size (tested).
- 2026-11-18 Checkpoint: npz (weights, Adam, normalization) + state.json (counters, RNG states, config) + manifest; load-tested; LAST_GOOD pointer. About 360 KB.
- 2026-11-18 Persistence: a git branch `genesis-store` in github.com/mohamad-alakkad/genesis-lab (PUBLIC, user-provided), one orphan commit force-pushed, verified by remote commit hash, plus a SHA-256 re-check on fetch. Remote `origin` is resolved from the clone, so it works through the cloud git proxy.
- 2026-11-18 Resource guard (this server): Windows Job Object 1800 MB, BELOW_NORMAL priority, affinity 2 of 4 cores; the over-allocation test was stopped at 1.7 GB. Watchdog at 1.4/1.6 GB.
- 2026-11-18 Compute (user directive, overrides earlier choices): training runs in **Claude Code Cloud** sandboxes in short, published cycles. No GitHub Actions compute. No training on the Windows server without explicit authorization (only one 3-minute pipeline validation ran there, before the directive).
- 2026-11-18 Design pack: kept local-only (`design/`, git-ignored) until the user approves publishing the assets in the public repo.
