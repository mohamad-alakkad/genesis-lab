# GENESIS: learning to survive in a small living world
### Technical report — working draft (updated per generation)

**Status:** gen1–gen6 complete (world 3.0.0 is the current world). All numbers below come from files in `docs/reports/` or store branches on the public repository. Nothing in this report is synthetic.

## Abstract
We train a small agent, Vela, from random weights with PPO in a 2D predator–prey ecology and study why defensive behavior does or does not emerge. Across six generations of controlled, multi-seed experiments evaluated on fresh held-out seeds: (1) a short credit-assignment horizon was the first bottleneck (γ 0.99 → 0.997: +320–440 survival ticks, captures roughly halved); (2) a memory of the agent's own last predator percept helped only with the short horizon; (3) training against faster predators did not produce condition-dependent hiding; (4) **the objective was the dominant cause of being caught**: replacing the per-seed reward with a reward for staying alive (+0.01 per tick, nothing else) lengthened survival with separated confidence intervals in all 9 predator conditions and cut captures by about two-thirds (13% vs 39% in the standard world; 36% vs 76% against a predator faster than the agent), with equal performance when no predator is present. The defensive strategy that emerged is not shelter use but **stealth**: the agent creeps three times as often, moves slower and keeps farther away, exploiting the world rule that detection range shrinks with prey speed. We report the hypotheses we falsified along the way. The world runs identically in Python (training) and JavaScript (browser), verified by tick-level parity tests, so the interactive experience executes exactly the trained policy.

## 1. Motivation
Artificial-life demonstrations often blur scripted and learned behavior. GENESIS keeps the two strictly separate: the world (physics, ecology, the predator) is scripted and versioned; every creature decision is the argmax of a network trained from random initialization. The research question is practical: *what does a small, memoryless or lightly-memoried policy need in order to develop defensive behavior against a predator, and what does it actually develop?*

## 2. Environment
A continuous 24 × 16 arena at 10 Hz (episode cap 1500 ticks). Full rules: `docs/WORLD_SPEC.md`; constants: `genesis/world_constants.json` (shared byte-for-byte with the browser port).

| version | lineage | what changed |
|---|---|---|
| 1.0.0 | gen1 | 16 × 10 arena, seeds, vents, slates, kinematic agent |
| 2.0.0 | gen2 | momentum, stamina, rest, shelters, meadows with regrowth delay, pools, rain, day/night, stalking predator (patrol → stalk → pounce → search → rest) |
| 2.1.0 | gen3 | meadows at thicket edges; stalking audible at 2.5 u |
| 2.2.0 | gen4 | +5 memory inputs (the agent's own last predator percept) |
| 3.0.0 | gen5 | fertility cycles, a rich hollow beside the vents, withering thickets, fissures with sensed warnings, injury, five "acts" shared by nature and visitors |

**Observation** (v2.2: 94, v3.0: 95 floats): 11 rays over ±120° with per-ray proximity channels (solid, seed, vent/fissure, shelter, predator within light-limited sight, water), internal state (energy, stamina, speed, turn, cover, hazard, water, contact), light, rain, day phase, hearing (proximity and bearing, radius depends on predator mode and rain), memory (v2.2+), injury (v3.0), last action. **Actions:** 8 discrete (cruise, veer ±, dash, turn ±, creep, rest). **Reward:** +1 per seed, −0.05 per tick in an active vent, −1 starvation, −2 caught. Nothing rewards hiding, resting or fleeing.

## 3. Agent and training
PPO (clip 0.2, GAE λ 0.95, 4 epochs, minibatch 2048, Adam 3e-4 with linear decay, entropy 0.01, value 0.5, grad-norm 0.5), two tanh MLPs (128 × 128) for policy and value, orthogonal init, implemented in NumPy with gradients verified against finite differences. 256 vectorized environments × 128 steps per update; running observation normalization frozen at export. Training runs in Claude Code Cloud sandboxes (4 CPU, 16 GB) as 45-minute cycles of a plain Python trainer, 4 processes per sandbox (≈10–13k steps/s each), publishing SHA-256-verified checkpoints to git store branches every 8 minutes. Later cycles resume from the store; cross-sandbox resumption was verified.

## 4. Evaluation protocol
Deterministic (argmax) policies on held-out seeds (≥ 3·10⁹, disjoint from training seeds < 2³¹), 100–200 episodes per checkpoint, per family. Because "best" checkpoints are selected on one seed block, final verdicts use a **disjoint fresh block** (3·10⁹ + 50 000 …). Baselines are evaluation-only scripts that cannot reach the browser bundle: random, *forager* (greedy, ignores the predator), *survivor* (flee to a sensed shelter, rest inside while the predator is sensed). Two further reference policies quantify strategy ceilings: a privileged *oracle hider* (true predator state) and a *sensory hider* (only the agent's observations plus an explicit memory window).

## 5. Results by generation

### gen1 (world 1.0.0) — the pipeline learns
Held-out seeds eaten per episode: 0.08 → 27.66 (seed 1) and 0.30 → 29.05 (seed 2) in 3.3M steps, vs 3.38 for a greedy heuristic and 0.26 for random.

### gen2 (world 2.0.0) — good forager, poor survivor
Held-out survival plateaued at ≈750 ticks (4 seeds; two stopped by the automatic plateau rule). Behavioral diagnosis (best checkpoint, 60 episodes): 666 ticks, caught 70%, 11.9 seeds, **0.3%** of threatened time in shelter, vs the scripted survivor's 640 ticks, 47%, 7.8 seeds, 22%.
*Hypothesis:* shelters too far from food, so hiding is never discovered.

### gen3 (world 2.1.0) — hypothesis falsified
Moving meadows to thicket edges changed little: 689 ticks, caught 78%, shelter under threat 1.7% (survivor 724, 60%, 23%). A trajectory decomposition located the failure precisely:

| | gen3 | survivor |
|---|---|---|
| captures without any cue in the prior 8 s | 0% | 0% |
| median warning before capture (ticks) | 50 | 17 |
| threat bouts with shelter entry | 12.5% | 50% |
| left shelter while still threatened | 91% | 17% |
| predator still sensed when it left | 73% | 8% |
| dash share inside a shelter under threat | 64% | 0% (89% rest) |

Perception was sufficient; the agent ran *through* shelters. Strategy ceilings in the same world:

| reference policy (60 episodes) | survival | caught |
|---|---|---|
| oracle: hide only once hunted | 645 | 70% |
| oracle: hide whenever the predator is within 5 u | 1069 | 17% |
| sensory hider, memory 0 / 30 / 80 / 150 ticks | 724 / 846 / 899 / 933 | 60% / 40% / 33% / 22% |

Reactive hiding is nearly worthless; preemptive avoidance works, and with the agent's own senses its value grows with memory.

### gen4 (world 2.2.0) — a controlled 2 × 2: memory × horizon
Memory inputs on/masked (masking applied identically in training, evaluation, export and browser) × γ 0.99/0.997; 2 paired seeds per arm; 40M steps; fresh-seed verdict, 200 episodes per checkpoint (400 per arm).

| arm (best checkpoints) | survival (95% CI) | caught | seeds | sheltered under threat |
|---|---|---|---|---|
| memory, γ 0.997 | **1059** [1011, 1107] | 36% | 21.6 | 3.5% |
| no memory, γ 0.997 | **1074** [1027, 1121] | 38% | 22.1 | 3.5% |
| memory, γ 0.99 | 739 [689, 789] | 64% | 14.8 | 2.5% |
| no memory, γ 0.99 | 635 [589, 680] | 71% | 12.6 | 1.2% |

Pre-registered rule (CI-separated survival): the **horizon effect** holds in both memory conditions; the **memory effect** holds only at γ 0.99. Every γ 0.997 run, best or final, beats every γ 0.99 run.

**What was learned.** A trace of the deployed brain (gen4-mem-g997-s1 @ 28.0M) shows evasive flight: 47% dash when the predator is sensed (9% otherwise), shelter entry in 27% of threat bouts but exit while still threatened in 72%, median predator distance 7 u, 20 captures in 60 episodes (gen3: 47), median warning 74 ticks.

**Why not hiding?** The evader matches the privileged preemptive hider's survival (≈1070 ticks) while eating ≈40% more (≈21.8 vs 15.5 seeds): ≈37% more return. Under this reward and this predator, concealment is the inferior strategy. We therefore do not count its absence as a learning failure.

### gen5 (world 3.0.0) — faster predators do not create conditional hiding
World 3.0.0 adds fertility cycles, a rich hollow, withering thickets, fissures with warnings, injury and five acts shared by nature and visitors. Arms: predator-speed range 0.10–0.17 (faster than the agent's cruise 0.12) vs 0.10–0.145; 4 paired seeds each; γ 0.997; memory on; 50M steps. Evaluation: best checkpoints, 10 families, fresh seeds, 400 episodes per arm per family.
No family showed a CI-separated difference between arms, and sheltering under threat did not rise where flight fails (relentless 2.6–2.9% vs standard 3.6–3.7%). Generalization was robust except against fast or frequently alerted predators (relentless ≈770, lure-heavy ≈835 ticks). References on the same seeds explain why: in the relentless world a privileged preemptive hider is caught 31% (vs 75%) and lives longer (854 vs ≈770 ticks), but the per-seed reward still favors the evader (≈14.5 vs ≈9.7 return).

### gen6 (world 3.0.0) — the objective
Single change: reward for staying alive (+0.01 per tick alive; food, hiding, resting and fleeing are only instrumental) vs the per-seed reward. Same world, predator range 0.10–0.17, γ 0.997, memory, 4 paired seeds, 50M steps; same evaluation.

| family | per-seed reward: survival / caught | life reward: survival / caught |
|---|---|---|
| standard | 1064 / 39% | **1235 / 13%** |
| relentless | 785 / 76% | **1093 / 36%** |
| lure-heavy | 855 / 62% | **1135 / 28%** |
| drought | 1020 / 44% | **1176 / 17%** |
| perturbed | 959 / 56% | **1184 / 26%** |
| no predator (control) | 1446 / 0% | 1444 / 0% |

Survival is CI-separated in favor of the life reward in all 9 predator families and equal without a predator. The pre-registered prediction that the life reward would increase sheltering under threat was **falsified** (≈4% in both). Trajectory analysis of the best checkpoints (60 episodes) shows **stealth**: the life brain creeps 9% of the time vs 3%, moves slower (0.080 vs 0.089), is within the predator's sensed range less (6% vs 9% of ticks) and keeps farther away (7.8 vs 7.3 u); captures 1/60 vs 25/60 in the standard world and 23/60 vs 45/60 against the faster predator. Detection in this world scales with prey speed (×0.65 when creeping), and the policy discovered and uses that rule.

## 6. Engineering for credibility
- **One world, two implementations.** Python (vectorized NumPy, float64) trains; JavaScript runs the browser. Parity: 22 recorded episodes × 900 ticks across 11 families, including every visitor act, match within 1e-6 per observation input.
- **Portable randomness.** Each episode is a mulberry32 stream seeded by an integer; draw order is specified.
- **Policy fidelity.** Exported bundles carry normalization statistics, both networks, the input mask and the world version; the browser refuses mismatched versions. Logit parity is tested against Python.
- **Checkpoints.** Write → SHA-256 → load-test → manifest → git store; corruption falls back to the previous verified checkpoint (tested). Bugs found by these checks are listed in `docs/DECISIONS.md` (e.g. CRLF rewriting of stored JSON on Windows; a temp-file race between concurrent trainers).

## 7. Limitations and threats to validity
- Two seeds per arm in gen4, four in gen5 and gen6. Intervals are per-episode, not per-seed; seed-level variance is reported in the per-run tables.
- The predator is a hand-written rule system; conclusions about "defense" are relative to it.
- Hand-designed memory features (which quantities to remember) were tested against masking, not against a learned recurrent memory.
- "Best" checkpoints are re-evaluated on fresh seeds; earlier generations (gen2/gen3 diagnoses) used the selection seeds.
- The browser presentation (animation, sound, labels) is derived from simulation state and policy outputs, but its emotional reading is an interpretation, not a measurement.

## 8. Reproducibility
Code, configs and evaluation scripts: github.com/mohamad-alakkad/genesis-lab. Checkpoint stores: branches `genesis-store-<run>`. Evaluation: `python -m genesis.experiment cloud/conditions.json 200 50000`; trajectory analysis: `python -m genesis.evalonly.trace <checkpoint_dir>`.

## 9. Status of the release
The public experience runs world 3.0.0 with the gen6 life-reward brain (`gen6-life-s1` best). Visitor interventions are tagged and excluded from all evaluation data.
