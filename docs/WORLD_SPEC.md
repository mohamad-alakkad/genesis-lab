# GENESIS world specification

**Version:** 1.0.0 · **Status:** frozen for training lineage 1 · **Changelog:** 1.0.0 initial (2026-11-18)

Numbers live in `genesis/world_constants.json` (shared by Python and the JS port; never hand-copy).
Reference implementation: `genesis/world.py` (vectorized NumPy, float64). Coordinates: x right, y **down**, origin top-left.

## 1. Units and time
- Arena 16 × 10 u, walls on all sides. Creature radius 0.3 u.
- Tick: 0.1 s world time (10 Hz); the policy decides once per tick. Episode cap: 1000 ticks (truncation, not death).

## 2. Entities
- **Creature (Vela):** kinematic, no inertia. Heading θ, direction (cos θ, sin θ).
- **Seeds (food):** `food_n` active (5; 3 in `harder`), radius 0.25, +0.3 energy (capped at 1.0). Eaten when center distance < 0.55. Respawn immediately by the sampler: clear of slates by r+0.35, outside active vents, > 1.0 u from the creature.
- **Vents (hazards):** `n_vents` circles (radius 0.6–1.0). Each has `present`, `active` and a countdown timer: active lasts 150–399 ticks, dormant 100–299, then toggles. Start active with p = 0.7. Inside an active vent: extra drain 0.025/tick and reward −0.05/tick. "Wake" (visitor or perturbation) sets active with timer 300.
- **Slates (obstacles):** circles, up to 10 slots. Count and radius per family. At layout time each slate keeps ≥ 0.65 u (creature diameter + 0.05) from walls and from other slates, so the creature can always pass.

## 3. Dynamics (fixed order per tick)
1. Heading += Δθ(action), wrapped to [0, 2π). 2. Position += speed(action) · direction.
3. Collisions: slates in slot order 0..9 (push out along the center normal to distance r_c + r_slate; zero-distance → +x), then clamp to [r_c, W−r_c] × [r_c, H−r_c]. `contact` = any push or clamp. Max step 0.15 < every collider radius → no tunnelling.
4. Energy −= 0.0035 + 0.0015·[speed > 0]. 5. Active vent contact: energy −= 0.025, reward −0.05.
6. Eating in seed index order 0..4: energy = min(1, e + 0.3), reward +1, respawn (sampler).
7. Vent timers −1, toggle at ≤ 0 (new duration drawn). 8. Random perturbation (§8). 9. t += 1.
10. Death if energy ≤ 0 (energy clamped to 0, reward −1). Truncation if t ≥ 1000.

## 4. Actions
| id | name | Δθ (rad) | speed (u/tick) |
|---|---|---|---|
| 0 | forward | 0 | 0.15 |
| 1 | forward_left | −0.15 | 0.12 |
| 2 | forward_right | +0.15 | 0.12 |
| 3 | turn_left | −0.3 | 0 |
| 4 | turn_right | +0.3 | 0 |
| 5 | stay | 0 | 0 |
"Left" is −θ, which is counter-clockwise on screen because y points down.

## 5. Observations (42 floats, all in [0, 1])
| index | meaning |
|---|---|
| 3k+0, 3k+1, 3k+2 for ray k = 0..10 | proximity to the nearest solid (wall or slate), seed, active vent along ray k |
| 33 | energy |
| 34 | in active vent (0/1) |
| 35 | contact this tick (0/1) |
| 36–41 | one-hot of the last action (reset: stay) |
- Rays: 11, evenly spaced from θ−120° to θ+120° (k = 0 is the leftmost), range 6 u, cast from the creature center.
- Proximity = 1 − min(t, 6)/6 (0 = nothing in range). Ray–circle: the nearest non-negative intersection; inside a circle → t = 0. Walls: the minimum positive axis distance.
- Policy input normalization: running mean/var frozen at checkpoint, clip ±10, ε = 1e-8.

## 6. Reward
| event | reward |
|---|---|
| seed eaten | +1 |
| tick inside an active vent | −0.05 |
| starvation (energy ≤ 0) | −1 |
An idle living creature outside vents gets 0 per step, so nothing teaches early death.

## 7. Randomness
- PRNG: **mulberry32**, one uint32 stream per env, state = episode seed. `next`: s += 0x6D2B79F5; t = (s ^ s>>15)·(s|1); t = (t + ((t ^ t>>7)·(t|61))) ^ t; return (t ^ t>>14) / 2^32 (uint32 arithmetic).
- Draw order at reset: n_slates; per slate (radius, then sampler); per vent (radius, sampler, active flag, duration); creature (sampler, heading); per seed (sampler).
- Sampler: up to 32 attempts, each draws x then y in [m, W−m] × [m, H−m]; the first clear point wins, otherwise the last candidate (slate insertion is skipped instead).
- Training seeds: [0, 2^31). Held-out evaluation: 3 000 000 000 + family·100 000 + i. Demo seeds: 4 000 000 000+.

## 8. Perturbations (per tick, one draw u)
u < p_food → relocate a random seed (index draw, sampler) · else u < p_food+p_obs → insert a slate (radius draw, sampler; not within r+r_c+0.5 of the creature, not on seeds; skipped when 10 slots are used) · else u < p_food+p_obs+p_vent → wake a random vent.
Family rates: see `world_constants.json` (`perturbed` ≈ 7× the training rates). The demo's visitor actions map to the same three mutations, with the same clearance rules.

## 9. Numerics and parity
World math float64 in both implementations; the policy runs in float64 in Python and in JS. Tolerances: observations 1e-6, logits 1e-4.
Avoided: hash-ordered iteration; NumPy RNG for anything in the world; `Math.random`.
