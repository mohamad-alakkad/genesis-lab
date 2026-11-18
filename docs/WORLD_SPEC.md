# GENESIS world specification

**Version:** 2.2.0 · **Status:** frozen for lineage gen4 · **Changelog:** 2.2.0 adds 5 memory inputs (Vela's own last predator percept; updated at the end of each step/reset; obs 89 → 94) · 1.0.0 seeds/vents/slates, tag `v1-baseline` · 2.0.0 living world (gen2) · 2.1.0 meadows at thicket edges (2–4 u from a shelter edge), stalk hearing 1.5 → 2.5 (gen2 never discovered hiding: 0.3% shelter use under threat)

All numbers live in `genesis/world_constants.json`, shared byte-for-byte with the JS port. Reference implementation: `genesis/world.py`. Coordinates: x right, y **down**.

## 1. Units, time, places
- Arena 24 × 16 u (the observer camera follows Vela), walls on all sides. Tick 0.1 s; one decision per tick. Episode cap 1500 ticks (2.5 days).
- **Day/night:** day = 600 ticks; light = ½ + ½·cos(2π(t + t0)/600) (1 = noon). t0 is random per episode (fixed in the `night` family).
- **Rain:** per episode, rain starts with p = 0.3. Dry spells last 400–899 ticks, rain 150–399, then the state toggles. Rain: the predator's detection ×0.7, the creature's hearing ×0.6, seeds regrow twice as fast.

## 2. Entities
- **Vela (creature):** radius 0.3; speed v and turn rate ω with momentum (§3); energy (hunger) and stamina (exertion), both in [0, 1].
- **Seeds (food):** `food_n` slots, each bound to meadow i mod 3. Placed within 2.2 u of the meadow center (polar draw), never in slates, shelters, pools or active vents, and > 1 u from Vela. Eating: +0.25 energy (cap 1), reward +1. The slot then **regrows** after 120–249 ticks (twice as fast in rain) at a new point of the same meadow.
- **Meadows (3):** open ground 2.0–4.0 u from the nearest shelter edge (the forest-edge ecotone: food in the open, cover within reach), ≥ 1 u from pools.
- **Shelters (thickets, 2–4):** circles r 0.9–1.3. Inside a shelter, Vela cannot be seen or caught. The predator cannot enter (it is pushed out).
- **Pools (1–2):** circles r 1.2–2.0. Inside, Vela's target speed ×0.55 and the predator's speed ×0.6.
- **Vents (2):** circles r 0.6–0.9 with active/dormant cycles (active 150–399, dormant 100–299). In an active vent: energy −0.02/tick and reward −0.05/tick.
- **Slates (4–8, up to 12):** circles; they block movement and the predator's line of sight. Layout gap ≥ 0.65 u.
- **The predator (Shade, a scripted world rule):** radius 0.35. Modes: **patrol** (0.065 u/tick to random waypoints) → **hunt** when it sees Vela → **search** at Vela's last seen position (0.09, 50 ticks) → patrol; **rest** for 60 ticks after a long hunt (fatigue ≥ 120). During a hunt it **stalks** at 0.05 (quiet) until within 2.8 u, then **pounces** at its hunt speed (train U[0.10, 0.145]; standard U[0.125, 0.145]; relentless 0.16). It sees Vela if Vela is not in a shelter, line of sight is clear of slates, and distance < detect × visibility, where detect = lerp(4.6 at night, 2.2 at noon) × family multiplier × (0.7 if raining) and visibility by Vela's speed: <0.02 → 0.45, <0.07 → 0.65, <0.15 → 1.0, else 1.35. It loses Vela beyond 1.4× that range. Catch: center distance < 0.6 with Vela outside a shelter → episode ends.

## 3. Dynamics (fixed order per tick)
1. Action target (v*, ω*) from §4. Dash needs stamina ≥ 0.1, otherwise it acts as cruise. Stamina < 0.2 → v* × 0.6. In a pool → v* × 0.55.
2. v += (v* − v)·0.3; ω += (ω* − ω)·0.4; heading = (heading + ω) mod 2π; position += v·(cos, sin).
3. Collisions: slates in slot order (push out along the normal), then wall clamp. `contact` flag.
4. in_shelter, in_water. Resting = action rest ∧ v < 0.02.
5. Energy −= (resting ? 0.0011 : 0.0022) + 0.05·v². Stamina += dash −0.014 · cruise/veer −0.0005 · turn/creep +0.003 · rest +0.008 (+0.008 in a shelter), clipped to [0, 1].
6. Active vent contact. 7. Eating in slot order. 8. Seed regrowth in slot order. 9. Vent timers. 10. Weather timer.
11. Predator update (mode transitions, then movement, push-out from slates and shelters, wall clamp, then waypoint draws). Catch test.
12. Random perturbation (§8). 13. t += 1. 14. Death: starvation (energy ≤ 0, reward −1) or caught (reward −2). Truncation at 1500.

## 4. Actions (discrete, 8)
| id | name | v* (u/tick) | ω* (rad/tick) |
|---|---|---|---|
| 0 | cruise | 0.12 | 0 |
| 1 | veer_left | 0.10 | −0.12 |
| 2 | veer_right | 0.10 | +0.12 |
| 3 | dash | 0.22 | 0 |
| 4 | turn_left | 0.03 | −0.25 |
| 5 | turn_right | 0.03 | +0.25 |
| 6 | creep | 0.05 | 0 |
| 7 | rest | 0 | 0 |

## 5. Observations (89 floats in [−1, 1])
- Rays: 11 over ±120° (k = 0 leftmost), range 6. Channels per ray (index 6k + c): 0 solid (wall/slate), 1 seed, 2 active vent, 3 shelter, 4 predator (only if within sight = lerp(2.5 at night, 6 at noon)), 5 pool. Value = 1 − min(t, 6)/6.
- 66 energy · 67 stamina · 68 v/0.22 · 69 ω/0.25 · 70 in_shelter · 71 in_vent · 72 in_water · 73 contact · 74 light · 75 raining · 76 sin(day phase) · 77 cos(day phase)
- 78–80 hearing: predator within hearing radius (pounce 6, stalk 2.5, patrol/search 3; ×0.6 in rain; silent when resting) → proximity 1 − d/r, sin and cos of the bearing relative to heading (0 when not heard)
- 81–85 memory of the last predator percept (seen on a ray or heard): recency exp(−Δt/60), sin and cos of the bearing to where it was sensed relative to the current heading, remembered proximity max(0, 1 − d/8), was-pouncing; all × recency, 0 if never sensed
- 86–93 one-hot of the last action (reset: rest)

## 6. Reward
seed +1 · tick in an active vent −0.05 · starvation −1 · caught −2. Nothing else: no reward for hiding, resting or fleeing. Those must be learned as means to survive and eat.

## 7. Randomness
- mulberry32 per env, state = episode seed (algorithm in v1.0.0, unchanged).
- Reset draw order: n_slates; per slate (r, sampler) → n_shelters; per shelter (r, sampler) → n_pools; per pool (r, sampler) → per vent (r, sampler, active, duration) → per meadow (sampler) → creature (sampler, heading) → predator (sampler, hunt speed, waypoint sampler) → per seed (polar sampler: angle, radius per attempt) → t0 (unless fixed) → rain flag, weather duration.
- Sampler: ≤ 32 attempts (x then y); the first clear candidate wins, otherwise the last one (slate insertion is skipped instead).
- Seeds: training [0, 2^31) · held-out 3 000 000 000 + family·100 000 + i · demo 4 000 000 000+.

## 8. Perturbations and visitor actions (same rules)
One draw u per tick: u < p_food → relocate a random live seed · then p_obs → insert a slate (clear of Vela, the predator, seeds, shelters and pools) · then p_vent → wake a random vent · then p_alert → alert the predator (it searches Vela's current position). Visitor actions map to: move seed, place slate, wake vent, make a sound (alert).

## 9. Numerics and parity
float64 world and policy in both implementations; tolerances: observations 1e-6, logits 1e-4. The predator uses atan2 for heading only (presentation); its movement uses normalized vectors.
