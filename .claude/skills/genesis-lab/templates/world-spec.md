# GENESIS world specification
<!-- Copy to docs/WORLD_SPEC.md. This is the single source of truth for the Python reference and the JS port.
     Every number lives here (or in a generated shared constants JSON). Bump the version on any rule, observation or action change. -->

**Version:** 1.0.0 · **Status:** draft | frozen · **Changelog:** <one line per version>

## 1. Units and time
- Length unit: <e.g. 1 u = creature radius> · Arena: <W × H u>, walls on all sides
- Tick: <dt s of world time>; the policy decides once per tick · Episode cap: <ticks>

## 2. Entities
- Creature: radius, max forward speed, turn rate, (no inertia | inertia rule)
- Food: count, radius, energy value, respawn rule (where, when)
- Hazards: count, shape, effect (energy drain per tick | termination), penalty
- Obstacles: count range, shapes (circles / axis-aligned boxes), size range, placement rules

## 3. Dynamics (in order of evaluation per tick)
1. Apply action → heading and velocity
2. Integrate position
3. Resolve collisions (walls, obstacles): rule, order, no tunnelling guarantee
4. Eating: contact test, energy gain, respawn
5. Hazard contact effect
6. Energy update: base decay + movement cost
7. Termination checks (energy ≤ 0, tick cap)

## 4. Actions
| id | name | effect |
|---|---|---|

## 5. Observations (layout, order, normalization)
| index range | meaning | raw range | normalization |
|---|---|---|---|
- Rays: count K, field of view, max range, angle order (left → right), hit types and one-hot order
- Ray/shape intersection rules (closest hit; ties broken by <rule>)

## 6. Reward
| event | reward |
|---|---|
- Net per-step reward for an idle living creature: <value, must not teach early death>

## 7. Randomness
- PRNG: <name>, exact algorithm + seeding; used for layout generation and perturbations in both implementations
- Training seed range: <…> · Evaluation (held-out) seed range: <…> · Demo seed range: <…>
- Randomization bounds: layouts, counts, start pose

## 8. Perturbations (training and demo share these bounds)
- Food relocation: when, how often, where allowed
- Obstacle insertion: max total obstacles, min distance from the creature, per-episode limit
- Demo interactions map to: <list>

## 9. Numerics and parity
- World math precision: float64 · Policy: float32 · Parity tolerances: observations 1e-6, logits 1e-4
- Known non-portable operations avoided: <list>
