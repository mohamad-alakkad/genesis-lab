# GENESIS v3 — a world that does not promise safety

Status: design (branch `v3`). Lineage: gen4 on world 3.0.0, trained only after gen3 tells us whether hiding is learnable.

## Principle: the Invisible Hand
Every visitor power is something the world also does by itself. During training "nature" fires the same
events at random (the perturbation channel), so the policy has met them. A visitor only chooses where and
when. Visitors never touch the policy, never write to observations (except the clearly labelled
out-of-distribution sense experiments in the Research Console), and never script outcomes.

Every effect is one of three kinds, and the interface labels which:
- **World**: changes the simulation (the creature can sense it through its normal inputs).
- **Senses**: changes only what the network receives (Research Console, out-of-distribution, labelled).
- **Presentation**: rendering or sound only (camera, time scale, sound mix). Never a world consequence.

Influence is limited: the visitor holds **3 motes of influence**. Each act spends one, and they return slowly
(one per ~20 s of world time). Small, deliberate acts; no god mode.

## Ecology (new rules; each has a cause, an observable warning, and a consequence)
| System | Cause | Warning the creature can sense | Consequence |
|---|---|---|---|
| Meadow fertility | each meadow has a slow fertility cycle (≈1 day, phase-shifted); rain raises it | fewer seeds visible; seeds regrow slower | food moves around the map: memory and exploration pay |
| Rich hollow | one meadow sits beside the vent field | seeds visible, vents visible | richer feeding at higher risk |
| Fissures (unstable ground) | tremors open a fissure where the ground is "tired" (near vents) | 40-tick warning: the vent channel at that spot rises gradually before it opens | an open fissure drains energy like a vent, then closes |
| Withering thickets | long dry spells wilt a thicket; rain regrows it | the shelter ray channel scales with thicket health | a wilted thicket hides nothing: safety is temporary |
| Injury | active vent / open fissure contact | in-hazard input | energy loss now; stamina recovery slowed for 50 ticks |

## Visitor acts (world kind unless stated)
| Act | Effect | Hidden trade-off |
|---|---|---|
| **Plant** (tap ground) | a seed sprouts and ripens over 30 ticks | draws Vela into the open |
| **Raise a thicket** | a young thicket grows over 40 ticks, lives ~300 ticks, then withers | can block an escape path |
| **Call the rain** | starts a rain spell | hides Vela from sight, but dulls its hearing |
| **Lure** (tap a point) | the predator goes to search there | can pull it past a hidden Vela |
| **Wake the ground** | starts a fissure (with its warning) | danger, or a barrier between Vela and the predator |
| Slow time, follow camera, mute (presentation) | none in the world | none |

Each act gets its own visual and sound signature, and the timeline in the Research Console records it with its kind.

## Observations (v3 adds, replaces nothing semantically)
Ray channels: solid, seed, vent+fissure (warning shows as a partial value), shelter (scaled by health), predator, water.
Self: + injury timer, + meadow-fertility hint is NOT given (must be inferred from what is seen).

## Training plan
gen4 from random init on world 3.0.0, 4 seeds, cloud cycles. Nature's rates for the five acts are chosen so
each happens a few times per episode. Evaluation families: standard, drought (no rain), lure-heavy,
fissure-heavy, one-thicket, no-predator, plus "visitor scripted" episodes replaying recorded visitor acts.
Gates: beats random and the scripted survivor on survival with disjoint CIs; hiding under threat > 15%;
reward-hacking audit (no exploits around fissure warnings or thicket timing).
