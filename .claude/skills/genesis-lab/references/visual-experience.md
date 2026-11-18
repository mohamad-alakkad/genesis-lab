# Visual experience and the policy bridge

**Summary.** How the demo turns the real policy's decisions into a cinematic 3D presentation without
faking capability: separation of rendering from decisions, the exported policy bundle and the JS
forward pass, milestone "brain ages", the sensor view and the "What's real here" honesty layer,
bounded visitor interactions, design-reference handling, performance and accessibility targets, and
deployment. Read it in W9 and W10.

## Separation (MUST)

Rendering, animation and effects consume the world state and the action the policy chose. They
never choose, delay, veto or alter actions. Animation may *express* real actions (anticipation before
a turn, squash and stretch, a glow on eating, a dim when energy is low). It must not invent capability
(no fake "looking around" toward food the sensors cannot see, no path previews).

Loop per world tick: JS world step → observation → policy forward pass → argmax action → world
applies it. Between ticks, render at display rate by interpolating poses. The demo's tick equals the
training tick (from the world spec).

## Policy bridge

- Export (`export` command) produces a versioned bundle (schema in contracts.md): weights, normalization
  constants (frozen), observation layout and spec version, action semantics, checksum, source checkpoint
  id, training step, milestone label.
- For a tiny MLP, prefer a **hand-written TypeScript forward pass** (matrix-vector products and
  activations, a few dozen lines, no runtime dependency). ONNX + a web runtime is an acceptable
  alternative if the architecture grows. Record the choice.
- **Deterministic selection (argmax)** in the demo, stated in the "What's real here" text.
- The JS side verifies the bundle checksum and spec version at load and refuses mismatches with a visible error.
- Parity tests (world + policy) from architecture-and-world.md MUST pass before a bundle ships.
- The heuristic baseline is never in the web build (see training-and-evaluation.md).

## Honesty layer (the most fascinating feature)

- **Brain age switcher.** Ship milestone brains: untrained (step 0), early, middle, final, all exported
  from real checkpoints, same world. Visitors watch it go from flailing to competent.
- **"See what it senses"** toggle: sensor rays colored by hit type, energy bar, live action
  probabilities. It shows the real observation vector and policy output.
- **"What's real here"**: 4–6 plain sentences. Learned: every movement decision, made by a small
  neural network that began random and learned only from simulated experience (state steps and
  training time). Not learned/rendered: the 3D art, lighting, animation, effects and the world's rules.
  Visitors change the world; they never teach the creature.
- Optional: one honest number per brain (e.g. average food found in the evaluation suite), taken from the eval report.

## Interactions (bounded, never training)

A small set: move food, place obstacle (max count and placement rules from the world spec's
perturbation bounds; never on top of the creature), reset world (new seed from the demo seed
range), switch brain age, toggle sensor view. Every interaction maps to a perturbation the training
distribution covered. Anything outside the bounds is disabled in the UI, not silently allowed.

## Visual direction

**User references are authoritative.** Look in `design/` (images, notes, `design/README.md`). Follow
them over your own taste. If `design/` is empty or missing, build a coherent, replaceable placeholder
quickly, and do not sink time into speculative art.

Default direction without references: premium, cinematic, dark atmospheric world; a small distinctive
creature with bioluminescent accents; controlled lighting and depth (fog, rim light, soft bloom used
sparingly); smooth expressive motion; elegant typography; a strong first five seconds (the creature
is already alive and moving when the page appears; the title fades in over it).

Stack: Three.js or React Three Fiber suggested, not mandated. Keep dependencies few. The creature lives
on the 2D plane; the camera and art are 3D.

## Performance, accessibility, fallback

- Targets: 60 fps on desktop; ≥ 30 fps on a mid-range phone; initial load ideally < 2 MB JS + assets
  excluding fonts; policy forward pass < 1 ms per tick.
- Adaptive quality (pixel ratio cap, effect toggles) on low frame rates.
- **No-WebGL fallback:** a 2D canvas top-down view driven by the same JS world and policy, which is still honest.
- `prefers-reduced-motion`: no camera drift or shake, reduced effects; the creature still moves (that is the content).
- Keyboard-operable controls, readable contrast, text alternatives for the honesty layer.
- Local builds and headless-browser checks count against the 2 GB RAM budget: never run them while
  training unless the combined measured peak fits (see architecture-and-world.md).

## Deployment

Static hosting is enough for the MVP (no backend). Any public host, DNS or domain change requires user
permission, and on this server follow the host's infrastructure rules. Deliver a deployment plan with:
the build command, the output size, the target host, verification steps (loads over HTTPS, the bundle
checksum is accepted, parity self-check passes) and a rollback (previous build kept).
