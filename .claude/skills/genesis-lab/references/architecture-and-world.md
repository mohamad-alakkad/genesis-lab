# Architecture, runtime, resource budget and world

**Summary.** Covers the five layers, how training runs survive session and container
lifetimes (bounded segments, portable `resume`, an optional non-LLM scheduler), the hard
**2 GB RAM budget** and how it is enforced, the default world/observation/action design, and
the rules that keep the Python simulation and the browser port in agreement. Read it in W1, W2,
W3 and W9, and before launching anything heavy.

## Layers

| Layer | Actor | Responsibility | Must never |
|---|---|---|---|
| 1 | Meta-prompt author | Wrote the brief | — |
| 2 | Skill Architect | Wrote this Skill | Build the project |
| 3 | Claude Code (you) | Design, implement, validate, launch, diagnose, report | Babysit training or act as the training loop |
| 4 | Plain Python processes | Simulate, train, checkpoint, publish, evaluate, write status | Require an LLM to make progress |
| 5 | Public web experience | Show the trained creature to visitors | Override or fake the policy's decisions |

## Runtime model: segments, not servers

A Claude Code session is not a training server. Do not assume a background process outlives the
session, or that a container lives for days. Design so it does not matter:

- Training runs in **bounded segments** (default 60 min wall time, configurable 30–120). Each
  segment ends with a verified checkpoint and a remote publish. A segment interrupted mid-way loses
  at most one checkpoint interval locally, and at most one remote-publish interval across environments.
- **One portable command** continues training anywhere: `resume`. It must work on this server, a
  local machine, a cheap VM, a CI runner or a cloud container, with only the repo plus the remote store.
- `resume --segments N` (or `--until <budget>`) may chain segments in one detached process; each
  segment boundary is a full checkpoint + publish. Segments are an accounting unit, not a reason to restart processes.
- **Detaching.** The launcher starts the trainer as a detached child (POSIX: new session via
  `setsid`/`start_new_session=True`; Windows: `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`), stdout/stderr to
  a size-rotated log under `runs/`. The launcher returns only after the child writes its first
  heartbeat and its first verified checkpoint (schedule an early first checkpoint, ~60 s), or exits
  non-zero with the reason within ≤ 3 min. On Windows, never use `CREATE_BREAKAWAY_FROM_JOB` for the
  trainer — it would escape the memory job (see below).

### Session lifetime probe (W1)

Launch a detached heartbeat-only process (writes timestamp + pid every 60 s to `runs/probe/`,
exits by itself after 24 h, uses negligible memory). End the session. Next session: read the last
timestamp once. Record "background processes survive session end: yes/no, observed ≥ X h". Do not
wait for the answer inside a session.

### Non-LLM scheduler (only if sessions or containers prove short-lived)

On a host where detached processes survive (e.g. this server, if the probe says so), a detached
`resume --segments N` is enough. Elsewhere, evaluate a plain scheduler. Leading candidate: **GitHub
Actions**. Each job runs `resume` for a bounded time below the job timeout (historically 6 h on hosted
runners), publishes, and the next scheduled or chained run continues. Before proposing it, verify the
current runner CPU/RAM, job timeout and minute quotas (public vs private repo), and remember the 2 GB
budget applies to this server only. **Ask the user before enabling it** (cost + repo settings).
Alternatives: OS scheduler on a VM (cron/systemd timer/Task Scheduler). On this shared server,
creating a scheduled task is an operational change, so follow the host's infrastructure rules first.

Forbidden: a Claude session or Claude API call as the loop that keeps training alive.

## Resource budget and enforcement

The user authorized GENESIS to use this server under a **hard cap of 2 GB RAM for all GENESIS
processes combined**. Server stability and existing services come first. Rules:

1. **Hard OS limit before any heavy process.** Run training, evaluation sweeps, builds and headless
   browsers inside a memory-limited container that the OS enforces:
   - Windows: a **Job Object** with `JOB_OBJECT_LIMIT_JOB_MEMORY` (job-wide committed memory) set to
     **1800 MB**, plus `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` only if the launcher stays alive (otherwise
     omit it so the trainer survives the launcher). Child processes inherit the job automatically.
     Implement with `ctypes` (no extra dependency) or `pywin32`.
   - Linux: cgroup v2 via `systemd-run --user --scope -p MemoryMax=1800M -p MemorySwapMax=0 …`, or
     `docker run --memory=1800m --memory-swap=1800m`. Avoid `ulimit -v` (PyTorch reserves large virtual
     ranges and fails spuriously).
   - Test the wrapper in W1: a child that allocates 2.5 GB must be stopped by the OS, and the
     rest of the server must stay unaffected. Record the result.
2. **In-process watchdog** (inside the trainer, every heartbeat): sum RSS of the GENESIS process tree
   (`psutil`). At **1.4 GB** → checkpoint, write end reason `resource_limit`, exit cleanly. At
   **1.6 GB** → immediate checkpoint attempt then exit. The OS limit at 1.8 GB is the backstop, never the plan.
3. **Protect the host.** Before launching, read system free RAM. Refuse to start if free RAM is below
   `2 GB + 1 GB margin`. While running, if system available RAM drops below 10 % of total (other
   services need it), checkpoint and exit with `host_pressure`. `status` reports it; `resume` retries only on the next launch.
4. **CPU courtesy.** Lower priority (Windows `BELOW_NORMAL_PRIORITY_CLASS`; POSIX `nice 10`). Use at
   most half the measured logical cores for GENESIS (minimum 1). Set `torch.set_num_threads`,
   `OMP_NUM_THREADS` and `MKL_NUM_THREADS` explicitly.
5. **One heavy GENESIS process at a time.** Trainer, `npm` build, headless browser and long
   evaluation never overlap unless their combined measured peak fits under 1.4 GB. To build the demo
   or run browser tests while training: stop training gracefully (stop file), do the work, `resume`.
6. **Memory-lean design defaults** (these usually decide whether 2 GB is enough):
   - **Single process, batch-vectorized NumPy env** (all N envs as arrays stepped together) instead of
     multiprocess workers. Each worker process that imports PyTorch costs ~300–500 MB; on Windows,
     spawn-based workers re-import everything. If subprocess workers are ever needed, they MUST NOT import torch.
   - Tiny MLP, modest rollout buffer (`n_envs × n_steps × obs_dim` floats — compute it, it is small).
   - No replay buffer larger than ~200 MB if an off-policy method is chosen.
   - Evaluation runs in the same process at scheduled points, or in a separate process only when the trainer is stopped.
   - Logs and metrics streamed to disk with rotation; no in-memory history growth.
7. **Record evidence.** Status and the segment history carry `peak_rss_mb` (process tree) and the
   enforced limit. MVP item 8 requires it.

## World design defaults (revise with evidence, record in decisions log)

**Behavior set (default):** find food to restore energy, avoid a hazard, route around obstacles,
under an energy budget. This is the smallest set that is visibly fascinating and measurable.
Simplify further if learning does not emerge; richness never beats correct reward and dynamics.

**World:** continuous 2D arena (bounded, walls), a few food items (respawn on eat), 1–3 hazard zones
(energy drain or episode end), several convex obstacles (circles or axis-aligned boxes keep collision
and ray tests cheap and exactly portable). Fixed tick (e.g. 10 Hz world time). Energy decays per tick
plus a movement cost; eating restores energy; episode ends at zero energy or a step cap.

**Observations (default):** compact, egocentric, normalized:
- `K` ray casts over a forward field of view (e.g. 9–15 rays); per ray: normalized distance to the
  first hit plus a one-hot hit type (none, wall, obstacle, food, hazard);
- own energy, forward speed, angular velocity; optionally a short "last action" one-hot.
No images, no absolute coordinates (they hurt generalization across layouts and bloat the network).

**Actions (default):** small discrete set (e.g. forward, forward-left, forward-right, turn-left,
turn-right, stay). Discrete actions make deterministic demo selection (argmax) and Python/JS parity
trivial. Continuous control is acceptable only with a stated reason.

**Randomization:** layouts, food/hazard/obstacle placement, start pose, and **mid-episode
perturbations** matching what visitors can do (food relocation; obstacle insertion up to the demo's
maximum, never on top of the creature). Declare the bounds in the world spec; the demo stays inside
them. Training and evaluation seed spaces are disjoint (e.g. training seeds < 10^9, evaluation seeds in a
reserved range) and recorded in the spec.

**Throughput:** target thousands of env steps/sec per core. Measure and record it in Stage A; if it is
far below, profile the ray casts first (vectorize over rays and envs).

## One world, two implementations

`docs/WORLD_SPEC.md` (from [../templates/world-spec.md](../templates/world-spec.md)) is the source of
truth: units, tick, sensor geometry, energy rules, collision rules, observation layout, action
semantics, normalization, RNG, randomization bounds, version. Python is the reference implementation;
the browser gets a TypeScript/JS port of the same 2D rules. 3D art sits on top of the 2D plane.

Determinism rules that make parity achievable:
- **Portable PRNG.** Layout generation and perturbations use a specified integer PRNG (e.g. mulberry32
  or PCG32, defined in the spec) implemented identically in both languages. Do not use NumPy's
  generator for anything the browser must reproduce. Training may use NumPy RNG for non-world
  randomness (e.g. action sampling).
- **Float64 for world math** in both (JS numbers are float64). The policy may run in float32; the
  parity tolerance accounts for it.
- Fixed order of operations for collision resolution and eating; no hash-map iteration order dependencies.
- Same constants file: generate the JS constants from the spec or a shared JSON, never hand-copy.

**Parity tests (mandatory, run in CI or `npm test`):**
1. *World parity:* recorded seeds + action sequences (incl. perturbation events) exported from Python
   as JSON fixtures. The JS port replays them. Observations must match within `1e-6` absolute per step for
   ≥ 500 steps per fixture, on ≥ 20 fixtures. If chaotic divergence appears at branch thresholds, also run
   per-step checks (load reference state → one step → compare), and keep open-loop runs as long as they stay exact.
2. *Policy parity:* the exported bundle evaluated in JS on ≥ 1,000 recorded observations must
   reproduce Python logits within `1e-4` and identical argmax actions (≥ 99.9 %; any mismatch must be on
   near-ties and listed).

**Decision rate:** the demo decides exactly once per world tick, as in training. Rendering interpolates
between ticks at display frame rate and never changes the decided action.
